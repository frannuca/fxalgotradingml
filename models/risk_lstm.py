"""Risk-attenuation LSTM: a second network, parallel to PortfolioLSTM, that
learns when to scale the whole portfolio down.

models/portfolio_lstm.py's PortfolioLSTM decides WHICH assets to hold and
in what proportion. It has no notion of "I'm not confident right now" -
softmax/tanh_norm always redistribute the full book across the pairs,
trend or no trend. This module adds a second, independent network whose
only job is to say HOW MUCH of each of those proposed positions to
actually take - one attenuation factor PER ASSET, in [max_attenuation, 1]:
close to 1 when it looks like a normal, tradeable period for that asset,
down towards max_attenuation when its recent log returns look
directionless - so the strategy scales down its exposure to that specific
asset rather than betting full size on a coin flip, without necessarily
touching the others.

Pipeline: JOINT training (the common case), or a frozen fallback for
loaded portfolios
------------------------------------------------------------------------
The common case - training a fresh PortfolioLSTM with a risk overlay
(--risk-overlay, without --load-portfolio) - trains PortfolioLSTM and
RiskLSTM TOGETHER, end-to-end, on one shared objective:
1. PortfolioLSTM proposes raw weights; models/portfolio_lstm.py's
   scale_weights_to_target_vol rescales them to --target-vol.
2. RiskLSTM reads the SAME log-return window, plus those (vol-targeted)
   weights, and outputs a per-asset attenuation factor. Rather than raw
   returns, RiskLSTM's own LSTM actually reads each asset's rolling
   standard deviation, skewness, and excess kurtosis over trailing
   `--risk-rolling-window` days (see rolling_moments()) - the moments a
   risk manager actually looks at (vol = realized risk, skewness =
   asymmetric tail risk, kurtosis = fat-tail/regime instability) - mapped
   to one attenuation factor per asset in [max_attenuation, 1] (a hard
   floor - see --max-attenuation, default 0.33 - so an asset is de-risked,
   never fully zeroed out).
3. final_weights = vol_targeted_weights * attenuation (elementwise);
   portfolio_return = dot(final_weights, next_returns).
4. ONE optimizer updates BOTH networks' parameters from the gradient of
   the SAME (negated) Sharpe ratio of that final return series - not two
   separate optimizers/training loops with PortfolioLSTM frozen partway
   through. This lets RiskLSTM's attenuation shape what PortfolioLSTM
   learns to propose in the first place (e.g. favoring positions that are
   easier to de-risk cleanly), and lets PortfolioLSTM's weights adapt
   knowing they will be attenuated downstream. --n-seeds/--restart-strategy
   restarts reseed and retrain BOTH networks together per restart (see
   run_pipeline_multi_seed() below).

Exactly like models/portfolio_lstm.py, this is full-batch training on the
whole train-period return path - Sharpe is a property of the return
distribution, not of individual samples.

Fallback: if --load-portfolio is given, that PortfolioLSTM is fixed (it
was explicitly loaded for inference, per its own documented contract) -
joint training doesn't apply, so RiskLSTM is instead trained alone on top
of it, frozen (see add_risk_overlay()/train_risk_model() below) - this is
the same two-stage behavior as before.

This module is a LIBRARY - it has no CLI of its own. The single entry
point for the whole project is main.py at the repo root, which reads a
JSON config file and orchestrates training/evaluation by calling the
functions here (and in models/portfolio_lstm.py) directly.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from models.portfolio_lstm import (
    EnsemblePortfolioLSTM,
    NoisyLinear,
    PortfolioLSTM,
    PortfolioResult,
    _PreparedData,
    _prepare_data,
    apply_transaction_costs_torch,
    evaluate_portfolio_model,
    load_pipeline as load_portfolio_pipeline,
    scale_weights_to_target_vol,
    sharpe_ratio,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Rolling risk statistics: RiskLSTM's actual input features
# --------------------------------------------------------------------------

def rolling_moments(x: torch.Tensor, window: int, eps: float = 1e-6) -> torch.Tensor:
    """Compute each asset's rolling standard deviation, skewness, and excess
    kurtosis over trailing `window`-day sub-windows of `x`.

    x: (batch, lookback, n_assets) log returns.
    Returns: (batch, lookback - window + 1, 3 * n_assets) - for every day
    from `window` onward inside the lookback window, the trailing
    `window`-day std/skewness/kurtosis of every asset, concatenated along
    the feature axis in that order.

    These three moments are a more direct read on "how risky does this
    asset look right now" than raw returns are: std is realized volatility
    (the classic risk signal), skewness flags asymmetric tails (crash risk
    vs. melt-up), and excess kurtosis (0 for a normal distribution) flags
    fat tails / regime instability. Feeding these directly, rather than
    hoping an LSTM rediscovers them from raw returns, is a more direct and
    sample-efficient way to teach a small network "is there a clear,
    stable trend or not."
    """
    if x.shape[1] < window:
        raise ValueError(f"lookback ({x.shape[1]}) must be >= rolling window ({window})")

    # (batch, T', n_assets, window): one trailing `window`-day slice ending
    # at each of the T' = lookback - window + 1 valid days.
    windows = x.unfold(dimension=1, size=window, step=1)

    mean = windows.mean(dim=-1, keepdim=True)
    centered = windows - mean
    variance = (centered ** 2).mean(dim=-1)
    std = torch.sqrt(variance + eps)

    skewness = (centered ** 3).mean(dim=-1) / (std ** 3 + eps)
    kurtosis = (centered ** 4).mean(dim=-1) / (std ** 4 + eps) - 3.0  # excess kurtosis: 0 for a normal dist

    return torch.cat([std, skewness, kurtosis], dim=-1)  # (batch, T', 3 * n_assets)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class RiskLSTM(nn.Module):
    """Maps (a window of multi-pair log returns, the portfolio weights
    PortfolioLSTM proposed for that same window) to a PER-ASSET
    attenuation vector, each entry independently in [max_attenuation, 1].

    Rather than reading raw log returns, this network's own LSTM reads the
    ROLLING STD/SKEWNESS/KURTOSIS of every asset (see rolling_moments()
    above), computed over trailing `rolling_window`-day sub-windows -
    directly-computed risk statistics instead of raw returns the LSTM
    would otherwise have to rediscover them from. Its final hidden state
    is concatenated with the proposed weight vector, then passed through
    two Linear layers (`head` -> LeakyReLU -> `head_2` -> sigmoid) that
    map down to one attenuation value per asset - the weights tell it WHAT
    position is being considered, the moments tell it whether each asset's
    recent behavior looks stable enough to hold it at full size. LeakyReLU
    (rather than plain ReLU) keeps a small gradient flowing even for units
    with a negative pre-activation, avoiding "dying ReLU" units that get
    stuck and stop learning.

    `max_attenuation` (default 0.33) is a FLOOR, not a ceiling: each
    asset's sigmoid output is rescaled into [max_attenuation, 1] -
        max_attenuation + (1 - max_attenuation) * sigmoid(logit)
    - a smooth rescaling (not a hard clamp), so there's no dead gradient
    zone. sigmoid=0 gives max_attenuation (the most this asset can ever be
    de-risked to - it's never zeroed out entirely), sigmoid=1 gives 1 (no
    attenuation at all, full-size position).
    """

    def __init__(
        self,
        n_assets: int,
        hidden_size: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0,
        max_attenuation: float = 0.33,
        rolling_window: int = 10,
        noisy_head: bool = False,
    ):
        super().__init__()
        if not (0.0 < max_attenuation <= 1.0):
            raise ValueError(f"max_attenuation must be in (0, 1], got {max_attenuation}")
        if rolling_window < 2:
            raise ValueError(f"rolling_window must be >= 2 (need >=2 points for std/skew/kurtosis), got {rolling_window}")
        # Stashed so save_model() can persist enough to reconstruct this
        # exact architecture on load, mirroring PortfolioLSTM.
        self.n_assets = n_assets
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.max_attenuation = max_attenuation
        self.rolling_window = rolling_window
        self.noisy_head = noisy_head
        n_moment_features = 3 * n_assets  # std, skewness, excess kurtosis per asset
        self.lstm = nn.LSTM(
            input_size=n_moment_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        # + n_assets for the concatenated portfolio weight vector.
        self.head = nn.Linear(hidden_size + n_assets, hidden_size // 2)
        self.head_2 = NoisyLinear(hidden_size // 2, n_assets) if noisy_head else nn.Linear(hidden_size // 2, n_assets)
        # LeakyReLU (not plain ReLU) between the two Linear layers: a
        # standard ReLU zeroes both the output AND the gradient for any
        # unit with a negative pre-activation, so a unit that lands there
        # early in training can get permanently stuck ("dying ReLU") and
        # never update again. LeakyReLU's small negative slope keeps a
        # (small) gradient flowing through those units too.
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor, portfolio_weights: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, n_assets) RAW log returns; portfolio_weights:
        # (batch, n_assets). Moments are computed here, inside forward, so
        # every caller (training and inference alike) always passes the
        # same raw window - there is exactly one place this transform
        # happens, so training and inference can never drift out of sync.
        moments = rolling_moments(x, self.rolling_window)  # (batch, T', 3*n_assets)
        _, (h_n, _) = self.lstm(moments)
        hidden = self.dropout(h_n[-1])                             # (batch, hidden_size)
        combined = torch.cat([hidden, portfolio_weights], dim=-1)  # (batch, hidden_size + n_assets)
        features = self.leaky_relu(self.head(combined))            # (batch, hidden_size // 2) - LeakyReLU in
        # between the two Linear layers is what makes head_2 an actual second
        # layer rather than collapsing into one big linear transform.
        attenuation_logit = torch.sigmoid(self.head_2(features))   # (batch, n_assets), one per asset
        # Map each asset's sigmoid output independently into [max_attenuation, 1]:
        # sigmoid=0 -> max_attenuation (floor - never de-risk further than
        # this), sigmoid=1 -> 1 (no attenuation - full-size position).
        return self.max_attenuation + (1.0 - self.max_attenuation) * attenuation_logit

    def _checkpoint_dict(self, pairs: list[str]) -> dict:
        """Build the checkpoint dict shared by save_model() (-> local file)
        and save_to_db() (-> quant.model_registry blob). Unlike
        PortfolioLSTM, no standardization stats are needed here - RiskLSTM
        consumes the same already-standardized X the (separately
        saved/loaded) PortfolioLSTM does. `pairs` is still persisted (the
        ordered FX pairs this risk model was trained/attenuated on) purely
        so a loaded RiskLSTM can be checked against the portfolio it's
        paired with at inference time (see add_risk_overlay) - attenuation
        i must line up with the same asset as portfolio weight i.
        """
        return {
            "config": {
                "n_assets": self.n_assets,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout_p,
                "max_attenuation": self.max_attenuation,
                "rolling_window": self.rolling_window,
                "noisy_head": self.noisy_head,
            },
            "state_dict": self.state_dict(),
            "pairs": list(pairs),
        }

    @classmethod
    def _from_checkpoint(cls, checkpoint: dict) -> "RiskLSTM":
        """Reconstruct a RiskLSTM from a checkpoint dict (however it was
        loaded - from a local file or a DB blob)."""
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        model.pairs = checkpoint["pairs"]
        return model

    def save_model(self, path: str = "models/risk_lstm.pt", *, pairs: list[str]) -> None:
        """Persist trained weights and the architecture config needed to reconstruct this model."""
        torch.save(self._checkpoint_dict(pairs), path)
        logger.info("Saved model weights to %s", path)

    def save_to_db(self, name: str, *, pairs: list[str], description: str = "") -> None:
        """Serialize the same checkpoint save_model() would write, and
        upsert it into quant.model_registry under `name` instead of a
        local file.
        """
        from data.model_registry import save_model_blob

        buffer = io.BytesIO()
        torch.save(self._checkpoint_dict(pairs), buffer)
        save_model_blob(name, buffer.getvalue(), model_type="risk", description=description)

    @classmethod
    def load_model(cls, path: str) -> "RiskLSTM":
        """Reconstruct a RiskLSTM from a checkpoint saved by save_model()."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        return cls._from_checkpoint(checkpoint)

    @classmethod
    def load_from_db(cls, name: str) -> "RiskLSTM":
        """Reconstruct a RiskLSTM from a checkpoint saved by save_to_db()."""
        from data.model_registry import load_model_blob

        checkpoint = torch.load(io.BytesIO(load_model_blob(name)), map_location="cpu", weights_only=True)
        return cls._from_checkpoint(checkpoint)


class EnsembleRiskLSTM(nn.Module):
    """Wraps several independently (jointly-)trained RiskLSTMs and averages
    their predicted per-asset attenuation - mirrors EnsemblePortfolioLSTM's
    role on the portfolio side, for the same reason: averaging tends to
    cancel out each individual restart's idiosyncratic overfitting to its
    own attenuation local optimum.

    Unlike EnsemblePortfolioLSTM's weight-averaging, no renormalization is
    needed here: every member's attenuation already lives in
    [max_attenuation, 1], and a plain average of several values in that
    range stays in that same range.
    """

    def __init__(self, models: list[RiskLSTM]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor, portfolio_weights: torch.Tensor) -> torch.Tensor:
        attenuations = torch.stack([m(x, portfolio_weights) for m in self.models], dim=0)
        return attenuations.mean(dim=0)

    def _checkpoint_dict(self, pairs: list[str]) -> dict:
        """Build the checkpoint dict shared by save_model() (-> local file)
        and save_to_db() (-> quant.model_registry blob)."""
        return {
            "members": [
                {
                    "config": {
                        "n_assets": m.n_assets,
                        "hidden_size": m.hidden_size,
                        "num_layers": m.num_layers,
                        "dropout": m.dropout_p,
                        "max_attenuation": m.max_attenuation,
                        "rolling_window": m.rolling_window,
                        "noisy_head": m.noisy_head,
                    },
                    "state_dict": m.state_dict(),
                }
                for m in self.models
            ],
            "pairs": list(pairs),
        }

    @classmethod
    def _from_checkpoint(cls, checkpoint: dict) -> "EnsembleRiskLSTM":
        """Reconstruct every member from a checkpoint dict (however it was
        loaded - from a local file or a DB blob)."""
        members = []
        for member_checkpoint in checkpoint["members"]:
            member = RiskLSTM(**member_checkpoint["config"])
            member.load_state_dict(member_checkpoint["state_dict"])
            member.eval()
            members.append(member)
        ensemble = cls(members)
        ensemble.eval()
        ensemble.pairs = checkpoint["pairs"]
        return ensemble

    def save_model(self, path: str = "models/risk_lstm_ensemble.pt", *, pairs: list[str]) -> None:
        """Persist every member's config + weights so the ensemble can be reloaded without retraining."""
        torch.save(self._checkpoint_dict(pairs), path)
        logger.info("Saved ensemble risk weights (%d members) to %s", len(self.models), path)

    def save_to_db(self, name: str, *, pairs: list[str], description: str = "") -> None:
        """Serialize the same checkpoint save_model() would write, and
        upsert it into quant.model_registry under `name` instead of a
        local file.
        """
        from data.model_registry import save_model_blob

        buffer = io.BytesIO()
        torch.save(self._checkpoint_dict(pairs), buffer)
        save_model_blob(name, buffer.getvalue(), model_type="risk_ensemble", description=description)

    @classmethod
    def load_model(cls, path: str) -> "EnsembleRiskLSTM":
        """Reconstruct every member from a checkpoint saved by save_model()."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        return cls._from_checkpoint(checkpoint)

    @classmethod
    def load_from_db(cls, name: str) -> "EnsembleRiskLSTM":
        """Reconstruct every member from a checkpoint saved by save_to_db()."""
        from data.model_registry import load_model_blob

        checkpoint = torch.load(io.BytesIO(load_model_blob(name)), map_location="cpu", weights_only=True)
        return cls._from_checkpoint(checkpoint)


def load_risk_model(path: str) -> nn.Module:
    """Load a RiskLSTM or EnsembleRiskLSTM checkpoint from a local file,
    auto-detecting which kind it is from the file's structure - mirrors
    models/portfolio_lstm.py's load_portfolio_model.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "members" in checkpoint:
        return EnsembleRiskLSTM.load_model(path)
    return RiskLSTM.load_model(path)


def load_risk_model_from_db(name: str) -> nn.Module:
    """Load a RiskLSTM or EnsembleRiskLSTM checkpoint from
    quant.model_registry by name, auto-detecting single vs. ensemble the
    same way load_risk_model() does for local files.
    """
    from data.model_registry import load_model_blob

    blob = load_model_blob(name)
    checkpoint = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    if "members" in checkpoint:
        return EnsembleRiskLSTM._from_checkpoint(checkpoint)
    return RiskLSTM._from_checkpoint(checkpoint)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train_risk_model(
    risk_model: nn.Module,
    X_train: torch.Tensor,
    portfolio_weights_train: torch.Tensor,
    next_returns_train: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float = 0.0,
    noise_std: float = 0.0,
    transaction_cost: float = 0.0,
) -> None:
    """Full-batch training of RiskLSTM only - `portfolio_weights_train` is a
    fixed input (already detached from PortfolioLSTM's graph by the caller),
    so gradients here only ever update the risk network's own parameters.

    `X_train` is the RAW log-return window (not a pre-computed rolling
    statistic) - risk_model.forward() computes rolling std/skewness/
    kurtosis internally (see rolling_moments()), so training and inference
    always apply the identical transform and can never drift out of sync.

    `transaction_cost` > 0 nets the training return series against turnover
    (of the FINAL, attenuated weights) before computing Sharpe - see
    apply_transaction_costs_torch in portfolio_lstm.py - so RiskLSTM learns
    to avoid attenuation patterns that cause abrupt, costly weight swings.
    """
    optimizer = torch.optim.Adam(risk_model.parameters(), lr=lr, weight_decay=weight_decay)

    risk_model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        noisy_X_train = X_train + torch.randn_like(X_train) * noise_std if noise_std > 0 else X_train
        attenuation = risk_model(noisy_X_train, portfolio_weights_train)        # (n_train, n_assets)
        scaled_weights = portfolio_weights_train * attenuation                  # elementwise, per asset
        portfolio_returns = (scaled_weights * next_returns_train).sum(dim=-1)   # (n_train,)
        portfolio_returns = apply_transaction_costs_torch(scaled_weights, portfolio_returns, transaction_cost)
        loss = -sharpe_ratio(portfolio_returns)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % max(epochs // 10, 1) == 0:
            logger.info(
                "epoch %d/%d - train Sharpe (attenuated, net of costs) %.4f | mean attenuation %.3f",
                epoch, epochs, -loss.item(), attenuation.mean().item(),
            )


def train_joint_model(
    portfolio_model: nn.Module,
    risk_model: nn.Module,
    X_train: torch.Tensor,
    X_train_raw: torch.Tensor,
    next_returns_train: torch.Tensor,
    target_vol: float,
    epochs: int,
    lr: float,
    weight_decay: float = 0.0,
    noise_std: float = 0.0,
    freeze_risk: bool = False,
    transaction_cost: float = 0.0,
) -> None:
    """Train PortfolioLSTM and RiskLSTM TOGETHER, end-to-end, on one shared
    objective: the Sharpe ratio of the FINAL portfolio returns after both
    volatility targeting and risk attenuation. ONE optimizer updates both
    networks' parameters from the same gradient signal every epoch -
    unlike train_risk_model() above, which trains RiskLSTM alone on top of
    an already-frozen PortfolioLSTM.

    This lets RiskLSTM's attenuation shape what PortfolioLSTM learns to
    propose in the first place (e.g. favoring positions that are easier to
    de-risk cleanly without wrecking Sharpe), and lets PortfolioLSTM's
    weights adapt to the fact that they'll be attenuated downstream -
    neither network trains "blind" to the other's existence.

    `freeze_risk=True` excludes risk_model's parameters from the optimizer
    (used when a --load-risk checkpoint is supplied: risk_model is fixed,
    but PortfolioLSTM still trains "aware" of it, since gradients still
    flow through risk_model's forward pass into the shared weights it
    consumes as input - only its own parameters don't move).

    `transaction_cost` > 0 nets the training return series against turnover
    of the FINAL (vol-targeted + attenuated) weights before computing
    Sharpe (see apply_transaction_costs_torch in portfolio_lstm.py) - both
    networks then learn, from the same gradient, that abrupt day-to-day
    weight swings carry a real cost, not just whatever Sharpe on gross
    returns happens to reward.
    """
    params = list(portfolio_model.parameters())
    if not freeze_risk:
        params += list(risk_model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    portfolio_model.train()
    risk_model.train(mode=not freeze_risk)  # eval() (no dropout) if frozen, train() otherwise
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        noisy_X_train = X_train + torch.randn_like(X_train) * noise_std if noise_std > 0 else X_train
        raw_weights = portfolio_model(noisy_X_train)
        weights = scale_weights_to_target_vol(raw_weights, X_train_raw, target_vol)
        attenuation = risk_model(noisy_X_train, weights)                       # (n_train, n_assets)
        scaled_weights = weights * attenuation                                # elementwise, per asset
        portfolio_returns = (scaled_weights * next_returns_train).sum(dim=-1)  # (n_train,)
        portfolio_returns = apply_transaction_costs_torch(scaled_weights, portfolio_returns, transaction_cost)
        loss = -sharpe_ratio(portfolio_returns)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % max(epochs // 10, 1) == 0:
            logger.info(
                "epoch %d/%d - train Sharpe (joint, attenuated, net of costs) %.4f | mean attenuation %.3f",
                epoch, epochs, -loss.item(), attenuation.mean().item(),
            )


# --------------------------------------------------------------------------
# End-to-end pipeline
# --------------------------------------------------------------------------

@dataclass
class RiskResult:
    """Everything needed to score and plot the risk-attenuated portfolio,
    alongside the unattenuated (vol-targeted PortfolioLSTM) baseline and
    the fully-raw (pre-vol-targeting) baseline for comparison.

    Three levels of the pipeline, all available here:
      1. portfolio_result.returns_train_unscaled / returns_val_unscaled -
         PortfolioLSTM's raw output, before volatility targeting.
      2. returns_train_raw / returns_val_raw (this class) - after
         volatility targeting (--target-vol), before the risk overlay.
         Sourced directly from portfolio_result.returns_train/returns_val.
      3. returns_train_scaled / returns_val_scaled (this class) - after
         BOTH volatility targeting AND the risk overlay's per-asset
         attenuation. The risk overlay only ever reduces stage 2's
         weights further; it never re-applies any volatility scaling.
    """

    portfolio_result: PortfolioResult
    risk_model: nn.Module  # RiskLSTM or EnsembleRiskLSTM
    dates_train: pd.DatetimeIndex
    dates_val: pd.DatetimeIndex
    attenuation_train: np.ndarray        # (n_train, n_assets), each entry in [max_attenuation, 1]
    attenuation_val: np.ndarray          # (n_val, n_assets), each entry in [max_attenuation, 1]
    returns_train_raw: np.ndarray        # vol-targeted portfolio returns WITHOUT attenuation
    returns_val_raw: np.ndarray
    returns_train_scaled: np.ndarray     # vol-targeted portfolio returns WITH attenuation
    returns_val_scaled: np.ndarray


def risk_model_name(args: argparse.Namespace) -> str:
    """Deterministic quant.model_registry name for a RiskLSTM/-ensemble
    trained with `args` - mirrors models/portfolio_lstm.py's
    portfolio_model_name(), built from the characteristics that actually
    change the trained risk network.
    """
    from data.model_registry import build_model_name

    is_ensemble = args.n_seeds > 1 and args.restart_strategy == "ensemble"
    return build_model_name(
        "risk_ensemble" if is_ensemble else "risk",
        pairs=sorted(args.pairs),
        lookback=args.lookback,
        risk_hidden_size=args.risk_hidden_size,
        risk_rolling_window=args.risk_rolling_window,
        max_attenuation=args.max_attenuation,
    )


def _check_pairs_match(risk_pairs: list[str], portfolio_pairs: list[str]) -> None:
    """A loaded RiskLSTM's attenuation output i must line up with the same
    asset as the paired PortfolioLSTM's weight i - raise loudly rather than
    silently attenuating the wrong assets if the two checkpoints disagree
    on which pairs (or what order) they were trained on.
    """
    if list(risk_pairs) != list(portfolio_pairs):
        raise ValueError(
            f"Risk model was trained on pairs {risk_pairs}, but the paired portfolio "
            f"model uses {portfolio_pairs} - the two checkpoints are incompatible."
        )


def load_risk_model_auto(value: str) -> nn.Module:
    """Load a RiskLSTM/-ensemble from either a local file path or a
    quant.model_registry name - mirrors
    models/portfolio_lstm.py's load_portfolio_model_auto.
    """
    if os.path.exists(value):
        return load_risk_model(value)
    return load_risk_model_from_db(value)


def _build_risk_result(portfolio_result: PortfolioResult, risk_model: nn.Module) -> RiskResult:
    """Given a PortfolioResult and an already-trained-or-loaded RiskLSTM/
    EnsembleRiskLSTM, compute attenuation and the final (vol-targeted AND
    attenuated) returns, and package a RiskResult. Shared by both the
    joint-training path and the frozen-portfolio fallback path below -
    the only difference between them is HOW risk_model got trained, not
    how its output turns into a RiskResult.
    """
    weights_train = torch.tensor(portfolio_result.weights_train)  # (n_train, n_assets), already vol-targeted
    weights_val = torch.tensor(portfolio_result.weights_val)

    risk_model.eval()
    with torch.no_grad():
        attenuation_train = risk_model(portfolio_result.X_train, weights_train).numpy()  # (n_train, n_assets)
        attenuation_val = risk_model(portfolio_result.X_val, weights_val).numpy()          # (n_val, n_assets)

    scaled_weights_train = weights_train.numpy() * attenuation_train  # elementwise, per asset
    scaled_weights_val = weights_val.numpy() * attenuation_val

    returns_train_scaled = (scaled_weights_train * portfolio_result.next_returns_train).sum(axis=1)
    returns_val_scaled = (scaled_weights_val * portfolio_result.next_returns_val).sum(axis=1)

    return RiskResult(
        portfolio_result=portfolio_result,
        risk_model=risk_model,
        dates_train=portfolio_result.dates_train,
        dates_val=portfolio_result.dates_val,
        attenuation_train=attenuation_train,
        attenuation_val=attenuation_val,
        returns_train_raw=portfolio_result.returns_train,
        returns_val_raw=portfolio_result.returns_val,
        returns_train_scaled=returns_train_scaled,
        returns_val_scaled=returns_val_scaled,
    )


def add_risk_overlay(portfolio_result: PortfolioResult, args: argparse.Namespace) -> RiskResult:
    """Fallback path: given an ALREADY-TRAINED-OR-LOADED, FROZEN
    PortfolioResult (used when --load-portfolio was given, so joint
    training doesn't apply - there's nothing left to jointly train on the
    portfolio side), train a RiskLSTM alone on top of it, or load one via
    --load-risk.

    For the common case (a FRESH PortfolioLSTM with --risk-overlay, no
    --load-portfolio), see run_pipeline_multi_seed() below instead - that
    path trains PortfolioLSTM and RiskLSTM TOGETHER rather than freezing
    the portfolio first.
    """
    portfolio_model = portfolio_result.model
    portfolio_model.eval()
    for param in portfolio_model.parameters():
        param.requires_grad_(False)

    # portfolio_result.weights_train/weights_val are already the FINAL,
    # volatility-targeted weights (PortfolioLSTM's raw output rescaled to
    # --target-vol - see scale_weights_to_target_vol in portfolio_lstm.py).
    weights_train = torch.tensor(portfolio_result.weights_train)  # (n_train, n_assets)

    if args.load_risk:
        risk_model = load_risk_model_auto(args.load_risk)
        _check_pairs_match(risk_model.pairs, portfolio_result.pairs)
    else:
        next_returns_train = torch.tensor(portfolio_result.next_returns_train)
        risk_model = RiskLSTM(
            n_assets=len(portfolio_result.pairs),
            hidden_size=args.risk_hidden_size,
            dropout=args.dropout,
            max_attenuation=args.max_attenuation,
            rolling_window=args.risk_rolling_window,
            noisy_head=args.noisy_head,
        )
        train_risk_model(
            risk_model, portfolio_result.X_train, weights_train, next_returns_train,
            epochs=args.risk_epochs, lr=args.risk_lr,
            weight_decay=args.weight_decay, noise_std=args.noise_std,
            transaction_cost=args.transaction_cost,
        )

    return _build_risk_result(portfolio_result, risk_model)


def _train_and_evaluate_joint(data: _PreparedData, args: argparse.Namespace) -> RiskResult:
    """Construct one PortfolioLSTM and one RiskLSTM (whatever random seed
    is currently set - seeding it once before calling this affects BOTH
    networks' initialization, since they're constructed back to back),
    train them TOGETHER via train_joint_model(), and evaluate the pair on
    both splits.
    """
    portfolio_model = PortfolioLSTM(
        n_assets=len(data.pairs),
        hidden_size=args.hidden_size,
        weight_scheme=args.weight_scheme,
        dropout=args.dropout,
        noisy_head=args.noisy_head,
    )

    freeze_risk = bool(args.load_risk)
    if freeze_risk:
        risk_model = load_risk_model_auto(args.load_risk)
        _check_pairs_match(risk_model.pairs, data.pairs)
    else:
        risk_model = RiskLSTM(
            n_assets=len(data.pairs),
            hidden_size=args.risk_hidden_size,
            dropout=args.dropout,
            max_attenuation=args.max_attenuation,
            rolling_window=args.risk_rolling_window,
            noisy_head=args.noisy_head,
        )

    train_joint_model(
        portfolio_model, risk_model, data.X_train, data.X_train_raw,
        torch.tensor(data.next_returns_train_raw),
        target_vol=args.target_vol,
        epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, noise_std=args.noise_std,
        freeze_risk=freeze_risk,
        transaction_cost=args.transaction_cost,
    )

    portfolio_result = evaluate_portfolio_model(portfolio_model, data, args.weight_scheme, args.target_vol)
    return _build_risk_result(portfolio_result, risk_model)


def run_pipeline_multi_seed(args: argparse.Namespace) -> RiskResult:
    """Train (or load) PortfolioLSTM and RiskLSTM for the --risk-overlay
    pipeline, honoring --n-seeds/--restart-strategy for BOTH networks
    together when both are trained fresh:

      - --load-portfolio given: the portfolio is fixed - joint training
        doesn't apply - fall back to add_risk_overlay() (RiskLSTM trained,
        or loaded via --load-risk, alone on top of the frozen portfolio).
        --n-seeds doesn't apply either in this case (nothing to restart on
        the portfolio side), matching models/portfolio_lstm.py's own
        run_pipeline_multi_seed().
      - Otherwise (the common case): train --n-seeds independent
        (PortfolioLSTM, RiskLSTM) pairs TOGETHER (see train_joint_model),
        each restart reseeding and retraining BOTH networks, and combine
        the restarts via --restart-strategy (on the ATTENUATED validation
        Sharpe, since that is the actual end-to-end objective now).
    """
    if args.load_portfolio:
        portfolio_result = load_portfolio_pipeline(args)
        return add_risk_overlay(portfolio_result, args)

    if args.n_seeds < 1:
        raise ValueError(f"--n-seeds must be >= 1, got {args.n_seeds}")

    data = _prepare_data(args)  # load/split/standardize once, reused by every seed

    results = []
    for seed in range(args.n_seeds):
        torch.manual_seed(seed)  # seeds BOTH PortfolioLSTM and RiskLSTM's initialization
        logger.info("--- joint restart %d/%d (seed=%d) ---", seed + 1, args.n_seeds, seed)
        results.append(_train_and_evaluate_joint(data, args))

    if len(results) == 1:
        return results[0]

    if args.restart_strategy == "best":
        best_idx, best = max(
            enumerate(results), key=lambda item: float(sharpe_ratio(torch.tensor(item[1].returns_val_scaled)))
        )
        logger.info(
            "Best of %d joint restarts: #%d (validation attenuated Sharpe %.3f)",
            len(results), best_idx, float(sharpe_ratio(torch.tensor(best.returns_val_scaled))),
        )
        return best

    # "ensemble": average both networks' outputs across restarts.
    ensemble_portfolio = EnsemblePortfolioLSTM([r.portfolio_result.model for r in results])
    ensemble_risk = EnsembleRiskLSTM([r.risk_model for r in results])
    portfolio_result = evaluate_portfolio_model(ensemble_portfolio, data, args.weight_scheme, args.target_vol)
    ensemble_result = _build_risk_result(portfolio_result, ensemble_risk)
    logger.info(
        "Ensemble of %d joint restarts: validation attenuated Sharpe %.3f",
        len(results), float(sharpe_ratio(torch.tensor(ensemble_result.returns_val_scaled))),
    )
    return ensemble_result


def print_risk_sharpe(result: RiskResult) -> None:
    """Log raw (pre-vol-targeting) -> vol-targeted -> attenuated Sharpe
    ratio and mean attenuation, for both splits. Used by main.py after
    training/loading."""
    unscaled_train_sharpe = float(sharpe_ratio(torch.tensor(result.portfolio_result.returns_train_unscaled)))
    unscaled_val_sharpe = float(sharpe_ratio(torch.tensor(result.portfolio_result.returns_val_unscaled)))
    raw_train_sharpe = float(sharpe_ratio(torch.tensor(result.returns_train_raw)))
    scaled_train_sharpe = float(sharpe_ratio(torch.tensor(result.returns_train_scaled)))
    raw_val_sharpe = float(sharpe_ratio(torch.tensor(result.returns_val_raw)))
    scaled_val_sharpe = float(sharpe_ratio(torch.tensor(result.returns_val_scaled)))

    logger.info(
        "In-sample  (train): raw Sharpe %.3f -> vol-targeted Sharpe %.3f -> attenuated Sharpe %.3f "
        "| mean attenuation %.3f",
        unscaled_train_sharpe, raw_train_sharpe, scaled_train_sharpe, result.attenuation_train.mean(),
    )
    logger.info(
        "Out-of-sample (val): raw Sharpe %.3f -> vol-targeted Sharpe %.3f -> attenuated Sharpe %.3f "
        "| mean attenuation %.3f",
        unscaled_val_sharpe, raw_val_sharpe, scaled_val_sharpe, result.attenuation_val.mean(),
    )
