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

Pipeline (two stages, not a single joint model)
------------------------------------------------
1. Train PortfolioLSTM exactly as in models/portfolio_lstm.py (unchanged -
   this reuses run_pipeline_multi_seed() from that module directly, so
   --n-seeds/--restart-strategy apply here too).
2. Freeze it. For every window (train and validation), take its predicted
   weights as a FIXED input - no gradient flows back into PortfolioLSTM
   from here on, so the risk network can only learn to scale the given
   position, not reshape which assets it favors.
3. RiskLSTM does NOT look at raw log returns directly. Instead, for every
   trailing `--risk-rolling-window` days inside the lookback window, it
   computes each asset's rolling standard deviation, skewness, and excess
   kurtosis - the moments a risk manager actually looks at (vol = realized
   risk, skewness = asymmetric tail risk, kurtosis = fat-tail/regime
   instability) - and feeds THAT rolling-moment sequence through its own
   LSTM. The final hidden state is concatenated with the frozen portfolio
   weights, and the combined vector maps to one attenuation factor per
   asset in [max_attenuation, 1] - see --max-attenuation (default 0.33): a
   hard floor on exposure per asset that holds regardless of how
   unconfident the network gets, so it can de-risk an asset without ever
   fully zeroing it out.
4. It is trained on its own (a separate optimizer, only its own
   parameters) to maximize the Sharpe ratio of the ATTENUATED portfolio:
       final_weights = portfolio_weights * attenuation   (elementwise, per asset)
       portfolio_return = dot(final_weights, next_returns)
   Exactly like models/portfolio_lstm.py, this is full-batch training on
   the whole train-period return path - Sharpe is a property of the
   return distribution, not of individual samples.

Usage
-----
    python -m models.risk_lstm \
        --pairs EURUSD GBPUSD USDJPY \
        --lookback 30 --weight-scheme softmax --epochs 300 \
        --risk-hidden-size 16 --risk-epochs 200
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from models.portfolio_lstm import (
    PortfolioResult,
    build_arg_parser as build_portfolio_arg_parser,
    run_pipeline_multi_seed as run_portfolio_pipeline,
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
        self.head_2 = nn.Linear(hidden_size // 2, n_assets)
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

    def save_model(self, path: str = "models/risk_lstm.pt") -> None:
        """Persist trained weights and the architecture config needed to
        reconstruct this model. Unlike PortfolioLSTM, no standardization
        stats are needed here - RiskLSTM consumes the same already-standardized
        X the (separately saved/loaded) PortfolioLSTM does.
        """
        torch.save(
            {
                "config": {
                    "n_assets": self.n_assets,
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout_p,
                    "max_attenuation": self.max_attenuation,
                    "rolling_window": self.rolling_window,
                },
                "state_dict": self.state_dict(),
            },
            path,
        )
        logger.info("Saved model weights to %s", path)

    @classmethod
    def load_model(cls, path: str) -> "RiskLSTM":
        """Reconstruct a RiskLSTM from a checkpoint saved by save_model()."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model


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
) -> None:
    """Full-batch training of RiskLSTM only - `portfolio_weights_train` is a
    fixed input (already detached from PortfolioLSTM's graph by the caller),
    so gradients here only ever update the risk network's own parameters.

    `X_train` is the RAW log-return window (not a pre-computed rolling
    statistic) - risk_model.forward() computes rolling std/skewness/
    kurtosis internally (see rolling_moments()), so training and inference
    always apply the identical transform and can never drift out of sync.
    """
    optimizer = torch.optim.Adam(risk_model.parameters(), lr=lr, weight_decay=weight_decay)

    risk_model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        noisy_X_train = X_train + torch.randn_like(X_train) * noise_std if noise_std > 0 else X_train
        attenuation = risk_model(noisy_X_train, portfolio_weights_train)        # (n_train, n_assets)
        scaled_weights = portfolio_weights_train * attenuation                  # elementwise, per asset
        portfolio_returns = (scaled_weights * next_returns_train).sum(dim=-1)   # (n_train,)
        loss = -sharpe_ratio(portfolio_returns)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % max(epochs // 10, 1) == 0:
            logger.info(
                "epoch %d/%d - train Sharpe (attenuated) %.4f | mean attenuation %.3f",
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
    risk_model: RiskLSTM
    dates_train: pd.DatetimeIndex
    dates_val: pd.DatetimeIndex
    attenuation_train: np.ndarray        # (n_train, n_assets), each entry in [max_attenuation, 1]
    attenuation_val: np.ndarray          # (n_val, n_assets), each entry in [max_attenuation, 1]
    returns_train_raw: np.ndarray        # vol-targeted portfolio returns WITHOUT attenuation
    returns_val_raw: np.ndarray
    returns_train_scaled: np.ndarray     # vol-targeted portfolio returns WITH attenuation
    returns_val_scaled: np.ndarray


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Reuses every models/portfolio_lstm.py argument as-is - that parser
    already defines --risk-overlay/--risk-hidden-size/--risk-epochs/--risk-lr
    (RiskLSTM is a much simpler task than picking the portfolio itself, so
    those default to a smaller network and fewer epochs than PortfolioLSTM's)."""
    return build_portfolio_arg_parser(description)


def add_risk_overlay(portfolio_result: PortfolioResult, args: argparse.Namespace) -> RiskResult:
    """Stage 2 only: given an already-trained PortfolioResult (from
    models/portfolio_lstm.py's run_pipeline), freeze it and train a
    RiskLSTM on top to attenuate its weights.

    Split out from run_pipeline() below so models/portfolio_lstm.py can
    call this directly on a PortfolioResult it already has, via its
    `--risk-overlay` flag, without training PortfolioLSTM a second time.
    `args` needs `--risk-hidden-size`/`--risk-epochs`/`--risk-lr` plus the
    `--dropout`/`--weight-decay`/`--noise-std` also used for PortfolioLSTM.
    """
    # Freeze PortfolioLSTM: no more training, and no gradients flow back
    # into it through the weights RiskLSTM consumes as input.
    portfolio_model = portfolio_result.model
    portfolio_model.eval()
    for param in portfolio_model.parameters():
        param.requires_grad_(False)

    # portfolio_result.weights_train/weights_val are already the FINAL,
    # volatility-targeted weights (PortfolioLSTM's raw output rescaled to
    # --target-vol - see scale_weights_to_target_vol in portfolio_lstm.py).
    # Reuse them directly rather than recomputing the forward pass, so
    # RiskLSTM always attenuates the exact same weights reported/plotted
    # everywhere else. Per design, RiskLSTM only ever reduces these
    # weights further - it never re-applies any volatility scaling.
    weights_train = torch.tensor(portfolio_result.weights_train)  # (n_train, n_assets)
    weights_val = torch.tensor(portfolio_result.weights_val)

    # Stage 2 - load a previously-trained risk overlay, or train a new one
    # on top of the frozen, vol-targeted weights.
    if args.load_risk:
        risk_model = RiskLSTM.load_model(args.load_risk)
    else:
        next_returns_train = torch.tensor(portfolio_result.next_returns_train)
        risk_model = RiskLSTM(
            n_assets=len(portfolio_result.pairs),
            hidden_size=args.risk_hidden_size,
            dropout=args.dropout,
            max_attenuation=args.max_attenuation,
            rolling_window=args.risk_rolling_window,
        )
        train_risk_model(
            risk_model, portfolio_result.X_train, weights_train, next_returns_train,
            epochs=args.risk_epochs, lr=args.risk_lr,
            weight_decay=args.weight_decay, noise_std=args.noise_std,
        )

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


def run_pipeline(args: argparse.Namespace) -> RiskResult:
    """Stage 1: train PortfolioLSTM (unchanged, via models/portfolio_lstm.py).
    Stage 2: add_risk_overlay() freezes it and trains RiskLSTM on top.
    """
    portfolio_result = run_portfolio_pipeline(args)  # identical to `python -m models.portfolio_lstm`
    return add_risk_overlay(portfolio_result, args)


def main() -> None:
    parser = build_arg_parser("Train a risk-attenuation LSTM on top of the PortfolioLSTM allocator.")
    args = parser.parse_args()
    result = run_pipeline(args)
    if not args.load_portfolio:
        result.portfolio_result.model.save_model(x_mean=result.portfolio_result.x_mean, x_std=result.portfolio_result.x_std)
    if not args.load_risk:
        result.risk_model.save_model()

    raw_train_sharpe = float(sharpe_ratio(torch.tensor(result.returns_train_raw)))
    scaled_train_sharpe = float(sharpe_ratio(torch.tensor(result.returns_train_scaled)))
    raw_val_sharpe = float(sharpe_ratio(torch.tensor(result.returns_val_raw)))
    scaled_val_sharpe = float(sharpe_ratio(torch.tensor(result.returns_val_scaled)))

    logger.info(
        "In-sample  (train): raw Sharpe %.3f -> attenuated Sharpe %.3f | mean attenuation %.3f",
        raw_train_sharpe, scaled_train_sharpe, result.attenuation_train.mean(),
    )
    logger.info(
        "Out-of-sample (val): raw Sharpe %.3f -> attenuated Sharpe %.3f | mean attenuation %.3f",
        raw_val_sharpe, scaled_val_sharpe, result.attenuation_val.mean(),
    )


if __name__ == "__main__":
    main()
