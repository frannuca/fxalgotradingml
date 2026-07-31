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

Pipeline: fully SEPARATED - PortfolioLSTM first, RiskLSTM after
------------------------------------------------------------------------
The risk overlay is optional (--risk-overlay / model.risk_overlay - the
user picks whether to run it, same as always) and, when enabled, ALWAYS
runs in two independent stages - the two networks are never trained
together:

1. PortfolioLSTM is trained (or loaded) COMPLETELY on its own, via
   models/portfolio_lstm.py's own run_pipeline_multi_seed - whatever
   --n-seeds/--restart-strategy/--objective/--target-vol settings apply,
   they apply to PortfolioLSTM alone. It has no notion that a risk overlay
   might follow; nothing about its training changes whether risk_overlay
   is on or off.
2. That result is then FROZEN (eval mode, gradients disabled) and handed
   to RiskLSTM (see add_risk_overlay()): RiskLSTM reads the same log-return
   window, plus the frozen portfolio's (already vol-targeted) weights, and
   outputs a per-asset attenuation factor. Rather than raw returns,
   RiskLSTM's own LSTM reads each asset's PER-ASSET WEIGHTED PNL - the raw
   return path multiplied by that same decision-point weight (see
   RiskLSTM.forward) - and computes ITS rolling standard deviation,
   skewness, and excess kurtosis over trailing `--risk-rolling-window` days
   (see rolling_moments()): the moments a risk manager actually looks at
   (vol = realized risk, skewness = asymmetric tail risk, kurtosis =
   fat-tail/regime instability), computed on the asset's ACTUAL exposure
   (which may be leveraged via vol-targeting or short via tanh_norm) rather
   than the asset's own unlevered volatility - mapped to one attenuation
   factor per asset in [max_attenuation, 1] (a hard floor - see
   --max-attenuation, default 0.33 - so an asset is de-risked, never fully
   zeroed out). RiskLSTM's own optimizer (see
   train_risk_model()) updates only ITS parameters, maximizing the Sharpe
   ratio of final_weights = vol_targeted_weights * attenuation
   (elementwise) - i.e. it's trained purely to IMPROVE the already-decided
   portfolio's realized Sharpe, never to reshape what that portfolio
   proposes in the first place.

Exactly like models/portfolio_lstm.py, this is full-batch training on the
whole train-period return path - Sharpe is a property of the return
distribution, not of individual samples.

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
    NoisyLinear,
    PortfolioResult,
    TemporalAttentionPool,
    TrainingStopped,
    _check_stop,
    _report_epoch,
    apply_transaction_costs_torch,
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


def cross_sectional_features(x: torch.Tensor, window: int, eps: float = 1e-6) -> torch.Tensor:
    """Compute, at every trailing `window`-day position, three PORTFOLIO-
    WIDE (cross-sectional) risk-concentration signals from the rolling
    correlation structure ACROSS assets - the complement to
    rolling_moments' per-asset (marginal) risk view: an asset can look
    individually calm while the whole book is one bad day away from moving
    together (a regime rolling_moments alone can't see).

    x: (batch, T, n_assets) RAW (unweighted) per-asset log returns - the
    correlation structure should reflect actual market co-movement, not an
    artifact of the current portfolio's own weights (which can be exactly
    0 for some asset, making a WEIGHTED return series constant and its
    correlation undefined).

    Returns (batch, T - window + 1, 3): for each valid position,
      [0] average pairwise correlation - mean of the off-diagonal
          correlation matrix entries; high when everything is moving
          together (diversification isn't actually helping right now).
      [1] correlation dispersion - std of those same off-diagonal entries;
          low dispersion + high average = a genuinely one-factor market;
          high dispersion means "some pairs correlated, others not" -
          a different (messier) regime than a uniformly high average.
      [2] top eigenvalue share - the largest eigenvalue of the rolling
          correlation matrix divided by the sum of all its eigenvalues
          (which always equals n_assets) - the standard PCA-based
          "how much of the day-to-day covariation is explained by one
          common factor" concentration measure; close to 1/n_assets means
          no dominant factor (assets moving independently), close to 1
          means one shared risk factor is driving nearly everything.

    With fewer than 2 assets, pairwise correlation is undefined - returns
    an all-zero (batch, T - window + 1, 3) tensor in that case (n_assets=1
    only happens when has_cash is on with a single real pair, an edge case
    with no meaningful cross-sectional structure anyway).
    """
    batch, T, n_assets = x.shape
    n_valid = T - window + 1
    if n_assets < 2:
        return torch.zeros(batch, n_valid, 3, dtype=x.dtype, device=x.device)

    windows = x.unfold(dimension=1, size=window, step=1)  # (batch, T', n_assets, window)
    mean = windows.mean(dim=-1, keepdim=True)
    centered = windows - mean
    std = centered.pow(2).mean(dim=-1, keepdim=True).clamp(min=eps ** 2).sqrt()  # (batch, T', n_assets, 1)
    normalized = centered / std  # (batch, T', n_assets, window)

    # Correlation matrix per (batch, T'): (batch, T', n_assets, n_assets).
    corr = torch.matmul(normalized, normalized.transpose(-1, -2)) / window

    eye = torch.eye(n_assets, dtype=x.dtype, device=x.device)
    off_diag_mask = 1.0 - eye
    n_pairs = float(n_assets * (n_assets - 1))  # each unordered pair counted twice - cancels in mean/std alike

    off_diag = corr * off_diag_mask
    avg_corr = off_diag.sum(dim=(-1, -2)) / n_pairs  # (batch, T')

    sq_diff = ((corr - avg_corr.unsqueeze(-1).unsqueeze(-1)) ** 2) * off_diag_mask
    corr_dispersion = (sq_diff.sum(dim=(-1, -2)) / n_pairs).clamp(min=0.0).sqrt()  # (batch, T')

    # corr is symmetric by construction - eigvalsh is the appropriate (and
    # cheaper/more stable) op, returning REAL eigenvalues in ascending order.
    eigenvalues = torch.linalg.eigvalsh(corr)  # (batch, T', n_assets)
    top_eig_share = eigenvalues[..., -1] / eigenvalues.sum(dim=-1).clamp(min=eps)

    return torch.stack([avg_corr, corr_dispersion, top_eig_share], dim=-1)  # (batch, T', 3)


def make_risk_sequences(
    X_raw: torch.Tensor,
    next_returns_raw: np.ndarray,
    weights: np.ndarray,
    lookback: int,
    rolling_window: int,
    use_cross_sectional: bool = False,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Build RiskLSTM's actual (batch, lookback, 4 * n_assets) input: for
    every sample, the per-asset WEIGHTED PnL return itself alongside its
    rolling std/skew/kurtosis - computed with `rolling_window` sized
    completely independently of `lookback` (previously, RiskLSTM's own
    rolling-moments computation ran on a single already-lookback-windowed
    slice, so it was bound by `lookback >= rolling_window`; not anymore).

    Order of operations (stats first, then lookback-window the result):
      1. Reconstruct the CONTINUOUS raw-return series underlying
         X_raw/next_returns_raw - X_raw[0] is the earliest window, and every
         next_returns_raw[i] is exactly the one NEW day sample i+1's window
         adds, so concatenating them recovers the whole series with no
         extra data needed.
      2. For each sample, take an EXTENDED slice of that series
         (lookback + rolling_window - 1 days, not just `lookback`) ending at
         the SAME decision day as before, weighted by THAT sample's own
         (single, forward-looking) portfolio weight - exactly the prior
         per-sample weighting, just over a longer span so there's enough
         history to compute independent rolling stats.
      3. Compute rolling std/skew/kurtosis over `rolling_window`-day
         sub-windows of that extended, weighted series (rolling_moments) -
         `rolling_window` is now free to be ANY size relative to `lookback`.
      4. Keep the per-asset weighted PnL return ITSELF (not just its
         moments) as an extra feature channel, aligned to the same trailing
         positions used in (3).
      5. Concatenating (4) and (3) gives exactly `lookback` timesteps per
         sample - (lookback + rolling_window - 1) - rolling_window + 1 ==
         lookback - so this always produces a (n_valid, lookback,
         4 * n_assets) tensor: B x S x F, ready for RiskLSTM's own LSTM.

    The first `rolling_window - 1` samples of THIS split don't have enough
    leading context to build a full extended slice and are dropped - a
    small, FIXED cost (independent of lookback), unlike the old
    lookback-bound computation. `weights`/`next_returns_raw` are returned
    sliced to match (`aligned_weights`, `aligned_next_returns`), so callers
    reuse the exact same alignment for training targets/reporting.

    When `use_cross_sectional` is set, 3 more PORTFOLIO-WIDE (not
    per-asset) channels are appended per timestep - see
    cross_sectional_features(): average pairwise correlation, correlation
    dispersion, and top-eigenvalue share of the rolling cross-asset
    correlation matrix, computed on the RAW (unweighted) return path. This
    gives the attenuation head a view of portfolio-level risk
    CONCENTRATION (are all assets moving together right now?) that no
    amount of PER-ASSET (marginal) statistics alone can express - widens
    the output to (n_valid, lookback, 4 * n_assets + 3).
    """
    n_samples, lookback_dim, n_assets = X_raw.shape
    if n_samples == 0:
        # An empty split (e.g. the test split when test_frac=0, or any
        # split that happens to have zero samples) - return empty, correctly-
        # shaped arrays rather than erroring, so callers (e.g.
        # _make_risk_features_for_split) don't need to special-case "no
        # test set" themselves.
        n_channels = 4 * n_assets + (3 if use_cross_sectional else 0)
        return (
            torch.zeros(0, lookback, n_channels, dtype=X_raw.dtype),
            weights[:0],
            next_returns_raw[:0],
        )
    n_valid = n_samples - rolling_window + 1
    if n_valid < 1:
        raise ValueError(
            f"Not enough samples ({n_samples}) for rolling_window={rolling_window} - "
            f"need at least {rolling_window}."
        )

    # Reconstruct the continuous raw series: the first window in full, then
    # exactly the one new day each subsequent sample's next_returns adds.
    full_series = torch.cat(
        [X_raw[0], torch.as_tensor(next_returns_raw, dtype=X_raw.dtype)], dim=0
    )  # (lookback + n_samples, n_assets)

    extended_len = lookback + rolling_window - 1
    # (n_windows, n_assets, extended_len) -> (n_windows, extended_len, n_assets);
    # keep only the first n_valid windows - the trailing ones would need a
    # sample beyond the last one that actually exists.
    extended_windows = full_series.unfold(0, extended_len, 1).permute(0, 2, 1)[:n_valid]

    aligned_weights = weights[rolling_window - 1:]
    aligned_next_returns = next_returns_raw[rolling_window - 1:]

    weights_t = torch.as_tensor(aligned_weights, dtype=X_raw.dtype)
    weighted = extended_windows * weights_t.unsqueeze(1)  # (n_valid, extended_len, n_assets)

    moments = rolling_moments(weighted, rolling_window)          # (n_valid, lookback, 3*n_assets)
    aligned_returns = weighted[:, rolling_window - 1:, :]        # (n_valid, lookback, n_assets)
    features = torch.cat([aligned_returns, moments], dim=-1)     # (n_valid, lookback, 4*n_assets)

    if use_cross_sectional:
        # Computed on extended_windows (RAW, unweighted returns) - NOT
        # `weighted` - so a zero-weighted asset's degenerate (constant)
        # series never enters the correlation structure; see
        # cross_sectional_features' own docstring.
        cross_sectional = cross_sectional_features(extended_windows, rolling_window)  # (n_valid, lookback, 3)
        features = torch.cat([features, cross_sectional], dim=-1)  # (n_valid, lookback, 4*n_assets + 3)

    return features, aligned_weights, aligned_next_returns


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class RiskLSTM(nn.Module):
    """Maps (a precomputed per-asset weighted-PnL + rolling-moments feature
    sequence, see make_risk_sequences(), and the portfolio weights
    PortfolioLSTM proposed for that same window) to a PER-ASSET attenuation
    vector, each entry independently in [max_attenuation, 1].

    Rather than reading raw log returns, this network's own LSTM reads, per
    timestep, each asset's PER-ASSET WEIGHTED PNL return itself ALONGSIDE
    its rolling STD/SKEWNESS/KURTOSIS (see rolling_moments()) - "weighted"
    meaning the raw return path multiplied by the same proposed weight (see
    make_risk_sequences()): directly-computed risk statistics of the
    asset's ACTUAL exposure (which the weight may leverage up via
    vol-targeting, or flip via a short position), rather than of the raw
    returns the LSTM would otherwise have to rediscover them from and that
    ignore how big a bet is actually being made. This model no longer
    computes those features itself in forward() - make_risk_sequences()
    computes the rolling statistics FIRST, independently of `lookback`, and
    only then slices the result into `lookback`-length sequences, so
    `rolling_window` no longer needs to be <= `lookback` the way it did
    when this computation ran inside forward() on an already-lookback-sized
    slice. Its final hidden state is concatenated with the proposed weight
    vector, then passed through two Linear layers (`head` -> LeakyReLU ->
    `head_2` -> sigmoid) that map down to one attenuation value per asset -
    the weights tell it WHAT position is being considered, the features
    tell it whether that SAME position's recent behavior looks stable
    enough to hold it at full size. LeakyReLU (rather than plain ReLU)
    keeps a small gradient flowing even for units with a negative
    pre-activation, avoiding "dying ReLU" units that get stuck and stop
    learning.

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
        use_cross_sectional: bool = False,
        pooling: str = "last",
    ):
        super().__init__()
        if not (0.0 < max_attenuation <= 1.0):
            raise ValueError(f"max_attenuation must be in (0, 1], got {max_attenuation}")
        if rolling_window < 2:
            raise ValueError(f"rolling_window must be >= 2 (need >=2 points for std/skew/kurtosis), got {rolling_window}")
        if pooling not in ("last", "attention"):
            raise ValueError(f"Unknown pooling: {pooling!r}")
        # Stashed so save_model() can persist enough to reconstruct this
        # exact architecture on load, mirroring PortfolioLSTM.
        self.n_assets = n_assets
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.max_attenuation = max_attenuation
        self.rolling_window = rolling_window
        self.noisy_head = noisy_head
        self.use_cross_sectional = use_cross_sectional
        self.pooling = pooling
        # weighted-PnL return, std, skewness, excess kurtosis per asset,
        # + 3 portfolio-wide cross-sectional channels (see
        # cross_sectional_features) when use_cross_sectional is on.
        n_feature_channels = 4 * n_assets + (3 if use_cross_sectional else 0)
        self.lstm = nn.LSTM(
            input_size=n_feature_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        # See TemporalAttentionPool (models/portfolio_lstm.py) - learned
        # attention pooling over EVERY timestep's hidden state, instead of
        # only h_n[-1], when pooling="attention".
        if pooling == "attention":
            self.temporal_pool = TemporalAttentionPool(hidden_size)
        # + n_assets for the concatenated portfolio weight vector. Both
        # Linear layers become NoisyLinear when noisy_head=True (not just
        # the last one) - full NoisyNet coverage of the decision path, same
        # as PortfolioLSTM's single-layer head gets.
        if noisy_head:
            self.head = NoisyLinear(hidden_size + n_assets, hidden_size // 2)
            self.head_2 = NoisyLinear(hidden_size // 2, n_assets)
        else:
            self.head = nn.Linear(hidden_size + n_assets, hidden_size // 2)
            self.head_2 = nn.Linear(hidden_size // 2, n_assets)
        # LeakyReLU (not plain ReLU) between the two Linear layers: a
        # standard ReLU zeroes both the output AND the gradient for any
        # unit with a negative pre-activation, so a unit that lands there
        # early in training can get permanently stuck ("dying ReLU") and
        # never update again. LeakyReLU's small negative slope keeps a
        # (small) gradient flowing through those units too.
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, features: torch.Tensor, portfolio_weights: torch.Tensor) -> torch.Tensor:
        # features: (batch, lookback, 4*n_assets) - PRECOMPUTED per-asset
        # weighted-PnL return + rolling std/skew/kurtosis, see
        # make_risk_sequences(). portfolio_weights: (batch, n_assets) - the
        # SAME single decision-point weight vector PortfolioLSTM proposed
        # for that window (there's only one weight per window, not one per
        # day inside it).
        #
        # Unlike the earlier design, this model does NOT compute per-asset
        # PnL or its rolling moments itself anymore - make_risk_sequences()
        # does that ONCE, ahead of time, with `rolling_window` free to be
        # any size relative to `lookback` (previously bound by needing a
        # full `lookback`-day slice to already contain >= rolling_window
        # days). Every caller (training and inference alike) is expected to
        # have called make_risk_sequences() first, so there's exactly one
        # place that transform happens and training/inference can never
        # drift out of sync.
        lstm_out, (h_n, _) = self.lstm(features)
        pooled = self.temporal_pool(lstm_out) if self.pooling == "attention" else h_n[-1]
        hidden = self.dropout(pooled)                              # (batch, hidden_size)
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
                "use_cross_sectional": self.use_cross_sectional,
                "pooling": self.pooling,
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
    """Wraps several independently-trained RiskLSTMs and averages their
    predicted per-asset attenuation - mirrors EnsemblePortfolioLSTM's role
    on the portfolio side (models/portfolio_lstm.py), for the same reason:
    averaging tends to cancel out each individual restart's idiosyncratic
    overfitting to its own attenuation local optimum. Built when
    --restart-strategy=ensemble and --n-seeds > 1 (see add_risk_overlay,
    which trains --n-seeds independent RiskLSTMs on top of the same frozen
    portfolio), or reconstructed by loading a previously-saved ensemble
    checkpoint (see load_risk_model/load_risk_model_from_db).

    Unlike EnsemblePortfolioLSTM's weight-averaging, no renormalization is
    needed here: every member's attenuation already lives in
    [max_attenuation, 1], and a plain average of several values in that
    range stays in that same range.
    """

    def __init__(self, models: list[RiskLSTM]):
        super().__init__()
        self.models = nn.ModuleList(models)

    @property
    def rolling_window(self) -> int:
        """All members are restarts of the same config, so they share one
        rolling_window - callers (e.g. _build_risk_result) that need to
        build make_risk_sequences() features don't have to special-case
        ensembles vs single models."""
        return self.models[0].rolling_window

    @property
    def use_cross_sectional(self) -> bool:
        """Mirrors .rolling_window - all members share one feature config."""
        return self.models[0].use_cross_sectional

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
                        "use_cross_sectional": m.use_cross_sectional,
                        "pooling": m.pooling,
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
    features_train: torch.Tensor,
    portfolio_weights_train: torch.Tensor,
    next_returns_train: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float = 0.0,
    noise_std: float = 0.0,
    transaction_cost: float = 0.0,
    features_val: torch.Tensor | None = None,
    portfolio_weights_val: torch.Tensor | None = None,
    next_returns_val: torch.Tensor | None = None,
    features_test: torch.Tensor | None = None,
    portfolio_weights_test: torch.Tensor | None = None,
    next_returns_test: torch.Tensor | None = None,
) -> None:
    """Full-batch training of RiskLSTM only - `portfolio_weights_train` is a
    fixed input (already detached from PortfolioLSTM's graph by the caller),
    so gradients here only ever update the risk network's own parameters.

    `features_train`/`features_val` must already be built by
    make_risk_sequences() (per-asset weighted-PnL return + rolling
    std/skew/kurtosis, `rolling_window` sized independently of `lookback`)
    - and `portfolio_weights_train`/`next_returns_train` (and the `_val`
    equivalents) must already be the ALIGNED arrays make_risk_sequences()
    returns alongside it (a few leading samples get dropped - see that
    function). This function itself is agnostic to how the input sequence
    was built; it just trains RiskLSTM against whatever it's given.

    `transaction_cost` > 0 nets the training return series against turnover
    (of the FINAL, attenuated weights) before computing Sharpe - see
    apply_transaction_costs_torch in portfolio_lstm.py - so RiskLSTM learns
    to avoid attenuation patterns that cause abrupt, costly weight swings.

    `noise_std` > 0 perturbs `features_train` directly each epoch (not the
    raw returns underneath it, since those are no longer recomputed inside
    this loop) - a coarser but equivalent-in-spirit version of
    train_portfolio_model's input-noise regularization.

    If `features_val`/`portfolio_weights_val`/`next_returns_val` are given,
    the (already-frozen) portfolio's out-of-sample return series under the
    CURRENT epoch's attenuation is also computed, purely for reporting (see
    _report_epoch) - unlike train_portfolio_model, this does NOT drive a
    best-checkpoint selection here; add_risk_overlay's restart selection
    already picks the best restart by post-training validation Sharpe.

    `features_test`/`portfolio_weights_test`/`next_returns_test`, if given,
    are computed the same way EVERY epoch too, purely for live reporting
    (the Training view's third PnL chart) - never influencing any
    training/selection decision, exactly like train_portfolio_model's own
    test tracking.
    """
    optimizer = torch.optim.Adam(risk_model.parameters(), lr=lr, weight_decay=weight_decay)
    track_val = features_val is not None and portfolio_weights_val is not None and next_returns_val is not None
    track_test = (
        features_test is not None and portfolio_weights_test is not None and next_returns_test is not None
    )

    risk_model.train()
    try:
        for epoch in range(1, epochs + 1):
            _check_stop()
            optimizer.zero_grad()
            noisy_features_train = (
                features_train + torch.randn_like(features_train) * noise_std if noise_std > 0 else features_train
            )
            attenuation = risk_model(noisy_features_train, portfolio_weights_train)  # (n_train, n_assets)
            scaled_weights = portfolio_weights_train * attenuation                  # elementwise, per asset
            portfolio_returns = (scaled_weights * next_returns_train).sum(dim=-1)   # (n_train,)
            portfolio_returns = apply_transaction_costs_torch(scaled_weights, portfolio_returns, transaction_cost)
            loss = -sharpe_ratio(portfolio_returns)
            loss.backward()
            optimizer.step()

            if epoch == 1 or epoch % max(epochs // 10, 1) == 0:
                val_returns = None
                test_returns = None
                val_msg = ""
                if track_val:
                    risk_model.eval()
                    with torch.no_grad():
                        val_attenuation = risk_model(features_val, portfolio_weights_val)
                        val_scaled_weights = portfolio_weights_val * val_attenuation
                        val_returns = (val_scaled_weights * next_returns_val).sum(dim=-1)
                        val_returns = apply_transaction_costs_torch(val_scaled_weights, val_returns, transaction_cost)
                        val_msg = f" | val Sharpe {float(sharpe_ratio(val_returns)):.4f}"
                        if track_test:
                            test_attenuation = risk_model(features_test, portfolio_weights_test)
                            test_scaled_weights = portfolio_weights_test * test_attenuation
                            test_returns = (test_scaled_weights * next_returns_test).sum(dim=-1)
                            test_returns = apply_transaction_costs_torch(
                                test_scaled_weights, test_returns, transaction_cost,
                            )
                    risk_model.train()
                logger.info(
                    "epoch %d/%d - train Sharpe (attenuated, net of costs) %.4f | mean attenuation %.3f%s",
                    epoch, epochs, -loss.item(), attenuation.mean().item(), val_msg,
                )
                _report_epoch("risk_overlay", epoch, epochs, portfolio_returns, val_returns, test_returns)
    except TrainingStopped:
        logger.info("Training stopped early at epoch %d/%d", epoch, epochs)
        raise


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
    dates_test: pd.DatetimeIndex
    attenuation_train: np.ndarray        # (n_train_risk, n_assets), each entry in [max_attenuation, 1]
    attenuation_val: np.ndarray          # (n_val_risk, n_assets), each entry in [max_attenuation, 1]
    attenuation_test: np.ndarray         # (n_test_risk, n_assets), each entry in [max_attenuation, 1]
    returns_train_raw: np.ndarray        # vol-targeted portfolio returns WITHOUT attenuation
    returns_val_raw: np.ndarray
    returns_test_raw: np.ndarray
    returns_train_scaled: np.ndarray     # vol-targeted portfolio returns WITH attenuation
    returns_val_scaled: np.ndarray
    returns_test_scaled: np.ndarray
    # ALIGNED to this class's own dates_train/dates_val/dates_test - NOT the
    # same length as portfolio_result.weights_train/next_returns_train,
    # since make_risk_sequences() drops the first `rolling_window - 1`
    # samples of each split (not enough leading context to build a full
    # feature window). Anything reported/plotted ALONGSIDE
    # attenuation_train/attenuation_val/attenuation_test must use THESE,
    # not portfolio_result's own (longer) arrays, or lengths won't line up.
    weights_train: np.ndarray            # (n_train_risk, n_assets), vol-targeted (pre-attenuation)
    weights_val: np.ndarray
    weights_test: np.ndarray
    next_returns_train: np.ndarray       # (n_train_risk, n_assets), raw log-return scale
    next_returns_val: np.ndarray
    next_returns_test: np.ndarray
    # Inverse-vol benchmark, trimmed from portfolio_result's own (see
    # PortfolioResult) to the SAME `rolling_window - 1:` alignment as
    # everything else on this class.
    benchmark_returns_train: np.ndarray
    benchmark_returns_val: np.ndarray
    benchmark_returns_test: np.ndarray


def risk_model_name(args: argparse.Namespace) -> str:
    """Deterministic quant.model_registry name for a RiskLSTM/-ensemble
    trained with `args` - mirrors models/portfolio_lstm.py's
    portfolio_model_name(), built from the characteristics that actually
    change the trained risk network. RiskLSTM now honors the SAME
    --n-seeds/--restart-strategy as PortfolioLSTM's own pipeline (see
    add_risk_overlay), so "risk_ensemble" applies here exactly like
    "portfolio_ensemble" does on the portfolio side.
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
        use_cross_sectional=getattr(args, "use_cross_sectional", False),
        pooling=getattr(args, "pooling", "last"),
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


def _unstandardize(X: torch.Tensor, x_mean: np.ndarray, x_std: np.ndarray) -> torch.Tensor:
    """Undo PortfolioLSTM's input standardization - make_risk_sequences()
    needs REAL log-return-scale values (to compute meaningful weighted-PnL/
    moments), not the standardized values PortfolioLSTM itself consumes.

    X may live on an accelerator (MPS/CUDA - see portfolio_lstm.get_device);
    .cpu() first since the rest of this reconstruction (and
    make_risk_sequences/rolling_moments/cross_sectional_features downstream)
    is one-off numpy/CPU feature engineering, not the repeated per-epoch
    work a GPU actually helps with - see _make_risk_features_for_split's
    caller, which moves the FINAL features back to the target device before
    RiskLSTM's own training loop.
    """
    return torch.as_tensor(X.cpu().numpy() * x_std + x_mean, dtype=X.dtype)


def _make_risk_features_for_split(
    portfolio_result: PortfolioResult, split: str, rolling_window: int, use_cross_sectional: bool = False,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    """Build make_risk_sequences() features for one split ("train" or
    "val") of a PortfolioResult - shared by _build_risk_result (reporting)
    and _train_one_risk_model (training), so both build features the exact
    same way. Returns (features, aligned_weights, aligned_next_returns,
    aligned_weights_tensor, aligned_next_returns_tensor).

    The feature-engineering itself (make_risk_sequences and everything it
    calls) always runs on CPU - it's one-off per split, not the repeated
    per-epoch work a GPU/MPS accelerator helps with. `device`, when given,
    moves only the FINAL tensors RiskLSTM's own training loop actually
    consumes every epoch (features, aligned_weights_t, aligned_next_returns_t)
    onto it, matching whatever device the paired PortfolioLSTM lives on.
    """
    X = getattr(portfolio_result, f"X_{split}")
    X_raw_full = _unstandardize(X, portfolio_result.x_mean, portfolio_result.x_std)
    # X may carry extra per-asset feature channels (carry, vol-normalized
    # return - see build_feature_dataframe) beyond the raw return itself.
    # make_risk_sequences needs REAL RETURNS ONLY (one per asset) to
    # reconstruct a continuous return series and compute weighted-PnL/
    # moments - channel 0 of every asset's block is always the raw return
    # (asset-major layout), so slicing every n_channels-th column recovers
    # exactly that, regardless of how many extra channels are enabled.
    n_channels = getattr(portfolio_result.model, "n_channels", 1)
    X_raw = X_raw_full[:, :, 0::n_channels]
    weights = getattr(portfolio_result, f"weights_{split}")
    next_returns = getattr(portfolio_result, f"next_returns_{split}")
    features, aligned_weights, aligned_next_returns = make_risk_sequences(
        X_raw, next_returns, weights, lookback=portfolio_result.lookback, rolling_window=rolling_window,
        use_cross_sectional=use_cross_sectional,
    )
    aligned_weights_t = torch.as_tensor(aligned_weights, dtype=features.dtype)
    aligned_next_returns_t = torch.as_tensor(aligned_next_returns, dtype=features.dtype)
    if device is not None:
        features = features.to(device)
        aligned_weights_t = aligned_weights_t.to(device)
        aligned_next_returns_t = aligned_next_returns_t.to(device)
    return features, aligned_weights, aligned_next_returns, aligned_weights_t, aligned_next_returns_t


def _build_risk_result(portfolio_result: PortfolioResult, risk_model: nn.Module) -> RiskResult:
    """Given a PortfolioResult and an already-trained-or-loaded RiskLSTM/
    EnsembleRiskLSTM, compute attenuation and the final (vol-targeted AND
    attenuated) returns, and package a RiskResult - used by add_risk_overlay
    regardless of whether risk_model was freshly trained or loaded via
    --load-risk.

    Builds make_risk_sequences() features for both splits itself (rather
    than requiring the caller to have already built them) - self-contained
    regardless of whether risk_model just finished training or was loaded
    fresh from a checkpoint. The first `rolling_window - 1` samples of each
    split get dropped (see make_risk_sequences), so dates_train/dates_val
    and everything else on the returned RiskResult are the ALIGNED
    (slightly shorter) arrays, not portfolio_result's own full-length ones.
    """
    rolling_window = risk_model.rolling_window
    use_cross_sectional = getattr(risk_model, "use_cross_sectional", False)
    # risk_model's OWN device is authoritative here (it may be freshly
    # trained or loaded from a checkpoint - either way, its parameters'
    # device is what features/weights must match to run it).
    device = next(risk_model.parameters()).device
    (
        features_train, aligned_weights_train, aligned_next_returns_train, weights_train, _,
    ) = _make_risk_features_for_split(portfolio_result, "train", rolling_window, use_cross_sectional, device)
    (
        features_val, aligned_weights_val, aligned_next_returns_val, weights_val, _,
    ) = _make_risk_features_for_split(portfolio_result, "val", rolling_window, use_cross_sectional, device)
    (
        features_test, aligned_weights_test, aligned_next_returns_test, weights_test, _,
    ) = _make_risk_features_for_split(portfolio_result, "test", rolling_window, use_cross_sectional, device)

    risk_model.eval()
    with torch.no_grad():
        # .cpu() before .numpy(): attenuation may live on an accelerator;
        # everything from here on (scaling, PnL) is plain numpy reporting.
        attenuation_train = risk_model(features_train, weights_train).cpu().numpy()  # (n_train_risk, n_assets)
        attenuation_val = risk_model(features_val, weights_val).cpu().numpy()        # (n_val_risk, n_assets)
        attenuation_test = risk_model(features_test, weights_test).cpu().numpy()     # (n_test_risk, n_assets)

    scaled_weights_train = aligned_weights_train * attenuation_train  # elementwise, per asset
    scaled_weights_val = aligned_weights_val * attenuation_val
    scaled_weights_test = aligned_weights_test * attenuation_test

    returns_train_scaled = (scaled_weights_train * aligned_next_returns_train).sum(axis=1)
    returns_val_scaled = (scaled_weights_val * aligned_next_returns_val).sum(axis=1)
    returns_test_scaled = (scaled_weights_test * aligned_next_returns_test).sum(axis=1)

    dates_train = portfolio_result.dates_train[rolling_window - 1:]
    dates_val = portfolio_result.dates_val[rolling_window - 1:]
    dates_test = portfolio_result.dates_test[rolling_window - 1:]
    returns_train_raw = portfolio_result.returns_train[rolling_window - 1:]
    returns_val_raw = portfolio_result.returns_val[rolling_window - 1:]
    returns_test_raw = portfolio_result.returns_test[rolling_window - 1:]

    return RiskResult(
        portfolio_result=portfolio_result,
        risk_model=risk_model,
        dates_train=dates_train,
        dates_val=dates_val,
        dates_test=dates_test,
        attenuation_train=attenuation_train,
        attenuation_val=attenuation_val,
        attenuation_test=attenuation_test,
        returns_train_raw=returns_train_raw,
        returns_val_raw=returns_val_raw,
        returns_test_raw=returns_test_raw,
        returns_train_scaled=returns_train_scaled,
        returns_val_scaled=returns_val_scaled,
        returns_test_scaled=returns_test_scaled,
        weights_train=aligned_weights_train,
        weights_val=aligned_weights_val,
        weights_test=aligned_weights_test,
        next_returns_train=aligned_next_returns_train,
        next_returns_val=aligned_next_returns_val,
        next_returns_test=aligned_next_returns_test,
        benchmark_returns_train=portfolio_result.benchmark_returns_train[rolling_window - 1:],
        benchmark_returns_val=portfolio_result.benchmark_returns_val[rolling_window - 1:],
        benchmark_returns_test=portfolio_result.benchmark_returns_test[rolling_window - 1:],
    )


def _train_one_risk_model(portfolio_result: PortfolioResult, args: argparse.Namespace, seed: int) -> RiskResult:
    """Construct and train ONE fresh RiskLSTM (seeded with `seed`) on top of
    the (already frozen) portfolio_result. One restart's worth of work for
    add_risk_overlay's --n-seeds loop.
    """
    torch.manual_seed(seed)
    rolling_window = args.risk_rolling_window
    use_cross_sectional = getattr(args, "use_cross_sectional", False)
    # Train RiskLSTM on the SAME device as its paired (frozen) PortfolioLSTM
    # - see portfolio_lstm.get_device.
    device = next(portfolio_result.model.parameters()).device
    features_train, _, _, weights_train, next_returns_train = _make_risk_features_for_split(
        portfolio_result, "train", rolling_window, use_cross_sectional, device,
    )
    features_val, _, _, weights_val, next_returns_val = _make_risk_features_for_split(
        portfolio_result, "val", rolling_window, use_cross_sectional, device,
    )
    features_test, _, _, weights_test, next_returns_test = _make_risk_features_for_split(
        portfolio_result, "test", rolling_window, use_cross_sectional, device,
    )
    risk_model = RiskLSTM(
        n_assets=len(portfolio_result.pairs),
        hidden_size=args.risk_hidden_size,
        dropout=args.dropout,
        max_attenuation=args.max_attenuation,
        rolling_window=rolling_window,
        noisy_head=args.noisy_head,
        use_cross_sectional=use_cross_sectional,
        pooling=getattr(args, "pooling", "last"),
    ).to(device)
    train_risk_model(
        risk_model, features_train, weights_train, next_returns_train,
        epochs=args.risk_epochs, lr=args.risk_lr,
        weight_decay=args.weight_decay, noise_std=args.noise_std,
        transaction_cost=args.transaction_cost,
        features_val=features_val, portfolio_weights_val=weights_val, next_returns_val=next_returns_val,
        features_test=features_test, portfolio_weights_test=weights_test, next_returns_test=next_returns_test,
    )
    return _build_risk_result(portfolio_result, risk_model)


def add_risk_overlay(portfolio_result: PortfolioResult, args: argparse.Namespace) -> RiskResult:
    """Given an ALREADY-TRAINED-OR-LOADED, FROZEN PortfolioResult, train a
    RiskLSTM alone on top of it (or load one via --load-risk).

    This is the ONLY path for --risk-overlay: PortfolioLSTM is always
    trained (or loaded) completely on its own first, with no notion that a
    risk overlay might follow, and RiskLSTM is trained afterward purely to
    improve THAT already-decided portfolio's realized Sharpe by learning
    when to size it down. The two networks are never trained together -
    gradients never flow from RiskLSTM's objective back into
    PortfolioLSTM's parameters, and PortfolioLSTM's own training doesn't
    even know a risk overlay exists.

    When training fresh (no --load-risk), honors the SAME --n-seeds/
    --restart-strategy PortfolioLSTM's own pipeline used - RiskLSTM's
    Sharpe-maximizing objective is just as non-convex (same LSTM/sigmoid
    nonlinearities), so it benefits from the same multi-restart treatment:
    train --n-seeds independent RiskLSTMs and either keep the one with the
    best validation (attenuated) Sharpe or average them all
    (EnsembleRiskLSTM). --n-seeds doesn't apply when --load-risk is given -
    there's nothing to restart on, matching PortfolioLSTM's own
    load_pipeline().
    """
    portfolio_model = portfolio_result.model
    portfolio_model.eval()
    for param in portfolio_model.parameters():
        param.requires_grad_(False)

    if args.load_risk:
        risk_model = load_risk_model_auto(args.load_risk)
        _check_pairs_match(risk_model.pairs, portfolio_result.pairs)
        # load_risk_model_auto reconstructs on CPU (map_location="cpu")
        # regardless of what device it was trained on - match the paired
        # (frozen) PortfolioLSTM's device instead, same as load_pipeline
        # does for a loaded PortfolioLSTM.
        risk_model = risk_model.to(next(portfolio_model.parameters()).device)
        return _build_risk_result(portfolio_result, risk_model)

    if args.n_seeds < 1:
        raise ValueError(f"--n-seeds must be >= 1, got {args.n_seeds}")

    results = []
    for seed in range(args.n_seeds):
        logger.info("--- risk-overlay restart %d/%d (seed=%d) ---", seed + 1, args.n_seeds, seed)
        results.append(_train_one_risk_model(portfolio_result, args, seed))

    if len(results) == 1:
        return results[0]

    if args.restart_strategy == "best":
        best_idx, best = max(
            enumerate(results), key=lambda item: float(sharpe_ratio(torch.tensor(item[1].returns_val_scaled)))
        )
        logger.info(
            "Best of %d risk-overlay restarts: #%d (validation attenuated Sharpe %.3f)",
            len(results), best_idx, float(sharpe_ratio(torch.tensor(best.returns_val_scaled))),
        )
        return best

    # "ensemble": average every restart's predicted attenuation.
    ensemble_risk = EnsembleRiskLSTM([r.risk_model for r in results])
    ensemble_result = _build_risk_result(portfolio_result, ensemble_risk)
    logger.info(
        "Ensemble of %d risk-overlay restarts: validation attenuated Sharpe %.3f",
        len(results), float(sharpe_ratio(torch.tensor(ensemble_result.returns_val_scaled))),
    )
    return ensemble_result


def run_pipeline_multi_seed(args: argparse.Namespace) -> RiskResult:
    """The --risk-overlay pipeline, entry point for both main.py and the
    API: train (or load) PortfolioLSTM COMPLETELY on its own, via
    models/portfolio_lstm.py's own run_pipeline_multi_seed - honoring
    whatever --n-seeds/--restart-strategy/--objective apply to IT alone,
    with no awareness that a risk overlay will follow - then train (or
    load) RiskLSTM SEPARATELY on top of that already-decided, FROZEN
    portfolio (see add_risk_overlay) to improve its realized Sharpe.

    The risk overlay is entirely optional - args.risk_overlay (checked by
    main.py/the API before calling this at all) is the only thing that
    decides whether this runs; when it's off, only
    models/portfolio_lstm.py's pipeline ever runs.
    """
    portfolio_result = run_portfolio_pipeline(args)
    return add_risk_overlay(portfolio_result, args)


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
