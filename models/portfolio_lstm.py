"""LSTM portfolio allocator: learns weights that directly maximize Sharpe ratio.

This module is a LIBRARY - it has no CLI of its own. The single entry
point for the whole project is main.py at the repo root, which reads a
JSON config file and orchestrates training/evaluation by calling the
functions here (and in models/risk_lstm.py) directly. See main.py's
module docstring for the JSON schema.

This model's output IS a trading decision: at each day it looks at the
last `lookback` days of log returns for every pair in the portfolio and
outputs a weight per pair for the following day. It is trained end-to-end
to maximize the Sharpe ratio of the resulting portfolio, not to predict
returns accurately - Sharpe ratio (mean return / return volatility) is
what a portfolio manager actually cares about, and it is differentiable,
so we can optimize it directly with gradient descent instead of using a
proxy loss like MSE.

Data flow
---------
1. Load daily close prices for the portfolio's FX pairs via data/db.py
   (see load_close_prices below).
2. Convert to log returns.
3. Slide a window over the returns:
       X[i]            = log returns of every pair over `lookback` days
       next_returns[i] = log returns of every pair on the single day right
                         after the window - i.e. what a weight decided at
                         the end of the window would actually earn.
4. Split by time into TRAIN (earliest) and VALIDATION (most recent, held
   out) - never shuffle a time series, that would leak future information
   into training.
5. The LSTM maps each X[i] window to a weight vector w[i] (one weight per
   pair). Two weight schemes are supported (--weight-scheme):
     - "softmax":   long-only. w = softmax(logits): every weight in (0, 1)
                    and the weights always sum to exactly 1.
     - "tanh_norm": long/short. w = tanh(logits) / sum(|tanh(logits)|):
                    every weight in (-1, 1), fully invested in absolute
                    terms (sum(|w|) == 1), but the signed sum can be
                    anything in [-1, 1] - allowing shorts is fundamentally
                    incompatible with also pinning the signed sum to 1.
6. The realized portfolio return at each step is dot(w[i], next_returns[i]).
   Training maximizes the Sharpe ratio of that whole return series over
   the TRAIN split in one full-batch step per epoch (Sharpe is a property
   of the entire return distribution, not of individual samples, so there
   is no per-sample loss to average over mini-batches).
7. The cumulative PnL of the strategy is just cumsum(portfolio returns),
   reported separately for the train (in-sample) and validation
   (out-of-sample) periods.

Regularization (this model overfits easily: full-batch training on a
noisy Sharpe objective for hundreds of epochs will happily memorize the
training period's noise). Three purely training-time regularizers are
applied, none of which touch the validation split - it stays a clean
blind holdout:
  - `noise_std`: Gaussian noise added to the (standardized) input
    window on every training epoch, regenerated fresh each time. This is
    the standard "add noise to the input" regularizer: the model can no
    longer fit the exact training sequence of returns, only patterns
    robust to small perturbations of it.
  - `dropout`: dropout on the LSTM's final hidden state before the
    linear head, so the head can't rely on any single hidden unit.
  - `weight_decay`: L2 penalty on the model weights (Adam's
    `weight_decay`), shrinking the model towards simpler solutions.
"""

from __future__ import annotations

import argparse
import contextvars
import copy
import io
import logging
import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

from data.db import get_time_series, upsert_pairs
from data.fx_downloader import FXDownloader, MAJOR_FX_PAIRS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_device(device: str = "auto") -> torch.device:
    """Resolve a `device` config string to an actual torch.device.

    "auto" (the default) picks the best available accelerator: Apple
    Silicon's Metal backend (MPS) if built and available (e.g. this
    project's target M-series Macs), else CUDA if available, else CPU.
    Explicit "cpu"/"mps"/"cuda" bypass detection entirely - useful for
    forcing CPU (e.g. to reproduce a result bit-for-bit, since MPS's
    numerics can differ very slightly from CPU's) or for a machine where
    autodetection picks the "wrong" device for some reason.

    Training here is full-batch (the whole split's worth of samples in one
    forward/backward pass, not mini-batched - see train_portfolio_model),
    so there's no per-batch host<->device transfer overhead once the data
    is moved once at the start; a GPU/MPS backend meaningfully speeds up
    the LSTM/attention matrix multiplications repeated every epoch.
    """
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)

# Optional hook so a caller (e.g. api/server.py, for live-updating charts in
# the Training view) can observe interim in-sample AND out-of-sample results
# DURING training, without train_portfolio_model/train_risk_model needing
# any new PUBLIC parameters - mirrors api/server.py's own _current_job_id/
# log-handler pattern (a contextvar, not a plain global, so concurrent
# training jobs in different background threads never see each other's
# callback). Library code (this module) only ever READS this; nothing here
# ever sets it.
_epoch_report_callback: contextvars.ContextVar[
    Callable[[str, int, int, np.ndarray, np.ndarray | None, np.ndarray | None], None] | None
] = contextvars.ContextVar("_epoch_report_callback", default=None)


def _report_epoch(
    stage: str,
    epoch: int,
    epochs: int,
    train_returns: torch.Tensor,
    val_returns: torch.Tensor | None = None,
    test_returns: torch.Tensor | None = None,
) -> None:
    """Call the registered interim-results callback, if any, with this
    epoch's in-sample (and, when available, out-of-sample validation AND
    test) portfolio return series - a no-op when nothing has registered one
    (e.g. the CLI / main.py path). `stage` is "portfolio" or "risk_overlay",
    so a caller tracking both training stages of the --risk-overlay
    pipeline can tell which one an update belongs to. `test_returns` is
    reported purely for LIVE DISPLAY (e.g. the Training view's third PnL
    chart) - it never influences any training/checkpoint-selection
    decision, unlike val_returns (see train_portfolio_model's best-epoch
    selection).
    """
    callback = _epoch_report_callback.get()
    if callback is not None:
        # .cpu() before .numpy(): these returns may live on an accelerator
        # (MPS/CUDA - see get_device); the callback (e.g. api/server.py's
        # live-chart reporting) only ever needs plain numpy from here.
        callback(
            stage, epoch, epochs, train_returns.detach().cpu().numpy(),
            val_returns.detach().cpu().numpy() if val_returns is not None else None,
            test_returns.detach().cpu().numpy() if test_returns is not None else None,
        )


class TrainingStopped(Exception):
    """Raised from inside train_portfolio_model/train_risk_model's epoch
    loop (see _check_stop) when a caller-registered stop-check callback
    reports a stop was requested. Deliberately a plain exception, not a
    return-value/flag threaded through every call site: it propagates
    uninterrupted through any number of nested restarts (--n-seeds) and
    sequential stages (portfolio training, then risk-overlay training), so
    only the top-level caller (api/server.py's _run_training_job) needs to
    catch it, once, to end the job cleanly instead of treating it as an
    error.
    """


# Optional hook (same contextvar pattern as _epoch_report_callback above) so
# a caller can request that an in-progress training run stop early - e.g. a
# "Stop training" button in the Training view. Checked once per epoch;
# there is deliberately no way to resume a stopped run, only to end it
# cleanly wherever it currently is.
_stop_check_callback: contextvars.ContextVar[Callable[[], bool] | None] = (
    contextvars.ContextVar("_stop_check_callback", default=None)
)


def _check_stop() -> None:
    """Raise TrainingStopped if the registered stop-check callback (if any)
    reports a stop was requested. A no-op when nothing has registered one
    (e.g. the CLI / main.py path, which has no way to request a stop)."""
    callback = _stop_check_callback.get()
    if callback is not None and callback():
        raise TrainingStopped()


# --------------------------------------------------------------------------
# Data sourcing + preparation
# --------------------------------------------------------------------------

def load_close_prices(symbols: list[str], years: int) -> pd.DataFrame:
    """Load daily close prices for `symbols` from Postgres via db.py.

    db.py's `get_time_series` is the single source of truth here. If a
    symbol has no rows in Postgres yet (e.g. the very first run), it is
    downloaded via FXDownloader and upserted (`upsert_pairs`), then the
    Postgres read is repeated - so the model always ends up training on
    whatever is in the database, not on a one-off in-memory download.
    Returns a wide DataFrame: date index, one column per symbol.
    """
    end = date.today()
    start = end - timedelta(days=365 * years)

    prices = get_time_series(symbols, start, end)

    missing = [s for s in symbols if s not in prices.columns or prices[s].dropna().empty]
    if missing:
        logger.info("Postgres has no history for %s yet - downloading and upserting", missing)
        downloader = FXDownloader(years=years)
        fresh = {
            symbol: downloader.download_pair(symbol, MAJOR_FX_PAIRS.get(symbol, f"{symbol}=X"))
            for symbol in missing
        }
        upsert_pairs(fresh)
        prices = get_time_series(symbols, start, end)

    # Inner-join on date: only keep days where every pair has a quote.
    return prices.dropna()


def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert a price DataFrame to log returns: r_t = log(p_t / p_t-1)."""
    return np.log(prices / prices.shift(1)).dropna()


def _parse_pair(pair: str) -> tuple[str, str]:
    """Split a 6-character FX ticker like "EURUSD" into (base, quote) =
    ("EUR", "USD"). Assumes the standard 3+3-letter currency code
    convention every pair in this project already follows."""
    return pair[:3], pair[3:]


def load_carry(pairs: list[str], years: int, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """For each pair, the interest-RATE DIFFERENTIAL (base currency's rate
    minus quote currency's) - the single best-documented FX predictor
    (uncovered interest parity says this SHOULD be offset by expected
    depreciation of the higher-rate currency; empirically it mostly isn't,
    which is exactly the carry trade). Reindexed/forward-filled onto
    `dates` (the daily returns index) - the underlying FRED series
    (data/rates_downloader.py) is monthly, so most days repeat the most
    recent month's differential.

    Returns a DataFrame indexed like `dates`, one column per pair, in
    DECIMAL form (a rate difference of "2.5" percentage points becomes
    0.025) - comparable in scale to a log return, not a raw percentage.
    """
    currencies = sorted({c for pair in pairs for c in _parse_pair(pair)})
    end = date.today()
    start = end - timedelta(days=365 * years)
    rates = get_time_series(currencies, start, end, source="fred", field="rate")

    carry = pd.DataFrame(index=dates)
    for pair in pairs:
        base, quote = _parse_pair(pair)
        if base not in rates.columns or quote not in rates.columns:
            logger.warning("No rate data for %s or %s - carry for %s will be all-zero", base, quote, pair)
            carry[pair] = 0.0
            continue
        diff = (rates[base] - rates[quote]) / 100.0  # percentage points -> decimal
        # Monthly series -> reindex onto the daily returns dates, forward-
        # filling (a rate differential set at the start of the month is
        # "known" every day until the next print). Deliberately NOT
        # back-filled: any day before the first available rate print gets
        # 0.0 (a neutral "no signal yet"), not a later value - back-filling
        # would leak future information into those early rows.
        carry[pair] = diff.reindex(carry.index, method="ffill").fillna(0.0)
    return carry


def vol_normalized_returns(returns: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """For each horizon in `horizons`, each asset's trailing `horizon`-day
    cumulative return divided by its trailing `horizon`-day realized
    volatility - a simple, standard risk-adjusted momentum/reversal signal
    (comparable across assets and regimes, unlike a raw cumulative return,
    since it's already scaled by how volatile getting there was).

    Returns a DataFrame with `len(horizons) * len(returns.columns)` columns,
    named "{pair}_vol{h}", aligned to `returns`'s own index. The first
    `max(horizons)` rows are necessarily NaN (not enough trailing history)
    - left for the caller to handle (see build_feature_dataframe, which
    forward/back-fills the whole feature set together).
    """
    eps = 1e-8
    out = pd.DataFrame(index=returns.index)
    for h in horizons:
        cum_return = returns.rolling(h).sum()
        vol = returns.rolling(h).std() + eps
        normalized = cum_return / vol
        for col in returns.columns:
            out[f"{col}_vol{h}"] = normalized[col]
    return out


def build_feature_dataframe(
    returns: pd.DataFrame,
    pairs: list[str],
    use_carry: bool,
    vol_horizons: list[int],
    years: int,
) -> pd.DataFrame:
    """Build the model's actual (wider) input feature set: for each pair,
    in order, [raw log return, carry?, vol-normalized return at each of
    `vol_horizons`?] - deliberately ASSET-MAJOR (all of one asset's
    channels together, then the next asset's), not channel-major, so a
    contiguous slice of the feature axis always belongs to one asset (this
    is what a future per-asset encoder would slice on).

    Channel 0 for every asset is ALWAYS its raw log return (see
    _prepare_data, which relies on this to recover the raw-returns-only
    view needed for volatility targeting) - carry/vol-normalized features
    are additional context, never a substitute for it.
    """
    n_channels = 1 + int(use_carry) + len(vol_horizons)
    carry = load_carry(pairs, years, returns.index) if use_carry else None
    vol_norm = vol_normalized_returns(returns, vol_horizons) if vol_horizons else None

    features = pd.DataFrame(index=returns.index)
    for pair in pairs:
        features[f"{pair}_ret"] = returns[pair]
        if use_carry:
            features[f"{pair}_carry"] = carry[pair]
        for h in vol_horizons:
            features[f"{pair}_vol{h}"] = vol_norm[f"{pair}_vol{h}"]

    # vol_normalized_returns' rolling windows leave NaN at the very start
    # (not enough trailing history yet) - forward-fill (for any mid-series
    # gaps) then treat any still-leading NaN as a neutral 0.0, NOT
    # back-filled: back-filling would leak a later day's value into an
    # earlier row. Raw returns themselves are never NaN here.
    features = features.ffill().fillna(0.0)
    assert features.shape[1] == n_channels * len(pairs)
    return features


def standardize(values: np.ndarray, axis) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std computed over `axis`, with a small epsilon to avoid /0."""
    return values.mean(axis=axis), values.std(axis=axis) + 1e-8


def make_portfolio_sequences(
    returns: pd.DataFrame, lookback: int, feature_returns: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Slide a window over `returns` to build (X, next_returns) pairs.

    X[i]: `lookback` days of log returns for every pair, i.e. everything
          known through day t (the window's last day). The weight the
          model computes from X[i] is the allocation decided AT day t.
    next_returns[i]: the log returns of every pair on day t+1 - the single
          day right after the window. This is what that day-t weight
          actually earns; day t+1's data never appears anywhere in X[i],
          so a weight can never be trained on the return it's applied to.
    dates[i]: the date of day t+1 (i.e. of next_returns[i]).

    `feature_returns`, if given (see build_feature_dataframe), is a WIDER
    DataFrame - same row count/date index as `returns`, extra columns
    (carry, vol-normalized returns at multiple horizons) - used to build X
    INSTEAD of `returns` itself. `next_returns`/`dates` always come from
    `returns` alone: carry/vol-normalized features are inputs the model
    reads, never a tradeable "return" a decided weight could be scored
    against.
    """
    values = returns.to_numpy(dtype=np.float32)  # shape (T, n_pairs)
    feature_values = feature_returns.to_numpy(dtype=np.float32) if feature_returns is not None else values
    index = returns.index

    X, next_returns, dates = [], [], []
    last_start = len(values) - lookback  # need exactly 1 day (t+1) after the window
    for start in range(max(last_start, 0)):
        day_t = start + lookback - 1     # last day inside the window - the decision day
        day_t_plus_1 = day_t + 1         # the day the decision's return is realized on

        X.append(feature_values[start:day_t + 1])   # rows start..day_t inclusive - never includes day_t+1
        next_returns.append(values[day_t_plus_1])
        dates.append(index[day_t_plus_1])

    if not X:
        raise ValueError(f"Not enough history ({len(values)} rows) for lookback={lookback}.")

    X, next_returns = np.stack(X), np.stack(next_returns)
    # Every window must be exactly `lookback` long. The no-look-ahead
    # invariant itself (day_t_plus_1's row is never one of X[i]'s rows) is
    # already guaranteed structurally above by slicing disjoint index
    # ranges (values[start:day_t+1] vs values[day_t_plus_1]) - it does NOT
    # need a runtime value-equality check. Real FX data legitimately
    # contains repeated/flat return values (e.g. holidays or stale closes
    # producing a return of exactly 0.0 on more than one day within the
    # same window), so comparing by VALUE would raise false positives on
    # perfectly valid data.
    assert X.shape[1] == lookback
    return X, next_returns, pd.DatetimeIndex(dates)


# --------------------------------------------------------------------------
# Noisy layers: parameter-space noise (NoisyNet, Fortunato et al. 2017)
# --------------------------------------------------------------------------

class NoisyLinear(nn.Module):
    """Drop-in replacement for nn.Linear whose weight/bias are themselves
    noisy: `weight = weight_mu + weight_sigma * epsilon`, with epsilon
    resampled fresh from N(0, 1) every forward() call while training.

    This is the "noisy cell" analogue of `noise_std` (which perturbs the
    INPUT window instead): rather than the model seeing a slightly
    different input each epoch, it has to produce a good Sharpe ratio under
    a slightly different version of its OWN decision boundary each epoch.
    That makes it harder for the model to lock onto one razor-sharp weight
    configuration that happens to maximize Sharpe on this exact training
    sample but doesn't generalize - the two noise sources compose (both can
    be enabled together) rather than replace each other.

    In eval() mode, weight/bias fall back to their mu-only (noise-free)
    values, so inference is deterministic - mirroring how dropout also
    turns itself off outside training.

    `sigma_init` sets the initial noise magnitude relative to fan-in (0.5 is
    the standard NoisyNet default); sigma is itself a learned parameter, so
    the model can shrink it over training if noise stops helping.
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        # Buffers, not parameters: resampled every forward() call (while
        # training and not frozen - see freeze_noise), never trained via
        # gradient descent themselves.
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self._sigma_init = sigma_init
        # See freeze_noise(): PortfolioLSTM.forward_sequence's sequential
        # (use_prev_weight=True) loop calls forward() many times per single
        # logical "pass" over the training period - resampling noise on
        # EVERY one of those calls would repeatedly mutate weight_epsilon/
        # bias_epsilon IN PLACE while earlier steps' still-unresolved
        # backward computation needs their OWN step's epsilon value,
        # corrupting it (a real bug this fixes: "one of the variables
        # needed for gradient computation has been modified by an in-place
        # operation"). Freezing lets a caller resample once, then hold that
        # SAME noise fixed across every step of one such sequential pass.
        self._frozen = False
        self._reset_parameters()
        self._reset_noise()

    def _reset_parameters(self) -> None:
        bound = 1.0 / (self.in_features ** 0.5)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, self._sigma_init * bound)
        nn.init.constant_(self.bias_sigma, self._sigma_init * bound)

    def _reset_noise(self) -> None:
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def resample_noise(self) -> None:
        """Public entry point for a caller that needs to control exactly
        WHEN noise gets resampled (see freeze_noise) rather than relying on
        forward()'s automatic per-call resampling."""
        self._reset_noise()

    def freeze_noise(self, frozen: bool = True) -> None:
        """While frozen, forward() reuses whatever weight_epsilon/
        bias_epsilon currently hold instead of resampling - see
        PortfolioLSTM.forward_sequence's sequential loop for the caller
        that needs this."""
        self._frozen = frozen

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            if not self._frozen:
                self._reset_noise()
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)


class TemporalAttentionPool(nn.Module):
    """Learned attention pooling over an LSTM's FULL output sequence,
    shared by both PortfolioLSTM and RiskLSTM as an alternative to using
    only the final hidden state (h_n[-1]) to summarize a window.

    h_n[-1] forces the LSTM to compress everything relevant about the
    whole lookback window into whatever it happens to be carrying at the
    very last timestep - fine for a short window, but a real limitation
    for a longer one where the most decision-relevant days (e.g. a vol
    spike) might sit in the middle rather than at the end. Attention
    pooling instead lets the network learn, per timestep, how relevant
    that day's hidden state is to the current decision, and combines all
    of them into one summary vector - the standard Bahdanau-style additive
    attention pooling:
        score_t = v^T tanh(W h_t)          (a learned scalar per timestep)
        weight  = softmax_t(score_t)        (attention weights over time)
        output  = sum_t weight_t * h_t      (the pooled summary vector)
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (*, T, hidden_size) -> (*, hidden_size). The leading
        # dimension is generic (a plain batch, or batch*n_assets for
        # PortfolioLSTM's per_asset encoder) - attention pooling is applied
        # independently within each row of it.
        scores = self.score(lstm_out)            # (*, T, 1)
        weights = torch.softmax(scores, dim=-2)   # (*, T, 1)
        return (weights * lstm_out).sum(dim=-2)   # (*, hidden_size)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class PortfolioLSTM(nn.Module):
    """Maps a window of multi-pair log returns to a portfolio weight vector
    (one weight per pair) for the following day.

    The network does NOT predict a weight directly. For each asset it
    predicts a single continuous coefficient, and the final weight is:
        weight = risk_parity_weights(window_returns_raw) * coefficient
    where `risk_parity_weights` is a fixed, long-only, inverse-volatility
    ("risk parity") baseline allocation (not learned - see that function).
    The network's whole job is choosing DIRECTION (the coefficient's sign,
    when allowed) and CONVICTION (its magnitude) per asset; how much
    capital an asset gets when fully committed is fixed by its own
    trailing volatility, never learned. `position_mode` selects the
    coefficient's range:
      "long_short" (default) - coefficient = tanh(logit) in (-1, 1): the
                    network can go short (negative coefficient), flat
                    (near 0), or long (positive), same sign convention as
                    the risk-parity baseline itself flipped.
      "long_only"  - coefficient = sigmoid(logit) in (0, 1): the network
                    can only ever scale the (already long-only) baseline
                    DOWN toward flat, never flip an asset short.
    Either way, gross exposure is bounded by construction: since
    risk_parity_weights sums to 1 and |coefficient| < 1, sum(|weight|) < 1
    always, BEFORE volatility targeting (which still applies on top
    exactly as before, and may leverage this up - see
    scale_weights_to_target_vol). This composes with everything downstream
    unchanged (the risk overlay, transaction costs, every training
    objective) - it only changes how `forward()` computes its "raw"
    weight, before any of that.

    Two encoder architectures (`encoder_type`):
      "concat"    - the original design: one LSTM consumes every asset's
                    features concatenated into a single per-timestep vector
                    (input_size = n_assets * n_channels), and one Linear
                    head maps its final hidden state directly to all
                    n_assets logits at once. Simple, but the input/head
                    weight matrices are tied to one fixed asset COUNT and
                    ORDER - retraining is required to add/remove/reorder
                    assets.
      "per_asset" - one SHARED LSTM (input_size = n_channels) is run
                    independently over each asset's own feature window
                    (same weights reused for every asset, batched as
                    batch*n_assets), producing one hidden state per asset.
                    A cross-asset combiner (`asset_combiner`) then lets
                    each asset's representation see the others - either
                    self-attention across the asset dimension ("attention",
                    the default: lets the model learn correlation/hedge
                    relationships between specific assets) or a simple
                    permutation-invariant mean-pool ("mean": one shared
                    market-wide context, cheaper and more stable with few
                    assets/short history). A single shared per-asset head
                    (also reused across assets) then maps each asset's own
                    hidden state + its cross-asset context to ONE logit.
                    Because every learned weight is reused across assets
                    (nothing depends on n_assets except the bookkeeping
                    used to reshape input/broadcast prev_weight), this
                    architecture is permutation-equivariant (reordering
                    the input pairs reorders the output weights the same
                    way) and, in principle, universe-size invariant (the
                    same weights could run on a different NUMBER of assets
                    than trained on - not currently exposed as a supported
                    workflow, but nothing in the architecture prevents it).
    """

    def __init__(
        self,
        n_assets: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        noisy_head: bool = False,
        use_prev_weight: bool = False,
        n_channels: int = 1,
        encoder_type: str = "concat",
        asset_combiner: str = "attention",
        n_attn_heads: int = 2,
        pooling: str = "last",
        position_mode: str = "long_short",
    ):
        super().__init__()
        if encoder_type not in ("concat", "per_asset"):
            raise ValueError(f"Unknown encoder_type: {encoder_type!r}")
        if asset_combiner not in ("attention", "mean"):
            raise ValueError(f"Unknown asset_combiner: {asset_combiner!r}")
        if pooling not in ("last", "attention"):
            raise ValueError(f"Unknown pooling: {pooling!r}")
        if position_mode not in ("long_short", "long_only"):
            raise ValueError(f"Unknown position_mode: {position_mode!r}")
        # Stashed (not just passed to submodules) so save_model() can
        # persist enough to reconstruct this exact architecture on load,
        # without the caller having to re-supply matching CLI flags.
        self.n_assets = n_assets
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.position_mode = position_mode
        self.noisy_head = noisy_head
        self.use_prev_weight = use_prev_weight
        # n_channels: how many input features per asset (1 = raw return
        # only; >1 when carry and/or vol-normalized-return features are
        # enabled - see build_feature_dataframe). Input width is always
        # n_assets * n_channels, asset-major ordered.
        self.n_channels = n_channels
        self.encoder_type = encoder_type
        self.asset_combiner = asset_combiner
        self.n_attn_heads = n_attn_heads
        self.pooling = pooling
        # Dropout on the pooled hidden state(s) only - a no-op (p=0) when
        # `dropout` isn't set, and inactive automatically in model.eval().
        self.dropout = nn.Dropout(dropout)
        # One shared TemporalAttentionPool works for BOTH encoder types
        # (see forward()) - it pools over whatever the leading "batch" dim
        # happens to be (plain batch for "concat", batch*n_assets for
        # "per_asset"), each row independently.
        if pooling == "attention":
            self.temporal_pool = TemporalAttentionPool(hidden_size)

        if encoder_type == "concat":
            self.lstm = nn.LSTM(
                input_size=n_assets * n_channels,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,  # inputs shaped (batch, time, features)
            )
            # + n_assets when use_prev_weight: the classic Moody & Saffell
            # recurrent-policy trick - concatenate the PREVIOUS day's own
            # decided weight into the head, so the model knows what
            # position it's already holding and can learn whether a
            # rebalance is worth its transaction cost, rather than
            # deciding each window's weight as if starting from flat
            # every time.
            head_input_size = hidden_size + n_assets if use_prev_weight else hidden_size
            self.head = (
                NoisyLinear(head_input_size, n_assets) if noisy_head
                else nn.Linear(head_input_size, n_assets)
            )
        else:  # "per_asset"
            # ONE LSTM, weight-shared across every asset (run as an
            # effective batch of batch*n_assets independent sequences of
            # width n_channels) - this is what makes the encoder itself
            # permutation/universe-size invariant.
            self.asset_lstm = nn.LSTM(
                input_size=n_channels,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            if asset_combiner == "attention":
                self.asset_attn = nn.MultiheadAttention(
                    embed_dim=hidden_size, num_heads=n_attn_heads, batch_first=True,
                )
            # Per-asset head input: this asset's own hidden state,
            # concatenated with the cross-asset context (attention output
            # or mean-pooled market context - same width, hidden_size) and
            # +1 for this asset's OWN previous weight (a scalar, not the
            # whole vector - each asset only needs to know ITS OWN
            # previously held position) when use_prev_weight.
            head_input_size = 2 * hidden_size + (1 if use_prev_weight else 0)
            # Applied per-asset with SHARED weights (nn.Linear/NoisyLinear
            # act on the last dim regardless of leading batch dims), so one
            # set of head weights produces every asset's logit.
            self.head = (
                NoisyLinear(head_input_size, 1) if noisy_head
                else nn.Linear(head_input_size, 1)
            )

    def _weights_from_coefficients(
        self,
        logits: torch.Tensor,
        risk_parity_baseline: torch.Tensor,
    ) -> torch.Tensor:
        """Turn per-asset logits into a weight vector.

        logits: (batch, n_assets) - one raw value per asset.
        risk_parity_baseline: (batch, n_assets) - PRECOMPUTED (see
        precompute_risk_parity_baseline), long-only, sums to 1 - never
        solved here, since it depends only on raw returns, never on this
        model's parameters (see precompute_risk_parity_baseline's
        docstring for why recomputing it per forward() call is wasted work).

        coefficient is tanh(logits) in (-1, 1) when self.position_mode is
        "long_short" (sign = direction, magnitude = conviction; can flip
        the risk-parity baseline short), or sigmoid(logits) in (0, 1)
        when "long_only" (can only scale the baseline down toward flat,
        never flip its sign) - see this class's own docstring. The final
        weight scales the (fixed, non-learned) risk-parity baseline by
        this coefficient.
        """
        if self.position_mode == "long_short":
            coefficients = torch.tanh(logits)      # (batch, n_assets), in (-1, 1)
        else:  # "long_only"
            coefficients = torch.sigmoid(logits)   # (batch, n_assets), in (0, 1)
        return risk_parity_baseline * coefficients

    def forward(
        self,
        x: torch.Tensor,
        prev_weight: torch.Tensor | None = None,
        risk_parity_baseline: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (batch, lookback, n_assets * n_channels). prev_weight:
        # (batch, n_assets) - only meaningful when use_prev_weight=True;
        # ignored otherwise. This is the "given a KNOWN previous weight per
        # sample" entry point - fine for a single live prediction (one
        # sample, one real previous position) or for use_prev_weight=False
        # (no cross-sample dependency at all, so a normal single batched
        # call is exact). TRAINING and full-history evaluation, where
        # use_prev_weight=True means sample i's previous weight IS sample
        # i-1's own output (a genuine dependency across the whole period,
        # not independent per-sample batching), must use forward_sequence()
        # instead - see that method.
        #
        # risk_parity_baseline: (batch, n_assets) - REQUIRED, PRECOMPUTED
        # (see precompute_risk_parity_baseline) risk-parity weights this
        # forward pass's coefficients scale (see _weights_from_coefficients).
        # Never solved here - it doesn't depend on this model's parameters,
        # only on raw returns, so solving it once outside the training/
        # inference loop and reusing it is both correct and far cheaper.
        if risk_parity_baseline is None:
            raise ValueError(
                "PortfolioLSTM.forward requires risk_parity_baseline "
                "(see precompute_risk_parity_baseline) to scale into its final weight."
            )

        if self.encoder_type == "concat":
            lstm_out, (h_n, _) = self.lstm(x)
            pooled = self.temporal_pool(lstm_out) if self.pooling == "attention" else h_n[-1]
            hidden = self.dropout(pooled)
            if self.use_prev_weight:
                if prev_weight is None:
                    prev_weight = torch.zeros(x.shape[0], self.n_assets, dtype=x.dtype, device=x.device)
                combined = torch.cat([hidden, prev_weight], dim=-1)
            else:
                combined = hidden
            logits = self.head(combined)  # (batch, n_assets)
            return self._weights_from_coefficients(logits, risk_parity_baseline)

        # "per_asset": run the SAME LSTM independently over every asset's
        # own [lookback, n_channels] window (batched as batch*n_assets),
        # then let assets attend to (or mean-pool over) each other before a
        # SHARED per-asset head turns each asset's own representation into
        # one logit.
        batch, lookback, _ = x.shape
        # (batch, lookback, n_assets, n_channels) -> (batch, n_assets, lookback, n_channels)
        x_per_asset = x.view(batch, lookback, self.n_assets, self.n_channels).permute(0, 2, 1, 3)
        x_flat = x_per_asset.reshape(batch * self.n_assets, lookback, self.n_channels)
        lstm_out, (h_n, _) = self.asset_lstm(x_flat)
        pooled = self.temporal_pool(lstm_out) if self.pooling == "attention" else h_n[-1]  # (batch*n_assets, hidden_size)
        h = self.dropout(pooled).view(batch, self.n_assets, self.hidden_size)  # (batch, n_assets, hidden_size)

        if self.asset_combiner == "attention":
            context, _ = self.asset_attn(h, h, h, need_weights=False)  # (batch, n_assets, hidden_size)
        else:  # "mean"
            context = h.mean(dim=1, keepdim=True).expand(-1, self.n_assets, -1)

        combined = torch.cat([h, context], dim=-1)  # (batch, n_assets, 2*hidden_size)
        if self.use_prev_weight:
            if prev_weight is None:
                prev_weight = torch.zeros(batch, self.n_assets, dtype=x.dtype, device=x.device)
            combined = torch.cat([combined, prev_weight.unsqueeze(-1)], dim=-1)  # (batch, n_assets, 2*hidden_size + 1)
        logits = self.head(combined).squeeze(-1)  # (batch, n_assets)
        return self._weights_from_coefficients(logits, risk_parity_baseline)

    def forward_sequence(
        self,
        X: torch.Tensor,
        initial_weight: torch.Tensor | None = None,
        risk_parity_baseline: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute weights for a whole ORDERED sequence of samples X (e.g.
        an entire train or validation split), correctly handling
        use_prev_weight=True's cross-sample dependency: sample i's input to
        the head includes sample (i-1)'s OWN decided weight, not a
        placeholder.

        When use_prev_weight=False there's no such dependency, so this is
        just one ordinary vectorized batch call (X.shape[0] independent
        samples processed in parallel) - identical speed to before this
        feature existed.

        When use_prev_weight=True, samples must be processed ONE AT A TIME
        in a Python loop (a genuine data dependency, not just a modeling
        choice) - `prev` is DETACHED before being fed to the next step
        (a "recurrent policy" that observes its own last action as state,
        not full backprop-through-time over the whole period, which would
        be both computationally intractable and numerically unstable over
        hundreds/thousands of sequential steps). Gradients still flow
        normally through EACH step's own local computation using the same
        shared LSTM/head parameters, so a loss summed over the whole
        resulting weight/return path still trains those shared parameters
        using signal from every step - standard for this kind of
        recurrent-policy setup.

        `initial_weight` (default: flat/all-zero) is the position assumed
        to be already held going into X[0] - pass the previous split's
        final weight (detached) to continue a position across a
        train/validation boundary rather than restarting from flat.

        `risk_parity_baseline` (n, n_assets) - PRECOMPUTED (see
        precompute_risk_parity_baseline), matching X 1:1 - REQUIRED, the
        fixed risk-parity weights each step's coefficients scale (see
        forward()). Precomputed ONCE for the whole split by the caller,
        not solved per-step here.
        """
        if not self.use_prev_weight:
            return self.forward(X, risk_parity_baseline=risk_parity_baseline)

        n = X.shape[0]
        prev = (
            initial_weight.detach() if initial_weight is not None
            else torch.zeros(self.n_assets, dtype=X.dtype, device=X.device)
        )
        # NoisyLinear's forward() would otherwise resample its noise on
        # EVERY one of the n sequential forward() calls below, mutating its
        # epsilon buffers IN PLACE while an earlier step's not-yet-run
        # backward pass still needs THAT step's own epsilon value - a real
        # bug ("...modified by an in-place operation") once the whole
        # period's loss is finally backpropagated. Resample ONCE here (this
        # whole sequential pass gets one fresh noise draw, same as one
        # ordinary forward() call would get) and hold it fixed for the
        # loop's duration.
        freeze = self.noisy_head and self.training
        if freeze:
            self.head.resample_noise()
            self.head.freeze_noise(True)
        try:
            weights = []
            for i in range(n):
                step_baseline = risk_parity_baseline[i : i + 1] if risk_parity_baseline is not None else None
                w = self.forward(
                    X[i : i + 1], prev.unsqueeze(0), risk_parity_baseline=step_baseline,
                ).squeeze(0)
                weights.append(w)
                prev = w.detach()
            return torch.stack(weights, dim=0)
        finally:
            if freeze:
                self.head.freeze_noise(False)

    def _checkpoint_dict(
        self,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        pairs: list[str],
        lookback: int,
        use_carry: bool = False,
        vol_horizons: list[int] | None = None,
    ) -> dict:
        """Build the checkpoint dict shared by save_model() (-> local file)
        and save_to_db() (-> quant.model_registry blob) - one definition of
        what a PortfolioLSTM checkpoint contains, serialized to either target.

        `pairs` is the ordered list of FX pairs this model was trained on -
        weight i in the model's output always corresponds to pairs[i], so
        this must be persisted alongside the weights and restored on load
        (see _from_checkpoint) rather than trusted to whatever pair list/
        order a caller happens to pass in later.

        `lookback` is the sequence length (days per input window) this
        model was trained on. The LSTM itself doesn't enforce a fixed
        sequence length (nn.LSTM runs on any T), so feeding it windows of a
        different length than training wouldn't error - it would just
        silently evaluate the model outside the regime it learned. Storing
        it lets load_pipeline rebuild windows the same size the model was
        actually trained on, rather than trusting whatever lookback a
        caller passes in later.

        `use_carry`/`vol_horizons` say exactly WHICH extra per-asset feature
        channels (beyond n_channels itself, already in `config`) this model
        expects - unlike `pairs`/`lookback`, there's no way to re-derive
        these from anything else already stored, so they must be persisted
        explicitly for load_pipeline to rebuild an identical feature set.
        """
        return {
            "config": {
                "n_assets": self.n_assets,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout_p,
                "noisy_head": self.noisy_head,
                "use_prev_weight": self.use_prev_weight,
                "n_channels": self.n_channels,
                "encoder_type": self.encoder_type,
                "asset_combiner": self.asset_combiner,
                "n_attn_heads": self.n_attn_heads,
                "pooling": self.pooling,
                "position_mode": self.position_mode,
            },
            "state_dict": self.state_dict(),
            "x_mean": torch.as_tensor(x_mean),
            "x_std": torch.as_tensor(x_std),
            "pairs": list(pairs),
            "lookback": lookback,
            "use_carry": use_carry,
            "vol_horizons": list(vol_horizons) if vol_horizons else [],
        }

    @classmethod
    def _from_checkpoint(cls, checkpoint: dict) -> "PortfolioLSTM":
        """Reconstruct a PortfolioLSTM from a checkpoint dict (however it was
        loaded - from a local file or a DB blob). The returned model also
        carries `.x_mean`/`.x_std` (as numpy arrays), `.pairs` (the ordered
        FX pairs it was trained on), `.lookback` (the sequence length it
        was trained on), and `.use_carry`/`.vol_horizons` (which extra
        feature channels it expects) so callers can standardize new input
        windows and rebuild the exact same feature set without having to
        re-supply any of it themselves.
        """
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        model.x_mean = checkpoint["x_mean"].numpy()
        model.x_std = checkpoint["x_std"].numpy()
        model.pairs = checkpoint["pairs"]
        model.lookback = checkpoint["lookback"]
        model.use_carry = checkpoint.get("use_carry", False)
        model.vol_horizons = checkpoint.get("vol_horizons", [])
        return model

    def save_model(
        self,
        path: str = "models/portfolio_lstm.pt",
        *,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        pairs: list[str],
        lookback: int,
        use_carry: bool = False,
        vol_horizons: list[int] | None = None,
    ) -> None:
        """Persist trained weights, the architecture config, the input
        standardization stats (x_mean/x_std), the ordered FX pairs, and the
        sequence length this model was trained on - a self-contained
        checkpoint that load_model() can rebuild and run inference from
        without retraining, without needing a fresh training split to
        re-derive x_mean/x_std, and without risking weights being applied
        to the wrong assets or windows of the wrong length.
        """
        torch.save(self._checkpoint_dict(x_mean, x_std, pairs, lookback, use_carry, vol_horizons), path)
        logger.info("Saved model weights to %s", path)

    def save_to_db(
        self,
        name: str,
        *,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        pairs: list[str],
        lookback: int,
        use_carry: bool = False,
        vol_horizons: list[int] | None = None,
        description: str = "",
    ) -> None:
        """Serialize the same checkpoint save_model() would write, and
        upsert it into quant.model_registry under `name` instead of a
        local file - see data/model_registry.py.
        """
        from data.model_registry import save_model_blob

        buffer = io.BytesIO()
        torch.save(self._checkpoint_dict(x_mean, x_std, pairs, lookback, use_carry, vol_horizons), buffer)
        save_model_blob(name, buffer.getvalue(), model_type="portfolio", description=description)

    @classmethod
    def load_model(cls, path: str) -> "PortfolioLSTM":
        """Reconstruct a PortfolioLSTM from a checkpoint saved by save_model()."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        return cls._from_checkpoint(checkpoint)

    @classmethod
    def load_from_db(cls, name: str) -> "PortfolioLSTM":
        """Reconstruct a PortfolioLSTM from a checkpoint saved by save_to_db()."""
        from data.model_registry import load_model_blob

        checkpoint = torch.load(io.BytesIO(load_model_blob(name)), map_location="cpu", weights_only=True)
        return cls._from_checkpoint(checkpoint)


class EnsemblePortfolioLSTM(nn.Module):
    """Wraps several independently-trained PortfolioLSTMs (different random
    seeds, so different local optima of the non-convex training landscape)
    and averages their predicted weights.

    Averaging tends to cancel out each individual run's idiosyncratic
    overfitting to the particular local optimum it landed in - a standard,
    cheap way to get a more robust result out of a non-convex training
    problem than trusting a single run.
    """

    def __init__(self, models: list[PortfolioLSTM]):
        super().__init__()
        self.models = nn.ModuleList(models)
        # Purely informational (mirrors the field every member already has).
        self.use_prev_weight = models[0].use_prev_weight
        self.n_assets = models[0].n_assets
        self.n_channels = models[0].n_channels

    def forward(
        self,
        x: torch.Tensor,
        prev_weight: torch.Tensor | None = None,
        risk_parity_baseline: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Every member sees the SAME prev_weight - there's only one actual
        # position held (the ensemble average), so all members reason
        # about a rebalance from that same real starting point, not their
        # own individual (never-actually-held) previous prediction.
        # risk_parity_baseline is the same PRECOMPUTED tensor for every
        # member (none of this is model-specific, just the shared
        # risk-parity baseline each member scales by its own coefficient -
        # see PortfolioLSTM.forward).
        weights = torch.stack(
            [m(x, prev_weight, risk_parity_baseline) for m in self.models], dim=0,
        )  # (n_models, batch, n_assets)
        # No re-normalization needed: each member's weight is
        # risk_parity_weights(...) * coefficient_i with |coefficient_i| < 1
        # and the SAME risk-parity baseline for every member, so
        # |mean_i(weight_i)| = baseline * |mean_i(coefficient_i)| <=
        # baseline * 1 - averaging alone preserves the gross-exposure bound.
        return weights.mean(dim=0)  # (batch, n_assets)

    def forward_sequence(
        self,
        X: torch.Tensor,
        initial_weight: torch.Tensor | None = None,
        risk_parity_baseline: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Ensemble counterpart of PortfolioLSTM.forward_sequence() - see
        that method's docstring. Feeds the ENSEMBLE's own (averaged)
        previous decision back to every member at each step, since that
        average is the only position actually held."""
        if not self.use_prev_weight:
            return self.forward(X, risk_parity_baseline=risk_parity_baseline)
        n = X.shape[0]
        prev = (
            initial_weight.detach() if initial_weight is not None
            else torch.zeros(self.n_assets, dtype=X.dtype, device=X.device)
        )
        # Same NoisyLinear in-place-mutation hazard as
        # PortfolioLSTM.forward_sequence (see its comment) - here across
        # EVERY member's own head, since self.forward() calls each member's
        # forward() once per sequential step.
        noisy_members = [m for m in self.models if m.noisy_head and m.training]
        for m in noisy_members:
            m.head.resample_noise()
            m.head.freeze_noise(True)
        try:
            weights = []
            for i in range(n):
                step_baseline = risk_parity_baseline[i : i + 1] if risk_parity_baseline is not None else None
                w = self.forward(
                    X[i : i + 1], prev.unsqueeze(0), risk_parity_baseline=step_baseline,
                ).squeeze(0)
                weights.append(w)
                prev = w.detach()
            return torch.stack(weights, dim=0)
        finally:
            for m in noisy_members:
                m.head.freeze_noise(False)

    def _checkpoint_dict(
        self,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        pairs: list[str],
        lookback: int,
        use_carry: bool = False,
        vol_horizons: list[int] | None = None,
    ) -> dict:
        """Build the checkpoint dict shared by save_model() (-> local file)
        and save_to_db() (-> quant.model_registry blob). All restarts were
        trained on the same standardized data, so one x_mean/x_std pair, one
        ordered `pairs` list, one `lookback`, and one `use_carry`/
        `vol_horizons` (see PortfolioLSTM._checkpoint_dict) cover the whole
        ensemble.
        """
        return {
            "members": [
                {
                    "config": {
                        "n_assets": m.n_assets,
                        "hidden_size": m.hidden_size,
                        "num_layers": m.num_layers,
                        "dropout": m.dropout_p,
                        "noisy_head": m.noisy_head,
                        "use_prev_weight": m.use_prev_weight,
                        "n_channels": m.n_channels,
                        "encoder_type": m.encoder_type,
                        "asset_combiner": m.asset_combiner,
                        "n_attn_heads": m.n_attn_heads,
                        "pooling": m.pooling,
                        "position_mode": m.position_mode,
                    },
                    "state_dict": m.state_dict(),
                }
                for m in self.models
            ],
            "x_mean": torch.as_tensor(x_mean),
            "x_std": torch.as_tensor(x_std),
            "pairs": list(pairs),
            "lookback": lookback,
            "use_carry": use_carry,
            "vol_horizons": list(vol_horizons) if vol_horizons else [],
        }

    @classmethod
    def _from_checkpoint(cls, checkpoint: dict) -> "EnsemblePortfolioLSTM":
        """Reconstruct every member from a checkpoint dict (however it was
        loaded - from a local file or a DB blob). The returned ensemble
        also carries `.x_mean`/`.x_std` (as numpy arrays), `.pairs`,
        `.lookback`, and `.use_carry`/`.vol_horizons`, same as
        PortfolioLSTM._from_checkpoint()."""
        members = []
        for member_checkpoint in checkpoint["members"]:
            member = PortfolioLSTM(**member_checkpoint["config"])
            member.load_state_dict(member_checkpoint["state_dict"])
            member.eval()
            members.append(member)
        ensemble = cls(members)
        ensemble.eval()
        ensemble.x_mean = checkpoint["x_mean"].numpy()
        ensemble.x_std = checkpoint["x_std"].numpy()
        ensemble.pairs = checkpoint["pairs"]
        ensemble.lookback = checkpoint["lookback"]
        ensemble.use_carry = checkpoint.get("use_carry", False)
        ensemble.vol_horizons = checkpoint.get("vol_horizons", [])
        return ensemble

    def save_model(
        self,
        path: str = "models/portfolio_lstm_ensemble.pt",
        *,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        pairs: list[str],
        lookback: int,
        use_carry: bool = False,
        vol_horizons: list[int] | None = None,
    ) -> None:
        """Persist every member's config + weights so the ensemble can be reloaded without retraining any of them."""
        torch.save(self._checkpoint_dict(x_mean, x_std, pairs, lookback, use_carry, vol_horizons), path)
        logger.info("Saved ensemble weights (%d members) to %s", len(self.models), path)

    def save_to_db(
        self,
        name: str,
        *,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        pairs: list[str],
        lookback: int,
        use_carry: bool = False,
        vol_horizons: list[int] | None = None,
        description: str = "",
    ) -> None:
        """Serialize the same checkpoint save_model() would write, and
        upsert it into quant.model_registry under `name` instead of a
        local file.
        """
        from data.model_registry import save_model_blob

        buffer = io.BytesIO()
        torch.save(self._checkpoint_dict(x_mean, x_std, pairs, lookback, use_carry, vol_horizons), buffer)
        save_model_blob(name, buffer.getvalue(), model_type="portfolio_ensemble", description=description)

    @classmethod
    def load_model(cls, path: str) -> "EnsemblePortfolioLSTM":
        """Reconstruct every member from a checkpoint saved by save_model()."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        return cls._from_checkpoint(checkpoint)

    @classmethod
    def load_from_db(cls, name: str) -> "EnsemblePortfolioLSTM":
        """Reconstruct every member from a checkpoint saved by save_to_db()."""
        from data.model_registry import load_model_blob

        checkpoint = torch.load(io.BytesIO(load_model_blob(name)), map_location="cpu", weights_only=True)
        return cls._from_checkpoint(checkpoint)


def load_portfolio_model(path: str) -> nn.Module:
    """Load a PortfolioLSTM or EnsemblePortfolioLSTM checkpoint from a local
    file, auto-detecting which kind it is from the file's structure -
    callers (e.g. load_pipeline() below) don't need to know in advance
    whether `path` holds a single model (saved with --restart-strategy
    best or --n-seeds 1) or an ensemble (saved with --restart-strategy
    ensemble).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "members" in checkpoint:
        return EnsemblePortfolioLSTM.load_model(path)
    return PortfolioLSTM.load_model(path)


def load_portfolio_model_from_db(name: str) -> nn.Module:
    """Load a PortfolioLSTM or EnsemblePortfolioLSTM checkpoint from
    quant.model_registry by name, auto-detecting single vs. ensemble the
    same way load_portfolio_model() does for local files.
    """
    from data.model_registry import load_model_blob

    blob = load_model_blob(name)
    checkpoint = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    if "members" in checkpoint:
        return EnsemblePortfolioLSTM._from_checkpoint(checkpoint)
    return PortfolioLSTM._from_checkpoint(checkpoint)


# --------------------------------------------------------------------------
# Sharpe ratio: the training objective
# --------------------------------------------------------------------------

def sharpe_ratio(
    portfolio_returns: torch.Tensor, periods_per_year: int = 252, eps: float = 1e-8
) -> torch.Tensor:
    """Annualized Sharpe ratio of a portfolio return series:
    mean(returns) / std(returns) * sqrt(periods_per_year).

    Differentiable in `portfolio_returns`, so this can be used directly as
    a (negated) training loss - no proxy metric like MSE is needed.
    """
    mean = portfolio_returns.mean()
    std = portfolio_returns.std() + eps
    return mean / std * (periods_per_year ** 0.5)


def rolling_window_ratio(
    portfolio_returns: torch.Tensor,
    window: int,
    downside: bool = True,
    periods_per_year: int = 252,
    eps: float = 1e-8,
) -> torch.Tensor:
    """A more robust training objective than sharpe_ratio()'s single
    whole-period ratio: the MEAN ratio computed over many overlapping
    `window`-day sub-windows of `portfolio_returns` instead of one ratio
    over the entire period.

    A single whole-period Sharpe lets a model with enough capacity find one
    weight pattern that happens to nail this exact historical sequence - it
    can be "great on net" while being terrible in some sub-periods and great
    in others, as long as the two average out favorably; that's exactly the
    kind of fit that maximizes in-sample Sharpe but doesn't generalize.
    Scoring (and averaging) the SAME ratio across many overlapping
    sub-windows instead forces the model to be consistently good across
    different segments of the training history, not just good in
    aggregate. Falls back to a single whole-period window when there isn't
    enough history for even one `window`-day slice.

    `downside=True` computes a Sortino-style ratio (mean / downside
    deviation, i.e. the std of only the below-zero returns, with zeros
    substituted for the rest so the denominator's sample size stays fixed)
    instead of Sharpe's plain mean/std - same shape of objective, still a
    single number with no extra combined penalty weight to tune, just a
    risk measure that isn't inflated by the upside volatility a trader is
    happy to keep.

    Used as the TRAINING loss only (train_portfolio_model/train_joint_model)
    - reported/plotted Sharpe elsewhere in the app still uses the plain
    whole-period sharpe_ratio(), so what gets shown to the user is unchanged;
    only what the optimizer is pushed toward changes.
    """
    n = portfolio_returns.shape[0]
    window = min(window, n)  # not enough history for a full window - fall back to one whole-period "window"
    windows = portfolio_returns.unfold(0, window, 1)  # (n_windows, window)
    means = windows.mean(dim=1)
    if downside:
        downside_sq = windows.clamp(max=0.0) ** 2
        spread = torch.sqrt(downside_sq.mean(dim=1) + eps)
    else:
        spread = windows.std(dim=1) + eps
    ratios = means / spread * (periods_per_year ** 0.5)
    return ratios.mean()


def kelly_loss(portfolio_returns: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Negative expected log-wealth growth rate (the Kelly criterion):
    -E[log(1 + portfolio_return)].

    Unlike Sharpe/Sortino (ratios, not convex or concave in general), this
    is a genuinely CONCAVE function of the weights - log(1 + affine
    function of weights) is concave, and expectation preserves concavity -
    so maximizing expected log-wealth is a convex minimization problem in
    weight-space (on top of whatever the LSTM itself contributes). It
    directly targets long-run COMPOUNDED growth rather than a single-period
    mean/std ratio: two return pa ths with identical mean and std can
    compound to very different wealth outcomes if one has fatter tails, and
    log-wealth penalizes that automatically, with no extra tunable weight.

    `portfolio_returns` is treated as an (approximately arithmetic) period
    return, consistent with how the rest of this module already treats the
    weights-dot-log-returns quantity (see e.g. apply_transaction_costs).
    Clamped at `eps` before taking the log so a very bad, highly levered day
    (1 + return <= 0) can't produce a NaN/-inf gradient - the clamp is a
    numerical safety net, not expected to bind for a well-behaved,
    vol-targeted portfolio.
    """
    return -torch.log(torch.clamp(1.0 + portfolio_returns, min=eps)).mean()


def cvar(portfolio_returns: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
    """Empirical Conditional Value-at-Risk (a.k.a. Expected Shortfall) of
    `portfolio_returns` at confidence level `alpha`: the average LOSS on the
    worst `1 - alpha` fraction of days (e.g. alpha=0.95 -> the average of
    the worst 5% of days).

    CVaR is a well-known CONVEX risk measure (Rockafellar & Uryasev, 2000) -
    unlike variance, it only penalizes the tail that actually hurts (the
    downside), and unlike max drawdown it stays convex/differentiable. This
    uses the standard empirical estimator (mean of the k worst realized
    losses, via topk) rather than the auxiliary-variable LP formulation
    Rockafellar & Uryasev derive for classical convex solvers - equivalent
    for a fixed empirical sample, and differentiable via topk's gradient
    (flows through whichever k days are currently the worst).
    """
    losses = -portfolio_returns
    n = losses.shape[0]
    k = max(1, int(round((1.0 - alpha) * n)))
    worst_losses, _ = torch.topk(losses, k)
    return worst_losses.mean()


def mean_cvar_loss(
    portfolio_returns: torch.Tensor, alpha: float = 0.95, kappa: float = 1.0, periods_per_year: int = 252,
) -> torch.Tensor:
    """Mean-CVaR training loss: -annualized_mean(returns) + kappa * annualized_CVaR_alpha(returns).

    This is the textbook convex portfolio objective (Rockafellar & Uryasev,
    2000): both `-mean` (linear/affine) and `CVaR` (convex) are convex in
    the weights, and a nonnegative linear combination of convex functions is
    convex, so this whole loss is convex in weight-space - unlike a RATIO
    of return to risk (Sharpe, Sortino, or a naive mean/CVaR ratio), which
    is not convex/concave in general. The tradeoff for that guarantee is
    `kappa`: a FIXED (not learned) risk-aversion weight balancing "how much
    expected return is worth giving up to reduce tail risk" - there is no
    way to combine two different quantities (return, tail risk) into one
    strictly convex scalar without some such weight; `kappa` plays the same
    structural role as mean-variance's lambda, just applied to CVaR
    (tail risk) instead of variance (symmetric risk).

    Both terms are annualized the SAME way sharpe_ratio() annualizes mean
    and std (mean scaled by `periods_per_year`, the spread-like CVaR term
    by its square root) before being combined - not just cosmetic scaling.
    On raw daily FX returns, the mean is tiny (a fraction of a percent) while
    CVaR (average of the worst days) is roughly the same order of magnitude
    as daily volatility - 10-30x larger in practice - so combining them
    UNANNUALIZED with kappa=1.0 made the loss almost entirely about
    minimizing CVaR, with the mean term contributing well under 10% of the
    gradient signal: the model barely learned to seek return at all.
    Annualizing first brings both terms to a comparable order of magnitude
    (a plausible annual return vs. a plausible annual tail-loss figure), so
    kappa=1.0 is a genuinely balanced default instead of one that
    accidentally starves the return term.
    """
    annualized_mean = portfolio_returns.mean() * periods_per_year
    annualized_cvar = cvar(portfolio_returns, alpha=alpha) * (periods_per_year ** 0.5)
    return -annualized_mean + kappa * annualized_cvar


def compute_training_loss(
    portfolio_returns: torch.Tensor,
    objective: str,
    sharpe_window: int = 60,
    cvar_alpha: float = 0.95,
    cvar_kappa: float = 1.0,
) -> torch.Tensor:
    """Dispatch to whichever training objective `objective` selects - the
    single place train_portfolio_model/train_joint_model compute their loss,
    so all three objectives are available to both.

    - "sharpe": -rolling_window_ratio(..., downside=True) - the default;
      a rolling-window, downside-risk-adjusted Sortino-style ratio (see
      rolling_window_ratio). Not convex, but a ratio directly comparable to
      the Sharpe numbers reported everywhere else in the app.
    - "kelly": kelly_loss(...) - maximize expected log-wealth growth.
      Convex, no extra tunable weight.
    - "cvar": mean_cvar_loss(...) - maximize mean return net of a tail-risk
      (CVaR) penalty at a fixed weight. Convex, but needs `cvar_kappa`.
    """
    if objective == "sharpe":
        return -rolling_window_ratio(portfolio_returns, window=sharpe_window, downside=True)
    if objective == "kelly":
        return kelly_loss(portfolio_returns)
    if objective == "cvar":
        return mean_cvar_loss(portfolio_returns, alpha=cvar_alpha, kappa=cvar_kappa)
    raise ValueError(f"Unknown objective: {objective!r} (expected 'sharpe', 'kelly', or 'cvar')")


# --------------------------------------------------------------------------
# Volatility targeting: rescale PortfolioLSTM's raw weights to a target vol
# --------------------------------------------------------------------------

def estimate_covariance(
    window_returns: torch.Tensor, estimator: str = "sample", ewma_lambda: float = 0.94,
) -> torch.Tensor:
    """Estimate each sample's (n_assets, n_assets) return covariance matrix
    from its own `window_returns` (batch, lookback, n_assets) window, under
    one of three estimators:

    "sample" (the original/default): every day in the window weighted
    equally - simple, but on a short window (the default lookback is 30
    days) a handful of coincidentally-calm or coincidentally-correlated
    days can swing the whole estimate, and with lookback <= n_assets it can
    even be singular. This is the estimator portfolio_volatility always
    used before covariance_estimator existed.

    "ewma" (RiskMetrics-style exponentially-weighted covariance):
    more recent days weighted more heavily via geometric decay
    `ewma_lambda` (0.94 is the RiskMetrics daily default - a ~11-day
    half-life), so the estimate reacts faster to a genuine regime change
    (a real vol spike) while still using the whole window for stability -
    a strict improvement over equal-weighting for a quantity (volatility)
    that's well known to cluster/decay over time.

    "ledoit_wolf" (Ledoit & Wolf, 2004 - shrinkage toward scaled identity):
    blends the sample covariance with a well-conditioned target
    (`mu * I`, where mu is the average sample variance) using the
    analytically-derived optimal shrinkage intensity, rather than a fixed
    blend weight. Directly addresses the sample estimator's worst failure
    mode for vol-targeting (see scale_weights_to_target_vol's max_leverage
    docstring): a short window's sample covariance can be near-singular
    (spuriously tiny estimated risk) purely by chance, producing an
    enormous leverage scale-up right before a normal-sized move wipes out
    the position. Shrinkage inflates small eigenvalues toward the target,
    so the estimate is never THAT close to singular.

    Returns: (batch, n_assets, n_assets).
    """
    if estimator not in ("sample", "ewma", "ledoit_wolf"):
        raise ValueError(f"Unknown covariance_estimator: {estimator!r} (expected 'sample', 'ewma', or 'ledoit_wolf')")

    lookback = window_returns.shape[1]

    if estimator == "ewma":
        # Weight day t (0 = oldest, lookback-1 = most recent) by
        # (1 - lambda) * lambda^(lookback-1-t), then renormalize to sum to
        # exactly 1 (a finite window truncates the infinite geometric
        # series, so the raw weights alone sum to slightly under 1).
        age = torch.arange(lookback - 1, -1, -1, dtype=window_returns.dtype, device=window_returns.device)
        raw_w = (1.0 - ewma_lambda) * (ewma_lambda ** age)
        w = (raw_w / raw_w.sum()).view(1, lookback, 1)  # (1, lookback, 1) - broadcasts over batch/assets
        mean = (window_returns * w).sum(dim=1, keepdim=True)
        centered = window_returns - mean
        # Weighted outer product sum: (batch, n_assets, n_assets).
        return torch.einsum("bti,btj->bij", centered * w, centered)

    # "sample" and "ledoit_wolf" both start from the plain sample covariance.
    centered = window_returns - window_returns.mean(dim=1, keepdim=True)  # (batch, lookback, n_assets)
    sample_cov = torch.einsum("bti,btj->bij", centered, centered) / max(lookback - 1, 1)

    if estimator == "sample":
        return sample_cov

    # "ledoit_wolf": shrink toward F = mu * I, mu = average sample variance,
    # with the analytically optimal shrinkage intensity (Ledoit & Wolf,
    # 2004, eq. 14): delta* = clamp(pi_hat / gamma_hat, 0, 1), where
    # pi_hat estimates the total variance of the sample covariance's own
    # entries (how noisy the estimate itself is) and gamma_hat is the
    # squared Frobenius distance between the sample covariance and the
    # target (how much bias shrinking toward that target would introduce).
    n_assets = window_returns.shape[-1]
    mu = torch.diagonal(sample_cov, dim1=-2, dim2=-1).mean(dim=-1)  # (batch,)
    eye = torch.eye(n_assets, dtype=window_returns.dtype, device=window_returns.device)
    target = mu.view(-1, 1, 1) * eye  # (batch, n_assets, n_assets)

    # Per-day outer products x_t x_t' (batch, lookback, n_assets, n_assets),
    # each an unbiased-in-expectation single-day covariance estimate;
    # pi_hat is how much these vary sample-to-sample around their average
    # (the sample covariance itself).
    outer = torch.einsum("bti,btj->btij", centered, centered)
    deviation = outer - sample_cov.unsqueeze(1)  # (batch, lookback, n_assets, n_assets)
    pi_hat = (deviation ** 2).sum(dim=(-1, -2)).mean(dim=1) / max(lookback - 1, 1)  # (batch,)
    gamma_hat = ((sample_cov - target) ** 2).sum(dim=(-1, -2))  # (batch,)
    delta = (pi_hat / gamma_hat.clamp(min=1e-12)).clamp(min=0.0, max=1.0)  # (batch,)

    delta = delta.view(-1, 1, 1)
    return delta * target + (1.0 - delta) * sample_cov


def portfolio_volatility(
    window_returns: torch.Tensor,
    weights: torch.Tensor,
    periods_per_year: int = 252,
    covariance_estimator: str = "sample",
    ewma_lambda: float = 0.94,
) -> torch.Tensor:
    """Estimate each sample's ANNUALIZED portfolio volatility from the
    estimated covariance (see estimate_covariance - "sample", "ewma", or
    "ledoit_wolf") of every asset's log returns over `window_returns` (the
    same RAW, real-scale lookback window the weights were computed from)
    and the proposed `weights`.

    window_returns: (batch, lookback, n_assets), real (unstandardized) log
    returns - real units are required since this gets compared against a
    real target like 20% annualized volatility.
    weights: (batch, n_assets).
    Returns: (batch,) - annualized volatility of dot(weights, daily_return)
    under each sample's own window covariance estimate (w^T . Sigma . w).
    """
    cov = estimate_covariance(window_returns, covariance_estimator, ewma_lambda)
    portfolio_var = torch.einsum("bi,bij,bj->b", weights, cov, weights).clamp(min=1e-12)
    return torch.sqrt(portfolio_var) * (periods_per_year ** 0.5)


def scale_weights_to_target_vol(
    weights: torch.Tensor,
    window_returns: torch.Tensor,
    target_vol: float,
    periods_per_year: int = 252,
    max_leverage: float = 10.0,
    covariance_estimator: str = "sample",
    ewma_lambda: float = 0.94,
) -> torch.Tensor:
    """Rescale `weights` uniformly per sample so the resulting portfolio's
    estimated annualized volatility (see portfolio_volatility) matches
    `target_vol` - standard volatility targeting / risk-parity-style
    leverage, applied right after PortfolioLSTM's own weight computation.

    FX portfolios are typically low-vol day to day, so hitting a target
    like 20% annualized usually means scaling weights UP - sum(weights)
    can end up above 1 (real leverage). That's the intended effect of
    vol-targeting, not a bug: everything downstream (training objective,
    reported PnL, the risk overlay) operates on these already-scaled
    weights, and nothing scales them again afterwards.

    `max_leverage` caps the scale factor itself (not just a clamp deep
    inside portfolio_volatility): whenever a sample's window happens to
    look coincidentally calm, the ESTIMATED covariance can be tiny, and
    dividing target_vol by a near-zero estimate produces an enormous scale
    - confirmed to reach 1000x+ on realistic synthetic data. That one
    sample can then realize a catastrophic (<-100%) return the moment the
    actual next-day move isn't equally calm. This hurts every training
    objective's numerical stability, but is especially severe for
    log-wealth (Kelly): log(1 + return) has a singularity at return = -1,
    and the safety clamp used there produces an exact ZERO gradient for
    any return at or below it - the model gets no corrective signal from
    precisely the sample that most needs one. Capping leverage at a sane
    multiple keeps every sample's worst-case return bounded and finite,
    fixing that dead-gradient failure mode at its source rather than
    papering over the symptom in each loss function individually.
    """
    if target_vol == 0:
        return weights

    vol = portfolio_volatility(
        window_returns, weights, periods_per_year, covariance_estimator, ewma_lambda,
    ).clamp(min=1e-8)
    scale = (target_vol / vol).clamp(max=max_leverage)
    return weights * scale.unsqueeze(-1)


# --------------------------------------------------------------------------
# Benchmark: inverse-volatility (risk-weighted) portfolio
# --------------------------------------------------------------------------

def inverse_vol_benchmark_returns(window_returns: np.ndarray, next_returns: np.ndarray) -> np.ndarray:
    """A simple, UN-LEARNED "risk-weighted" benchmark allocator: every day,
    weight each asset inversely to its own trailing realized volatility
    over that SAME day's lookback window, normalized so weights sum to 1 -
    the classic inverse-volatility / naive risk-parity portfolio (no
    learning, no target-vol leverage, no transaction-cost awareness - just
    "hold less of whatever's been noisier lately").

    window_returns: (n_days, lookback, n_assets), the SAME raw per-sample
    windows the model itself saw for each of those days (so the benchmark
    uses exactly as much information as the model did, no more/less).
    next_returns: (n_days, n_assets) - the realized next-day return
    actually applied on each of those days.

    Returns: (n_days,) - this benchmark's own raw (unscaled) daily
    returns. Callers comparing it against a model's PnL should first
    rescale it to the model's own realized volatility - see
    vol_match_benchmark - since a plain risk-parity book runs at whatever
    volatility the market happens to produce, not any particular target.
    """
    if len(window_returns) == 0:
        return np.zeros(0, dtype=np.float64)
    std = window_returns.std(axis=1) + 1e-8  # (n_days, n_assets)
    inv_vol = 1.0 / std
    weights = inv_vol / inv_vol.sum(axis=1, keepdims=True)
    return (weights * next_returns).sum(axis=1)


def vol_match_benchmark(
    benchmark_returns: np.ndarray, model_returns: np.ndarray, periods_per_year: int = 252,
) -> np.ndarray:
    """Rescale `benchmark_returns` by ONE constant multiplier (uniform
    across the whole period, not day-by-day) so its realized annualized
    volatility matches `model_returns`' own - so a benchmark-vs-model PnL
    comparison isolates "did the LEARNED allocation add value" from "did
    the model just happen to run hotter or colder than the benchmark this
    period", which would otherwise be an apples-to-oranges comparison (the
    model is actively vol-targeted; a plain risk-parity book is not).
    Matched SEPARATELY per split (train/val/test), since realized
    volatility can differ a lot between them.
    """
    if len(benchmark_returns) == 0:
        return benchmark_returns
    benchmark_vol = benchmark_returns.std() * (periods_per_year ** 0.5)
    model_vol = model_returns.std() * (periods_per_year ** 0.5)
    if benchmark_vol < 1e-12:
        return benchmark_returns
    return benchmark_returns * (model_vol / benchmark_vol)


# --------------------------------------------------------------------------
# Risk-parity baseline allocation (see PortfolioLSTM's own docstring)
# --------------------------------------------------------------------------

def risk_parity_weights(
    window_returns: torch.Tensor,
    covariance_estimator: str = "sample",
    ewma_lambda: float = 0.94,
    n_iter: int = 20,
    eps: float = 1e-8,
) -> torch.Tensor:
    """TRUE, covariance-based risk-parity weights: the long-only portfolio
    where every asset contributes EQUALLY to total portfolio risk (equal
    risk contribution / ERC) - not merely marginal inverse-volatility
    weighting, which ignores how assets move together.

    The covariance matrix is estimated from `window_returns` via
    estimate_covariance, using the SAME `covariance_estimator`/
    `ewma_lambda` choice (and the same window) already used for volatility
    targeting elsewhere (see scale_weights_to_target_vol) - one consistent
    view of "current risk" drives both how much of each asset to hold
    (this function) and how much overall leverage to apply (target-vol
    scaling), rather than two independently-configured risk estimates.

    Solved via cyclical coordinate descent (Griveau-Billiotte, Richard &
    Roncalli, 2013) on Spinu's (2013) convex log-barrier reformulation of
    the ERC problem:
        minimize_{w > 0}  0.5 w^T Sigma w  -  (1/N) * sum_i log(w_i)
    whose minimizer, renormalized to sum to 1, IS the ERC portfolio -
    at that minimum, every asset's contribution to portfolio variance
    (w_i * (Sigma w)_i) is equal, by construction of the log-barrier's
    first-order condition. Each coordinate update (holding every other
    w_j fixed) has a closed-form solution - the positive root of a
    quadratic in w_i - so no matrix inversion or line search is needed,
    only `n_iter` sweeps over the (typically small, a handful of FX pairs)
    N assets; this converges to high precision within ~20 sweeps for a
    well-conditioned Sigma, and is cheap enough to re-solve for every
    sample, every epoch.

    window_returns: (batch, lookback, n_assets) raw (unstandardized) log
    returns - the SAME raw window used for volatility targeting, never a
    carry/vol-normalized channel.
    Returns: (batch, n_assets), non-negative, summing to 1.
    """
    cov = estimate_covariance(window_returns, covariance_estimator, ewma_lambda)  # (batch, n_assets, n_assets)
    batch, n_assets, _ = cov.shape
    diag = torch.diagonal(cov, dim1=-2, dim2=-1).clamp(min=eps)  # (batch, n_assets) - each asset's own variance

    # w[i] is a (batch,) tensor per asset; reassigned (never mutated
    # in-place) each coordinate update, so this stays autograd-safe over
    # the unrolled sweeps.
    w = [torch.full((batch,), 1.0 / n_assets, dtype=cov.dtype, device=cov.device) for _ in range(n_assets)]
    for _ in range(n_iter):
        for i in range(n_assets):
            w_vec = torch.stack(w, dim=-1)  # (batch, n_assets) - latest value of every coordinate
            sigma_w_i = torch.einsum("bj,bj->b", cov[:, i, :], w_vec)  # (Sigma w)_i under the CURRENT w
            b_i = sigma_w_i - cov[:, i, i] * w[i]  # sum_{j != i} Sigma[i,j] * w_j (drop asset i's own term)
            a_i = diag[:, i]
            # Positive root of: a_i * w_i^2 + b_i * w_i - 1/n_assets = 0
            # (the log-barrier's first-order condition for coordinate i).
            w[i] = ((-b_i + torch.sqrt(b_i ** 2 + 4 * a_i / n_assets)) / (2 * a_i)).clamp(min=eps)

    w_vec = torch.stack(w, dim=-1)
    return w_vec / w_vec.sum(dim=-1, keepdim=True)


def precompute_risk_parity_baseline(
    window_returns: torch.Tensor,
    covariance_estimator: str = "sample",
    ewma_lambda: float = 0.94,
) -> torch.Tensor:
    """Solve risk_parity_weights ONCE for every window in `window_returns`
    and return the result detached, as a fixed constant to reuse for the
    rest of training/evaluation.

    risk_parity_weights depends only on raw (unstandardized) return
    windows - never on model parameters - so, unlike the model's own
    logits, it is the SAME value every epoch for a given sample. Calling
    it from inside forward() (as an earlier version of this code did)
    re-ran its ~20-sweep coordinate-descent solve for every sample, every
    epoch, for an answer that never changed - pure wasted compute (and
    wasted autograd-graph bookkeeping, even though the result never needed
    gradients in the first place, since it doesn't depend on any learned
    parameter). Calling this once per split (train/val/test), before the
    epoch loop starts, and reusing the result is exactly equivalent and
    orders of magnitude cheaper.

    window_returns: (n, lookback, n_assets) - every window in a split
    (n = however many samples that split has), not just one batch.
    Returns: (n, n_assets), detached, no_grad.
    """
    with torch.no_grad():
        return risk_parity_weights(window_returns, covariance_estimator, ewma_lambda)


# --------------------------------------------------------------------------
# Transaction costs
# --------------------------------------------------------------------------

def apply_transaction_costs(
    weights: np.ndarray, returns: np.ndarray, transaction_cost_bps: float
) -> np.ndarray:
    """Subtract an estimated transaction-cost drag from a realized return
    series, based on day-to-day TURNOVER in `weights` - the standard way to
    approximate the real-world cost of actually executing a strategy's
    rebalancing (spread, slippage, commissions).

    Numpy version, used for REPORTING (plots/API responses) once weights
    are already fixed numbers. See apply_transaction_costs_torch for the
    differentiable version used INSIDE training.

    weights: (n_days, n_assets) - the FINAL weights (e.g. risk-attenuated)
    the `returns` series was realized from.
    returns: (n_days,) - realized portfolio returns for the same days.
    transaction_cost_bps: cost per unit of turnover, in basis points (1 bps
    = 0.0001 = 0.01% of notional). E.g. 5.0 means fully rebuilding the book
    from flat costs 0.05% of NAV.

    turnover_t = sum(|weights_t - weights_{t-1}|) - how much of the book
    was bought/sold moving from yesterday's position to today's. The very
    first day charges sum(|weights_0|), the cost of putting the initial
    position on from flat.
    """
    turnover = np.empty(len(weights), dtype=weights.dtype)
    turnover[0] = np.abs(weights[0]).sum()
    turnover[1:] = np.abs(np.diff(weights, axis=0)).sum(axis=1)
    cost = (transaction_cost_bps / 10_000) * turnover
    return returns - cost


def apply_transaction_costs_torch(
    weights: torch.Tensor, returns: torch.Tensor, transaction_cost_bps: float
) -> torch.Tensor:
    """Differentiable twin of apply_transaction_costs, for use INSIDE the
    training loop (see `transaction_cost` in train_portfolio_model/
    train_joint_model/train_risk_model) rather than only as a post-hoc
    reporting adjustment.

    Training the Sharpe objective directly on these net-of-cost returns
    (instead of gross returns) makes the model favor smoother weight paths
    over abrupt day-to-day swings whenever `transaction_cost_bps` > 0 -
    without adding any extra tunable penalty term: it reuses the same
    turnover-based cost that was already being reported, just moved inside
    the loss instead of applied only afterwards. `transaction_cost_bps=0`
    (the default) reproduces the original gross-return objective exactly.

    weights: (n_days, n_assets), requires_grad-tracked - the FINAL weights
    (post vol-targeting and, if present, risk attenuation) for the whole
    training period this epoch.
    returns: (n_days,) - the (differentiable) gross portfolio returns for
    the same days.
    """
    if transaction_cost_bps <= 0:
        return returns
    turnover_first = weights[:1].abs().sum(dim=-1)          # (1,) - cost of the initial position from flat
    turnover_rest = (weights[1:] - weights[:-1]).abs().sum(dim=-1)  # (n_days - 1,)
    turnover = torch.cat([turnover_first, turnover_rest], dim=0)    # (n_days,)
    cost = (transaction_cost_bps / 10_000) * turnover
    return returns - cost


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train_portfolio_model(
    model: nn.Module,
    X_train: torch.Tensor,
    X_train_raw: torch.Tensor,
    next_returns_train: torch.Tensor,
    target_vol: float,
    epochs: int,
    lr: float,
    weight_decay: float = 0.0,
    noise_std: float = 0.0,
    transaction_cost: float = 0.0,
    max_leverage: float = 10.0,
    objective: str = "sharpe",
    sharpe_window: int = 60,
    cvar_alpha: float = 0.95,
    cvar_kappa: float = 1.0,
    covariance_estimator: str = "sample",
    ewma_lambda: float = 0.94,
    X_val: torch.Tensor | None = None,
    X_val_raw: torch.Tensor | None = None,
    next_returns_val: torch.Tensor | None = None,
    X_test: torch.Tensor | None = None,
    X_test_raw: torch.Tensor | None = None,
    next_returns_test: torch.Tensor | None = None,
) -> None:
    """Full-batch training: at every epoch, allocate weights for the WHOLE
    training period at once, rescale them to --target-vol (see
    scale_weights_to_target_vol), compute the resulting return series, and
    take a gradient step on `objective`'s loss (see compute_training_loss -
    "sharpe": a rolling-window, downside-risk-adjusted ratio; "kelly":
    negative expected log-wealth growth, convex; "cvar": mean return net of
    a fixed-weight CVaR tail-risk penalty, convex). The default ("sharpe")
    scores the model across many overlapping `sharpe_window`-day
    sub-periods of training history instead of one ratio over the whole
    period, so it can't just find one weight pattern that happens to nail
    the exact historical sequence.

    This is full-batch rather than mini-batch on purpose: these ratios are
    statistics of a return distribution (mean/std), so they can only be
    evaluated meaningfully over a full set of returns, not one sample at a
    time as with a per-sample loss like MSE.

    `noise_std` > 0 adds fresh Gaussian noise to the (already standardized)
    input window on every epoch - the model sees a slightly different
    version of the training data each time, so it can't memorize the exact
    training sequence, only patterns that survive small perturbations of
    it. Note the vol-targeting scale is computed from the CLEAN raw window
    (`X_train_raw`), not the noised one - noise regularizes what the LSTM
    sees, it shouldn't distort the real volatility estimate. `weight_decay`
    is passed straight through to Adam as an L2 penalty.

    `transaction_cost` > 0 (basis points per unit of turnover - the same
    knob apply_transaction_costs uses for reporting) subtracts a turnover-
    based cost from the return series BEFORE computing the training ratio
    (see apply_transaction_costs_torch), so the model is trained to
    maximize its objective net of trading costs directly - it learns to
    avoid abrupt, expensive day-to-day weight swings whenever they're not
    worth their cost, rather than being scored on gross returns and only
    having costs applied afterwards as a reporting adjustment.

    If `X_val`/`X_val_raw`/`next_returns_val` are given, the model is
    scored on genuine held-out validation Sharpe (the plain whole-period
    metric, matching what's reported elsewhere) after every epoch, and the
    state_dict from whichever epoch had the BEST validation Sharpe is
    restored at the end - not just whichever epoch the fixed `epochs`
    budget happened to end on. This directly targets an in-sample-great,
    out-of-sample-poor training curve: rather than trusting the last epoch,
    the run keeps the point that actually generalized best. This is a
    genuine use of validation data (an early-stopping / checkpoint-
    selection criterion, not part of the loss/gradient itself), the
    standard reason a validation split exists.

    If `X_test`/`X_test_raw`/`next_returns_test` are ALSO given, test
    returns are computed every epoch too - but PURELY for live reporting
    (see _report_epoch/the Training view's third PnL chart), never for
    checkpoint selection: best_state is chosen by validation Sharpe alone,
    exactly as without a test set. This is the whole point of a separate
    test split (see PortfolioResult's docstring) - it stays a genuinely
    unbiased read on generalization, never influencing any decision the
    way validation does.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    track_val = X_val is not None and X_val_raw is not None and next_returns_val is not None
    track_test = X_test is not None and X_test_raw is not None and next_returns_test is not None
    best_val_sharpe = -float("inf")
    best_state: dict | None = None

    # Risk-parity baseline depends only on raw returns, never on model
    # parameters, so it's the SAME value every epoch - solve it ONCE per
    # split here (see precompute_risk_parity_baseline) instead of
    # re-running its ~20-sweep coordinate-descent solve inside
    # forward_sequence on every single epoch.
    train_baseline = precompute_risk_parity_baseline(X_train_raw, covariance_estimator, ewma_lambda)
    val_baseline = (
        precompute_risk_parity_baseline(X_val_raw, covariance_estimator, ewma_lambda) if track_val else None
    )
    test_baseline = (
        precompute_risk_parity_baseline(X_test_raw, covariance_estimator, ewma_lambda) if track_test else None
    )

    model.train()
    try:
        for epoch in range(1, epochs + 1):
            _check_stop()
            optimizer.zero_grad()
            noisy_X_train = X_train + torch.randn_like(X_train) * noise_std if noise_std > 0 else X_train
            # risk_parity_baseline=train_baseline (precomputed from the
            # CLEAN window, not noisy_X_train) - same reasoning as
            # vol-targeting already using the clean window, not the noised
            # one, for its own volatility estimate.
            raw_weights = model.forward_sequence(
                noisy_X_train, risk_parity_baseline=train_baseline,
            )    # (n_train, n_assets)
            weights = scale_weights_to_target_vol(
                raw_weights, X_train_raw, target_vol, max_leverage=max_leverage,
                covariance_estimator=covariance_estimator, ewma_lambda=ewma_lambda,
            )
            portfolio_returns = (weights * next_returns_train).sum(dim=-1)  # (n_train,)
            portfolio_returns = apply_transaction_costs_torch(weights, portfolio_returns, transaction_cost)
            loss = compute_training_loss(
                portfolio_returns, objective, sharpe_window=sharpe_window, cvar_alpha=cvar_alpha, cvar_kappa=cvar_kappa,
            )
            loss.backward()
            optimizer.step()

            if track_val:
                model.eval()
                with torch.no_grad():
                    # Continue the position from train's own last decided
                    # weight (this epoch) rather than restarting flat at
                    # the train/val boundary - real portfolios don't reset.
                    val_initial = weights[-1].detach() if model.use_prev_weight else None
                    val_weights = scale_weights_to_target_vol(
                        model.forward_sequence(
                            X_val, initial_weight=val_initial, risk_parity_baseline=val_baseline,
                        ), X_val_raw, target_vol,
                        max_leverage=max_leverage,
                        covariance_estimator=covariance_estimator, ewma_lambda=ewma_lambda,
                    )
                    val_returns = (val_weights * next_returns_val).sum(dim=-1)
                    val_returns = apply_transaction_costs_torch(val_weights, val_returns, transaction_cost)
                    val_sharpe = float(sharpe_ratio(val_returns))

                    test_returns = None
                    if track_test:
                        # Continue from val's own last decided weight, same
                        # reasoning as train->val above. Purely for live
                        # display - see track_test's docstring note above -
                        # never touches best_state/checkpoint selection.
                        test_initial = val_weights[-1].detach() if model.use_prev_weight and val_weights.shape[0] > 0 else val_initial
                        test_weights = scale_weights_to_target_vol(
                            model.forward_sequence(
                                X_test, initial_weight=test_initial, risk_parity_baseline=test_baseline,
                            ), X_test_raw, target_vol,
                            max_leverage=max_leverage,
                            covariance_estimator=covariance_estimator, ewma_lambda=ewma_lambda,
                        )
                        test_returns = (test_weights * next_returns_test).sum(dim=-1)
                        test_returns = apply_transaction_costs_torch(test_weights, test_returns, transaction_cost)
                model.train()
                if val_sharpe > best_val_sharpe:
                    best_val_sharpe = val_sharpe
                    best_state = copy.deepcopy(model.state_dict())

            if epoch == 1 or epoch % max(epochs // 10, 1) == 0:
                val_msg = f" | val Sharpe {val_sharpe:.4f}" if track_val else ""
                logger.info(
                    "epoch %d/%d - train loss (%s, net of costs) %.4f%s",
                    epoch, epochs, objective, loss.item(), val_msg,
                )
                _report_epoch(
                    "portfolio", epoch, epochs, portfolio_returns,
                    val_returns if track_val else None,
                    # test tracking piggybacks on val's continuation chain
                    # (test_initial is derived from val_weights above), so
                    # it's only computed/reported when val tracking is ALSO on.
                    test_returns if track_val and track_test else None,
                )
    except TrainingStopped:
        logger.info("Training stopped early at epoch %d/%d", epoch, epochs)
        if track_val and best_state is not None:
            model.load_state_dict(best_state)
            logger.info("Restored best-validation-Sharpe checkpoint (val Sharpe %.4f)", best_val_sharpe)
        raise

    if track_val and best_state is not None:
        model.load_state_dict(best_state)
        logger.info("Restored best-validation-Sharpe checkpoint (val Sharpe %.4f)", best_val_sharpe)


# --------------------------------------------------------------------------
# End-to-end pipeline: shared by this script's CLI and models/portfolio_postprocess.py
# --------------------------------------------------------------------------

@dataclass
class PortfolioResult:
    """Everything needed to score and plot a trained portfolio allocator.

    Also carries the standardized input tensors and raw next-day returns
    (X_train/X_val, next_returns_train/next_returns_val) so a downstream
    model - e.g. models/risk_lstm.py's risk overlay - can reuse the exact
    same data and train/validation split without loading and re-slicing
    everything a second time.

    `weights_train`/`weights_val` and `returns_train`/`returns_val` are the
    FINAL, volatility-targeted weights/returns (PortfolioLSTM's raw output
    rescaled to `target_vol` via scale_weights_to_target_vol) - this is
    what's used for training's Sharpe objective, what the risk overlay
    attenuates, and what gets reported everywhere. `weights_train_unscaled`
    /`weights_val_unscaled` and `returns_train_unscaled`/`returns_val_unscaled`
    keep PortfolioLSTM's ORIGINAL (pre-vol-targeting) weights/returns around
    too, purely for comparison (e.g. the raw-vs-vol-targeted-vs-attenuated
    PnL plot in models/risk_postprocess.py).
    """

    model: PortfolioLSTM
    pairs: list[str]
    lookback: int
    target_vol: float
    dates_train: pd.DatetimeIndex
    dates_val: pd.DatetimeIndex
    returns_train: np.ndarray   # realized daily portfolio log returns, in-sample (vol-targeted)
    returns_val: np.ndarray     # realized daily portfolio log returns, out-of-sample (vol-targeted)
    weights_train: np.ndarray   # (n_train, n_assets), vol-targeted
    weights_val: np.ndarray     # (n_val, n_assets), vol-targeted
    returns_train_unscaled: np.ndarray  # pre-vol-targeting, in-sample
    returns_val_unscaled: np.ndarray    # pre-vol-targeting, out-of-sample
    weights_train_unscaled: np.ndarray  # (n_train, n_assets), pre-vol-targeting
    weights_val_unscaled: np.ndarray    # (n_val, n_assets), pre-vol-targeting
    X_train: torch.Tensor       # standardized input windows, in-sample
    X_val: torch.Tensor         # standardized input windows, out-of-sample
    next_returns_train: np.ndarray  # raw log-return scale, (n_train, n_assets)
    next_returns_val: np.ndarray    # raw log-return scale, (n_val, n_assets)
    x_mean: np.ndarray          # standardization stats X was fit/loaded with
    x_std: np.ndarray
    # Test split (see DEFAULT_CONFIG's test_frac) - held out from every
    # training/model-selection decision, unlike validation. Empty arrays
    # (n_test=0) when test_frac=0. Mirrors the train/val fields above.
    dates_test: pd.DatetimeIndex
    returns_test: np.ndarray
    weights_test: np.ndarray
    returns_test_unscaled: np.ndarray
    weights_test_unscaled: np.ndarray
    X_test: torch.Tensor
    next_returns_test: np.ndarray
    # Inverse-volatility (risk-weighted) benchmark - a plain, un-learned
    # allocator (see inverse_vol_benchmark_returns) rescaled to match this
    # model's OWN realized volatility on the SAME period (see
    # vol_match_benchmark) - so the comparison isolates "did the LEARNED
    # allocation add value" from "did the model just run hotter/colder",
    # separately for each split (in-sample and out-of-sample dynamics can
    # differ a lot).
    benchmark_returns_train: np.ndarray
    benchmark_returns_val: np.ndarray
    benchmark_returns_test: np.ndarray
    # Per-asset coefficient the model applied to the risk-parity baseline
    # each day (tanh(logits) if position_mode="long_short", else
    # sigmoid(logits) - see PortfolioLSTM._weights_from_coefficients),
    # i.e. weights_*_unscaled / risk_parity_baseline elementwise: this is
    # the model's own learned "conviction" signal, separate from both the
    # (fixed, un-learned) risk-parity baseline and vol-targeting's overall
    # leverage scale - what the Evaluation view's coefficient chart plots.
    coefficients_train: np.ndarray  # (n_train, n_assets), in (-1, 1) or (0, 1)
    coefficients_val: np.ndarray    # (n_val, n_assets)
    coefficients_test: np.ndarray   # (n_test, n_assets)


#: Default configuration - main.py (the project's single entry point) reads
#: a JSON file and merges it on top of this dict before wrapping the result
#: in an argparse.Namespace, so a JSON config only needs to specify keys
#: that differ from these defaults. "pairs" has no sensible default and
#: must always be provided by the caller.
DEFAULT_CONFIG: dict = {
    # Data
    "pairs": None,  # REQUIRED - e.g. ["EURUSD", "GBPUSD", "USDJPY"]
    "lookback": 30,
    "years": 8,
    "train_frac": 0.8,
    # Chronological 3-way split: the most recent `test_frac` fraction of
    # ALL sequences is carved off FIRST as the test set - held out
    # completely from every training/model-selection decision (unlike
    # validation, which best-epoch checkpoint selection DOES see - see
    # train_portfolio_model's docstring) - so test performance is a genuine
    # unbiased read on generalization, not just "the split that happened to
    # look best." `train_frac` keeps its original meaning applied to
    # whatever remains AFTER carving out test: e.g. train_frac=0.8,
    # test_frac=0.1 -> the oldest 72% of all data is train, the next 18%
    # is validation, and the most recent 10% is test. 0.0 disables the test
    # split entirely (train/val only, the original 2-way behavior).
    "test_frac": 0.1,
    # PortfolioLSTM architecture / training
    # Every asset's final weight is a fixed, non-learned risk-parity
    # (inverse-volatility) baseline multiplied by a per-asset coefficient
    # the network predicts (see PortfolioLSTM's own docstring and
    # risk_parity_weights). "position_mode" selects that coefficient's
    # range: "long_short" (tanh, in (-1,1) - can go short) or "long_only"
    # (sigmoid, in (0,1) - can only scale the baseline down toward flat).
    "position_mode": "long_short",
    "hidden_size": 32,
    "epochs": 300,
    "lr": 1e-3,
    "dropout": 0.1,
    "weight_decay": 1e-4,
    "noise_std": 0.05,
    "target_vol": 0.20,
    # Caps the leverage vol-targeting can impose on any single sample (see
    # scale_weights_to_target_vol) - without this, a window whose ESTIMATED
    # covariance happens to be near zero can get scaled up by 100x-1000x+,
    # risking a catastrophic realized loss the moment the actual next-day
    # move isn't equally calm. This is what makes the "kelly"/"cvar"
    # training objectives numerically safe (log-wealth in particular has a
    # singularity at a -100% return); it also protects "sharpe" from rare
    # but arbitrarily large outlier gradients.
    "max_leverage": 10.0,
    # Parameter-space noise (NoisyNet) on the output head(s) - PortfolioLSTM's
    # head, and BOTH of RiskLSTM's head/head_2 layers when risk_overlay is
    # on - resampled every forward() call during training, mu-only
    # (deterministic) at eval time.
    # Composes with noise_std rather than replacing it: noise_std perturbs
    # the input, this perturbs the model's own decision boundary, so neither
    # alone is enough to memorize one exact Sharpe-maximizing configuration.
    "noisy_head": False,
    # Feed the previous day's OWN decided weight into the head alongside
    # the current window (the Moody & Saffell recurrent-policy trick) - so
    # the model knows what position it already holds and can learn whether
    # a rebalance is worth its transaction cost, rather than deciding each
    # window as if starting flat every time. Pairs naturally with
    # `transaction_cost` > 0. Forces training/evaluation to process the
    # whole split as a genuine sequential recurrence (see
    # PortfolioLSTM.forward_sequence) instead of one parallel batched call,
    # so it's meaningfully slower - off by default.
    "use_prev_weight": False,
    # Adds one extra "CASH" pseudo-pair (constant `cash_return` every day,
    # so it's a zero-variance asset - see _prepare_data) to `pairs`, letting
    # the allocator choose to de-risk by holding cash directly instead of
    # relying solely on the (separate) risk overlay to shrink positions.
    # Enables the ablation: does the risk overlay still add value once the
    # allocator itself can hold cash? cash_return=0.0 means cash earns
    # nothing (the standard, simplest choice; a nonzero short rate can be
    # supplied instead).
    "has_cash": False,
    "cash_return": 0.0,
    # Training objective (see compute_training_loss): which quantity
    # train_portfolio_model/train_joint_model take a gradient step on.
    #   "sharpe" - rolling-window, downside-risk-adjusted ratio (see
    #              rolling_window_ratio) over `sharpe_window`-day
    #              sub-windows of the training period, not one ratio over
    #              the whole period - so it can't just find one weight
    #              pattern that happens to nail the exact historical
    #              sequence. Not convex, but directly comparable to the
    #              Sharpe numbers reported everywhere else in the app.
    #   "kelly"  - negative expected log-wealth growth (Kelly criterion).
    #              Convex, no extra tunable weight.
    #   "cvar"   - mean return net of a fixed-weight CVaR tail-risk penalty
    #              (`cvar_kappa`). Convex, at the cost of that fixed weight.
    "objective": "sharpe",
    # Also reused by RiskLSTM's own training loss (train_risk_model) - it
    # scores the SAME rolling-window ratio, regardless of PortfolioLSTM's
    # own `objective` choice, rather than a single whole-period Sharpe: a
    # single aggregate scalar over the whole training period is a very
    # weak gradient signal for genuinely DAY-TO-DAY attenuation, and
    # empirically converges toward an attenuation that's nearly constant
    # per asset over time - directly undermining the point of a risk
    # overlay that's supposed to react to changing conditions.
    "sharpe_window": 60,
    "cvar_alpha": 0.95,  # confidence level for CVaR - the worst (1 - alpha) fraction of days
    "cvar_kappa": 1.0,   # fixed risk-aversion weight on CVaR in the "cvar" objective
    # Whichever epoch has the best VALIDATION Sharpe (always the plain,
    # whole-period metric, regardless of `objective`) is kept at the end
    # instead of always the last one - see train_portfolio_model's
    # docstring for why.
    # Multi-seed restarts (see run_pipeline_multi_seed)
    "n_seeds": 1,
    "restart_strategy": "best",  # "best" or "ensemble"
    # Risk overlay (see models/risk_lstm.py)
    "risk_overlay": False,
    "risk_hidden_size": 16,
    "risk_epochs": 200,
    "risk_lr": 1e-3,
    "max_attenuation": 0.33,
    "risk_rolling_window": 10,
    # Portfolio-wide cross-sectional risk features (average pairwise
    # correlation, correlation dispersion, top eigenvalue share of the
    # rolling cross-asset correlation matrix - see risk_lstm.py's
    # cross_sectional_features) appended to RiskLSTM's per-timestep input
    # alongside its existing per-asset (marginal) statistics. Off by
    # default - the attenuation head then sees only marginal risk, exactly
    # as before this feature existed.
    "use_cross_sectional": False,
    # Transaction costs, in basis points per unit of turnover - 0 disables
    # them entirely. Applied BOTH inside the training objective (Sharpe is
    # computed net of turnover-based costs - see apply_transaction_costs_torch
    # - so the model learns to avoid abrupt, costly weight swings) and to
    # reported/plotted returns (see apply_transaction_costs).
    "transaction_cost": 0.0,
    # Extra per-asset input feature channels (see build_feature_dataframe).
    # use_carry adds the interest-rate differential (data/rates_downloader.py
    # via db.py's get_time_series); vol_horizons adds, for each horizon (in
    # days), that asset's trailing cumulative-return-over-realized-vol ratio.
    # Both default OFF - the model then sees raw log returns only, exactly
    # as before either feature existed.
    "use_carry": False,
    "vol_horizons": [],
    # Encoder architecture (see PortfolioLSTM's docstring): "concat" (the
    # original design - one LSTM over all assets' concatenated features) or
    # "per_asset" (one shared per-asset LSTM + cross-asset attention/mean-
    # pooling - permutation/universe-size invariant). asset_combiner and
    # n_attn_heads only matter when encoder_type="per_asset".
    "encoder_type": "concat",
    "asset_combiner": "attention",  # "attention" or "mean"
    "n_attn_heads": 2,
    # Covariance estimator for volatility targeting (see estimate_covariance
    # in scale_weights_to_target_vol/portfolio_volatility): "sample" (the
    # original plain equal-weighted covariance over the lookback window),
    # "ewma" (RiskMetrics-style exponentially-weighted - reacts faster to a
    # genuine vol regime change), or "ledoit_wolf" (shrinkage toward a
    # well-conditioned scaled-identity target with the analytically optimal
    # shrinkage intensity - directly addresses the near-singular-covariance
    # failure mode max_leverage was added to cap the SYMPTOM of).
    # ewma_lambda only matters when covariance_estimator="ewma" (0.94 is the
    # RiskMetrics daily default, a ~11-day half-life).
    "covariance_estimator": "sample",
    "ewma_lambda": 0.94,
    # Time pooling (see TemporalAttentionPool): "last" (the original design
    # - use the LSTM's final hidden state, h_n[-1], as the whole window's
    # summary) or "attention" (learned attention pooling over EVERY
    # timestep's hidden state - lets the network learn which days in the
    # window matter most, rather than trusting the last day alone to have
    # retained everything relevant). Applies to both PortfolioLSTM and
    # RiskLSTM.
    "pooling": "last",
    # Compute device (see get_device): "auto" picks Apple Silicon's Metal
    # backend (MPS) if available, else CUDA, else CPU. Both PortfolioLSTM
    # and (if enabled) RiskLSTM are moved to this device for training and
    # evaluation; only their own hot per-epoch forward/backward passes run
    # on it - one-off feature engineering (pandas/numpy) stays on CPU.
    "device": "auto",
    # Persistence: each of load_portfolio/load_risk accepts EITHER a local
    # .pt file path OR a quant.model_registry name (see
    # load_portfolio_model_auto below); save_db additionally persists
    # whatever gets trained/loaded to Postgres under a deterministic name.
    "load_portfolio": None,
    "load_risk": None,
    "save_db": False,
    "model_description": "",
    # Plot output paths (main.py always saves cumulative PnL; the rest are
    # only used when risk_overlay is true).
    "output": "models/portfolio_pnl.png",
    "position_output": "models/risk_position.png",
    "vol_matched_output": "models/risk_vol_matched_pnl.png",
    "histogram_output": "models/risk_return_histogram.png",
    "transaction_cost_output": "models/risk_transaction_cost_pnl.png",
}


def portfolio_model_name(args: argparse.Namespace) -> str:
    """Deterministic quant.model_registry name for a PortfolioLSTM/-ensemble
    trained with `args` - built from the characteristics that actually
    change the trained model (pairs, position mode, lookback, hidden size,
    target vol), so the same configuration always maps to the same name
    and re-saving under it is a natural update.
    """
    from data.model_registry import build_model_name

    is_ensemble = args.n_seeds > 1 and args.restart_strategy == "ensemble"
    return build_model_name(
        "portfolio_ensemble" if is_ensemble else "portfolio",
        pairs=sorted(args.pairs),
        position_mode=getattr(args, "position_mode", "long_short"),
        lookback=args.lookback,
        hidden_size=args.hidden_size,
        target_vol=args.target_vol,
        has_cash=getattr(args, "has_cash", False),
        use_prev_weight=getattr(args, "use_prev_weight", False),
        use_carry=getattr(args, "use_carry", False),
        vol_horizons=sorted(getattr(args, "vol_horizons", []) or []),
        encoder_type=getattr(args, "encoder_type", "concat"),
        asset_combiner=getattr(args, "asset_combiner", "attention"),
        covariance_estimator=getattr(args, "covariance_estimator", "sample"),
        pooling=getattr(args, "pooling", "last"),
    )


def load_portfolio_model_auto(value: str) -> nn.Module:
    """Load a PortfolioLSTM/-ensemble from either a local file path or a
    quant.model_registry name - tries the local file first (so an existing
    "load_portfolio": "<path>.pt" config keeps working unchanged), and
    falls back to the database if no such file exists.
    """
    if os.path.exists(value):
        return load_portfolio_model(value)
    return load_portfolio_model_from_db(value)


@dataclass
class _PreparedData:
    """Data shared across restarts - loaded and split once, not once per seed."""

    pairs: list[str]
    lookback: int
    n_channels: int
    dates_train: pd.DatetimeIndex
    dates_val: pd.DatetimeIndex
    dates_test: pd.DatetimeIndex
    X_train: torch.Tensor
    X_val: torch.Tensor
    X_test: torch.Tensor
    X_train_raw: torch.Tensor   # real (unstandardized) log-return windows - for volatility targeting
    X_val_raw: torch.Tensor
    X_test_raw: torch.Tensor
    next_returns_train_raw: np.ndarray
    next_returns_val_raw: np.ndarray
    next_returns_test_raw: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    device: torch.device   # X_train/X_val/X_test/*_raw already live here - see get_device


def _prepare_data(
    args: argparse.Namespace,
    x_mean: np.ndarray | None = None,
    x_std: np.ndarray | None = None,
    pairs: list[str] | None = None,
    lookback: int | None = None,
    use_carry: bool | None = None,
    vol_horizons: list[int] | None = None,
) -> _PreparedData:
    """Load data (via db.py), build sequences, split by time, and
    standardize - everything a training (or inference) run needs that
    doesn't depend on the model's random seed.

    If `x_mean`/`x_std` are given (loading a previously-saved model), they
    are used as-is instead of being freshly fit - inference must standardize
    new data exactly the way the loaded model was trained, not according to
    whatever training split happens to be in front of it today.

    If `pairs`/`lookback` are given, they OVERRIDE args.pairs/args.lookback
    - used when loading a saved model, whose checkpoint carries the exact
    ordered pair list AND sequence length it was trained on (see
    PortfolioLSTM._from_checkpoint/load_pipeline below). Model weight i
    always corresponds to pairs[i], and the LSTM's learned dynamics assume
    windows of exactly the trained lookback, so a loaded model's own stored
    values must win over whatever a caller (e.g. a UI form) passes in.

    Likewise, `use_carry`/`vol_horizons`, when given (not None), OVERRIDE
    args.use_carry/args.vol_horizons - a loaded model's own stored feature
    configuration (see PortfolioLSTM._from_checkpoint) must win, since the
    LSTM's input width is fixed at training time and would silently
    mismatch data built with a different feature configuration.

    "CASH" (see `args.has_cash`/`args.cash_return`) is added to `pairs`
    (and to `returns`, as a constant-value column) here - AFTER fetching
    real price data for the real pairs only, since CASH has no ticker - so
    it's treated as JUST ANOTHER ASSET everywhere downstream (vol-
    targeting, transaction costs, reporting, per-asset charts) with no
    special-casing needed: a constant daily return gives it exactly zero
    realized variance, which scale_weights_to_target_vol's own covariance
    estimate already handles correctly (a risk-free asset contributes no
    variance, exactly as it should). If `pairs` was passed in already
    containing "CASH" (a loaded model that was trained with has_cash=True),
    that's honored regardless of `args.has_cash` - it's a property of the
    model, not a free evaluation-time choice.
    """
    has_cash = getattr(args, "has_cash", False)
    cash_return = getattr(args, "cash_return", 0.0)

    if pairs is None:
        real_pairs = list(dict.fromkeys(args.pairs))  # de-duplicate, keep order
    else:
        has_cash = has_cash or ("CASH" in pairs)
        real_pairs = [p for p in pairs if p != "CASH"]
    if lookback is None:
        lookback = args.lookback

    logger.info("Loading %s via db.py", real_pairs)
    prices = load_close_prices(real_pairs, years=args.years)
    returns = to_log_returns(prices)

    if has_cash:
        returns = returns.copy()
        returns["CASH"] = cash_return
        pairs = real_pairs + ["CASH"]
    else:
        pairs = real_pairs

    min_rows = lookback + 1  # need `lookback` history days plus 1 realized return to form even one sequence
    if len(returns) < min_rows:
        raise ValueError(
            f"Only {len(returns)} days of history available for {pairs}, but this model needs at least "
            f"{min_rows} (sequence length {lookback} + 1) - increase 'years' to fetch more history."
        )

    if use_carry is None:
        use_carry = getattr(args, "use_carry", False)
    if vol_horizons is None:
        vol_horizons = getattr(args, "vol_horizons", []) or []
    n_channels = 1 + int(use_carry) + len(vol_horizons)
    feature_returns = (
        build_feature_dataframe(returns, pairs, use_carry, vol_horizons, args.years)
        if n_channels > 1 else None
    )

    X, next_returns, dates = make_portfolio_sequences(returns, lookback=lookback, feature_returns=feature_returns)

    # Chronological 3-way split: the most recent `test_frac` fraction is
    # carved off FIRST as the test set (held out from every training/
    # model-selection decision - see PortfolioResult's docstring), then
    # `train_frac` splits whatever REMAINS into train/validation, exactly
    # as before test_frac existed. test_frac=0 (the default off-state for
    # any caller that doesn't set it) reproduces the original 2-way split
    # precisely: n_test=0, n_remaining=len(X).
    test_frac = getattr(args, "test_frac", 0.0) or 0.0
    n_total = len(X)
    n_test = int(n_total * test_frac)
    n_remaining = n_total - n_test
    n_train = int(n_remaining * args.train_frac)

    X_full_train_raw, X_full_val_raw, X_full_test_raw = X[:n_train], X[n_train:n_remaining], X[n_remaining:]
    next_returns_train_raw = next_returns[:n_train]
    next_returns_val_raw = next_returns[n_train:n_remaining]
    next_returns_test_raw = next_returns[n_remaining:]
    dates_train, dates_val, dates_test = dates[:n_train], dates[n_train:n_remaining], dates[n_remaining:]

    # Standardize the LSTM's FULL (possibly multi-channel: return + carry +
    # vol-normalized-return per asset) input features. next_returns stay on
    # the real log-return scale - they're multiplied by the weights to get
    # real portfolio P&L, so they must not be rescaled.
    if x_mean is None or x_std is None:
        x_mean, x_std = standardize(X_full_train_raw, axis=(0, 1))  # TRAIN-only stats

    # Every tensor that feeds the network's own hot per-epoch forward/
    # backward pass (the standardized X_*, and the raw-return X_*_raw used
    # every epoch by vol-targeting) is moved to `device` ONCE here - see
    # get_device. Everything upstream (pandas/numpy feature engineering)
    # stays on CPU; only the repeated matrix-multiply-heavy work benefits
    # from an accelerator, and this is full-batch training (one transfer
    # per split, not one per mini-batch), so there's no per-step overhead.
    device = get_device(getattr(args, "device", "auto"))
    X_train = torch.tensor((X_full_train_raw - x_mean) / x_std, device=device)
    X_val = torch.tensor((X_full_val_raw - x_mean) / x_std, device=device)
    X_test = torch.tensor((X_full_test_raw - x_mean) / x_std, device=device)

    # Volatility targeting (scale_weights_to_target_vol/portfolio_volatility)
    # needs the RAW REAL log-return of each asset only - never carry or a
    # vol-normalized ratio, which aren't returns and would corrupt the
    # covariance estimate. Channel 0 of every asset's block is always the
    # raw return (see build_feature_dataframe's asset-major layout), so
    # slicing every n_channels-th column recovers exactly that.
    X_train_raw = X_full_train_raw[:, :, 0::n_channels]
    X_val_raw = X_full_val_raw[:, :, 0::n_channels]
    X_test_raw = X_full_test_raw[:, :, 0::n_channels]

    return _PreparedData(
        pairs=pairs,
        lookback=lookback,
        n_channels=n_channels,
        dates_train=dates_train,
        dates_val=dates_val,
        dates_test=dates_test,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        X_train_raw=torch.tensor(X_train_raw, device=device),
        X_val_raw=torch.tensor(X_val_raw, device=device),
        X_test_raw=torch.tensor(X_test_raw, device=device),
        next_returns_train_raw=next_returns_train_raw,
        next_returns_val_raw=next_returns_val_raw,
        next_returns_test_raw=next_returns_test_raw,
        x_mean=x_mean,
        x_std=x_std,
        device=device,
    )


def evaluate_portfolio_model(
    model: nn.Module, data: _PreparedData, target_vol: float, max_leverage: float = 10.0,
    covariance_estimator: str = "sample", ewma_lambda: float = 0.94,
) -> PortfolioResult:
    """Run an already-trained-or-loaded model (eval mode, no grad) over all
    three splits (train/validation/test - test is empty when test_frac=0),
    rescale its raw weights to `target_vol` (see scale_weights_to_target_vol),
    and package the result - keeping both the pre-scaling ("unscaled") and
    post-scaling (vol-targeted) weights/returns so callers can compare them,
    plus an inverse-volatility benchmark vol-matched to the model on each
    split. Shared by the train path (after fitting) and the load path
    (skips fitting entirely).
    """
    # Risk-parity baseline depends only on raw returns, never on the
    # model - solve it ONCE per split here (see
    # precompute_risk_parity_baseline) rather than inside forward_sequence.
    train_baseline = precompute_risk_parity_baseline(data.X_train_raw, covariance_estimator, ewma_lambda)
    val_baseline = precompute_risk_parity_baseline(data.X_val_raw, covariance_estimator, ewma_lambda)
    test_baseline = precompute_risk_parity_baseline(data.X_test_raw, covariance_estimator, ewma_lambda)

    model.eval()
    with torch.no_grad():
        raw_weights_train = model.forward_sequence(
            data.X_train, risk_parity_baseline=train_baseline,
        )
        use_prev = getattr(model, "use_prev_weight", False)
        val_initial = raw_weights_train[-1].detach() if use_prev and raw_weights_train.shape[0] > 0 else None
        raw_weights_val = model.forward_sequence(
            data.X_val, initial_weight=val_initial, risk_parity_baseline=val_baseline,
        )
        test_initial = raw_weights_val[-1].detach() if use_prev and raw_weights_val.shape[0] > 0 else val_initial
        raw_weights_test = model.forward_sequence(
            data.X_test, initial_weight=test_initial, risk_parity_baseline=test_baseline,
        )
        weights_train_t = scale_weights_to_target_vol(
            raw_weights_train, data.X_train_raw, target_vol, max_leverage=max_leverage,
            covariance_estimator=covariance_estimator, ewma_lambda=ewma_lambda,
        )
        weights_val_t = scale_weights_to_target_vol(
            raw_weights_val, data.X_val_raw, target_vol, max_leverage=max_leverage,
            covariance_estimator=covariance_estimator, ewma_lambda=ewma_lambda,
        )
        weights_test_t = scale_weights_to_target_vol(
            raw_weights_test, data.X_test_raw, target_vol, max_leverage=max_leverage,
            covariance_estimator=covariance_estimator, ewma_lambda=ewma_lambda,
        )

    # .cpu() before .numpy(): these tensors may live on an accelerator
    # (MPS/CUDA - see get_device); torch's .numpy() only works on CPU
    # tensors, and everything from here on (PnL, benchmark, reporting) is
    # plain numpy - no further accelerator benefit past this point.
    weights_train_unscaled = raw_weights_train.cpu().numpy()
    weights_val_unscaled = raw_weights_val.cpu().numpy()
    weights_test_unscaled = raw_weights_test.cpu().numpy()
    weights_train = weights_train_t.cpu().numpy()
    weights_val = weights_val_t.cpu().numpy()
    weights_test = weights_test_t.cpu().numpy()

    # coefficient = raw (pre-vol-targeting) weight / risk-parity baseline,
    # elementwise - see PortfolioResult's docstring on coefficients_train.
    # Safe to divide directly: risk_parity_weights clamps every entry to
    # >= eps, so the baseline is never exactly zero.
    coefficients_train = (raw_weights_train / train_baseline).cpu().numpy()
    coefficients_val = (raw_weights_val / val_baseline).cpu().numpy()
    coefficients_test = (raw_weights_test / test_baseline).cpu().numpy()

    returns_train_unscaled = (weights_train_unscaled * data.next_returns_train_raw).sum(axis=1)
    returns_val_unscaled = (weights_val_unscaled * data.next_returns_val_raw).sum(axis=1)
    returns_test_unscaled = (weights_test_unscaled * data.next_returns_test_raw).sum(axis=1)
    returns_train = (weights_train * data.next_returns_train_raw).sum(axis=1)
    returns_val = (weights_val * data.next_returns_val_raw).sum(axis=1)
    returns_test = (weights_test * data.next_returns_test_raw).sum(axis=1)

    benchmark_returns_train = vol_match_benchmark(
        inverse_vol_benchmark_returns(data.X_train_raw.cpu().numpy(), data.next_returns_train_raw), returns_train,
    )
    benchmark_returns_val = vol_match_benchmark(
        inverse_vol_benchmark_returns(data.X_val_raw.cpu().numpy(), data.next_returns_val_raw), returns_val,
    )
    benchmark_returns_test = vol_match_benchmark(
        inverse_vol_benchmark_returns(data.X_test_raw.cpu().numpy(), data.next_returns_test_raw), returns_test,
    )

    return PortfolioResult(
        model=model,
        pairs=data.pairs,
        lookback=data.lookback,
        target_vol=target_vol,
        dates_train=data.dates_train,
        dates_val=data.dates_val,
        dates_test=data.dates_test,
        returns_train=returns_train,
        returns_val=returns_val,
        returns_test=returns_test,
        weights_train=weights_train,
        weights_val=weights_val,
        weights_test=weights_test,
        returns_train_unscaled=returns_train_unscaled,
        returns_val_unscaled=returns_val_unscaled,
        returns_test_unscaled=returns_test_unscaled,
        weights_train_unscaled=weights_train_unscaled,
        weights_val_unscaled=weights_val_unscaled,
        weights_test_unscaled=weights_test_unscaled,
        X_train=data.X_train,
        X_val=data.X_val,
        X_test=data.X_test,
        next_returns_train=data.next_returns_train_raw,
        next_returns_val=data.next_returns_val_raw,
        next_returns_test=data.next_returns_test_raw,
        x_mean=data.x_mean,
        x_std=data.x_std,
        benchmark_returns_train=benchmark_returns_train,
        benchmark_returns_val=benchmark_returns_val,
        benchmark_returns_test=benchmark_returns_test,
        coefficients_train=coefficients_train,
        coefficients_val=coefficients_val,
        coefficients_test=coefficients_test,
    )


def _train_and_evaluate(data: _PreparedData, args: argparse.Namespace) -> PortfolioResult:
    """Train one PortfolioLSTM (whatever random seed is currently set) on
    already-prepared data, and evaluate it on both splits.
    """
    model = PortfolioLSTM(
        n_assets=len(data.pairs),
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        noisy_head=args.noisy_head,
        use_prev_weight=args.use_prev_weight,
        n_channels=data.n_channels,
        encoder_type=getattr(args, "encoder_type", "concat"),
        asset_combiner=getattr(args, "asset_combiner", "attention"),
        n_attn_heads=getattr(args, "n_attn_heads", 2),
        pooling=getattr(args, "pooling", "last"),
        position_mode=getattr(args, "position_mode", "long_short"),
    ).to(data.device)
    covariance_estimator = getattr(args, "covariance_estimator", "sample")
    ewma_lambda = getattr(args, "ewma_lambda", 0.94)
    train_portfolio_model(
        model, data.X_train, data.X_train_raw,
        torch.tensor(data.next_returns_train_raw, device=data.device),
        target_vol=args.target_vol,
        epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, noise_std=args.noise_std,
        transaction_cost=args.transaction_cost, max_leverage=args.max_leverage,
        objective=args.objective, sharpe_window=args.sharpe_window,
        cvar_alpha=args.cvar_alpha, cvar_kappa=args.cvar_kappa,
        covariance_estimator=covariance_estimator, ewma_lambda=ewma_lambda,
        X_val=data.X_val, X_val_raw=data.X_val_raw,
        next_returns_val=torch.tensor(data.next_returns_val_raw, device=data.device),
        X_test=data.X_test, X_test_raw=data.X_test_raw,
        next_returns_test=torch.tensor(data.next_returns_test_raw, device=data.device),
    )
    return evaluate_portfolio_model(
        model, data, args.target_vol, max_leverage=args.max_leverage,
        covariance_estimator=covariance_estimator, ewma_lambda=ewma_lambda,
    )


def load_pipeline(args: argparse.Namespace) -> PortfolioResult:
    """Load a previously-trained PortfolioLSTM (or ensemble) from
    `args.load_portfolio` (a local file path OR a quant.model_registry
    name - see load_portfolio_model_auto) and evaluate it on freshly-loaded
    data - no training happens at all. The checkpoint carries its own
    x_mean/x_std (fit during its original training run), the ordered list
    of FX pairs, and the sequence length (lookback) it was trained on, so
    new data is standardized/windowed identically without needing to
    reconstruct the original training split or trust whatever
    `args.pairs`/`args.lookback` a caller passed in - those are genuinely
    part of the model, not free evaluation-time parameters. Only the
    amount of history to fetch (`args.years`), how much of it counts as
    "recent" (`args.train_frac`), and post-hoc knobs (`args.target_vol`,
    transaction cost) remain caller-controlled.
    """
    model = load_portfolio_model_auto(args.load_portfolio)
    requested_pairs = list(dict.fromkeys(args.pairs)) if args.pairs else None
    if requested_pairs is not None and set(requested_pairs) != set(model.pairs):
        logger.warning(
            "Requested pairs %s don't match the pairs %s this model was trained on - "
            "using the model's own pairs (weight order must match training).",
            requested_pairs, model.pairs,
        )
    if args.lookback and args.lookback != model.lookback:
        logger.warning(
            "Requested lookback %d doesn't match the sequence length %d this model was trained on - "
            "using the model's own lookback.",
            args.lookback, model.lookback,
        )
    data = _prepare_data(
        args, x_mean=model.x_mean, x_std=model.x_std, pairs=model.pairs, lookback=model.lookback,
        use_carry=getattr(model, "use_carry", False), vol_horizons=getattr(model, "vol_horizons", []),
    )
    # load_portfolio_model_auto reconstructs the checkpoint on CPU
    # (torch.load(..., map_location="cpu")) regardless of what device it was
    # originally trained on - move it to match data's device (see
    # _prepare_data/get_device) before running it against data.X_train/etc.
    model = model.to(data.device)
    return evaluate_portfolio_model(
        model, data, args.target_vol, max_leverage=args.max_leverage,
        covariance_estimator=getattr(args, "covariance_estimator", "sample"),
        ewma_lambda=getattr(args, "ewma_lambda", 0.94),
    )


def run_pipeline(args: argparse.Namespace) -> PortfolioResult:
    """Load data (via db.py), build sequences, train the portfolio LSTM on
    the train split, and evaluate realized returns on both the train
    (in-sample) and validation (out-of-sample) splits. Single run, whatever
    the ambient random seed is - see run_pipeline_multi_seed() for restarts.
    """
    if args.load_portfolio:
        return load_pipeline(args)
    return _train_and_evaluate(_prepare_data(args), args)


def run_pipeline_multi_seed(args: argparse.Namespace) -> PortfolioResult:
    """Train --n-seeds independent PortfolioLSTMs on the same data (only the
    random seed differs) and combine them via --restart-strategy.

    The Sharpe-ratio training objective is non-convex in the LSTM's
    parameters (the LSTM/softmax nonlinearities, not the Sharpe ratio
    itself, are what make it non-convex - that's true of any neural net
    regardless of loss function), so different random initializations can
    land in meaningfully different local optima. Training several and
    either keeping the best-validated one or averaging them is a standard,
    cheap way to get a more robust result than trusting a single run.

    If --load-portfolio is set, restarts don't apply at all - there's
    nothing to train, so this just delegates to load_pipeline().
    """
    if args.load_portfolio:
        return load_pipeline(args)

    if args.n_seeds < 1:
        raise ValueError(f"--n-seeds must be >= 1, got {args.n_seeds}")

    data = _prepare_data(args)  # load/split/standardize once, reused by every seed

    results = []
    for seed in range(args.n_seeds):
        torch.manual_seed(seed)
        logger.info("--- restart %d/%d (seed=%d) ---", seed + 1, args.n_seeds, seed)
        results.append(_train_and_evaluate(data, args))

    if len(results) == 1:
        return results[0]

    if args.restart_strategy == "best":
        best_idx, best = max(
            enumerate(results), key=lambda item: float(sharpe_ratio(torch.tensor(item[1].returns_val)))
        )
        logger.info(
            "Best of %d restarts: #%d (validation Sharpe %.3f)",
            len(results), best_idx, float(sharpe_ratio(torch.tensor(best.returns_val))),
        )
        return best

    # "ensemble": average every restart's predicted weights.
    ensemble_model = EnsemblePortfolioLSTM([r.model for r in results])
    ensemble_result = evaluate_portfolio_model(
        ensemble_model, data, args.target_vol, max_leverage=args.max_leverage,
    )
    logger.info(
        "Ensemble of %d restarts: validation Sharpe %.3f",
        len(results), float(sharpe_ratio(torch.tensor(ensemble_result.returns_val))),
    )
    return ensemble_result


def print_portfolio_sharpe(result: PortfolioResult) -> None:
    """Log raw (pre-vol-targeting) vs vol-targeted Sharpe ratio and
    cumulative PnL, for both splits. Used by main.py after training/loading."""
    unscaled_train_sharpe = float(sharpe_ratio(torch.tensor(result.returns_train_unscaled)))
    unscaled_val_sharpe = float(sharpe_ratio(torch.tensor(result.returns_val_unscaled)))
    train_sharpe = float(sharpe_ratio(torch.tensor(result.returns_train)))
    val_sharpe = float(sharpe_ratio(torch.tensor(result.returns_val)))
    logger.info(
        "In-sample  (train): raw Sharpe %.3f -> vol-targeted (%.0f%%) Sharpe %.3f | cumulative PnL %.4f",
        unscaled_train_sharpe, result.target_vol * 100, train_sharpe, float(result.returns_train.sum()),
    )
    logger.info(
        "Out-of-sample (val): raw Sharpe %.3f -> vol-targeted (%.0f%%) Sharpe %.3f | cumulative PnL %.4f",
        unscaled_val_sharpe, result.target_vol * 100, val_sharpe, float(result.returns_val.sum()),
    )
