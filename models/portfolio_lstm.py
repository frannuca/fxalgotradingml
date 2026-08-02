"""Directional-probability predictor for FX pairs: N independent per-asset
LSTMs, cross-informed by richer input rather than a separate cross-asset
stage.

This module is a LIBRARY - it has no CLI of its own. The single entry
point for the whole project is main.py at the repo root, which reads a
JSON config file and orchestrates training/evaluation by calling the
functions here directly. See main.py's module docstring for the JSON
schema.

This is a PURE PREDICTION system - there is no portfolio, no weight, no
PnL, no Sharpe ratio, no trading decision anywhere in this file. Its only
job: for each asset, predict a Z-SCORE for its cumulative log return over
the next `direction_horizon` days (see make_sequences), convert that to a
calibrated probability via the probit link (see probit()), and measure
how often the predicted direction is right (hit rate / confusion matrix)
on train, validation, and test data. What a trading system would
eventually DO with that probability is entirely out of scope here.

Architecture (see PredictionModel/AssetLSTM)
-----------------------------------------------------------------------
PredictionModel is N INDEPENDENT per-asset LSTMs (AssetLSTM), one per
asset, no parameters shared between them - since different pairs can have
genuinely different dynamics. Each asset's own LSTM, however, reads the
FULL cross-sectional feature block at every timestep - every asset's own
[log return, rolling volatility/skewness/kurtosis, optional carry], not
just this asset's own - so cross-asset correlation (e.g. a USD-wide move
visible across several pairs the same day) can still be learned, directly
inside each asset's own recurrence, without a separate cross-asset
"copula" stage. (An earlier version of this code had exactly that - a
second, shared CopulaLSTM stage mixing each asset's already-compressed
z-score across assets - but it was a bottleneck: CopulaLSTM only ever saw
one scalar per asset per day, never the raw features, so any cross-asset
structure stage 1 didn't already encode into that single number was
unrecoverable. Feeding every asset's own LSTM the full cross-sectional
input directly removes that bottleneck without adding a stage.)

Each AssetLSTM's recurrent output is followed by a CAUSAL self-attention
layer over the time axis (masked so day k only attends to days <= k,
preserving the no-look-ahead property every day's dense supervision
relies on - see AssetLSTM's own docstring) before the final head, giving
the network a second route to long-range within-window dependencies
beyond whatever the LSTM's fixed-size recurrent state carries forward.
The whole network is deterministic - no NoisyNet head, no input-noise
regularization; dropout (training mode only) is the only stochasticity.

Each AssetLSTM outputs a heteroscedastic (mu, sigma) pair at EVERY day in
the window (not just the last one) - a dense, per-day, per-asset
supervised signal. mu is the predicted conditional MEAN of that day's
forward z-score; sigma is the predicted conditional STD, strictly
positive. P(positive) = probit(mu / sigma) - see probit().

Training is FULLY INDEPENDENT per asset (see train_prediction_model/
_train_and_evaluate): each asset's LSTM gets its OWN optimizer (separate
Adam momentum/variance state) and its OWN loss - Gaussian NLL (fits the
full predictive distribution) plus a BCE direction term (the
anti-mean-collapse term - see gaussian_nll/direction_bce), applied densely
over every (sample, day) pair for THAT asset alone, never averaged
together with any other asset's loss before its own backward() call. The
assets never shared weights; this decouples the optimization too, so nothing about
how one asset trains - gradient magnitude, Adam's per-parameter state,
which epoch its own best-validation checkpoint comes from - depends on
what any other asset is doing. If an asset is linked to another pair's
features via `cross_pairs`, only what its LSTM READS is widened - what
optimizes it stays entirely its own.

All N independent LSTMs are persisted TOGETHER as one PredictionModel
checkpoint (see save_model/save_to_db/load_model/load_from_db) - loading
one file/DB row is enough to run the full pipeline on new data.
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
    forward/backward pass, not mini-batched), so there's no per-batch
    host<->device transfer overhead once the data is moved once at the
    start; a GPU/MPS backend meaningfully speeds up the repeated LSTM
    matrix multiplications every epoch.
    """
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


# Optional hook so a caller (e.g. api/server.py, for live-updating charts in
# the Training view) can observe interim train/validation loss and hit rate
# DURING training, without train_prediction_model needing any new PUBLIC
# parameters - a contextvar, not a plain global, so
# concurrent training jobs in different background threads never see each
# other's callback. Library code (this module) only ever READS this;
# nothing here ever sets it.
_epoch_report_callback: contextvars.ContextVar[
    Callable[[str, int, int, float, float, float | None, float | None], None] | None
] = contextvars.ContextVar("_epoch_report_callback", default=None)


def _report_epoch(
    stage: str,
    epoch: int,
    epochs: int,
    train_loss: float,
    train_hit_rate: float,
    val_loss: float | None = None,
    val_hit_rate: float | None = None,
) -> None:
    """Call the registered interim-results callback, if any, with this
    epoch's training BCE loss and mean (decision-day) hit rate, plus
    validation versions of both when available - a no-op when nothing has
    registered one (e.g. the CLI / main.py path). `stage` is "marginal" or
    "copula", so a caller tracking both training phases can tell which one
    an update belongs to.
    """
    callback = _epoch_report_callback.get()
    if callback is not None:
        callback(stage, epoch, epochs, train_loss, train_hit_rate, val_loss, val_hit_rate)


class TrainingStopped(Exception):
    """Raised from inside train_prediction_model's epoch loop (see
    _check_stop) when a caller-registered stop-check callback reports a
    stop was requested. Deliberately a plain exception, not a
    return-value/flag threaded through every call site: it propagates
    uninterrupted through any number of nested restarts (--n-seeds), so
    only the top-level caller (api/server.py's _run_training_job) needs to
    catch it, once, to end
    the job cleanly instead of treating it as an error.
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


def rolling_moment_features(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """For each asset, its own trailing `window`-day rolling standard
    deviation, skewness, and excess kurtosis of log returns - the moments
    a risk manager actually looks at (vol = realized risk, skewness =
    asymmetric tail risk, kurtosis = fat-tail/regime instability), used
    here as STAGE-1 INPUT FEATURES so each asset's own AssetLSTM sees more
    than just the raw return.

    Returns a DataFrame with 3 columns per asset ("{pair}_vol"/"_skew"/
    "_kurt"), aligned to `returns`'s own index. The first `window` or so
    rows are necessarily NaN (not enough trailing history for a stable
    estimate, and skew/kurtosis need slightly more) - left for the caller
    to handle (see build_feature_dataframe).
    """
    vol = returns.rolling(window).std()
    skew = returns.rolling(window).skew()
    kurt = returns.rolling(window).kurt()  # excess kurtosis (pandas already subtracts 3)
    out = pd.DataFrame(index=returns.index)
    for col in returns.columns:
        out[f"{col}_vol"] = vol[col]
        out[f"{col}_skew"] = skew[col]
        out[f"{col}_kurt"] = kurt[col]
    return out


#: Every feature build_feature_dataframe knows how to produce, in this
#: FIXED canonical order (a caller's `features` list is a SELECTION out of
#: this catalog, not an ordering directive - avoids ambiguity over where
#: "cma"'s multiple expanded columns would land relative to the rest).
#: "cma" is special: it contributes one channel PER (short, long) window
#: pair in `cma_windows`, not just one - see n_channels_per_pair.
FEATURE_CATALOG: tuple[str, ...] = ("log_return", "vol", "skew", "kurt", "carry", "cma")

#: Default feature selection - matches this module's original always-on
#: set (raw return + rolling vol/skew/kurtosis), with carry and CMAs both
#: opt-in.
DEFAULT_FEATURES: list[str] = ["log_return", "vol", "skew", "kurt"]

#: Default (short, long) trailing windows for "cma" (see
#: cross_moving_averages) when "cma" is selected but no windows are given.
DEFAULT_CMA_WINDOWS: list[tuple[int, int]] = [[10, 50]]


def n_channels_per_pair(features: list[str], cma_windows: list | None = None) -> int:
    """How many input channels build_feature_dataframe produces PER PAIR
    for this feature selection: every base feature in FEATURE_CATALOG
    contributes exactly one channel; "cma" instead contributes one channel
    PER (short, long) window pair in `cma_windows` (zero if "cma" isn't
    selected, regardless of what `cma_windows` contains).
    """
    unknown = set(features) - set(FEATURE_CATALOG)
    if unknown:
        raise ValueError(f"Unknown feature(s) {sorted(unknown)} - choose from {FEATURE_CATALOG}")
    base = sum(1 for f in features if f != "cma")
    cma_count = len(cma_windows or []) if "cma" in features else 0
    return base + cma_count


def cross_moving_averages(returns: pd.DataFrame, cma_windows: list) -> dict:
    """For each (short, long) window pair, the TREND signal
    `rolling_mean(returns, short) - rolling_mean(returns, long)` per pair -
    positive when the recent (short-window) trend in log returns is
    running above the longer-run (long-window) trend, the return-space
    analogue of a classic price moving-average crossover (a fast MA
    crossing above a slow one signals a new uptrend). Purely trailing
    (min_periods=window, no bfill - same "exclude rather than paper over"
    policy as make_sequences' trailing_vol), so it never looks ahead.

    Returns a dict keyed by (pair, short, long) -> pd.Series aligned to
    `returns`'s own index.
    """
    out = {}
    for short, long_ in cma_windows:
        if short >= long_:
            raise ValueError(f"cma_windows short window ({short}) must be < long window ({long_})")
        short_ma = returns.rolling(short, min_periods=short).mean()
        long_ma = returns.rolling(long_, min_periods=long_).mean()
        diff = short_ma - long_ma
        for pair in returns.columns:
            out[(pair, short, long_)] = diff[pair]
    return out


def build_feature_dataframe(
    returns: pd.DataFrame,
    pairs: list[str],
    features: list[str],
    rolling_stats_window: int,
    cma_windows: list,
    years: int,
) -> pd.DataFrame:
    """Build PredictionModel's actual (wider) input feature set: for each
    pair, in order, whichever of [raw log return, rolling vol, rolling
    skew, rolling kurtosis, carry, cma...] are selected in `features` (see
    FEATURE_CATALOG for the fixed column order) - deliberately ASSET-MAJOR
    (all of one asset's channels together, then the next asset's).

    Which of these channels actually feed a given asset's OWN AssetLSTM is
    a SEPARATE question, decided by `cross_pairs` at the PredictionModel
    level (see its own docstring) - this function always builds the FULL
    block for every pair in `pairs`, regardless of cross_pairs; slicing
    down to what one asset's LSTM actually sees happens later, in
    PredictionModel.forward().
    """
    n_channels = n_channels_per_pair(features, cma_windows)
    moments = rolling_moment_features(returns, rolling_stats_window) if any(f in features for f in ("vol", "skew", "kurt")) else None
    carry = load_carry(pairs, years, returns.index) if "carry" in features else None
    cmas = cross_moving_averages(returns, cma_windows) if "cma" in features else None

    features_df = pd.DataFrame(index=returns.index)
    for pair in pairs:
        if "log_return" in features:
            features_df[f"{pair}_ret"] = returns[pair]
        if "vol" in features:
            features_df[f"{pair}_vol"] = moments[f"{pair}_vol"]
        if "skew" in features:
            features_df[f"{pair}_skew"] = moments[f"{pair}_skew"]
        if "kurt" in features:
            features_df[f"{pair}_kurt"] = moments[f"{pair}_kurt"]
        if "carry" in features:
            features_df[f"{pair}_carry"] = carry[pair]
        if "cma" in features:
            for short, long_ in cma_windows:
                features_df[f"{pair}_cma_{short}_{long_}"] = cmas[(pair, short, long_)]

    # Rolling-window features leave NaN at the very start (not enough
    # trailing history yet) - forward-fill (for any mid-series gaps) then
    # treat any still-leading NaN as a neutral 0.0, NOT back-filled:
    # back-filling would leak a later day's value into an earlier row.
    # Raw returns themselves are never NaN here.
    features_df = features_df.ffill().fillna(0.0)
    assert features_df.shape[1] == n_channels * len(pairs)
    return features_df


def standardize(values: np.ndarray, axis) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std computed over `axis`, with a small epsilon to avoid /0."""
    return values.mean(axis=axis), values.std(axis=axis) + 1e-8


def _forward_zscore_labels(
    values: np.ndarray, direction_horizon: int, trailing_vol: np.ndarray, eps: float = 1e-8,
) -> np.ndarray:
    """(T, n_pairs) raw log returns + (T, n_pairs) trailing volatility (see
    rolling_moment_features - the SAME estimate PredictionModel also gets
    as an input feature, known as of day d, no look-ahead) -> (T, n_pairs)
    continuous z-score labels:
        z[d] = (cumulative log return over [d+1, d+H]) / (trailing_vol[d] * sqrt(H))
    - the forward move's size in units of "how many trailing-volatility
    standard deviations", under the standard i.i.d.-returns assumption
    that an H-day sum's std scales as the daily std times sqrt(H). This
    keeps each move's MAGNITUDE, not just its sign (unlike a binary
    up/down label), and normalizes it consistently with the vol input
    feature so its scale is stable across assets/regimes - a 1% move on a
    quiet pair and a 3% move on a volatile one can land at a similar
    z-score if both are equally "surprising" relative to their own recent
    volatility. This is what PredictionModel is trained to predict (as the
    mean of a Gaussian - see gaussian_nll/train_prediction_model);
    sign(z) is still exactly the direction (what hit rate/confusion
    matrices use), and probit() recovers a calibrated probability from it.

    The last `direction_horizon` rows have no full forward window and are
    NaN - make_sequences never windows into them (see its own docstring).
    Vectorized via a cumulative-sum trick rather than an explicit rolling
    loop: sum(values[d+1 : d+1+H]) = cumsum[d+H] - cumsum[d].
    """
    T, n = values.shape
    padded_cumsum = np.concatenate([np.zeros((1, n), dtype=values.dtype), np.cumsum(values, axis=0)], axis=0)  # (T+1, n)
    z = np.full((T, n), np.nan, dtype=np.float32)
    valid = T - direction_horizon
    if valid > 0:
        forward_sum = padded_cumsum[direction_horizon + 1 : valid + direction_horizon + 1] - padded_cumsum[1 : valid + 1]
        denom = trailing_vol[:valid] * np.sqrt(direction_horizon) + eps
        z[:valid] = forward_sum / denom
    return z


def make_sequences(
    returns: pd.DataFrame, lookback: int, feature_returns: pd.DataFrame | None = None,
    direction_horizon: int = 5, rolling_stats_window: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Slide a window over `returns` to build (X, next_returns, z_labels) triples.

    X[i]: `lookback` days of features for every pair, i.e. everything
          known through day t (the window's last day, the "decision day").
    next_returns[i]: the RAW log return of every pair on day t+1 - kept
          purely for reporting (e.g. a cumulative-return chart of the
          asset itself), never used as a training target.
    z_labels[i]: (lookback, n_pairs) - one CONTINUOUS z-score label PER DAY
          within the window, not just at the end (see
          _forward_zscore_labels): z_labels[i][k] is pair p's forward
          `direction_horizon`-day cumulative return starting right after
          window-day k, expressed in trailing-volatility standard
          deviations. This is what PredictionModel is supervised against
          at EVERY timestep (dense, per-day-per-asset labels - see
          train_prediction_model); z_labels[i][-1] (the decision day's own
          label) is what hit rate/confusion matrices are computed from
          (via sign(z_labels[i][-1])) everywhere else in this file. Every one of these
          labels' forward window (day k+1..k+H) is entirely AFTER day k,
          so this is never look-ahead leakage - it's simply "does
          everything computable at day k's close accurately predict day
          k's own forward outcome", for every k, not just the last one.
    dates[i]: the date of day t+1 (i.e. of next_returns[i]).

    `feature_returns`, if given (see build_feature_dataframe), is a WIDER
    DataFrame - same row count/date index as `returns`, extra columns
    (rolling vol/skew/kurtosis, carry) - used to build X INSTEAD of
    `returns` itself. `next_returns`/`dates`/z_labels always come from
    `returns` alone: engineered features are inputs the model reads,
    never something a label is computed from.

    `rolling_stats_window` is the SAME trailing window used for the vol
    input feature (see rolling_moment_features) - reused here to
    normalize z_labels, so the label's scale matches what the model
    itself can observe about "how volatile has this asset been lately".
    """
    values = returns.to_numpy(dtype=np.float32)  # shape (T, n_pairs)
    feature_values = feature_returns.to_numpy(dtype=np.float32) if feature_returns is not None else values
    index = returns.index

    # Requires a FULL rolling_stats_window of history (min_periods equal to
    # the window itself, no bfill) - an under-filled trailing std from a
    # handful of points is a noisy, easily-tiny denominator that would
    # otherwise blow up the earliest z-score labels (Huber caps the
    # gradient damage, but the label itself is still garbage). The first
    # `rolling_stats_window - 1` days are simply excluded from every split
    # below (via first_start) rather than papered over with a fallback.
    trailing_vol = (
        returns.rolling(rolling_stats_window, min_periods=rolling_stats_window).std().to_numpy(dtype=np.float32)
    )
    full_z = _forward_zscore_labels(values, direction_horizon, trailing_vol)  # (T, n_pairs), NaN in the first rolling_stats_window-1 and last direction_horizon rows

    X, next_returns, z_labels, dates = [], [], [], []
    first_start = rolling_stats_window - 1  # trailing_vol[k] is only defined for k >= this
    last_start = len(values) - lookback - (direction_horizon - 1)
    for start in range(max(first_start, 0), max(last_start, 0)):
        day_t = start + lookback - 1     # last day inside the window - the decision day
        day_t_plus_1 = day_t + 1         # the day right after the window

        X.append(feature_values[start:day_t + 1])       # rows start..day_t inclusive
        next_returns.append(values[day_t_plus_1])
        z_labels.append(full_z[start:day_t + 1])         # (lookback, n_pairs) - one z-score label per window day
        dates.append(index[day_t_plus_1])

    if not X:
        raise ValueError(
            f"Not enough history ({len(values)} rows) for lookback={lookback} + "
            f"direction_horizon={direction_horizon}."
        )

    X, next_returns, z_labels = np.stack(X), np.stack(next_returns), np.stack(z_labels)
    assert X.shape[1] == lookback
    assert not np.isnan(z_labels).any()  # every window's labels were constructed to stay inside the valid range
    return X, next_returns, z_labels, pd.DatetimeIndex(dates)


# --------------------------------------------------------------------------
# Model: N independent, cross-informed per-asset predictors
# --------------------------------------------------------------------------

class AssetLSTM(nn.Module):
    """ONE asset's own LSTM, followed by a CAUSAL self-attention layer over
    the time axis. No parameters are shared with any other asset's
    AssetLSTM (see PredictionModel) - each asset gets its own dedicated
    network, since different pairs can have genuinely different dynamics.
    Its INPUT, however, is the FULL cross-sectional feature block - every
    asset's own [log return, rolling vol/skew/kurtosis, optional carry] at
    each timestep, not just this asset's own - so this asset's recurrence
    can still react to any other asset's moves (e.g. a JPY pair's LSTM
    learning to key off a USD-wide move visible in EURUSD/GBPUSD that same
    day), without a separate cross-asset stage or the information
    bottleneck of only seeing a compressed summary of it.

    The whole network is deterministic: no NoisyNet head, no input-noise
    regularization (see train_prediction_model) - dropout (in training
    mode only) is the sole source of any stochasticity, same as any
    ordinary supervised model.

    Attention layer: the LSTM's hidden state at day k already summarizes
    x[0..k] recurrently, but that summary is a fixed-size bottleneck - the
    attention layer lets day k's representation pull directly from any
    EARLIER day's hidden state instead of relying purely on what the
    recurrence chose to carry forward. It MUST be causally masked (day k
    can only attend to days <= k): this network outputs a prediction at
    EVERY day in the window (see below), each one supervised as if it were
    computed using only information through that day - an unmasked
    (bidirectional) attention layer would let day k's prediction see days
    AFTER k, which is exactly the label look-ahead leakage the rest of
    this module (purge/embargo, trailing-only features) is built to avoid.
    Implemented as a residual block (attention output added back to the
    LSTM output, then layer-normed), the standard post-norm Transformer
    pattern - so early training, before the attention weights have learned
    anything useful, still passes the LSTM's own signal through unchanged.

    Outputs a heteroscedastic (mu, sigma) pair at EVERY timestep, not just
    the last one - a dense, per-day supervised signal:
      - mu:    (batch, lookback) - the predicted conditional MEAN of that
               day's forward z-score, unbounded.
      - sigma: (batch, lookback) - the predicted conditional STD of that
               same z-score, strictly positive (softplus + floor).
    P(positive) = probit(mu / sigma) - see probit(). Trained with Gaussian
    NLL + a BCE direction term (see gaussian_nll/direction_bce/
    train_prediction_model): a bare regression loss is minimized by the
    label's unconditional mean - one constant sign for every sample when
    the target is barely predictable (the degenerate all-recall/zero-
    specificity confusion matrix) - while BCE is minimized by matching
    each sample's OWN sign, forcing mu/sigma to spread across 0 wherever
    the (now cross-sectional) features actually discriminate.
    """

    def __init__(
        self, input_size: int, hidden_size: int = 16, num_layers: int = 1, dropout: float = 0.0,
        n_attn_heads: int = 4,
    ):
        super().__init__()
        if hidden_size % n_attn_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by n_attn_heads ({n_attn_heads})")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.n_attn_heads = n_attn_heads
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(hidden_size, n_attn_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, 2)
        # Bias the sigma half so softplus(bias) ~ 1.0 at init: the model
        # starts out saying "predictive std = 1" (the label's own
        # unconditional scale), i.e. p ~ probit(mu) ~ 0.5 while mu is
        # still small - a calibrated-ignorance starting point, rather
        # than random over/under-confidence it must first unlearn.
        with torch.no_grad():
            self.head.bias[1].fill_(0.5413)  # softplus(0.5413) = 1.0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, lookback, input_size) - the FULL cross-sectional
        # feature block (every asset's own channels), the SAME input every
        # asset's own AssetLSTM instance receives (see PredictionModel).
        # Returns (mu, sigma), each (batch, lookback) - this asset's own
        # prediction at EVERY day in the window.
        lstm_out, _ = self.lstm(x)               # (batch, lookback, hidden_size)
        lstm_out = self.dropout(lstm_out)
        lookback = lstm_out.shape[1]
        # Causal mask: position i may attend to positions <= i only (see
        # this class's own docstring on why an unmasked layer would leak).
        causal_mask = torch.triu(
            torch.full((lookback, lookback), float("-inf"), device=x.device, dtype=lstm_out.dtype), diagonal=1,
        )
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out, attn_mask=causal_mask, need_weights=False)
        h = self.attn_norm(lstm_out + attn_out)   # residual + norm
        out = self.head(h)                        # (batch, lookback, 2)
        mu = out[..., 0]
        sigma = nn.functional.softplus(out[..., 1]) + 1e-3
        return mu, sigma


# --------------------------------------------------------------------------
# Model: the combined, persistable pipeline
# --------------------------------------------------------------------------

class PredictionModel(nn.Module):
    """N independent per-asset LSTMs (AssetLSTM - see its docstring) as
    ONE persistable unit. forward() runs every asset's own LSTM for
    inference on new data after loading (see load_model/load_from_db) -
    see train_prediction_model for how they're actually TRAINED (jointly,
    one optimizer over every parameter of every asset's LSTM - see this
    module's own docstring).

    Which OTHER pairs' features (if any) feed a given asset's own LSTM is
    controlled by `cross_pairs`: a `{pair: [other_pair, ...]}` dict. A
    pair's own features are ALWAYS included; `cross_pairs` only adds MORE.
    DEFAULT is `{}` - i.e. every pair NOT listed as a key gets no cross
    pairs at all, so by default every asset's LSTM sees ONLY its own
    features (fully independent per-asset networks, no cross-asset mixing
    unless explicitly linked). This replaces the earlier "every asset sees
    every pair, always" design - unconditional full mixing was itself an
    earlier experiment (see this module's own docstring on the removed
    CopulaLSTM) with the same bottleneck-avoidance goal, but no way to
    dial it back down when it wasn't paying for itself on a given pair;
    `cross_pairs` makes that a per-pair, explicit, opt-in choice instead.

    forward() returns (mu, sigma), each (batch, lookback, n_assets) - the
    dense per-day, per-asset predictive mean and std (see AssetLSTM), NOT
    a probability. The window's LAST timestep is the "decision day" - what
    hit rate/confusion matrices/live inference actually use.
    P(positive) = probit(mu / (sigma * sigma_hat)) - see
    evaluate_prediction_model, which is where every reporting consumer in
    this codebase (hit rate, confusion matrix, the frontend) actually gets
    its probabilities from, and where the residual calibration sigma_hat
    is fit.
    """

    def __init__(
        self,
        n_assets: int,
        pairs: list[str],
        n_channels: int = 4,
        hidden_size: int = 16,
        num_layers: int = 1,
        dropout: float = 0.1,
        n_attn_heads: int = 4,
        cross_pairs: dict | None = None,
    ):
        super().__init__()
        if n_assets != len(pairs):
            raise ValueError(f"n_assets ({n_assets}) must match len(pairs) ({len(pairs)})")
        self.n_assets = n_assets
        self.pairs = list(pairs)
        self.n_channels = n_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.n_attn_heads = n_attn_heads
        self.cross_pairs = {k: list(v) for k, v in (cross_pairs or {}).items()}

        # For each target pair, which GLOBAL pair-indices' channel blocks
        # feed its OWN LSTM - itself always first, then its configured
        # cross_pairs (deduplicated, unknown pairs rejected loudly rather
        # than silently ignored).
        pair_index = {p: i for i, p in enumerate(self.pairs)}
        self.included_indices: list[list[int]] = []
        for pair in self.pairs:
            ordered = [pair] + [p for p in self.cross_pairs.get(pair, []) if p != pair]
            seen: set[str] = set()
            indices = []
            for p in ordered:
                if p in seen:
                    continue
                if p not in pair_index:
                    raise ValueError(f"cross_pairs[{pair!r}] references unknown pair {p!r} - not in {self.pairs}")
                seen.add(p)
                indices.append(pair_index[p])
            self.included_indices.append(indices)

        self.assets = nn.ModuleList([
            AssetLSTM(len(indices) * n_channels, hidden_size, num_layers, dropout, n_attn_heads)
            for indices in self.included_indices
        ])

    def asset_input(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """The channel-sliced input asset `i`'s OWN LSTM actually reads
        from the full asset-major `x` (see forward()) - itself, and
        whichever other pairs `cross_pairs` links it to (see
        self.included_indices, built in __init__). Used by both forward()
        (all assets at once) and train_prediction_model (one asset at a
        time, its own independent optimizer/loss - see that function's own
        docstring for why they're no longer run through forward() during
        training).
        """
        indices = self.included_indices[i]
        blocks = [x[:, :, idx * self.n_channels : (idx + 1) * self.n_channels] for idx in indices]
        return blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, lookback, n_assets * n_channels), ASSET-MAJOR in
        # self.pairs' order (see build_feature_dataframe). Returns
        # (mu, sigma), each (batch, lookback, n_assets).
        outs = [self.assets[i](self.asset_input(x, i)) for i in range(self.n_assets)]
        mu = torch.stack([o[0] for o in outs], dim=-1)
        sigma = torch.stack([o[1] for o in outs], dim=-1)
        return mu, sigma

    def _checkpoint_dict(
        self, x_mean: np.ndarray, x_std: np.ndarray, pairs: list[str], lookback: int,
        features: list[str] | None = None, cma_windows: list | None = None,
        sigma_hat: np.ndarray | None = None, neutral_band: float = 0.0,
    ) -> dict:
        """Build the checkpoint dict shared by save_model() (-> local file)
        and save_to_db() (-> quant.model_registry blob).

        `pairs`/`cross_pairs` (in `config`, needed to reconstruct the
        model's own per-asset input-slicing - see __init__) and
        `lookback`/`features`/`cma_windows` (top-level, needed to rebuild
        an IDENTICAL feature dataframe for new data - see
        build_feature_dataframe) are persisted the same way as before:
        properties of the trained model, not free evaluation-time
        parameters a caller could change later without invalidating the
        weights.

        `sigma_hat` (see evaluate_prediction_model's docstring on
        calibration) is the per-asset residual-std estimate the probit
        link needs to turn a raw (mu, sigma) into a CALIBRATED
        probability - fit once on validation data at training time, then
        reused as-is at live-inference time (see
        api/server.py's _predict_latest_probabilities), since there is no
        validation set available when only a single new window is being
        scored. Defaults to all-ones (i.e. uncalibrated) when not given,
        so old checkpoints without it still load.
        """
        return {
            "config": {
                "n_assets": self.n_assets,
                "pairs": self.pairs,
                "n_channels": self.n_channels,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout_p,
                "n_attn_heads": self.n_attn_heads,
                "cross_pairs": self.cross_pairs,
            },
            "state_dict": self.state_dict(),
            "x_mean": torch.as_tensor(x_mean),
            "x_std": torch.as_tensor(x_std),
            "pairs": list(pairs),
            "lookback": lookback,
            "features": list(features) if features is not None else list(DEFAULT_FEATURES),
            "cma_windows": [list(w) for w in (cma_windows or [])],
            "sigma_hat": torch.as_tensor(sigma_hat if sigma_hat is not None else np.ones(self.n_assets, dtype=np.float32)),
            # Half-width of the abstention region around p=0.5 (see
            # apply_neutral_band) - persisted so live inference abstains
            # exactly the way the reported backtest metrics did.
            "neutral_band": float(neutral_band),
        }

    @classmethod
    def _from_checkpoint(cls, checkpoint: dict) -> "PredictionModel":
        """Reconstruct a PredictionModel (every asset's LSTM, and its
        per-asset cross_pairs input slicing) from a checkpoint dict
        (however it was loaded - local file or DB blob). The returned
        model also carries `.x_mean`/`.x_std` (numpy), `.pairs`,
        `.lookback`, `.features`, `.cma_windows`, `.sigma_hat` (numpy, see
        _checkpoint_dict), so callers can standardize new input windows,
        rebuild the exact same feature set, and produce calibrated
        probabilities without re-supplying any of it themselves.
        """
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        model.x_mean = checkpoint["x_mean"].numpy()
        model.x_std = checkpoint["x_std"].numpy()
        model.pairs = checkpoint["pairs"]
        model.lookback = checkpoint["lookback"]
        model.features = checkpoint.get("features", list(DEFAULT_FEATURES))
        model.cma_windows = checkpoint.get("cma_windows", [])
        n_assets = checkpoint["config"]["n_assets"]
        sigma_hat = checkpoint.get("sigma_hat")
        model.sigma_hat = sigma_hat.numpy() if sigma_hat is not None else np.ones(n_assets, dtype=np.float32)
        model.neutral_band = float(checkpoint.get("neutral_band", 0.0))
        return model

    def save_model(
        self, path: str = "models/prediction_model.pt", *,
        x_mean: np.ndarray, x_std: np.ndarray, pairs: list[str], lookback: int,
        features: list[str] | None = None, cma_windows: list | None = None,
        sigma_hat: np.ndarray | None = None, neutral_band: float = 0.0,
    ) -> None:
        """Persist every asset's trained LSTM weights, architecture
        config (including per-asset cross_pairs input slicing), input
        standardization stats, the ordered FX pairs, the sequence length,
        the feature selection, and the validation-fit calibration scale -
        a self-contained checkpoint load_model() can rebuild and run
        calibrated inference from without retraining.
        """
        torch.save(self._checkpoint_dict(x_mean, x_std, pairs, lookback, features, cma_windows, sigma_hat, neutral_band), path)
        logger.info("Saved model weights to %s", path)

    def save_to_db(
        self, name: str, *,
        x_mean: np.ndarray, x_std: np.ndarray, pairs: list[str], lookback: int,
        features: list[str] | None = None, cma_windows: list | None = None,
        sigma_hat: np.ndarray | None = None, neutral_band: float = 0.0,
        description: str = "",
    ) -> None:
        """Serialize the same checkpoint save_model() would write, and
        upsert it into quant.model_registry under `name` instead of a
        local file - see data/model_registry.py.
        """
        from data.model_registry import save_model_blob

        buffer = io.BytesIO()
        torch.save(self._checkpoint_dict(x_mean, x_std, pairs, lookback, features, cma_windows, sigma_hat, neutral_band), buffer)
        save_model_blob(name, buffer.getvalue(), model_type="prediction", description=description)

    @classmethod
    def load_model(cls, path: str) -> "PredictionModel":
        """Reconstruct a PredictionModel from a checkpoint saved by save_model()."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        return cls._from_checkpoint(checkpoint)

    @classmethod
    def load_from_db(cls, name: str) -> "PredictionModel":
        """Reconstruct a PredictionModel from a checkpoint saved by save_to_db()."""
        from data.model_registry import load_model_blob

        checkpoint = torch.load(io.BytesIO(load_model_blob(name)), map_location="cpu", weights_only=True)
        return cls._from_checkpoint(checkpoint)


def load_prediction_model(path: str) -> PredictionModel:
    """Load a PredictionModel checkpoint from a local file."""
    return PredictionModel.load_model(path)


def load_prediction_model_from_db(name: str) -> PredictionModel:
    """Load a PredictionModel checkpoint from quant.model_registry by name."""
    return PredictionModel.load_from_db(name)


def load_prediction_model_auto(value: str) -> PredictionModel:
    """Load a PredictionModel from either a local file path or a
    quant.model_registry name - tries the local file first (so an existing
    path always wins even if it happens to collide with a DB name)."""
    if os.path.exists(value):
        return load_prediction_model(value)
    return load_prediction_model_from_db(value)


# --------------------------------------------------------------------------
# Probit link: z-score -> calibrated probability (reporting only)
# --------------------------------------------------------------------------

def probit(z: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF: P(N(0, 1) draw <= z). The standard "probit
    link" for recovering a calibrated probability from a continuous
    z-score - PredictionModel/AssetLSTM predict a raw (mu, sigma) pair
    (see their docstrings), trained via Gaussian NLL + a BCE direction
    term on that pair (see train_prediction_model), never via sigmoid+BCE
    on a bounded probability directly. This is applied afterward, purely
    for reporting (see evaluate_prediction_model, the only place it's
    called) - never inside the model, never inside its own loss.

    z=0 (no predicted edge) maps to exactly 0.5, so hit rate/confusion-
    matrix code elsewhere (which all threshold at 0.5) matches sign(z)
    exactly.
    """
    return torch.special.ndtr(z)


def gaussian_nll(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Gaussian negative log-likelihood of `target` under
    N(mu, sigma^2), per element:  0.5 * (log sigma^2 + (target - mu)^2 / sigma^2).

    The proper scoring rule for a heteroscedastic (mu, sigma) head (see
    AssetLSTM): unlike a plain point-estimate loss, it lets the model
    LOWER its loss on unpredictable samples by RAISING sigma there (honest
    "I don't know" -> p ~ 0.5) while keeping sigma small - and getting
    full credit - on the samples where mu really discriminates. Constant
    additive terms (log 2*pi) are dropped; they don't affect gradients.
    """
    return (0.5 * (2.0 * torch.log(sigma) + ((target - mu) / sigma) ** 2)).mean()


def direction_bce(mu: torch.Tensor, sigma: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy of the model's implied direction probability
    probit(mu / sigma) against the realized direction sign(target_z).

    This is the term that directly attacks the mean-collapse failure
    mode: Gaussian NLL (or Huber) alone is minimized by mu ~ E[z] - one
    constant sign for every sample when the target is barely predictable
    (the all-TP/all-FN=0 confusion matrix). BCE is instead minimized by
    matching each SAMPLE's own sign, so any feature that discriminates
    up-days from down-days gets pulled into mu/sigma's spread - and where
    nothing discriminates, the optimum is exactly p = base rate ~ 0.5,
    which the neutral band (see apply_neutral_band) then reports as an
    abstention rather than a coin-flip directional call.

    Computed via ndtr (torch.special.ndtr, same as probit()) plus a
    clamped log, NOT torch.special.log_ndtr: the latter is numerically
    nicer (avoids materializing a probability that could round to exactly
    0 or 1) but isn't implemented on the MPS backend as of this
    codebase's target PyTorch versions, and this runs every training
    epoch - reaching for it would force a CPU fallback (PYTORCH_ENABLE_MPS_
    FALLBACK) on every single call rather than running natively on MPS.
    The clamp floor only ever binds for astronomically extreme/miscalibrated
    scores, far outside anything a z-score head actually produces.
    """
    score = mu / sigma
    y = (target_z > 0).to(score.dtype)
    eps = 1e-12
    log_p = torch.log(torch.clamp(torch.special.ndtr(score), min=eps))         # log P(positive)
    log_not_p = torch.log(torch.clamp(torch.special.ndtr(-score), min=eps))    # log P(negative), by symmetry
    return -(y * log_p + (1.0 - y) * log_not_p).mean()


# --------------------------------------------------------------------------
# Neutral band: abstain (report exactly 0.5) when the edge is too small
# --------------------------------------------------------------------------

def apply_neutral_band(probabilities: np.ndarray, neutral_band: float) -> np.ndarray:
    """Snap every probability inside (0.5 - band, 0.5 + band) to EXACTLY
    0.5 - the model's explicit "no call" output. Outside the band the
    probability passes through untouched. band <= 0 disables snapping.

    Downstream, confusion_matrix_counts/hit rates treat only samples
    OUTSIDE the band as actual directional calls (see their docstrings) -
    so the reported precision/recall describe the model's behavior when
    it chooses to speak, and `coverage` reports how often that is.
    """
    if neutral_band <= 0:
        return probabilities
    out = probabilities.copy()
    out[np.abs(out - 0.5) < neutral_band] = 0.5
    return out


def confusion_matrix_counts(probabilities: np.ndarray, labels: np.ndarray, neutral_band: float = 0.0) -> dict:
    """Per-asset confusion-matrix counts from (n_samples, n_assets)
    predicted probabilities and (n_samples, n_assets) realized binary
    labels: predicted positive iff probability > 0.5 + neutral_band,
    negative iff probability < 0.5 - neutral_band; anything in between
    (including probabilities snapped to exactly 0.5 by apply_neutral_band)
    is an ABSTENTION - excluded from tp/fp/tn/fn entirely and counted in
    "abstained" instead.

    Returns a dict of (n_assets,) int arrays: "tp"/"fp"/"tn"/"fn" plus
    "abstained" and "n_decided" (= tp+fp+tn+fn). With neutral_band=0 this
    reduces exactly to the classic 2x2 matrix over every sample.
    """
    predicted_positive = probabilities > 0.5 + neutral_band
    predicted_negative = probabilities < 0.5 - neutral_band
    decided = predicted_positive | predicted_negative
    actual_positive = labels > 0.5
    tp = (predicted_positive & actual_positive).sum(axis=0)
    fp = (predicted_positive & ~actual_positive).sum(axis=0)
    tn = (predicted_negative & ~actual_positive).sum(axis=0)
    fn = (predicted_negative & actual_positive).sum(axis=0)
    abstained = (~decided).sum(axis=0)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "abstained": abstained, "n_decided": tp + fp + tn + fn}


def confusion_matrix_metrics(
    probabilities: np.ndarray, labels: np.ndarray, eps: float = 1e-8, neutral_band: float = 0.0,
) -> dict:
    """Per-asset confusion-matrix counts PLUS the standard derived metrics,
    computed over DECIDED samples only (see confusion_matrix_counts -
    with neutral_band=0 every sample is decided, restoring the classic
    definitions):
      - accuracy (== hit rate among calls): (tp + tn) / n_decided
      - precision: tp / (tp + fp) - of days called positive, how many actually were
      - recall (sensitivity): tp / (tp + fn) - of decided actually-positive days, how many were called
      - specificity: tn / (tn + fp) - of decided actually-negative days, how many were called
      - f1: harmonic mean of precision and recall
      - coverage: n_decided / (n_decided + abstained) - how often the model
        makes a call at all; accuracy and coverage must be read TOGETHER
        (100% accuracy at 2% coverage is a very different claim from 55%
        at 80%).
    Every value is an (n_assets,) array. `eps` avoids /0 for an asset with
    a degenerate split (e.g. every label the same sign, or all abstained).
    """
    counts = confusion_matrix_counts(probabilities, labels, neutral_band=neutral_band)
    tp, fp, tn, fn = counts["tp"].astype(np.float64), counts["fp"].astype(np.float64), counts["tn"].astype(np.float64), counts["fn"].astype(np.float64)
    abstained = counts["abstained"].astype(np.float64)
    n_decided = tp + fp + tn + fn
    accuracy = (tp + tn) / (n_decided + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    coverage = n_decided / (n_decided + abstained + eps)
    return {
        **counts, "accuracy": accuracy, "precision": precision, "recall": recall,
        "specificity": specificity, "f1": f1, "coverage": coverage,
    }


def binary_cross_entropy_np(probabilities: np.ndarray, labels: np.ndarray, eps: float = 1e-7) -> float:
    """Mean binary cross-entropy (log loss) of `probabilities` against
    `labels`, over every element (plain numpy - no autograd needed here,
    this is a REPORTING/model-comparison metric, not a training loss).
    `eps`-clipped so a probability of exactly 0 or 1 can't produce -inf.

    Used instead of hit rate/accuracy for comparing already-trained
    restarts (see run_pipeline_multi_seed): accuracy saturates once every
    sample's sign is correct and can't discriminate any further between
    two restarts that are both "fully correct" but very differently
    confident/calibrated, while log loss (a proper scoring rule) keeps
    rewarding the better-calibrated one.
    """
    p = np.clip(probabilities, eps, 1 - eps)
    return float(-(labels * np.log(p) + (1 - labels) * np.log(1 - p)).mean())


# --------------------------------------------------------------------------
# Training: every asset's LSTM, fully independently
# --------------------------------------------------------------------------

def train_prediction_model(
    model: PredictionModel,
    X_train: torch.Tensor,
    z_labels_train: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float = 0.0,
    bce_weight: float = 1.0,
    X_val: torch.Tensor | None = None,
    z_labels_val: torch.Tensor | None = None,
) -> None:
    """Train every asset's LSTM FULLY INDEPENDENTLY - each asset gets its
    OWN `torch.optim.Adam` instance (own momentum/variance state, entirely
    separate from every other asset's optimizer), its OWN loss (computed
    from just that asset's own (mu, sigma) and label - never averaged or
    summed together with any other asset's loss before backward()), and
    its OWN best-validation-loss checkpoint restored independently at the
    end (asset A's own best epoch can differ from asset B's). The assets
    never shared WEIGHTS (see AssetLSTM/PredictionModel) - this decouples
    the OPTIMIZATION too, so nothing about how one asset is trained
    (gradient magnitude, Adam's adaptive per-parameter state, which epoch
    its own checkpoint gets restored from) depends on how many OTHER
    assets there are or how they happen to be doing. An asset linked to
    another pair's features via `cross_pairs` still trains this way - only
    what its LSTM READS is widened, never what optimizes it.

    Per asset i, each epoch:
        mu_i, sigma_i = model.assets[i](model.asset_input(X_train, i))  # dense, (n_train, lookback)
        loss_i = GaussianNLL(mu_i, sigma_i, z_label_i)                   [see gaussian_nll]
               + bce_weight * BCE(probit(mu_i/sigma_i), sign(z_label_i)) [see direction_bce]
        loss_i.backward(); optimizer_i.step()

    Applied to the FULL dense (batch, lookback) output, not just the
    window's last ("decision") day - every day in every window is its own
    supervised sample (see AssetLSTM's docstring), giving each epoch far
    more gradient signal than decision-day-only supervision would, and
    since consecutive windows slide by one day, a given calendar day is
    trained on repeatedly (once per window it falls inside) rather than
    just once as a decision day - so there's no need to separately
    up-weight it.

    NLL fits the full predictive DISTRIBUTION: mu is pulled toward each
    sample's conditional mean, and sigma toward the actual residual spread
    - per sample, so quiet-regime confidence and wild-regime uncertainty
    both become expressible. BCE is the anti-collapse term: a bare
    regression loss is minimized by the label's unconditional mean - ONE
    sign for every sample when the target is barely predictable, which is
    exactly the degenerate all-recall/zero-specificity confusion matrix -
    while BCE is minimized by matching each sample's OWN sign, forcing
    mu/sigma to spread across 0 wherever the features discriminate at all.
    `bce_weight=0` recovers pure distributional regression (collapse risk
    and all).

    Deterministic inputs: no input-noise regularization is applied here
    (unlike an earlier version of this code, which added fresh Gaussian
    noise to the standardized input window every epoch) - training data is
    used exactly as standardized. `weight_decay` is passed straight to
    each asset's own Adam as an L2 penalty.

    If `X_val`/`z_labels_val` are given, the same per-asset loss - but
    computed on the DECISION DAY ONLY, matching what's actually
    evaluated/reported (unlike the dense training loss above) - is tracked
    per asset after every epoch, and each asset's OWN state_dict from
    whichever epoch had ITS OWN lowest validation loss is restored
    independently at the end - not just whichever epoch `epochs` happened
    to end on, and deliberately NOT the epoch with the best validation HIT
    RATE (computed here from sign(mu), purely for logging): hit rate is a
    coarse, saturating metric (once every sample's sign is already
    correct, it cannot improve further no matter how much more
    confident/well-calibrated the predictions get), so selecting on it
    tends to freeze onto whichever epoch FIRST reached its ceiling - often
    very early - and discard every later epoch that kept reducing loss but
    couldn't move hit rate any higher.

    The epoch counter and log line are shared across all assets (one line
    per epoch, an average across assets) purely for readable progress
    reporting - it does not mean the assets are optimized together;
    reread the per-asset loop above for what actually happens.
    """
    n_assets = model.n_assets
    optimizers = [
        torch.optim.Adam(model.assets[i].parameters(), lr=lr, weight_decay=weight_decay)
        for i in range(n_assets)
    ]
    track_val = X_val is not None and z_labels_val is not None
    best_val_loss = [float("inf")] * n_assets
    best_state: list[dict | None] = [None] * n_assets
    val_losses: list[float | None] = [None] * n_assets
    val_hit_rates: list[float | None] = [None] * n_assets

    def _restore_best() -> None:
        for i in range(n_assets):
            if best_state[i] is not None:
                model.assets[i].load_state_dict(best_state[i])
        mean_best = sum(v for v in best_val_loss if v != float("inf")) / n_assets
        logger.info("Restored each asset's own best-validation-loss checkpoint (mean val loss %.4f)", mean_best)

    model.train()
    try:
        for epoch in range(1, epochs + 1):
            _check_stop()
            train_losses = []
            train_hits = []
            for i in range(n_assets):
                optimizers[i].zero_grad()
                x_i = model.asset_input(X_train, i)
                mu_i, sigma_i = model.assets[i](x_i)  # each (n_train, lookback)
                target_i = z_labels_train[:, :, i]
                loss_i = gaussian_nll(mu_i, sigma_i, target_i) + bce_weight * direction_bce(mu_i, sigma_i, target_i)
                loss_i.backward()
                optimizers[i].step()
                train_losses.append(loss_i.item())
                with torch.no_grad():
                    train_hits.append(float(((mu_i[:, -1] > 0) == (target_i[:, -1] > 0)).float().mean()))
            train_loss = sum(train_losses) / n_assets
            train_hit_rate = sum(train_hits) / n_assets

            if track_val:
                model.eval()
                with torch.no_grad():
                    for i in range(n_assets):
                        x_i_val = model.asset_input(X_val, i)
                        val_mu_i, val_sigma_i = model.assets[i](x_i_val)
                        val_mu_dec, val_sigma_dec = val_mu_i[:, -1], val_sigma_i[:, -1]
                        val_target_i = z_labels_val[:, -1, i]
                        val_loss_i = float(
                            gaussian_nll(val_mu_dec, val_sigma_dec, val_target_i)
                            + bce_weight * direction_bce(val_mu_dec, val_sigma_dec, val_target_i)
                        )
                        val_losses[i] = val_loss_i
                        val_hit_rates[i] = float(((val_mu_dec > 0) == (val_target_i > 0)).float().mean())
                        if val_loss_i < best_val_loss[i]:
                            best_val_loss[i] = val_loss_i
                            best_state[i] = copy.deepcopy(model.assets[i].state_dict())
                model.train()

            if epoch == 1 or epoch % max(epochs // 10, 1) == 0:
                val_loss_mean = sum(val_losses) / n_assets if track_val else None
                val_hit_mean = sum(val_hit_rates) / n_assets if track_val else None
                val_msg = f" | val loss {val_loss_mean:.4f} | val hit rate {val_hit_mean:.4f}" if track_val else ""
                logger.info(
                    "epoch %d/%d - train loss %.4f | train hit rate %.4f%s",
                    epoch, epochs, train_loss, train_hit_rate, val_msg,
                )
                _report_epoch("train", epoch, epochs, train_loss, train_hit_rate, val_loss_mean, val_hit_mean)
    except TrainingStopped:
        logger.info("Training stopped early at epoch %d/%d", epoch, epochs)
        if track_val:
            _restore_best()
        raise

    if track_val:
        _restore_best()


# --------------------------------------------------------------------------
# End-to-end pipeline
# --------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """Everything needed to score a trained PredictionModel: predicted
    joint probabilities, realized direction labels, per-asset hit rate,
    and raw next-day returns (for a cumulative-return chart), for all
    three splits.
    """

    model: PredictionModel
    pairs: list[str]
    lookback: int
    dates_train: pd.DatetimeIndex
    dates_val: pd.DatetimeIndex
    dates_test: pd.DatetimeIndex
    # PredictionModel's own predicted probability per asset, per window's
    # decision day - P(asset positive over the next direction_horizon
    # days). (n_train/n_val/n_test, n_assets), in (0, 1).
    probabilities_train: np.ndarray
    probabilities_val: np.ndarray
    probabilities_test: np.ndarray
    # Realized ground truth for the SAME decision day - what probabilities_*
    # is compared against for hit rate/confusion matrices.
    direction_labels_train: np.ndarray
    direction_labels_val: np.ndarray
    direction_labels_test: np.ndarray
    # The SAME decision day's realized z-score, CONTINUOUS (not reduced to
    # sign) - the "actual" half of the forecast-vs-actual distribution
    # comparison (see api/server.py's _distribution_payload).
    z_labels_train: np.ndarray
    z_labels_val: np.ndarray
    z_labels_test: np.ndarray
    # The model's own predicted (mu, sigma) for that SAME decision day -
    # mu raw, sigma CALIBRATED (sigma * sigma_hat, see below) - i.e. the
    # model's own claimed N(mu, sigma^2) belief about that day's z-score.
    # The "forecasted" half of the distribution comparison: sampling once
    # from each row's own N(mu, sigma) and histogramming the result
    # alongside z_labels_* shows whether the model's predictive
    # distributions, in aggregate, actually look like the realized outcomes.
    mu_train: np.ndarray
    mu_val: np.ndarray
    mu_test: np.ndarray
    sigma_train: np.ndarray
    sigma_val: np.ndarray
    sigma_test: np.ndarray
    # Fraction of samples where (probability > 0.5) matched the realized
    # label, per asset - see confusion_matrix_metrics for the fuller
    # breakdown (precision/recall/specificity/F1), computed on demand from
    # probabilities_*/direction_labels_* rather than stored redundantly.
    hit_rate_train: np.ndarray
    hit_rate_val: np.ndarray
    hit_rate_test: np.ndarray
    # RAW (unstandardized) single-day-ahead log returns - not a training
    # target, kept purely so a caller can plot each asset's own cumulative
    # return path (cumsum), colored by whether that day's prediction hit.
    next_returns_train: np.ndarray
    next_returns_val: np.ndarray
    next_returns_test: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    # Calibration scale (see evaluate_prediction_model): per-asset residual
    # std of the STANDARDIZED residual (z - mu) / sigma, estimated on
    # VALIDATION ONLY (~1.0 when the per-sample sigmas are already
    # well-calibrated), used as probit(mu / (sigma * sigma_hat)).
    # Persisted alongside the model (see
    # PredictionModel.save_model/save_to_db) for live single-window
    # inference, where no validation set exists to refit it.
    sigma_hat: np.ndarray
    # Half-width of the abstention region around p=0.5 that hit_rate_*
    # above was computed with (see apply_neutral_band/_decided_hit_rate) -
    # probabilities_* themselves are stored RAW (never snapped), so a
    # caller can recompute a confusion matrix/hit rate/colored-return
    # chart for a DIFFERENT band from this same result without retraining
    # (see api/server.py, which ships this alongside raw probabilities and
    # realized labels for exactly that purpose).
    neutral_band: float


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
    # validation, which best-epoch checkpoint selection DOES see). Of what
    # remains, `train_frac` is train and the rest is validation.
    "test_frac": 0.1,
    # How many days AHEAD the z-score label (see make_sequences) looks:
    # an asset's CUMULATIVE log return over the next `direction_horizon`
    # days, expressed in trailing-volatility standard deviations - what
    # training's Huber loss regresses against (see train_prediction_model),
    # and what hit rate/confusion matrices are computed from (via the
    # label's sign).
    "direction_horizon": 5,
    # Trailing window (days) rolling vol/skew/kurtosis (see
    # rolling_moment_features) are computed over - one of the possible
    # input features (see "features" below), and the SAME window the
    # z-score label's own volatility normalization uses (see
    # make_sequences), so the label's scale matches what the model itself
    # can observe about recent volatility, regardless of whether "vol"
    # itself is a selected input feature.
    "rolling_stats_window": 20,
    # Which per-pair input channels to build (see build_feature_dataframe/
    # FEATURE_CATALOG for the full list: "log_return", "vol", "skew",
    # "kurt", "carry", "cma"). "carry" adds the interest-rate differential
    # (FRED via data/rates_downloader.py); "cma" adds one channel PER
    # (short, long) window pair in "cma_windows" below (a trailing
    # moving-average crossover / trend signal in return space).
    "features": list(DEFAULT_FEATURES),
    "cma_windows": [],  # e.g. [[10, 50], [20, 100]] - only used if "cma" is in "features"
    # Architecture: N independent per-asset LSTMs, each followed by a
    # CAUSAL self-attention layer over the time axis (see AssetLSTM/
    # PredictionModel) - no parameters shared between assets. By DEFAULT
    # every pair's own LSTM sees ONLY its own features - fully
    # independent, no cross-asset mixing. `cross_pairs` (a {pair:
    # [other_pair, ...]} dict) opts specific pairs INTO also seeing
    # specific other pairs' full feature blocks, e.g.
    # {"EURUSD": ["GBPUSD", "USDJPY"]} - EURUSD's LSTM then also sees
    # GBPUSD's and USDJPY's channels (its own features are always
    # included regardless). Fully deterministic - no NoisyNet head, no
    # input-noise regularization.
    "cross_pairs": {},
    "hidden_size": 16,
    "num_layers": 1,
    "dropout": 0.1,
    # Attention heads per asset's causal self-attention layer (see
    # AssetLSTM) - hidden_size must be divisible by this.
    "n_attn_heads": 4,
    # Joint training (see train_prediction_model): every asset's LSTM
    # trained together, one optimizer over every parameter, one combined
    # loss - Gaussian NLL on (mu, sigma) + `bce_weight` * BCE on the
    # implied direction probability probit(mu/sigma), applied densely
    # over every day in the window, not just the decision day.
    "epochs": 300,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    # Weight of the BCE direction term (see direction_bce) - the
    # anti-collapse term that forces mu/sigma to separate up-days from
    # down-days per sample instead of settling on the label's
    # unconditional mean (one sign for everything). 0 disables it
    # (pure distributional regression). Accepts EITHER a single float OR a
    # list to SWEEP (e.g. [1.0, 1.5, 1.75, 2.0, 3.0]) - see
    # run_pipeline_multi_seed, which trains every value under every seed
    # (same initial weights across the sweep, so only the value differs)
    # and keeps whichever validated best, per seed and then overall.
    "bce_weight": 1.0,
    # Neutral band (see apply_neutral_band): probabilities within
    # 0.5 +/- this half-width are snapped to exactly 0.5 - the model
    # ABSTAINS instead of making a near-coin-flip call. Hit rate/
    # confusion-matrix metrics are then computed over decided samples
    # only, alongside a `coverage` metric (how often the model speaks).
    # 0.0 disables abstention entirely.
    "neutral_band": 0.05,
    # Multi-seed restarts: train `n_seeds` independent PredictionModels
    # and keep whichever restart had the lowest validation log loss.
    "n_seeds": 1,
    # Compute device (see get_device): "auto" picks Apple Silicon's Metal
    # backend (MPS) if available, else CUDA, else CPU.
    "device": "auto",
    # Persistence: accepts EITHER a local .pt file path OR a
    # quant.model_registry name (see load_prediction_model_auto below);
    # save_db additionally persists whatever gets trained/loaded to
    # Postgres under a deterministic name.
    "load_model": None,
    "save_db": False,
    "model_description": "",
}


def prediction_model_name(args: argparse.Namespace) -> str:
    """Deterministic quant.model_registry name for a PredictionModel
    trained with `args` - built from the characteristics that actually
    change the trained model, so the same configuration always maps to
    the same name and re-saving under it is a natural update.
    """
    from data.model_registry import build_model_name

    return build_model_name(
        "prediction",
        pairs=sorted(args.pairs),
        lookback=args.lookback,
        direction_horizon=getattr(args, "direction_horizon", 5),
        features=sorted(getattr(args, "features", None) or DEFAULT_FEATURES),
        hidden_size=args.hidden_size,
    )


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
    # Per-day, per-asset CONTINUOUS z-score labels (see make_sequences) -
    # (n, lookback, n_assets). z_labels_*[:, -1, :] is the decision day's
    # own label - what hit-rate/confusion-matrix derive a binary direction
    # from via sign(), and what evaluate_prediction_model's calibration is
    # fit against.
    z_labels_train: torch.Tensor
    z_labels_val: torch.Tensor
    z_labels_test: torch.Tensor
    next_returns_train: np.ndarray
    next_returns_val: np.ndarray
    next_returns_test: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    device: torch.device


def _prepare_data(
    args: argparse.Namespace,
    x_mean: np.ndarray | None = None,
    x_std: np.ndarray | None = None,
    pairs: list[str] | None = None,
    lookback: int | None = None,
    features: list[str] | None = None,
    cma_windows: list | None = None,
) -> _PreparedData:
    """Load data (via db.py), build sequences, split by time, and
    standardize - everything a training (or inference) run needs that
    doesn't depend on the model's random seed.

    If `x_mean`/`x_std` are given (loading a previously-saved model), they
    are used as-is instead of being freshly fit - inference must
    standardize new data exactly the way the loaded model was trained.

    If `pairs`/`lookback`/`features`/`cma_windows` are given, they OVERRIDE
    args.pairs/args.lookback/args.features/args.cma_windows - used when
    loading a saved model, whose checkpoint carries the exact values it
    was trained with (see PredictionModel._from_checkpoint/load_pipeline
    below); a loaded model's own stored values must win over whatever a
    caller passes in.
    """
    if pairs is None:
        pairs = list(dict.fromkeys(args.pairs))  # de-duplicate, keep order
    if lookback is None:
        lookback = args.lookback
    if features is None:
        features = getattr(args, "features", None) or list(DEFAULT_FEATURES)
    if cma_windows is None:
        cma_windows = getattr(args, "cma_windows", None) or []

    logger.info("Loading %s via db.py", pairs)
    prices = load_close_prices(pairs, years=args.years)
    returns = to_log_returns(prices)

    direction_horizon = getattr(args, "direction_horizon", 5) or 5
    rolling_stats_window = getattr(args, "rolling_stats_window", 20) or 20
    min_rows = lookback + direction_horizon + rolling_stats_window
    if len(returns) < min_rows:
        raise ValueError(
            f"Only {len(returns)} days of history available for {pairs}, but this model needs at least "
            f"{min_rows} (lookback {lookback} + direction_horizon {direction_horizon} + "
            f"rolling_stats_window {rolling_stats_window}) - increase 'years' to fetch more history."
        )

    n_channels = n_channels_per_pair(features, cma_windows)
    feature_returns = build_feature_dataframe(returns, pairs, features, rolling_stats_window, cma_windows, args.years)

    X, next_returns, z_labels, dates = make_sequences(
        returns, lookback=lookback, feature_returns=feature_returns, direction_horizon=direction_horizon,
        rolling_stats_window=rolling_stats_window,
    )

    # Chronological 3-way split: the most recent `test_frac` fraction of
    # ALL sequences is carved off FIRST as the test set - held out
    # completely from every training/model-selection decision - then
    # `train_frac` splits whatever REMAINS into train/validation.
    test_frac = getattr(args, "test_frac", 0.0) or 0.0
    n_total = len(X)
    n_test = int(n_total * test_frac)
    n_remaining = n_total - n_test
    n_train = int(n_remaining * args.train_frac)

    # Purge/embargo (Lopez de Prado, "purged CV"): consecutive sequences
    # are stride-1, so a sample's forward label window ([d+1, d+H], see
    # _forward_zscore_labels) and its lookback INPUT window overlap
    # heavily with its neighbors'. Right at a split boundary this means
    # the last training sample's label is computed from days that are
    # also inside the first validation sample's lookback window (and
    # vice versa at the val/test boundary) - i.e. information the model
    # was trained to predict leaks into what validation "sees" as history,
    # optimistically biasing the validation loss that drives checkpoint
    # and seed selection. Dropping `lookback + direction_horizon` samples
    # at each boundary (from the END of the earlier split, entirely
    # excluded - not reassigned to the later split) removes every such
    # overlap.
    purge_gap = lookback + direction_horizon
    train_end = max(n_train - purge_gap, 0)
    val_end = max(n_remaining - purge_gap, n_train)
    if train_end == 0:
        raise ValueError(
            f"Purging {purge_gap} samples (lookback {lookback} + direction_horizon {direction_horizon}) at each "
            f"split boundary left an empty training set out of {n_train} pre-purge train sequences - increase "
            f"'years' to fetch more history, or reduce lookback/direction_horizon/train_frac."
        )

    X_full_train, X_full_val, X_full_test = X[:train_end], X[n_train:val_end], X[n_remaining:]
    next_returns_train = next_returns[:train_end]
    next_returns_val = next_returns[n_train:val_end]
    next_returns_test = next_returns[n_remaining:]
    z_labels_train_raw = z_labels[:train_end]
    z_labels_val_raw = z_labels[n_train:val_end]
    z_labels_test_raw = z_labels[n_remaining:]
    dates_train, dates_val, dates_test = dates[:train_end], dates[n_train:val_end], dates[n_remaining:]

    if x_mean is None or x_std is None:
        x_mean, x_std = standardize(X_full_train, axis=(0, 1))  # TRAIN-only stats

    device = get_device(getattr(args, "device", "auto"))
    X_train = torch.tensor((X_full_train - x_mean) / x_std, device=device)
    X_val = torch.tensor((X_full_val - x_mean) / x_std, device=device)
    X_test = torch.tensor((X_full_test - x_mean) / x_std, device=device)

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
        z_labels_train=torch.tensor(z_labels_train_raw, device=device),
        z_labels_val=torch.tensor(z_labels_val_raw, device=device),
        z_labels_test=torch.tensor(z_labels_test_raw, device=device),
        next_returns_train=next_returns_train,
        next_returns_val=next_returns_val,
        next_returns_test=next_returns_test,
        x_mean=x_mean,
        x_std=x_std,
        device=device,
    )


def _decided_hit_rate(probabilities: np.ndarray, labels: np.ndarray, neutral_band: float) -> np.ndarray:
    """Per-asset fraction of DECIDED samples (|p - 0.5| > band; see
    apply_neutral_band) whose direction call matched the realized label.
    Assets with zero decided samples report 0.5 (chance) rather than
    dividing by zero - read alongside `coverage` from
    confusion_matrix_metrics, which is 0 in that case."""
    predicted_positive = probabilities > 0.5 + neutral_band
    predicted_negative = probabilities < 0.5 - neutral_band
    decided = predicted_positive | predicted_negative
    correct = (predicted_positive & (labels > 0.5)) | (predicted_negative & (labels <= 0.5))
    n_decided = decided.sum(axis=0)
    return np.where(n_decided > 0, correct.sum(axis=0) / np.maximum(n_decided, 1), 0.5).astype(np.float32)


def evaluate_prediction_model(
    model: PredictionModel, data: _PreparedData, neutral_band: float = 0.0,
) -> PredictionResult:
    """Run an already-trained-or-loaded model (eval mode, no grad) over all
    three splits (train/validation/test - test is empty when test_frac=0),
    and package hit rate + everything needed for a confusion matrix and a
    per-asset cumulative-return chart. Shared by the train path (after
    fitting) and the load path (skips fitting entirely).

    The model's own forward() returns (mu, sigma), each dense over every
    day in the window (see PredictionModel's docstring) - only the LAST
    timestep (the "decision day") is used here, matching what hit
    rate/confusion matrices/live inference all report. The probability is
    probit(mu / (sigma * sigma_hat)):
      - mu / sigma is the model's own per-sample signal-to-noise ratio -
        confidence varies day by day, unlike a single global residual
        scale ever could;
      - sigma_hat is a residual GLOBAL calibration factor: the std of the
        standardized residual (z - mu) / sigma, estimated ONCE on the
        VALIDATION split only (never train - calibration would fit the
        same noise the model overfit; never test - that would leak the
        held-out set into a quantity that ships with the model). If the
        NLL-trained sigmas are already honest, sigma_hat ~ 1 and this is
        a no-op; if they are collectively over/under-confident, sigma_hat
        corrects the shared factor.

    `neutral_band` is used ONLY to compute hit_rate_*/PredictionResult's
    own reported metrics (see _decided_hit_rate) - it is pure
    POSTPROCESSING, never part of training (see train_prediction_model,
    which never receives it), so it's applied here, after the model is
    already fixed, not baked into it. probabilities_train/val/test
    themselves are stored RAW (unsnapped) - callers that want a specific
    band's abstention behavior (a confusion matrix, a colored return
    chart) apply it themselves via apply_neutral_band/
    confusion_matrix_metrics, which is exactly what lets a caller (e.g.
    the frontend) recompute those for a DIFFERENT band without retraining
    or even a new evaluation pass - see api/server.py, which ships raw
    probabilities plus realized labels precisely so this can happen
    client-side.
    """
    model.eval()
    with torch.no_grad():
        # Decision day only (the window's last timestep) - see this
        # function's own docstring.
        mu_train, sig_train = (t[:, -1, :] for t in model(data.X_train))
        mu_val, sig_val = (t[:, -1, :] for t in model(data.X_val))
        mu_test, sig_test = (t[:, -1, :] for t in model(data.X_test))

    sigma_floor = 1e-3
    n_assets = len(data.pairs)
    if mu_val.shape[0] > 0:
        # Standardized residual: (realized - mu) / sigma. Std ~ 1 already
        # if the per-sample sigmas are honest.
        resid_val = ((data.z_labels_val[:, -1, :] - mu_val) / sig_val).cpu().numpy()
        sigma_hat = np.maximum(resid_val.std(axis=0), sigma_floor).astype(np.float32)
    else:
        # No validation samples (e.g. purge_gap ate the whole split on a
        # tiny dataset) - fall back to trusting the model's own sigmas
        # rather than dividing by an undefined std.
        sigma_hat = np.ones(n_assets, dtype=np.float32)
    model.sigma_hat = sigma_hat
    model.neutral_band = float(neutral_band)

    # RAW probabilities - deliberately NOT band-snapped (see this
    # function's own docstring). confusion_matrix_metrics/_decided_hit_rate
    # below apply the band via a threshold check, not by requiring an
    # already-snapped input, so this is equivalent either way for THIS
    # function's own hit_rate_*/PredictionResult output - it's storing
    # probabilities_* raw that actually matters, so any caller can later
    # recompute for a different band.
    sigma_hat_t = torch.as_tensor(sigma_hat, device=mu_train.device, dtype=mu_train.dtype)
    with torch.no_grad():
        probs_train = probit(mu_train / (sig_train * sigma_hat_t)).cpu().numpy()
        probs_val = probit(mu_val / (sig_val * sigma_hat_t)).cpu().numpy()
        probs_test = probit(mu_test / (sig_test * sigma_hat_t)).cpu().numpy()

    # The decision day's own label only (the last timestep in each
    # window) - CONTINUOUS (z_labels_*) and reduced to sign (labels_*).
    # Sign is what the probability is actually compared against (probit is
    # monotone with probit(0) = 0.5, so p > 0.5 iff mu > 0, matching
    # sign(z_labels) exactly); the continuous value is the "actual" half
    # of the forecast-vs-actual distribution comparison (see
    # PredictionResult's docstring on z_labels_train).
    z_labels_train = data.z_labels_train[:, -1, :].cpu().numpy()
    z_labels_val = data.z_labels_val[:, -1, :].cpu().numpy()
    z_labels_test = data.z_labels_test[:, -1, :].cpu().numpy()
    labels_train = (z_labels_train > 0).astype(np.float32)
    labels_val = (z_labels_val > 0).astype(np.float32)
    labels_test = (z_labels_test > 0).astype(np.float32)

    hit_rate_train = _decided_hit_rate(probs_train, labels_train, neutral_band)
    hit_rate_val = _decided_hit_rate(probs_val, labels_val, neutral_band)
    hit_rate_test = _decided_hit_rate(probs_test, labels_test, neutral_band)

    # CALIBRATED sigma (sigma * sigma_hat) - the model's own claimed
    # N(mu, sigma^2) belief, in the same scale probit() actually uses -
    # see PredictionResult's docstring on mu_train/sigma_train.
    sig_train_calibrated = (sig_train * sigma_hat_t).cpu().numpy()
    sig_val_calibrated = (sig_val * sigma_hat_t).cpu().numpy()
    sig_test_calibrated = (sig_test * sigma_hat_t).cpu().numpy()
    mu_train_np = mu_train.cpu().numpy()
    mu_val_np = mu_val.cpu().numpy()
    mu_test_np = mu_test.cpu().numpy()

    return PredictionResult(
        model=model,
        pairs=data.pairs,
        lookback=data.lookback,
        dates_train=data.dates_train,
        dates_val=data.dates_val,
        dates_test=data.dates_test,
        probabilities_train=probs_train,
        probabilities_val=probs_val,
        probabilities_test=probs_test,
        direction_labels_train=labels_train,
        direction_labels_val=labels_val,
        direction_labels_test=labels_test,
        z_labels_train=z_labels_train,
        z_labels_val=z_labels_val,
        z_labels_test=z_labels_test,
        mu_train=mu_train_np,
        mu_val=mu_val_np,
        mu_test=mu_test_np,
        sigma_train=sig_train_calibrated,
        sigma_val=sig_val_calibrated,
        sigma_test=sig_test_calibrated,
        hit_rate_train=hit_rate_train,
        hit_rate_val=hit_rate_val,
        hit_rate_test=hit_rate_test,
        next_returns_train=data.next_returns_train,
        next_returns_val=data.next_returns_val,
        next_returns_test=data.next_returns_test,
        x_mean=data.x_mean,
        x_std=data.x_std,
        sigma_hat=sigma_hat,
        neutral_band=float(neutral_band),
    )


def _train_and_evaluate(data: _PreparedData, args: argparse.Namespace) -> PredictionResult:
    """Train one PredictionModel (whatever random seed is currently set) on
    already-prepared data - every asset's LSTM trained together, one
    optimizer over every parameter (see train_prediction_model) - and
    evaluate it on all three splits.
    """
    model = PredictionModel(
        n_assets=len(data.pairs),
        pairs=data.pairs,
        n_channels=data.n_channels,
        hidden_size=args.hidden_size,
        num_layers=getattr(args, "num_layers", 1),
        dropout=args.dropout,
        n_attn_heads=getattr(args, "n_attn_heads", 4),
        cross_pairs=getattr(args, "cross_pairs", None),
    ).to(data.device)

    train_prediction_model(
        model, data.X_train, data.z_labels_train,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        bce_weight=getattr(args, "bce_weight", 1.0),
        X_val=data.X_val, z_labels_val=data.z_labels_val,
    )

    return evaluate_prediction_model(model, data, neutral_band=getattr(args, "neutral_band", 0.0))


def load_pipeline(args: argparse.Namespace) -> PredictionResult:
    """Load a previously-trained PredictionModel from `args.load_model` (a
    local file path OR a quant.model_registry name - see
    load_prediction_model_auto) and evaluate it on freshly-loaded data - no
    training happens at all. The checkpoint carries its own x_mean/x_std,
    ordered FX pairs, sequence length, and feature selection, so new data
    is standardized/windowed identically without needing to reconstruct
    the original training split.
    """
    model = load_prediction_model_auto(args.load_model)
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
        features=getattr(model, "features", None), cma_windows=getattr(model, "cma_windows", None),
    )
    # load_prediction_model_auto reconstructs the checkpoint on CPU
    # (torch.load(..., map_location="cpu")) regardless of what device it
    # was originally trained on - move it to match data's device (see
    # _prepare_data/get_device) before running it against data.X_train/etc.
    model = model.to(data.device)
    # The checkpoint's own band wins over the config's: metrics must
    # reflect how the model was actually calibrated/saved.
    return evaluate_prediction_model(
        model, data, neutral_band=getattr(model, "neutral_band", getattr(args, "neutral_band", 0.0)),
    )


def run_pipeline(args: argparse.Namespace) -> PredictionResult:
    """Load data (via db.py), build sequences, train every asset's LSTM,
    and evaluate realized hit rate on all three splits. Single run,
    whatever the ambient random seed is - see run_pipeline_multi_seed()
    for restarts.
    """
    if args.load_model:
        return load_pipeline(args)
    return _train_and_evaluate(_prepare_data(args), args)


def run_pipeline_multi_seed(args: argparse.Namespace) -> PredictionResult:
    """Train `n_seeds` independent PredictionModels - each seed optionally
    SWEPT over every `bce_weight` value given (pass a list, e.g.
    `[1.0, 1.5, 1.75, 2.0, 3.0]`, instead of a single float) - and keep
    the best, in two selection stages:

    1. Per seed: if `bce_weight` is a list, train one model per value,
       with `torch.manual_seed(seed)` reset right before EACH one - so
       every lambda in the sweep starts from the SAME initial weights and
       sees the SAME input-noise draws under that seed, isolating the
       lambda's own effect from initialization luck - and keep whichever
       value validated best for that seed.
    2. Across seeds: keep whichever seed's winner validated best overall.

    Both stages select by validation BCE loss (see below), never
    validation hit rate.

    Training is non-convex in the LSTMs' parameters regardless of loss
    function, so different random initializations can land in meaningfully
    different local optima - training several and keeping the best-
    validated one is a standard, cheap way to get a more robust result
    than trusting a single run. `bce_weight` (see direction_bce) trades
    off distributional fit (Gaussian NLL) against anti-collapse sign-
    separation (BCE); sweeping it lets the DATA pick that tradeoff via
    validation performance, rather than committing to one value a priori.

    Selecting by validation loss rather than validation hit rate for the
    same reason train_prediction_model does (see its own docstring): hit
    rate saturates once every sample's sign is correct, so two runs that
    are both "fully correct" but very differently confident/calibrated
    would otherwise be indistinguishable (or worse, the LESS confident one
    could win on a coin-flip tie) - log loss keeps discriminating between
    them.

    If --load-model is set, restarts don't apply at all - there's nothing
    to train, so this just delegates to load_pipeline().
    """
    if args.load_model:
        return load_pipeline(args)

    n_seeds = getattr(args, "n_seeds", 1) or 1
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1, got {n_seeds}")

    bce_weight = getattr(args, "bce_weight", 1.0)
    bce_weights = list(bce_weight) if isinstance(bce_weight, (list, tuple)) else [bce_weight]
    if not bce_weights:
        raise ValueError("bce_weight sweep must have at least one value")
    n_lambdas = len(bce_weights)

    data = _prepare_data(args)  # load/split/standardize once, reused by every seed/lambda combo

    def _val_bce(result: PredictionResult) -> float:
        return binary_cross_entropy_np(result.probabilities_val, result.direction_labels_val)

    seed_winners = []
    for seed in range(n_seeds):
        candidates = []
        for lambda_idx, bw in enumerate(bce_weights):
            torch.manual_seed(seed)  # reset per lambda: same init/noise draws, only bw differs
            logger.info(
                "--- restart %d/%d (seed=%d), bce_weight %s (%d/%d) ---",
                seed + 1, n_seeds, seed, bw, lambda_idx + 1, n_lambdas,
            )
            lambda_args = argparse.Namespace(**{**vars(args), "bce_weight": bw})
            candidates.append(_train_and_evaluate(data, lambda_args))

        if n_lambdas == 1:
            seed_best = candidates[0]
        else:
            best_lambda_idx, seed_best = min(enumerate(candidates), key=lambda item: _val_bce(item[1]))
            logger.info(
                "Seed %d: best bce_weight %s of %s (validation BCE %.4f)",
                seed, bce_weights[best_lambda_idx], bce_weights, _val_bce(seed_best),
            )
        seed_winners.append(seed_best)

    if len(seed_winners) == 1:
        return seed_winners[0]

    best_idx, best = min(enumerate(seed_winners), key=lambda item: _val_bce(item[1]))
    logger.info(
        "Best of %d restarts: #%d (validation BCE %.4f, mean validation hit rate %.3f)",
        len(seed_winners), best_idx, _val_bce(best), float(best.hit_rate_val.mean()),
    )
    return best


def print_hit_rates(result: PredictionResult) -> None:
    """Log per-asset hit rate for all three splits. Used by main.py after
    training/loading."""
    for pair_idx, pair in enumerate(result.pairs):
        logger.info(
            "%s: train hit rate %.3f | val hit rate %.3f | test hit rate %.3f",
            pair, result.hit_rate_train[pair_idx], result.hit_rate_val[pair_idx], result.hit_rate_test[pair_idx],
        )
    logger.info(
        "Mean across assets: train %.3f | val %.3f | test %.3f",
        result.hit_rate_train.mean(), result.hit_rate_val.mean(), result.hit_rate_test.mean(),
    )
