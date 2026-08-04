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
from models.portfolio_pnl import (
    DEFAULT_COST_BPS,
    DEFAULT_COV_WINDOW,
    DEFAULT_TARGET_VOL,
    MAX_VOL_SCALE,
    TRADING_DAYS_PER_YEAR,
    compute_portfolio,
    precompute_risk_parity,
)

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
    Callable[..., None] | None
] = contextvars.ContextVar("_epoch_report_callback", default=None)


def _report_epoch(
    stage: str,
    epoch: int,
    epochs: int,
    train_loss: float,
    train_hit_rate: float,
    val_loss: float | None = None,
    val_hit_rate: float | None = None,
    train_sharpe: float | None = None,
    val_sharpe: float | None = None,
    best_score: float | None = None,
) -> None:
    """Call the registered interim-results callback, if any, with this
    epoch's training loss and mean (decision-day) hit rate, plus
    validation versions of both when available - a no-op when nothing has
    registered one (e.g. the CLI / main.py path). `stage` is always
    "train" now that training is a single joint phase - kept as a field
    purely for payload-shape stability with older consumers.

    `train_sharpe`/`val_sharpe` (only when the optional Sharpe phase or
    checkpoint_metric="sharpe" is actually computing them that epoch - see
    train_prediction_model) are the SAME train/val rolling-Sharpe numbers
    already in the console log line, surfaced here too so a live UI can
    plot them. `best_score` is the RUNNING best checkpoint-selection score
    found so far this run, in human-legible units matching
    `checkpoint_metric` (best hit rate/Sharpe SO FAR if that metric is
    selected - higher is better; best val loss so far if "val_loss" -
    lower is better) - None until at least one epoch has updated a
    checkpoint (i.e. no validation split, or the very first logged epoch).
    """
    callback = _epoch_report_callback.get()
    if callback is not None:
        callback(
            stage, epoch, epochs, train_loss, train_hit_rate, val_loss, val_hit_rate,
            train_sharpe=train_sharpe, val_sharpe=val_sharpe, best_score=best_score,
        )


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

#: Default `cutoff_date` (see _resolve_cutoff_date): None, meaning "no
#: cutoff configured" - resolved dynamically to whatever `date.today()`
#: actually is at call time, so it's always today's real date rather than
#: a stale value baked in once at import/config-write time.
DEFAULT_CUTOFF_DATE: str | None = None


def _resolve_cutoff_date(cutoff_date: str | date | None) -> date:
    """Turn a caller-supplied cutoff (an ISO "YYYY-MM-DD" string, a `date`,
    or None) into the actual END date every data-loading function below
    should query up to.

    This is the single mechanism that guarantees a walk-forward backtest
    never trains/validates on data the caller considers "not yet known":
    load_close_prices/load_carry use this instead of `date.today()`
    directly, so passing e.g. cutoff_date="2024-06-30" makes the ENTIRE
    pipeline (prices, returns, carry, features, sequences, splits) behave
    as if today were 2024-06-30 - no later row is ever fetched from
    Postgres, so it cannot leak into training, validation, or test.

    None (DEFAULT_CUTOFF_DATE, "no cutoff configured") resolves to
    `date.today()` - i.e. use all data available up to today, computed
    fresh on every call rather than a fixed sentinel date. A cutoff in the
    future is capped at today the same way (`min(..., date.today())`),
    since "not yet known" can never mean "later than today".
    """
    if cutoff_date is None:
        return date.today()
    if isinstance(cutoff_date, str):
        cutoff_date = date.fromisoformat(cutoff_date)
    return min(cutoff_date, date.today())


def load_close_prices(
    symbols: list[str], years: int, cutoff_date: str | date | None = DEFAULT_CUTOFF_DATE,
) -> pd.DataFrame:
    """Load daily close prices for `symbols` from Postgres via db.py.

    db.py's `get_time_series` is the single source of truth here. If a
    symbol has no rows in Postgres yet (e.g. the very first run), it is
    downloaded via FXDownloader and upserted (`upsert_pairs`), then the
    Postgres read is repeated - so the model always ends up training on
    whatever is in the database, not on a one-off in-memory download.

    `cutoff_date` (see _resolve_cutoff_date) caps the END of the fetched
    window - never `date.today()` directly - so a caller can guarantee no
    row after that date is ever read, regardless of what's since been
    upserted into Postgres (e.g. by /api/quotes/refresh). Left at its
    default (DEFAULT_CUTOFF_DATE, far in the future), this is a no-op:
    every day up to today is used, exactly as before this parameter
    existed.

    Returns a wide DataFrame: date index, one column per symbol.
    """
    end = _resolve_cutoff_date(cutoff_date)
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


def load_carry(
    pairs: list[str], years: int, dates: pd.DatetimeIndex,
    cutoff_date: str | date | None = DEFAULT_CUTOFF_DATE,
) -> pd.DataFrame:
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
    end = _resolve_cutoff_date(cutoff_date)
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
#: "cma"'s/"bandpass"'s multiple expanded columns would land relative to
#: the rest). "cma" and "bandpass" are both special: each contributes one
#: channel PER (short, long) window pair in cma_windows/bandpass_windows
#: respectively, not just one - see n_channels_per_pair.
FEATURE_CATALOG: tuple[str, ...] = ("log_return", "vol", "skew", "kurt", "carry", "cma", "bandpass")

#: Default feature selection - matches this module's original always-on
#: set (raw return + rolling vol/skew/kurtosis), with carry/cma/bandpass
#: all opt-in.
DEFAULT_FEATURES: list[str] = ["log_return", "vol", "skew", "kurt"]

#: Default (short, long) trailing windows for "cma" (see
#: cross_moving_averages) when "cma" is selected but no windows are given.
DEFAULT_CMA_WINDOWS: list[tuple[int, int]] = [[10, 50]]

#: Default (short, long) PERIODS (days) for "bandpass" (see
#: butterworth_bandpass_features) when "bandpass" is selected but no
#: windows are given.
DEFAULT_BANDPASS_WINDOWS: list[tuple[int, int]] = [[10, 50]]

#: Default Butterworth filter order (see butterworth_bandpass_features) -
#: higher = steeper rolloff/more selective, but more phase lag and more
#: ringing; lower = faster reaction, less noise rejection.
DEFAULT_BANDPASS_ORDER: int = 3


def n_channels_per_pair(
    features: list[str], cma_windows: list | None = None, bandpass_windows: list | None = None,
) -> int:
    """How many input channels build_feature_dataframe produces PER PAIR
    for this feature selection: every base feature in FEATURE_CATALOG
    contributes exactly one channel; "cma"/"bandpass" instead each
    contribute one channel PER (short, long) window pair in
    `cma_windows`/`bandpass_windows` respectively (zero if that feature
    isn't selected, regardless of what its windows list contains).
    """
    unknown = set(features) - set(FEATURE_CATALOG)
    if unknown:
        raise ValueError(f"Unknown feature(s) {sorted(unknown)} - choose from {FEATURE_CATALOG}")
    base = sum(1 for f in features if f not in ("cma", "bandpass"))
    cma_count = len(cma_windows or []) if "cma" in features else 0
    bandpass_count = len(bandpass_windows or []) if "bandpass" in features else 0
    return base + cma_count + bandpass_count


def cross_moving_averages(returns: pd.DataFrame, cma_windows: list) -> dict:
    """For each (short, long) window pair, the TREND signal
    `rolling_mean(returns, short) - rolling_mean(returns, long)` per pair -
    positive when the recent (short-window) trend in log returns is
    running above the longer-run (long-window) trend, the return-space
    analogue of a classic price moving-average crossover (a fast MA
    crossing above a slow one signals a new uptrend). Purely trailing
    (min_periods=window, no bfill - same "exclude rather than paper over"
    policy as make_sequences' trailing_vol), so it never looks ahead.

    See butterworth_bandpass_features for a faster-reacting alternative -
    an SMA difference is a crude, high-lag approximation of a band-pass
    filter; a proper Butterworth design gets comparable noise rejection
    with less lag, at the cost of being harder to reason about by eye.

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


def butterworth_bandpass_features(returns: pd.DataFrame, bandpass_windows: list, order: int = 3) -> dict:
    """For each (short_period, long_period) pair - DAYS, the band of cycle
    lengths to pass, same short/long framing as cross_moving_averages' own
    windows - a CAUSAL Butterworth band-pass filter of each pair's log
    returns, via `scipy.signal.lfilter` applied FORWARD-ONLY.

    Deliberately NOT `scipy.signal.filtfilt`: filtfilt is the usual way
    Butterworth filters get applied (it runs the filter forward then
    backward to cancel phase distortion entirely), but that's NON-CAUSAL -
    the backward pass uses samples AFTER the point being filtered, which
    would leak future returns into today's feature value. `lfilter` keeps
    every output strictly a function of past and current samples only,
    the same causality guarantee every other feature/label in this module
    (trailing_vol, rolling_moment_features, cross_moving_averages,
    AssetLSTM's own causal attention mask) already has to hold - at the
    cost of some unavoidable phase lag, since a genuinely causal filter
    can never fully cancel it. It still reacts FASTER than
    cross_moving_averages (an SMA-difference, itself a crude high-lag
    band-pass) for a comparable amount of noise rejection, because a
    proper Butterworth design achieves its passband/stopband shape far
    more efficiently than differencing two plain averages - that's the
    actual point of using one.

    `short_period` sets the HIGH cutoff (the fastest cycle length that
    still passes); `long_period` sets the LOW cutoff (cycles slower than
    this - i.e. the long-run trend/DC component - are removed).
    `order` (default 3, see DEFAULT_BANDPASS_ORDER) trades lag against
    selectivity: higher = steeper rolloff and more noise rejection but
    more phase lag and more ringing; lower = faster reaction, noisier.

    The first `long_period` rows of each column are set to NaN (the
    filter's own transient response hasn't had enough history to settle
    over that little data yet - the same "need at least this many days"
    logic as a `long_period`-day rolling window) - left for the caller
    (build_feature_dataframe) to ffill/fillna(0.0), same as every other
    rolling-window input feature.

    Returns a dict keyed by (pair, short_period, long_period) ->
    pd.Series aligned to `returns`'s own index.
    """
    from scipy.signal import butter, lfilter

    nyquist = 0.5  # cycles/day, for once-daily-sampled data (sampling rate = 1/day)
    out = {}
    for short_period, long_period in bandpass_windows:
        if not (0 < short_period < long_period):
            raise ValueError(
                f"bandpass_windows short period ({short_period}) must be > 0 and < long period ({long_period})"
            )
        low = (1.0 / long_period) / nyquist
        high = (1.0 / short_period) / nyquist
        if not (0 < low < high < 1):
            raise ValueError(
                f"bandpass_windows [{short_period}, {long_period}] gives an invalid normalized passband "
                f"({low:.4f}, {high:.4f}) - short_period must be > {1 / nyquist:.0f} days (the Nyquist limit for "
                f"daily data), and long_period must be finite."
            )
        b, a = butter(order, [low, high], btype="bandpass")
        for pair in returns.columns:
            filtered = lfilter(b, a, returns[pair].to_numpy(dtype=np.float64))
            series = pd.Series(filtered, index=returns.index, dtype=np.float32)
            series.iloc[:long_period] = np.nan
            out[(pair, short_period, long_period)] = series
    return out


def build_feature_dataframe(
    returns: pd.DataFrame,
    pairs: list[str],
    features: list[str],
    rolling_stats_window: int,
    cma_windows: list,
    years: int,
    bandpass_windows: list | None = None,
    bandpass_order: int = DEFAULT_BANDPASS_ORDER,
    cutoff_date: str | date | None = DEFAULT_CUTOFF_DATE,
) -> pd.DataFrame:
    """Build PredictionModel's actual (wider) input feature set: for each
    pair, in order, whichever of [raw log return, rolling vol, rolling
    skew, rolling kurtosis, carry, cma, bandpass...] are selected in
    `features` (see FEATURE_CATALOG for the fixed column order) -
    deliberately ASSET-MAJOR (all of one asset's channels together, then
    the next asset's).

    Which of these channels actually feed a given asset's OWN AssetLSTM is
    a SEPARATE question, decided by `cross_pairs` at the PredictionModel
    level (see its own docstring) - this function always builds the FULL
    block for every pair in `pairs`, regardless of cross_pairs; slicing
    down to what one asset's LSTM actually sees happens later, in
    PredictionModel.forward().
    """
    bandpass_windows = bandpass_windows or []
    n_channels = n_channels_per_pair(features, cma_windows, bandpass_windows)
    moments = rolling_moment_features(returns, rolling_stats_window) if any(f in features for f in ("vol", "skew", "kurt")) else None
    carry = load_carry(pairs, years, returns.index, cutoff_date) if "carry" in features else None
    cmas = cross_moving_averages(returns, cma_windows) if "cma" in features else None
    bandpasses = butterworth_bandpass_features(returns, bandpass_windows, bandpass_order) if "bandpass" in features else None

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
        if "bandpass" in features:
            for short, long_ in bandpass_windows:
                features_df[f"{pair}_bp_{short}_{long_}"] = bandpasses[(pair, short, long_)]

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
        target_vol: float = DEFAULT_TARGET_VOL,
        bandpass_windows: list | None = None, bandpass_order: int = DEFAULT_BANDPASS_ORDER,
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
            "bandpass_windows": [list(w) for w in (bandpass_windows or [])],
            "bandpass_order": int(bandpass_order),
            "sigma_hat": torch.as_tensor(sigma_hat if sigma_hat is not None else np.ones(self.n_assets, dtype=np.float32)),
            # Half-width of the abstention region around p=0.5 (see
            # apply_neutral_band) - persisted so live inference abstains
            # exactly the way the reported backtest metrics did.
            "neutral_band": float(neutral_band),
            # Annualized volatility the evaluation-mode portfolio PnL
            # calculator (models/portfolio_pnl.py) scales this model's
            # positions to - a property of how the model is meant to be
            # traded, not a free evaluation-time parameter, so it's
            # persisted the same way as neutral_band above.
            "target_vol": float(target_vol),
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
        model.bandpass_windows = checkpoint.get("bandpass_windows", [])
        model.bandpass_order = int(checkpoint.get("bandpass_order", DEFAULT_BANDPASS_ORDER))
        n_assets = checkpoint["config"]["n_assets"]
        sigma_hat = checkpoint.get("sigma_hat")
        model.sigma_hat = sigma_hat.numpy() if sigma_hat is not None else np.ones(n_assets, dtype=np.float32)
        model.neutral_band = float(checkpoint.get("neutral_band", 0.0))
        model.target_vol = float(checkpoint.get("target_vol", DEFAULT_TARGET_VOL))
        return model

    def save_model(
        self, path: str = "models/prediction_model.pt", *,
        x_mean: np.ndarray, x_std: np.ndarray, pairs: list[str], lookback: int,
        features: list[str] | None = None, cma_windows: list | None = None,
        sigma_hat: np.ndarray | None = None, neutral_band: float = 0.0,
        target_vol: float = DEFAULT_TARGET_VOL,
        bandpass_windows: list | None = None, bandpass_order: int = DEFAULT_BANDPASS_ORDER,
    ) -> None:
        """Persist every asset's trained LSTM weights, architecture
        config (including per-asset cross_pairs input slicing), input
        standardization stats, the ordered FX pairs, the sequence length,
        the feature selection, and the validation-fit calibration scale -
        a self-contained checkpoint load_model() can rebuild and run
        calibrated inference from without retraining.
        """
        torch.save(
            self._checkpoint_dict(
                x_mean, x_std, pairs, lookback, features, cma_windows, sigma_hat, neutral_band, target_vol,
                bandpass_windows, bandpass_order,
            ),
            path,
        )
        logger.info("Saved model weights to %s", path)

    def save_to_db(
        self, name: str, *,
        x_mean: np.ndarray, x_std: np.ndarray, pairs: list[str], lookback: int,
        features: list[str] | None = None, cma_windows: list | None = None,
        sigma_hat: np.ndarray | None = None, neutral_band: float = 0.0,
        target_vol: float = DEFAULT_TARGET_VOL,
        bandpass_windows: list | None = None, bandpass_order: int = DEFAULT_BANDPASS_ORDER,
        description: str = "",
    ) -> None:
        """Serialize the same checkpoint save_model() would write, and
        upsert it into quant.model_registry under `name` instead of a
        local file - see data/model_registry.py.
        """
        from data.model_registry import save_model_blob

        buffer = io.BytesIO()
        torch.save(
            self._checkpoint_dict(
                x_mean, x_std, pairs, lookback, features, cma_windows, sigma_hat, neutral_band, target_vol,
                bandpass_windows, bandpass_order,
            ),
            buffer,
        )
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

    Applied purely for reporting almost everywhere it's used (see
    evaluate_prediction_model) - the one exception is
    _portfolio_sharpe_loss below (train_prediction_model's optional
    sharpe_weight phase), which needs an actual probability to turn into a
    trading signal, not just a reported number.
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

def _trailing_moving_average_torch(x: torch.Tensor, window: int) -> torch.Tensor:
    """Trailing simple moving average over `window` rows (INCLUDING row t),
    expanding (mean over whatever's available) for the first `window - 1`
    rows - the differentiable, torch-side counterpart of
    models/portfolio_pnl.py's own _trailing_moving_average, used only
    inside _portfolio_sharpe_loss below. Unlike that numpy version, there's
    no NaN case to special-case here: `x` is always a model's own signal
    (finite by construction - see _portfolio_sharpe_loss), never a
    precomputed-from-data quantity with a warm-up NaN region.
    """
    t = x.shape[0]
    cumsum = torch.cumsum(x, dim=0)
    windowed_sum = cumsum.clone()
    if t > window:
        windowed_sum[window:] = cumsum[window:] - cumsum[:-window]
    counts = torch.clamp(torch.arange(1, t + 1, device=x.device, dtype=x.dtype), max=float(window))
    return windowed_sum / counts.unsqueeze(-1)


def _non_overlapping_sharpe_torch(pnl: torch.Tensor, window: int, eps: float = 1e-8) -> torch.Tensor:
    """Sharpe ratio (mean/std of `pnl`, NOT annualized - a plain per-day
    ratio) computed over NON-OVERLAPPING `window`-day chunks, walking
    BACKWARD from the LAST day: chunk 0 is the final `window` days, chunk
    1 is the `window` days immediately before that, and so on - each day
    contributes to AT MOST one chunk, unlike a trailing/rolling window
    (see this function's own git history), where the same day is reused
    across up to `window` overlapping positions. Returned as a
    (n_chunks,) series, most recent chunk first.
    train_prediction_model's sharpe_weight phase averages this into one
    scalar loss - an AVERAGED Sharpe across several INDEPENDENT
    sub-periods of the window, not a single end-of-period value and not a
    smoothed/autocorrelated rolling statistic, so the objective reflects
    consistency across genuinely distinct stretches of the training
    window.

    If `T` (the series length) doesn't divide evenly by `window`, the
    leftover OLDEST `T mod window` days are DROPPED entirely, never
    contributing a chunk of their own - every chunk that DOES get a
    Sharpe has the SAME full `window` length (no averaging together
    Sharpes computed over different sample sizes/variances). If `T` is
    SHORTER than `window` itself, there isn't even one full chunk -
    returns an EMPTY tensor rather than shrinking the window to fit (that
    would silently compute a Sharpe over fewer than `window` points,
    exactly what dropping the leftover above is meant to avoid) - see
    _portfolio_sharpe_loss_from_predictions for how an empty result is
    handled (a neutral, zero-gradient contribution, not a NaN).
    """
    t = pnl.shape[0]
    window = max(window, 2)
    n_full_chunks = t // window  # 0 if t < window - see this function's own docstring
    if n_full_chunks == 0:
        # Short-circuit rather than falling through to a (0, window)
        # reshape + std() - PyTorch warns ("degrees of freedom is <= 0")
        # on a std() reduction over a zero-sized tensor even though the
        # result (correctly) still comes out empty; skip the warning
        # entirely since this is an expected, handled case, not a bug.
        return pnl.new_zeros((0,))

    # Walk backward from the end: reverse once, so a plain reshape into
    # (n_full_chunks, window) rows gives row 0 = the LAST `window` days,
    # row 1 = the `window` days before that, etc. (any leftover oldest
    # days, past n_full_chunks * window, are simply never included in the
    # reshape - dropped). mean()/std() are order-invariant WITHIN a row,
    # so the reversal never needs undoing.
    reversed_pnl = pnl.flip(0)
    full_chunks = reversed_pnl[: n_full_chunks * window].reshape(n_full_chunks, window)
    return full_chunks.mean(dim=-1) / (full_chunks.std(dim=-1, unbiased=True) + eps)


def _portfolio_sharpe_loss_from_predictions(
    mu_dec: torch.Tensor,
    sigma_dec: torch.Tensor,
    next_returns: torch.Tensor,
    rp_weights: torch.Tensor,
    cov: torch.Tensor,
    direction_horizon: int,
    sharpe_window: int,
    target_vol: float,
    cost_bps: float = DEFAULT_COST_BPS,
) -> torch.Tensor:
    """Differentiable, JOINT (whole-book) negative averaged Sharpe (over
    non-overlapping `sharpe_window`-day chunks - see
    _non_overlapping_sharpe_torch) - the torch-side counterpart of models/portfolio_pnl.py's
    compute_portfolio (same strategy: risk-parity weight x probability
    signal, smoothed over direction_horizon, scaled to target_vol),
    computed for ONE split (train or val), from ALREADY-COMPUTED
    decision-day predictions (`mu_dec`/`sigma_dec`, each (T, n_assets) -
    the window's LAST timestep, see PredictionModel.forward's docstring).

    Takes predictions rather than `model`+`X` so a caller that ALSO needs
    (mu, sigma) for something else that epoch (see train_prediction_model,
    which needs the SAME forward pass's dense output for its own per-asset
    NLL/BCE terms) can compute the shared forward pass exactly ONCE and
    reuse it here - both for a single combined `backward()` across every
    loss term (see train_prediction_model's own docstring on why this
    matters), and simply to avoid a redundant forward pass. `_portfolio_sharpe_loss`
    below is the convenience wrapper for a caller that only needs this
    term on its own (e.g. a standalone before/after comparison).

    NOT decomposable per-asset like gaussian_nll/direction_bce, since the
    vol-scaling denominator `w' cov w` mixes EVERY asset's current signal
    together - this is the ONE place in this module's training loop where
    one asset's loss genuinely depends on every other asset's CURRENT
    output.

    `rp_weights`/`cov` are the PRECOMPUTED, data-only constants from
    models/portfolio_pnl.py's precompute_risk_parity (see _PreparedData) -
    never a function of the model, so no gradient flows through them, only
    through each asset's own probability signal. Rows where they're NaN
    (not enough trailing covariance history yet) are zeroed via
    nan_to_num BEFORE any arithmetic (not masked after) - a NaN multiplied
    by anything, including 0, is still NaN in IEEE754, so zeroing early is
    what actually makes those rows contribute exactly 0 to every downstream
    quantity (signal weight, portfolio variance, position, pnl) instead of
    poisoning the whole batch with NaN gradients.

    No sigma_hat (calibration scale) is applied here, unlike
    evaluate_prediction_model's reported probabilities - sigma_hat is fit
    from THIS model's own validation residuals AFTER training finishes, so
    it doesn't exist yet mid-training. The sign/relative ordering a trading
    signal actually needs doesn't depend on that calibration scale.
    """
    signal = (probit(mu_dec / sigma_dec.clamp_min(1e-8)) - 0.5) * 2.0  # (T, n_assets), in [-1, 1]

    rp_weights_safe = torch.nan_to_num(rp_weights, nan=0.0)
    cov_safe = torch.nan_to_num(cov, nan=0.0)
    # Smooth the PRODUCT (rp_weight * signal), not the signal alone, then
    # rp_weight-at-t - matches compute_portfolio's own raw_modulated ->
    # _trailing_moving_average ordering exactly (models/portfolio_pnl.py).
    # rp_weight itself drifts day to day (a rolling risk-parity solve), so
    # these two orderings are NOT equivalent - MA(w*s) != w * MA(s) - and
    # this function's whole point is to be the differentiable counterpart
    # of THAT exact eval-time strategy, not a similar-looking one.
    raw_modulated = rp_weights_safe * signal
    w_mod = _trailing_moving_average_torch(raw_modulated, direction_horizon)
    port_var = torch.einsum("ti,tij,tj->t", w_mod, cov_safe, w_mod)
    target_vol_daily = target_vol / (TRADING_DAYS_PER_YEAR ** 0.5)
    scale = torch.clamp(target_vol_daily / torch.sqrt(port_var.clamp_min(1e-16)), max=MAX_VOL_SCALE)
    positions = w_mod * scale.unsqueeze(-1)

    # NET daily pnl: gross minus linear transaction costs on daily position
    # CHANGE (day 0 pays for establishing from flat) - the differentiable
    # counterpart of compute_portfolio's own cost deduction
    # (models/portfolio_pnl.py), so the Sharpe being maximized here is the
    # SAME net-of-costs Sharpe evaluation reports; without this the
    # objective would happily crank turnover a real book pays for. |x| is
    # differentiable a.e. - the same subgradient story as any L1 penalty.
    gross_pnl = (positions * next_returns).sum(dim=-1)  # (T,) - whole-book daily pnl
    position_deltas = torch.diff(positions, dim=0, prepend=torch.zeros_like(positions[:1]))
    daily_costs = cost_bps * 1e-4 * position_deltas.abs().sum(dim=-1)
    daily_pnl = gross_pnl - daily_costs
    period_sharpes = _non_overlapping_sharpe_torch(daily_pnl, sharpe_window)
    if period_sharpes.numel() == 0:
        # Fewer than `sharpe_window` days available (e.g. a short split) -
        # _non_overlapping_sharpe_torch deliberately returns empty rather
        # than shrinking the window (see its own docstring), so there's no
        # well-defined Sharpe to compute yet. A NEUTRAL zero loss (no
        # gradient contribution from this term at all) is the honest
        # result - not NaN (mean() of an empty tensor), which would
        # corrupt every other loss term summed with this one that epoch
        # (see train_prediction_model's own combined-loss docstring).
        return torch.zeros((), device=daily_pnl.device, dtype=daily_pnl.dtype)
    return -period_sharpes.mean()


def _portfolio_sharpe_loss(
    model: "PredictionModel",
    X: torch.Tensor,
    next_returns: torch.Tensor,
    rp_weights: torch.Tensor,
    cov: torch.Tensor,
    direction_horizon: int,
    sharpe_window: int,
    target_vol: float,
    cost_bps: float = DEFAULT_COST_BPS,
) -> torch.Tensor:
    """Convenience wrapper around _portfolio_sharpe_loss_from_predictions
    for a caller that only needs the Sharpe term on its own (a standalone
    before/after comparison, a smoke test, etc.) and doesn't already have
    `model`'s own forward pass computed - runs one itself. Everywhere this
    loss is needed ALONGSIDE the dense NLL/BCE forward pass (see
    train_prediction_model) calls _portfolio_sharpe_loss_from_predictions
    directly instead, reusing that SAME pass rather than paying for a
    second one.
    """
    mu, sigma = model(X)  # each (T, lookback, n_assets)
    return _portfolio_sharpe_loss_from_predictions(
        mu[:, -1, :], sigma[:, -1, :], next_returns, rp_weights, cov, direction_horizon, sharpe_window, target_vol,
        cost_bps,
    )


def _assert_finite_grad(params, context: str) -> None:
    """Raise immediately if any parameter's gradient (just populated by a
    backward() call) is NaN/Inf - a loud, actionable failure AT THE EPOCH
    IT HAPPENS, instead of silently continuing to train (and checkpoint-
    select) on a corrupted model for however many epochs remain, which is
    what happened before this check existed: a real run (num_layers=2,
    hidden_size=128, device="mps") produced a finite loss but a NaN
    gradient on its very first backward() - confirmed to be a PyTorch MPS
    multi-layer-LSTM backward-kernel issue specific to that exact
    (hidden_size, num_layers) combination (the same forward/backward pass
    on `device="cpu"` produces a normal, finite gradient - see
    smoke_test.py's own regression test for this). This check can't fix
    that underlying MPS kernel bug, but it turns "300 silently-wasted
    epochs producing a NaN checkpoint" into an immediate, clear error.
    """
    for p in params:
        if p.grad is not None and not torch.isfinite(p.grad).all():
            raise RuntimeError(
                f"Non-finite gradient during {context} - training stopped immediately rather than continuing on "
                f"a corrupted model. If you're on device='mps' with num_layers > 1 (especially with a large "
                f"hidden_size), this matches a known PyTorch MPS multi-layer-LSTM backward-pass bug - try "
                f"device='cpu' or num_layers=1 first to confirm, before assuming it's a data/config issue."
            )


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
    sharpe_weight: float = 0.0,
    sharpe_window: int = 20,
    direction_horizon: int = 5,
    target_vol: float = DEFAULT_TARGET_VOL,
    cost_bps: float = DEFAULT_COST_BPS,
    next_returns_train: torch.Tensor | None = None,
    next_returns_val: torch.Tensor | None = None,
    rp_weights_train: torch.Tensor | None = None,
    rp_weights_val: torch.Tensor | None = None,
    cov_train: torch.Tensor | None = None,
    cov_val: torch.Tensor | None = None,
    checkpoint_metric: str = "val_loss",
    on_checkpoint_update: Callable[[int, list], None] | None = None,
) -> None:
    """Train every asset's LSTM with its OWN `torch.optim.Adam` instance
    (own momentum/variance state, entirely separate from every other
    asset's optimizer) and its OWN best-validation-loss checkpoint
    restored independently at the end (asset A's own best epoch can differ
    from asset B's). The assets never shared WEIGHTS (see AssetLSTM/
    PredictionModel) - so even though (see below) every loss term is
    summed into ONE scalar before a SINGLE `backward()` call each epoch,
    each asset's OWN accumulated gradient still only ever contains terms
    that actually depend on THAT asset's own parameters. An asset linked
    to another pair's features via `cross_pairs` still trains this way -
    only what its LSTM READS is widened, never what optimizes it.

    ONE combined step per epoch
    ----------------------------
    Every loss term this epoch - each asset's own NLL+BCE, and (if
    `sharpe_weight > 0`) the portfolio-Sharpe term - is summed into ONE
    scalar and passed to a SINGLE `backward()` call, evaluated at the SAME
    pre-step weights, before any optimizer steps at all:

        mu, sigma = model(X_train)          # ONE dense forward pass, every asset at once, (n_train, lookback, n_assets)
        total_loss = sum(
            gaussian_nll(mu[:, :, i], sigma[:, :, i], z_label_i)
            + bce_weight * direction_bce(mu[:, :, i], sigma[:, :, i], z_label_i)
            for i in range(n_assets)
        )
        if sharpe_weight > 0:
            total_loss = total_loss + sharpe_weight * _portfolio_sharpe_loss_from_predictions(
                mu[:, -1, :], sigma[:, -1, :], next_returns_train, rp_weights_train, cov_train, ...
            )
        total_loss.backward()
        for each asset's own optimizer: optimizer.step()

    Applied to the FULL dense (batch, lookback) output for the NLL/BCE
    terms, not just the window's last ("decision") day - every day in
    every window is its own supervised sample (see AssetLSTM's docstring),
    giving each epoch far more gradient signal than decision-day-only
    supervision would, and since consecutive windows slide by one day, a
    given calendar day is trained on repeatedly (once per window it falls
    inside) rather than just once as a decision day - so there's no need
    to separately up-weight it. The Sharpe term (when enabled) uses only
    the decision day, matching how it's actually traded/evaluated (see
    models/portfolio_pnl.py's compute_portfolio).

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

    Why ONE step, not two sequential ones
    ----------------------------------------
    An earlier version of this function ran two SEPARATE phases each
    epoch: a full per-asset NLL+BCE backward/step, immediately followed by
    a SEPARATE Sharpe backward/step computed at the ALREADY-UPDATED
    weights. That was a first-order OPERATOR SPLIT of the combined update
    (the Lie-Trotter formula: `exp(lr*A) . exp(lr*B) = exp(lr*(A+B)) +
    O(lr^2)`) - a legitimate technique (the same idea behind GAN G/D
    alternation, ADMM, etc.), but not free: whenever the two gradients
    pointed in different directions, the SECOND phase could genuinely claw
    back some of the FIRST phase's own improvement that same epoch (and
    vice versa the next epoch, in the other order) - an avoidable
    per-epoch approximation error whose size scales with the two
    gradients' disagreement and with `lr`. Summing both terms into ONE
    scalar before ONE `backward()` removes that error entirely: both
    gradients are now computed at the IDENTICAL starting point and added
    together BEFORE stepping - exactly what jointly minimizing
    `sum_i loss_i + sharpe_weight * sharpe_loss` means, with no splitting
    artifact left over.

    This changes WHEN the Sharpe gradient is computed relative to the
    NLL+BCE gradient, not whose parameters end up receiving it - it does
    NOT reintroduce weight-sharing. `sharpe_weight=0` remains BYTE-
    IDENTICAL to training every asset in complete isolation, one at a
    time (see smoke_test.py's own regression test for this): the NLL/BCE
    terms are architecturally independent per asset (no shared weights),
    so summing N independent scalars into one before backward() cannot
    introduce a dependency between them that wasn't already there.
    `sharpe_weight > 0` remains the ONE case with a genuine cross-asset
    gradient term (see _portfolio_sharpe_loss_from_predictions's own
    docstring on why Sharpe can't decompose per-asset) - it's just now
    included in the SAME step instead of a follow-up one.

    If `X_val`/`z_labels_val` are given, the same per-asset loss - but
    computed on the DECISION DAY ONLY, matching what's actually
    evaluated/reported (unlike the dense training loss above) - is tracked
    per asset after every epoch, and each asset's OWN state_dict from
    whichever epoch had ITS OWN lowest validation loss is restored
    independently at the end - not just whichever epoch `epochs` happened
    to end on, and (by default - see `checkpoint_metric` below)
    deliberately NOT the epoch with the best validation HIT RATE
    (computed here from sign(mu), purely for logging): hit rate is a
    coarse, saturating metric (once every sample's sign is already
    correct, it cannot improve further no matter how much more
    confident/well-calibrated the predictions get), so selecting on it
    tends to freeze onto whichever epoch FIRST reached its ceiling - often
    very early - and discard every later epoch that kept reducing loss but
    couldn't move hit rate any higher.

    The epoch counter and log line are shared across all assets (one line
    per epoch, an average across assets) purely for readable progress
    reporting.

    `rp_weights`/`cov` are precomputed ONCE from realized data before
    training even starts (models/portfolio_pnl.py's precompute_risk_parity,
    threaded through via _PreparedData) - never a function of the model,
    so nothing here is optimizing the risk-parity solve itself (which
    isn't even differentiable - it's an external SLSQP call).

    When `X_val`/`z_labels_val` are also given, the SAME joint Sharpe is
    computed on the validation split (no backward - pure diagnostic +
    selection signal) and, if `checkpoint_metric="val_loss"` (the
    default), SUBTRACTED (weighted by `sharpe_weight`) from each asset's
    own `val_loss_i` before the best-checkpoint comparison - i.e.
    checkpoint selection uses the exact same composite objective training
    optimizes, instead of switching to a different, noisier metric
    (validation-period Sharpe alone, over a few hundred days, is a
    high-variance statistic - selecting on it in isolation risks picking
    whichever epoch's realized path happened to align with the validation
    window rather than a genuinely better model).

    `checkpoint_metric` (independent of `sharpe_weight` - which controls
    what TRAINS the weights above): which per-epoch VALIDATION metric
    selects each asset's own restored checkpoint at the end.
      - "val_loss" (default): as described above - own NLL+BCE val loss,
        minus sharpe_weight * val_sharpe if sharpe_weight > 0. Unchanged
        from this function's original behavior.
      - "hit_rate": the epoch with the best validation directional hit
        rate wins instead. This function's own docstring above explains
        why this is NOT the default (hit rate saturates once every sign is
        already correct, so it tends to freeze onto whichever epoch first
        reached its ceiling and ignore further loss improvement) - it's
        offered as an explicit, informed choice for a caller who wants
        checkpoints selected on the same coarse metric the Training/
        Evaluation views report front and center, not a recommendation.
      - "sharpe": the epoch with the best validation portfolio Sharpe
        (see _portfolio_sharpe_loss) wins - the SAME joint (whole-book,
        not per-asset) value applies to every asset's own comparison, same
        as the "val_loss" branch's sharpe_weight-adjustment above. Works
        even when `sharpe_weight=0` (training stays pure NLL+BCE; only
        which CHECKPOINT gets kept changes) - the caller must still supply
        next_returns_val/rp_weights_val/cov_val for this to be computable.

    `on_checkpoint_update`, if given, is called after every VALIDATED
    epoch with `best_state` itself (the SAME list object, whose per-asset
    ENTRIES get reassigned - never mutated in place - to a fresh
    `copy.deepcopy` whenever a new best is found) - cheap (no evaluation,
    just handing over a reference), letting a caller elsewhere (e.g.
    api/server.py's "save best model so far" endpoint) always read the
    CURRENT best checkpoint on demand, without this function needing to
    know anything about full evaluation, checkpoint metadata, or the DB -
    see models/portfolio_lstm.py's summarize_checkpoint for that part.
    """
    n_assets = model.n_assets
    optimizers = [
        torch.optim.Adam(model.assets[i].parameters(), lr=lr, weight_decay=weight_decay)
        for i in range(n_assets)
    ]
    if checkpoint_metric not in ("val_loss", "hit_rate", "sharpe"):
        raise ValueError(f"checkpoint_metric must be 'val_loss', 'hit_rate', or 'sharpe' - got {checkpoint_metric!r}")
    track_val = X_val is not None and z_labels_val is not None
    track_sharpe = sharpe_weight > 0  # does SHARPE actually contribute to the combined training loss?
    # Does validation-time Sharpe need computing at all this run? Either
    # because track_sharpe needs it for the "val_loss" branch's own
    # composite score (unchanged original behavior), or because
    # checkpoint_metric itself selects on it directly - independent of
    # whether sharpe_weight trains anything.
    track_sharpe_val = track_val and (track_sharpe or checkpoint_metric == "sharpe")
    if track_sharpe and (next_returns_train is None or rp_weights_train is None or cov_train is None):
        raise ValueError("sharpe_weight > 0 requires next_returns_train/rp_weights_train/cov_train")
    if track_sharpe_val and (next_returns_val is None or rp_weights_val is None or cov_val is None):
        raise ValueError(
            "sharpe_weight > 0 (with validation tracking) or checkpoint_metric='sharpe' requires "
            "next_returns_val/rp_weights_val/cov_val"
        )
    best_val_loss = [float("inf")] * n_assets
    best_state: list[dict | None] = [None] * n_assets
    val_losses: list[float | None] = [None] * n_assets
    val_hit_rates: list[float | None] = [None] * n_assets

    def _restore_best() -> None:
        for i in range(n_assets):
            if best_state[i] is not None:
                model.assets[i].load_state_dict(best_state[i])
        mean_best = sum(v for v in best_val_loss if v != float("inf")) / n_assets
        logger.info(
            "Restored each asset's own best-validation-%s checkpoint (mean score %.4f)",
            checkpoint_metric, mean_best,
        )

    model.train()
    try:
        for epoch in range(1, epochs + 1):
            _check_stop()
            for opt in optimizers:
                opt.zero_grad()

            # ONE dense forward pass, every asset at once - see this
            # function's own docstring on why every loss term below is
            # summed into ONE scalar and backward()'d together, rather
            # than as two sequential phases.
            mu, sigma = model(X_train)  # each (n_train, lookback, n_assets)
            per_asset_losses = []
            train_hits = []
            for i in range(n_assets):
                mu_i, sigma_i = mu[:, :, i], sigma[:, :, i]
                target_i = z_labels_train[:, :, i]
                loss_i = gaussian_nll(mu_i, sigma_i, target_i) + bce_weight * direction_bce(mu_i, sigma_i, target_i)
                per_asset_losses.append(loss_i)
                with torch.no_grad():
                    train_hits.append(float(((mu_i[:, -1] > 0) == (target_i[:, -1] > 0)).float().mean()))
            total_loss = torch.stack(per_asset_losses).sum()
            train_loss = total_loss.item() / n_assets
            train_hit_rate = sum(train_hits) / n_assets

            train_sharpe = None
            if track_sharpe:
                sharpe_loss = _portfolio_sharpe_loss_from_predictions(
                    mu[:, -1, :], sigma[:, -1, :], next_returns_train, rp_weights_train, cov_train,
                    direction_horizon, sharpe_window, target_vol, cost_bps,
                )
                total_loss = total_loss + sharpe_weight * sharpe_loss
                train_sharpe = -sharpe_loss.detach().item()

            total_loss.backward()
            for i in range(n_assets):
                _assert_finite_grad(model.assets[i].parameters(), f"epoch {epoch}, asset {i} ({model.pairs[i]})")
            for opt in optimizers:
                opt.step()

            if track_val:
                model.eval()
                with torch.no_grad():
                    # Same idea as the training pass above: ONE dense
                    # forward pass over X_val, reused for every asset's own
                    # NLL+BCE val loss AND (if needed) the joint val Sharpe -
                    # values only, no backward, but no reason to pay for N+1
                    # separate forward passes when one covers everything.
                    val_mu, val_sigma = model(X_val)
                    val_mu_dec, val_sigma_dec = val_mu[:, -1, :], val_sigma[:, -1, :]
                    for i in range(n_assets):
                        val_target_i = z_labels_val[:, -1, i]
                        val_loss_i = float(
                            gaussian_nll(val_mu_dec[:, i], val_sigma_dec[:, i], val_target_i)
                            + bce_weight * direction_bce(val_mu_dec[:, i], val_sigma_dec[:, i], val_target_i)
                        )
                        val_losses[i] = val_loss_i
                        val_hit_rates[i] = float(((val_mu_dec[:, i] > 0) == (val_target_i > 0)).float().mean())

                    # val_sharpe is a single JOINT (whole-book) value - see
                    # _portfolio_sharpe_loss's own docstring on why Sharpe
                    # can't decompose per-asset - so it applies equally to
                    # every asset's own comparison below, both in the
                    # "val_loss" branch's composite score (unchanged
                    # original behavior: the SAME objective training
                    # optimizes, not a separate Sharpe-only metric) and in
                    # the "sharpe" branch, where it's the WHOLE score.
                    val_sharpe = None
                    if track_sharpe_val:
                        val_sharpe_loss = _portfolio_sharpe_loss_from_predictions(
                            val_mu_dec, val_sigma_dec, next_returns_val, rp_weights_val, cov_val,
                            direction_horizon, sharpe_window, target_vol, cost_bps,
                        )
                        val_sharpe = -val_sharpe_loss.item()
                    for i in range(n_assets):
                        # LOWER score is always better (matches best_val_loss's
                        # own "<" comparison below) - "hit_rate"/"sharpe" are
                        # maximized quantities, so negate them to fit that
                        # convention; "val_loss" (default) is unchanged from
                        # this function's original behavior.
                        if checkpoint_metric == "hit_rate":
                            score_i = -val_hit_rates[i]
                        elif checkpoint_metric == "sharpe":
                            score_i = -val_sharpe
                        else:
                            score_i = val_losses[i] - (sharpe_weight * val_sharpe if track_sharpe else 0.0)
                        if score_i < best_val_loss[i]:
                            best_val_loss[i] = score_i
                            best_state[i] = copy.deepcopy(model.assets[i].state_dict())
                model.train()
                if on_checkpoint_update is not None:
                    on_checkpoint_update(epoch, best_state)

            if epoch == 1 or epoch % max(epochs // 10, 1) == 0:
                val_loss_mean = sum(val_losses) / n_assets if track_val else None
                val_hit_mean = sum(val_hit_rates) / n_assets if track_val else None
                val_msg = f" | val loss {val_loss_mean:.4f} | val hit rate {val_hit_mean:.4f}" if track_val else ""
                sharpe_msg = f" | train sharpe {train_sharpe:.4f}" if track_sharpe else ""
                if track_sharpe_val:
                    sharpe_msg += f" | val sharpe {val_sharpe:.4f}"

                # Running best CHECKPOINT-SELECTION score so far this run,
                # in human-legible units matching checkpoint_metric (see
                # this function's own docstring) - "val_loss" is stored/
                # displayed as-is (lower better); "hit_rate"/"sharpe" are
                # stored NEGATED (see the score_i computation above, to fit
                # best_val_loss's shared "<" convention), so un-negate them
                # here for display (higher better). None until at least one
                # epoch has actually updated a checkpoint.
                best_score = None
                if track_val:
                    finite_bests = [v for v in best_val_loss if v != float("inf")]
                    if finite_bests:
                        mean_best_score = sum(finite_bests) / len(finite_bests)
                        best_score = mean_best_score if checkpoint_metric == "val_loss" else -mean_best_score
                if best_score is not None:
                    sharpe_msg += f" | best {checkpoint_metric} {best_score:.4f}"

                logger.info(
                    "epoch %d/%d - train loss %.4f | train hit rate %.4f%s%s",
                    epoch, epochs, train_loss, train_hit_rate, val_msg, sharpe_msg,
                )
                _report_epoch(
                    "train", epoch, epochs, train_loss, train_hit_rate, val_loss_mean, val_hit_mean,
                    train_sharpe=train_sharpe, val_sharpe=val_sharpe if track_sharpe_val else None,
                    best_score=best_score,
                )
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
    # Caps every fetched date range (prices, carry) at this date - never
    # later, regardless of what's since landed in Postgres - guaranteeing
    # training/validation/test never see a day after it (see
    # _resolve_cutoff_date). None (the default) means "no cutoff": use all
    # data available up to today, resolved fresh on every run rather than
    # a fixed date. Set to an ISO "YYYY-MM-DD" string (e.g. "2024-06-30")
    # to walk-forward-backtest as of a specific historical date.
    "cutoff_date": DEFAULT_CUTOFF_DATE,
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
    # training's Gaussian NLL + BCE loss fits (see train_prediction_model),
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
    # (short, long) period-in-days pairs for a causal Butterworth band-pass
    # filter (see butterworth_bandpass_features) - a faster-reacting
    # trend-strength alternative to "cma"; only used if "bandpass" is in
    # "features". Uses scipy.signal.lfilter (forward-only), NEVER filtfilt
    # (zero-phase/non-causal - would leak future samples into today's
    # feature value).
    "bandpass_windows": [],
    "bandpass_order": DEFAULT_BANDPASS_ORDER,
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
    # Training (see train_prediction_model): every asset's LSTM trains
    # FULLY INDEPENDENTLY - its own optimizer, its own loss (Gaussian NLL
    # on (mu, sigma) + `bce_weight` * BCE on the implied direction
    # probability probit(mu/sigma)), applied densely over every day in the
    # window, not just the decision day.
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
    # Optional COMPLEMENTARY training objective (see train_prediction_model's
    # own docstring on its "ONE combined step per epoch"): 0 (default)
    # disables it entirely - training is then IDENTICAL to the description
    # above. > 0 adds a JOINT Sharpe term to the SAME combined loss every
    # epoch (one shared backward(), not a separate phase): the model's own
    # probability signal, multiplied by a PRECOMPUTED (data-only, never
    # optimized) risk-parity weight and scaled to `target_vol`, drives an
    # averaged Sharpe ratio - over NON-OVERLAPPING `sharpe_window`-day
    # chunks walking backward from the last day, see
    # _non_overlapping_sharpe_torch - that gets maximized.
    # This is NOT decomposable per-asset like NLL/BCE - the target-vol
    # scaling denominator mixes every asset's current signal - so unlike
    # everything else in this config, sharpe_weight > 0 means one asset's
    # training genuinely depends on every other asset's current output
    # (see models/portfolio_pnl.py/train_prediction_model docstrings).
    "sharpe_weight": 0.0,
    # Chunk size (days) each non-overlapping period Sharpe (see
    # _non_overlapping_sharpe_torch) is computed over.
    "sharpe_window": 20,
    # Linear transaction cost, in BASIS POINTS per unit of daily position
    # change - applied both to every reported portfolio PnL/Sharpe
    # (models/portfolio_pnl.py's compute_portfolio) and inside the
    # sharpe_weight training objective itself, so what's optimized and
    # what's reported are the same NET-of-costs quantity. ~1bp is a
    # realistic all-in spread for FX majors at modest size; 0 recovers the
    # frictionless book.
    "cost_bps": DEFAULT_COST_BPS,
    # Which per-epoch VALIDATION metric selects each asset's own restored
    # checkpoint (see train_prediction_model's own docstring) - independent
    # of "sharpe_weight" above, which controls what TRAINS the weights, not
    # which epoch's weights get kept. "val_loss" (default, unchanged
    # original behavior): own NLL+BCE val loss, minus sharpe_weight *
    # val_sharpe if sharpe_weight > 0. "hit_rate": best validation
    # directional hit rate wins (NOT the default - see the docstring on why
    # this is a coarser, saturating metric). "sharpe": best validation
    # portfolio Sharpe wins - works even with sharpe_weight=0 (training
    # stays pure NLL+BCE; only checkpoint selection changes).
    "checkpoint_metric": "val_loss",
    # Neutral band (see apply_neutral_band): probabilities within
    # 0.5 +/- this half-width are snapped to exactly 0.5 - the model
    # ABSTAINS instead of making a near-coin-flip call. Hit rate/
    # confusion-matrix metrics are then computed over decided samples
    # only, alongside a `coverage` metric (how often the model speaks).
    # 0.0 disables abstention entirely.
    "neutral_band": 0.05,
    # Annualized volatility the evaluation-mode portfolio PnL calculator
    # (models/portfolio_pnl.py) scales this model's risk-parity-weighted,
    # probability-modulated positions to (a property of how the model is
    # meant to be traded, persisted alongside it - see
    # PredictionModel._checkpoint_dict). Also used DURING training itself
    # if "sharpe_weight" > 0 above - the same target the trained strategy
    # is evaluated at is what it's optimized toward.
    "target_vol": DEFAULT_TARGET_VOL,
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
    # Risk-parity weights + covariance, precomputed ONCE from REALIZED data
    # only (models/portfolio_pnl.py's precompute_risk_parity) - never a
    # function of any model's predictions, so safe to compute here and
    # reuse across every seed/restart. Torch tensors (constant,
    # requires_grad=False) so train_prediction_model's optional portfolio-
    # Sharpe training phase (sharpe_weight > 0) can use them directly
    # without leaving numpy. NaN on early days without enough trailing
    # history (see rolling_covariance_matrices) - unused when
    # sharpe_weight == 0 (the default).
    rp_weights_train: torch.Tensor
    rp_weights_val: torch.Tensor
    cov_train: torch.Tensor
    cov_val: torch.Tensor
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
    bandpass_windows: list | None = None,
    bandpass_order: int | None = None,
) -> _PreparedData:
    """Load data (via db.py), build sequences, split by time, and
    standardize - everything a training (or inference) run needs that
    doesn't depend on the model's random seed.

    If `x_mean`/`x_std` are given (loading a previously-saved model), they
    are used as-is instead of being freshly fit - inference must
    standardize new data exactly the way the loaded model was trained.

    If `pairs`/`lookback`/`features`/`cma_windows`/`bandpass_windows`/
    `bandpass_order` are given, they OVERRIDE the matching `args.*` value -
    used when loading a saved model, whose checkpoint carries the exact
    values it was trained with (see PredictionModel._from_checkpoint/
    load_pipeline below); a loaded model's own stored values must win over
    whatever a caller passes in.
    """
    if pairs is None:
        pairs = list(dict.fromkeys(args.pairs))  # de-duplicate, keep order
    if lookback is None:
        lookback = args.lookback
    if features is None:
        features = getattr(args, "features", None) or list(DEFAULT_FEATURES)
    if cma_windows is None:
        cma_windows = getattr(args, "cma_windows", None) or []
    if bandpass_windows is None:
        bandpass_windows = getattr(args, "bandpass_windows", None) or []
    if bandpass_order is None:
        bandpass_order = getattr(args, "bandpass_order", None) or DEFAULT_BANDPASS_ORDER
    cutoff_date = getattr(args, "cutoff_date", None) or DEFAULT_CUTOFF_DATE

    logger.info("Loading %s via db.py (cutoff_date=%s)", pairs, cutoff_date)
    prices = load_close_prices(pairs, years=args.years, cutoff_date=cutoff_date)
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

    n_channels = n_channels_per_pair(features, cma_windows, bandpass_windows)
    feature_returns = build_feature_dataframe(
        returns, pairs, features, rolling_stats_window, cma_windows, args.years, bandpass_windows, bandpass_order,
        cutoff_date,
    )

    X, next_returns, z_labels, dates = make_sequences(
        returns, lookback=lookback, feature_returns=feature_returns, direction_horizon=direction_horizon,
        rolling_stats_window=rolling_stats_window,
    )

    # Computed over the FULL (pre-split) next_returns, like every other
    # rolling-window quantity in this function (feature_returns, trailing
    # vol, etc.) - so validation/test's own early days aren't needlessly
    # NaN just for lacking history that's actually sitting right before
    # them in the training split. Purely a function of realized data (see
    # precompute_risk_parity's own docstring), so slicing this the same
    # way as next_returns below introduces no leakage: cov[t]/rp_weights[t]
    # still only ever look at returns strictly before t, split boundary or not.
    cov_window = getattr(args, "cov_window", None) or DEFAULT_COV_WINDOW
    rp_weights_full, cov_full = precompute_risk_parity(next_returns, cov_window)

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
    # Purging only makes sense at a boundary that actually SEPARATES two
    # splits - if `train_frac=1.0`/`test_frac=0.0` (evaluation mode: the
    # whole fetched period is one block, see api/server.py's evaluate()),
    # there's no val/test after train to protect from leakage, so trimming
    # here would just needlessly throw away the freshest, most valuable
    # days for no reason. Train IS followed by another split whenever a
    # validation split exists (n_train < n_remaining) OR - the
    # `train_frac=1.0` + `test_frac>0` corner - the test split begins
    # IMMEDIATELY at n_train == n_remaining, which needs the exact same
    # gap (the last training labels would otherwise overlap the first
    # test samples' lookback windows).
    purge_gap = lookback + direction_horizon
    train_followed_by_split = n_train < n_remaining or n_test > 0
    train_end = max(n_train - purge_gap, 0) if train_followed_by_split else n_train
    val_end = max(n_remaining - purge_gap, n_train) if n_test > 0 else n_remaining
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
    rp_weights_train_raw = rp_weights_full[:train_end]
    rp_weights_val_raw = rp_weights_full[n_train:val_end]
    cov_train_raw = cov_full[:train_end]
    cov_val_raw = cov_full[n_train:val_end]
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
        rp_weights_train=torch.tensor(rp_weights_train_raw.astype(np.float32), device=device),
        rp_weights_val=torch.tensor(rp_weights_val_raw.astype(np.float32), device=device),
        cov_train=torch.tensor(cov_train_raw.astype(np.float32), device=device),
        cov_val=torch.tensor(cov_val_raw.astype(np.float32), device=device),
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
    n_assets = len(data.pairs)

    def _forward_last_timestep(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # An empty split (e.g. evaluation mode's train_frac=1.0 leaves
        # X_val/X_test with 0 rows) must never reach the LSTM: PyTorch's
        # MPS backend hard-crashes ("Placeholder tensor is empty!") on a
        # zero-length input, while CPU handles it fine - so short-circuit
        # to an empty, correctly-shaped/-typed/-deviced result instead of
        # ever calling model() on it.
        if x.shape[0] == 0:
            empty = x.new_zeros((0, n_assets))
            return empty, empty
        mu, sigma = model(x)
        return mu[:, -1, :], sigma[:, -1, :]

    with torch.no_grad():
        # Decision day only (the window's last timestep) - see this
        # function's own docstring.
        mu_train, sig_train = _forward_last_timestep(data.X_train)
        mu_val, sig_val = _forward_last_timestep(data.X_val)
        mu_test, sig_test = _forward_last_timestep(data.X_test)

    sigma_floor = 1e-3
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


def _run_train_prediction_model(
    model: PredictionModel, data: "_PreparedData", args: argparse.Namespace,
    on_checkpoint_update: Callable[[int, list], None] | None = None,
) -> None:
    """Shared by _train_and_evaluate (fresh random-init model) and
    continue_training (warm-started from an existing checkpoint's
    state_dict) - builds the sharpe_weight/checkpoint_metric-dependent
    tensors and calls train_prediction_model identically either way. The
    only difference between the two callers is how `model`'s WEIGHTS got
    to this point before this function ever runs.

    `on_checkpoint_update` is passed straight through to
    train_prediction_model - see its own docstring.
    """
    sharpe_weight = getattr(args, "sharpe_weight", 0.0) or 0.0
    checkpoint_metric = getattr(args, "checkpoint_metric", "val_loss") or "val_loss"
    # next_returns_val_t is needed whenever validation-time Sharpe is
    # needed AT ALL - either because sharpe_weight>0 (the combined loss's
    # own composite val score) or because checkpoint_metric="sharpe"
    # selects on it directly, even with sharpe_weight=0 (pure NLL+BCE
    # training, Sharpe only decides which checkpoint gets kept - see
    # train_prediction_model's own docstring on checkpoint_metric).
    # next_returns_train_t is only ever needed when sharpe_weight>0 itself.
    next_returns_train_t, next_returns_val_t = None, None
    if sharpe_weight > 0:
        next_returns_train_t = torch.as_tensor(data.next_returns_train, dtype=torch.float32, device=data.device)
    if sharpe_weight > 0 or checkpoint_metric == "sharpe":
        next_returns_val_t = torch.as_tensor(data.next_returns_val, dtype=torch.float32, device=data.device)

    train_prediction_model(
        model, data.X_train, data.z_labels_train,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        bce_weight=getattr(args, "bce_weight", 1.0),
        X_val=data.X_val, z_labels_val=data.z_labels_val,
        sharpe_weight=sharpe_weight,
        sharpe_window=getattr(args, "sharpe_window", 20) or 20,
        direction_horizon=getattr(args, "direction_horizon", 5) or 5,
        target_vol=getattr(args, "target_vol", None) or DEFAULT_TARGET_VOL,
        cost_bps=getattr(args, "cost_bps", DEFAULT_COST_BPS),
        next_returns_train=next_returns_train_t, next_returns_val=next_returns_val_t,
        rp_weights_train=data.rp_weights_train, rp_weights_val=data.rp_weights_val,
        cov_train=data.cov_train, cov_val=data.cov_val,
        checkpoint_metric=checkpoint_metric,
        on_checkpoint_update=on_checkpoint_update,
    )


def _train_and_evaluate(
    data: _PreparedData, args: argparse.Namespace,
    on_best_checkpoint: Callable[[PredictionModel, _PreparedData, argparse.Namespace, int, list], None] | None = None,
) -> PredictionResult:
    """Train one PredictionModel (whatever random seed is currently set) on
    already-prepared data - every asset's LSTM trained FULLY
    INDEPENDENTLY, its own optimizer and loss (see train_prediction_model)
    - and evaluate it on all three splits.

    `on_best_checkpoint`, if given, is called after every VALIDATED epoch
    with `(model, data, args, epoch, best_state)` - `model`/`data`/`args` are
    this call's OWN (architecture, prepared data, config), closed over
    here so a caller several layers up (see api/server.py's "save best
    model so far" endpoint) can build a full evaluated snapshot from
    `best_state` at any later time - e.g. models/portfolio_lstm.py's
    summarize_checkpoint - without needing to plumb `model`/`data`/`args`
    through itself. See train_prediction_model's own docstring on why
    handing over `best_state` here is cheap (no evaluation happens now).
    """
    num_layers = getattr(args, "num_layers", 1)
    if str(data.device) == "mps" and num_layers > 1:
        logger.warning(
            "device=mps with num_layers=%d and hidden_size=%d: a multi-layer LSTM on Apple's Metal backend has a "
            "known PyTorch backward-pass bug that can produce NaN gradients (confirmed on this exact hidden_size/"
            "num_layers combination - the identical forward/backward pass is fine on device='cpu'). "
            "train_prediction_model will raise immediately if it hits this, rather than training silently on a "
            "corrupted model - if it does, retry with device='cpu' or num_layers=1.",
            num_layers, args.hidden_size,
        )
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

    on_checkpoint_update = None
    if on_best_checkpoint is not None:
        on_checkpoint_update = lambda epoch, best_state: on_best_checkpoint(model, data, args, epoch, best_state)  # noqa: E731

    _run_train_prediction_model(model, data, args, on_checkpoint_update=on_checkpoint_update)
    return evaluate_prediction_model(model, data, neutral_band=getattr(args, "neutral_band", 0.0))


def continue_training(
    args: argparse.Namespace, base_model: PredictionModel,
    on_best_checkpoint: Callable[[PredictionModel, "_PreparedData", argparse.Namespace, int, list], None] | None = None,
) -> PredictionResult:
    """Warm-start training from an ALREADY-TRAINED model's weights
    (`base_model` - loaded by the caller, e.g. via load_prediction_model_auto)
    instead of a freshly random-initialized PredictionModel - i.e. keep
    optimizing an existing model rather than starting over.

    Every ARCHITECTURE-defining property - n_assets/pairs, n_channels,
    hidden_size, num_layers, dropout, n_attn_heads, cross_pairs, lookback,
    features, cma_windows, bandpass_windows, bandpass_order - and the
    model's own input standardization stats (x_mean/x_std: the weights
    were fit expecting inputs scaled EXACTLY this way) are recovered from
    `base_model` and used AS-IS - never from `args` - since changing any
    of them would produce a shape-incompatible or silently-miscalibrated
    model, not a genuine continuation of training. (api/server.py's
    _run_training_job additionally overwrites `args`'s own copies of these
    fields to match `base_model` before calling this, purely so the
    caller's OWN post-training bookkeeping - e.g. what it saves the result
    checkpoint with - stays consistent too; this function does not depend
    on that having happened.)

    Everything else - epochs, lr, weight_decay, bce_weight, sharpe_weight,
    sharpe_window, direction_horizon, checkpoint_metric, target_vol,
    neutral_band, and the data window (years, cutoff_date, train_frac,
    test_frac, device) - comes from `args`, exactly like a fresh training
    run (see _train_and_evaluate) - e.g. to continue training over freshly
    extended history, or with a different optimization objective, without
    touching what the network architecturally is. Optimizer momentum
    (Adam's own per-parameter state) is NOT persisted in a checkpoint and
    so is NOT resumed - this is a warm-started fresh optimizer run over
    pretrained weights, not a bit-exact resume of a previous run.

    `on_best_checkpoint`, if given, is called after every VALIDATED epoch
    with `(model, data, args, epoch, best_state)` - see _train_and_evaluate's own
    docstring, identical contract here.
    """
    data = _prepare_data(
        args, x_mean=base_model.x_mean, x_std=base_model.x_std, pairs=base_model.pairs,
        lookback=base_model.lookback, features=base_model.features, cma_windows=base_model.cma_windows,
        bandpass_windows=base_model.bandpass_windows, bandpass_order=base_model.bandpass_order,
    )
    model = PredictionModel(
        n_assets=len(base_model.pairs),
        pairs=base_model.pairs,
        n_channels=base_model.n_channels,
        hidden_size=base_model.hidden_size,
        num_layers=base_model.num_layers,
        dropout=base_model.dropout_p,
        n_attn_heads=base_model.n_attn_heads,
        cross_pairs=base_model.cross_pairs,
    ).to(data.device)
    model.load_state_dict(base_model.state_dict())

    on_checkpoint_update = None
    if on_best_checkpoint is not None:
        on_checkpoint_update = lambda epoch, best_state: on_best_checkpoint(model, data, args, epoch, best_state)  # noqa: E731

    _run_train_prediction_model(model, data, args, on_checkpoint_update=on_checkpoint_update)
    return evaluate_prediction_model(model, data, neutral_band=getattr(args, "neutral_band", 0.0))


def summarize_checkpoint(
    model_template: PredictionModel, data: "_PreparedData", args: argparse.Namespace, best_state: list,
) -> tuple[PredictionModel, PredictionResult, dict] | None:
    """Build a fresh, fully-evaluated snapshot from `best_state` (the SAME
    list _train_and_evaluate/continue_training's `on_best_checkpoint`
    callback hands over - see their own docstrings) - the "save best model
    so far" endpoint's whole job. Returns None if any asset doesn't have a
    validated checkpoint yet (best_state[i] is still None - e.g. before
    the first validated epoch, or that asset's every score so far was
    NaN) - there is no well-defined "best" for that asset yet.

    `model_template` supplies the ARCHITECTURE only (hidden_size,
    num_layers, dropout, n_attn_heads, cross_pairs, n_channels) - never
    its CURRENT weights, which keep changing throughout training and are
    irrelevant here; a brand new PredictionModel is constructed and only
    `best_state`'s own (already-deep-copied) per-asset weights are loaded
    into it, so this never touches - or races with - the live training
    model's own parameters.

    Returns `(snapshot_model, result, summary)`:
      - `snapshot_model`: the newly built, weight-loaded PredictionModel -
        ready to hand to save_model/save_to_db exactly like any other
        trained model.
      - `result`: its full PredictionResult (see evaluate_prediction_model) -
        everything the Training view's own result panel needs (hit rate,
        confusion matrix, cumulative returns, probabilities, ...).
      - `summary`: `{"train"/"val"/"test": {"loss", "hit_rate", "sharpe"}}` -
        `loss` is NLL + bce_weight*BCE (using the FIRST value if
        `bce_weight` is a sweep list) over that split's decision-day
        predictions (see PredictionResult's own mu_*/sigma_*/z_labels_*),
        `hit_rate` the same mean already in `result.hit_rate_*`, and
        `sharpe` the ANNUALIZED Sharpe of that split's realized daily
        whole-book PnL (risk-parity weight x probability signal, scaled
        to `target_vol` - see models/portfolio_pnl.py's compute_portfolio,
        the same strategy the Evaluation/Training views' own portfolio
        charts use) - None if that split is too short to define a Sharpe
        (fewer than 2 days, or zero variance).
    """
    if any(s is None for s in best_state):
        return None
    n_assets = len(data.pairs)
    snapshot = PredictionModel(
        n_assets=n_assets,
        pairs=data.pairs,
        n_channels=data.n_channels,
        hidden_size=model_template.hidden_size,
        num_layers=model_template.num_layers,
        dropout=model_template.dropout_p,
        n_attn_heads=model_template.n_attn_heads,
        cross_pairs=model_template.cross_pairs,
    ).to(data.device)
    for i in range(n_assets):
        snapshot.assets[i].load_state_dict(best_state[i])

    result = evaluate_prediction_model(snapshot, data, neutral_band=getattr(args, "neutral_band", 0.0))

    direction_horizon = getattr(args, "direction_horizon", 5) or 5
    target_vol = getattr(args, "target_vol", None) or DEFAULT_TARGET_VOL
    cost_bps = getattr(args, "cost_bps", DEFAULT_COST_BPS)
    bce_weight = getattr(args, "bce_weight", 1.0)
    if isinstance(bce_weight, (list, tuple)):
        bce_weight = bce_weight[0] if bce_weight else 1.0
    band = result.neutral_band

    summary = {}
    for split in ("train", "val", "test"):
        mu = torch.as_tensor(getattr(result, f"mu_{split}"))
        # result.sigma_* is CALIBRATED (sigma * sigma_hat - see
        # PredictionResult's docstring); divide sigma_hat back out so this
        # loss is the SAME raw-sigma NLL+BCE quantity training and
        # checkpoint selection actually minimized - comparable with the
        # training log's own val-loss numbers, not offset by the
        # calibration factor.
        sigma_raw = torch.as_tensor(getattr(result, f"sigma_{split}")) / torch.as_tensor(result.sigma_hat)
        z = torch.as_tensor(getattr(result, f"z_labels_{split}"))
        loss = float(gaussian_nll(mu, sigma_raw, z) + bce_weight * direction_bce(mu, sigma_raw, z))
        hit_rate = float(np.mean(getattr(result, f"hit_rate_{split}")))

        probs = getattr(result, f"probabilities_{split}")
        next_returns = getattr(result, f"next_returns_{split}")
        portfolio = compute_portfolio(
            apply_neutral_band(probs, band), next_returns, direction_horizon, target_vol, cost_bps=cost_bps,
        )
        pnl = portfolio["pnl_modulated"]
        sharpe = (
            float(pnl.mean() / pnl.std() * (TRADING_DAYS_PER_YEAR ** 0.5))
            if len(pnl) > 1 and pnl.std() > 0 else None
        )
        summary[split] = {"loss": loss, "hit_rate": hit_rate, "sharpe": sharpe}

    return snapshot, result, summary


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
        bandpass_windows=getattr(model, "bandpass_windows", None), bandpass_order=getattr(model, "bandpass_order", None),
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


def run_pipeline_multi_seed(
    args: argparse.Namespace,
    on_best_checkpoint: Callable[[PredictionModel, "_PreparedData", argparse.Namespace, int, list], None] | None = None,
) -> PredictionResult:
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

    `on_best_checkpoint`, if given, is forwarded to every _train_and_evaluate
    call (see its own docstring) - so it always reflects whichever
    seed/bce_weight combo is CURRENTLY running, not necessarily the
    eventual overall winner across every restart (a caller that only
    cares about n_seeds=1 sees no difference).
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
            candidates.append(_train_and_evaluate(data, lambda_args, on_best_checkpoint=on_best_checkpoint))

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
