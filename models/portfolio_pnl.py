"""Portfolio PnL simulation: risk-parity weights modulated by a model's
directional probabilities, scaled to a target volatility.

Most of this module is pure EVALUATION-time postprocessing on top of a
trained PredictionModel's own outputs (see PredictionResult in
models/portfolio_lstm.py) - it never touches training. The one exception
is `precompute_risk_parity`: it's a pure function of REALIZED DATA (never
of any model's predictions), so it's also reused - unchanged, still just
data - by models/portfolio_lstm.py's train_prediction_model when
`sharpe_weight > 0` (see that module's own docstring) to compute a
differentiable, torch-side version of the SAME strategy as an optional
complementary training objective.

Strategy
--------
Each day t, the model reports a raw probability p_i(t) that pair i's
`direction_horizon`-day-forward move is positive (see PredictionResult's
`probabilities_*`, which are RAW/unsnapped - the neutral band never applies
here, this strategy always has a view). That's converted to a signal in
[-1, 1] via `(p - 0.5) * 2`, multiplied by a risk-parity weight (equal risk
contribution, computed from a rolling covariance of realized returns), then
the whole book is scaled so its OWN volatility hits a configured annualized
`target_vol`.

Trade frequency vs. forecast horizon
-------------------------------------
The portfolio rebalances DAILY, but the probability is a forecast about a
`direction_horizon`-day-forward move - consecutive days' signals are
therefore forecasts of heavily overlapping windows (they share
`direction_horizon - 1` days), which is highly autocorrelated, noisy
information to trade fresh every single day. The probability-modulated
weight is smoothed with a trailing simple moving average over a
`direction_horizon`-day window BEFORE vol-scaling - the standard treatment
for overlapping-horizon forecasts - so the traded position reflects the
model's persistent view rather than day-to-day jitter in when exactly the
window rolled. The unmodulated risk-parity baseline isn't smoothed (it
carries no forecast, so there's no overlap to average out) but IS scaled to
the same target_vol, for a like-for-like comparison.

Causality
---------
Every quantity used to set day t's position - the covariance matrix and the
smoothing window - only ever looks at returns/signals strictly BEFORE t (see
rolling_covariance_matrices/_trailing_moving_average below). Day t's
position is then applied to `next_returns[t]`, the realized return that
`probabilities[t]` was itself a forecast about - the same day/return pairing
PredictionResult already uses for hit-rate/confusion-matrix reporting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

DEFAULT_TARGET_VOL = 0.10  # 10% annualized
DEFAULT_COV_WINDOW = 60  # trailing days for rolling_covariance_matrices/precompute_risk_parity
TRADING_DAYS_PER_YEAR = 252
_COV_EPS = 1e-10  # diagonal regularization - avoids a singular covariance on short/degenerate windows
MAX_VOL_SCALE = 20.0  # caps leverage when a book's own volatility is near zero (e.g. all-neutral signals)


def rolling_covariance_matrices(returns: np.ndarray, window: int, min_periods: int = 20) -> np.ndarray:
    """(T, n) realized daily log returns -> (T, n, n) covariance matrices.

    cov[t] uses ONLY returns[max(0, t-window):t] - STRICTLY before t, so a
    position set using cov[t] never sees the return it's about to be
    applied to. Below `min_periods` observations the sample covariance is
    too noisy to size a position from (2-3 points can look almost
    perfectly, spuriously correlated) - cov[t] stays NaN until at least
    `min_periods` prior returns are available, same warm-up floor as the
    rest of this codebase's rolling features (see rolling_stats_window).
    """
    t, n = returns.shape
    cov = np.full((t, n, n), np.nan, dtype=np.float64)
    for i in range(1, t):
        window_data = returns[max(0, i - window):i]
        if len(window_data) < min_periods:
            continue
        cov[i] = np.cov(window_data, rowvar=False).reshape(n, n) + _COV_EPS * np.eye(n)
    return cov


def risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    """Equal-risk-contribution weights for ONE (n, n) covariance matrix -
    long-only, sum to 1: asset i contributes w_i * (cov @ w)_i, and every
    asset's contribution is pushed to the same share of total portfolio
    variance. No closed form once assets are correlated, so this is solved
    numerically (SLSQP) starting from an equal-weight guess.

    `cov` is rescaled to O(1) diagonal entries before optimizing - daily
    return variances are ~1e-6, which puts the raw objective at ~1e-12;
    SLSQP's finite-difference gradient rounds to noise at that scale and
    it exits after one iteration, silently returning the untouched
    equal-weight starting guess (confirmed against a real FX covariance
    matrix - see smoke_test.py's 8a). The ERC solution is scale-invariant
    (cov -> k*cov leaves argmin w unchanged), so this changes nothing about
    the answer, only its numerical conditioning.
    """
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])

    scale = np.trace(cov) / n
    cov = cov / scale

    def objective(w: np.ndarray) -> float:
        port_var = w @ cov @ w
        contrib = w * (cov @ w)
        return float(np.sum((contrib - port_var / n) ** 2))

    result = minimize(
        objective, np.full(n, 1.0 / n), method="SLSQP",
        bounds=[(1e-6, 1.0)] * n, constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"maxiter": 500, "ftol": 1e-14},
    )
    w = np.clip(result.x, 1e-6, None)
    return w / w.sum()


def _trailing_moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing simple moving average over `window` rows (INCLUDING row i),
    skipping rows that are still NaN (no valid signal that day yet) - NaN
    only where NONE of [i-window+1, i] are valid.
    """
    t = x.shape[0]
    out = np.full_like(x, np.nan)
    for i in range(t):
        chunk = x[max(0, i - window + 1):i + 1]
        valid = ~np.isnan(chunk).any(axis=1)
        if valid.any():
            out[i] = np.nanmean(chunk[valid], axis=0)
    return out


def _scale_to_target_vol(weights: np.ndarray, cov: np.ndarray, target_vol_daily: float) -> np.ndarray:
    port_vol = float(np.sqrt(max(weights @ cov @ weights, 1e-16)))
    scale = min(target_vol_daily / port_vol, MAX_VOL_SCALE)
    return weights * scale


def precompute_risk_parity(next_returns: np.ndarray, cov_window: int = DEFAULT_COV_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """(T, n) realized daily log returns -> ((T, n) risk-parity weights,
    (T, n, n) covariance matrices), both causal (see
    rolling_covariance_matrices) and a pure function of REALIZED DATA -
    never of any model's predictions. rp_weights[t] is NaN wherever cov[t]
    is (not enough trailing history yet - see rolling_covariance_matrices).

    Depending on nothing but data means this can be computed ONCE - before
    training starts, before any seed/restart - and reused as a genuine
    CONSTANT everywhere it's needed: by compute_portfolio (evaluation-time
    postprocessing, below) and by train_prediction_model's optional
    portfolio-Sharpe training phase (models/portfolio_lstm.py) alike. It is
    NEVER optimized - there's nothing to backprop through here, the
    equal-risk-contribution solve (risk_parity_weights) isn't even
    differentiable (it's an external SLSQP call, not a torch op).
    """
    t, n = next_returns.shape
    cov = rolling_covariance_matrices(next_returns, cov_window)
    rp_weights = np.full((t, n), np.nan)
    for i in range(t):
        if np.isnan(cov[i]).any():
            continue
        rp_weights[i] = risk_parity_weights(cov[i])
    return rp_weights, cov


def compute_portfolio(
    probabilities: np.ndarray,
    next_returns: np.ndarray,
    direction_horizon: int,
    target_vol: float = DEFAULT_TARGET_VOL,
    cov_window: int = DEFAULT_COV_WINDOW,
) -> dict:
    """Drive the whole strategy over a (T, n) probabilities/next_returns
    pair (one split - train, val, or test; see PredictionResult).

    Returns a dict of (T, n) arrays `positions_modulated`/`positions_baseline`
    (NaN on days without enough history to size a position - the first
    `cov_window`+`direction_horizon`-ish days), (T, n) arrays
    `pnl_per_asset_modulated`/`pnl_per_asset_baseline` (each asset's OWN
    `position_i * next_return_i` - these sum across assets, per day, to
    `pnl_modulated`/`pnl_baseline` below) and their cumulative
    (`cumulative_pnl_per_asset_modulated`/`_baseline`), and (T,) arrays
    `pnl_modulated`/`pnl_baseline`/`cumulative_pnl_modulated`/
    `cumulative_pnl_baseline` (the WHOLE BOOK, summed across assets). All
    0 (not NaN) on days without enough history to size a position, so
    every cumsum is always defined.
    """
    t, n = probabilities.shape
    target_vol_daily = target_vol / np.sqrt(TRADING_DAYS_PER_YEAR)

    rp_weights, cov = precompute_risk_parity(next_returns, cov_window)

    raw_modulated = np.full((t, n), np.nan)
    for i in range(t):
        if np.isnan(cov[i]).any():
            continue
        signal = (probabilities[i] - 0.5) * 2.0
        raw_modulated[i] = rp_weights[i] * signal

    smoothed_modulated = _trailing_moving_average(raw_modulated, direction_horizon)

    positions_modulated = np.full((t, n), np.nan)
    positions_baseline = np.full((t, n), np.nan)
    for i in range(t):
        if np.isnan(cov[i]).any():
            continue
        positions_baseline[i] = _scale_to_target_vol(rp_weights[i], cov[i], target_vol_daily)
        if not np.isnan(smoothed_modulated[i]).any():
            positions_modulated[i] = _scale_to_target_vol(smoothed_modulated[i], cov[i], target_vol_daily)

    def _pnl_per_asset(positions: np.ndarray) -> np.ndarray:
        # A day with NO valid position (any asset NaN, e.g. still in the
        # warm-up window) contributes exactly 0 to EVERY asset that day -
        # not just the ones that happen to be NaN - so a given asset's own
        # cumulative pnl never silently skips a day the whole book sat out.
        valid = ~np.isnan(positions).any(axis=1, keepdims=True)
        return np.where(valid, np.nan_to_num(positions) * next_returns, 0.0)

    pnl_per_asset_modulated = _pnl_per_asset(positions_modulated)
    pnl_per_asset_baseline = _pnl_per_asset(positions_baseline)
    pnl_modulated = pnl_per_asset_modulated.sum(axis=1)
    pnl_baseline = pnl_per_asset_baseline.sum(axis=1)

    return {
        "positions_modulated": positions_modulated,
        "positions_baseline": positions_baseline,
        "pnl_per_asset_modulated": pnl_per_asset_modulated,
        "pnl_per_asset_baseline": pnl_per_asset_baseline,
        "cumulative_pnl_per_asset_modulated": np.cumsum(pnl_per_asset_modulated, axis=0),
        "cumulative_pnl_per_asset_baseline": np.cumsum(pnl_per_asset_baseline, axis=0),
        "pnl_modulated": pnl_modulated,
        "pnl_baseline": pnl_baseline,
        "cumulative_pnl_modulated": np.cumsum(pnl_modulated),
        "cumulative_pnl_baseline": np.cumsum(pnl_baseline),
    }


def annual_sharpe_table(dates, pnl_modulated: np.ndarray, pnl_baseline: np.ndarray) -> dict:
    """Group daily whole-book PnL (see compute_portfolio's `pnl_modulated`/
    `pnl_baseline`) by CALENDAR YEAR and report each year's own annualized
    Sharpe ratio (`mean(daily_pnl) / std(daily_pnl) * sqrt(252)`) for both
    the probability-modulated book and the unmodulated risk-parity
    baseline, alongside the year's day count and cumulative return - a
    reviewer's first question about any backtest is usually "is this
    driven by one lucky year", which a single all-period Sharpe number
    can't answer.

    A year with fewer than 2 days (e.g. the partial first/last calendar
    year of a split) reports `None` for its Sharpe (std of 0-1 points is
    undefined) but still reports its day count and cumulative return.
    """
    years = pd.DatetimeIndex(dates).year
    table: dict[str, dict] = {}
    for year in sorted(set(years.tolist())):
        mask = years == year
        n_days = int(mask.sum())
        mod, base = pnl_modulated[mask], pnl_baseline[mask]

        def _sharpe(pnl: np.ndarray) -> float | None:
            if len(pnl) < 2 or pnl.std() <= 0:
                return None
            return float(pnl.mean() / pnl.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

        table[str(year)] = {
            "n_days": n_days,
            "sharpe_modulated": _sharpe(mod),
            "sharpe_baseline": _sharpe(base),
            "return_modulated": float(mod.sum()),
            "return_baseline": float(base.sum()),
        }
    return table


def latest_position(
    probabilities: np.ndarray,
    next_returns: np.ndarray,
    latest_probabilities: np.ndarray,
    direction_horizon: int,
    target_vol: float = DEFAULT_TARGET_VOL,
    cov_window: int = DEFAULT_COV_WINDOW,
) -> dict:
    """The position to book for TODAY - the not-yet-realized day
    `latest_probabilities` (shape (n,), today's probability per pair, e.g.
    from api/server.py's _predict_latest_probabilities) forecasts.

    `probabilities`/`next_returns` are the most recent historical split
    (typically the test split) - just enough trailing history to (a) size
    today's covariance matrix and (b) fill out the direction_horizon-day
    moving average the modulated weight is smoothed over (see
    compute_portfolio's docstring on overlapping-horizon forecasts) so
    today isn't averaged in isolation. Appends `latest_probabilities` as
    one extra day (with a placeholder 0 return, since today hasn't
    realized yet) and reuses compute_portfolio's own causal machinery -
    the appended day's covariance matrix only ever looks at the REAL
    historical returns before it, never the placeholder.
    """
    extended_probabilities = np.vstack([probabilities, np.asarray(latest_probabilities).reshape(1, -1)])
    extended_returns = np.vstack([next_returns, np.zeros((1, next_returns.shape[1]))])
    out = compute_portfolio(extended_probabilities, extended_returns, direction_horizon, target_vol, cov_window)
    return {
        "position_modulated": out["positions_modulated"][-1],
        "position_baseline": out["positions_baseline"][-1],
    }
