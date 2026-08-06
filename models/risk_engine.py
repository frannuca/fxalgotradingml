"""Risk engine: a SECOND, independently-trained model that sits ON TOP of
an already-trained PredictionModel (frozen - see train_risk_engine), and
learns a per-asset, per-day ATTENUATION factor applied to the prediction
stage's own already-fully-formed (target-vol-scaled) position, in an
attempt to reduce exposure ahead of realized losses.

Where attenuation is applied
-----------------------------
Critically, attenuation multiplies the position AFTER target-vol scaling
(models/portfolio_lstm.py's compute_target_vol_positions_torch /
models/portfolio_pnl.py's compute_portfolio's own `_scale_to_target_vol`),
never before: `_scale_to_target_vol` renormalizes the whole book to hit a
FIXED annualized vol every day, so attenuating the pre-scaling weight
would just get undone by that very next step and have ZERO net effect.
Applying it downstream instead means the realized book vol legitimately
drops below `target_vol` whenever the engine de-risks - the intended
behavior, not a bug (see api/server.py's own portfolio payload, which
reports a `positions_risk_attenuated`/`cumulative_pnl_risk_attenuated`
series alongside the existing `positions_modulated` one so both can be
compared directly).

Inputs
------
Per asset, per day: [pre-attenuation position weight, rolling skewness,
rolling excess kurtosis, cross-moving-average(s), Butterworth
bandpass(s)] - see n_risk_channels_per_asset/build_risk_stats_dataframe.
Skewness/kurtosis/CMA/bandpass are computed HERE, independently of
whatever the underlying PredictionModel's own `features` config selects
(the prediction stage may or may not ALSO use CMA/bandpass as directional
inputs - see models/portfolio_lstm.py's DEFAULT_FEATURES; skewness/
kurtosis, by contrast, are risk-engine-exclusive by design and no longer
part of the default per-asset predictor input set at all). Every value is
strictly CAUSAL - a function of data through day t only, the same
discipline as every other rolling feature in this codebase.

The position weight itself is precomputed ONCE (no_grad - the underlying
PredictionModel is FROZEN throughout, see train_risk_engine) via
models/portfolio_lstm.py's compute_target_vol_positions_torch - the exact
same quantity models/portfolio_pnl.py's compute_portfolio calls
`positions_modulated`.

`risk_lookback` counts DECISION DAYS (one row per day the underlying
PredictionModel actually produced a decision-day output for), not raw
calendar days - see make_risk_sequences.

Architecture
------------
ONE joint model (not one per asset, unlike PredictionModel's independent
per-asset AssetLSTMs) - risk is fundamentally a portfolio-level,
cross-asset-coupled concept (the same target-vol scaling that produces the
input weights already mixes every asset's variance jointly via `w' cov w`
- see compute_target_vol_positions_torch), and the output is a PER-ASSET
attenuation vector, so a single model reading every asset's weight/stats
together, free to differentiate its output per asset, is the natural fit.
Same LSTM + causal self-attention block as AssetLSTM (see
models/portfolio_lstm.py's PredictionModel), just reshaped for this joint,
per-asset-output role: input width is `n_assets * risk_channels_per_asset`
(every asset's own channels concatenated), output width is `n_assets` (one
raw score per asset, not a (mu, sigma) pair).

Bounds
------
Raw output -> sigmoid -> linearly mapped into (min_risk_att, max_risk_att)
- the SAME "linear map into a user-provided (min, max)" pattern as
models/portfolio_lstm.py's resolve_signal_bounds/signal_range. Recommended
default (0.0, 1.0): attenuation can only REDUCE exposure, never amplify
it - matching "try to prevent losses" - though nothing stops a caller from
choosing a wider or asymmetric range (e.g. allowing >1.0 to also permit
the engine to LEAN IN during favorable conditions).

Training objective
-------------------
A downside-focused (Sortino-style) objective - see
_non_overlapping_sortino_torch - rather than plain Sharpe: a Sharpe-style
objective symmetrically penalizes upside variance too, which isn't "risk"
in the sense this engine is meant to manage. `full_exposure_penalty` (see
train_risk_engine) regularizes the output toward max_risk_att (full
exposure) so the engine must actually EARN a de-risking move via the
objective, rather than trivially collapsing exposure to minimize variance
over a noisy, short backtest - a real failure mode for any model free to
"solve" a risk-adjusted-return objective by simply not participating.
"""

from __future__ import annotations

import argparse
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

from models.portfolio_lstm import (
    DEFAULT_BANDPASS_ORDER,
    DEFAULT_CUTOFF_DATE,
    TrainingStopped,
    _assert_finite_grad,
    _prepare_data,
    butterworth_bandpass_features,
    compute_target_vol_positions_torch,
    cross_moving_averages,
    get_device,
    load_close_prices,
    logger,
    resolve_signal_bounds,
    rolling_moment_features,
    to_log_returns,
)
from models.portfolio_pnl import DEFAULT_COST_BPS, DEFAULT_TARGET_VOL

#: Defaults for a NEW risk engine - deliberately narrower than
#: PredictionModel's own (hidden_size=16 default there too, but this model
#: has a much narrower job and less data-per-parameter, so keeping its
#: capacity modest is itself part of guarding against overfitting a
#: secondary risk signal - see this module's own docstring).
DEFAULT_RISK_LOOKBACK = 20
#: Trailing window (days) build_risk_stats_dataframe's own skew/kurtosis
#: computation uses - INDEPENDENT of the base predictor's own
#: `rolling_stats_window` (see models/portfolio_lstm.py's DEFAULT_CONFIG):
#: since skew/kurtosis are risk-engine-exclusive inputs now (never fed to
#: the per-asset predictor - see this module's own docstring), tying this
#: to the predictor's window would coincidentally couple two otherwise-
#: unrelated hyperparameters. Set via `args.risk_rolling_stats_window` in
#: train_risk_engine - see its own docstring.
DEFAULT_RISK_ROLLING_STATS_WINDOW = 20
DEFAULT_MIN_RISK_ATT = 0.0
DEFAULT_MAX_RISK_ATT = 1.0
DEFAULT_RISK_HIDDEN_SIZE = 16
DEFAULT_RISK_NUM_LAYERS = 1
DEFAULT_RISK_DROPOUT = 0.1
DEFAULT_RISK_N_ATTN_HEADS = 4
DEFAULT_RISK_SORTINO_WINDOW = 20
#: Weight of the "stay near max_risk_att (full exposure)" regularizer
#: added to the Sortino-maximizing loss - see this module's own docstring
#: on why a risk-reduction overlay needs an anchor against collapsing
#: exposure by default. 0.0 disables it entirely.
DEFAULT_FULL_EXPOSURE_PENALTY = 0.05


def n_risk_channels_per_asset(risk_cma_windows: list | None = None, risk_bandpass_windows: list | None = None) -> int:
    """weight + skew + kurt (always) + one channel per (short, long) window
    pair in `risk_cma_windows`/`risk_bandpass_windows` (each independently
    opt-in, mirroring models/portfolio_lstm.py's own n_channels_per_pair
    convention for "cma"/"bandpass")."""
    return 3 + len(risk_cma_windows or []) + len(risk_bandpass_windows or [])


def build_risk_stats_dataframe(
    returns: pd.DataFrame,
    pairs: list[str],
    rolling_stats_window: int,
    risk_cma_windows: list | None = None,
    risk_bandpass_windows: list | None = None,
    risk_bandpass_order: int = DEFAULT_BANDPASS_ORDER,
) -> pd.DataFrame:
    """Asset-major [skew, kurt, cma..., bandpass...] channels (NOT
    including the position weight - that's a MODEL output, not a data
    function, added separately by the caller - see make_risk_sequences)
    for every pair in `pairs`, aligned to `returns`'s own index. Computed
    independently of whatever the underlying PredictionModel's own
    `features` config selects - see this module's own docstring.

    Reuses models/portfolio_lstm.py's rolling_moment_features (for skew/
    kurt only - vol is deliberately not included here; this module's own
    spec is skewness/kurtosis/CMA/bandpass, not volatility, which the
    predictor's own `target_vol` already accounts for downstream) /
    cross_moving_averages / butterworth_bandpass_features directly, so any
    future change to those functions' own causality/definition applies
    here too, automatically.
    """
    risk_cma_windows = risk_cma_windows or []
    risk_bandpass_windows = risk_bandpass_windows or []
    moments = rolling_moment_features(returns, rolling_stats_window)
    cmas = cross_moving_averages(returns, risk_cma_windows) if risk_cma_windows else None
    bandpasses = (
        butterworth_bandpass_features(returns, risk_bandpass_windows, risk_bandpass_order)
        if risk_bandpass_windows else None
    )

    out = pd.DataFrame(index=returns.index)
    for pair in pairs:
        out[f"{pair}_skew"] = moments[f"{pair}_skew"]
        out[f"{pair}_kurt"] = moments[f"{pair}_kurt"]
        for short, long_ in risk_cma_windows:
            out[f"{pair}_cma_{short}_{long_}"] = cmas[(pair, short, long_)]
        for short, long_ in risk_bandpass_windows:
            out[f"{pair}_bp_{short}_{long_}"] = bandpasses[(pair, short, long_)]

    # Same warm-up handling as models/portfolio_lstm.py's build_feature_dataframe:
    # forward-fill mid-series gaps, then treat any still-leading NaN (not
    # enough trailing history yet) as a neutral 0.0 - never back-filled,
    # which would leak a later day's value into an earlier row.
    return out.ffill().fillna(0.0)


def _make_windows(values: np.ndarray, lookback: int) -> np.ndarray:
    """(T, C) -> (T - lookback + 1, lookback, C) sliding windows - window i
    covers values[i : i+lookback], decision day = values[i+lookback-1].
    Mirrors models/portfolio_lstm.py's make_sequences own "last row is the
    decision day" convention, without needing a forward label: the risk
    engine's own training objective (see train_risk_engine) is driven
    directly by REALIZED next-day returns applied to the FINAL attenuated
    position, not a supervised regression target. A plain Python loop
    (values.shape[0] is at most a few thousand rows, called once per
    split) - the same simplicity models/portfolio_pnl.py's
    rolling_covariance_matrices already accepts for a comparable one-time
    precomputation.
    """
    t, c = values.shape
    n = t - lookback + 1
    if n <= 0:
        return np.empty((0, lookback, c), dtype=values.dtype)
    out = np.empty((n, lookback, c), dtype=values.dtype)
    for i in range(n):
        out[i] = values[i : i + lookback]
    return out


class RiskEngine(nn.Module):
    """See this module's own docstring for the full design. ONE joint
    LSTM + causal self-attention block (architecturally identical to
    models/portfolio_lstm.py's AssetLSTM) over EVERY asset's own
    [weight, skew, kurt, cma..., bandpass...] channels concatenated,
    outputting a RAW (unbounded) score per asset at every timestep - only
    the window's LAST (decision-day) timestep is ever actually used (see
    train_risk_engine/evaluate_risk_engine) - mapped into
    (min_risk_att, max_risk_att) via `attenuation_from_raw` below, kept
    OUTSIDE this module's own forward() so a caller can defer the bounds
    map until after slicing out the decision day (cheaper - the sigmoid
    only needs to run over what's actually used).
    """

    def __init__(
        self,
        n_assets: int,
        risk_channels_per_asset: int,
        hidden_size: int = DEFAULT_RISK_HIDDEN_SIZE,
        num_layers: int = DEFAULT_RISK_NUM_LAYERS,
        dropout: float = DEFAULT_RISK_DROPOUT,
        n_attn_heads: int = DEFAULT_RISK_N_ATTN_HEADS,
    ):
        super().__init__()
        if hidden_size % n_attn_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by n_attn_heads ({n_attn_heads})")
        self.n_assets = n_assets
        self.risk_channels_per_asset = risk_channels_per_asset
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.n_attn_heads = n_attn_heads
        input_size = n_assets * risk_channels_per_asset
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(hidden_size, n_attn_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, n_assets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, n_assets * risk_channels_per_asset).
        # Returns (batch, lookback, n_assets) - a RAW (unbounded) score per
        # asset at every timestep; see attenuation_from_raw for the bounds
        # map, and this class's own docstring on why only the last
        # timestep is actually used downstream.
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        lookback = lstm_out.shape[1]
        causal_mask = torch.triu(
            torch.full((lookback, lookback), float("-inf"), device=x.device, dtype=lstm_out.dtype), diagonal=1,
        )
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out, attn_mask=causal_mask, need_weights=False)
        h = self.attn_norm(lstm_out + attn_out)
        return self.head(h)  # (batch, lookback, n_assets)


def attenuation_from_raw(raw: torch.Tensor, min_risk_att: float, max_risk_att: float) -> torch.Tensor:
    """RiskEngine's own raw score -> attenuation, linearly mapped into
    (min_risk_att, max_risk_att) via a sigmoid - the same "bounded map"
    pattern as models/portfolio_lstm.py's resolve_signal_bounds/
    signal_range, just with a sigmoid (raw score unbounded in both
    directions) instead of the [0, 1]-already probability those use.
    """
    return min_risk_att + (max_risk_att - min_risk_att) * torch.sigmoid(raw)


def risk_engine_checkpoint_dict(
    risk_engine: RiskEngine, pairs: list[str], risk_lookback: int,
    risk_cma_windows: list | None, risk_bandpass_windows: list | None, risk_bandpass_order: int,
    rolling_stats_window: int, min_risk_att: float, max_risk_att: float,
) -> dict:
    """Build the nested checkpoint dict PredictionModel._checkpoint_dict
    bundles under its own "risk_engine" key when a risk engine is attached
    (see models/portfolio_lstm.py) - None there for any model that never
    had one trained, so every EXISTING checkpoint stays loadable unchanged.

    `pairs` is persisted here (not just recovered from the parent
    PredictionModel's own `pairs`) so the risk engine's own per-asset
    output order is self-describing even if read independently of its
    parent model. `rolling_stats_window` (the window build_risk_stats_dataframe's
    own skew/kurt computation used at training time) is a genuinely
    INDEPENDENT hyperparameter from the parent model's OWN
    `rolling_stats_window` (see DEFAULT_RISK_ROLLING_STATS_WINDOW's own
    comment - skew/kurtosis are risk-engine-exclusive inputs now, so
    there's no reason for the two windows to be coupled at all) - so
    evaluate_risk_engine never has to reach into its parent to reproduce
    the exact same feature values it trained on.
    """
    return {
        "config": {
            "n_assets": risk_engine.n_assets,
            "risk_channels_per_asset": risk_engine.risk_channels_per_asset,
            "hidden_size": risk_engine.hidden_size,
            "num_layers": risk_engine.num_layers,
            "dropout": risk_engine.dropout_p,
            "n_attn_heads": risk_engine.n_attn_heads,
        },
        "state_dict": risk_engine.state_dict(),
        "pairs": list(pairs),
        "risk_lookback": int(risk_lookback),
        "risk_cma_windows": [list(w) for w in (risk_cma_windows or [])],
        "risk_bandpass_windows": [list(w) for w in (risk_bandpass_windows or [])],
        "risk_bandpass_order": int(risk_bandpass_order),
        "rolling_stats_window": int(rolling_stats_window),
        "min_risk_att": float(min_risk_att),
        "max_risk_att": float(max_risk_att),
    }


def risk_engine_from_checkpoint(checkpoint: dict) -> RiskEngine:
    """Inverse of risk_engine_checkpoint_dict - reconstructs a RiskEngine
    (weights + every hyperparameter needed to rebuild its own input
    pipeline) from a PredictionModel checkpoint's "risk_engine" sub-dict.
    """
    engine = RiskEngine(**checkpoint["config"])
    engine.load_state_dict(checkpoint["state_dict"])
    engine.eval()
    engine.pairs = checkpoint["pairs"]
    engine.risk_lookback = int(checkpoint["risk_lookback"])
    engine.risk_cma_windows = checkpoint.get("risk_cma_windows", [])
    engine.risk_bandpass_windows = checkpoint.get("risk_bandpass_windows", [])
    engine.risk_bandpass_order = int(checkpoint.get("risk_bandpass_order", DEFAULT_BANDPASS_ORDER))
    engine.rolling_stats_window = int(checkpoint.get("rolling_stats_window", 20))
    engine.min_risk_att = float(checkpoint["min_risk_att"])
    engine.max_risk_att = float(checkpoint["max_risk_att"])
    return engine


def _non_overlapping_sortino_torch(pnl: torch.Tensor, window: int, target: float = 0.0, eps: float = 1e-8) -> torch.Tensor:
    """Sortino ratio (mean excess-over-target / downside deviation) over
    NON-OVERLAPPING `window`-day chunks, walking BACKWARD from the last
    day - the SAME chunking convention as models/portfolio_lstm.py's
    _non_overlapping_sharpe_torch (see its own docstring: chunk 0 is the
    final `window` days, the oldest leftover remainder is dropped
    entirely, T < window returns an empty tensor rather than shrinking the
    window) - only the DENOMINATOR differs: instead of the full standard
    deviation, only NEGATIVE deviations below `target` (default 0.0 - "any
    daily loss") count toward the risk measure, so a day that
    OUTPERFORMS `target` never penalizes the ratio the way it would under
    plain Sharpe - the point of a downside-focused objective for a
    risk-reduction overlay whose job is preventing LOSSES, not smoothing
    away genuinely profitable variance too.

    `downside deviation = sqrt(mean(min(pnl - target, 0) ** 2) + eps)` -
    the population mean (not `unbiased=True`'s N-1 correction
    _non_overlapping_sharpe_torch's own std() uses), the standard Sortino
    convention, over EVERY day in the chunk (not just the down days). `eps`
    is added INSIDE the sqrt, not just the final division - a chunk with
    zero down days has `mean(downside**2)` of EXACTLY 0, and
    `d/du sqrt(u)` at `u=0` is infinite, so without this a perfectly
    loss-free chunk would silently blow up the backward pass into NaN
    gradients (confirmed the hard way - see this module's own test
    coverage) rather than correctly reading as "no downside risk at all".
    """
    t = pnl.shape[0]
    window = max(window, 2)
    n_full_chunks = t // window
    if n_full_chunks == 0:
        return pnl.new_zeros((0,))
    reversed_pnl = pnl.flip(0)
    full_chunks = reversed_pnl[: n_full_chunks * window].reshape(n_full_chunks, window)
    downside = torch.clamp(target - full_chunks, min=0.0)
    downside_dev = torch.sqrt((downside ** 2).mean(dim=-1) + eps)
    return (full_chunks.mean(dim=-1) - target) / (downside_dev + eps)


def _risk_loss_from_attenuation(
    attenuation: torch.Tensor, positions: torch.Tensor, next_returns: torch.Tensor,
    sortino_window: int, cost_bps: float, max_risk_att: float, full_exposure_penalty: float,
) -> tuple[torch.Tensor, float | None]:
    """Shared by train_risk_engine's train/val passes: applies
    `attenuation` (T, n_assets) to the already-target-vol-scaled
    `positions` (T, n_assets - see compute_target_vol_positions_torch),
    nets out linear transaction costs on the FINAL (post-attenuation)
    position's own day-to-day change (same convention as
    models/portfolio_lstm.py's _portfolio_sharpe_loss_from_predictions),
    and returns `(loss, realized_sortino_or_None)` - `loss` is the
    negative averaged Sortino (see _non_overlapping_sortino_torch) PLUS
    `full_exposure_penalty * mean((max_risk_att - attenuation) ** 2)` (see
    this module's own docstring on why de-risking must be earned, not
    trivially defaulted to), `realized_sortino_or_None` is the plain
    (unregularized) Sortino alone, for human-legible reporting - None if
    there were too few days for even one `sortino_window`-day chunk (see
    _non_overlapping_sortino_torch), in which case `loss` is JUST the
    regularization term (a neutral, non-crashing fallback for a too-short
    split, same spirit as the prediction model's own empty-Sharpe handling).
    """
    final_position = positions * attenuation
    gross_pnl = (final_position * next_returns).sum(dim=-1)
    position_deltas = torch.diff(final_position, dim=0, prepend=torch.zeros_like(final_position[:1]))
    daily_costs = cost_bps * 1e-4 * position_deltas.abs().sum(dim=-1)
    daily_pnl = gross_pnl - daily_costs
    penalty = full_exposure_penalty * ((max_risk_att - attenuation) ** 2).mean()

    period_sortinos = _non_overlapping_sortino_torch(daily_pnl, sortino_window)
    if period_sortinos.numel() == 0:
        return penalty, None
    sortino = period_sortinos.mean()
    return -sortino + penalty, float(sortino.item())


def train_risk_engine(
    base_model: "PredictionModel",  # noqa: F821 - forward ref, see models.portfolio_lstm
    args: argparse.Namespace,
    on_epoch: Callable[[int, int, float | None, float | None, float | None], None] | None = None,
    stop_check: Callable[[], bool] | None = None,
) -> tuple[RiskEngine, dict]:
    """Train a NEW RiskEngine on top of `base_model`, which is used
    STRICTLY FROZEN throughout (no gradient ever flows into its own
    parameters - see this module's own docstring on why this is a
    freeze-and-stack SECOND stage, not a joint retraining) - mirrors
    models/portfolio_lstm.py's continue_training in spirit (warm-start
    from an existing, already-trained model) but trains an entirely
    DIFFERENT set of parameters on top, rather than continuing to update
    `base_model`'s own.

    `args` carries the SAME data-window fields continue_training's own
    `args` does (years, cutoff_date, train_frac, test_frac, device) plus
    this module's own hyperparameters: risk_lookback,
    risk_rolling_stats_window, risk_cma_windows, risk_bandpass_windows,
    risk_bandpass_order, min_risk_att, max_risk_att, risk_hidden_size,
    risk_num_layers, risk_dropout, risk_n_attn_heads, risk_epochs, risk_lr,
    risk_weight_decay, risk_sortino_window, full_exposure_penalty, cost_bps.

    `on_epoch(epoch, epochs, train_sortino, val_sortino, best_val_sortino)`,
    if given, is called after every epoch - `train_sortino`/`val_sortino`
    are the realized (unregularized) Sortino ratios that epoch (None if
    that split had too few days for even one sortino_window-day chunk -
    see _risk_loss_from_attenuation), `best_val_sortino` the running best
    validation Sortino so far this run (None until the first validated
    epoch). `stop_check()`, if given and returning True, raises
    TrainingStopped at the next epoch boundary - same cooperative-stop
    contract as models/portfolio_lstm.py's train_prediction_model.

    Returns `(risk_engine, summary)` - `summary` is
    `{"train_sortino", "val_sortino"}`, the FINAL (best-checkpoint-restored)
    realized Sortino on each split.
    """
    device = get_device(getattr(args, "device", "auto"))
    base_model = base_model.to(device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    data = _prepare_data(
        args, x_mean=base_model.x_mean, x_std=base_model.x_std, pairs=base_model.pairs,
        lookback=base_model.lookback, features=base_model.features, cma_windows=base_model.cma_windows,
        bandpass_windows=base_model.bandpass_windows, bandpass_order=base_model.bandpass_order,
        direction_horizon=getattr(base_model, "direction_horizon", None),
        rolling_stats_window=getattr(base_model, "rolling_stats_window", None),
    )

    risk_lookback = getattr(args, "risk_lookback", None) or DEFAULT_RISK_LOOKBACK
    risk_cma_windows = getattr(args, "risk_cma_windows", None) or []
    risk_bandpass_windows = getattr(args, "risk_bandpass_windows", None) or []
    risk_bandpass_order = getattr(args, "risk_bandpass_order", None) or DEFAULT_BANDPASS_ORDER
    min_risk_att = getattr(args, "min_risk_att", None)
    min_risk_att = DEFAULT_MIN_RISK_ATT if min_risk_att is None else min_risk_att
    max_risk_att = getattr(args, "max_risk_att", None)
    max_risk_att = DEFAULT_MAX_RISK_ATT if max_risk_att is None else max_risk_att
    if min_risk_att >= max_risk_att:
        raise ValueError(f"min_risk_att ({min_risk_att}) must be < max_risk_att ({max_risk_att})")
    cost_bps = getattr(args, "cost_bps", None)
    cost_bps = DEFAULT_COST_BPS if cost_bps is None else cost_bps
    target_vol = getattr(args, "target_vol", None) or getattr(base_model, "target_vol", DEFAULT_TARGET_VOL)
    neutral_band = getattr(args, "neutral_band", None)
    neutral_band = getattr(base_model, "neutral_band", 0.0) if neutral_band is None else neutral_band
    sortino_window = getattr(args, "risk_sortino_window", None) or DEFAULT_RISK_SORTINO_WINDOW
    full_exposure_penalty = getattr(args, "full_exposure_penalty", None)
    full_exposure_penalty = DEFAULT_FULL_EXPOSURE_PENALTY if full_exposure_penalty is None else full_exposure_penalty
    epochs = getattr(args, "risk_epochs", None) or 100
    lr = getattr(args, "risk_lr", None) or 1e-3
    weight_decay = getattr(args, "risk_weight_decay", None) or 0.0

    signal_min, signal_max = resolve_signal_bounds(data.pairs, getattr(base_model, "signal_range", None))
    signal_min_t = torch.as_tensor(signal_min, dtype=torch.float32, device=device)
    signal_max_t = torch.as_tensor(signal_max, dtype=torch.float32, device=device)

    # Raw daily log returns over the SAME fetched window _prepare_data used
    # (same pairs/years/cutoff_date) - re-fetched here (a cheap Postgres
    # read, not a live API call - see models/portfolio_lstm.py's
    # load_close_prices) rather than threading a new field through the
    # widely-shared _PreparedData dataclass for this one, second-stage-only
    # consumer.
    cutoff_date = getattr(args, "cutoff_date", None) or DEFAULT_CUTOFF_DATE
    prices = load_close_prices(data.pairs, years=args.years, cutoff_date=cutoff_date)
    returns = to_log_returns(prices)
    # The risk engine's OWN dedicated window (see DEFAULT_RISK_ROLLING_STATS_WINDOW's
    # own comment) - NOT the base predictor's `rolling_stats_window`, which
    # has nothing to do with skew/kurtosis anymore now that they're
    # risk-engine-exclusive inputs.
    risk_rolling_stats_window = getattr(args, "risk_rolling_stats_window", None) or DEFAULT_RISK_ROLLING_STATS_WINDOW
    risk_stats = build_risk_stats_dataframe(
        returns, data.pairs, risk_rolling_stats_window, risk_cma_windows, risk_bandpass_windows, risk_bandpass_order,
    )

    n_assets = len(data.pairs)
    per_asset_channels = n_risk_channels_per_asset(risk_cma_windows, risk_bandpass_windows) - 1  # excluding weight
    n_input_channels = n_assets * (per_asset_channels + 1)
    # Recovered from the FROZEN base_model's own checkpoint, matching
    # exactly what _prepare_data above was told to use - the risk engine's
    # own weight input must be computed with the SAME smoothing window the
    # base model's positions were always meant to use, not a free `args`
    # override (unlike continue_training, where direction_horizon changing
    # the RETRAINED model's own labels is a deliberate, separate knob).
    direction_horizon = getattr(base_model, "direction_horizon", 5) or 5

    def _precompute_split(X, next_returns_np, rp_weights, cov, dates):
        """Frozen-model forward pass -> pre-attenuation positions (no_grad,
        base_model never trains here) + this split's own risk-stats rows,
        aligned to `dates` (see this module's own docstring on why
        risk_lookback counts DECISION days), interleaved per asset as
        [weight_i, skew_i, kurt_i, cma_i..., bp_i...] -> risk_lookback-day
        windows. Returns (windows_x, positions, next_returns_t) - all
        length `n_decision_days - risk_lookback + 1` in the first two dims
        (see _make_windows), or empty tensors if there aren't enough
        decision days for even one window.
        """
        if X.shape[0] == 0:
            return X.new_zeros((0, risk_lookback, n_input_channels)), X.new_zeros((0, n_assets)), X.new_zeros((0, n_assets))
        with torch.no_grad():
            mu, sigma = base_model(X)
            positions = compute_target_vol_positions_torch(
                mu[:, -1, :], sigma[:, -1, :], rp_weights, cov, direction_horizon,
                target_vol, signal_min_t, signal_max_t, neutral_band,
            )
        t = len(dates)
        weight_np = positions.detach().cpu().numpy().reshape(t, n_assets, 1)
        stats_np = risk_stats.reindex(dates).to_numpy(dtype=np.float32).reshape(t, n_assets, per_asset_channels)
        combined = np.concatenate([weight_np, stats_np], axis=2).reshape(t, -1)
        windows = _make_windows(combined, risk_lookback)
        if windows.shape[0] == 0:
            return X.new_zeros((0, risk_lookback, n_input_channels)), X.new_zeros((0, n_assets)), X.new_zeros((0, n_assets))
        windows_t = torch.as_tensor(windows, dtype=torch.float32, device=device)
        positions_dec = positions[risk_lookback - 1:]
        next_returns_t = torch.as_tensor(next_returns_np[risk_lookback - 1:], dtype=torch.float32, device=device)
        return windows_t, positions_dec, next_returns_t

    X_train_w, pos_train, ret_train = _precompute_split(
        data.X_train, data.next_returns_train, data.rp_weights_train, data.cov_train, data.dates_train,
    )
    X_val_w, pos_val, ret_val = _precompute_split(
        data.X_val, data.next_returns_val, data.rp_weights_val, data.cov_val, data.dates_val,
    )

    risk_engine = RiskEngine(
        n_assets=len(data.pairs),
        risk_channels_per_asset=n_risk_channels_per_asset(risk_cma_windows, risk_bandpass_windows),
        hidden_size=getattr(args, "risk_hidden_size", None) or DEFAULT_RISK_HIDDEN_SIZE,
        num_layers=getattr(args, "risk_num_layers", None) or DEFAULT_RISK_NUM_LAYERS,
        dropout=getattr(args, "risk_dropout", None) if getattr(args, "risk_dropout", None) is not None else DEFAULT_RISK_DROPOUT,
        n_attn_heads=getattr(args, "risk_n_attn_heads", None) or DEFAULT_RISK_N_ATTN_HEADS,
    ).to(device)

    optimizer = torch.optim.Adam(risk_engine.parameters(), lr=lr, weight_decay=weight_decay)
    track_val = X_val_w.shape[0] > 0
    best_val_sortino = None
    best_state = None
    train_sortino_final, val_sortino_final = None, None

    risk_engine.train()
    for epoch in range(1, epochs + 1):
        if stop_check is not None and stop_check():
            raise TrainingStopped()
        optimizer.zero_grad()
        raw_train = risk_engine(X_train_w)[:, -1, :]
        att_train = attenuation_from_raw(raw_train, min_risk_att, max_risk_att)
        loss, train_sortino = _risk_loss_from_attenuation(
            att_train, pos_train, ret_train, sortino_window, cost_bps, max_risk_att, full_exposure_penalty,
        )
        loss.backward()
        _assert_finite_grad(risk_engine.parameters(), f"risk engine epoch {epoch}")
        optimizer.step()
        train_sortino_final = train_sortino

        val_sortino = None
        if track_val:
            risk_engine.eval()
            with torch.no_grad():
                raw_val = risk_engine(X_val_w)[:, -1, :]
                att_val = attenuation_from_raw(raw_val, min_risk_att, max_risk_att)
                _, val_sortino = _risk_loss_from_attenuation(
                    att_val, pos_val, ret_val, sortino_window, cost_bps, max_risk_att, 0.0,
                )
            risk_engine.train()
            val_sortino_final = val_sortino
            if val_sortino is not None and (best_val_sortino is None or val_sortino > best_val_sortino):
                best_val_sortino = val_sortino
                best_state = {k: v.detach().clone() for k, v in risk_engine.state_dict().items()}

        if epoch == 1 or epoch % max(epochs // 10, 1) == 0:
            logger.info(
                "risk engine epoch %d/%d - train sortino %s | val sortino %s | best val sortino %s",
                epoch, epochs,
                f"{train_sortino:.4f}" if train_sortino is not None else "n/a",
                f"{val_sortino:.4f}" if val_sortino is not None else "n/a",
                f"{best_val_sortino:.4f}" if best_val_sortino is not None else "n/a",
            )
        if on_epoch is not None:
            on_epoch(epoch, epochs, train_sortino, val_sortino, best_val_sortino)

    if best_state is not None:
        risk_engine.load_state_dict(best_state)
        logger.info("Restored risk engine's own best-validation-Sortino checkpoint (%.4f)", best_val_sortino)

    risk_engine.eval()
    risk_engine.pairs = data.pairs
    risk_engine.risk_lookback = risk_lookback
    risk_engine.risk_cma_windows = risk_cma_windows
    risk_engine.risk_bandpass_windows = risk_bandpass_windows
    risk_engine.risk_bandpass_order = risk_bandpass_order
    risk_engine.rolling_stats_window = risk_rolling_stats_window
    risk_engine.min_risk_att = min_risk_att
    risk_engine.max_risk_att = max_risk_att
    return risk_engine, {"train_sortino": train_sortino_final, "val_sortino": val_sortino_final}


def evaluate_risk_engine(
    risk_engine: RiskEngine, positions_modulated: np.ndarray, returns: pd.DataFrame, dates: pd.DatetimeIndex,
) -> np.ndarray:
    """Dense, eval-time (numpy) attenuation series - (T, n_assets), same T
    as `positions_modulated`/`dates` - for a caller that already has the
    FULL, already-computed `positions_modulated` series (e.g.
    api/server.py's evaluate()/summarize-style callers, which already run
    the frozen PredictionModel over the whole requested window). Days
    before `risk_engine.risk_lookback - 1` (not enough trailing decision
    days for even one window) default to `max_risk_att` - i.e. NO
    attenuation, full exposure - rather than leaving them undefined: the
    prediction stage already has a valid position by then (see this
    module's own docstring), so there's no reason to blank out the whole
    position just because the RISK overlay's own, separate warm-up hasn't
    finished yet.

    `dates` MAY include one date past `returns`'s own last realized row
    (e.g. api/server.py's evaluate() appending a placeholder "today" date,
    the same trick models/portfolio_pnl.py's latest_position uses for
    `next_returns`) - the `.ffill()` after `.reindex(dates)` below carries
    that day's skew/kurt/CMA/bandpass forward from the last REAL row
    rather than reindex leaving it NaN, i.e. "today's" own stats are
    treated as unchanged from the most recent actual close, the same
    "nothing FORWARD-looking, just not literally updated yet" spirit as
    every other trailing feature in this codebase.
    """
    pairs = risk_engine.pairs
    per_asset_channels = risk_engine.risk_channels_per_asset - 1
    risk_stats = build_risk_stats_dataframe(
        returns, pairs, getattr(risk_engine, "rolling_stats_window", 20), risk_engine.risk_cma_windows,
        risk_engine.risk_bandpass_windows, risk_engine.risk_bandpass_order,
    ).reindex(dates).ffill().to_numpy(dtype=np.float32)
    t = len(dates)
    stats_reshaped = risk_stats.reshape(t, len(pairs), per_asset_channels)
    weight_reshaped = positions_modulated.reshape(t, len(pairs), 1)
    combined = np.concatenate([weight_reshaped, stats_reshaped], axis=2).reshape(t, -1)

    attenuation = np.full((t, len(pairs)), risk_engine.max_risk_att, dtype=np.float32)
    windows = _make_windows(combined, risk_engine.risk_lookback)
    if windows.shape[0] == 0:
        return attenuation
    device = next(risk_engine.parameters()).device
    with torch.no_grad():
        raw = risk_engine(torch.as_tensor(windows, dtype=torch.float32, device=device))[:, -1, :]
        att = attenuation_from_raw(raw, risk_engine.min_risk_att, risk_engine.max_risk_att).cpu().numpy()
    attenuation[risk_engine.risk_lookback - 1:] = att
    return attenuation
