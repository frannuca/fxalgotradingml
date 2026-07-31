"""Postprocessing for the risk-attenuation LSTM: plot raw vs attenuated
cumulative PnL and print the Sharpe ratio of both, for the in-sample
(train) and out-of-sample (validation) periods.

Kept in its own file, mirroring models/portfolio_postprocess.py. This
module is a LIBRARY - it has no CLI of its own; main.py at the repo root
calls these functions after running models/risk_lstm.py's
run_pipeline_multi_seed() (which trains/loads PortfolioLSTM completely on
its own first, then trains/loads RiskLSTM SEPARATELY on top of it - see
that module's docstring), so the plot and metrics always reflect the exact
same paired portfolio/risk models.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.portfolio_lstm import apply_transaction_costs, sharpe_ratio
from models.risk_lstm import RiskResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def print_sharpe_ratios(result: RiskResult) -> None:
    """Print raw vs attenuated Sharpe ratio and mean attenuation, for both splits."""
    raw_train = float(sharpe_ratio(torch.tensor(result.returns_train_raw)))
    scaled_train = float(sharpe_ratio(torch.tensor(result.returns_train_scaled)))
    raw_val = float(sharpe_ratio(torch.tensor(result.returns_val_raw)))
    scaled_val = float(sharpe_ratio(torch.tensor(result.returns_val_scaled)))

    print(f"In-sample  (train): raw Sharpe {raw_train:.3f} -> attenuated Sharpe {scaled_train:.3f} "
          f"| mean attenuation {result.attenuation_train.mean():.3f}")
    print(f"Out-of-sample (val): raw Sharpe {raw_val:.3f} -> attenuated Sharpe {scaled_val:.3f} "
          f"| mean attenuation {result.attenuation_val.mean():.3f}")


def _suffixed_path(path: str, suffix: str) -> str:
    """Insert `suffix` before the file extension, e.g.
    ("pnl.png", "insample") -> "pnl_insample.png"."""
    p = Path(path)
    return str(p.with_name(f"{p.stem}_{suffix}{p.suffix}"))


def _plot_single(dates, raw_pnl, scaled_pnl, title: str, output_path: str) -> None:
    """Draw one raw-vs-attenuated cumulative PnL chart and save it."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, raw_pnl, label="raw (no attenuation)", color="tab:gray", linewidth=1)
    ax.plot(dates, scaled_pnl, label="attenuated", color="black", linewidth=1)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("cumulative log-return P&L")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    logger.info("Saved plot to %s", output_path)


def _plot_position_and_scaling(
    dates, weights: np.ndarray, attenuation: np.ndarray, pairs: list[str], title: str, output_path: str
) -> None:
    """Draw one chart overlaying each pair's raw position (portfolio weight,
    solid line) with that SAME pair's risk-overlay attenuation (dashed
    line, matching color), on two separate y-axes.

    Attenuation is per-asset now (RiskLSTM outputs one factor per pair, not
    a single global scaling for the whole book), so each pair gets two
    lines linked by color rather than one shared scaling line - lets you
    see e.g. one pair being de-risked while another stays near full size.
    These are two different quantities on two different scales - a weight
    vs. an attenuation factor in [max_attenuation, 1] - so they're drawn on
    separate y-axes sharing the same time axis.
    """
    fig, ax_position = plt.subplots(figsize=(12, 5))
    ax_scaling = ax_position.twinx()

    # Fixed color order for the per-pair lines, consistent across the
    # train/validation figures; each pair's position and attenuation share
    # a color so they're visually linked.
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, pair in enumerate(pairs):
        color = colors[i % len(colors)]
        ax_position.plot(dates, weights[:, i], label=f"{pair} position", color=color, linewidth=1)
        ax_scaling.plot(
            dates, attenuation[:, i], label=f"{pair} attenuation",
            color=color, linewidth=1, linestyle="--", alpha=0.8,
        )

    ax_position.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax_position.set_xlabel("date")
    ax_position.set_ylabel("position (portfolio weight)")
    ax_scaling.set_ylabel("attenuation (per-asset risk overlay scaling)")
    ax_scaling.set_ylim(0, 1)

    # One combined legend for both axes (2 columns - position + attenuation
    # entries per pair adds up fast).
    position_handles, position_labels = ax_position.get_legend_handles_labels()
    scaling_handles, scaling_labels = ax_scaling.get_legend_handles_labels()
    ax_position.legend(
        position_handles + scaling_handles, position_labels + scaling_labels,
        loc="upper left", ncol=2, fontsize=8,
    )

    ax_position.set_title(title)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    logger.info("Saved plot to %s", output_path)


def plot_position_and_scaling(result: RiskResult, output_path: str) -> None:
    """Plot position (raw portfolio weights) vs. risk-overlay scaling as two
    SEPARATE figures - one for the in-sample (train) period, one for the
    out-of-sample (validation) period - each saved to its own file derived
    from `output_path`.
    """
    # result.weights_train/weights_val (NOT portfolio_result's own) are
    # ALIGNED to result.dates_train/dates_val and result.attenuation_train/
    # attenuation_val - make_risk_sequences() drops the first
    # `rolling_window - 1` samples of each split, so portfolio_result's own
    # (longer) arrays would silently misalign here.
    pairs = result.portfolio_result.pairs
    _plot_position_and_scaling(
        dates=result.dates_train,
        weights=result.weights_train,
        attenuation=result.attenuation_train,
        pairs=pairs,
        title="In-sample (train): position vs. risk-overlay scaling",
        output_path=_suffixed_path(output_path, "insample"),
    )
    _plot_position_and_scaling(
        dates=result.dates_val,
        weights=result.weights_val,
        attenuation=result.attenuation_val,
        pairs=pairs,
        title="Out-of-sample (validation): position vs. risk-overlay scaling",
        output_path=_suffixed_path(output_path, "outsample"),
    )


def annualized_vol(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Realized annualized volatility of a daily log-return series."""
    return float(np.std(returns) * np.sqrt(periods_per_year))


def scale_to_target_vol(returns: np.ndarray, target_vol: float, periods_per_year: int = 252) -> np.ndarray:
    """Rescale `returns` by a single constant factor so its REALIZED
    annualized volatility over the given period equals `target_vol`.

    This is a plotting-time normalization, distinct from
    portfolio_lstm.py's scale_weights_to_target_vol (which rescales
    weights day-by-day, using each day's own trailing-window covariance
    estimate, before the fact). Here the series is already fixed - we're
    just rescaling it after the fact, using its own REALIZED volatility
    over the whole period, purely so several return series that naturally
    run at different risk levels can be compared on the same chart: any
    remaining difference in cumulative PnL then reflects differences in
    SKILL/SHAPE (drawdowns, smoothness) rather than just how much risk
    each one happened to take.
    """
    vol = annualized_vol(returns, periods_per_year)
    if vol < 1e-12:
        return returns  # degenerate all-zero series - nothing to scale
    return returns * (target_vol / vol)


def plot_vol_matched_pnl(result: RiskResult, output_path: str, target_vol: float = 0.20) -> None:
    """Plot OUT-OF-SAMPLE ONLY cumulative PnL for three strategies, each
    independently rescaled (by its own realized volatility) to the SAME
    annualized volatility target:
      - "raw": PortfolioLSTM's weights before --target-vol scaling.
      - "risk-weighted": after --target-vol scaling (the normal, un-
        attenuated pipeline output), before the risk overlay.
      - "attenuated": after the risk overlay's per-asset attenuation on
        top of the risk-weighted position.

    Comparing cumulative PnL directly can be misleading when the
    strategies run at very different risk levels - a naturally lower-vol
    strategy looks "worse" purely from being smaller. Vol-matching removes
    that confound.
    """
    # portfolio_result.returns_val_unscaled is portfolio_result's own
    # (longer) array - align it to result.dates_val/returns_val_raw's
    # range the same way make_risk_sequences() trimmed those (drops the
    # first `rolling_window - 1` samples of the split).
    trim = result.risk_model.rolling_window - 1
    raw = scale_to_target_vol(result.portfolio_result.returns_val_unscaled[trim:], target_vol)
    risk_weighted = scale_to_target_vol(result.returns_val_raw, target_vol)
    attenuated = scale_to_target_vol(result.returns_val_scaled, target_vol)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(result.dates_val, np.cumsum(raw), label="raw (pre vol-target)", color="tab:gray", linewidth=1)
    ax.plot(result.dates_val, np.cumsum(risk_weighted), label="risk-weighted (vol-targeted)", color="tab:blue", linewidth=1)
    ax.plot(result.dates_val, np.cumsum(attenuated), label="attenuated (risk overlay)", color="black", linewidth=1)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(f"Out-of-sample cumulative PnL, all matched to {target_vol:.0%} annualized volatility")
    ax.set_xlabel("date")
    ax.set_ylabel("cumulative log-return P&L (vol-matched)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    logger.info("Saved plot to %s", output_path)


def plot_pnl(result: RiskResult, output_path: str) -> None:
    """Plot raw vs attenuated cumulative PnL as two SEPARATE figures - one
    for the in-sample (train) period, one for the out-of-sample
    (validation) period - each saved to its own file derived from
    `output_path`.
    """
    raw_train = float(sharpe_ratio(torch.tensor(result.returns_train_raw)))
    scaled_train = float(sharpe_ratio(torch.tensor(result.returns_train_scaled)))
    raw_val = float(sharpe_ratio(torch.tensor(result.returns_val_raw)))
    scaled_val = float(sharpe_ratio(torch.tensor(result.returns_val_scaled)))

    _plot_single(
        dates=result.dates_train,
        raw_pnl=np.cumsum(result.returns_train_raw),
        scaled_pnl=np.cumsum(result.returns_train_scaled),
        title=f"In-sample (train) cumulative PnL - Sharpe raw {raw_train:.2f} / attenuated {scaled_train:.2f}",
        output_path=_suffixed_path(output_path, "insample"),
    )
    _plot_single(
        dates=result.dates_val,
        raw_pnl=np.cumsum(result.returns_val_raw),
        scaled_pnl=np.cumsum(result.returns_val_scaled),
        title=f"Out-of-sample (validation) cumulative PnL - Sharpe raw {raw_val:.2f} / attenuated {scaled_val:.2f}",
        output_path=_suffixed_path(output_path, "outsample"),
    )


def plot_return_histograms(result: RiskResult, output_path: str, bins: int = 40) -> None:
    """Plot OUT-OF-SAMPLE daily return histograms for the risk-weighted
    (pre-attenuation) and risk-attenuated (post-attenuation) return series,
    overlaid - lets you compare their distribution SHAPE (tails, spread),
    not just their summary Sharpe ratio.
    """
    raw = result.returns_val_raw
    scaled = result.returns_val_scaled

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(raw, bins=bins, alpha=0.5, label="risk-weighted (no attenuation)", color="tab:gray", density=True)
    ax.hist(scaled, bins=bins, alpha=0.5, label="risk-attenuated", color="black", density=True)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title("Out-of-sample daily return distribution")
    ax.set_xlabel("daily log return")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    logger.info("Saved plot to %s", output_path)


def plot_transaction_cost_pnl(result: RiskResult, output_path: str, transaction_cost_bps: float) -> None:
    """Plot OUT-OF-SAMPLE cumulative PnL for the risk-attenuated strategy,
    gross vs NET of an estimated transaction-cost drag (see
    apply_transaction_costs in models/portfolio_lstm.py) - the final,
    most "realistic" view of the strategy's net-of-costs performance.
    """
    # result.weights_val (not portfolio_result's own, longer array) is
    # aligned to result.attenuation_val/dates_val - see make_risk_sequences.
    final_weights_val = result.weights_val * result.attenuation_val  # already-attenuated weights
    net_returns = apply_transaction_costs(final_weights_val, result.returns_val_scaled, transaction_cost_bps)

    gross_sharpe = float(sharpe_ratio(torch.tensor(result.returns_val_scaled)))
    net_sharpe = float(sharpe_ratio(torch.tensor(net_returns)))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        result.dates_val, np.cumsum(result.returns_val_scaled),
        label=f"gross (no costs) - Sharpe {gross_sharpe:.2f}", color="tab:gray", linewidth=1,
    )
    ax.plot(
        result.dates_val, np.cumsum(net_returns),
        label=f"net ({transaction_cost_bps:.1f}bps/turnover) - Sharpe {net_sharpe:.2f}", color="black", linewidth=1,
    )
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(f"Out-of-sample cumulative PnL: risk-attenuated, gross vs net of {transaction_cost_bps:.1f}bps costs")
    ax.set_xlabel("date")
    ax.set_ylabel("cumulative log-return P&L")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    logger.info("Saved plot to %s", output_path)


