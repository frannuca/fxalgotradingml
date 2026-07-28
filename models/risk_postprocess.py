"""Postprocessing for the risk-attenuation LSTM: plot raw vs attenuated
cumulative PnL and print the Sharpe ratio of both, for the in-sample
(train) and out-of-sample (validation) periods.

Kept in its own file, mirroring models/portfolio_postprocess.py. Reuses
`run_pipeline()` from risk_lstm.py, so the plot and metrics always reflect
the exact same trained PortfolioLSTM + RiskLSTM pair.

Usage
-----
    python -m models.risk_postprocess \
        --pairs EURUSD GBPUSD USDJPY \
        --lookback 30 --weight-scheme softmax --epochs 300 \
        --risk-hidden-size 16 --risk-epochs 200 \
        --output models/risk_pnl.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.portfolio_lstm import sharpe_ratio
from models.risk_lstm import RiskResult, build_arg_parser, run_pipeline

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
    pairs = result.portfolio_result.pairs
    _plot_position_and_scaling(
        dates=result.dates_train,
        weights=result.portfolio_result.weights_train,
        attenuation=result.attenuation_train,
        pairs=pairs,
        title="In-sample (train): position vs. risk-overlay scaling",
        output_path=_suffixed_path(output_path, "insample"),
    )
    _plot_position_and_scaling(
        dates=result.dates_val,
        weights=result.portfolio_result.weights_val,
        attenuation=result.attenuation_val,
        pairs=pairs,
        title="Out-of-sample (validation): position vs. risk-overlay scaling",
        output_path=_suffixed_path(output_path, "outsample"),
    )


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


def main() -> None:
    parser = build_arg_parser("Plot raw vs risk-attenuated portfolio PnL and print the Sharpe ratio.")
    parser.add_argument("--output", default="models/risk_pnl.png", help="Path to save the PnL plot images")
    parser.add_argument(
        "--position-output", default="models/risk_position.png",
        help="Path to save the position-vs-scaling plot images",
    )
    args = parser.parse_args()

    result = run_pipeline(args)

    print_sharpe_ratios(result)
    plot_pnl(result, args.output)
    plot_position_and_scaling(result, args.position_output)


if __name__ == "__main__":
    main()
