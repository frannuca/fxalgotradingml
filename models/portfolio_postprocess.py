"""Postprocessing for the LSTM portfolio allocator: plot cumulative PnL and
print the Sharpe ratio, for both the in-sample (train) and out-of-sample
(validation) periods.

Kept in its own file so training/data logic (models/portfolio_lstm.py)
stays separate from presentation (matplotlib, printouts), mirroring
models/postprocess.py for the return-forecasting model. Reuses
`run_pipeline_multi_seed()` from portfolio_lstm.py, so the plot and Sharpe
ratio always reflect the exact same train/validation split (and, if
`--n-seeds` > 1, the same restart-combination) the model was trained with.

`--risk-overlay` is also honored here (not just in models/risk_lstm.py /
models/risk_postprocess.py): passing it trains RiskLSTM on top and
produces the full 4-plot risk output instead of the plain 2-plot one -
this script and models/risk_postprocess.py both end up calling the exact
same underlying functions either way, so which one you run is just a
matter of preference.

Usage
-----
    python -m models.portfolio_postprocess \
        --pairs EURUSD GBPUSD USDJPY \
        --lookback 30 --weight-scheme softmax --epochs 300 \
        --output models/portfolio_pnl.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.portfolio_lstm import PortfolioResult, build_arg_parser, run_pipeline_multi_seed, sharpe_ratio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def print_sharpe_ratios(result: PortfolioResult) -> None:
    """Print in-sample and out-of-sample Sharpe ratio and cumulative PnL."""
    train_sharpe = float(sharpe_ratio(torch.tensor(result.returns_train)))
    val_sharpe = float(sharpe_ratio(torch.tensor(result.returns_val)))
    print(f"In-sample (train) Sharpe ratio:      {train_sharpe:.3f} | cumulative PnL {result.returns_train.sum():.4f}")
    print(f"Out-of-sample (validation) Sharpe ratio: {val_sharpe:.3f} | cumulative PnL {result.returns_val.sum():.4f}")


def _suffixed_path(path: str, suffix: str) -> str:
    """Insert `suffix` before the file extension, e.g.
    ("pnl.png", "insample") -> "pnl_insample.png"."""
    p = Path(path)
    return str(p.with_name(f"{p.stem}_{suffix}{p.suffix}"))


def _plot_single(dates, cumulative_pnl, title: str, output_path: str) -> None:
    """Draw one cumulative-PnL line chart and save it to `output_path`."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, cumulative_pnl, color="black", linewidth=1)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("cumulative log-return P&L")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    logger.info("Saved plot to %s", output_path)


def plot_pnl(result: PortfolioResult, output_path: str) -> None:
    """Plot cumulative PnL (cumsum of realized portfolio log returns) as two
    SEPARATE figures - one for the in-sample (train) period, one for the
    out-of-sample (validation) period - each saved to its own file derived
    from `output_path`.
    """
    train_sharpe = float(sharpe_ratio(torch.tensor(result.returns_train)))
    val_sharpe = float(sharpe_ratio(torch.tensor(result.returns_val)))

    _plot_single(
        dates=result.dates_train,
        cumulative_pnl=np.cumsum(result.returns_train),
        title=f"In-sample (train) cumulative PnL - Sharpe {train_sharpe:.2f}",
        output_path=_suffixed_path(output_path, "insample"),
    )
    _plot_single(
        dates=result.dates_val,
        cumulative_pnl=np.cumsum(result.returns_val),
        title=f"Out-of-sample (validation) cumulative PnL - Sharpe {val_sharpe:.2f}",
        output_path=_suffixed_path(output_path, "outsample"),
    )


def main() -> None:
    parser = build_arg_parser("Plot LSTM portfolio cumulative PnL and print the Sharpe ratio.")
    parser.add_argument("--output", default="models/portfolio_pnl.png", help="Path to save the plot images")
    parser.add_argument(
        "--position-output", default="models/risk_position.png",
        help="Only used with --risk-overlay: path to save the position-vs-scaling plot images",
    )
    args = parser.parse_args()

    result = run_pipeline_multi_seed(args)
    if not args.load_portfolio:
        result.model.save_model(x_mean=result.x_mean, x_std=result.x_std)

    if not args.risk_overlay:
        print_sharpe_ratios(result)
        plot_pnl(result, args.output)
        return

    # --risk-overlay: hand off to the exact same functions
    # models/risk_postprocess.py uses, so both entry points produce
    # identical output - a single raw-vs-attenuated PnL plot (not the
    # plain one above, which --output would otherwise collide with) plus
    # the position-vs-scaling plot, for both splits.
    from models.risk_lstm import add_risk_overlay
    from models.risk_postprocess import (
        plot_pnl as plot_risk_pnl,
        plot_position_and_scaling,
        print_sharpe_ratios as print_risk_sharpe_ratios,
    )

    risk_result = add_risk_overlay(result, args)
    if not args.load_risk:
        risk_result.risk_model.save_model()
    print_risk_sharpe_ratios(risk_result)
    plot_risk_pnl(risk_result, args.output)
    plot_position_and_scaling(risk_result, args.position_output)


if __name__ == "__main__":
    main()
