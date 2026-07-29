"""Postprocessing for the LSTM portfolio allocator: plot cumulative PnL and
print the Sharpe ratio, for both the in-sample (train) and out-of-sample
(validation) periods.

Kept in its own file so training/data logic (models/portfolio_lstm.py)
stays separate from presentation (matplotlib, printouts). This module is a
LIBRARY - it has no CLI of its own; main.py at the repo root calls these
functions after running models/portfolio_lstm.py's or models/risk_lstm.py's
pipeline, so the plot and Sharpe ratio always reflect the exact model that
was just trained/loaded.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.portfolio_lstm import PortfolioResult, sharpe_ratio

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


