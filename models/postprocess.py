"""Postprocessing for the LSTM FX forecaster: plot forecast vs actual and
print the hit rate, for both the in-sample (train) and out-of-sample
(validation) periods.

Kept in its own file so training/data logic (models/lstm_forecaster.py)
stays separate from presentation (matplotlib, printouts). This module
reuses `run_pipeline()` from lstm_forecaster.py, so the plot and the hit
rate always reflect the exact same train/validation split and
standardization the model was trained with - nothing is recomputed twice.

Produces two separate figures - one for the in-sample (train) period and
one for the out-of-sample (validation) period - rather than one combined
plot, so each period's fit can be inspected on its own y-axis scale.

Usage
-----
    python -m models.postprocess \
        --pairs EURUSD GBPUSD USDJPY --target EURUSD \
        --lookback 30 --horizon 5 --epochs 100 \
        --output models/forecast_plot.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

from models.lstm_forecaster import PipelineResult, build_arg_parser, hit_rate, run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def print_hit_rates(result: PipelineResult) -> None:
    """Print in-sample and out-of-sample hit rate (sign-match accuracy)."""
    train_hr = hit_rate(result.y_train_pred, result.y_train_actual)
    val_hr = hit_rate(result.y_val_pred, result.y_val_actual)
    print(f"In-sample (train) hit rate:      {train_hr:.2%}")
    print(f"Out-of-sample (validation) hit rate: {val_hr:.2%}")


def _suffixed_path(path: str, suffix: str) -> str:
    """Insert `suffix` before the file extension, e.g.
    ("forecast.png", "insample") -> "forecast_insample.png"."""
    p = Path(path)
    return str(p.with_name(f"{p.stem}_{suffix}{p.suffix}"))


def _plot_single(dates, actual, forecast, title: str, output_path: str) -> None:
    """Draw one actual-vs-forecast line chart and save it to `output_path`."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, actual, label="actual", color="black", linewidth=1)
    ax.plot(dates, forecast, label="forecast", color="tab:orange", linewidth=1, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("log return")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    logger.info("Saved plot to %s", output_path)


def plot_forecast(result: PipelineResult, output_path: str) -> None:
    """Plot actual vs forecasted 1-day-ahead log returns for the target
    symbol as two SEPARATE figures - one for the in-sample (train) period,
    one for the out-of-sample (validation) period - each saved to its own
    file derived from `output_path`.

    Only the first horizon step (h=1, "next day's return") is plotted per
    window: each window predicts a `horizon`-long vector, and every window
    overlaps with the next one, so plotting every step of every window
    would draw many overlapping lines for the same date. The h=1 series
    instead gives exactly one point per date - a clean, readable series.
    """
    train_hr = hit_rate(result.y_train_pred, result.y_train_actual)
    val_hr = hit_rate(result.y_val_pred, result.y_val_actual)

    _plot_single(
        dates=result.dates_train,
        actual=result.y_train_actual[:, 0],
        forecast=result.y_train_pred[:, 0],
        title=f"{result.target}: in-sample (train) next-day log return - hit rate {train_hr:.2%}",
        output_path=_suffixed_path(output_path, "insample"),
    )
    _plot_single(
        dates=result.dates_val,
        actual=result.y_val_actual[:, 0],
        forecast=result.y_val_pred[:, 0],
        title=f"{result.target}: out-of-sample (validation) next-day log return - hit rate {val_hr:.2%}",
        output_path=_suffixed_path(output_path, "outsample"),
    )


def main() -> None:
    parser = build_arg_parser("Plot LSTM forecast vs actual FX log returns and print the hit rate.")
    parser.add_argument("--output", default="models/forecast_plot.png", help="Path to save the plot image")
    args = parser.parse_args()

    result = run_pipeline(args)

    print_hit_rates(result)
    plot_forecast(result, args.output)


if __name__ == "__main__":
    main()
