"""LSTM forecaster for FX log returns.

Data flow
---------
1. Load daily close prices for a set of FX pairs through ``data/db.py``
   (``get_time_series``), which reads from the project's Postgres store.
   If Postgres doesn't have a symbol's history yet, it is downloaded once
   via the existing ``FXDownloader`` and upserted (``upsert_pairs``) so
   every subsequent read still goes through ``db.py``.
2. Convert prices to log returns: r_t = log(close_t / close_t-1).
   Log returns (rather than raw prices) are used as both model input and
   target because they are roughly stationary, which is what an LSTM needs
   to find a stable pattern instead of just memorising a price level.
3. Slide a fixed-size window over the returns to build a supervised
   dataset:
       X[i] = log returns of ALL input pairs over `lookback` days
              -> shape (lookback, n_pairs)
       y[i] = log returns of the TARGET pair over the following
              `horizon` days -> shape (horizon,)
4. Split by time into a TRAIN set (earliest rows) and a VALIDATION set
   (most recent rows) — never shuffle a time series, that would leak
   future information into training. The validation set is what lets us
   measure genuine out-of-sample error and hit rate.
5. Standardize X and y using only the training split's mean/std, then feed
   them to a small LSTM that predicts all `horizon` steps at once.

Usage
-----
    python -m models.lstm_forecaster \
        --pairs EURUSD GBPUSD USDJPY \
        --target EURUSD \
        --lookback 30 --horizon 5 --epochs 100
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data.db import get_time_series, upsert_pairs
from data.fx_downloader import FXDownloader, MAJOR_FX_PAIRS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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


def make_sequences(
    returns: pd.DataFrame, target: str, lookback: int, horizon: int
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Slide a window over `returns` to build (X, y) supervised pairs.

    X[i]: `lookback` days of log returns for every input pair (multivariate).
    y[i]: the following `horizon` days of log returns for `target` only.
    dates[i]: the date of y[i]'s first step - used later to align forecasts
              with actual dates on a plot.
    """
    values = returns.to_numpy(dtype=np.float32)  # shape (T, n_pairs)
    target_col = returns.columns.get_loc(target)
    index = returns.index

    X, y, dates = [], [], []
    last_start = len(values) - lookback - horizon + 1
    for start in range(max(last_start, 0)):
        window_end = start + lookback
        X.append(values[start:window_end])
        y.append(values[window_end:window_end + horizon, target_col])
        dates.append(index[window_end])

    if not X:
        raise ValueError(
            f"Not enough history ({len(values)} rows) for lookback={lookback} "
            f"+ horizon={horizon}."
        )
    return np.stack(X), np.stack(y), pd.DatetimeIndex(dates)


def standardize(values: np.ndarray, axis) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std computed over `axis`, with a small epsilon to avoid /0.

    Public (no leading underscore) so models/portfolio_lstm.py can reuse it
    without duplicating the same two lines.
    """
    return values.mean(axis=axis), values.std(axis=axis) + 1e-8


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class LSTMForecaster(nn.Module):
    """Maps a window of multi-pair log returns to an N-step-ahead forecast
    of a single target pair's log returns.

    This predicts all `horizon` steps directly from the LSTM's final hidden
    state (a "direct" multi-step forecast), which is simpler and avoids
    compounding errors compared to feeding predictions back in recursively.
    """

    def __init__(self, n_features: int, horizon: int, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # inputs shaped (batch, time, features)
        )
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, n_features)
        _, (h_n, _) = self.lstm(x)
        last_layer_hidden = h_n[-1]        # (batch, hidden_size): final layer's end-of-sequence state
        return self.head(last_layer_hidden)  # (batch, horizon)

    def save_model(self, path: str = "models/lstm_forecaster.pt") -> None:
        """Persist trained weights so they can be reloaded without retraining."""
        torch.save(self.state_dict(), path)
        logger.info("Saved model weights to %s", path)


# --------------------------------------------------------------------------
# Hit rate: directional accuracy metric
# --------------------------------------------------------------------------

def hit_rate(forecast: np.ndarray, actual: np.ndarray) -> float:
    """Fraction of forecasts whose sign matches the actual return's sign.

    A "hit" means the model correctly called the direction of the move
    (up vs down), regardless of how close the magnitude was — the metric
    traders usually care about most for a return forecast.

    Arrays are flattened before comparing, so this works for a single
    horizon vector (e.g. `hit_rate(pred[0], actual[0])`) or a whole batch
    of them (`hit_rate(pred, actual)`, averaging over every step of every
    sample).
    """
    forecast = np.asarray(forecast).ravel()
    actual = np.asarray(actual).ravel()
    if forecast.shape != actual.shape:
        raise ValueError(f"Shape mismatch: forecast{forecast.shape} vs actual{actual.shape}")
    return float((np.sign(forecast) == np.sign(actual)).mean())


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    lr: float,
    batch_size: int,
) -> None:
    """Plain supervised training loop: MSE loss between predicted and
    actual (standardized) future log returns, optimized with Adam."""
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        if epoch == 1 or epoch % max(epochs // 10, 1) == 0:
            logger.info("epoch %d/%d - train MSE %.6f", epoch, epochs, epoch_loss / len(X_train))


# --------------------------------------------------------------------------
# End-to-end pipeline: shared by this script's CLI and models/postprocess.py
# --------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Everything needed to score and plot a trained forecaster.

    All y_* arrays are on the REAL log-return scale (already
    un-standardized), shaped (n_windows, horizon).
    """

    model: LSTMForecaster
    target: str
    horizon: int
    dates_train: pd.DatetimeIndex  # date of each training window's 1st forecasted day
    dates_val: pd.DatetimeIndex    # same, for the held-out validation windows
    y_train_actual: np.ndarray
    y_train_pred: np.ndarray       # in-sample forecasts
    y_val_actual: np.ndarray
    y_val_pred: np.ndarray         # out-of-sample forecasts
    future_returns: np.ndarray     # (horizon,) forecast beyond the last known date


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """CLI arguments shared by this script and models/postprocess.py."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--pairs", nargs="+", required=True, help="Input FX pairs, e.g. EURUSD GBPUSD USDJPY")
    parser.add_argument("--target", required=True, help="Symbol to forecast, e.g. EURUSD")
    parser.add_argument("--lookback", type=int, default=30, help="Days of history fed to the LSTM")
    parser.add_argument("--horizon", type=int, default=5, help="N future days to forecast")
    parser.add_argument("--years", type=int, default=20, help="Years of history to load")
    parser.add_argument(
        "--train-frac", type=float, default=0.8,
        help="Fraction of sequences used for training; the rest is the out-of-sample validation set",
    )
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser


def run_pipeline(args: argparse.Namespace) -> PipelineResult:
    """Load data (via db.py), build sequences, train the LSTM on the train
    split, and score it on both the train (in-sample) and validation
    (out-of-sample) splits.
    """
    # The target's own past returns are a useful feature for forecasting
    # itself, so make sure it's always part of the input set.
    symbols = list(dict.fromkeys([*args.pairs, args.target]))  # de-duplicate, keep order

    logger.info("Loading %s (target=%s) via db.py", symbols, args.target)
    prices = load_close_prices(symbols, years=args.years)
    returns = to_log_returns(prices)

    X, y, dates = make_sequences(returns, target=args.target, lookback=args.lookback, horizon=args.horizon)

    # Time-based split: train on the earlier segment, validate on the later
    # (most recent) one - this is the out-of-sample set.
    n_train = int(len(X) * args.train_frac)
    X_train_raw, X_val_raw = X[:n_train], X[n_train:]
    y_train_raw, y_val_raw = y[:n_train], y[n_train:]
    dates_train, dates_val = dates[:n_train], dates[n_train:]

    # Standardize using TRAIN-only statistics to avoid leaking future data.
    x_mean, x_std = standardize(X_train_raw, axis=(0, 1))
    y_mean, y_std = standardize(y_train_raw, axis=None)

    X_train = torch.tensor((X_train_raw - x_mean) / x_std)
    X_val = torch.tensor((X_val_raw - x_mean) / x_std)
    y_train = torch.tensor((y_train_raw - y_mean) / y_std)

    model = LSTMForecaster(n_features=len(symbols), horizon=args.horizon, hidden_size=args.hidden_size)
    train_model(model, X_train, y_train, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)

    model.eval()
    with torch.no_grad():
        # In-sample: how well the model fits data it was trained on.
        y_train_pred = model(X_train).numpy() * y_std + y_mean
        # Out-of-sample: the real test - data the model never saw in training.
        y_val_pred = model(X_val).numpy() * y_std + y_mean

        # One more step: forecast `horizon` days beyond the last known date.
        last_window = returns.to_numpy(dtype=np.float32)[-args.lookback:]
        last_window_std = torch.tensor((last_window - x_mean) / x_std).unsqueeze(0)  # add batch dim
        future_returns = model(last_window_std).numpy()[0] * y_std + y_mean

    return PipelineResult(
        model=model,
        target=args.target,
        horizon=args.horizon,
        dates_train=dates_train,
        dates_val=dates_val,
        y_train_actual=y_train_raw,
        y_train_pred=y_train_pred,
        y_val_actual=y_val_raw,
        y_val_pred=y_val_pred,
        future_returns=future_returns,
    )


def main() -> None:
    parser = build_arg_parser("Train an LSTM to forecast FX log returns.")
    args = parser.parse_args()
    result = run_pipeline(args)
    result.model.save_model()
    train_mse = float(np.mean((result.y_train_pred - result.y_train_actual) ** 2))
    val_mse = float(np.mean((result.y_val_pred - result.y_val_actual) ** 2))
    train_hr = hit_rate(result.y_train_pred, result.y_train_actual)
    val_hr = hit_rate(result.y_val_pred, result.y_val_actual)

    logger.info("In-sample  (train) MSE %.6f | hit rate %.2f%%", train_mse, train_hr * 100)
    logger.info("Out-of-sample (val) MSE %.6f | hit rate %.2f%%", val_mse, val_hr * 100)
    logger.info(
        "Forecast next %d log returns for %s: %s",
        result.horizon, result.target, np.round(result.future_returns, 6),
    )


if __name__ == "__main__":
    main()
