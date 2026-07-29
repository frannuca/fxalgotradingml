"""LSTM portfolio allocator: learns weights that directly maximize Sharpe ratio.

This module is a LIBRARY - it has no CLI of its own. The single entry
point for the whole project is main.py at the repo root, which reads a
JSON config file and orchestrates training/evaluation by calling the
functions here (and in models/risk_lstm.py) directly. See main.py's
module docstring for the JSON schema.

This model's output IS a trading decision: at each day it looks at the
last `lookback` days of log returns for every pair in the portfolio and
outputs a weight per pair for the following day. It is trained end-to-end
to maximize the Sharpe ratio of the resulting portfolio, not to predict
returns accurately - Sharpe ratio (mean return / return volatility) is
what a portfolio manager actually cares about, and it is differentiable,
so we can optimize it directly with gradient descent instead of using a
proxy loss like MSE.

Data flow
---------
1. Load daily close prices for the portfolio's FX pairs via data/db.py
   (see load_close_prices below).
2. Convert to log returns.
3. Slide a window over the returns:
       X[i]            = log returns of every pair over `lookback` days
       next_returns[i] = log returns of every pair on the single day right
                         after the window - i.e. what a weight decided at
                         the end of the window would actually earn.
4. Split by time into TRAIN (earliest) and VALIDATION (most recent, held
   out) - never shuffle a time series, that would leak future information
   into training.
5. The LSTM maps each X[i] window to a weight vector w[i] (one weight per
   pair). Two weight schemes are supported (--weight-scheme):
     - "softmax":   long-only. w = softmax(logits): every weight in (0, 1)
                    and the weights always sum to exactly 1.
     - "tanh_norm": long/short. w = tanh(logits) / sum(|tanh(logits)|):
                    every weight in (-1, 1), fully invested in absolute
                    terms (sum(|w|) == 1), but the signed sum can be
                    anything in [-1, 1] - allowing shorts is fundamentally
                    incompatible with also pinning the signed sum to 1.
6. The realized portfolio return at each step is dot(w[i], next_returns[i]).
   Training maximizes the Sharpe ratio of that whole return series over
   the TRAIN split in one full-batch step per epoch (Sharpe is a property
   of the entire return distribution, not of individual samples, so there
   is no per-sample loss to average over mini-batches).
7. The cumulative PnL of the strategy is just cumsum(portfolio returns),
   reported separately for the train (in-sample) and validation
   (out-of-sample) periods.

Regularization (this model overfits easily: full-batch training on a
noisy Sharpe objective for hundreds of epochs will happily memorize the
training period's noise). Three purely training-time regularizers are
applied, none of which touch the validation split - it stays a clean
blind holdout:
  - `noise_std`: Gaussian noise added to the (standardized) input
    window on every training epoch, regenerated fresh each time. This is
    the standard "add noise to the input" regularizer: the model can no
    longer fit the exact training sequence of returns, only patterns
    robust to small perturbations of it.
  - `dropout`: dropout on the LSTM's final hidden state before the
    linear head, so the head can't rely on any single hidden unit.
  - `weight_decay`: L2 penalty on the model weights (Adam's
    `weight_decay`), shrinking the model towards simpler solutions.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import torch
from torch import nn

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


def standardize(values: np.ndarray, axis) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std computed over `axis`, with a small epsilon to avoid /0."""
    return values.mean(axis=axis), values.std(axis=axis) + 1e-8


def make_portfolio_sequences(
    returns: pd.DataFrame, lookback: int
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Slide a window over `returns` to build (X, next_returns) pairs.

    X[i]: `lookback` days of log returns for every pair, i.e. everything
          known through day t (the window's last day). The weight the
          model computes from X[i] is the allocation decided AT day t.
    next_returns[i]: the log returns of every pair on day t+1 - the single
          day right after the window. This is what that day-t weight
          actually earns; day t+1's data never appears anywhere in X[i],
          so a weight can never be trained on the return it's applied to.
    dates[i]: the date of day t+1 (i.e. of next_returns[i]).
    """
    values = returns.to_numpy(dtype=np.float32)  # shape (T, n_pairs)
    index = returns.index

    X, next_returns, dates = [], [], []
    last_start = len(values) - lookback  # need exactly 1 day (t+1) after the window
    for start in range(max(last_start, 0)):
        day_t = start + lookback - 1     # last day inside the window - the decision day
        day_t_plus_1 = day_t + 1         # the day the decision's return is realized on

        X.append(values[start:day_t + 1])          # rows start..day_t inclusive - never includes day_t+1
        next_returns.append(values[day_t_plus_1])
        dates.append(index[day_t_plus_1])

    if not X:
        raise ValueError(f"Not enough history ({len(values)} rows) for lookback={lookback}.")

    X, next_returns = np.stack(X), np.stack(next_returns)
    # Every window must be exactly `lookback` long. The no-look-ahead
    # invariant itself (day_t_plus_1's row is never one of X[i]'s rows) is
    # already guaranteed structurally above by slicing disjoint index
    # ranges (values[start:day_t+1] vs values[day_t_plus_1]) - it does NOT
    # need a runtime value-equality check. Real FX data legitimately
    # contains repeated/flat return values (e.g. holidays or stale closes
    # producing a return of exactly 0.0 on more than one day within the
    # same window), so comparing by VALUE would raise false positives on
    # perfectly valid data.
    assert X.shape[1] == lookback
    return X, next_returns, pd.DatetimeIndex(dates)


# --------------------------------------------------------------------------
# Noisy layers: parameter-space noise (NoisyNet, Fortunato et al. 2017)
# --------------------------------------------------------------------------

class NoisyLinear(nn.Module):
    """Drop-in replacement for nn.Linear whose weight/bias are themselves
    noisy: `weight = weight_mu + weight_sigma * epsilon`, with epsilon
    resampled fresh from N(0, 1) every forward() call while training.

    This is the "noisy cell" analogue of `noise_std` (which perturbs the
    INPUT window instead): rather than the model seeing a slightly
    different input each epoch, it has to produce a good Sharpe ratio under
    a slightly different version of its OWN decision boundary each epoch.
    That makes it harder for the model to lock onto one razor-sharp weight
    configuration that happens to maximize Sharpe on this exact training
    sample but doesn't generalize - the two noise sources compose (both can
    be enabled together) rather than replace each other.

    In eval() mode, weight/bias fall back to their mu-only (noise-free)
    values, so inference is deterministic - mirroring how dropout also
    turns itself off outside training.

    `sigma_init` sets the initial noise magnitude relative to fan-in (0.5 is
    the standard NoisyNet default); sigma is itself a learned parameter, so
    the model can shrink it over training if noise stops helping.
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        # Buffers, not parameters: resampled every forward() call, never
        # trained via gradient descent themselves.
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self._sigma_init = sigma_init
        self._reset_parameters()
        self._reset_noise()

    def _reset_parameters(self) -> None:
        bound = 1.0 / (self.in_features ** 0.5)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, self._sigma_init * bound)
        nn.init.constant_(self.bias_sigma, self._sigma_init * bound)

    def _reset_noise(self) -> None:
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._reset_noise()
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class PortfolioLSTM(nn.Module):
    """Maps a window of multi-pair log returns to a portfolio weight vector
    (one weight per pair) for the following day.

    See the module docstring for the two supported `weight_scheme` values.
    """

    def __init__(
        self,
        n_assets: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        weight_scheme: str = "softmax",
        dropout: float = 0.0,
        noisy_head: bool = False,
    ):
        super().__init__()
        if weight_scheme not in ("softmax", "tanh_norm"):
            raise ValueError(f"Unknown weight_scheme: {weight_scheme!r}")
        # Stashed (not just passed to submodules) so save_model() can
        # persist enough to reconstruct this exact architecture on load,
        # without the caller having to re-supply matching CLI flags.
        self.n_assets = n_assets
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.weight_scheme = weight_scheme
        self.dropout_p = dropout
        self.noisy_head = noisy_head
        self.lstm = nn.LSTM(
            input_size=n_assets,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # inputs shaped (batch, time, features)
        )
        # Dropout on the final hidden state only - a no-op (p=0) when
        # `dropout` isn't set, and inactive automatically in model.eval().
        self.dropout = nn.Dropout(dropout)
        self.head = NoisyLinear(hidden_size, n_assets) if noisy_head else nn.Linear(hidden_size, n_assets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, n_assets)
        _, (h_n, _) = self.lstm(x)
        hidden = self.dropout(h_n[-1])
        logits = self.head(hidden)  # (batch, n_assets)

        if self.weight_scheme == "softmax":
            # Long-only: non-negative weights that sum to exactly 1.
            return torch.softmax(logits, dim=-1)

        # "tanh_norm": each weight in (-1, 1), then L1-normalized so the
        # book is fully invested (sum of absolute weights == 1).
        raw = torch.tanh(logits)
        return raw / (raw.abs().sum(dim=-1, keepdim=True) + 1e-8)

    def _checkpoint_dict(
        self, x_mean: np.ndarray, x_std: np.ndarray, pairs: list[str], lookback: int,
    ) -> dict:
        """Build the checkpoint dict shared by save_model() (-> local file)
        and save_to_db() (-> quant.model_registry blob) - one definition of
        what a PortfolioLSTM checkpoint contains, serialized to either target.

        `pairs` is the ordered list of FX pairs this model was trained on -
        weight i in the model's output always corresponds to pairs[i], so
        this must be persisted alongside the weights and restored on load
        (see _from_checkpoint) rather than trusted to whatever pair list/
        order a caller happens to pass in later.

        `lookback` is the sequence length (days per input window) this
        model was trained on. The LSTM itself doesn't enforce a fixed
        sequence length (nn.LSTM runs on any T), so feeding it windows of a
        different length than training wouldn't error - it would just
        silently evaluate the model outside the regime it learned. Storing
        it lets load_pipeline rebuild windows the same size the model was
        actually trained on, rather than trusting whatever lookback a
        caller passes in later.
        """
        return {
            "config": {
                "n_assets": self.n_assets,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "weight_scheme": self.weight_scheme,
                "dropout": self.dropout_p,
                "noisy_head": self.noisy_head,
            },
            "state_dict": self.state_dict(),
            "x_mean": torch.as_tensor(x_mean),
            "x_std": torch.as_tensor(x_std),
            "pairs": list(pairs),
            "lookback": lookback,
        }

    @classmethod
    def _from_checkpoint(cls, checkpoint: dict) -> "PortfolioLSTM":
        """Reconstruct a PortfolioLSTM from a checkpoint dict (however it was
        loaded - from a local file or a DB blob). The returned model also
        carries `.x_mean`/`.x_std` (as numpy arrays), `.pairs` (the ordered
        FX pairs it was trained on), and `.lookback` (the sequence length it
        was trained on) so callers can standardize new input windows
        identically to training and build them with the correct asset
        order/length without having to re-supply either themselves.
        """
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        model.x_mean = checkpoint["x_mean"].numpy()
        model.x_std = checkpoint["x_std"].numpy()
        model.pairs = checkpoint["pairs"]
        model.lookback = checkpoint["lookback"]
        return model

    def save_model(
        self,
        path: str = "models/portfolio_lstm.pt",
        *,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        pairs: list[str],
        lookback: int,
    ) -> None:
        """Persist trained weights, the architecture config, the input
        standardization stats (x_mean/x_std), the ordered FX pairs, and the
        sequence length this model was trained on - a self-contained
        checkpoint that load_model() can rebuild and run inference from
        without retraining, without needing a fresh training split to
        re-derive x_mean/x_std, and without risking weights being applied
        to the wrong assets or windows of the wrong length.
        """
        torch.save(self._checkpoint_dict(x_mean, x_std, pairs, lookback), path)
        logger.info("Saved model weights to %s", path)

    def save_to_db(
        self,
        name: str,
        *,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        pairs: list[str],
        lookback: int,
        description: str = "",
    ) -> None:
        """Serialize the same checkpoint save_model() would write, and
        upsert it into quant.model_registry under `name` instead of a
        local file - see data/model_registry.py.
        """
        from data.model_registry import save_model_blob

        buffer = io.BytesIO()
        torch.save(self._checkpoint_dict(x_mean, x_std, pairs, lookback), buffer)
        save_model_blob(name, buffer.getvalue(), model_type="portfolio", description=description)

    @classmethod
    def load_model(cls, path: str) -> "PortfolioLSTM":
        """Reconstruct a PortfolioLSTM from a checkpoint saved by save_model()."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        return cls._from_checkpoint(checkpoint)

    @classmethod
    def load_from_db(cls, name: str) -> "PortfolioLSTM":
        """Reconstruct a PortfolioLSTM from a checkpoint saved by save_to_db()."""
        from data.model_registry import load_model_blob

        checkpoint = torch.load(io.BytesIO(load_model_blob(name)), map_location="cpu", weights_only=True)
        return cls._from_checkpoint(checkpoint)


class EnsemblePortfolioLSTM(nn.Module):
    """Wraps several independently-trained PortfolioLSTMs (different random
    seeds, so different local optima of the non-convex training landscape)
    and averages their predicted weights.

    Averaging tends to cancel out each individual run's idiosyncratic
    overfitting to the particular local optimum it landed in - a standard,
    cheap way to get a more robust result out of a non-convex training
    problem than trusting a single run.
    """

    def __init__(self, models: list[PortfolioLSTM]):
        super().__init__()
        self.models = nn.ModuleList(models)
        # Purely informational (mirrors the field every member already has);
        # all members share the same scheme since they're restarts of the
        # same training run.
        self.weight_scheme = models[0].weight_scheme

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.stack([m(x) for m in self.models], dim=0)  # (n_models, batch, n_assets)
        averaged = weights.mean(dim=0)                              # (batch, n_assets)
        # A simple average of several softmax vectors already sums to
        # exactly 1 (a convex combination of points on the simplex stays on
        # the simplex) - this re-normalization is a no-op there. For
        # tanh_norm it's not a no-op: individual models can disagree on
        # sign per asset, so naive averaging can shrink the sum of absolute
        # weights below 1. Dividing by it here restores the same "fully
        # invested" invariant each individual model already satisfies.
        return averaged / (averaged.abs().sum(dim=-1, keepdim=True) + 1e-8)

    def _checkpoint_dict(
        self, x_mean: np.ndarray, x_std: np.ndarray, pairs: list[str], lookback: int,
    ) -> dict:
        """Build the checkpoint dict shared by save_model() (-> local file)
        and save_to_db() (-> quant.model_registry blob). All restarts were
        trained on the same standardized data, so one x_mean/x_std pair, one
        ordered `pairs` list, and one `lookback` (see
        PortfolioLSTM._checkpoint_dict) cover the whole ensemble.
        """
        return {
            "members": [
                {
                    "config": {
                        "n_assets": m.n_assets,
                        "hidden_size": m.hidden_size,
                        "num_layers": m.num_layers,
                        "weight_scheme": m.weight_scheme,
                        "dropout": m.dropout_p,
                        "noisy_head": m.noisy_head,
                    },
                    "state_dict": m.state_dict(),
                }
                for m in self.models
            ],
            "x_mean": torch.as_tensor(x_mean),
            "x_std": torch.as_tensor(x_std),
            "pairs": list(pairs),
            "lookback": lookback,
        }

    @classmethod
    def _from_checkpoint(cls, checkpoint: dict) -> "EnsemblePortfolioLSTM":
        """Reconstruct every member from a checkpoint dict (however it was
        loaded - from a local file or a DB blob). The returned ensemble
        also carries `.x_mean`/`.x_std` (as numpy arrays), `.pairs`, and
        `.lookback`, same as PortfolioLSTM._from_checkpoint()."""
        members = []
        for member_checkpoint in checkpoint["members"]:
            member = PortfolioLSTM(**member_checkpoint["config"])
            member.load_state_dict(member_checkpoint["state_dict"])
            member.eval()
            members.append(member)
        ensemble = cls(members)
        ensemble.eval()
        ensemble.x_mean = checkpoint["x_mean"].numpy()
        ensemble.x_std = checkpoint["x_std"].numpy()
        ensemble.pairs = checkpoint["pairs"]
        ensemble.lookback = checkpoint["lookback"]
        return ensemble

    def save_model(
        self,
        path: str = "models/portfolio_lstm_ensemble.pt",
        *,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        pairs: list[str],
        lookback: int,
    ) -> None:
        """Persist every member's config + weights so the ensemble can be reloaded without retraining any of them."""
        torch.save(self._checkpoint_dict(x_mean, x_std, pairs, lookback), path)
        logger.info("Saved ensemble weights (%d members) to %s", len(self.models), path)

    def save_to_db(
        self,
        name: str,
        *,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        pairs: list[str],
        lookback: int,
        description: str = "",
    ) -> None:
        """Serialize the same checkpoint save_model() would write, and
        upsert it into quant.model_registry under `name` instead of a
        local file.
        """
        from data.model_registry import save_model_blob

        buffer = io.BytesIO()
        torch.save(self._checkpoint_dict(x_mean, x_std, pairs, lookback), buffer)
        save_model_blob(name, buffer.getvalue(), model_type="portfolio_ensemble", description=description)

    @classmethod
    def load_model(cls, path: str) -> "EnsemblePortfolioLSTM":
        """Reconstruct every member from a checkpoint saved by save_model()."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        return cls._from_checkpoint(checkpoint)

    @classmethod
    def load_from_db(cls, name: str) -> "EnsemblePortfolioLSTM":
        """Reconstruct every member from a checkpoint saved by save_to_db()."""
        from data.model_registry import load_model_blob

        checkpoint = torch.load(io.BytesIO(load_model_blob(name)), map_location="cpu", weights_only=True)
        return cls._from_checkpoint(checkpoint)


def load_portfolio_model(path: str) -> nn.Module:
    """Load a PortfolioLSTM or EnsemblePortfolioLSTM checkpoint from a local
    file, auto-detecting which kind it is from the file's structure -
    callers (e.g. load_pipeline() below) don't need to know in advance
    whether `path` holds a single model (saved with --restart-strategy
    best or --n-seeds 1) or an ensemble (saved with --restart-strategy
    ensemble).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "members" in checkpoint:
        return EnsemblePortfolioLSTM.load_model(path)
    return PortfolioLSTM.load_model(path)


def load_portfolio_model_from_db(name: str) -> nn.Module:
    """Load a PortfolioLSTM or EnsemblePortfolioLSTM checkpoint from
    quant.model_registry by name, auto-detecting single vs. ensemble the
    same way load_portfolio_model() does for local files.
    """
    from data.model_registry import load_model_blob

    blob = load_model_blob(name)
    checkpoint = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    if "members" in checkpoint:
        return EnsemblePortfolioLSTM._from_checkpoint(checkpoint)
    return PortfolioLSTM._from_checkpoint(checkpoint)


# --------------------------------------------------------------------------
# Sharpe ratio: the training objective
# --------------------------------------------------------------------------

def sharpe_ratio(
    portfolio_returns: torch.Tensor, periods_per_year: int = 252, eps: float = 1e-8
) -> torch.Tensor:
    """Annualized Sharpe ratio of a portfolio return series:
    mean(returns) / std(returns) * sqrt(periods_per_year).

    Differentiable in `portfolio_returns`, so this can be used directly as
    a (negated) training loss - no proxy metric like MSE is needed.
    """
    mean = portfolio_returns.mean()
    std = portfolio_returns.std() + eps
    return mean / std * (periods_per_year ** 0.5)


# --------------------------------------------------------------------------
# Volatility targeting: rescale PortfolioLSTM's raw weights to a target vol
# --------------------------------------------------------------------------

def portfolio_volatility(
    window_returns: torch.Tensor, weights: torch.Tensor, periods_per_year: int = 252
) -> torch.Tensor:
    """Estimate each sample's ANNUALIZED portfolio volatility from the
    realized covariance of every asset's log returns over `window_returns`
    (the same RAW, real-scale lookback window the weights were computed
    from) and the proposed `weights`.

    window_returns: (batch, lookback, n_assets), real (unstandardized) log
    returns - real units are required since this gets compared against a
    real target like 20% annualized volatility.
    weights: (batch, n_assets).
    Returns: (batch,) - annualized volatility of dot(weights, daily_return)
    under each sample's own window covariance estimate (w^T . Sigma . w).
    """
    lookback = window_returns.shape[1]
    centered = window_returns - window_returns.mean(dim=1, keepdim=True)  # (batch, lookback, n_assets)
    # Per-sample covariance matrix: (batch, n_assets, n_assets).
    cov = torch.einsum("bti,btj->bij", centered, centered) / max(lookback - 1, 1)
    portfolio_var = torch.einsum("bi,bij,bj->b", weights, cov, weights).clamp(min=1e-12)
    return torch.sqrt(portfolio_var) * (periods_per_year ** 0.5)


def scale_weights_to_target_vol(
    weights: torch.Tensor, window_returns: torch.Tensor, target_vol: float, periods_per_year: int = 252
) -> torch.Tensor:
    """Rescale `weights` uniformly per sample so the resulting portfolio's
    estimated annualized volatility (see portfolio_volatility) matches
    `target_vol` - standard volatility targeting / risk-parity-style
    leverage, applied right after PortfolioLSTM's own weight computation.

    FX portfolios are typically low-vol day to day, so hitting a target
    like 20% annualized usually means scaling weights UP - sum(weights)
    can end up above 1 (real leverage). That's the intended effect of
    vol-targeting, not a bug: everything downstream (Sharpe training
    objective, reported PnL, the risk overlay) operates on these
    already-scaled weights, and nothing scales them again afterwards.
    """
    vol = portfolio_volatility(window_returns, weights, periods_per_year).clamp(min=1e-8)
    scale = target_vol / vol
    return weights * scale.unsqueeze(-1)


# --------------------------------------------------------------------------
# Transaction costs
# --------------------------------------------------------------------------

def apply_transaction_costs(
    weights: np.ndarray, returns: np.ndarray, transaction_cost_bps: float
) -> np.ndarray:
    """Subtract an estimated transaction-cost drag from a realized return
    series, based on day-to-day TURNOVER in `weights` - the standard way to
    approximate the real-world cost of actually executing a strategy's
    rebalancing (spread, slippage, commissions).

    Numpy version, used for REPORTING (plots/API responses) once weights
    are already fixed numbers. See apply_transaction_costs_torch for the
    differentiable version used INSIDE training.

    weights: (n_days, n_assets) - the FINAL weights (e.g. risk-attenuated)
    the `returns` series was realized from.
    returns: (n_days,) - realized portfolio returns for the same days.
    transaction_cost_bps: cost per unit of turnover, in basis points (1 bps
    = 0.0001 = 0.01% of notional). E.g. 5.0 means fully rebuilding the book
    from flat costs 0.05% of NAV.

    turnover_t = sum(|weights_t - weights_{t-1}|) - how much of the book
    was bought/sold moving from yesterday's position to today's. The very
    first day charges sum(|weights_0|), the cost of putting the initial
    position on from flat.
    """
    turnover = np.empty(len(weights), dtype=weights.dtype)
    turnover[0] = np.abs(weights[0]).sum()
    turnover[1:] = np.abs(np.diff(weights, axis=0)).sum(axis=1)
    cost = (transaction_cost_bps / 10_000) * turnover
    return returns - cost


def apply_transaction_costs_torch(
    weights: torch.Tensor, returns: torch.Tensor, transaction_cost_bps: float
) -> torch.Tensor:
    """Differentiable twin of apply_transaction_costs, for use INSIDE the
    training loop (see `transaction_cost` in train_portfolio_model/
    train_joint_model/train_risk_model) rather than only as a post-hoc
    reporting adjustment.

    Training the Sharpe objective directly on these net-of-cost returns
    (instead of gross returns) makes the model favor smoother weight paths
    over abrupt day-to-day swings whenever `transaction_cost_bps` > 0 -
    without adding any extra tunable penalty term: it reuses the same
    turnover-based cost that was already being reported, just moved inside
    the loss instead of applied only afterwards. `transaction_cost_bps=0`
    (the default) reproduces the original gross-return objective exactly.

    weights: (n_days, n_assets), requires_grad-tracked - the FINAL weights
    (post vol-targeting and, if present, risk attenuation) for the whole
    training period this epoch.
    returns: (n_days,) - the (differentiable) gross portfolio returns for
    the same days.
    """
    if transaction_cost_bps <= 0:
        return returns
    turnover_first = weights[:1].abs().sum(dim=-1)          # (1,) - cost of the initial position from flat
    turnover_rest = (weights[1:] - weights[:-1]).abs().sum(dim=-1)  # (n_days - 1,)
    turnover = torch.cat([turnover_first, turnover_rest], dim=0)    # (n_days,)
    cost = (transaction_cost_bps / 10_000) * turnover
    return returns - cost


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train_portfolio_model(
    model: nn.Module,
    X_train: torch.Tensor,
    X_train_raw: torch.Tensor,
    next_returns_train: torch.Tensor,
    target_vol: float,
    epochs: int,
    lr: float,
    weight_decay: float = 0.0,
    noise_std: float = 0.0,
    transaction_cost: float = 0.0,
) -> None:
    """Full-batch training: at every epoch, allocate weights for the WHOLE
    training period at once, rescale them to --target-vol (see
    scale_weights_to_target_vol), compute the resulting return series, and
    take a gradient step on its (negated) Sharpe ratio - so the model is
    trained to directly maximize the Sharpe ratio of the volatility-
    targeted portfolio it will actually be evaluated on, not the raw one.

    This is full-batch rather than mini-batch on purpose: Sharpe ratio is a
    statistic of the entire return distribution (mean/std), so it can only
    be evaluated meaningfully over a full set of returns, not one sample at
    a time as with a per-sample loss like MSE.

    `noise_std` > 0 adds fresh Gaussian noise to the (already standardized)
    input window on every epoch - the model sees a slightly different
    version of the training data each time, so it can't memorize the exact
    training sequence, only patterns that survive small perturbations of
    it. Note the vol-targeting scale is computed from the CLEAN raw window
    (`X_train_raw`), not the noised one - noise regularizes what the LSTM
    sees, it shouldn't distort the real volatility estimate. `weight_decay`
    is passed straight through to Adam as an L2 penalty.

    `transaction_cost` > 0 (basis points per unit of turnover - the same
    knob apply_transaction_costs uses for reporting) subtracts a turnover-
    based cost from the return series BEFORE computing Sharpe (see
    apply_transaction_costs_torch), so the model is trained to maximize
    Sharpe net of trading costs directly - it learns to avoid abrupt,
    expensive day-to-day weight swings whenever they're not worth their
    cost, rather than being scored on gross returns and only having costs
    applied afterwards as a reporting adjustment.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        noisy_X_train = X_train + torch.randn_like(X_train) * noise_std if noise_std > 0 else X_train
        raw_weights = model(noisy_X_train)                                       # (n_train, n_assets)
        weights = scale_weights_to_target_vol(raw_weights, X_train_raw, target_vol)
        portfolio_returns = (weights * next_returns_train).sum(dim=-1)  # (n_train,)
        portfolio_returns = apply_transaction_costs_torch(weights, portfolio_returns, transaction_cost)
        loss = -sharpe_ratio(portfolio_returns)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % max(epochs // 10, 1) == 0:
            logger.info("epoch %d/%d - train Sharpe (vol-targeted, net of costs) %.4f", epoch, epochs, -loss.item())


# --------------------------------------------------------------------------
# End-to-end pipeline: shared by this script's CLI and models/portfolio_postprocess.py
# --------------------------------------------------------------------------

@dataclass
class PortfolioResult:
    """Everything needed to score and plot a trained portfolio allocator.

    Also carries the standardized input tensors and raw next-day returns
    (X_train/X_val, next_returns_train/next_returns_val) so a downstream
    model - e.g. models/risk_lstm.py's risk overlay - can reuse the exact
    same data and train/validation split without loading and re-slicing
    everything a second time.

    `weights_train`/`weights_val` and `returns_train`/`returns_val` are the
    FINAL, volatility-targeted weights/returns (PortfolioLSTM's raw output
    rescaled to `target_vol` via scale_weights_to_target_vol) - this is
    what's used for training's Sharpe objective, what the risk overlay
    attenuates, and what gets reported everywhere. `weights_train_unscaled`
    /`weights_val_unscaled` and `returns_train_unscaled`/`returns_val_unscaled`
    keep PortfolioLSTM's ORIGINAL (pre-vol-targeting) weights/returns around
    too, purely for comparison (e.g. the raw-vs-vol-targeted-vs-attenuated
    PnL plot in models/risk_postprocess.py).
    """

    model: PortfolioLSTM
    pairs: list[str]
    lookback: int
    weight_scheme: str
    target_vol: float
    dates_train: pd.DatetimeIndex
    dates_val: pd.DatetimeIndex
    returns_train: np.ndarray   # realized daily portfolio log returns, in-sample (vol-targeted)
    returns_val: np.ndarray     # realized daily portfolio log returns, out-of-sample (vol-targeted)
    weights_train: np.ndarray   # (n_train, n_assets), vol-targeted
    weights_val: np.ndarray     # (n_val, n_assets), vol-targeted
    returns_train_unscaled: np.ndarray  # pre-vol-targeting, in-sample
    returns_val_unscaled: np.ndarray    # pre-vol-targeting, out-of-sample
    weights_train_unscaled: np.ndarray  # (n_train, n_assets), pre-vol-targeting
    weights_val_unscaled: np.ndarray    # (n_val, n_assets), pre-vol-targeting
    X_train: torch.Tensor       # standardized input windows, in-sample
    X_val: torch.Tensor         # standardized input windows, out-of-sample
    next_returns_train: np.ndarray  # raw log-return scale, (n_train, n_assets)
    next_returns_val: np.ndarray    # raw log-return scale, (n_val, n_assets)
    x_mean: np.ndarray          # standardization stats X was fit/loaded with
    x_std: np.ndarray


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
    # PortfolioLSTM architecture / training
    "weight_scheme": "softmax",  # "softmax" (long-only) or "tanh_norm" (long/short)
    "hidden_size": 32,
    "epochs": 300,
    "lr": 1e-3,
    "dropout": 0.1,
    "weight_decay": 1e-4,
    "noise_std": 0.05,
    "target_vol": 0.20,
    # Parameter-space noise (NoisyNet) on the output head(s) - PortfolioLSTM's
    # head, and RiskLSTM's head_2 when risk_overlay is on - resampled every
    # forward() call during training, mu-only (deterministic) at eval time.
    # Composes with noise_std rather than replacing it: noise_std perturbs
    # the input, this perturbs the model's own decision boundary, so neither
    # alone is enough to memorize one exact Sharpe-maximizing configuration.
    "noisy_head": False,
    # Multi-seed restarts (see run_pipeline_multi_seed)
    "n_seeds": 1,
    "restart_strategy": "best",  # "best" or "ensemble"
    # Risk overlay (see models/risk_lstm.py)
    "risk_overlay": False,
    "risk_hidden_size": 16,
    "risk_epochs": 200,
    "risk_lr": 1e-3,
    "max_attenuation": 0.33,
    "risk_rolling_window": 10,
    # Transaction costs, in basis points per unit of turnover - 0 disables
    # them entirely. Applied BOTH inside the training objective (Sharpe is
    # computed net of turnover-based costs - see apply_transaction_costs_torch
    # - so the model learns to avoid abrupt, costly weight swings) and to
    # reported/plotted returns (see apply_transaction_costs).
    "transaction_cost": 0.0,
    # Persistence: each of load_portfolio/load_risk accepts EITHER a local
    # .pt file path OR a quant.model_registry name (see
    # load_portfolio_model_auto below); save_db additionally persists
    # whatever gets trained/loaded to Postgres under a deterministic name.
    "load_portfolio": None,
    "load_risk": None,
    "save_db": False,
    "model_description": "",
    # Plot output paths (main.py always saves cumulative PnL; the rest are
    # only used when risk_overlay is true).
    "output": "models/portfolio_pnl.png",
    "position_output": "models/risk_position.png",
    "vol_matched_output": "models/risk_vol_matched_pnl.png",
    "histogram_output": "models/risk_return_histogram.png",
    "transaction_cost_output": "models/risk_transaction_cost_pnl.png",
}


def portfolio_model_name(args: argparse.Namespace) -> str:
    """Deterministic quant.model_registry name for a PortfolioLSTM/-ensemble
    trained with `args` - built from the characteristics that actually
    change the trained model (pairs, weight scheme, lookback, hidden size,
    target vol), so the same configuration always maps to the same name
    and re-saving under it is a natural update.
    """
    from data.model_registry import build_model_name

    is_ensemble = args.n_seeds > 1 and args.restart_strategy == "ensemble"
    return build_model_name(
        "portfolio_ensemble" if is_ensemble else "portfolio",
        pairs=sorted(args.pairs),
        weight_scheme=args.weight_scheme,
        lookback=args.lookback,
        hidden_size=args.hidden_size,
        target_vol=args.target_vol,
    )


def load_portfolio_model_auto(value: str) -> nn.Module:
    """Load a PortfolioLSTM/-ensemble from either a local file path or a
    quant.model_registry name - tries the local file first (so an existing
    "load_portfolio": "<path>.pt" config keeps working unchanged), and
    falls back to the database if no such file exists.
    """
    if os.path.exists(value):
        return load_portfolio_model(value)
    return load_portfolio_model_from_db(value)


@dataclass
class _PreparedData:
    """Data shared across restarts - loaded and split once, not once per seed."""

    pairs: list[str]
    lookback: int
    dates_train: pd.DatetimeIndex
    dates_val: pd.DatetimeIndex
    X_train: torch.Tensor
    X_val: torch.Tensor
    X_train_raw: torch.Tensor   # real (unstandardized) log-return windows - for volatility targeting
    X_val_raw: torch.Tensor
    next_returns_train_raw: np.ndarray
    next_returns_val_raw: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray


def _prepare_data(
    args: argparse.Namespace,
    x_mean: np.ndarray | None = None,
    x_std: np.ndarray | None = None,
    pairs: list[str] | None = None,
    lookback: int | None = None,
) -> _PreparedData:
    """Load data (via db.py), build sequences, split by time, and
    standardize - everything a training (or inference) run needs that
    doesn't depend on the model's random seed.

    If `x_mean`/`x_std` are given (loading a previously-saved model), they
    are used as-is instead of being freshly fit - inference must standardize
    new data exactly the way the loaded model was trained, not according to
    whatever training split happens to be in front of it today.

    If `pairs`/`lookback` are given, they OVERRIDE args.pairs/args.lookback
    - used when loading a saved model, whose checkpoint carries the exact
    ordered pair list AND sequence length it was trained on (see
    PortfolioLSTM._from_checkpoint/load_pipeline below). Model weight i
    always corresponds to pairs[i], and the LSTM's learned dynamics assume
    windows of exactly the trained lookback, so a loaded model's own stored
    values must win over whatever a caller (e.g. a UI form) passes in.
    """
    if pairs is None:
        pairs = list(dict.fromkeys(args.pairs))  # de-duplicate, keep order
    if lookback is None:
        lookback = args.lookback

    logger.info("Loading %s via db.py", pairs)
    prices = load_close_prices(pairs, years=args.years)
    returns = to_log_returns(prices)

    min_rows = lookback + 1  # need `lookback` history days plus 1 realized return to form even one sequence
    if len(returns) < min_rows:
        raise ValueError(
            f"Only {len(returns)} days of history available for {pairs}, but this model needs at least "
            f"{min_rows} (sequence length {lookback} + 1) - increase 'years' to fetch more history."
        )

    X, next_returns, dates = make_portfolio_sequences(returns, lookback=lookback)

    # Time-based split: train on the earlier segment, validate on the later
    # (most recent) one - this is the out-of-sample set.
    n_train = int(len(X) * args.train_frac)
    X_train_raw, X_val_raw = X[:n_train], X[n_train:]
    next_returns_train_raw, next_returns_val_raw = next_returns[:n_train], next_returns[n_train:]
    dates_train, dates_val = dates[:n_train], dates[n_train:]

    # Standardize only the LSTM's input features. next_returns stay on the
    # real log-return scale - they're multiplied by the weights to get real
    # portfolio P&L, so they must not be rescaled.
    if x_mean is None or x_std is None:
        x_mean, x_std = standardize(X_train_raw, axis=(0, 1))  # TRAIN-only stats
    X_train = torch.tensor((X_train_raw - x_mean) / x_std)
    X_val = torch.tensor((X_val_raw - x_mean) / x_std)

    return _PreparedData(
        pairs=pairs,
        lookback=lookback,
        dates_train=dates_train,
        dates_val=dates_val,
        X_train=X_train,
        X_val=X_val,
        X_train_raw=torch.tensor(X_train_raw),
        X_val_raw=torch.tensor(X_val_raw),
        next_returns_train_raw=next_returns_train_raw,
        next_returns_val_raw=next_returns_val_raw,
        x_mean=x_mean,
        x_std=x_std,
    )


def evaluate_portfolio_model(model: nn.Module, data: _PreparedData, weight_scheme: str, target_vol: float) -> PortfolioResult:
    """Run an already-trained-or-loaded model (eval mode, no grad) over both
    splits, rescale its raw weights to `target_vol` (see
    scale_weights_to_target_vol), and package the result - keeping both the
    pre-scaling ("unscaled") and post-scaling (vol-targeted) weights/returns
    so callers can compare them. Shared by the train path (after fitting)
    and the load path (skips fitting entirely).
    """
    model.eval()
    with torch.no_grad():
        raw_weights_train = model(data.X_train)
        raw_weights_val = model(data.X_val)
        weights_train_t = scale_weights_to_target_vol(raw_weights_train, data.X_train_raw, target_vol)
        weights_val_t = scale_weights_to_target_vol(raw_weights_val, data.X_val_raw, target_vol)

    weights_train_unscaled = raw_weights_train.numpy()
    weights_val_unscaled = raw_weights_val.numpy()
    weights_train = weights_train_t.numpy()
    weights_val = weights_val_t.numpy()

    returns_train_unscaled = (weights_train_unscaled * data.next_returns_train_raw).sum(axis=1)
    returns_val_unscaled = (weights_val_unscaled * data.next_returns_val_raw).sum(axis=1)
    returns_train = (weights_train * data.next_returns_train_raw).sum(axis=1)
    returns_val = (weights_val * data.next_returns_val_raw).sum(axis=1)

    return PortfolioResult(
        model=model,
        pairs=data.pairs,
        lookback=data.lookback,
        weight_scheme=weight_scheme,
        target_vol=target_vol,
        dates_train=data.dates_train,
        dates_val=data.dates_val,
        returns_train=returns_train,
        returns_val=returns_val,
        weights_train=weights_train,
        weights_val=weights_val,
        returns_train_unscaled=returns_train_unscaled,
        returns_val_unscaled=returns_val_unscaled,
        weights_train_unscaled=weights_train_unscaled,
        weights_val_unscaled=weights_val_unscaled,
        X_train=data.X_train,
        X_val=data.X_val,
        next_returns_train=data.next_returns_train_raw,
        next_returns_val=data.next_returns_val_raw,
        x_mean=data.x_mean,
        x_std=data.x_std,
    )


def _train_and_evaluate(data: _PreparedData, args: argparse.Namespace) -> PortfolioResult:
    """Train one PortfolioLSTM (whatever random seed is currently set) on
    already-prepared data, and evaluate it on both splits.
    """
    model = PortfolioLSTM(
        n_assets=len(data.pairs),
        hidden_size=args.hidden_size,
        weight_scheme=args.weight_scheme,
        dropout=args.dropout,
        noisy_head=args.noisy_head,
    )
    train_portfolio_model(
        model, data.X_train, data.X_train_raw, torch.tensor(data.next_returns_train_raw),
        target_vol=args.target_vol,
        epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, noise_std=args.noise_std,
        transaction_cost=args.transaction_cost,
    )
    return evaluate_portfolio_model(model, data, args.weight_scheme, args.target_vol)


def load_pipeline(args: argparse.Namespace) -> PortfolioResult:
    """Load a previously-trained PortfolioLSTM (or ensemble) from
    `args.load_portfolio` (a local file path OR a quant.model_registry
    name - see load_portfolio_model_auto) and evaluate it on freshly-loaded
    data - no training happens at all. The checkpoint carries its own
    x_mean/x_std (fit during its original training run), the ordered list
    of FX pairs, and the sequence length (lookback) it was trained on, so
    new data is standardized/windowed identically without needing to
    reconstruct the original training split or trust whatever
    `args.pairs`/`args.lookback` a caller passed in - those are genuinely
    part of the model, not free evaluation-time parameters. Only the
    amount of history to fetch (`args.years`), how much of it counts as
    "recent" (`args.train_frac`), and post-hoc knobs (`args.target_vol`,
    transaction cost) remain caller-controlled.
    """
    model = load_portfolio_model_auto(args.load_portfolio)
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
    data = _prepare_data(args, x_mean=model.x_mean, x_std=model.x_std, pairs=model.pairs, lookback=model.lookback)
    return evaluate_portfolio_model(model, data, model.weight_scheme, args.target_vol)


def run_pipeline(args: argparse.Namespace) -> PortfolioResult:
    """Load data (via db.py), build sequences, train the portfolio LSTM on
    the train split, and evaluate realized returns on both the train
    (in-sample) and validation (out-of-sample) splits. Single run, whatever
    the ambient random seed is - see run_pipeline_multi_seed() for restarts.
    """
    if args.load_portfolio:
        return load_pipeline(args)
    return _train_and_evaluate(_prepare_data(args), args)


def run_pipeline_multi_seed(args: argparse.Namespace) -> PortfolioResult:
    """Train --n-seeds independent PortfolioLSTMs on the same data (only the
    random seed differs) and combine them via --restart-strategy.

    The Sharpe-ratio training objective is non-convex in the LSTM's
    parameters (the LSTM/softmax nonlinearities, not the Sharpe ratio
    itself, are what make it non-convex - that's true of any neural net
    regardless of loss function), so different random initializations can
    land in meaningfully different local optima. Training several and
    either keeping the best-validated one or averaging them is a standard,
    cheap way to get a more robust result than trusting a single run.

    If --load-portfolio is set, restarts don't apply at all - there's
    nothing to train, so this just delegates to load_pipeline().
    """
    if args.load_portfolio:
        return load_pipeline(args)

    if args.n_seeds < 1:
        raise ValueError(f"--n-seeds must be >= 1, got {args.n_seeds}")

    data = _prepare_data(args)  # load/split/standardize once, reused by every seed

    results = []
    for seed in range(args.n_seeds):
        torch.manual_seed(seed)
        logger.info("--- restart %d/%d (seed=%d) ---", seed + 1, args.n_seeds, seed)
        results.append(_train_and_evaluate(data, args))

    if len(results) == 1:
        return results[0]

    if args.restart_strategy == "best":
        best_idx, best = max(
            enumerate(results), key=lambda item: float(sharpe_ratio(torch.tensor(item[1].returns_val)))
        )
        logger.info(
            "Best of %d restarts: #%d (validation Sharpe %.3f)",
            len(results), best_idx, float(sharpe_ratio(torch.tensor(best.returns_val))),
        )
        return best

    # "ensemble": average every restart's predicted weights.
    ensemble_model = EnsemblePortfolioLSTM([r.model for r in results])
    ensemble_result = evaluate_portfolio_model(ensemble_model, data, args.weight_scheme, args.target_vol)
    logger.info(
        "Ensemble of %d restarts: validation Sharpe %.3f",
        len(results), float(sharpe_ratio(torch.tensor(ensemble_result.returns_val))),
    )
    return ensemble_result


def print_portfolio_sharpe(result: PortfolioResult) -> None:
    """Log raw (pre-vol-targeting) vs vol-targeted Sharpe ratio and
    cumulative PnL, for both splits. Used by main.py after training/loading."""
    unscaled_train_sharpe = float(sharpe_ratio(torch.tensor(result.returns_train_unscaled)))
    unscaled_val_sharpe = float(sharpe_ratio(torch.tensor(result.returns_val_unscaled)))
    train_sharpe = float(sharpe_ratio(torch.tensor(result.returns_train)))
    val_sharpe = float(sharpe_ratio(torch.tensor(result.returns_val)))
    logger.info(
        "In-sample  (train): raw Sharpe %.3f -> vol-targeted (%.0f%%) Sharpe %.3f | cumulative PnL %.4f",
        unscaled_train_sharpe, result.target_vol * 100, train_sharpe, float(result.returns_train.sum()),
    )
    logger.info(
        "Out-of-sample (val): raw Sharpe %.3f -> vol-targeted (%.0f%%) Sharpe %.3f | cumulative PnL %.4f",
        unscaled_val_sharpe, result.target_vol * 100, val_sharpe, float(result.returns_val.sum()),
    )
