"""LSTM portfolio allocator: learns weights that directly maximize Sharpe ratio.

Unlike models/lstm_forecaster.py (which predicts a return value and is
trained to minimize forecast error), this model's output IS a trading
decision: at each day it looks at the last `lookback` days of log returns
for every pair in the portfolio and outputs a weight per pair for the
following day. It is trained end-to-end to maximize the Sharpe ratio of
the resulting portfolio, not to predict returns accurately - Sharpe ratio
(mean return / return volatility) is what a portfolio manager actually
cares about, and it is differentiable, so we can optimize it directly with
gradient descent instead of using a proxy loss like MSE.

Data flow
---------
1. Load daily close prices for the portfolio's FX pairs via data/db.py
   (see load_close_prices in models/lstm_forecaster.py - reused here).
2. Convert to log returns.
3. Slide a window over the returns:
       X[i]            = log returns of every pair over `lookback` days
       next_returns[i] = log returns of every pair on the single day right
                         after the window - i.e. what a weight decided at
                         the end of the window would actually earn.
4. Split by time into TRAIN (earliest) and VALIDATION (most recent, held
   out) - exactly as in lstm_forecaster.py.
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
  - `--noise-std`: Gaussian noise added to the (standardized) input
    window on every training epoch, regenerated fresh each time. This is
    the standard "add noise to the input" regularizer: the model can no
    longer fit the exact training sequence of returns, only patterns
    robust to small perturbations of it.
  - `--dropout`: dropout on the LSTM's final hidden state before the
    linear head, so the head can't rely on any single hidden unit.
  - `--weight-decay`: L2 penalty on the model weights (Adam's
    `weight_decay`), shrinking the model towards simpler solutions.

Usage
-----
    python -m models.portfolio_lstm \
        --pairs EURUSD GBPUSD USDJPY \
        --lookback 30 --weight-scheme softmax --epochs 300
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from models.lstm_forecaster import load_close_prices, standardize, to_log_returns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------

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
    # Guard the no-look-ahead invariant: every window must be exactly
    # `lookback` long, and the day it predicts must never be one of the
    # rows fed into the model.
    assert X.shape[1] == lookback
    assert not np.any(np.all(X == next_returns[:, None, :], axis=2))
    return X, next_returns, pd.DatetimeIndex(dates)


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
        self.lstm = nn.LSTM(
            input_size=n_assets,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # inputs shaped (batch, time, features)
        )
        # Dropout on the final hidden state only - a no-op (p=0) when
        # `dropout` isn't set, and inactive automatically in model.eval().
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, n_assets)

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

    def save_model(self, path: str = "models/portfolio_lstm.pt", *, x_mean: np.ndarray, x_std: np.ndarray) -> None:
        """Persist trained weights, the architecture config, and the input
        standardization stats (x_mean/x_std) this model was trained with -
        a self-contained checkpoint that load_model() can rebuild and run
        inference from without retraining or needing a fresh training split
        to re-derive x_mean/x_std.
        """
        torch.save(
            {
                "config": {
                    "n_assets": self.n_assets,
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "weight_scheme": self.weight_scheme,
                    "dropout": self.dropout_p,
                },
                "state_dict": self.state_dict(),
                "x_mean": torch.as_tensor(x_mean),
                "x_std": torch.as_tensor(x_std),
            },
            path,
        )
        logger.info("Saved model weights to %s", path)

    @classmethod
    def load_model(cls, path: str) -> "PortfolioLSTM":
        """Reconstruct a PortfolioLSTM from a checkpoint saved by save_model().
        The returned model also carries `.x_mean`/`.x_std` (as numpy arrays)
        so callers can standardize new input windows identically to training."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        model.x_mean = checkpoint["x_mean"].numpy()
        model.x_std = checkpoint["x_std"].numpy()
        return model


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

    def save_model(
        self, path: str = "models/portfolio_lstm_ensemble.pt", *, x_mean: np.ndarray, x_std: np.ndarray
    ) -> None:
        """Persist every member's config + weights (all restarts were trained
        on the same standardized data, so one x_mean/x_std pair covers the
        whole ensemble) so it can be reloaded without retraining any of them.
        """
        torch.save(
            {
                "members": [
                    {
                        "config": {
                            "n_assets": m.n_assets,
                            "hidden_size": m.hidden_size,
                            "num_layers": m.num_layers,
                            "weight_scheme": m.weight_scheme,
                            "dropout": m.dropout_p,
                        },
                        "state_dict": m.state_dict(),
                    }
                    for m in self.models
                ],
                "x_mean": torch.as_tensor(x_mean),
                "x_std": torch.as_tensor(x_std),
            },
            path,
        )
        logger.info("Saved ensemble weights (%d members) to %s", len(self.models), path)

    @classmethod
    def load_model(cls, path: str) -> "EnsemblePortfolioLSTM":
        """Reconstruct every member from a checkpoint saved by save_model().
        The returned ensemble also carries `.x_mean`/`.x_std` (as numpy
        arrays), same as PortfolioLSTM.load_model()."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
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
        return ensemble


def load_portfolio_model(path: str) -> nn.Module:
    """Load a PortfolioLSTM or EnsemblePortfolioLSTM checkpoint, auto-detecting
    which kind it is from the file's structure - callers (e.g. load_pipeline()
    below) don't need to know in advance whether `path` holds a single model
    (saved with --restart-strategy best or --n-seeds 1) or an ensemble
    (saved with --restart-strategy ensemble).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "members" in checkpoint:
        return EnsemblePortfolioLSTM.load_model(path)
    return PortfolioLSTM.load_model(path)


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
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        noisy_X_train = X_train + torch.randn_like(X_train) * noise_std if noise_std > 0 else X_train
        raw_weights = model(noisy_X_train)                                       # (n_train, n_assets)
        weights = scale_weights_to_target_vol(raw_weights, X_train_raw, target_vol)
        portfolio_returns = (weights * next_returns_train).sum(dim=-1)  # (n_train,)
        loss = -sharpe_ratio(portfolio_returns)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % max(epochs // 10, 1) == 0:
            logger.info("epoch %d/%d - train Sharpe (vol-targeted) %.4f", epoch, epochs, -loss.item())


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


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """CLI arguments shared by this script and models/portfolio_postprocess.py."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--pairs", nargs="+", required=True, help="FX pairs in the portfolio, e.g. EURUSD GBPUSD USDJPY")
    parser.add_argument("--lookback", type=int, default=30, help="Days of history fed to the LSTM")
    parser.add_argument("--years", type=int, default=8, help="Years of history to load")
    parser.add_argument(
        "--train-frac", type=float, default=0.8,
        help="Fraction of sequences used for training; the rest is the out-of-sample validation set",
    )
    parser.add_argument(
        "--weight-scheme", choices=["softmax", "tanh_norm"], default="softmax",
        help="softmax = long-only, weights in (0,1) summing to 1. "
             "tanh_norm = long/short, weights in (-1,1), normalized so sum(|w|)=1",
    )
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout applied to the LSTM's final hidden state, before the linear head (0 disables it)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4,
        help="L2 penalty on the model weights, passed to Adam (0 disables it)",
    )
    parser.add_argument(
        "--noise-std", type=float, default=0.05,
        help="Std-dev of Gaussian noise added to the standardized training inputs each epoch (0 disables it)",
    )
    parser.add_argument(
        "--target-vol", type=float, default=0.20,
        help="Target ANNUALIZED portfolio volatility, e.g. 0.20 = 20%%. Applied right after "
             "PortfolioLSTM's own weight computation (in training and at evaluation time): weights "
             "are uniformly rescaled per day so the portfolio's estimated volatility - from the "
             "realized covariance of the same lookback window - matches this target. FX portfolios "
             "are typically low-vol, so this usually means scaling weights UP (real leverage); "
             "nothing downstream (the risk overlay included) scales them again afterwards.",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=1,
        help="Train this many independent random restarts (different seeds) and combine them via "
             "--restart-strategy, instead of trusting a single non-convex training run",
    )
    parser.add_argument(
        "--restart-strategy", choices=["best", "ensemble"], default="best",
        help="Only used when --n-seeds > 1. 'best' keeps the single restart with the highest "
             "validation Sharpe. 'ensemble' averages all restarts' predicted weights.",
    )
    parser.add_argument(
        "--risk-overlay", action="store_true",
        help="Also train a RiskLSTM on top (see models/risk_lstm.py) that attenuates these weights "
             "in (0,1) to de-risk periods with no clear trend, and report raw vs attenuated results",
    )
    parser.add_argument("--risk-hidden-size", type=int, default=16, help="Only used with --risk-overlay")
    parser.add_argument("--risk-epochs", type=int, default=200, help="Only used with --risk-overlay")
    parser.add_argument("--risk-lr", type=float, default=1e-3, help="Only used with --risk-overlay")
    parser.add_argument(
        "--max-attenuation", type=float, default=0.33,
        help="Only used with --risk-overlay. Hard FLOOR on each asset's attenuation factor, in (0, 1]: "
             "even at minimum confidence that asset is never de-risked below this fraction of its "
             "(already vol-targeted) weight; 1 means no attenuation at all (full-size position).",
    )
    parser.add_argument(
        "--risk-rolling-window", type=int, default=10,
        help="Only used with --risk-overlay. Trailing window (in days, must be < --lookback) the risk "
             "overlay computes each asset's rolling volatility, skewness, and excess kurtosis over - "
             "these moments, not raw returns, are what the risk overlay's LSTM actually reads.",
    )
    parser.add_argument(
        "--load-portfolio", default=None,
        help="Path to a PortfolioLSTM/-ensemble checkpoint saved by --n-seeds/--restart-strategy. "
             "If given, skips training entirely and just loads it for inference (positions only).",
    )

    parser.add_argument(
        "--save-portfolio", action="store_true",
        help="Used by models/portfolio_postprocess.py: save the trained PortfolioLSTM/-ensemble "
             "checkpoint after plotting (models/portfolio_lstm.py's own CLI always saves by default; "
             "this flag opts postprocess's plotting run into saving too).",
    )
    parser.add_argument(
        "--load-risk", default=None,
        help="Path to a RiskLSTM checkpoint (only used with --risk-overlay). If given alongside "
             "--load-portfolio, skips training the risk overlay too and just loads it.",
    )

    return parser


@dataclass
class _PreparedData:
    """Data shared across restarts - loaded and split once, not once per seed."""

    pairs: list[str]
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
    args: argparse.Namespace, x_mean: np.ndarray | None = None, x_std: np.ndarray | None = None
) -> _PreparedData:
    """Load data (via db.py), build sequences, split by time, and
    standardize - everything a training (or inference) run needs that
    doesn't depend on the model's random seed.

    If `x_mean`/`x_std` are given (loading a previously-saved model), they
    are used as-is instead of being freshly fit - inference must standardize
    new data exactly the way the loaded model was trained, not according to
    whatever training split happens to be in front of it today.
    """
    pairs = list(dict.fromkeys(args.pairs))  # de-duplicate, keep order

    logger.info("Loading %s via db.py", pairs)
    prices = load_close_prices(pairs, years=args.years)
    returns = to_log_returns(prices)

    X, next_returns, dates = make_portfolio_sequences(returns, lookback=args.lookback)

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


def _evaluate(model: nn.Module, data: _PreparedData, weight_scheme: str, target_vol: float) -> PortfolioResult:
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
    )
    train_portfolio_model(
        model, data.X_train, data.X_train_raw, torch.tensor(data.next_returns_train_raw),
        target_vol=args.target_vol,
        epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, noise_std=args.noise_std,
    )
    return _evaluate(model, data, args.weight_scheme, args.target_vol)


def load_pipeline(args: argparse.Namespace) -> PortfolioResult:
    """Load a previously-trained PortfolioLSTM (or ensemble) from
    --load-portfolio and evaluate it on freshly-loaded data - no training
    happens at all. The checkpoint carries its own x_mean/x_std (fit during
    its original training run), so new data is standardized identically
    without needing to reconstruct the original training split.
    """
    model = load_portfolio_model(args.load_portfolio)
    data = _prepare_data(args, x_mean=model.x_mean, x_std=model.x_std)
    return _evaluate(model, data, args.weight_scheme, args.target_vol)


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
    ensemble_result = _evaluate(ensemble_model, data, args.weight_scheme, args.target_vol)
    logger.info(
        "Ensemble of %d restarts: validation Sharpe %.3f",
        len(results), float(sharpe_ratio(torch.tensor(ensemble_result.returns_val))),
    )
    return ensemble_result


def main() -> None:
    parser = build_arg_parser("Train an LSTM portfolio allocator that maximizes Sharpe ratio.")
    args = parser.parse_args()
    result = run_pipeline_multi_seed(args)
    if not args.load_portfolio:
        result.model.save_model(x_mean=result.x_mean, x_std=result.x_std)

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

    if args.risk_overlay:
        # Deferred import: models/risk_lstm.py imports build_arg_parser and
        # run_pipeline from this module at load time, so importing it back
        # at module level here would be a circular import. Importing inside
        # main() instead defers it until both modules are already loaded.
        from models.risk_lstm import add_risk_overlay

        risk_result = add_risk_overlay(result, args)
        if not args.load_risk:
            risk_result.risk_model.save_model()

        raw_train_sharpe = float(sharpe_ratio(torch.tensor(risk_result.returns_train_raw)))
        scaled_train_sharpe = float(sharpe_ratio(torch.tensor(risk_result.returns_train_scaled)))
        raw_val_sharpe = float(sharpe_ratio(torch.tensor(risk_result.returns_val_raw)))
        scaled_val_sharpe = float(sharpe_ratio(torch.tensor(risk_result.returns_val_scaled)))

        logger.info(
            "In-sample  (train) risk overlay: raw Sharpe %.3f -> attenuated Sharpe %.3f | mean attenuation %.3f",
            raw_train_sharpe, scaled_train_sharpe, risk_result.attenuation_train.mean(),
        )
        logger.info(
            "Out-of-sample (val) risk overlay: raw Sharpe %.3f -> attenuated Sharpe %.3f | mean attenuation %.3f",
            raw_val_sharpe, scaled_val_sharpe, risk_result.attenuation_val.mean(),
        )


if __name__ == "__main__":
    main()
