"""FastAPI backend for the FX portfolio / risk-overlay pipeline.

Thin HTTP wrapper around models/portfolio_lstm.py and models/risk_lstm.py -
no business logic lives here, just request/response shaping so the React
frontend (frontend/) has a JSON API to call.

Endpoints
---------
GET  /api/pairs             - available FX pair tickers, for pair pickers
POST /api/quotes/refresh     - download + upsert the latest close prices
GET  /api/models             - list models saved in quant.model_registry
POST /api/train               - kick off a training run (background job)
GET  /api/train/{job_id}     - poll a training job's status/result
POST /api/train/{job_id}/stop - request an in-progress job stop early
POST /api/evaluate            - load model(s) by name and run inference:
                                 returns PnL series (risk-weighted baseline
                                 vs with-risk-overlay vs with-risk+costs),
                                 return histograms, and the model's
                                 recommended weights for the next day.

Run with (from the repo root):
    uvicorn api.server:app --reload --port 8000
"""

from __future__ import annotations

import contextvars
import io
import logging
import re
import threading
import uuid
from argparse import Namespace
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data.db import get_connection
from data.fx_downloader import MAJOR_FX_PAIRS
from models.portfolio_lstm import (
    DEFAULT_CONFIG,
    TrainingStopped,
    _epoch_report_callback,
    _stop_check_callback,
    apply_transaction_costs,
    load_close_prices,
    portfolio_model_name,
    scale_weights_to_target_vol,
    sharpe_ratio,
    to_log_returns,
)

# Number of sequential decisions to "warm up" a use_prev_weight=True
# model's recurrence with, before predicting the actual next-day weight -
# see _predict_latest_weights. Arbitrary but small: prev is detached
# between steps (see PortfolioLSTM.forward_sequence), so this only needs
# enough steps for the (approximate, since the true historical position
# isn't observable here) recurrence to settle, not a long context window.
PREV_WEIGHT_WARMUP_DAYS = 20

app = FastAPI(title="FX Portfolio API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev only - tighten before deploying anywhere shared
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# GET /api/pairs
# --------------------------------------------------------------------------

@app.get("/api/pairs")
def get_pairs() -> list[str]:
    """The seven major FX pairs FXDownloader knows the Yahoo ticker for -
    other pairs still work (via the "<PAIR>=X" convention), this is just
    what a pair picker can offer without the user typing a raw ticker."""
    return list(MAJOR_FX_PAIRS.keys())


# --------------------------------------------------------------------------
# POST /api/quotes/refresh
# --------------------------------------------------------------------------

class RefreshQuotesRequest(BaseModel):
    pairs: list[str]
    years: int = 8


@app.post("/api/quotes/refresh")
def refresh_quotes(req: RefreshQuotesRequest) -> dict:
    """Download the latest daily closes for `pairs` and upsert them into
    Postgres - the same path models/portfolio_lstm.py's load_close_prices
    falls back to automatically, exposed here so the frontend can trigger
    it explicitly ("get the freshest data") before evaluating.
    """
    from data.db import upsert_pairs
    from data.fx_downloader import FXDownloader

    downloader = FXDownloader(years=req.years)
    downloaded = {}
    for pair in req.pairs:
        ticker = MAJOR_FX_PAIRS.get(pair, f"{pair}=X")
        downloaded[pair] = downloader.download_pair(pair, ticker)
    upsert_pairs(downloaded)
    return {"status": "ok", "pairs": req.pairs, "rows": {p: len(df) for p, df in downloaded.items()}}


# --------------------------------------------------------------------------
# GET /api/models
# --------------------------------------------------------------------------

@app.get("/api/models")
def list_models() -> list[dict]:
    """List every model saved in quant.model_registry, newest first - the
    frontend's model picker (for the Evaluation view) reads this.

    Includes each model's own `pairs` and (for portfolio/portfolio_ensemble
    models) `lookback`, decoded from its checkpoint blob, so the frontend
    can auto-select the FX pairs and display the sequence length a chosen
    model was actually trained on, instead of requiring the user to supply
    values that must match it exactly - see api/server.py's evaluate()/
    models/portfolio_lstm.py's load_pipeline() for why the model's own
    pairs/lookback are authoritative over anything a caller might guess.
    Risk models don't store their own lookback (it's a property of the
    portfolio pipeline they were trained alongside), so `lookback` is null
    for `model_type` "risk"/"risk_ensemble".
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT name, model_type, description, created_at, updated_at, blob "
            "FROM quant.model_registry ORDER BY updated_at DESC"
        )
        rows = cur.fetchall()
    result = []
    for name, model_type, description, created_at, updated_at, blob in rows:
        checkpoint = torch.load(io.BytesIO(bytes(blob)), map_location="cpu", weights_only=True)
        result.append({
            "name": name,
            "model_type": model_type,
            "description": description,
            "created_at": created_at.isoformat(),
            "updated_at": updated_at.isoformat(),
            "size_bytes": len(blob),
            "pairs": checkpoint.get("pairs"),
            "lookback": checkpoint.get("lookback"),
        })
    return result


# --------------------------------------------------------------------------
# POST /api/train  +  GET /api/train/{job_id}
# --------------------------------------------------------------------------

# In-memory job store - fine for a local, single-process dev server; a
# multi-worker/production deployment would need a real job queue instead.
_JOBS: dict[str, dict[str, Any]] = {}

# Which job_id (if any) is training on the CURRENT thread - contextvars are
# thread-local, and each training job runs its own dedicated background
# thread (see start_training), so setting this once at the top of
# _run_training_job keeps every subsequent logger.info(...) call inside
# that thread correctly attributed to that job, with no cross-job mixing.
_current_job_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_job_id", default=None)

_EPOCH_RE = re.compile(r"epoch (\d+)/(\d+)")
_RESTART_RE = re.compile(r"restart (\d+)/(\d+) \(seed=(\d+)\)")
_MAX_LOG_LINES = 500


class _JobLogHandler(logging.Handler):
    """Captures the training pipeline's EXISTING logger.info(...) calls
    (models/portfolio_lstm.py's and models/risk_lstm.py's per-epoch and
    per-restart progress lines) into the current job's state - so the
    frontend's polling GET /api/train/{job_id} can show a live log window
    and a progress bar, without any changes to the training loops
    themselves (they already log exactly this, for the CLI's benefit).
    """

    def emit(self, record: logging.LogRecord) -> None:
        job_id = _current_job_id.get()
        job = _JOBS.get(job_id) if job_id else None
        if job is None:
            return

        message = self.format(record)
        logs = job.setdefault("logs", [])
        logs.append(message)
        if len(logs) > _MAX_LOG_LINES:
            del logs[: len(logs) - _MAX_LOG_LINES]

        progress = job.setdefault(
            "progress", {"seed_index": 1, "n_seeds": 1, "epoch": 0, "total_epochs": 1, "percent": 0.0},
        )

        restart_match = _RESTART_RE.search(message)
        if restart_match:
            progress["seed_index"] = int(restart_match.group(1))
            progress["n_seeds"] = int(restart_match.group(2))
            progress["epoch"] = 0
            self._update_percent(progress)
            return

        epoch_match = _EPOCH_RE.search(message)
        if epoch_match:
            progress["epoch"] = int(epoch_match.group(1))
            progress["total_epochs"] = int(epoch_match.group(2))
            self._update_percent(progress)

    @staticmethod
    def _update_percent(progress: dict) -> None:
        n_seeds = max(progress["n_seeds"], 1)
        total_epochs = max(progress["total_epochs"], 1)
        completed_seeds_fraction = (progress["seed_index"] - 1) / n_seeds
        current_seed_fraction = (progress["epoch"] / total_epochs) / n_seeds
        progress["percent"] = round((completed_seeds_fraction + current_seed_fraction) * 100, 1)


# Attach once, to the shared "models" ancestor logger - INFO records from
# models.portfolio_lstm's and models.risk_lstm's own loggers (each named
# after their module) propagate up to it by default, so this single
# handler sees every training job's progress regardless of which of the
# two modules is actually doing the logging at that moment.
_job_log_handler = _JobLogHandler()
_job_log_handler.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger("models").addHandler(_job_log_handler)
logging.getLogger("models").setLevel(logging.INFO)


class TrainRequest(BaseModel):
    pairs: list[str]
    lookback: int = 30
    years: int = 8
    train_frac: float = 0.8
    test_frac: float = 0.1
    weight_scheme: str = "softmax"
    hidden_size: int = 32
    epochs: int = 300
    lr: float = 1e-3
    dropout: float = 0.1
    weight_decay: float = 1e-4
    noise_std: float = 0.05
    target_vol: float = 0.20
    noisy_head: bool = False
    use_prev_weight: bool = False
    has_cash: bool = False
    cash_return: float = 0.0
    use_carry: bool = False
    vol_horizons: list[int] = []
    encoder_type: str = "concat"  # "concat" or "per_asset" - see PortfolioLSTM's docstring
    asset_combiner: str = "attention"  # "attention" or "mean" - only used when encoder_type="per_asset"
    n_attn_heads: int = 2
    covariance_estimator: str = "sample"  # "sample", "ewma", or "ledoit_wolf" - see estimate_covariance
    ewma_lambda: float = 0.94
    pooling: str = "last"  # "last" or "attention" - see TemporalAttentionPool; applies to both networks
    device: str = "auto"  # "auto" (Metal/MPS on Apple Silicon, else CUDA, else CPU), "cpu", "mps", or "cuda" - see get_device
    objective: str = "sharpe"  # "sharpe", "kelly", or "cvar" - see models/portfolio_lstm.py's compute_training_loss
    sharpe_window: int = 60
    cvar_alpha: float = 0.95
    cvar_kappa: float = 1.0
    n_seeds: int = 1
    restart_strategy: str = "best"
    risk_overlay: bool = False
    risk_hidden_size: int = 16
    risk_epochs: int = 200
    risk_lr: float = 1e-3
    max_attenuation: float = 0.33
    risk_rolling_window: int = 10
    use_cross_sectional: bool = False
    transaction_cost: float = 0.0
    save_db: bool = True
    model_description: str = ""


def _series_payload(dates, **named_arrays: np.ndarray) -> dict:
    """Build a {dates: [...], <name>: [...], ...} dict that's directly
    JSON-serializable and easy for the frontend to zip into chart points."""
    payload: dict[str, Any] = {"dates": [str(d.date()) if hasattr(d, "date") else str(d) for d in dates]}
    for name, arr in named_arrays.items():
        payload[name] = np.asarray(arr).tolist()
    return payload


def _positions_payload(dates, pairs: list[str], weights: np.ndarray) -> dict:
    """Build a {dates: [...], <pair>: [...], ...} dict from a (n_days,
    n_assets) array - one series per pair, for a per-asset line chart
    (used for both portfolio POSITIONS and per-asset ATTENUATION)."""
    payload: dict[str, Any] = {"dates": [str(d.date()) if hasattr(d, "date") else str(d) for d in dates]}
    weights = np.asarray(weights)
    for i, pair in enumerate(pairs):
        payload[pair] = weights[:, i].tolist()
    return payload


def _asset_returns_payload(
    pairs: list[str], dates_val, next_returns_val: np.ndarray, dates_train=None, next_returns_train: np.ndarray = None,
) -> dict:
    """Cumulative log-return of each underlying FX pair itself (not the
    portfolio's PnL) - lets the frontend show what the market actually did,
    for context alongside the portfolio/position charts.

    If `dates_train`/`next_returns_train` are given (Training view: full
    history), the cumulative sum runs continuously across train+val
    concatenated, and `split_date` marks where the out-of-sample period
    begins (for a vertical marker line). Without them (Evaluation view:
    out-of-sample only, matching that view's scope), it's just the
    validation-period cumulative return with no split_date.
    """
    if dates_train is not None:
        all_dates = list(dates_train) + list(dates_val)
        all_returns = np.concatenate([next_returns_train, next_returns_val], axis=0)
    else:
        all_dates = list(dates_val)
        all_returns = np.asarray(next_returns_val)

    payload = _positions_payload(all_dates, pairs, np.cumsum(all_returns, axis=0))
    if dates_train is not None and len(dates_val):
        first_val_date = dates_val[0]
        payload["split_date"] = str(first_val_date.date()) if hasattr(first_val_date, "date") else str(first_val_date)
    else:
        payload["split_date"] = None
    return payload


def _run_training_job(job_id: str, config: dict) -> None:
    """Runs on a background thread - trains PortfolioLSTM (and, if
    risk_overlay was requested, then trains RiskLSTM SEPARATELY on top of
    it - see models/risk_lstm.py's run_pipeline_multi_seed), always saves
    locally, and to quant.model_registry if save_db was requested, then
    records the result: Sharpe ratios, in-sample AND out-of-sample PnL
    series, per-asset positions, and (with a risk overlay) per-asset
    attenuation - everything the Training view plots once the job is done.
    """
    _current_job_id.set(job_id)  # scopes _JobLogHandler's capture to this thread/job for its whole lifetime

    def _interim_callback(
        stage: str, epoch: int, epochs: int, train_returns: np.ndarray,
        val_returns: np.ndarray | None, test_returns: np.ndarray | None = None,
    ) -> None:
        """Registered below via _epoch_report_callback - called from INSIDE
        train_portfolio_model/train_risk_model's own epoch loop (same
        ~10%-of-epochs cadence as their progress logging), so the Training
        view can show live-updating in-sample, validation, AND test PnL/
        Sharpe charts instead of only a progress bar and log text.
        `val_returns`/`test_returns` are None whenever the caller didn't
        have that split to report (test_returns in particular is None
        whenever test_frac=0 - no test split configured).
        """
        job = _JOBS.get(job_id)
        if job is None:
            return
        interim = {
            "stage": stage,
            "epoch": epoch,
            "total_epochs": epochs,
            "cumulative_pnl": np.cumsum(train_returns).tolist(),
            "sharpe": float(sharpe_ratio(torch.tensor(train_returns))),
        }
        if val_returns is not None:
            interim["val_cumulative_pnl"] = np.cumsum(val_returns).tolist()
            interim["val_sharpe"] = float(sharpe_ratio(torch.tensor(val_returns)))
        if test_returns is not None:
            interim["test_cumulative_pnl"] = np.cumsum(test_returns).tolist()
            interim["test_sharpe"] = float(sharpe_ratio(torch.tensor(test_returns)))
        job["interim"] = interim

    _epoch_report_callback.set(_interim_callback)
    _stop_check_callback.set(lambda: _JOBS.get(job_id, {}).get("stop_requested", False))
    _JOBS[job_id]["status"] = "running"
    try:
        args = Namespace(**{**DEFAULT_CONFIG, **config, "load_portfolio": None, "load_risk": None})

        if args.risk_overlay:
            from models.risk_lstm import risk_model_name, run_pipeline_multi_seed as run_risk_overlay_pipeline

            result = run_risk_overlay_pipeline(args)
            result.portfolio_result.model.save_model(
                x_mean=result.portfolio_result.x_mean, x_std=result.portfolio_result.x_std,
                pairs=result.portfolio_result.pairs, lookback=result.portfolio_result.lookback,
                use_carry=args.use_carry, vol_horizons=args.vol_horizons,
            )
            result.risk_model.save_model(pairs=result.portfolio_result.pairs)

            portfolio_name = portfolio_model_name(args)
            risk_name = risk_model_name(args)
            if args.save_db:
                result.portfolio_result.model.save_to_db(
                    portfolio_name,
                    x_mean=result.portfolio_result.x_mean, x_std=result.portfolio_result.x_std,
                    pairs=result.portfolio_result.pairs, lookback=result.portfolio_result.lookback,
                    use_carry=args.use_carry, vol_horizons=args.vol_horizons,
                    description=args.model_description,
                )
                result.risk_model.save_to_db(
                    risk_name, pairs=result.portfolio_result.pairs, description=args.model_description,
                )

            pr = result.portfolio_result
            pairs = pr.pairs  # the model's OWN pairs (includes "CASH" when has_cash was used) - not args.pairs
            _JOBS[job_id]["result"] = {
                "portfolio_model_name": portfolio_name,
                "risk_model_name": risk_name,
                "train_sharpe_raw": float(sharpe_ratio(torch.tensor(pr.returns_train_unscaled))),
                "train_sharpe_vol_targeted": float(sharpe_ratio(torch.tensor(result.returns_train_raw))),
                "train_sharpe_with_risk": float(sharpe_ratio(torch.tensor(result.returns_train_scaled))),
                "val_sharpe_raw": float(sharpe_ratio(torch.tensor(pr.returns_val_unscaled))),
                "val_sharpe_vol_targeted": float(sharpe_ratio(torch.tensor(result.returns_val_raw))),
                "val_sharpe_with_risk": float(sharpe_ratio(torch.tensor(result.returns_val_scaled))),
                "test_sharpe_raw": float(sharpe_ratio(torch.tensor(pr.returns_test_unscaled))),
                "test_sharpe_vol_targeted": float(sharpe_ratio(torch.tensor(result.returns_test_raw))),
                "test_sharpe_with_risk": float(sharpe_ratio(torch.tensor(result.returns_test_scaled))),
                "pnl_train": _series_payload(
                    result.dates_train,
                    vol_targeted=np.cumsum(result.returns_train_raw),
                    with_risk=np.cumsum(result.returns_train_scaled),
                    benchmark=np.cumsum(result.benchmark_returns_train),
                ),
                "pnl_val": _series_payload(
                    result.dates_val,
                    vol_targeted=np.cumsum(result.returns_val_raw),
                    with_risk=np.cumsum(result.returns_val_scaled),
                    benchmark=np.cumsum(result.benchmark_returns_val),
                ),
                "pnl_test": _series_payload(
                    result.dates_test,
                    vol_targeted=np.cumsum(result.returns_test_raw),
                    with_risk=np.cumsum(result.returns_test_scaled),
                    benchmark=np.cumsum(result.benchmark_returns_test),
                ),
                # Aligned to result.dates_train/dates_val/dates_test
                # (make_risk_sequences drops the first
                # `risk_rolling_window - 1` samples of each split) - NOT
                # pr.dates_train/pr.weights_train, which are longer, so
                # positions/attenuation stay on the same axis.
                "positions_train": _positions_payload(result.dates_train, pairs, result.weights_train),
                "positions_val": _positions_payload(result.dates_val, pairs, result.weights_val),
                "attenuation_train": _positions_payload(result.dates_train, pairs, result.attenuation_train),
                "attenuation_val": _positions_payload(result.dates_val, pairs, result.attenuation_val),
                "asset_returns": _asset_returns_payload(
                    pairs, pr.dates_val, pr.next_returns_val, pr.dates_train, pr.next_returns_train,
                ),
            }
        else:
            from models.portfolio_lstm import run_pipeline_multi_seed

            result = run_pipeline_multi_seed(args)
            pairs = result.pairs  # the model's OWN pairs (includes "CASH" when has_cash was used) - not args.pairs
            result.model.save_model(
                x_mean=result.x_mean, x_std=result.x_std, pairs=result.pairs, lookback=result.lookback,
                use_carry=args.use_carry, vol_horizons=args.vol_horizons,
            )

            portfolio_name = portfolio_model_name(args)
            if args.save_db:
                result.model.save_to_db(
                    portfolio_name, x_mean=result.x_mean, x_std=result.x_std,
                    pairs=result.pairs, lookback=result.lookback,
                    use_carry=args.use_carry, vol_horizons=args.vol_horizons,
                    description=args.model_description,
                )

            _JOBS[job_id]["result"] = {
                "portfolio_model_name": portfolio_name,
                "risk_model_name": None,
                "train_sharpe_raw": float(sharpe_ratio(torch.tensor(result.returns_train_unscaled))),
                "train_sharpe_vol_targeted": float(sharpe_ratio(torch.tensor(result.returns_train))),
                "train_sharpe_with_risk": None,
                "val_sharpe_raw": float(sharpe_ratio(torch.tensor(result.returns_val_unscaled))),
                "val_sharpe_vol_targeted": float(sharpe_ratio(torch.tensor(result.returns_val))),
                "val_sharpe_with_risk": None,
                "test_sharpe_raw": float(sharpe_ratio(torch.tensor(result.returns_test_unscaled))),
                "test_sharpe_vol_targeted": float(sharpe_ratio(torch.tensor(result.returns_test))),
                "test_sharpe_with_risk": None,
                "pnl_train": _series_payload(
                    result.dates_train, vol_targeted=np.cumsum(result.returns_train),
                    benchmark=np.cumsum(result.benchmark_returns_train),
                ),
                "pnl_val": _series_payload(
                    result.dates_val, vol_targeted=np.cumsum(result.returns_val),
                    benchmark=np.cumsum(result.benchmark_returns_val),
                ),
                "pnl_test": _series_payload(
                    result.dates_test, vol_targeted=np.cumsum(result.returns_test),
                    benchmark=np.cumsum(result.benchmark_returns_test),
                ),
                "positions_train": _positions_payload(result.dates_train, pairs, result.weights_train),
                "positions_val": _positions_payload(result.dates_val, pairs, result.weights_val),
                "attenuation_train": None,
                "attenuation_val": None,
                "asset_returns": _asset_returns_payload(
                    pairs, result.dates_val, result.next_returns_val, result.dates_train, result.next_returns_train,
                ),
            }

        # The last CAPTURED epoch log line may fall short of the true final
        # epoch (it's only logged every epochs//10 epochs), so snap the bar
        # to 100% explicitly now that training has actually finished.
        progress = _JOBS[job_id]["progress"]
        progress["epoch"] = progress["total_epochs"]
        progress["percent"] = 100.0
        _JOBS[job_id]["status"] = "done"
    except TrainingStopped:
        # Requested via POST /api/train/{job_id}/stop - not an error.
        # Nothing gets saved (locally or to the db): the job ends wherever
        # training happened to be, showing whatever the live interim/log
        # panels already displayed.
        _JOBS[job_id]["status"] = "stopped"
    except Exception as exc:  # noqa: BLE001 - surface any failure to the polling client
        _JOBS[job_id]["status"] = "error"
        _JOBS[job_id]["error"] = str(exc)


@app.post("/api/train")
def start_training(req: TrainRequest) -> dict:
    """Kick off training on a background thread and return immediately -
    training can take from seconds to minutes, too long for one HTTP
    request, so the frontend polls GET /api/train/{job_id} instead.
    """
    job_id = str(uuid.uuid4())
    _JOBS[job_id] = {
        "status": "pending",
        "result": None,
        "error": None,
        "logs": [],
        "progress": {
            "seed_index": 1, "n_seeds": req.n_seeds,
            "epoch": 0, "total_epochs": req.risk_epochs if req.risk_overlay else req.epochs,
            "percent": 0.0,
        },
        "interim": None,  # live in-sample PnL/Sharpe snapshot, updated during training - see _run_training_job
        "stop_requested": False,
    }
    thread = threading.Thread(target=_run_training_job, args=(job_id, req.model_dump()), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.get("/api/train/{job_id}")
def get_training_status(job_id: str) -> dict:
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return {"job_id": job_id, **job}


@app.post("/api/train/{job_id}/stop")
def stop_training(job_id: str) -> dict:
    """Request that a running training job stop as soon as possible - it's
    checked once per epoch (see models/portfolio_lstm.py's _check_stop), so
    it takes effect within a few epochs, not instantly. Cooperative, not a
    thread kill: nothing gets saved for a stopped job (see
    _run_training_job's TrainingStopped handling), and there's no way to
    resume - starting again means a fresh job.
    """
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    if job["status"] not in ("pending", "running"):
        raise HTTPException(status_code=400, detail=f"Job is already {job['status']!r}, nothing to stop")
    job["stop_requested"] = True
    return {"job_id": job_id, "status": "stopping"}


# --------------------------------------------------------------------------
# POST /api/evaluate
# --------------------------------------------------------------------------

class EvaluateRequest(BaseModel):
    """Note: no `pairs` or `lookback` fields - both are properties of the
    trained model itself (restored from its checkpoint, see
    models/portfolio_lstm.py's load_pipeline), not free evaluation
    parameters. Only what's genuinely evaluation-time - how much history to
    fetch, how much of it counts as the scored/"recent" window, and
    reporting-only knobs - is exposed here.
    """

    portfolio_model: str  # a quant.model_registry name, or a local .pt path
    risk_model: str | None = None  # same; omit to evaluate PortfolioLSTM alone
    years: int = 8
    train_frac: float = 0.8
    test_frac: float = 0.1
    target_vol: float = 0.20
    transaction_cost: float = 0.0


def _histogram(values: np.ndarray, bins: int = 30) -> dict:
    """Bin `values` server-side (numpy) so the frontend just renders bars,
    with no statistics logic duplicated in JS."""
    counts, edges = np.histogram(np.asarray(values), bins=bins)
    return {"counts": counts.tolist(), "bin_edges": edges.tolist()}


def _predict_latest_weights(args: Namespace, model, risk_model=None) -> dict[str, float]:
    """Compute the model's recommended weights for the NEXT (as-yet-
    unrealized) day, using the most recent `lookback`-day window of real
    data - distinct from the historical validation-period weights (which
    are for backtesting): this is "run the model to get today's position".

    Uses `model.pairs`/`model.lookback` - restored from the model's own
    checkpoint - rather than `args.pairs`/`args.lookback` (whatever the
    request/UI asked for), so the fetched data, window length, and returned
    weight labels always match what the model was actually trained on.

    When `risk_model` is given, builds its make_risk_sequences()-equivalent
    input for this single live window: an extra `risk_model.rolling_window
    - 1` days of history beyond `lookback` are fetched so the SAME
    per-asset weighted-PnL + rolling-moments features the model was trained
    on (see models/risk_lstm.py's make_risk_sequences) can be computed here
    too, using TODAY's own decided weight (consistent with how each
    training sample uses its own decision weight, not a different day's).

    When `model.use_prev_weight` is True, tomorrow's decision also needs
    today's ACTUAL held position as an input (see PortfolioLSTM.forward) -
    not available directly, so a short PREV_WEIGHT_WARMUP_DAYS-day
    recurrence is run first (forward_sequence, starting flat) purely to
    arrive at a plausible current position; only its LAST output is used
    (prev is detached between steps - see forward_sequence - so this
    warmup doesn't need to be long to be accurate, just enough for the
    detached recurrence to reach a steady position).
    """
    pairs = model.pairs
    lookback = model.lookback
    # model may live on an accelerator (MPS/CUDA - see
    # models/portfolio_lstm.get_device); every tensor built below is moved
    # to match before being passed to model()/risk_model() - this is a
    # single tiny inference call, so there's no real GPU benefit, but it
    # must still land on whatever device the (possibly GPU-trained) model
    # actually lives on or the forward pass would raise a device-mismatch error.
    device = next(model.parameters()).device
    use_carry = getattr(model, "use_carry", False)
    vol_horizons = getattr(model, "vol_horizons", [])
    n_channels = 1 + int(use_carry) + len(vol_horizons)
    extra_days = (risk_model.rolling_window - 1) if risk_model is not None else 0
    prev_weight_warmup = PREV_WEIGHT_WARMUP_DAYS if getattr(model, "use_prev_weight", False) else 0
    min_days = lookback + extra_days + prev_weight_warmup

    # "CASH" (see has_cash/_prepare_data) has no real ticker - has no price
    # history to fetch. Strip it before load_close_prices, then add it back
    # as a constant-return column (cash_return isn't persisted on the model
    # checkpoint - see DEFAULT_CONFIG's docstring on cash_return - so, like
    # _prepare_data, this trusts the CURRENT request's args.cash_return,
    # not something baked in at training time), reordered to match
    # model.pairs exactly.
    real_pairs = [p for p in pairs if p != "CASH"]
    prices = load_close_prices(real_pairs, years=args.years)
    returns = to_log_returns(prices)
    if "CASH" in pairs:
        returns = returns.copy()
        returns["CASH"] = getattr(args, "cash_return", 0.0)
        returns = returns[pairs]
    if len(returns) < min_days:
        raise ValueError(
            f"Only {len(returns)} days of history available for {pairs}, but this model needs "
            f"the most recent {min_days} days - increase 'years' to fetch more history."
        )
    last_window_raw = returns.to_numpy(dtype=np.float32)[-lookback:]  # (lookback, n_assets) - raw returns only

    if n_channels > 1:
        from models.portfolio_lstm import build_feature_dataframe

        feature_returns = build_feature_dataframe(returns, pairs, use_carry, vol_horizons, args.years)
    else:
        feature_returns = returns
    last_window_features = feature_returns.to_numpy(dtype=np.float32)[-lookback:]  # (lookback, n_assets * n_channels)

    X = torch.tensor((last_window_features - model.x_mean) / model.x_std, device=device).unsqueeze(0)  # (1, lookback, n_assets * n_channels)
    X_raw = torch.tensor(last_window_raw, device=device).unsqueeze(0)  # raw returns only - for vol-targeting/risk features

    model.eval()
    with torch.no_grad():
        prev_weight = None
        if prev_weight_warmup > 0:
            # `lookback + prev_weight_warmup - 1` days of FEATURES, ending
            # the day BEFORE today's own window ends, sliding a lookback-day
            # window step=1 across them gives exactly `prev_weight_warmup`
            # sequential decisions whose LAST one ends exactly at
            # "yesterday" - i.e. the decision immediately preceding today's.
            warmup_raw = feature_returns.to_numpy(dtype=np.float32)[-(lookback + prev_weight_warmup) : -1]
            warmup_windows = torch.tensor(warmup_raw).unfold(0, lookback, 1).permute(0, 2, 1)  # (prev_weight_warmup, lookback, n_assets * n_channels)
            warmup_standardized = (warmup_windows.numpy() - model.x_mean) / model.x_std
            warmup_sequence = model.forward_sequence(torch.tensor(warmup_standardized, dtype=X.dtype, device=device))
            prev_weight = warmup_sequence[-1].detach()  # yesterday's decision = today's ACTUAL held position

        raw_weights = model(X, prev_weight.unsqueeze(0) if prev_weight is not None else None)
        weights = scale_weights_to_target_vol(
            raw_weights, X_raw, args.target_vol, max_leverage=args.max_leverage,
            covariance_estimator=getattr(args, "covariance_estimator", "sample"),
            ewma_lambda=getattr(args, "ewma_lambda", 0.94),
        )
        if risk_model is not None:
            from models.risk_lstm import cross_sectional_features, rolling_moments

            extended_raw = returns.to_numpy(dtype=np.float32)[-min_days:]  # (lookback + rolling_window - 1, n_assets)
            extended = torch.tensor(extended_raw, device=device).unsqueeze(0)  # (1, extended_len, n_assets)
            weighted = extended * weights.unsqueeze(1)  # (1, extended_len, n_assets) - TODAY's weight, broadcast
            moments = rolling_moments(weighted, risk_model.rolling_window)      # (1, lookback, 3*n_assets)
            aligned_returns = weighted[:, risk_model.rolling_window - 1:, :]    # (1, lookback, n_assets)
            features = torch.cat([aligned_returns, moments], dim=-1)           # (1, lookback, 4*n_assets)
            if getattr(risk_model, "use_cross_sectional", False):
                # Same RAW (unweighted) extended window used for moments'
                # weighted series - see cross_sectional_features' docstring
                # for why correlation structure uses raw, not weighted, returns.
                # Computed on CPU regardless of `device`: it uses
                # torch.linalg.eigvalsh, which MPS does not implement
                # (confirmed: "aten::_linalg_eigh.eigenvalues... not
                # currently implemented for the MPS device") - then moved
                # back to match `features` before concatenating.
                cross_sectional = cross_sectional_features(extended.cpu(), risk_model.rolling_window).to(device)  # (1, lookback, 3)
                features = torch.cat([features, cross_sectional], dim=-1)      # (1, lookback, 4*n_assets + 3)

            risk_model.eval()
            attenuation = risk_model(features, weights)
            weights = weights * attenuation

    return {pair: float(w) for pair, w in zip(pairs, weights[0].cpu().numpy())}


@app.post("/api/evaluate")
def evaluate(req: EvaluateRequest) -> dict:
    """Load a previously-trained model (by quant.model_registry name or
    local path) and run it purely for inference - no training happens.
    Returns everything the Evaluation view plots:
      - pnl: cumulative baseline ("risk-weighted", pre-attenuation) vs
        with-risk-overlay vs with-risk-overlay-and-transaction-costs,
        out-of-sample only.
      - sharpe: matching Sharpe ratios for each series.
      - histograms: return distributions for baseline and with-risk.
      - positions: per-asset portfolio weight over time, out-of-sample.
      - attenuation: per-asset risk-overlay attenuation over time (only
        present when a risk_model was given).
      - latest_weights: the model's recommended allocation for the next
        (not-yet-realized) day, using the freshest available data.

    `pairs` and `lookback` are deliberately NOT accepted from `req` - both
    are recovered from the loaded model's own checkpoint (see
    models/portfolio_lstm.py's load_pipeline), since they're properties of
    the trained model, not free evaluation parameters. Only `years` (how
    much history to fetch) is caller-controlled; if it's not enough to
    cover the model's own sequence length, load_pipeline/_predict_latest_weights
    raise a ValueError, turned into a 400 below.
    """
    args = Namespace(**{
        **DEFAULT_CONFIG,
        "pairs": None,
        "lookback": None,
        "years": req.years,
        "train_frac": req.train_frac,
        "test_frac": req.test_frac,
        "target_vol": req.target_vol,
        "load_portfolio": req.portfolio_model,
        "load_risk": req.risk_model,
    })

    try:
        if req.risk_model:
            from models.risk_lstm import run_pipeline_multi_seed as run_risk_overlay_pipeline

            result = run_risk_overlay_pipeline(args)  # load_portfolio+load_risk set -> pure inference, no training
            pairs = result.portfolio_result.pairs  # the model's own stored pairs, restored by load_pipeline

            # result.weights_val/next_returns_val/dates_val are ALIGNED to
            # the risk model's own valid range (make_risk_sequences drops
            # the first `risk_rolling_window - 1` samples) - NOT
            # result.portfolio_result's own (longer) arrays, which would
            # silently misalign against result.attenuation_val's length.
            final_weights_val = result.weights_val * result.attenuation_val
            net_returns_val = apply_transaction_costs(
                final_weights_val, result.returns_val_scaled, req.transaction_cost,
            )

            latest_weights = _predict_latest_weights(
                args, result.portfolio_result.model, result.risk_model,
            )

            return {
                "pairs": pairs,
                "latest_weights": latest_weights,
                "pnl": _series_payload(
                    result.dates_val,
                    baseline=np.cumsum(result.returns_val_raw),
                    with_risk=np.cumsum(result.returns_val_scaled),
                    with_risk_and_costs=np.cumsum(net_returns_val),
                ),
                "sharpe": {
                    "baseline": float(sharpe_ratio(torch.tensor(result.returns_val_raw))),
                    "with_risk": float(sharpe_ratio(torch.tensor(result.returns_val_scaled))),
                    "with_risk_and_costs": float(sharpe_ratio(torch.tensor(net_returns_val))),
                },
                "histograms": {
                    "baseline": _histogram(result.returns_val_raw),
                    "with_risk": _histogram(result.returns_val_scaled),
                },
                "positions": _positions_payload(result.dates_val, pairs, result.weights_val),
                "attenuation": _positions_payload(result.dates_val, pairs, result.attenuation_val),
                "asset_returns": _asset_returns_payload(pairs, result.dates_val, result.next_returns_val),
            }

        from models.portfolio_lstm import run_pipeline_multi_seed

        result = run_pipeline_multi_seed(args)  # load_portfolio set -> pure inference, no training
        pairs = result.pairs
        latest_weights = _predict_latest_weights(args, result.model)

        return {
            "pairs": pairs,
            "latest_weights": latest_weights,
            "pnl": _series_payload(result.dates_val, baseline=np.cumsum(result.returns_val)),
            "sharpe": {"baseline": float(sharpe_ratio(torch.tensor(result.returns_val)))},
            "histograms": {"baseline": _histogram(result.returns_val)},
            "positions": _positions_payload(result.dates_val, pairs, result.weights_val),
            "attenuation": None,
            "asset_returns": _asset_returns_payload(pairs, result.dates_val, result.next_returns_val),
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Model not found: {exc}") from exc
    except ValueError as exc:
        # e.g. "not enough history for this model's sequence length" - a
        # clear, actionable error from load_pipeline/_predict_latest_weights,
        # not a 500.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
