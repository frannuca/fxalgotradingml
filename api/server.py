"""FastAPI backend for the FX direction-prediction pipeline.

Thin HTTP wrapper around models/portfolio_lstm.py - no business logic
lives here, just request/response shaping so the React frontend
(frontend/) has a JSON API to call.

Endpoints
---------
GET  /api/pairs             - available FX pair tickers, for pair pickers
POST /api/quotes/refresh     - download + upsert the latest close prices
GET  /api/models             - list models saved in quant.model_registry
POST /api/train               - kick off a training run (background job)
GET  /api/train/{job_id}     - poll a training job's status/result
POST /api/train/{job_id}/stop - request an in-progress job stop early
POST /api/train/{job_id}/save-best - save whichever checkpoint is
                                 CURRENTLY best (mid-run or after) to
                                 quant.model_registry under a new name,
                                 and return a train/val/test loss/hit-rate/
                                 Sharpe summary
POST /api/evaluate            - load a model by name and run inference:
                                 returns per-asset hit rate, confusion
                                 matrices, cumulative-return series (with
                                 per-day hit/miss), and the model's
                                 predicted probabilities for the next day.
POST /api/evaluate/{eval_id}/portfolio - recompute portfolio PnL/annual
                                 Sharpe/today's position for a DIFFERENT
                                 neutral band than the one POST /api/evaluate
                                 used, from cached raw probabilities/returns
                                 (no re-fetch, no model re-run).

Run with (from the repo root):
    uvicorn api.server:app --reload --port 8000
"""

from __future__ import annotations

import contextvars
import io
import logging
import os
import re
import threading
import uuid
from argparse import Namespace
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import models.risk_engine as risk_engine_module
from data.db import get_connection
from data.fx_downloader import MAJOR_FX_PAIRS
from models.portfolio_lstm import (
    DEFAULT_BANDPASS_ORDER,
    DEFAULT_CONFIG,
    DEFAULT_FEATURES,
    PredictionModel,
    TrainingStopped,
    _epoch_report_callback,
    _stop_check_callback,
    apply_neutral_band,
    build_feature_dataframe,
    confusion_matrix_metrics,
    load_close_prices,
    load_prediction_model_auto,
    logger,
    prediction_model_name,
    probit,
    resolve_signal_bounds,
    summarize_checkpoint,
    to_log_returns,
)
from models.portfolio_pnl import (
    DEFAULT_COST_BPS,
    DEFAULT_TARGET_VOL,
    annual_sharpe_table,
    compute_portfolio,
    latest_position,
)

app = FastAPI(title="FX Direction Prediction API")
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
    frontend's model picker (for the Evaluation and Continue Training
    views) reads this.

    Includes each model's own `pairs` and `lookback`, decoded from its
    checkpoint blob, so the frontend can auto-select the FX pairs and
    display the sequence length a chosen model was actually trained on,
    instead of requiring the user to supply values that must match it
    exactly - see api/server.py's evaluate()/models/portfolio_lstm.py's
    load_pipeline() for why the model's own pairs/lookback are
    authoritative over anything a caller might guess.

    Also includes every other ARCHITECTURE-defining property (n_channels,
    hidden_size, num_layers, dropout, n_attn_heads, cross_pairs, features,
    cma_windows, bandpass_windows, bandpass_order) plus every other
    persisted, non-architecture model property (direction_horizon,
    rolling_stats_window, neutral_band, target_vol, signal_range) - not
    needed by the Evaluation view to actually RUN inference (it recovers
    them server-side via the model's own checkpoint - see load_pipeline),
    but both the Evaluation and Continue Training views' whole point in
    showing this is transparency: a chosen model's full configuration
    READ-ONLY (see /api/train's `continue_from`, which re-derives the
    architecture fields from the checkpoint itself server-side too - this
    payload is for DISPLAY only, never trusted as a source of truth for
    training).
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
        config = checkpoint.get("config", {})
        result.append({
            "name": name,
            "model_type": model_type,
            "description": description,
            "created_at": created_at.isoformat(),
            "updated_at": updated_at.isoformat(),
            "size_bytes": len(blob),
            "pairs": checkpoint.get("pairs"),
            "lookback": checkpoint.get("lookback"),
            "n_channels": config.get("n_channels"),
            "hidden_size": config.get("hidden_size"),
            "num_layers": config.get("num_layers"),
            "dropout": config.get("dropout"),
            "n_attn_heads": config.get("n_attn_heads"),
            "cross_pairs": config.get("cross_pairs"),
            "features": checkpoint.get("features"),
            "cma_windows": checkpoint.get("cma_windows"),
            "bandpass_windows": checkpoint.get("bandpass_windows"),
            "bandpass_order": checkpoint.get("bandpass_order"),
            "direction_horizon": checkpoint.get("direction_horizon", 5),
            "rolling_stats_window": checkpoint.get("rolling_stats_window", 20),
            "neutral_band": checkpoint.get("neutral_band"),
            "target_vol": checkpoint.get("target_vol"),
            "signal_range": checkpoint.get("signal_range", {}),
            "risk_engine": _risk_engine_summary(checkpoint.get("risk_engine")),
        })
    return result


def _risk_engine_summary(risk_engine_checkpoint: dict | None) -> dict | None:
    """DISPLAY-only summary of a model's attached RiskEngine (see
    models/risk_engine.py), if any - None for a model with none. Used by
    list_models() (so the frontend can show whether a model already has a
    risk engine, and its own hyperparameters, without decoding the full
    nested checkpoint itself) - never used server-side as a source of
    truth for training (recompute_portfolio/evaluate() always reconstruct
    the actual RiskEngine via PredictionModel._from_checkpoint instead).
    """
    if risk_engine_checkpoint is None:
        return None
    return {
        "risk_lookback": risk_engine_checkpoint.get("risk_lookback"),
        "risk_cma_windows": risk_engine_checkpoint.get("risk_cma_windows", []),
        "risk_bandpass_windows": risk_engine_checkpoint.get("risk_bandpass_windows", []),
        "risk_bandpass_order": risk_engine_checkpoint.get("risk_bandpass_order"),
        "min_risk_att": risk_engine_checkpoint.get("min_risk_att"),
        "max_risk_att": risk_engine_checkpoint.get("max_risk_att"),
    }


# --------------------------------------------------------------------------
# POST /api/train  +  GET /api/train/{job_id}
# --------------------------------------------------------------------------

# In-memory job store - fine for a local, single-process dev server; a
# multi-worker/production deployment would need a real job queue instead.
# Every value in here must stay JSON-serializable: GET /api/train/{job_id}
# (see get_training_status) returns `{"job_id": job_id, **job}` - anything
# non-serializable (a torch model, a _PreparedData) stored under a job_id
# here would crash EVERY subsequent status poll with a Pydantic
# serialization error, not just whichever endpoint actually needed it. Raw
# model/data/args/best_state references (see POST .../save-best) live in
# the SEPARATE _BEST_CHECKPOINT_STATE dict below instead, precisely so
# they can never leak into this one's wholesale dict-spread.
_JOBS: dict[str, dict[str, Any]] = {}

# job_id -> {"model", "data", "args", "epoch", "best_state"} - the raw
# (non-JSON-serializable) references POST /api/train/{job_id}/save-best
# needs to build a snapshot on demand (see _run_training_job's
# on_best_checkpoint callback and models/portfolio_lstm.py's
# summarize_checkpoint). Deliberately NOT part of _JOBS - see its own
# comment on why.
_BEST_CHECKPOINT_STATE: dict[str, dict[str, Any]] = {}

# eval_id -> everything POST /api/evaluate/{eval_id}/portfolio needs to
# recompute the portfolio PnL/annual Sharpe/today's position for a
# DIFFERENT neutral band, without re-fetching data or re-running the model
# (see evaluate()'s own comment on why the portfolio can't just use
# whatever band the frontend's live-adjustable slider is currently at -
# this cache is what lets the Evaluation view's slider actually move it).
# Fine for a local, single-process dev server - see _JOBS's own comment on
# the same tradeoff; no eviction.
_EVAL_CACHE: dict[str, dict[str, Any]] = {}

# Which job_id (if any) is training on the CURRENT thread - contextvars are
# thread-local, and each training job runs its own dedicated background
# thread (see start_training), so setting this once at the top of
# _run_training_job keeps every subsequent logger.info(...) call inside
# that thread correctly attributed to that job, with no cross-job mixing.
_current_job_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_job_id", default=None)

# Training runs every asset's LSTM fully independently (own optimizer, own
# loss - see models/portfolio_lstm.py's train_prediction_model) but on a
# SHARED epoch counter, one phase per (seed, bce_weight) combo (see
# run_pipeline_multi_seed - every bce_weight value is swept under every
# seed) - captured from its own log lines ("--- restart %d/%d (seed=%d),
# bce_weight %s (%d/%d) ---" and "epoch %d/%d - train loss ...") so the
# progress bar can track progress across the WHOLE sweep, not just seeds.
_EPOCH_RE = re.compile(r"epoch (\d+)/(\d+) - train")
_RESTART_RE = re.compile(r"restart (\d+)/(\d+) \(seed=\d+\), bce_weight \S+ \((\d+)/(\d+)\)")
_MAX_LOG_LINES = 500


class _JobLogHandler(logging.Handler):
    """Captures the training pipeline's EXISTING logger.info(...) calls
    (models/portfolio_lstm.py's per-epoch and per-restart progress lines)
    into the current job's state - so the frontend's polling
    GET /api/train/{job_id} can show a live log window and a progress bar,
    without any changes to the training loops themselves (they already
    log exactly this, for the CLI's benefit).
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
            "progress", {
                "seed_index": 1, "n_seeds": 1, "lambda_index": 1, "n_lambdas": 1,
                "epoch": 0, "total_epochs": 1, "percent": 0.0,
            },
        )

        restart_match = _RESTART_RE.search(message)
        if restart_match:
            progress["seed_index"] = int(restart_match.group(1))
            progress["n_seeds"] = int(restart_match.group(2))
            progress["lambda_index"] = int(restart_match.group(3))
            progress["n_lambdas"] = int(restart_match.group(4))
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
        # Total units of work is every (seed, bce_weight) combo, not just
        # seeds - a sweep of N lambdas under each seed is N times the work
        # a single value would be, and the bar must reflect that instead
        # of resetting to the SAME per-seed share for every lambda (which
        # would make it visibly jump backward each time a new lambda
        # starts).
        n_seeds = max(progress["n_seeds"], 1)
        n_lambdas = max(progress["n_lambdas"], 1)
        total_epochs = max(progress["total_epochs"], 1)
        total_combos = n_seeds * n_lambdas
        completed_combos = (progress["seed_index"] - 1) * n_lambdas + (progress["lambda_index"] - 1)
        current_combo_fraction = (progress["epoch"] / total_epochs) / total_combos
        progress["percent"] = round((completed_combos / total_combos + current_combo_fraction) * 100, 1)


# Attach once, to the shared "models" ancestor logger - INFO records from
# models.portfolio_lstm's own logger propagate up to it by default, so
# this single handler sees every training job's progress.
_job_log_handler = _JobLogHandler()
_job_log_handler.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger("models").addHandler(_job_log_handler)
logging.getLogger("models").setLevel(logging.INFO)


class TrainRequest(BaseModel):
    # A quant.model_registry name or local .pt path (see
    # load_prediction_model_auto) to CONTINUE TRAINING from instead of a
    # fresh random init - see models/portfolio_lstm.py's continue_training.
    # When set: `pairs`/`lookback`/`hidden_size`/`num_layers`/`dropout`/
    # `n_attn_heads`/`cross_pairs`/`features`/`cma_windows`/
    # `bandpass_windows`/`bandpass_order` below are ALL ignored (the
    # server recovers them from the base model's own checkpoint instead -
    # every architecture-defining property must match exactly, so none of
    # them are free parameters here) - only training-behavior fields
    # (epochs, lr, weight_decay, bce_weight, sharpe_weight, sharpe_window,
    # direction_horizon, checkpoint_metric, neutral_band, target_vol,
    # signal_range) and the data window (years, cutoff_date, train_frac,
    # test_frac, device) actually apply. `pairs` above may be sent as `[]`
    # in this mode.
    continue_from: str | None = None
    pairs: list[str]
    lookback: int = 30
    years: int = 8
    # Caps every fetched date range (prices, carry) at this date - never
    # later, regardless of what's since landed in Postgres (e.g. via
    # /api/quotes/refresh) - guaranteeing training/validation/test never
    # see a day after it (see models/portfolio_lstm.py's
    # _resolve_cutoff_date). None (the default) means "no cutoff": use all
    # data available up to today. An ISO "YYYY-MM-DD" string walk-forward-
    # backtests as of that historical date instead.
    cutoff_date: str | None = None
    train_frac: float = 0.8
    test_frac: float = 0.1
    direction_horizon: int = 5  # forward days the z-score label (see make_sequences) looks
    rolling_stats_window: int = 20  # trailing window for the "vol" input feature + z-score label normalization
    # Which per-pair input channels to build (see models/portfolio_lstm.py's
    # FEATURE_CATALOG: "log_return", "vol", "carry", "cma", "bandpass").
    # "skew"/"kurt" remain in FEATURE_CATALOG for backward compatibility
    # with OLD checkpoints that still list them (build_feature_dataframe
    # still knows how to build them), but are no longer offered as a
    # default or a frontend option - they're risk-engine-exclusive inputs
    # now (see models/risk_engine.py, `risk_rolling_stats_window` below).
    features: list[str] = ["log_return", "vol"]
    cma_windows: list[list[int]] = []  # [[short, long], ...] - only used if "cma" is in `features`
    # (short, long) period-in-days pairs for a causal Butterworth band-pass
    # filter (see build_feature_dataframe's own bandpass_windows param) - a
    # faster-reacting trend-strength alternative to "cma"; only used if
    # "bandpass" is in `features`.
    bandpass_windows: list[list[int]] = []
    bandpass_order: int = DEFAULT_BANDPASS_ORDER
    # Architecture: N independent per-asset LSTMs, each followed by a
    # CAUSAL self-attention layer over the time axis (see AssetLSTM/
    # PredictionModel) - no weights shared between assets. By DEFAULT
    # every pair's LSTM sees ONLY its own features - fully independent, no
    # cross-asset mixing. `cross_pairs` (a {pair: [other_pair, ...]} dict)
    # opts specific pairs INTO also seeing specific other pairs' full
    # feature blocks (own features are always included regardless).
    # Fully deterministic - no NoisyNet head, no input-noise regularization.
    cross_pairs: dict[str, list[str]] = {}
    hidden_size: int = 16
    num_layers: int = 1
    dropout: float = 0.1
    n_attn_heads: int = 4  # attention heads per asset's causal self-attention layer - hidden_size must divide evenly
    # Training (see train_prediction_model): every asset's LSTM trains
    # fully independently - its own optimizer, its own loss.
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-4
    # Weight of the BCE direction term (see direction_bce) - the
    # anti-mean-collapse term. Either a single value, or a list to SWEEP
    # (e.g. [1.0, 1.5, 1.75, 2.0, 3.0]) - see run_pipeline_multi_seed,
    # which trains every value under every seed and keeps whichever
    # validated best, per seed and then overall.
    bce_weight: float | list[float] = 1.0
    # Optional COMPLEMENTARY training objective (see train_prediction_model's
    # own docstring on its "ONE combined step per epoch") - 0 (default)
    # disables it, training is then unchanged. > 0 adds a joint portfolio-
    # Sharpe term to the SAME combined loss as NLL+BCE: risk-parity weight
    # x this model's own probability signal, scaled to target_vol below,
    # maximizing an averaged Sharpe over NON-OVERLAPPING sharpe_window-day
    # chunks (see _non_overlapping_sharpe_torch). Unlike every other
    # training parameter, > 0 means one asset's training depends on every
    # other asset's current output (the target-vol scaling is joint).
    sharpe_weight: float = 0.0
    sharpe_window: int = 20
    # Which per-epoch VALIDATION metric selects each asset's own restored
    # checkpoint (see models/portfolio_lstm.py's train_prediction_model
    # docstring on checkpoint_metric) - independent of sharpe_weight above,
    # which controls what TRAINS the weights, not which epoch's weights
    # get kept. "val_loss" (default), "hit_rate", or "sharpe".
    checkpoint_metric: str = "val_loss"
    neutral_band: float = 0.05  # abstention half-width around p=0.5 (see apply_neutral_band); 0 disables
    # Annualized volatility the evaluation-mode portfolio PnL calculator
    # (models/portfolio_pnl.py) scales this model's positions to - a
    # property of the model, persisted in its checkpoint alongside
    # neutral_band, not a free evaluation-time parameter.
    target_vol: float = DEFAULT_TARGET_VOL
    # Per-asset (min, max) bounds the decision-day probability is linearly
    # mapped into before multiplying the risk-parity baseline weight (see
    # models/portfolio_lstm.py's resolve_signal_bounds and
    # models/portfolio_pnl.py's compute_portfolio) - a {pair: [min, max]}
    # dict, persisted in the checkpoint alongside target_vol/neutral_band.
    # A pair missing from this dict defaults to (-1, 1) - the original
    # fixed `(p - 0.5) * 2` signed-direction map. Narrowing the range
    # changes what the value MEANS, not just its scale: e.g.
    # {"EURUSD": [0.0, 1.0]} makes EURUSD long-only (a [0, 1] factor on the
    # risk-parity weight, never negated) - an abstained day (p=0.5) then
    # sizes a HALF-SIZE position rather than flat, since 0 is no longer the
    # range's midpoint.
    signal_range: dict[str, list[float]] = {}
    # Linear transaction cost (basis points per unit of daily position
    # change) charged in every reported portfolio PnL/Sharpe AND inside the
    # sharpe_weight training objective - see models/portfolio_pnl.py.
    cost_bps: float = DEFAULT_COST_BPS
    n_seeds: int = 1
    device: str = "auto"  # "auto" (Metal/MPS on Apple Silicon, else CUDA, else CPU), "cpu", "mps", or "cuda" - see get_device
    save_db: bool = True
    model_description: str = ""

    # --- Optional risk-engine second phase (see models/risk_engine.py) ---
    # When True, _run_training_job continues STRAIGHT into training a NEW
    # RiskEngine on top of THIS run's own just-trained (frozen) prediction
    # model, in the SAME job/thread, right after phase 1 finishes - no
    # separate save/reload round trip, no separate job submission. The
    # bundled result (predictor + risk engine together, under ONE model
    # name) is exactly what a separate POST /api/train-risk-engine call
    # would produce, just concatenated into one optimization run with
    # phase-aware progress reporting (see _run_training_job's own
    # `progress["phase"]`). POST /api/train-risk-engine (TrainRiskEngineRequest)
    # remains available separately, for retrofitting a risk engine onto an
    # ALREADY-SAVED model without retraining its predictor.
    train_risk_engine: bool = False
    risk_lookback: int = risk_engine_module.DEFAULT_RISK_LOOKBACK
    # Trailing window (days) for the risk engine's OWN skew/kurtosis
    # computation - see models/risk_engine.py's DEFAULT_RISK_ROLLING_STATS_WINDOW;
    # independent of `rolling_stats_window` above, which only affects the
    # per-asset predictor's own "vol" feature now (skew/kurt are
    # risk-engine-exclusive inputs - see `features` above, which no longer
    # offers them).
    risk_rolling_stats_window: int = risk_engine_module.DEFAULT_RISK_ROLLING_STATS_WINDOW
    risk_cma_windows: list[list[int]] = []
    risk_bandpass_windows: list[list[int]] = []
    risk_bandpass_order: int = DEFAULT_BANDPASS_ORDER
    min_risk_att: float = risk_engine_module.DEFAULT_MIN_RISK_ATT
    max_risk_att: float = risk_engine_module.DEFAULT_MAX_RISK_ATT
    risk_hidden_size: int = risk_engine_module.DEFAULT_RISK_HIDDEN_SIZE
    risk_num_layers: int = risk_engine_module.DEFAULT_RISK_NUM_LAYERS
    risk_dropout: float = risk_engine_module.DEFAULT_RISK_DROPOUT
    risk_n_attn_heads: int = risk_engine_module.DEFAULT_RISK_N_ATTN_HEADS
    risk_epochs: int = 100
    risk_lr: float = 1e-3
    risk_weight_decay: float = 0.0
    risk_sortino_window: int = risk_engine_module.DEFAULT_RISK_SORTINO_WINDOW
    full_exposure_penalty: float = risk_engine_module.DEFAULT_FULL_EXPOSURE_PENALTY


def _hit_rate_payload(pairs: list[str], hit_rate: np.ndarray) -> dict:
    """Build a {<pair>: rate, ...} dict from a (n_assets,) array - one
    scalar directional hit rate per asset (see PredictionResult's
    docstring on hit_rate_train)."""
    hit_rate = np.asarray(hit_rate)
    return {pair: float(hit_rate[i]) for i, pair in enumerate(pairs)}


def _confusion_matrix_payload(
    pairs: list[str], probabilities: np.ndarray, labels: np.ndarray, neutral_band: float = 0.0,
) -> dict:
    """Build a {<pair>: {tp, fp, tn, fn, abstained, coverage, accuracy,
    precision, recall, specificity, f1}, ...} dict - the full per-asset
    confusion-matrix breakdown (see models/portfolio_lstm.py's
    confusion_matrix_metrics), for the Training/Evaluation views'
    confusion-matrix table. With a neutral band, tp/fp/tn/fn and every
    derived metric cover DECIDED samples only; `abstained` counts the
    no-call days and `coverage` their complement's share, so the frontend
    can show accuracy and coverage side by side.
    """
    metrics = confusion_matrix_metrics(probabilities, labels, neutral_band=neutral_band)
    return {
        pair: {
            "tp": int(metrics["tp"][i]), "fp": int(metrics["fp"][i]),
            "tn": int(metrics["tn"][i]), "fn": int(metrics["fn"][i]),
            "abstained": int(metrics["abstained"][i]),
            "coverage": float(metrics["coverage"][i]),
            "accuracy": float(metrics["accuracy"][i]),
            "precision": float(metrics["precision"][i]),
            "recall": float(metrics["recall"][i]),
            "specificity": float(metrics["specificity"][i]),
            "f1": float(metrics["f1"][i]),
        }
        for i, pair in enumerate(pairs)
    }


def _cumulative_return_payload(dates, pairs: list[str], next_returns: np.ndarray) -> dict:
    """Build a {dates: [...], <pair>: {cumulative: [...]}, ...} dict: each
    asset's own cumulative (single-day-ahead) log-return path. Deliberately
    does NOT compute hit/abstained here - that requires a neutral band (see
    apply_neutral_band), which is pure postprocessing (see
    evaluate_prediction_model's docstring), not baked into anything at
    evaluation time - the frontend computes hit/abstained itself, from
    _probability_payload's raw probability + realized label, against
    whatever band the user currently has selected (so it can recolor this
    same chart for a different band without a new request - see
    frontend/src/metrics.js).
    """
    payload: dict[str, Any] = {"dates": [str(d.date()) if hasattr(d, "date") else str(d) for d in dates]}
    cumulative = np.cumsum(np.asarray(next_returns), axis=0)
    for i, pair in enumerate(pairs):
        payload[pair] = {"cumulative": cumulative[:, i].tolist()}
    return payload


def _probability_payload(dates, pairs: list[str], probabilities: np.ndarray, direction_labels: np.ndarray) -> dict:
    """Build a {dates: [...], <pair>: {probability: [...], label: [...]}, ...}
    dict: each asset's own predicted probability path (RAW - see
    evaluate_prediction_model's docstring, NOT neutral-band-snapped)
    alongside the realized direction label for the SAME date - together,
    everything a caller needs to derive hit/miss/abstained or a full
    confusion matrix for ANY neutral band, without another request (see
    frontend/src/metrics.js, which is what the Training/Evaluation pages'
    neutral-band control actually recomputes against client-side).
    """
    payload: dict[str, Any] = {"dates": [str(d.date()) if hasattr(d, "date") else str(d) for d in dates]}
    probabilities = np.asarray(probabilities)
    direction_labels = np.asarray(direction_labels)
    for i, pair in enumerate(pairs):
        payload[pair] = {"probability": probabilities[:, i].tolist(), "label": direction_labels[:, i].tolist()}
    return payload


def _distribution_payload(pairs: list[str], z_labels: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> dict:
    """Build a {<pair>: {actual: [...], forecasted: [...]}, ...} dict for
    the "forecasted vs actual" distribution histograms: `actual` is the
    realized decision-day z-score (data.z_labels_*, see PredictionResult);
    `forecasted` draws ONE random sample from EACH row's own model-implied
    N(mu_i, sigma_i) (sigma already calibrated by sigma_hat, see
    evaluate_prediction_model) - the same length/pairing as `actual`, so a
    caller can histogram both and compare shape/spread/skew directly. If
    the model's predictive distributions are well-calibrated, the two
    histograms should look statistically similar even though no individual
    pair of values need match.
    """
    z_labels = np.asarray(z_labels)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    forecasted = np.random.default_rng().normal(mu, sigma)
    return {
        pair: {"actual": z_labels[:, i].tolist(), "forecasted": forecasted[:, i].tolist()}
        for i, pair in enumerate(pairs)
    }


def _portfolio_payload(dates, pairs: list[str], portfolio: dict) -> dict:
    """Build a {dates: [...], <pair>: {position_modulated: [...],
    position_baseline: [...], cumulative_pnl: [...]}, ...,
    cumulative_pnl_modulated: [...], cumulative_pnl_baseline: [...]} dict -
    the Evaluation view's portfolio PnL chart (see
    models/portfolio_pnl.py's compute_portfolio).
    `position_modulated` is the risk-parity weight times the probability
    signal `(p - 0.5) * 2`, smoothed over the model's own direction_horizon
    and scaled to the model's own persisted target_vol; `position_baseline`
    is the SAME risk-parity weights with no signal applied (unmodulated),
    scaled to the same target_vol, for a like-for-like comparison. NaN
    entries (not enough trailing history yet to size a position) are sent
    as `null` - the frontend chart simply skips them. Each pair's own
    `cumulative_pnl` (MODULATED strategy only) is that asset's own
    `position_modulated * next_return`, cumulatively summed - these sum
    ACROSS pairs, per day, to top-level `cumulative_pnl_modulated` (the
    whole book), so the Evaluation page can plot per-asset contributions
    alongside the book total in the same chart.

    If `portfolio` has a `"positions_risk_attenuated"` key (see
    compute_portfolio's own `attenuation` param - only present when a
    RiskEngine is attached, models/risk_engine.py), the SAME per-pair
    `position_risk_attenuated`/`cumulative_pnl_risk_attenuated` and
    top-level `cumulative_pnl_risk_attenuated` are included alongside the
    existing modulated/baseline series - never REPLACING them, so the
    Evaluation view can plot all three lines together.
    """
    payload: dict[str, Any] = {"dates": [str(d.date()) if hasattr(d, "date") else str(d) for d in dates]}
    has_risk = "positions_risk_attenuated" in portfolio
    for i, pair in enumerate(pairs):
        payload[pair] = {
            "position_modulated": [None if np.isnan(v) else float(v) for v in portfolio["positions_modulated"][:, i]],
            "position_baseline": [None if np.isnan(v) else float(v) for v in portfolio["positions_baseline"][:, i]],
            "cumulative_pnl": portfolio["cumulative_pnl_per_asset_modulated"][:, i].tolist(),
        }
        if has_risk:
            payload[pair]["position_risk_attenuated"] = [
                None if np.isnan(v) else float(v) for v in portfolio["positions_risk_attenuated"][:, i]
            ]
            payload[pair]["cumulative_pnl_risk_attenuated"] = portfolio["cumulative_pnl_per_asset_risk_attenuated"][:, i].tolist()
    payload["cumulative_pnl_modulated"] = portfolio["cumulative_pnl_modulated"].tolist()
    payload["cumulative_pnl_baseline"] = portfolio["cumulative_pnl_baseline"].tolist()
    if has_risk:
        payload["cumulative_pnl_risk_attenuated"] = portfolio["cumulative_pnl_risk_attenuated"].tolist()
    return payload


def _run_training_job(job_id: str, config: dict) -> None:
    """Runs on a background thread - trains a PredictionModel (every
    asset's LSTM together - see models/portfolio_lstm.py's own docstring),
    always saves locally, and to quant.model_registry if save_db was
    requested, then records the result: per-asset hit rate, confusion
    matrices, and cumulative-return series (with per-day hit/miss) for all
    three splits - everything the Training view plots once the job is
    done.
    """
    _current_job_id.set(job_id)  # scopes _JobLogHandler's capture to this thread/job for its whole lifetime

    def _interim_callback(
        stage: str, epoch: int, epochs: int, train_loss: float, train_hit_rate: float,
        val_loss: float | None = None, val_hit_rate: float | None = None,
        train_sharpe: float | None = None, val_sharpe: float | None = None,
        best_score: float | None = None,
    ) -> None:
        """Registered below via _epoch_report_callback - called from INSIDE
        train_prediction_model's own epoch loop (same ~10%-of-epochs
        cadence as its progress logging), so the Training view can show a
        live-updating loss/hit-rate curve instead of only a progress bar
        and log text. `stage` is always "train" now (kept as a field for
        payload-shape stability).

        `train_sharpe`/`val_sharpe` (only present when the optional
        portfolio-Sharpe phase or checkpoint_metric="sharpe" is actually
        computing them - see train_prediction_model) and `best_score` (the
        running best checkpoint-selection score so far, in units matching
        `config["checkpoint_metric"]` - see _report_epoch's own docstring)
        let the Training view show the SAME "what is checkpoint selection
        actually optimizing" picture live, not just afterward.

        `best_score` is scoped to the CURRENT restart only - it resets to
        "no best yet" every time run_pipeline_multi_seed starts a new
        seed/bce_weight combo (a fresh train_prediction_model call, its own
        fresh best_val_loss/best_state - see that function's own
        docstring), so on its own it can visibly DROP the moment a new
        restart begins even though a stronger checkpoint from an earlier
        restart still exists (and may still end up the one actually kept -
        see run_pipeline_multi_seed's own seed-selection, which compares
        validation BCE across restarts independently of checkpoint_metric).
        `best_score_overall`, computed here from `job`'s own running
        history rather than from train_prediction_model, is the running
        best across EVERY restart this job has completed so far - never
        decreases within a job - so the Training view can show a
        genuinely monotonic "best so far" instead.
        """
        job = _JOBS.get(job_id)
        if job is None:
            return
        checkpoint_metric = config.get("checkpoint_metric") or "val_loss"
        interim = {
            "phase": "prediction", "stage": stage, "epoch": epoch, "total_epochs": epochs,
            "train_loss": train_loss, "train_hit_rate": train_hit_rate,
            "checkpoint_metric": checkpoint_metric,
        }
        if val_loss is not None:
            interim["val_loss"] = val_loss
            interim["val_hit_rate"] = val_hit_rate
        if train_sharpe is not None:
            interim["train_sharpe"] = train_sharpe
        if val_sharpe is not None:
            interim["val_sharpe"] = val_sharpe
        if best_score is not None:
            interim["best_score"] = best_score
            # "hit_rate"/"sharpe" are MAXIMIZED; "val_loss" is MINIMIZED -
            # same direction train_prediction_model's own score_i
            # comparison uses (see its docstring), just applied here across
            # restarts instead of across epochs within one.
            higher_is_better = checkpoint_metric in ("hit_rate", "sharpe")
            overall = job.get("best_score_overall")
            if overall is None or (best_score > overall if higher_is_better else best_score < overall):
                job["best_score_overall"] = best_score
            interim["best_score_overall"] = job["best_score_overall"]
        job["interim"] = interim
        history = job.setdefault("interim_history", [])
        history.append(interim)
        if len(history) > 1000:
            del history[: len(history) - 1000]

    def _on_best_checkpoint(model, data, run_args, epoch: int, best_state: list, best_score: float | None) -> None:
        """Registered below via run_pipeline_multi_seed/continue_training's
        own `on_best_checkpoint` param - fires after every VALIDATED epoch
        with (that call's own model/data/args, the epoch, best_state,
        best_score) - see models/portfolio_lstm.py's train_prediction_model
        docstring on why this is cheap (no evaluation happens here, just
        storing references for POST /api/train/{job_id}/save-best to build
        a full snapshot from ON DEMAND, only when a user actually clicks
        "Save best model so far" - not on every epoch).

        BUG THIS FIXES: `best_state` here is scoped to whichever restart is
        CURRENTLY running (see train_prediction_model's own docstring) -
        with n_seeds/a bce_weight sweep > 1, this fires for EVERY restart
        in turn, and an earlier restart's genuinely BETTER best_state must
        not be silently replaced by a later restart's own (possibly worse,
        especially early in ITS OWN training) current best just because it
        happened to fire more recently. Previously this function
        overwrote `_BEST_CHECKPOINT_STATE[job_id]` unconditionally on every
        call - so "Save best model so far" could save a checkpoint visibly
        WORSE than the "best so far" figure the Training view was showing
        (which already correctly tracks the true cross-restart best via
        `job["best_score_overall"]` - see _interim_callback above): the
        display and the save were reading from two different notions of
        "best". Now this compares `best_score` against whatever score is
        already stored before overwriting, using the SAME higher-is-
        better/lower-is-better direction _interim_callback uses - so
        _BEST_CHECKPOINT_STATE[job_id] can only ever improve within a job,
        exactly like the display.
        """
        job = _JOBS.get(job_id)
        if job is None or best_score is None:
            return
        checkpoint_metric = getattr(run_args, "checkpoint_metric", "val_loss") or "val_loss"
        higher_is_better = checkpoint_metric in ("hit_rate", "sharpe")
        stored = _BEST_CHECKPOINT_STATE.get(job_id)
        if stored is not None:
            stored_score = stored["score"]
            improved = best_score > stored_score if higher_is_better else best_score < stored_score
            if not improved:
                return
        _BEST_CHECKPOINT_STATE[job_id] = {
            "model": model, "data": data, "args": run_args, "epoch": epoch, "best_state": best_state,
            "score": best_score,
        }
        # Keep the DISPLAYED "best so far" (see _interim_callback) in sync
        # immediately - this fires every validated epoch, more often than
        # _interim_callback's own sparse ~10%-of-epochs cadence, so it's
        # frequently the FIRST to see a genuine new best.
        overall = job.get("best_score_overall")
        if overall is None or (best_score > overall if higher_is_better else best_score < overall):
            job["best_score_overall"] = best_score

    def _on_risk_epoch(epoch: int, epochs: int, train_sortino: float | None, val_sortino: float | None, best_val_sortino: float | None) -> None:
        """Registered as train_risk_engine's own `on_epoch` for this job's
        OPTIONAL phase 2 (see `config["train_risk_engine"]` below) - same
        shape as standalone _run_risk_engine_job's own `_on_epoch`, except
        `job["progress"]`/`job["interim"]` here carry a `"phase":
        "risk_engine"` tag (set once, right before this phase starts - see
        below) so the SAME polling payload phase 1 already used can tell
        the Training view which phase is live without a second job/poll.
        """
        job = _JOBS.get(job_id)
        if job is None:
            return
        progress = job["progress"]
        progress["epoch"] = epoch
        progress["total_epochs"] = epochs
        progress["percent"] = round(epoch / max(epochs, 1) * 100, 1)
        interim = {
            "phase": "risk_engine",
            "epoch": epoch, "total_epochs": epochs,
            "train_sortino": train_sortino, "val_sortino": val_sortino, "best_val_sortino": best_val_sortino,
        }
        job["interim"] = interim
        history = job.setdefault("interim_history", [])
        history.append(interim)
        if len(history) > 1000:
            del history[: len(history) - 1000]

    _epoch_report_callback.set(_interim_callback)
    _stop_check_callback.set(lambda: _JOBS.get(job_id, {}).get("stop_requested", False))
    _JOBS[job_id]["status"] = "running"
    try:
        continue_from = config.get("continue_from")
        if continue_from:
            # Continue-training mode (see models/portfolio_lstm.py's
            # continue_training): load the base model ONCE here, then
            # overwrite `args`'s own copies of every ARCHITECTURE-defining
            # field with the base model's actual values - not because
            # continue_training itself needs `args` to carry them (it
            # reads straight from `base_model`), but so the REST of this
            # function (the save_model/save_to_db calls a few lines below,
            # which read `args.features`/`args.cma_windows`/etc, not
            # `result`'s) stays self-consistent with what was actually
            # trained, exactly as if the user had submitted a normal
            # config matching this model from scratch.
            from models.portfolio_lstm import continue_training, load_prediction_model_auto

            base_model = load_prediction_model_auto(continue_from)
            args = Namespace(**{
                **DEFAULT_CONFIG, **config, "load_model": None,
                "pairs": base_model.pairs, "lookback": base_model.lookback,
                "features": base_model.features, "cma_windows": base_model.cma_windows,
                "bandpass_windows": base_model.bandpass_windows, "bandpass_order": base_model.bandpass_order,
                "hidden_size": base_model.hidden_size, "num_layers": base_model.num_layers,
                "dropout": base_model.dropout_p, "n_attn_heads": base_model.n_attn_heads,
                "cross_pairs": base_model.cross_pairs,
            })
            # Base name "save best model so far" (below) suffixes with ITS
            # OWN timestamp - see that endpoint - distinct from the
            # "_continued_<finish-time>" name this run's FINAL result gets
            # a few lines down, so an early snapshot is never silently
            # overwritten by (or collides with) the eventual final save.
            _JOBS[job_id]["model_base_name"] = os.path.splitext(os.path.basename(continue_from))[0]
            result = continue_training(args, base_model, on_best_checkpoint=_on_best_checkpoint)
        else:
            args = Namespace(**{**DEFAULT_CONFIG, **config, "load_model": None})
            _JOBS[job_id]["model_base_name"] = prediction_model_name(args)

            from models.portfolio_lstm import run_pipeline_multi_seed

            result = run_pipeline_multi_seed(args, on_best_checkpoint=_on_best_checkpoint)
        pairs = result.pairs

        # --- Optional phase 2: risk engine (see TrainRequest.train_risk_engine) ---
        # Trained INSIDE this same job/thread, right after phase 1 - no
        # separate save/reload round trip. `result.model` itself doesn't
        # carry x_mean/x_std/features/etc as instance attributes (those
        # only get set by PredictionModel._from_checkpoint, i.e. after a
        # save/load round trip - see that method's own docstring); rather
        # than duplicate its attribute list here, build the SAME checkpoint
        # dict save_model()/save_to_db() below use and reconstruct through
        # _from_checkpoint - purely in-memory (no disk I/O), and guaranteed
        # to stay consistent with save/load if that attribute list ever
        # changes.
        risk_engine_trained = None
        risk_engine_summary = None
        if getattr(args, "train_risk_engine", False):
            logger.info("Phase 1 (prediction model) complete - continuing with phase 2 (risk engine)")
            frozen_checkpoint = result.model._checkpoint_dict(
                x_mean=result.x_mean, x_std=result.x_std, pairs=result.pairs, lookback=result.lookback,
                features=args.features, cma_windows=args.cma_windows,
                sigma_hat=result.sigma_hat, neutral_band=result.neutral_band, target_vol=args.target_vol,
                bandpass_windows=args.bandpass_windows, bandpass_order=args.bandpass_order,
                signal_range=args.signal_range,
                direction_horizon=args.direction_horizon, rolling_stats_window=args.rolling_stats_window,
            )
            frozen_base = PredictionModel._from_checkpoint(frozen_checkpoint)
            progress = _JOBS[job_id]["progress"]
            progress["phase"] = "risk_engine"
            progress["phase_index"] = 2
            progress["epoch"] = 0
            progress["total_epochs"] = getattr(args, "risk_epochs", None) or 100
            progress["percent"] = 0.0
            risk_engine_trained, risk_engine_summary = risk_engine_module.train_risk_engine(
                frozen_base, args, on_epoch=_on_risk_epoch,
                stop_check=lambda: _JOBS.get(job_id, {}).get("stop_requested", False),
            )
            progress["epoch"] = progress["total_epochs"]
            progress["percent"] = 100.0

        result.model.save_model(
            x_mean=result.x_mean, x_std=result.x_std, pairs=result.pairs, lookback=result.lookback,
            features=args.features, cma_windows=args.cma_windows,
            sigma_hat=result.sigma_hat, neutral_band=result.neutral_band, target_vol=args.target_vol,
            bandpass_windows=args.bandpass_windows, bandpass_order=args.bandpass_order,
            signal_range=args.signal_range,
            direction_horizon=args.direction_horizon, rolling_stats_window=args.rolling_stats_window,
            risk_engine=risk_engine_trained,
        )

        if continue_from:
            # Deliberately NOT prediction_model_name(args) - in continue
            # mode `args`'s architecture fields now MATCH the base model
            # exactly (see above), so that deterministic name would
            # collide with (and silently overwrite) the base model itself.
            # A distinct name, suffixed with THIS run's own finish time,
            # guarantees the base model is never overwritten and every
            # continued run gets its own separately browsable entry.
            name = f"{_JOBS[job_id]['model_base_name']}_continued_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
        else:
            name = _JOBS[job_id]["model_base_name"]
        if args.save_db:
            result.model.save_to_db(
                name, x_mean=result.x_mean, x_std=result.x_std, pairs=result.pairs, lookback=result.lookback,
                features=args.features, cma_windows=args.cma_windows, description=args.model_description,
                sigma_hat=result.sigma_hat, neutral_band=result.neutral_band, target_vol=args.target_vol,
                bandpass_windows=args.bandpass_windows, bandpass_order=args.bandpass_order,
                signal_range=args.signal_range,
                direction_horizon=args.direction_horizon, rolling_stats_window=args.rolling_stats_window,
                risk_engine=risk_engine_trained,
            )

        # Portfolio PnL + per-year Sharpe (see models/portfolio_pnl.py) for
        # all three splits - same risk-parity-weight x probability-signal
        # strategy the Evaluation page reports, computed here too so a
        # freshly-trained model's Sharpe-by-year can be read off right
        # after training, without a separate evaluation round-trip.
        direction_horizon = getattr(args, "direction_horizon", 5) or 5
        target_vol = args.target_vol
        cost_bps = getattr(args, "cost_bps", DEFAULT_COST_BPS)
        band = result.neutral_band
        signal_min, signal_max = resolve_signal_bounds(pairs, args.signal_range)
        portfolio_train = compute_portfolio(
            result.probabilities_train, result.next_returns_train, direction_horizon, target_vol,
            cost_bps=cost_bps, signal_min=signal_min, signal_max=signal_max, neutral_band=band,
        )
        portfolio_val = compute_portfolio(
            result.probabilities_val, result.next_returns_val, direction_horizon, target_vol,
            cost_bps=cost_bps, signal_min=signal_min, signal_max=signal_max, neutral_band=band,
        )
        portfolio_test = compute_portfolio(
            result.probabilities_test, result.next_returns_test, direction_horizon, target_vol,
            cost_bps=cost_bps, signal_min=signal_min, signal_max=signal_max, neutral_band=band,
        )

        _JOBS[job_id]["result"] = {
            "model_name": name,
            "pairs": pairs,
            # Initial value for the frontend's neutral-band control - the
            # band this training run was configured with. hit_rate/
            # confusion_matrix below are computed at THIS band purely as
            # the initial display; probabilities/cumulative_returns carry
            # everything needed to recompute both for a different band
            # entirely client-side (see _probability_payload's docstring).
            "neutral_band": result.neutral_band,
            "hit_rate": {
                "train": _hit_rate_payload(pairs, result.hit_rate_train),
                "val": _hit_rate_payload(pairs, result.hit_rate_val),
                "test": _hit_rate_payload(pairs, result.hit_rate_test),
            },
            "confusion_matrix": {
                "train": _confusion_matrix_payload(pairs, result.probabilities_train, result.direction_labels_train, result.neutral_band),
                "val": _confusion_matrix_payload(pairs, result.probabilities_val, result.direction_labels_val, result.neutral_band),
                "test": _confusion_matrix_payload(pairs, result.probabilities_test, result.direction_labels_test, result.neutral_band),
            },
            "cumulative_returns": {
                "train": _cumulative_return_payload(result.dates_train, pairs, result.next_returns_train),
                "val": _cumulative_return_payload(result.dates_val, pairs, result.next_returns_val),
                "test": _cumulative_return_payload(result.dates_test, pairs, result.next_returns_test),
            },
            "probabilities": {
                "train": _probability_payload(result.dates_train, pairs, result.probabilities_train, result.direction_labels_train),
                "val": _probability_payload(result.dates_val, pairs, result.probabilities_val, result.direction_labels_val),
                "test": _probability_payload(result.dates_test, pairs, result.probabilities_test, result.direction_labels_test),
            },
            "distribution": {
                "train": _distribution_payload(pairs, result.z_labels_train, result.mu_train, result.sigma_train),
                "val": _distribution_payload(pairs, result.z_labels_val, result.mu_val, result.sigma_val),
                "test": _distribution_payload(pairs, result.z_labels_test, result.mu_test, result.sigma_test),
            },
            "annual_sharpe": {
                "train": annual_sharpe_table(result.dates_train, portfolio_train["pnl_modulated"], portfolio_train["pnl_baseline"]),
                "val": annual_sharpe_table(result.dates_val, portfolio_val["pnl_modulated"], portfolio_val["pnl_baseline"]),
                "test": annual_sharpe_table(result.dates_test, portfolio_test["pnl_modulated"], portfolio_test["pnl_baseline"]),
            },
            # Only present when train_risk_engine was set - see phase 2
            # above. The Evaluation view's risk-attenuated PnL series come
            # from re-running evaluate_risk_engine on the SAVED model
            # (models.risk_engine attached under this same name) rather
            # than from here - this is just the final train/val Sortino a
            # user can read off right after training finishes.
            "risk_engine": risk_engine_summary,
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
    n_lambdas = len(req.bce_weight) if isinstance(req.bce_weight, list) else 1
    _JOBS[job_id] = {
        "status": "pending",
        "result": None,
        "error": None,
        "logs": [],
        "progress": {
            "seed_index": 1, "n_seeds": req.n_seeds,
            "lambda_index": 1, "n_lambdas": n_lambdas,
            "epoch": 0, "total_epochs": req.epochs,
            "percent": 0.0,
            # "prediction" (phase 1) throughout, unless train_risk_engine
            # is set, in which case _run_training_job switches this to
            # "risk_engine" once phase 1 finishes - see its own
            # _on_risk_epoch. n_phases/phase_index let the Training view
            # show "Phase 2/2" without guessing from `phase` alone.
            "phase": "prediction",
            "n_phases": 2 if req.train_risk_engine else 1,
            "phase_index": 1,
        },
        "interim": None,  # live train/val loss+hit-rate snapshot, updated during training - see _run_training_job
        "interim_history": [],
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


@app.post("/api/train/{job_id}/save-best")
def save_best_checkpoint(job_id: str) -> dict:
    """Build a full snapshot from whichever checkpoint is CURRENTLY best
    (per models/portfolio_lstm.py's train_prediction_model - each asset's
    OWN lowest-validation-score epoch, by "checkpoint_metric"), save it to
    quant.model_registry under a NEW, timestamped name (never overwriting
    the base model or this run's own eventual final save - see
    _run_training_job's own naming), and return a global summary (loss,
    hit rate, and annualized Sharpe on ALL THREE splits) - usable at any
    point during (or after) a training run, not just once it's finished.

    Works whether the run is a fresh training job or a continue_training
    one (see /api/train's `continue_from`) - both register the same
    `on_best_checkpoint` hook. 400s if no epoch has been validated yet, or
    (a rare edge case - see summarize_checkpoint) if some asset's every
    score so far has been NaN, so it has no validated checkpoint of its
    own yet.
    """
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    state = _BEST_CHECKPOINT_STATE.get(job_id)
    if state is None:
        raise HTTPException(
            status_code=400,
            detail="No validated checkpoint yet - this needs at least one completed validation epoch.",
        )
    snapshot = summarize_checkpoint(state["model"], state["data"], state["args"], state["best_state"])
    if snapshot is None:
        raise HTTPException(
            status_code=400,
            detail="Not every asset has a validated checkpoint yet (e.g. a NaN score so far) - try again shortly.",
        )
    snapshot_model, result, summary = snapshot
    run_args = state["args"]
    base_name = job.get("model_base_name") or prediction_model_name(run_args)
    name = f"{base_name}_bestsofar_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    snapshot_model.save_to_db(
        name, x_mean=result.x_mean, x_std=result.x_std, pairs=result.pairs, lookback=result.lookback,
        features=run_args.features, cma_windows=run_args.cma_windows, description=run_args.model_description,
        sigma_hat=result.sigma_hat, neutral_band=result.neutral_band, target_vol=run_args.target_vol,
        bandpass_windows=run_args.bandpass_windows, bandpass_order=run_args.bandpass_order,
        signal_range=getattr(run_args, "signal_range", None),
        direction_horizon=getattr(run_args, "direction_horizon", 5) or 5,
        rolling_stats_window=getattr(run_args, "rolling_stats_window", 20) or 20,
    )
    return {"model_name": name, "epoch": state["epoch"], "summary": summary}


# --------------------------------------------------------------------------
# POST /api/train-risk-engine
# --------------------------------------------------------------------------

class TrainRiskEngineRequest(BaseModel):
    """Train a NEW RiskEngine (see models/risk_engine.py) on top of an
    already-trained, EXISTING PredictionModel, which is used strictly
    FROZEN throughout - see train_risk_engine's own docstring on why this
    is a separate second stage, not a joint retraining. The result is
    saved as a NEW model (base predictor + attached risk engine bundled
    together - see PredictionModel._checkpoint_dict's own `risk_engine`
    key), never overwriting `base_model`.
    """

    base_model: str  # a quant.model_registry name or local .pt path
    years: int = 8
    cutoff_date: str | None = None
    train_frac: float = 0.8
    test_frac: float = 0.1
    device: str = "auto"
    risk_lookback: int = risk_engine_module.DEFAULT_RISK_LOOKBACK
    # Trailing window (days) for the risk engine's OWN skew/kurtosis
    # computation - INDEPENDENT of the base predictor's own
    # rolling_stats_window (see models/risk_engine.py's own
    # DEFAULT_RISK_ROLLING_STATS_WINDOW comment: skew/kurtosis are
    # risk-engine-exclusive inputs now, so there's no reason to couple the
    # two windows).
    risk_rolling_stats_window: int = risk_engine_module.DEFAULT_RISK_ROLLING_STATS_WINDOW
    # (short, long) window pairs - independent of, and in addition to,
    # whatever the base predictor's OWN cma_windows/bandpass_windows are -
    # see models/risk_engine.py's own module docstring.
    risk_cma_windows: list[list[int]] = []
    risk_bandpass_windows: list[list[int]] = []
    risk_bandpass_order: int = DEFAULT_BANDPASS_ORDER
    # Per-asset attenuation bounds - see models/risk_engine.py's own
    # attenuation_from_raw. (0, 1) (the default) can only REDUCE exposure,
    # never amplify it.
    min_risk_att: float = risk_engine_module.DEFAULT_MIN_RISK_ATT
    max_risk_att: float = risk_engine_module.DEFAULT_MAX_RISK_ATT
    risk_hidden_size: int = risk_engine_module.DEFAULT_RISK_HIDDEN_SIZE
    risk_num_layers: int = risk_engine_module.DEFAULT_RISK_NUM_LAYERS
    risk_dropout: float = risk_engine_module.DEFAULT_RISK_DROPOUT
    risk_n_attn_heads: int = risk_engine_module.DEFAULT_RISK_N_ATTN_HEADS
    risk_epochs: int = 100
    risk_lr: float = 1e-3
    risk_weight_decay: float = 0.0
    # Chunk size (days, in DECISION-day units - see
    # models/risk_engine.py's own train_risk_engine docstring) each
    # non-overlapping period Sortino is computed over.
    risk_sortino_window: int = risk_engine_module.DEFAULT_RISK_SORTINO_WINDOW
    # Regularizes the engine's own output toward max_risk_att (full
    # exposure) - see models/risk_engine.py's own module docstring on why
    # a risk-reduction overlay needs this anchor by default.
    full_exposure_penalty: float = risk_engine_module.DEFAULT_FULL_EXPOSURE_PENALTY
    cost_bps: float = DEFAULT_COST_BPS
    save_db: bool = True
    model_description: str = ""


def _run_risk_engine_job(job_id: str, config: dict) -> None:
    """Runs on a background thread, same shape as _run_training_job (see
    its own docstring) - reuses the SAME `_JOBS` dict, GET /api/train/{job_id}
    polling endpoint, and POST /api/train/{job_id}/stop endpoint, since a
    risk-engine job's `status`/`progress`/`interim`/`logs`/`stop_requested`
    fields are structurally identical; only `interim`'s own CONTENTS differ
    (train/val Sortino here, not NLL+BCE+Sharpe - see
    models/risk_engine.py's train_risk_engine `on_epoch` callback).
    """
    _current_job_id.set(job_id)

    def _on_epoch(epoch, epochs, train_sortino, val_sortino, best_val_sortino) -> None:
        job = _JOBS.get(job_id)
        if job is None:
            return
        progress = job["progress"]
        progress["epoch"] = epoch
        progress["total_epochs"] = epochs
        progress["percent"] = round(epoch / max(epochs, 1) * 100, 1)
        interim = {
            "epoch": epoch, "total_epochs": epochs,
            "train_sortino": train_sortino, "val_sortino": val_sortino, "best_val_sortino": best_val_sortino,
        }
        job["interim"] = interim
        history = job.setdefault("interim_history", [])
        history.append(interim)
        if len(history) > 1000:
            del history[: len(history) - 1000]

    _stop_check_callback.set(lambda: _JOBS.get(job_id, {}).get("stop_requested", False))
    _JOBS[job_id]["status"] = "running"
    try:
        base_model = load_prediction_model_auto(config["base_model"])
        args = Namespace(**{**DEFAULT_CONFIG, **config})
        base_name = os.path.splitext(os.path.basename(config["base_model"]))[0]
        _JOBS[job_id]["model_base_name"] = base_name

        engine, summary = risk_engine_module.train_risk_engine(
            base_model, args, on_epoch=_on_epoch, stop_check=lambda: _JOBS.get(job_id, {}).get("stop_requested", False),
        )

        # Distinct from base_model's own name, suffixed with THIS run's own
        # finish time - never overwrites the base predictor (same naming
        # discipline as continue_training's own resave - see
        # _run_training_job).
        name = f"{base_name}_risk_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
        save_kwargs = dict(
            x_mean=base_model.x_mean, x_std=base_model.x_std, pairs=base_model.pairs, lookback=base_model.lookback,
            features=base_model.features, cma_windows=base_model.cma_windows,
            sigma_hat=base_model.sigma_hat, neutral_band=base_model.neutral_band, target_vol=base_model.target_vol,
            bandpass_windows=base_model.bandpass_windows, bandpass_order=base_model.bandpass_order,
            signal_range=getattr(base_model, "signal_range", None),
            direction_horizon=base_model.direction_horizon, rolling_stats_window=base_model.rolling_stats_window,
            risk_engine=engine,
        )
        base_model.save_model(**save_kwargs)
        if args.save_db:
            base_model.save_to_db(name, description=args.model_description, **save_kwargs)

        _JOBS[job_id]["result"] = {
            "model_name": name, "base_model": config["base_model"],
            "train_sortino": summary["train_sortino"], "val_sortino": summary["val_sortino"],
        }
        progress = _JOBS[job_id]["progress"]
        progress["epoch"] = progress["total_epochs"]
        progress["percent"] = 100.0
        _JOBS[job_id]["status"] = "done"
    except TrainingStopped:
        _JOBS[job_id]["status"] = "stopped"
    except Exception as exc:  # noqa: BLE001 - surface any failure to the polling client
        _JOBS[job_id]["status"] = "error"
        _JOBS[job_id]["error"] = str(exc)


@app.post("/api/train-risk-engine")
def start_risk_engine_training(req: TrainRiskEngineRequest) -> dict:
    """Kick off risk-engine training on a background thread - see
    _run_risk_engine_job. Polled/stopped via the SAME GET /api/train/{job_id}
    and POST /api/train/{job_id}/stop endpoints a normal training job uses.
    """
    job_id = str(uuid.uuid4())
    _JOBS[job_id] = {
        "status": "pending",
        "result": None,
        "error": None,
        "logs": [],
        "progress": {
            "seed_index": 1, "n_seeds": 1, "lambda_index": 1, "n_lambdas": 1,
            "epoch": 0, "total_epochs": req.risk_epochs, "percent": 0.0,
        },
        "interim": None,
        "interim_history": [],
        "stop_requested": False,
    }
    thread = threading.Thread(target=_run_risk_engine_job, args=(job_id, req.model_dump()), daemon=True)
    thread.start()
    return {"job_id": job_id}


# --------------------------------------------------------------------------
# POST /api/evaluate
# --------------------------------------------------------------------------

class EvaluateRequest(BaseModel):
    """Note: no `pairs` or `lookback` fields - both are properties of the
    trained model itself (restored from its checkpoint, see
    models/portfolio_lstm.py's load_pipeline), not free evaluation
    parameters. No `train_frac`/`test_frac` either - evaluation is no
    longer split into train/validation/test at all (that distinction only
    means something DURING training, for checkpoint/seed selection and a
    genuinely held-out test read); here the model is just run over the
    WHOLE fetched period as one continuous evaluation. Only `years` (how
    much history to fetch), `cutoff_date`, and `device` are genuinely
    evaluation-time.
    """

    model_name: str  # a quant.model_registry name, or a local .pt path
    years: int = 8
    # Caps the fetched date range (and "latest_probabilities") at this
    # date - never later, regardless of what's since landed in Postgres -
    # so evaluation never sees a day after it (see models/portfolio_lstm.py's
    # _resolve_cutoff_date). None (the default) means "no cutoff": use all
    # data up to today. An ISO "YYYY-MM-DD" string walk-forward-backtests
    # this model as of that historical date instead.
    cutoff_date: str | None = None
    device: str = "auto"  # "auto" (Metal/MPS on Apple Silicon, else CUDA, else CPU), "cpu", "mps", or "cuda" - see get_device
    # Linear transaction cost (basis points per unit of daily position
    # change) charged in the reported portfolio PnL/annual Sharpe - see
    # models/portfolio_pnl.py. Positions themselves are cost-independent.
    cost_bps: float = DEFAULT_COST_BPS


def _predict_latest_probabilities(args: Namespace, model) -> tuple[dict[str, float], str]:
    """Compute the model's predicted probability for the NEXT (as-yet-
    unrealized) `direction_horizon`-day-forward outcome, using the most
    recent `lookback`-day window of real data - distinct from the
    historical validation/test-period probabilities (which are for
    backtesting): this is "run the model on today's window".

    Uses `model.pairs`/`model.lookback`/`model.features`/`model.cma_windows`/
    `model.bandpass_windows`/`model.bandpass_order` - restored from the
    model's own checkpoint - rather than anything from
    the request, so the fetched data, window length, and returned feature
    set always match what the model was actually trained on.

    Returns `(probabilities, as_of_date)`: `as_of_date` is the most recent
    date with a REAL realized close (the last row of `returns`, i.e. the
    lookback window's own last day) - the day the model is actually
    treating as "today" when it reports these probabilities, which can lag
    behind the calendar date this endpoint is called on (a weekend, a
    holiday, or `cutoff_date`/stale quotes - see /api/quotes/refresh). The
    forecast itself is for the `direction_horizon` TRADING days after this
    date, not a specific calendar date (holidays/weekends make that exact
    date caller-dependent, so it's left to the frontend to describe
    relative to `as_of_date` rather than computed here).
    """
    pairs = model.pairs
    lookback = model.lookback
    # model may live on an accelerator (MPS/CUDA - see
    # models/portfolio_lstm.get_device); every tensor built below is moved
    # to match before being passed to model() - this is a single tiny
    # inference call, so there's no real GPU benefit, but it must still
    # land on whatever device the (possibly GPU-trained) model actually
    # lives on or the forward pass would raise a device-mismatch error.
    device = next(model.parameters()).device
    features = getattr(model, "features", None) or list(DEFAULT_FEATURES)
    cma_windows = getattr(model, "cma_windows", None) or []
    bandpass_windows = getattr(model, "bandpass_windows", None) or []
    bandpass_order = getattr(model, "bandpass_order", None) or DEFAULT_BANDPASS_ORDER
    # From the model's own checkpoint, NOT the request - rolling_stats_window
    # changes the actual feature values (rolling vol/skew/kurt) the network
    # reads; a mismatch here would feed it data unlike anything it trained on.
    rolling_stats_window = getattr(model, "rolling_stats_window", 20) or 20
    min_days = lookback + rolling_stats_window
    cutoff_date = getattr(args, "cutoff_date", None)

    prices = load_close_prices(pairs, years=args.years, cutoff_date=cutoff_date)
    returns = to_log_returns(prices)
    if len(returns) < min_days:
        raise ValueError(
            f"Only {len(returns)} days of history available for {pairs}, but this model needs "
            f"the most recent {min_days} days - increase 'years' to fetch more history."
        )

    feature_returns = build_feature_dataframe(
        returns, pairs, features, rolling_stats_window, cma_windows, args.years, bandpass_windows, bandpass_order,
        cutoff_date,
    )
    last_window_features = feature_returns.to_numpy(dtype=np.float32)[-lookback:]  # (lookback, n_assets * n_channels)
    X = torch.tensor((last_window_features - model.x_mean) / model.x_std, device=device).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        # model(X) returns (mu, sigma), dense over every day in the window
        # (see PredictionModel's docstring) - only the LAST timestep (the
        # "decision day") is used here, matching backtest reporting.
        # probit(mu / (sigma * sigma_hat)) converts them to a CALIBRATED
        # probability - sigma_hat is the validation-fit standardized-
        # residual std persisted in the checkpoint (see
        # evaluate_prediction_model's docstring on calibration); there's
        # no validation set here to refit it against (this is a single
        # live window), so the training-time estimate is reused as-is.
        sigma_hat = getattr(model, "sigma_hat", np.ones(len(pairs), dtype=np.float32))
        sigma_hat_t = torch.as_tensor(sigma_hat, device=device, dtype=torch.float32)
        mu, sigma = model(X)
        probs = probit(mu[:, -1, :] / (sigma[:, -1, :] * sigma_hat_t)).cpu().numpy()
    # Same abstention rule the backtest metrics used (see
    # apply_neutral_band): inside the band the model reports exactly 0.5 -
    # an explicit "no call today", not a weak directional lean.
    probs = apply_neutral_band(probs, getattr(model, "neutral_band", 0.0))
    as_of_date = str(returns.index[-1].date()) if hasattr(returns.index[-1], "date") else str(returns.index[-1])
    return {pair: float(p) for pair, p in zip(pairs, probs[0])}, as_of_date


def _compute_portfolio_and_today(
    probabilities: np.ndarray, next_returns: np.ndarray, latest_probabilities_array: np.ndarray,
    direction_horizon: int, target_vol: float, cost_bps: float, signal_min: np.ndarray, signal_max: np.ndarray,
    band: float, dates: pd.DatetimeIndex, risk_engine, returns_df: pd.DataFrame | None,
) -> tuple[dict, dict, np.ndarray | None]:
    """Shared by evaluate()/recompute_portfolio: compute_portfolio +
    latest_position, PLUS (if `risk_engine` is given - an already-trained
    RiskEngine, see models/risk_engine.py) a SECOND pass that additionally
    reports a risk-attenuated series.

    Two passes, not one, because the risk engine's own input is the
    prediction stage's OWN `positions_modulated` (see
    models/risk_engine.py's own module docstring on why attenuation must
    be computed FROM an already-fully-formed position, not folded into
    computing it) - the first pass produces that; the second re-runs
    compute_portfolio/latest_position WITH the resulting attenuation
    applied. `returns_df` (raw log returns, needed for the risk engine's
    own skew/kurt/CMA/bandpass inputs) is REQUIRED whenever `risk_engine`
    is given.

    "Today"'s own attenuation is computed by appending `today`'s own
    (pre-attenuation) position as one more day onto the historical
    `positions_modulated` series (see models/risk_engine.py's
    evaluate_risk_engine's own docstring on why a synthetic date past
    `returns_df`'s own last row is safe here - its own ffill carries the
    last REAL day's skew/kurt/CMA/bandpass forward) - the same
    "positions and dates get an extra placeholder day" pattern
    models/portfolio_pnl.py's own latest_position already uses for
    `next_returns`.

    Returns `(portfolio, today, attenuation_hist_or_None)` - the third
    element lets a caller (see _EVAL_CACHE) cache the historical
    attenuation series without recomputing it from scratch elsewhere.
    """
    portfolio = compute_portfolio(
        probabilities, next_returns, direction_horizon, target_vol, cost_bps=cost_bps,
        signal_min=signal_min, signal_max=signal_max, neutral_band=band,
    )
    today = latest_position(
        probabilities, next_returns, latest_probabilities_array, direction_horizon, target_vol,
        signal_min=signal_min, signal_max=signal_max, neutral_band=band,
    )
    if risk_engine is None:
        return portfolio, today, None

    from models.risk_engine import evaluate_risk_engine

    extended_positions = np.vstack([portfolio["positions_modulated"], today["position_modulated"].reshape(1, -1)])
    extended_dates = dates.append(pd.DatetimeIndex([dates[-1] + pd.Timedelta(days=1)]))
    attenuation_full = evaluate_risk_engine(risk_engine, extended_positions, returns_df, extended_dates)
    attenuation_hist, attenuation_today = attenuation_full[:-1], attenuation_full[-1]

    portfolio = compute_portfolio(
        probabilities, next_returns, direction_horizon, target_vol, cost_bps=cost_bps,
        signal_min=signal_min, signal_max=signal_max, neutral_band=band, attenuation=attenuation_hist,
    )
    today = latest_position(
        probabilities, next_returns, latest_probabilities_array, direction_horizon, target_vol,
        signal_min=signal_min, signal_max=signal_max, neutral_band=band,
        attenuation=attenuation_hist, latest_attenuation=attenuation_today,
    )
    return portfolio, today, attenuation_hist


@app.post("/api/evaluate")
def evaluate(req: EvaluateRequest) -> dict:
    """Load a previously-trained model (by quant.model_registry name or
    local path) and run it purely for inference - no training happens.
    Unlike training, evaluation is NOT split into train/validation/test -
    that distinction only matters DURING training (val drives checkpoint
    selection, test is held out for an unbiased read); here the model is
    simply run over the WHOLE fetched period as one continuous evaluation
    (`train_frac=1.0`/`test_frac=0.0` internally - see _prepare_data's own
    purge-boundary handling for why this doesn't needlessly discard the
    freshest days). Returns everything the Evaluation view plots:
      - hit_rate: per-asset directional hit rate.
      - confusion_matrix: per-asset TP/FP/TN/FN + accuracy/precision/
        recall/specificity/F1.
      - cumulative_returns: per-asset cumulative (single-day-ahead) return
        path, with a parallel per-day hit/miss boolean array for coloring.
      - latest_probabilities: the model's predicted probability per asset
        for the next (not-yet-realized) `direction_horizon`-day outcome,
        using the freshest available data.
      - portfolio: per-asset risk-parity positions, both probability-
        modulated (scaled to the model's own target_vol) and an
        unmodulated risk-parity baseline for comparison - see
        models/portfolio_pnl.py.
      - latest_position: today's not-yet-booked position per asset (same
        modulated/baseline split) alongside the probability it was sized
        from - what a trader would book for today's EoD.
      - eval_id: opaque key for POST /api/evaluate/{eval_id}/portfolio,
        which recomputes portfolio/annual_sharpe/latest_position for a
        DIFFERENT neutral band without re-fetching data or re-running the
        model - see that endpoint and _EVAL_CACHE's own comment.

    `pairs` and `lookback` are deliberately NOT accepted from `req` - both
    are recovered from the loaded model's own checkpoint (see
    models/portfolio_lstm.py's load_pipeline), since they're properties of
    the trained model, not free evaluation parameters. Only `years` (how
    much history to fetch) is caller-controlled; if it's not enough to
    cover the model's own sequence length, load_pipeline/
    _predict_latest_probabilities raise a ValueError, turned into a 400 below.
    """
    args = Namespace(**{
        **DEFAULT_CONFIG,
        "pairs": None,
        "lookback": None,
        "years": req.years,
        "cutoff_date": req.cutoff_date,
        "train_frac": 1.0,
        "test_frac": 0.0,
        "device": req.device,
        "load_model": req.model_name,
    })

    try:
        from models.portfolio_lstm import run_pipeline_multi_seed

        result = run_pipeline_multi_seed(args)  # load_model set -> pure inference, no training
        pairs = result.pairs
        latest_probabilities, latest_as_of_date = _predict_latest_probabilities(args, result.model)

        # Portfolio PnL (see models/portfolio_pnl.py): risk-parity weights
        # modulated by the model's own probabilities, scaled to the
        # model's own persisted target_vol (a checkpoint property, like
        # neutral_band - see PredictionModel._checkpoint_dict). Uses the
        # model's own persisted neutral_band as the INITIAL value - an
        # in-band day rides the unmodulated risk-parity weight (see
        # compute_portfolio's own docstring) - the frontend's
        # live-adjustable display band can recompute this against a
        # DIFFERENT band via POST /api/evaluate/{eval_id}/portfolio (see
        # _EVAL_CACHE) without a full re-run, so a band change actually
        # moves the PnL/positions below, not just hit rate/confusion matrix.
        direction_horizon = getattr(result.model, "direction_horizon", 5) or 5
        target_vol = getattr(result.model, "target_vol", DEFAULT_TARGET_VOL)
        band = result.neutral_band
        signal_min, signal_max = resolve_signal_bounds(pairs, getattr(result.model, "signal_range", None))
        latest_probabilities_array = np.array([latest_probabilities[p] for p in pairs], dtype=np.float32)

        risk_engine = getattr(result.model, "risk_engine", None)
        returns_df = None
        if risk_engine is not None:
            # Raw log returns - the risk engine's own skew/kurt/CMA/bandpass
            # inputs (see models/risk_engine.py) - a cheap extra Postgres
            # read (see load_close_prices), not a live API call.
            prices = load_close_prices(pairs, years=req.years, cutoff_date=req.cutoff_date)
            returns_df = to_log_returns(prices)

        portfolio, today, attenuation_hist = _compute_portfolio_and_today(
            result.probabilities_train, result.next_returns_train, latest_probabilities_array, direction_horizon,
            target_vol, req.cost_bps, signal_min, signal_max, band, result.dates_train, risk_engine, returns_df,
        )

        eval_id = str(uuid.uuid4())
        _EVAL_CACHE[eval_id] = {
            "pairs": pairs,
            "probabilities_train": result.probabilities_train,
            "next_returns_train": result.next_returns_train,
            "dates_train": result.dates_train,
            "direction_horizon": direction_horizon,
            "target_vol": target_vol,
            "cost_bps": req.cost_bps,
            "signal_min": signal_min,
            "signal_max": signal_max,
            "latest_probabilities": latest_probabilities,
            "latest_probabilities_array": latest_probabilities_array,
            "risk_engine": risk_engine,
            "returns_df": returns_df,
        }

        return {
            "eval_id": eval_id,
            "pairs": pairs,
            "latest_probabilities": latest_probabilities,
            # The most recent date with a REAL realized close - the day the
            # model is treating as "today" (its lookback window's last day)
            # when it reports latest_probabilities/latest_position below -
            # see _predict_latest_probabilities's own docstring on why this
            # can lag the calendar date this endpoint is called on.
            "latest_as_of_date": latest_as_of_date,
            "neutral_band": result.neutral_band,  # initial value for the frontend's neutral-band control - see start_training's result dict
            "target_vol": target_vol,
            "signal_range": getattr(result.model, "signal_range", {}),
            "direction_horizon": direction_horizon,
            "rolling_stats_window": getattr(result.model, "rolling_stats_window", 20) or 20,
            "latest_position": {
                pair: {
                    "position_modulated": float(today["position_modulated"][i]),
                    "position_baseline": float(today["position_baseline"][i]),
                    **({"position_risk_attenuated": float(today["position_risk_attenuated"][i])} if risk_engine is not None else {}),
                    "probability": latest_probabilities[pair],
                }
                for i, pair in enumerate(pairs)
            },
            "has_risk_engine": risk_engine is not None,
            "portfolio": _portfolio_payload(result.dates_train, pairs, portfolio),
            "annual_sharpe": annual_sharpe_table(
                result.dates_train, portfolio["pnl_modulated"], portfolio["pnl_baseline"],
                portfolio.get("pnl_risk_attenuated"),
            ),
            "hit_rate": _hit_rate_payload(pairs, result.hit_rate_train),
            "confusion_matrix": _confusion_matrix_payload(pairs, result.probabilities_train, result.direction_labels_train, result.neutral_band),
            "cumulative_returns": _cumulative_return_payload(result.dates_train, pairs, result.next_returns_train),
            "probabilities": _probability_payload(result.dates_train, pairs, result.probabilities_train, result.direction_labels_train),
            "distribution": _distribution_payload(pairs, result.z_labels_train, result.mu_train, result.sigma_train),
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Model not found: {exc}") from exc
    except ValueError as exc:
        # e.g. "not enough history for this model's sequence length" - a
        # clear, actionable error from load_pipeline/
        # _predict_latest_probabilities, not a 500.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class RecomputePortfolioRequest(BaseModel):
    neutral_band: float


@app.post("/api/evaluate/{eval_id}/portfolio")
def recompute_portfolio(eval_id: str, req: RecomputePortfolioRequest) -> dict:
    """Recompute portfolio/annual_sharpe/latest_position for a DIFFERENT
    neutral band than whatever POST /api/evaluate originally used, from
    the cached raw probabilities/returns it stored under `eval_id` (see
    _EVAL_CACHE) - no data re-fetch, no model forward pass, just re-running
    compute_portfolio/latest_position/annual_sharpe_table on already-
    computed arrays, so this is cheap enough to call on every tick of the
    Evaluation view's neutral-band slider.

    Inside the band, compute_portfolio treats that day as having no view -
    the position rides the UNMODULATED risk-parity weight (see its own
    docstring) rather than a directional bet, so widening the band actually
    shifts more days' positions toward the diversified baseline, not just
    the reported hit rate.
    """
    cached = _EVAL_CACHE.get(eval_id)
    if cached is None:
        raise HTTPException(status_code=404, detail="Unknown eval_id - re-run evaluation first")

    pairs = cached["pairs"]
    risk_engine = cached.get("risk_engine")
    portfolio, today, _ = _compute_portfolio_and_today(
        cached["probabilities_train"], cached["next_returns_train"], cached["latest_probabilities_array"],
        cached["direction_horizon"], cached["target_vol"], cached["cost_bps"], cached["signal_min"], cached["signal_max"],
        req.neutral_band, cached["dates_train"], risk_engine, cached.get("returns_df"),
    )
    latest_probabilities = cached["latest_probabilities"]

    return {
        "neutral_band": req.neutral_band,
        "latest_position": {
            pair: {
                "position_modulated": float(today["position_modulated"][i]),
                "position_baseline": float(today["position_baseline"][i]),
                **({"position_risk_attenuated": float(today["position_risk_attenuated"][i])} if risk_engine is not None else {}),
                "probability": latest_probabilities[pair],
            }
            for i, pair in enumerate(pairs)
        },
        "portfolio": _portfolio_payload(cached["dates_train"], pairs, portfolio),
        "annual_sharpe": annual_sharpe_table(
            cached["dates_train"], portfolio["pnl_modulated"], portfolio["pnl_baseline"], portfolio.get("pnl_risk_attenuated"),
        ),
    }
