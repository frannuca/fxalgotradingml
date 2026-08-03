"""Generates the numeric results embedded in methodology.tex: trains three
model "modes" on EURUSD/USDJPY/AUDUSD against the real database and
serializes everything the LaTeX doc's tables/figures need to
docs/methodology/results/<mode>.json (metrics) and
docs/methodology/results/<mode>.npz (raw arrays for plotting).

Modes
-----
A "baseline":  DEFAULT_FEATURES only (log_return, vol, skew, kurt), no
               cross_pairs, sharpe_weight=0 - pure distributional (NLL+BCE)
               training, independent per asset.
B "enriched":  + carry + cma + bandpass features, still sharpe_weight=0 -
               isolates the effect of richer per-asset input features alone.
C "pnl_mode":  same enriched features, sharpe_weight=1.0 - adds the joint
               portfolio-Sharpe training phase on top of B.

Run from the repo root: python docs/methodology/run_experiments.py
"""
from __future__ import annotations

import json
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

from models.portfolio_lstm import DEFAULT_CONFIG, apply_neutral_band, run_pipeline_multi_seed
from models.portfolio_pnl import annual_sharpe_table, compute_portfolio

PAIRS = ["EURUSD", "USDJPY", "AUDUSD"]
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

MODES = {
    "A_baseline": {
        "features": ["log_return", "vol", "skew", "kurt"],
        "cma_windows": [], "bandpass_windows": [], "cross_pairs": {}, "sharpe_weight": 0.0,
    },
    "B_enriched": {
        "features": ["log_return", "vol", "skew", "kurt", "carry", "cma", "bandpass"],
        "cma_windows": [[10, 50]], "bandpass_windows": [[10, 50]], "cross_pairs": {}, "sharpe_weight": 0.0,
    },
    "C_pnl_mode": {
        "features": ["log_return", "vol", "skew", "kurt", "carry", "cma", "bandpass"],
        "cma_windows": [[10, 50]], "bandpass_windows": [[10, 50]], "cross_pairs": {}, "sharpe_weight": 1.0,
    },
}

COMMON = dict(
    pairs=PAIRS, lookback=30, years=8, train_frac=0.8, test_frac=0.1,
    direction_horizon=5, rolling_stats_window=20,
    hidden_size=16, num_layers=1, dropout=0.1, n_attn_heads=4,
    epochs=300, lr=1e-3, weight_decay=1e-4, bce_weight=1.0,
    sharpe_window=20, neutral_band=0.05, target_vol=0.10,
    n_seeds=1, device="cpu", save_db=False, load_model=None,
)


def run_mode(name: str, overrides: dict) -> None:
    print(f"\n=== Mode {name} ===", flush=True)
    t0 = time.time()
    args = Namespace(**{**DEFAULT_CONFIG, **COMMON, **overrides})
    result = run_pipeline_multi_seed(args)
    pairs = result.pairs
    band = result.neutral_band
    target_vol = args.target_vol

    metrics = {"mode": name, "pairs": pairs, "elapsed_sec": None}
    arrays = {}

    for split in ("train", "val", "test"):
        probs = getattr(result, f"probabilities_{split}")
        labels = getattr(result, f"direction_labels_{split}")
        next_returns = getattr(result, f"next_returns_{split}")
        dates = getattr(result, f"dates_{split}")
        hit_rate = getattr(result, f"hit_rate_{split}")
        z_labels = getattr(result, f"z_labels_{split}")
        mu = getattr(result, f"mu_{split}")
        sigma = getattr(result, f"sigma_{split}")

        from models.portfolio_lstm import confusion_matrix_metrics
        cm = confusion_matrix_metrics(probs, labels, neutral_band=band)

        banded = apply_neutral_band(probs, band)
        portfolio = compute_portfolio(banded, next_returns, args.direction_horizon, target_vol)
        annual = annual_sharpe_table(dates, portfolio["pnl_modulated"], portfolio["pnl_baseline"])

        overall_sharpe_mod = float(
            portfolio["pnl_modulated"].mean() / (portfolio["pnl_modulated"].std() + 1e-12) * np.sqrt(252)
        )
        overall_sharpe_base = float(
            portfolio["pnl_baseline"].mean() / (portfolio["pnl_baseline"].std() + 1e-12) * np.sqrt(252)
        )

        metrics[split] = {
            "n_days": int(len(dates)),
            "hit_rate": {p: float(hit_rate[i]) for i, p in enumerate(pairs)},
            "confusion_matrix": {
                p: {k: (float(cm[k][i]) if k not in ("tp", "fp", "tn", "fn", "abstained") else int(cm[k][i]))
                    for k in cm} for i, p in enumerate(pairs)
            },
            "annual_sharpe": annual,
            "overall_sharpe_modulated": overall_sharpe_mod,
            "overall_sharpe_baseline": overall_sharpe_base,
            "cumulative_return_modulated": float(portfolio["cumulative_pnl_modulated"][-1]),
            "cumulative_return_baseline": float(portfolio["cumulative_pnl_baseline"][-1]),
        }

        arrays[f"{split}_dates"] = np.array([str(d.date()) for d in dates])
        arrays[f"{split}_probabilities"] = probs
        arrays[f"{split}_labels"] = labels
        arrays[f"{split}_next_returns"] = next_returns
        arrays[f"{split}_z_labels"] = z_labels
        arrays[f"{split}_mu"] = mu
        arrays[f"{split}_sigma"] = sigma
        arrays[f"{split}_cumulative_return_asset"] = np.cumsum(next_returns, axis=0)
        arrays[f"{split}_positions_modulated"] = portfolio["positions_modulated"]
        arrays[f"{split}_positions_baseline"] = portfolio["positions_baseline"]
        arrays[f"{split}_cumulative_pnl_modulated"] = portfolio["cumulative_pnl_modulated"]
        arrays[f"{split}_cumulative_pnl_baseline"] = portfolio["cumulative_pnl_baseline"]
        arrays[f"{split}_cumulative_pnl_per_asset_modulated"] = portfolio["cumulative_pnl_per_asset_modulated"]

    arrays["pairs"] = np.array(pairs)
    metrics["elapsed_sec"] = round(time.time() - t0, 1)
    with open(OUT_DIR / f"{name}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez_compressed(OUT_DIR / f"{name}.npz", **arrays)
    print(f"=== Mode {name} done in {metrics['elapsed_sec']}s ===", flush=True)


if __name__ == "__main__":
    for name, overrides in MODES.items():
        run_mode(name, overrides)
    print("\nAll modes complete.")
