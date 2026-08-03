"""Generates docs/methodology/figures/*.pdf from docs/methodology/results/*
(see run_experiments.py), styled to match the web app's own chart palette
(frontend/src/theme.js) as closely as matplotlib allows.

Run from the repo root: python docs/methodology/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# --- frontend/src/theme.js, verbatim ---
HIT_COLOR = "#059669"
MISS_COLOR = "#dc2626"
ABSTAIN_COLOR = "#94a3b8"
PAIR_PALETTE = ["#2563eb", "#d97706", "#059669", "#dc2626", "#7c3aed", "#0891b2", "#db2777", "#65a30d"]
MODULATED_COLOR = "#2563eb"
BASELINE_COLOR = "#94a3b8"
SPLIT_COLOR = {"train": "#0891b2", "val": "#db2777", "test": "#65a30d"}
GRID_COLOR = "#e2e8f0"
AXIS_COLOR = "#94a3b8"
PROBABILITY_COLOR = "#2563eb"
ACTUAL_DIST_COLOR = "#7c3aed"
FORECAST_DIST_COLOR = "#d97706"

MODES = ["A_baseline", "B_enriched", "C_pnl_mode"]
MODE_LABEL = {"A_baseline": "A: Baseline", "B_enriched": "B: Enriched features", "C_pnl_mode": "C: PnL-complementary"}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID_COLOR, "axes.linewidth": 0.8,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, zorder=0)
    ax.tick_params(colors=AXIS_COLOR, labelsize=8)


def load(mode):
    metrics = json.load(open(RESULTS_DIR / f"{mode}.json"))
    arrays = np.load(RESULTS_DIR / f"{mode}.npz", allow_pickle=True)
    return metrics, arrays


# --- Figure 1: hit rate, test split, grouped by mode x pair ---
def fig_hit_rate():
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    pairs = None
    width = 0.25
    for mi, mode in enumerate(MODES):
        metrics, _ = load(mode)
        hr = metrics["test"]["hit_rate"]
        if pairs is None:
            pairs = list(hr.keys())
        x = np.arange(len(pairs)) + (mi - 1) * width
        ax.bar(x, [hr[p] for p in pairs], width=width * 0.9, label=MODE_LABEL[mode],
               color=PAIR_PALETTE[mi * 2 % len(PAIR_PALETTE)], zorder=3)
    ax.axhline(0.5, color=AXIS_COLOR, linestyle="--", linewidth=1, zorder=2)
    ax.set_xticks(np.arange(len(pairs)))
    ax.set_xticklabels(pairs)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Directional hit rate")
    ax.set_title("Test-split hit rate by mode and pair")
    _style_ax(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.savefig(FIG_DIR / "hit_rate_by_mode.pdf")
    plt.close(fig)


# --- Figure 2: cumulative return colored by hit/miss, EURUSD, test, mode C ---
def fig_colored_return(mode="C_pnl_mode", pair="EURUSD", split="test"):
    metrics, arrays = load(mode)
    pairs = list(arrays["pairs"])
    i = pairs.index(pair)
    dates = np.array([np.datetime64(d) for d in arrays[f"{split}_dates"]])
    cum = arrays[f"{split}_cumulative_return_asset"][:, i]
    probs = arrays[f"{split}_probabilities"][:, i]
    labels = arrays[f"{split}_labels"][:, i]
    band = 0.05
    hit = (probs > 0.5) == (labels > 0.5)
    abstained = np.abs(probs - 0.5) < band

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.plot(dates, cum, color=BASELINE_COLOR, linewidth=1, zorder=2)
    colors = np.where(abstained, ABSTAIN_COLOR, np.where(hit, HIT_COLOR, MISS_COLOR))
    ax.scatter(dates, cum, c=colors, s=7, zorder=3, linewidths=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_title(f"{pair} cumulative return, colored by prediction hit/miss ({MODE_LABEL[mode]}, {split})")
    ax.set_ylabel("Cumulative log return")
    _style_ax(ax)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=HIT_COLOR, markersize=6, label="Hit"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=MISS_COLOR, markersize=6, label="Miss"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=ABSTAIN_COLOR, markersize=6, label="Abstained"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="upper left")
    fig.autofmt_xdate()
    fig.savefig(FIG_DIR / f"colored_return_{pair}_{mode}.pdf")
    plt.close(fig)


# --- Figure 3: portfolio PnL - total modulated vs baseline + per-asset, test split ---
def fig_portfolio_pnl(mode="C_pnl_mode", split="test"):
    metrics, arrays = load(mode)
    pairs = list(arrays["pairs"])
    dates = np.array([np.datetime64(d) for d in arrays[f"{split}_dates"]])
    cum_mod = arrays[f"{split}_cumulative_pnl_modulated"]
    cum_base = arrays[f"{split}_cumulative_pnl_baseline"]
    cum_per_asset = arrays[f"{split}_cumulative_pnl_per_asset_modulated"]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, pair in enumerate(pairs):
        ax.plot(dates, cum_per_asset[:, i], color=PAIR_PALETTE[i % len(PAIR_PALETTE)],
                linewidth=1, label=f"{pair} (modulated)", zorder=2)
    ax.plot(dates, cum_mod, color=MODULATED_COLOR, linewidth=2, linestyle=(0, (5, 2)), label="Total (modulated)", zorder=3)
    ax.plot(dates, cum_base, color=BASELINE_COLOR, linewidth=2, linestyle=(0, (5, 2)), label="Total (risk parity, unmodulated)", zorder=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_title(f"Portfolio PnL, {split} split ({MODE_LABEL[mode]})")
    ax.set_ylabel("Cumulative PnL")
    _style_ax(ax)
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
    fig.autofmt_xdate()
    fig.savefig(FIG_DIR / f"portfolio_pnl_{mode}_{split}.pdf")
    plt.close(fig)


# --- Figure 4: predicted probability path, EURUSD, test, mode C ---
def fig_probability(mode="C_pnl_mode", pair="EURUSD", split="test"):
    metrics, arrays = load(mode)
    pairs = list(arrays["pairs"])
    i = pairs.index(pair)
    dates = np.array([np.datetime64(d) for d in arrays[f"{split}_dates"]])
    probs = arrays[f"{split}_probabilities"][:, i]

    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.plot(dates, probs, color=PROBABILITY_COLOR, linewidth=1)
    ax.axhline(0.5, color=AXIS_COLOR, linestyle="--", linewidth=1)
    ax.fill_between(dates, 0.45, 0.55, color=ABSTAIN_COLOR, alpha=0.25, linewidth=0, label="Neutral band")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_title(f"{pair} predicted P(positive) ({MODE_LABEL[mode]}, {split})")
    _style_ax(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.autofmt_xdate()
    fig.savefig(FIG_DIR / f"probability_{pair}_{mode}.pdf")
    plt.close(fig)


# --- Figure 5: forecast vs actual distribution, EURUSD, test, mode C ---
def fig_distribution(mode="C_pnl_mode", pair="EURUSD", split="test"):
    rng = np.random.default_rng(0)
    metrics, arrays = load(mode)
    pairs = list(arrays["pairs"])
    i = pairs.index(pair)
    z_actual = arrays[f"{split}_z_labels"][:, i]
    mu = arrays[f"{split}_mu"][:, i]
    sigma = arrays[f"{split}_sigma"][:, i]
    forecasted = rng.normal(mu, sigma)

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    bins = np.linspace(min(z_actual.min(), forecasted.min()), max(z_actual.max(), forecasted.max()), 30)
    ax.hist(z_actual, bins=bins, color=ACTUAL_DIST_COLOR, alpha=0.6, label="Actual", zorder=2)
    ax.hist(forecasted, bins=bins, color=FORECAST_DIST_COLOR, alpha=0.6, label="Forecasted", zorder=2)
    ax.set_title(f"{pair} forecasted vs actual z-score distribution ({MODE_LABEL[mode]}, {split})")
    _style_ax(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIG_DIR / f"distribution_{pair}_{mode}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_hit_rate()
    fig_colored_return()
    fig_portfolio_pnl()
    fig_probability()
    fig_distribution()
    print("Figures written to", FIG_DIR)
