"""Single entry point for the FX portfolio / risk-overlay pipeline.

Usage
-----
    python main.py path/to/config.json

The JSON file is the ONLY input - every training/evaluation option (data,
architecture, regularization, volatility targeting, the risk overlay,
multi-seed restarts, save/load, plot output paths) is a key in it. Any key
left out falls back to models.portfolio_lstm.DEFAULT_CONFIG's default.
Only "pairs" has no default and must always be provided.

Example config - risk overlay on top of an ensemble of 5 PortfolioLSTM
restarts, persisted to the database as well as to a local .pt file:

    {
        "pairs": ["EURUSD", "GBPUSD", "USDJPY"],
        "lookback": 30,
        "position_mode": "long_short",
        "epochs": 300,
        "target_vol": 0.20,
        "risk_overlay": true,
        "risk_hidden_size": 16,
        "risk_epochs": 200,
        "risk_rolling_window": 10,
        "n_seeds": 5,
        "restart_strategy": "ensemble",
        "save_db": true,
        "model_description": "EUR/GBP/JPY majors, 30d lookback, ensemble-of-5",
        "output": "models/risk_pnl.png"
    }

A minimal config just needs "pairs" - everything else takes its default:

    {"pairs": ["EURUSD", "GBPUSD", "USDJPY"]}

To load a previously-trained model instead of training (see
models/portfolio_lstm.py's DEFAULT_CONFIG and data/model_registry.py),
set "load_portfolio" (and/or "load_risk") to either a local .pt file path
or a name previously saved with "save_db": true.

What happens, in order:
1. Load the config, merge over defaults, validate "pairs" is present.
2. Train (or load) PortfolioLSTM via models/portfolio_lstm.py's
   run_pipeline_multi_seed - completely independent of the risk overlay,
   whatever "n_seeds"/"restart_strategy"/"objective" apply, they apply to
   PortfolioLSTM alone. If "risk_overlay" is true, that (now fixed, frozen)
   result is then handed to models/risk_lstm.py's run_pipeline_multi_seed,
   which trains (or loads) RiskLSTM SEPARATELY on top of it to improve its
   realized Sharpe (see that module's docstring) - the risk overlay never
   feeds back into how PortfolioLSTM itself was trained.
3. Save whatever was freshly trained to a local .pt file, and, if
   "save_db" is true, also to Postgres (quant.model_registry) under a name
   derived from the config's characteristics (see portfolio_model_name()/
   risk_model_name()) - printed so it can be reused in a later config's
   "load_portfolio"/"load_risk".
4. Print Sharpe ratios (raw / vol-targeted / attenuated as applicable).
5. Save plots: cumulative PnL always; for a risk overlay, also
   position-vs-attenuation and the out-of-sample vol-matched 3-way
   comparison.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace

from models.portfolio_lstm import (
    DEFAULT_CONFIG,
    portfolio_model_name,
    run_pipeline_multi_seed as run_portfolio_pipeline,
)
from models.portfolio_postprocess import (
    plot_pnl as plot_portfolio_pnl,
    print_sharpe_ratios as print_portfolio_sharpe_ratios,
)


def load_config(path: str) -> Namespace:
    """Read a JSON config file and merge it over DEFAULT_CONFIG."""
    with open(path) as f:
        user_config = json.load(f)

    config = {**DEFAULT_CONFIG, **user_config}
    if not config.get("pairs"):
        raise ValueError(
            'Config must include a non-empty "pairs" list, e.g. ["EURUSD", "GBPUSD", "USDJPY"]'
        )
    return Namespace(**config)


def run_portfolio_only(args: Namespace) -> None:
    """Train (or load) PortfolioLSTM alone: save, print Sharpe, plot PnL."""
    result = run_portfolio_pipeline(args)

    if not args.load_portfolio:
        result.model.save_model(
            x_mean=result.x_mean, x_std=result.x_std, pairs=result.pairs, lookback=result.lookback,
        )
    if args.save_db:
        name = portfolio_model_name(args)
        result.model.save_to_db(
            name, x_mean=result.x_mean, x_std=result.x_std, pairs=result.pairs, lookback=result.lookback,
            description=args.model_description,
        )
        print(f"Persisted portfolio model to database as {name!r}")

    print_portfolio_sharpe_ratios(result)
    plot_portfolio_pnl(result, args.output)


def run_with_risk_overlay(args: Namespace) -> None:
    """Train (or load) PortfolioLSTM, then train (or load) RiskLSTM
    SEPARATELY on top of it (see models/risk_lstm.py's run_pipeline_multi_seed):
    save, print Sharpe, plot PnL + position/attenuation + the vol-matched
    comparison.

    Deferred import: models/risk_lstm.py imports from models/portfolio_lstm.py
    at module load time, so importing it back at this module's top level
    would be a circular import.
    """
    from models.risk_lstm import risk_model_name, run_pipeline_multi_seed as run_risk_overlay_pipeline
    from models.risk_postprocess import (
        plot_pnl as plot_risk_pnl,
        plot_position_and_scaling,
        plot_return_histograms,
        plot_transaction_cost_pnl,
        plot_vol_matched_pnl,
        print_sharpe_ratios as print_risk_sharpe_ratios,
    )

    result = run_risk_overlay_pipeline(args)

    if not args.load_portfolio:
        result.portfolio_result.model.save_model(
            x_mean=result.portfolio_result.x_mean, x_std=result.portfolio_result.x_std,
            pairs=result.portfolio_result.pairs, lookback=result.portfolio_result.lookback,
        )
    if not args.load_risk:
        result.risk_model.save_model(pairs=result.portfolio_result.pairs)
    if args.save_db:
        portfolio_name = portfolio_model_name(args)
        result.portfolio_result.model.save_to_db(
            portfolio_name,
            x_mean=result.portfolio_result.x_mean, x_std=result.portfolio_result.x_std,
            pairs=result.portfolio_result.pairs, lookback=result.portfolio_result.lookback,
            description=args.model_description,
        )
        risk_name = risk_model_name(args)
        result.risk_model.save_to_db(
            risk_name, pairs=result.portfolio_result.pairs, description=args.model_description,
        )
        print(f"Persisted portfolio model to database as {portfolio_name!r}")
        print(f"Persisted risk model to database as {risk_name!r}")

    print_risk_sharpe_ratios(result)
    plot_risk_pnl(result, args.output)
    plot_position_and_scaling(result, args.position_output)
    plot_vol_matched_pnl(result, args.vol_matched_output, target_vol=args.target_vol)
    plot_return_histograms(result, args.histogram_output)
    plot_transaction_cost_pnl(result, args.transaction_cost_output, transaction_cost_bps=args.transaction_cost)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python main.py path/to/config.json", file=sys.stderr)
        sys.exit(1)

    args = load_config(sys.argv[1])

    if args.risk_overlay:
        run_with_risk_overlay(args)
    else:
        run_portfolio_only(args)


if __name__ == "__main__":
    main()
