"""Single entry point for the FX direction-prediction pipeline.

Usage
-----
    python main.py path/to/config.json

The JSON file is the ONLY input - every option (data, architecture,
regularization, multi-seed restarts, save/load) is a key in it. Any key
left out falls back to models.portfolio_lstm.DEFAULT_CONFIG's default.
Only "pairs" has no default and must always be provided.

Example config - carry and a moving-average-crossover feature added, one
pair (EURUSD) also linked to GBPUSD/USDJPY's own features, 5 restarts,
persisted to the database as well as to a local .pt file:

    {
        "pairs": ["EURUSD", "GBPUSD", "USDJPY"],
        "lookback": 30,
        "direction_horizon": 5,
        "features": ["log_return", "vol", "skew", "kurt", "carry", "cma"],
        "cma_windows": [[10, 50]],
        "cross_pairs": {"EURUSD": ["GBPUSD", "USDJPY"]},
        "n_seeds": 5,
        "save_db": true,
        "model_description": "EUR/GBP/JPY majors, 30d lookback, carry + cma"
    }

`"features"` (see models/portfolio_lstm.py's FEATURE_CATALOG) selects
which per-pair input channels to build; `"cma_windows"` is only used if
`"cma"` is in `"features"`. `"cross_pairs"` (a `{pair: [other_pair, ...]}`
dict) is the ONLY way cross-asset information ever reaches a pair's own
LSTM - by DEFAULT (`{}`) every pair's LSTM sees ONLY its own features,
fully independent; a pair not listed as a `cross_pairs` key stays fully
independent even if OTHER pairs are configured.

"bce_weight" also accepts a LIST instead of a single number, to SWEEP it
alongside the seeds (see models/portfolio_lstm.py's
run_pipeline_multi_seed): every value is trained under every seed (same
initial weights across the sweep, so only the value differs) and
whichever validated best wins, per seed and then overall -

    {
        "pairs": ["EURUSD", "GBPUSD", "USDJPY"],
        "n_seeds": 5,
        "bce_weight": [1.0, 1.5, 1.75, 2.0, 3.0]
    }

A minimal config just needs "pairs" - everything else takes its default:

    {"pairs": ["EURUSD", "GBPUSD", "USDJPY"]}

To load a previously-trained model instead of training (see
models/portfolio_lstm.py's DEFAULT_CONFIG and data/model_registry.py),
set "load_model" to either a local .pt file path or a name previously
saved with "save_db": true.

What happens, in order:
1. Load the config, merge over defaults, validate "pairs" is present.
2. Train (or load) the PredictionModel via models/portfolio_lstm.py's
   run_pipeline_multi_seed - N independent per-asset LSTMs (by default
   each sees only its own features; "cross_pairs" opts specific pairs into
   also seeing specific others'), each trained with its OWN optimizer and
   loss, fully independently of every other asset - see that module's own
   docstring.
3. Save whatever was freshly trained to a local .pt file, and, if
   "save_db" is true, also to Postgres (quant.model_registry) under a
   name derived from the config's characteristics (see
   prediction_model_name()) - printed so it can be reused in a later
   config's "load_model".
4. Print per-asset hit rate and the full confusion-matrix breakdown
   (precision/recall/specificity/F1, plus abstained/coverage - see
   apply_neutral_band) for train, validation, and test.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace

from models.portfolio_lstm import (
    DEFAULT_CONFIG,
    prediction_model_name,
    run_pipeline_multi_seed,
)
from models.portfolio_postprocess import print_confusion_matrices, print_hit_rates


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


def run(args: Namespace) -> None:
    """Train (or load) the PredictionModel: save, print hit rate and
    confusion matrices."""
    result = run_pipeline_multi_seed(args)

    if not args.load_model:
        result.model.save_model(
            x_mean=result.x_mean, x_std=result.x_std, pairs=result.pairs, lookback=result.lookback,
            features=getattr(args, "features", None), cma_windows=getattr(args, "cma_windows", None),
            sigma_hat=result.sigma_hat, neutral_band=result.neutral_band,
        )
    if args.save_db:
        name = prediction_model_name(args)
        result.model.save_to_db(
            name, x_mean=result.x_mean, x_std=result.x_std, pairs=result.pairs, lookback=result.lookback,
            features=getattr(args, "features", None), cma_windows=getattr(args, "cma_windows", None),
            description=args.model_description,
            sigma_hat=result.sigma_hat, neutral_band=result.neutral_band,
        )
        print(f"Persisted model to database as {name!r}")

    print_hit_rates(result)
    print_confusion_matrices(result)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python main.py path/to/config.json", file=sys.stderr)
        sys.exit(1)

    args = load_config(sys.argv[1])
    run(args)


if __name__ == "__main__":
    main()
