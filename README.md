# fx-forecasting

LSTM portfolio allocator for FX pairs, trained end-to-end to maximize
Sharpe ratio, with an optional risk-attenuation overlay and volatility
targeting. One JSON-driven entry point (`main.py`) runs the whole thing -
data loading, training (or loading a saved model), volatility targeting,
the risk overlay, multi-seed restarts, saving, printing, and plotting.

## 1. Install dependencies

```bash
uv sync
```

## 2. Configure Postgres

All data access goes through `data/db.py`, which expects these environment
variables (a `.env` file in the project root is loaded automatically):

```
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=<your db>
POSTGRES_USER=<your user>
POSTGRES_PASSWORD=<your password>
```

The database must have the `quant.quote_keys`, `quant.market_data` and
`quant.metrics` tables that `data/db.py` reads/writes for price history,
plus `quant.model_registry` for saved models - create it with:

```bash
psql "$DATABASE_URL" -f data/sql/create_model_registry.sql
```

(safe to re-run - uses `CREATE SCHEMA/TABLE IF NOT EXISTS`.)

## 3. (Optional) Pre-populate FX price history

The pipeline auto-downloads and upserts any symbol that isn't in Postgres
yet the first time it's asked for, so this step isn't strictly required.
Run it up front if you'd rather control the download (e.g. more years of
history, or every major pair at once):

```bash
python -m data.fx_downloader --pairs EURUSD GBPUSD USDJPY --years 8 --upload
```

`--upload` is what upserts the downloaded closes into Postgres. Omit
`--pairs` to fetch all seven major pairs.

## 4. Run the pipeline

```bash
python main.py path/to/config.json
```

The JSON file is the **only** input. Every option - data, architecture,
regularization, volatility targeting, the risk overlay, multi-seed
restarts, save/load, plot output paths - is a key in it. Any key left out
falls back to `models/portfolio_lstm.py`'s `DEFAULT_CONFIG`. Only `pairs`
has no default and must always be provided.

Minimal config (everything else takes its default):

```json
{"pairs": ["EURUSD", "GBPUSD", "USDJPY"]}
```

Full config showing every option and its default:

```json
{
    "pairs": ["EURUSD", "GBPUSD", "USDJPY"],

    "lookback": 30,
    "years": 8,
    "train_frac": 0.8,

    "weight_scheme": "softmax",
    "hidden_size": 32,
    "epochs": 300,
    "lr": 0.001,
    "dropout": 0.1,
    "weight_decay": 0.0001,
    "noise_std": 0.05,
    "target_vol": 0.20,

    "n_seeds": 1,
    "restart_strategy": "best",

    "risk_overlay": false,
    "risk_hidden_size": 16,
    "risk_epochs": 200,
    "risk_lr": 0.001,
    "max_attenuation": 0.33,
    "risk_rolling_window": 10,

    "load_portfolio": null,
    "load_risk": null,
    "save_db": false,
    "model_description": "",

    "output": "models/portfolio_pnl.png",
    "position_output": "models/risk_position.png",
    "vol_matched_output": "models/risk_vol_matched_pnl.png"
}
```

### What happens, in order

1. Load the config, merge over defaults, validate `pairs` is present.
2. If `risk_overlay` is `true`: train (or load) PortfolioLSTM and RiskLSTM
   **together** (see below). Otherwise: train (or load) PortfolioLSTM alone.
3. Save whatever was freshly trained to a local `.pt` file, and, if
   `save_db` is `true`, also to Postgres (see step 7) under a name derived
   from the config's characteristics - printed so you can reuse it later
   in another config's `load_portfolio`/`load_risk`.
4. Print Sharpe ratios (raw / vol-targeted / attenuated, as applicable).
5. Save plots: cumulative PnL always; with `risk_overlay`, also
   position-vs-attenuation and the out-of-sample vol-matched comparison.

## 5. PortfolioLSTM: the allocator (Sharpe ratio optimization)

Instead of forecasting a return value, PortfolioLSTM's output **is** the
trading decision - a weight per FX pair - trained end-to-end via
full-batch gradient descent to maximize the Sharpe ratio of the resulting
portfolio.

- `weight_scheme`:
  - `"softmax"` (default) - long-only; weights in (0, 1) summing to 1.
  - `"tanh_norm"` - long/short; weights in (-1, 1), L1-normalized so the
    book is fully invested (sum of `|weight|` == 1); the signed sum floats
    freely in [-1, 1] since allowing shorts rules out also pinning it to 1.

**Volatility targeting** (`target_vol`, default `0.20` = 20% annualized):
right after PortfolioLSTM computes its raw weights, they're uniformly
rescaled per day so the portfolio's annualized volatility - estimated from
the realized covariance of the same lookback window - matches this target.
FX portfolios are naturally low-vol (a few percent annualized), so hitting
20% usually means leveraging weights up (`sum(weights)` can end up well
above 1) - that's the intended effect, not a bug. This scaling is baked
into training itself (the Sharpe objective is computed on the vol-targeted
returns) and into evaluation; the risk overlay only ever reduces these
already-scaled weights further, it never re-scales them. Set `target_vol`
to something tiny (e.g. `1e-6`) to effectively disable it.

**Regularization** (this model overfits easily: full-batch training for
hundreds of epochs on a noisy Sharpe objective will happily memorize the
training period) - three training-time regularizers, on by default (`0`
disables each):

- `noise_std` (default 0.05): fresh Gaussian noise added to the
  standardized input window every epoch, so the model can't fit the exact
  training sequence, only patterns that survive small perturbations of it.
- `dropout` (default 0.1): dropout on the LSTM's final hidden state,
  before the linear head.
- `weight_decay` (default 0.0001): L2 penalty on the model weights (Adam's
  `weight_decay`).

Other techniques worth trying if overfitting is still a problem (not
implemented, to keep the pipeline simple): early stopping on a third
held-out dev split; a turnover penalty (`-|weights[t]-weights[t-1]|`) to
discourage chasing noisy day-to-day signals; a smaller model / shorter
`lookback`; walk-forward cross-validation instead of one fixed split.

**Multi-seed restarts**: the Sharpe-ratio objective is non-convex in the
LSTM's parameters, so different random initializations can land in
meaningfully different local optima. `n_seeds > 1` trains that many
independent restarts on the same data and combines them via
`restart_strategy`:

- `"best"` (default): keeps the single restart with the highest
  validation Sharpe.
- `"ensemble"`: averages every restart's predicted weights (re-normalized
  to keep the weight-scheme invariant - a no-op for `softmax`, a real
  renormalization for `tanh_norm` since restarts can disagree on sign per
  asset). Averaging tends to cancel out each restart's idiosyncratic
  overfitting.

`n_seeds: 1` (the default) skips all of this. Data is loaded once and
reused across every restart, so the cost of `n_seeds: N` is roughly N
training runs, not N full pipelines.

## 6. RiskLSTM: the risk-attenuation overlay (`risk_overlay: true`)

PortfolioLSTM only ever decides *which* assets to hold and in what
proportion - it has no notion of "I'm not confident right now". RiskLSTM's
job is to say *how much* of each proposed position to actually take - one
attenuation factor **per asset**, in `[max_attenuation, 1]`: close to 1 for
an asset in a normal, tradeable period, down towards `max_attenuation` when
that asset's recent behavior looks directionless - so the strategy can
de-risk one pair without necessarily touching the others, and never
zeroes any asset out entirely.

PortfolioLSTM and RiskLSTM are trained **together**, end-to-end, on one
shared objective (not frozen/sequential):

1. PortfolioLSTM proposes raw weights; volatility targeting rescales them
   to `target_vol`.
2. RiskLSTM does NOT read raw log returns. For every trailing
   `risk_rolling_window` days inside the lookback window, it computes each
   asset's rolling **standard deviation, skewness, and excess kurtosis** -
   the moments a risk manager actually looks at (vol = realized risk,
   skewness = asymmetric tail risk, kurtosis = fat-tail/regime
   instability) - feeds that rolling-moment sequence through its own LSTM,
   concatenates the final hidden state with the (vol-targeted) weights,
   and maps to one attenuation factor per asset.
3. `final_weights = vol_targeted_weights * attenuation` (elementwise);
   `portfolio_return = dot(final_weights, next_returns)`.
4. **One optimizer** updates both networks' parameters from the gradient
   of the same (negated) Sharpe ratio of that final return series - so
   RiskLSTM's attenuation can shape what PortfolioLSTM learns to propose,
   and PortfolioLSTM's weights adapt knowing they'll be attenuated
   downstream. `n_seeds`/`restart_strategy` reseed and retrain **both**
   networks together per restart - a restart's seed affects RiskLSTM's
   initialization too, not just PortfolioLSTM's.

**Exception**: if `load_portfolio` is given, that PortfolioLSTM is fixed
(loaded explicitly for inference), so joint training doesn't apply -
`n_seeds` is ignored and RiskLSTM is instead trained alone on top of the
frozen portfolio (or loaded too, via `load_risk`).

- `risk_hidden_size` (default 16): RiskLSTM's hidden size - attenuation is
  a simpler task than picking the portfolio, so this defaults smaller
  than `hidden_size`.
- `risk_epochs`/`risk_lr`: RiskLSTM's own training length/rate, when
  trained jointly these still control the SHARED optimizer's epoch count/rate.
- `max_attenuation` (default 0.33): a hard **floor**, not a ceiling, on
  each asset's attenuation factor, in (0, 1]. Even at minimum confidence
  that asset is never de-risked below this fraction of its proposed
  weight; 1 means no attenuation at all (full-size position).
- `risk_rolling_window` (default 10, must be `< lookback`): the trailing
  window (in days) the rolling volatility/skewness/kurtosis features are
  computed over.

`dropout`/`weight_decay`/`noise_std` are shared between both networks'
training.

## 7. Model persistence: local files or Postgres, by name

Every trained model is saved locally after training:

- `models/portfolio_lstm.pt` (or `models/portfolio_lstm_ensemble.pt` if
  `restart_strategy: "ensemble"` was used)
- `models/risk_lstm.pt` (or `models/risk_lstm_ensemble.pt`) with
  `risk_overlay: true`

Each checkpoint is self-contained: architecture config, weights, and (for
the portfolio model) the exact input standardization stats it was trained
with.

**Database persistence** (`quant.model_registry` - see step 2): set
`"save_db": true` to ALSO persist the trained model(s) to Postgres, under
a name deterministically derived from the config's characteristics (e.g.
`portfolio_hidden_size=32_lookback=30_pairs=EURUSD-GBPUSD-USDJPY_target_vol=0.2_weight_scheme=softmax`)
- printed to the console so you can copy it into another config. The same
training configuration always maps to the same name, so re-saving under
it is a natural update, not a collision.

**Loading**: set `load_portfolio` (and/or `load_risk`) to either

- a local `.pt` file path, or
- a name previously saved with `save_db: true`

`main.py` tries the local file first, then falls back to the database.
Loading skips training entirely for that model (all of
`n_seeds`/`restart_strategy`/`epochs`/architecture options are ignored for
it) and just uses it for inference; it auto-detects whether a checkpoint
holds a single model or an ensemble.

Example - load a previously-trained pair purely by name and just re-plot:

```json
{
    "pairs": ["EURUSD", "GBPUSD", "USDJPY"],
    "lookback": 30,
    "risk_overlay": true,
    "load_portfolio": "portfolio_hidden_size=32_lookback=30_pairs=EURUSD-GBPUSD-USDJPY_target_vol=0.2_weight_scheme=softmax",
    "load_risk": "risk_lookback=30_max_attenuation=0.33_pairs=EURUSD-GBPUSD-USDJPY_risk_hidden_size=16_risk_rolling_window=10"
}
```

`pairs`/`lookback`/`years`/`train_frac` still need to be supplied (they
control what data is fetched and how it's split for evaluation); everything
architecture-related is ignored in favor of what's baked into the checkpoint.

## 8. Plots

Cumulative PnL is always saved (`output`, two files: `<name>_insample.png`
/ `<name>_outsample.png`, each with its Sharpe ratio in the title).

With `risk_overlay: true`, two more are saved:

- `position_output`: per-pair position (portfolio weight, solid line) vs.
  that same pair's attenuation (dashed line, same color, right y-axis
  fixed to [0, 1]) - since attenuation is per-asset, each pair gets a
  matched line pair, so you can see e.g. one pair being de-risked while
  another stays near full size.
- `vol_matched_output`: **out-of-sample only**, three cumulative-PnL
  curves - raw (pre `target_vol`), risk-weighted (post `target_vol`, pre
  risk overlay), and attenuated (post risk overlay) - each independently
  rescaled by its own realized volatility to `target_vol`, so differences
  between the curves reflect differences in shape/skill (drawdowns,
  smoothness) rather than just how much risk each one happened to run.
