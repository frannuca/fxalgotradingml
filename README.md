# fx-forecasting

A probabilistic direction predictor for FX pairs: N independent per-asset
LSTMs (no weights shared between assets) each predict the probability
their own pair's return is positive over the next few days. By default
each pair's LSTM sees ONLY its own features - fully independent; a
configurable `cross_pairs` link (see step 5) opts specific pairs into also
seeing specific other pairs' full feature blocks, so cross-asset
correlation can be learned directly inside an independent LSTM without a
separate cross-asset stage, wherever (and only where) it's explicitly
asked for. Feature selection is itself configurable too - raw log return,
rolling vol/skew/kurtosis, carry, and configurable moving-average-
crossover ("cma") channels, see step 5. Pure prediction - no portfolio
weights, no PnL, no trading decision anywhere in this codebase; everything
is measured as hit rate and confusion-matrix metrics against realized
direction. Two ways to run it:

- **CLI**: a JSON-driven entry point (`main.py`) - see steps 1-7 below.
- **Web app**: a FastAPI backend (`api/`) + React frontend (`frontend/`)
  with a Training page (set parameters, train, save) and an Evaluation
  page (pick a saved model, refresh quotes, run, view hit rate/confusion
  matrices/colored return charts) - see step 8.

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

The JSON file is the **only** input. Every option - data, features,
architecture, regularization, multi-seed restarts, save/load - is a key
in it. Any key left out falls back to `models/portfolio_lstm.py`'s
`DEFAULT_CONFIG`. Only `pairs` has no default and must always be provided.

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
    "test_frac": 0.1,
    "direction_horizon": 5,
    "rolling_stats_window": 20,
    "features": ["log_return", "vol", "skew", "kurt"],
    "cma_windows": [],
    "cross_pairs": {},

    "hidden_size": 16,
    "num_layers": 1,
    "dropout": 0.1,
    "n_attn_heads": 4,

    "epochs": 300,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "bce_weight": 1.0,
    "neutral_band": 0.05,

    "n_seeds": 1,
    "device": "auto",

    "load_model": null,
    "save_db": false,
    "model_description": ""
}
```

### What happens, in order

1. Load the config, merge over defaults, validate `pairs` is present.
2. Train (or load) the `PredictionModel` via `models/portfolio_lstm.py`'s
   `run_pipeline_multi_seed` - N independent per-asset LSTMs, trained
   together in one optimizer/one backward pass per epoch (see step 5).
3. Save whatever was freshly trained to a local `.pt` file, and, if
   `save_db` is `true`, also to Postgres (see step 6) under a name derived
   from the config's characteristics - printed so you can reuse it later
   in another config's `load_model`.
4. Print per-asset hit rate and the full confusion-matrix breakdown
   (precision/recall/specificity/F1, plus abstained/coverage - see step 5's
   "neutral band" section) for train, validation, and test.

## 5. Architecture: N independent per-asset LSTMs, optionally cross-linked

**The prediction target is a Z-SCORE, not a binary label**: for every day
`d`, the label is that asset's cumulative log return over
`[d+1, d+direction_horizon]`, divided by its own trailing volatility (the
SAME `rolling_stats_window`-day rolling std used as an input feature,
scaled by `sqrt(direction_horizon)`) - "how many standard deviations did
this move by, relative to how volatile this asset has recently been".
Unlike a thresholded up/down label, this keeps each move's MAGNITUDE, not
just its sign, and normalizes it consistently across assets/regimes (see
`models/portfolio_lstm.py`'s `make_sequences`/`_forward_zscore_labels`).

**`PredictionModel`/`AssetLSTM`**: N **independent** per-asset LSTMs - no
parameters shared between assets, since different pairs can have
genuinely different dynamics. By DEFAULT, each asset's own LSTM sees ONLY
its own feature block - fully independent, no cross-asset information at
all. `cross_pairs` (a `{pair: [other_pair, ...]}` dict, default `{}`) opts
SPECIFIC pairs into ALSO seeing SPECIFIC other pairs' full feature blocks
at every timestep - e.g. `{"EURUSD": ["GBPUSD", "USDJPY"]}` gives EURUSD's
LSTM the concatenation of its own, GBPUSD's, and USDJPY's channels (own
features are always included regardless of `cross_pairs`); GBPUSD and
USDJPY, absent as their own key, stay fully independent. This is a
per-pair, explicit, opt-in choice - not a global switch - because whether
mixing in another pair's raw features actually helps is an empirical
question that can differ pair by pair; compare validation loss/hit rate
with and without a given `cross_pairs` link before assuming it does. (An earlier
version of this code had a DIFFERENT, non-optional way of getting
cross-asset information - first a separate `CopulaLSTM` stage mixing each
asset's already-compressed z-score across assets (a bottleneck - it never
saw raw features), then, after removing that stage, ALL pairs seeing ALL
other pairs' full feature blocks unconditionally. `cross_pairs` replaces
that blanket default with an explicit per-pair choice.)

Each `AssetLSTM`'s recurrent output is followed by a **causal self-attention
layer** over the time axis (masked so day k only attends to days <= k)
before the final head - a second route to long-range within-window
dependencies beyond whatever the LSTM's fixed-size recurrent state
carries forward. The mask matters: this network outputs a prediction at
EVERY day in the window (see below), each one supervised as if computed
using only information through that day, so an unmasked (bidirectional)
attention layer would let day k's prediction see days AFTER k - exactly
the label look-ahead leakage the rest of this module (purge/embargo,
trailing-only features) is built to avoid. The whole network is
**deterministic**: no NoisyNet head, no input-noise regularization -
dropout (training mode only) is the only stochasticity.

Each `AssetLSTM` predicts a heteroscedastic **(mu, sigma)** pair at EVERY
day in the window, not just the last one (a dense, per-day, per-asset
supervised signal): `mu` is the predicted conditional MEAN of that day's
forward z-score, `sigma` its predicted conditional STD (strictly
positive).

- `direction_horizon` (default 5): how many days ahead the label looks.
- `rolling_stats_window` (default 20): trailing window the rolling vol/
  skew/kurtosis input features AND the z-score label's own volatility
  normalization are computed over (the latter regardless of whether "vol"
  is itself a selected feature).
- `features` (default `["log_return", "vol", "skew", "kurt"]`): which
  per-pair input channels to build - see `models/portfolio_lstm.py`'s
  `FEATURE_CATALOG` for the full list (also `"carry"`: the interest-rate
  differential, base minus quote, from FRED via
  `data/rates_downloader.py` - the single best-documented FX predictor;
  `"cma"` and `"bandpass"`: see below).
- `cma_windows` (default `[]`, e.g. `[[10, 50], [20, 100]]`): only used if
  `"cma"` is in `features` - one input channel PER `[short, long]` window
  pair, each `rolling_mean(returns, short) - rolling_mean(returns, long)`
  (see `cross_moving_averages`) - a trailing moving-average-crossover /
  trend signal in return space, positive when the recent trend runs above
  the longer-run one.
- `bandpass_windows`/`bandpass_order` (default `[]`/`3`): only used if
  `"bandpass"` is in `features` - one input channel PER `[short_period,
  long_period]` (in days), a CAUSAL Butterworth band-pass filter of returns
  (see `butterworth_bandpass_features`) - a faster-reacting alternative to
  a CMA crossover for the same window pair. Uses `scipy.signal.lfilter`
  (forward-only), deliberately NOT `filtfilt` (the usual zero-phase way to
  apply a Butterworth filter) - `filtfilt` runs the filter backward too,
  which would leak future returns into today's feature value.
  `short_period` sets the fastest cycle length that still passes (the high
  cutoff); `long_period` sets the slowest (the low cutoff - removes the
  long-run trend/DC component). `bandpass_order` (default 3) trades lag
  against selectivity - higher rolls off more sharply but adds phase lag.
- `cross_pairs` (default `{}`): see above - which OTHER pairs' feature
  blocks additionally feed a given pair's own LSTM.
- `n_attn_heads` (default 4): attention heads in each asset's causal
  self-attention layer - `hidden_size` must be divisible by this.

**Loss function - Gaussian NLL + a BCE anti-collapse term, not Huber**:
every asset's LSTM is trained with `gaussian_nll(mu, sigma, z_label)` +
`bce_weight * direction_bce(mu, sigma, z_label)`, applied densely over
every (sample, day, asset) triple. NLL fits the full predictive
distribution (mu toward the conditional mean, sigma toward the actual
residual spread, per sample - so quiet-regime confidence and wild-regime
uncertainty both become expressible). BCE is the anti-collapse term: a
bare regression loss is minimized by the label's unconditional mean - ONE
sign for every sample when the target is barely predictable, which is
exactly the degenerate all-recall/zero-specificity confusion matrix -
while BCE is minimized by matching each sample's OWN sign, forcing
mu/sigma to spread across 0 wherever the features actually discriminate.
`bce_weight=0` recovers pure distributional regression (collapse risk and
all).

**Recovering a probability - the probit link, calibrated on validation**:
`probit(z)` (the standard normal CDF, `models/portfolio_lstm.py`'s
`probit`) is the standard link from a z-score to a probability.
`P(positive) = probit(mu / (sigma * sigma_hat))`:
- `mu / sigma` is the model's own per-sample signal-to-noise ratio -
  confidence varies day by day, unlike a single global residual scale
  ever could;
- `sigma_hat` is a residual GLOBAL calibration factor: the std of the
  standardized residual `(z - mu) / sigma`, estimated ONCE on the
  VALIDATION split only (never train, which would fit the calibration to
  noise the model already overfit to; never test, which must stay
  untouched by anything baked into the model). If the NLL-trained sigmas
  are already honest, `sigma_hat ~ 1` and this is a no-op; if they are
  collectively over/under-confident, `sigma_hat` corrects the shared
  factor. Persisted in the checkpoint (see step 6) so live single-window
  inference, which has no validation set to refit against, reuses the
  training-time estimate.

`z = 0` still maps to exactly `0.5` regardless of `sigma_hat`.

**Neutral band - abstaining on thin edge, PURE postprocessing**:
probabilities inside `0.5 ± neutral_band` (default 0.05) count as
ABSTENTION (see `apply_neutral_band`) rather than a near-coin-flip
directional call. Hit rate/confusion-matrix metrics are then computed
over DECIDED samples only, alongside a `coverage` metric (how often the
model actually makes a call) - accuracy and coverage must always be read
together (100% accuracy at 2% coverage is a very different claim from 55%
at 80%). `neutral_band: 0` disables abstention entirely.

Critically, the band is NEVER part of training - `train_prediction_model`
never receives it, and `evaluate_prediction_model` stores
`probabilities_train/val/test` RAW (unsnapped); the band only enters when
`confusion_matrix_metrics`/`_decided_hit_rate` are computed from those raw
probabilities. This means a confusion matrix/hit rate/colored-return chart
for a DIFFERENT band never requires retraining or even a new evaluation
pass - the web app's Training/Evaluation pages expose a live "Neutral
band" control that recomputes all three entirely in the browser (see
`frontend/src/metrics.js`, a JS port of the same threshold logic,
numerically verified against the Python implementation) from the raw
probability + realized label pairs the API ships alongside every result.

**Training is FULLY INDEPENDENT per asset**: each asset's LSTM gets its
OWN `torch.optim.Adam` instance (separate momentum/variance state) and
its OWN loss, computed from just that asset's own `(mu, sigma)` and
label - never averaged or summed with any other asset's before its own
`backward()`/`optimizer.step()` call. Applied to the FULL dense
`(batch, lookback)` output for that asset - not just the window's last
("decision") day - since every day in every window is its own supervised
sample, and consecutive windows slide by one day, so a given calendar day
is trained on repeatedly rather than just once. The assets never shared
weights (see step 5's own architecture section) - this decouples the
OPTIMIZATION too: an asset's own gradient magnitude, Adam's per-parameter
adaptive state, and which epoch its own checkpoint gets restored from
depend on nothing but that asset's own loss curve, regardless of how many
other assets are being trained alongside it in the same
`PredictionModel`, or how they're doing. An asset linked to another pair's
features via `cross_pairs` still trains this way - `cross_pairs` only
widens what its LSTM READS, never what optimizes it. Checkpoint selection
keeps, per asset, whichever epoch had THAT asset's own LOWEST validation
loss (computed on the decision day only, matching what's actually
evaluated/reported) - not just whichever epoch the fixed epoch budget
happened to end on, not the SAME epoch forced across every asset, and
deliberately not validation hit rate, which saturates once every sample's
sign is already correct and so can freeze onto a very early,
barely-trained checkpoint while discarding every later epoch that kept
getting more confident/better-calibrated without being able to push
accuracy any higher.

**Purge/embargo at split boundaries**: sequences are stride-1, so
consecutive samples share most of their forward label window and lookback
input window with their neighbors. Right at the train/validation or
validation/test boundary, this means information the model was trained to
predict would otherwise leak into what the next split "sees" as recent
history, optimistically biasing the validation loss that drives
checkpoint and seed selection (purged cross-validation, per López de
Prado). `_prepare_data` drops `lookback + direction_horizon` samples
entirely (not reassigned to either split) at each boundary before any
model ever sees the data.

**Multi-seed restarts, and sweeping `bce_weight` alongside them**:
training is non-convex in the LSTMs' parameters regardless of loss
function, so different random initializations can land in meaningfully
different local optima. `n_seeds > 1` trains that many independent
restarts on the same data and keeps whichever had the lowest validation
log loss (binary cross-entropy of the probit-converted probabilities
against the realized labels - same "don't select on a saturating metric"
reasoning as epoch checkpoint selection above). `n_seeds: 1` (the
default) skips this - data is loaded once and reused across every
restart, so the cost of `n_seeds: N` is roughly N training runs, not N
full pipelines.

`bce_weight` accepts EITHER a single number OR a **list**, e.g.
`"bce_weight": [1.0, 1.5, 1.75, 2.0, 3.0]`, to sweep it together with the
seeds via `run_pipeline_multi_seed`: every value is trained under EVERY
seed - with `torch.manual_seed(seed)` reset right before each one, so
every value in the sweep starts from the SAME initial weights and sees
the SAME input-noise draws for that seed, isolating the value's own
effect from initialization luck - and whichever value validated best
wins, first per seed, then the overall best across seeds. This lets
validation performance pick the NLL/anti-collapse tradeoff instead of
committing to one value a priori. The cost is `n_seeds * len(bce_weight)`
training runs.

## 6. Model persistence: local files or Postgres, by name

Every asset's LSTM is saved TOGETHER as one checkpoint after
training:

- `models/prediction_model.pt` - every asset's LSTM weights, architecture
  config, input standardization stats, the ordered FX pairs, the sequence
  length, and the validation-fit calibration scale (`sigma_hat` - see
  step 5's "probit link" section): self-contained, enough to rebuild and
  run calibrated inference on new data without retraining anything.

**Database persistence** (`quant.model_registry` - see step 2): set
`"save_db": true` to ALSO persist the trained model to Postgres, under a
name deterministically derived from the config's characteristics (e.g.
`prediction_direction_horizon=5_features=kurt-log_return-skew-vol_hidden_size=16_lookback=30_pairs=EURUSD-GBPUSD-USDJPY`)
- printed to the console so you can copy it into another config. The same
training configuration always maps to the same name, so re-saving under
it is a natural update, not a collision.

**Loading**: set `load_model` to either

- a local `.pt` file path, or
- a name previously saved with `save_db: true`

`main.py` tries the local file first, then falls back to the database.
Loading skips training entirely and just uses the checkpoint for
inference - `pairs`/`lookback`/architecture options are all ignored in
favor of what's baked into it.

Example - load a previously-trained model purely by name and re-evaluate:

```json
{
    "pairs": ["EURUSD", "GBPUSD", "USDJPY"],
    "lookback": 30,
    "load_model": "prediction_direction_horizon=5_features=kurt-log_return-skew-vol_hidden_size=16_lookback=30_pairs=EURUSD-GBPUSD-USDJPY"
}
```

`pairs`/`lookback`/`years`/`train_frac`/`test_frac` still need to be
supplied (they control what data is fetched and how it's split for
evaluation); everything architecture-related is ignored in favor of
what's baked into the checkpoint.

## 7. Evaluation metrics: hit rate and confusion matrix

Everything is reported per asset, for all three splits, from
`models/portfolio_lstm.py`'s `confusion_matrix_metrics` - over DECIDED
samples only (see step 5's "neutral band" section):

- **hit rate** (== accuracy): the fraction of DECIDED samples where
  `(probability > 0.5)` matched the realized `direction_horizon`-day-
  forward label. 0.5 is random chance.
- **precision**: of decided days predicted positive, how many actually were.
- **recall (sensitivity)**: of decided days actually positive, how many
  were predicted positive.
- **specificity**: of decided days actually negative, how many were
  predicted negative.
- **F1**: harmonic mean of precision and recall.
- **abstained**/**coverage**: how many samples fell inside the neutral
  band (no call made) and their complement's share - must be read
  alongside accuracy, since 100% accuracy at 2% coverage is a very
  different claim from 55% at 80%.

`main.py` prints all of these (see `models/portfolio_postprocess.py`);
the web app's Training/Evaluation pages show them as a bar chart (hit
rate) and a table (full confusion matrix, including abstained/coverage),
plus a per-asset cumulative-return chart with each day colored green
(predicted direction correct), red (wrong), or slate (abstained).

**Forecasted vs actual return distribution**: a per-asset, per-split
histogram comparing the realized decision-day z-scores (`actual`) against
one sample drawn from each row's own model-predicted `N(mu, sigma)`
(`forecasted`, sigma already calibrated by `sigma_hat`) - see
`api/server.py`'s `_distribution_payload`. This is a distributional check,
not a point-accuracy one: if the model's predictive distributions are
well-calibrated, the two histograms should look statistically similar in
center/spread/shape even though no individual pair of values need match -
a systematically narrower "forecasted" histogram means the model is
overconfident (understating its own uncertainty), a shifted one means a
biased `mu`.

## 8. Web app: Training + Evaluation in the browser

A FastAPI backend (`api/`) and React frontend (`frontend/`) wrap the same
pipeline in a UI - useful for setting parameters interactively and
browsing results without re-running the CLI each time.

**Backend** (run from the repo root, same Postgres env vars as above):

```bash
uvicorn api.server:app --reload --port 8000
```

**Frontend**:

```bash
cd frontend
npm install
npm run dev
```

Then open the printed local URL (typically `http://localhost:5173`). If
the backend isn't on `http://127.0.0.1:8000`, copy `frontend/.env.example`
to `frontend/.env` and set `VITE_API_BASE_URL`.

**Training page**: pick FX pairs (from the majors, or type a custom
ticker), set data/feature/architecture/training/multi-seed parameters,
and train. Training runs as a background job on the server (polled every 2s)
since it can take longer than one HTTP request should block for; on
completion the trained model is saved locally and, if requested, to
`quant.model_registry`. Results show per-asset hit rate, the full
confusion matrix, and per-asset cumulative-return charts colored by
prediction hit/miss, for train, validation, and test.

**Evaluation page**: pick a saved model from a dropdown populated from
`quant.model_registry`, optionally refresh its pairs' latest quotes into
Postgres, set evaluation parameters (years/train_frac/test_frac), and
run. Loads the model purely for inference (no training) and shows the
same hit rate/confusion matrix/colored cumulative-return charts, plus the
model's predicted probability per asset for the next (not-yet-realized)
day.

**Portfolio PnL calculator** (`models/portfolio_pnl.py`, evaluation-only -
never touches training): turns a model's per-asset probabilities into a
long/short book.

1. **Risk parity**: each day's realized returns (trailing 60-day window,
   causal - never looks at the day being sized) fit a covariance matrix;
   `risk_parity_weights` solves for the long-only, sum-to-1 weights that
   equalize every asset's contribution to total portfolio variance (not
   just inverse-vol - it accounts for correlation).
2. **Probability modulation**: each risk-parity weight is multiplied by
   that asset's signal `(p - 0.5) * 2` (so `p < 0.5` flips the position
   short, magnitude scaled by conviction).
3. **Horizon smoothing**: the portfolio rebalances daily, but the
   probability forecasts a `direction_horizon`-day-forward move, so
   consecutive days' signals are heavily overlapping forecasts. The
   modulated weight is smoothed with a trailing `direction_horizon`-day
   moving average before scaling - the standard treatment for
   overlapping-horizon forecasts - so the book reflects the model's
   persistent view rather than day-to-day jitter in exactly when the
   window rolled.
4. **Target-vol scaling**: the smoothed weight is scaled so the book's own
   `w' C w` hits a configured annualized `target_vol` (10% by default) -
   a model property, set at training time and persisted in its checkpoint
   alongside `neutral_band`, so evaluation always uses what the model was
   actually meant to be traded at.

The unmodulated risk-parity weights (no probability signal, same
target-vol scaling) are shown alongside as a baseline for comparison. The
Evaluation page plots both, per asset and as whole-book cumulative PnL,
for train/val/test, and shows a "today's position" table - the
not-yet-booked position and probability per asset a trader would book for
today's EoD, sized from the freshest available data.

See `api/server.py`'s module docstring for the full endpoint list
(`GET /api/pairs`, `POST /api/quotes/refresh`, `GET /api/models`,
`POST /api/train` + `GET /api/train/{job_id}`, `POST /api/evaluate`).
