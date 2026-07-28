# fx-forecasting

LSTM forecasting of FX pair log returns.

## 1. Install dependencies

```bash
uv sync
```

## 2. Configure Postgres

All scripts read/write price history through `data/db.py`, which expects
these environment variables (a `.env` file in the project root is loaded
automatically):

```
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=<your db>
POSTGRES_USER=<your user>
POSTGRES_PASSWORD=<your password>
```

The database must already have the `quant.quote_keys`, `quant.market_data`
and `quant.metrics` tables that `data/db.py` reads and writes.

## 3. (Optional) Pre-populate FX price history

`models/lstm_forecaster.py` will auto-download and upsert any symbol that
isn't in Postgres yet the first time it's asked for it, so this step isn't
strictly required. Run it up front if you'd rather control the download
(e.g. to fetch more years of history, or every major pair at once):

```bash
python -m data.fx_downloader --pairs EURUSD GBPUSD USDJPY --years 8 --upload
```

`--upload` is what upserts the downloaded closes into Postgres via
`data/db.py`. Omit `--pairs` to fetch all seven major pairs.

## 4. (Optional) Download interest-rate series

Only needed if a model/feature you're building uses the FX carry (rate
differential) feature; the LSTM forecaster below doesn't require it.

```bash
python -m data.rates_downloader --currencies USD EUR JPY --upload
```

## 5. Train the LSTM and check out-of-sample metrics

```bash
python -m models.lstm_forecaster \
    --pairs EURUSD GBPUSD USDJPY \
    --target EURUSD \
    --lookback 30 --horizon 5 --epochs 100
```

- `--pairs`: input FX pairs (log returns fed to the LSTM as features).
- `--target`: the pair being forecast (auto-added to `--pairs` if missing).
- `--lookback`: days of history per input window.
- `--horizon`: N days ahead to forecast.

Data is loaded from Postgres (falling back to download+upsert if a symbol
isn't cached there yet, as in step 3). Prints training progress, then
in-sample (train) and out-of-sample (validation) MSE and hit rate, plus a
forecast for the next `--horizon` days.

## 6. Plot forecast vs actual and print the hit rate

```bash
python -m models.postprocess \
    --pairs EURUSD GBPUSD USDJPY \
    --target EURUSD \
    --lookback 30 --horizon 5 --epochs 100 \
    --output models/forecast_plot.png
```

Same arguments and pipeline as step 5 (it retrains the model the same way,
so keep the arguments consistent), plus `--output` for the saved plots.
Prints in-sample and out-of-sample hit rate, and saves two separate charts
of next-day forecast vs actual log return (each with its own hit rate in
the title) derived from `--output`, e.g. `models/forecast_plot.png` ->

- `models/forecast_plot_insample.png` (train period)
- `models/forecast_plot_outsample.png` (validation period)

## 7. Train the portfolio allocator (Sharpe ratio optimization)

`models/portfolio_lstm.py` is a different model from step 5: instead of
forecasting a return value, its output IS the trading decision - a weight
per FX pair - and it's trained end-to-end to maximize the Sharpe ratio of
the resulting portfolio, not to minimize forecast error.

```bash
python -m models.portfolio_lstm \
    --pairs EURUSD GBPUSD USDJPY \
    --lookback 30 --weight-scheme softmax --epochs 300
```

- `--pairs`: the FX pairs in the portfolio (inputs and allocation targets).
- `--weight-scheme`:
  - `softmax` (default) - long-only; weights in (0, 1) and always sum to
    exactly 1.
  - `tanh_norm` - long/short; weights in (-1, 1), L1-normalized so the book
    is fully invested (sum of `|weight|` == 1); the signed sum floats
    freely in [-1, 1] since allowing shorts rules out also pinning the
    signed sum to 1.

Data is loaded via `data/db.py` exactly as in step 5. Trains with full-batch
gradient descent on the (negated) Sharpe ratio of the whole train-period
return path, then prints in-sample (train) and out-of-sample (validation)
Sharpe ratio and cumulative PnL.

Add `--risk-overlay` to also train the risk-attenuation network from step
9 on top, in the same run:

```bash
python -m models.portfolio_lstm \
    --pairs EURUSD GBPUSD USDJPY \
    --lookback 30 --weight-scheme softmax --epochs 300 \
    --risk-overlay --risk-hidden-size 16 --risk-epochs 200
```

Without `--risk-overlay`, behavior is unchanged. With it, after training
PortfolioLSTM as usual, it's frozen and a RiskLSTM is trained on top (see
step 9), and raw vs attenuated Sharpe ratio is printed for both splits.

This model overfits easily (full-batch training for hundreds of epochs on
a noisy Sharpe objective will happily memorize the training period), so
three training-time regularizers are on by default:

- `--noise-std` (default 0.05): fresh Gaussian noise added to the
  standardized input window every epoch, so the model can't fit the exact
  training sequence, only patterns that survive small perturbations of it.
- `--dropout` (default 0.1): dropout on the LSTM's final hidden state,
  before the linear head.
- `--weight-decay` (default 1e-5): L2 penalty on the model weights (Adam's
  `weight_decay`).

Set any of them to `0` to disable. Other techniques worth trying if
overfitting is still a problem (not implemented here, to keep the
pipeline simple):

- **Early stopping** on a third held-out dev split (stop training once
  dev-set Sharpe stops improving) - the cleanest lever, but needs a
  train/dev/test split instead of train/validation.
- **Turnover penalty**: subtract a cost proportional to
  `|weights[t] - weights[t-1]|` from the Sharpe objective, so the model is
  discouraged from chasing noisy day-to-day signals.
- **Smaller model**: fewer hidden units / a shorter `--lookback` reduces
  capacity relative to how much data there is.
- **Walk-forward cross-validation**: retrain on a rolling window instead
  of one fixed train/validation split, to check the Sharpe estimate is
  stable across different time periods.

The Sharpe-ratio training objective is also non-convex in the LSTM's
parameters (that's the LSTM/softmax nonlinearity, not the Sharpe ratio
itself - true of any neural net regardless of loss function), so different
random initializations can land in meaningfully different local optima.
`--n-seeds` trains that many independent restarts on the same data and
combines them via `--restart-strategy`:

```bash
python -m models.portfolio_lstm \
    --pairs EURUSD GBPUSD USDJPY \
    --lookback 30 --weight-scheme softmax --epochs 300 \
    --n-seeds 5 --restart-strategy best
```

```bash
python -m models.portfolio_lstm \
    --pairs EURUSD GBPUSD USDJPY \
    --lookback 30 --weight-scheme softmax --epochs 300 \
    --n-seeds 5 --restart-strategy ensemble
```

- `--restart-strategy best` (default): keeps the single restart with the
  highest validation Sharpe - model selection.
- `--restart-strategy ensemble`: averages every restart's predicted
  weights (re-normalized to keep the same weight-scheme invariant - a
  no-op for `softmax`, since an average of simplex points stays on the
  simplex; a real renormalization for `tanh_norm`, since restarts can
  disagree on sign per asset). Averaging tends to cancel out each
  individual restart's idiosyncratic overfitting.

`--n-seeds 1` (the default) skips all of this and behaves exactly as
before. Data is loaded once and reused across every restart, so the cost
of `--n-seeds N` is roughly N training runs, not N full pipelines.

## 8. Plot cumulative PnL and print the Sharpe ratio

```bash
python -m models.portfolio_postprocess \
    --pairs EURUSD GBPUSD USDJPY \
    --lookback 30 --weight-scheme softmax --epochs 300 \
    --output models/portfolio_pnl.png
```

Same arguments and pipeline as step 7 (including `--n-seeds`/`--restart-strategy`),
plus `--output` for the saved plots. Prints in-sample and out-of-sample
Sharpe ratio and cumulative PnL, and saves two separate cumulative-PnL
charts (each with its Sharpe ratio in the title) derived from `--output`,
e.g. `models/portfolio_pnl.png` ->

- `models/portfolio_pnl_insample.png` (train period)
- `models/portfolio_pnl_outsample.png` (validation period)

Passing `--risk-overlay` here also works, and switches this script's
output to the full risk pipeline (identical to running
`models.risk_postprocess` - step 10 below): it trains RiskLSTM on top,
skips the plain 2-plot output above, and instead saves the same 4 plots
step 10 describes (raw-vs-attenuated PnL + position-vs-scaling, both
splits) to `--output`/`--position-output`.

## 9. Train the risk-attenuation overlay

`models/risk_lstm.py` adds a second, independent network on top of
PortfolioLSTM. PortfolioLSTM only ever decides *which* assets to hold and
in what proportion - it has no notion of "I'm not confident right now".
RiskLSTM's job is to say *how much* of each proposed position to actually
take - one attenuation factor **per asset**, in `[max_attenuation, 1]`:
close to 1 for an asset in a normal, tradeable period, down towards
`max_attenuation` when that asset's recent log returns look
directionless - so the strategy can de-risk one pair without necessarily
touching the others, and never zeroes any asset out entirely.

This is a two-stage pipeline, not one joint model:

1. Train PortfolioLSTM exactly as in step 7 (reuses that same code).
2. Freeze it, and use its predicted weights as a fixed input (no
   gradients flow back into PortfolioLSTM from this stage).
3. RiskLSTM looks at the same log-return window plus those frozen
   weights, and is trained on its own to maximize the Sharpe ratio of the
   *attenuated* portfolio (`weights * attenuation`, elementwise per asset).

```bash
python -m models.risk_lstm \
    --pairs EURUSD GBPUSD USDJPY \
    --lookback 30 --weight-scheme softmax --epochs 300 \
    --risk-hidden-size 16 --risk-epochs 200
```

(Equivalent to step 7's `python -m models.portfolio_lstm ... --risk-overlay`
- both call the same `add_risk_overlay()`. Use whichever entry point is
more convenient; this one exists mainly so `models/risk_postprocess.py`
below has a plain function to call.)

Accepts every `models/portfolio_lstm.py` argument (for stage 1) plus:

- `--risk-hidden-size` (default 16): RiskLSTM's hidden size - attenuation
  is a simpler task than picking the portfolio, so this defaults smaller
  than `--hidden-size`.
- `--risk-epochs` (default 200), `--risk-lr` (default 1e-3): RiskLSTM's
  own training length/rate, independent of PortfolioLSTM's `--epochs`/`--lr`.
- `--max-attenuation` (default 0.33): a hard **floor**, not a ceiling, on
  each asset's attenuation factor, in (0, 1]. Each asset's sigmoid output
  is rescaled into `[max_attenuation, 1]` rather than the full `(0, 1)`,
  so even at minimum confidence that asset is never de-risked below this
  fraction of its proposed weight, and 1 means no attenuation at all
  (full-size position) - a limit that holds regardless of what the
  network learns, not just a training-time preference for smaller bets.

`--dropout`/`--weight-decay`/`--noise-std` are shared between both
networks' training stages. Prints raw (unattenuated) vs attenuated Sharpe
ratio and mean attenuation, for both the train and validation splits.

## 10. Plot raw vs attenuated PnL and print the Sharpe ratio

This single command runs the entire positions + risk process end to end -
trains PortfolioLSTM (optionally with `--n-seeds`/`--restart-strategy`
restarts), trains RiskLSTM on top, prints both Sharpe ratios, and saves
all four plots from steps 9-10 in one go - nothing else needs to be run
separately:

```bash
python -m models.risk_postprocess \
    --pairs EURUSD GBPUSD USDJPY \
    --lookback 30 --weight-scheme softmax --epochs 300 \
    --n-seeds 5 --restart-strategy best \
    --risk-hidden-size 16 --risk-epochs 200 \
    --output models/risk_pnl.png
```

Same arguments and pipeline as step 9, plus `--output` for the saved
plots. Prints raw vs attenuated Sharpe ratio and mean attenuation for both
splits, and saves two separate charts - each overlaying the raw and
attenuated cumulative PnL curves so the risk overlay's effect (smaller
drawdowns, usually a smaller but steadier curve) is visible directly -
derived from `--output`, e.g. `models/risk_pnl.png` ->

- `models/risk_pnl_insample.png` (train period)
- `models/risk_pnl_outsample.png` (validation period)

It also saves a second pair of charts (`--position-output`, default
`models/risk_position.png`) overlaying each pair's raw position (portfolio
weight, solid line, left y-axis) with that SAME pair's attenuation (dashed
line, same color, right y-axis fixed to [0, 1]) - since attenuation is now
per asset rather than one global scaling factor, each pair gets a matched
position/attenuation line pair, so you can see e.g. one pair being
de-risked while another stays near full size ->

- `models/risk_position_insample.png`
- `models/risk_position_outsample.png`

## 11. Inference only - load saved models instead of retraining

Every script above saves its model(s) after training (`models/portfolio_lstm.pt`,
or `models/portfolio_lstm_ensemble.pt` when `--restart-strategy ensemble` was
used, plus `models/risk_lstm.pt` with `--risk-overlay`). Each checkpoint is
self-contained: architecture config, weights, and (for the portfolio model)
the exact input standardization stats (`x_mean`/`x_std`) it was trained
with - so `--load-portfolio`/`--load-risk` reload it without needing to
know which flags originally trained it.

`--load-portfolio <path>` skips PortfolioLSTM training entirely (all of
`--n-seeds`/`--restart-strategy`/`--epochs`/etc. are ignored) and just
loads it for inference; it auto-detects whether the file holds a single
model or an ensemble, and if an ensemble, loads every member and averages
their responses exactly as it did at training time. Works with any script
above, e.g. to just re-plot from saved models without retraining:

```bash
python -m models.risk_postprocess \
    --pairs EURUSD GBPUSD USDJPY --lookback 30 \
    --load-portfolio models/portfolio_lstm.pt \
    --load-risk models/risk_lstm.pt \
    --output models/risk_pnl.png
```

`--pairs`/`--lookback`/`--years`/`--train-frac` still need to be supplied
(they control what data is fetched and how it's split for evaluation);
everything architecture-related (`--hidden-size`, `--weight-scheme`,
`--dropout`, etc.) is ignored in favor of what's baked into the checkpoint.
`--load-portfolio` alone (without `--risk-overlay`/`--load-risk`) also
works with `models/portfolio_lstm.py`/`models/portfolio_postprocess.py`
for positions-only inference.
