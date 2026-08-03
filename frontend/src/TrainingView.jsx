import { useEffect, useRef, useState } from "react";
import { getPairs, startTraining, getTrainingStatus, stopTraining } from "./api";
import HitRateChart from "./charts/HitRateChart";
import ColoredReturnChart from "./charts/ColoredReturnChart";
import ProbabilityChart from "./charts/ProbabilityChart";
import ReturnDistributionChart from "./charts/ReturnDistributionChart";
import ConfusionMatrixTable from "./ConfusionMatrixTable";
import AnnualSharpeTable from "./AnnualSharpeTable";
import { SPLIT_LABEL } from "./theme";
import { confusionMatrixForSplit, hitAbstainedSeries, hitRateForSplit } from "./metrics";

// Every feature build_feature_dataframe knows how to produce (see
// models/portfolio_lstm.py's FEATURE_CATALOG) - "cma" is special, it
// expands to one channel PER window pair in form.cma_windows, not one.
const FEATURE_OPTIONS = [
  ["log_return", "Log return"],
  ["vol", "Rolling volatility"],
  ["skew", "Rolling skewness"],
  ["kurt", "Rolling excess kurtosis"],
  ["carry", "Carry (interest-rate differential)"],
  ["cma", "Cross moving averages (trend)"],
  ["bandpass", "Butterworth bandpass (faster-reacting trend)"],
];

const DEVICE_OPTIONS = [
  ["auto", "Auto (Metal/MPS if available, else CUDA, else CPU)"],
  ["cpu", "CPU"],
  ["mps", "MPS (Apple Silicon)"],
];

const DEFAULT_FORM = {
  pairs: [],
  lookback: 30,
  years: 8,
  // "" = no cutoff (backend default: use all data up to today, computed
  // fresh on every run). An ISO "YYYY-MM-DD" date caps every fetched
  // series (prices + carry) at that date - never later - so a
  // walk-forward backtest never trains/validates on data past it (see
  // models/portfolio_lstm.py's _resolve_cutoff_date).
  cutoff_date: "",
  train_frac: 0.8,
  test_frac: 0.1,
  direction_horizon: 5,
  rolling_stats_window: 20,
  features: ["log_return", "vol", "skew", "kurt"],
  cma_windows: [], // [{short, long}, ...] - UI shape; converted to [[short,long],...] on submit
  bandpass_windows: [], // same UI shape as cma_windows - short/long PERIODS in days for a causal Butterworth bandpass filter
  bandpass_order: 3,
  // {pair: [other_pair, ...]} - which OTHER pairs' features also feed a
  // given pair's own LSTM (its own features are always included
  // regardless). DEFAULT is every pair fully independent (see
  // models/portfolio_lstm.py's PredictionModel docstring) - a pair
  // missing here, or with an empty list, sees ONLY its own features.
  cross_pairs: {},
  hidden_size: 16,
  num_layers: 1,
  dropout: 0.1,
  n_attn_heads: 4,
  epochs: 300,
  lr: 0.001,
  weight_decay: 0.0001,
  bce_weight: "1.0", // free text: a single number, or comma-separated to SWEEP (see parseBceWeight)
  // Optional COMPLEMENTARY training objective - 0 (default) disables it,
  // training is then unchanged (own optimizer/loss per asset only). > 0
  // adds a joint portfolio-Sharpe phase: risk-parity weight x this
  // model's own probability signal, scaled to target_vol below,
  // maximizing an averaged rolling Sharpe over sharpe_window days - see
  // models/portfolio_lstm.py's train_prediction_model docstring.
  sharpe_weight: 0,
  sharpe_window: 20,
  neutral_band: 0.05,
  // Annualized volatility the Evaluation view's portfolio PnL calculator
  // scales this model's positions to (see models/portfolio_pnl.py) -
  // persisted with the model, like neutral_band, not a free
  // evaluation-time parameter. Also used DURING training if
  // sharpe_weight above is > 0.
  target_vol: 0.1,
  n_seeds: 1,
  device: "auto",
  save_db: true,
  model_description: "",
};

const NUMERIC_FIELDS = new Set([
  "lookback", "years", "train_frac", "test_frac", "direction_horizon", "rolling_stats_window",
  "hidden_size", "num_layers", "dropout", "n_attn_heads",
  "epochs", "lr", "weight_decay", "neutral_band", "target_vol",
  "n_seeds", "bandpass_order", "sharpe_weight", "sharpe_window",
]);

// "1.0" -> 1.0 (single value); "1.0, 1.5, 2.0" -> [1.0, 1.5, 2.0] (sweep -
// see models/portfolio_lstm.py's run_pipeline_multi_seed, which trains
// every value under every seed and keeps whichever validated best).
function parseBceWeight(text) {
  const values = text.split(",").map((s) => Number(s.trim())).filter((n) => !Number.isNaN(n));
  if (values.length === 0) return 1.0;
  return values.length === 1 ? values[0] : values;
}

export default function TrainingView() {
  const [availablePairs, setAvailablePairs] = useState([]);
  const [form, setForm] = useState(DEFAULT_FORM);
  const [customPair, setCustomPair] = useState("");
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null); // "pending" | "running" | "done" | "error" | "stopped"
  const [progress, setProgress] = useState(null);
  const [logs, setLogs] = useState([]);
  const [interim, setInterim] = useState(null);
  const [result, setResult] = useState(null);
  // Neutral band shown/editable in the results section, separate from
  // form.neutral_band (which only controls what a NEW training run uses) -
  // initialized to whatever this run was configured with, then freely
  // adjustable to recompute hit rate/confusion matrix/colored returns
  // client-side (see ../metrics.js) without retraining or a new request,
  // since the band is pure postprocessing, never part of training (see
  // models/portfolio_lstm.py's evaluate_prediction_model docstring).
  const [displayBand, setDisplayBand] = useState(0.05);
  const [error, setError] = useState(null);
  const pollRef = useRef(null);
  const logBoxRef = useRef(null);

  useEffect(() => {
    if (logBoxRef.current) {
      logBoxRef.current.scrollTop = logBoxRef.current.scrollHeight;
    }
  }, [logs]);

  useEffect(() => {
    getPairs().then(setAvailablePairs).catch((e) => setError(e.message));
    return () => clearInterval(pollRef.current);
  }, []);

  function updateField(name, value) {
    setForm((f) => ({ ...f, [name]: NUMERIC_FIELDS.has(name) ? Number(value) : value }));
  }

  function togglePair(pair) {
    setForm((f) => {
      const wasIncluded = f.pairs.includes(pair);
      const pairs = wasIncluded ? f.pairs.filter((p) => p !== pair) : [...f.pairs, pair];
      // Removing a pair also drops it as a cross_pairs KEY, and out of
      // every OTHER pair's linked list - a removed pair can't stay
      // referenced as an input source.
      let cross_pairs = f.cross_pairs;
      if (wasIncluded) {
        cross_pairs = Object.fromEntries(
          Object.entries(f.cross_pairs)
            .filter(([p]) => p !== pair)
            .map(([p, others]) => [p, others.filter((o) => o !== pair)])
            .filter(([, others]) => others.length > 0),
        );
      }
      return { ...f, pairs, cross_pairs };
    });
  }

  function addCustomPair() {
    const pair = customPair.trim().toUpperCase();
    if (pair && !form.pairs.includes(pair)) {
      setForm((f) => ({ ...f, pairs: [...f.pairs, pair] }));
    }
    setCustomPair("");
  }

  function toggleFeature(feature) {
    setForm((f) => ({
      ...f,
      features: f.features.includes(feature) ? f.features.filter((x) => x !== feature) : [...f.features, feature],
    }));
  }

  function addCmaWindow() {
    setForm((f) => ({ ...f, cma_windows: [...f.cma_windows, { short: 10, long: 50 }] }));
  }

  function updateCmaWindow(index, field, value) {
    setForm((f) => ({
      ...f,
      cma_windows: f.cma_windows.map((w, i) => (i === index ? { ...w, [field]: Number(value) } : w)),
    }));
  }

  function removeCmaWindow(index) {
    setForm((f) => ({ ...f, cma_windows: f.cma_windows.filter((_, i) => i !== index) }));
  }

  function addBandpassWindow() {
    setForm((f) => ({ ...f, bandpass_windows: [...f.bandpass_windows, { short: 10, long: 50 }] }));
  }

  function updateBandpassWindow(index, field, value) {
    setForm((f) => ({
      ...f,
      bandpass_windows: f.bandpass_windows.map((w, i) => (i === index ? { ...w, [field]: Number(value) } : w)),
    }));
  }

  function removeBandpassWindow(index) {
    setForm((f) => ({ ...f, bandpass_windows: f.bandpass_windows.filter((_, i) => i !== index) }));
  }

  function toggleCrossPair(pair, otherPair) {
    setForm((f) => {
      const current = f.cross_pairs[pair] || [];
      const updated = current.includes(otherPair) ? current.filter((p) => p !== otherPair) : [...current, otherPair];
      const cross_pairs = { ...f.cross_pairs };
      if (updated.length === 0) {
        delete cross_pairs[pair];
      } else {
        cross_pairs[pair] = updated;
      }
      return { ...f, cross_pairs };
    });
  }

  async function submit(e) {
    e.preventDefault();
    setError(null);
    setResult(null);
    setProgress(null);
    setLogs([]);
    setInterim(null);
    if (form.pairs.length < 1) {
      setError("Select at least one FX pair.");
      return;
    }
    try {
      const { job_id } = await startTraining({
        ...form,
        cutoff_date: form.cutoff_date || null,
        bce_weight: parseBceWeight(form.bce_weight),
        cma_windows: form.cma_windows.map((w) => [w.short, w.long]),
        bandpass_windows: form.bandpass_windows.map((w) => [w.short, w.long]),
      });
      setJobId(job_id);
      setStatus("pending");
      pollRef.current = setInterval(async () => {
        const job = await getTrainingStatus(job_id);
        setStatus(job.status);
        setProgress(job.progress || null);
        setLogs(job.logs || []);
        setInterim(job.interim || null);
        if (job.status === "done") {
          clearInterval(pollRef.current);
          setResult(job.result);
          setDisplayBand(job.result.neutral_band);
        } else if (job.status === "error") {
          clearInterval(pollRef.current);
          setError(job.error);
        } else if (job.status === "stopped") {
          clearInterval(pollRef.current);
        }
      }, 1500);
    } catch (err) {
      setError(err.message);
    }
  }

  async function stop() {
    if (!jobId) return;
    try {
      await stopTraining(jobId);
    } catch (err) {
      setError(err.message);
    }
  }

  const allPairs = Array.from(new Set([...availablePairs, ...form.pairs]));
  const isBusy = status === "pending" || status === "running";

  // Recomputed from raw probability + realized label (see ../metrics.js)
  // every time displayBand changes - the band never touches training, so
  // this is the ONLY place it's actually applied to this result.
  const hitRate = result && {
    train: hitRateForSplit(result.probabilities.train, result.pairs, displayBand),
    val: hitRateForSplit(result.probabilities.val, result.pairs, displayBand),
    test: hitRateForSplit(result.probabilities.test, result.pairs, displayBand),
  };
  const confusionMatrix = result && {
    train: confusionMatrixForSplit(result.probabilities.train, result.pairs, displayBand),
    val: confusionMatrixForSplit(result.probabilities.val, result.pairs, displayBand),
    test: confusionMatrixForSplit(result.probabilities.test, result.pairs, displayBand),
  };

  return (
    <div>
      <form onSubmit={submit}>
        <div className="panel">
          <h2 style={{ marginTop: 0 }}>FX pairs</h2>
          <div className="pair-picker">
            {allPairs.map((pair) => (
              <span
                key={pair}
                className={`pair-chip${form.pairs.includes(pair) ? " selected" : ""}`}
                onClick={() => togglePair(pair)}
              >
                {pair}
              </span>
            ))}
          </div>
          <div style={{ marginTop: 10, display: "flex", gap: 8 }}>
            <input
              type="text"
              placeholder="Add custom pair, e.g. EURJPY"
              value={customPair}
              onChange={(e) => setCustomPair(e.target.value)}
              style={{ padding: "6px 8px", border: "1px solid #cbd5e1", borderRadius: 6, fontSize: 13 }}
            />
            <button type="button" className="secondary" onClick={addCustomPair}>Add</button>
          </div>
        </div>

        <div className="panel">
          <h2 style={{ marginTop: 0 }}>Data &amp; features</h2>
          <div className="form-grid">
            <NumField label="Lookback (days)" name="lookback" value={form.lookback} onChange={updateField} />
            <NumField label="Years of history" name="years" value={form.years} onChange={updateField} />
            <DateField
              label="Cutoff date (blank = today, no cutoff)"
              name="cutoff_date"
              value={form.cutoff_date}
              onChange={updateField}
            />
            <NumField label="Train fraction (of non-test data)" name="train_frac" step="0.05" value={form.train_frac} onChange={updateField} />
            <NumField label="Test fraction (held out, most recent)" name="test_frac" step="0.05" value={form.test_frac} onChange={updateField} />
            <NumField
              label="Direction horizon (forward days)"
              name="direction_horizon"
              value={form.direction_horizon}
              onChange={updateField}
            />
            <NumField
              label="Rolling stats window (vol/skew/kurtosis)"
              name="rolling_stats_window"
              value={form.rolling_stats_window}
              onChange={updateField}
            />
          </div>
          <p className="status-line" style={{ marginTop: 4 }}>
            Chronological 3-way split: the most recent {(form.test_frac * 100).toFixed(0)}% of history is held out as
            a TEST set, never used for training or checkpoint selection - a genuinely unbiased read on
            generalization. Of what remains, the oldest {(form.train_frac * 100).toFixed(0)}% is train and the rest
            is validation (which DOES influence best-epoch checkpoint selection).
          </p>
          <h3 style={{ marginBottom: 6 }}>Features</h3>
          <div className="pair-picker">
            {FEATURE_OPTIONS.map(([key, label]) => (
              <span
                key={key}
                className={`pair-chip${form.features.includes(key) ? " selected" : ""}`}
                onClick={() => toggleFeature(key)}
              >
                {label}
              </span>
            ))}
          </div>
          <p className="status-line" style={{ marginTop: 4 }}>
            Each selected feature becomes one input channel per pair (see models/portfolio_lstm.py's
            FEATURE_CATALOG) - "Rolling volatility/skewness/kurtosis" use the {form.rolling_stats_window}-day window
            above; "Carry" is the interest-rate differential (FRED); "Cross moving averages" adds a trend/momentum
            channel PER window pair below (short-window mean return minus long-window mean return - positive when
            the recent trend runs above the longer-run one).
          </p>
          {form.features.includes("cma") && (
            <div style={{ marginTop: 8 }}>
              {form.cma_windows.map((w, i) => (
                <div key={i} style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 6 }}>
                  <div className="field" style={{ maxWidth: 100 }}>
                    <label>Short window</label>
                    <input type="number" value={w.short} onChange={(e) => updateCmaWindow(i, "short", e.target.value)} />
                  </div>
                  <div className="field" style={{ maxWidth: 100 }}>
                    <label>Long window</label>
                    <input type="number" value={w.long} onChange={(e) => updateCmaWindow(i, "long", e.target.value)} />
                  </div>
                  <button type="button" className="secondary" onClick={() => removeCmaWindow(i)} style={{ alignSelf: "end" }}>
                    Remove
                  </button>
                </div>
              ))}
              <button type="button" className="secondary" onClick={addCmaWindow}>Add CMA window</button>
              {form.cma_windows.length === 0 && (
                <p className="status-line" style={{ marginTop: 4, color: "#b45309" }}>
                  "Cross moving averages" is selected but no windows are configured - add at least one.
                </p>
              )}
            </div>
          )}
          {form.features.includes("bandpass") && (
            <div style={{ marginTop: 8 }}>
              <p className="status-line" style={{ marginTop: 0 }}>
                A causal Butterworth band-pass filter (forward-only - never the zero-phase filtfilt, which would leak
                future samples) reacting to cycles between the short and long period below, in days - a
                faster-reacting alternative to a CMA crossover for the same window pair.
              </p>
              {form.bandpass_windows.map((w, i) => (
                <div key={i} style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 6 }}>
                  <div className="field" style={{ maxWidth: 100 }}>
                    <label>Short period (days)</label>
                    <input type="number" value={w.short} onChange={(e) => updateBandpassWindow(i, "short", e.target.value)} />
                  </div>
                  <div className="field" style={{ maxWidth: 100 }}>
                    <label>Long period (days)</label>
                    <input type="number" value={w.long} onChange={(e) => updateBandpassWindow(i, "long", e.target.value)} />
                  </div>
                  <button type="button" className="secondary" onClick={() => removeBandpassWindow(i)} style={{ alignSelf: "end" }}>
                    Remove
                  </button>
                </div>
              ))}
              <button type="button" className="secondary" onClick={addBandpassWindow}>Add bandpass window</button>
              {form.bandpass_windows.length === 0 && (
                <p className="status-line" style={{ marginTop: 4, color: "#b45309" }}>
                  "Butterworth bandpass" is selected but no windows are configured - add at least one.
                </p>
              )}
              <div className="field" style={{ maxWidth: 140, marginTop: 6 }}>
                <label>Filter order</label>
                <input type="number" value={form.bandpass_order} onChange={(e) => updateField("bandpass_order", e.target.value)} />
              </div>
            </div>
          )}
        </div>

        {form.pairs.length > 1 && (
          <div className="panel">
            <h2 style={{ marginTop: 0 }}>Cross-asset inputs</h2>
            <p className="status-line" style={{ marginTop: 0 }}>
              By DEFAULT every pair's own LSTM sees ONLY its own features - fully independent, no cross-asset
              mixing. Link a pair to OTHER pairs below to also feed their full feature blocks into that pair's own
              LSTM (its own features are always included regardless) - see models/portfolio_lstm.py's
              PredictionModel docstring.
            </p>
            {form.pairs.map((pair) => (
              <div key={pair} style={{ marginBottom: 10 }}>
                <strong style={{ fontSize: 13 }}>{pair}</strong>'s LSTM also sees:
                <div className="pair-picker" style={{ marginTop: 4 }}>
                  {form.pairs.filter((p) => p !== pair).map((other) => (
                    <span
                      key={other}
                      className={`pair-chip${(form.cross_pairs[pair] || []).includes(other) ? " selected" : ""}`}
                      onClick={() => toggleCrossPair(pair, other)}
                    >
                      {other}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        <div className="panel">
          <h2 style={{ marginTop: 0 }}>Architecture</h2>
          <div className="form-grid">
            <NumField label="Hidden size" name="hidden_size" value={form.hidden_size} onChange={updateField} />
            <NumField label="LSTM layers" name="num_layers" value={form.num_layers} onChange={updateField} />
            <NumField label="Dropout" name="dropout" step="0.01" value={form.dropout} onChange={updateField} />
            <NumField label="Attention heads" name="n_attn_heads" value={form.n_attn_heads} onChange={updateField} />
          </div>
          <p className="status-line" style={{ marginTop: 4 }}>
            Each asset's own LSTM is followed by a CAUSAL self-attention layer over the time axis (day k can only
            attend to days ≤ k, preserving the no-look-ahead property every day's supervision relies on), then
            outputs a heteroscedastic (μ, σ) pair at every day in the window: μ is the predicted conditional mean
            z-score, σ its predicted conditional std. P(positive) = Φ(μ / (σ · σ̂)) - the probit link, rescaled by a
            validation-fit calibration factor σ̂ - is what the probability/hit rate/confusion matrix below are
            computed from. The whole network is deterministic - no NoisyNet head, no input-noise regularization.
            "Attention heads" must divide "Hidden size" evenly.
          </p>
        </div>

        <div className="panel">
          <h2 style={{ marginTop: 0 }}>Training</h2>
          <div className="form-grid">
            <NumField label="Epochs" name="epochs" value={form.epochs} onChange={updateField} />
            <NumField label="Learning rate" name="lr" step="0.0001" value={form.lr} onChange={updateField} />
            <NumField label="Weight decay" name="weight_decay" step="0.0001" value={form.weight_decay} onChange={updateField} />
            <div className="field">
              <label>BCE direction weight (comma-separated to sweep)</label>
              <input
                type="text"
                value={form.bce_weight}
                onChange={(e) => setForm((f) => ({ ...f, bce_weight: e.target.value }))}
                placeholder="e.g. 1.0  or  1.0, 1.5, 1.75, 2.0, 3.0"
              />
            </div>
            <NumField label="Neutral band (abstention half-width)" name="neutral_band" step="0.01" value={form.neutral_band} onChange={updateField} />
            <NumField label="Target vol (annualized, for portfolio PnL)" name="target_vol" step="0.01" value={form.target_vol} onChange={updateField} />
            <NumField label="Sharpe weight (0 = disabled)" name="sharpe_weight" step="0.1" value={form.sharpe_weight} onChange={updateField} />
            <NumField label="Sharpe window (days)" name="sharpe_window" value={form.sharpe_window} onChange={updateField} />
          </div>
          <p className="status-line" style={{ marginTop: 4 }}>
            Every asset's LSTM trains FULLY INDEPENDENTLY: its own optimizer, its own loss - never averaged or
            shared with any other asset's - Gaussian NLL (fits the full predictive distribution) plus "BCE direction
            weight" × a binary cross-entropy term on the implied direction probability. The BCE term is the
            anti-collapse fix: a bare regression loss is minimized by the label's unconditional mean - one sign for
            everything when the target is barely predictable - while BCE is minimized by matching each sample's OWN
            sign, so any discriminating feature gets pulled into spreading μ across zero. Applied densely, over
            every day in every window, not just the last one. Each asset's own checkpoint is restored from ITS OWN
            best validation epoch independently, too - one asset's convergence never drags another's along.
          </p>
          <p className="status-line" style={{ marginTop: 4 }}>
            "Sharpe weight" 0 keeps the above unchanged. {'>'} 0 adds a SECOND, complementary training phase: the
            model's own probability signal × a precomputed (data-only, never optimized) risk-parity weight, scaled
            to "Target vol", drives an averaged rolling Sharpe (over "Sharpe window" days) that gets maximized -
            the same strategy the Evaluation page's portfolio PnL calculator reports. Unlike everything else here,
            this phase is NOT per-asset independent: the vol-scaling denominator mixes every asset's current
            signal, so one asset's training genuinely depends on what the others are currently predicting. Keep
            this modest relative to "BCE direction weight" - directly maximizing Sharpe on noisy daily returns can
            drift toward low-variance, low-conviction bets rather than genuine predictive signal.
          </p>
          <p className="status-line" style={{ marginTop: 4 }}>
            Enter several comma-separated values to SWEEP the BCE weight alongside the seeds below: every value is
            trained under every seed (same initial weights across the sweep, so only the value differs) and
            whichever validated best wins - first per seed, then overall.
          </p>
          <p className="status-line" style={{ marginTop: 4 }}>
            "Neutral band": probabilities within 0.5 ± this half-width are snapped to exactly 0.5 - the model
            ABSTAINS instead of making a near-coin-flip call. Hit rate/confusion-matrix metrics below are then
            computed over decided samples only, alongside a coverage metric (how often the model actually speaks) -
            accuracy and coverage must be read together. 0 disables abstention entirely.
          </p>
        </div>

        <div className="panel">
          <h2 style={{ marginTop: 0 }}>Multi-seed restarts &amp; execution</h2>
          <div className="form-grid">
            <NumField label="Number of seeds" name="n_seeds" value={form.n_seeds} onChange={updateField} />
            <div className="field">
              <label>Execution device</label>
              <select value={form.device} onChange={(e) => updateField("device", e.target.value)}>
                {DEVICE_OPTIONS.map(([key, label]) => (
                  <option key={key} value={key}>{label}</option>
                ))}
              </select>
            </div>
          </div>
          <p className="status-line" style={{ marginTop: 4 }}>
            Trains this many independent restarts and keeps whichever had the lowest validation log loss (not hit
            rate - see the model's own docs for why a saturating accuracy metric is the wrong thing to compare
            restarts on). "Auto" picks Apple Silicon's Metal backend (MPS) if available, else CUDA, else CPU - if
            training fails with a non-finite-gradient error, try "CPU" (a known PyTorch MPS bug affects some
            multi-layer, large-hidden-size architectures - see models/portfolio_lstm.py's train_prediction_model).
          </p>
        </div>

        <div className="panel">
          <h2 style={{ marginTop: 0 }}>Save</h2>
          <div className="form-grid">
            <label className="field checkbox">
              <input
                type="checkbox"
                checked={form.save_db}
                onChange={(e) => setForm((f) => ({ ...f, save_db: e.target.checked }))}
              />
              Persist to database (quant.model_registry)
            </label>
            <div className="field" style={{ gridColumn: "span 2" }}>
              <label>Model description</label>
              <input
                type="text"
                value={form.model_description}
                onChange={(e) => updateField("model_description", e.target.value)}
                placeholder="Optional note stored alongside the model"
              />
            </div>
          </div>
        </div>

        <div className="actions">
          <button className="primary" type="submit" disabled={isBusy}>
            {isBusy ? "Training…" : "Start training"}
          </button>
          {isBusy && (
            <button type="button" className="secondary" onClick={stop}>
              Stop training
            </button>
          )}
          {status && <span className="status-line">Status: {status}</span>}
        </div>
      </form>

      {status === "stopped" && (
        <p className="status-line">Training stopped - nothing was saved. The log below shows where it left off.</p>
      )}

      {progress && (status === "running" || status === "pending" || status === "done" || status === "stopped") && (
        <div className="panel">
          <h3 style={{ marginBottom: 6 }}>
            Progress{progress.n_seeds > 1 ? ` — restart ${progress.seed_index}/${progress.n_seeds}` : ""}
            {progress.n_lambdas > 1 ? ` — bce_weight ${progress.lambda_index}/${progress.n_lambdas}` : ""}
            {" "}— epoch {progress.epoch}/{progress.total_epochs}
          </h3>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${progress.percent}%` }} />
          </div>
          <div className="progress-percent">{progress.percent.toFixed(1)}%</div>

          {interim && (
            <table className="weights-table" style={{ marginTop: 12 }}>
              <thead>
                <tr><th></th><th>Train</th>{interim.val_loss != null && <th>Validation</th>}</tr>
              </thead>
              <tbody>
                <tr><td>Loss (NLL + BCE)</td><td>{interim.train_loss.toFixed(4)}</td>{interim.val_loss != null && <td>{interim.val_loss.toFixed(4)}</td>}</tr>
                <tr><td>Decision-day hit rate</td><td>{(interim.train_hit_rate * 100).toFixed(1)}%</td>{interim.val_hit_rate != null && <td>{(interim.val_hit_rate * 100).toFixed(1)}%</td>}</tr>
              </tbody>
            </table>
          )}

          <h3 style={{ marginTop: 16, marginBottom: 6 }}>Training log</h3>
          <div className="log-window" ref={logBoxRef}>
            {logs.length === 0 ? (
              <div className="log-line log-line-muted">Waiting for the first log line…</div>
            ) : (
              logs.map((line, i) => (
                <div key={i} className="log-line">{line}</div>
              ))
            )}
          </div>
        </div>
      )}

      {error && <p className="status-line error">{error}</p>}

      {result && (
        <>
          <div className="result-box">
            <strong>Training complete.</strong>
            <table>
              <tbody>
                <tr><td>Model saved as</td><td><code>{result.model_name}</code></td></tr>
              </tbody>
            </table>
          </div>

          <div className="panel">
            <h2 style={{ marginTop: 0 }}>Neutral band</h2>
            <div className="form-grid">
              <div className="field">
                <label>Abstention half-width around p=0.5</label>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  max="0.5"
                  value={displayBand}
                  onChange={(e) => setDisplayBand(Number(e.target.value))}
                />
              </div>
            </div>
            <p className="status-line" style={{ marginTop: 4 }}>
              Pure postprocessing - never part of training (see the model's own docs). Everything below is
              recomputed live from this model's raw predicted probabilities as you change it, with no retraining and
              no new request. Trained with {result.neutral_band.toFixed(2)}.
            </p>
          </div>

          <h2>Hit rate</h2>
          <div className="chart-grid">
            <HitRateChart pairs={result.pairs} hitRate={hitRate} />
          </div>

          <h2>Confusion matrix</h2>
          <ConfusionMatrixTable pairs={result.pairs} confusionMatrix={confusionMatrix} />

          <h2>Portfolio Sharpe by year (risk parity, probability-modulated)</h2>
          {["train", "val", "test"].map((split) => (
            <div key={split}>
              <h3>{SPLIT_LABEL[split]}</h3>
              <AnnualSharpeTable annualSharpe={result.annual_sharpe[split]} />
            </div>
          ))}

          <h2>Cumulative returns (colored by prediction hit/miss)</h2>
          {["train", "val", "test"].map((split) => (
            <div key={split}>
              <h3>{SPLIT_LABEL[split]}</h3>
              <div className="chart-grid">
                {result.pairs.map((pair) => {
                  const { probability, label } = result.probabilities[split][pair];
                  const { hit, abstained } = hitAbstainedSeries(probability, label, displayBand);
                  return (
                    <ColoredReturnChart
                      key={pair}
                      title={pair}
                      dates={result.cumulative_returns[split].dates}
                      series={{ cumulative: result.cumulative_returns[split][pair].cumulative, hit, abstained }}
                    />
                  );
                })}
              </div>
            </div>
          ))}

          <h2>Predicted probability</h2>
          {["train", "val", "test"].map((split) => (
            <div key={split}>
              <h3>{SPLIT_LABEL[split]}</h3>
              <div className="chart-grid">
                {result.pairs.map((pair) => (
                  <ProbabilityChart
                    key={pair}
                    title={pair}
                    dates={result.probabilities[split].dates}
                    series={result.probabilities[split][pair]}
                    band={displayBand}
                  />
                ))}
              </div>
            </div>
          ))}

          <h2>Forecasted vs actual return distribution</h2>
          {["train", "val", "test"].map((split) => (
            <div key={split}>
              <h3>{SPLIT_LABEL[split]}</h3>
              <div className="chart-grid">
                {result.pairs.map((pair) => (
                  <ReturnDistributionChart key={pair} title={pair} series={result.distribution[split][pair]} />
                ))}
              </div>
            </div>
          ))}
        </>
      )}
    </div>
  );
}

function NumField({ label, name, value, onChange, step = "1" }) {
  return (
    <div className="field">
      <label>{label}</label>
      <input type="number" step={step} value={value} onChange={(e) => onChange(name, e.target.value)} />
    </div>
  );
}

function DateField({ label, name, value, onChange }) {
  return (
    <div className="field">
      <label>{label}</label>
      <input type="date" value={value} onChange={(e) => onChange(name, e.target.value)} />
    </div>
  );
}
