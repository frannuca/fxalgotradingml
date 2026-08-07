import { useEffect, useRef, useState } from "react";
import { getPairs, startTraining, getTrainingStatus, stopTraining } from "./api";
import TrainingProgressPanel from "./TrainingProgressPanel";
import TrainingResultPanel from "./TrainingResultPanel";
import { confusionMatrixForSplit, hitRateForSplit } from "./metrics";

// Every feature build_feature_dataframe knows how to produce (see
// models/portfolio_lstm.py's FEATURE_CATALOG) - "cma" is special, it
// expands to one channel PER window pair in form.cma_windows, not one.
const FEATURE_OPTIONS = [
  ["log_return", "Log return"],
  ["vol", "Rolling volatility"],
  ["carry", "Carry (interest-rate differential)"],
  ["cma", "Cross moving averages (trend)"],
  ["bandpass", "Butterworth bandpass (faster-reacting trend)"],
];

const DEVICE_OPTIONS = [
  ["auto", "Auto (Metal/MPS if available, else CUDA, else CPU)"],
  ["cpu", "CPU"],
  ["mps", "MPS (Apple Silicon)"],
];

// Which per-epoch VALIDATION metric selects each asset's own restored
// checkpoint (see models/portfolio_lstm.py's train_prediction_model
// docstring on checkpoint_metric) - independent of "Sharpe weight", which
// controls what TRAINS the weights, not which epoch's weights get kept.
const CHECKPOINT_METRIC_OPTIONS = [
  ["val_loss", "Validation loss (default)"],
  ["hit_rate", "Validation hit rate"],
  ["sharpe", "Validation Sharpe"],
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
  features: ["log_return", "vol"],
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
  // adds a joint portfolio-Sharpe term to the SAME combined training
  // loss: risk-parity weight x this model's own probability signal,
  // scaled to target_vol below, maximizing an averaged Sharpe over
  // NON-OVERLAPPING sharpe_window-day chunks (walking backward from the
  // last day) - see models/portfolio_lstm.py's
  // _non_overlapping_sharpe_torch/train_prediction_model docstrings.
  sharpe_weight: 0,
  sharpe_window: 20,
  // Which per-epoch VALIDATION metric selects each asset's own restored
  // checkpoint - independent of sharpe_weight above (see
  // models/portfolio_lstm.py's train_prediction_model docstring on
  // checkpoint_metric). "val_loss" (default): unchanged original
  // behavior. "hit_rate"/"sharpe": select on that instead.
  checkpoint_metric: "val_loss",
  neutral_band: 0.05,
  // Annualized volatility the Evaluation view's portfolio PnL calculator
  // scales this model's positions to (see models/portfolio_pnl.py) -
  // persisted with the model, like neutral_band, not a free
  // evaluation-time parameter. Also used DURING training if
  // sharpe_weight above is > 0.
  target_vol: 0.1,
  // {pair: [min, max]} - per-asset bounds the decision-day probability is
  // linearly mapped into before scaling the risk-parity baseline weight
  // (see models/portfolio_lstm.py's resolve_signal_bounds/
  // models/portfolio_pnl.py's compute_portfolio). A pair missing here
  // defaults to (-1, 1) - the original signed-direction map. Narrowing a
  // pair's range (e.g. [0, 1]) makes it long-only instead of a signed bet.
  signal_range: {},
  n_seeds: 1,
  device: "auto",
  save_db: true,
  model_description: "",

  // --- Optional purged K-fold cross-validation (see
  // models/portfolio_lstm.py's run_kfold_pipeline) - a more robust MODEL
  // SELECTION alternative to the single train/val split above: trains
  // n_folds independent models on purged, embargoed splits (still
  // respecting "Number of seeds"/BCE-weight sweep above, per fold) and
  // keeps whichever fold validated best by "Checkpoint selection metric"
  // above - reuses that SAME selector and "Years of history"/"Cutoff date"
  // above rather than duplicating them here.
  use_kfold_cv: false,
  n_folds: 5,

  // --- Optional risk-engine second phase (see models/risk_engine.py) ---
  // When true, api/server.py's _run_training_job continues STRAIGHT into
  // training a NEW RiskEngine on top of THIS run's own just-trained
  // (frozen) model, right after phase 1 finishes, in the SAME job - one
  // bundled checkpoint gets saved, not two separate models.
  train_risk_engine: false,
  risk_lookback: 20,
  risk_rolling_stats_window: 20,
  risk_cma_windows: [], // UI shape [{short, long}, ...] - converted to [[short,long],...] on submit
  risk_bandpass_windows: [],
  risk_bandpass_order: 3,
  min_risk_att: 0.0,
  max_risk_att: 1.0,
  risk_hidden_size: 16,
  risk_num_layers: 1,
  risk_dropout: 0.1,
  risk_n_attn_heads: 4,
  risk_epochs: 100,
  risk_lr: 0.001,
  risk_weight_decay: 0.0,
  risk_sortino_window: 20,
  full_exposure_penalty: 0.05,
};

const NUMERIC_FIELDS = new Set([
  "lookback", "years", "train_frac", "test_frac", "direction_horizon", "rolling_stats_window",
  "hidden_size", "num_layers", "dropout", "n_attn_heads",
  "epochs", "lr", "weight_decay", "neutral_band", "target_vol",
  "n_seeds", "bandpass_order", "sharpe_weight", "sharpe_window", "n_folds",
  "risk_lookback", "risk_rolling_stats_window", "risk_bandpass_order", "min_risk_att", "max_risk_att",
  "risk_hidden_size", "risk_num_layers", "risk_dropout", "risk_n_attn_heads",
  "risk_epochs", "risk_lr", "risk_weight_decay", "risk_sortino_window", "full_exposure_penalty",
]);

// "1.0" -> 1.0 (single value); "1.0, 1.5, 2.0" -> [1.0, 1.5, 2.0] (sweep -
// see models/portfolio_lstm.py's run_pipeline_multi_seed, which trains
// every value under every seed and keeps whichever validated best).
function parseBceWeight(text) {
  const values = text.split(",").map((s) => Number(s.trim())).filter((n) => !Number.isNaN(n));
  if (values.length === 0) return 1.0;
  return values.length === 1 ? values[0] : values;
}

// Inverse of submit()/downloadConfig()'s own transform - turns a raw
// models/portfolio_lstm.py DEFAULT_CONFIG-shaped JSON (see main.py's own
// docstring/configs/*.json, and this file's own "Save config" button)
// back into this form's UI shape. Merged onto DEFAULT_FORM rather than
// replacing it outright, matching main.py's own "any key left out falls
// back to the default" behavior for a config that only sets SOME keys.
function parseConfigToForm(raw) {
  return {
    ...DEFAULT_FORM,
    ...raw,
    pairs: raw.pairs || [],
    cross_pairs: raw.cross_pairs || {},
    signal_range: raw.signal_range || {},
    features: raw.features || DEFAULT_FORM.features,
    cutoff_date: raw.cutoff_date || "",
    bce_weight: raw.bce_weight == null
      ? DEFAULT_FORM.bce_weight
      : Array.isArray(raw.bce_weight)
        ? raw.bce_weight.join(", ")
        : String(raw.bce_weight),
    cma_windows: (raw.cma_windows || []).map(([short, long]) => ({ short, long })),
    bandpass_windows: (raw.bandpass_windows || []).map(([short, long]) => ({ short, long })),
    risk_cma_windows: (raw.risk_cma_windows || []).map(([short, long]) => ({ short, long })),
    risk_bandpass_windows: (raw.risk_bandpass_windows || []).map(([short, long]) => ({ short, long })),
  };
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
  const loadConfigInputRef = useRef(null);

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
      // referenced as an input source. Same for signal_range: a removed
      // pair can't stay configured with a bound that no longer applies.
      let cross_pairs = f.cross_pairs;
      let signal_range = f.signal_range;
      if (wasIncluded) {
        cross_pairs = Object.fromEntries(
          Object.entries(f.cross_pairs)
            .filter(([p]) => p !== pair)
            .map(([p, others]) => [p, others.filter((o) => o !== pair)])
            .filter(([, others]) => others.length > 0),
        );
        signal_range = Object.fromEntries(Object.entries(f.signal_range).filter(([p]) => p !== pair));
      }
      return { ...f, pairs, cross_pairs, signal_range };
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

  function addRiskCmaWindow() {
    setForm((f) => ({ ...f, risk_cma_windows: [...f.risk_cma_windows, { short: 10, long: 50 }] }));
  }

  function updateRiskCmaWindow(index, field, value) {
    setForm((f) => ({
      ...f,
      risk_cma_windows: f.risk_cma_windows.map((w, i) => (i === index ? { ...w, [field]: Number(value) } : w)),
    }));
  }

  function removeRiskCmaWindow(index) {
    setForm((f) => ({ ...f, risk_cma_windows: f.risk_cma_windows.filter((_, i) => i !== index) }));
  }

  function addRiskBandpassWindow() {
    setForm((f) => ({ ...f, risk_bandpass_windows: [...f.risk_bandpass_windows, { short: 10, long: 50 }] }));
  }

  function updateRiskBandpassWindow(index, field, value) {
    setForm((f) => ({
      ...f,
      risk_bandpass_windows: f.risk_bandpass_windows.map((w, i) => (i === index ? { ...w, [field]: Number(value) } : w)),
    }));
  }

  function removeRiskBandpassWindow(index) {
    setForm((f) => ({ ...f, risk_bandpass_windows: f.risk_bandpass_windows.filter((_, i) => i !== index) }));
  }

  // (min, max) default to (-1, 1) - the same default the backend applies
  // to any pair absent from form.signal_range (see models/portfolio_lstm.py's
  // resolve_signal_bounds) - so the field always shows a meaningful value
  // even before the pair has an explicit override.
  function updateSignalRange(pair, which, value) {
    setForm((f) => {
      const [min, max] = f.signal_range[pair] || [-1, 1];
      const updated = which === "min" ? [Number(value), max] : [min, Number(value)];
      return { ...f, signal_range: { ...f.signal_range, [pair]: updated } };
    });
  }

  function resetSignalRange(pair) {
    setForm((f) => {
      const signal_range = { ...f.signal_range };
      delete signal_range[pair];
      return { ...f, signal_range };
    });
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

  // Same transform submit() sends to POST /api/train, reused by
  // downloadConfig() below - this IS the shape models/portfolio_lstm.py's
  // DEFAULT_CONFIG expects (see main.py's own docstring/configs/*.json),
  // so a saved file is directly usable as `python main.py path/to/this.json`
  // as well as re-importable back into this form (see parseConfigToForm,
  // its own inverse, used by handleLoadConfig below).
  function buildConfig() {
    return {
      ...form,
      cutoff_date: form.cutoff_date || null,
      bce_weight: parseBceWeight(form.bce_weight),
      cma_windows: form.cma_windows.map((w) => [w.short, w.long]),
      bandpass_windows: form.bandpass_windows.map((w) => [w.short, w.long]),
      risk_cma_windows: form.risk_cma_windows.map((w) => [w.short, w.long]),
      risk_bandpass_windows: form.risk_bandpass_windows.map((w) => [w.short, w.long]),
    };
  }

  function downloadConfig() {
    if (form.pairs.length < 1) {
      setError("Select at least one FX pair.");
      return;
    }
    const blob = new Blob([JSON.stringify(buildConfig(), null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    a.href = url;
    a.download = `training_config_${form.pairs.join("-")}_${stamp}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  function handleLoadConfig(e) {
    const file = e.target.files[0];
    e.target.value = ""; // reset so re-selecting the SAME file still fires onChange next time
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const raw = JSON.parse(reader.result);
        setForm(parseConfigToForm(raw));
        setError(null);
        setResult(null);
        setProgress(null);
        setLogs([]);
        setInterim(null);
      } catch (err) {
        setError(`Could not load config file: ${err.message}`);
      }
    };
    reader.onerror = () => setError("Could not read the selected file.");
    reader.readAsText(file);
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
    if (form.train_risk_engine && form.min_risk_att >= form.max_risk_att) {
      setError("Min risk attenuation must be less than max risk attenuation.");
      return;
    }
    if (form.use_kfold_cv && form.n_folds < 2) {
      setError("Number of folds must be at least 2.");
      return;
    }
    try {
      const { job_id } = await startTraining(buildConfig());
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
              label="Rolling stats window (volatility)"
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
            FEATURE_CATALOG) - "Rolling volatility" uses the {form.rolling_stats_window}-day window above; "Carry" is
            the interest-rate differential (FRED); "Cross moving averages" adds a trend/momentum channel PER window
            pair below (short-window mean return minus long-window mean return - positive when the recent trend runs
            above the longer-run one). Rolling skewness/kurtosis are risk-engine-only inputs now - configure them
            below under "Risk engine".
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
            <div className="field">
              <label>Checkpoint selection metric</label>
              <select value={form.checkpoint_metric} onChange={(e) => updateField("checkpoint_metric", e.target.value)}>
                {CHECKPOINT_METRIC_OPTIONS.map(([key, label]) => (
                  <option key={key} value={key}>{label}</option>
                ))}
              </select>
            </div>
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
            "Sharpe weight" 0 keeps the above unchanged. {'>'} 0 adds a Sharpe term to the SAME combined loss every
            epoch (one shared backward pass together with NLL+BCE, not a separate step - this avoids the two
            gradients ever partially undoing each other, which a naive two-step version would risk): the model's
            own probability signal × a precomputed (data-only, never optimized) risk-parity weight, scaled to
            "Target vol", drives an averaged Sharpe - computed over NON-OVERLAPPING "Sharpe window"-day chunks,
            walking backward from the most recent day, never a smoothed/overlapping rolling statistic - that gets
            maximized, the same strategy the Evaluation page's portfolio PnL calculator reports. Unlike everything else here, this
            term is NOT per-asset independent: the vol-scaling denominator mixes every asset's current signal, so
            one asset's training genuinely depends on what the others are currently predicting. Keep this modest
            relative to "BCE direction weight" - directly maximizing Sharpe on noisy daily returns can drift toward
            low-variance, low-conviction bets rather than genuine predictive signal.
          </p>
          <p className="status-line" style={{ marginTop: 4 }}>
            "Checkpoint selection metric" controls which epoch's weights each asset keeps at the end - independent
            of "Sharpe weight" above, which controls what TRAINS the weights, not which epoch's checkpoint gets
            restored. "Validation loss" (default) selects the epoch with the lowest own NLL+BCE validation loss
            (minus the Sharpe term if "Sharpe weight" {'>'} 0). "Hit rate" instead keeps whichever epoch had the best
            validation directional hit rate - a coarser, saturating metric (once every sample's sign is already
            correct it can't improve further, so this tends to lock onto an early epoch and ignore later loss
            improvements) offered as an explicit choice, not a recommendation. "Sharpe" keeps whichever epoch had
            the best validation portfolio Sharpe - works even with "Sharpe weight" left at 0 (training stays pure
            NLL+BCE; only which checkpoint gets kept changes).
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

        {form.pairs.length > 0 && (
          <div className="panel">
            <h2 style={{ marginTop: 0 }}>Signal range (portfolio position limits)</h2>
            <p className="status-line" style={{ marginTop: 0 }}>
              The decision-day probability is linearly mapped into a (min, max) range, then multiplied by that
              pair's risk-parity baseline weight to size its position - by DEFAULT (-1, 1) for every pair, a signed
              direction (positive probability -&gt; long, negative -&gt; short). Narrow a pair's range to change what the
              mapped value MEANS: e.g. (0, 1) makes that pair LONG-ONLY (a 0-1 factor scaling the risk-parity
              weight, never negated); (-0.1, 1.0) allows only a little short. A day inside the "Neutral band" above
              is treated as no view regardless of this range - it rides the UNMODULATED risk-parity weight (the same
              diversified position "Risk-parity baseline" reports), not a value from this map.
            </p>
            {form.pairs.map((pair) => {
              const [min, max] = form.signal_range[pair] || [-1, 1];
              const isDefault = !form.signal_range[pair];
              return (
                <div key={pair} style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 6 }}>
                  <strong style={{ fontSize: 13, minWidth: 70 }}>{pair}</strong>
                  <div className="field" style={{ maxWidth: 110 }}>
                    <label>Min</label>
                    <input type="number" step="0.05" value={min} onChange={(e) => updateSignalRange(pair, "min", e.target.value)} />
                  </div>
                  <div className="field" style={{ maxWidth: 110 }}>
                    <label>Max</label>
                    <input type="number" step="0.05" value={max} onChange={(e) => updateSignalRange(pair, "max", e.target.value)} />
                  </div>
                  {!isDefault && (
                    <button type="button" className="secondary" onClick={() => resetSignalRange(pair)} style={{ alignSelf: "end" }}>
                      Reset to default (-1, 1)
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        )}

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
          <h2 style={{ marginTop: 0 }}>Cross-validation (K-fold)</h2>
          <label className="field checkbox">
            <input
              type="checkbox"
              checked={form.use_kfold_cv}
              onChange={(e) => setForm((f) => ({ ...f, use_kfold_cv: e.target.checked }))}
            />
            Use purged K-fold cross-validation for model selection instead of a single validation split
          </label>
          <p className="status-line" style={{ marginTop: 4 }}>
            A single validation split is a noisy, small-sample estimate - a config can look "best" by luck as much
            as by genuine quality, especially once several seeds/BCE-weight values above are all competing for it.
            Enabling this instead splits the data (still governed by "Years of history"/"Cutoff date" above,
            outside "Test fraction", which stays a genuinely untouched final holdout) into "Number of folds" purged,
            embargoed blocks (a buffer of lookback + direction_horizon decision days is dropped on EACH side of
            every validation block, so no training sample's label or lookback window ever overlaps what it's
            validated against - see models/portfolio_lstm.py's generate_purged_kfold_splits) and trains one fully
            independent model per fold (each still respecting "Number of seeds"/the BCE-weight sweep above,
            multiplying total training cost by "Number of folds"). "Checkpoint selection metric" above scores every
            fold's own validation split, exactly as it already scores epochs/restarts - whichever fold validates
            best is the one that gets saved; the full per-fold score distribution (mean and spread, not just the
            winner) is reported once training finishes.
          </p>
          {form.use_kfold_cv && (
            <div className="form-grid" style={{ marginTop: 8 }}>
              <NumField label="Number of folds" name="n_folds" value={form.n_folds} onChange={updateField} />
            </div>
          )}
        </div>

        <div className="panel">
          <h2 style={{ marginTop: 0 }}>Risk engine (optional second phase)</h2>
          <label className="field checkbox">
            <input
              type="checkbox"
              checked={form.train_risk_engine}
              onChange={(e) => setForm((f) => ({ ...f, train_risk_engine: e.target.checked }))}
            />
            Train a risk engine on top of this run's own model, right after it finishes
          </label>
          <p className="status-line" style={{ marginTop: 4 }}>
            A second, independently-trained LSTM that reads THIS run's own (pre-attenuation) position weights
            alongside each pair's rolling skewness/kurtosis, cross moving averages, and Butterworth bandpass below,
            and learns a per-asset ATTENUATION FACTOR (bounded between "Min risk attenuation"/"Max risk
            attenuation") applied to the position AFTER it's already been scaled to target volatility - trained with
            a downside-focused (Sortino) objective, with the just-trained predictor above used strictly FROZEN. Runs
            as PHASE 2 of this SAME job, right after phase 1 (the prediction model above) finishes, with its own
            progress reporting below - one bundled checkpoint gets saved at the end, not two separate models. Skew
            and kurtosis are risk-engine-exclusive inputs now (see "Features" above, which no longer offers them for
            the predictor itself).
          </p>

          {form.train_risk_engine && (
            <>
              <div className="form-grid" style={{ marginTop: 8 }}>
                <NumField label="Risk lookback (decision days)" name="risk_lookback" value={form.risk_lookback} onChange={updateField} />
                <NumField label="Rolling stats window for skew/kurtosis (days)" name="risk_rolling_stats_window" value={form.risk_rolling_stats_window} onChange={updateField} />
                <NumField label="Min risk attenuation" name="min_risk_att" step="0.05" value={form.min_risk_att} onChange={updateField} />
                <NumField label="Max risk attenuation" name="max_risk_att" step="0.05" value={form.max_risk_att} onChange={updateField} />
                <NumField label="Hidden size" name="risk_hidden_size" value={form.risk_hidden_size} onChange={updateField} />
                <NumField label="LSTM layers" name="risk_num_layers" value={form.risk_num_layers} onChange={updateField} />
                <NumField label="Dropout" name="risk_dropout" step="0.01" value={form.risk_dropout} onChange={updateField} />
                <NumField label="Attention heads" name="risk_n_attn_heads" value={form.risk_n_attn_heads} onChange={updateField} />
                <NumField label="Epochs" name="risk_epochs" value={form.risk_epochs} onChange={updateField} />
                <NumField label="Learning rate" name="risk_lr" step="0.0001" value={form.risk_lr} onChange={updateField} />
                <NumField label="Weight decay" name="risk_weight_decay" step="0.0001" value={form.risk_weight_decay} onChange={updateField} />
                <NumField label="Sortino window (decision days)" name="risk_sortino_window" value={form.risk_sortino_window} onChange={updateField} />
                <NumField label="Full-exposure penalty" name="full_exposure_penalty" step="0.01" value={form.full_exposure_penalty} onChange={updateField} />
              </div>
              <p className="status-line" style={{ marginTop: 4 }}>
                Default attenuation bounds (0, 1): the engine can only REDUCE exposure, never amplify it. "Full-
                exposure penalty" regularizes the engine's own output toward "Max risk attenuation" so it must
                actually EARN a de-risking move via the Sortino objective rather than trivially collapsing exposure -
                0 disables this.
              </p>

              <div style={{ marginTop: 8 }}>
                <strong style={{ fontSize: 13 }}>
                  Cross moving average windows (risk engine - independent of the predictor's own above)
                </strong>
                {form.risk_cma_windows.map((w, i) => (
                  <div key={i} style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 6 }}>
                    <div className="field" style={{ maxWidth: 100 }}>
                      <label>Short window</label>
                      <input type="number" value={w.short} onChange={(e) => updateRiskCmaWindow(i, "short", e.target.value)} />
                    </div>
                    <div className="field" style={{ maxWidth: 100 }}>
                      <label>Long window</label>
                      <input type="number" value={w.long} onChange={(e) => updateRiskCmaWindow(i, "long", e.target.value)} />
                    </div>
                    <button type="button" className="secondary" onClick={() => removeRiskCmaWindow(i)} style={{ alignSelf: "end" }}>
                      Remove
                    </button>
                  </div>
                ))}
                <button type="button" className="secondary" onClick={addRiskCmaWindow} style={{ marginTop: 6 }}>
                  Add CMA window
                </button>
              </div>

              <div style={{ marginTop: 12 }}>
                <strong style={{ fontSize: 13 }}>Butterworth bandpass windows (risk engine)</strong>
                {form.risk_bandpass_windows.map((w, i) => (
                  <div key={i} style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 6 }}>
                    <div className="field" style={{ maxWidth: 100 }}>
                      <label>Short period</label>
                      <input type="number" value={w.short} onChange={(e) => updateRiskBandpassWindow(i, "short", e.target.value)} />
                    </div>
                    <div className="field" style={{ maxWidth: 100 }}>
                      <label>Long period</label>
                      <input type="number" value={w.long} onChange={(e) => updateRiskBandpassWindow(i, "long", e.target.value)} />
                    </div>
                    <button type="button" className="secondary" onClick={() => removeRiskBandpassWindow(i)} style={{ alignSelf: "end" }}>
                      Remove
                    </button>
                  </div>
                ))}
                <button type="button" className="secondary" onClick={addRiskBandpassWindow} style={{ marginTop: 6 }}>
                  Add bandpass window
                </button>
                <div className="field" style={{ maxWidth: 140, marginTop: 8 }}>
                  <label>Filter order</label>
                  <input
                    type="number"
                    value={form.risk_bandpass_order}
                    onChange={(e) => updateField("risk_bandpass_order", e.target.value)}
                  />
                </div>
              </div>
            </>
          )}
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
          <button type="button" className="secondary" onClick={downloadConfig} disabled={isBusy}>
            Save config
          </button>
          <button type="button" className="secondary" onClick={() => loadConfigInputRef.current.click()} disabled={isBusy}>
            Load config
          </button>
          <input
            type="file"
            accept=".json,application/json"
            ref={loadConfigInputRef}
            onChange={handleLoadConfig}
            style={{ display: "none" }}
          />
          {isBusy && (
            <button type="button" className="secondary" onClick={stop}>
              Stop training
            </button>
          )}
          {status && <span className="status-line">Status: {status}</span>}
        </div>
        <p className="status-line" style={{ marginTop: 4 }}>
          "Save config" downloads the CURRENT form as a JSON file matching this project's config format (see
          main.py's own docstring/configs/*.json) - directly runnable as <code>python main.py path/to/file.json</code>,
          with no training run required first. "Load config" reads one back in (any key it leaves out keeps this
          form's current value) - use it to replay a saved run's exact settings, or hand-edit a file and load it
          back rather than clicking through every field.
        </p>
      </form>

      {status === "stopped" && (
        <p className="status-line">Training stopped - nothing was saved. The log below shows where it left off.</p>
      )}

      {progress && (status === "running" || status === "pending" || status === "done" || status === "stopped") && (
        <TrainingProgressPanel progress={progress} interim={interim} logs={logs} logBoxRef={logBoxRef} jobId={jobId} />
      )}

      {error && <p className="status-line error">{error}</p>}

      <TrainingResultPanel
        result={result}
        hitRate={hitRate}
        confusionMatrix={confusionMatrix}
        displayBand={displayBand}
        setDisplayBand={setDisplayBand}
      />
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
