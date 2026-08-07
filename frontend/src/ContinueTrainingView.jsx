import { useEffect, useRef, useState } from "react";
import { getModels, startTraining, getTrainingStatus, stopTraining } from "./api";
import TrainingProgressPanel from "./TrainingProgressPanel";
import TrainingResultPanel from "./TrainingResultPanel";
import { confusionMatrixForSplit, hitRateForSplit } from "./metrics";

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

// Only fields that DON'T define the network's architecture - epochs,
// optimization knobs, and the data window - are editable here (see
// models/portfolio_lstm.py's continue_training docstring: every
// architecture-defining property - pairs, lookback, hidden_size,
// num_layers, dropout, n_attn_heads, cross_pairs, features, cma_windows,
// bandpass_windows, bandpass_order - is recovered from the base model's
// own checkpoint server-side and CANNOT be changed from here, hence no
// form fields for any of them; see the read-only architecture table below
// instead).
const DEFAULT_FORM = {
  years: 8,
  cutoff_date: "",
  train_frac: 0.8,
  test_frac: 0.1,
  direction_horizon: 5,
  rolling_stats_window: 20,
  epochs: 100,
  lr: 0.001,
  weight_decay: 0.0001,
  bce_weight: "1.0",
  sharpe_weight: 0,
  sharpe_window: 20,
  checkpoint_metric: "val_loss",
  neutral_band: 0.05,
  target_vol: 0.1,
  // {pair: [min, max]} - per-asset bounds the decision-day probability is
  // linearly mapped into before scaling the risk-parity baseline weight
  // (see models/portfolio_lstm.py's resolve_signal_bounds). Pre-filled
  // from the chosen base model's own persisted value in selectModel()
  // below, same as neutral_band/target_vol - still freely editable.
  signal_range: {},
  device: "auto",
  save_db: true,
  model_description: "",
};

const NUMERIC_FIELDS = new Set([
  "years", "train_frac", "test_frac", "direction_horizon", "rolling_stats_window",
  "epochs", "lr", "weight_decay", "neutral_band", "target_vol", "sharpe_weight", "sharpe_window",
]);

// "1.0" -> 1.0 (single value); "1.0, 1.5, 2.0" -> [1.0, 1.5, 2.0] (sweep -
// see models/portfolio_lstm.py's run_pipeline_multi_seed). Continue-
// training itself is always a single run (no seed/lambda sweep - see
// continue_training), but a multi-value bce_weight is harmless to accept
// and just takes the first value server-side would be surprising, so this
// view keeps it a single free-text number rather than a sweep-capable one.
function parseBceWeight(text) {
  const value = Number(text);
  return Number.isNaN(value) ? 1.0 : value;
}

export default function ContinueTrainingView() {
  const [models, setModels] = useState([]);
  const [modelName, setModelName] = useState("");
  const [form, setForm] = useState(DEFAULT_FORM);
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null); // "pending" | "running" | "done" | "error" | "stopped"
  const [progress, setProgress] = useState(null);
  const [logs, setLogs] = useState([]);
  const [interim, setInterim] = useState(null);
  const [result, setResult] = useState(null);
  const [displayBand, setDisplayBand] = useState(0.05);
  const [error, setError] = useState(null);
  const pollRef = useRef(null);
  const logBoxRef = useRef(null);

  useEffect(() => {
    reloadModels();
    return () => clearInterval(pollRef.current);
  }, []);

  useEffect(() => {
    if (logBoxRef.current) {
      logBoxRef.current.scrollTop = logBoxRef.current.scrollHeight;
    }
  }, [logs]);

  function reloadModels() {
    getModels().then(setModels).catch((e) => setError(e.message));
  }

  const selectedModel = models.find((m) => m.name === modelName) || null;

  function selectModel(name) {
    setModelName(name);
    // Pre-fill every editable field this model has its OWN persisted value
    // for (how it's meant to be traded, AND what its labels/features mean -
    // direction_horizon/rolling_stats_window) - a reasonable starting point
    // for continuing it, still freely overridable (unlike the architecture
    // table below, none of these are actually LOCKED by continue_training -
    // see models/portfolio_lstm.py's own docstring on why).
    const model = models.find((m) => m.name === name);
    if (model) {
      setForm((f) => ({
        ...f,
        neutral_band: model.neutral_band ?? f.neutral_band,
        target_vol: model.target_vol ?? f.target_vol,
        signal_range: model.signal_range ?? f.signal_range,
        direction_horizon: model.direction_horizon ?? f.direction_horizon,
        rolling_stats_window: model.rolling_stats_window ?? f.rolling_stats_window,
      }));
    }
  }

  function updateField(name, value) {
    setForm((f) => ({ ...f, [name]: NUMERIC_FIELDS.has(name) ? Number(value) : value }));
  }

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

  async function submit(e) {
    e.preventDefault();
    setError(null);
    setResult(null);
    setProgress(null);
    setLogs([]);
    setInterim(null);
    if (!modelName) {
      setError("Select a model to continue training.");
      return;
    }
    try {
      // `pairs`/`lookback` are required by the request shape but IGNORED
      // server-side whenever `continue_from` is set (recovered from the
      // base model's own checkpoint instead - see api/server.py's
      // _run_training_job) - sending the base model's own pairs here is
      // just a courtesy, not load-bearing.
      const { job_id } = await startTraining({
        continue_from: modelName,
        pairs: selectedModel ? selectedModel.pairs : [],
        ...form,
        cutoff_date: form.cutoff_date || null,
        bce_weight: parseBceWeight(form.bce_weight),
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

  const isBusy = status === "pending" || status === "running";

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
      <p className="status-line" style={{ marginTop: 0 }}>
        Warm-start training from an EXISTING model's weights instead of a fresh random init - i.e. keep optimizing
        a model you already trained, rather than starting over. Only epochs, optimization/objective knobs, and the
        data window are adjustable; the network's own architecture (shown read-only below) is recovered from the
        chosen model's checkpoint and cannot change - editing it would produce a different, incompatible network,
        not a genuine continuation. The result is always saved under a NEW name (the base model's own name, suffixed
        with this run's finish time) - the base model itself is never overwritten.
      </p>

      <form onSubmit={submit}>
        <div className="panel">
          <h2 style={{ marginTop: 0 }}>1. Select base model</h2>
          <div className="model-select-row">
            <div className="field">
              <label>Model</label>
              <select value={modelName} onChange={(e) => selectModel(e.target.value)}>
                <option value="">— choose —</option>
                {models.map((m) => (
                  <option key={m.name} value={m.name}>
                    {m.name}{m.is_ensemble ? ` (ensemble of ${m.n_members})` : ""}
                  </option>
                ))}
              </select>
            </div>
            <button type="button" className="secondary" onClick={reloadModels} style={{ alignSelf: "end" }}>
              Refresh model list
            </button>
          </div>

          {selectedModel && (
            <div className="result-box" style={{ marginTop: 14 }}>
              <strong>Architecture (recovered from the checkpoint - read only, cannot be changed here):</strong>
              <table>
                <tbody>
                  <tr><td>FX pairs</td><td>{(selectedModel.pairs || []).join(", ") || "—"}</td></tr>
                  <tr><td>Sequence length (lookback)</td><td>{selectedModel.lookback != null ? `${selectedModel.lookback} days` : "—"}</td></tr>
                  <tr><td>Features</td><td>{(selectedModel.features || []).join(", ") || "—"}</td></tr>
                  <tr><td>Cross moving average windows</td><td>{(selectedModel.cma_windows || []).map((w) => `${w[0]}/${w[1]}`).join(", ") || "—"}</td></tr>
                  <tr><td>Bandpass windows</td><td>{(selectedModel.bandpass_windows || []).map((w) => `${w[0]}/${w[1]}`).join(", ") || "—"}</td></tr>
                  <tr><td>Bandpass filter order</td><td>{selectedModel.bandpass_order ?? "—"}</td></tr>
                  <tr><td>Hidden size</td><td>{selectedModel.hidden_size ?? "—"}</td></tr>
                  <tr><td>LSTM layers</td><td>{selectedModel.num_layers ?? "—"}</td></tr>
                  <tr><td>Dropout</td><td>{selectedModel.dropout ?? "—"}</td></tr>
                  <tr><td>Attention heads</td><td>{selectedModel.n_attn_heads ?? "—"}</td></tr>
                  <tr>
                    <td>Cross-pair links</td>
                    <td>
                      {selectedModel.cross_pairs && Object.keys(selectedModel.cross_pairs).length > 0
                        ? Object.entries(selectedModel.cross_pairs).map(([p, others]) => `${p} ← ${others.join(", ")}`).join("; ")
                        : "none (every pair fully independent)"}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          )}

          {selectedModel && selectedModel.is_ensemble && (
            <p className="status-line" style={{ marginTop: 8, color: "#b45309" }}>
              This is a {selectedModel.n_members}-member ensemble (from K-fold cross-validation). Continuing it
              continues EVERY member independently with the settings below, then saves a new ensemble bundling all
              {selectedModel.n_members} continued members - not a single model. Any risk engines already attached are
              never carried forward (they were trained against each member's OLD weights); risk-engine retraining
              isn't exposed in this view yet.
            </p>
          )}

          {selectedModel && (
            <div className="result-box" style={{ marginTop: 14 }}>
              <strong>Currently saved as (pre-filled into the editable fields below - freely changeable there):</strong>
              <table>
                <tbody>
                  <tr><td>Direction horizon</td><td>{selectedModel.direction_horizon != null ? `${selectedModel.direction_horizon} days` : "—"}</td></tr>
                  <tr><td>Rolling stats window</td><td>{selectedModel.rolling_stats_window != null ? `${selectedModel.rolling_stats_window} days` : "—"}</td></tr>
                  <tr><td>Neutral band</td><td>{selectedModel.neutral_band ?? "—"}</td></tr>
                  <tr><td>Target vol (annualized)</td><td>{selectedModel.target_vol != null ? `${(selectedModel.target_vol * 100).toFixed(1)}%` : "—"}</td></tr>
                  <tr>
                    <td>Signal range (min, max)</td>
                    <td>
                      {selectedModel.signal_range && Object.keys(selectedModel.signal_range).length > 0
                        ? Object.entries(selectedModel.signal_range).map(([p, [min, max]]) => `${p}: (${min}, ${max})`).join("; ")
                        : "default (-1, 1) for every pair"}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div className="panel">
          <h2 style={{ marginTop: 0 }}>2. Data window</h2>
          <div className="form-grid">
            <NumField label="Years of history" name="years" value={form.years} onChange={updateField} />
            <DateField
              label="Cutoff date (blank = today, no cutoff)"
              name="cutoff_date"
              value={form.cutoff_date}
              onChange={updateField}
            />
            <NumField label="Train fraction (of non-test data)" name="train_frac" step="0.05" value={form.train_frac} onChange={updateField} />
            <NumField label="Test fraction (held out, most recent)" name="test_frac" step="0.05" value={form.test_frac} onChange={updateField} />
            <NumField label="Direction horizon (forward days)" name="direction_horizon" value={form.direction_horizon} onChange={updateField} />
            <NumField label="Rolling stats window (vol/skew/kurtosis)" name="rolling_stats_window" value={form.rolling_stats_window} onChange={updateField} />
          </div>
          <p className="status-line" style={{ marginTop: 4 }}>
            Free to pick a different (e.g. more recent, or extended) window than the base model's own original
            training run - continuation re-fetches and re-splits data fresh from these settings, still standardized
            with the BASE MODEL's own mean/std (never refit), since its weights were trained expecting that exact
            scaling.
          </p>
        </div>

        <div className="panel">
          <h2 style={{ marginTop: 0 }}>3. Training &amp; objective</h2>
          <div className="form-grid">
            <NumField label="Additional epochs" name="epochs" value={form.epochs} onChange={updateField} />
            <NumField label="Learning rate" name="lr" step="0.0001" value={form.lr} onChange={updateField} />
            <NumField label="Weight decay" name="weight_decay" step="0.0001" value={form.weight_decay} onChange={updateField} />
            <div className="field">
              <label>BCE direction weight</label>
              <input
                type="text"
                value={form.bce_weight}
                onChange={(e) => setForm((f) => ({ ...f, bce_weight: e.target.value }))}
                placeholder="e.g. 1.0"
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
            "Neutral band"/"Target vol" are pre-filled from the base model's own saved values when you pick it above,
            but stay freely editable - same as every other field here. "Sharpe weight" &gt; 0 adds the same joint
            portfolio-Sharpe training phase a fresh training run would (see the Training page's own explanation) -
            it works starting from ANY base model, including one that was never trained with it before.
          </p>
        </div>

        {selectedModel && (
          <div className="panel">
            <h2 style={{ marginTop: 0 }}>Signal range (portfolio position limits)</h2>
            <p className="status-line" style={{ marginTop: 0 }}>
              Per-pair (min, max) the decision-day probability is linearly mapped into before scaling the
              risk-parity baseline weight - pre-filled from the base model's own saved value, still freely editable.
              Default (-1, 1) is a signed direction; narrowing a pair's range (e.g. (0, 1)) makes it long-only
              instead. A day inside "Neutral band" below rides the unmodulated risk-parity weight regardless of this
              range, not a value from this map.
            </p>
            {selectedModel.pairs.map((pair) => {
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
          <h2 style={{ marginTop: 0 }}>4. Save</h2>
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
                onChange={(e) => setForm((f) => ({ ...f, model_description: e.target.value }))}
                placeholder="Optional note stored alongside the model"
              />
            </div>
          </div>
          <p className="status-line" style={{ marginTop: 4 }}>
            Saved as <code>{modelName || "<base model>"}_continued_&lt;finish-timestamp&gt;</code> - never overwrites
            the base model itself.
          </p>
        </div>

        <div className="actions">
          <button className="primary" type="submit" disabled={isBusy || !modelName}>
            {isBusy ? "Training…" : "Continue training"}
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
