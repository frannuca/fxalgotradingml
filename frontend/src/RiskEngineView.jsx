import { useEffect, useRef, useState } from "react";
import { getModels, trainRiskEngine, getTrainingStatus, stopTraining } from "./api";

const DEVICE_OPTIONS = [
  ["auto", "Auto (Metal/MPS if available, else CUDA, else CPU)"],
  ["cpu", "CPU"],
  ["mps", "MPS (Apple Silicon)"],
];

const DEFAULT_FORM = {
  years: 8,
  cutoff_date: "",
  train_frac: 0.8,
  test_frac: 0.1,
  device: "auto",
  risk_lookback: 20,
  risk_rolling_stats_window: 20,
  cma_windows: [], // UI shape: [{short, long}, ...] - converted to risk_cma_windows on submit
  bandpass_windows: [],
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
  cost_bps: 1.0,
  save_db: true,
  model_description: "",
};

const NUMERIC_FIELDS = new Set([
  "years", "train_frac", "test_frac", "risk_lookback", "risk_rolling_stats_window", "risk_bandpass_order",
  "min_risk_att", "max_risk_att", "risk_hidden_size", "risk_num_layers", "risk_dropout", "risk_n_attn_heads",
  "risk_epochs", "risk_lr", "risk_weight_decay", "risk_sortino_window", "full_exposure_penalty", "cost_bps",
]);

export default function RiskEngineView() {
  const [models, setModels] = useState([]);
  const [modelName, setModelName] = useState("");
  const [form, setForm] = useState(DEFAULT_FORM);
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null);
  const [progress, setProgress] = useState(null);
  const [interim, setInterim] = useState(null);
  const [logs, setLogs] = useState([]);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const pollRef = useRef(null);

  useEffect(() => {
    reloadModels();
    return () => clearInterval(pollRef.current);
  }, []);

  function reloadModels() {
    getModels().then(setModels).catch((e) => setError(e.message));
  }

  const selectedModel = models.find((m) => m.name === modelName) || null;

  function updateField(name, value) {
    setForm((f) => ({ ...f, [name]: NUMERIC_FIELDS.has(name) ? Number(value) : value }));
  }

  function addCmaWindow() {
    setForm((f) => ({ ...f, cma_windows: [...f.cma_windows, { short: 10, long: 50 }] }));
  }
  function updateCmaWindow(index, field, value) {
    setForm((f) => ({
      ...f, cma_windows: f.cma_windows.map((w, i) => (i === index ? { ...w, [field]: Number(value) } : w)),
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
      ...f, bandpass_windows: f.bandpass_windows.map((w, i) => (i === index ? { ...w, [field]: Number(value) } : w)),
    }));
  }
  function removeBandpassWindow(index) {
    setForm((f) => ({ ...f, bandpass_windows: f.bandpass_windows.filter((_, i) => i !== index) }));
  }

  async function submit(e) {
    e.preventDefault();
    setError(null);
    setResult(null);
    setProgress(null);
    setLogs([]);
    setInterim(null);
    if (!modelName) {
      setError("Select a base model.");
      return;
    }
    if (form.min_risk_att >= form.max_risk_att) {
      setError("Min risk attenuation must be less than max risk attenuation.");
      return;
    }
    try {
      const { job_id } = await trainRiskEngine({
        base_model: modelName,
        years: form.years,
        cutoff_date: form.cutoff_date || null,
        train_frac: form.train_frac,
        test_frac: form.test_frac,
        device: form.device,
        risk_lookback: form.risk_lookback,
        risk_rolling_stats_window: form.risk_rolling_stats_window,
        risk_cma_windows: form.cma_windows.map((w) => [w.short, w.long]),
        risk_bandpass_windows: form.bandpass_windows.map((w) => [w.short, w.long]),
        risk_bandpass_order: form.risk_bandpass_order,
        min_risk_att: form.min_risk_att,
        max_risk_att: form.max_risk_att,
        risk_hidden_size: form.risk_hidden_size,
        risk_num_layers: form.risk_num_layers,
        risk_dropout: form.risk_dropout,
        risk_n_attn_heads: form.risk_n_attn_heads,
        risk_epochs: form.risk_epochs,
        risk_lr: form.risk_lr,
        risk_weight_decay: form.risk_weight_decay,
        risk_sortino_window: form.risk_sortino_window,
        full_exposure_penalty: form.full_exposure_penalty,
        cost_bps: form.cost_bps,
        save_db: form.save_db,
        model_description: form.model_description,
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

  return (
    <div>
      <p className="status-line" style={{ marginTop: 0 }}>
        Train a RISK ENGINE on top of an already-trained model - a second, independently-trained LSTM that reads the
        prediction stage's own (pre-attenuation) position weights alongside each pair's rolling skewness, kurtosis,
        cross moving averages, and Butterworth bandpass, and learns a per-asset ATTENUATION FACTOR (bounded between
        "Min risk attenuation"/"Max risk attenuation") applied to the position AFTER it's already been scaled to
        target volatility - an overlay that tries to reduce exposure ahead of losses, trained with a downside-focused
        (Sortino) objective. The base model is used strictly FROZEN throughout - only the new risk engine's own
        weights are trained. The result is saved as a NEW model (base predictor + attached risk engine bundled
        together) - the base model itself is never overwritten.
      </p>

      <form onSubmit={submit}>
        <div className="panel">
          <h2 style={{ marginTop: 0 }}>1. Select base model</h2>
          <div className="model-select-row">
            <div className="field">
              <label>Model</label>
              <select value={modelName} onChange={(e) => setModelName(e.target.value)}>
                <option value="">— choose —</option>
                {models.map((m) => (
                  <option key={m.name} value={m.name}>
                    {m.name}{m.risk_engine ? " (already has a risk engine)" : ""}
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
              <strong>Base model (read only):</strong>
              <table>
                <tbody>
                  <tr><td>FX pairs</td><td>{(selectedModel.pairs || []).join(", ") || "—"}</td></tr>
                  <tr><td>Sequence length (lookback)</td><td>{selectedModel.lookback != null ? `${selectedModel.lookback} days` : "—"}</td></tr>
                  <tr><td>Direction horizon</td><td>{selectedModel.direction_horizon != null ? `${selectedModel.direction_horizon} days` : "—"}</td></tr>
                  <tr><td>Target vol</td><td>{selectedModel.target_vol != null ? `${(selectedModel.target_vol * 100).toFixed(1)}%` : "—"}</td></tr>
                </tbody>
              </table>
              {selectedModel.risk_engine && (
                <p className="status-line" style={{ marginTop: 8, color: "#b45309" }}>
                  This model already has a risk engine attached (lookback {selectedModel.risk_engine.risk_lookback},
                  bounds ({selectedModel.risk_engine.min_risk_att}, {selectedModel.risk_engine.max_risk_att})).
                  Training a new one here does not touch it - the base model is loaded fresh and re-saved under a
                  new name either way.
                </p>
              )}
            </div>
          )}
        </div>

        <div className="panel">
          <h2 style={{ marginTop: 0 }}>2. Data window</h2>
          <div className="form-grid">
            <NumField label="Years of history" name="years" value={form.years} onChange={updateField} />
            <DateField label="Cutoff date (blank = today, no cutoff)" name="cutoff_date" value={form.cutoff_date} onChange={updateField} />
            <NumField label="Train fraction (of non-test data)" name="train_frac" step="0.05" value={form.train_frac} onChange={updateField} />
            <NumField label="Test fraction (held out, most recent)" name="test_frac" step="0.05" value={form.test_frac} onChange={updateField} />
            <div className="field">
              <label>Execution device</label>
              <select value={form.device} onChange={(e) => updateField("device", e.target.value)}>
                {DEVICE_OPTIONS.map(([key, label]) => <option key={key} value={key}>{label}</option>)}
              </select>
            </div>
          </div>
        </div>

        <div className="panel">
          <h2 style={{ marginTop: 0 }}>3. Risk engine inputs</h2>
          <p className="status-line" style={{ marginTop: 0 }}>
            "Risk lookback" counts DECISION days (one per day the base model itself produced a decision - not raw
            calendar days). Rolling skewness/kurtosis are always included; cross moving averages/Butterworth
            bandpass below are independent of - and in addition to - whatever the base predictor's own features are.
          </p>
          <div className="form-grid">
            <NumField label="Risk lookback (decision days)" name="risk_lookback" value={form.risk_lookback} onChange={updateField} />
            <NumField label="Rolling stats window for skew/kurtosis (days)" name="risk_rolling_stats_window" value={form.risk_rolling_stats_window} onChange={updateField} />
          </div>
          <div style={{ marginTop: 8 }}>
            <strong style={{ fontSize: 13 }}>Cross moving average windows</strong>
            {form.cma_windows.map((w, i) => (
              <div key={i} style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 6 }}>
                <div className="field" style={{ maxWidth: 100 }}>
                  <label>Short window</label>
                  <input type="number" value={w.short} onChange={(e) => updateCmaWindow(i, "short", e.target.value)} />
                </div>
                <div className="field" style={{ maxWidth: 100 }}>
                  <label>Long window</label>
                  <input type="number" value={w.long} onChange={(e) => updateCmaWindow(i, "long", e.target.value)} />
                </div>
                <button type="button" className="secondary" onClick={() => removeCmaWindow(i)} style={{ alignSelf: "end" }}>Remove</button>
              </div>
            ))}
            <button type="button" className="secondary" onClick={addCmaWindow} style={{ marginTop: 6 }}>Add CMA window</button>
          </div>
          <div style={{ marginTop: 12 }}>
            <strong style={{ fontSize: 13 }}>Butterworth bandpass windows</strong>
            {form.bandpass_windows.map((w, i) => (
              <div key={i} style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 6 }}>
                <div className="field" style={{ maxWidth: 100 }}>
                  <label>Short period</label>
                  <input type="number" value={w.short} onChange={(e) => updateBandpassWindow(i, "short", e.target.value)} />
                </div>
                <div className="field" style={{ maxWidth: 100 }}>
                  <label>Long period</label>
                  <input type="number" value={w.long} onChange={(e) => updateBandpassWindow(i, "long", e.target.value)} />
                </div>
                <button type="button" className="secondary" onClick={() => removeBandpassWindow(i)} style={{ alignSelf: "end" }}>Remove</button>
              </div>
            ))}
            <button type="button" className="secondary" onClick={addBandpassWindow} style={{ marginTop: 6 }}>Add bandpass window</button>
            <div className="field" style={{ maxWidth: 140, marginTop: 8 }}>
              <label>Filter order</label>
              <input type="number" value={form.risk_bandpass_order} onChange={(e) => updateField("risk_bandpass_order", e.target.value)} />
            </div>
          </div>
        </div>

        <div className="panel">
          <h2 style={{ marginTop: 0 }}>4. Attenuation bounds &amp; architecture</h2>
          <div className="form-grid">
            <NumField label="Min risk attenuation" name="min_risk_att" step="0.05" value={form.min_risk_att} onChange={updateField} />
            <NumField label="Max risk attenuation" name="max_risk_att" step="0.05" value={form.max_risk_att} onChange={updateField} />
            <NumField label="Hidden size" name="risk_hidden_size" value={form.risk_hidden_size} onChange={updateField} />
            <NumField label="LSTM layers" name="risk_num_layers" value={form.risk_num_layers} onChange={updateField} />
            <NumField label="Dropout" name="risk_dropout" step="0.01" value={form.risk_dropout} onChange={updateField} />
            <NumField label="Attention heads" name="risk_n_attn_heads" value={form.risk_n_attn_heads} onChange={updateField} />
          </div>
          <p className="status-line" style={{ marginTop: 4 }}>
            Default (0, 1): the engine can only REDUCE exposure, never amplify it. The output is a per-asset factor
            multiplying that asset's already target-vol-scaled position - 0.5 halves it, 1.0 leaves it unchanged.
          </p>
        </div>

        <div className="panel">
          <h2 style={{ marginTop: 0 }}>5. Training</h2>
          <div className="form-grid">
            <NumField label="Epochs" name="risk_epochs" value={form.risk_epochs} onChange={updateField} />
            <NumField label="Learning rate" name="risk_lr" step="0.0001" value={form.risk_lr} onChange={updateField} />
            <NumField label="Weight decay" name="risk_weight_decay" step="0.0001" value={form.risk_weight_decay} onChange={updateField} />
            <NumField label="Sortino window (decision days)" name="risk_sortino_window" value={form.risk_sortino_window} onChange={updateField} />
            <NumField label="Full-exposure penalty" name="full_exposure_penalty" step="0.01" value={form.full_exposure_penalty} onChange={updateField} />
            <NumField label="Transaction cost (bps)" name="cost_bps" step="0.1" value={form.cost_bps} onChange={updateField} />
          </div>
          <p className="status-line" style={{ marginTop: 4 }}>
            Trained with a downside-focused (Sortino) objective, not plain Sharpe - only negative deviations count
            as risk, so genuinely profitable variance isn't penalized. "Full-exposure penalty" regularizes the
            engine's own output toward "Max risk attenuation" (full exposure) so it must actually EARN a de-risking
            move via the objective rather than trivially collapsing exposure - 0 disables this.
          </p>
        </div>

        <div className="panel">
          <h2 style={{ marginTop: 0 }}>6. Save</h2>
          <div className="form-grid">
            <label className="field checkbox">
              <input type="checkbox" checked={form.save_db} onChange={(e) => setForm((f) => ({ ...f, save_db: e.target.checked }))} />
              Persist to database (quant.model_registry)
            </label>
            <div className="field" style={{ gridColumn: "span 2" }}>
              <label>Model description</label>
              <input
                type="text" value={form.model_description}
                onChange={(e) => updateField("model_description", e.target.value)}
                placeholder="Optional note stored alongside the model"
              />
            </div>
          </div>
          <p className="status-line" style={{ marginTop: 4 }}>
            Saved as <code>{modelName || "<base model>"}_risk_&lt;finish-timestamp&gt;</code> - never overwrites the
            base model itself.
          </p>
        </div>

        <div className="actions">
          <button className="primary" type="submit" disabled={isBusy || !modelName}>
            {isBusy ? "Training…" : "Train risk engine"}
          </button>
          {isBusy && (
            <button type="button" className="secondary" onClick={stop}>Stop training</button>
          )}
          {status && <span className="status-line">Status: {status}</span>}
        </div>
      </form>

      {status === "stopped" && (
        <p className="status-line">Training stopped - nothing was saved. The log below shows where it left off.</p>
      )}

      {progress && (
        <div className="panel">
          <h3 style={{ marginBottom: 6 }}>Progress — epoch {progress.epoch}/{progress.total_epochs}</h3>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${progress.percent}%` }} />
          </div>
          <div className="progress-percent">{progress.percent.toFixed(1)}%</div>

          {interim && (
            <table className="weights-table" style={{ marginTop: 12 }}>
              <thead>
                <tr><th></th><th>Train</th><th>Validation</th></tr>
              </thead>
              <tbody>
                <tr>
                  <td>Sortino ratio</td>
                  <td>{interim.train_sortino != null ? interim.train_sortino.toFixed(4) : "—"}</td>
                  <td>
                    {interim.val_sortino != null ? interim.val_sortino.toFixed(4) : "—"}
                    {interim.best_val_sortino != null && (
                      <strong> (best so far: {interim.best_val_sortino.toFixed(4)})</strong>
                    )}
                  </td>
                </tr>
              </tbody>
            </table>
          )}

          <h3 style={{ marginTop: 16, marginBottom: 6 }}>Training log</h3>
          <div className="log-window">
            {logs.length === 0 ? (
              <div className="log-line log-line-muted">Waiting for the first log line…</div>
            ) : (
              logs.map((line, i) => <div key={i} className="log-line">{line}</div>)
            )}
          </div>
        </div>
      )}

      {error && <p className="status-line error">{error}</p>}

      {result && (
        <div className="panel">
          <h3 style={{ marginTop: 0 }}>Done</h3>
          <table>
            <tbody>
              <tr><td>Model saved as</td><td><code>{result.model_name}</code></td></tr>
              <tr><td>Base model</td><td><code>{result.base_model}</code></td></tr>
              <tr><td>Final train Sortino</td><td>{result.train_sortino != null ? result.train_sortino.toFixed(4) : "—"}</td></tr>
              <tr><td>Final validation Sortino</td><td>{result.val_sortino != null ? result.val_sortino.toFixed(4) : "—"}</td></tr>
            </tbody>
          </table>
          <p className="status-line" style={{ marginTop: 8 }}>
            Load <code>{result.model_name}</code> in the Evaluation view to see the risk-attenuated portfolio
            plotted alongside the modulated/baseline series.
          </p>
        </div>
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
