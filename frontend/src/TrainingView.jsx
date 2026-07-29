import { useEffect, useRef, useState } from "react";
import { getPairs, startTraining, getTrainingStatus } from "./api";
import PnlChart from "./charts/PnlChart";
import SeriesByPairChart from "./charts/SeriesByPairChart";

const DEFAULT_FORM = {
  pairs: [],
  lookback: 30,
  years: 8,
  train_frac: 0.8,
  weight_scheme: "softmax",
  hidden_size: 32,
  epochs: 300,
  lr: 0.001,
  dropout: 0.1,
  weight_decay: 0.0001,
  noise_std: 0.05,
  target_vol: 0.2,
  noisy_head: false,
  n_seeds: 1,
  restart_strategy: "best",
  risk_overlay: false,
  risk_hidden_size: 16,
  risk_epochs: 200,
  risk_lr: 0.001,
  max_attenuation: 0.33,
  risk_rolling_window: 10,
  transaction_cost: 0,
  save_db: true,
  model_description: "",
};

const NUMERIC_FIELDS = new Set([
  "lookback", "years", "train_frac", "hidden_size", "epochs", "lr", "dropout",
  "weight_decay", "noise_std", "target_vol", "n_seeds", "risk_hidden_size",
  "risk_epochs", "risk_lr", "max_attenuation", "risk_rolling_window", "transaction_cost",
]);

export default function TrainingView() {
  const [availablePairs, setAvailablePairs] = useState([]);
  const [form, setForm] = useState(DEFAULT_FORM);
  const [customPair, setCustomPair] = useState("");
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null); // "pending" | "running" | "done" | "error"
  const [progress, setProgress] = useState(null);
  const [logs, setLogs] = useState([]);
  const [result, setResult] = useState(null);
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
    setForm((f) => ({
      ...f,
      pairs: f.pairs.includes(pair) ? f.pairs.filter((p) => p !== pair) : [...f.pairs, pair],
    }));
  }

  function addCustomPair() {
    const pair = customPair.trim().toUpperCase();
    if (pair && !form.pairs.includes(pair)) {
      setForm((f) => ({ ...f, pairs: [...f.pairs, pair] }));
    }
    setCustomPair("");
  }

  async function submit(e) {
    e.preventDefault();
    setError(null);
    setResult(null);
    setProgress(null);
    setLogs([]);
    if (form.pairs.length < 2) {
      setError("Select at least two FX pairs.");
      return;
    }
    try {
      const { job_id } = await startTraining(form);
      setJobId(job_id);
      setStatus("pending");
      pollRef.current = setInterval(async () => {
        const job = await getTrainingStatus(job_id);
        setStatus(job.status);
        setProgress(job.progress || null);
        setLogs(job.logs || []);
        if (job.status === "done") {
          clearInterval(pollRef.current);
          setResult(job.result);
        } else if (job.status === "error") {
          clearInterval(pollRef.current);
          setError(job.error);
        }
      }, 1500);
    } catch (err) {
      setError(err.message);
    }
  }

  const allPairs = Array.from(new Set([...availablePairs, ...form.pairs]));
  const isBusy = status === "pending" || status === "running";

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
          <h2 style={{ marginTop: 0 }}>Data &amp; portfolio allocator</h2>
          <div className="form-grid">
            <NumField label="Lookback (days)" name="lookback" value={form.lookback} onChange={updateField} />
            <NumField label="Years of history" name="years" value={form.years} onChange={updateField} />
            <NumField label="Train fraction" name="train_frac" step="0.05" value={form.train_frac} onChange={updateField} />
            <SelectField
              label="Weight scheme"
              name="weight_scheme"
              value={form.weight_scheme}
              onChange={updateField}
              options={[["softmax", "Long-only (softmax)"], ["tanh_norm", "Long/short (tanh)"]]}
            />
            <NumField label="Hidden size" name="hidden_size" value={form.hidden_size} onChange={updateField} />
            <NumField label="Epochs" name="epochs" value={form.epochs} onChange={updateField} />
            <NumField label="Learning rate" name="lr" step="0.0001" value={form.lr} onChange={updateField} />
            <NumField label="Dropout" name="dropout" step="0.01" value={form.dropout} onChange={updateField} />
            <NumField label="Weight decay" name="weight_decay" step="0.0001" value={form.weight_decay} onChange={updateField} />
            <NumField label="Noise std" name="noise_std" step="0.01" value={form.noise_std} onChange={updateField} />
            <NumField label="Target volatility" name="target_vol" step="0.01" value={form.target_vol} onChange={updateField} />
            <NumField label="Transaction cost (bps)" name="transaction_cost" step="0.5" value={form.transaction_cost} onChange={updateField} />
            <label className="field checkbox">
              <input
                type="checkbox"
                checked={form.noisy_head}
                onChange={(e) => setForm((f) => ({ ...f, noisy_head: e.target.checked }))}
              />
              Noisy output head (NoisyNet)
            </label>
          </div>
          {form.noisy_head && (
            <p className="status-line" style={{ marginTop: 10 }}>
              Adds learnable weight noise (resampled every training step) to the portfolio's output layer, and to
              the risk overlay's output layer when enabled below - composes with "Noise std" above. Deterministic
              at evaluation time.
            </p>
          )}
        </div>

        <div className="panel">
          <h2 style={{ marginTop: 0 }}>Multi-seed restarts</h2>
          <div className="form-grid">
            <NumField label="Number of seeds" name="n_seeds" value={form.n_seeds} onChange={updateField} />
            <SelectField
              label="Restart strategy"
              name="restart_strategy"
              value={form.restart_strategy}
              onChange={updateField}
              options={[["best", "Best of restarts"], ["ensemble", "Ensemble average"]]}
            />
          </div>
        </div>

        <div className="panel">
          <h2 style={{ marginTop: 0 }}>
            <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={form.risk_overlay}
                onChange={(e) => setForm((f) => ({ ...f, risk_overlay: e.target.checked }))}
              />
              Risk-attenuation overlay
            </label>
          </h2>
          {form.risk_overlay && (
            <div className="form-grid">
              <NumField label="Risk hidden size" name="risk_hidden_size" value={form.risk_hidden_size} onChange={updateField} />
              <NumField label="Risk epochs" name="risk_epochs" value={form.risk_epochs} onChange={updateField} />
              <NumField label="Risk learning rate" name="risk_lr" step="0.0001" value={form.risk_lr} onChange={updateField} />
              <NumField label="Max attenuation floor" name="max_attenuation" step="0.01" value={form.max_attenuation} onChange={updateField} />
              <NumField label="Risk rolling window" name="risk_rolling_window" value={form.risk_rolling_window} onChange={updateField} />
            </div>
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
          {status && <span className="status-line">Status: {status}</span>}
        </div>
      </form>

      {progress && (status === "running" || status === "pending" || status === "done") && (
        <div className="panel">
          <h3 style={{ marginBottom: 6 }}>
            Progress{progress.n_seeds > 1 ? ` — restart ${progress.seed_index}/${progress.n_seeds}` : ""}
            {" "}— epoch {progress.epoch}/{progress.total_epochs}
          </h3>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${progress.percent}%` }} />
          </div>
          <div className="progress-percent">{progress.percent.toFixed(1)}%</div>

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
                <tr><td>Portfolio model saved as</td><td><code>{result.portfolio_model_name}</code></td></tr>
                {result.risk_model_name && (
                  <tr><td>Risk model saved as</td><td><code>{result.risk_model_name}</code></td></tr>
                )}
              </tbody>
            </table>
          </div>

          <h2>Sharpe ratio</h2>
          <div className="panel">
            <table className="weights-table">
              <thead>
                <tr>
                  <th></th>
                  <th>Raw</th>
                  <th>Vol-targeted</th>
                  {result.risk_model_name && <th>With risk overlay</th>}
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>In-sample</td>
                  <td>{result.train_sharpe_raw.toFixed(3)}</td>
                  <td>{result.train_sharpe_vol_targeted.toFixed(3)}</td>
                  {result.risk_model_name && <td>{result.train_sharpe_with_risk.toFixed(3)}</td>}
                </tr>
                <tr>
                  <td>Out-of-sample</td>
                  <td>{result.val_sharpe_raw.toFixed(3)}</td>
                  <td>{result.val_sharpe_vol_targeted.toFixed(3)}</td>
                  {result.risk_model_name && <td>{result.val_sharpe_with_risk.toFixed(3)}</td>}
                </tr>
              </tbody>
            </table>
          </div>

          <h2>Cumulative PnL</h2>
          <div className="chart-grid">
            <PnlChart
              title="In-sample"
              pnl={result.pnl_train}
              seriesKeys={result.risk_model_name ? ["vol_targeted", "with_risk"] : ["vol_targeted"]}
            />
            <PnlChart
              title="Out-of-sample"
              pnl={result.pnl_val}
              seriesKeys={result.risk_model_name ? ["vol_targeted", "with_risk"] : ["vol_targeted"]}
            />
          </div>

          <h2>Cumulative FX pair returns</h2>
          <div className="chart-grid">
            <SeriesByPairChart
              title="Full history (dashed line marks out-of-sample start)"
              series={result.asset_returns}
              pairs={form.pairs}
              splitDate={result.asset_returns.split_date}
              height={340}
            />
          </div>

          <h2>Portfolio positions</h2>
          <div className="chart-grid">
            <SeriesByPairChart title="In-sample" series={result.positions_train} pairs={form.pairs} />
            <SeriesByPairChart title="Out-of-sample" series={result.positions_val} pairs={form.pairs} />
          </div>

          {result.risk_model_name && (
            <>
              <h2>Risk attenuation factor</h2>
              <div className="chart-grid">
                <SeriesByPairChart title="In-sample" series={result.attenuation_train} pairs={form.pairs} />
                <SeriesByPairChart title="Out-of-sample" series={result.attenuation_val} pairs={form.pairs} />
              </div>
            </>
          )}
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

function SelectField({ label, name, value, onChange, options }) {
  return (
    <div className="field">
      <label>{label}</label>
      <select value={value} onChange={(e) => onChange(name, e.target.value)}>
        {options.map(([val, text]) => (
          <option key={val} value={val}>{text}</option>
        ))}
      </select>
    </div>
  );
}
