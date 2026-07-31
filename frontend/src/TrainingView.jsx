import { useEffect, useRef, useState } from "react";
import { getPairs, startTraining, getTrainingStatus, stopTraining } from "./api";
import PnlChart from "./charts/PnlChart";
import SeriesByPairChart from "./charts/SeriesByPairChart";

const DEFAULT_FORM = {
  pairs: [],
  lookback: 30,
  years: 8,
  train_frac: 0.8,
  test_frac: 0.1,
  weight_scheme: "softmax",
  hidden_size: 32,
  epochs: 300,
  lr: 0.001,
  dropout: 0.1,
  weight_decay: 0.0001,
  noise_std: 0.05,
  target_vol: 0.2,
  noisy_head: false,
  use_prev_weight: false,
  has_cash: false,
  cash_return: 0.0,
  use_carry: false,
  vol_horizons: [],
  encoder_type: "concat",
  asset_combiner: "attention",
  n_attn_heads: 2,
  covariance_estimator: "sample",
  ewma_lambda: 0.94,
  pooling: "last",
  device: "auto",
  objective: "sharpe",
  sharpe_window: 60,
  cvar_alpha: 0.95,
  cvar_kappa: 1.0,
  n_seeds: 1,
  restart_strategy: "best",
  risk_overlay: false,
  risk_hidden_size: 16,
  risk_epochs: 200,
  risk_lr: 0.001,
  max_attenuation: 0.33,
  risk_rolling_window: 10,
  use_cross_sectional: false,
  transaction_cost: 0,
  save_db: true,
  model_description: "",
};

const NUMERIC_FIELDS = new Set([
  "lookback", "years", "train_frac", "test_frac", "hidden_size", "epochs", "lr", "dropout",
  "weight_decay", "noise_std", "target_vol", "sharpe_window", "cvar_alpha", "cvar_kappa",
  "n_seeds", "risk_hidden_size", "risk_epochs", "risk_lr", "max_attenuation",
  "risk_rolling_window", "transaction_cost", "cash_return", "n_attn_heads", "ewma_lambda",
]);

export default function TrainingView() {
  const [availablePairs, setAvailablePairs] = useState([]);
  const [form, setForm] = useState(DEFAULT_FORM);
  const [customPair, setCustomPair] = useState("");
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null); // "pending" | "running" | "done" | "error"
  const [progress, setProgress] = useState(null);
  const [logs, setLogs] = useState([]);
  const [interim, setInterim] = useState(null);
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
    setInterim(null);
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

  const allPairs = Array.from(new Set([...availablePairs, ...form.pairs]));
  const isBusy = status === "pending" || status === "running";
  // The backend appends "CASH" as one more asset when has_cash is on (see
  // models/portfolio_lstm.py's _prepare_data) - result.positions_train/
  // attenuation_train etc. include a "CASH" series, so the pairs list used
  // to render them must too, or that line is silently omitted.
  const chartPairs = form.has_cash ? [...form.pairs, "CASH"] : form.pairs;

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
            <NumField label="Train fraction (of non-test data)" name="train_frac" step="0.05" value={form.train_frac} onChange={updateField} />
            <NumField label="Test fraction (held out, most recent)" name="test_frac" step="0.05" value={form.test_frac} onChange={updateField} />
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
            <SelectField
              label="Covariance estimator (vol targeting)"
              name="covariance_estimator"
              value={form.covariance_estimator}
              onChange={updateField}
              options={[
                ["sample", "Sample covariance (original)"],
                ["ewma", "EWMA (RiskMetrics-style)"],
                ["ledoit_wolf", "Ledoit-Wolf shrinkage"],
              ]}
            />
            {form.covariance_estimator === "ewma" && (
              <NumField label="EWMA lambda (decay)" name="ewma_lambda" step="0.01" value={form.ewma_lambda} onChange={updateField} />
            )}
            <NumField label="Transaction cost (bps)" name="transaction_cost" step="0.5" value={form.transaction_cost} onChange={updateField} />
            <SelectField
              label="Training objective"
              name="objective"
              value={form.objective}
              onChange={updateField}
              options={[
                ["sharpe", "Rolling-window Sharpe (Sortino-style)"],
                ["kelly", "Log-wealth (Kelly criterion)"],
                ["cvar", "Mean-CVaR"],
              ]}
            />
            {form.objective === "sharpe" && (
              <NumField label="Sharpe window (days)" name="sharpe_window" value={form.sharpe_window} onChange={updateField} />
            )}
            {form.objective === "cvar" && (
              <>
                <NumField label="CVaR confidence" name="cvar_alpha" step="0.01" value={form.cvar_alpha} onChange={updateField} />
                <NumField label="CVaR risk weight (kappa)" name="cvar_kappa" step="0.1" value={form.cvar_kappa} onChange={updateField} />
              </>
            )}
            <label className="field checkbox">
              <input
                type="checkbox"
                checked={form.noisy_head}
                onChange={(e) => setForm((f) => ({ ...f, noisy_head: e.target.checked }))}
              />
              Noisy output head (NoisyNet)
            </label>
            <label className="field checkbox">
              <input
                type="checkbox"
                checked={form.use_prev_weight}
                onChange={(e) => setForm((f) => ({ ...f, use_prev_weight: e.target.checked }))}
              />
              Feed previous weight into allocator
            </label>
            <label className="field checkbox">
              <input
                type="checkbox"
                checked={form.has_cash}
                onChange={(e) => setForm((f) => ({ ...f, has_cash: e.target.checked }))}
              />
              Add cash as an asset
            </label>
            {form.has_cash && (
              <NumField label="Cash return (daily)" name="cash_return" step="0.0001" value={form.cash_return} onChange={updateField} />
            )}
            <label className="field checkbox">
              <input
                type="checkbox"
                checked={form.use_carry}
                onChange={(e) => setForm((f) => ({ ...f, use_carry: e.target.checked }))}
              />
              Add FX carry (interest-rate differential)
            </label>
            <label className="field">
              Vol-normalized return horizons (days, comma-separated)
              <input
                type="text"
                value={form.vol_horizons.join(",")}
                onChange={(e) =>
                  setForm((f) => ({
                    ...f,
                    vol_horizons: e.target.value
                      .split(",")
                      .map((s) => parseInt(s.trim(), 10))
                      .filter((n) => Number.isFinite(n) && n > 0),
                  }))
                }
                placeholder="e.g. 5,20"
              />
            </label>
            <SelectField
              label="Encoder architecture"
              name="encoder_type"
              value={form.encoder_type}
              onChange={updateField}
              options={[
                ["concat", "Concatenated input (one LSTM over all assets)"],
                ["per_asset", "Shared per-asset encoder + cross-asset attention"],
              ]}
            />
            {form.encoder_type === "per_asset" && (
              <>
                <SelectField
                  label="Cross-asset combiner"
                  name="asset_combiner"
                  value={form.asset_combiner}
                  onChange={updateField}
                  options={[
                    ["attention", "Self-attention across assets"],
                    ["mean", "Mean-pool across assets"],
                  ]}
                />
                {form.asset_combiner === "attention" && (
                  <NumField label="Attention heads" name="n_attn_heads" value={form.n_attn_heads} onChange={updateField} />
                )}
              </>
            )}
            <SelectField
              label="Time pooling (both networks)"
              name="pooling"
              value={form.pooling}
              onChange={updateField}
              options={[
                ["last", "Last hidden state (h_n[-1])"],
                ["attention", "Attention pooling over all timesteps"],
              ]}
            />
            <SelectField
              label="Compute device"
              name="device"
              value={form.device}
              onChange={updateField}
              options={[
                ["auto", "Auto (Metal/MPS GPU if available, else CPU)"],
                ["cpu", "CPU"],
                ["mps", "Force Metal (MPS) GPU"],
              ]}
            />
          </div>
          <p className="status-line" style={{ marginTop: 4 }}>
            Chronological 3-way split: the most recent {(form.test_frac * 100).toFixed(0)}% of history is held out as
            a TEST set, never used for training or checkpoint selection - a genuinely unbiased read on
            generalization. Of what remains, the oldest {(form.train_frac * 100).toFixed(0)}% is train and the rest
            is validation (which DOES influence best-epoch checkpoint selection).
          </p>
          {form.has_cash && (
            <p className="status-line" style={{ marginTop: 4 }}>
              Adds "CASH" as one more asset the allocator can hold (constant daily return above, zero by default) -
              lets it de-risk directly instead of relying only on the risk overlay. Enable both to compare: does the
              risk overlay still help once the allocator can just hold cash?
            </p>
          )}
          {(form.use_carry || form.vol_horizons.length > 0) && (
            <p className="status-line" style={{ marginTop: 4 }}>
              Adds extra per-asset input channels alongside the raw return: carry (base minus quote interest rate,
              from FRED via data/rates_downloader.py) and/or each horizon's trailing cumulative-return-over-vol
              ratio - more context per asset, without changing the allocator's output (still one weight per asset).
            </p>
          )}
          {form.use_prev_weight && (
            <p className="status-line" style={{ marginTop: 4 }}>
              The allocator sees its own previous day's position alongside each window, so it can learn whether a
              rebalance is worth its transaction cost - pairs naturally with "Transaction cost" above. Training and
              evaluation become a genuine day-by-day recurrence instead of one parallel batch, so this is
              meaningfully slower.
            </p>
          )}
          {form.encoder_type === "per_asset" && (
            <p className="status-line" style={{ marginTop: 4 }}>
              One shared LSTM encodes each asset's own window independently, then assets{" "}
              {form.asset_combiner === "attention" ? "attend to each other" : "share a mean-pooled market context"}{" "}
              before a shared per-asset head produces each asset's weight - unlike the concatenated design, this
              treats every asset with the same learned weights (permutation-invariant), rather than tying one input
              column to one specific asset.
            </p>
          )}
          {form.pooling === "attention" && (
            <p className="status-line" style={{ marginTop: 4 }}>
              Both the allocator and (if enabled) the risk overlay summarize their lookback window with learned
              attention pooling over every timestep's hidden state, instead of just the LSTM's final hidden state -
              lets the network learn which days in the window matter most for the current decision.
            </p>
          )}
          {form.covariance_estimator !== "sample" && (
            <p className="status-line" style={{ marginTop: 4 }}>
              {form.covariance_estimator === "ewma"
                ? "Volatility targeting weights recent days more heavily (exponential decay) instead of treating every day in the lookback window equally - reacts faster to a genuine vol regime change."
                : "Volatility targeting shrinks the sample covariance toward a well-conditioned target using the analytically optimal Ledoit-Wolf intensity - avoids the near-singular covariance estimates that can otherwise produce extreme leverage on a coincidentally calm window."}
            </p>
          )}
          {form.objective === "sharpe" && (
            <p className="status-line" style={{ marginTop: 10 }}>
              Scores the mean Sortino-style ratio over overlapping {form.sharpe_window}-day windows of the
              training period (not one ratio over the whole period) - not convex, but directly comparable to the
              Sharpe numbers reported elsewhere.
            </p>
          )}
          {form.objective === "kelly" && (
            <p className="status-line" style={{ marginTop: 10 }}>
              Maximizes expected log-wealth growth (Kelly criterion) - convex in the portfolio weights, no extra
              tunable weight, and targets long-run compounded growth rather than a single-period ratio.
            </p>
          )}
          {form.objective === "cvar" && (
            <p className="status-line" style={{ marginTop: 10 }}>
              Maximizes mean return net of a CVaR tail-risk penalty (average loss on the worst {" "}
              {((1 - form.cvar_alpha) * 100).toFixed(0)}% of days, weighted by kappa) - convex, at the cost of a
              fixed risk-aversion weight (kappa) to combine return and tail risk into one objective.
            </p>
          )}
          <p className="status-line" style={{ marginTop: 4 }}>
            Whichever epoch has the best validation Sharpe (always the plain, whole-period metric) is kept at the
            end instead of always the last one - aimed at generalizing better out-of-sample instead of maximizing
            the in-sample fit.
          </p>
          {form.noisy_head && (
            <p className="status-line" style={{ marginTop: 4 }}>
              Noisy head adds learnable weight noise (resampled every training step) to the portfolio's output
              layer, and to the risk overlay's output layer when enabled below - composes with "Noise std" above.
              Deterministic at evaluation time.
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
              <label className="field checkbox">
                <input
                  type="checkbox"
                  checked={form.use_cross_sectional}
                  onChange={(e) => setForm((f) => ({ ...f, use_cross_sectional: e.target.checked }))}
                />
                Add cross-sectional correlation features
              </label>
            </div>
          )}
          {form.risk_overlay && form.use_cross_sectional && (
            <p className="status-line" style={{ marginTop: 4 }}>
              Adds 3 portfolio-wide signals from the rolling cross-asset correlation matrix (average pairwise
              correlation, correlation dispersion, top eigenvalue share) alongside the existing per-asset
              std/skew/kurtosis - lets the attenuation head see market-wide risk concentration, not just each
              asset's own marginal risk.
            </p>
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
          {isBusy && (
            <button type="button" className="secondary" onClick={stop}>
              Stop training
            </button>
          )}
          {status && <span className="status-line">Status: {status}</span>}
        </div>
      </form>

      {status === "stopped" && (
        <p className="status-line">Training stopped - nothing was saved. The charts/log below show where it left off.</p>
      )}

      {progress && (status === "running" || status === "pending" || status === "done" || status === "stopped") && (
        <div className="panel">
          <h3 style={{ marginBottom: 6 }}>
            Progress{progress.n_seeds > 1 ? ` — restart ${progress.seed_index}/${progress.n_seeds}` : ""}
            {" "}— epoch {progress.epoch}/{progress.total_epochs}
          </h3>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${progress.percent}%` }} />
          </div>
          <div className="progress-percent">{progress.percent.toFixed(1)}%</div>

          {interim && (
            <div style={{ marginTop: 16 }}>
              <PnlChart
                title={
                  (interim.stage === "risk_overlay" ? "Risk overlay training" : "Portfolio training") +
                  ` — in-sample, epoch ${interim.epoch}/${interim.total_epochs}` +
                  ` (Sharpe ${interim.sharpe.toFixed(2)})`
                }
                pnl={{ dates: interim.cumulative_pnl.map((_, i) => i), live: interim.cumulative_pnl }}
                seriesKeys={["live"]}
                height={260}
              />
            </div>
          )}

          {interim && interim.val_cumulative_pnl && (
            <div style={{ marginTop: 16 }}>
              <PnlChart
                title={
                  (interim.stage === "risk_overlay" ? "Risk overlay training" : "Portfolio training") +
                  ` — validation, epoch ${interim.epoch}/${interim.total_epochs}` +
                  ` (Sharpe ${interim.val_sharpe.toFixed(2)})`
                }
                pnl={{ dates: interim.val_cumulative_pnl.map((_, i) => i), live: interim.val_cumulative_pnl }}
                seriesKeys={["live"]}
                height={260}
              />
            </div>
          )}

          {interim && interim.test_cumulative_pnl && (
            <div style={{ marginTop: 16 }}>
              <PnlChart
                title={
                  (interim.stage === "risk_overlay" ? "Risk overlay training" : "Portfolio training") +
                  ` — test, epoch ${interim.epoch}/${interim.total_epochs}` +
                  ` (Sharpe ${interim.test_sharpe.toFixed(2)})`
                }
                pnl={{ dates: interim.test_cumulative_pnl.map((_, i) => i), live: interim.test_cumulative_pnl }}
                seriesKeys={["live"]}
                height={260}
              />
            </div>
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
                  <td>Validation</td>
                  <td>{result.val_sharpe_raw.toFixed(3)}</td>
                  <td>{result.val_sharpe_vol_targeted.toFixed(3)}</td>
                  {result.risk_model_name && <td>{result.val_sharpe_with_risk.toFixed(3)}</td>}
                </tr>
                <tr>
                  <td>Test</td>
                  <td>{result.test_sharpe_raw.toFixed(3)}</td>
                  <td>{result.test_sharpe_vol_targeted.toFixed(3)}</td>
                  {result.risk_model_name && <td>{result.test_sharpe_with_risk.toFixed(3)}</td>}
                </tr>
              </tbody>
            </table>
          </div>

          <h2>Cumulative PnL</h2>
          <div className="chart-grid">
            <PnlChart
              title="In-sample"
              pnl={result.pnl_train}
              seriesKeys={
                result.risk_model_name
                  ? ["vol_targeted", "with_risk", "benchmark"]
                  : ["vol_targeted", "benchmark"]
              }
            />
            <PnlChart
              title="Validation"
              pnl={result.pnl_val}
              seriesKeys={
                result.risk_model_name
                  ? ["vol_targeted", "with_risk", "benchmark"]
                  : ["vol_targeted", "benchmark"]
              }
            />
            <PnlChart
              title="Test"
              pnl={result.pnl_test}
              seriesKeys={
                result.risk_model_name
                  ? ["vol_targeted", "with_risk", "benchmark"]
                  : ["vol_targeted", "benchmark"]
              }
            />
          </div>
          <p className="status-line" style={{ marginTop: 4 }}>
            "Inverse-vol benchmark" is a simple, un-learned risk-parity allocator (weight each asset inversely to its
            own trailing volatility), rescaled to match the MODEL's own realized volatility on each split
            separately - so the comparison isolates whether the learned allocation adds value, not just whether the
            model happened to run hotter or colder than the benchmark.
          </p>

          <h2>Cumulative FX pair returns</h2>
          <div className="chart-grid">
            <SeriesByPairChart
              title="Full history (dashed line marks out-of-sample start)"
              series={result.asset_returns}
              pairs={chartPairs}
              splitDate={result.asset_returns.split_date}
              height={340}
            />
          </div>

          <h2>Portfolio positions</h2>
          <div className="chart-grid">
            <SeriesByPairChart title="In-sample" series={result.positions_train} pairs={chartPairs} />
            <SeriesByPairChart title="Out-of-sample" series={result.positions_val} pairs={chartPairs} />
          </div>

          {result.risk_model_name && (
            <>
              <h2>Risk attenuation factor</h2>
              <div className="chart-grid">
                <SeriesByPairChart title="In-sample" series={result.attenuation_train} pairs={chartPairs} />
                <SeriesByPairChart title="Out-of-sample" series={result.attenuation_val} pairs={chartPairs} />
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
