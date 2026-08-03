import { useEffect, useState } from "react";
import { getModels, refreshQuotes, evaluate } from "./api";
import HitRateChart from "./charts/HitRateChart";
import ColoredReturnChart from "./charts/ColoredReturnChart";
import ProbabilityChart from "./charts/ProbabilityChart";
import ReturnDistributionChart from "./charts/ReturnDistributionChart";
import PortfolioPositionChart from "./charts/PortfolioPositionChart";
import PortfolioPnlChart from "./charts/PortfolioPnlChart";
import ConfusionMatrixTable from "./ConfusionMatrixTable";
import AnnualSharpeTable from "./AnnualSharpeTable";
import { confusionMatrixForSplit, hitAbstainedSeries, hitRateForSplit } from "./metrics";

const DEVICE_OPTIONS = [
  ["auto", "Auto (Metal/MPS if available, else CUDA, else CPU)"],
  ["cpu", "CPU"],
  ["mps", "MPS (Apple Silicon)"],
];

export default function EvaluationView() {
  const [models, setModels] = useState([]);
  const [pairs, setPairs] = useState([]);
  const [lookback, setLookback] = useState(null);
  const [modelName, setModelName] = useState("");
  const [params, setParams] = useState({ years: 8, cutoff_date: "", device: "auto" });
  const [refreshYears, setRefreshYears] = useState(1);
  const [refreshStatus, setRefreshStatus] = useState(null);
  const [evalStatus, setEvalStatus] = useState(null);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  // See TrainingView.jsx's own displayBand - same idea: pure
  // postprocessing, freely adjustable without re-running evaluation.
  const [displayBand, setDisplayBand] = useState(0.05);

  useEffect(() => {
    reloadModels();
  }, []);

  function reloadModels() {
    getModels().then(setModels).catch((e) => setError(e.message));
  }

  function selectModel(name) {
    setModelName(name);
    // Pairs and lookback (sequence length) are properties of the trained
    // model itself, not free evaluation parameters - recover both from its
    // checkpoint (see /api/models) instead of asking the user to supply
    // values that must happen to match it exactly.
    const model = models.find((m) => m.name === name);
    setPairs(model && model.pairs ? model.pairs : []);
    setLookback(model ? model.lookback : null);
  }

  function updateParam(name, value) {
    setParams((p) => ({ ...p, [name]: Number(value) }));
  }

  async function doRefreshQuotes() {
    setError(null);
    setRefreshStatus("refreshing");
    try {
      const res = await refreshQuotes({ pairs, years: refreshYears });
      setRefreshStatus(`Updated ${Object.values(res.rows).reduce((a, b) => a + b, 0)} rows across ${pairs.length} pair(s).`);
    } catch (e) {
      setRefreshStatus(null);
      setError(e.message);
    }
  }

  async function runEvaluation(e) {
    e.preventDefault();
    setError(null);
    setResult(null);
    if (!modelName) {
      setError("Select a model.");
      return;
    }
    setEvalStatus("running");
    try {
      // No `pairs`/`lookback` sent - the backend recovers both from the
      // selected model's own checkpoint (see api/server.py's evaluate()).
      const res = await evaluate({ ...params, cutoff_date: params.cutoff_date || null, model_name: modelName });
      setResult(res);
      setDisplayBand(res.neutral_band);
      setEvalStatus("done");
    } catch (err) {
      setEvalStatus(null);
      setError(err.message);
    }
  }

  const minYears = lookback ? Math.ceil(lookback / 252 + 0.2) : null; // rough padding above the raw day count

  // Recomputed from raw probability + realized label (see ./metrics.js)
  // every time displayBand changes - the band never touches evaluation
  // itself, so this is the ONLY place it's actually applied to this
  // result. Evaluation is a single continuous period now (no train/val/
  // test split - see api/server.py's evaluate()), so these cover the
  // whole thing directly.
  const hitRate = result && hitRateForSplit(result.probabilities, result.pairs, displayBand);
  const confusionMatrix = result && confusionMatrixForSplit(result.probabilities, result.pairs, displayBand);

  return (
    <div>
      <form onSubmit={runEvaluation}>
        <div className="panel">
          <h2 style={{ marginTop: 0 }}>1. Select model</h2>
          <div className="model-select-row">
            <div className="field">
              <label>Model</label>
              <select value={modelName} onChange={(e) => selectModel(e.target.value)}>
                <option value="">— choose —</option>
                {models.map((m) => (
                  <option key={m.name} value={m.name}>{m.name}</option>
                ))}
              </select>
            </div>
            <button type="button" className="secondary" onClick={reloadModels} style={{ alignSelf: "end" }}>
              Refresh model list
            </button>
          </div>

          {modelName && (
            <div className="result-box" style={{ marginTop: 14 }}>
              <strong>Recovered from the model itself (not editable here):</strong>
              <table>
                <tbody>
                  <tr><td>FX pairs</td><td>{pairs.join(", ") || "—"}</td></tr>
                  <tr><td>Sequence length (lookback)</td><td>{lookback != null ? `${lookback} days` : "—"}</td></tr>
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div className="panel">
          <h2 style={{ marginTop: 0 }}>2. Download latest quotes</h2>
          <div className="actions" style={{ marginTop: 0 }}>
            <div className="field" style={{ maxWidth: 140 }}>
              <label>Years to fetch</label>
              <input type="number" value={refreshYears} onChange={(e) => setRefreshYears(Number(e.target.value))} />
            </div>
            <button
              type="button"
              className="secondary"
              onClick={doRefreshQuotes}
              disabled={refreshStatus === "refreshing" || pairs.length === 0}
            >
              {refreshStatus === "refreshing" ? "Refreshing…" : "Refresh quotes into DB"}
            </button>
            {typeof refreshStatus === "string" && refreshStatus !== "refreshing" && (
              <span className="status-line">{refreshStatus}</span>
            )}
          </div>
        </div>

        <div className="panel">
          <h2 style={{ marginTop: 0 }}>3. Evaluation parameters</h2>
          <p className="status-line" style={{ marginTop: 0 }}>
            {minYears
              ? `This model needs at least ${lookback} days of history - "years" must fetch more than that (roughly ${minYears}+ years).`
              : "How much historical data to use - the model's own pairs/lookback aren't editable here. Evaluation runs over the WHOLE fetched period as one continuous read, not split into train/validation/test (that split only matters during training)."}
          </p>
          <div className="form-grid">
            <NumField label="Years of history" name="years" value={params.years} onChange={updateParam} />
            <div className="field">
              <label>Cutoff date (blank = today, no cutoff)</label>
              <input
                type="date"
                value={params.cutoff_date}
                onChange={(e) => setParams((p) => ({ ...p, cutoff_date: e.target.value }))}
              />
            </div>
            <div className="field">
              <label>Execution device</label>
              <select value={params.device} onChange={(e) => setParams((p) => ({ ...p, device: e.target.value }))}>
                {DEVICE_OPTIONS.map(([key, label]) => (
                  <option key={key} value={key}>{label}</option>
                ))}
              </select>
            </div>
          </div>
        </div>

        <div className="actions">
          <button className="primary" type="submit" disabled={evalStatus === "running"}>
            {evalStatus === "running" ? "Running…" : "Run model"}
          </button>
        </div>
      </form>

      {error && <p className="status-line error">{error}</p>}

      {result && (
        <>
          {JSON.stringify([...result.pairs].sort()) !== JSON.stringify([...pairs].sort()) && (
            <p className="status-line" style={{ color: "#b45309" }}>
              Note: this model was trained on {result.pairs.join(", ")} - results below use those pairs,
              not the {pairs.join(", ")} selected above.
            </p>
          )}

          <h2>Predicted probabilities (next day)</h2>
          <div className="panel">
            <table className="weights-table">
              <thead>
                <tr><th>Pair</th><th>P(positive)</th></tr>
              </thead>
              <tbody>
                {Object.entries(result.latest_probabilities).map(([pair, p]) => (
                  <tr key={pair}>
                    <td>{pair}</td>
                    <td>{(p * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h2>Today&apos;s position (EoD booking sheet)</h2>
          <div className="panel">
            <p className="status-line" style={{ marginTop: 0 }}>
              Risk-parity weight times the probability signal <code>(p - 0.5) * 2</code>, smoothed over this model&apos;s own
              direction horizon and scaled to its target volatility of {(result.target_vol * 100).toFixed(1)}% annualized
              (a model property, recovered from its checkpoint) - see the Portfolio PnL section below for the full
              backtest this is based on.
            </p>
            <table className="weights-table">
              <thead>
                <tr><th>Pair</th><th>P(positive)</th><th>Modulated position</th><th>Risk-parity baseline</th></tr>
              </thead>
              <tbody>
                {result.pairs.map((pair) => {
                  const row = result.latest_position[pair];
                  return (
                    <tr key={pair}>
                      <td>{pair}</td>
                      <td>{(row.probability * 100).toFixed(1)}%</td>
                      <td style={{ color: row.position_modulated >= 0 ? "#059669" : "#dc2626" }}>
                        {row.position_modulated >= 0 ? "long " : "short "}
                        {Math.abs(row.position_modulated).toFixed(3)}
                      </td>
                      <td>{row.position_baseline.toFixed(3)}</td>
                    </tr>
                  );
                })}
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
              Pure postprocessing - never part of training or evaluation itself. Everything below is recomputed live
              from this model's raw predicted probabilities as you change it, with no new request. Evaluated with{" "}
              {result.neutral_band.toFixed(2)}.
            </p>
          </div>

          <h2>Hit rate</h2>
          <div className="chart-grid">
            <HitRateChart pairs={result.pairs} hitRate={hitRate} />
          </div>

          <h2>Confusion matrix</h2>
          <ConfusionMatrixTable pairs={result.pairs} confusionMatrix={confusionMatrix} />

          <h2>Cumulative returns (colored by prediction hit/miss)</h2>
          <div className="chart-grid">
            {result.pairs.map((pair) => {
              const { probability, label } = result.probabilities[pair];
              const { hit, abstained } = hitAbstainedSeries(probability, label, displayBand);
              return (
                <ColoredReturnChart
                  key={pair}
                  title={pair}
                  dates={result.cumulative_returns.dates}
                  series={{ cumulative: result.cumulative_returns[pair].cumulative, hit, abstained }}
                />
              );
            })}
          </div>

          <h2>Portfolio PnL (risk parity, probability-modulated)</h2>
          <AnnualSharpeTable annualSharpe={result.annual_sharpe} />
          <PortfolioPnlChart
            dates={result.portfolio.dates}
            pairs={result.pairs}
            cumulativeModulated={result.portfolio.cumulative_pnl_modulated}
            cumulativeBaseline={result.portfolio.cumulative_pnl_baseline}
            perAssetCumulativePnl={Object.fromEntries(
              result.pairs.map((pair) => [pair, result.portfolio[pair].cumulative_pnl])
            )}
          />
          <div className="chart-grid">
            {result.pairs.map((pair) => (
              <PortfolioPositionChart
                key={pair}
                title={pair}
                dates={result.portfolio.dates}
                series={result.portfolio[pair]}
              />
            ))}
          </div>

          <h2>Predicted probability</h2>
          <div className="chart-grid">
            {result.pairs.map((pair) => (
              <ProbabilityChart
                key={pair}
                title={pair}
                dates={result.probabilities.dates}
                series={result.probabilities[pair]}
                band={displayBand}
              />
            ))}
          </div>

          <h2>Forecasted vs actual return distribution</h2>
          <div className="chart-grid">
            {result.pairs.map((pair) => (
              <ReturnDistributionChart key={pair} title={pair} series={result.distribution[pair]} />
            ))}
          </div>
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
