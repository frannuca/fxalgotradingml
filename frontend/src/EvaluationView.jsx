import { useEffect, useState } from "react";
import { getModels, refreshQuotes, evaluate } from "./api";
import PnlChart from "./charts/PnlChart";
import HistogramChart from "./charts/HistogramChart";
import SeriesByPairChart from "./charts/SeriesByPairChart";
import { SERIES_COLOR } from "./theme";

const PORTFOLIO_TYPES = new Set(["portfolio", "portfolio_ensemble"]);
const RISK_TYPES = new Set(["risk", "risk_ensemble"]);

export default function EvaluationView() {
  const [models, setModels] = useState([]);
  const [pairs, setPairs] = useState([]);
  const [lookback, setLookback] = useState(null);
  const [portfolioModel, setPortfolioModel] = useState("");
  const [riskModel, setRiskModel] = useState("");
  const [params, setParams] = useState({
    years: 8, train_frac: 0.8, target_vol: 0.2, transaction_cost: 5,
  });
  const [refreshYears, setRefreshYears] = useState(1);
  const [refreshStatus, setRefreshStatus] = useState(null);
  const [evalStatus, setEvalStatus] = useState(null);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  useEffect(() => {
    reloadModels();
  }, []);

  function reloadModels() {
    getModels().then(setModels).catch((e) => setError(e.message));
  }

  function selectPortfolioModel(name) {
    setPortfolioModel(name);
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
    if (!portfolioModel) {
      setError("Select a portfolio model.");
      return;
    }
    setEvalStatus("running");
    try {
      // No `pairs`/`lookback` sent - the backend recovers both from the
      // selected model's own checkpoint (see api/server.py's evaluate()).
      const res = await evaluate({
        ...params,
        portfolio_model: portfolioModel,
        risk_model: riskModel || null,
      });
      setResult(res);
      setEvalStatus("done");
    } catch (err) {
      setEvalStatus(null);
      setError(err.message);
    }
  }

  const portfolioModels = models.filter((m) => PORTFOLIO_TYPES.has(m.model_type));
  const riskModels = models.filter((m) => RISK_TYPES.has(m.model_type));
  const hasRisk = Boolean(result && result.pnl.with_risk);
  const minYears = lookback ? Math.ceil(lookback / 252 + 0.2) : null; // rough padding above the raw day count

  return (
    <div>
      <form onSubmit={runEvaluation}>
        <div className="panel">
          <h2 style={{ marginTop: 0 }}>1. Select model(s)</h2>
          <div className="model-select-row">
            <div className="field">
              <label>Portfolio model (required)</label>
              <select value={portfolioModel} onChange={(e) => selectPortfolioModel(e.target.value)}>
                <option value="">— choose —</option>
                {portfolioModels.map((m) => (
                  <option key={m.name} value={m.name}>{m.name}</option>
                ))}
              </select>
            </div>
            <div className="field">
              <label>Risk overlay model (optional)</label>
              <select value={riskModel} onChange={(e) => setRiskModel(e.target.value)}>
                <option value="">— none —</option>
                {riskModels.map((m) => (
                  <option key={m.name} value={m.name}>{m.name}</option>
                ))}
              </select>
            </div>
            <button type="button" className="secondary" onClick={reloadModels} style={{ alignSelf: "end" }}>
              Refresh model list
            </button>
          </div>

          {portfolioModel && (
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
              : "How much historical data to use, and reporting-only settings - the model's own pairs/lookback aren't editable here."}
          </p>
          <div className="form-grid">
            <NumField label="Years of history" name="years" value={params.years} onChange={updateParam} />
            <NumField label="Train fraction" name="train_frac" step="0.05" value={params.train_frac} onChange={updateParam} />
            <NumField label="Target volatility" name="target_vol" step="0.01" value={params.target_vol} onChange={updateParam} />
            <NumField label="Transaction cost (bps)" name="transaction_cost" step="0.5" value={params.transaction_cost} onChange={updateParam} />
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

          <h2>Recommended weights (next day)</h2>
          <div className="panel">
            <table className="weights-table">
              <thead>
                <tr><th>Pair</th><th>Weight</th></tr>
              </thead>
              <tbody>
                {Object.entries(result.latest_weights).map(([pair, weight]) => (
                  <tr key={pair}>
                    <td>{pair}</td>
                    <td>{weight.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h2>Cumulative PnL (out-of-sample)</h2>
          <div className="chart-grid">
            <PnlChart
              title={
                hasRisk
                  ? `Risk-weighted baseline vs. with risk overlay vs. with risk overlay + transaction costs ` +
                    `(Sharpe ${result.sharpe.baseline.toFixed(2)} / ${result.sharpe.with_risk.toFixed(2)} / ${result.sharpe.with_risk_and_costs.toFixed(2)})`
                  : `Risk-weighted portfolio - baseline (Sharpe ${result.sharpe.baseline.toFixed(2)})`
              }
              pnl={result.pnl}
              seriesKeys={hasRisk ? ["baseline", "with_risk", "with_risk_and_costs"] : ["baseline"]}
              height={380}
            />
          </div>

          <h2>Return distribution (out-of-sample)</h2>
          <div className="chart-grid">
            <HistogramChart
              title="Risk-weighted (baseline) daily returns"
              histogram={result.histograms.baseline}
              color={SERIES_COLOR.baseline}
            />
            {hasRisk && (
              <HistogramChart
                title="With risk overlay daily returns"
                histogram={result.histograms.with_risk}
                color={SERIES_COLOR.with_risk}
              />
            )}
          </div>

          <h2>Cumulative FX pair returns (out-of-sample)</h2>
          <div className="chart-grid">
            <SeriesByPairChart title="Out-of-sample" series={result.asset_returns} pairs={result.pairs} height={340} />
          </div>

          <h2>Portfolio positions (out-of-sample)</h2>
          <div className="chart-grid">
            <SeriesByPairChart title="Out-of-sample" series={result.positions} pairs={result.pairs} />
          </div>

          {result.attenuation && (
            <>
              <h2>Risk attenuation factor (out-of-sample)</h2>
              <div className="chart-grid">
                <SeriesByPairChart title="Out-of-sample" series={result.attenuation} pairs={result.pairs} />
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
