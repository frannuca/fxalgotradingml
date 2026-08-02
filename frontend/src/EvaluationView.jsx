import { useEffect, useState } from "react";
import { getModels, refreshQuotes, evaluate } from "./api";
import HitRateChart from "./charts/HitRateChart";
import ColoredReturnChart from "./charts/ColoredReturnChart";
import ProbabilityChart from "./charts/ProbabilityChart";
import ReturnDistributionChart from "./charts/ReturnDistributionChart";
import ConfusionMatrixTable from "./ConfusionMatrixTable";
import { SPLIT_LABEL } from "./theme";
import { confusionMatrixForSplit, hitAbstainedSeries, hitRateForSplit } from "./metrics";

export default function EvaluationView() {
  const [models, setModels] = useState([]);
  const [pairs, setPairs] = useState([]);
  const [lookback, setLookback] = useState(null);
  const [modelName, setModelName] = useState("");
  const [params, setParams] = useState({ years: 8, train_frac: 0.8, test_frac: 0.1 });
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
      const res = await evaluate({ ...params, model_name: modelName });
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
  // itself, so this is the ONLY place it's actually applied to this result.
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
              : "How much historical data to use, and how the train/val/test split is drawn - the model's own pairs/lookback aren't editable here."}
          </p>
          <div className="form-grid">
            <NumField label="Years of history" name="years" value={params.years} onChange={updateParam} />
            <NumField label="Train fraction" name="train_frac" step="0.05" value={params.train_frac} onChange={updateParam} />
            <NumField label="Test fraction" name="test_frac" step="0.05" value={params.test_frac} onChange={updateParam} />
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
