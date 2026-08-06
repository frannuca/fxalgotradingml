import { useEffect, useState } from "react";
import { getModels, refreshQuotes, evaluate, recomputePortfolio } from "./api";
import HitRateChart from "./charts/HitRateChart";
import ColoredReturnChart from "./charts/ColoredReturnChart";
import ProbabilityChart from "./charts/ProbabilityChart";
import ReturnDistributionChart from "./charts/ReturnDistributionChart";
import PortfolioPositionChart from "./charts/PortfolioPositionChart";
import PortfolioPnlChart from "./charts/PortfolioPnlChart";
import PortfolioAssetPnlChart from "./charts/PortfolioAssetPnlChart";
import ConfusionMatrixTable from "./ConfusionMatrixTable";
import AnnualSharpeTable from "./AnnualSharpeTable";
import { confusionMatrixForSplit, findPlotStartIndex, hitAbstainedSeries, hitRateForSplit, rebaseCumulativeSeries, sliceArraysFrom } from "./metrics";

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
  // Unlike hit rate/confusion matrix (recomputed purely client-side from
  // raw probabilities - see ./metrics.js), the portfolio PnL/annual
  // Sharpe/today's position need the full risk-parity machinery
  // (models/portfolio_pnl.py) - so changing this band triggers a debounced
  // POST /api/evaluate/{eval_id}/portfolio call (see the effect below)
  // instead, cheap since it reuses the already-fetched data/model output
  // cached server-side rather than re-running either.
  const [displayBand, setDisplayBand] = useState(0.05);
  const [portfolioLoading, setPortfolioLoading] = useState(false);
  // Optional DISPLAY-only crop: "years" above still controls how much
  // history is FETCHED and fed through feature-building/covariance
  // warm-up (see api/server.py's evaluate()) - this only hides days
  // before plotStartDate from the charts/tables below, entirely in the
  // browser, no new request. "" (default) shows the whole fetched period,
  // unchanged from before this existed.
  const [plotStartDate, setPlotStartDate] = useState("");

  useEffect(() => {
    reloadModels();
  }, []);

  // `evalId` (a plain string, not the whole `result` object) is the
  // dependency here on purpose - re-firing only on a genuinely NEW
  // evaluation or a band edit, never on result's own object identity
  // changing for unrelated reasons. Every band edit (including right after
  // a fresh "Run model") schedules a recompute - a redundant call the very
  // first time is harmless (same band the initial evaluation already used,
  // server responds instantly from _EVAL_CACHE) and buys certainty that a
  // dropped edit can never silently leave the portfolio panel stale.
  const evalId = result ? result.eval_id : null;
  useEffect(() => {
    if (!evalId) return;
    const timer = setTimeout(async () => {
      setPortfolioLoading(true);
      try {
        const res = await recomputePortfolio(evalId, displayBand);
        setResult((r) => (r && r.eval_id === evalId
          ? { ...r, portfolio: res.portfolio, annual_sharpe: res.annual_sharpe, latest_position: res.latest_position }
          : r));
      } catch (err) {
        setError(err.message);
      } finally {
        setPortfolioLoading(false);
      }
    }, 400);
    return () => clearTimeout(timer);
  }, [displayBand, evalId]);

  const selectedModel = models.find((m) => m.name === modelName) || null;

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

  // Crop every date-aligned series down to plotStartDate (see
  // ./metrics.js's sliceArraysFrom/findPlotStartIndex) - "" (no crop) is a
  // no-op (startIndex 0), so every chart/table below is unaffected until
  // the user actually sets a start date. `totalDays` is the UNCROPPED
  // length shared by every one of these series (they're all built from
  // the same `dates_train` server-side - see api/server.py's evaluate()),
  // used to tell a date-aligned array apart from an incidental one.
  const totalDays = result ? result.cumulative_returns.dates.length : 0;
  const plotStartIndex = result ? findPlotStartIndex(result.cumulative_returns.dates, plotStartDate) : 0;
  const croppedCumulativeReturns = result && sliceArraysFrom(result.cumulative_returns, plotStartIndex, totalDays);
  const croppedPortfolio = result && sliceArraysFrom(result.portfolio, plotStartIndex, totalDays);
  const croppedProbabilities = result && sliceArraysFrom(result.probabilities, plotStartIndex, totalDays);
  const croppedDistribution = result && sliceArraysFrom(result.distribution, plotStartIndex, totalDays);
  // With an explicit plotStartDate, every CUMULATIVE series above (per-asset
  // cumulative return, portfolio cumulative PnL - see rebaseCumulativeSeries'
  // own docstring) is rebased to 0 at its own first visible point, so the
  // chart reads as "return SINCE plotStartDate" rather than still carrying
  // forward everything accumulated earlier in the fetched history. Left
  // alone (identity) when no start date is set - unchanged, whole-period
  // behavior from before this existed.
  const displayCumulativeReturns = plotStartDate ? rebaseCumulativeSeries(croppedCumulativeReturns) : croppedCumulativeReturns;
  const displayPortfolio = plotStartDate ? rebaseCumulativeSeries(croppedPortfolio) : croppedPortfolio;
  // annual_sharpe is pre-aggregated PER CALENDAR YEAR server-side (no
  // per-day pnl is shipped to recompute this client-side) - coarser than
  // the other crops: drops whole years entirely before plotStartDate's own
  // year, but a year plotStartDate falls IN THE MIDDLE of still reports
  // its Sharpe over its FULL year, not just the visible tail (see
  // AnnualSharpeTable's own display, which flags this).
  const croppedAnnualSharpe = result && (
    plotStartDate
      ? Object.fromEntries(Object.entries(result.annual_sharpe).filter(([year]) => year >= plotStartDate.slice(0, 4)))
      : result.annual_sharpe
  );

  // Recomputed from raw probability + realized label (see ./metrics.js)
  // every time displayBand OR plotStartDate changes - the band never
  // touches evaluation itself, so this is the ONLY place it's actually
  // applied to this result. Evaluation is a single continuous period now
  // (no train/val/test split - see api/server.py's evaluate()), so these
  // cover the whole (cropped) thing directly.
  const hitRate = result && hitRateForSplit(croppedProbabilities, result.pairs, displayBand);
  const confusionMatrix = result && confusionMatrixForSplit(croppedProbabilities, result.pairs, displayBand);

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
                  <option key={m.name} value={m.name}>
                    {m.name}{m.risk_engine ? " (has risk engine)" : ""}
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
              <strong>Recovered from the model itself (not editable here) - see api/server.py's evaluate()/load_pipeline:</strong>
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
                  <tr>
                    <td>Risk engine</td>
                    <td>
                      {selectedModel.risk_engine
                        ? `attached - lookback ${selectedModel.risk_engine.risk_lookback} decision days, attenuation bounds (${selectedModel.risk_engine.min_risk_att}, ${selectedModel.risk_engine.max_risk_att})`
                        : "none - train one in the Risk Engine view, then re-select this model's saved output here"}
                    </td>
                  </tr>
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
            <p className="status-line" style={{ marginTop: 0 }}>
              As of <strong>{result.latest_as_of_date}</strong> - the most recent close the model actually has data
              for (its lookback window's last day), not necessarily today's calendar date (a weekend, a holiday, or
              stale quotes - see "Refresh quotes into DB" above - can push this back). Forecasts the{" "}
              {result.direction_horizon}-trading-day move from that date forward.
            </p>
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
              As of <strong>{result.latest_as_of_date}</strong> (see "Predicted probabilities" above) - risk-parity
              weight times the probability signal <code>(p - 0.5) * 2</code>, smoothed over this model&apos;s own
              direction horizon and scaled to its target volatility of {(result.target_vol * 100).toFixed(1)}% annualized
              (a model property, recovered from its checkpoint) - see the Portfolio PnL section below for the full
              backtest this is based on.
            </p>
            <table className="weights-table">
              <thead>
                <tr>
                  <th>Pair</th><th>P(positive)</th><th>Modulated position</th>
                  {result.has_risk_engine && <th>Risk-attenuated</th>}
                  <th>Risk-parity baseline</th>
                </tr>
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
                      {result.has_risk_engine && (
                        <td style={{ color: row.position_risk_attenuated >= 0 ? "#059669" : "#dc2626" }}>
                          {row.position_risk_attenuated >= 0 ? "long " : "short "}
                          {Math.abs(row.position_risk_attenuated).toFixed(3)}
                        </td>
                      )}
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
              Pure postprocessing - never part of training or evaluation itself. Hit rate/confusion matrix/colored
              return charts below are recomputed live from this model's raw predicted probabilities as you change
              it, with no new request. Portfolio PnL/annual Sharpe/today's position (further below) need a
              server-side recompute - a probability inside the band is treated as no view, riding the unmodulated
              risk-parity weight for that pair instead of a near-coin-flip directional bet - fired automatically,
              shortly after you stop typing.{portfolioLoading ? " Recomputing portfolio…" : ""} Evaluated with{" "}
              {result.neutral_band.toFixed(2)}.
            </p>
          </div>

          <div className="panel">
            <h2 style={{ marginTop: 0 }}>Plot start date</h2>
            <div className="form-grid">
              <div className="field">
                <label>Show charts/tables from (blank = full fetched history)</label>
                <input
                  type="date"
                  value={plotStartDate}
                  min={result.cumulative_returns.dates[0]}
                  max={result.cumulative_returns.dates[result.cumulative_returns.dates.length - 1]}
                  onChange={(e) => setPlotStartDate(e.target.value)}
                />
              </div>
              {plotStartDate && (
                <button type="button" className="secondary" onClick={() => setPlotStartDate("")} style={{ alignSelf: "end" }}>
                  Clear
                </button>
              )}
            </div>
            <p className="status-line" style={{ marginTop: 4 }}>
              Pure DISPLAY crop, entirely client-side - "Years of history" above still controls how much data is
              FETCHED and fed through feature/covariance warm-up, so setting this doesn't lose any of that context;
              it only hides days before this date from the charts and tables below (hit rate, confusion matrix,
              cumulative returns, portfolio PnL/positions, predicted probability, and the return distribution). The
              annual Sharpe table below is the one exception - it's pre-aggregated PER CALENDAR YEAR server-side, so
              a year this date falls in the MIDDLE of still reports that whole year's Sharpe, not just the visible
              tail.
              {plotStartIndex >= totalDays && plotStartDate && (
                <span style={{ color: "#b45309" }}> No data on or after {plotStartDate} - charts below will be empty.</span>
              )}
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
              const { probability, label } = croppedProbabilities[pair];
              const { hit, abstained } = hitAbstainedSeries(probability, label, displayBand);
              return (
                <ColoredReturnChart
                  key={pair}
                  title={pair}
                  dates={displayCumulativeReturns.dates}
                  series={{ cumulative: displayCumulativeReturns[pair].cumulative, hit, abstained }}
                />
              );
            })}
          </div>

          <h2>Portfolio PnL (risk parity, probability-modulated)</h2>
          <AnnualSharpeTable annualSharpe={croppedAnnualSharpe} />
          <PortfolioPnlChart
            dates={displayPortfolio.dates}
            cumulativeModulated={displayPortfolio.cumulative_pnl_modulated}
            cumulativeBaseline={displayPortfolio.cumulative_pnl_baseline}
            cumulativeRiskAttenuated={displayPortfolio.cumulative_pnl_risk_attenuated ?? null}
          />
          <PortfolioAssetPnlChart
            dates={displayPortfolio.dates}
            pairs={result.pairs}
            perAssetCumulativePnl={Object.fromEntries(
              result.pairs.map((pair) => [pair, displayPortfolio[pair].cumulative_pnl])
            )}
          />
          <div className="chart-grid">
            {result.pairs.map((pair) => (
              <PortfolioPositionChart
                key={pair}
                title={pair}
                dates={displayPortfolio.dates}
                series={displayPortfolio[pair]}
              />
            ))}
          </div>

          <h2>Predicted probability</h2>
          <div className="chart-grid">
            {result.pairs.map((pair) => (
              <ProbabilityChart
                key={pair}
                title={pair}
                dates={croppedProbabilities.dates}
                series={croppedProbabilities[pair]}
                band={displayBand}
              />
            ))}
          </div>

          <h2>Forecasted vs actual return distribution</h2>
          <div className="chart-grid">
            {result.pairs.map((pair) => (
              <ReturnDistributionChart key={pair} title={pair} series={croppedDistribution[pair]} />
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
