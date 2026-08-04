import HitRateChart from "./charts/HitRateChart";
import ColoredReturnChart from "./charts/ColoredReturnChart";
import ProbabilityChart from "./charts/ProbabilityChart";
import ReturnDistributionChart from "./charts/ReturnDistributionChart";
import ConfusionMatrixTable from "./ConfusionMatrixTable";
import AnnualSharpeTable from "./AnnualSharpeTable";
import { SPLIT_LABEL } from "./theme";
import { hitAbstainedSeries } from "./metrics";

// Shared by TrainingView and ContinueTrainingView - renders a finished
// training job's full result (see api/server.py's _run_training_job,
// whose response shape is identical whether the run was a fresh
// PredictionModel or a continue_training warm start - see
// models/portfolio_lstm.py's continue_training). `hitRate`/
// `confusionMatrix` are recomputed client-side from raw probabilities as
// `displayBand` changes (see ../metrics.js) - pure postprocessing, never
// part of training itself.
export default function TrainingResultPanel({ result, hitRate, confusionMatrix, displayBand, setDisplayBand }) {
  if (!result) return null;
  return (
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
  );
}
