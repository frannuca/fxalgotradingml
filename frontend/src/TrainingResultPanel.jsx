import HitRateChart from "./charts/HitRateChart";
import ColoredReturnChart from "./charts/ColoredReturnChart";
import ProbabilityChart from "./charts/ProbabilityChart";
import ReturnDistributionChart from "./charts/ReturnDistributionChart";
import ConfusionMatrixTable from "./ConfusionMatrixTable";
import AnnualSharpeTable from "./AnnualSharpeTable";
import { SPLIT_LABEL } from "./theme";
import { hitAbstainedSeries } from "./metrics";

const KFOLD_METRIC_LABEL = {
  val_loss: "Validation loss (NLL + BCE)",
  hit_rate: "Validation hit rate",
  sharpe: "Validation Sharpe (annualized)",
};

// result.kfold's scores are HIGHER-IS-BETTER always (see
// api/server.py's job result payload - val_loss is stored NEGATED so a
// single max() picks the winner regardless of metric) - flip val_loss back
// to natural "loss, lower is better" units for display, matching every
// other loss figure in this app; hit_rate/sharpe are already natural.
function displayKfoldScore(rawScore, checkpointMetric) {
  const natural = checkpointMetric === "val_loss" ? -rawScore : rawScore;
  if (checkpointMetric === "hit_rate") return `${(natural * 100).toFixed(1)}%`;
  if (checkpointMetric === "sharpe") return natural.toFixed(3);
  return natural.toFixed(4);
}

// Spread (std) is sign-invariant - never negate this one, even for
// val_loss (std(-X) == std(X)), just scale it the same way per metric.
function displayKfoldSpread(stdScore, checkpointMetric) {
  if (checkpointMetric === "hit_rate") return `${(stdScore * 100).toFixed(1)}%`;
  if (checkpointMetric === "sharpe") return stdScore.toFixed(3);
  return stdScore.toFixed(4);
}

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

  // An ensemble job (see result.kfold above) has no single train/val
  // split of its own - "train" is sent back empty (see api/server.py's
  // own ensemble branch) rather than meaningful, so it's skipped here
  // entirely; "val" carries the full leakage-aware historical curve
  // instead of an ordinary validation split, relabeled below to say so.
  const splitsToShow = result.kfold ? ["val", "test"] : ["train", "val", "test"];
  const splitLabel = (split) => (result.kfold && split === "val" ? "Historical (leakage-aware)" : SPLIT_LABEL[split]);
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

      {result.kfold && (
        <div className="panel">
          <h2 style={{ marginTop: 0 }}>Cross-validation (K-fold) results</h2>
          <p className="status-line" style={{ marginTop: 0 }}>
            Trained {result.kfold.n_folds} independent models on purged, embargoed folds (a {result.kfold.purge_gap}-decision-day
            buffer dropped on each side of every validation block, so no training sample's label or lookback window
            overlaps what it's validated against - see models/portfolio_lstm.py's generate_purged_kfold_splits).
            EVERY fold's own model{result.kfold.has_risk_engine ? " (and its own risk engine)" : ""} is kept as one
            member of a saved ENSEMBLE, not just whichever validated best - picking a single "best" fold would
            reintroduce the same luck-driven selection problem K-fold exists to avoid, one level up. At evaluation
            time every member runs its own complete forward pass and their outputs are averaged. "Validation"
            below is this job's own leakage-aware historical curve (every historical day averaged only across
            members that never trained on it - see _historical_ensemble_curve), and "Test" the shared, never-
            trained-on split, averaged across all {result.kfold.n_members} members. Read the per-fold scores' mean
            and spread together: a high mean with a wide spread means this CONFIGURATION got lucky on some folds
            and unlucky on others, not a robustly good one - the single-split number this replaces could never
            have shown you that.
          </p>
          <table className="weights-table">
            <thead>
              <tr><th>Fold</th><th>{KFOLD_METRIC_LABEL[result.kfold.checkpoint_metric] || result.kfold.checkpoint_metric}</th></tr>
            </thead>
            <tbody>
              {result.kfold.fold_scores.map((score, i) => (
                <tr key={i}>
                  <td>{`Fold ${i + 1}`}</td>
                  <td>{displayKfoldScore(score, result.kfold.checkpoint_metric)}</td>
                </tr>
              ))}
              <tr>
                <td><strong>Mean ± std across folds</strong></td>
                <td>
                  <strong>
                    {displayKfoldScore(result.kfold.mean_score, result.kfold.checkpoint_metric)}
                    {" ± "}
                    {displayKfoldSpread(result.kfold.std_score, result.kfold.checkpoint_metric)}
                  </strong>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      )}

      {result.ensemble && (
        <div className="panel">
          <h2 style={{ marginTop: 0 }}>Continued ensemble</h2>
          <p className="status-line" style={{ marginTop: 0 }}>
            This continued EVERY one of the base model's {result.ensemble.n_members} members independently (same
            extended data window and settings for all), then saved a new {result.ensemble.n_members}-member ensemble -
            not a single model. {result.ensemble.has_risk_engine
              ? "Each member also got a freshly-trained risk engine (any it had before was never carried forward - it was trained against that member's OLD weights)."
              : "No risk engines were trained for this run."} Unlike a fresh K-fold cross-validation run, this DID
            NOT re-run cross-validation itself - it just extended each existing member's own training - so there's
            no new per-fold score distribution to report here; everything below (train/val/test) is the ensemble's
            own averaged result, exactly like any other trained model.
          </p>
        </div>
      )}

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
      {splitsToShow.map((split) => (
        <div key={split}>
          <h3>{splitLabel(split)}</h3>
          <AnnualSharpeTable annualSharpe={result.annual_sharpe[split]} />
        </div>
      ))}

      <h2>Cumulative returns (colored by prediction hit/miss)</h2>
      {splitsToShow.map((split) => (
        <div key={split}>
          <h3>{splitLabel(split)}</h3>
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
      {splitsToShow.map((split) => (
        <div key={split}>
          <h3>{splitLabel(split)}</h3>
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
      {splitsToShow.map((split) => (
        <div key={split}>
          <h3>{splitLabel(split)}</h3>
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
