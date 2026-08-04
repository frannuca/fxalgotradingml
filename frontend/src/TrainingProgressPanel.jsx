import { useState } from "react";
import { saveBestCheckpoint } from "./api";

// Shared by TrainingView and ContinueTrainingView - the live progress
// bar + interim loss/hit-rate/Sharpe table + "save best model so far"
// action + scrolling log window, driven by GET /api/train/{job_id}
// polling (see api/server.py's _run_training_job/_JobLogHandler).
// Identical for both a fresh training run and a continue-training run -
// the underlying job/interim/log shape is the same either way (see
// models/portfolio_lstm.py's continue_training, which returns the exact
// same PredictionResult shape run_pipeline_multi_seed does).

// Best-so-far annotation appended to a Validation cell (see
// models/portfolio_lstm.py's train_prediction_model docstring on
// checkpoint_metric) - only shown on the ONE row that's actually driving
// checkpoint selection this run, so it's clear which quantity "best"
// refers to instead of a generic, metric-agnostic label.
function bestSuffix(interim, metric, formatValue) {
  if (interim.best_score == null || interim.checkpoint_metric !== metric) return null;
  return ` (best so far: ${formatValue(interim.best_score)})`;
}

const SPLIT_COLUMNS = [["train", "Train"], ["val", "Validation"], ["test", "Test"]];

export default function TrainingProgressPanel({ progress, interim, logs, logBoxRef, jobId }) {
  const [bestSummary, setBestSummary] = useState(null); // { model_name, epoch, summary: {train/val/test: {loss, hit_rate, sharpe}}}
  const [savingBest, setSavingBest] = useState(false);
  const [saveBestError, setSaveBestError] = useState(null);

  if (!progress) return null;

  async function handleSaveBest() {
    setSavingBest(true);
    setSaveBestError(null);
    try {
      const res = await saveBestCheckpoint(jobId);
      setBestSummary(res);
    } catch (err) {
      setSaveBestError(err.message);
    } finally {
      setSavingBest(false);
    }
  }

  return (
    <div className="panel">
      <h3 style={{ marginBottom: 6 }}>
        Progress{progress.n_seeds > 1 ? ` — restart ${progress.seed_index}/${progress.n_seeds}` : ""}
        {progress.n_lambdas > 1 ? ` — bce_weight ${progress.lambda_index}/${progress.n_lambdas}` : ""}
        {" "}— epoch {progress.epoch}/{progress.total_epochs}
      </h3>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${progress.percent}%` }} />
      </div>
      <div className="progress-percent">{progress.percent.toFixed(1)}%</div>

      {interim && (
        <>
          <table className="weights-table" style={{ marginTop: 12 }}>
            <thead>
              <tr><th></th><th>Train</th>{interim.val_loss != null && <th>Validation</th>}</tr>
            </thead>
            <tbody>
              <tr>
                <td>Loss (NLL + BCE)</td>
                <td>{interim.train_loss.toFixed(4)}</td>
                {interim.val_loss != null && (
                  <td>
                    {interim.val_loss.toFixed(4)}
                    <strong>{bestSuffix(interim, "val_loss", (v) => v.toFixed(4))}</strong>
                  </td>
                )}
              </tr>
              <tr>
                <td>Decision-day hit rate</td>
                <td>{(interim.train_hit_rate * 100).toFixed(1)}%</td>
                {interim.val_hit_rate != null && (
                  <td>
                    {(interim.val_hit_rate * 100).toFixed(1)}%
                    <strong>{bestSuffix(interim, "hit_rate", (v) => `${(v * 100).toFixed(1)}%`)}</strong>
                  </td>
                )}
              </tr>
              {(interim.train_sharpe != null || interim.val_sharpe != null) && (
                <tr>
                  <td>Portfolio Sharpe (rolling, not annualized)</td>
                  {/* train_sharpe is null when checkpoint_metric="sharpe" alone triggered this row
                      (sharpe_weight=0 - Phase B never ran, so there's no TRAINING sharpe to show,
                      only the validation-side one checkpoint selection actually used). */}
                  <td>{interim.train_sharpe != null ? interim.train_sharpe.toFixed(4) : "—"}</td>
                  {interim.val_sharpe != null && (
                    <td>
                      {interim.val_sharpe.toFixed(4)}
                      <strong>{bestSuffix(interim, "sharpe", (v) => v.toFixed(4))}</strong>
                    </td>
                  )}
                </tr>
              )}
            </tbody>
          </table>
          {interim.best_score != null && (
            <p className="status-line" style={{ marginTop: 6 }}>
              "Best so far" is this run's checkpoint-selection score (averaged across assets; each asset's own
              best epoch is restored independently at the end) - shown on whichever row "Checkpoint selection
              metric" is currently set to.
            </p>
          )}
        </>
      )}

      <div style={{ marginTop: 16 }}>
        <button type="button" className="secondary" onClick={handleSaveBest} disabled={savingBest || !jobId}>
          {savingBest ? "Saving…" : "Save best model so far"}
        </button>
        <p className="status-line" style={{ marginTop: 4 }}>
          Saves whichever checkpoint is CURRENTLY best (per asset, by "Checkpoint selection metric" above) to the
          model registry right now, under its own new name - the run keeps training uninterrupted, and this never
          overwrites the base model or the run's own eventual final save.
        </p>
      </div>

      {saveBestError && <p className="status-line error">{saveBestError}</p>}

      {bestSummary && (
        <div style={{ marginTop: 8 }}>
          <div className="result-box">
            <strong>Saved best model so far.</strong>
            <table>
              <tbody>
                <tr><td>Model saved as</td><td><code>{bestSummary.model_name}</code></td></tr>
                <tr><td>As of epoch</td><td>{bestSummary.epoch}</td></tr>
              </tbody>
            </table>
          </div>
          <table className="weights-table" style={{ marginTop: 8 }}>
            <thead>
              <tr><th></th>{SPLIT_COLUMNS.map(([key, label]) => <th key={key}>{label}</th>)}</tr>
            </thead>
            <tbody>
              <tr>
                <td>Loss (NLL + BCE)</td>
                {SPLIT_COLUMNS.map(([key]) => (
                  <td key={key}>{bestSummary.summary[key].loss.toFixed(4)}</td>
                ))}
              </tr>
              <tr>
                <td>Decision-day hit rate</td>
                {SPLIT_COLUMNS.map(([key]) => (
                  <td key={key}>{(bestSummary.summary[key].hit_rate * 100).toFixed(1)}%</td>
                ))}
              </tr>
              <tr>
                <td>Portfolio Sharpe (annualized)</td>
                {SPLIT_COLUMNS.map(([key]) => (
                  <td key={key}>
                    {bestSummary.summary[key].sharpe != null ? bestSummary.summary[key].sharpe.toFixed(3) : "—"}
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
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
  );
}
