// Client-side mirror of models/portfolio_lstm.py's apply_neutral_band /
// confusion_matrix_counts / confusion_matrix_metrics / _decided_hit_rate.
//
// The neutral band is pure POSTPROCESSING - never part of training (see
// evaluate_prediction_model's docstring: train_prediction_model never
// receives it) - so the API ships RAW probabilities plus the realized
// direction label per date (see api/server.py's _probability_payload)
// instead of a value already snapped to one band, and everything here
// recomputes hit rate / confusion matrix / colored-return coloring for
// whatever band the user currently has selected, entirely in the browser:
// no retraining, no new request, just re-deriving from data already on
// the page.

const EPS = 1e-8;

// Mirrors apply_neutral_band: snaps a probability inside 0.5 +/- band to
// exactly 0.5 ("no call"). band <= 0 disables snapping.
export function applyNeutralBand(probability, band) {
  if (band <= 0) return probability;
  return Math.abs(probability - 0.5) < band ? 0.5 : probability;
}

// Mirrors confusion_matrix_metrics for ONE asset: `probabilities`/`labels`
// are parallel arrays (raw probability, realized 0/1 label). Returns the
// same shape ConfusionMatrixTable already expects from the backend.
export function confusionMatrixMetrics(probabilities, labels, band) {
  let tp = 0, fp = 0, tn = 0, fn = 0, abstained = 0;
  for (let i = 0; i < probabilities.length; i++) {
    const p = probabilities[i];
    const actualPositive = labels[i] > 0.5;
    const predictedPositive = p > 0.5 + band;
    const predictedNegative = p < 0.5 - band;
    if (predictedPositive) {
      actualPositive ? tp++ : fp++;
    } else if (predictedNegative) {
      actualPositive ? fn++ : tn++;
    } else {
      abstained++;
    }
  }
  const nDecided = tp + fp + tn + fn;
  const precision = tp / (tp + fp + EPS);
  const recall = tp / (tp + fn + EPS);
  return {
    tp, fp, tn, fn, abstained,
    coverage: nDecided / (nDecided + abstained + EPS),
    accuracy: (tp + tn) / (nDecided + EPS),
    precision,
    recall,
    specificity: tn / (tn + fp + EPS),
    f1: (2 * precision * recall) / (precision + recall + EPS),
  };
}

// Mirrors _decided_hit_rate: fraction of DECIDED samples whose call
// matched the label - 0.5 (chance) if nothing was decided, matching the
// Python fallback exactly (NOT the same as confusionMatrixMetrics's own
// accuracy field, which would report 0 in that edge case instead).
export function decidedHitRate(probabilities, labels, band) {
  let correct = 0, decided = 0;
  for (let i = 0; i < probabilities.length; i++) {
    const p = probabilities[i];
    const actualPositive = labels[i] > 0.5;
    const predictedPositive = p > 0.5 + band;
    const predictedNegative = p < 0.5 - band;
    if (predictedPositive || predictedNegative) {
      decided++;
      if (predictedPositive === actualPositive) correct++;
    }
  }
  return decided > 0 ? correct / decided : 0.5;
}

// Per-day hit/abstained booleans for ColoredReturnChart - mirrors the
// hit/abstained logic the backend used to compute server-side (now
// computed here instead, per the currently-selected band).
export function hitAbstainedSeries(probabilities, labels, band) {
  const hit = [];
  const abstained = [];
  for (let i = 0; i < probabilities.length; i++) {
    const p = probabilities[i];
    hit.push((p > 0.5) === (labels[i] > 0.5));
    abstained.push(Math.abs(p - 0.5) < band);
  }
  return { hit, abstained };
}

// Build the {<pair>: {tp,fp,...}} shape ConfusionMatrixTable expects,
// from api/server.py's _probability_payload shape ({<pair>: {probability,
// label}}), for one split.
export function confusionMatrixForSplit(probabilitiesSplit, pairs, band) {
  const out = {};
  for (const pair of pairs) {
    const { probability, label } = probabilitiesSplit[pair];
    out[pair] = confusionMatrixMetrics(probability, label, band);
  }
  return out;
}

// Build the {<pair>: rate} shape HitRateChart expects, for one split.
export function hitRateForSplit(probabilitiesSplit, pairs, band) {
  const out = {};
  for (const pair of pairs) {
    const { probability, label } = probabilitiesSplit[pair];
    out[pair] = decidedHitRate(probability, label, band);
  }
  return out;
}
