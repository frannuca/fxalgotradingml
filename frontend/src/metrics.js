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

// First index in `dates` (ISO "YYYY-MM-DD" strings, so plain string
// comparison sorts correctly) whose date is >= startDate. `dates.length`
// (i.e. "nothing visible") if every date is BEFORE startDate - the plot
// start date is later than all available history - rather than -1, so a
// caller can always slice(idx) safely without a wrap-around bug.
// startDate falsy (not set) means "no crop": returns 0.
export function findPlotStartIndex(dates, startDate) {
  if (!startDate) return 0;
  const idx = dates.findIndex((d) => d >= startDate);
  return idx === -1 ? dates.length : idx;
}

// Recursively slices every array found in `payload` (including nested
// arrays under per-pair keys, e.g. api/server.py's _cumulative_return_payload/
// _portfolio_payload/_probability_payload shapes) from `startIndex` onward -
// crops a date-aligned evaluation payload down to a chosen display window
// for the CHARTS/TABLES only, without re-fetching or recomputing anything
// server-side (the backend still fetches/computes over the model's full
// requested history - see EvaluationView.jsx's own "years" field - this is
// purely what gets DRAWN). Only arrays whose length equals `total` (the
// UNCROPPED series length, e.g. `dates.length`) are treated as
// date-aligned and sliced; anything else (numbers, strings, a shorter
// array that isn't one of these per-day series) passes through untouched -
// this is what lets the SAME function crop `dates` itself, every per-pair
// sub-object, AND api/server.py's `distribution` payload (which has no
// `dates` field of its own, but its `actual`/`forecasted` arrays are the
// SAME length as every other split's `dates`, so passing that length in
// as `total` crops it consistently too).
export function sliceArraysFrom(payload, startIndex, total) {
  if (Array.isArray(payload)) {
    return payload.length === total ? payload.slice(startIndex) : payload;
  }
  if (payload && typeof payload === "object") {
    const out = {};
    for (const [key, value] of Object.entries(payload)) {
      out[key] = sliceArraysFrom(value, startIndex, total);
    }
    return out;
  }
  return payload;
}

// Field names that are genuinely CUMULATIVE series (running sums from the
// evaluation's own day 0 - see api/server.py's _cumulative_return_payload/
// _portfolio_payload) as opposed to a per-day snapshot (positions,
// probabilities, distribution samples) that merely happens to be nested in
// the same payload shape.
const CUMULATIVE_KEYS = new Set([
  "cumulative", "cumulative_pnl",
  "cumulative_pnl_modulated", "cumulative_pnl_baseline", "cumulative_pnl_risk_attenuated",
]);

// Subtracts each cumulative array's own FIRST value from every point in it,
// so a series already cropped to start partway through (see
// sliceArraysFrom/findPlotStartIndex) reads as "cumulative return SINCE the
// crop start" - 0 at the first visible day - instead of carrying forward
// whatever it had already accumulated before the crop (e.g. a chart cropped
// to the last year of a 3-year evaluation would otherwise still open at 3
// years' worth of accumulated return). Only touches keys in CUMULATIVE_KEYS -
// positions/probabilities/distribution samples are structurally NOT
// cumulative and must pass through untouched - recurses through the same
// nested {<pair>: {...}} shape sliceArraysFrom does. Call this on an
// ALREADY-CROPPED payload (its own first element becomes the new zero
// point), never on the full, uncropped one.
export function rebaseCumulativeSeries(payload) {
  if (Array.isArray(payload)) return payload;
  if (payload && typeof payload === "object") {
    const out = {};
    for (const [key, value] of Object.entries(payload)) {
      if (CUMULATIVE_KEYS.has(key) && Array.isArray(value) && value.length > 0) {
        const base = value[0];
        out[key] = value.map((v) => v - base);
      } else {
        out[key] = rebaseCumulativeSeries(value);
      }
    }
    return out;
  }
  return payload;
}
