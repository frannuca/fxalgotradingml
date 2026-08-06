// Fixed color identity per series, used consistently across every chart in
// the app - color always follows the SAME entity (never re-assigned by
// rank/position), and legends/tooltips are never the only way to tell
// series apart (labels are always visible too).

export const GRID_COLOR = "#e2e8f0";
export const AXIS_COLOR = "#94a3b8";
export const TEXT_COLOR = "#334155";

// Fixed categorical palette for per-asset series whose keys are FX pair
// names chosen by the user, not known in advance. Assigned by POSITION in
// the pairs list, never by rank/value, so a given pair keeps the same
// color across every chart it appears in during a single run.
const PAIR_PALETTE = [
  "#2563eb", // blue
  "#d97706", // amber
  "#059669", // green
  "#dc2626", // red
  "#7c3aed", // violet
  "#0891b2", // cyan
  "#db2777", // pink
  "#65a30d", // lime
];

export function pairColor(pairs, pair) {
  const idx = pairs.indexOf(pair);
  return PAIR_PALETTE[idx % PAIR_PALETTE.length];
}

// The cumulative-return chart's per-day correctness coloring: whether the
// model's predicted direction matched the realized outcome that day.
export const HIT_COLOR = "#059669"; // green - predicted direction correct
export const MISS_COLOR = "#dc2626"; // red - predicted direction wrong
export const ABSTAIN_COLOR = "#94a3b8"; // slate - inside the neutral band, no call made (see apply_neutral_band)
export const RETURN_LINE_COLOR = "#94a3b8"; // neutral slate for the connecting equity-curve line itself

// Dedicated to the hit-rate bar chart's three bars per asset (train/val/
// test) - distinct from PAIR_PALETTE (different chart type/meaning, so no
// semantic clash), fixed order: train, val, test.
export const SPLIT_COLOR = {
  train: "#0891b2", // cyan - in-sample
  val: "#db2777", // pink - validation (out-of-sample, influenced checkpoint selection)
  test: "#65a30d", // lime - test (out-of-sample, never influenced anything)
};

export const SPLIT_LABEL = {
  train: "In-sample",
  val: "Validation",
  test: "Test",
};

// The probability chart's single series: each asset's own predicted
// probability path.
export const PROBABILITY_COLOR = "#2563eb"; // blue

// The forecast-vs-actual distribution histogram's two series: the
// realized z-score outcomes vs. one sample drawn from the model's own
// predicted N(mu, sigma) per row - distinct from PAIR_PALETTE (different
// chart type/meaning, so no semantic clash).
export const ACTUAL_DIST_COLOR = "#7c3aed"; // violet - realized outcomes
export const FORECAST_DIST_COLOR = "#d97706"; // amber - model-implied outcomes

// The portfolio PnL calculator's two series (see models/portfolio_pnl.py):
// the probability-modulated, vol-targeted position/PnL vs. the unmodulated
// risk-parity baseline it's compared against - distinct from PAIR_PALETTE
// (this pairing means something different: strategy vs. baseline, not
// asset identity).
export const MODULATED_COLOR = "#2563eb"; // blue - probability-modulated, vol-targeted
export const BASELINE_COLOR = "#94a3b8"; // slate - unmodulated risk-parity baseline
// A third, optional series - only present when the loaded model has a
// RiskEngine attached (see models/risk_engine.py) - the SAME modulated
// book with its per-asset attenuation applied on top. Green: this line is
// the one specifically trying to reduce losses, not just a third
// arbitrary category - distinct from MODULATED_COLOR/BASELINE_COLOR and
// from every PAIR_PALETTE entry.
export const RISK_ATTENUATED_COLOR = "#059669";
