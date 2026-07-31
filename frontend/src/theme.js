// Fixed color identity per series, used consistently across every chart in
// the app - color always follows the SAME entity (never re-assigned by
// rank/position), and legends/tooltips are never the only way to tell
// series apart (labels are always visible too).
export const SERIES_COLOR = {
  baseline: "#64748b", // slate - "risk-weighted" baseline, no attenuation
  vol_targeted: "#64748b", // slate - same series identity as "baseline", used in training payloads
  with_risk: "#2563eb", // blue - risk-overlay attenuated
  with_risk_and_costs: "#d97706", // amber - attenuated, net of transaction costs
  live: "#059669", // green - live PnL snapshot while training is running
  benchmark: "#7c3aed", // violet - inverse-vol (risk-weighted, un-learned) benchmark, vol-matched to the model
  model_minus_benchmark: "#dc2626", // red - model's PnL minus the benchmark's, i.e. the value the model itself added
};

export const SERIES_LABEL = {
  baseline: "Risk-weighted (baseline)",
  vol_targeted: "Risk-weighted (vol-targeted)",
  with_risk: "With risk overlay",
  with_risk_and_costs: "With risk overlay + transaction costs",
  live: "Cumulative PnL (live)",
  benchmark: "Inverse-vol benchmark (vol-matched)",
  model_minus_benchmark: "Model − benchmark",
};

export const GRID_COLOR = "#e2e8f0";
export const AXIS_COLOR = "#94a3b8";
export const TEXT_COLOR = "#334155";

// Fixed categorical palette for per-asset series (positions, attenuation,
// cumulative returns) whose keys are FX pair names chosen by the user, not
// known in advance. Assigned by POSITION in the pairs list, never by rank/
// value, so a given pair keeps the same color across every chart it appears
// in during a single training/evaluation run.
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

export const SPLIT_LINE_COLOR = "#94a3b8";
