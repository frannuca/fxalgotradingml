import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS_COLOR, BASELINE_COLOR, GRID_COLOR, MODULATED_COLOR, RISK_ATTENUATED_COLOR } from "../theme";

// Whole-book cumulative PnL ONLY (summed across every pair's position *
// that day's realized return - see models/portfolio_pnl.py's
// compute_portfolio): the probability-modulated, target-vol-scaled
// strategy against the unmodulated risk-parity baseline. Deliberately
// does NOT plot per-asset contributions (see the sibling
// PortfolioAssetPnlChart, rendered directly below this one in
// EvaluationView.jsx with the SAME `dates`/margins/YAxis width so the two
// charts' x-axes line up) - keeping the book-level comparison as its own
// plot means its y-scale is never distorted by a single volatile asset's
// swings.
//
// `cumulativeRiskAttenuated` (optional - only present when the loaded
// model has a RiskEngine attached, see models/risk_engine.py) adds a
// THIRD whole-book line: the SAME modulated strategy with its per-asset
// attenuation applied on top, so the risk overlay's effect is visible
// directly against both the unattenuated modulated line and the baseline.
export default function PortfolioPnlChart({
  dates, cumulativeModulated, cumulativeBaseline, cumulativeRiskAttenuated = null, height = 320,
}) {
  const data = dates.map((date, i) => {
    const row = { date, modulated: cumulativeModulated[i], baseline: cumulativeBaseline[i] };
    if (cumulativeRiskAttenuated) row.riskAttenuated = cumulativeRiskAttenuated[i];
    return row;
  });

  return (
    <div className="chart-card">
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid stroke={GRID_COLOR} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 11, fill: AXIS_COLOR }}
            minTickGap={40}
            axisLine={{ stroke: GRID_COLOR }}
            tickLine={false}
          />
          <YAxis tick={{ fontSize: 11, fill: AXIS_COLOR }} axisLine={{ stroke: GRID_COLOR }} tickLine={false} width={64} />
          <Tooltip
            formatter={(value, name) => [Number(value).toFixed(4), name]}
            contentStyle={{ fontSize: 12, borderRadius: 6 }}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Line
            type="monotone"
            dataKey="modulated"
            name="Total (modulated)"
            stroke={MODULATED_COLOR}
            strokeWidth={2.5}
            strokeDasharray="6 3"
            dot={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="baseline"
            name="Total (risk parity, unmodulated)"
            stroke={BASELINE_COLOR}
            strokeWidth={2.5}
            strokeDasharray="6 3"
            dot={false}
            isAnimationActive={false}
          />
          {cumulativeRiskAttenuated && (
            <Line
              type="monotone"
              dataKey="riskAttenuated"
              name="Total (risk-attenuated)"
              stroke={RISK_ATTENUATED_COLOR}
              strokeWidth={2.5}
              dot={false}
              isAnimationActive={false}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
