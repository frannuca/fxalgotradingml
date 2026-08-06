import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS_COLOR, BASELINE_COLOR, GRID_COLOR, MODULATED_COLOR, RISK_ATTENUATED_COLOR } from "../theme";

// One asset's own position over time: the probability-modulated,
// target-vol-scaled weight (see models/portfolio_pnl.py's compute_portfolio)
// against the SAME asset's unmodulated risk-parity baseline weight (also
// scaled to the model's target_vol, for a like-for-like comparison) - both
// can go negative (short) since the modulated weight flips sign with the
// probability signal. `series` is {position_modulated: [...],
// position_baseline: [...], [position_risk_attenuated: [...]]} (see
// api/server.py's _portfolio_payload) - the risk-attenuated line (only
// present when the loaded model has a RiskEngine attached) is the SAME
// modulated weight with its per-day attenuation applied on top.
export default function PortfolioPositionChart({ title, dates, series, height = 260 }) {
  const hasRisk = series.position_risk_attenuated != null;
  const data = dates.map((date, i) => ({
    date, modulated: series.position_modulated[i], baseline: series.position_baseline[i],
    ...(hasRisk ? { riskAttenuated: series.position_risk_attenuated[i] } : {}),
  }));

  return (
    <div className="chart-card">
      <h3>{title}</h3>
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
          <YAxis tick={{ fontSize: 11, fill: AXIS_COLOR }} axisLine={{ stroke: GRID_COLOR }} tickLine={false} width={56} />
          <ReferenceLine y={0} stroke={AXIS_COLOR} />
          <Tooltip
            formatter={(value) => (value == null ? ["—"] : [Number(value).toFixed(3), "weight"])}
            contentStyle={{ fontSize: 12, borderRadius: 6 }}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Line
            type="monotone"
            dataKey="modulated"
            name="Modulated (vol-targeted)"
            stroke={MODULATED_COLOR}
            strokeWidth={1.5}
            dot={false}
            connectNulls={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="baseline"
            name="Risk parity (unmodulated)"
            stroke={BASELINE_COLOR}
            strokeWidth={1.5}
            dot={false}
            connectNulls={false}
            isAnimationActive={false}
          />
          {hasRisk && (
            <Line
              type="monotone"
              dataKey="riskAttenuated"
              name="Risk-attenuated"
              stroke={RISK_ATTENUATED_COLOR}
              strokeWidth={1.5}
              dot={false}
              connectNulls={false}
              isAnimationActive={false}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
