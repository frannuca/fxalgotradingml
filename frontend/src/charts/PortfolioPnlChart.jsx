import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS_COLOR, BASELINE_COLOR, GRID_COLOR, MODULATED_COLOR } from "../theme";

// Whole-book cumulative PnL (summed across every pair's position * that
// day's realized return - see models/portfolio_pnl.py's compute_portfolio):
// the probability-modulated, target-vol-scaled strategy against the
// unmodulated risk-parity baseline.
export default function PortfolioPnlChart({ dates, cumulativeModulated, cumulativeBaseline, height = 300 }) {
  const data = dates.map((date, i) => ({
    date, modulated: cumulativeModulated[i], baseline: cumulativeBaseline[i],
  }));

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
            formatter={(value) => [Number(value).toFixed(4), "cumulative pnl"]}
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
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="baseline"
            name="Risk parity (unmodulated)"
            stroke={BASELINE_COLOR}
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
