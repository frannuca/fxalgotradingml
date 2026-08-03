import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS_COLOR, BASELINE_COLOR, GRID_COLOR, MODULATED_COLOR, pairColor } from "../theme";

// Whole-book cumulative PnL (summed across every pair's position * that
// day's realized return - see models/portfolio_pnl.py's compute_portfolio):
// the probability-modulated, target-vol-scaled strategy against the
// unmodulated risk-parity baseline, PLUS each pair's own modulated
// contribution in the same plot (see api/server.py's _portfolio_payload -
// `perAssetCumulativePnl[pair]` sums, per day, to `cumulativeModulated`).
// Per-asset lines use the same fixed categorical palette every other
// per-pair chart in this app uses (see theme.js's pairColor), thinner than
// the two book-level lines so the aggregate comparison stays the visual
// headline.
export default function PortfolioPnlChart({
  dates, pairs, cumulativeModulated, cumulativeBaseline, perAssetCumulativePnl, height = 320,
}) {
  const data = dates.map((date, i) => {
    const row = { date, modulated: cumulativeModulated[i], baseline: cumulativeBaseline[i] };
    for (const pair of pairs) {
      row[pair] = perAssetCumulativePnl[pair][i];
    }
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
          {pairs.map((pair) => (
            <Line
              key={pair}
              type="monotone"
              dataKey={pair}
              name={`${pair} (modulated)`}
              stroke={pairColor(pairs, pair)}
              strokeWidth={1}
              dot={false}
              isAnimationActive={false}
            />
          ))}
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
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
