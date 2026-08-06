import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS_COLOR, GRID_COLOR, pairColor } from "../theme";

// Each pair's own modulated cumulative PnL contribution (see
// api/server.py's _portfolio_payload - `perAssetCumulativePnl[pair]` sums,
// per day, to PortfolioPnlChart's `cumulativeModulated`) - rendered as its
// own plot, directly below PortfolioPnlChart in EvaluationView.jsx, rather
// than mixed into the book-level totals chart: a single volatile asset's
// swings no longer distort the total/baseline comparison's y-scale, and
// this plot's own y-scale is free to fit the per-asset lines instead. Uses
// the SAME `dates` array, margins, and YAxis width as PortfolioPnlChart so
// the two charts' x-axes land at the same pixel positions when stacked.
export default function PortfolioAssetPnlChart({ dates, pairs, perAssetCumulativePnl, height = 320 }) {
  const data = dates.map((date, i) => {
    const row = { date };
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
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
