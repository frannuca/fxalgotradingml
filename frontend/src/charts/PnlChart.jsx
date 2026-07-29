import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AXIS_COLOR, GRID_COLOR, SERIES_COLOR, SERIES_LABEL } from "../theme";

// Renders one or more cumulative-PnL lines sharing a date axis. `pnl` is
// the {dates, <seriesKey>: [...], ...} shape /api/evaluate returns;
// `seriesKeys` picks which of its series to draw, in the order given.
export default function PnlChart({ title, pnl, seriesKeys, height = 320 }) {
  const data = pnl.dates.map((date, i) => {
    const point = { date };
    for (const key of seriesKeys) point[key] = pnl[key][i];
    return point;
  });

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
          <YAxis
            tick={{ fontSize: 11, fill: AXIS_COLOR }}
            axisLine={{ stroke: GRID_COLOR }}
            tickLine={false}
            width={56}
          />
          <Tooltip
            formatter={(value, key) => [value.toFixed(4), SERIES_LABEL[key] || key]}
            contentStyle={{ fontSize: 12, borderRadius: 6 }}
          />
          <Legend
            formatter={(key) => SERIES_LABEL[key] || key}
            wrapperStyle={{ fontSize: 12 }}
          />
          {seriesKeys.map((key) => (
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              stroke={SERIES_COLOR[key] || "#333"}
              strokeWidth={1.75}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
