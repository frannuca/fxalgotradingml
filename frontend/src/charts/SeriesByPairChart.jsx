import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AXIS_COLOR, GRID_COLOR, pairColor, SPLIT_LINE_COLOR } from "../theme";

// Renders one line per FX pair sharing a date axis - used for positions,
// attenuation factors, and cumulative asset returns. `series` is the
// {dates, <pair>: [...], ...} shape the API returns; `pairs` fixes both the
// set of lines to draw and their color assignment (by position, not value).
// `splitDate`, when given, draws a vertical reference line (e.g. marking
// where the out-of-sample period starts).
export default function SeriesByPairChart({ title, series, pairs, splitDate, height = 300 }) {
  const data = series.dates.map((date, i) => {
    const point = { date };
    for (const pair of pairs) point[pair] = series[pair][i];
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
            formatter={(value, key) => [Number(value).toFixed(4), key]}
            contentStyle={{ fontSize: 12, borderRadius: 6 }}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {splitDate && (
            <ReferenceLine
              x={splitDate}
              stroke={SPLIT_LINE_COLOR}
              strokeDasharray="4 4"
              label={{ value: "out-of-sample start", position: "insideTopRight", fontSize: 10, fill: SPLIT_LINE_COLOR }}
            />
          )}
          {pairs.map((pair) => (
            <Line
              key={pair}
              type="monotone"
              dataKey={pair}
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
