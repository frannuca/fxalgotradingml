import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AXIS_COLOR, GRID_COLOR, SPLIT_COLOR, SPLIT_LABEL } from "../theme";

// Grouped bar chart: one asset per group, three bars (train/val/test) -
// `hitRate` is {train: {<pair>: rate}, val: {...}, test: {...}} (see
// api/server.py's _hit_rate_payload). A dashed reference line at 0.5 marks
// random-chance (no directional skill) so a bar's position relative to it
// is immediately readable.
export default function HitRateChart({ pairs, hitRate, height = 300 }) {
  const data = pairs.map((pair) => ({
    pair,
    [SPLIT_LABEL.train]: hitRate.train[pair],
    [SPLIT_LABEL.val]: hitRate.val[pair],
    [SPLIT_LABEL.test]: hitRate.test[pair],
  }));

  return (
    <div className="chart-card">
      <h3>Directional hit rate</h3>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid stroke={GRID_COLOR} strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="pair" tick={{ fontSize: 11, fill: AXIS_COLOR }} axisLine={{ stroke: GRID_COLOR }} tickLine={false} />
          <YAxis
            domain={[0, 1]}
            tickFormatter={(v) => `${Math.round(v * 100)}%`}
            tick={{ fontSize: 11, fill: AXIS_COLOR }}
            axisLine={{ stroke: GRID_COLOR }}
            tickLine={false}
            width={44}
          />
          <Tooltip
            formatter={(value, key) => [`${(value * 100).toFixed(1)}%`, key]}
            contentStyle={{ fontSize: 12, borderRadius: 6 }}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <ReferenceLine
            y={0.5}
            stroke={AXIS_COLOR}
            strokeDasharray="4 4"
            label={{ value: "random chance (50%)", position: "insideTopRight", fontSize: 10, fill: AXIS_COLOR }}
          />
          <Bar dataKey={SPLIT_LABEL.train} fill={SPLIT_COLOR.train} radius={[2, 2, 0, 0]} isAnimationActive={false} />
          <Bar dataKey={SPLIT_LABEL.val} fill={SPLIT_COLOR.val} radius={[2, 2, 0, 0]} isAnimationActive={false} />
          <Bar dataKey={SPLIT_LABEL.test} fill={SPLIT_COLOR.test} radius={[2, 2, 0, 0]} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
