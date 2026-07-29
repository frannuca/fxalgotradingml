import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS_COLOR, GRID_COLOR } from "../theme";

// Renders one pre-binned histogram ({counts, bin_edges} from /api/evaluate,
// computed server-side with numpy so no statistics logic is duplicated here).
export default function HistogramChart({ title, histogram, color, height = 260 }) {
  const { counts, bin_edges: binEdges } = histogram;
  const data = counts.map((count, i) => ({
    binStart: binEdges[i],
    binLabel: ((binEdges[i] + binEdges[i + 1]) / 2).toFixed(4),
    count,
  }));

  return (
    <div className="chart-card">
      <h3>{title}</h3>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid stroke={GRID_COLOR} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="binLabel"
            tick={{ fontSize: 10, fill: AXIS_COLOR }}
            minTickGap={30}
            axisLine={{ stroke: GRID_COLOR }}
            tickLine={false}
          />
          <YAxis tick={{ fontSize: 11, fill: AXIS_COLOR }} axisLine={{ stroke: GRID_COLOR }} tickLine={false} width={40} />
          <Tooltip
            formatter={(value) => [value, "days"]}
            labelFormatter={(label) => `return ≈ ${label}`}
            contentStyle={{ fontSize: 12, borderRadius: 6 }}
          />
          <Bar dataKey="count" fill={color} radius={[2, 2, 0, 0]} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
