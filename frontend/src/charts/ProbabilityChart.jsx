import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AXIS_COLOR, GRID_COLOR, PROBABILITY_COLOR } from "../theme";
import { applyNeutralBand } from "../metrics";

// One asset's own predicted probability path - `series` is
// {probability: [...], label: [...]} (see api/server.py's
// _probability_payload), RAW (not pre-snapped). `band`, if given, is
// applied client-side (see metrics.js's applyNeutralBand) so the line
// flattens to exactly 0.5 during abstention for whatever band is
// currently selected - updates live as the user adjusts it, no new
// request. A dashed 0.5 reference line marks the long/short decision
// boundary.
export default function ProbabilityChart({ title, dates, series, band = 0, height = 260 }) {
  const data = dates.map((date, i) => ({ date, probability: applyNeutralBand(series.probability[i], band) }));

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
            domain={[0, 1]}
            tickFormatter={(v) => `${Math.round(v * 100)}%`}
            tick={{ fontSize: 11, fill: AXIS_COLOR }}
            axisLine={{ stroke: GRID_COLOR }}
            tickLine={false}
            width={44}
          />
          <Tooltip
            formatter={(value) => [`${(value * 100).toFixed(1)}%`, "probability"]}
            contentStyle={{ fontSize: 12, borderRadius: 6 }}
          />
          <ReferenceLine y={0.5} stroke={AXIS_COLOR} strokeDasharray="4 4" />
          <Line
            type="monotone"
            dataKey="probability"
            stroke={PROBABILITY_COLOR}
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
