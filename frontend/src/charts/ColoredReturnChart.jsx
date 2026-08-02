import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ABSTAIN_COLOR, AXIS_COLOR, GRID_COLOR, HIT_COLOR, MISS_COLOR, RETURN_LINE_COLOR } from "../theme";

// Colored dot per day: green if that day's predicted direction matched
// the realized outcome, red if it didn't, slate if the model abstained
// (inside the neutral band - see apply_neutral_band) - neither a hit nor
// a miss, so it must not read as either. See api/server.py's
// _cumulative_return_payload. The connecting line itself stays a neutral
// color - the dots carry the hit/miss/abstain signal, not the line's own
// color, since correctness doesn't come in contiguous runs.
function HitDot({ cx, cy, payload }) {
  if (cx == null || cy == null) return null;
  const fill = payload.abstained ? ABSTAIN_COLOR : payload.hit ? HIT_COLOR : MISS_COLOR;
  return <circle cx={cx} cy={cy} r={2.5} fill={fill} stroke="none" />;
}

// One asset's own cumulative (single-day-ahead) log-return path, dotted
// green/red/slate by whether the model's prediction that day was
// correct, wrong, or abstained. `series` is {cumulative: [...],
// hit: [...], abstained: [...]} (see api/server.py's
// _cumulative_return_payload); `dates` is the split's own shared date
// array.
export default function ColoredReturnChart({ title, dates, series, height = 280 }) {
  const data = dates.map((date, i) => ({
    date, cumulative: series.cumulative[i], hit: series.hit[i], abstained: series.abstained[i],
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
          <Tooltip
            formatter={(value, _key, item) => [
              `${Number(value).toFixed(4)} (${item.payload.abstained ? "abstained" : item.payload.hit ? "hit" : "miss"})`,
              "cumulative return",
            ]}
            contentStyle={{ fontSize: 12, borderRadius: 6 }}
          />
          <Line
            type="monotone"
            dataKey="cumulative"
            stroke={RETURN_LINE_COLOR}
            strokeWidth={1}
            dot={<HitDot />}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
      <div style={{ display: "flex", gap: 16, fontSize: 12, color: AXIS_COLOR, marginTop: 4 }}>
        <span>
          <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: HIT_COLOR, marginRight: 4 }} />
          predicted direction correct
        </span>
        <span>
          <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: MISS_COLOR, marginRight: 4 }} />
          predicted direction wrong
        </span>
        <span>
          <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: ABSTAIN_COLOR, marginRight: 4 }} />
          abstained (inside neutral band)
        </span>
      </div>
    </div>
  );
}
