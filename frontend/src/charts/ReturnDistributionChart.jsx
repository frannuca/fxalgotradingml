import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ACTUAL_DIST_COLOR, AXIS_COLOR, FORECAST_DIST_COLOR, GRID_COLOR } from "../theme";

// Bucket two equal-length arrays into a shared set of bins and return each
// bin's share (%) of its own series - density, not raw count, so the two
// series stay comparable even if array lengths ever differ.
function buildHistogram(actual, forecasted, binCount) {
  const all = actual.concat(forecasted);
  const min = Math.min(...all);
  const max = Math.max(...all);
  const width = (max - min || 1) / binCount;
  const bins = Array.from({ length: binCount }, (_, i) => ({
    z: min + (i + 0.5) * width, // bin center, used as the x label
    actualCount: 0,
    forecastedCount: 0,
  }));
  const bucket = (v) => Math.min(Math.max(Math.floor((v - min) / width), 0), binCount - 1);
  actual.forEach((v) => bins[bucket(v)].actualCount++);
  forecasted.forEach((v) => bins[bucket(v)].forecastedCount++);
  const nActual = actual.length || 1;
  const nForecasted = forecasted.length || 1;
  return bins.map((b) => ({
    z: b.z,
    Actual: (b.actualCount / nActual) * 100,
    Forecasted: (b.forecastedCount / nForecasted) * 100,
  }));
}

function mean(xs) {
  return xs.reduce((a, b) => a + b, 0) / (xs.length || 1);
}
function std(xs) {
  const m = mean(xs);
  return Math.sqrt(mean(xs.map((x) => (x - m) ** 2)));
}

// One asset's forecasted-vs-actual z-score distribution: `series` is
// {actual: [...], forecasted: [...]} (see api/server.py's
// _distribution_payload) - `actual` is the realized decision-day z-score,
// `forecasted` is one sample drawn from the model's own predicted
// N(mu, sigma) per row. If the model's predictive distributions are
// well-calibrated, these two histograms should look statistically similar
// (same center, spread, shape) even though no individual pair of values
// need match - this is a distributional check, not a point-accuracy one.
export default function ReturnDistributionChart({ title, series, height = 260, binCount = 24 }) {
  const { actual, forecasted } = series;
  const data = buildHistogram(actual, forecasted, binCount);
  const actualMean = mean(actual);
  const actualStd = std(actual);
  const forecastedMean = mean(forecasted);
  const forecastedStd = std(forecasted);

  return (
    <div className="chart-card">
      <h3>{title}</h3>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }} barGap={0} barCategoryGap="10%">
          <CartesianGrid stroke={GRID_COLOR} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="z"
            tickFormatter={(v) => v.toFixed(1)}
            tick={{ fontSize: 11, fill: AXIS_COLOR }}
            axisLine={{ stroke: GRID_COLOR }}
            tickLine={false}
            interval="preserveStartEnd"
            label={{ value: "z-score (vol-normalized return)", position: "insideBottom", offset: -4, fontSize: 10, fill: AXIS_COLOR }}
          />
          <YAxis
            tickFormatter={(v) => `${v.toFixed(0)}%`}
            tick={{ fontSize: 11, fill: AXIS_COLOR }}
            axisLine={{ stroke: GRID_COLOR }}
            tickLine={false}
            width={40}
          />
          <Tooltip
            formatter={(value) => [`${Number(value).toFixed(1)}%`, undefined]}
            labelFormatter={(v) => `z ≈ ${Number(v).toFixed(2)}`}
            contentStyle={{ fontSize: 12, borderRadius: 6 }}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Bar dataKey="Actual" fill={ACTUAL_DIST_COLOR} fillOpacity={0.75} isAnimationActive={false} />
          <Bar dataKey="Forecasted" fill={FORECAST_DIST_COLOR} fillOpacity={0.75} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
      <div style={{ fontSize: 12, color: AXIS_COLOR, marginTop: 4 }}>
        actual: mean {actualMean.toFixed(2)}, std {actualStd.toFixed(2)} &nbsp;·&nbsp;
        forecasted: mean {forecastedMean.toFixed(2)}, std {forecastedStd.toFixed(2)}
      </div>
    </div>
  );
}
