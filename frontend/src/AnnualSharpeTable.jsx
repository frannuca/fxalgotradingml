// `annualSharpe` is {<year>: {n_days, sharpe_modulated, sharpe_baseline,
// return_modulated, return_baseline}} (see models/portfolio_pnl.py's
// annual_sharpe_table) - one row per calendar year, so a reviewer can see
// whether a strategy's overall Sharpe is driven by one lucky year rather
// than being consistent across the backtest. `sharpe_*` is `null` for a
// year with fewer than 2 days (std undefined) - rendered as "—".
export default function AnnualSharpeTable({ annualSharpe }) {
  const years = Object.keys(annualSharpe).sort();
  if (years.length === 0) return null;

  return (
    <div className="panel">
      <h3 style={{ marginTop: 0, marginBottom: 6 }}>Sharpe ratio by year</h3>
      <div style={{ overflowX: "auto" }}>
        <table className="weights-table">
          <thead>
            <tr>
              <th>Year</th>
              <th>Days</th>
              <th>Sharpe (modulated)</th>
              <th>Sharpe (risk parity)</th>
              <th>Return (modulated)</th>
              <th>Return (risk parity)</th>
            </tr>
          </thead>
          <tbody>
            {years.map((year) => {
              const row = annualSharpe[year];
              return (
                <tr key={year}>
                  <td>{year}</td>
                  <td>{row.n_days}</td>
                  <td>{row.sharpe_modulated == null ? "—" : row.sharpe_modulated.toFixed(2)}</td>
                  <td>{row.sharpe_baseline == null ? "—" : row.sharpe_baseline.toFixed(2)}</td>
                  <td>{(row.return_modulated * 100).toFixed(2)}%</td>
                  <td>{(row.return_baseline * 100).toFixed(2)}%</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
