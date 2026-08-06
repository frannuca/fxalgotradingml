// `annualSharpe` is {<year>: {n_days, sharpe_modulated, sharpe_baseline,
// return_modulated, return_baseline, [sharpe_risk_attenuated,
// return_risk_attenuated]}} (see models/portfolio_pnl.py's
// annual_sharpe_table) - one row per calendar year, so a reviewer can see
// whether a strategy's overall Sharpe is driven by one lucky year rather
// than being consistent across the backtest. `sharpe_*` is `null` for a
// year with fewer than 2 days (std undefined) - rendered as "—". The
// risk-attenuated columns (only present when the loaded model has a
// RiskEngine attached) report a year's Sharpe/return for the FULL
// calendar year even if the chosen "plot start date" falls in the middle
// of it - see EvaluationView.jsx's own note on this.
export default function AnnualSharpeTable({ annualSharpe }) {
  const years = Object.keys(annualSharpe).sort();
  if (years.length === 0) return null;
  const hasRisk = years.some((year) => annualSharpe[year].sharpe_risk_attenuated !== undefined);

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
              {hasRisk && <th>Sharpe (risk-attenuated)</th>}
              <th>Return (modulated)</th>
              <th>Return (risk parity)</th>
              {hasRisk && <th>Return (risk-attenuated)</th>}
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
                  {hasRisk && <td>{row.sharpe_risk_attenuated == null ? "—" : row.sharpe_risk_attenuated.toFixed(2)}</td>}
                  <td>{(row.return_modulated * 100).toFixed(2)}%</td>
                  <td>{(row.return_baseline * 100).toFixed(2)}%</td>
                  {hasRisk && <td>{(row.return_risk_attenuated * 100).toFixed(2)}%</td>}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
