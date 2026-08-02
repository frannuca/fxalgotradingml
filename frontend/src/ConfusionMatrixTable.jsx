import { SPLIT_LABEL } from "./theme";

// `confusionMatrix` is {train: {<pair>: {tp,fp,tn,fn,abstained,coverage,
// accuracy,precision,recall,specificity,f1}}, val: {...}, test: {...}}
// (see api/server.py's _confusion_matrix_payload) - one table per split,
// one row per asset. With a neutral band, every metric except `abstained`
// itself is computed over DECIDED samples only - accuracy and coverage
// must be read together (100% accuracy at 2% coverage is a very
// different claim from 55% at 80%).
export default function ConfusionMatrixTable({ pairs, confusionMatrix }) {
  return (
    <div className="panel">
      {["train", "val", "test"].map((split) => (
        <div key={split} style={{ marginBottom: 20 }}>
          <h3 style={{ marginBottom: 6 }}>{SPLIT_LABEL[split]}</h3>
          <div style={{ overflowX: "auto" }}>
            <table className="weights-table">
              <thead>
                <tr>
                  <th>Pair</th>
                  <th>TP</th>
                  <th>FP</th>
                  <th>TN</th>
                  <th>FN</th>
                  <th>Abstained</th>
                  <th>Coverage</th>
                  <th>Accuracy</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>Specificity</th>
                  <th>F1</th>
                </tr>
              </thead>
              <tbody>
                {pairs.map((pair) => {
                  const m = confusionMatrix[split][pair];
                  return (
                    <tr key={pair}>
                      <td>{pair}</td>
                      <td>{m.tp}</td>
                      <td>{m.fp}</td>
                      <td>{m.tn}</td>
                      <td>{m.fn}</td>
                      <td>{m.abstained}</td>
                      <td>{(m.coverage * 100).toFixed(1)}%</td>
                      <td>{(m.accuracy * 100).toFixed(1)}%</td>
                      <td>{(m.precision * 100).toFixed(1)}%</td>
                      <td>{(m.recall * 100).toFixed(1)}%</td>
                      <td>{(m.specificity * 100).toFixed(1)}%</td>
                      <td>{m.f1.toFixed(3)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      ))}
    </div>
  );
}
