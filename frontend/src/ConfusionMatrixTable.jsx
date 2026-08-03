import { SPLIT_LABEL } from "./theme";

// `confusionMatrix` is EITHER {train: {<pair>: {...}}, val: {...}, test:
// {...}} (Training view - three splits genuinely mean something there:
// val drives checkpoint selection, test is held out) OR a FLAT
// {<pair>: {tp,fp,tn,fn,abstained,coverage,accuracy,precision,recall,
// specificity,f1}} (Evaluation view - a single continuous period, no
// split - see api/server.py's evaluate()/_confusion_matrix_payload).
// Detected by whether a `train` key is present, so this one component
// serves both without either view needing to know about the other's shape.
export default function ConfusionMatrixTable({ pairs, confusionMatrix }) {
  const isSplit = confusionMatrix && typeof confusionMatrix.train === "object";
  const splits = isSplit ? ["train", "val", "test"] : [null];

  return (
    <div className="panel">
      {splits.map((split) => (
        <div key={split ?? "flat"} style={{ marginBottom: 20 }}>
          {split && <h3 style={{ marginBottom: 6 }}>{SPLIT_LABEL[split]}</h3>}
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
                  const m = split ? confusionMatrix[split][pair] : confusionMatrix[pair];
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
