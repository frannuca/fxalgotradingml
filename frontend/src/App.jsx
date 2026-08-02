import { NavLink, Route, Routes, Navigate } from "react-router-dom";
import TrainingView from "./TrainingView";
import EvaluationView from "./EvaluationView";

export default function App() {
  return (
    <div className="app">
      <h1>FX Direction Prediction</h1>
      <p className="subtitle">Two-stage probabilistic predictor: independent per-asset LSTMs + a cross-asset copula LSTM</p>

      <nav className="app-nav">
        <NavLink to="/training" className={({ isActive }) => (isActive ? "active" : "")}>
          Training
        </NavLink>
        <NavLink to="/evaluation" className={({ isActive }) => (isActive ? "active" : "")}>
          Evaluation
        </NavLink>
      </nav>

      <Routes>
        <Route path="/" element={<Navigate to="/training" replace />} />
        <Route path="/training" element={<TrainingView />} />
        <Route path="/evaluation" element={<EvaluationView />} />
      </Routes>
    </div>
  );
}
