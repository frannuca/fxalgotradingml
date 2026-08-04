// Thin client for the FastAPI backend (api/server.py). Every function
// returns a parsed JSON body or throws with the backend's error detail.

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      // response wasn't JSON - keep statusText
    }
    throw new Error(detail);
  }
  return res.json();
}

export function getPairs() {
  return request("/api/pairs");
}

export function getModels() {
  return request("/api/models");
}

export function refreshQuotes({ pairs, years }) {
  return request("/api/quotes/refresh", {
    method: "POST",
    body: JSON.stringify({ pairs, years }),
  });
}

export function startTraining(config) {
  return request("/api/train", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

export function getTrainingStatus(jobId) {
  return request(`/api/train/${jobId}`);
}

export function stopTraining(jobId) {
  return request(`/api/train/${jobId}/stop`, { method: "POST" });
}

export function saveBestCheckpoint(jobId) {
  return request(`/api/train/${jobId}/save-best`, { method: "POST" });
}

export function evaluate(config) {
  return request("/api/evaluate", {
    method: "POST",
    body: JSON.stringify(config),
  });
}
