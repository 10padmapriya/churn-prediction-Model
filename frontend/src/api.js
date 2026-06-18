// src/api.js
// Centralised API client. All components import from here — never from fetch() directly.
// Swap BASE_URL in .env for production deployment.

const BASE_URL = import.meta?.env?.VITE_API_URL ?? "http://localhost:8000";

async function request(method, path, body = null, isForm = false) {
  const opts = {
    method,
    headers: isForm ? undefined : { "Content-Type": "application/json" },
    body: body
      ? isForm
        ? body
        : JSON.stringify(body)
      : undefined,
  };
  const res = await fetch(`${BASE_URL}${path}`, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ── Endpoints ────────────────────────────────────────────────────

export const api = {
  health: () => request("GET", "/health"),

  modelPerformance: () => request("GET", "/model/performance"),

  predict: (customer) => request("POST", "/predict", customer),

  explain: (customer) => request("POST", "/explain", customer),

  predictBatch: (file) => {
    const form = new FormData();
    form.append("file", file);
    return request("POST", "/predict/batch", form, true);
  },
};

export default api;
