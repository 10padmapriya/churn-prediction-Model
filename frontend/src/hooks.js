// src/hooks.js
// Custom React hooks wrapping API calls with loading / error state.

import { useState, useEffect, useCallback } from "react";
import { api } from "./api";

// Generic fetch hook ─────────────────────────────────────────────
export function useFetch(apiFn, deps = []) {
  const [data, setData]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]   = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await apiFn();
      setData(result);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, deps); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { load(); }, [load]);

  return { data, loading, error, reload: load };
}

// Model performance ──────────────────────────────────────────────
export function useModelPerformance() {
  return useFetch(api.modelPerformance);
}

// Health check ───────────────────────────────────────────────────
export function useHealth() {
  return useFetch(api.health);
}

// Predict (manual trigger) ───────────────────────────────────────
export function usePredict() {
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  const predict = useCallback(async (customer) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await api.predict(customer);
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  return { result, loading, error, predict };
}

// Explain (manual trigger) ───────────────────────────────────────
export function useExplain() {
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  const explain = useCallback(async (customer) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await api.explain(customer);
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  return { result, loading, error, explain };
}

// Batch predict (manual trigger) ─────────────────────────────────
export function useBatch() {
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  const submit = useCallback(async (file) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await api.predictBatch(file);
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  return { result, loading, error, submit };
}
