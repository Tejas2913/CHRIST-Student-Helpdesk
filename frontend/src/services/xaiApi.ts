// XAI API service for CHRIST Student Helpdesk Explainability Lab

import type {
  SHAPResult,
  LIMEResult,
  AttributionResult,
  XAIComparison,
  EvidentlyReport,
  XAIBenchmarkReport,
  ExplanationHistoryItem,
} from '../types/xai';

const BASE = '/api';

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const err = await r.text();
    throw new Error(`${path} failed (${r.status}): ${err}`);
  }
  return r.json() as Promise<T>;
}

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(`${path} failed (${r.status})`);
  return r.json() as Promise<T>;
}

// --- SHAP ---
export async function fetchSHAP(
  query: string,
  model: string,
  target: 'retrieval_relevance' | 'intent_classification',
  chunks?: unknown[],
): Promise<SHAPResult> {
  return post('/xai/shap', { query, model, target, chunks });
}

// --- LIME ---
export async function fetchLIME(
  query: string,
  model: string,
  target: 'retrieval_relevance' | 'intent_classification',
  chunks?: unknown[],
): Promise<LIMEResult> {
  return post('/xai/lime', { query, model, target, chunks });
}

// --- Attribution ---
export async function fetchAttribution(
  query: string,
  answer: string,
  model: string,
  chunks?: unknown[],
): Promise<AttributionResult> {
  return post('/xai/attribution', { query, answer, model, target: 'attribution', chunks });
}

// --- SHAP vs LIME comparison ---
export async function fetchComparison(
  query: string,
  model: string,
  target: 'retrieval_relevance' | 'intent_classification',
  chunks?: unknown[],
): Promise<XAIComparison> {
  return post('/xai/compare', { query, model, target, chunks });
}

// --- History ---
export async function fetchXAIHistory(limit = 20, explainerType?: string): Promise<ExplanationHistoryItem[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (explainerType) params.append('explainer_type', explainerType);
  return get(`/xai/history?${params}`);
}

// --- Global metrics ---
export async function fetchGlobalSHAPMetrics(): Promise<{
  global_importance: Record<string, number>;
  retrieval_features?: Record<string, number>;
  linguistic_tokens?: Record<string, number>;
  n_explanations: number;
}> {
  return get('/xai/metrics/global');
}

// --- Benchmark ---
export async function runXAIBenchmark(model: string, nQueries = 5): Promise<XAIBenchmarkReport> {
  return get(`/xai/benchmark?model=${encodeURIComponent(model)}&n_queries=${nQueries}`);
}

// --- Evidently ---
export async function generateEvidentlyReport(
  reportType: 'rag_quality' | 'drift',
  model?: string,
  limit = 100,
): Promise<EvidentlyReport> {
  return post('/evidently/generate', { report_type: reportType, model, limit });
}

export async function fetchEvidentlyReports(limit = 10): Promise<EvidentlyReport[]> {
  return get(`/evidently/reports?limit=${limit}`);
}

export async function fetchDriftReport(model?: string): Promise<EvidentlyReport> {
  const params = model ? `?model=${encodeURIComponent(model)}` : '';
  return get(`/evidently/drift${params}`);
}
