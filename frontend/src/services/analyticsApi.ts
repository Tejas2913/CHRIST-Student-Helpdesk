// Analytics API service — mirrors existing api.ts pattern

import type {
  AnalyticsSummary,
  ModelStats,
  PaginatedQueryLogs,
  QueryLogEntry,
  VolumePoint,
  LatencyPoint,
  CategoryPoint,
  ErrorBreakdownPoint,
  EvalReport,
} from '../types/analytics';

const BASE = 'http://localhost:8000/api/analytics';

async function get<T>(path: string, params?: Record<string, string | number | undefined>): Promise<T> {
  const url = new URL(`${BASE}${path}`);
  if (params) {
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== '' && v !== null) url.searchParams.set(k, String(v));
    });
  }
  const res = await fetch(url.toString());
  if (!res.ok) throw new Error(`Analytics API error ${res.status}: ${path}`);
  return res.json();
}

export const analyticsApi = {
  getSummary: () => get<AnalyticsSummary>('/summary'),

  getModels: () => get<ModelStats[]>('/models'),

  getQueries: (params: {
    page?: number;
    limit?: number;
    model?: string;
    status?: string;
    date_from?: string;
    date_to?: string;
    search?: string;
    query_type?: string;
    sort_by?: string;
    sort_dir?: string;
  }) => get<PaginatedQueryLogs>('/queries', params as Record<string, string | number | undefined>),

  getQueryDetail: (id: string) => get<QueryLogEntry>(`/queries/${id}`),

  getVolume: (group_by: string) => get<VolumePoint[]>('/volume', { group_by }),

  getLatencyTrend: (group_by: string) => get<LatencyPoint[]>('/latency-trend', { group_by }),

  getCategories: () => get<CategoryPoint[]>('/categories'),

  getErrors: () => get<ErrorBreakdownPoint[]>('/errors'),

  getEval: () => get<EvalReport>('/eval'),

  getExportUrl: (params: {
    model?: string;
    status?: string;
    date_from?: string;
    date_to?: string;
  }) => {
    const url = new URL(`${BASE}/export/csv`);
    Object.entries(params).forEach(([k, v]) => {
      if (v) url.searchParams.set(k, v);
    });
    return url.toString();
  },
};
