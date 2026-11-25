// Analytics TypeScript types for the dashboard

export interface AnalyticsSummary {
  total_queries: number;
  queries_today: number;
  total_responses: number;
  avg_latency_ms: number;
  avg_retrieval_ms: number;
  avg_generation_ms: number;
  error_rate_pct: number;
  best_model: string | null;
  models_active: number;
}

export interface ModelStats {
  model: string;
  provider: string;
  total_queries: number;
  avg_latency_ms: number;
  avg_retrieval_ms: number;
  avg_generation_ms: number;
  successes: number;
  errors: number;
  success_rate_pct: number;
  error_rate_pct: number;
  avg_sources: number;
  avg_tokens: number | null;
  p50_latency_ms: number;
  p95_latency_ms: number;
}

export interface QueryLogEntry {
  id: string;
  timestamp: string;
  query: string;
  provider: string;
  model: string;
  query_type: string | null;
  answer_preview: string | null;
  confidence: string | null;
  total_latency_ms: number;
  retrieval_latency_ms: number;
  generation_latency_ms: number;
  embedding_latency_ms: number;
  rerank_latency_ms: number;
  sources_retrieved: number;
  source_pages: number[];
  retrieved_chunk_ids: string[];
  token_count: number | null;
  status: string;
  error: string | null;
  eval_correctness: number | null;
  eval_faithfulness: number | null;
  eval_relevance: number | null;
}

export interface PaginatedQueryLogs {
  data: QueryLogEntry[];
  total: number;
  page: number;
  limit: number;
  pages: number;
}

export interface VolumePoint {
  period: string;
  model: string;
  count: number;
}

export interface LatencyPoint {
  period: string;
  model: string;
  avg_latency_ms: number;
  avg_retrieval_ms: number;
  avg_generation_ms: number;
  count: number;
}

export interface CategoryPoint {
  query_type: string;
  count: number;
}

export interface ErrorBreakdownPoint {
  model: string;
  status: string;
  count: number;
}

export interface EvalReport {
  timestamp?: string;
  total_queries?: number;
  hit_rate_at_1?: number;
  hit_rate_at_5?: number;
  mrr?: number;
  avg_retrieval_ms?: number;
  error?: string;
  [key: string]: unknown;
}

export interface DashboardFilters {
  model: string;
  status: string;
  dateFrom: string;
  dateTo: string;
  search: string;
  groupBy: 'hour' | 'day' | 'week';
}
