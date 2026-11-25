export interface SourceCitation {
  chunk_id: number;
  page: number;
  score: number;
  text: string;
}

export interface LatencyMetrics {
  retrieval_ms: number;
  generation_ms: number;
  total_ms: number;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  sources?: SourceCitation[];
  provider?: string;
  model?: string;
  latency?: LatencyMetrics;
  error?: string;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

export interface ModelInfo {
  id: string;
  name: string;
  provider: string;
  is_available: boolean;
  is_default: boolean;
  is_selected?: boolean;
  notes?: string;
}

export interface HealthStatus {
  status: string;
  app_name: string;
  version: string;
  vector_count: number;
  chunk_count: number;
  active_provider: string;
  active_model: string;
  provider_available: boolean;
  handbook_loaded: boolean;
}

export interface MonitoringStats {
  total_requests: number;
  avg_total_latency_ms: number;
  avg_retrieval_latency_ms: number;
  avg_generation_latency_ms: number;
  error_rate_pct: number;
  providers: Record<string, number>;
}
