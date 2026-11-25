import type { HealthStatus, ModelInfo, MonitoringStats, SourceCitation } from '../types';

const API_BASE = '/api';

export interface ChatApiPayload {
  message: string;
  provider?: string;
  model?: string;
  top_k?: number;
  conversation_history?: Array<{ role: string; content: string }>;
}

export interface ChatApiResponse {
  answer: string;
  sources: SourceCitation[];
  provider: string;
  model: string;
  latency: {
    retrieval_ms: number;
    generation_ms: number;
    total_ms: number;
  };
  timestamp: number;
  error?: string | null;
}

export const api = {
  async getHealth(): Promise<HealthStatus> {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) {
      throw new Error(`Health check failed: ${res.statusText}`);
    }
    return res.json();
  },

  async getModels(): Promise<ModelInfo[]> {
    const res = await fetch(`${API_BASE}/models`);
    if (!res.ok) {
      throw new Error(`Failed to load models: ${res.statusText}`);
    }
    return res.json();
  },

  async selectModel(provider: string, model: string): Promise<void> {
    const res = await fetch(`${API_BASE}/models/select`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider, model }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Failed to switch model');
    }
  },

  async sendMessage(payload: ChatApiPayload): Promise<ChatApiResponse> {
    const res = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${res.status}`);
    }

    return res.json();
  },

  async getMonitoringStats(): Promise<MonitoringStats> {
    const res = await fetch(`${API_BASE}/monitoring/stats`);
    if (!res.ok) {
      throw new Error('Failed to load metrics');
    }
    return res.json();
  },
};
