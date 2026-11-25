// XAI TypeScript types for CHRIST Student Helpdesk Explainability Lab

export interface SHAPChunkExplanation {
  chunk_id: number | string;
  page: number | null;
  section: string;
  score: number;
  predicted_relevance: number;
  shap_contributions: Record<string, number>;
  top_positive: { feature: string; value: number }[];
  top_negative: { feature: string; value: number }[];
}

export interface SHAPResult {
  target: string;
  base_value: number;
  feature_names: string[];
  global_importance: Record<string, number>;
  chunk_explanations: SHAPChunkExplanation[];
  word_importance?: { word: string; shap_value: number }[];
  predicted_class?: string;
  confidence?: number;
  class_probabilities?: Record<string, number>;
  runtime_ms: number;
  n_chunks_explained?: number;
  surrogate_type?: string;
  error?: string;
  query?: string;
  model?: string;
  explanation_id?: string;
}

export interface LIMEWordImportance {
  word: string;
  lime_weight: number;
  positive: boolean;
}

export interface LIMEFeatureImportance {
  feature_description: string;
  lime_weight: number;
  positive: boolean;
}

export interface LIMEResult {
  target: string;
  query?: string;
  model?: string;
  predicted_class?: string;
  confidence?: number;
  class_probabilities?: Record<string, number>;
  predicted_relevance?: number;
  word_importance?: LIMEWordImportance[];
  feature_importance?: LIMEFeatureImportance[];
  positive_words?: LIMEWordImportance[];
  negative_words?: LIMEWordImportance[];
  positive_features?: LIMEFeatureImportance[];
  negative_features?: LIMEFeatureImportance[];
  local_fidelity: number | null;
  runtime_ms: number;
  num_samples?: number;
  surrogate?: string;
  error?: string;
  explanation_id?: string;
}

export interface SentenceAttribution {
  sentence_id: number;
  sentence: string;
  support_level: 'supported' | 'partial' | 'unsupported';
  similarity_score: number;
  attributed_chunk_id: number | string;
  attributed_page: number | null;
  attributed_section: string;
  attributed_subsection: string;
  document: string;
  supporting_passage: string;
  all_chunk_sims?: number[];
}

export interface AttributionCoverage {
  total_sentences: number;
  supported: number;
  partial: number;
  unsupported: number;
  support_rate: number;
  coverage_rate: number;
}

export interface AttributionResult {
  query: string;
  model?: string;
  answer_preview: string;
  sentence_attributions: SentenceAttribution[];
  coverage: AttributionCoverage;
  runtime_ms: number;
  support_threshold: number;
  partial_threshold: number;
  n_chunks: number;
  error?: string;
}

export interface ComparisonMetric {
  shap: number | null;
  lime: number | null;
  note: string;
  value?: number;
}

export interface XAIComparison {
  query: string;
  metrics: {
    faithfulness: ComparisonMetric;
    stability: ComparisonMetric;
    local_fidelity: ComparisonMetric;
    runtime_ms: ComparisonMetric;
    sparsity: ComparisonMetric;
    feature_consistency: { value: number; note: string };
  };
  top_shap_features: { feature: string; importance: number }[];
  top_lime_features: LIMEFeatureImportance[];
  summary: string;
  shap_detail?: SHAPResult;
  lime_detail?: LIMEResult;
  error?: string;
}

export interface EvidentlyLatency {
  mean_ms: number;
  p50_ms: number;
  p95_ms: number;
  p99_ms: number;
}

export interface EvidentlyReport {
  report_id: string;
  report_type: string;
  model: string;
  generated_at: string;
  n_records?: number;
  runtime_ms: number;
  error?: string;
  metrics?: {
    total_queries: number;
    success_rate: number;
    error_rate: number;
    latency: EvidentlyLatency;
    confidence: { mean: number; std: number; below_0_5: number };
    query_types: Record<string, number>;
    models: Record<string, number>;
    error_types: Record<string, number>;
  };
  drift_summary?: {
    available: boolean;
    overall_drift_detected?: boolean;
    latency_drift?: { ks_statistic: number; p_value: number; drift_detected: boolean };
    query_type_distribution_drift?: {
      query_type: string;
      reference_pct: number;
      current_pct: number;
      delta: number;
      drift: boolean;
    }[];
  };
  drift?: {
    available: boolean;
    overall_drift_detected?: boolean;
    latency_drift?: { ks_statistic: number; p_value: number; drift_detected: boolean };
    query_type_distribution_drift?: {
      query_type: string;
      reference_pct: number;
      current_pct: number;
      delta: number;
      drift: boolean;
    }[];
  };
  report?: EvidentlyReport;
}

export interface XAIBenchmarkResult {
  query: string;
  shap_runtime_ms: number;
  lime_runtime_ms: number;
  feature_consistency: number;
  shap_sparsity: number;
  lime_sparsity: number;
  lime_local_fidelity: number | null;
}

export interface XAIBenchmarkReport {
  model: string;
  n_queries: number;
  results: XAIBenchmarkResult[];
  timestamp: string;
}

export interface ExplanationHistoryItem {
  id: string;
  query_id: string;
  model: string;
  explainer_type: string;
  prediction_target: string;
  runtime_ms: number;
  timestamp: string;
  faithfulness_score?: number;
  local_fidelity_score?: number;
  sparsity_score?: number;
}
