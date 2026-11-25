import React from 'react';
import type { ModelStats } from '../../types/analytics';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface ModelComparisonCardsProps {
  models: ModelStats[];
  loading: boolean;
}

const fmtMs = (ms: number) =>
  ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`;

const MODEL_COLORS: Record<string, { bg: string; border: string; badge: string; text: string }> = {
  'qwen/qwen3.8-27b':     { bg: 'bg-indigo-50',  border: 'border-indigo-200', badge: 'bg-indigo-100 text-indigo-700',  text: 'text-indigo-700' },
  'openai/gpt-oss-120b':  { bg: 'bg-violet-50',  border: 'border-violet-200', badge: 'bg-violet-100 text-violet-700',  text: 'text-violet-700' },
  'openai/gpt-oss-20b':   { bg: 'bg-sky-50',     border: 'border-sky-200',    badge: 'bg-sky-100 text-sky-700',        text: 'text-sky-700'    },
};

const DEFAULT_COLOR = { bg: 'bg-slate-50', border: 'border-slate-200', badge: 'bg-slate-100 text-slate-700', text: 'text-slate-700' };

function getColor(model: string) {
  return MODEL_COLORS[model] || DEFAULT_COLOR;
}

interface MetricRowProps {
  label: string;
  value: string;
  highlight?: 'best' | 'worst' | 'neutral';
}

const MetricRow: React.FC<MetricRowProps> = ({ label, value, highlight }) => (
  <div className="flex items-center justify-between py-1 border-b border-slate-100 last:border-0">
    <span className="text-xs text-slate-500">{label}</span>
    <span className={`text-xs font-semibold flex items-center gap-1 ${
      highlight === 'best' ? 'text-emerald-600' :
      highlight === 'worst' ? 'text-rose-500' : 'text-slate-700'
    }`}>
      {highlight === 'best' && <TrendingUp className="w-3 h-3" />}
      {highlight === 'worst' && <TrendingDown className="w-3 h-3" />}
      {highlight === 'neutral' && <Minus className="w-3 h-3 text-slate-300" />}
      {value}
    </span>
  </div>
);

function bestIdx(models: ModelStats[], key: keyof ModelStats, lower = true): number {
  if (!models.length) return -1;
  let best = -1;
  let bestVal = lower ? Infinity : -Infinity;
  models.forEach((m, i) => {
    const v = m[key] as number;
    if (typeof v === 'number') {
      if (lower ? v < bestVal : v > bestVal) { bestVal = v; best = i; }
    }
  });
  return best;
}

export const ModelComparisonCards: React.FC<ModelComparisonCardsProps> = ({ models, loading }) => {
  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[0, 1, 2].map(i => (
          <div key={i} className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 animate-pulse">
            <div className="h-4 bg-slate-100 rounded w-3/4 mb-3" />
            <div className="space-y-2">
              {[0,1,2,3,4].map(j => <div key={j} className="h-3 bg-slate-100 rounded" />)}
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (!models.length) {
    return <div className="text-slate-400 text-sm text-center py-8">No model data yet — run some queries first.</div>;
  }

  const bestLatency = bestIdx(models, 'avg_latency_ms', true);
  const bestSuccess = bestIdx(models, 'success_rate_pct', false);
  const bestSources = bestIdx(models, 'avg_sources', false);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
      {models.map((m, i) => {
        const c = getColor(m.model);
        return (
          <div key={m.model} className={`${c.bg} rounded-xl border ${c.border} shadow-sm p-5`}>
            {/* Header */}
            <div className="flex items-start justify-between mb-3">
              <div>
                <p className="text-xs text-slate-500 uppercase tracking-wide font-medium">{m.provider}</p>
                <h3 className={`font-bold text-sm ${c.text} break-all leading-tight mt-0.5`}>{m.model}</h3>
              </div>
              <span className={`text-[10px] font-bold uppercase px-2 py-0.5 rounded-full ${c.badge} flex-shrink-0 ml-2`}>
                {m.total_queries} req
              </span>
            </div>

            {/* Metrics */}
            <div className="space-y-0">
              <MetricRow
                label="Avg Latency"
                value={fmtMs(m.avg_latency_ms)}
                highlight={i === bestLatency ? 'best' : i === models.length - 1 ? 'worst' : 'neutral'}
              />
              <MetricRow label="P50 Latency" value={fmtMs(m.p50_latency_ms)} />
              <MetricRow label="P95 Latency" value={fmtMs(m.p95_latency_ms)} />
              <MetricRow label="Avg Retrieval" value={fmtMs(m.avg_retrieval_ms)} />
              <MetricRow label="Avg Generation" value={fmtMs(m.avg_generation_ms)} />
              <MetricRow
                label="Success Rate"
                value={`${m.success_rate_pct.toFixed(1)}%`}
                highlight={i === bestSuccess ? 'best' : undefined}
              />
              <MetricRow label="Error Rate" value={`${m.error_rate_pct.toFixed(1)}%`} />
              <MetricRow
                label="Avg Sources"
                value={m.avg_sources ? m.avg_sources.toFixed(1) : '—'}
                highlight={i === bestSources ? 'best' : undefined}
              />
              {m.avg_tokens && (
                <MetricRow label="Avg Tokens" value={Math.round(m.avg_tokens).toLocaleString()} />
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};
