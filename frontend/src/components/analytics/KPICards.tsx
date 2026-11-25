import React from 'react';
import type { AnalyticsSummary } from '../../types/analytics';
import {
  MessageSquare, Calendar, CheckCircle, Clock, Zap, AlertTriangle, Database, Trophy
} from 'lucide-react';

interface KPICardsProps {
  summary: AnalyticsSummary | null;
  loading: boolean;
}

const fmt = (n: number, dec = 0) =>
  n === undefined || n === null ? '—' : n.toLocaleString('en-IN', { maximumFractionDigits: dec });

const fmtMs = (ms: number) =>
  ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`;

interface CardProps {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub?: string;
  accent: string;
  loading?: boolean;
}

const Card: React.FC<CardProps> = ({ icon, label, value, sub, accent, loading }) => (
  <div className={`bg-white rounded-xl border border-slate-200 shadow-sm p-4 flex items-start gap-3 hover:shadow-md transition-shadow`}>
    <div className={`p-2.5 rounded-lg ${accent} flex-shrink-0`}>
      {icon}
    </div>
    <div className="min-w-0">
      <p className="text-xs text-slate-500 font-medium uppercase tracking-wide">{label}</p>
      {loading ? (
        <div className="h-6 w-20 bg-slate-100 rounded animate-pulse mt-1" />
      ) : (
        <p className="text-2xl font-bold text-slate-800 leading-tight">{value}</p>
      )}
      {sub && <p className="text-xs text-slate-400 mt-0.5">{sub}</p>}
    </div>
  </div>
);

export const KPICards: React.FC<KPICardsProps> = ({ summary: s, loading }) => {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-2 lg:grid-cols-4 gap-3">
      <Card
        icon={<MessageSquare className="w-5 h-5 text-indigo-600" />}
        label="Total Queries"
        value={s ? fmt(s.total_queries) : '—'}
        sub="All time"
        accent="bg-indigo-50"
        loading={loading}
      />
      <Card
        icon={<Calendar className="w-5 h-5 text-violet-600" />}
        label="Queries Today"
        value={s ? fmt(s.queries_today) : '—'}
        sub="Since midnight UTC"
        accent="bg-violet-50"
        loading={loading}
      />
      <Card
        icon={<CheckCircle className="w-5 h-5 text-emerald-600" />}
        label="Successful"
        value={s ? fmt(s.total_responses) : '—'}
        sub={s ? `${(100 - s.error_rate_pct).toFixed(1)}% success rate` : ''}
        accent="bg-emerald-50"
        loading={loading}
      />
      <Card
        icon={<AlertTriangle className="w-5 h-5 text-rose-500" />}
        label="Error Rate"
        value={s ? `${s.error_rate_pct.toFixed(1)}%` : '—'}
        sub="Failed requests"
        accent="bg-rose-50"
        loading={loading}
      />
      <Card
        icon={<Clock className="w-5 h-5 text-amber-600" />}
        label="Avg Latency"
        value={s ? fmtMs(s.avg_latency_ms) : '—'}
        sub="End-to-end"
        accent="bg-amber-50"
        loading={loading}
      />
      <Card
        icon={<Database className="w-5 h-5 text-cyan-600" />}
        label="Avg Retrieval"
        value={s ? fmtMs(s.avg_retrieval_ms) : '—'}
        sub="Vector + rerank"
        accent="bg-cyan-50"
        loading={loading}
      />
      <Card
        icon={<Zap className="w-5 h-5 text-orange-500" />}
        label="Avg Generation"
        value={s ? fmtMs(s.avg_generation_ms) : '—'}
        sub="LLM inference"
        accent="bg-orange-50"
        loading={loading}
      />
      <Card
        icon={<Trophy className="w-5 h-5 text-yellow-600" />}
        label="Best Model"
        value={s?.best_model ? s.best_model.split('/').pop() || s.best_model : '—'}
        sub="Lowest avg latency"
        accent="bg-yellow-50"
        loading={loading}
      />
    </div>
  );
};
