import React from 'react';
import type { EvalReport } from '../../types/analytics';

interface EvalSectionProps {
  report: EvalReport | null;
  loading: boolean;
}

const fmtPct = (n: number | undefined) => n !== undefined ? `${(n * 100).toFixed(1)}%` : '—';
const fmtMs = (ms: number | undefined) => ms !== undefined ? `${Math.round(ms)}ms` : '—';

export const EvalSection: React.FC<EvalSectionProps> = ({ report: r, loading }) => {
  if (loading) {
    return (
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 animate-pulse">
        <div className="h-4 bg-slate-100 rounded w-1/4 mb-4" />
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[0,1,2,3].map(i => <div key={i} className="h-14 bg-slate-100 rounded-lg" />)}
        </div>
      </div>
    );
  }

  if (!r || r.error) {
    return (
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
        <h3 className="text-sm font-semibold text-slate-700 mb-2">Retrieval Evaluation Report</h3>
        <p className="text-xs text-slate-400">
          {r?.error || 'No evaluation report available. Run the benchmark: '}
          {!r?.error && <code className="bg-slate-100 px-1 rounded text-slate-600">cd backend && python eval/run_eval.py</code>}
        </p>
      </div>
    );
  }

  const metrics = [
    { label: 'Hit Rate @1', value: fmtPct(r.hit_rate_at_1 as number) },
    { label: 'Hit Rate @5', value: fmtPct(r.hit_rate_at_5 as number) },
    { label: 'MRR', value: fmtPct(r.mrr as number) },
    { label: 'Avg Retrieval', value: fmtMs(r.avg_retrieval_ms as number) },
    { label: 'Total Queries', value: r.total_queries ?? '—' },
  ];

  // Pick any extra keys
  const extraKeys = Object.keys(r).filter(
    k => !['timestamp','total_queries','hit_rate_at_1','hit_rate_at_5','mrr','avg_retrieval_ms','error'].includes(k)
  );

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-slate-700">Retrieval Evaluation Report</h3>
        {r.timestamp && (
          <p className="text-[10px] text-slate-400">
            Generated: {new Date(r.timestamp as string).toLocaleString()}
          </p>
        )}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
        {metrics.map(m => (
          <div key={m.label} className="bg-slate-50 rounded-lg px-3 py-3 text-center">
            <p className="text-lg font-bold text-slate-800">{String(m.value)}</p>
            <p className="text-[10px] text-slate-400 mt-0.5">{m.label}</p>
          </div>
        ))}
      </div>

      {extraKeys.length > 0 && (
        <div className="mt-3 pt-3 border-t border-slate-100">
          <p className="text-[10px] text-slate-400 uppercase tracking-wide font-medium mb-2">Additional Metrics</p>
          <div className="flex flex-wrap gap-2">
            {extraKeys.map(k => (
              <span key={k} className="text-xs bg-slate-100 text-slate-600 rounded px-2 py-1">
                <span className="font-medium">{k}:</span>{' '}
                {typeof r[k] === 'number' ? (r[k] as number).toFixed(3) : String(r[k])}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
