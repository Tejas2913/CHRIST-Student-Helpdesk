import React, { useState } from 'react';
import type { QueryLogEntry } from '../../types/analytics';
import { X, Clock, Database, Zap, FileText, ChevronDown, ChevronUp } from 'lucide-react';

interface QueryDetailModalProps {
  query: QueryLogEntry | null;
  onClose: () => void;
}

const fmtMs = (ms: number) => ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`;

const statusColor: Record<string, string> = {
  SUCCESS: 'bg-emerald-100 text-emerald-700',
  ERROR: 'bg-rose-100 text-rose-600',
  UNANSWERABLE: 'bg-amber-100 text-amber-700',
};

const confidenceColor: Record<string, string> = {
  high: 'text-emerald-600',
  medium: 'text-amber-600',
  low: 'text-rose-500',
  unanswerable: 'text-slate-400',
};

export const QueryDetailModal: React.FC<QueryDetailModalProps> = ({ query: q, onClose }) => {
  const [showChunks, setShowChunks] = useState(false);

  if (!q) return null;

  const timingTotal = (q.retrieval_latency_ms || 0) + (q.generation_latency_ms || 0);
  const retPct = timingTotal > 0 ? (q.retrieval_latency_ms / timingTotal) * 100 : 50;
  const genPct = timingTotal > 0 ? (q.generation_latency_ms / timingTotal) * 100 : 50;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-sm">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl max-h-[90vh] flex flex-col border border-slate-200">
        {/* Header */}
        <div className="flex items-start justify-between px-5 py-4 border-b border-slate-100">
          <div>
            <h2 className="text-base font-bold text-slate-800">Query Detail</h2>
            <p className="text-xs text-slate-400 mt-0.5 font-mono">{q.id}</p>
          </div>
          <button onClick={onClose} className="p-1.5 hover:bg-slate-100 rounded-lg transition">
            <X className="w-4 h-4 text-slate-500" />
          </button>
        </div>

        <div className="overflow-y-auto flex-1 px-5 py-4 space-y-4">
          {/* Query */}
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-400 font-medium mb-1">Query</p>
            <p className="text-sm text-slate-800 font-medium leading-relaxed bg-slate-50 rounded-lg px-3 py-2">
              {q.query}
            </p>
          </div>

          {/* Meta row */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            <div className="bg-slate-50 rounded-lg px-3 py-2">
              <p className="text-[10px] text-slate-400 uppercase">Model</p>
              <p className="text-xs font-semibold text-slate-700 break-all">{q.model}</p>
            </div>
            <div className="bg-slate-50 rounded-lg px-3 py-2">
              <p className="text-[10px] text-slate-400 uppercase">Status</p>
              <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${statusColor[q.status] || 'bg-slate-100 text-slate-600'}`}>
                {q.status}
              </span>
            </div>
            <div className="bg-slate-50 rounded-lg px-3 py-2">
              <p className="text-[10px] text-slate-400 uppercase">Type</p>
              <p className="text-xs font-semibold text-slate-700">{q.query_type || '—'}</p>
            </div>
            <div className="bg-slate-50 rounded-lg px-3 py-2">
              <p className="text-[10px] text-slate-400 uppercase">Confidence</p>
              <p className={`text-xs font-semibold capitalize ${confidenceColor[q.confidence || ''] || 'text-slate-700'}`}>
                {q.confidence || '—'}
              </p>
            </div>
            <div className="bg-slate-50 rounded-lg px-3 py-2">
              <p className="text-[10px] text-slate-400 uppercase">Tokens</p>
              <p className="text-xs font-semibold text-slate-700">{q.token_count ?? '—'}</p>
            </div>
            <div className="bg-slate-50 rounded-lg px-3 py-2">
              <p className="text-[10px] text-slate-400 uppercase">Timestamp</p>
              <p className="text-xs text-slate-600">{new Date(q.timestamp).toLocaleString()}</p>
            </div>
          </div>

          {/* Timing breakdown */}
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-400 font-medium mb-2">Timing Breakdown</p>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Clock className="w-3.5 h-3.5 text-slate-400 flex-shrink-0" />
                <span className="text-xs text-slate-500 w-28">Total End-to-End</span>
                <div className="flex-1 bg-slate-100 rounded-full h-2">
                  <div className="bg-indigo-500 h-2 rounded-full" style={{ width: '100%' }} />
                </div>
                <span className="text-xs font-mono font-semibold text-slate-700 w-16 text-right">
                  {fmtMs(q.total_latency_ms)}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <Database className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
                <span className="text-xs text-slate-500 w-28">Retrieval</span>
                <div className="flex-1 bg-slate-100 rounded-full h-2">
                  <div className="bg-cyan-400 h-2 rounded-full" style={{ width: `${retPct}%` }} />
                </div>
                <span className="text-xs font-mono font-semibold text-slate-700 w-16 text-right">
                  {fmtMs(q.retrieval_latency_ms)}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <Zap className="w-3.5 h-3.5 text-amber-400 flex-shrink-0" />
                <span className="text-xs text-slate-500 w-28">Generation</span>
                <div className="flex-1 bg-slate-100 rounded-full h-2">
                  <div className="bg-amber-400 h-2 rounded-full" style={{ width: `${genPct}%` }} />
                </div>
                <span className="text-xs font-mono font-semibold text-slate-700 w-16 text-right">
                  {fmtMs(q.generation_latency_ms)}
                </span>
              </div>
            </div>
          </div>

          {/* Answer preview */}
          {q.answer_preview && (
            <div>
              <p className="text-xs uppercase tracking-wide text-slate-400 font-medium mb-1">Answer Preview</p>
              <p className="text-xs text-slate-700 leading-relaxed bg-slate-50 rounded-lg px-3 py-2">
                {q.answer_preview}
                {q.answer_preview.length >= 200 && <span className="text-slate-400">...</span>}
              </p>
            </div>
          )}

          {/* Sources */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <p className="text-xs uppercase tracking-wide text-slate-400 font-medium">
                Retrieved Sources ({q.sources_retrieved})
              </p>
              {q.retrieved_chunk_ids?.length > 0 && (
                <button
                  onClick={() => setShowChunks(v => !v)}
                  className="flex items-center gap-1 text-[10px] text-indigo-600 hover:text-indigo-800 transition"
                >
                  <FileText className="w-3 h-3" />
                  {showChunks ? 'Hide' : 'Show'} chunk IDs
                  {showChunks ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                </button>
              )}
            </div>
            {q.source_pages?.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {q.source_pages.map((p, i) => (
                  <span key={i} className="text-[10px] bg-indigo-50 text-indigo-700 border border-indigo-100 rounded px-1.5 py-0.5 font-medium">
                    p.{p}
                  </span>
                ))}
              </div>
            )}
            {showChunks && q.retrieved_chunk_ids?.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1">
                {q.retrieved_chunk_ids.map((id, i) => (
                  <span key={i} className="text-[10px] bg-slate-100 text-slate-500 rounded px-1.5 py-0.5 font-mono">
                    {id}
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Error */}
          {q.error && (
            <div className="bg-rose-50 border border-rose-100 rounded-lg px-3 py-2">
              <p className="text-xs font-semibold text-rose-600 mb-1">Error</p>
              <p className="text-xs text-rose-500 font-mono break-all">{q.error}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
