import React, { useState, useEffect, useCallback } from 'react';
import { analyticsApi } from '../../services/analyticsApi';
import type { QueryLogEntry, PaginatedQueryLogs } from '../../types/analytics';
import { QueryDetailModal } from './QueryDetailModal';
import { ChevronLeft, ChevronRight, ExternalLink, ArrowUpDown } from 'lucide-react';

interface QueryLogsTableProps {
  model: string;
  status: string;
  dateFrom: string;
  dateTo: string;
  search: string;
}

const fmtMs = (ms: number) => ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`;

const statusBadge: Record<string, string> = {
  SUCCESS: 'bg-emerald-100 text-emerald-700',
  ERROR: 'bg-rose-100 text-rose-600',
  UNANSWERABLE: 'bg-amber-100 text-amber-700',
};

const confidenceBadge: Record<string, string> = {
  high: 'bg-emerald-50 text-emerald-600',
  medium: 'bg-amber-50 text-amber-600',
  low: 'bg-rose-50 text-rose-500',
  unanswerable: 'bg-slate-50 text-slate-400',
};

export const QueryLogsTable: React.FC<QueryLogsTableProps> = ({
  model, status, dateFrom, dateTo, search,
}) => {
  const [page, setPage] = useState(1);
  const [sortBy, setSortBy] = useState<string>('timestamp');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');
  const [data, setData] = useState<PaginatedQueryLogs | null>(null);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<QueryLogEntry | null>(null);

  const LIMIT = 20;

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await analyticsApi.getQueries({
        page, limit: LIMIT,
        model: model || undefined,
        status: status || undefined,
        date_from: dateFrom || undefined,
        date_to: dateTo || undefined,
        search: search || undefined,
        sort_by: sortBy,
        sort_dir: sortDir,
      });
      setData(res);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, [page, model, status, dateFrom, dateTo, search, sortBy, sortDir]);

  useEffect(() => { setPage(1); }, [model, status, dateFrom, dateTo, search]);
  useEffect(() => { load(); }, [load]);

  const toggleSort = (col: string) => {
    if (sortBy === col) setSortDir(d => d === 'desc' ? 'asc' : 'desc');
    else { setSortBy(col); setSortDir('desc'); }
  };

  const SortBtn: React.FC<{ col: string }> = ({ col }) => (
    <button onClick={() => toggleSort(col)} className="inline-flex items-center gap-0.5 hover:text-indigo-600 transition">
      <ArrowUpDown className={`w-3 h-3 ${sortBy === col ? 'text-indigo-600' : 'text-slate-300'}`} />
    </button>
  );

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
      <div className="px-4 py-3 border-b border-slate-100 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-700">
          Query Logs
          {data && <span className="ml-2 text-xs text-slate-400 font-normal">({data.total} total)</span>}
        </h3>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="bg-slate-50 border-b border-slate-100">
              <th className="text-left px-3 py-2.5 text-slate-500 font-semibold w-36">
                Timestamp <SortBtn col="timestamp" />
              </th>
              <th className="text-left px-3 py-2.5 text-slate-500 font-semibold">Query</th>
              <th className="text-left px-3 py-2.5 text-slate-500 font-semibold w-28">Model</th>
              <th className="text-left px-3 py-2.5 text-slate-500 font-semibold w-20">Type</th>
              <th className="text-right px-3 py-2.5 text-slate-500 font-semibold w-20">
                Latency <SortBtn col="total_latency_ms" />
              </th>
              <th className="text-right px-3 py-2.5 text-slate-500 font-semibold w-16">Sources</th>
              <th className="text-left px-3 py-2.5 text-slate-500 font-semibold w-20">Status</th>
              <th className="text-left px-3 py-2.5 text-slate-500 font-semibold w-20">Confidence</th>
              <th className="w-8"></th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr>
                <td colSpan={9} className="text-center py-8 text-slate-400">Loading…</td>
              </tr>
            )}
            {!loading && data?.data.length === 0 && (
              <tr>
                <td colSpan={9} className="text-center py-8 text-slate-400">No queries match the current filters.</td>
              </tr>
            )}
            {!loading && data?.data.map(row => (
              <tr
                key={row.id}
                className="border-b border-slate-50 hover:bg-indigo-50/40 cursor-pointer transition"
                onClick={() => setSelected(row)}
              >
                <td className="px-3 py-2 text-slate-400 font-mono whitespace-nowrap">
                  {new Date(row.timestamp).toLocaleString('en-IN', {
                    month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit'
                  })}
                </td>
                <td className="px-3 py-2 text-slate-700 max-w-[220px]">
                  <span className="line-clamp-2">{row.query}</span>
                </td>
                <td className="px-3 py-2 text-slate-500 font-mono text-[10px] break-all">
                  {row.model?.split('/').pop()}
                </td>
                <td className="px-3 py-2">
                  {row.query_type && (
                    <span className="text-[10px] bg-slate-100 text-slate-600 rounded px-1.5 py-0.5">{row.query_type}</span>
                  )}
                </td>
                <td className="px-3 py-2 text-right font-mono text-slate-700">{fmtMs(row.total_latency_ms)}</td>
                <td className="px-3 py-2 text-right text-slate-500">{row.sources_retrieved}</td>
                <td className="px-3 py-2">
                  <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded-full ${statusBadge[row.status] || 'bg-slate-100 text-slate-600'}`}>
                    {row.status}
                  </span>
                </td>
                <td className="px-3 py-2">
                  {row.confidence && (
                    <span className={`text-[10px] px-1.5 py-0.5 rounded capitalize ${confidenceBadge[row.confidence] || 'bg-slate-50 text-slate-500'}`}>
                      {row.confidence}
                    </span>
                  )}
                </td>
                <td className="px-3 py-2">
                  <ExternalLink className="w-3 h-3 text-slate-300" />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {data && data.pages > 1 && (
        <div className="flex items-center justify-between px-4 py-2.5 border-t border-slate-100">
          <span className="text-xs text-slate-400">
            Page {data.page} of {data.pages}
          </span>
          <div className="flex gap-1">
            <button
              disabled={page === 1}
              onClick={() => setPage(p => p - 1)}
              className="p-1 rounded hover:bg-slate-100 disabled:opacity-30 transition"
            >
              <ChevronLeft className="w-4 h-4 text-slate-500" />
            </button>
            <button
              disabled={page >= data.pages}
              onClick={() => setPage(p => p + 1)}
              className="p-1 rounded hover:bg-slate-100 disabled:opacity-30 transition"
            >
              <ChevronRight className="w-4 h-4 text-slate-500" />
            </button>
          </div>
        </div>
      )}

      <QueryDetailModal query={selected} onClose={() => setSelected(null)} />
    </div>
  );
};
