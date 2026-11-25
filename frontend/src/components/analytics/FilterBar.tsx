import React from 'react';
import type { ModelStats } from '../../types/analytics';
import { Search, Download, RefreshCw } from 'lucide-react';
import { analyticsApi } from '../../services/analyticsApi';

interface FilterBarProps {
  model: string;
  onModelChange: (m: string) => void;
  status: string;
  onStatusChange: (s: string) => void;
  dateFrom: string;
  onDateFromChange: (d: string) => void;
  dateTo: string;
  onDateToChange: (d: string) => void;
  search: string;
  onSearchChange: (s: string) => void;
  groupBy: 'hour' | 'day' | 'week';
  onGroupByChange: (g: 'hour' | 'day' | 'week') => void;
  availableModels: ModelStats[];
  onRefresh: () => void;
  isRefreshing: boolean;
  lastUpdated: Date | null;
}

export const FilterBar: React.FC<FilterBarProps> = ({
  model,
  onModelChange,
  status,
  onStatusChange,
  dateFrom,
  onDateFromChange,
  dateTo,
  onDateToChange,
  search,
  onSearchChange,
  groupBy,
  onGroupByChange,
  availableModels,
  onRefresh,
  isRefreshing,
  lastUpdated,
}) => {
  const handleExportCSV = () => {
    const url = analyticsApi.getExportUrl({
      model: model || undefined,
      status: status || undefined,
      date_from: dateFrom || undefined,
      date_to: dateTo || undefined,
    });
    window.open(url, '_blank');
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4 space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        {/* Left Side Filters */}
        <div className="flex flex-wrap items-center gap-2.5 flex-1 min-w-[280px]">
          {/* Search Box */}
          <div className="relative min-w-[200px] flex-1 max-w-xs">
            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search user query text..."
              value={search}
              onChange={(e) => onSearchChange(e.target.value)}
              className="w-full pl-8 pr-3 py-1.5 text-xs bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:bg-white transition"
            />
          </div>

          {/* Model Selector */}
          <select
            value={model}
            onChange={(e) => onModelChange(e.target.value)}
            className="text-xs bg-slate-50 border border-slate-200 rounded-lg px-2.5 py-1.5 text-slate-700 focus:outline-none focus:ring-1 focus:ring-indigo-500"
          >
            <option value="">All Models</option>
            {availableModels.map((m) => (
              <option key={m.model} value={m.model}>
                {m.model}
              </option>
            ))}
          </select>

          {/* Status Filter */}
          <select
            value={status}
            onChange={(e) => onStatusChange(e.target.value)}
            className="text-xs bg-slate-50 border border-slate-200 rounded-lg px-2.5 py-1.5 text-slate-700 focus:outline-none focus:ring-1 focus:ring-indigo-500"
          >
            <option value="">All Statuses</option>
            <option value="SUCCESS">Success Only</option>
            <option value="ERROR">Errors Only</option>
            <option value="UNANSWERABLE">Unanswerable</option>
          </select>

          {/* Date Range */}
          <div className="flex items-center gap-1.5 text-xs text-slate-500">
            <input
              type="date"
              value={dateFrom}
              onChange={(e) => onDateFromChange(e.target.value)}
              className="bg-slate-50 border border-slate-200 rounded-lg px-2 py-1 text-xs text-slate-700 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            />
            <span>to</span>
            <input
              type="date"
              value={dateTo}
              onChange={(e) => onDateToChange(e.target.value)}
              className="bg-slate-50 border border-slate-200 rounded-lg px-2 py-1 text-xs text-slate-700 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            />
          </div>

          {/* Group By Granularity */}
          <div className="flex items-center bg-slate-100 rounded-lg p-0.5 border border-slate-200 text-[11px]">
            {(['hour', 'day', 'week'] as const).map((g) => (
              <button
                key={g}
                onClick={() => onGroupByChange(g)}
                className={`px-2 py-1 rounded-md font-medium capitalize transition ${
                  groupBy === g
                    ? 'bg-white text-indigo-600 shadow-sm'
                    : 'text-slate-500 hover:text-slate-700'
                }`}
              >
                {g}
              </button>
            ))}
          </div>
        </div>

        {/* Right Side Actions */}
        <div className="flex items-center gap-2">
          {lastUpdated && (
            <span className="text-[11px] text-slate-400 hidden sm:inline">
              Updated {lastUpdated.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
            </span>
          )}

          <button
            onClick={onRefresh}
            disabled={isRefreshing}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-lg transition disabled:opacity-50"
            title="Auto-refreshes every 30s"
          >
            <RefreshCw className={`w-3.5 h-3.5 text-slate-500 ${isRefreshing ? 'animate-spin' : ''}`} />
            Refresh
          </button>

          <button
            onClick={handleExportCSV}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-indigo-600 hover:bg-indigo-700 rounded-lg shadow-sm transition"
          >
            <Download className="w-3.5 h-3.5" />
            Export CSV
          </button>
        </div>
      </div>
    </div>
  );
};
