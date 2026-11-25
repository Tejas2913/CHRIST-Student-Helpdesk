import React, { useState, useEffect, useCallback } from 'react';
import { analyticsApi } from '../services/analyticsApi';
import type {
  AnalyticsSummary,
  ModelStats,
  VolumePoint,
  LatencyPoint,
  CategoryPoint,
  ErrorBreakdownPoint,
  EvalReport,
} from '../types/analytics';

import { KPICards } from '../components/analytics/KPICards';
import { ModelComparisonCards } from '../components/analytics/ModelComparisonCards';
import { FilterBar } from '../components/analytics/FilterBar';
import {
  QueriesByModelChart,
  LatencyComparisonChart,
  LatencyDistributionChart,
  QueryVolumeChart,
  LatencyTrendChart,
  SuccessFailureChart,
  CategoryDistributionChart,
  AccuracyLatencyScatter,
} from '../components/analytics/Charts';
import { QueryLogsTable } from '../components/analytics/QueryLogsTable';
import { EvalSection } from '../components/analytics/EvalSection';

export const AnalyticsDashboard: React.FC = () => {
  // Global filter state
  const [model, setModel] = useState<string>('');
  const [status, setStatus] = useState<string>('');
  const [dateFrom, setDateFrom] = useState<string>('');
  const [dateTo, setDateTo] = useState<string>('');
  const [search, setSearch] = useState<string>('');
  const [groupBy, setGroupBy] = useState<'hour' | 'day' | 'week'>('day');

  // Dashboard Data State
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null);
  const [models, setModels] = useState<ModelStats[]>([]);
  const [volume, setVolume] = useState<VolumePoint[]>([]);
  const [latencyTrend, setLatencyTrend] = useState<LatencyPoint[]>([]);
  const [categories, setCategories] = useState<CategoryPoint[]>([]);
  const [errors, setErrors] = useState<ErrorBreakdownPoint[]>([]);
  const [evalReport, setEvalReport] = useState<EvalReport | null>(null);

  // Loading & Refresh State
  const [loading, setLoading] = useState<boolean>(true);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchDashboardData = useCallback(async (isBackground = false) => {
    if (!isBackground) setLoading(true);
    setIsRefreshing(true);

    try {
      const [sumRes, modRes, volRes, latRes, catRes, errRes, evRes] = await Promise.allSettled([
        analyticsApi.getSummary(),
        analyticsApi.getModels(),
        analyticsApi.getVolume(groupBy),
        analyticsApi.getLatencyTrend(groupBy === 'week' ? 'day' : groupBy),
        analyticsApi.getCategories(),
        analyticsApi.getErrors(),
        analyticsApi.getEval(),
      ]);

      if (sumRes.status === 'fulfilled') setSummary(sumRes.value);
      if (modRes.status === 'fulfilled') setModels(modRes.value);
      if (volRes.status === 'fulfilled') setVolume(volRes.value);
      if (latRes.status === 'fulfilled') setLatencyTrend(latRes.value);
      if (catRes.status === 'fulfilled') setCategories(catRes.value);
      if (errRes.status === 'fulfilled') setErrors(errRes.value);
      if (evRes.status === 'fulfilled') setEvalReport(evRes.value);

      setLastUpdated(new Date());
    } catch (e) {
      console.error('Failed fetching analytics dashboard data:', e);
    } finally {
      setLoading(false);
      setIsRefreshing(false);
    }
  }, [groupBy]);

  // Initial load and whenever groupBy changes
  useEffect(() => {
    fetchDashboardData();
  }, [fetchDashboardData]);

  // Auto-refresh interval (30 seconds)
  useEffect(() => {
    const timer = setInterval(() => {
      fetchDashboardData(true);
    }, 30000);
    return () => clearInterval(timer);
  }, [fetchDashboardData]);

  return (
    <div className="flex-1 overflow-y-auto bg-slate-50 p-4 sm:p-6 space-y-6">
      {/* Top Banner & Header info */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-200 pb-4">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-slate-800 tracking-tight flex items-center gap-2">
            RAG Performance & Analytics Dashboard
          </h1>
          <p className="text-xs sm:text-sm text-slate-500 mt-1">
            Real-time LLMOps metrics, model latency comparisons, and retrieval diagnostics for CHRIST Student Helpdesk.
          </p>
        </div>
      </div>

      {/* Global Filter Bar */}
      <FilterBar
        model={model}
        onModelChange={setModel}
        status={status}
        onStatusChange={setStatus}
        dateFrom={dateFrom}
        onDateFromChange={setDateFrom}
        dateTo={dateTo}
        onDateToChange={setDateTo}
        search={search}
        onSearchChange={setSearch}
        groupBy={groupBy}
        onGroupByChange={setGroupBy}
        availableModels={models}
        onRefresh={() => fetchDashboardData(false)}
        isRefreshing={isRefreshing}
        lastUpdated={lastUpdated}
      />

      {/* Top-Level KPI Cards */}
      <section>
        <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2.5">
          High-Level Metrics
        </h2>
        <KPICards summary={summary} loading={loading} />
      </section>

      {/* Side-by-Side Model Comparison Cards */}
      <section>
        <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2.5">
          Model Comparison (Side-by-Side)
        </h2>
        <ModelComparisonCards models={models} loading={loading} />
      </section>

      {/* Performance & Quality Charts Grid */}
      <section className="space-y-4">
        <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
          Performance & Reliability Analytics
        </h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <QueriesByModelChart models={models} />
          <LatencyComparisonChart models={models} />
          <LatencyDistributionChart models={models} />
          <QueryVolumeChart data={volume} />
          <LatencyTrendChart data={latencyTrend} />
          <SuccessFailureChart errors={errors} />
          <CategoryDistributionChart data={categories} />
          <AccuracyLatencyScatter models={models} />
        </div>
      </section>

      {/* Retrieval Benchmark Evaluation Report */}
      <section>
        <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2.5">
          Benchmark & Ground Truth Evaluation
        </h2>
        <EvalSection report={evalReport} loading={loading} />
      </section>

      {/* Query Logs Data Table */}
      <section>
        <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2.5">
          Detailed Query Logs
        </h2>
        <QueryLogsTable
          model={model}
          status={status}
          dateFrom={dateFrom}
          dateTo={dateTo}
          search={search}
        />
      </section>
    </div>
  );
};
export default AnalyticsDashboard;
