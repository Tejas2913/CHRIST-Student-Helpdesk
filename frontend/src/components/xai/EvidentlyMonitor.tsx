import { useState } from 'react';
import type { EvidentlyReport } from '../../types/xai';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, PieChart, Pie, Legend } from 'recharts';

interface Props {
  report: EvidentlyReport | null;
  driftReport: EvidentlyReport | null;
  loading: boolean;
  onGenerate: (type: 'rag_quality' | 'drift') => void;
}

const COLORS = ['#7c3aed', '#059669', '#f59e0b', '#dc2626', '#3b82f6', '#ec4899'];

export default function EvidentlyMonitor({ report, driftReport, loading, onGenerate }: Props) {
  const [activeTab, setActiveTab] = useState<'quality' | 'drift'>('quality');

  const m = report?.metrics;
  const hasValidMetrics = Boolean(m && typeof m.total_queries === 'number' && m.total_queries > 0 && m.latency);
  const drift = driftReport?.drift || driftReport?.drift_summary;

  const queryTypeData = m?.query_types
    ? Object.entries(m.query_types)
        .filter(([k, v]) => Boolean(k) && typeof v === 'number' && v > 0)
        .map(([k, v]) => ({ name: k, value: v }))
    : [];

  const latencyData = m?.latency
    ? [
        { name: 'Mean', value: Math.round(m.latency.mean_ms || 0) },
        { name: 'P50', value: Math.round(m.latency.p50_ms || 0) },
        { name: 'P95', value: Math.round(m.latency.p95_ms || 0) },
        { name: 'P99', value: Math.round(m.latency.p99_ms || 0) },
      ]
    : [];

  const driftTypeData = (drift?.query_type_distribution_drift || []).slice(0, 8).map(d => ({
    name: d.query_type,
    reference: parseFloat((d.reference_pct * 100).toFixed(1)),
    current: parseFloat((d.current_pct * 100).toFixed(1)),
    drift: d.drift,
  }));

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h3 className="text-base font-semibold text-gray-900">Evidently AI Monitoring</h3>
          <p className="text-xs text-gray-500 mt-0.5">RAG quality metrics, latency analysis, and query distribution drift detection</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => onGenerate('rag_quality')}
            disabled={loading}
            className="px-3 py-1.5 bg-purple-700 text-white text-xs rounded-lg hover:bg-purple-800 disabled:opacity-50 transition-colors"
          >
            {loading ? 'Generating…' : '↻ Quality Report'}
          </button>
          <button
            onClick={() => onGenerate('drift')}
            disabled={loading}
            className="px-3 py-1.5 bg-indigo-600 text-white text-xs rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition-colors"
          >
            {loading ? '…' : '↻ Drift Report'}
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200">
        {[{ id: 'quality', label: 'Quality Report' }, { id: 'drift', label: 'Drift Detection' }].map(t => (
          <button
            key={t.id}
            onClick={() => setActiveTab(t.id as any)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === t.id ? 'border-purple-600 text-purple-700' : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Quality Report */}
      {activeTab === 'quality' && (
        <div className="space-y-4">
          {!report && !loading && (
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-6 text-center text-gray-500 text-sm">
              Click "Quality Report" to generate a fresh Evidently report from query logs.
            </div>
          )}

          {loading && (
            <div className="flex items-center justify-center h-32 text-gray-500">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-700 mr-3" />
              Generating report…
            </div>
          )}

          {report?.error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700 text-sm">
              {report.error}
            </div>
          )}

          {hasValidMetrics && m && (
            <>
              {/* KPI Row */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {[
                  { label: 'Total Queries', value: m.total_queries ?? 0, color: 'text-gray-800' },
                  { label: 'Success Rate', value: `${((m.success_rate ?? 0) * 100).toFixed(1)}%`, color: 'text-green-700' },
                  { label: 'Avg Latency', value: `${(m.latency?.mean_ms ?? 0).toFixed(0)} ms`, color: 'text-blue-700' },
                  { label: 'P95 Latency', value: `${(m.latency?.p95_ms ?? 0).toFixed(0)} ms`, color: 'text-orange-700' },
                ].map(kpi => (
                  <div key={kpi.label} className="bg-white border border-gray-200 rounded-lg p-3 text-center shadow-sm">
                    <div className={`text-xl font-bold ${kpi.color}`}>{kpi.value}</div>
                    <div className="text-xs text-gray-500 mt-0.5">{kpi.label}</div>
                  </div>
                ))}
              </div>

              {/* Latency chart */}
              <div>
                <h4 className="text-sm font-medium text-gray-700 mb-2">Latency Distribution</h4>
                <ResponsiveContainer width="100%" height={160}>
                  <BarChart data={latencyData} margin={{ left: 0, right: 10 }}>
                    <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip formatter={(v: any) => [`${Number(v || 0).toFixed(0)} ms`, 'Latency']} contentStyle={{ fontSize: 12 }} />
                    <Bar dataKey="value" fill="#7c3aed" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Query type pie */}
              {queryTypeData.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Query Type Distribution</h4>
                  <ResponsiveContainer width="100%" height={200}>
                    <PieChart>
                      <Pie
                        data={queryTypeData}
                        dataKey="value"
                        nameKey="name"
                        cx="40%"
                        cy="50%"
                        outerRadius={70}
                        label={({ name, percent }: any) =>
                          name && typeof percent === 'number' ? `${name} ${(percent * 100).toFixed(0)}%` : (name || '')
                        }
                        labelLine={false}
                      >
                        {queryTypeData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                      </Pie>
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* Report metadata */}
              <div className="text-xs text-gray-400">
                Report generated: {new Date(report.generated_at).toLocaleString()} ·
                {report.n_records ?? m.total_queries} records · Runtime: {report.runtime_ms?.toFixed(0)} ms
              </div>
            </>
          )}
        </div>
      )}

      {/* Drift Detection */}
      {activeTab === 'drift' && (
        <div className="space-y-4">
          {!driftReport && !loading && (
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-6 text-center text-gray-500 text-sm">
              Click "Drift Report" to compare recent vs older query distributions.
            </div>
          )}

          {loading && (
            <div className="flex items-center justify-center h-32 text-gray-500">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600 mr-3" />
              Computing drift…
            </div>
          )}

          {driftReport && !drift?.available && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-yellow-800 text-sm">
              {driftReport.error || 'Drift analysis unavailable. Need more query history.'}
            </div>
          )}

          {drift?.available && (
            <>
              {/* Overall drift badge */}
              <div className={`flex items-center gap-3 p-3 rounded-lg border ${
                drift.overall_drift_detected
                  ? 'bg-red-50 border-red-300'
                  : 'bg-green-50 border-green-300'
              }`}>
                <span className="text-2xl">{drift.overall_drift_detected ? '⚠️' : '✅'}</span>
                <div>
                  <div className={`font-semibold text-sm ${drift.overall_drift_detected ? 'text-red-800' : 'text-green-800'}`}>
                    {drift.overall_drift_detected ? 'Distribution Drift Detected' : 'No Significant Drift'}
                  </div>
                  {drift.latency_drift && (
                    <div className="text-xs text-gray-600 mt-0.5">
                      Latency KS stat: {drift.latency_drift.ks_statistic?.toFixed(3)} · p={drift.latency_drift.p_value?.toFixed(3)}
                    </div>
                  )}
                </div>
              </div>

              {/* Query type drift chart */}
              {driftTypeData.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Query Type Shift (reference vs current)</h4>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={driftTypeData} layout="vertical" margin={{ left: 10, right: 20 }}>
                      <XAxis type="number" tick={{ fontSize: 11 }} unit="%" />
                      <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={120} />
                      <Tooltip formatter={(v: any) => [`${v}%`, '']} contentStyle={{ fontSize: 12 }} />
                      <Bar dataKey="reference" fill="#9ca3af" name="Reference %" radius={[0, 3, 3, 0]} />
                      <Bar dataKey="current" name="Current %" radius={[0, 3, 3, 0]}>
                        {driftTypeData.map((d, i) => (
                          <Cell key={i} fill={d.drift ? '#dc2626' : '#059669'} />
                        ))}
                      </Bar>
                      <Legend />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
