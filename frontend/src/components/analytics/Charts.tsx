import React from 'react';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, ScatterChart,
  Scatter
} from 'recharts';
import type { VolumePoint, LatencyPoint, CategoryPoint, ErrorBreakdownPoint, ModelStats } from '../../types/analytics';

// Shared colour palette
export const MODEL_PALETTE: Record<string, string> = {
  'qwen/qwen3.8-27b':    '#6366f1',
  'openai/gpt-oss-120b': '#8b5cf6',
  'openai/gpt-oss-20b':  '#0ea5e9',
};
const FALLBACK_COLORS = ['#f59e0b', '#10b981', '#ef4444', '#ec4899', '#14b8a6'];
function colorFor(model: string, idx = 0) {
  return MODEL_PALETTE[model] || FALLBACK_COLORS[idx % FALLBACK_COLORS.length];
}

const ChartCard: React.FC<{ title: string; children: React.ReactNode; className?: string }> = ({
  title, children, className = ''
}) => (
  <div className={`bg-white rounded-xl border border-slate-200 shadow-sm p-4 ${className}`}>
    <h3 className="text-sm font-semibold text-slate-700 mb-3">{title}</h3>
    {children}
  </div>
);

// ── Chart 1: Queries by Model ───────────────────────────────────────────────
export const QueriesByModelChart: React.FC<{ models: ModelStats[] }> = ({ models }) => (
  <ChartCard title="Queries Processed by Model">
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={models} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
        <XAxis dataKey="model" tick={{ fontSize: 10 }} tickFormatter={v => v.split('/').pop()} />
        <YAxis tick={{ fontSize: 10 }} />
        <Tooltip formatter={(v: any) => [v ?? 0, 'Queries']} />
        <Bar dataKey="total_queries" name="Total Queries" radius={[4, 4, 0, 0]}>
          {models.map((m, i) => (
            <Cell key={m.model} fill={colorFor(m.model, i)} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  </ChartCard>
);

// ── Chart 2: Latency Comparison ─────────────────────────────────────────────
export const LatencyComparisonChart: React.FC<{ models: ModelStats[] }> = ({ models }) => {
  const data = models.map(m => ({
    model: m.model.split('/').pop(),
    'Avg Total': Math.round(m.avg_latency_ms),
    'Avg Retrieval': Math.round(m.avg_retrieval_ms),
    'Avg Generation': Math.round(m.avg_generation_ms),
  }));
  return (
    <ChartCard title="Latency Breakdown by Model (ms)">
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis dataKey="model" tick={{ fontSize: 10 }} />
          <YAxis tick={{ fontSize: 10 }} />
          <Tooltip formatter={(v: any) => [`${v ?? 0}ms`, '']} />
          <Legend iconSize={10} wrapperStyle={{ fontSize: '11px' }} />
          <Bar dataKey="Avg Total" fill="#6366f1" radius={[3, 3, 0, 0]} />
          <Bar dataKey="Avg Retrieval" fill="#0ea5e9" radius={[3, 3, 0, 0]} />
          <Bar dataKey="Avg Generation" fill="#f59e0b" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
};

// ── Chart 3: P50/P95 Latency ─────────────────────────────────────────────────
export const LatencyDistributionChart: React.FC<{ models: ModelStats[] }> = ({ models }) => {
  const data = models.map(m => ({
    model: m.model.split('/').pop(),
    P50: Math.round(m.p50_latency_ms),
    P95: Math.round(m.p95_latency_ms),
    Avg: Math.round(m.avg_latency_ms),
  }));
  return (
    <ChartCard title="Latency Percentiles (ms) — P50 / P95 / Avg">
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis dataKey="model" tick={{ fontSize: 10 }} />
          <YAxis tick={{ fontSize: 10 }} />
          <Tooltip formatter={(v: any) => [`${v ?? 0}ms`, '']} />
          <Legend iconSize={10} wrapperStyle={{ fontSize: '11px' }} />
          <Bar dataKey="P50" fill="#10b981" radius={[3, 3, 0, 0]} />
          <Bar dataKey="Avg" fill="#6366f1" radius={[3, 3, 0, 0]} />
          <Bar dataKey="P95" fill="#f43f5e" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
};

// ── Chart 4: Query Volume Over Time ──────────────────────────────────────────
export const QueryVolumeChart: React.FC<{ data: VolumePoint[] }> = ({ data }) => {
  const models = [...new Set(data.map(d => d.model))];
  const periods = [...new Set(data.map(d => d.period))].sort();
  const pivoted = periods.map(p => {
    const row: Record<string, string | number> = { period: p };
    models.forEach(m => {
      const found = data.find(d => d.period === p && d.model === m);
      row[m.split('/').pop() || m] = found ? found.count : 0;
    });
    return row;
  });

  return (
    <ChartCard title="Query Volume Over Time">
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={pivoted} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis dataKey="period" tick={{ fontSize: 9 }} />
          <YAxis tick={{ fontSize: 10 }} allowDecimals={false} />
          <Tooltip />
          <Legend iconSize={10} wrapperStyle={{ fontSize: '11px' }} />
          {models.map((m, i) => (
            <Bar key={m} dataKey={m.split('/').pop() || m} stackId="a" fill={colorFor(m, i)} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
};

// ── Chart 5: Latency Trend ───────────────────────────────────────────────────
export const LatencyTrendChart: React.FC<{ data: LatencyPoint[] }> = ({ data }) => {
  const models = [...new Set(data.map(d => d.model))];
  const periods = [...new Set(data.map(d => d.period))].sort();
  const pivoted = periods.map(p => {
    const row: Record<string, string | number> = { period: p };
    models.forEach(m => {
      const found = data.find(d => d.period === p && d.model === m);
      row[m.split('/').pop() || m] = found ? Math.round(found.avg_latency_ms) : 0;
    });
    return row;
  });
  return (
    <ChartCard title="Latency Trend Over Time (ms)">
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={pivoted} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis dataKey="period" tick={{ fontSize: 9 }} />
          <YAxis tick={{ fontSize: 10 }} />
          <Tooltip formatter={(v: any) => [`${v ?? 0}ms`, '']} />
          <Legend iconSize={10} wrapperStyle={{ fontSize: '11px' }} />
          {models.map((m, i) => (
            <Line
              key={m} type="monotone" dataKey={m.split('/').pop() || m}
              stroke={colorFor(m, i)} strokeWidth={2} dot={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  );
};

// ── Chart 6: Success vs Failure ──────────────────────────────────────────────
export const SuccessFailureChart: React.FC<{ errors: ErrorBreakdownPoint[] }> = ({ errors }) => {
  const models = [...new Set(errors.map(e => e.model))];
  const data = models.map(m => {
    const row: Record<string, string | number> = { model: m.split('/').pop() || m };
    const recs = errors.filter(e => e.model === m);
    recs.forEach(r => { row[r.status] = r.count; });
    return row;
  });
  const statuses = [...new Set(errors.map(e => e.status))];
  const statusColors: Record<string, string> = { SUCCESS: '#10b981', ERROR: '#f43f5e', UNANSWERABLE: '#f59e0b' };
  return (
    <ChartCard title="Success vs Failure by Model">
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis dataKey="model" tick={{ fontSize: 10 }} />
          <YAxis tick={{ fontSize: 10 }} allowDecimals={false} />
          <Tooltip />
          <Legend iconSize={10} wrapperStyle={{ fontSize: '11px' }} />
          {statuses.map(s => (
            <Bar key={s} dataKey={s} stackId="a" fill={statusColors[s] || '#94a3b8'} radius={[3, 3, 0, 0]} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
};

// ── Chart 7: Category Distribution ──────────────────────────────────────────
const CAT_COLORS = ['#6366f1','#8b5cf6','#0ea5e9','#10b981','#f59e0b','#f43f5e','#ec4899','#14b8a6','#64748b'];

export const CategoryDistributionChart: React.FC<{ data: CategoryPoint[] }> = ({ data }) => (
  <ChartCard title="Query Category Distribution">
    <ResponsiveContainer width="100%" height={220}>
      <PieChart>
        <Pie data={data} dataKey="count" nameKey="query_type" cx="50%" cy="50%"
          outerRadius={80} label={(entry: any) =>
            `${entry.query_type || 'unknown'} ${((entry.percent || 0) * 100).toFixed(0)}%`
          }
          labelLine={false}
        >
          {data.map((_, i) => <Cell key={i} fill={CAT_COLORS[i % CAT_COLORS.length]} />)}
        </Pie>
        <Tooltip formatter={(v: any) => [v ?? 0, 'queries']} />
      </PieChart>
    </ResponsiveContainer>
  </ChartCard>
);

// ── Chart 8: Accuracy vs Latency scatter ────────────────────────────────────
export const AccuracyLatencyScatter: React.FC<{ models: ModelStats[] }> = ({ models }) => {
  const data = models.map(m => ({
    name: m.model.split('/').pop() || m.model,
    latency: Math.round(m.avg_latency_ms),
    success: m.success_rate_pct,
    model: m.model,
  }));
  return (
    <ChartCard title="Model Trade-off: Success Rate vs Avg Latency">
      <ResponsiveContainer width="100%" height={220}>
        <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis dataKey="latency" name="Avg Latency (ms)" type="number" tick={{ fontSize: 10 }}
            label={{ value: 'Latency (ms)', position: 'insideBottom', offset: -4, fontSize: 10 }} />
          <YAxis dataKey="success" name="Success Rate (%)" type="number" domain={[0, 100]}
            tick={{ fontSize: 10 }} label={{ value: 'Success %', angle: -90, position: 'insideLeft', fontSize: 10 }} />
          <Tooltip cursor={{ strokeDasharray: '3 3' }}
            content={({ payload }) => {
              if (!payload?.length) return null;
              const p = payload[0].payload;
              return (
                <div className="bg-white border border-slate-200 rounded-lg p-2 text-xs shadow">
                  <p className="font-bold text-slate-700">{p.name}</p>
                  <p className="text-slate-500">Latency: {p.latency}ms</p>
                  <p className="text-slate-500">Success: {p.success.toFixed(1)}%</p>
                </div>
              );
            }}
          />
          {data.map((d, i) => (
            <Scatter key={d.model} name={d.name} data={[d]} fill={colorFor(d.model, i)} />
          ))}
          <Legend />
        </ScatterChart>
      </ResponsiveContainer>
    </ChartCard>
  );
};
