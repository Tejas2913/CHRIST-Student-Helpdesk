import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import type { XAIComparison } from '../../types/xai';

interface Props {
  data: XAIComparison | null;
  loading: boolean;
  error?: string;
}

function MetricRow({
  label,
  shap,
  lime,
  note,
}: {
  label: string;
  shap?: number | null;
  lime?: number | null;
  note: string;
  higher?: string;
}) {
  const fmt = (v: number | null | undefined) =>
    v == null ? '—' : typeof v === 'number' ? v.toFixed(3) : String(v);

  return (
    <tr className="border-b border-gray-100 hover:bg-gray-50">
      <td className="py-2 px-3 text-sm font-medium text-gray-800 w-36">{label}</td>
      <td className="py-2 px-3 text-sm text-center font-mono text-purple-700">{fmt(shap)}</td>
      <td className="py-2 px-3 text-sm text-center font-mono text-emerald-700">{fmt(lime)}</td>
      <td className="py-2 px-3 text-xs text-gray-500">{note}</td>
    </tr>
  );
}

export default function SHAPvLIMEComparison({ data, loading, error }: Props) {
  if (loading) {
    return (
      <div className="flex items-center justify-center h-48 text-gray-500">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600 mr-3" />
        Running SHAP vs LIME comparison…
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-amber-800 text-sm">
        {error || 'No comparison data. Click "Compare SHAP vs LIME" to run analysis.'}
      </div>
    );
  }

  if (data.error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-800 text-sm">
        Comparison error: {data.error}
      </div>
    );
  }

  const { metrics } = data;

  // Runtime comparison bar chart
  const runtimeData = [
    { name: 'SHAP', value: metrics.runtime_ms.shap ?? 0, fill: '#7c3aed' },
    { name: 'LIME', value: metrics.runtime_ms.lime ?? 0, fill: '#059669' },
  ];

  // Feature consistency bar
  const consistency = metrics.feature_consistency.value ?? 0;

  // Top SHAP features
  const topSHAP = (data.top_shap_features || []).slice(0, 5).map(f => ({
    name: f.feature.replace(/_/g, ' '),
    value: f.importance,
  }));

  return (
    <div className="space-y-5">
      {/* Header */}
      <div>
        <h3 className="text-base font-semibold text-gray-900">SHAP vs LIME Comparison</h3>
        <p className="text-xs text-gray-500 mt-0.5">
          Both methods explain the same prediction target on the same input.
          Results reflect surrogate model approximations — not direct LLM internals.
        </p>
      </div>

      {/* Disclaimer */}
      <div className="bg-blue-50 border-l-4 border-blue-400 p-3 text-xs text-blue-800 rounded">
        <strong>Experimental design:</strong> SHAP uses additive feature attributions on a GBM surrogate.
        LIME fits a local linear model around the instance. Neither can claim universal superiority —
        each has distinct theoretical properties. Results here are specific to the retrieval relevance
        prediction target.
      </div>

      {/* Metrics table */}
      <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
        <table className="w-full">
          <thead className="bg-gray-50">
            <tr>
              <th className="py-2 px-3 text-left text-xs font-semibold text-gray-700">Metric</th>
              <th className="py-2 px-3 text-center text-xs font-semibold text-purple-700">SHAP</th>
              <th className="py-2 px-3 text-center text-xs font-semibold text-emerald-700">LIME</th>
              <th className="py-2 px-3 text-left text-xs font-semibold text-gray-700">Note</th>
            </tr>
          </thead>
          <tbody>
            <MetricRow
              label="Faithfulness"
              shap={metrics.faithfulness.shap}
              lime={metrics.faithfulness.lime}
              note={metrics.faithfulness.note}
            />
            <MetricRow
              label="Stability"
              shap={metrics.stability.shap}
              lime={metrics.stability.lime}
              note={metrics.stability.note}
            />
            <MetricRow
              label="Local Fidelity"
              shap={metrics.local_fidelity.shap}
              lime={metrics.local_fidelity.lime}
              note={metrics.local_fidelity.note}
            />
            <MetricRow
              label="Runtime (ms)"
              shap={metrics.runtime_ms.shap}
              lime={metrics.runtime_ms.lime}
              note={metrics.runtime_ms.note}
            />
            <MetricRow
              label="Sparsity"
              shap={metrics.sparsity.shap}
              lime={metrics.sparsity.lime}
              note={metrics.sparsity.note}
            />
            <tr className="border-b border-gray-100 hover:bg-gray-50">
              <td className="py-2 px-3 text-sm font-medium text-gray-800">Feature Consistency</td>
              <td className="py-2 px-3 text-sm text-center font-mono text-indigo-700" colSpan={2}>
                {consistency.toFixed(3)} (Spearman r)
              </td>
              <td className="py-2 px-3 text-xs text-gray-500">{metrics.feature_consistency.note}</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Runtime chart */}
      <div>
        <h4 className="text-sm font-medium text-gray-700 mb-2">Runtime Comparison (ms)</h4>
        <ResponsiveContainer width="100%" height={100}>
          <BarChart data={runtimeData} layout="vertical" margin={{ left: 10, right: 20 }}>
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={50} />
            <Tooltip formatter={(v: any) => [`${Number(v || 0).toFixed(0)} ms`, 'Runtime']} contentStyle={{ fontSize: 12 }} />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {runtimeData.map((d, i) => <Cell key={i} fill={d.fill} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Feature consistency gauge */}
      <div>
        <h4 className="text-sm font-medium text-gray-700 mb-2">
          Feature Ranking Consistency: {consistency.toFixed(3)}
          <span className="ml-2 text-xs text-gray-500">
            {consistency > 0.7 ? '(High)' : consistency > 0.4 ? '(Moderate)' : '(Low)'}
          </span>
        </h4>
        <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all"
            style={{
              width: `${Math.max(0, consistency) * 100}%`,
              background: consistency > 0.7 ? '#059669' : consistency > 0.4 ? '#f59e0b' : '#dc2626',
            }}
          />
        </div>
        <div className="flex justify-between text-xs text-gray-400 mt-1">
          <span>-1 (disagreement)</span><span>0</span><span>+1 (agreement)</span>
        </div>
      </div>

      {/* Top SHAP features */}
      {topSHAP.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-2">Top SHAP Features</h4>
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={topSHAP} layout="vertical" margin={{ left: 10, right: 20 }}>
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={160} />
              <Tooltip formatter={(v: any) => [Number(v || 0).toFixed(5), 'Mean |SHAP|']} contentStyle={{ fontSize: 12 }} />
              <Bar dataKey="value" fill="#7c3aed" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Summary */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 text-sm text-gray-700">
        <strong>Summary:</strong> {data.summary}
      </div>
    </div>
  );
}
