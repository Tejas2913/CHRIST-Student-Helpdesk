import { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine,
} from 'recharts';
import type { SHAPResult } from '../../types/xai';

interface Props {
  data: SHAPResult | null;
  loading: boolean;
  error?: string;
}

const POSITIVE_COLOR = '#7c3aed';
const NEGATIVE_COLOR = '#f59e0b';

function formatFeatureName(name: string): string {
  return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

export default function SHAPVisualization({ data, loading, error }: Props) {
  const [view, setView] = useState<'global' | 'chunk' | 'intent'>('global');

  if (loading) {
    return (
      <div className="flex items-center justify-center h-48 text-gray-500">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-700 mr-3" />
        Computing SHAP values…
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-amber-800 text-sm">
        {error || 'No SHAP data available. Submit a query to generate explanations.'}
      </div>
    );
  }

  // Global importance chart data
  const globalData = Object.entries(data.global_importance || {})
    .map(([feature, value]) => ({ feature: formatFeatureName(feature), value: Math.abs(value), raw: feature }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 8);

  // Chunk-level data for first chunk
  const topChunk = data.chunk_explanations?.[0];
  const chunkContribData = topChunk
    ? Object.entries(topChunk.shap_contributions)
        .map(([feature, value]) => ({ feature: formatFeatureName(feature), value: parseFloat(value.toFixed(5)) }))
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    : [];

  // Intent word importance
  const wordData = (data.word_importance || []).slice(0, 15);

  const tabs = [
    { id: 'global', label: 'Global Importance' },
    { id: 'chunk', label: 'Chunk Attribution', disabled: !topChunk },
    { id: 'intent', label: 'Intent Words' },
  ];

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h3 className="text-base font-semibold text-gray-900">SHAP Explanation</h3>
          <p className="text-xs text-gray-500 mt-0.5">
            Target: <span className="font-medium text-purple-700">{data.target}</span>
            &nbsp;·&nbsp;Surrogate: {data.surrogate_type || 'GBM'}
            &nbsp;·&nbsp;Runtime: {data.runtime_ms?.toFixed(0)} ms
          </p>
        </div>
        <div className="bg-purple-100 text-purple-800 text-xs px-2 py-1 rounded-full font-medium">
          {data.n_chunks_explained ?? 0} chunks explained
        </div>
      </div>

      {/* Disclaimer */}
      <div className="bg-blue-50 border-l-4 border-blue-400 p-3 text-xs text-blue-800 rounded">
        <strong>Note:</strong> SHAP values explain a surrogate regression model trained on retrieval features —
        they do <em>not</em> explain internal Groq LLM weights. Higher |SHAP| = stronger feature influence
        on the surrogate's relevance prediction.
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200">
        {tabs.map(t => (
          <button
            key={t.id}
            onClick={() => !t.disabled && setView(t.id as any)}
            disabled={t.disabled}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              view === t.id
                ? 'border-purple-600 text-purple-700'
                : t.disabled
                ? 'border-transparent text-gray-300 cursor-not-allowed'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Global importance */}
      {view === 'global' && (
        <div>
          <p className="text-xs text-gray-500 mb-2">Mean |SHAP| across all retrieved chunks</p>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={globalData} layout="vertical" margin={{ left: 10, right: 20 }}>
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={160} />
              <Tooltip
                formatter={(v: any) => [Number(v || 0).toFixed(5), 'Mean |SHAP|']}
                contentStyle={{ fontSize: 12 }}
              />
              <Bar dataKey="value" radius={[0, 3, 3, 0]}>
                {globalData.map((_, i) => (
                  <Cell key={i} fill={i === 0 ? POSITIVE_COLOR : i === 1 ? '#8b5cf6' : '#a78bfa'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Chunk-level waterfall */}
      {view === 'chunk' && topChunk && (
        <div>
          <div className="text-xs text-gray-500 mb-2">
            Top chunk · Page {topChunk.page} · {topChunk.section || 'No section'} · Score: {topChunk.score.toFixed(3)}
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={chunkContribData} layout="vertical" margin={{ left: 10, right: 20 }}>
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={160} />
              <ReferenceLine x={0} stroke="#9ca3af" />
              <Tooltip
                formatter={(v: any) => [Number(v || 0).toFixed(6), 'SHAP contribution']}
                contentStyle={{ fontSize: 12 }}
              />
              <Bar dataKey="value" radius={[0, 3, 3, 0]}>
                {chunkContribData.map((d, i) => (
                  <Cell key={i} fill={d.value >= 0 ? POSITIVE_COLOR : NEGATIVE_COLOR} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="flex gap-4 mt-2 text-xs text-gray-500 justify-end">
            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm inline-block" style={{ background: POSITIVE_COLOR }} /> Positive</span>
            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm inline-block" style={{ background: NEGATIVE_COLOR }} /> Negative</span>
          </div>
        </div>
      )}

      {/* Intent word importance */}
      {view === 'intent' && (
        <div>
          {wordData.length > 0 ? (
            <>
              <p className="text-xs text-gray-500 mb-2">
                Predicted class: <strong>{data.predicted_class}</strong>
                {data.confidence !== undefined && <> · Confidence: <strong>{(data.confidence * 100).toFixed(1)}%</strong></>}
              </p>
              <div className="flex flex-wrap gap-2">
                {wordData.map((w, i) => {
                  const intensity = Math.min(Math.abs(w.shap_value) / 0.1, 1);
                  const isPos = w.shap_value >= 0;
                  return (
                    <span
                      key={i}
                      title={`SHAP: ${w.shap_value.toFixed(5)}`}
                      className="px-3 py-1 rounded-full text-sm font-medium border"
                      style={{
                        background: isPos ? `rgba(124,58,237,${0.1 + intensity * 0.5})` : `rgba(245,158,11,${0.1 + intensity * 0.5})`,
                        borderColor: isPos ? '#7c3aed' : '#f59e0b',
                        color: isPos ? '#4c1d95' : '#78350f',
                      }}
                    >
                      {w.word}
                    </span>
                  );
                })}
              </div>
            </>
          ) : (
            <div className="p-6 text-center text-xs text-gray-500 bg-gray-50 rounded-lg border border-dashed">
              No token attributions calculated yet. Click &quot;Run Explainability Analysis&quot; to compute query token SHAP values.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
