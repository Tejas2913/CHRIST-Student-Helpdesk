import { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import type { LIMEResult } from '../../types/xai';

interface Props {
  data: LIMEResult | null;
  loading: boolean;
  error?: string;
}

const POS_COLOR = '#059669';
const NEG_COLOR = '#dc2626';

export default function LIMEVisualization({ data, loading, error }: Props) {
  const [view, setView] = useState<'text' | 'tabular'>('text');

  if (loading) {
    return (
      <div className="flex items-center justify-center h-48 text-gray-500">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-emerald-600 mr-3" />
        Computing LIME explanation…
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-amber-800 text-sm">
        {error || 'No LIME data available. Submit a query to generate explanations.'}
      </div>
    );
  }

  const hasText = (data.word_importance?.length ?? 0) > 0;
  const hasTabular = (data.feature_importance?.length ?? 0) > 0;

  const tabularData = (data.feature_importance || [])
    .slice(0, 8)
    .map(f => ({
      feature: f.feature_description.length > 28 ? f.feature_description.slice(0, 28) + '…' : f.feature_description,
      value: parseFloat(f.lime_weight.toFixed(5)),
    }));

  const wordImportance = data.word_importance || [];

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h3 className="text-base font-semibold text-gray-900">LIME Explanation</h3>
          <p className="text-xs text-gray-500 mt-0.5">
            Target: <span className="font-medium text-emerald-700">{data.target}</span>
            &nbsp;·&nbsp;Surrogate: {data.surrogate || 'LimeTextExplainer'}
            &nbsp;·&nbsp;Runtime: {data.runtime_ms?.toFixed(0)} ms
            {data.local_fidelity !== null && data.local_fidelity !== undefined && (
              <>&nbsp;·&nbsp;Local R²: <strong>{data.local_fidelity?.toFixed(3)}</strong></>
            )}
          </p>
        </div>
        {data.predicted_class && (
          <div className="bg-emerald-100 text-emerald-800 text-xs px-2 py-1 rounded-full font-medium">
            {data.predicted_class} ({((data.confidence ?? 0) * 100).toFixed(1)}%)
          </div>
        )}
      </div>

      {/* Disclaimer */}
      <div className="bg-blue-50 border-l-4 border-blue-400 p-3 text-xs text-blue-800 rounded">
        <strong>Note:</strong> LIME fits a local linear surrogate model. Word/feature weights reflect the
        surrogate's approximation around this input — not causal proof of LLM behaviour.
      </div>

      {/* Tabs */}
      {hasText && hasTabular && (
        <div className="flex border-b border-gray-200">
          {[{ id: 'text', label: 'Word Importance' }, { id: 'tabular', label: 'Feature Weights' }].map(t => (
            <button
              key={t.id}
              onClick={() => setView(t.id as any)}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                view === t.id ? 'border-emerald-600 text-emerald-700' : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
      )}

      {/* Text / word highlight */}
      {(view === 'text' || !hasTabular) && hasText && (
        <div>
          <p className="text-xs text-gray-500 mb-3">
            Word contributions to predicted class. Green = supports prediction · Red = contradicts.
          </p>
          <div className="flex flex-wrap gap-2 mb-4">
            {wordImportance.map((w, i) => {
              const scale = Math.min(Math.abs(w.lime_weight) / 0.3, 1);
              return (
                <span
                  key={i}
                  title={`LIME weight: ${w.lime_weight.toFixed(5)}`}
                  className="px-3 py-1 rounded-full text-sm font-medium border cursor-help"
                  style={{
                    background: w.positive
                      ? `rgba(5,150,105,${0.08 + scale * 0.45})`
                      : `rgba(220,38,38,${0.08 + scale * 0.45})`,
                    borderColor: w.positive ? POS_COLOR : NEG_COLOR,
                    color: w.positive ? '#065f46' : '#7f1d1d',
                  }}
                >
                  {w.word}
                </span>
              );
            })}
          </div>

          {/* Positive / negative breakdown */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-green-50 rounded-lg p-3">
              <h4 className="text-xs font-semibold text-green-800 mb-2">↑ Positive influence</h4>
              {(data.positive_words || []).map((w, i) => (
                <div key={i} className="flex justify-between text-xs text-green-700">
                  <span>"{w.word}"</span><span>+{w.lime_weight.toFixed(4)}</span>
                </div>
              ))}
            </div>
            <div className="bg-red-50 rounded-lg p-3">
              <h4 className="text-xs font-semibold text-red-800 mb-2">↓ Negative influence</h4>
              {(data.negative_words || []).map((w, i) => (
                <div key={i} className="flex justify-between text-xs text-red-700">
                  <span>"{w.word}"</span><span>{w.lime_weight.toFixed(4)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Tabular / feature chart */}
      {(view === 'tabular' || !hasText) && hasTabular && (
        <div>
          <p className="text-xs text-gray-500 mb-2">Feature contributions to retrieval relevance prediction</p>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={tabularData} layout="vertical" margin={{ left: 10, right: 20 }}>
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={190} />
              <Tooltip
                formatter={(v: any) => [Number(v || 0).toFixed(5), 'LIME weight']}
                contentStyle={{ fontSize: 12 }}
              />
              <Bar dataKey="value" radius={[0, 3, 3, 0]}>
                {tabularData.map((d, i) => (
                  <Cell key={i} fill={d.value >= 0 ? POS_COLOR : NEG_COLOR} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="flex gap-4 mt-2 text-xs text-gray-500 justify-end">
            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm inline-block" style={{ background: POS_COLOR }} /> Positive</span>
            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm inline-block" style={{ background: NEG_COLOR }} /> Negative</span>
          </div>
        </div>
      )}
    </div>
  );
}
