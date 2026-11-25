import type { AttributionResult, SentenceAttribution } from '../../types/xai';

interface Props {
  data: AttributionResult | null;
  loading: boolean;
  error?: string;
}

const supportColors: Record<string, { bg: string; text: string; border: string; badge: string }> = {
  supported: {
    bg: 'bg-green-50',
    text: 'text-green-900',
    border: 'border-green-300',
    badge: 'bg-green-100 text-green-800',
  },
  partial: {
    bg: 'bg-yellow-50',
    text: 'text-yellow-900',
    border: 'border-yellow-300',
    badge: 'bg-yellow-100 text-yellow-800',
  },
  unsupported: {
    bg: 'bg-red-50',
    text: 'text-red-900',
    border: 'border-red-300',
    badge: 'bg-red-100 text-red-800',
  },
};

function SentenceCard({ s }: { s: SentenceAttribution }) {
  const colors = supportColors[s.support_level];
  return (
    <div className={`rounded-lg border p-4 ${colors.bg} ${colors.border}`}>
      <div className="flex items-start justify-between gap-2 mb-2">
        <p className={`text-sm font-medium ${colors.text}`}>"{s.sentence}"</p>
        <span className={`shrink-0 text-xs font-semibold px-2 py-0.5 rounded-full ${colors.badge}`}>
          {s.support_level}
        </span>
      </div>
      {s.support_level !== 'unsupported' && (
        <div className="mt-2 space-y-1 text-xs text-gray-600">
          <div className="flex flex-wrap gap-2">
            {s.attributed_page && (
              <span className="bg-white border border-gray-200 rounded px-2 py-0.5">
                📄 Page {s.attributed_page}
              </span>
            )}
            {s.attributed_section && (
              <span className="bg-white border border-gray-200 rounded px-2 py-0.5">
                § {s.attributed_section}
              </span>
            )}
            <span className="bg-white border border-gray-200 rounded px-2 py-0.5">
              sim: {s.similarity_score.toFixed(3)}
            </span>
          </div>
          {s.supporting_passage && (
            <div className="mt-1 bg-white border border-gray-200 rounded p-2 text-gray-500 italic text-xs">
              "{s.supporting_passage.slice(0, 220)}{s.supporting_passage.length > 220 ? '…' : ''}"
            </div>
          )}
        </div>
      )}
      {s.support_level === 'unsupported' && (
        <p className="text-xs text-red-600 mt-1">
          ⚠ No retrieved chunk matched with sufficient similarity (threshold {'>'}0.18).
          This statement could not be attributed to source documents.
        </p>
      )}
    </div>
  );
}

export default function SourceAttributionView({ data, loading, error }: Props) {
  if (loading) {
    return (
      <div className="flex items-center justify-center h-48 text-gray-500">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mr-3" />
        Computing sentence-level attributions…
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-amber-800 text-sm">
        {error || 'No attribution data. Provide a query and answer to compute attribution.'}
      </div>
    );
  }

  const { coverage, sentence_attributions: attrs } = data;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div>
        <h3 className="text-base font-semibold text-gray-900">Sentence-Level Source Attribution</h3>
        <p className="text-xs text-gray-500 mt-0.5">
          Embedding cosine similarity between answer sentences and retrieved chunks.
          Runtime: {data.runtime_ms?.toFixed(0)} ms · {data.n_chunks} chunks
        </p>
      </div>

      {/* Coverage summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {[
          { label: 'Sentences', value: coverage.total_sentences, color: 'text-gray-800' },
          { label: 'Supported', value: coverage.supported, color: 'text-green-700' },
          { label: 'Partial', value: coverage.partial, color: 'text-yellow-700' },
          { label: 'Unsupported', value: coverage.unsupported, color: 'text-red-700' },
        ].map(m => (
          <div key={m.label} className="bg-white border border-gray-200 rounded-lg p-3 text-center shadow-sm">
            <div className={`text-2xl font-bold ${m.color}`}>{m.value}</div>
            <div className="text-xs text-gray-500">{m.label}</div>
          </div>
        ))}
      </div>

      {/* Progress bars */}
      <div className="bg-gray-50 rounded-lg p-3">
        <div className="flex justify-between text-xs text-gray-500 mb-1">
          <span>Support coverage</span>
          <span>{(coverage.coverage_rate * 100).toFixed(0)}%</span>
        </div>
        <div className="h-2 bg-gray-200 rounded-full overflow-hidden flex">
          <div
            className="h-full bg-green-500 transition-all"
            style={{ width: `${(coverage.supported / Math.max(coverage.total_sentences, 1)) * 100}%` }}
          />
          <div
            className="h-full bg-yellow-400 transition-all"
            style={{ width: `${(coverage.partial / Math.max(coverage.total_sentences, 1)) * 100}%` }}
          />
          <div
            className="h-full bg-red-400 transition-all"
            style={{ width: `${(coverage.unsupported / Math.max(coverage.total_sentences, 1)) * 100}%` }}
          />
        </div>
        <div className="flex gap-4 mt-1 text-xs text-gray-400">
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-green-500 inline-block" />Supported</span>
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-yellow-400 inline-block" />Partial</span>
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-red-400 inline-block" />Unsupported</span>
        </div>
      </div>

      {/* Attribution disclaimer */}
      <div className="bg-blue-50 border-l-4 border-blue-400 p-3 text-xs text-blue-800 rounded">
        <strong>Note:</strong> Attribution uses cosine similarity between answer text and retrieved chunks.
        High similarity suggests overlap — it does <em>not</em> guarantee factual correctness.
        "Unsupported" means no retrieved chunk closely matched; it does not mean the statement is false.
      </div>

      {/* Sentence cards */}
      <div className="space-y-3">
        {attrs.map(s => (
          <SentenceCard key={s.sentence_id} s={s} />
        ))}
      </div>
    </div>
  );
}
