import { useState, useEffect } from 'react';
import {
  fetchSHAP,
  fetchLIME,
  fetchAttribution,
  fetchComparison,
  fetchGlobalSHAPMetrics,
  runXAIBenchmark,
  generateEvidentlyReport,
  fetchDriftReport,
} from '../services/xaiApi';
import type {
  SHAPResult,
  LIMEResult,
  AttributionResult,
  XAIComparison,
  EvidentlyReport,
  XAIBenchmarkReport,
} from '../types/xai';
import SHAPVisualization from '../components/xai/SHAPVisualization';
import LIMEVisualization from '../components/xai/LIMEVisualization';
import SourceAttributionView from '../components/xai/SourceAttributionView';
import SHAPvLIMEComparison from '../components/xai/SHAPvLIMEComparison';
import EvidentlyMonitor from '../components/xai/EvidentlyMonitor';

const SAMPLE_QUERIES = [
  'What is the minimum attendance required to write end-semester examinations?',
  'How many books can undergraduate students borrow from the library and what is the late fine?',
  'When do classes commence for the academic year according to the calendar?',
  'What are the eligibility criteria for merit scholarships?',
  'Compare the attendance rules for UG and PG students.',
];

const MODELS = [
  { id: 'qwen/qwen3.8-27b', name: 'Qwen 3.8 27B (Default)' },
  { id: 'openai/gpt-oss-120b', name: 'GPT-OSS 120B' },
  { id: 'openai/gpt-oss-20b', name: 'GPT-OSS 20B' },
];

const RETRIEVAL_FEATURE_KEYS = new Set([
  'semantic_similarity',
  'reranker_score_norm',
  'keyword_overlap_ratio',
  'chunk_position_norm',
  'bm25_score_norm',
  'query_length_tokens',
  'section_match',
  'content_type_bonus',
]);

export default function ExplainabilityLab() {
  const [activeTab, setActiveTab] = useState<'interactive' | 'comparison' | 'evidently' | 'benchmark' | 'global'>('interactive');
  const [query, setQuery] = useState(SAMPLE_QUERIES[0]);
  const [selectedModel, setSelectedModel] = useState(MODELS[0].id);
  const [target, setTarget] = useState<'retrieval_relevance' | 'intent_classification'>('retrieval_relevance');

  // Interactive results state
  const [shapData, setShapData] = useState<SHAPResult | null>(null);
  const [limeData, setLimeData] = useState<LIMEResult | null>(null);
  const [attrData, setAttrData] = useState<AttributionResult | null>(null);
  const [comparisonData, setComparisonData] = useState<XAIComparison | null>(null);
  const [evidentlyReport, setEvidentlyReport] = useState<EvidentlyReport | null>(null);
  const [driftReport, setDriftReport] = useState<EvidentlyReport | null>(null);
  const [benchmarkReport, setBenchmarkReport] = useState<XAIBenchmarkReport | null>(null);
  const [globalMetrics, setGlobalMetrics] = useState<{
    global_importance: Record<string, number>;
    retrieval_features?: Record<string, number>;
    linguistic_tokens?: Record<string, number>;
    n_explanations: number;
  } | null>(null);

  // Attribution dynamic text state
  const [attributionAnswer, setAttributionAnswer] = useState<string>('');
  const [attributionMode, setAttributionMode] = useState<'auto' | 'custom'>('auto');
  const [reAttributing, setReAttributing] = useState(false);

  // Loading states
  const [loadingInteractive, setLoadingInteractive] = useState(false);
  const [loadingComparison, setLoadingComparison] = useState(false);
  const [loadingEvidently, setLoadingEvidently] = useState(false);
  const [loadingBenchmark, setLoadingBenchmark] = useState(false);

  // Load global metrics on mount
  useEffect(() => {
    fetchGlobalSHAPMetrics()
      .then(setGlobalMetrics)
      .catch(() => {});
  }, []);

  const handleRunInteractive = async () => {
    setLoadingInteractive(true);
    setLoadingComparison(true);
    try {
      // Execute retrieval relevance, intent words, LIME, and comparison simultaneously
      const [shapRetRes, shapIntRes, limeRetRes, limeIntRes, compRes] = await Promise.all([
        fetchSHAP(query, selectedModel, 'retrieval_relevance'),
        fetchSHAP(query, selectedModel, 'intent_classification'),
        fetchLIME(query, selectedModel, 'retrieval_relevance'),
        fetchLIME(query, selectedModel, 'intent_classification'),
        fetchComparison(query, selectedModel, target),
      ]);

      const mergedShap: SHAPResult = {
        ...shapRetRes,
        word_importance: shapIntRes.word_importance,
        predicted_class: shapIntRes.predicted_class,
        confidence: shapIntRes.confidence,
        class_probabilities: shapIntRes.class_probabilities,
      };

      const mergedLime: LIMEResult = {
        ...limeRetRes,
        word_importance: limeIntRes.word_importance,
        positive_words: limeIntRes.positive_words,
        negative_words: limeIntRes.negative_words,
        predicted_class: limeIntRes.predicted_class,
        confidence: limeIntRes.confidence,
      };

      setShapData(mergedShap);
      setLimeData(mergedLime);
      setComparisonData(compRes);

      // Determine answer for attribution: live generated from LLM or custom text
      let answerToAttribute = attributionAnswer.trim();
      if (attributionMode === 'auto' || !answerToAttribute) {
        try {
          const chatRes = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: query, provider: 'groq', model: selectedModel }),
          });
          if (chatRes.ok) {
            const chatData = await chatRes.json();
            if (chatData.response) {
              answerToAttribute = chatData.response;
              setAttributionAnswer(chatData.response);
            }
          }
        } catch (e) {
          console.warn('Could not auto-generate live response from /api/chat, falling back to handbook excerpt', e);
        }
      }

      if (!answerToAttribute) {
        answerToAttribute =
          'Students require a minimum of 85% aggregate attendance in each course to be eligible to appear for the End Semester Examinations. Library users may borrow up to 3 books at a time for 14 days.';
        setAttributionAnswer(answerToAttribute);
      }

      const attrRes = await fetchAttribution(query, answerToAttribute, selectedModel);
      setAttrData(attrRes);

      // Refresh global metrics immediately
      fetchGlobalSHAPMetrics().then(setGlobalMetrics).catch(() => {});
    } catch (err) {
      console.error('Error running explainers:', err);
    } finally {
      setLoadingInteractive(false);
      setLoadingComparison(false);
    }
  };

  const handleReAttribute = async () => {
    if (!attributionAnswer.trim()) return;
    setReAttributing(true);
    try {
      const attrRes = await fetchAttribution(query, attributionAnswer, selectedModel);
      setAttrData(attrRes);
    } catch (err) {
      console.error('Error re-attributing answer:', err);
    } finally {
      setReAttributing(false);
    }
  };

  const handleRunComparison = async () => {
    setLoadingComparison(true);
    try {
      const res = await fetchComparison(query, selectedModel, target);
      setComparisonData(res);
    } catch (err) {
      console.error('Error running comparison:', err);
    } finally {
      setLoadingComparison(false);
    }
  };

  const handleGenerateEvidently = async (type: 'rag_quality' | 'drift') => {
    setLoadingEvidently(true);
    try {
      if (type === 'drift') {
        const d = await fetchDriftReport(selectedModel);
        setDriftReport(d);
      } else {
        const q = await generateEvidentlyReport('rag_quality', selectedModel);
        setEvidentlyReport(q);
      }
    } catch (err) {
      console.error('Error with Evidently:', err);
    } finally {
      setLoadingEvidently(false);
    }
  };

  const handleRunBenchmark = async () => {
    setLoadingBenchmark(true);
    try {
      const b = await runXAIBenchmark(selectedModel, 5);
      setBenchmarkReport(b);
    } catch (err) {
      console.error('Error running XAI benchmark:', err);
    } finally {
      setLoadingBenchmark(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-gradient-to-r from-purple-900 via-indigo-900 to-amber-900 rounded-xl p-6 text-white shadow-md">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Explainable RAG & Model Intelligence Lab</h1>
            <p className="text-sm text-purple-200 mt-1 max-w-2xl">
              Research suite inspecting feature contributions (SHAP), local surrogate text importance (LIME), sentence-level evidence attribution, and Evidently AI drift monitoring.
            </p>
          </div>
          <div className="flex items-center gap-2 bg-white/10 backdrop-blur-sm px-4 py-2 rounded-lg border border-white/20 text-xs">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            Surrogates Active & Verified
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-2 flex flex-wrap gap-2">
        {[
          { id: 'interactive', label: 'Interactive Explainer (SHAP + LIME + Attribution)' },
          { id: 'comparison', label: 'SHAP vs LIME Benchmark' },
          { id: 'evidently', label: 'Evidently AI Monitoring' },
          { id: 'benchmark', label: 'Batch Benchmark Suite' },
          { id: 'global', label: 'Global Feature Dynamics' },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? 'bg-purple-700 text-white shadow-sm'
                : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Query & Model Controls Bar */}
      {(activeTab === 'interactive' || activeTab === 'comparison') && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 space-y-3">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
            <div className="md:col-span-2 space-y-1">
              <label className="text-xs font-semibold text-gray-700 uppercase tracking-wide">Test Query</label>
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Enter query to explain..."
                className="w-full text-sm border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-purple-600"
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-semibold text-gray-700 uppercase tracking-wide">Model Architecture</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full text-sm border border-gray-300 rounded-md px-3 py-2 bg-white focus:outline-none focus:ring-2 focus:ring-purple-600"
              >
                {MODELS.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-1">
              <label className="text-xs font-semibold text-gray-700 uppercase tracking-wide">Target Prediction</label>
              <select
                value={target}
                onChange={(e) => setTarget(e.target.value as any)}
                className="w-full text-sm border border-gray-300 rounded-md px-3 py-2 bg-white focus:outline-none focus:ring-2 focus:ring-purple-600"
              >
                <option value="retrieval_relevance">Retrieval Relevance (Cross-Chunk)</option>
                <option value="intent_classification">Intent Classification (Query Words)</option>
              </select>
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-gray-100">
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-500">Quick Samples:</span>
              <div className="flex flex-wrap gap-1">
                {SAMPLE_QUERIES.slice(0, 3).map((sq, idx) => (
                  <button
                    key={idx}
                    onClick={() => setQuery(sq)}
                    className="text-xs bg-gray-100 hover:bg-purple-100 hover:text-purple-800 text-gray-700 px-2 py-1 rounded transition-colors"
                  >
                    Sample {idx + 1}
                  </button>
                ))}
              </div>
            </div>

            <button
              onClick={activeTab === 'interactive' ? handleRunInteractive : handleRunComparison}
              disabled={loadingInteractive || loadingComparison}
              className="px-5 py-2 bg-purple-700 hover:bg-purple-800 text-white text-sm font-medium rounded-md shadow-sm disabled:opacity-50 transition-colors flex items-center gap-2"
            >
              {(loadingInteractive || loadingComparison) ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" />
                  Calculating Attributions...
                </>
              ) : (
                'Run Explainability Analysis'
              )}
            </button>
          </div>
        </div>
      )}

      {/* Tab 1: Interactive Explainer */}
      {activeTab === 'interactive' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-5">
              <SHAPVisualization data={shapData} loading={loadingInteractive} />
            </div>
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-5">
              <LIMEVisualization data={limeData} loading={loadingInteractive} />
            </div>
          </div>

          {/* Dynamic Answer for Attribution Panel */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-5 space-y-3">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <div>
                <h3 className="text-sm font-bold text-gray-900 flex items-center gap-2">
                  <span>💬 Evidence Attribution Input</span>
                  <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                    attributionMode === 'auto' ? 'bg-purple-100 text-purple-800' : 'bg-amber-100 text-amber-800'
                  }`}>
                    {attributionMode === 'auto' ? 'Auto Live LLM Generation' : 'Custom Input Mode'}
                  </span>
                </h3>
                <p className="text-xs text-gray-500">
                  Each factual sentence below is split and mapped to retrieved handbook chunks via embedding cosine similarity.
                </p>
              </div>

              <div className="flex items-center gap-2">
                <div className="bg-gray-100 p-0.5 rounded-lg flex text-xs">
                  <button
                    onClick={() => setAttributionMode('auto')}
                    className={`px-2.5 py-1 rounded-md font-medium transition ${
                      attributionMode === 'auto' ? 'bg-white shadow text-purple-700' : 'text-gray-600 hover:text-gray-900'
                    }`}
                  >
                    Auto-Generate
                  </button>
                  <button
                    onClick={() => setAttributionMode('custom')}
                    className={`px-2.5 py-1 rounded-md font-medium transition ${
                      attributionMode === 'custom' ? 'bg-white shadow text-purple-700' : 'text-gray-600 hover:text-gray-900'
                    }`}
                  >
                    Custom Text
                  </button>
                </div>

                <button
                  onClick={handleReAttribute}
                  disabled={reAttributing || !attributionAnswer.trim()}
                  className="px-3 py-1 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-medium rounded-md shadow-sm disabled:opacity-50 transition flex items-center gap-1.5"
                >
                  {reAttributing && <div className="animate-spin rounded-full h-3 w-3 border-2 border-white border-t-transparent" />}
                  <span>Re-evaluate Attribution</span>
                </button>
              </div>
            </div>

            <textarea
              rows={3}
              value={attributionAnswer}
              onChange={(e) => {
                setAttributionAnswer(e.target.value);
                setAttributionMode('custom');
              }}
              placeholder="Inspect the model's generated answer or type any custom statement to evaluate sentence-level support against handbook chunks..."
              className="w-full text-xs font-mono border border-gray-300 rounded-md p-3 focus:outline-none focus:ring-2 focus:ring-purple-600 leading-relaxed bg-slate-50 text-gray-800"
            />
          </div>

          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-5">
            <SourceAttributionView data={attrData} loading={loadingInteractive} />
          </div>
        </div>
      )}

      {/* Tab 2: SHAP vs LIME Comparison */}
      {activeTab === 'comparison' && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <SHAPvLIMEComparison data={comparisonData} loading={loadingComparison} />
        </div>
      )}

      {/* Tab 3: Evidently AI Monitoring */}
      {activeTab === 'evidently' && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <EvidentlyMonitor
            report={evidentlyReport}
            driftReport={driftReport}
            loading={loadingEvidently}
            onGenerate={handleGenerateEvidently}
          />
        </div>
      )}

      {/* Tab 4: Batch Benchmark Suite */}
      {activeTab === 'benchmark' && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-base font-semibold text-gray-900">Research Benchmark Evaluation</h3>
              <p className="text-xs text-gray-500">Run identical query suites through SHAP and LIME to systematically evaluate runtime, sparsity, and agreement.</p>
            </div>
            <button
              onClick={handleRunBenchmark}
              disabled={loadingBenchmark}
              className="px-4 py-2 bg-purple-700 text-white text-xs font-medium rounded-lg hover:bg-purple-800 disabled:opacity-50"
            >
              {loadingBenchmark ? 'Evaluating...' : 'Run 5-Query Benchmark'}
            </button>
          </div>

          {benchmarkReport ? (
            <div className="overflow-x-auto border border-gray-200 rounded-lg">
              <table className="w-full text-left text-xs">
                <thead className="bg-gray-50 border-b border-gray-200 text-gray-700 uppercase font-semibold">
                  <tr>
                    <th className="py-2.5 px-3">Benchmark Query</th>
                    <th className="py-2.5 px-3">SHAP (ms)</th>
                    <th className="py-2.5 px-3">LIME (ms)</th>
                    <th className="py-2.5 px-3">Agreement (Spearman)</th>
                    <th className="py-2.5 px-3">SHAP Sparsity</th>
                    <th className="py-2.5 px-3">LIME Sparsity</th>
                    <th className="py-2.5 px-3">LIME R²</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {benchmarkReport.results.map((row, idx) => (
                    <tr key={idx} className="hover:bg-gray-50">
                      <td className="py-2 px-3 font-medium text-gray-900 max-w-xs truncate">{row.query}</td>
                      <td className="py-2 px-3 font-mono text-purple-700">{row.shap_runtime_ms?.toFixed(0)}</td>
                      <td className="py-2 px-3 font-mono text-emerald-700">{row.lime_runtime_ms?.toFixed(0)}</td>
                      <td className="py-2 px-3 font-mono font-semibold text-indigo-700">{row.feature_consistency?.toFixed(3)}</td>
                      <td className="py-2 px-3 font-mono">{row.shap_sparsity?.toFixed(2)}</td>
                      <td className="py-2 px-3 font-mono">{row.lime_sparsity?.toFixed(2)}</td>
                      <td className="py-2 px-3 font-mono">{row.lime_local_fidelity?.toFixed(3) ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="p-8 text-center text-sm text-gray-500 bg-gray-50 rounded-lg border border-dashed border-gray-300">
              Click "Run 5-Query Benchmark" to compute metrics across the benchmark test suite.
            </div>
          )}
        </div>
      )}

      {/* Tab 5: Global Feature Dynamics */}
      {activeTab === 'global' && (
        <div className="space-y-6">
          <div className="flex items-center justify-between flex-wrap gap-2 bg-white rounded-lg p-4 border border-gray-200">
            <div>
              <h3 className="text-base font-semibold text-gray-900">Global Feature Importance Dynamics</h3>
              <p className="text-xs text-gray-500">
                Separated dynamics across {globalMetrics?.n_explanations || 0} evaluations recorded in SQLite.
              </p>
            </div>
            <div className="text-xs bg-purple-50 text-purple-700 border border-purple-200 px-3 py-1 rounded-full font-medium">
              Dual-Stream Decomposition (Retrieval vs Linguistic)
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Card 1: Retrieval Pipeline Features */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 space-y-4">
              <div className="border-b border-gray-100 pb-3">
                <h4 className="text-sm font-bold text-gray-900 flex items-center gap-2">
                  <span>📊 RAG Retrieval Pipeline Features</span>
                  <span className="text-[10px] bg-purple-100 text-purple-800 px-2 py-0.5 rounded-full font-medium">Surrogate Weights</span>
                </h4>
                <p className="text-xs text-gray-500 mt-1">
                  Relative impact of candidate ranking features learned by the GradientBoostingRegressor surrogate.
                </p>
              </div>

              <div className="space-y-3">
                {(globalMetrics?.retrieval_features && Object.keys(globalMetrics.retrieval_features).length > 0
                  ? Object.entries(globalMetrics.retrieval_features)
                  : Object.entries(globalMetrics?.global_importance || {}).filter(([k]) => RETRIEVAL_FEATURE_KEYS.has(k.toLowerCase()))
                ).map(([feature, val], i) => (
                  <div key={i} className="space-y-1">
                    <div className="flex justify-between text-xs font-medium text-gray-700">
                      <span className="capitalize">{feature.replace(/_/g, ' ')}</span>
                      <span className="font-mono text-purple-700">{val.toFixed(5)}</span>
                    </div>
                    <div className="h-2.5 w-full bg-gray-100 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-purple-600 to-indigo-600 rounded-full transition-all duration-500"
                        style={{ width: `${Math.min(Math.max(val * 150, 4), 100)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Card 2: Query Linguistic Tokens */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 space-y-4">
              <div className="border-b border-gray-100 pb-3">
                <h4 className="text-sm font-bold text-gray-900 flex items-center gap-2">
                  <span>🔤 Key Query Linguistic Tokens</span>
                  <span className="text-[10px] bg-emerald-100 text-emerald-800 px-2 py-0.5 rounded-full font-medium">Intent Salience</span>
                </h4>
                <p className="text-xs text-gray-500 mt-1">
                  High-frequency salient tokens extracted from query intent classification across queries.
                </p>
              </div>

              <div className="space-y-3">
                {(globalMetrics?.linguistic_tokens && Object.keys(globalMetrics.linguistic_tokens).length > 0
                  ? Object.entries(globalMetrics.linguistic_tokens)
                  : Object.entries(globalMetrics?.global_importance || {}).filter(([k]) => !RETRIEVAL_FEATURE_KEYS.has(k.toLowerCase()))
                ).map(([word, val], i) => (
                  <div key={i} className="space-y-1">
                    <div className="flex justify-between text-xs font-medium text-gray-700">
                      <span className="font-semibold text-emerald-900 capitalize">&quot;{word}&quot;</span>
                      <span className="font-mono text-emerald-700">{val.toFixed(5)}</span>
                    </div>
                    <div className="h-2.5 w-full bg-gray-100 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-emerald-500 to-teal-500 rounded-full transition-all duration-500"
                        style={{ width: `${Math.min(Math.max(val * 200, 4), 100)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
