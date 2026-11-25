import React, { useState } from 'react';
import {
  Plus,
  MessageSquare,
  Trash2,
  Database,
  Cpu,
  BarChart2,
  X,
  Info,
  CheckCircle2,
} from 'lucide-react';
import type { Conversation, ModelInfo, HealthStatus, MonitoringStats } from '../types';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  conversations: Conversation[];
  activeConversationId: string;
  onSelectConversation: (id: string) => void;
  onNewConversation: () => void;
  onDeleteConversation: (id: string) => void;
  models: ModelInfo[];
  activeProvider: string;
  activeModel: string;
  onSelectModel: (provider: string, model: string) => void;
  health: HealthStatus | null;
  stats: MonitoringStats | null;
}

export const Sidebar: React.FC<SidebarProps> = ({
  isOpen,
  onClose,
  conversations,
  activeConversationId,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
  models,
  activeProvider,
  activeModel,
  onSelectModel,
  health,
  stats,
}) => {
  const [showStatsModal, setShowStatsModal] = useState(false);

  // Group models by provider
  const groqModels = models.filter((m) => m.provider === 'groq');

  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div
          onClick={onClose}
          className="fixed inset-0 bg-black/40 backdrop-blur-sm z-30 md:hidden transition-opacity"
        />
      )}

      {/* Sidebar Panel */}
      <aside
        className={`fixed md:static inset-y-0 left-0 w-72 sm:w-80 bg-white border-r border-slate-200 z-40 flex flex-col transform transition-transform duration-300 ease-in-out ${
          isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
        }`}
      >
        {/* Brand & Close */}
        <div className="p-4 border-b border-slate-200 flex items-center justify-between bg-slate-50/50">
          <div>
            <span className="text-xs font-bold text-christ-maroon tracking-wider uppercase">
              CHRIST UNIVERSITY
            </span>
            <p className="text-sm font-bold text-slate-800">Student Support Helpdesk</p>
          </div>
          <button
            onClick={onClose}
            className="md:hidden p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-200"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* New Chat Button */}
        <div className="p-3.5 border-b border-slate-100">
          <button
            onClick={onNewConversation}
            className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl bg-christ-maroon text-white font-medium text-xs hover:bg-christ-dark transition shadow-sm"
          >
            <Plus className="w-4 h-4" />
            <span>New Conversation</span>
          </button>
        </div>

        {/* Model & Provider Switcher */}
        <div className="p-3.5 border-b border-slate-100 bg-slate-50/30 space-y-2">
          <label className="text-[11px] font-bold text-slate-500 uppercase tracking-wider flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-christ-maroon" />
            <span>LLM Inference Engine</span>
          </label>

          <select
            value={`${activeProvider}::${activeModel}`}
            onChange={(e) => {
              const [prov, mod] = e.target.value.split('::');
              onSelectModel(prov, mod);
            }}
            className="w-full text-xs bg-white border border-slate-300 rounded-lg p-2 text-slate-800 focus:outline-none focus:ring-2 focus:ring-christ-maroon/20 focus:border-christ-maroon font-medium"
          >
            {groqModels.map((m) => (
              <option key={`groq::${m.id}`} value={`groq::${m.id}`}>
                {m.id} {m.is_available ? '✓ (Online)' : '(Key required)'}
              </option>
            ))}
          </select>

          <div className="p-2 rounded-lg bg-emerald-50 border border-emerald-200 text-[11px] text-emerald-800 flex items-center gap-1.5">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 flex-shrink-0" />
            <span>Connected to Groq Cloud API for ultra-fast RAG inference.</span>
          </div>
        </div>

        {/* Conversation History */}
        <div className="flex-1 overflow-y-auto p-3 space-y-1">
          <div className="px-2 py-1 text-[11px] font-bold text-slate-400 uppercase tracking-wider">
            Conversations
          </div>
          {conversations.length === 0 ? (
            <p className="px-2 py-4 text-xs text-slate-400 italic text-center">
              No conversations yet
            </p>
          ) : (
            conversations.map((conv) => {
              const isActive = conv.id === activeConversationId;
              return (
                <div
                  key={conv.id}
                  onClick={() => onSelectConversation(conv.id)}
                  className={`group flex items-center justify-between p-2.5 rounded-xl cursor-pointer text-xs transition ${
                    isActive
                      ? 'bg-christ-light text-christ-maroon font-semibold shadow-xs border border-christ-maroon/20'
                      : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
                  }`}
                >
                  <div className="flex items-center gap-2 truncate">
                    <MessageSquare className="w-3.5 h-3.5 flex-shrink-0 text-christ-maroon/70" />
                    <span className="truncate">{conv.title}</span>
                  </div>

                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteConversation(conv.id);
                    }}
                    className="opacity-0 group-hover:opacity-100 text-slate-400 hover:text-rose-600 p-1 rounded transition"
                    title="Delete Conversation"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              );
            })
          )}
        </div>

        {/* Knowledge Base Status & LLMOps Button */}
        <div className="p-3 border-t border-slate-200 bg-slate-50 text-[11px] space-y-2">
          <div className="flex items-center justify-between text-slate-600">
            <span className="flex items-center gap-1.5 font-medium">
              <Database className="w-3.5 h-3.5 text-christ-maroon" />
              <span>Knowledge Base:</span>
            </span>
            <span className="font-mono font-bold text-slate-800">
              {health?.chunk_count || 379} chunks
            </span>
          </div>

          <div className="text-[10px] text-slate-400">
            MiniLM-L6-v2 Embeddings • FAISS IP Cosine Search
          </div>

          <button
            onClick={() => setShowStatsModal(true)}
            className="w-full flex items-center justify-center gap-1.5 py-1.5 rounded-lg border border-slate-300 text-slate-700 bg-white hover:bg-slate-50 transition text-xs font-medium"
          >
            <BarChart2 className="w-3.5 h-3.5 text-christ-gold" />
            <span>LLMOps Request Metrics</span>
          </button>
        </div>
      </aside>

      {/* Monitoring Modal */}
      {showStatsModal && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl max-w-md w-full shadow-2xl border border-slate-200 overflow-hidden">
            <div className="p-4 bg-gradient-to-r from-christ-deep to-christ-maroon text-white flex items-center justify-between">
              <div className="flex items-center gap-2">
                <BarChart2 className="w-5 h-5 text-christ-gold" />
                <h3 className="font-bold text-sm">LLMOps Request Analytics</h3>
              </div>
              <button
                onClick={() => setShowStatsModal(false)}
                className="p-1 rounded-lg text-white/80 hover:text-white hover:bg-white/10"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="p-5 space-y-4 text-xs text-slate-700">
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                  <span className="text-slate-400 text-[10px] uppercase font-bold">Total Queries</span>
                  <p className="text-xl font-bold text-slate-800 mt-0.5">
                    {stats?.total_requests ?? 0}
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                  <span className="text-slate-400 text-[10px] uppercase font-bold">Avg Total Latency</span>
                  <p className="text-xl font-bold text-christ-maroon mt-0.5">
                    {stats?.avg_total_latency_ms ? `${(stats.avg_total_latency_ms / 1000).toFixed(2)}s` : '0s'}
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                  <span className="text-slate-400 text-[10px] uppercase font-bold">Retrieval Latency</span>
                  <p className="text-lg font-bold text-slate-700 mt-0.5">
                    {stats?.avg_retrieval_latency_ms ?? 0} ms
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                  <span className="text-slate-400 text-[10px] uppercase font-bold">Error Rate</span>
                  <p className="text-lg font-bold text-emerald-600 mt-0.5">
                    {stats?.error_rate_pct ?? 0}%
                  </p>
                </div>
              </div>

              <div className="p-3 rounded-xl bg-rose-50/50 border border-rose-100 flex items-start gap-2">
                <Info className="w-4 h-4 text-christ-maroon flex-shrink-0 mt-0.5" />
                <p className="text-[11px] text-slate-600 leading-relaxed">
                  Metrics are logged to structured JSONL logs in <code className="bg-white px-1 py-0.5 rounded border border-slate-200 font-mono">backend/logs/llmops_requests.jsonl</code> following our experimental notebook methodology.
                </p>
              </div>
            </div>

            <div className="p-3 bg-slate-50 border-t border-slate-200 text-right">
              <button
                onClick={() => setShowStatsModal(false)}
                className="px-4 py-2 rounded-xl bg-slate-800 text-white font-medium text-xs hover:bg-slate-900 transition"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
