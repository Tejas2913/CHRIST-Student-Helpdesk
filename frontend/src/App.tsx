import React, { useState, useEffect, useRef } from 'react';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { ChatMessage } from './components/ChatMessage';
import { ChatInput } from './components/ChatInput';
import { QuickPrompts } from './components/QuickPrompts';
import { AnalyticsDashboard } from './pages/AnalyticsDashboard';
import ExplainabilityLab from './pages/ExplainabilityLab';
import type { Conversation, Message, ModelInfo, HealthStatus, MonitoringStats } from './types';
import { api } from './services/api';
import { Sparkles, GraduationCap } from 'lucide-react';

const STORAGE_KEY = 'christ_student_helpdesk_convs_v1';

const WELCOME_TEXT =
  "Hello! Welcome to CHRIST StudentAssist. I can help you with university regulations, attendance, examinations, leave policies and student support services. What would you like to know?";

function createInitialConversation(): Conversation {
  const id = Date.now().toString();
  return {
    id,
    title: 'New Student Query',
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: [
      {
        id: 'msg-welcome',
        role: 'assistant',
        content: WELCOME_TEXT,
        timestamp: Date.now(),
      },
    ],
  };
}

export const App: React.FC = () => {
  const [conversations, setConversations] = useState<Conversation[]>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) return JSON.parse(saved);
    } catch {
      // fallback
    }
    return [createInitialConversation()];
  });

  const [activeConvId, setActiveConvId] = useState<string>(() => {
    return conversations[0]?.id || Date.now().toString();
  });

  const [activeView, setActiveView] = useState<'chat' | 'analytics' | 'xai'>('chat');
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [activeProvider, setActiveProvider] = useState<string>('groq');
  const [activeModel, setActiveModel] = useState<string>('qwen/qwen3.8-27b');
  const [stats, setStats] = useState<MonitoringStats | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Sync conversations to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
    } catch (e) {
      console.error('Storage error:', e);
    }
  }, [conversations]);

  // Load backend health and model metadata on mount
  useEffect(() => {
    async function initBackend() {
      try {
        const h = await api.getHealth();
        setHealth(h);
        if (h.active_provider === 'groq') {
          setActiveProvider('groq');
          setActiveModel(h.active_model);
        } else {
          setActiveProvider('groq');
          setActiveModel('qwen/qwen3.8-27b');
          api.selectModel('groq', 'qwen/qwen3.8-27b').catch(() => {});
        }
      } catch (err) {
        console.warn('Backend not responding yet:', err);
      }

      try {
        const m = await api.getModels();
        setModels(m);
      } catch (err) {
        console.warn('Could not load models list:', err);
      }

      try {
        const st = await api.getMonitoringStats();
        setStats(st);
      } catch {
        // quiet fallback
      }
    }

    initBackend();
  }, []);

  const activeConversation =
    conversations.find((c) => c.id === activeConvId) || conversations[0];

  // Auto-scroll to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activeConversation?.messages]);

  const handleSelectModel = async (provider: string, model: string) => {
    setActiveProvider(provider);
    setActiveModel(model);
    try {
      await api.selectModel(provider, model);
    } catch (err) {
      console.error('Failed to notify backend of model change:', err);
    }
  };

  const handleNewConversation = () => {
    const newConv = createInitialConversation();
    setConversations((prev) => [newConv, ...prev]);
    setActiveConvId(newConv.id);
    setIsSidebarOpen(false);
  };

  const handleDeleteConversation = (id: string) => {
    setConversations((prev) => {
      const filtered = prev.filter((c) => c.id !== id);
      if (filtered.length === 0) {
        const fresh = createInitialConversation();
        setActiveConvId(fresh.id);
        return [fresh];
      }
      if (activeConvId === id) {
        setActiveConvId(filtered[0].id);
      }
      return filtered;
    });
  };

  const handleSendMessage = async (text: string) => {
    if (!text.trim() || isLoading) return;

    const userMsg: Message = {
      id: `usr-${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: Date.now(),
    };

    // Update conversation with user message
    setConversations((prev) =>
      prev.map((c) => {
        if (c.id === activeConvId) {
          const isFirstUserMsg = c.messages.filter((m) => m.role === 'user').length === 0;
          return {
            ...c,
            title: isFirstUserMsg ? (text.length > 28 ? `${text.slice(0, 28)}...` : text) : c.title,
            messages: [...c.messages, userMsg],
            updatedAt: Date.now(),
          };
        }
        return c;
      })
    );

    setIsLoading(true);

    try {
      const history = (activeConversation?.messages || [])
        .slice(-6)
        .map((m) => ({ role: m.role, content: m.content }));

      const res = await api.sendMessage({
        message: text,
        provider: activeProvider,
        model: activeModel,
        conversation_history: history,
      });

      const assistantMsg: Message = {
        id: `asst-${Date.now()}`,
        role: 'assistant',
        content: res.answer,
        timestamp: Date.now(),
        sources: res.sources,
        provider: res.provider,
        model: res.model,
        latency: res.latency,
      };

      setConversations((prev) =>
        prev.map((c) => {
          if (c.id === activeConvId) {
            return {
              ...c,
              messages: [...c.messages, assistantMsg],
              updatedAt: Date.now(),
            };
          }
          return c;
        })
      );

      // Refresh stats
      api.getMonitoringStats().then(setStats).catch(() => {});
    } catch (err: any) {
      const errorMsg: Message = {
        id: `err-${Date.now()}`,
        role: 'assistant',
        content: err.message || 'An unexpected error occurred while communicating with the backend.',
        timestamp: Date.now(),
        error: 'INFERENCE_ERROR',
      };

      setConversations((prev) =>
        prev.map((c) => {
          if (c.id === activeConvId) {
            return {
              ...c,
              messages: [...c.messages, errorMsg],
              updatedAt: Date.now(),
            };
          }
          return c;
        })
      );
    } finally {
      setIsLoading(false);
    }
  };

  const showQuickPrompts = (activeConversation?.messages?.length || 0) <= 1;

  return (
    <div className="flex h-screen bg-slate-50 overflow-hidden font-sans">
      {/* Sidebar */}
      <Sidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        conversations={conversations}
        activeConversationId={activeConvId}
        onSelectConversation={(id) => {
          setActiveConvId(id);
          setIsSidebarOpen(false);
        }}
        onNewConversation={handleNewConversation}
        onDeleteConversation={handleDeleteConversation}
        models={models}
        activeProvider={activeProvider}
        activeModel={activeModel}
        onSelectModel={handleSelectModel}
        health={health}
        stats={stats}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden">
        {/* Navigation Header */}
        <Header
          health={health}
          activeProvider={activeProvider}
          activeModel={activeModel}
          isModelAvailable={
            models.find((m) => m.provider === activeProvider && m.id === activeModel)?.is_available ??
            (activeProvider === 'groq')
          }
          onToggleSidebar={() => setIsSidebarOpen(true)}
          activeView={activeView}
          onViewChange={setActiveView}
        />

        {activeView === 'analytics' ? (
          <AnalyticsDashboard />
        ) : activeView === 'xai' ? (
          <div className="flex-1 overflow-y-auto p-4 sm:p-6 max-w-7xl mx-auto w-full">
            <ExplainabilityLab />
          </div>
        ) : (
          <>
            {/* Chat Messages Feed */}
            <div className="flex-1 overflow-y-auto">
              {/* Welcome Banner Card */}
              <div className="max-w-4xl mx-auto px-4 sm:px-6 pt-5 pb-2">
                <div className="p-4 sm:p-5 rounded-2xl bg-gradient-to-r from-christ-deep to-christ-maroon text-white shadow-sm flex items-start gap-3.5 border-b-2 border-christ-gold">
                  <div className="p-2.5 rounded-xl bg-white/10 text-christ-gold border border-white/20">
                    <GraduationCap className="w-6 h-6" />
                  </div>
                  <div>
                    <h2 className="text-base sm:text-lg font-bold flex items-center gap-2">
                      Official CHRIST University Student Helpdesk
                    </h2>
                    <p className="text-xs text-rose-100/90 mt-1 leading-relaxed">
                      Answers to student regulations, attendance policies, examination guidelines, and grievances grounded strictly in the 2026-27 Student Handbook.
                    </p>
                  </div>
                </div>
              </div>

              {/* Quick Prompts if no user messages yet */}
              {showQuickPrompts && (
                <QuickPrompts
                  onSelectPrompt={handleSendMessage}
                  disabled={isLoading}
                />
              )}

              {/* Conversation Messages */}
              <div className="max-w-4xl mx-auto divide-y divide-slate-100 pb-6">
                {activeConversation?.messages.map((msg) => (
                  <ChatMessage key={msg.id} message={msg} />
                ))}

                {/* Loading Indicator */}
                {isLoading && (
                  <div className="py-4 px-4 sm:px-6 flex gap-3.5 bg-slate-50/80 border-y border-slate-100 animate-pulse">
                    <div className="w-8 h-8 rounded-full bg-christ-maroon text-white flex items-center justify-center">
                      <Sparkles className="w-4 h-4 text-christ-gold animate-spin" />
                    </div>
                    <div className="flex-1 space-y-2 py-1">
                      <div className="h-3 bg-slate-200 rounded w-1/4"></div>
                      <div className="h-3 bg-slate-200 rounded w-3/4"></div>
                      <div className="h-3 bg-slate-200 rounded w-1/2"></div>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>
            </div>

            {/* Input Bar */}
            <ChatInput
              onSend={handleSendMessage}
              isLoading={isLoading}
            />
          </>
        )}
      </div>
    </div>
  );
};

export default App;
