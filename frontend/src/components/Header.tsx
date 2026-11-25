import React from 'react';
import { GraduationCap, Sparkles, CheckCircle2, AlertCircle, AlertTriangle, Menu, MessageSquare, BarChart3, Microscope } from 'lucide-react';
import type { HealthStatus } from '../types';

interface HeaderProps {
  health: HealthStatus | null;
  activeProvider: string;
  activeModel: string;
  isModelAvailable?: boolean;
  onToggleSidebar: () => void;
  activeView: 'chat' | 'analytics' | 'xai';
  onViewChange: (view: 'chat' | 'analytics' | 'xai') => void;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  activeProvider,
  activeModel,
  isModelAvailable = true,
  onToggleSidebar,
  activeView,
  onViewChange,
}) => {
  const isHealthy = health?.status === 'healthy';

  return (
    <header className="bg-gradient-to-r from-christ-deep via-christ-maroon to-christ-dark text-white border-b-2 border-christ-gold shadow-md sticky top-0 z-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        
        {/* Left: Mobile Toggle & University Branding */}
        <div className="flex items-center space-x-3">
          <button
            onClick={onToggleSidebar}
            className="md:hidden p-2 rounded-lg hover:bg-white/10 transition"
            aria-label="Toggle Navigation"
          >
            <Menu className="w-5 h-5 text-white" />
          </button>

          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-white/10 border border-white/20 flex items-center justify-center shadow-inner">
              <GraduationCap className="w-6 h-6 text-christ-gold" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-1.5">
                  CHRIST StudentAssist
                </h1>
                <span className="hidden sm:inline-block px-2 py-0.5 text-[10px] uppercase tracking-wider font-semibold bg-christ-gold/20 text-christ-gold border border-christ-gold/40 rounded-full">
                  Official Helpdesk
                </span>
              </div>
              <p className="text-xs text-rose-100/80 hidden sm:block">
                CHRIST (Deemed to be University) • Excellence and Service
              </p>
            </div>
          </div>
        </div>

        {/* Center: Navigation Tabs (Chat | Analytics Dashboard | Explainability Lab) */}
        <div className="flex items-center bg-black/25 p-1 rounded-xl border border-white/10 shadow-inner">
          <button
            onClick={() => onViewChange('chat')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
              activeView === 'chat'
                ? 'bg-white text-christ-deep shadow'
                : 'text-white/80 hover:text-white hover:bg-white/10'
            }`}
          >
            <MessageSquare className="w-3.5 h-3.5" />
            <span>Chat</span>
          </button>
          <button
            onClick={() => onViewChange('analytics')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
              activeView === 'analytics'
                ? 'bg-white text-christ-deep shadow'
                : 'text-white/80 hover:text-white hover:bg-white/10'
            }`}
          >
            <BarChart3 className="w-3.5 h-3.5" />
            <span>Analytics Dashboard</span>
          </button>
          <button
            onClick={() => onViewChange('xai')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
              activeView === 'xai'
                ? 'bg-white text-christ-deep shadow'
                : 'text-white/80 hover:text-white hover:bg-white/10'
            }`}
          >
            <Microscope className="w-3.5 h-3.5" />
            <span>Explainability Lab</span>
          </button>
        </div>

        {/* Right: Active Model & System Health Badges */}
        <div className="flex items-center space-x-2 sm:space-x-3">
          {/* Active Model Indicator - reflects true availability */}
          {isModelAvailable ? (
            <div className="hidden sm:flex items-center space-x-1.5 bg-black/30 backdrop-blur-sm border border-emerald-500/40 px-3 py-1 rounded-full text-xs text-emerald-200">
              <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
              <span className="font-medium capitalize">{activeProvider}:</span>
              <span className="text-white font-mono font-semibold truncate max-w-[140px]">
                {activeModel}
              </span>
              <span className="text-[10px] bg-emerald-500/20 text-emerald-300 px-1.5 py-0.5 rounded font-semibold uppercase">
                Active
              </span>
            </div>
          ) : (
            <div className="hidden sm:flex items-center space-x-1.5 bg-amber-950/60 backdrop-blur-sm border border-amber-500/60 px-3 py-1 rounded-full text-xs text-amber-200">
              <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
              <span className="font-medium capitalize">{activeProvider}:</span>
              <span className="text-amber-100 font-mono font-semibold truncate max-w-[140px]">
                {activeModel}
              </span>
              <span className="text-[10px] bg-amber-500/30 text-amber-300 px-1.5 py-0.5 rounded font-bold uppercase tracking-wider">
                Offline
              </span>
            </div>
          )}

          {/* Backend Knowledge Base Status */}
          <div
            className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${
              isHealthy
                ? 'bg-emerald-500/20 text-emerald-200 border-emerald-500/40'
                : 'bg-rose-500/20 text-rose-200 border-rose-500/40'
            }`}
          >
            {isHealthy ? (
              <>
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                <span className="hidden md:inline">Handbook Online</span>
              </>
            ) : (
              <>
                <AlertCircle className="w-3.5 h-3.5 text-rose-400" />
                <span className="hidden md:inline">Connecting...</span>
              </>
            )}
          </div>
        </div>

      </div>
    </header>
  );
};
