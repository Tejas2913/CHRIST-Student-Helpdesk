import React from 'react';
import ReactMarkdown from 'react-markdown';
import { User, GraduationCap, Zap, AlertTriangle } from 'lucide-react';
import type { Message } from '../types';
import { CitationCard } from './CitationCard';

interface ChatMessageProps {
  message: Message;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const isUser = message.role === 'user';
  const timeFormatted = new Date(message.timestamp).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });

  return (
    <div
      className={`py-4 px-4 sm:px-6 flex gap-3.5 sm:gap-4 transition-colors ${
        isUser ? 'bg-white' : 'bg-slate-50/80 border-y border-slate-100'
      }`}
    >
      {/* Avatar */}
      <div className="flex-shrink-0">
        {isUser ? (
          <div className="w-8 h-8 rounded-full bg-slate-200 text-slate-700 flex items-center justify-center shadow-sm">
            <User className="w-4 h-4" />
          </div>
        ) : (
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-christ-maroon to-christ-dark text-white flex items-center justify-center shadow-md border border-christ-gold/40">
            <GraduationCap className="w-4 h-4 text-christ-gold" />
          </div>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span className="font-semibold text-xs text-slate-800">
            {isUser ? 'You' : 'CHRIST StudentAssist'}
          </span>
          <span className="text-[11px] text-slate-400 font-mono">
            {timeFormatted}
          </span>
          {message.model && !isUser && (
            <span className="hidden sm:inline-flex items-center gap-1 text-[10px] bg-slate-200/70 text-slate-600 px-2 py-0.5 rounded-full font-mono">
              <Zap className="w-3 h-3 text-christ-gold" />
              {message.model}
              {message.latency?.total_ms && (
                <span>• {(message.latency.total_ms / 1000).toFixed(2)}s</span>
              )}
            </span>
          )}
        </div>

        {/* Message Body */}
        {message.error ? (
          <div className="p-3 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-sm flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 text-rose-600 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold">Request Error</p>
              <p className="text-xs text-rose-700 mt-0.5">{message.content}</p>
            </div>
          </div>
        ) : isUser ? (
          <p className="text-sm text-slate-800 whitespace-pre-wrap leading-relaxed">
            {message.content}
          </p>
        ) : (
          <div className="prose prose-sm max-w-none text-slate-800 leading-relaxed font-sans prose-headings:text-christ-maroon prose-headings:font-bold prose-headings:mb-2 prose-p:my-2 prose-ul:my-2 prose-li:my-0.5 prose-strong:text-slate-900">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        )}

        {/* Citations if available */}
        {message.sources && message.sources.length > 0 && (
          <CitationCard sources={message.sources} />
        )}
      </div>
    </div>
  );
};
