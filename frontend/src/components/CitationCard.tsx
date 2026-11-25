import React, { useState } from 'react';
import { BookOpen, ChevronDown, ChevronUp, Copy, Check } from 'lucide-react';
import type { SourceCitation } from '../types';

interface CitationCardProps {
  sources: SourceCitation[];
}

export const CitationCard: React.FC<CitationCardProps> = ({ sources }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null);

  if (!sources || sources.length === 0) return null;

  const handleCopy = (text: string, idx: number) => {
    navigator.clipboard.writeText(text);
    setCopiedIdx(idx);
    setTimeout(() => setCopiedIdx(null), 2000);
  };

  return (
    <div className="mt-3 border border-slate-200 rounded-xl overflow-hidden bg-slate-50/75 text-xs text-slate-700">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-3.5 py-2.5 bg-slate-100/80 hover:bg-slate-200/60 transition text-slate-800 font-medium"
      >
        <span className="flex items-center gap-2">
          <BookOpen className="w-3.5 h-3.5 text-christ-maroon" />
          <span>Verified Handbook Sources ({sources.length} citations)</span>
        </span>
        {isOpen ? (
          <ChevronUp className="w-4 h-4 text-slate-500" />
        ) : (
          <ChevronDown className="w-4 h-4 text-slate-500" />
        )}
      </button>

      {isOpen && (
        <div className="p-3 space-y-2.5 divide-y divide-slate-200/60">
          {sources.map((src, idx) => (
            <div key={idx} className={idx > 0 ? 'pt-2.5' : ''}>
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-2">
                  <span className="px-2 py-0.5 rounded font-bold bg-christ-maroon text-white text-[10px] tracking-wide">
                    Page {src.page}
                  </span>
                  <span className="text-[11px] text-slate-500 font-mono">
                    Relevance: {(src.score * 100).toFixed(1)}%
                  </span>
                </div>
                <button
                  onClick={() => handleCopy(src.text, idx)}
                  className="p-1 rounded text-slate-400 hover:text-slate-700 hover:bg-slate-200/60 transition"
                  title="Copy handbook excerpt"
                >
                  {copiedIdx === idx ? (
                    <Check className="w-3.5 h-3.5 text-emerald-600" />
                  ) : (
                    <Copy className="w-3.5 h-3.5" />
                  )}
                </button>
              </div>
              <p className="text-[11px] text-slate-600 italic bg-white p-2.5 rounded-lg border border-slate-200/70 leading-relaxed font-sans">
                "{src.text}"
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
