import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2 } from 'lucide-react';

interface ChatInputProps {
  onSend: (message: string) => void;
  isLoading: boolean;
  disabled?: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({ onSend, isLoading, disabled }) => {
  const [text, setText] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`;
    }
  }, [text]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSubmit = () => {
    if (!text.trim() || isLoading || disabled) return;
    onSend(text.trim());
    setText('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  return (
    <div className="border-t border-slate-200 bg-white p-3 sm:p-4">
      <div className="max-w-4xl mx-auto">
        <div className="relative flex items-end border border-slate-300 rounded-2xl bg-white shadow-sm focus-within:border-christ-maroon focus-within:ring-2 focus-within:ring-christ-maroon/20 transition">
          <textarea
            ref={textareaRef}
            rows={1}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={disabled || isLoading}
            placeholder="Ask a question about attendance, exams, leave, discipline..."
            className="w-full resize-none bg-transparent px-4 py-3.5 text-sm text-slate-800 placeholder-slate-400 focus:outline-none disabled:bg-slate-50 disabled:cursor-not-allowed max-h-40"
          />

          <button
            onClick={handleSubmit}
            disabled={!text.trim() || isLoading || disabled}
            className="m-2 p-2.5 rounded-xl bg-christ-maroon text-white hover:bg-christ-dark transition disabled:opacity-40 disabled:cursor-not-allowed flex-shrink-0"
            aria-label="Send message"
          >
            {isLoading ? (
              <Loader2 className="w-4 h-4 animate-spin text-white" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </div>

        <div className="flex items-center justify-between mt-2 px-1 text-[11px] text-slate-400">
          <span>Official answers grounded in CHRIST Student Handbook 2026-27</span>
          <span className="hidden sm:inline">Press Enter to send, Shift+Enter for newline</span>
        </div>
      </div>
    </div>
  );
};
