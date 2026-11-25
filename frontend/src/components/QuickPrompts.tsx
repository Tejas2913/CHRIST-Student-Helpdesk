import React from 'react';
import { CalendarCheck, FileText, ClipboardList, ShieldAlert, MessageSquareWarning } from 'lucide-react';

interface QuickPromptsProps {
  onSelectPrompt: (prompt: string) => void;
  disabled?: boolean;
}

export const QuickPrompts: React.FC<QuickPromptsProps> = ({ onSelectPrompt, disabled }) => {
  const prompts = [
    {
      title: 'Attendance Requirements',
      query: 'What is the minimum attendance requirement to write end semester examinations?',
      icon: CalendarCheck,
      desc: '85% rule, detention, & condonation policies',
    },
    {
      title: 'Examination Eligibility',
      query: 'What are the eligibility criteria and regulations for semester examinations?',
      icon: FileText,
      desc: 'Hall ticket, MSE/ESE criteria, & revaluation',
    },
    {
      title: 'Leave Application',
      query: 'What is the procedure for applying for leave of absence or medical leave?',
      icon: ClipboardList,
      desc: 'Student Services forms & 2-week absence rule',
    },
    {
      title: 'Code of Conduct',
      query: 'What is the student code of conduct and what acts are classified as misconduct?',
      icon: ShieldAlert,
      desc: 'Disciplinary regulations & dress code',
    },
    {
      title: 'Grievance Redressal',
      query: 'How does the student grievance redressal mechanism work at CHRIST?',
      icon: MessageSquareWarning,
      desc: 'Student complaints & Redressal Committee',
    },
  ];

  return (
    <div className="py-6 px-4 sm:px-8 max-w-4xl mx-auto">
      <div className="text-center mb-6">
        <h2 className="text-lg font-bold text-slate-900">
          Frequently Consulted Handbook Policies
        </h2>
        <p className="text-xs text-slate-500 mt-1">
          Select a topic below to inspect official university regulations and cited handbook pages.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {prompts.map((p, idx) => {
          const Icon = p.icon;
          return (
            <button
              key={idx}
              onClick={() => onSelectPrompt(p.query)}
              disabled={disabled}
              className="flex items-start text-left p-3.5 rounded-xl border border-slate-200/90 bg-white hover:border-christ-maroon hover:shadow-sm transition group disabled:opacity-60 disabled:cursor-not-allowed"
            >
              <div className="p-2 rounded-lg bg-christ-light text-christ-maroon mr-3 group-hover:bg-christ-maroon group-hover:text-white transition">
                <Icon className="w-4 h-4" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs font-bold text-slate-800 group-hover:text-christ-maroon transition">
                  {p.title}
                </p>
                <p className="text-[11px] text-slate-500 line-clamp-1 mt-0.5">
                  {p.desc}
                </p>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};
