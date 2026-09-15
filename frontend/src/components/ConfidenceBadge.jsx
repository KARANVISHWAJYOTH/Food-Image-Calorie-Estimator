import React from 'react';
import { ShieldCheck, AlertTriangle, Sparkles, CheckCircle2 } from 'lucide-react';

export default function ConfidenceBadge({ score = 0.94, percentage = null, size = 'normal' }) {
  const pct = percentage !== null ? percentage : Math.round(score * 1000) / 10;

  let colorClass = 'bg-emerald-50 text-emerald-700 border-emerald-200/80';
  let Icon = ShieldCheck;
  let statusText = 'High Confidence';
  let dotColor = 'bg-emerald-500';

  if (pct < 65) {
    colorClass = 'bg-rose-50 text-rose-700 border-rose-200/80';
    Icon = AlertTriangle;
    statusText = 'Low Confidence';
    dotColor = 'bg-rose-500';
  } else if (pct < 85) {
    colorClass = 'bg-amber-50 text-amber-700 border-amber-200/80';
    Icon = Sparkles;
    statusText = 'Moderate Confidence';
    dotColor = 'bg-amber-500';
  }

  if (size === 'sm') {
    return (
      <span
        className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-bold border ${colorClass}`}
      >
        <span className={`w-1.5 h-1.5 rounded-full ${dotColor}`} />
        <span>{pct}%</span>
      </span>
    );
  }

  return (
    <div
      className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-bold border ${colorClass} shadow-sm`}
    >
      <Icon className="w-4 h-4" />
      <span>{pct}% confidence</span>
      <span className="opacity-60">•</span>
      <span className="font-semibold text-[11px]">{statusText}</span>
    </div>
  );
}
