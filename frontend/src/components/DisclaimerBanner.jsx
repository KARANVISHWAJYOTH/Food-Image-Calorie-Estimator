import React from 'react';
import { Info, ShieldAlert } from 'lucide-react';

export default function DisclaimerBanner({ className = '' }) {
  return (
    <div
      className={`rounded-2xl p-4 bg-slate-100/80 border border-slate-200/80 text-slate-600 text-xs flex items-start sm:items-center gap-3 ${className}`}
    >
      <Info className="w-4 h-4 text-slate-400 flex-shrink-0 mt-0.5 sm:mt-0" />
      <div className="flex-1 leading-relaxed">
        <p className="font-semibold text-slate-700">
          AI nutrition estimates are approximate and should not be considered medical or dietary advice.
        </p>
        <p className="text-[11px] text-slate-500 mt-0.5">
          Nutrition values are AI-generated estimates and may vary depending on ingredients, preparation method, and portion size.
        </p>
      </div>
    </div>
  );
}
