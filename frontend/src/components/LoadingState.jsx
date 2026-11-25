import React, { useState, useEffect } from 'react';
import { Cpu, CheckCircle2, Loader2 } from 'lucide-react';

export default function LoadingState({ message = 'Analyzing your image…' }) {
  const [step, setStep] = useState(0);

  const steps = [
    'Resizing and normalising the image',
    'Running the ResNet-50 model',
    'Scoring the predicted food class',
    'Reading the nutrition outputs',
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setStep((prev) => (prev < steps.length - 1 ? prev + 1 : prev));
    }, 450);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="bg-white rounded-3xl p-8 sm:p-12 border border-slate-200/80 shadow-card text-center max-w-xl mx-auto space-y-8">
      <div className="relative w-24 h-24 mx-auto flex items-center justify-center">
        <div className="absolute inset-0 rounded-full bg-brand-500/20 animate-ping" />
        <div className="w-20 h-20 rounded-2xl bg-gradient-to-tr from-brand-600 to-emerald-400 flex items-center justify-center text-white shadow-glow-brand z-10">
          <Cpu className="w-10 h-10 animate-spin" style={{ animationDuration: '6s' }} />
        </div>
      </div>

      <div className="space-y-2">
        <h3 className="text-2xl font-extrabold text-slate-900 tracking-tight flex items-center justify-center gap-2">
          <span>{message}</span>
        </h3>
        <p className="text-sm text-slate-500 font-medium max-w-sm mx-auto">
          The photo is being classified and its calorie and macronutrient outputs computed.
        </p>
      </div>

      {/* Pipeline Step Progress */}
      <div className="space-y-3 text-left max-w-md mx-auto pt-2">
        {steps.map((label, idx) => {
          const isDone = idx < step;
          const isCurrent = idx === step;
          return (
            <div
              key={idx}
              className={`flex items-center gap-3 p-3 rounded-xl transition-all duration-300 ${
                isCurrent
                  ? 'bg-brand-50 border border-brand-200 text-brand-900 font-bold'
                  : isDone
                  ? 'bg-slate-50 text-slate-700 font-medium'
                  : 'text-slate-400 font-medium'
              }`}
            >
              {isDone ? (
                <CheckCircle2 className="w-5 h-5 text-emerald-600 flex-shrink-0" />
              ) : isCurrent ? (
                <Loader2 className="w-5 h-5 text-brand-600 animate-spin flex-shrink-0" />
              ) : (
                <div className="w-5 h-5 rounded-full border border-slate-300 flex-shrink-0" />
              )}
              <span className="text-xs sm:text-sm">{label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
