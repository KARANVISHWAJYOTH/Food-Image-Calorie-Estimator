import React, { useEffect } from 'react';
import { CheckCircle2, AlertCircle, Info, X } from 'lucide-react';

export default function Toast({ message, type = 'success', onClose, duration = 3500 }) {
  useEffect(() => {
    if (!message) return;
    const timer = setTimeout(() => {
      onClose();
    }, duration);
    return () => clearTimeout(timer);
  }, [message, duration, onClose]);

  if (!message) return null;

  const typeConfigs = {
    success: {
      bg: 'bg-emerald-900 text-white border-emerald-700',
      icon: CheckCircle2,
      iconColor: 'text-emerald-400',
    },
    error: {
      bg: 'bg-rose-900 text-white border-rose-700',
      icon: AlertCircle,
      iconColor: 'text-rose-400',
    },
    info: {
      bg: 'bg-slate-900 text-white border-slate-700',
      icon: Info,
      iconColor: 'text-brand-400',
    },
  };

  const config = typeConfigs[type] || typeConfigs.info;
  const Icon = config.icon;

  return (
    <div className="fixed bottom-6 right-6 z-50 animate-bounce-in max-w-sm">
      <div
        className={`flex items-center gap-3 px-4 py-3 rounded-2xl shadow-elevated border ${config.bg} backdrop-blur-md`}
      >
        <Icon className={`w-5 h-5 flex-shrink-0 ${config.iconColor}`} />
        <p className="text-xs font-semibold flex-1">{message}</p>
        <button
          onClick={onClose}
          className="p-1 rounded-lg text-slate-400 hover:text-white transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
