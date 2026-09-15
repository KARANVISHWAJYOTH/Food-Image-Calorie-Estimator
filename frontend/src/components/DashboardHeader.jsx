import React from 'react';
import { Sparkles, Calendar, PlusCircle } from 'lucide-react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function DashboardHeader({ title, subtitle, showAction = true }) {
  const { user } = useAuth();

  const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return 'Good morning';
    if (hour < 17) return 'Good afternoon';
    return 'Good evening';
  };

  const formattedDate = new Intl.DateTimeFormat('en-US', {
    weekday: 'long',
    month: 'short',
    day: 'numeric',
  }).format(new Date());

  return (
    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-slate-200/80">
      <div>
        <div className="flex items-center gap-2 text-xs font-semibold text-brand-700 uppercase tracking-wider mb-1">
          <Calendar className="w-3.5 h-3.5" />
          <span>{formattedDate}</span>
          <span>•</span>
          <span className="flex items-center gap-1 text-slate-500 font-medium">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
            AI Ready
          </span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
          {title || `${getGreeting()}, ${user?.name || 'Karan'} 👋`}
        </h1>
        <p className="text-sm text-slate-500 mt-1 font-medium">
          {subtitle || 'Upload a food image to get your instant AI nutrition breakdown and calorie estimation.'}
        </p>
      </div>

      {showAction && (
        <div className="flex items-center gap-3">
          <Link
            to="/analyze"
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-brand-600 to-emerald-500 hover:from-brand-700 hover:to-emerald-600 text-white text-sm font-semibold shadow-md shadow-brand-500/25 transition-all hover:scale-[1.02] active:scale-[0.98]"
          >
            <Sparkles className="w-4 h-4" />
            <span>Analyze Food</span>
          </Link>
        </div>
      )}
    </div>
  );
}
