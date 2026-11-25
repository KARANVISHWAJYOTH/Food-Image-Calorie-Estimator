import React from 'react';
import { Utensils, Flame, Beef, Zap } from 'lucide-react';
import { useHistory } from '../context/HistoryContext';

export default function QuickStats() {
  const { getStats } = useHistory();
  const stats = getStats();
  const hasData = stats.totalMeals > 0;

  const cards = [
    {
      title: 'Total Meals Analyzed',
      value: stats.totalMeals,
      unit: 'meals',
      detail: `${stats.thisWeekCount} in the last 7 days`,
      icon: Utensils,
      bgLight: 'bg-emerald-50',
      textColor: 'text-emerald-600',
    },
    {
      title: 'Average Calories',
      value: stats.avgCalories,
      unit: 'kcal / meal',
      detail: 'Per saved scan',
      icon: Flame,
      bgLight: 'bg-amber-50',
      textColor: 'text-amber-600',
    },
    {
      title: 'Average Protein',
      value: stats.avgProtein,
      unit: 'g / meal',
      detail: 'Per saved scan',
      icon: Beef,
      bgLight: 'bg-cyan-50',
      textColor: 'text-cyan-600',
    },
    {
      title: 'Analyses This Week',
      value: stats.thisWeekCount,
      unit: 'scans',
      detail: 'Last 7 days',
      icon: Zap,
      bgLight: 'bg-purple-50',
      textColor: 'text-purple-600',
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-5">
      {cards.map((card) => {
        const Icon = card.icon;
        return (
          <div
            key={card.title}
            className="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-soft hover:shadow-card transition-all duration-200 group"
          >
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-semibold text-slate-500">{card.title}</span>
              <div
                className={`w-10 h-10 rounded-xl ${card.bgLight} ${card.textColor} flex items-center justify-center group-hover:scale-110 transition-transform`}
              >
                <Icon className="w-5 h-5" />
              </div>
            </div>

            <div className="flex items-baseline gap-2 mb-2">
              <span className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
                {card.value}
              </span>
              <span className="text-xs font-semibold text-slate-500">{card.unit}</span>
            </div>

            <div className="text-xs font-medium text-slate-500 pt-2 border-t border-slate-100">
              {hasData ? card.detail : 'No saved scans yet'}
            </div>
          </div>
        );
      })}
    </div>
  );
}
