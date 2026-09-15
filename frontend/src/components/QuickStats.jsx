import React from 'react';
import { Utensils, Flame, Beef, Zap, TrendingUp } from 'lucide-react';
import { useHistory } from '../context/HistoryContext';

export default function QuickStats() {
  const { getStats } = useHistory();
  const stats = getStats();

  const cards = [
    {
      title: 'Total Meals Analyzed',
      value: stats.totalMeals || 0,
      unit: 'meals',
      trend: '+12% this month',
      trendPositive: true,
      icon: Utensils,
      color: 'brand',
      bgLight: 'bg-emerald-50',
      textColor: 'text-emerald-600',
      borderColor: 'border-emerald-100',
    },
    {
      title: 'Average Calories',
      value: stats.avgCalories || 0,
      unit: 'kcal / meal',
      trend: 'Target: 600-750 kcal',
      trendPositive: true,
      icon: Flame,
      color: 'amber',
      bgLight: 'bg-amber-50',
      textColor: 'text-amber-600',
      borderColor: 'border-amber-100',
    },
    {
      title: 'Average Protein',
      value: stats.avgProtein || 0,
      unit: 'g / meal',
      trend: 'High Protein Ratio',
      trendPositive: true,
      icon: Beef,
      color: 'cyan',
      bgLight: 'bg-cyan-50',
      textColor: 'text-cyan-600',
      borderColor: 'border-cyan-100',
    },
    {
      title: 'Analyses This Week',
      value: stats.thisWeekCount || 0,
      unit: 'scans',
      trend: 'Active tracking',
      trendPositive: true,
      icon: Zap,
      color: 'purple',
      bgLight: 'bg-purple-50',
      textColor: 'text-purple-600',
      borderColor: 'border-purple-100',
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-5">
      {cards.map((card, idx) => {
        const Icon = card.icon;
        return (
          <div
            key={idx}
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

            <div className="flex items-center gap-1.5 text-xs font-medium text-slate-500 pt-2 border-t border-slate-100">
              <TrendingUp className="w-3.5 h-3.5 text-brand-600" />
              <span>{card.trend}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
