import React from 'react';
import { Flame, Beef, Wheat, Droplet } from 'lucide-react';

export default function NutritionCard({
  type = 'calories',
  value = 0,
  unit = 'kcal',
  dailyGoal = 2000,
  portionMultiplier = 1,
}) {
  const adjustedValue = Math.round(Number(value) * portionMultiplier * 10) / 10;
  const percentage = Math.min(100, Math.round((adjustedValue / dailyGoal) * 100));

  const configs = {
    calories: {
      label: 'Calories',
      unitLabel: 'kcal',
      icon: Flame,
      textColor: 'text-amber-600',
      bgColor: 'bg-amber-50',
      borderColor: 'border-amber-200/70',
      barColor: 'bg-gradient-to-r from-amber-400 to-amber-500',
      badgeBg: 'bg-amber-100 text-amber-800',
    },
    protein: {
      label: 'Protein',
      unitLabel: 'g',
      icon: Beef,
      textColor: 'text-cyan-600',
      bgColor: 'bg-cyan-50',
      borderColor: 'border-cyan-200/70',
      barColor: 'bg-gradient-to-r from-cyan-400 to-cyan-500',
      badgeBg: 'bg-cyan-100 text-cyan-800',
    },
    carbohydrates: {
      label: 'Carbohydrates',
      unitLabel: 'g',
      icon: Wheat,
      textColor: 'text-emerald-600',
      bgColor: 'bg-emerald-50',
      borderColor: 'border-emerald-200/70',
      barColor: 'bg-gradient-to-r from-emerald-400 to-emerald-500',
      badgeBg: 'bg-emerald-100 text-emerald-800',
    },
    fat: {
      label: 'Total Fat',
      unitLabel: 'g',
      icon: Droplet,
      textColor: 'text-rose-600',
      bgColor: 'bg-rose-50',
      borderColor: 'border-rose-200/70',
      barColor: 'bg-gradient-to-r from-rose-400 to-rose-500',
      badgeBg: 'bg-rose-100 text-rose-800',
    },
  };

  const config = configs[type] || configs.calories;
  const Icon = config.icon;

  return (
    <div
      className={`rounded-2xl p-5 sm:p-6 border ${config.borderColor} bg-white shadow-soft hover:shadow-card transition-all duration-200 group flex flex-col justify-between`}
    >
      <div>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <div
              className={`w-9 h-9 rounded-xl ${config.bgColor} ${config.textColor} flex items-center justify-center group-hover:scale-110 transition-transform`}
            >
              <Icon className="w-5 h-5" />
            </div>
            <span className="text-sm font-bold text-slate-700">{config.label}</span>
          </div>
          <span className={`text-[11px] font-bold px-2 py-0.5 rounded-full ${config.badgeBg}`}>
            {percentage}% DV
          </span>
        </div>

        <div className="flex items-baseline gap-1.5 my-2">
          <span className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
            {adjustedValue}
          </span>
          <span className="text-sm font-bold text-slate-500">{unit || config.unitLabel}</span>
        </div>
      </div>

      <div className="mt-3">
        <div className="flex justify-between text-[11px] font-semibold text-slate-400 mb-1.5">
          <span>Daily Allowance</span>
          <span>
            {adjustedValue} / {dailyGoal} {config.unitLabel}
          </span>
        </div>
        <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${config.barColor}`}
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>
    </div>
  );
}
