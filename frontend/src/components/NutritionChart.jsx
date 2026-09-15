import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { PieChart as PieIcon, Info } from 'lucide-react';

export default function NutritionChart({ protein = 0, carbs = 0, fat = 0, portionMultiplier = 1 }) {
  const p = (Number(protein) || 0) * portionMultiplier;
  const c = (Number(carbs) || 0) * portionMultiplier;
  const f = (Number(fat) || 0) * portionMultiplier;

  // Caloric calculations (Protein 4 kcal/g, Carbs 4 kcal/g, Fat 9 kcal/g)
  const proteinCals = p * 4;
  const carbCals = c * 4;
  const fatCals = f * 9;
  const totalCals = proteinCals + carbCals + fatCals || 1;

  const proteinPct = Math.round((proteinCals / totalCals) * 100);
  const carbPct = Math.round((carbCals / totalCals) * 100);
  const fatPct = Math.round((fatCals / totalCals) * 100);

  const data = [
    { name: 'Protein', value: Math.round(proteinCals), grams: Math.round(p * 10) / 10, pct: proteinPct, color: '#06b6d4' },
    { name: 'Carbohydrates', value: Math.round(carbCals), grams: Math.round(c * 10) / 10, pct: carbPct, color: '#10b981' },
    { name: 'Fat', value: Math.round(fatCals), grams: Math.round(f * 10) / 10, pct: fatPct, color: '#f43f5e' },
  ];

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const item = payload[0].payload;
      return (
        <div className="bg-slate-900/90 backdrop-blur-md text-white text-xs p-3 rounded-xl shadow-xl border border-slate-700">
          <p className="font-bold text-sm" style={{ color: item.color }}>
            {item.name}
          </p>
          <div className="mt-1 space-y-0.5 text-slate-300">
            <p>Weight: <strong className="text-white">{item.grams}g</strong></p>
            <p>Energy: <strong className="text-white">{item.value} kcal</strong></p>
            <p>Caloric Share: <strong className="text-white">{item.pct}%</strong></p>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-white rounded-3xl p-6 sm:p-7 border border-slate-200/85 shadow-soft space-y-5">
      <div className="flex items-center justify-between pb-3 border-b border-slate-100">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-brand-50 text-brand-600 flex items-center justify-center">
            <PieIcon className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-base font-extrabold text-slate-900">Macronutrient Energy Distribution</h3>
            <p className="text-xs text-slate-500">Relative caloric contribution from macros</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-center">
        {/* Donut Chart */}
        <div className="md:col-span-6 h-52 relative flex items-center justify-center">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Tooltip content={<CustomTooltip />} />
              <Pie
                data={data}
                innerRadius={55}
                outerRadius={80}
                paddingAngle={4}
                dataKey="value"
                stroke="none"
              >
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
          
          {/* Centered Donut Label */}
          <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Total</span>
            <span className="text-xl font-extrabold text-slate-900 tracking-tight">
              {Math.round(totalCals)}
            </span>
            <span className="text-[10px] font-bold text-slate-500">kcal</span>
          </div>
        </div>

        {/* Breakdown Badges */}
        <div className="md:col-span-6 space-y-3">
          {data.map((item, idx) => (
            <div
              key={idx}
              className="p-3 rounded-2xl bg-slate-50 border border-slate-200/60 flex items-center justify-between hover:bg-slate-100/60 transition-colors"
            >
              <div className="flex items-center gap-3">
                <div
                  className="w-3.5 h-3.5 rounded-full shadow-sm"
                  style={{ backgroundColor: item.color }}
                />
                <div>
                  <p className="text-xs font-bold text-slate-900">{item.name}</p>
                  <p className="text-[11px] text-slate-500 font-medium">{item.grams} grams</p>
                </div>
              </div>

              <div className="text-right">
                <p className="text-sm font-extrabold text-slate-900">{item.pct}%</p>
                <p className="text-[11px] font-semibold text-slate-400">{item.value} kcal</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
