import React from 'react';
import { Sparkles, Flame, Check } from 'lucide-react';
import { SAMPLE_FOODS } from '../data/sampleFoods';

export default function SampleFoodPicker({ onSelect, selectedKey }) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-brand-600" />
          <h3 className="text-sm font-bold text-slate-900">
            Quick Test Samples (1-Click Demo)
          </h3>
        </div>
        <span className="text-xs text-slate-500">Pick any image to test prediction</span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
        {SAMPLE_FOODS.map((food) => {
          const isSelected = selectedKey === food.key || selectedKey === food.id;
          return (
            <button
              key={food.id}
              type="button"
              onClick={() => onSelect(food)}
              className={`group relative text-left rounded-xl overflow-hidden border transition-all duration-200 p-2 flex flex-col justify-between ${
                isSelected
                  ? 'border-brand-500 bg-brand-50/70 ring-2 ring-brand-500/20 shadow-md'
                  : 'border-slate-200 bg-white hover:border-brand-300 hover:shadow-sm'
              }`}
            >
              <div className="relative aspect-video sm:aspect-square w-full rounded-lg overflow-hidden mb-2">
                <img
                  src={food.imageUrl}
                  alt={food.name}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                  loading="lazy"
                />
                {food.isLowConfidence && (
                  <span className="absolute top-1 right-1 px-1.5 py-0.5 rounded bg-amber-500/90 text-white text-[9px] font-bold">
                    Low Conf
                  </span>
                )}
                {isSelected && (
                  <div className="absolute inset-0 bg-brand-900/40 flex items-center justify-center">
                    <div className="w-6 h-6 rounded-full bg-brand-500 text-white flex items-center justify-center shadow-md">
                      <Check className="w-3.5 h-3.5 stroke-[3]" />
                    </div>
                  </div>
                )}
              </div>

              <div>
                <p className="text-xs font-bold text-slate-900 truncate group-hover:text-brand-700">
                  {food.name}
                </p>
                <div className="flex items-center gap-1.5 text-[11px] text-slate-500 mt-0.5">
                  <Flame className="w-3 h-3 text-amber-500" />
                  <span>{food.calories} kcal</span>
                  <span>•</span>
                  <span>{food.protein}g P</span>
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
