import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import confetti from 'canvas-confetti';
import {
  Sparkles,
  ArrowLeft,
  RotateCcw,
  BookmarkCheck,
  History,
  Share2,
  Download,
  Copy,
  Check,
  Cpu,
  Flame,
  Info,
  Sliders,
  Scale
} from 'lucide-react';
import Sidebar from '../components/Sidebar';
import NutritionCard from '../components/NutritionCard';
import NutritionChart from '../components/NutritionChart';
import ConfidenceBadge from '../components/ConfidenceBadge';
import LowConfidenceAlert from '../components/LowConfidenceAlert';
import DisclaimerBanner from '../components/DisclaimerBanner';
import Toast from '../components/Toast';
import { useHistory } from '../context/HistoryContext';
import { SAMPLE_FOODS } from '../data/sampleFoods';

export default function Result() {
  const { activeResult, addHistoryItem } = useHistory();
  const navigate = useNavigate();

  // Fallback to first sample food if accessed directly without upload
  const result = activeResult || {
    ...SAMPLE_FOODS[0],
    imageUrl: SAMPLE_FOODS[0].imageUrl,
  };

  const [portion, setPortion] = useState(1.0);
  const [toastMsg, setToastMsg] = useState(null);
  const [isSaved, setIsSaved] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    // Trigger celebratory confetti if high confidence
    if (result.confidenceScore >= 0.85 && !result.isLowConfidence) {
      try {
        confetti({
          particleCount: 40,
          spread: 60,
          origin: { y: 0.6 },
          colors: ['#10b981', '#34d399', '#059669', '#f59e0b'],
        });
      } catch (e) {}
    }
  }, [result]);

  const isLowConfidence = result.isLowConfidence || result.confidenceScore < 0.65;

  const handleSave = async () => {
    await addHistoryItem({
      ...result,
      portion,
      calories: Math.round(result.estimatedCalories * portion),
      protein: Math.round(result.protein * portion * 10) / 10,
      carbohydrates: Math.round(result.carbohydrates * portion * 10) / 10,
      fat: Math.round(result.fat * portion * 10) / 10,
    });
    setIsSaved(true);
    setToastMsg('Result saved to your analysis history!');
  };

  const handleCopySummary = () => {
    const text = `🥗 NutriVision AI Nutrition Report:
Food: ${result.foodClass} (${Math.round(result.confidenceScore * 100)}% confidence)
Portion: ${portion}x (${result.servingSize || '1 serving'})
• Energy: ${Math.round(result.estimatedCalories * portion)} kcal
• Protein: ${Math.round(result.protein * portion * 10) / 10}g
• Carbs: ${Math.round(result.carbohydrates * portion * 10) / 10}g
• Fat: ${Math.round(result.fat * portion * 10) / 10}g
Analyzed by NutriVision AI PyTorch Engine.`;

    navigator.clipboard.writeText(text);
    setCopied(true);
    setToastMsg('Nutrition summary copied to clipboard!');
    setTimeout(() => setCopied(false), 2500);
  };

  const handleExportJSON = () => {
    const exportData = {
      ...result,
      selectedPortionMultiplier: portion,
      adjustedCalories: Math.round(result.estimatedCalories * portion),
      adjustedProtein: Math.round(result.protein * portion * 10) / 10,
      adjustedCarbs: Math.round(result.carbohydrates * portion * 10) / 10,
      adjustedFat: Math.round(result.fat * portion * 10) / 10,
      exportedAt: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `nutrivision-${result.foodClass.toLowerCase().replace(/\s+/g, '-')}.json`;
    a.click();
    URL.revokeObjectURL(url);
    setToastMsg('Analysis exported as JSON file!');
  };

  return (
    <div className="min-h-screen bg-slate-50/60 flex">
      <Sidebar />

      <main className="flex-1 lg:pl-72 flex flex-col min-w-0">
        <div className="max-w-6xl w-full mx-auto p-4 sm:p-6 lg:p-8 space-y-8">
          {/* Top Navigation & Action Row */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-200/80">
            <Link
              to="/analyze"
              className="inline-flex items-center gap-2 text-sm font-bold text-slate-600 hover:text-slate-900 transition-colors"
            >
              <ArrowLeft className="w-4 h-4" />
              <span>Back to Scanner</span>
            </Link>

            <div className="flex flex-wrap items-center gap-2.5">
              <button
                onClick={handleCopySummary}
                className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-white border border-slate-200 text-xs font-bold text-slate-700 hover:bg-slate-50 shadow-soft transition-colors"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                <span>{copied ? 'Copied!' : 'Copy Summary'}</span>
              </button>

              <button
                onClick={handleExportJSON}
                className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-white border border-slate-200 text-xs font-bold text-slate-700 hover:bg-slate-50 shadow-soft transition-colors"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Export JSON</span>
              </button>

              <button
                onClick={handleSave}
                disabled={isSaved}
                className={`inline-flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-bold shadow-soft transition-all ${
                  isSaved
                    ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                    : 'bg-brand-600 hover:bg-brand-700 text-white shadow-brand-500/20'
                }`}
              >
                <BookmarkCheck className="w-4 h-4" />
                <span>{isSaved ? 'Saved in History' : 'Save Result'}</span>
              </button>
            </div>
          </div>

          {/* Low Confidence Warning Card (Requirement #5) */}
          {isLowConfidence && (
            <LowConfidenceAlert
              confidenceScore={result.confidenceScore}
              onRetry={() => navigate('/analyze')}
            />
          )}

          {/* Identification & Image Showcase Card */}
          <div className="bg-white rounded-3xl p-6 sm:p-8 border border-slate-200/80 shadow-soft">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
              {/* Food Image */}
              <div className="lg:col-span-5 relative aspect-square sm:aspect-video lg:aspect-square rounded-2xl overflow-hidden shadow-inner border border-slate-100">
                <img
                  src={result.imageUrl}
                  alt={result.foodClass}
                  className="w-full h-full object-cover"
                />
                <div className="absolute top-3 left-3">
                  <ConfidenceBadge score={result.confidenceScore} />
                </div>
              </div>

              {/* Food Information & AI Tagging */}
              <div className="lg:col-span-7 space-y-4">
                <div className="space-y-1">
                  <div className="flex items-center gap-2 text-xs font-bold text-brand-600 uppercase tracking-wider">
                    <Sparkles className="w-4 h-4" />
                    <span>{result.category || 'Identified Dish'}</span>
                  </div>
                  <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
                    {result.foodClass}
                  </h1>
                  <p className="text-xs sm:text-sm text-slate-500 font-medium">
                    Serving: <strong className="text-slate-700">{result.servingSize || '1 standard serving (approx. 350g)'}</strong>
                  </p>
                </div>

                {/* Dietary Tags */}
                {result.dietaryTags && result.dietaryTags.length > 0 && (
                  <div className="flex flex-wrap gap-2 pt-1">
                    {result.dietaryTags.map((tag, idx) => (
                      <span
                        key={idx}
                        className="px-2.5 py-1 rounded-lg bg-slate-100 text-slate-700 text-xs font-semibold"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                )}

                {/* AI Health Tip */}
                {result.healthTips && (
                  <div className="p-4 rounded-2xl bg-brand-50/70 border border-brand-200/70 text-brand-950 text-xs sm:text-sm leading-relaxed">
                    <p className="font-bold text-brand-800 mb-0.5 flex items-center gap-1.5">
                      <Sparkles className="w-3.5 h-3.5 text-brand-600" />
                      AI Nutritional Insight
                    </p>
                    <p>{result.healthTips}</p>
                  </div>
                )}

                {/* Portion Size Scaling Slider / Buttons */}
                <div className="pt-3 border-t border-slate-100">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-bold text-slate-700 flex items-center gap-1.5">
                      <Scale className="w-3.5 h-3.5 text-slate-500" />
                      Adjust Portion Size:
                    </span>
                    <span className="text-xs font-extrabold text-brand-600">{portion}x Portion</span>
                  </div>

                  <div className="grid grid-cols-4 gap-2">
                    {[0.5, 1.0, 1.5, 2.0].map((val) => (
                      <button
                        key={val}
                        type="button"
                        onClick={() => setPortion(val)}
                        className={`py-1.5 rounded-xl text-xs font-bold transition-all ${
                          portion === val
                            ? 'bg-slate-900 text-white shadow-sm'
                            : 'bg-slate-100 hover:bg-slate-200 text-slate-700'
                        }`}
                      >
                        {val === 0.5 ? 'Half (0.5x)' : val === 1.0 ? 'Regular (1x)' : val === 1.5 ? 'Large (1.5x)' : 'Double (2x)'}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* 4 Large Nutrition Summary Cards (Requirement #4) */}
          <div className="space-y-3">
            <h2 className="text-lg font-extrabold text-slate-900 tracking-tight flex items-center gap-2">
              <Flame className="w-5 h-5 text-amber-500" />
              <span>Nutritional Summary</span>
            </h2>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-5">
              <NutritionCard
                type="calories"
                value={result.estimatedCalories}
                unit="kcal"
                dailyGoal={2200}
                portionMultiplier={portion}
              />
              <NutritionCard
                type="protein"
                value={result.protein}
                unit="g"
                dailyGoal={130}
                portionMultiplier={portion}
              />
              <NutritionCard
                type="carbohydrates"
                value={result.carbohydrates}
                unit="g"
                dailyGoal={240}
                portionMultiplier={portion}
              />
              <NutritionCard
                type="fat"
                value={result.fat}
                unit="g"
                dailyGoal={65}
                portionMultiplier={portion}
              />
            </div>
          </div>

          {/* Nutrition Energy Chart & Micronutrient Breakdown */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            {/* Macro Ratio Chart */}
            <div className="lg:col-span-7">
              <NutritionChart
                protein={result.protein}
                carbs={result.carbohydrates}
                fat={result.fat}
                portionMultiplier={portion}
              />
            </div>

            {/* Micronutrients & Detected Ingredients */}
            <div className="lg:col-span-5 space-y-6">
              {/* Micronutrients Table */}
              <div className="bg-white rounded-3xl p-6 border border-slate-200/80 shadow-soft space-y-4">
                <h3 className="text-sm font-extrabold text-slate-900 pb-2 border-b border-slate-100">
                  Key Micronutrients & Minerals
                </h3>
                <div className="space-y-2.5 text-xs">
                  <div className="flex justify-between py-1 border-b border-slate-50">
                    <span className="text-slate-500 font-medium">Dietary Fiber</span>
                    <span className="font-extrabold text-slate-800">
                      {Math.round((result.fiber || 0) * portion * 10) / 10} g
                    </span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-slate-50">
                    <span className="text-slate-500 font-medium">Sugars</span>
                    <span className="font-extrabold text-slate-800">
                      {Math.round((result.sugar || 0) * portion * 10) / 10} g
                    </span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-slate-50">
                    <span className="text-slate-500 font-medium">Sodium</span>
                    <span className="font-extrabold text-slate-800">
                      {Math.round((result.sodium || 0) * portion)} mg
                    </span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-slate-500 font-medium">Potassium</span>
                    <span className="font-extrabold text-slate-800">
                      {Math.round((result.potassium || 0) * portion)} mg
                    </span>
                  </div>
                </div>
              </div>

              {/* Detected Ingredients */}
              {result.ingredients && result.ingredients.length > 0 && (
                <div className="bg-white rounded-3xl p-6 border border-slate-200/80 shadow-soft space-y-3">
                  <h3 className="text-sm font-extrabold text-slate-900">
                    Common Ingredients Detected
                  </h3>
                  <div className="flex flex-wrap gap-1.5">
                    {result.ingredients.map((ing, i) => (
                      <span
                        key={i}
                        className="px-2.5 py-1 rounded-lg bg-slate-50 border border-slate-200/70 text-slate-700 text-xs font-medium"
                      >
                        {ing}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* AI Prediction Technical Diagnostics Box (Requirement #4) */}
          <div className="bg-white rounded-3xl p-6 border border-slate-200/80 shadow-soft">
            <h3 className="text-sm font-extrabold text-slate-900 mb-4 flex items-center gap-2">
              <Cpu className="w-4 h-4 text-brand-600" />
              <span>AI Prediction Diagnostics</span>
            </h3>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
              <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-100">
                <span className="text-slate-400 font-medium block">Food Class</span>
                <span className="text-slate-900 font-bold mt-1 block truncate">{result.foodClass}</span>
              </div>
              <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-100">
                <span className="text-slate-400 font-medium block">Confidence Score</span>
                <span className="text-slate-900 font-bold mt-1 block">{Math.round(result.confidenceScore * 1000) / 10}%</span>
              </div>
              <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-100">
                <span className="text-slate-400 font-medium block">Estimated Base Energy</span>
                <span className="text-slate-900 font-bold mt-1 block">{result.estimatedCalories} kcal</span>
              </div>
              <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-100">
                <span className="text-slate-400 font-medium block">Inference Runtime</span>
                <span className="text-slate-900 font-bold mt-1 block">{result.inferenceTimeMs || 45.0} ms</span>
              </div>
            </div>
          </div>

          {/* Action Buttons Row (Requirement #4) */}
          <div className="flex flex-wrap items-center justify-between gap-4 p-6 bg-white rounded-3xl border border-slate-200/80 shadow-soft">
            <div className="flex flex-wrap items-center gap-3">
              <Link
                to="/analyze"
                className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-brand-600 hover:bg-brand-700 text-white font-bold text-sm shadow-md shadow-brand-500/20 transition-all hover:scale-[1.02]"
              >
                <RotateCcw className="w-4 h-4" />
                <span>Analyze Another Food</span>
              </Link>

              <Link
                to="/history"
                className="inline-flex items-center gap-2 px-5 py-3 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 font-bold text-sm transition-colors"
              >
                <History className="w-4 h-4" />
                <span>View Scan History</span>
              </Link>
            </div>

            <Link
              to="/dashboard"
              className="text-xs font-bold text-slate-500 hover:text-slate-900"
            >
              Return to Dashboard →
            </Link>
          </div>

          {/* ML Disclaimer (Requirement #4 & #11) */}
          <DisclaimerBanner />
        </div>
      </main>

      {/* Toast feedback */}
      <Toast message={toastMsg} onClose={() => setToastMsg(null)} />
    </div>
  );
}
