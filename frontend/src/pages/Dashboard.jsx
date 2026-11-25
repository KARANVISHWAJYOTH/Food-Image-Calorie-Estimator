import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Sidebar from '../components/Sidebar';
import DashboardHeader from '../components/DashboardHeader';
import QuickStats from '../components/QuickStats';
import FoodUploader from '../components/FoodUploader';
import LoadingState from '../components/LoadingState';
import DisclaimerBanner from '../components/DisclaimerBanner';
import { useHistory } from '../context/HistoryContext';
import { predictFood, resolveAssetUrl } from '../services/api';
import { Flame, ArrowRight, History, Sparkles } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function Dashboard() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const { history, setActiveResult, addHistoryItem } = useHistory();
  const navigate = useNavigate();

  const handleAnalyze = async ({ file, previewUrl }) => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await predictFood(file);
      if (res.success) {
        const resultData = {
          ...res.data,
          imageUrl: previewUrl || res.data.imageUrl,
        };
        setActiveResult(resultData);
        try {
          await addHistoryItem(resultData);
        } catch (saveError) {
          console.warn('Prediction succeeded but history save failed:', saveError);
        }
        navigate('/result');
      }
    } catch (err) {
      console.error('Analysis error:', err);
      setError(
        err.response?.data?.detail
          || err.message
          || 'The model could not analyze this image. Make sure the backend and trained model are available.',
      );
    } finally {
      setIsLoading(false);
    }
  };

  const recentHistory = history.slice(0, 3);

  return (
    <div className="min-h-screen bg-slate-50/60 flex flex-col">
      {/* Navigation Sidebar */}
      <Sidebar />

      {/* Main Workspace */}
      <main className="flex-1 lg:pl-72 flex flex-col min-w-0">
        <div className="max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8 space-y-8">
          {/* Header */}
          <DashboardHeader
            subtitle="Upload a food image to get your nutrition analysis."
            showAction={false}
          />

          {/* Summary cards */}
          <QuickStats />

          {/* Analysis Workspace or Loader */}
          {isLoading ? (
            <LoadingState message="Analyzing your food…" />
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
              {/* Main Upload Card */}
              <div className="lg:col-span-8 space-y-6">
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <h2 className="text-lg font-extrabold text-slate-900 tracking-tight flex items-center gap-2">
                      <Sparkles className="w-5 h-5 text-brand-600" />
                      <span>Scan a Meal</span>
                    </h2>
                    <span className="text-xs font-semibold text-slate-400">
                      ResNet-50 model
                    </span>
                  </div>
                  <FoodUploader onAnalyze={handleAnalyze} isLoading={isLoading} />
                  {error && (
                    <div className="mt-4 p-3.5 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-xs">
                      {error}
                    </div>
                  )}
                </div>
              </div>

              {/* Right Side: Recent Activity & Daily Target */}
              <div className="lg:col-span-4 space-y-6">
                {/* Recent Scans Widget */}
                <div className="bg-white rounded-3xl p-6 border border-slate-200/80 shadow-soft space-y-4">
                  <div className="flex items-center justify-between pb-3 border-b border-slate-100">
                    <div className="flex items-center gap-2">
                      <History className="w-4 h-4 text-brand-600" />
                      <h3 className="text-sm font-extrabold text-slate-900">Recent Scans</h3>
                    </div>
                    <Link
                      to="/history"
                      className="text-xs font-bold text-brand-600 hover:text-brand-700"
                    >
                      View All
                    </Link>
                  </div>

                  {recentHistory.length > 0 ? (
                    <div className="space-y-3">
                      {recentHistory.map((item) => (
                        <div
                          key={item.id}
                          onClick={() => {
                            setActiveResult(item);
                            navigate('/result');
                          }}
                          className="flex items-center gap-3 p-2.5 rounded-2xl hover:bg-slate-50 border border-transparent hover:border-slate-200/60 cursor-pointer transition-all group"
                        >
                          <img
                            src={resolveAssetUrl(item.imageUrl)}
                            alt={item.foodClass}
                            className="w-12 h-12 rounded-xl object-cover border border-slate-200 group-hover:scale-105 transition-transform"
                          />
                          <div className="min-w-0 flex-1">
                            <p className="text-xs font-bold text-slate-900 truncate group-hover:text-brand-600">
                              {item.foodClass}
                            </p>
                            <div className="flex items-center gap-2 text-[11px] text-slate-500 mt-0.5">
                              <span className="flex items-center gap-1 font-semibold text-amber-600">
                                <Flame className="w-3 h-3 fill-amber-500 text-amber-500" />
                                {item.calories} kcal
                              </span>
                              <span>•</span>
                              <span className="text-cyan-600 font-semibold">{item.protein}g P</span>
                            </div>
                          </div>
                          <ArrowRight className="w-3.5 h-3.5 text-slate-300 group-hover:text-brand-600 group-hover:translate-x-0.5 transition-all" />
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-6 text-slate-400 text-xs">
                      No meals analyzed yet. Upload your first dish above!
                    </div>
                  )}
                </div>

                {/* Model info card */}
                <div className="bg-gradient-to-br from-brand-900 to-slate-900 text-white rounded-3xl p-6 shadow-card space-y-3 relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-32 h-32 bg-brand-400/20 rounded-full blur-2xl pointer-events-none" />
                  <div className="flex items-center gap-2 text-brand-300 text-xs font-bold uppercase tracking-wider">
                    <Sparkles className="w-4 h-4" />
                    <span>Model Info</span>
                  </div>
                  <h4 className="text-base font-extrabold tracking-tight">
                    How the Estimate Is Made
                  </h4>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    A ResNet-50 backbone feeds one classification head (Food-101 classes) and one regression head that predicts
                    calories, protein, carbohydrates, and fat in a single forward pass.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Disclaimer */}
          <DisclaimerBanner />
        </div>
      </main>
    </div>
  );
}
