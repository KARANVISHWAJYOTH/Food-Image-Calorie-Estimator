import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Sidebar from '../components/Sidebar';
import DashboardHeader from '../components/DashboardHeader';
import FoodUploader from '../components/FoodUploader';
import LoadingState from '../components/LoadingState';
import DisclaimerBanner from '../components/DisclaimerBanner';
import { useHistory } from '../context/HistoryContext';
import { predictFood } from '../services/api';
import { Sparkles, Scan, HelpCircle, ShieldCheck } from 'lucide-react';

export default function Analyze() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const { setActiveResult, addHistoryItem } = useHistory();
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
        await addHistoryItem(resultData);
        navigate('/result');
      }
    } catch (err) {
      console.error('Analyze error:', err);
      setError(err.response?.data?.detail || 'The model could not analyze this image. Make sure the backend and trained model are available.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50/60 flex">
      <Sidebar />

      <main className="flex-1 lg:pl-72 flex flex-col min-w-0">
        <div className="max-w-5xl w-full mx-auto p-4 sm:p-6 lg:p-8 space-y-8">
          <DashboardHeader
            title="Analyze Food Photo"
            subtitle="Upload an image or pick a sample to identify food and estimate macronutrient values."
            showAction={false}
          />

          {isLoading ? (
            <LoadingState message="AI is analyzing your food…" />
          ) : (
            <div className="space-y-6">
              <FoodUploader onAnalyze={handleAnalyze} isLoading={isLoading} />
              {error && <p className="text-sm text-rose-600">{error}</p>}

              {/* Best Practice Tips */}
              <div className="bg-white rounded-3xl p-6 border border-slate-200/80 shadow-soft">
                <div className="flex items-center gap-2 mb-3">
                  <Sparkles className="w-4 h-4 text-brand-600" />
                  <h3 className="text-sm font-bold text-slate-900">
                    Tips for Maximum Prediction Accuracy
                  </h3>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs text-slate-600">
                  <div className="p-3 rounded-xl bg-slate-50 border border-slate-100">
                    <p className="font-bold text-slate-800 mb-1">1. Good Illumination</p>
                    <p>Natural overhead light helps the CNN detect texture and ingredient layers clearly.</p>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-50 border border-slate-100">
                    <p className="font-bold text-slate-800 mb-1">2. Full Plate View</p>
                    <p>Ensure the entire dish is inside the viewfinder for accurate portion estimation.</p>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-50 border border-slate-100">
                    <p className="font-bold text-slate-800 mb-1">3. Clear Focus</p>
                    <p>Avoid motion blur so individual garnishes and grains can be categorized.</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          <DisclaimerBanner />
        </div>
      </main>
    </div>
  );
}
