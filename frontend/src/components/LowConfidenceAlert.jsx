import React from 'react';
import { AlertTriangle, RefreshCw, HelpCircle, Camera, Lightbulb } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function LowConfidenceAlert({ confidenceScore = 0.52, onRetry }) {
  const percentage = Math.round(confidenceScore * 1000) / 10;

  return (
    <div className="rounded-3xl p-6 sm:p-7 bg-gradient-to-br from-amber-500/10 via-amber-500/5 to-transparent border-2 border-amber-300/80 shadow-soft">
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-5">
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 rounded-2xl bg-amber-100 text-amber-700 flex items-center justify-center flex-shrink-0 shadow-sm border border-amber-200">
            <AlertTriangle className="w-6 h-6 animate-pulse" />
          </div>

          <div className="space-y-1">
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="text-lg font-extrabold text-slate-900">
                We’re not completely sure about this food.
              </h3>
              <span className="px-2.5 py-0.5 rounded-full bg-amber-100 text-amber-800 text-xs font-bold border border-amber-200">
                Confidence: {percentage}%
              </span>
            </div>
            <p className="text-sm text-slate-600 leading-relaxed max-w-2xl">
              The model detected visual ambiguity (e.g., dim lighting, unusual angle, or partial crop).
              Estimated nutrition values below may vary from actual content.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 w-full md:w-auto">
          {onRetry ? (
            <button
              onClick={onRetry}
              className="w-full md:w-auto inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl bg-amber-600 hover:bg-amber-700 text-white font-bold text-sm shadow-md shadow-amber-500/20 transition-all hover:scale-[1.02]"
            >
              <RefreshCw className="w-4 h-4" />
              <span>Try Another Image</span>
            </button>
          ) : (
            <Link
              to="/analyze"
              className="w-full md:w-auto inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl bg-amber-600 hover:bg-amber-700 text-white font-bold text-sm shadow-md shadow-amber-500/20 transition-all hover:scale-[1.02]"
            >
              <RefreshCw className="w-4 h-4" />
              <span>Try Another Image</span>
            </Link>
          )}
        </div>
      </div>

      {/* Helpful photography tips */}
      <div className="mt-5 pt-4 border-t border-amber-200/60 grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs text-slate-700">
        <div className="flex items-center gap-2">
          <Lightbulb className="w-4 h-4 text-amber-600 flex-shrink-0" />
          <span>Shoot in bright, natural daylight</span>
        </div>
        <div className="flex items-center gap-2">
          <Camera className="w-4 h-4 text-amber-600 flex-shrink-0" />
          <span>Keep the entire plate in frame</span>
        </div>
        <div className="flex items-center gap-2">
          <HelpCircle className="w-4 h-4 text-amber-600 flex-shrink-0" />
          <span>Avoid blurry or high-shadow angles</span>
        </div>
      </div>
    </div>
  );
}
