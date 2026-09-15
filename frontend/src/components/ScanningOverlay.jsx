import React from 'react';
import { Sparkles, Cpu } from 'lucide-react';

export default function ScanningOverlay({ isScanning = true, label = 'AI Neural Feature Extractor' }) {
  if (!isScanning) return null;

  return (
    <div className="absolute inset-0 z-20 pointer-events-none overflow-hidden rounded-2xl sm:rounded-3xl">
      {/* Dark tint overlay */}
      <div className="absolute inset-0 bg-brand-950/20 backdrop-blur-[1px]" />

      {/* Animated Laser Scanning Beam */}
      <div className="absolute left-0 right-0 h-1.5 scanner-beam animate-scan-line z-30" />

      {/* Scanning Reticle Corners */}
      <div className="absolute top-4 left-4 w-8 h-8 border-t-2 border-l-2 border-brand-400 rounded-tl-lg" />
      <div className="absolute top-4 right-4 w-8 h-8 border-t-2 border-r-2 border-brand-400 rounded-tr-lg" />
      <div className="absolute bottom-4 left-4 w-8 h-8 border-b-2 border-l-2 border-brand-400 rounded-bl-lg" />
      <div className="absolute bottom-4 right-4 w-8 h-8 border-b-2 border-r-2 border-brand-400 rounded-br-lg" />

      {/* Target Focus Reticle Center */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="w-32 h-32 sm:w-44 sm:h-44 rounded-full border border-brand-400/40 border-dashed animate-spin" style={{ animationDuration: '12s' }} />
        <div className="absolute w-2 h-2 rounded-full bg-brand-400 animate-ping" />
      </div>

      {/* Real-time AI Tracking Pill */}
      <div className="absolute top-5 left-1/2 -translate-x-1/2 flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900/80 backdrop-blur-md border border-brand-500/40 text-white text-xs font-semibold shadow-lg">
        <Cpu className="w-3.5 h-3.5 text-brand-400 animate-spin" style={{ animationDuration: '3s' }} />
        <span className="tracking-wide">{label}</span>
      </div>

      {/* Dynamic Bounding Box simulation */}
      <div className="absolute top-[28%] left-[22%] w-[56%] h-[48%] border border-brand-400/70 rounded-xl bg-brand-500/10">
        <div className="absolute -top-3 left-2 px-2 py-0.5 rounded bg-brand-600 text-white text-[10px] font-bold uppercase tracking-wider flex items-center gap-1 shadow-sm">
          <Sparkles className="w-2.5 h-2.5" />
          <span>Food Segment • 98.4%</span>
        </div>
      </div>
    </div>
  );
}
