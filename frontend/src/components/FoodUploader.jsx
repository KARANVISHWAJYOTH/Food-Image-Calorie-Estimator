import React, { useState, useRef } from 'react';
import {
  UploadCloud,
  Image as ImageIcon,
  Camera,
  X,
  Sparkles,
  ArrowRight,
  AlertCircle,
  FileCheck2
} from 'lucide-react';
import ScanningOverlay from './ScanningOverlay';

export default function FoodUploader({ onAnalyze, isLoading = false }) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  const fileInputRef = useRef(null);
  const cameraInputRef = useRef(null);

  const validateAndProcessFile = (file) => {
    setErrorMsg(null);
    if (!file) return;

    // Check type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type)) {
      setErrorMsg('Invalid file format. Please upload a JPG, JPEG, PNG, or WEBP image.');
      return;
    }

    // Check size (15MB limit)
    if (file.size > 15 * 1024 * 1024) {
      setErrorMsg('Image size exceeds 15 MB. Please select a smaller photo.');
      return;
    }

    setSelectedFile(file);

    const reader = new FileReader();
    reader.onload = (e) => {
      setPreviewUrl(e.target.result);
    };
    reader.readAsDataURL(file);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndProcessFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      validateAndProcessFile(e.target.files[0]);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setErrorMsg(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
    if (cameraInputRef.current) cameraInputRef.current.value = '';
  };

  const handleTriggerAnalysis = () => {
    if (!previewUrl) {
      setErrorMsg('Please upload a food photo first.');
      return;
    }
    onAnalyze({
      file: selectedFile,
      previewUrl: previewUrl,
    });
  };

  return (
    <div className="space-y-6">
      {/* Upload Zone Container */}
      <div
        className={`relative bg-white rounded-3xl p-6 sm:p-8 border-2 transition-all duration-200 shadow-soft ${
          dragActive
            ? 'border-brand-500 bg-brand-50/40 ring-4 ring-brand-500/15'
            : previewUrl
            ? 'border-brand-200 bg-white'
            : 'border-dashed border-slate-300 hover:border-brand-400 hover:bg-slate-50/50'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp,image/jpg"
          onChange={handleFileChange}
          className="hidden"
          id="food-file-input"
        />
        <input
          ref={cameraInputRef}
          type="file"
          accept="image/*"
          capture="environment"
          onChange={handleFileChange}
          className="hidden"
          id="food-camera-input"
        />

        {!previewUrl ? (
          /* Empty / Drop state */
          <div className="flex flex-col items-center justify-center text-center py-8 sm:py-12">
            <div className="w-20 h-20 rounded-2xl bg-brand-50 text-brand-600 flex items-center justify-center mb-4 shadow-sm border border-brand-100 group-hover:scale-105 transition-transform">
              <UploadCloud className="w-10 h-10 stroke-[1.75]" />
            </div>

            <h3 className="text-xl sm:text-2xl font-extrabold text-slate-900 tracking-tight mb-2">
              Drop your food image here
            </h3>
            <p className="text-sm text-slate-500 max-w-md mb-6 leading-relaxed">
              Upload any meal photo. NutriVision AI will segment the dish, classify ingredients, and estimate calories & macros.
            </p>

            <div className="flex flex-wrap items-center justify-center gap-3">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-brand-600 hover:bg-brand-700 text-white text-sm font-semibold shadow-md shadow-brand-500/20 transition-all hover:scale-[1.02]"
              >
                <ImageIcon className="w-4 h-4" />
                <span>Browse from device</span>
              </button>
              <button
                type="button"
                onClick={() => cameraInputRef.current?.click()}
                className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-sm font-semibold transition-colors"
              >
                <Camera className="w-4 h-4" />
                <span>Take Photo</span>
              </button>
            </div>

            <p className="text-[12px] text-slate-400 font-medium mt-4">
              Supported formats: <strong className="text-slate-600">JPG, JPEG, PNG, WEBP</strong> (Up to 15MB)
            </p>
          </div>
        ) : (
          /* Preview state */
          <div className="space-y-5">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100">
              <div className="flex items-center gap-2 text-sm font-bold text-slate-900">
                <FileCheck2 className="w-4 h-4 text-brand-600" />
                <span>
                  {selectedFile?.name || 'Uploaded Food Photo'}
                </span>
              </div>
              <button
                type="button"
                onClick={handleClear}
                disabled={isLoading}
                className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-semibold text-slate-500 hover:text-rose-600 hover:bg-rose-50 transition-colors disabled:opacity-50"
              >
                <X className="w-3.5 h-3.5" />
                <span>Change Image</span>
              </button>
            </div>

            {/* Food Image with scanning preview */}
            <div className="relative aspect-video sm:aspect-[21/9] max-h-[380px] w-full rounded-2xl overflow-hidden bg-slate-900 border border-slate-200 shadow-inner group">
              <img
                src={previewUrl}
                alt="Selected Food Preview"
                className="w-full h-full object-cover"
              />
              {/* Active laser scan simulation */}
              <ScanningOverlay isScanning={isLoading} label={isLoading ? 'Extracting Deep CNN Features...' : 'Ready for AI Inference'} />
            </div>

            {/* Action Bar */}
            <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-2">
              <div className="text-xs text-slate-500 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                <span>Multi-Task PyTorch Model Loaded • ResNet50 Backbone</span>
              </div>

              <button
                type="button"
                onClick={handleTriggerAnalysis}
                disabled={isLoading}
                className="w-full sm:w-auto inline-flex items-center justify-center gap-2.5 px-8 py-3.5 rounded-xl bg-gradient-to-r from-brand-600 to-emerald-500 hover:from-brand-700 hover:to-emerald-600 text-white font-bold text-base shadow-lg shadow-brand-500/25 transition-all hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:pointer-events-none"
              >
                {isLoading ? (
                  <>
                    <Sparkles className="w-5 h-5 animate-spin" />
                    <span>AI is analyzing your food…</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-5 h-5" />
                    <span>Analyze Food Nutrition</span>
                    <ArrowRight className="w-4 h-4" />
                  </>
                )}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Error alert */}
      {errorMsg && (
        <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-sm flex items-start gap-3">
          <AlertCircle className="w-5 h-5 flex-shrink-0 text-rose-500 mt-0.5" />
          <div>
            <p className="font-semibold">Upload Error</p>
            <p className="text-xs mt-0.5">{errorMsg}</p>
          </div>
        </div>
      )}

    </div>
  );
}
