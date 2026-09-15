import React from 'react';
import { Link } from 'react-router-dom';
import {
  Sparkles,
  ArrowRight,
  ShieldCheck,
  Flame,
  Beef,
  Wheat,
  Droplet,
  Zap,
  Cpu,
  Layers,
  BarChart3,
  CheckCircle2,
  ScanLine,
  Image as ImageIcon,
  ChevronRight
} from 'lucide-react';
import Navbar from '../components/Navbar';
import ScanningOverlay from '../components/ScanningOverlay';
import DisclaimerBanner from '../components/DisclaimerBanner';

export default function Home() {
  const features = [
    {
      title: 'AI Food Recognition',
      desc: 'Identify food from uploaded images using a Multi-task CNN trained on thousands of global cuisine images.',
      icon: Cpu,
      color: 'bg-emerald-50 text-emerald-600 border-emerald-100',
    },
    {
      title: 'Calorie Estimation',
      desc: 'Get an accurate estimated calorie value (kcal) computed through deep visual feature regression.',
      icon: Flame,
      color: 'bg-amber-50 text-amber-600 border-amber-100',
    },
    {
      title: 'Nutrition Breakdown',
      desc: 'View comprehensive estimated protein, carbohydrates, and fat breakdown with % daily value gauges.',
      icon: BarChart3,
      color: 'bg-cyan-50 text-cyan-600 border-cyan-100',
    },
    {
      title: 'Fast AI Analysis',
      desc: 'Upload an image and receive complete multi-task predictions in under 50ms via our high-speed API.',
      icon: Zap,
      color: 'bg-purple-50 text-purple-600 border-purple-100',
    },
  ];

  const steps = [
    {
      num: '01',
      title: 'Upload Food Image',
      desc: 'Snap a picture with your phone camera or drag and drop any meal image into the upload zone.',
    },
    {
      num: '02',
      title: 'Image Preprocessing',
      desc: 'The image is normalized, resized, and tensor-transformed for optimal neural network inference.',
    },
    {
      num: '03',
      title: 'Multi-Task AI Analysis',
      desc: 'PyTorch deep CNN backbone extracts visual latent vectors for simultaneous classification & macro regression.',
    },
    {
      num: '04',
      title: 'Nutrition Report',
      desc: 'View calories, macronutrient donut breakdown, micronutrients, confidence scores, and dietary tips.',
    },
  ];

  const technologies = [
    { name: 'Python 3.11+', role: 'Backend Runtime', badge: 'Core' },
    { name: 'PyTorch', role: 'Deep Learning Engine', badge: 'AI Model' },
    { name: 'Torchvision', role: 'Transform Pipeline', badge: 'CV' },
    { name: 'FastAPI', role: 'High-Performance REST API', badge: 'API' },
    { name: 'React.js', role: 'Interactive Frontend SPA', badge: 'UI' },
    { name: 'Tailwind CSS', role: 'Design System', badge: 'Styles' },
    { name: 'Pandas & NumPy', role: 'Nutritional Data Analysis', badge: 'Data' },
    { name: 'Docker', role: 'Containerized Deployment', badge: 'DevOps' },
  ];

  return (
    <div className="min-h-screen flex flex-col bg-slate-50/50">
      <Navbar />

      {/* Hero Section */}
      <section className="relative overflow-hidden pt-12 pb-20 md:pt-20 md:pb-28">
        {/* Background glow orbs */}
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-gradient-to-tr from-brand-300/20 to-emerald-200/20 rounded-full blur-3xl pointer-events-none -z-10" />

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-8 items-center">
            {/* Left Hero Text */}
            <div className="lg:col-span-7 space-y-6 text-center lg:text-left">
              {/* PyTorch AI Badge */}
              <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-brand-50 border border-brand-200/80 text-brand-800 text-xs font-bold shadow-sm animate-pulse-subtle">
                <Sparkles className="w-4 h-4 text-brand-600" />
                <span>Powered by PyTorch AI & Multi-Task CNN</span>
              </div>

              <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold text-slate-900 tracking-tight leading-[1.12]">
                Know What’s <br className="hidden sm:inline" />
                <span className="bg-gradient-to-r from-brand-600 via-emerald-600 to-teal-600 bg-clip-text text-transparent">
                  On Your Plate.
                </span>
              </h1>

              <p className="text-lg sm:text-xl text-slate-600 leading-relaxed max-w-2xl mx-auto lg:mx-0 font-normal">
                Upload a photo of your food and let AI identify it, estimate calories, and break down its nutrition in seconds.
              </p>

              {/* Action Buttons */}
              <div className="flex flex-wrap items-center justify-center lg:justify-start gap-4 pt-2">
                <Link
                  to="/analyze"
                  className="inline-flex items-center gap-2.5 px-7 py-3.5 rounded-2xl bg-gradient-to-r from-brand-600 to-emerald-500 hover:from-brand-700 hover:to-emerald-600 text-white font-bold text-base shadow-lg shadow-brand-500/25 transition-all hover:scale-[1.02] active:scale-[0.98]"
                >
                  <Sparkles className="w-5 h-5" />
                  <span>Analyze Food</span>
                  <ArrowRight className="w-4 h-4" />
                </Link>

                <a
                  href="#how-it-works"
                  className="inline-flex items-center gap-2 px-6 py-3.5 rounded-2xl bg-white hover:bg-slate-100 text-slate-700 font-bold text-base border border-slate-200 shadow-soft transition-colors"
                >
                  <span>How It Works</span>
                </a>
              </div>

              {/* Trust stats */}
              <div className="pt-6 border-t border-slate-200/70 flex flex-wrap items-center justify-center lg:justify-start gap-6 text-xs text-slate-500 font-semibold">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                  <span>94.7% Average Accuracy</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                  <span>Multi-Task Regression</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                  <span>USDA Ground Truth</span>
                </div>
              </div>
            </div>

            {/* Right Hero Visual with Scanner & Floating Nutrition Cards */}
            <div className="lg:col-span-5 relative flex items-center justify-center">
              {/* Main Food Card Container */}
              <div className="relative w-full max-w-md bg-white rounded-3xl p-4 shadow-elevated border border-slate-200/80">
                {/* Food Image with Active Scanner */}
                <div className="relative aspect-square w-full rounded-2xl overflow-hidden shadow-inner group">
                  <img
                    src="https://images.unsplash.com/photo-1546069901-ba9599a7e63c?auto=format&fit=crop&w=800&q=80"
                    alt="Grilled Chicken Rice Bowl"
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                  />
                  {/* AI Scanning Beam Overlay */}
                  <ScanningOverlay isScanning={true} label="Multi-Task CNN • Segmenting" />
                </div>

                {/* Identification Banner Under Image */}
                <div className="p-3 mt-2 flex items-center justify-between">
                  <div>
                    <h3 className="text-base font-extrabold text-slate-900">
                      Grilled Chicken Rice Bowl
                    </h3>
                    <p className="text-xs text-slate-500">1 standard bowl (approx. 420g)</p>
                  </div>
                  <div className="px-2.5 py-1 rounded-xl bg-brand-50 border border-brand-200 text-brand-700 text-xs font-bold flex items-center gap-1">
                    <ShieldCheck className="w-3.5 h-3.5" />
                    <span>94.7%</span>
                  </div>
                </div>

                {/* Floating Nutrition Cards */}
                {/* Floating Card: Calories */}
                <div className="absolute -top-5 -left-6 bg-white/95 backdrop-blur-md rounded-2xl p-3 shadow-card border border-amber-200/80 flex items-center gap-3 animate-float-slow z-30">
                  <div className="w-9 h-9 rounded-xl bg-amber-50 text-amber-600 flex items-center justify-center">
                    <Flame className="w-5 h-5 fill-amber-500" />
                  </div>
                  <div>
                    <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Calories</p>
                    <p className="text-sm font-extrabold text-slate-900">650 kcal</p>
                  </div>
                </div>

                {/* Floating Card: Protein */}
                <div className="absolute -bottom-5 -left-4 bg-white/95 backdrop-blur-md rounded-2xl p-3 shadow-card border border-cyan-200/80 flex items-center gap-3 animate-float-reverse z-30">
                  <div className="w-9 h-9 rounded-xl bg-cyan-50 text-cyan-600 flex items-center justify-center">
                    <Beef className="w-5 h-5 text-cyan-600" />
                  </div>
                  <div>
                    <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Protein</p>
                    <p className="text-sm font-extrabold text-slate-900">38 g</p>
                  </div>
                </div>

                {/* Floating Card: Carbs */}
                <div className="absolute top-1/3 -right-6 bg-white/95 backdrop-blur-md rounded-2xl p-3 shadow-card border border-emerald-200/80 flex items-center gap-3 animate-float-slow z-30">
                  <div className="w-9 h-9 rounded-xl bg-emerald-50 text-emerald-600 flex items-center justify-center">
                    <Wheat className="w-5 h-5 text-emerald-600" />
                  </div>
                  <div>
                    <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Carbs</p>
                    <p className="text-sm font-extrabold text-slate-900">72 g</p>
                  </div>
                </div>

                {/* Floating Card: Fat */}
                <div className="absolute -bottom-4 -right-4 bg-white/95 backdrop-blur-md rounded-2xl p-3 shadow-card border border-rose-200/80 flex items-center gap-3 animate-float-reverse z-30">
                  <div className="w-9 h-9 rounded-xl bg-rose-50 text-rose-600 flex items-center justify-center">
                    <Droplet className="w-5 h-5 text-rose-600" />
                  </div>
                  <div>
                    <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Fat</p>
                    <p className="text-sm font-extrabold text-slate-900">19 g</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 bg-white border-y border-slate-200/80">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-2xl mx-auto mb-16 space-y-3">
            <span className="text-xs font-bold uppercase tracking-wider text-brand-600 px-3 py-1 rounded-full bg-brand-50 border border-brand-200/60">
              Core Capabilities
            </span>
            <h2 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
              Smarter Nutrition Through Deep Learning
            </h2>
            <p className="text-slate-500 text-sm sm:text-base">
              Harness multi-task convolutional neural networks designed to simultaneously classify meals and predict continuous macro density.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((item, idx) => {
              const Icon = item.icon;
              return (
                <div
                  key={idx}
                  className="rounded-3xl p-6 bg-slate-50/70 border border-slate-200/80 hover:bg-white hover:border-brand-300 shadow-soft hover:shadow-card transition-all duration-300 group flex flex-col justify-between"
                >
                  <div>
                    <div
                      className={`w-12 h-12 rounded-2xl ${item.color} flex items-center justify-center mb-5 shadow-sm group-hover:scale-110 transition-transform`}
                    >
                      <Icon className="w-6 h-6" />
                    </div>
                    <h3 className="text-lg font-bold text-slate-900 mb-2 group-hover:text-brand-700 transition-colors">
                      {item.title}
                    </h3>
                    <p className="text-xs sm:text-sm text-slate-500 leading-relaxed">
                      {item.desc}
                    </p>
                  </div>
                  <div className="pt-4 mt-4 border-t border-slate-200/60 flex items-center text-xs font-bold text-brand-600 group-hover:translate-x-1 transition-transform">
                    <span>Learn more</span>
                    <ChevronRight className="w-3.5 h-3.5 ml-0.5" />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-20 bg-slate-50/60">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-2xl mx-auto mb-16 space-y-3">
            <span className="text-xs font-bold uppercase tracking-wider text-brand-600 px-3 py-1 rounded-full bg-brand-50 border border-brand-200/60">
              Visual Pipeline
            </span>
            <h2 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
              How NutriVision AI Works
            </h2>
            <p className="text-slate-500 text-sm sm:text-base">
              From raw image pixel tensors to calibrated nutritional intelligence in 4 automated steps.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 relative">
            {steps.map((step, idx) => (
              <div
                key={idx}
                className="relative bg-white rounded-3xl p-6 border border-slate-200/80 shadow-soft hover:shadow-card transition-all group"
              >
                <div className="flex items-center justify-between mb-4">
                  <span className="text-3xl font-black text-brand-500/80 font-mono">
                    {step.num}
                  </span>
                  <div className="w-8 h-8 rounded-full bg-brand-50 text-brand-600 flex items-center justify-center text-xs font-bold">
                    ✓
                  </div>
                </div>
                <h3 className="text-base font-bold text-slate-900 mb-2">
                  {step.title}
                </h3>
                <p className="text-xs sm:text-sm text-slate-500 leading-relaxed">
                  {step.desc}
                </p>
              </div>
            ))}
          </div>

          {/* Process Flow Summary Bar */}
          <div className="mt-12 p-4 rounded-2xl bg-white border border-brand-200/80 shadow-soft flex flex-wrap items-center justify-center gap-3 text-xs sm:text-sm font-bold text-slate-700">
            <span className="text-brand-600 font-extrabold">01 Upload</span>
            <ArrowRight className="w-4 h-4 text-slate-300" />
            <span className="text-brand-600 font-extrabold">02 Preprocess</span>
            <ArrowRight className="w-4 h-4 text-slate-300" />
            <span className="text-brand-600 font-extrabold">03 AI Analysis</span>
            <ArrowRight className="w-4 h-4 text-slate-300" />
            <span className="text-brand-600 font-extrabold">04 Nutrition Report</span>
          </div>
        </div>
      </section>

      {/* Technology Section */}
      <section id="technology" className="py-20 bg-white border-y border-slate-200/80">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-2xl mx-auto mb-16 space-y-3">
            <span className="text-xs font-bold uppercase tracking-wider text-brand-600 px-3 py-1 rounded-full bg-brand-50 border border-brand-200/60">
              Tech Stack
            </span>
            <h2 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
              Enterprise AI & Full-Stack Architecture
            </h2>
            <p className="text-slate-500 text-sm sm:text-base">
              Built using cutting-edge deep learning, asynchronous API microservices, and reactive frontend architecture.
            </p>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-2 md:grid-cols-4 gap-4 sm:gap-6">
            {technologies.map((tech, idx) => (
              <div
                key={idx}
                className="p-5 rounded-2xl bg-slate-50/80 border border-slate-200/80 hover:bg-white hover:border-brand-300 hover:shadow-soft transition-all"
              >
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-sm sm:text-base font-extrabold text-slate-900">
                    {tech.name}
                  </h4>
                  <span className="text-[10px] uppercase font-bold px-2 py-0.5 rounded bg-brand-100 text-brand-800">
                    {tech.badge}
                  </span>
                </div>
                <p className="text-xs text-slate-500 font-medium">{tech.role}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-br from-slate-900 via-slate-900 to-brand-950 text-white relative overflow-hidden">
        <div className="absolute top-0 right-0 w-96 h-96 bg-brand-500/15 rounded-full blur-3xl pointer-events-none" />
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center space-y-6 relative z-10">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-brand-600 to-emerald-400 flex items-center justify-center text-white mx-auto shadow-glow-brand">
            <Sparkles className="w-8 h-8" />
          </div>

          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold tracking-tight">
            Ready to analyze your meal?
          </h2>

          <p className="text-slate-300 text-base sm:text-lg max-w-xl mx-auto leading-relaxed">
            Upload your meal image and see the instant breakdown of calories, protein, carbohydrates, and fat.
          </p>

          <div className="pt-2">
            <Link
              to="/analyze"
              className="inline-flex items-center gap-2.5 px-8 py-4 rounded-2xl bg-gradient-to-r from-brand-500 to-emerald-400 hover:from-brand-600 hover:to-emerald-500 text-slate-950 font-extrabold text-base shadow-lg shadow-brand-500/30 transition-all hover:scale-105"
            >
              <Sparkles className="w-5 h-5 text-slate-950" />
              <span>Analyze Your Food</span>
              <ArrowRight className="w-4 h-4 text-slate-950" />
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200/80 py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-8">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div className="md:col-span-2 space-y-3">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-xl bg-brand-600 flex items-center justify-center text-white font-bold">
                  <Sparkles className="w-4 h-4" />
                </div>
                <span className="text-lg font-extrabold text-slate-900">
                  NutriVision<span className="text-brand-600">.AI</span>
                </span>
              </div>
              <p className="text-xs text-slate-500 max-w-sm leading-relaxed">
                Turn your food photo into nutrition insights. Advanced deep learning platform providing instantaneous calorie and macronutrient regression from single food images.
              </p>
            </div>

            <div>
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-900 mb-3">
                Platform Links
              </h4>
              <ul className="space-y-2 text-xs font-medium text-slate-500">
                <li><Link to="/analyze" className="hover:text-brand-600">Analyze Food</Link></li>
                <li><Link to="/dashboard" className="hover:text-brand-600">Dashboard</Link></li>
                <li><Link to="/history" className="hover:text-brand-600">Scan History</Link></li>
                <li><Link to="/profile" className="hover:text-brand-600">Profile & Goals</Link></li>
              </ul>
            </div>

            <div>
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-900 mb-3">
                Technology
              </h4>
              <ul className="space-y-2 text-xs font-medium text-slate-500">
                <li>PyTorch & Torchvision</li>
                <li>FastAPI REST Backend</li>
                <li>React & Tailwind CSS</li>
                <li>Docker Ready</li>
              </ul>
            </div>
          </div>

          <DisclaimerBanner />

          <div className="pt-6 border-t border-slate-100 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-400 gap-4">
            <p>© 2026 NutriVision AI. All rights reserved.</p>
            <p>Built with PyTorch Multi-Task Deep Learning & FastAPI</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
