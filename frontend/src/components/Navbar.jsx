import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Sparkles, Activity, ShieldCheck, Menu, X, ArrowRight, LayoutDashboard, LogIn } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export default function Navbar() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const { isAuthenticated, user } = useAuth();
  const location = useLocation();

  const isCurrent = (path) => location.pathname === path;

  return (
    <header className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-slate-200/80 transition-all">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 md:h-20">
          {/* Logo & Brand */}
          <Link to="/" className="flex items-center gap-3 group">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-brand-700 to-brand-500 flex items-center justify-center text-white shadow-md shadow-brand-500/20 group-hover:scale-105 transition-transform duration-200">
              <Sparkles className="w-5 h-5 text-white animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="text-xl font-extrabold tracking-tight text-slate-900 font-sans">
                  NutriVision<span className="text-brand-600">.AI</span>
                </span>
                <span className="text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.5 rounded bg-brand-50 text-brand-700 border border-brand-200/60">
                  PyTorch
                </span>
              </div>
              <p className="text-[11px] text-slate-500 font-medium hidden sm:block">
                AI Food & Calorie Estimator
              </p>
            </div>
          </Link>

          {/* Desktop Navigation Links */}
          <nav className="hidden md:flex items-center gap-8">
            <Link
              to="/"
              className={`text-sm font-semibold transition-colors ${
                isCurrent('/') ? 'text-brand-600' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Home
            </Link>
            <a
              href="#how-it-works"
              className="text-sm font-semibold text-slate-600 hover:text-slate-900 transition-colors"
            >
              How It Works
            </a>
            <a
              href="#features"
              className="text-sm font-semibold text-slate-600 hover:text-slate-900 transition-colors"
            >
              Features
            </a>
            <a
              href="#technology"
              className="text-sm font-semibold text-slate-600 hover:text-slate-900 transition-colors"
            >
              Technology
            </a>
          </nav>

          {/* Action / Auth Buttons */}
          <div className="hidden md:flex items-center gap-3">
            {isAuthenticated ? (
              <Link
                to="/dashboard"
                className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 text-white text-sm font-semibold shadow-sm transition-all hover:shadow-md"
              >
                <LayoutDashboard className="w-4 h-4 text-brand-400" />
                <span>Go to Dashboard</span>
                <ArrowRight className="w-4 h-4" />
              </Link>
            ) : (
              <>
                <Link
                  to="/login"
                  className="px-4 py-2.5 text-sm font-semibold text-slate-700 hover:text-brand-600 transition-colors"
                >
                  Sign In
                </Link>
                <Link
                  to="/analyze"
                  className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-brand-600 to-emerald-500 hover:from-brand-700 hover:to-emerald-600 text-white text-sm font-semibold shadow-md shadow-brand-500/25 transition-all hover:scale-[1.02]"
                >
                  <Sparkles className="w-4 h-4" />
                  <span>Analyze Food</span>
                </Link>
              </>
            )}
          </div>

          {/* Mobile menu toggle button */}
          <button
            onClick={() => setMobileOpen(!mobileOpen)}
            className="md:hidden p-2 rounded-lg text-slate-600 hover:text-slate-900 hover:bg-slate-100"
            aria-label="Toggle Navigation Menu"
          >
            {mobileOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>
      </div>

      {/* Mobile Drawer */}
      {mobileOpen && (
        <div className="md:hidden bg-white border-b border-slate-200 px-4 pt-2 pb-6 space-y-3">
          <Link
            to="/"
            onClick={() => setMobileOpen(false)}
            className="block py-2 text-base font-semibold text-slate-800 hover:text-brand-600"
          >
            Home
          </Link>
          <a
            href="#how-it-works"
            onClick={() => setMobileOpen(false)}
            className="block py-2 text-base font-semibold text-slate-800 hover:text-brand-600"
          >
            How It Works
          </a>
          <a
            href="#features"
            onClick={() => setMobileOpen(false)}
            className="block py-2 text-base font-semibold text-slate-800 hover:text-brand-600"
          >
            Features
          </a>
          <a
            href="#technology"
            onClick={() => setMobileOpen(false)}
            className="block py-2 text-base font-semibold text-slate-800 hover:text-brand-600"
          >
            Technology
          </a>

          <div className="pt-4 border-t border-slate-100 space-y-2">
            {isAuthenticated ? (
              <Link
                to="/dashboard"
                onClick={() => setMobileOpen(false)}
                className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-brand-600 text-white font-semibold text-sm shadow-md"
              >
                <LayoutDashboard className="w-4 h-4" />
                <span>Open Dashboard</span>
              </Link>
            ) : (
              <>
                <Link
                  to="/login"
                  onClick={() => setMobileOpen(false)}
                  className="w-full flex items-center justify-center py-2.5 rounded-xl border border-slate-300 text-slate-700 font-semibold text-sm"
                >
                  Log In
                </Link>
                <Link
                  to="/analyze"
                  onClick={() => setMobileOpen(false)}
                  className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl bg-brand-600 text-white font-semibold text-sm shadow-md shadow-brand-500/20"
                >
                  <Sparkles className="w-4 h-4" />
                  <span>Analyze Food Now</span>
                </Link>
              </>
            )}
          </div>
        </div>
      )}
    </header>
  );
}
