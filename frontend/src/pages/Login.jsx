import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Sparkles, Mail, Lock, ArrowRight, ShieldCheck, Zap, AlertCircle } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const { login, loginDemo, isLoading } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    if (!email || !password) {
      setError('Please enter both email and password.');
      return;
    }

    const res = await login(email, password);
    if (res.success) {
      navigate('/dashboard');
    } else {
      setError(res.error || 'Invalid credentials. Try the demo account below.');
    }
  };

  const handleDemoLogin = async () => {
    setError('');
    const res = await loginDemo();
    if (res.success) {
      navigate('/dashboard');
    }
  };

  return (
    <div className="min-h-screen bg-slate-50/70 flex flex-col justify-center py-12 px-4 sm:px-6 lg:px-8 relative overflow-hidden">
      {/* Glow Orbs */}
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-brand-400/10 rounded-full blur-3xl pointer-events-none" />

      <div className="sm:mx-auto sm:w-full sm:max-w-md text-center space-y-3">
        <Link to="/" className="inline-flex items-center gap-2 group mb-2">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-brand-700 to-brand-500 flex items-center justify-center text-white shadow-md shadow-brand-500/20 group-hover:scale-105 transition-transform">
            <Sparkles className="w-5 h-5 text-white animate-pulse" />
          </div>
          <span className="text-2xl font-extrabold tracking-tight text-slate-900">
            NutriVision<span className="text-brand-600">.AI</span>
          </span>
        </Link>
        <h2 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
          Welcome Back
        </h2>
        <p className="text-xs sm:text-sm text-slate-500 font-medium">
          Log in to access your meal analysis history & personalized dietary goals.
        </p>
      </div>

      <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
        <div className="bg-white py-8 px-6 sm:px-10 rounded-3xl border border-slate-200/90 shadow-card space-y-6">
          {/* Quick Demo 1-Click Button */}
          <button
            type="button"
            onClick={handleDemoLogin}
            disabled={isLoading}
            className="w-full flex items-center justify-center gap-2 py-3 px-4 rounded-xl bg-brand-50 hover:bg-brand-100 border border-brand-300 text-brand-800 text-sm font-bold shadow-sm transition-all hover:scale-[1.01] active:scale-[0.99] disabled:opacity-50"
          >
            <Zap className="w-4 h-4 text-brand-600 fill-brand-600" />
            <span>Continue with Demo Account (1-Click)</span>
          </button>

          <div className="relative flex items-center justify-center">
            <div className="border-t border-slate-200 w-full" />
            <span className="bg-white px-3 text-[11px] font-bold text-slate-400 uppercase tracking-wider">
              Or with credentials
            </span>
          </div>

          {error && (
            <div className="p-3.5 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-xs flex items-center gap-2.5">
              <AlertCircle className="w-4 h-4 flex-shrink-0 text-rose-500" />
              <span>{error}</span>
            </div>
          )}

          <form className="space-y-4" onSubmit={handleSubmit}>
            <div>
              <label className="block text-xs font-bold text-slate-700 mb-1.5" htmlFor="email">
                Email Address
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
                  <Mail className="w-4 h-4" />
                </div>
                <input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="karan@example.com"
                  className="w-full pl-10 pr-4 py-2.5 rounded-xl border border-slate-300 bg-white text-sm text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-brand-500 transition-colors"
                />
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="block text-xs font-bold text-slate-700" htmlFor="password">
                  Password
                </label>
                <a href="#forgot" onClick={(e) => { e.preventDefault(); alert("Demo mode: Use any password with 4+ characters or click 'Continue with Demo Account'."); }} className="text-xs font-semibold text-brand-600 hover:text-brand-700">
                  Forgot password?
                </a>
              </div>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
                  <Lock className="w-4 h-4" />
                </div>
                <input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full pl-10 pr-4 py-2.5 rounded-xl border border-slate-300 bg-white text-sm text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-brand-500 transition-colors"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full flex items-center justify-center gap-2 py-3 px-4 rounded-xl bg-slate-900 hover:bg-slate-800 text-white font-bold text-sm shadow-md transition-all hover:scale-[1.01] active:scale-[0.99] disabled:opacity-50"
            >
              <span>{isLoading ? 'Signing in...' : 'Sign In'}</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </form>

          <p className="text-center text-xs text-slate-500 font-medium pt-2 border-t border-slate-100">
            Don't have an account?{' '}
            <Link to="/signup" className="font-bold text-brand-600 hover:text-brand-700">
              Create account
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
