import React, { useState, useEffect } from 'react';
import { NavLink, useNavigate, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  ScanLine,
  History,
  User,
  LogOut,
  Sparkles,
  ChevronRight,
  Activity,
  Menu,
  X,
  Flame,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { checkApiHealth } from '../services/api';

export default function Sidebar() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [apiStatus, setApiStatus] = useState({ checked: false, isLive: false });

  useEffect(() => {
    let isMounted = true;
    checkApiHealth().then((res) => {
      if (isMounted) setApiStatus({ checked: true, isLive: res.isLive });
    });
    return () => {
      isMounted = false;
    };
  }, []);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const navItems = [
    {
      name: 'Dashboard',
      path: '/dashboard',
      icon: LayoutDashboard,
    },
    {
      name: 'Analyze Food',
      path: '/analyze',
      icon: ScanLine,
      badge: 'AI Model',
    },
    {
      name: 'History',
      path: '/history',
      icon: History,
    },
    {
      name: 'Profile & Goals',
      path: '/profile',
      icon: User,
    },
  ];

  return (
    <>
      {/* Mobile Top Bar */}
      <div className="lg:hidden flex items-center justify-between px-4 py-3 bg-white border-b border-slate-200 sticky top-0 z-40">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-brand-700 to-brand-500 flex items-center justify-center text-white shadow-sm">
            <Sparkles className="w-4 h-4" />
          </div>
          <span className="font-extrabold text-slate-900 tracking-tight text-lg">
            NutriVision<span className="text-brand-600">.AI</span>
          </span>
        </div>
        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="p-2 text-slate-600 hover:text-slate-900 rounded-lg hover:bg-slate-100"
          aria-label="Toggle Navigation"
        >
          {mobileOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
        </button>
      </div>

      {/* Backdrop for Mobile */}
      {mobileOpen && (
        <div
          className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm z-40 lg:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Sidebar Container */}
      <aside
        className={`fixed top-0 bottom-0 left-0 z-50 w-72 bg-white border-r border-slate-200/90 flex flex-col justify-between transition-transform duration-300 ease-in-out lg:translate-x-0 ${
          mobileOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Top Header */}
        <div>
          <div className="p-6 border-b border-slate-100 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-brand-700 to-brand-500 flex items-center justify-center text-white shadow-md shadow-brand-500/20">
                <Sparkles className="w-5 h-5 text-white animate-pulse" />
              </div>
              <div>
                <div className="flex items-center gap-1.5">
                  <span className="text-lg font-extrabold tracking-tight text-slate-900">
                    NutriVision<span className="text-brand-600">.AI</span>
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 font-medium">PyTorch Multi-Task CNN</p>
              </div>
            </div>
            {mobileOpen && (
              <button
                onClick={() => setMobileOpen(false)}
                className="lg:hidden p-1.5 text-slate-400 hover:text-slate-700"
              >
                <X className="w-5 h-5" />
              </button>
            )}
          </div>

          {/* Nav links */}
          <div className="px-4 py-6 space-y-1.5">
            <p className="px-3 text-[11px] font-bold uppercase tracking-wider text-slate-400 mb-2">
              Menu Navigation
            </p>
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              return (
                <NavLink
                  key={item.path}
                  to={item.path}
                  onClick={() => setMobileOpen(false)}
                  className={`flex items-center justify-between px-3.5 py-3 rounded-xl text-sm font-semibold transition-all duration-150 ${
                    isActive
                      ? 'bg-brand-50 text-brand-700 font-bold shadow-sm border border-brand-200/60'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <Icon
                      className={`w-5 h-5 ${
                        isActive ? 'text-brand-600' : 'text-slate-400 group-hover:text-slate-600'
                      }`}
                    />
                    <span>{item.name}</span>
                  </div>
                  {item.badge ? (
                    <span className="text-[10px] uppercase font-bold tracking-wide px-2 py-0.5 rounded-full bg-brand-500/10 text-brand-700 border border-brand-300/40">
                      {item.badge}
                    </span>
                  ) : isActive ? (
                    <ChevronRight className="w-4 h-4 text-brand-600" />
                  ) : null}
                </NavLink>
              );
            })}
          </div>

          {/* Quick Calorie Streak Box */}
          <div className="mx-4 p-4 rounded-2xl bg-gradient-to-br from-slate-900 to-slate-800 text-white shadow-md relative overflow-hidden">
            <div className="absolute -right-4 -bottom-4 w-20 h-20 bg-brand-500/20 rounded-full blur-xl pointer-events-none" />
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                <Flame className="w-4 h-4 text-amber-400 fill-amber-400 animate-bounce" />
                Daily Streak
              </span>
              <span className="text-xs font-bold text-brand-400 bg-brand-950/60 px-2 py-0.5 rounded-full border border-brand-500/30">
                5 Days
              </span>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              Target: <span className="text-white font-bold">{user?.dailyCalorieGoal || 2200} kcal</span>
            </p>
            <div className="w-full bg-slate-700 h-2 rounded-full mt-2.5 overflow-hidden">
              <div className="bg-gradient-to-r from-brand-400 to-emerald-400 h-full rounded-full w-3/4" />
            </div>
          </div>
        </div>

        {/* Bottom Section: API Status & User Profile */}
        <div className="p-4 border-t border-slate-100 space-y-3">
          {/* Backend Status Indicator */}
          <div className="flex items-center justify-between px-3 py-2 rounded-xl bg-slate-50 border border-slate-200/70 text-xs">
            <span className="text-slate-500 font-medium">Backend Engine:</span>
            <div className="flex items-center gap-1.5 font-semibold">
              {apiStatus.isLive ? (
                <>
                  <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                  <span className="text-emerald-700">FastAPI Live</span>
                </>
              ) : (
                <>
                  <span className="w-2 h-2 rounded-full bg-blue-500" />
                  <span className="text-blue-700">AI Simulation</span>
                </>
              )}
            </div>
          </div>

          {/* User profile card & logout */}
          <div className="flex items-center justify-between p-2 rounded-xl hover:bg-slate-50 transition-colors">
            <div className="flex items-center gap-2.5 min-w-0">
              <img
                src={user?.avatarUrl || 'https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?auto=format&fit=crop&w=120&q=80'}
                alt={user?.name || 'User'}
                className="w-9 h-9 rounded-full object-cover border border-slate-200"
              />
              <div className="min-w-0">
                <p className="text-xs font-bold text-slate-900 truncate">{user?.name || 'Karan'}</p>
                <p className="text-[11px] text-slate-400 truncate">{user?.email || 'karan@nutrivision.ai'}</p>
              </div>
            </div>
            <button
              onClick={handleLogout}
              title="Log out"
              className="p-1.5 text-slate-400 hover:text-rose-600 hover:bg-rose-50 rounded-lg transition-colors"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>
      </aside>
    </>
  );
}
