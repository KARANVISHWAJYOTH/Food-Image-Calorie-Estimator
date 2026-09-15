import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Sidebar from '../components/Sidebar';
import DashboardHeader from '../components/DashboardHeader';
import DisclaimerBanner from '../components/DisclaimerBanner';
import Toast from '../components/Toast';
import { useAuth } from '../context/AuthContext';
import { useHistory } from '../context/HistoryContext';
import {
  User,
  Mail,
  Calendar,
  Sparkles,
  Flame,
  Beef,
  Wheat,
  Droplet,
  Save,
  Lock,
  LogOut,
  ShieldCheck,
  CheckCircle2,
  Settings
} from 'lucide-react';

export default function Profile() {
  const { user, updateProfile, logout } = useAuth();
  const { history, getStats } = useHistory();
  const stats = getStats();
  const navigate = useNavigate();

  const [name, setName] = useState(user?.name || 'Karan');
  const [calorieGoal, setCalorieGoal] = useState(user?.dailyCalorieGoal || 2200);
  const [proteinGoal, setProteinGoal] = useState(user?.dailyProteinGoal || 130);
  const [dietaryPreference, setDietaryPreference] = useState(user?.dietaryPreference || 'Balanced & High-Protein');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [toastMsg, setToastMsg] = useState(null);

  const handleUpdateProfile = (e) => {
    e.preventDefault();
    updateProfile({
      name,
      dailyCalorieGoal: Number(calorieGoal),
      dailyProteinGoal: Number(proteinGoal),
      dietaryPreference,
    });
    setToastMsg('Profile settings updated successfully!');
  };

  const handlePasswordChange = (e) => {
    e.preventDefault();
    if (!password || password.length < 4) {
      setToastMsg('Password must be at least 4 characters');
      return;
    }
    if (password !== confirmPassword) {
      setToastMsg('Passwords do not match');
      return;
    }
    setPassword('');
    setConfirmPassword('');
    setToastMsg('Security password updated successfully!');
  };

  const formattedJoinDate = user?.createdAt
    ? new Intl.DateTimeFormat('en-US', { month: 'long', year: 'numeric' }).format(new Date(user.createdAt))
    : 'January 2026';

  return (
    <div className="min-h-screen bg-slate-50/60 flex">
      <Sidebar />

      <main className="flex-1 lg:pl-72 flex flex-col min-w-0">
        <div className="max-w-5xl w-full mx-auto p-4 sm:p-6 lg:p-8 space-y-8">
          <DashboardHeader
            title="Profile & Nutrition Goals"
            subtitle="Manage your personal profile, daily caloric allowances, and dietary preferences."
            showAction={false}
          />

          {/* User Profile Card */}
          <div className="bg-white rounded-3xl p-6 sm:p-8 border border-slate-200/80 shadow-soft">
            <div className="flex flex-col sm:flex-row items-center sm:items-start gap-6">
              <div className="relative">
                <img
                  src={user?.avatarUrl || 'https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?auto=format&fit=crop&w=200&q=80'}
                  alt={user?.name || 'User'}
                  className="w-24 h-24 rounded-3xl object-cover border-2 border-brand-200 shadow-md"
                />
                <span className="absolute -bottom-2 -right-2 px-2 py-0.5 rounded-full bg-brand-500 text-white text-[10px] font-bold shadow-sm">
                  PRO
                </span>
              </div>

              <div className="space-y-2 text-center sm:text-left flex-1">
                <div className="flex flex-wrap items-center justify-center sm:justify-start gap-2">
                  <h2 className="text-2xl font-extrabold text-slate-900 tracking-tight">
                    {user?.name || 'Karan'}
                  </h2>
                  <span className="px-2.5 py-0.5 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-700 text-xs font-bold">
                    Active Member
                  </span>
                </div>

                <p className="text-xs sm:text-sm text-slate-500 font-medium flex items-center justify-center sm:justify-start gap-1.5">
                  <Mail className="w-3.5 h-3.5" />
                  <span>{user?.email || 'karan@nutrivision.ai'}</span>
                </p>

                <div className="pt-2 flex flex-wrap items-center justify-center sm:justify-start gap-4 text-xs text-slate-500">
                  <span className="flex items-center gap-1">
                    <Calendar className="w-3.5 h-3.5 text-slate-400" />
                    Joined {formattedJoinDate}
                  </span>
                  <span>•</span>
                  <span className="flex items-center gap-1 font-bold text-slate-700">
                    <Sparkles className="w-3.5 h-3.5 text-brand-600" />
                    {stats.totalMeals} Total Scans
                  </span>
                </div>
              </div>

              <button
                onClick={() => {
                  logout();
                  navigate('/login');
                }}
                className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl border border-slate-200 text-xs font-bold text-rose-600 hover:bg-rose-50 transition-colors"
              >
                <LogOut className="w-4 h-4" />
                <span>Log Out</span>
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-start">
            {/* Nutrition Target Settings */}
            <div className="bg-white rounded-3xl p-6 sm:p-8 border border-slate-200/80 shadow-soft space-y-6">
              <div className="flex items-center gap-2 pb-3 border-b border-slate-100">
                <Flame className="w-5 h-5 text-brand-600" />
                <h3 className="text-base font-extrabold text-slate-900">
                  Daily Target Allowances
                </h3>
              </div>

              <form onSubmit={handleUpdateProfile} className="space-y-4">
                <div>
                  <label className="block text-xs font-bold text-slate-700 mb-1.5">
                    Display Name
                  </label>
                  <input
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    className="w-full px-4 py-2.5 rounded-xl border border-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 bg-slate-50/50"
                  />
                </div>

                <div>
                  <label className="block text-xs font-bold text-slate-700 mb-1.5">
                    Daily Calorie Target (kcal)
                  </label>
                  <input
                    type="number"
                    value={calorieGoal}
                    onChange={(e) => setCalorieGoal(e.target.value)}
                    className="w-full px-4 py-2.5 rounded-xl border border-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 bg-slate-50/50"
                  />
                </div>

                <div>
                  <label className="block text-xs font-bold text-slate-700 mb-1.5">
                    Daily Protein Target (grams)
                  </label>
                  <input
                    type="number"
                    value={proteinGoal}
                    onChange={(e) => setProteinGoal(e.target.value)}
                    className="w-full px-4 py-2.5 rounded-xl border border-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 bg-slate-50/50"
                  />
                </div>

                <div>
                  <label className="block text-xs font-bold text-slate-700 mb-1.5">
                    Dietary Focus / Style
                  </label>
                  <select
                    value={dietaryPreference}
                    onChange={(e) => setDietaryPreference(e.target.value)}
                    className="w-full px-4 py-2.5 rounded-xl border border-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 bg-slate-50/50 cursor-pointer"
                  >
                    <option value="Balanced & High-Protein">Balanced & High-Protein</option>
                    <option value="Low Carb / Keto">Low Carb / Keto</option>
                    <option value="Mediterranean">Mediterranean Health</option>
                    <option value="Vegetarian Muscle">Vegetarian Clean Bulk</option>
                    <option value="Caloric Deficit">Caloric Deficit / Fat Loss</option>
                  </select>
                </div>

                <button
                  type="submit"
                  className="w-full inline-flex items-center justify-center gap-2 py-3 px-4 rounded-xl bg-brand-600 hover:bg-brand-700 text-white font-bold text-sm shadow-md shadow-brand-500/20 transition-all hover:scale-[1.01]"
                >
                  <Save className="w-4 h-4" />
                  <span>Save Target Settings</span>
                </button>
              </form>
            </div>

            {/* Security & Password */}
            <div className="bg-white rounded-3xl p-6 sm:p-8 border border-slate-200/80 shadow-soft space-y-6">
              <div className="flex items-center gap-2 pb-3 border-b border-slate-100">
                <Lock className="w-5 h-5 text-slate-700" />
                <h3 className="text-base font-extrabold text-slate-900">
                  Security & Password
                </h3>
              </div>

              <form onSubmit={handlePasswordChange} className="space-y-4">
                <div>
                  <label className="block text-xs font-bold text-slate-700 mb-1.5">
                    New Password
                  </label>
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••"
                    className="w-full px-4 py-2.5 rounded-xl border border-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 bg-slate-50/50"
                  />
                </div>

                <div>
                  <label className="block text-xs font-bold text-slate-700 mb-1.5">
                    Confirm New Password
                  </label>
                  <input
                    type="password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    placeholder="••••••••"
                    className="w-full px-4 py-2.5 rounded-xl border border-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 bg-slate-50/50"
                  />
                </div>

                <p className="text-[11px] text-slate-400">
                  Passwords must be 4+ characters. Simulated credentials in demo mode.
                </p>

                <button
                  type="submit"
                  className="w-full inline-flex items-center justify-center gap-2 py-3 px-4 rounded-xl bg-slate-900 hover:bg-slate-800 text-white font-bold text-sm shadow-md transition-all hover:scale-[1.01]"
                >
                  <Lock className="w-4 h-4" />
                  <span>Update Password</span>
                </button>
              </form>
            </div>
          </div>

          <DisclaimerBanner />
        </div>
      </main>

      <Toast message={toastMsg} onClose={() => setToastMsg(null)} />
    </div>
  );
}
