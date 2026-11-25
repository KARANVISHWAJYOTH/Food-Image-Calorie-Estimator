import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Sidebar from '../components/Sidebar';
import DashboardHeader from '../components/DashboardHeader';
import DisclaimerBanner from '../components/DisclaimerBanner';
import Toast from '../components/Toast';
import { useAuth } from '../context/AuthContext';
import { useHistory } from '../context/HistoryContext';
import { changePasswordApi } from '../services/api';
import {
  Mail,
  Calendar,
  Sparkles,
  Flame,
  Save,
  Lock,
  LogOut,
} from 'lucide-react';

export default function Profile() {
  const { user, updateProfile, logout } = useAuth();
  const { getStats } = useHistory();
  const stats = getStats();
  const navigate = useNavigate();

  const [name, setName] = useState(user?.name || '');
  const [calorieGoal, setCalorieGoal] = useState(user?.dailyCalorieGoal ?? '');
  const [proteinGoal, setProteinGoal] = useState(user?.dailyProteinGoal ?? '');
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPasswordForm, setShowPasswordForm] = useState(false);
  const [isSavingProfile, setIsSavingProfile] = useState(false);
  const [isChangingPassword, setIsChangingPassword] = useState(false);
  const [toastMsg, setToastMsg] = useState(null);

  const formattedJoinDate = (() => {
    if (!user?.createdAt) return null;
    const date = new Date(user.createdAt);
    if (Number.isNaN(date.getTime())) return null;
    return new Intl.DateTimeFormat('en-US', { month: 'long', year: 'numeric' }).format(date);
  })();

  const initials = (user?.name || user?.email || '?')
    .split(/[\s@._-]+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0].toUpperCase())
    .join('');

  const handleUpdateProfile = async (e) => {
    e.preventDefault();
    setIsSavingProfile(true);
    try {
      const result = await updateProfile({
        name,
        dailyCalorieGoal: Number(calorieGoal),
        dailyProteinGoal: Number(proteinGoal),
      });
      setToastMsg(result.success ? 'Profile settings updated.' : result.error);
    } finally {
      setIsSavingProfile(false);
    }
  };

  const handlePasswordChange = async (e) => {
    e.preventDefault();
    if (newPassword.length < 8) {
      setToastMsg('New password must be at least 8 characters.');
      return;
    }
    if (newPassword !== confirmPassword) {
      setToastMsg('New passwords do not match.');
      return;
    }
    setIsChangingPassword(true);
    try {
      await changePasswordApi(currentPassword, newPassword);
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
      setShowPasswordForm(false);
      setToastMsg('Password updated.');
    } catch (error) {
      setToastMsg(error.message);
    } finally {
      setIsChangingPassword(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50/60 flex flex-col">
      <Sidebar />

      <main className="flex-1 lg:pl-72 flex flex-col min-w-0">
        <div className="max-w-5xl w-full mx-auto p-4 sm:p-6 lg:p-8 space-y-8">
          <DashboardHeader
            title="Profile & Nutrition Goals"
            subtitle="Manage your personal profile, daily nutrition targets, and account security."
            showAction={false}
          />

          <div className="bg-white rounded-3xl p-6 sm:p-8 border border-slate-200/80 shadow-soft">
            <div className="flex flex-col sm:flex-row items-center sm:items-start gap-6">
              <div className="w-24 h-24 rounded-3xl bg-brand-600 text-white flex items-center justify-center text-3xl font-extrabold shadow-md">
                {initials || '?'}
              </div>

              <div className="space-y-2 text-center sm:text-left flex-1">
                <div className="flex flex-wrap items-center justify-center sm:justify-start gap-2">
                  <h2 className="text-2xl font-extrabold text-slate-900 tracking-tight">
                    {user?.name}
                  </h2>
                  <span className="px-2.5 py-0.5 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-700 text-xs font-bold">
                    {user?.role || 'Member'}
                  </span>
                </div>

                <p className="text-xs sm:text-sm text-slate-500 font-medium flex items-center justify-center sm:justify-start gap-1.5">
                  <Mail className="w-3.5 h-3.5" />
                  <span>{user?.email}</span>
                </p>

                <div className="pt-2 flex flex-wrap items-center justify-center sm:justify-start gap-4 text-xs text-slate-500">
                  {formattedJoinDate && (
                    <span className="flex items-center gap-1">
                      <Calendar className="w-3.5 h-3.5 text-slate-400" />
                      Joined {formattedJoinDate}
                    </span>
                  )}
                  <span className="flex items-center gap-1 font-bold text-slate-700">
                    <Sparkles className="w-3.5 h-3.5 text-brand-600" />
                    {stats.totalMeals} Saved Scans
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
            <div className="bg-white rounded-3xl p-6 sm:p-8 border border-slate-200/80 shadow-soft space-y-6">
              <div className="flex items-center gap-2 pb-3 border-b border-slate-100">
                <Flame className="w-5 h-5 text-brand-600" />
                <h3 className="text-base font-extrabold text-slate-900">
                  Daily Target Allowances
                </h3>
              </div>

              <form onSubmit={handleUpdateProfile} className="space-y-4">
                <div>
                  <label className="block text-xs font-bold text-slate-700 mb-1.5" htmlFor="profile-name">
                    Display Name
                  </label>
                  <input
                    id="profile-name"
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    maxLength={120}
                    required
                    className="w-full px-4 py-2.5 rounded-xl border border-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 bg-slate-50/50"
                  />
                </div>

                <div>
                  <label className="block text-xs font-bold text-slate-700 mb-1.5" htmlFor="profile-calories">
                    Daily Calorie Target (kcal)
                  </label>
                  <input
                    id="profile-calories"
                    type="number"
                    min={500}
                    max={10000}
                    value={calorieGoal}
                    onChange={(e) => setCalorieGoal(e.target.value)}
                    required
                    className="w-full px-4 py-2.5 rounded-xl border border-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 bg-slate-50/50"
                  />
                </div>

                <div>
                  <label className="block text-xs font-bold text-slate-700 mb-1.5" htmlFor="profile-protein">
                    Daily Protein Target (grams)
                  </label>
                  <input
                    id="profile-protein"
                    type="number"
                    min={0}
                    max={1000}
                    value={proteinGoal}
                    onChange={(e) => setProteinGoal(e.target.value)}
                    required
                    className="w-full px-4 py-2.5 rounded-xl border border-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 bg-slate-50/50"
                  />
                </div>

                <button
                  type="submit"
                  disabled={isSavingProfile}
                  className="w-full inline-flex items-center justify-center gap-2 py-3 px-4 rounded-xl bg-brand-600 hover:bg-brand-700 disabled:opacity-60 text-white font-bold text-sm shadow-md shadow-brand-500/20 transition-all hover:scale-[1.01]"
                >
                  <Save className="w-4 h-4" />
                  <span>{isSavingProfile ? 'Saving…' : 'Save Target Settings'}</span>
                </button>
              </form>
            </div>

            <div className="bg-white rounded-3xl p-6 sm:p-8 border border-slate-200/80 shadow-soft space-y-6">
              <div className="flex items-center gap-2 pb-3 border-b border-slate-100">
                <Lock className="w-5 h-5 text-slate-700" />
                <h3 className="text-base font-extrabold text-slate-900">
                  Security & Password
                </h3>
              </div>

              {!showPasswordForm ? (
                <button
                  onClick={() => setShowPasswordForm(true)}
                  className="w-full inline-flex items-center justify-center gap-2 py-3 px-4 rounded-xl bg-slate-900 hover:bg-slate-800 text-white font-bold text-sm shadow-md transition-all hover:scale-[1.01]"
                >
                  <Lock className="w-4 h-4" />
                  <span>Change Password</span>
                </button>
              ) : (
                <form onSubmit={handlePasswordChange} className="space-y-4">
                  <div>
                    <label className="block text-xs font-bold text-slate-700 mb-1.5" htmlFor="current-password">
                      Current Password
                    </label>
                    <input
                      id="current-password"
                      type="password"
                      value={currentPassword}
                      onChange={(e) => setCurrentPassword(e.target.value)}
                      required
                      autoComplete="current-password"
                      className="w-full px-4 py-2.5 rounded-xl border border-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 bg-slate-50/50"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-bold text-slate-700 mb-1.5" htmlFor="new-password">
                      New Password
                    </label>
                    <input
                      id="new-password"
                      type="password"
                      value={newPassword}
                      onChange={(e) => setNewPassword(e.target.value)}
                      minLength={8}
                      required
                      autoComplete="new-password"
                      className="w-full px-4 py-2.5 rounded-xl border border-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 bg-slate-50/50"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-bold text-slate-700 mb-1.5" htmlFor="confirm-password">
                      Confirm New Password
                    </label>
                    <input
                      id="confirm-password"
                      type="password"
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      minLength={8}
                      required
                      autoComplete="new-password"
                      className="w-full px-4 py-2.5 rounded-xl border border-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 bg-slate-50/50"
                    />
                  </div>

                  <p className="text-[11px] text-slate-400">Passwords must be at least 8 characters.</p>

                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={() => {
                        setShowPasswordForm(false);
                        setCurrentPassword('');
                        setNewPassword('');
                        setConfirmPassword('');
                      }}
                      className="flex-1 py-3 px-4 rounded-xl border border-slate-200 text-slate-700 font-bold text-sm hover:bg-slate-50 transition-colors"
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      disabled={isChangingPassword}
                      className="flex-1 inline-flex items-center justify-center gap-2 py-3 px-4 rounded-xl bg-slate-900 hover:bg-slate-800 disabled:opacity-60 text-white font-bold text-sm shadow-md transition-all"
                    >
                      <Lock className="w-4 h-4" />
                      <span>{isChangingPassword ? 'Updating…' : 'Update Password'}</span>
                    </button>
                  </div>
                </form>
              )}
            </div>
          </div>

          <DisclaimerBanner />
        </div>
      </main>

      <Toast message={toastMsg} onClose={() => setToastMsg(null)} />
    </div>
  );
}
