import React, { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import Sidebar from '../components/Sidebar';
import DashboardHeader from '../components/DashboardHeader';
import ConfidenceBadge from '../components/ConfidenceBadge';
import DisclaimerBanner from '../components/DisclaimerBanner';
import Toast from '../components/Toast';
import { useHistory } from '../context/HistoryContext';
import {
  Search,
  Filter,
  ArrowUpDown,
  Trash2,
  ExternalLink,
  Flame,
  Beef,
  Wheat,
  Droplet,
  Calendar,
  Grid,
  List,
  Eye,
  X,
  Sparkles
} from 'lucide-react';

export default function HistoryPage() {
  const { history, deleteHistoryItem, setActiveResult, clearHistory } = useHistory();
  const navigate = useNavigate();

  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [sortBy, setSortBy] = useState('date-desc');
  const [viewMode, setViewMode] = useState('grid'); // 'grid' or 'table'
  const [activeModalItem, setActiveModalItem] = useState(null);
  const [toastMsg, setToastMsg] = useState(null);

  // Filter & Sort Pipeline
  const filteredHistory = useMemo(() => {
    return history
      .filter((item) => {
        const matchesSearch =
          item.foodClass.toLowerCase().includes(searchTerm.toLowerCase()) ||
          (item.dietaryTags && item.dietaryTags.some((t) => t.toLowerCase().includes(searchTerm.toLowerCase())));

        if (!matchesSearch) return false;

        if (selectedCategory === 'high-protein') {
          return (item.protein || 0) >= 25;
        } else if (selectedCategory === 'low-calorie') {
          return (item.calories || 0) <= 450;
        } else if (selectedCategory === 'high-calorie') {
          return (item.calories || 0) >= 650;
        }
        return true;
      })
      .sort((a, b) => {
        if (sortBy === 'date-desc') {
          return new Date(b.timestamp) - new Date(a.timestamp);
        } else if (sortBy === 'date-asc') {
          return new Date(a.timestamp) - new Date(b.timestamp);
        } else if (sortBy === 'cal-desc') {
          return (b.calories || 0) - (a.calories || 0);
        } else if (sortBy === 'cal-asc') {
          return (a.calories || 0) - (b.calories || 0);
        } else if (sortBy === 'protein-desc') {
          return (b.protein || 0) - (a.protein || 0);
        }
        return 0;
      });
  }, [history, searchTerm, selectedCategory, sortBy]);

  const handleOpenDetail = (item) => {
    setActiveResult(item);
    navigate('/result');
  };

  const handleDelete = async (e, id) => {
    e.stopPropagation();
    await deleteHistoryItem(id);
    setToastMsg('Record removed from history');
  };

  const formatDate = (isoString) => {
    try {
      const d = new Date(isoString);
      return new Intl.DateTimeFormat('en-US', {
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: 'numeric',
      }).format(d);
    } catch (e) {
      return 'Recently';
    }
  };

  return (
    <div className="min-h-screen bg-slate-50/60 flex">
      <Sidebar />

      <main className="flex-1 lg:pl-72 flex flex-col min-w-0">
        <div className="max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8 space-y-8">
          <DashboardHeader
            title="Analysis History"
            subtitle="Browse, search, and review all previous food scans and nutritional breakdowns."
            showAction={true}
          />

          {/* Controls Bar: Search, Filters, Sort, View Toggle */}
          <div className="bg-white rounded-3xl p-5 border border-slate-200/80 shadow-soft space-y-4">
            <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-4">
              {/* Search Bar */}
              <div className="relative flex-1">
                <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
                <input
                  type="text"
                  placeholder="Search by food name or tag (e.g. Biryani, Dosa, Protein)..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-10 pr-4 py-2.5 rounded-xl border border-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-brand-500 bg-slate-50/50"
                />
              </div>

              {/* Filters & Sorting */}
              <div className="flex flex-wrap items-center gap-2.5">
                {/* Category Filter */}
                <div className="flex items-center gap-1 bg-slate-50 border border-slate-200 rounded-xl px-2.5 py-1.5 text-xs font-semibold text-slate-700">
                  <Filter className="w-3.5 h-3.5 text-slate-500" />
                  <select
                    value={selectedCategory}
                    onChange={(e) => setSelectedCategory(e.target.value)}
                    className="bg-transparent focus:outline-none cursor-pointer"
                  >
                    <option value="all">All Foods</option>
                    <option value="high-protein">High Protein (25g+)</option>
                    <option value="low-calorie">Under 450 kcal</option>
                    <option value="high-calorie">650+ kcal</option>
                  </select>
                </div>

                {/* Sort dropdown */}
                <div className="flex items-center gap-1 bg-slate-50 border border-slate-200 rounded-xl px-2.5 py-1.5 text-xs font-semibold text-slate-700">
                  <ArrowUpDown className="w-3.5 h-3.5 text-slate-500" />
                  <select
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value)}
                    className="bg-transparent focus:outline-none cursor-pointer"
                  >
                    <option value="date-desc">Newest First</option>
                    <option value="date-asc">Oldest First</option>
                    <option value="cal-desc">Highest Calories</option>
                    <option value="cal-asc">Lowest Calories</option>
                    <option value="protein-desc">Highest Protein</option>
                  </select>
                </div>

                {/* Grid / Table Mode toggle */}
                <div className="flex items-center border border-slate-200 rounded-xl p-0.5 bg-slate-50">
                  <button
                    onClick={() => setViewMode('grid')}
                    className={`p-1.5 rounded-lg transition-colors ${
                      viewMode === 'grid' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-400 hover:text-slate-700'
                    }`}
                    title="Grid View"
                  >
                    <Grid className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => setViewMode('table')}
                    className={`p-1.5 rounded-lg transition-colors ${
                      viewMode === 'table' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-400 hover:text-slate-700'
                    }`}
                    title="Table View"
                  >
                    <List className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>

            {/* Quick Summary Pill */}
            <div className="flex items-center justify-between text-xs text-slate-500 pt-2 border-t border-slate-100">
              <span>
                Showing <strong className="text-slate-800">{filteredHistory.length}</strong> recorded analysis items
              </span>
              {history.length > 0 && (
                <button
                  onClick={() => {
                    if (confirm('Are you sure you want to clear your local analysis history?')) {
                      clearHistory();
                      setToastMsg('Analysis history cleared.');
                    }
                  }}
                  className="text-slate-400 hover:text-rose-600 transition-colors"
                >
                  Clear History
                </button>
              )}
            </div>
          </div>

          {/* Display Items: Grid or Table */}
          {filteredHistory.length === 0 ? (
            <div className="bg-white rounded-3xl p-12 text-center border border-slate-200/80 shadow-soft space-y-4">
              <div className="w-16 h-16 rounded-2xl bg-brand-50 text-brand-600 flex items-center justify-center mx-auto">
                <Search className="w-8 h-8" />
              </div>
              <h3 className="text-lg font-bold text-slate-900">No food records found</h3>
              <p className="text-xs text-slate-500 max-w-sm mx-auto">
                Try clearing your search filters or upload a new food photo to start building your history.
              </p>
            </div>
          ) : viewMode === 'grid' ? (
            /* Card Grid View */
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filteredHistory.map((item) => (
                <div
                  key={item.id}
                  onClick={() => handleOpenDetail(item)}
                  className="bg-white rounded-3xl overflow-hidden border border-slate-200/80 shadow-soft hover:shadow-card transition-all duration-200 cursor-pointer group flex flex-col justify-between"
                >
                  <div>
                    {/* Food Thumbnail */}
                    <div className="relative aspect-video w-full overflow-hidden bg-slate-100">
                      <img
                        src={item.imageUrl}
                        alt={item.foodClass}
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                        loading="lazy"
                      />
                      <div className="absolute top-3 left-3">
                        <ConfidenceBadge score={item.confidenceScore} size="sm" />
                      </div>
                      <button
                        onClick={(e) => handleDelete(e, item.id)}
                        className="absolute top-3 right-3 p-1.5 rounded-lg bg-slate-900/60 backdrop-blur-sm text-white hover:bg-rose-600 transition-colors opacity-0 group-hover:opacity-100"
                        title="Delete Record"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>

                    {/* Content */}
                    <div className="p-5 space-y-3">
                      <div>
                        <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
                          <span className="flex items-center gap-1">
                            <Calendar className="w-3 h-3" />
                            {formatDate(item.timestamp)}
                          </span>
                        </div>
                        <h3 className="text-base font-extrabold text-slate-900 truncate group-hover:text-brand-600 transition-colors">
                          {item.foodClass}
                        </h3>
                      </div>

                      {/* 4 Quick Macro Badges */}
                      <div className="grid grid-cols-4 gap-2 pt-2 border-t border-slate-100 text-center">
                        <div className="p-2 rounded-xl bg-amber-50/70 border border-amber-100">
                          <p className="text-[10px] font-bold text-amber-700">Calories</p>
                          <p className="text-xs font-extrabold text-slate-900 mt-0.5">{item.calories}</p>
                        </div>
                        <div className="p-2 rounded-xl bg-cyan-50/70 border border-cyan-100">
                          <p className="text-[10px] font-bold text-cyan-700">Protein</p>
                          <p className="text-xs font-extrabold text-slate-900 mt-0.5">{item.protein}g</p>
                        </div>
                        <div className="p-2 rounded-xl bg-emerald-50/70 border border-emerald-100">
                          <p className="text-[10px] font-bold text-emerald-700">Carbs</p>
                          <p className="text-xs font-extrabold text-slate-900 mt-0.5">{item.carbohydrates || item.carbs}g</p>
                        </div>
                        <div className="p-2 rounded-xl bg-rose-50/70 border border-rose-100">
                          <p className="text-[10px] font-bold text-rose-700">Fat</p>
                          <p className="text-xs font-extrabold text-slate-900 mt-0.5">{item.fat}g</p>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="px-5 py-3 bg-slate-50/70 border-t border-slate-100 flex items-center justify-between text-xs font-bold text-brand-600">
                    <span>View full analysis</span>
                    <ExternalLink className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition-transform" />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            /* Table View */
            <div className="bg-white rounded-3xl overflow-hidden border border-slate-200/80 shadow-soft">
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 font-bold uppercase tracking-wider text-[10px]">
                    <tr>
                      <th className="py-3.5 px-4">Dish</th>
                      <th className="py-3.5 px-4">Calories</th>
                      <th className="py-3.5 px-4">Protein</th>
                      <th className="py-3.5 px-4">Carbs</th>
                      <th className="py-3.5 px-4">Fat</th>
                      <th className="py-3.5 px-4">Confidence</th>
                      <th className="py-3.5 px-4">Date</th>
                      <th className="py-3.5 px-4 text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {filteredHistory.map((item) => (
                      <tr
                        key={item.id}
                        onClick={() => handleOpenDetail(item)}
                        className="hover:bg-slate-50 cursor-pointer transition-colors"
                      >
                        <td className="py-3.5 px-4">
                          <div className="flex items-center gap-3">
                            <img
                              src={item.imageUrl}
                              alt={item.foodClass}
                              className="w-10 h-10 rounded-xl object-cover border border-slate-200"
                            />
                            <span className="font-bold text-slate-900 text-sm">{item.foodClass}</span>
                          </div>
                        </td>
                        <td className="py-3.5 px-4 font-bold text-slate-900">{item.calories} kcal</td>
                        <td className="py-3.5 px-4 font-semibold text-cyan-600">{item.protein} g</td>
                        <td className="py-3.5 px-4 font-semibold text-emerald-600">{item.carbohydrates || item.carbs} g</td>
                        <td className="py-3.5 px-4 font-semibold text-rose-600">{item.fat} g</td>
                        <td className="py-3.5 px-4">
                          <ConfidenceBadge score={item.confidenceScore} size="sm" />
                        </td>
                        <td className="py-3.5 px-4 text-slate-500 font-medium">{formatDate(item.timestamp)}</td>
                        <td className="py-3.5 px-4 text-right">
                          <div className="flex items-center justify-end gap-2">
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                handleOpenDetail(item);
                              }}
                              className="p-1.5 text-slate-400 hover:text-brand-600 hover:bg-brand-50 rounded-lg transition-colors"
                              title="View Details"
                            >
                              <Eye className="w-4 h-4" />
                            </button>
                            <button
                              onClick={(e) => handleDelete(e, item.id)}
                              className="p-1.5 text-slate-400 hover:text-rose-600 hover:bg-rose-50 rounded-lg transition-colors"
                              title="Delete Record"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <DisclaimerBanner />
        </div>
      </main>

      <Toast message={toastMsg} onClose={() => setToastMsg(null)} />
    </div>
  );
}
