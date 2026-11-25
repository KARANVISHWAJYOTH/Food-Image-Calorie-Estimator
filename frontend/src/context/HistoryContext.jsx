import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import {
  getPredictionHistory,
  savePrediction,
  deletePrediction as apiDeletePrediction,
  clearPredictionHistory,
} from '../services/api';
import { useAuth } from './AuthContext';

const HistoryContext = createContext(null);

export const HistoryProvider = ({ children }) => {
  const { user, isInitializing } = useAuth();
  const [history, setHistory] = useState([]);
  const [activeResult, setActiveResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const refreshHistory = useCallback(async (limit = 50, offset = 0) => {
    if (!localStorage.getItem('nutrivision_token')) {
      setHistory([]);
      return;
    }

    setIsLoading(true);
    try {
      const items = await getPredictionHistory(limit, offset);
      setHistory(Array.isArray(items) ? items : []);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Reload history whenever the authenticated user changes (login/logout).
  useEffect(() => {
    if (isInitializing) return;
    if (!user) {
      setHistory([]);
      setActiveResult(null);
      return;
    }
    refreshHistory().catch(() => setHistory([]));
  }, [user, isInitializing, refreshHistory]);

  const addHistoryItem = async (item) => {
    const tempId = `pending_${Date.now()}`;
    const optimistic = {
      id: tempId,
      foodClass: item.foodClass,
      calories: item.estimatedCalories ?? item.calories,
      protein: item.protein,
      carbohydrates: item.carbohydrates ?? item.carbs,
      fat: item.fat,
      fiber: item.fiber ?? 0,
      confidenceScore: item.confidenceScore,
      confidencePercentage: item.confidencePercentage ?? (item.confidenceScore * 100),
      imageUrl: item.imageUrl,
      imagePath: item.imagePath,
      timestamp: item.timestamp || new Date().toISOString(),
      portion: item.portion ?? 1.0,
      servingSize: item.servingSize,
      healthRating: item.healthRating,
      dietaryTags: item.dietaryTags || [],
      healthTips: item.healthTips,
      isLowConfidence: item.isLowConfidence || false,
    };

    setHistory((prev) => [optimistic, ...prev]);
    try {
      const res = await savePrediction(optimistic);
      if (res?.item) {
        setHistory((prev) => prev.map((h) => (h.id === tempId ? res.item : h)));
      }
      return res?.item;
    } catch (error) {
      setHistory((prev) => prev.filter((h) => h.id !== tempId));
      throw error;
    }
  };

  const deleteHistoryItem = async (id) => {
    const removed = history.find((item) => item.id === id);
    setHistory((prev) => prev.filter((item) => item.id !== id));
    try {
      await apiDeletePrediction(id);
    } catch (error) {
      if (removed) {
        setHistory((prev) => [removed, ...prev]);
      }
      throw error;
    }
  };

  const clearHistory = async () => {
    const result = await clearPredictionHistory();
    setHistory([]);
    setActiveResult(null);
    return result;
  };

  const getStats = () => {
    if (!history.length) {
      return {
        totalMeals: 0,
        avgCalories: 0,
        avgProtein: 0,
        thisWeekCount: 0,
        todayCalories: 0,
      };
    }

    const totalMeals = history.length;
    const totalCalories = history.reduce((sum, h) => sum + (Number(h.calories) || 0), 0);
    const totalProtein = history.reduce((sum, h) => sum + (Number(h.protein) || 0), 0);

    const oneWeekAgo = Date.now() - 7 * 24 * 60 * 60 * 1000;
    const thisWeekCount = history.filter((h) => new Date(h.timestamp).getTime() >= oneWeekAgo).length;

    const startOfToday = new Date();
    startOfToday.setHours(0, 0, 0, 0);
    const todayCalories = history
      .filter((h) => new Date(h.timestamp).getTime() >= startOfToday.getTime())
      .reduce((sum, h) => sum + (Number(h.calories) || 0), 0);

    return {
      totalMeals,
      avgCalories: Math.round(totalCalories / totalMeals),
      avgProtein: Math.round((totalProtein / totalMeals) * 10) / 10,
      thisWeekCount,
      todayCalories,
    };
  };

  return (
    <HistoryContext.Provider
      value={{
        history,
        isLoading,
        activeResult,
        setActiveResult,
        addHistoryItem,
        deleteHistoryItem,
        clearHistory,
        refreshHistory,
        getStats,
      }}
    >
      {children}
    </HistoryContext.Provider>
  );
};

export const useHistory = () => {
  const context = useContext(HistoryContext);
  if (!context) {
    throw new Error('useHistory must be used within a HistoryProvider');
  }
  return context;
};
