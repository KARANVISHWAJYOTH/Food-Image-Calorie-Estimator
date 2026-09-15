import React, { createContext, useContext, useState, useEffect } from 'react';
import { MOCK_HISTORY_INITIAL } from '../data/sampleFoods';
import { getPredictionHistory, savePrediction, deletePrediction as apiDeletePrediction } from '../services/api';

const HistoryContext = createContext(null);

export const HistoryProvider = ({ children }) => {
  const [history, setHistory] = useState(() => {
    const saved = localStorage.getItem('nutrivision_history');
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch (e) {
        return MOCK_HISTORY_INITIAL;
      }
    }
    return MOCK_HISTORY_INITIAL;
  });

  const [activeResult, setActiveResult] = useState(null);

  useEffect(() => {
    localStorage.setItem('nutrivision_history', JSON.stringify(history));
  }, [history]);

  const addHistoryItem = async (item) => {
    const newRecord = {
      id: item.id || `hist_${Date.now()}`,
      foodClass: item.foodClass,
      calories: item.estimatedCalories || item.calories,
      protein: item.protein,
      carbohydrates: item.carbohydrates || item.carbs,
      fat: item.fat,
      fiber: item.fiber || 0,
      sugar: item.sugar || 0,
      sodium: item.sodium || 0,
      potassium: item.potassium || 0,
      confidenceScore: item.confidenceScore,
      confidencePercentage: item.confidencePercentage || (item.confidenceScore * 100),
      imageUrl: item.imageUrl || 'https://images.unsplash.com/photo-1546069901-ba9599a7e63c?auto=format&fit=crop&w=600&q=80',
      timestamp: item.timestamp || new Date().toISOString(),
      portion: item.portion || 1.0,
      servingSize: item.servingSize,
      healthRating: item.healthRating,
      dietaryTags: item.dietaryTags || [],
      ingredients: item.ingredients || [],
      healthTips: item.healthTips,
      isLowConfidence: item.isLowConfidence || false,
    };

    setHistory((prev) => [newRecord, ...prev]);
    await savePrediction(newRecord);
  };

  const deleteHistoryItem = async (id) => {
    setHistory((prev) => prev.filter((item) => item.id !== id));
    await apiDeletePrediction(id);
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem('nutrivision_history');
  };

  // Quick statistics calculator
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
      thisWeekCount: thisWeekCount || totalMeals,
      todayCalories: todayCalories || Math.round(totalCalories / totalMeals),
    };
  };

  return (
    <HistoryContext.Provider
      value={{
        history,
        activeResult,
        setActiveResult,
        addHistoryItem,
        deleteHistoryItem,
        clearHistory,
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
