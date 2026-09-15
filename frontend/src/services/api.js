import axios from 'axios';
import { SAMPLE_FOODS, MOCK_HISTORY_INITIAL } from '../data/sampleFoods';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
  headers: {
    'Accept': 'application/json',
  },
});

// Check if FastAPI backend server is alive
export const checkApiHealth = async () => {
  try {
    const response = await apiClient.get('/api/health');
    return { isLive: true, data: response.data };
  } catch (error) {
    return { isLive: false, data: null };
  }
};

/**
 * Predict food from uploaded image file.
 * Automatically tries live FastAPI backend first; if unavailable, falls back to realistic AI simulation.
 */
export const predictFood = async (fileOrBlob, presetHint = null) => {
  // If backend is running, attempt real multi-part upload
  try {
    const formData = new FormData();
    formData.append('image', fileOrBlob);

    const response = await apiClient.post('/api/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return {
      success: true,
      data: response.data,
      source: 'live_fastapi',
    };
  } catch (backendError) {
    // Graceful fallback to rich realistic ML simulation
    console.info('Backend API offline or unreachable. Using NutriVision AI Demo Simulation engine.');
    
    // Simulate realistic multi-task neural network inference latency
    await new Promise((resolve) => setTimeout(resolve, 1600));

    // Match sample food by preset hint or filename
    let matched = SAMPLE_FOODS[0]; // Default: Grilled Chicken Rice Bowl

    if (presetHint) {
      const found = SAMPLE_FOODS.find((f) => f.key === presetHint || f.id === presetHint);
      if (found) matched = found;
    } else if (fileOrBlob && fileOrBlob.name) {
      const name = fileOrBlob.name.toLowerCase();
      if (name.includes('biryani')) matched = SAMPLE_FOODS.find((f) => f.key === 'chicken_biryani');
      else if (name.includes('dosa')) matched = SAMPLE_FOODS.find((f) => f.key === 'masala_dosa');
      else if (name.includes('paneer')) matched = SAMPLE_FOODS.find((f) => f.key === 'paneer_butter_masala');
      else if (name.includes('salmon') || name.includes('quinoa')) matched = SAMPLE_FOODS.find((f) => f.key === 'salmon_quinoa_bowl');
      else if (name.includes('avocado') || name.includes('toast')) matched = SAMPLE_FOODS.find((f) => f.key === 'avocado_toast');
      else if (name.includes('caesar') || name.includes('salad')) matched = SAMPLE_FOODS.find((f) => f.key === 'caesar_salad');
      else if (name.includes('pizza')) matched = SAMPLE_FOODS.find((f) => f.key === 'pepperoni_pizza');
      else if (name.includes('oat') || name.includes('berry')) matched = SAMPLE_FOODS.find((f) => f.key === 'oatmeal_berry_bowl');
      else if (name.includes('low') || name.includes('uncertain') || name.includes('blur')) matched = SAMPLE_FOODS.find((f) => f.key === 'mixed_street_snack');
    }

    if (!matched) matched = SAMPLE_FOODS[0];

    const result = {
      id: `pred_${Math.random().toString(36).substr(2, 9)}`,
      foodClass: matched.name,
      category: matched.category,
      confidenceScore: matched.confidenceScore,
      confidencePercentage: matched.confidencePercentage,
      estimatedCalories: matched.calories,
      protein: matched.protein,
      carbohydrates: matched.carbohydrates,
      fat: matched.fat,
      fiber: matched.fiber,
      sugar: matched.sugar,
      sodium: matched.sodium,
      potassium: matched.potassium,
      servingSize: matched.servingSize,
      isLowConfidence: !!matched.isLowConfidence,
      healthRating: matched.healthRating,
      dietaryTags: matched.dietaryTags,
      ingredients: matched.ingredients,
      healthTips: matched.healthTips,
      imageUrl: matched.imageUrl,
      inferenceTimeMs: matched.inferenceTimeMs || 42.0,
      timestamp: new Date().toISOString(),
    };

    return {
      success: true,
      data: result,
      source: 'demo_simulation',
    };
  }
};

/**
 * Fetch past prediction history
 */
export const getPredictionHistory = async () => {
  try {
    const response = await apiClient.get('/api/history');
    return response.data.history;
  } catch (error) {
    const local = localStorage.getItem('nutrivision_history');
    if (local) {
      try {
        return JSON.parse(local);
      } catch (e) {
        return MOCK_HISTORY_INITIAL;
      }
    }
    return MOCK_HISTORY_INITIAL;
  }
};

/**
 * Save new prediction to history
 */
export const savePrediction = async (prediction) => {
  try {
    const response = await apiClient.post('/api/history', prediction);
    return response.data;
  } catch (error) {
    // Save to localStorage
    const local = localStorage.getItem('nutrivision_history');
    let historyList = local ? JSON.parse(local) : MOCK_HISTORY_INITIAL;
    historyList = [prediction, ...historyList];
    localStorage.setItem('nutrivision_history', JSON.stringify(historyList));
    return { status: 'saved_locally', item: prediction };
  }
};

/**
 * Delete a prediction item from history
 */
export const deletePrediction = async (id) => {
  try {
    await apiClient.delete(`/api/history/${id}`);
  } catch (error) {
    const local = localStorage.getItem('nutrivision_history');
    if (local) {
      const historyList = JSON.parse(local).filter((item) => item.id !== id);
      localStorage.setItem('nutrivision_history', JSON.stringify(historyList));
    }
  }
};

/**
 * Authentication mock & API helpers
 */
export const loginUser = async (email, password) => {
  // Simulate auth latency
  await new Promise((resolve) => setTimeout(resolve, 600));
  
  if (email && password.length >= 4) {
    const user = {
      id: 'usr_karan_01',
      name: email.split('@')[0] || 'Karan',
      email: email,
      role: 'Premium Member',
      avatarUrl: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=200&q=80',
      dailyCalorieGoal: 2200,
      dailyProteinGoal: 130,
      dailyCarbGoal: 240,
      dailyFatGoal: 65,
      createdAt: '2026-01-15T10:00:00Z',
    };
    return { success: true, user, token: 'jwt_mock_token_nutrivision' };
  }
  throw new Error('Invalid email or password. Password must be at least 4 characters.');
};

export const registerUser = async (username, email, password) => {
  await new Promise((resolve) => setTimeout(resolve, 800));
  const user = {
    id: `usr_${Math.random().toString(36).substr(2, 7)}`,
    name: username || 'Nutrition Explorer',
    email: email,
    role: 'Member',
    avatarUrl: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=200&q=80',
    dailyCalorieGoal: 2000,
    dailyProteinGoal: 120,
    dailyCarbGoal: 220,
    dailyFatGoal: 60,
    createdAt: new Date().toISOString(),
  };
  return { success: true, user, token: 'jwt_mock_token_nutrivision' };
};

export default apiClient;
