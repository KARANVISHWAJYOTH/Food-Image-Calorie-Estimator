import axios from 'axios';

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
 * Sends the image to the live model inference API.
 */
export const predictFood = async (fileOrBlob) => {
  const formData = new FormData();
  formData.append('image', fileOrBlob);

  const response = await apiClient.post('/api/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });

  return { success: true, data: response.data, source: 'live_fastapi' };
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
        return [];
      }
    }
    return [];
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
    let historyList = local ? JSON.parse(local) : [];
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
