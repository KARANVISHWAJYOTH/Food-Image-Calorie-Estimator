import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
  headers: {
    'Accept': 'application/json',
  },
});

// Request Interceptor: Attach JWT Bearer token to all outgoing requests
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('nutrivision_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response Interceptor: On 401 Unauthorized, clear session, broadcast event, and redirect to /login
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      const requestUrl = error.config?.url || '';
      const isAuthEndpoint =
        requestUrl.includes('/api/auth/login') ||
        requestUrl.includes('/api/auth/signup') ||
        requestUrl.includes('/api/auth/register');

      // Only trigger session expiration for protected resources, not failed login attempts
      if (!isAuthEndpoint) {
        localStorage.removeItem('nutrivision_token');
        localStorage.removeItem('nutrivision_user');

        const sessionExpiredMsg = 'Your session has expired. Please log in again.';
        sessionStorage.setItem('nutrivision_session_expired', sessionExpiredMsg);

        window.dispatchEvent(
          new CustomEvent('auth:session-expired', {
            detail: { message: sessionExpiredMsg },
          })
        );

        if (!window.location.pathname.includes('/login')) {
          window.location.href = '/login?expired=true';
        }
      }
    }
    return Promise.reject(error);
  }
);

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
 * Sends the image to the live model inference API with Bearer token authentication.
 */
export const predictFood = async (fileOrBlob) => {
  const formData = new FormData();
  formData.append('image', fileOrBlob);

  let response;
  try {
    response = await apiClient.post('/api/predict', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  } catch (error) {
    if (!error.response) {
      throw new Error(
        `Cannot connect to the backend at ${API_BASE_URL}. Start the FastAPI service on port 8000 and retry.`
      );
    }
    throw error;
  }

  return { success: true, data: response.data, source: 'live_fastapi' };
};

/**
 * Fetch past prediction history for current authenticated user
 */
export const getPredictionHistory = async () => {
  try {
    const response = await apiClient.get('/api/history');
    return response.data?.history || [];
  } catch (error) {
    if (error.response?.status === 401) {
      throw error;
    }
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
    if (error.response?.status === 401) {
      throw error;
    }
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
    const response = await apiClient.delete(`/api/history/${id}`);
    return response.data;
  } catch (error) {
    if (error.response?.status === 401) {
      throw error;
    }
    const local = localStorage.getItem('nutrivision_history');
    if (local) {
      const historyList = JSON.parse(local).filter((item) => item.id !== id);
      localStorage.setItem('nutrivision_history', JSON.stringify(historyList));
    }
    return { status: 'deleted_locally', id };
  }
};

/**
 * Real Authentication API endpoints
 */
export const loginUser = async (email, password) => {
  try {
    const response = await apiClient.post('/api/auth/login', { email, password });
    if (response.data?.token) {
      localStorage.setItem('nutrivision_token', response.data.token);
    }
    return {
      success: true,
      user: {
        ...response.data.user,
        dailyCalorieGoal: response.data.user.dailyCalorieGoal || 2000,
        dailyProteinGoal: response.data.user.dailyProteinGoal || 120,
      },
      token: response.data.token,
    };
  } catch (error) {
    const msg = error.response?.data?.detail || error.message || 'Login failed';
    throw new Error(msg);
  }
};

export const registerUser = async (username, email, password) => {
  try {
    const response = await apiClient.post('/api/auth/signup', {
      email,
      password,
      name: username,
    });
    if (response.data?.token) {
      localStorage.setItem('nutrivision_token', response.data.token);
    }
    return {
      success: true,
      user: response.data.user,
      token: response.data.token,
    };
  } catch (error) {
    const msg = error.response?.data?.detail || error.message || 'Registration failed';
    throw new Error(msg);
  }
};

export const fetchCurrentUser = async () => {
  const response = await apiClient.get('/api/auth/me');
  return response.data;
};

export const updateUserProfileApi = async (updates) => {
  try {
    const response = await apiClient.patch('/api/auth/profile', updates);
    return response.data;
  } catch (error) {
    const msg = error.response?.data?.detail || error.message || 'Profile update failed';
    throw new Error(msg);
  }
};

export default apiClient;
