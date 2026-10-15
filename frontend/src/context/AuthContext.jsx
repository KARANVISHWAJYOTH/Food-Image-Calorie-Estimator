import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import {
  loginUser as apiLogin,
  registerUser as apiRegister,
  fetchCurrentUser,
  updateUserProfileApi,
} from '../services/api';
import Toast from '../components/Toast';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(() => {
    const token = localStorage.getItem('nutrivision_token');
    const saved = localStorage.getItem('nutrivision_user');
    if (token && saved) {
      try {
        return JSON.parse(saved);
      } catch (e) {
        return null;
      }
    }
    return null;
  });

  const [isLoading, setIsLoading] = useState(false);
  const [toastMsg, setToastMsg] = useState(null);

  const logout = useCallback(() => {
    setUser(null);
    localStorage.removeItem('nutrivision_user');
    localStorage.removeItem('nutrivision_token');
  }, []);

  // Check backend session validity on initial app mount if token exists
  useEffect(() => {
    const token = localStorage.getItem('nutrivision_token');
    if (!token) return;

    let isMounted = true;
    fetchCurrentUser()
      .then((freshUser) => {
        if (isMounted && freshUser) {
          setUser(freshUser);
          localStorage.setItem('nutrivision_user', JSON.stringify(freshUser));
        }
      })
      .catch((err) => {
        if (err.response?.status === 401) {
          logout();
        }
      });

    return () => {
      isMounted = false;
    };
  }, [logout]);

  // Listen for session expiration events (from Axios 401 interceptor)
  useEffect(() => {
    // Check if session expired banner was queued in sessionStorage
    const storedMsg = sessionStorage.getItem('nutrivision_session_expired');
    if (storedMsg) {
      setToastMsg(storedMsg);
      sessionStorage.removeItem('nutrivision_session_expired');
    }

    const handleSessionExpired = (e) => {
      logout();
      setToastMsg(e.detail?.message || 'Your session has expired. Please log in again.');
    };

    window.addEventListener('auth:session-expired', handleSessionExpired);
    return () => {
      window.removeEventListener('auth:session-expired', handleSessionExpired);
    };
  }, [logout]);

  // Sync user state changes to localStorage
  useEffect(() => {
    if (user) {
      localStorage.setItem('nutrivision_user', JSON.stringify(user));
    } else {
      localStorage.removeItem('nutrivision_user');
    }
  }, [user]);

  const login = async (email, password) => {
    setIsLoading(true);
    try {
      const res = await apiLogin(email, password);
      setUser(res.user);
      return { success: true };
    } catch (err) {
      return { success: false, error: err.message || 'Login failed' };
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * 1-Click Demo Login: Authenticates against seeded demo user in SQLite DB,
   * acquiring a real 30-minute signed JWT access token.
   */
  const loginDemo = async () => {
    setIsLoading(true);
    try {
      const res = await apiLogin('demo@nutrivision.ai', 'demoPassword123');
      setUser(res.user);
      return { success: true };
    } catch (err) {
      return { success: false, error: err.message || 'Demo login failed' };
    } finally {
      setIsLoading(false);
    }
  };

  const register = async (username, email, password) => {
    setIsLoading(true);
    try {
      const res = await apiRegister(username, email, password);
      setUser(res.user);
      return { success: true };
    } catch (err) {
      return { success: false, error: err.message || 'Registration failed' };
    } finally {
      setIsLoading(false);
    }
  };

  const updateProfile = async (updatedFields) => {
    try {
      const updated = await updateUserProfileApi(updatedFields);
      if (updated) {
        setUser((prev) => ({
          ...prev,
          ...updated,
        }));
      }
      return { success: true };
    } catch (err) {
      return { success: false, error: err.message || 'Failed to update profile' };
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user,
        isLoading,
        login,
        loginDemo,
        register,
        updateProfile,
        logout,
        setToastMsg,
      }}
    >
      {children}
      <Toast
        message={toastMsg}
        type="error"
        onClose={() => setToastMsg(null)}
        duration={4500}
      />
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
