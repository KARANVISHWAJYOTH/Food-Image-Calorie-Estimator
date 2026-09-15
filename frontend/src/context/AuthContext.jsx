import React, { createContext, useContext, useState, useEffect } from 'react';
import { loginUser as apiLogin, registerUser as apiRegister } from '../services/api';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(() => {
    const saved = localStorage.getItem('nutrivision_user');
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch (e) {
        return null;
      }
    }
    // Default logged-in user for seamless first impression
    return {
      id: 'usr_karan_01',
      name: 'Karan',
      email: 'karan@nutrivision.ai',
      role: 'Pro Member',
      avatarUrl: 'https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?auto=format&fit=crop&w=200&q=80',
      dailyCalorieGoal: 2200,
      dailyProteinGoal: 130,
      dailyCarbGoal: 240,
      dailyFatGoal: 65,
      dietaryPreference: 'Balanced & High-Protein',
      createdAt: '2026-01-15T10:00:00Z',
    };
  });

  const [isLoading, setIsLoading] = useState(false);

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

  const loginDemo = async () => {
    setIsLoading(true);
    await new Promise((r) => setTimeout(r, 400));
    const demoUser = {
      id: 'usr_demo_karan',
      name: 'Karan',
      email: 'karan.demo@nutrivision.ai',
      role: 'Pro Member',
      avatarUrl: 'https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?auto=format&fit=crop&w=200&q=80',
      dailyCalorieGoal: 2200,
      dailyProteinGoal: 130,
      dailyCarbGoal: 240,
      dailyFatGoal: 65,
      dietaryPreference: 'High-Protein Athletic',
      createdAt: '2026-02-01T08:00:00Z',
    };
    setUser(demoUser);
    setIsLoading(false);
    return { success: true };
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

  const updateProfile = (updatedFields) => {
    setUser((prev) => ({
      ...prev,
      ...updatedFields,
    }));
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('nutrivision_user');
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
      }}
    >
      {children}
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
