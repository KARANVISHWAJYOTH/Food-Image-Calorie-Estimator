/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#ecfdf5',
          100: '#d1fae5',
          200: '#a7f3d0',
          300: '#6ee7b7',
          400: '#34d399',
          500: '#10b981',
          600: '#059669',
          700: '#047857',
          800: '#065f46',
          900: '#064e3b',
          950: '#022c22',
        },
        carb: {
          50: '#fffbeb',
          100: '#fef3c7',
          500: '#f59e0b',
          600: '#d97706',
          700: '#b45309',
        },
        protein: {
          50: '#ecfeff',
          100: '#cffafe',
          500: '#06b6d4',
          600: '#0891b2',
          700: '#0e7490',
        },
        fat: {
          50: '#fff1f2',
          100: '#ffe4e6',
          500: '#f43f5e',
          600: '#e11d48',
          700: '#be123c',
        }
      },
      fontFamily: {
        sans: ['Plus Jakarta Sans', 'Inter', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        'soft': '0 4px 20px -2px rgba(0, 0, 0, 0.05), 0 2px 6px -1px rgba(0, 0, 0, 0.02)',
        'card': '0 10px 30px -5px rgba(0, 0, 0, 0.06), 0 4px 10px -3px rgba(0, 0, 0, 0.02)',
        'elevated': '0 20px 40px -10px rgba(0, 0, 0, 0.08), 0 8px 16px -4px rgba(0, 0, 0, 0.03)',
        'glow-emerald': '0 0 25px -3px rgba(16, 185, 129, 0.25)',
        'glow-brand': '0 0 30px -5px rgba(5, 150, 105, 0.35)',
      },
      keyframes: {
        'scan': {
          '0%, 100%': { transform: 'translateY(0%)' },
          '50%': { transform: 'translateY(100%)' },
        },
        'scan-line': {
          '0%': { top: '0%', opacity: '0.8' },
          '50%': { top: '96%', opacity: '1' },
          '100%': { top: '0%', opacity: '0.8' },
        },
        'float-slow': {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-8px)' },
        },
        'float-reverse': {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(8px)' },
        },
        'pulse-subtle': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.7' },
        },
        'shimmer': {
          '100%': { transform: 'translateX(100%)' },
        }
      },
      animation: {
        'scan': 'scan 3s ease-in-out infinite',
        'scan-line': 'scan-line 2.5s ease-in-out infinite',
        'float-slow': 'float-slow 4s ease-in-out infinite',
        'float-reverse': 'float-reverse 4.5s ease-in-out infinite',
        'pulse-subtle': 'pulse-subtle 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'shimmer': 'shimmer 2s infinite',
      }
    },
  },
  plugins: [],
}
