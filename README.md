# 🥗 NutriVision AI – Food Image Calorie & Macro Estimator

> **“Turn your food photo into nutrition insights.”**

[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4-38B2AC?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

**NutriVision AI** is a modern, responsive, full-stack AI-powered nutrition analysis platform. Users upload a food image, and the system identifies the dish and estimates calories, protein, carbohydrates, and fat using a Multi-Task Deep Convolutional Neural Network (CNN).

---

## 🌟 Key Features

- **AI Food Recognition**: Real-time visual food categorization powered by deep convolutional feature extraction.
- **Calorie & Macro Regression**: Simultaneous estimation of energy (`kcal`), protein (`g`), carbohydrates (`g`), fat (`g`), fiber, and sodium.
- **AI Scanning Effect**: Animated laser scanning beams and neural reticle overlays on food previews.
- **Interactive Macro Breakdown**: Energy distribution donut charts and daily value (`% DV`) progress gauges.
- **Dynamic Portion Scaler**: Real-time recalculation of nutrition metrics across `0.5x`, `1.0x`, `1.5x`, and `2.0x` portions.
- **Low Confidence Safeguards**: Automatic detection of visually ambiguous or poorly lit photos (`< 65%` confidence) with actionable re-capture guidance.
- **1-Click Demo Mode & Presets**: Instant testing with curated dishes (Biryani, Masala Dosa, Grilled Chicken Bowl, Salmon Quinoa, Avocado Toast, etc.).
- **Scan History**: Search, sort (by calories, protein, date), filter, and delete past prediction records with LocalStorage and backend sync.
- **User Nutrition Goals**: Customizable daily caloric and protein targets with daily streak tracking.
- **Docker-Ready Architecture**: Multi-container Docker Compose configuration for one-command deployment.

---

## 🏗️ Project Architecture

```text
Food Image Calorie Estimator/
├── frontend/                     # React + Vite + Tailwind CSS Single Page Application
│   ├── src/
│   │   ├── components/          # Reusable UI components
│   │   │   ├── Navbar.jsx       # Public landing navigation
│   │   │   ├── Sidebar.jsx      # Dashboard sidebar & mobile drawer
│   │   │   ├── DashboardHeader.jsx # Personalized header & greeting
│   │   │   ├── QuickStats.jsx   # 4 Top metrics cards
│   │   │   ├── FoodUploader.jsx # Drag & drop zone, camera, preview
│   │   │   ├── ScanningOverlay.jsx # AI laser beam visual effect
│   │   │   ├── SampleFoodPicker.jsx # 1-click sample food gallery
│   │   │   ├── NutritionCard.jsx# Macro metric card with % DV bar
│   │   │   ├── NutritionChart.jsx # Donut & caloric share chart
│   │   │   ├── ConfidenceBadge.jsx# Dynamic confidence pill
│   │   │   ├── LowConfidenceAlert.jsx # Guidance for uncertain images
│   │   │   ├── LoadingState.jsx # Multi-stage AI inference loader
│   │   │   ├── DisclaimerBanner.jsx # Medical & estimation disclaimer
│   │   │   └── Toast.jsx        # Notification alert system
│   │   ├── context/
│   │   │   ├── AuthContext.jsx  # Authentication state & demo user
│   │   │   └── HistoryContext.jsx # Scan history & stats calculator
│   │   ├── data/
│   │   │   └── sampleFoods.js   # Rich mock database & USDA metrics
│   │   ├── pages/
│   │   │   ├── Home.jsx         # Landing page (Hero, Features, How It Works, Tech)
│   │   │   ├── Login.jsx        # Login & 1-click Demo Account
│   │   │   ├── Signup.jsx       # Registration with validation
│   │   │   ├── Dashboard.jsx    # User dashboard & recent scans
│   │   │   ├── Analyze.jsx      # Dedicated scanning workstation
│   │   │   ├── Result.jsx       # Detailed prediction breakdown & export
│   │   │   ├── History.jsx      # Search, filter, and history table
│   │   │   └── Profile.jsx      # Nutrition goals & account settings
│   │   ├── services/
│   │   │   └── api.js           # Axios service layer with backend/mock auto-switching
│   │   ├── App.jsx              # Routes & Protected Route wrapper
│   │   └── main.jsx             # Entry point
│   ├── Dockerfile               # Production Nginx container
│   ├── nginx.conf               # SPA routing Nginx config
│   └── package.json
│
├── backend/                      # FastAPI + PyTorch Backend API
│   ├── app/
│   │   ├── main.py              # FastAPI app with CORS & endpoints
│   │   ├── model.py             # PyTorch MultiTaskFoodCNN architecture
│   │   ├── predictor.py         # Torchvision transforms & inference engine
│   │   ├── schemas.py           # Pydantic request/response schemas
│   │   └── food_database.py     # USDA calibrated food database
│   ├── Dockerfile               # Backend Docker container
│   └── requirements.txt
│
├── docker-compose.yml           # Unified multi-container deployment
└── README.md
```

---

## 🧠 PyTorch Multi-Task CNN Architecture

The model architecture in `backend/app/model.py` utilizes a shared deep convolutional backbone (e.g. `ResNet50`) with three specialized parallel output heads:

```text
               ┌───────────────────────────────┐
               │    Input Food Photo (RGB)     │
               └───────────────┬───────────────┘
                               │
               ┌───────────────▼───────────────┐
               │   ResNet50 Backbone Features  │
               └───────────────┬───────────────┘
                               │
               ┌───────────────▼───────────────┐
               │    Shared Latent Bottleneck   │
               └───────┬───────────────┬───────┘
                       │               │
       ┌───────────────▼──────┐ ┌──────▼──────────────────┐
       │ Classification Head  │ │ Calorie Regression Head │
       │ (Multi-Class Logits) │ │ (Continuous kcal value) │
       └──────────────────────┘ └─────────────────────────┘
                                       │
                                ┌──────▼──────────────────┐
                                │ Macro Regression Head   │
                                │ [Protein, Carbs, Fat]   │
                                └─────────────────────────┘
```

---

## 🚀 Quickstart Guide

### 1. Running the Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser. The frontend starts in **Demo Mode** out of the box with instant simulated predictions and 1-click sample food presets.

### 2. Running the FastAPI Backend (Optional)

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

FastAPI interactive Swagger documentation will be available at: `http://localhost:8000/docs`.

### 3. Running with Docker Compose

To spin up the entire full-stack application with a single command:

```bash
docker-compose up --build
```

- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`

---

## 📡 API Specification

### Predict Food Image
- **Method**: `POST`
- **Endpoint**: `/api/predict`
- **Body**: `multipart/form-data` with `image` file
- **Response**:
```json
{
  "id": "pred_8fa21b9c",
  "foodClass": "Grilled Chicken Rice Bowl",
  "category": "Main Course",
  "confidenceScore": 0.947,
  "confidencePercentage": 94.7,
  "estimatedCalories": 650.0,
  "protein": 38.0,
  "carbohydrates": 72.0,
  "fat": 19.0,
  "fiber": 5.2,
  "sugar": 3.8,
  "sodium": 580.0,
  "potassium": 640.0,
  "servingSize": "1 bowl (approx. 420g)",
  "isLowConfidence": false,
  "healthRating": "High Protein & Balanced",
  "dietaryTags": ["High Protein", "Low Sugar", "Lean Meat"],
  "healthTips": "Excellent lean protein source combined with complex carbohydrates.",
  "inferenceTimeMs": 42.5
}
```

---

## ⚠️ Important ML Disclaimer

> **“AI nutrition estimates are approximate and should not be considered medical or dietary advice.”**
> Nutrition values are AI-generated estimates and may vary depending on ingredients, preparation method, and portion size.

---

## 📄 License

MIT License © 2026 NutriVision AI.
