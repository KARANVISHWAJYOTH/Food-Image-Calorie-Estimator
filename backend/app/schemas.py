from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

class NutrientBreakdown(BaseModel):
    calories: float = Field(..., description="Estimated energy in kilocalories (kcal)")
    protein: float = Field(..., description="Protein in grams (g)")
    carbohydrates: float = Field(..., description="Carbohydrates in grams (g)")
    fat: float = Field(..., description="Total Fat in grams (g)")
    fiber: Optional[float] = Field(default=0.0, description="Dietary Fiber in grams (g)")
    sugar: Optional[float] = Field(default=0.0, description="Sugars in grams (g)")
    sodium: Optional[float] = Field(default=0.0, description="Sodium in milligrams (mg)")
    potassium: Optional[float] = Field(default=0.0, description="Potassium in milligrams (mg)")

class PredictionResponse(BaseModel):
    id: str = Field(..., description="Unique prediction identifier")
    foodClass: str = Field(..., description="Predicted food category/dish name")
    category: Optional[str] = Field(default="Main Course", description="Meal category")
    confidenceScore: float = Field(..., description="Classification confidence probability (0.0 - 1.0)")
    confidencePercentage: float = Field(..., description="Confidence represented as percentage (0 - 100)")
    estimatedCalories: float = Field(..., description="Estimated calories (kcal)")
    protein: float = Field(..., description="Protein in grams (g)")
    carbohydrates: float = Field(..., description="Carbohydrates in grams (g)")
    fat: float = Field(..., description="Fat in grams (g)")
    fiber: float = Field(default=0.0, description="Fiber in grams (g)")
    sugar: float = Field(default=0.0, description="Sugar in grams (g)")
    sodium: float = Field(default=0.0, description="Sodium in mg")
    potassium: float = Field(default=0.0, description="Potassium in mg")
    servingSize: str = Field(default="1 standard serving (approx. 350g)", description="Estimated portion weight/serving")
    isLowConfidence: bool = Field(default=False, description="Flag indicating if prediction confidence is low (< 0.65)")
    healthRating: Optional[str] = Field(default="Balanced", description="Nutritional rating (e.g., High Protein, Balanced, Low Carb)")
    dietaryTags: List[str] = Field(default_factory=list, description="Tags such as Vegan, Gluten-Free, High-Protein")
    ingredients: List[str] = Field(default_factory=list, description="Common ingredients detected in this dish")
    healthTips: str = Field(..., description="AI generated nutritional insight and recommendation")
    inferenceTimeMs: float = Field(default=0.0, description="Measured model inference runtime in milliseconds")
    imageUrl: Optional[str] = Field(default=None, description="Accessible web URL of uploaded food image")
    imagePath: Optional[str] = Field(default=None, description="Local or volume file path of uploaded food image")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class HistoryItem(BaseModel):
    id: Optional[str] = None
    foodClass: str
    calories: Optional[float] = Field(default=None, ge=0)
    estimatedCalories: Optional[float] = Field(default=None, ge=0)
    protein: Optional[float] = Field(default=0.0, ge=0)
    carbohydrates: Optional[float] = Field(default=None, ge=0)
    carbs: Optional[float] = Field(default=None, ge=0)
    fat: Optional[float] = Field(default=0.0, ge=0)
    fiber: Optional[float] = Field(default=0.0, ge=0)
    confidenceScore: Optional[float] = Field(default=1.0, ge=0, le=1)
    confidencePercentage: Optional[float] = None
    imageUrl: Optional[str] = None
    imagePath: Optional[str] = None
    image_path: Optional[str] = None
    timestamp: Optional[str] = None
    portion: Optional[float] = 1.0
    servingSize: Optional[str] = None
    healthRating: Optional[str] = None
    dietaryTags: Optional[List[str]] = Field(default_factory=list)
    healthTips: Optional[str] = None
    isLowConfidence: Optional[bool] = False

class HealthResponse(BaseModel):
    status: str = "healthy"
    modelLoaded: bool = False
    modelReady: bool = False
    modelArchitecture: str = "MultiTaskFoodCNN (ResNet50 Backbone + Dual Heads)"
    modelVariant: Optional[str] = "fp32"
    device: str = "cpu"
    version: str = "1.0.0"
    checkpointPath: Optional[str] = None
    trainedMetrics: Optional[Dict[str, float]] = None
    modelValidation: Optional[Dict[str, Any]] = None

class UserRegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    username: Optional[str] = Field(default=None, max_length=80)
    name: Optional[str] = Field(default="Nutrition Explorer", max_length=120)
    dailyCalorieGoal: Optional[int] = Field(default=2000, ge=500, le=10000)
    dailyProteinGoal: Optional[int] = Field(default=120, ge=0, le=1000)

class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)

class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    username: Optional[str] = None
    role: str = "Member"
    dailyCalorieGoal: int = 2000
    dailyProteinGoal: int = 120
    createdAt: str

class AuthResponse(BaseModel):
    success: bool = True
    user: UserResponse
    token: str

class UserProfileUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, max_length=120)
    dailyCalorieGoal: Optional[int] = Field(default=None, ge=500, le=10000)
    dailyProteinGoal: Optional[int] = Field(default=None, ge=0, le=1000)

class PasswordChangeRequest(BaseModel):
    currentPassword: str = Field(min_length=1, max_length=128)
    newPassword: str = Field(min_length=8, max_length=128)

