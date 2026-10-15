from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

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
    inferenceTimeMs: float = Field(default=45.0, description="Model inference runtime in milliseconds")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class HistoryItem(BaseModel):
    id: Optional[str] = None
    foodClass: str
    calories: float
    protein: float
    carbohydrates: Optional[float] = None
    carbs: Optional[float] = None
    fat: float
    fiber: Optional[float] = 0.0
    confidenceScore: Optional[float] = 1.0
    confidencePercentage: Optional[float] = None
    imageUrl: Optional[str] = None
    timestamp: Optional[str] = None
    portion: Optional[float] = 1.0
    servingSize: Optional[str] = None
    healthRating: Optional[str] = None
    dietaryTags: Optional[List[str]] = Field(default_factory=list)
    healthTips: Optional[str] = None
    isLowConfidence: Optional[bool] = False

class HealthResponse(BaseModel):
    status: str = "healthy"
    modelLoaded: bool = True
    modelArchitecture: str = "MultiTaskFoodCNN (ResNet50 Backbone + Dual Heads)"
    modelVariant: Optional[str] = "fp32"
    device: str = "cpu"
    version: str = "1.0.0"

class UserRegisterRequest(BaseModel):
    email: str
    password: str
    name: Optional[str] = "Nutrition Explorer"
    dailyCalorieGoal: Optional[int] = 2000
    dailyProteinGoal: Optional[int] = 120

class UserLoginRequest(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    role: str = "Member"
    dailyCalorieGoal: int = 2000
    dailyProteinGoal: int = 120
    createdAt: str

class AuthResponse(BaseModel):
    success: bool = True
    user: UserResponse
    token: str

class UserProfileUpdateRequest(BaseModel):
    name: Optional[str] = None
    dailyCalorieGoal: Optional[int] = None
    dailyProteinGoal: Optional[int] = None

