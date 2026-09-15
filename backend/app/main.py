# pyrefly: ignore [missing-import]
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, status
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from typing import List, Optional

from .schemas import PredictionResponse, HealthResponse, HistoryItem
from .predictor import predictor_instance
from .food_database import FOOD_DATABASE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("nutrivision.main")

app = FastAPI(
    title="NutriVision AI - Food Image Calorie Estimator API",
    description="Production-ready FastAPI backend for Multi-Task CNN food identification and nutritional regression.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for local development and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for seamless development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory history cache for backend demo
HISTORY_STORE: List[dict] = [
    {
        "id": "pred_hist_01",
        "foodClass": "Chicken Biryani",
        "calories": 720,
        "protein": 34,
        "carbs": 84,
        "fat": 26,
        "confidenceScore": 0.962,
        "imageUrl": "https://images.unsplash.com/photo-1563379091339-03b21ab4a4f8?auto=format&fit=crop&w=600&q=80",
        "timestamp": "2026-09-09T19:30:00Z",
        "portion": 1.0
    },
    {
        "id": "pred_hist_02",
        "foodClass": "Masala Dosa",
        "calories": 420,
        "protein": 11.5,
        "carbs": 64,
        "fat": 14,
        "confidenceScore": 0.935,
        "imageUrl": "https://images.unsplash.com/photo-1589301760014-d929f3979dbc?auto=format&fit=crop&w=600&q=80",
        "timestamp": "2026-09-09T08:45:00Z",
        "portion": 1.0
    },
    {
        "id": "pred_hist_03",
        "foodClass": "Paneer Butter Masala",
        "calories": 560,
        "protein": 21,
        "carbs": 26,
        "fat": 42,
        "confidenceScore": 0.918,
        "imageUrl": "https://images.unsplash.com/photo-1631452180519-c014fe946bc7?auto=format&fit=crop&w=600&q=80",
        "timestamp": "2026-09-08T20:15:00Z",
        "portion": 1.0
    }
]

@app.get("/", tags=["General"])
def root():
    return {
        "service": "NutriVision AI API",
        "version": "1.0.0",
        "status": "online",
        "docs": "/docs",
        "model": "Multi-Task PyTorch Food CNN (ResNet50 + Dual Heads)"
    }

@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    return HealthResponse(
        status="healthy",
        modelLoaded=predictor_instance.is_torch_ready,
        modelArchitecture="MultiTaskFoodCNN (ResNet50 Backbone + Calorie/Macro Regression)",
        device=str(predictor_instance.device),
        version="1.0.0"
    )

@app.get("/api/foods", tags=["Database"])
def list_supported_foods():
    """Returns database of recognizable food items with baseline nutrition profile."""
    return {
        "total": len(FOOD_DATABASE),
        "items": [
            {
                "key": key,
                "name": val["name"],
                "category": val.get("category", "Main Course"),
                "calories": val["calories"],
                "protein": val["protein"],
                "carbs": val["carbohydrates"],
                "fat": val["fat"],
                "sampleImage": val.get("sampleImage", "")
            }
            for key, val in FOOD_DATABASE.items()
        ]
    }

@app.post("/api/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_food(image: UploadFile = File(...)):
    """
    Accepts an uploaded food photo via multipart/form-data.
    Returns classified food item, confidence score, and multi-task estimated nutritional breakdown.
    """
    # Validate content type
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/jpg"]
    if image.content_type and image.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image type: {image.content_type}. Please upload JPG, PNG, or WEBP."
        )

    try:
        contents = await image.read()
        if len(contents) > 15 * 1024 * 1024: # 15 MB limit
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Image file size exceeds 15 MB limit."
            )
            
        result = predictor_instance.predict_image(contents, filename=image.filename)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Inference error processing upload: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run model inference on uploaded image. Please try again."
        )

@app.get("/api/history", tags=["History"])
def get_prediction_history(limit: int = Query(20, ge=1, le=100)):
    """Returns past food estimation scans."""
    return {
        "total": len(HISTORY_STORE),
        "history": HISTORY_STORE[:limit]
    }

@app.post("/api/history", tags=["History"])
def save_to_history(item: HistoryItem):
    """Save an analysis to the backend history store."""
    HISTORY_STORE.insert(0, item.dict())
    return {"status": "saved", "item": item}

@app.delete("/api/history/{item_id}", tags=["History"])
def delete_history_item(item_id: str):
    """Remove an item from history."""
    global HISTORY_STORE
    HISTORY_STORE = [item for item in HISTORY_STORE if item["id"] != item_id]
    return {"status": "deleted", "id": item_id}
