# pyrefly: ignore [missing-import]
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, status
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from typing import List, Optional

from .schemas import PredictionResponse, HealthResponse, HistoryItem
from .predictor import predictor_instance

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

HISTORY_STORE: List[dict] = []

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
