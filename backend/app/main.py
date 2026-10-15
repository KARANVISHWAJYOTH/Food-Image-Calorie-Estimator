# pyrefly: ignore [missing-import]
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, status, Depends
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import List, Optional

from .schemas import (
    PredictionResponse,
    HealthResponse,
    HistoryItem,
    UserRegisterRequest,
    UserLoginRequest,
    UserResponse,
    AuthResponse,
    UserProfileUpdateRequest,
)
from .predictor import predictor_instance
from .database import (
    init_db,
    create_user_record,
    get_user_by_email,
    get_user_by_id,
    update_user_profile,
    insert_history_item,
    get_all_history,
    delete_history_item as db_delete_history,
)
from .auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user_optional,
    get_current_user_required,
)

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

@app.on_event("startup")
def startup_event():
    init_db()
    logger.info("Database initialized successfully.")


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
    if not predictor_instance.is_torch_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The trained inference model is not loaded. Set MODEL_WEIGHTS_PATH "
                "to a compatible trained checkpoint and restart the backend."
            ),
        )

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

# --- Authentication Endpoints ---

@app.post("/api/auth/register", response_model=AuthResponse, tags=["Authentication"])
def register_user(request: UserRegisterRequest):
    """Register a new user account with PBKDF2 hashed credentials."""
    existing = get_user_by_email(request.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email address already exists.",
        )

    if len(request.password) < 4:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 4 characters.",
        )

    pw_hash, salt = hash_password(request.password)
    user_data = create_user_record(
        email=request.email,
        name=request.name or "Nutrition Explorer",
        password_hash=pw_hash,
        password_salt=salt,
        daily_calorie_goal=request.dailyCalorieGoal or 2000,
        daily_protein_goal=request.dailyProteinGoal or 120,
    )

    token = create_access_token(user_id=user_data["id"], email=user_data["email"], role=user_data["role"])
    user_res = UserResponse(
        id=user_data["id"],
        email=user_data["email"],
        name=user_data["name"],
        role=user_data["role"],
        dailyCalorieGoal=user_data["daily_calorie_goal"],
        dailyProteinGoal=user_data["daily_protein_goal"],
        createdAt=user_data["created_at"],
    )
    return AuthResponse(success=True, user=user_res, token=token)


@app.post("/api/auth/login", response_model=AuthResponse, tags=["Authentication"])
def login_user(request: UserLoginRequest):
    """Authenticate credentials and return a signed JWT token."""
    user = get_user_by_email(request.email)
    if not user or not verify_password(request.password, user["password_hash"], user["password_salt"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(user_id=user["id"], email=user["email"], role=user["role"])
    user_res = UserResponse(
        id=user["id"],
        email=user["email"],
        name=user["name"],
        role=user["role"],
        dailyCalorieGoal=user["daily_calorie_goal"],
        dailyProteinGoal=user["daily_protein_goal"],
        createdAt=user["created_at"],
    )
    return AuthResponse(success=True, user=user_res, token=token)


@app.get("/api/auth/me", response_model=UserResponse, tags=["Authentication"])
def get_current_profile(current_user: dict = Depends(get_current_user_required)):
    """Fetch current logged-in user profile from database."""
    user = get_user_by_id(current_user["sub"])
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    return UserResponse(
        id=user["id"],
        email=user["email"],
        name=user["name"],
        role=user["role"],
        dailyCalorieGoal=user["daily_calorie_goal"],
        dailyProteinGoal=user["daily_protein_goal"],
        createdAt=user["created_at"],
    )


@app.patch("/api/auth/profile", response_model=UserResponse, tags=["Authentication"])
def update_profile(
    updates: UserProfileUpdateRequest,
    current_user: dict = Depends(get_current_user_required),
):
    """Update profile settings and daily nutrition targets."""
    updated = update_user_profile(
        user_id=current_user["sub"],
        name=updates.name,
        daily_calorie_goal=updates.dailyCalorieGoal,
        daily_protein_goal=updates.dailyProteinGoal,
    )
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    return UserResponse(
        id=updated["id"],
        email=updated["email"],
        name=updated["name"],
        role=updated["role"],
        dailyCalorieGoal=updated["daily_calorie_goal"],
        dailyProteinGoal=updated["daily_protein_goal"],
        createdAt=updated["created_at"],
    )


# --- Persistent History Endpoints ---

@app.get("/api/history", tags=["History"])
def get_prediction_history(
    limit: int = Query(50, ge=1, le=100),
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """Returns past food estimation scans from SQLite database."""
    user_id = current_user["sub"] if current_user else None
    history_items = get_all_history(limit=limit, user_id=user_id)
    return {
        "total": len(history_items),
        "history": history_items,
    }


@app.post("/api/history", tags=["History"])
def save_to_history(
    item: HistoryItem,
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    """Save an analysis to SQLite history database."""
    user_id = current_user["sub"] if current_user else None
    saved = insert_history_item(item.model_dump(), user_id=user_id)
    return {"status": "saved", "item": saved}


@app.delete("/api/history/{item_id}", tags=["History"])
def delete_history_item(item_id: str):
    """Remove an item from SQLite history."""
    deleted = db_delete_history(item_id)
    return {"status": "deleted" if deleted else "not_found", "id": item_id}

