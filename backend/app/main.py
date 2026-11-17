# pyrefly: ignore [missing-import]
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, status, Depends
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import io
import logging
import os
import uuid

from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from .schemas import (
    PredictionResponse,
    HealthResponse,
    HistoryItem,
    UserRegisterRequest,
    UserLoginRequest,
    UserResponse,
    AuthResponse,
    UserProfileUpdateRequest,
    PasswordChangeRequest,
)
from .predictor import predictor_instance
from .database import (
    init_db,
    UPLOADS_DIR,
    create_user_record,
    get_user_by_email,
    update_user_profile,
    update_user_password,
    insert_prediction_record,
    get_user_predictions,
    delete_user_prediction,
    clear_user_predictions,
)
from .auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("nutrivision.main")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    logger.info("Database initialized.")
    yield


app = FastAPI(
    title="NutriVision AI - Food Image Calorie Estimator API",
    description="Production-ready FastAPI backend for Multi-Task CNN food identification and nutritional regression.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Browser origins allowed to call this API. Set CORS_ORIGINS as a comma-separated list.
def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins or ["http://localhost:5173"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files to serve uploaded food images from volume
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

SUPPORTED_IMAGE_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
MAX_IMAGE_DIMENSION = 8000


def inspect_uploaded_image(contents: bytes) -> str:
    """Fully decode an upload with Pillow and return its canonical file extension."""
    if not contents:
        raise ValueError("Uploaded image is empty.")
    try:
        with Image.open(io.BytesIO(contents)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(contents)) as image:
            image_format = (image.format or "").upper()
            width, height = image.size
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise ValueError("File is not a readable image. Please upload a valid JPG, PNG, or WEBP photo.")

    if image_format not in SUPPORTED_IMAGE_FORMATS:
        raise ValueError(f"Unsupported image format: {image_format or 'unknown'}. Please upload JPG, PNG, or WEBP.")
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise ValueError(f"Image dimensions exceed the {MAX_IMAGE_DIMENSION}px limit.")
    return SUPPORTED_IMAGE_FORMATS[image_format]


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
    validation = predictor_instance.model_validation
    model_ready = bool(predictor_instance.is_torch_ready and predictor_instance.is_model_valid)
    return HealthResponse(
        status="healthy",
        modelLoaded=model_ready,
        modelReady=model_ready,
        modelArchitecture="MultiTaskFoodCNN (ResNet50 Backbone + Calorie/Macro Regression)",
        modelVariant=getattr(predictor_instance, "model_variant", "fp32"),
        device=str(predictor_instance.device),
        version="1.0.0",
        checkpointPath=predictor_instance.checkpoint_path,
        trainedMetrics=validation.get("val_metrics") or None,
        modelValidation=validation,
    )


@app.post("/api/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_food(
    image: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Accepts an uploaded food photo via multipart/form-data.
    Requires authentication via Bearer JWT.
    Saves the image to the mounted uploads volume with a UUID filename,
    runs model inference, and returns prediction with image path & URL.
    """
    if not predictor_instance.is_torch_ready or not predictor_instance.is_model_valid:
        validation_errors = predictor_instance.model_validation.get("errors") or [
            "checkpoint_unavailable"
        ]
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No deployable trained model is loaded. Checkpoint validation failed: "
                f"{', '.join(str(error) for error in validation_errors)}. "
                "Train the model and set MODEL_WEIGHTS_PATH (or replace backend/app/best_model.pth)."
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
        if len(contents) > 15 * 1024 * 1024:  # 15 MB limit
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Image file size exceeds 15 MB limit."
            )

        # Decode before touching disk so malformed uploads never create orphan files.
        extension = inspect_uploaded_image(contents)

        # CPU-bound inference must not block the event loop.
        result = await run_in_threadpool(predictor_instance.predict_image, contents, image.filename)

        unique_filename = f"{uuid.uuid4().hex}{extension}"
        saved_file_path = UPLOADS_DIR / unique_filename
        try:
            saved_file_path.write_bytes(contents)
        except OSError:
            saved_file_path.unlink(missing_ok=True)
            raise

        result.imagePath = str(saved_file_path)
        result.imageUrl = f"/uploads/{unique_filename}"
        return result
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error processing upload: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run model inference on uploaded image. Please try again."
        )


# --- Authentication Endpoints ---

@app.post("/api/auth/signup", response_model=AuthResponse, tags=["Authentication"])
@app.post("/api/auth/register", response_model=AuthResponse, tags=["Authentication"])
def register_user(request: UserRegisterRequest):
    """Register a new user account with bcrypt hashed credentials."""
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
        username=request.username or request.email.split("@")[0],
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
        username=user_data.get("username"),
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
    if not user or not verify_password(request.password, user["password_hash"], user.get("password_salt", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(user_id=user["id"], email=user["email"], role=user["role"])
    user_res = UserResponse(
        id=user["id"],
        email=user["email"],
        username=user.get("username"),
        name=user["name"],
        role=user["role"],
        dailyCalorieGoal=user["daily_calorie_goal"],
        dailyProteinGoal=user["daily_protein_goal"],
        createdAt=user["created_at"],
    )
    return AuthResponse(success=True, user=user_res, token=token)


@app.get("/api/auth/me", response_model=UserResponse, tags=["Authentication"])
def get_current_profile(current_user: dict = Depends(get_current_user)):
    """Fetch current logged-in user profile from database."""
    return UserResponse(
        id=current_user["id"],
        email=current_user["email"],
        username=current_user.get("username"),
        name=current_user["name"],
        role=current_user.get("role", "Member"),
        dailyCalorieGoal=current_user.get("daily_calorie_goal", 2000),
        dailyProteinGoal=current_user.get("daily_protein_goal", 120),
        createdAt=current_user.get("created_at", ""),
    )


@app.patch("/api/auth/profile", response_model=UserResponse, tags=["Authentication"])
def update_profile(
    updates: UserProfileUpdateRequest,
    current_user: dict = Depends(get_current_user),
):
    """Update profile settings and daily nutrition targets."""
    updated = update_user_profile(
        user_id=current_user["id"],
        name=updates.name,
        daily_calorie_goal=updates.dailyCalorieGoal,
        daily_protein_goal=updates.dailyProteinGoal,
    )
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    return UserResponse(
        id=updated["id"],
        email=updated["email"],
        username=updated.get("username"),
        name=updated["name"],
        role=updated["role"],
        dailyCalorieGoal=updated["daily_calorie_goal"],
        dailyProteinGoal=updated["daily_protein_goal"],
        createdAt=updated["created_at"],
    )


@app.post("/api/auth/change-password", tags=["Authentication"])
def change_password(
    request: PasswordChangeRequest,
    current_user: dict = Depends(get_current_user),
):
    """Verify the current password, then replace it with a new bcrypt hash."""
    user = get_user_by_email(current_user["email"])
    if not user or not verify_password(
        request.currentPassword, user["password_hash"], user.get("password_salt", "")
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect.",
        )
    if request.currentPassword == request.newPassword:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be different from the current password.",
        )

    new_hash, _ = hash_password(request.newPassword)
    if not update_user_password(current_user["id"], new_hash):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    token = create_access_token(
        user_id=user["id"], email=user["email"], role=user.get("role", "Member")
    )
    return {"status": "password_changed", "token": token}


@app.post("/api/auth/logout", tags=["Authentication"])
def logout_user():
    """Stateless JWT logout; the client discards its token. No server-side revocation list is used."""
    return {"status": "logged_out"}


# --- Persistent History Endpoints (Protected & User-Scoped) ---

@app.get("/api/history", tags=["History"])
def get_prediction_history(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user),
):
    """
    Returns paginated past food estimation scans ONLY for current authenticated user.
    Supports limit and offset parameters.
    """
    items, total_count = get_user_predictions(
        user_id=current_user["id"],
        limit=limit,
        offset=offset,
    )
    return {
        "total": total_count,
        "history": items,
        "limit": limit,
        "offset": offset,
    }


@app.post("/api/history", tags=["History"])
def save_to_history(
    item: HistoryItem,
    current_user: dict = Depends(get_current_user),
):
    """Save an analysis record to database for authenticated user."""
    saved = insert_prediction_record(item.model_dump(), user_id=current_user["id"])
    return {"status": "saved", "item": saved}


@app.delete("/api/history", tags=["History"])
def clear_prediction_history(
    current_user: dict = Depends(get_current_user),
):
    """Delete every history record belonging to the current authenticated user."""
    deleted_count = clear_user_predictions(user_id=current_user["id"])
    return {"status": "cleared", "deleted": deleted_count}


@app.delete("/api/history/{item_id}", tags=["History"])
def delete_history_item(
    item_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Remove an item from history ONLY if it belongs to the current authenticated user."""
    deleted = delete_user_prediction(item_id=item_id, user_id=current_user["id"])
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="History item not found or you do not have permission to delete it.",
        )
    return {"status": "deleted", "id": item_id}
