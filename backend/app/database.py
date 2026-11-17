"""Database persistence layer for NutriVision AI using SQLAlchemy 2.0.

Provides SQLite default storage with complete Postgres readiness via DATABASE_URL.
Handles session management, Alembic integration, and demo user seeding.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Generator, List, Optional
import uuid

from sqlalchemy import create_engine, select, delete, func
from sqlalchemy.orm import Session, sessionmaker

from .models import Base, User, FoodPrediction

# Database paths and URL configuration
DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Uploaded images directory for mounted volume
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", DATA_DIR / "uploads"))
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

raw_db_url = os.getenv("DATABASE_URL")
if not raw_db_url:
    db_file = DATA_DIR / "nutrivision.db"
    DATABASE_URL = f"sqlite:///{db_file.as_posix()}"
else:
    # Normalize postgres URL scheme for SQLAlchemy 2.0
    if raw_db_url.startswith("postgres://"):
        DATABASE_URL = raw_db_url.replace("postgres://", "postgresql://", 1)
    else:
        DATABASE_URL = raw_db_url

# Configure database engine
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Context manager for transactional database operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db_session() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Initialize database tables and, unless disabled, seed demo accounts."""
    Base.metadata.create_all(bind=engine)
    if os.getenv("SEED_DEMO", "true").strip().lower() not in {"0", "false", "no", "off"}:
        _seed_demo_accounts()


def _seed_demo_accounts() -> None:
    """Seed 1-click demo accounts with bcrypt password hashes if not already present."""
    try:
        from passlib.context import CryptContext
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        hashed_pw = pwd_context.hash("demoPassword123")
        now = datetime.now(timezone.utc)

        demo_users = [
            ("usr_demo_default", "demo_user", "demo@nutrivision.ai", "Demo Explorer", 2000, 120),
        ]

        with get_db() as session:
            for u_id, username, email, name, cal_goal, prot_goal in demo_users:
                stmt = select(User).where((User.email == email) | (User.username == username))
                existing = session.execute(stmt).scalar_one_or_none()
                if not existing:
                    user = User(
                        id=u_id,
                        username=username,
                        email=email,
                        hashed_password=hashed_pw,
                        name=name,
                        role="Pro Member",
                        daily_calorie_goal=cal_goal,
                        daily_protein_goal=prot_goal,
                        daily_carb_goal=240,
                        daily_fat_goal=65,
                        created_at=now,
                    )
                    session.add(user)
    except Exception:
        # Prevent app startup failure if seeding was already performed
        pass


# --- User Repository Functions ---

def create_user_record(
    email: str,
    name: str,
    password_hash: str,
    password_salt: str = "",
    username: Optional[str] = None,
    daily_calorie_goal: int = 2000,
    daily_protein_goal: int = 120,
    daily_carb_goal: int = 220,
    daily_fat_goal: int = 60,
) -> dict[str, Any]:
    """Insert a new user record using SQLAlchemy 2.0."""
    clean_email = email.lower().strip()
    clean_username = (username or clean_email.split("@")[0]).strip()
    clean_name = (name or clean_username).strip()
    user_id = f"usr_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc)

    with get_db() as session:
        # Ensure unique username
        existing_u = session.execute(select(User).where(User.username == clean_username)).scalar_one_or_none()
        if existing_u:
            clean_username = f"{clean_username}_{uuid.uuid4().hex[:4]}"

        new_user = User(
            id=user_id,
            username=clean_username,
            email=clean_email,
            hashed_password=password_hash,
            name=clean_name,
            role="Member",
            daily_calorie_goal=daily_calorie_goal,
            daily_protein_goal=daily_protein_goal,
            daily_carb_goal=daily_carb_goal,
            daily_fat_goal=daily_fat_goal,
            created_at=now,
        )
        session.add(new_user)
        session.flush()
        return new_user.to_dict()


def get_user_by_email(email: str) -> Optional[dict[str, Any]]:
    """Retrieve user dictionary by email."""
    with get_db() as session:
        stmt = select(User).where(User.email == email.lower().strip())
        user = session.execute(stmt).scalar_one_or_none()
        if not user:
            return None
        res = user.to_dict()
        res["password_hash"] = user.hashed_password
        res["password_salt"] = ""
        return res


def get_user_by_id(user_id: str) -> Optional[dict[str, Any]]:
    """Retrieve user dictionary by ID."""
    with get_db() as session:
        stmt = select(User).where(User.id == user_id)
        user = session.execute(stmt).scalar_one_or_none()
        if not user:
            return None
        res = user.to_dict()
        res["password_hash"] = user.hashed_password
        res["password_salt"] = ""
        return res


def get_user_by_username(username: str) -> Optional[dict[str, Any]]:
    """Retrieve user dictionary by username."""
    with get_db() as session:
        stmt = select(User).where(User.username == username.strip())
        user = session.execute(stmt).scalar_one_or_none()
        if not user:
            return None
        res = user.to_dict()
        res["password_hash"] = user.hashed_password
        res["password_salt"] = ""
        return res


def update_user_profile(
    user_id: str,
    name: Optional[str] = None,
    daily_calorie_goal: Optional[int] = None,
    daily_protein_goal: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    """Update user profile preferences."""
    with get_db() as session:
        user = session.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
        if not user:
            return None

        if name is not None and name.strip():
            user.name = name.strip()
        if daily_calorie_goal is not None and daily_calorie_goal > 0:
            user.daily_calorie_goal = daily_calorie_goal
        if daily_protein_goal is not None and daily_protein_goal > 0:
            user.daily_protein_goal = daily_protein_goal

        session.flush()
        return user.to_dict()


def update_user_password(user_id: str, password_hash: str) -> bool:
    """Replace the stored password hash for user_id."""
    with get_db() as session:
        user = session.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
        if not user:
            return False
        user.hashed_password = password_hash
        return True


# --- Food Prediction History Repository Functions ---
def insert_prediction_record(data: dict[str, Any], user_id: str) -> dict[str, Any]:
    """Insert a FoodPrediction record belonging to user_id.

    The record ID is always generated server-side; any client-supplied id is ignored.
    """
    pred_id = f"hist_{uuid.uuid4().hex[:12]}"
    food_class = data.get("foodClass") or data.get("food_class") or "Food Item"

    # Nutrition values
    cals = float(data.get("estimatedCalories") or data.get("calories") or 0.0)
    prot = float(data.get("protein") or 0.0)
    carbs = float(data.get("carbohydrates") or data.get("carbs") or 0.0)
    fat = float(data.get("fat") or 0.0)
    fiber = float(data.get("fiber") or 0.0)
    conf_score = float(data.get("confidenceScore") or 1.0)
    conf_pct = float(data.get("confidencePercentage") or round(conf_score * 100, 1))

    # Image path
    img_path = data.get("imagePath") or data.get("image_path") or data.get("imageUrl") or ""

    # UI metadata
    portion = float(data.get("portion") or 1.0)
    serving_size = data.get("servingSize") or "1 standard serving"
    health_rating = data.get("healthRating") or "Balanced"
    dietary_tags = data.get("dietaryTags")
    tags_str = ",".join(dietary_tags) if isinstance(dietary_tags, list) else str(dietary_tags or "")
    health_tips = data.get("healthTips") or ""
    is_low_conf = bool(data.get("isLowConfidence") or conf_score < 0.65)

    now = datetime.now(timezone.utc)

    with get_db() as session:
        prediction = FoodPrediction(
            id=pred_id,
            user_id=user_id,
            image_path=img_path,
            food_class=food_class,
            confidence_score=conf_score,
            estimated_calories=cals,
            protein=prot,
            carbohydrates=carbs,
            fat=fat,
            fiber=fiber,
            confidence_percentage=conf_pct,
            portion=portion,
            serving_size=serving_size,
            health_rating=health_rating,
            dietary_tags=tags_str,
            health_tips=health_tips,
            is_low_confidence=is_low_conf,
            created_at=now,
        )
        session.add(prediction)
        session.flush()
        return prediction.to_history_dict()


# Alias for compatibility
insert_history_item = insert_prediction_record


def get_user_predictions(
    user_id: str,
    limit: int = 50,
    offset: int = 0,
) -> tuple[List[dict[str, Any]], int]:
    """Retrieve paginated food predictions for user_id, returning (items, total_count)."""
    with get_db() as session:
        # Total count query for current user
        count_stmt = select(func.count(FoodPrediction.id)).where(FoodPrediction.user_id == user_id)
        total_count = session.execute(count_stmt).scalar() or 0

        # Paginated items query
        stmt = (
            select(FoodPrediction)
            .where(FoodPrediction.user_id == user_id)
            .order_by(FoodPrediction.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        rows = session.execute(stmt).scalars().all()
        items = [row.to_history_dict() for row in rows]
        return items, total_count


def get_all_history(
    limit: int = 50,
    offset: int = 0,
    user_id: Optional[str] = None,
) -> List[dict[str, Any]]:
    """Legacy compatibility helper: returns list of formatted history dictionaries."""
    if not user_id:
        return []
    items, _ = get_user_predictions(user_id=user_id, limit=limit, offset=offset)
    return items


def delete_user_prediction(item_id: str, user_id: str) -> bool:
    """Delete a prediction item ONLY if it belongs to the authenticated user."""
    with get_db() as session:
        stmt = delete(FoodPrediction).where(
            (FoodPrediction.id == item_id) & (FoodPrediction.user_id == user_id)
        )
        result = session.execute(stmt)
        return result.rowcount > 0


# Alias for compatibility
delete_history_item = delete_user_prediction


def clear_user_predictions(user_id: str) -> int:
    """Delete every prediction owned by user_id and return the number removed."""
    with get_db() as session:
        result = session.execute(
            delete(FoodPrediction).where(FoodPrediction.user_id == user_id)
        )
        return result.rowcount or 0
