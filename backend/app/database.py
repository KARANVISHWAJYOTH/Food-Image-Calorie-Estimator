"""Database persistence layer for NutriVision AI.

Provides SQLite persistence for users, credentials, and scan history.
Supports zero-configuration local deployment and volume persistence in Docker.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
from typing import Any, Generator, Optional
import uuid

DB_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "nutrivision.db"


def get_connection() -> sqlite3.Connection:
    """Create a connection with row factory enabled."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Context manager for transactional database operations."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Initialize database tables and indexes."""
    with get_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            password_salt TEXT NOT NULL,
            role TEXT DEFAULT 'Member',
            daily_calorie_goal INTEGER DEFAULT 2000,
            daily_protein_goal INTEGER DEFAULT 120,
            daily_carb_goal INTEGER DEFAULT 220,
            daily_fat_goal INTEGER DEFAULT 60,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS history (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            food_class TEXT NOT NULL,
            category TEXT,
            confidence_score REAL,
            confidence_percentage REAL,
            estimated_calories REAL,
            protein REAL,
            carbohydrates REAL,
            fat REAL,
            fiber REAL,
            portion REAL DEFAULT 1.0,
            serving_size TEXT,
            image_url TEXT,
            is_low_confidence INTEGER DEFAULT 0,
            health_rating TEXT,
            dietary_tags TEXT,
            health_tips TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
        );

        CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
        CREATE INDEX IF NOT EXISTS idx_history_user_id ON history(user_id);
        CREATE INDEX IF NOT EXISTS idx_history_created_at ON history(created_at DESC);
        """)


# --- User Repository Functions ---

def create_user_record(
    email: str,
    name: str,
    password_hash: str,
    password_salt: str,
    daily_calorie_goal: int = 2000,
    daily_protein_goal: int = 120,
    daily_carb_goal: int = 220,
    daily_fat_goal: int = 60,
) -> dict[str, Any]:
    """Insert a new user record."""
    user_id = f"usr_{uuid.uuid4().hex[:10]}"
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO users (
                id, email, name, password_hash, password_salt, role,
                daily_calorie_goal, daily_protein_goal, daily_carb_goal, daily_fat_goal, created_at
            ) VALUES (?, ?, ?, ?, ?, 'Member', ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                email.lower().strip(),
                name.strip(),
                password_hash,
                password_salt,
                daily_calorie_goal,
                daily_protein_goal,
                daily_carb_goal,
                daily_fat_goal,
                now,
            ),
        )
    return get_user_by_id(user_id)  # type: ignore[return-value]


def get_user_by_email(email: str) -> Optional[dict[str, Any]]:
    """Retrieve user dictionary by email."""
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: str) -> Optional[dict[str, Any]]:
    """Retrieve user dictionary by ID."""
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def update_user_profile(
    user_id: str,
    name: Optional[str] = None,
    daily_calorie_goal: Optional[int] = None,
    daily_protein_goal: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    """Update profile and nutrition goals."""
    updates = []
    params: list[Any] = []
    if name is not None:
        updates.append("name = ?")
        params.append(name.strip())
    if daily_calorie_goal is not None:
        updates.append("daily_calorie_goal = ?")
        params.append(daily_calorie_goal)
    if daily_protein_goal is not None:
        updates.append("daily_protein_goal = ?")
        params.append(daily_protein_goal)

    if not updates:
        return get_user_by_id(user_id)

    params.append(user_id)
    with get_db() as conn:
        conn.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = ?", params)
    return get_user_by_id(user_id)


# --- History Repository Functions ---

def insert_history_item(item: dict[str, Any], user_id: Optional[str] = None) -> dict[str, Any]:
    """Persist a prediction scan into the history table."""
    hist_id = item.get("id") or f"hist_{uuid.uuid4().hex[:10]}"
    now = item.get("timestamp") or datetime.now(timezone.utc).isoformat()
    is_low_conf = 1 if item.get("isLowConfidence") else 0
    dietary_tags = ",".join(item.get("dietaryTags", [])) if isinstance(item.get("dietaryTags"), list) else ""

    with get_db() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO history (
                id, user_id, food_class, category, confidence_score,
                confidence_percentage, estimated_calories, protein,
                carbohydrates, fat, fiber, portion, serving_size,
                image_url, is_low_confidence, health_rating,
                dietary_tags, health_tips, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                hist_id,
                user_id,
                item.get("foodClass", "Unknown Food"),
                item.get("category", "Detected food"),
                item.get("confidenceScore", 0.0),
                item.get("confidencePercentage", 0.0),
                item.get("estimatedCalories") or item.get("calories", 0.0),
                item.get("protein", 0.0),
                item.get("carbohydrates") or item.get("carbs", 0.0),
                item.get("fat", 0.0),
                item.get("fiber", 0.0),
                item.get("portion", 1.0),
                item.get("servingSize", "1 serving"),
                item.get("imageUrl"),
                is_low_conf,
                item.get("healthRating"),
                dietary_tags,
                item.get("healthTips"),
                now,
            ),
        )
    return get_history_item_by_id(hist_id)  # type: ignore[return-value]


def get_history_item_by_id(hist_id: str) -> Optional[dict[str, Any]]:
    """Fetch single history record by ID."""
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM history WHERE id = ?", (hist_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return _format_history_row(dict(row))


def get_all_history(limit: int = 50, user_id: Optional[str] = None) -> list[dict[str, Any]]:
    """Retrieve recent history items ordered by creation date."""
    with get_db() as conn:
        if user_id:
            cursor = conn.execute(
                "SELECT * FROM history WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            )
        else:
            cursor = conn.execute(
                "SELECT * FROM history ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        rows = cursor.fetchall()
        return [_format_history_row(dict(r)) for r in rows]


def delete_history_item(hist_id: str) -> bool:
    """Delete a history item by ID."""
    with get_db() as conn:
        cursor = conn.execute("DELETE FROM history WHERE id = ?", (hist_id,))
        return cursor.rowcount > 0


def _format_history_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert DB row dictionary to API-compatible HistoryItem structure."""
    tags = [t.strip() for t in row.get("dietary_tags", "").split(",") if t.strip()]
    return {
        "id": row["id"],
        "foodClass": row["food_class"],
        "calories": row["estimated_calories"],
        "protein": row["protein"],
        "carbohydrates": row["carbohydrates"],
        "fat": row["fat"],
        "fiber": row.get("fiber", 0.0),
        "confidenceScore": row["confidence_score"],
        "confidencePercentage": row["confidence_percentage"],
        "imageUrl": row.get("image_url"),
        "portion": row.get("portion", 1.0),
        "servingSize": row.get("serving_size"),
        "isLowConfidence": bool(row.get("is_low_confidence", 0)),
        "healthRating": row.get("health_rating"),
        "dietaryTags": tags,
        "healthTips": row.get("health_tips"),
        "timestamp": row["created_at"],
    }
