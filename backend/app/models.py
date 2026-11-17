"""SQLAlchemy 2.0 ORM models for NutriVision AI.

Models per PRD specification:
- User: {id, username, email, hashed_password, created_at}
- FoodPrediction: {id, user_id FK, image_path, food_class, confidence_score,
                   estimated_calories, protein, carbohydrates, fat, fiber, created_at}
"""

from datetime import datetime, timezone
from typing import List, Optional
import uuid

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    """User account model matching PRD specification."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: f"usr_{uuid.uuid4().hex[:12]}",
    )
    username: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    # Profile & goal attributes for user nutrition dashboard
    name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, default=None)
    role: Mapped[str] = mapped_column(String(64), default="Member")
    daily_calorie_goal: Mapped[int] = mapped_column(Integer, default=2000)
    daily_protein_goal: Mapped[int] = mapped_column(Integer, default=120)
    daily_carb_goal: Mapped[int] = mapped_column(Integer, default=220)
    daily_fat_goal: Mapped[int] = mapped_column(Integer, default=60)

    # 1-to-many relationship with food predictions
    predictions: Mapped[List["FoodPrediction"]] = relationship(
        "FoodPrediction",
        back_populates="user",
        cascade="all, delete-orphan",
        order_by="desc(FoodPrediction.created_at)",
    )

    @property
    def password_hash(self) -> str:
        """Compatibility property for auth modules."""
        return self.hashed_password

    @password_hash.setter
    def password_hash(self, value: str) -> None:
        self.hashed_password = value

    def to_dict(self) -> dict:
        """Convert User model to dictionary representation."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "name": self.name or self.username,
            "role": self.role,
            "daily_calorie_goal": self.daily_calorie_goal,
            "daily_protein_goal": self.daily_protein_goal,
            "daily_carb_goal": self.daily_carb_goal,
            "daily_fat_goal": self.daily_fat_goal,
            "created_at": (
                self.created_at.isoformat()
                if isinstance(self.created_at, datetime)
                else str(self.created_at)
            ),
        }


class FoodPrediction(Base):
    """FoodPrediction history model matching PRD specification."""

    __tablename__ = "food_predictions"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: f"hist_{uuid.uuid4().hex[:12]}",
    )
    user_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    image_path: Mapped[str] = mapped_column(Text)
    food_class: Mapped[str] = mapped_column(String(128), index=True)
    confidence_score: Mapped[float] = mapped_column(Float)
    estimated_calories: Mapped[float] = mapped_column(Float)
    protein: Mapped[float] = mapped_column(Float)
    carbohydrates: Mapped[float] = mapped_column(Float)
    fat: Mapped[float] = mapped_column(Float)
    fiber: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    # Presentation metadata for rich frontend UI contract
    confidence_percentage: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    portion: Mapped[Optional[float]] = mapped_column(Float, default=1.0, nullable=True)
    serving_size: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    health_rating: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    dietary_tags: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    health_tips: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_low_confidence: Mapped[Optional[bool]] = mapped_column(Boolean, default=False, nullable=True)

    # Relationship to user
    user: Mapped[Optional[User]] = relationship("User", back_populates="predictions")

    def to_history_dict(self) -> dict:
        """Format as dictionary matching frontend HistoryContext contract."""
        tags = [t.strip() for t in (self.dietary_tags or "").split(",") if t.strip()]

        # Generate accessible URL if image is saved as path or filename
        img_url = self.image_path
        if img_url and not img_url.startswith(("http://", "https://", "data:")):
            clean_name = img_url.replace("\\", "/").split("/")[-1]
            img_url = f"/uploads/{clean_name}"

        return {
            "id": self.id,
            "foodClass": self.food_class,
            "calories": self.estimated_calories,
            "estimatedCalories": self.estimated_calories,
            "protein": self.protein,
            "carbohydrates": self.carbohydrates,
            "carbs": self.carbohydrates,
            "fat": self.fat,
            "fiber": self.fiber,
            "confidenceScore": self.confidence_score,
            "confidencePercentage": (
                self.confidence_percentage
                if self.confidence_percentage is not None
                else round(self.confidence_score * 100, 1)
            ),
            "imagePath": self.image_path,
            "imageUrl": img_url,
            "portion": self.portion or 1.0,
            "servingSize": self.serving_size or "1 serving",
            "isLowConfidence": bool(self.is_low_confidence),
            "healthRating": self.health_rating or "Balanced",
            "dietaryTags": tags,
            "healthTips": self.health_tips or "",
            "timestamp": (
                self.created_at.isoformat()
                if isinstance(self.created_at, datetime)
                else str(self.created_at)
            ),
        }
