"""Authentication and JWT module for NutriVision AI.

Uses bcrypt (passlib) for password hashing and python-jose for issuing
signed 30-minute JWT access tokens. Provides get_current_user FastAPI dependency.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import os
from typing import Any, Optional

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from .database import get_user_by_id

# Load SECRET_KEY from environment with secure fallback
SECRET_KEY = os.getenv("SECRET_KEY", os.getenv("JWT_SECRET_KEY", "nutrivision_super_secret_jwt_key_2026_change_in_prod"))
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # 30-minute token lifespan

# Passlib CryptContext with bcrypt scheme
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security_bearer = HTTPBearer(auto_error=False)


# --- Password Hashing with Bcrypt ---

def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """Hash password using bcrypt via passlib.

    Returns:
        tuple (hash_str, salt_str). Bcrypt embeds salt into the hash string.
    """
    hashed = pwd_context.hash(password)
    return hashed, ""


def verify_password(plain_password: str, stored_hash: str, salt: Optional[str] = None) -> bool:
    """Verify password against bcrypt hash, with fallback for legacy PBKDF2 hashes."""
    if not stored_hash or not plain_password:
        return False

    # Check if bcrypt hash format ($2a$, $2b$, $2y$)
    if stored_hash.startswith(("$2a$", "$2b$", "$2y$")):
        try:
            return pwd_context.verify(plain_password, stored_hash)
        except Exception:
            return False

    # Legacy PBKDF2 fallback
    if salt:
        try:
            computed_bytes = hashlib.pbkdf2_hmac(
                "sha256",
                plain_password.encode("utf-8"),
                salt.encode("utf-8"),
                100000,
            )
            return hmac.compare_digest(computed_bytes.hex(), stored_hash)
        except Exception:
            pass

    return False


# --- JWT Token Generation & Verification with python-jose ---

def create_access_token(
    user_id: str,
    email: str,
    role: str = "Member",
    expires_delta_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES,
) -> str:
    """Issue a signed JWT access token with 30-minute expiration."""
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=expires_delta_minutes)
    payload = {
        "sub": str(user_id),
        "email": email,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[dict[str, Any]]:
    """Decode and validate a signed JWT token string."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
        # Verify expiration
        exp = payload.get("exp")
        if exp and exp < datetime.now(timezone.utc).timestamp():
            return None
        return payload
    except JWTError:
        return None
    except Exception:
        return None


# --- FastAPI Auth Dependencies ---

def extract_token_from_request(
    credentials: Optional[HTTPAuthorizationCredentials],
    authorization: Optional[str],
) -> Optional[str]:
    """Helper to extract raw token from HTTPBearer or Authorization header."""
    if credentials and credentials.credentials:
        return credentials.credentials.strip()
    if authorization and authorization.strip().lower().startswith("bearer "):
        parts = authorization.strip().split(" ", 1)
        if len(parts) == 2:
            return parts[1].strip()
    return None


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer),
    authorization: Optional[str] = Header(None),
) -> dict[str, Any]:
    """Dependency that requires a valid Bearer token and returns the database user record."""
    token = extract_token_from_request(credentials, authorization)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token is missing, expired, or invalid.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token is expired or invalid.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token claims.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account no longer exists.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user["sub"] = user["id"]
    return user


# Aliases for backwards compatibility
get_current_user_required = get_current_user


def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer),
    authorization: Optional[str] = Header(None),
) -> Optional[dict[str, Any]]:
    """Dependency that extracts user if valid token is provided, without raising 401."""
    token = extract_token_from_request(credentials, authorization)
    if not token:
        return None
    payload = decode_access_token(token)
    if not payload:
        return None
    user_id = payload.get("sub")
    if not user_id:
        return None
    user = get_user_by_id(user_id)
    if user:
        user["sub"] = user["id"]
    return user
