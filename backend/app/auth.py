"""Authentication and JWT module for NutriVision AI.

Uses PBKDF2-HMAC-SHA256 for secure password hashing and RFC-compliant
HMAC-SHA256 JWT tokens with zero external binary C-dependencies.
"""

import base64
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
import secrets
from typing import Any, Optional

from fastapi import Depends, HTTPException, Header, status

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "nutrivision_super_secret_jwt_key_2026_change_in_prod")
JWT_ALGORITHM = "HS256"
DEFAULT_TOKEN_EXPIRY_SECONDS = 7 * 24 * 3600  # 7 days


# --- Password Hashing with PBKDF2 ---

def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """Hash a password using PBKDF2-HMAC-SHA256 with 100,000 rounds."""
    if not salt:
        salt = secrets.token_hex(16)
    hashed_bytes = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000,
    )
    return hashed_bytes.hex(), salt


def verify_password(plain_password: str, stored_hash: str, salt: str) -> bool:
    """Verify password against stored PBKDF2 hash."""
    computed_hash, _ = hash_password(plain_password, salt=salt)
    return hmac.compare_digest(computed_hash, stored_hash)


# --- Secure RFC 7519 JWT Implementation ---

def _b64url_encode(data: bytes) -> str:
    """Encode bytes to base64url string without padding."""
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64url_decode(s: str) -> bytes:
    """Decode base64url string with padding restoration."""
    rem = len(s) % 4
    if rem > 0:
        s += "=" * (4 - rem)
    return base64.urlsafe_b64decode(s.encode("utf-8"))


def create_access_token(
    user_id: str,
    email: str,
    role: str = "Member",
    expires_delta_seconds: int = DEFAULT_TOKEN_EXPIRY_SECONDS,
) -> str:
    """Generate a signed HMAC-SHA256 JWT string."""
    now_ts = int(datetime.now(timezone.utc).timestamp())
    header = {"alg": JWT_ALGORITHM, "typ": "JWT"}
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "iat": now_ts,
        "exp": now_ts + expires_delta_seconds,
    }

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")

    unsigned_token = f"{_b64url_encode(header_bytes)}.{_b64url_encode(payload_bytes)}"
    signature = hmac.new(
        JWT_SECRET_KEY.encode("utf-8"),
        unsigned_token.encode("utf-8"),
        hashlib.sha256,
    ).digest()

    return f"{unsigned_token}.{_b64url_encode(signature)}"


def decode_access_token(token: str) -> Optional[dict[str, Any]]:
    """Decode and verify a signed JWT token string."""
    try:
        parts = token.strip().split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, signature_b64 = parts

        unsigned_token = f"{header_b64}.{payload_b64}"
        expected_sig = hmac.new(
            JWT_SECRET_KEY.encode("utf-8"),
            unsigned_token.encode("utf-8"),
            hashlib.sha256,
        ).digest()

        provided_sig = _b64url_decode(signature_b64)
        if not hmac.compare_digest(expected_sig, provided_sig):
            return None

        payload_bytes = _b64url_decode(payload_b64)
        payload = json.loads(payload_bytes.decode("utf-8"))

        now_ts = int(datetime.now(timezone.utc).timestamp())
        if payload.get("exp") and payload["exp"] < now_ts:
            return None  # Token expired

        return payload
    except Exception:
        return None


# --- FastAPI Auth Dependencies ---

def get_current_user_optional(authorization: Optional[str] = Header(None)) -> Optional[dict[str, Any]]:
    """Extract user claims if valid Bearer token is provided."""
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1]
    return decode_access_token(token)


def get_current_user_required(authorization: Optional[str] = Header(None)) -> dict[str, Any]:
    """Require a valid Bearer token or raise 401 Unauthorized."""
    user = get_current_user_optional(authorization)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token is missing, expired, or invalid.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
