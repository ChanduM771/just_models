"""
AgroVision AI - Authentication Utilities
PBKDF2 password hashing, JWT creation/verification
"""
import hashlib
import os
import hmac
from datetime import datetime, timedelta
from typing import Optional

import jwt

from config import JWT_SECRET, JWT_ALGORITHM, JWT_EXPIRY_HOURS


def hash_password(password: str) -> str:
    """
    Hash a password using PBKDF2 with a random salt.
    Returns format: salt:hash (both hex-encoded)
    """
    salt = os.urandom(16)
    hash_bytes = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        100000  # iterations
    )
    return f"{salt.hex()}:{hash_bytes.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """
    Verify a password against a stored PBKDF2 hash.
    """
    try:
        salt_hex, hash_hex = stored_hash.split(":")
        salt = bytes.fromhex(salt_hex)
        computed_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            100000
        )
        return hmac.compare_digest(computed_hash.hex(), hash_hex)
    except (ValueError, AttributeError):
        return False


def create_jwt(user_id: int, is_admin: bool = False) -> str:
    """
    Create a JWT token for the given user.
    """
    payload = {
        "user_id": user_id,
        "is_admin": is_admin,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_jwt(token: str) -> Optional[dict]:
    """
    Verify and decode a JWT token.
    Returns the payload dict if valid, None if invalid/expired.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def extract_token_from_header(authorization: str) -> Optional[str]:
    """
    Extract Bearer token from Authorization header.
    """
    if not authorization:
        return None
    parts = authorization.split(" ")
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None


def get_active_user(authorization: str) -> dict:
    """Validate a token against the current database user and ban status."""
    from fastapi import HTTPException
    from db.database import get_connection

    token = extract_token_from_header(authorization)
    payload = verify_jwt(token) if token else None
    if not payload or not payload.get("user_id"):
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    conn = get_connection()
    row = conn.execute(
        "SELECT id, is_admin, is_banned FROM users WHERE id = ?",
        (payload["user_id"],),
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=401, detail="User not found")
    if row["is_banned"]:
        raise HTTPException(status_code=403, detail="Account is banned")
    return {"user_id": row["id"], "is_admin": bool(row["is_admin"])}
