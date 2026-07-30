"""
Shamba AI - Authentication Routes
POST /api/auth/register, POST /api/auth/login, GET /api/auth/me
"""
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import Optional

from db.database import get_connection
from utils.auth_utils import hash_password, verify_password, create_jwt, verify_jwt, extract_token_from_header

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


# --- Pydantic Models ---

class RegisterRequest(BaseModel):
    name: str
    contact: str  # Phone or email
    county: Optional[str] = None
    password: str

class LoginRequest(BaseModel):
    contact: Optional[str] = None  # Phone or email
    email: Optional[str] = None   # Alternative: email directly
    password: str
    
    def get_contact(self):
        """Return contact from either field."""
        return self.contact or self.email or ""

class UserResponse(BaseModel):
    id: int
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    county: Optional[str] = None
    is_admin: bool = False


# --- Helper ---

def get_user_from_token(authorization: str = Header(None)):
    """Dependency to get current user from JWT token."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    token = extract_token_from_header(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    
    payload = verify_jwt(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return payload


# --- Routes ---

@router.post("/register")
async def register(req: RegisterRequest):
    """Register a new user account."""
    conn = get_connection()
    cursor = conn.cursor()

    # Check if user already exists
    cursor.execute("SELECT id FROM users WHERE email = ? OR phone = ?", (req.contact, req.contact))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="User with this email/phone already exists")

    # Determine if contact is email or phone
    email = req.contact if "@" in req.contact else None
    phone = req.contact if "@" not in req.contact else None

    # Hash password and insert
    pw_hash = hash_password(req.password)
    try:
        cursor.execute(
            "INSERT INTO users (name, email, phone, county, password_hash) VALUES (?, ?, ?, ?, ?)",
            (req.name, email, phone, req.county, pw_hash)
        )
        conn.commit()
        user_id = cursor.lastrowid
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

    # Create JWT token
    token = create_jwt(user_id, is_admin=False)
    conn.close()

    return {
        "message": "Registration successful",
        "token": token,
        "user": {
            "id": user_id,
            "name": req.name,
            "email": email,
            "phone": phone,
            "county": req.county,
            "is_admin": False
        }
    }


@router.post("/login")
async def login(req: LoginRequest):
    """Login and get JWT token."""
    conn = get_connection()
    cursor = conn.cursor()

    contact_value = req.get_contact()
    if not contact_value:
        raise HTTPException(status_code=400, detail="Email or phone is required")
    
    # Find user by email or phone
    cursor.execute(
        "SELECT id, name, email, phone, county, password_hash, is_admin, is_banned FROM users WHERE email = ? OR phone = ?",
        (contact_value, contact_value)
    )
    user = cursor.fetchone()
    conn.close()

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Check if banned
    if user["is_banned"]:
        raise HTTPException(status_code=403, detail="Account has been suspended")

    # Verify password
    if not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Create JWT token
    token = create_jwt(user["id"], is_admin=bool(user["is_admin"]))

    return {
        "message": "Login successful",
        "token": token,
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "phone": user["phone"],
            "county": user["county"],
            "is_admin": bool(user["is_admin"])
        }
    }


@router.get("/me")
async def get_current_user(authorization: str = Header(None)):
    """Get current user info from JWT token."""
    payload = get_user_from_token(authorization)
    user_id = payload["user_id"]

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, name, email, phone, county, is_admin, is_banned, created_at FROM users WHERE id = ?",
        (user_id,)
    )
    user = cursor.fetchone()
    conn.close()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "phone": user["phone"],
        "county": user["county"],
        "is_admin": bool(user["is_admin"]),
        "is_banned": bool(user["is_banned"]),
        "created_at": user["created_at"]
    }
