"""
AgroVision AI - Admin Routes
Dashboard stats, user management, disease CRUD operations
"""
import json
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import Optional, List

from db.database import get_connection
from utils.auth_utils import get_active_user

router = APIRouter(prefix="/api/admin", tags=["Admin"])


# --- Pydantic Models ---

class DiseaseCreate(BaseModel):
    name: str
    display_name: str
    crop: str
    severity: Optional[str] = "Medium"
    symptoms: Optional[List[str]] = []
    treatment_steps: Optional[List[dict]] = []
    products: Optional[List[dict]] = []
    prevention: Optional[List[str]] = []

class DiseaseUpdate(BaseModel):
    display_name: Optional[str] = None
    crop: Optional[str] = None
    severity: Optional[str] = None
    symptoms: Optional[List[str]] = None
    treatment_steps: Optional[List[dict]] = None
    products: Optional[List[dict]] = None
    prevention: Optional[List[str]] = None


# --- Auth Dependency ---

def require_admin(authorization: str = Header(None)):
    """Dependency that requires admin JWT token."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    payload = get_active_user(authorization)
    if not payload["is_admin"]:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return payload


# --- Dashboard Stats ---

@router.get("/stats")
async def get_admin_stats(authorization: str = Header(None)):
    """Get dashboard statistics."""
    require_admin(authorization)
    
    conn = get_connection()
    cursor = conn.cursor()

    # Total scans
    cursor.execute("SELECT COUNT(*) as count FROM detections")
    total_scans = cursor.fetchone()["count"]

    # Registered users
    cursor.execute("SELECT COUNT(*) as count FROM users WHERE is_admin = 0")
    registered_users = cursor.fetchone()["count"]

    # Diseases in DB
    cursor.execute("SELECT COUNT(*) as count FROM diseases")
    diseases_in_db = cursor.fetchone()["count"]

    # Top detected diseases
    cursor.execute("""
        SELECT disease_name, COUNT(*) as count 
        FROM detections 
        GROUP BY disease_name 
        ORDER BY count DESC 
        LIMIT 5
    """)
    top_diseases = [{"name": row["disease_name"], "count": row["count"]} for row in cursor.fetchall()]

    # Scans by crop
    cursor.execute("""
        SELECT crop_type, COUNT(*) as count 
        FROM detections 
        GROUP BY crop_type 
        ORDER BY count DESC
    """)
    scans_by_crop = [{"crop": row["crop_type"], "count": row["count"]} for row in cursor.fetchall()]

    # Average confidence
    cursor.execute("SELECT AVG(confidence) as avg_conf FROM detections")
    avg_result = cursor.fetchone()
    avg_confidence = round(avg_result["avg_conf"], 1) if avg_result["avg_conf"] else 0

    # Recent scans
    cursor.execute("""
        SELECT d.id, d.crop_type, d.disease_name, d.confidence, d.severity, d.created_at,
               COALESCE(u.name, 'Anonymous') as user_name
        FROM detections d
        LEFT JOIN users u ON d.user_id = u.id
        ORDER BY d.created_at DESC
        LIMIT 10
    """)
    recent_scans = [dict(row) for row in cursor.fetchall()]

    # Weekly scans (last 7 days)
    cursor.execute("""
        SELECT date(created_at) as scan_date, COUNT(*) as count
        FROM detections
        WHERE created_at >= date('now', '-7 days')
        GROUP BY date(created_at)
        ORDER BY scan_date
    """)
    weekly_scans = [{"date": row["scan_date"], "count": row["count"]} for row in cursor.fetchall()]

    # Scans by region (using user county)
    cursor.execute("""
        SELECT COALESCE(u.county, 'Unknown') as region, COUNT(*) as count
        FROM detections d
        LEFT JOIN users u ON d.user_id = u.id
        GROUP BY COALESCE(u.county, 'Unknown')
        ORDER BY count DESC
    """)
    region_scans = [{"region": row["region"], "count": row["count"]} for row in cursor.fetchall()]

    conn.close()

    return {
        "total_scans": total_scans,
        "registered_users": registered_users,
        "model_accuracy": 92.4,  # Simulated
        "diseases_in_db": diseases_in_db,
        "avg_confidence": avg_confidence,
        "top_diseases": top_diseases,
        "scans_by_crop": scans_by_crop,
        "recent_scans": recent_scans,
        "weekly_scans": weekly_scans,
        "region_scans": region_scans
    }


# --- User Management ---

@router.get("/users")
async def get_users(authorization: str = Header(None)):
    """List all users."""
    require_admin(authorization)
    
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, name, email, phone, county, is_admin, is_banned, created_at
        FROM users
        ORDER BY created_at DESC
    """)
    users = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return {"users": users}


@router.put("/users/{user_id}/ban")
async def toggle_ban_user(user_id: int, authorization: str = Header(None)):
    """Ban or unban a user."""
    require_admin(authorization)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, is_banned FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    
    if not user:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")
    
    new_status = 0 if user["is_banned"] else 1
    cursor.execute("UPDATE users SET is_banned = ? WHERE id = ?", (new_status, user_id))
    conn.commit()
    conn.close()
    
    action = "banned" if new_status else "unbanned"
    return {"message": f"User {action} successfully", "is_banned": bool(new_status)}


# --- Disease Database CRUD ---

@router.get("/diseases")
async def get_diseases(authorization: str = Header(None)):
    """List all diseases in the database."""
    require_admin(authorization)
    
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM diseases ORDER BY crop, name")
    diseases = []
    for row in cursor.fetchall():
        d = dict(row)
        # Parse JSON fields
        for field in ["symptoms", "treatment_steps", "products", "prevention"]:
            if d.get(field) and isinstance(d[field], str):
                try:
                    d[field] = json.loads(d[field])
                except json.JSONDecodeError:
                    d[field] = []
        diseases.append(d)
    conn.close()
    
    return {"diseases": diseases}


@router.post("/diseases")
async def create_disease(disease: DiseaseCreate, authorization: str = Header(None)):
    """Add a new disease to the database."""
    require_admin(authorization)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            """INSERT INTO diseases (name, display_name, crop, severity, symptoms, treatment_steps, products, prevention)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                disease.name,
                disease.display_name,
                disease.crop,
                disease.severity,
                json.dumps(disease.symptoms),
                json.dumps(disease.treatment_steps),
                json.dumps(disease.products),
                json.dumps(disease.prevention),
            )
        )
        conn.commit()
        disease_id = cursor.lastrowid
    except Exception as e:
        conn.close()
        if "UNIQUE constraint" in str(e):
            raise HTTPException(status_code=400, detail="Disease with this name already exists")
        raise HTTPException(status_code=500, detail=str(e))
    
    conn.close()
    return {"message": "Disease added successfully", "id": disease_id}


@router.put("/diseases/{disease_id}")
async def update_disease(disease_id: int, disease: DiseaseUpdate, authorization: str = Header(None)):
    """Update an existing disease."""
    require_admin(authorization)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM diseases WHERE id = ?", (disease_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Disease not found")
    
    # Build update query dynamically
    updates = []
    values = []
    
    for field, value in disease.model_dump(exclude_none=True).items():
        if field in ["symptoms", "treatment_steps", "products", "prevention"]:
            value = json.dumps(value)
        updates.append(f"{field} = ?")
        values.append(value)
    
    if not updates:
        conn.close()
        return {"message": "No changes to update"}
    
    values.append(disease_id)
    query = f"UPDATE diseases SET {', '.join(updates)} WHERE id = ?"
    cursor.execute(query, values)
    conn.commit()
    conn.close()
    
    return {"message": "Disease updated successfully"}


@router.delete("/diseases/{disease_id}")
async def delete_disease(disease_id: int, authorization: str = Header(None)):
    """Delete a disease from the database."""
    require_admin(authorization)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM diseases WHERE id = ?", (disease_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Disease not found")
    
    cursor.execute("DELETE FROM diseases WHERE id = ?", (disease_id,))
    conn.commit()
    conn.close()
    
    return {"message": "Disease deleted successfully"}
