"""
AgroVision AI - History Routes
GET /api/history - User scan history
GET /api/history/{id} - Specific scan result
"""

import json
from fastapi import APIRouter, HTTPException, Header, Query
from typing import Optional

from db.database import get_connection
from utils.auth_utils import get_active_user

router = APIRouter(prefix="/api/history", tags=["History"])


def get_user_id(authorization: str = Header(None)):
    """Extract user_id from JWT token."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization required")
    return get_active_user(authorization)["user_id"]


@router.get("")
async def get_scan_history(
    authorization: str = Header(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """Get the current user's scan history."""
    user_id = get_user_id(authorization)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """SELECT id, crop_type, disease_name, confidence, severity, created_at
           FROM detections
           WHERE user_id = ?
           ORDER BY created_at DESC
           LIMIT ? OFFSET ?""",
        (user_id, limit, offset),
    )
    scans = []
    for row in cursor.fetchall():
        scan = dict(row)
        # Get display name from diseases table
        cursor.execute(
            "SELECT display_name FROM diseases WHERE name = ?", (scan["disease_name"],)
        )
        disease = cursor.fetchone()
        scan["display_name"] = (
            disease["display_name"] if disease else scan["disease_name"]
        )
        scans.append(scan)

    # Get total count
    cursor.execute(
        "SELECT COUNT(*) as count FROM detections WHERE user_id = ?", (user_id,)
    )
    total = cursor.fetchone()["count"]

    conn.close()

    return {"scans": scans, "total": total, "limit": limit, "offset": offset}


@router.get("/{scan_id}")
async def get_scan_detail(scan_id: int, authorization: str = Header(None)):
    """Get details of a specific scan."""
    user_id = get_user_id(authorization)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """SELECT d.*, dis.display_name
           FROM detections d
           LEFT JOIN diseases dis ON d.disease_name = dis.name
           WHERE d.id = ? AND d.user_id = ?""",
        (scan_id, user_id),
    )
    scan = cursor.fetchone()

    if not scan:
        conn.close()
        raise HTTPException(status_code=404, detail="Scan not found")

    result = dict(scan)
    if not result.get("display_name"):
        raw = result.get("disease_name", "Unknown Disease")
        result["display_name"] = raw.replace("_", " ").title()

    # Parse JSON fields
    for field in ["symptoms", "treatment_steps", "products", "prevention"]:
        if result.get(field) and isinstance(result[field], str):
            try:
                result[field] = json.loads(result[field])
            except json.JSONDecodeError:
                result[field] = []

    # Fetch chat messages
    cursor.execute(
        "SELECT role, content, preview_urls, created_at FROM chat_messages WHERE detection_id = ? ORDER BY created_at ASC",
        (scan_id,),
    )
    history = []
    for row in cursor.fetchall():
        entry = dict(row)
        if entry.get("preview_urls"):
            try:
                entry["preview_urls"] = json.loads(entry["preview_urls"])
            except Exception:
                entry["preview_urls"] = []
        else:
            entry["preview_urls"] = []
        history.append(entry)
    result["chat_history"] = history

    conn.close()

    return result
