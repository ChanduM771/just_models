"""
AgroVision AI - Database initialization and seeding.
Uses raw SQLite with row_factory for dict-like rows.
All open() calls use encoding="utf-8" for Windows cp1252 compatibility.
"""

import json
import os
import sqlite3
import logging

from config import ADMIN_EMAIL, ADMIN_PASSWORD, DATABASE_URL

logger = logging.getLogger(__name__)

# Resolve the DB file path from the DATABASE_URL
# DATABASE_URL = "sqlite:///./shamba.db" => file = "./shamba.db"
_DB_PATH = DATABASE_URL.replace("sqlite:///", "").replace("sqlite://", "")
if not _DB_PATH:
    _DB_PATH = "shamba.db"
_DB_PATH = os.path.abspath(os.path.expanduser(_DB_PATH))
os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)


def get_connection() -> sqlite3.Connection:
    """Get a new SQLite connection with row_factory set for dict-like access."""
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            phone TEXT UNIQUE,
            county TEXT,
            password_hash TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0,
            is_banned INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS diseases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            display_name TEXT NOT NULL,
            crop TEXT NOT NULL,
            severity TEXT DEFAULT 'Medium',
            symptoms TEXT DEFAULT '[]',
            treatment_steps TEXT DEFAULT '[]',
            products TEXT DEFAULT '[]',
            prevention TEXT DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            image_path TEXT,
            crop_type TEXT,
            disease_name TEXT,
            confidence REAL,
            severity TEXT,
            symptoms TEXT,
            treatment_steps TEXT,
            products TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            detection_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            preview_urls TEXT DEFAULT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (detection_id) REFERENCES detections(id)
        )
    """)

    cursor.execute("PRAGMA table_info(chat_messages)")
    chat_columns = {row[1] for row in cursor.fetchall()}
    if "preview_urls" not in chat_columns:
        cursor.execute("ALTER TABLE chat_messages ADD COLUMN preview_urls TEXT DEFAULT NULL")

    # Create the configured development admin if no admin exists.
    cursor.execute("SELECT id FROM users WHERE is_admin = 1 LIMIT 1")
    if not cursor.fetchone():
        from utils.auth_utils import hash_password
        admin_hash = hash_password(ADMIN_PASSWORD)
        cursor.execute(
            "INSERT INTO users (name, email, password_hash, is_admin) VALUES (?, ?, ?, 1)",
            ("Admin", ADMIN_EMAIL, admin_hash),
        )
        logger.info("[DB] Admin account created for %s", ADMIN_EMAIL)

    conn.commit()
    conn.close()
    logger.info("[DB] Database initialized.")


def seed_demo_data():
    """Seed the diseases table with data from disease_advice.json."""
    conn = get_connection()
    cursor = conn.cursor()

    # Check if diseases already seeded
    cursor.execute("SELECT COUNT(*) as count FROM diseases")
    count = cursor.fetchone()["count"]
    if count > 0:
        conn.close()
        logger.info(f"[DB] Diseases already seeded ({count} entries). Skipping.")
        return

    # Load disease advice JSON
    advice_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "model", "disease_advice.json"
    )

    if not os.path.exists(advice_path):
        logger.warning(f"[DB] disease_advice.json not found at {advice_path}")
        conn.close()
        return

    try:
        with open(advice_path, "r", encoding="utf-8") as f:
            advice_data = json.load(f)
    except Exception as e:
        logger.error(f"[DB] Failed to load disease_advice.json: {e}")
        conn.close()
        return

    # Insert each disease into the database
    seeded = 0
    for disease_name, info in advice_data.items():
        # Build display name from class_name: Tomato_Early_blight -> Tomato Early Blight
        display = info.get("display_name", disease_name.replace("_", " ").title())

        # Determine crop from the disease name
        crop = info.get("crop", _extract_crop(disease_name))

        # Extract fields with proper format
        symptoms = info.get("symptoms", [])
        if isinstance(symptoms, str):
            symptoms = [symptoms]

        treatment_steps = info.get("treatment_steps", [])
        if isinstance(treatment_steps, str):
            treatment_steps = [{"step": 1, "text": treatment_steps}]

        products = info.get("products", [])
        if isinstance(products, str):
            products = [{"name": products, "type": "N/A", "availability": "Local agrovet"}]

        prevention = info.get("prevention", [])
        if isinstance(prevention, str):
            prevention = [prevention]

        severity = info.get("severity", "Medium")

        try:
            cursor.execute(
                """INSERT OR IGNORE INTO diseases
                   (name, display_name, crop, severity, symptoms, treatment_steps, products, prevention)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    disease_name,
                    display,
                    crop,
                    severity,
                    json.dumps(symptoms),
                    json.dumps(treatment_steps),
                    json.dumps(products),
                    json.dumps(prevention),
                ),
            )
            seeded += 1
        except Exception as e:
            logger.warning(f"[DB] Could not seed {disease_name}: {e}")

    conn.commit()
    conn.close()
    logger.info(f"[DB] Seeded {seeded} diseases into database.")


def _extract_crop(disease_name: str) -> str:
    """Extract crop name from disease class name like Tomato_Early_blight -> Tomato."""
    crop_keywords = {
        "Tomato": "Tomato",
        "Corn": "Maize",
        "Maize": "Maize",
        "Potato": "Potato",
        "Kale": "Kale",
        "Onion": "Onion",
        "Fall": "Maize",  # Fall Armyworm affects Maize
        "Apple": "Apple",
        "Grape": "Grape",
        "Pepper": "Pepper",
        "Peach": "Peach",
        "Cherry": "Cherry",
        "Strawberry": "Strawberry",
    }
    first_word = disease_name.split("_")[0]
    return crop_keywords.get(first_word, first_word)
