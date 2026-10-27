"""
AgroVision AI - FastAPI Application Entry Point
Mounts frontend as static files, includes all routers, initializes DB
"""

import os
import subprocess
import sys

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

sys.path.insert(0, str(Path(__file__).parent))

from config import APP_NAME, APP_VERSION
from db.database import init_db, seed_demo_data
from routes.auth import router as auth_router
from routes.detect import router as detect_router
from routes.admin import router as admin_router
from routes.history import router as history_router
from routes.chat import router as chat_router


# Lifespan handler for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    print(f"Starting {APP_NAME} v{APP_VERSION}...")
    try:
        import ddgs  # noqa: F401
    except ImportError:
        print("Installing ddgs...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ddgs", "-q"])
        print("ddgs installed.")
    init_db()
    seed_demo_data()
    print(f"{APP_NAME} is ready!")
    yield
    print(f"Shutting down {APP_NAME}...")


# Create FastAPI app
app = FastAPI(
    title=APP_NAME,
    description="Detect crop diseases instantly. Save your harvest. Save your income.",
    version=APP_VERSION,
    lifespan=lifespan,
)

# CORS middleware — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include API routers
app.include_router(auth_router)
app.include_router(detect_router)
app.include_router(admin_router)
app.include_router(history_router)
app.include_router(chat_router)

# Mount frontend static files
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

if FRONTEND_DIR.exists():
    # Serve static files (CSS, JS, images, etc.)
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    # Serve specific HTML pages
    @app.get("/")
    async def serve_index():
        return FileResponse(FRONTEND_DIR / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/result.html")
    async def serve_result():
        return FileResponse(FRONTEND_DIR / "result.html", headers={"Cache-Control": "no-store"})

    @app.get("/admin_dashboard.html")
    async def serve_admin():
        return FileResponse(FRONTEND_DIR / "admin_dashboard.html", headers={"Cache-Control": "no-store"})

    @app.get("/login.html")
    async def serve_login():
        return FileResponse(FRONTEND_DIR / "login.html", headers={"Cache-Control": "no-store"})

    @app.get("/history.html")
    async def serve_history():
        return FileResponse(FRONTEND_DIR / "history.html", headers={"Cache-Control": "no-store"})

else:
    print(f"⚠️ Frontend directory not found at {FRONTEND_DIR}")


# Health check endpoint
@app.get("/api/health")
async def health_check():
    from ai.agrovision import get_agrovision_registry
    from config import GEMINI_API_KEY
    agrovision = get_agrovision_registry()
    gemini_configured = GEMINI_API_KEY not in {
        "",
        "your-gemini-api-key-here",
        "local-development-placeholder",
    }
    from db.database import get_connection
    db_ok = False
    try:
        conn = get_connection()
        conn.execute("SELECT 1")
        conn.close()
        db_ok = True
    except Exception:
        pass
    
    return {
        "status": "healthy" if db_ok else "unhealthy",
        "app": APP_NAME,
        "version": APP_VERSION,
        "database": "connected" if db_ok else "disconnected",
        "agrovision_models": agrovision.status(),
        "supported_crops": agrovision.supported_crops,
        "gemini_api": "configured" if gemini_configured else "not_configured"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
