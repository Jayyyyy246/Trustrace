"""
Entrypoint script to start the TRUSTTRACE backend development server.
Usage:
    python run_backend.py
"""
import sys
from pathlib import Path
import uvicorn

# Ensure 'backend' directory is on sys.path
backend_dir = Path(__file__).resolve().parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", os.environ.get("TRUSTTRACE_PORT", "8001")))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"[TRUSTTRACE] Starting Forensic Backend Server on http://{host}:{port}...")
    print(f"[TRUSTTRACE] Interactive API Documentation available at http://{host}:{port}/docs")
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,
        app_dir=str(backend_dir),
    )
