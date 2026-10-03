"""
System health and capability inspection endpoint.
"""
from datetime import datetime, timezone
from typing import Dict, Any
from fastapi import APIRouter

from app.core.config import settings
from app.services.ocr_service import ocr_service
from app.services.ml_inference_service import ml_inference_service
from app.storage.local import local_storage

router = APIRouter()


@router.get("/health", summary="System Health & Diagnostic Capabilities")
def get_health() -> Dict[str, Any]:
    """
    Returns platform operational health, active capabilities,
    and analyzer availability flags.
    """
    ocr_available = ocr_service.is_engine_available()
    ml_available = ml_inference_service.is_available()

    return {
        "status": "healthy",
        "system": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "storage": {
            "status": "online" if settings.QUARANTINE_DIR.exists() else "offline",
            "provider": "LocalEvidenceStorage",
        },
        "analyzers": {
            "hashing": "ACTIVE",
            "metadata_exif": "ACTIVE",
            "image_cv": "ACTIVE",
            "compression_ela": "ACTIVE",
            "screenshot_geometry": "ACTIVE",
            "ocr_text": "ACTIVE" if ocr_available else "NOT_AVAILABLE",
            "ml_inference": "ACTIVE" if ml_available else "NOT_AVAILABLE",
        },
    }
