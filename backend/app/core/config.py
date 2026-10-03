"""
Core application configuration for TRUSTTRACE.
Production-ready settings managed via Pydantic Settings.
"""
from pathlib import Path
from typing import Set
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    PROJECT_NAME: str = "TRUSTTRACE"
    PROJECT_DESCRIPTION: str = "AI-Assisted Digital Evidence Verification & Tamper Detection Platform"
    VERSION: str = "1.0.0-phase1"
    API_V1_PREFIX: str = "/api/v1"
    
    # Environment
    ENVIRONMENT: str = Field(default="development", description="Environment mode")
    DEBUG: bool = Field(default=False, description="Debug mode")
    
    # Storage & Security bounds
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    QUARANTINE_DIR: Path = DATA_DIR / "quarantine"
    ARTIFACTS_DIR: Path = DATA_DIR / "artifacts"
    REPORTS_DIR: Path = DATA_DIR / "reports"
    
    # 25 MB max upload size
    MAX_UPLOAD_SIZE_BYTES: int = 25 * 1024 * 1024
    
    # Maximum image dimensions to prevent decompression bombs
    MAX_IMAGE_DIMENSION: int = 8192
    MAX_IMAGE_PIXELS: int = 100_000_000
    
    # Allowed MIME types and extensions
    ALLOWED_MIME_TYPES: Set[str] = {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/tiff",
        "image/bmp",
    }
    
    ALLOWED_EXTENSIONS: Set[str] = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".tif",
        ".tiff",
        ".bmp",
    }
    
    # CORS
    CORS_ORIGINS: list[str] = ["*"]
    
    model_config = {
        "env_prefix": "TRUSTTRACE_",
        "case_sensitive": True,
        "extra": "ignore",
    }


settings = Settings()

# Ensure critical data directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
settings.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
