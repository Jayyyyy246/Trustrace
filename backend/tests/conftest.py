"""
Pytest fixtures and test environment setup.
"""
import io
import sys
from pathlib import Path
from PIL import Image
import pytest
from fastapi.testclient import TestClient

# Ensure project root and backend directory are in python path
backend_path = Path(__file__).resolve().parent.parent
project_root = backend_path.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from app.main import app
from app.core.config import settings
from app.storage.local import local_storage


@pytest.fixture(scope="session")
def client() -> TestClient:
    """FastAPI TestClient fixture."""
    return TestClient(app)


@pytest.fixture
def sample_png_bytes() -> bytes:
    """Generates a valid 200x200 RGB PNG in memory."""
    img = Image.new("RGB", (200, 200), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    """Generates a valid 300x400 JPEG in memory."""
    img = Image.new("RGB", (300, 400), color=(120, 180, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture
def sample_iphone_screenshot_bytes() -> bytes:
    """Generates a 1170x2532 PNG matching standard iPhone viewport."""
    img = Image.new("RGB", (1170, 2532), color=(25, 25, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_tampered_metadata_jpeg_bytes() -> bytes:
    """Generates a JPEG with embedded Adobe Photoshop software metadata."""
    img = Image.new("RGB", (250, 250), color=(200, 50, 50))
    exif = img.getexif()
    # Tag 305 is Software
    exif[305] = "Adobe Photoshop 2026 (Windows)"
    # Tag 271 is Make
    exif[271] = "Apple"
    # Tag 272 is Model
    exif[272] = "iPhone 15 Pro"
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


@pytest.fixture
def invalid_text_file_bytes() -> bytes:
    """Generates plain text bytes disguised with .png extension."""
    return b"This is not a real image file. It should fail magic byte check."
