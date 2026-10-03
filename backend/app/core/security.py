"""
Security module for TRUSTTRACE.
Provides input validation, magic byte checking, path sanitization,
and bounds enforcement for untrusted evidence ingestion.
"""
import io
import re
import uuid
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image

from app.core.config import settings
from app.core.exceptions import (
    EvidenceValidationError,
    FileTooLargeError,
    UnsupportedMimeTypeError,
)

# Apply Pillow decompression bomb safety limit
Image.MAX_IMAGE_PIXELS = settings.MAX_IMAGE_PIXELS


# Magic byte signatures for supported image types
MAGIC_SIGNATURES = {
    "image/jpeg": [
        b"\xff\xd8\xff",
    ],
    "image/png": [
        b"\x89PNG\r\n\x1a\n",
    ],
    "image/webp": [
        # RIFF header + WEBP format
        b"RIFF",
    ],
    "image/tiff": [
        b"II*\x00",  # Little-endian
        b"MM\x00*",  # Big-endian
    ],
    "image/bmp": [
        b"BM",
    ],
}


def sanitize_filename(filename: str) -> str:
    """
    Sanitizes an untrusted filename by stripping path traversal characters,
    null bytes, and special shell metacharacters.
    """
    if not filename:
        return "unnamed_evidence"
    
    # Remove directory paths
    base_name = Path(filename).name
    # Strip null bytes
    base_name = base_name.replace("\x00", "")
    # Allow alphanumeric, underscore, hyphen, dot
    clean_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", base_name)
    # Ensure it's not empty
    if not clean_name or clean_name.startswith("."):
        clean_name = f"evidence_{clean_name}"
    return clean_name[:128]


def detect_mime_type_from_bytes(data: bytes) -> Optional[str]:
    """
    Deterministically inspects leading bytes (magic numbers) to detect real MIME type.
    Does NOT trust file extensions or client-supplied headers.
    """
    if not data or len(data) < 4:
        return None

    for mime_type, signatures in MAGIC_SIGNATURES.items():
        for sig in signatures:
            if data.startswith(sig):
                if mime_type == "image/webp":
                    # Check for WEBP marker at offset 8-12
                    if len(data) >= 12 and data[8:12] == b"WEBP":
                        return "image/webp"
                else:
                    return mime_type

    return None


def validate_evidence_payload(
    file_bytes: bytes,
    original_filename: str,
) -> Tuple[str, str, str]:
    """
    Validates evidence payload against security bounds:
    - Byte length bounds
    - Magic byte MIME verification
    - Pillow image integrity test
    - Dimension bounds test
    
    Returns:
        (sanitized_filename, validated_mime_type, file_extension)
    """
    file_size = len(file_bytes)
    
    # 1. Size bounds check
    if file_size == 0:
        raise EvidenceValidationError("Uploaded evidence file is empty (0 bytes).")
    
    if file_size > settings.MAX_UPLOAD_SIZE_BYTES:
        raise FileTooLargeError(
            max_bytes=settings.MAX_UPLOAD_SIZE_BYTES,
            actual_bytes=file_size,
        )

    # 2. Magic byte detection
    detected_mime = detect_mime_type_from_bytes(file_bytes)
    if not detected_mime or detected_mime not in settings.ALLOWED_MIME_TYPES:
        raise UnsupportedMimeTypeError(detected_mime or "unknown/binary")

    # 3. Filename sanitization & extension check
    safe_filename = sanitize_filename(original_filename)
    extension = Path(safe_filename).suffix.lower()
    if not extension or extension not in settings.ALLOWED_EXTENSIONS:
        # Default extension based on verified MIME
        mime_to_ext = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/tiff": ".tiff",
            "image/bmp": ".bmp",
        }
        extension = mime_to_ext.get(detected_mime, ".bin")
        safe_filename = f"{Path(safe_filename).stem}{extension}"

    # 4. Image decode integrity test (checks for corrupt headers or truncated byte streams)
    try:
        with Image.open(io.BytesIO(file_bytes)) as pil_img:
            pil_img.verify()
    except Exception as exc:
        raise EvidenceValidationError(f"Corrupt or malformed image data: {str(exc)}")

    return safe_filename, detected_mime, extension


def generate_evidence_id() -> str:
    """Generates a secure UUIDv4 for evidence identification."""
    return str(uuid.uuid4())
