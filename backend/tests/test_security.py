"""
Tests for security validation, magic byte checking, and sanitization.
"""
import pytest
from app.core.security import sanitize_filename, detect_mime_type_from_bytes, validate_evidence_payload
from app.core.exceptions import EvidenceValidationError, UnsupportedMimeTypeError


def test_sanitize_filename_traversal():
    assert sanitize_filename("../../../etc/passwd") == "passwd"
    assert sanitize_filename("..\\..\\windows\\system32\\cmd.exe") == "cmd.exe"
    assert sanitize_filename("suspicious;rm -rf;.png") == "suspicious_rm_-rf_.png"
    # Slashes are treated as path separators, taking base name
    assert sanitize_filename("path/to/evidence.jpg") == "evidence.jpg"


def test_detect_mime_type(sample_png_bytes: bytes, sample_jpeg_bytes: bytes, invalid_text_file_bytes: bytes):
    assert detect_mime_type_from_bytes(sample_png_bytes) == "image/png"
    assert detect_mime_type_from_bytes(sample_jpeg_bytes) == "image/jpeg"
    assert detect_mime_type_from_bytes(invalid_text_file_bytes) is None


def test_reject_spoofed_extension(invalid_text_file_bytes: bytes):
    with pytest.raises(UnsupportedMimeTypeError):
        validate_evidence_payload(
            file_bytes=invalid_text_file_bytes,
            original_filename="spoofed_image.png",
        )


def test_reject_empty_payload():
    with pytest.raises(EvidenceValidationError, match="empty"):
        validate_evidence_payload(file_bytes=b"", original_filename="empty.png")
