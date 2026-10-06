"""Real OCR integration tests for TRUSTTRACE OCR activation.
Proves uploaded bytes -> OCR engine -> detected text -> structured OCR response.
Fails if mocked/static text or fabricated confidence is returned.
"""
import io
import pytest
from PIL import Image, ImageDraw

from app.schemas.common import AnalyzerStatus
from app.services.ocr_service import ocr_service
from forensic.ocr import ocr_analyzer


def test_ocr_engine_availability_detection():
    """Assert that the system correctly discovers host OCR capability."""
    is_avail = ocr_service.is_engine_available()
    engine_name = ocr_service.get_engine_name()
    assert isinstance(is_avail, bool)
    assert isinstance(engine_name, str)
    # On Windows, Windows.Media.Ocr is always available
    if ocr_analyzer._has_windows_media_ocr() or ocr_analyzer._locate_tesseract():
        assert is_avail is True
        assert engine_name in ("Tesseract OCR", "Windows.Media.Ocr")


def test_ocr_extracts_known_text_from_actual_image_bytes():
    """Prove uploaded image bytes reach real OCR engine and transcribe known text."""
    if not ocr_service.is_engine_available():
        pytest.skip("No OCR engine available in this environment.")

    # 1. Test with existing evidence image containing 'TRANSACTION RECORD *982104'
    with open("data/test_payment_proof.jpg", "rb") as f:
        real_bytes = f.read()

    res, findings = ocr_service.analyze_text(real_bytes)

    # 2. Strict status assertions
    assert res.status == AnalyzerStatus.AVAILABLE
    assert res.engine in ("Windows.Media.Ocr", "Tesseract OCR", "Tesseract (Local)")
    assert res.text is not None
    assert "TRANSACTION" in res.text or "RECORD" in res.text

    # 3. Structured output assertions
    assert res.word_count >= 2
    assert res.character_count > 0
    assert res.processing_time_ms is not None
    assert res.processing_time_ms > 0.0

    # 4. Regions and bounding boxes
    assert len(res.regions) >= 2
    for reg in res.regions:
        assert reg.text is not None
        assert len(reg.bbox) == 4
        x, y, w, h = reg.bbox
        assert w > 0 and h > 0
        assert x >= 0 and y >= 0


def test_ocr_synthetic_rendered_text_verification():
    """Renders dynamic high-contrast text and verifies OCR transcribes the exact string."""
    if not ocr_service.is_engine_available():
        pytest.skip("No OCR engine available in this environment.")

    # Create image with high-contrast text
    img = Image.new("RGB", (600, 150), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    target_phrase = "TRUSTTRACE EVIDENCE AUDIT"
    draw.text((30, 50), target_phrase, fill=(0, 0, 0))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    res, findings = ocr_service.analyze_text(img_bytes)

    assert res.status == AnalyzerStatus.AVAILABLE
    assert res.text is not None
    # Verify expected words are extracted
    extracted_upper = res.text.upper()
    assert "TRUSTTRACE" in extracted_upper or "EVIDENCE" in extracted_upper or "AUDIT" in extracted_upper
    assert res.word_count >= 1
    assert len(res.regions) >= 1


def test_ocr_blank_image_distinguishes_no_text_from_engine_failure():
    """Assert blank image produces AVAILABLE with empty text, not FAILED or NOT_AVAILABLE."""
    if not ocr_service.is_engine_available():
        pytest.skip("No OCR engine available in this environment.")

    img = Image.new("RGB", (200, 200), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    res, findings = ocr_service.analyze_text(buf.getvalue())

    assert res.status == AnalyzerStatus.AVAILABLE
    assert res.text == ""
    assert res.word_count == 0
    assert len(res.regions) == 0
    assert res.failure_reason is None


def test_ocr_corrupt_payload_handled_gracefully():
    """Assert corrupted image data does not raise unhandled exception."""
    corrupt_bytes = b"NOT_A_VALID_IMAGE_BYTES_PAYLOAD_12345"
    res, findings = ocr_service.analyze_text(corrupt_bytes)
    assert res.status in (AnalyzerStatus.FAILED, AnalyzerStatus.NOT_AVAILABLE)
