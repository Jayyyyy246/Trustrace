"""
Unit tests for Module 5 — Forensic OCR.
"""
from forensic.ocr import ocr_analyzer


def test_ocr_analyzer_schema_and_graceful_handling(sample_png_bytes: bytes):
    res = ocr_analyzer.analyze(sample_png_bytes)
    assert res.status in ("completed", "not_available")
    assert "extracted_text" in res.metrics
    assert "word_count" in res.metrics
    assert "character_count" in res.metrics
    assert "bounding_boxes" in res.metrics
    assert "language" in res.metrics
    assert res.metrics["language"] == "eng"
    assert len(res.limitations) >= 2


def test_ocr_analyzer_limitations():
    res = ocr_analyzer.analyze(b"fakebytes")
    # Must never crash and should document limitations
    assert len(res.limitations) > 0
    assert any("accuracy" in lim.lower() or "readability" in lim.lower() or "path" in lim.lower() for lim in res.limitations)
