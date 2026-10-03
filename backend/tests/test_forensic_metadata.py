"""
Unit tests for Module 2 — Forensic Metadata.
"""
from forensic.metadata import metadata_analyzer


def test_metadata_analyzer_clean_image(sample_png_bytes: bytes):
    res = metadata_analyzer.analyze(sample_png_bytes)
    assert res.status == "completed"
    assert res.metrics["format"] == "PNG"
    assert res.metrics["exif_present"] is False
    assert res.metrics["editing_software_detected"] is False
    assert len(res.limitations) >= 2

    # Check finding for stripped EXIF
    stripped_f = [f for f in res.findings if f.finding_id == "FIND-META-STRIPPED"]
    assert len(stripped_f) == 1
    assert stripped_f[0].category == "METADATA"
    assert stripped_f[0].severity.value == "INFO"
    assert stripped_f[0].evidence is not None
    assert stripped_f[0].interpretation != ""
    assert stripped_f[0].limitation != ""


def test_metadata_analyzer_tampered_photoshop(sample_tampered_metadata_jpeg_bytes: bytes):
    res = metadata_analyzer.analyze(sample_tampered_metadata_jpeg_bytes)
    assert res.status == "completed"
    assert res.metrics["exif_present"] is True
    assert res.metrics["camera"]["make"] == "Apple"
    assert res.metrics["camera"]["model"] == "iPhone 15 Pro"
    assert "photoshop" in res.metrics["software"].lower()
    assert res.metrics["editing_software_detected"] is True

    # Check high-severity software finding
    software_f = [f for f in res.findings if f.finding_id == "FIND-META-SOFTWARE"]
    assert len(software_f) == 1
    assert software_f[0].severity.value == "HIGH"
    assert software_f[0].is_anomaly is True
    assert "photoshop" in str(software_f[0].evidence).lower()
    assert software_f[0].interpretation != ""
    # Must explicitly state that software signature does not prove malicious intent
    assert "extent of editing" in software_f[0].limitation.lower() or "not specify" in software_f[0].limitation.lower()
