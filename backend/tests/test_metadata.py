"""
Tests for metadata and EXIF analysis.
"""
from app.services.metadata_service import metadata_service
from app.schemas.common import AnalyzerStatus, FindingSeverity


def test_clean_png_metadata(sample_png_bytes: bytes):
    result, findings = metadata_service.extract_metadata(sample_png_bytes)
    assert result.status == AnalyzerStatus.COMPLETED
    assert result.editing_software_detected is False
    assert result.exif_present is False


def test_tampered_photoshop_metadata(sample_tampered_metadata_jpeg_bytes: bytes):
    result, findings = metadata_service.extract_metadata(sample_tampered_metadata_jpeg_bytes)
    assert result.status == AnalyzerStatus.COMPLETED
    assert result.editing_software_detected is True
    assert "photoshop" in result.software.lower()
    assert result.camera_make == "Apple"
    assert result.camera_model == "iPhone 15 Pro"
    
    # Verify high-severity anomaly finding was generated
    tamper_findings = [f for f in findings if f.finding_id == "FIND-META-001"]
    assert len(tamper_findings) == 1
    assert tamper_findings[0].severity == FindingSeverity.HIGH
    assert tamper_findings[0].is_anomaly is True
