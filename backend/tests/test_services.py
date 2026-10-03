"""
Tests for individual modular forensic services:
OCR, Screenshot, ML inference, and Evidence Fusion.
"""
from app.services.ocr_service import ocr_service
from app.services.screenshot_analysis_service import screenshot_analysis_service
from app.services.ml_inference_service import ml_inference_service
from app.services.evidence_fusion_service import evidence_fusion_service
from app.schemas.common import AnalyzerStatus, VerdictLabel
from app.schemas.forensic import (
    MetadataAnalyzerResult,
    ImageAnalyzerResult,
    OCRAnalyzerResult,
    ScreenshotAnalyzerResult,
    MLAnalyzerResult,
)


def test_ocr_honest_availability(sample_png_bytes: bytes):
    result, findings = ocr_service.analyze_text(sample_png_bytes)
    # The status must be either COMPLETED (if tesseract installed) or NOT_AVAILABLE
    assert result.status in (AnalyzerStatus.COMPLETED, AnalyzerStatus.NOT_AVAILABLE)
    if result.status == AnalyzerStatus.NOT_AVAILABLE:
        assert result.text is None
        assert "not found" in result.availability_note.lower()


def test_screenshot_viewport_detection():
    # Test standard iPhone 12/13/14 Pro dimensions: 1170x2532
    res_iphone, findings = screenshot_analysis_service.analyze_screenshot(
        width=1170, height=2532, aspect_ratio=round(1170 / 2532, 4)
    )
    assert res_iphone.is_common_viewport is True
    assert "iPhone" in res_iphone.matched_viewport

    # Test random arbitrary cropped dimension: 317x411
    res_cropped, _ = screenshot_analysis_service.analyze_screenshot(
        width=317, height=411, aspect_ratio=round(317 / 411, 4)
    )
    assert res_cropped.is_common_viewport is False


def test_ml_inference_not_available_in_phase_1(sample_png_bytes: bytes):
    result, findings = ml_inference_service.predict(sample_png_bytes)
    assert result.model_status == "NOT_AVAILABLE"
    assert result.predicted_label == "UNKNOWN"
    assert result.confidence is None
    assert len(findings) == 0  # No fake ML findings


def test_evidence_fusion_unknown_when_ml_missing_and_no_tampering():
    meta = MetadataAnalyzerResult(status=AnalyzerStatus.COMPLETED, editing_software_detected=False)
    img = ImageAnalyzerResult(
        status=AnalyzerStatus.COMPLETED,
        dimensions={"width": 100, "height": 100},
        channels=3,
        color_space="RGB",
        aspect_ratio=1.0,
        laplacian_variance=120.0,
        luminance_mean=128.0,
        luminance_std=30.0,
        ela_computed=True,
        ela_variance=20.0,
    )
    ocr = OCRAnalyzerResult(status=AnalyzerStatus.NOT_AVAILABLE)
    screen = ScreenshotAnalyzerResult(status=AnalyzerStatus.COMPLETED, aspect_ratio_standard=True)
    ml = MLAnalyzerResult(model_status="NOT_AVAILABLE")

    verdict, limitations = evidence_fusion_service.synthesize_verdict(
        metadata_res=meta,
        image_res=img,
        ocr_res=ocr,
        screenshot_res=screen,
        ml_res=ml,
        findings=[],
    )

    # When ML is not available and no deterministic tamper evidence is found:
    assert verdict.label == VerdictLabel.UNKNOWN
    assert verdict.confidence is None  # Strictly null as required!
    assert any("NOT_AVAILABLE" in lim for lim in limitations)


def test_evidence_fusion_deterministic_tampering_override():
    meta = MetadataAnalyzerResult(
        status=AnalyzerStatus.COMPLETED,
        editing_software_detected=True,
        anomalies=["Editing software artifact detected in tag 'Software': Adobe Photoshop"],
    )
    img = ImageAnalyzerResult(
        status=AnalyzerStatus.COMPLETED,
        dimensions={"width": 100, "height": 100},
        channels=3,
        color_space="RGB",
        aspect_ratio=1.0,
        laplacian_variance=120.0,
        luminance_mean=128.0,
        luminance_std=30.0,
        ela_computed=True,
    )
    ocr = OCRAnalyzerResult(status=AnalyzerStatus.NOT_AVAILABLE)
    screen = ScreenshotAnalyzerResult(status=AnalyzerStatus.COMPLETED)
    ml = MLAnalyzerResult(model_status="NOT_AVAILABLE")

    verdict, limitations = evidence_fusion_service.synthesize_verdict(
        metadata_res=meta,
        image_res=img,
        ocr_res=ocr,
        screenshot_res=screen,
        ml_res=ml,
        findings=[],
    )

    # Deterministic metadata tamper signature overrides to EDITED
    assert verdict.label == VerdictLabel.EDITED
    assert verdict.confidence > 0.8
    assert "RULE-01-METADATA-EDITING-TOOL-DETECTED" in verdict.decision_rules_triggered
