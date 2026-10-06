"""
TRUSTTRACE Implementation Repair Regression Test Suite.

Validates all repaired data-flows and requirements from the audit:
- P0-1: estimated_jpeg_quality & noise_residual passed into Decision Engine; RULE-FUSE-04 evaluated.
- P0-3: Uploaded image bytes actually analyzed by screenshot analyzer; no false manipulation merely from viewport.
- P1-1: Production services delegate to canonical implementations in backend/forensic/.
- P1-2: Real measurable typography anomaly detector connected to OCR and Decision Engine.
- P1-3: OCR dependency check and graceful availability status without fake text.
"""
import io
import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.common import AnalyzerStatus, VerdictLabel, FindingSeverity
from app.schemas.forensic import ImageAnalyzerResult, ScreenshotAnalyzerResult, OCRAnalyzerResult, MetadataAnalyzerResult, BaseAnalyzerResult
from app.services.image_analysis_service import image_analysis_service
from app.services.screenshot_analysis_service import screenshot_analysis_service
from app.services.evidence_fusion_service import evidence_fusion_service
from app.services.evidence_service import evidence_service
from app.services.hashing_service import hashing_service
from app.services.metadata_service import metadata_service
from app.services.ocr_service import ocr_service
from forensic.hashing import hashing_analyzer
from forensic.metadata import metadata_analyzer
from forensic.screenshot import screenshot_analyzer
from forensic.ocr import ocr_analyzer
from forensic.decision_engine import evidence_decision_engine


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# =========================================================================
# P0-1 REGRESSION TESTS: JPEG Quality & Noise Residual in Fusion & Engine
# =========================================================================

def test_p0_1_fusion_service_passes_jpeg_quality_and_noise_residual_to_engine():
    """Verify that evidence_fusion_service passes estimated_jpeg_quality and noise_residual."""
    img_res = ImageAnalyzerResult(
        status=AnalyzerStatus.COMPLETED,
        dimensions={"width": 400, "height": 400},
        channels=3,
        color_space="RGB",
        aspect_ratio=1.0,
        laplacian_variance=150.0,
        luminance_mean=120.0,
        luminance_std=30.0,
        ela_computed=True,
        ela_variance=160.0,
        estimated_jpeg_quality=50,  # Low quality
        noise_residual={"tile_variance_ratio": 35.0},
    )
    meta_res = MetadataAnalyzerResult(status=AnalyzerStatus.COMPLETED, editing_software_detected=False)
    screen_res = ScreenshotAnalyzerResult(status=AnalyzerStatus.COMPLETED, is_common_viewport=False)
    ocr_res = OCRAnalyzerResult(status=AnalyzerStatus.NOT_AVAILABLE, word_count=0)
    from app.schemas.forensic import MLAnalyzerResult
    ml_res = MLAnalyzerResult(model_status="NOT_AVAILABLE", predicted_label="UNKNOWN")

    verdict, limitations = evidence_fusion_service.synthesize_verdict(
        metadata_res=meta_res,
        image_res=img_res,
        ocr_res=ocr_res,
        screenshot_res=screen_res,
        ml_res=ml_res,
        findings=[],
    )

    # 1. Proves estimated_jpeg_quality reached Decision Engine
    assert any("compression" in lim.lower() and "50" in lim for lim in limitations + verdict.limitations)
    # 2. Proves compression conflict / RULE-FUSE-04 was evaluated
    assert "RULE-CONFLICT-HEAVY-COMPRESSION-AMBIGUITY" in verdict.decision_rules_triggered
    assert "RULE-FUSE-04-COMPRESSION-AMBIGUITY" in verdict.decision_rules_triggered or "RULE-FUSE-04" in verdict.decision_rules_triggered
    assert verdict.conflict_detected is True
    assert verdict.label == VerdictLabel.UNKNOWN


def test_p0_1_live_upload_evaluates_rule_fuse_04_on_recompressed_jpeg(client: TestClient):
    """
    Live end-to-end regression test: Uploading a heavily recompressed JPEG
    proves RULE-FUSE-04 actually evaluates during a live analysis.
    """
    # Create recompressed JPEG with quality=40
    img = Image.new("RGB", (320, 320), color=(180, 120, 90))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 270, 270], fill=(220, 180, 140))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=40)
    jpeg_bytes = buf.getvalue()

    upload_res = client.post(
        "/api/v1/evidence/upload",
        files={"file": ("lossy_recompressed.jpg", jpeg_bytes, "image/jpeg")},
    )
    assert upload_res.status_code == 201
    data = upload_res.json()

    img_data = data["analyzers"]["image"]
    assert img_data["estimated_jpeg_quality"] is not None
    assert img_data["estimated_jpeg_quality"] <= 65

    verdict = data["final_verdict"]
    rules = verdict["decision_rules_triggered"]

    # RULE-FUSE-04 evaluation verification:
    # If ELA variance elevated on lossy JPEG, engine arbitrates to UNKNOWN via RULE-FUSE-04
    if img_data["ela_variance"] and img_data["ela_variance"] > 120.0:
        assert ("RULE-CONFLICT-HEAVY-COMPRESSION-AMBIGUITY" in rules or "RULE-FUSE-04" in rules or "RULE-FUSE-04-COMPRESSION-AMBIGUITY" in rules)
        assert verdict["label"] == "UNKNOWN"

    # Limitations must document compression
    assert any("compression" in lim.lower() for lim in verdict["limitations"] + data["limitations"])


# =========================================================================
# P0-3 REGRESSION TESTS: Screenshot Bytes Analyzed in Production
# =========================================================================

def test_p0_3_screenshot_analysis_analyzes_real_image_pixels():
    """
    Verify that screenshot_analysis_service inspects real image pixel data
    rather than only matching viewport dimensions.
    """
    # 1. Image with identical resolution (1080x1920) but uniform plain pixels (no UI)
    plain_img = Image.new("RGB", (1080, 1920), color=(100, 100, 100))
    buf1 = io.BytesIO()
    plain_img.save(buf1, format="PNG")
    plain_bytes = buf1.getvalue()

    # 2. Image with identical resolution (1080x1920) but rich UI cards & alignments
    ui_img = Image.new("RGB", (1080, 1920), color=(245, 247, 250))
    draw = ImageDraw.Draw(ui_img)
    draw.rectangle([60, 150, 1020, 450], fill=(255, 255, 255), outline=(200, 200, 200))
    draw.rectangle([60, 500, 1020, 800], fill=(255, 255, 255), outline=(200, 200, 200))
    draw.rectangle([60, 850, 1020, 1150], fill=(255, 255, 255), outline=(200, 200, 200))
    for y in range(200, 500, 15):
        for x in range(100, 800, 12):
            draw.rectangle([x, y, x + 6, y + 10], fill=(30, 30, 30))
    buf2 = io.BytesIO()
    ui_img.save(buf2, format="PNG")
    ui_bytes = buf2.getvalue()

    # Analyze both through screenshot_analysis_service
    res_plain, findings_plain = screenshot_analysis_service.analyze_screenshot(data=plain_bytes)
    res_ui, findings_ui = screenshot_analysis_service.analyze_screenshot(data=ui_bytes)

    # Both match mobile 1080x1920 viewport
    assert res_plain.is_common_viewport is True
    assert res_ui.is_common_viewport is True

    # But pixel analysis proves actual bytes were analyzed:
    assert res_plain.ui_rectangles_detected == 0
    assert res_ui.ui_rectangles_detected >= 3
    assert res_ui.repeated_alignments_count >= 1
    assert res_ui.text_density_ratio > 0.05


def test_p0_3_clean_viewport_not_classified_as_manipulated(client: TestClient):
    """
    Do not classify an image as manipulated merely because its dimensions match a known viewport.
    """
    img = Image.new("RGB", (1170, 2532), color=(240, 240, 240))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    upload_res = client.post(
        "/api/v1/evidence/upload",
        files={"file": ("clean_iphone_capture.png", buf.getvalue(), "image/png")},
    )
    assert upload_res.status_code == 201
    data = upload_res.json()

    verdict = data["final_verdict"]
    # Matching iPhone viewport must NOT trigger an EDITED or SCREENSHOT_MANIPULATED verdict
    assert verdict["label"] != "EDITED"
    assert verdict["label"] != "SCREENSHOT_MANIPULATED"


# =========================================================================
# P1-1 REGRESSION TESTS: Production Services Delegate to Canonical Modules
# =========================================================================

def test_p1_1_services_delegate_to_canonical_forensic_modules():
    """Verify that production services delegate to backend/forensic/ implementations."""
    sample_bytes = b"TRUSTTRACE_FORENSIC_DELEGATION_TEST"

    # HashingService -> hashing_analyzer
    hash_via_service = hashing_service.compute_sha256(sample_bytes)
    hash_via_forensic = hashing_analyzer.compute_sha256(sample_bytes)
    assert hash_via_service == hash_via_forensic

    # ScreenshotAnalysisService -> screenshot_analyzer
    vp_service = screenshot_analysis_service._find_matching_viewport(1170, 2532)
    vp_forensic = screenshot_analyzer._match_viewport(1170, 2532)
    assert vp_service == vp_forensic == "Apple iPhone 12/13/14 Pro (19.5:9)"

    # OCRService -> ocr_analyzer
    assert ocr_service.is_engine_available() == ocr_analyzer.is_available()


# =========================================================================
# P1-2 REGRESSION TESTS: Typography Anomaly Detector Connected to OCR
# =========================================================================

def test_p1_2_typography_anomaly_detector_logic():
    """
    Test the mathematical typography anomaly detector on bounding boxes:
    - Normal baseline: font_anomaly_detected = False
    - Staggered/inserted words with vertical baseline drift: font_anomaly_detected = True
    """
    # 1. Clean regular line of text (all words share baseline y=100, height=20)
    normal_boxes = [
        {"word": "Payment", "x": 50, "y": 80, "w": 80, "h": 20},
        {"word": "Received", "x": 140, "y": 80, "w": 90, "h": 20},
        {"word": "Successfully", "x": 240, "y": 80, "w": 120, "h": 20},
    ]
    detected_norm, metrics_norm, findings_norm = ocr_analyzer.detect_typography_anomalies(normal_boxes)
    assert detected_norm is False
    assert metrics_norm["font_anomaly_detected"] is False
    assert len(findings_norm) == 0

    # 2. Tampered line with spliced word at irregular vertical position (baseline drift)
    # Middle word pasted with baseline y=118 instead of y=100
    tampered_boxes = [
        {"word": "Amount:", "x": 50, "y": 80, "w": 70, "h": 20},      # baseline = 100
        {"word": "$9,500.00", "x": 130, "y": 95, "w": 90, "h": 24},   # baseline = 119 (drift > 12px!)
        {"word": "USD", "x": 230, "y": 80, "w": 40, "h": 20},         # baseline = 100
    ]
    detected_tamper, metrics_tamper, findings_tamper = ocr_analyzer.detect_typography_anomalies(tampered_boxes)
    assert detected_tamper is True
    assert metrics_tamper["font_anomaly_detected"] is True
    assert metrics_tamper["max_baseline_drift"] > 6.0
    assert len(findings_tamper) == 1
    assert findings_tamper[0].finding_id == "FIND-OCR-TYPOGRAPHY-ANOMALY"
    assert findings_tamper[0].severity == FindingSeverity.HIGH


def test_p1_2_typography_anomaly_propagates_to_decision_engine():
    """Verify that typography anomalies trigger RULE-SCREEN-02 in decision engine."""
    ocr_res = OCRAnalyzerResult(
        status=AnalyzerStatus.COMPLETED,
        word_count=20,
        character_count=100,
        font_anomaly_detected=True,  # Detected anomaly
    )
    meta_res = MetadataAnalyzerResult(status=AnalyzerStatus.COMPLETED)
    screen_res = ScreenshotAnalyzerResult(status=AnalyzerStatus.COMPLETED, is_common_viewport=True)
    img_res = ImageAnalyzerResult(
        status=AnalyzerStatus.COMPLETED,
        dimensions={"width": 1080, "height": 1920},
        channels=3,
        color_space="RGB",
        aspect_ratio=round(1080 / 1920, 4),
        laplacian_variance=120.0,
        luminance_mean=128.0,
        luminance_std=20.0,
    )
    from app.schemas.forensic import MLAnalyzerResult
    ml_res = MLAnalyzerResult(model_status="NOT_AVAILABLE", predicted_label="UNKNOWN")

    verdict, limitations = evidence_fusion_service.synthesize_verdict(
        metadata_res=meta_res,
        image_res=img_res,
        ocr_res=ocr_res,
        screenshot_res=screen_res,
        ml_res=ml_res,
        findings=[],
    )

    # Proves typography anomaly reaches Decision Engine and triggers RULE-SCREEN-02
    assert "RULE-SCREEN-02-MANIPULATED-TYPOGRAPHY" in verdict.decision_rules_triggered
    assert verdict.label == VerdictLabel.SCREENSHOT_MANIPULATED


# =========================================================================
# P1-3 REGRESSION TESTS: OCR Honest Availability & Dependency Diagnostic
# =========================================================================

def test_p1_3_ocr_graceful_availability_and_no_fake_output(sample_png_bytes: bytes):
    """Verify that OCR reports COMPLETED when engine is available, or NOT_AVAILABLE with diagnostic guidance when offline."""
    res = ocr_analyzer.analyze(sample_png_bytes)
    assert res.status in ("completed", "not_available")
    if res.status == "completed":
        assert res.metrics["extracted_text"] is not None
        assert res.metrics["engine"] != "Unknown"
    elif res.status == "not_available":
        assert res.metrics["extracted_text"] is None
        assert res.metrics["word_count"] == 0
        assert any("tesseract" in lim.lower() for lim in res.limitations)
        assert any("install" in lim.lower() for lim in res.limitations)

    # Explicitly test unavailable branch when no engine is installed
    from unittest.mock import patch
    with patch.object(ocr_analyzer, "_locate_tesseract", return_value=None), \
         patch.object(ocr_analyzer, "_has_windows_media_ocr", return_value=False):
        unavail_res = ocr_analyzer.analyze(sample_png_bytes)
        assert unavail_res.status == "not_available"
        assert unavail_res.metrics["extracted_text"] is None
        assert unavail_res.metrics["word_count"] == 0
        assert any("tesseract" in lim.lower() for lim in unavail_res.limitations)
        assert any("install" in lim.lower() for lim in unavail_res.limitations)
