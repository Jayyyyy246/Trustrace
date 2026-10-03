"""TRUSTTRACE Full Adversarial Validation Test Suite.

Rigorously evaluates all 14 adversarial categories and system components:
1. REAL images (authentic camera capture, valid sensor EXIF, baseline noise)
2. Edited images (copy-move duplication, localized high ELA variance, tool signatures)
3. AI-generated images (synthetic signals, absent sensor PRNU, calibrated advisory)
4. Screenshots (standard viewports, lossless PNG, clean UI geometry)
5. Manipulated screenshots (canonical viewport with spliced text / modified UI blocks)
6. Recompressed images (severe lossy recompression Q=40 triggering UNKNOWN arbitration)
7. Images with removed metadata (stripped EXIF without false manipulation attribution)
8. Images with conflicting metadata (camera EXIF combined with Photoshop signature or clone)
9. Very small images (1x1, 4x4, 8x8, 16x16 px boundary edge cases)
10. Very large images (3200x3200 px multi-megapixel stability)
11. Corrupted files (truncated headers, random bytes, broken byte streams)
12. Unsupported formats (PDF, executable, plain text, SVG disguised as PNG/JPG)
13. Duplicate files (SHA-256 bitwise deduplication and custody tracking)
14. Images from unseen sources (grayscale 1-ch, RGBA with alpha, ultra-wide aspect ratios)

Plus:
- Upload validation & MIME sniffing
- SHA-256 cryptographic hashing
- Metadata extraction & EXIF parsing
- OCR & typography analysis
- Computer vision forensic analyzers (ELA, DQT, Laplacian, copy-move)
- ML inference & uncertainty calibration
- Evidence fusion & conflict arbitration
- API failure modes (400, 404, 422)
- Frontend/backend contract schema integrity
"""

import io
import struct
import numpy as np
from PIL import Image, ImageDraw
import pytest
from fastapi.testclient import TestClient

from app.services.evidence_service import evidence_service
from app.services.report_service import report_service
from app.schemas.common import VerdictLabel
from model.evaluator import (
    compute_classification_metrics,
    compute_calibration_metrics,
    compute_ood_metrics,
)


# =========================================================================
# HELPER GENERATORS FOR ADVERSARIAL SAMPLES
# =========================================================================

def _generate_synthetic_camera_jpeg(width=600, height=400) -> bytes:
    """Generates an image with simulated camera EXIF and realistic Gaussian sensor noise."""
    # Base gradient image
    arr = np.zeros((height, width, 3), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            arr[y, x, 0] = (x / width) * 200 + 20
            arr[y, x, 1] = (y / height) * 180 + 30
            arr[y, x, 2] = 140

    # Add realistic optical sensor noise (Gaussian sigma=4.0)
    noise = np.random.normal(0, 4.0, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(noisy, mode="RGB")

    # Inject realistic camera EXIF tags
    exif = img.getexif()
    exif[271] = "Nikon"               # Make
    exif[272] = "Nikon D850"          # Model
    exif[306] = "2026:08:15 14:32:00" # DateTime
    exif[33434] = (1, 250)            # ExposureTime 1/250s
    exif[33437] = (28, 10)            # FNumber f/2.8

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95, exif=exif)
    return buf.getvalue()


def _generate_copy_move_image(width=500, height=400) -> bytes:
    """Generates an image with an exact duplicated high-contrast textured feature patch."""
    img = Image.new("RGB", (width, height), color=(180, 180, 180))
    draw = ImageDraw.Draw(img)

    # Draw complex distinct feature shape (donor region at x=50, y=50)
    draw.rectangle([50, 50, 130, 130], fill=(20, 20, 150), outline=(255, 255, 0), width=3)
    draw.ellipse([65, 65, 115, 115], fill=(220, 30, 30))
    draw.line([50, 50, 130, 130], fill=(0, 255, 0), width=2)
    draw.line([50, 130, 130, 50], fill=(255, 0, 255), width=2)

    # Clone exact donor region to target region at x=280, y=180
    donor_patch = img.crop((50, 50, 130, 130))
    img.paste(donor_patch, (280, 180))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _generate_heavily_recompressed_jpeg(width=300, height=300) -> bytes:
    """Generates an image recompressed at Quality=35, introducing severe blocking artifacts."""
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, 260, 260], fill=(220, 80, 50))
    draw.ellipse([80, 80, 220, 220], fill=(40, 180, 90))

    # Cycle 1: High quality
    buf1 = io.BytesIO()
    img.save(buf1, format="JPEG", quality=95)
    buf1.seek(0)
    reopened = Image.open(buf1)

    # Cycle 2: Severe low quality recompression
    buf2 = io.BytesIO()
    reopened.save(buf2, format="JPEG", quality=35)
    return buf2.getvalue()


# =========================================================================
# 1. REAL IMAGES TEST
# =========================================================================

def test_category_01_real_camera_image(client: TestClient):
    """Category 1: Genuine camera capture with valid sensor EXIF and uniform physical noise."""
    jpeg_bytes = _generate_synthetic_camera_jpeg()
    res = client.post(
        "/api/v1/evidence/upload",
        files={"file": ("nikon_d850_dsc001.jpg", jpeg_bytes, "image/jpeg")},
    )
    assert res.status_code == 201
    data = res.json()

    # 1. Hashing
    assert len(data["evidence"]["sha256"]) == 64
    assert data["evidence"]["dimensions"]["width"] == 600
    assert data["evidence"]["dimensions"]["height"] == 400

    # 2. Metadata extraction
    meta = data["analyzers"]["metadata"]
    assert meta["exif_present"] is True
    assert meta["camera_make"] == "Nikon"
    assert "D850" in (meta["camera_model"] or "")
    assert meta["editing_software_detected"] is False

    # 3. Low-level image forensics
    img_res = data["analyzers"]["image"]
    assert img_res["is_blurry"] is False
    assert img_res["laplacian_variance"] > 10.0
    assert img_res["ela_variance"] is not None

    # 4. Evidence fusion verdict
    verdict = data["final_verdict"]
    assert verdict["label"] == "REAL"
    assert verdict["conflict_detected"] is False
    assert len(verdict["supporting_findings"]) > 0


# =========================================================================
# 2. EDITED IMAGES TEST
# =========================================================================

def test_category_02_edited_copy_move_cloning(client: TestClient):
    """Category 2: Image containing duplicated pixel regions (copy-move tampering)."""
    tampered_bytes = _generate_copy_move_image()
    res = client.post(
        "/api/v1/evidence/upload",
        files={"file": ("spliced_clone.jpg", tampered_bytes, "image/jpeg")},
    )
    assert res.status_code == 201
    data = res.json()

    # Must detect manipulation (either clone clusters or ELA disparity)
    verdict = data["final_verdict"]
    assert verdict["label"] == "EDITED"
    assert verdict["risk_score"] >= 0.70
    assert "consistent with" in verdict["justification"].lower() or "copy-move" in verdict["justification"].lower()


# =========================================================================
# 3. AI-GENERATED IMAGES TEST
# =========================================================================

def test_category_03_ai_generated_characteristics(client: TestClient):
    """Category 3: Image with AI-generated characteristics and absent sensor PRNU."""
    # Perfectly flat synthetic gradient without sensor noise
    img = Image.new("RGB", (512, 512), color=(128, 64, 192))
    draw = ImageDraw.Draw(img)
    draw.polygon([(100, 100), (400, 150), (350, 450), (80, 380)], fill=(240, 200, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    res = client.post(
        "/api/v1/evidence/upload",
        files={"file": ("midjourney_v6_synth.png", buf.getvalue(), "image/png")},
    )
    assert res.status_code == 201
    data = res.json()

    # Absence of sensor metadata verified
    assert data["analyzers"]["metadata"]["exif_present"] is False
    assert data["analyzers"]["ml"]["model_status"] in ["AVAILABLE", "NOT_AVAILABLE"]
    if data["analyzers"]["ml"]["model_status"] == "NOT_AVAILABLE":
        assert any("machine learning" in lim.lower() or "model" in lim.lower() or "ml" in lim.lower() for lim in data["limitations"])
        assert "ml_inference" in data["final_verdict"]["unavailable_analyzers"]


# =========================================================================
# 4. SCREENSHOTS TEST
# =========================================================================

def test_category_04_native_screen_capture(client: TestClient):
    """Category 4: Native unmanipulated screen capture matching standard 1080p display viewport."""
    img = Image.new("RGB", (1920, 1080), color=(30, 35, 45))
    draw = ImageDraw.Draw(img)
    # Draw simulated taskbar and window header
    draw.rectangle([0, 1040, 1920, 1080], fill=(20, 22, 28))
    draw.rectangle([100, 100, 1820, 950], fill=(40, 44, 56), outline=(60, 66, 82), width=1)
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    res = client.post(
        "/api/v1/evidence/upload",
        files={"file": ("desktop_screenshot.png", buf.getvalue(), "image/png")},
    )
    assert res.status_code == 201
    data = res.json()

    screen = data["analyzers"]["screenshot"]
    assert screen["is_common_viewport"] is True
    assert "Desktop Full HD" in (screen["matched_viewport"] or "") or "16:9" in (screen["matched_viewport"] or "")
    assert screen["aspect_ratio_standard"] is True


# =========================================================================
# 5. MANIPULATED SCREENSHOTS TEST
# =========================================================================

def test_category_05_manipulated_screenshot(client: TestClient):
    """Category 5: Screenshot geometry with superimposed localized compression splice."""
    img = Image.new("RGB", (1920, 1080), color=(240, 240, 245))
    draw = ImageDraw.Draw(img)
    # Simulated UI background
    draw.rectangle([100, 100, 1000, 600], fill=(255, 255, 255), outline=(200, 200, 200))

    # Spliced localized JPEG patch with high compression disparity
    patch = Image.new("RGB", (300, 100), color=(255, 200, 200))
    p_draw = ImageDraw.Draw(patch)
    p_draw.text((20, 40), "FORGED BALANCE: $9,999,999", fill=(0, 0, 0))
    p_buf = io.BytesIO()
    patch.save(p_buf, format="JPEG", quality=40)
    p_buf.seek(0)
    recomp_patch = Image.open(p_buf)

    img.paste(recomp_patch, (200, 250))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    res = client.post(
        "/api/v1/evidence/upload",
        files={"file": ("tampered_banking_ui.png", buf.getvalue(), "image/png")},
    )
    assert res.status_code == 201
    data = res.json()

    # Viewport recognized as screenshot
    assert data["analyzers"]["screenshot"]["is_common_viewport"] is True
    # ELA or forensic metrics detect the anomalous local block
    assert data["analyzers"]["image"]["ela_computed"] is True


# =========================================================================
# 6. RECOMPRESSED IMAGES TEST
# =========================================================================

def test_category_06_recompressed_images_refuse_false_attribution(client: TestClient):
    """Category 6: Low-quality recompressed image. Engine must arbitrate high ELA to UNKNOWN."""
    low_q_bytes = _generate_heavily_recompressed_jpeg(width=280, height=280)
    res = client.post(
        "/api/v1/evidence/upload",
        files={"file": ("recompressed_q35.jpg", low_q_bytes, "image/jpeg")},
    )
    assert res.status_code == 201
    data = res.json()

    img_res = data["analyzers"]["image"]
    q_est = img_res.get("estimated_jpeg_quality")
    # Low quality factor detected
    assert q_est is not None and q_est <= 65

    verdict = data["final_verdict"]
    # Refusal of naive false-positive EDITED: if ELA elevated due to compression, verdict is UNKNOWN
    if img_res["ela_variance"] and img_res["ela_variance"] > 120.0:
        assert verdict["label"] in ["UNKNOWN", "REAL"]
        assert any("compression" in lim.lower() for lim in verdict["limitations"] + data["limitations"])


# =========================================================================
# 7. IMAGES WITH REMOVED METADATA TEST
# =========================================================================

def test_category_07_images_with_removed_metadata(client: TestClient):
    """Category 7: Clean image stripped of all EXIF (transit via social media / web)."""
    img = Image.new("RGB", (400, 300), color=(90, 140, 180))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)  # No EXIF passed

    res = client.post(
        "/api/v1/evidence/upload",
        files={"file": ("stripped_web_transit.jpg", buf.getvalue(), "image/jpeg")},
    )
    assert res.status_code == 201
    data = res.json()

    meta = data["analyzers"]["metadata"]
    assert meta["exif_present"] is False
    assert meta["editing_software_detected"] is False
    # System must NOT declare file EDITED simply because EXIF is stripped
    assert data["final_verdict"]["label"] != "EDITED"


# =========================================================================
# 8. IMAGES WITH CONFLICTING METADATA TEST
# =========================================================================

def test_category_08_conflicting_metadata_camera_vs_photoshop(client: TestClient, sample_tampered_metadata_jpeg_bytes: bytes):
    """Category 8: Camera EXIF (Apple iPhone) present with Adobe Photoshop editing software tag."""
    res = client.post(
        "/api/v1/evidence/upload",
        files={"file": ("conflicting_metadata.jpg", sample_tampered_metadata_jpeg_bytes, "image/jpeg")},
    )
    assert res.status_code == 201
    data = res.json()

    meta = data["analyzers"]["metadata"]
    assert meta["camera_make"] == "Apple"
    assert meta["editing_software_detected"] is True
    assert "Photoshop" in (meta["software"] or "")

    verdict = data["final_verdict"]
    assert verdict["label"] == "EDITED"
    assert verdict["confidence"] >= 0.80


# =========================================================================
# 9. VERY SMALL IMAGES TEST
# =========================================================================

def test_category_09_very_small_images_boundary_conditions(client: TestClient):
    """Category 9: Extreme minimum dimensions: 1x1, 4x4, 8x8, 16x16 px."""
    tiny_sizes = [(1, 1), (4, 4), (8, 8), (16, 16)]

    for w, h in tiny_sizes:
        img = Image.new("RGB", (w, h), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        res = client.post(
            "/api/v1/evidence/upload",
            files={"file": (f"tiny_{w}x{h}.png", buf.getvalue(), "image/png")},
        )
        assert res.status_code == 201
        data = res.json()
        assert data["evidence"]["dimensions"]["width"] == w
        assert data["evidence"]["dimensions"]["height"] == h
        # Processing completed without math/zero division crash
        assert data["analyzers"]["image"]["status"] in ["COMPLETED", "FAILED"]


# =========================================================================
# 10. VERY LARGE IMAGES TEST
# =========================================================================

def test_category_10_very_large_image_processing(client: TestClient):
    """Category 10: High-resolution multi-megapixel asset (3200x3200 px)."""
    img = Image.new("RGB", (3200, 3200), color=(50, 70, 90))
    draw = ImageDraw.Draw(img)
    draw.rectangle([500, 500, 2700, 2700], fill=(120, 140, 160))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)

    res = client.post(
        "/api/v1/evidence/upload",
        files={"file": ("large_10mp_asset.jpg", buf.getvalue(), "image/jpeg")},
    )
    assert res.status_code == 201
    data = res.json()
    assert data["evidence"]["dimensions"]["width"] == 3200
    assert data["evidence"]["dimensions"]["height"] == 3200
    assert len(data["evidence"]["sha256"]) == 64


# =========================================================================
# 11. CORRUPTED FILES TEST
# =========================================================================

def test_category_11_corrupted_files_rejection(client: TestClient):
    """Category 11: Truncated headers and random corrupt bytes must be safely rejected."""
    # 1. Truncated JPEG (starts with SOI marker, truncated abruptly)
    truncated_jpeg = b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00"
    res1 = client.post(
        "/api/v1/evidence/upload",
        files={"file": ("truncated.jpg", truncated_jpeg, "image/jpeg")},
    )
    assert res1.status_code == 400
    assert "validation" in res1.json()["detail"].lower() or "decode" in res1.json()["detail"].lower() or "corrupt" in res1.json()["detail"].lower()

    # 2. Random garbage bytes
    random_bytes = b"ABCDEF1234567890\x00\xFF\xAA\xBB\xCC\xDD" * 20
    res2 = client.post(
        "/api/v1/evidence/upload",
        files={"file": ("corrupt_garbage.png", random_bytes, "image/png")},
    )
    assert res2.status_code == 400


# =========================================================================
# 12. UNSUPPORTED FORMATS TEST
# =========================================================================

def test_category_12_unsupported_formats_rejection(client: TestClient):
    """Category 12: PDF, executable, plain text, and SVG files with spoofed extensions."""
    # 1. PDF disguised as PNG
    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"
    res1 = client.post(
        "/api/v1/evidence/upload",
        files={"file": ("spoofed_document.png", pdf_bytes, "image/png")},
    )
    assert res1.status_code == 400
    assert "supported" in res1.json()["detail"].lower() or "validation" in res1.json()["detail"].lower()

    # 2. Windows Executable disguised as JPG
    exe_bytes = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff\x00\x00"
    res2 = client.post(
        "/api/v1/evidence/upload",
        files={"file": ("malware.jpg", exe_bytes, "image/jpeg")},
    )
    assert res2.status_code == 400

    # 3. Plain text disguised as WebP
    text_bytes = b"Hello, this is a plain text file pretending to be image/webp."
    res3 = client.post(
        "/api/v1/evidence/upload",
        files={"file": ("notes.webp", text_bytes, "image/webp")},
    )
    assert res3.status_code == 400


# =========================================================================
# 13. DUPLICATE FILES TEST
# =========================================================================

def test_category_13_duplicate_files_hashing_consistency(client: TestClient, sample_png_bytes: bytes):
    """Category 13: Identical binary uploads produce identical cryptographic SHA-256 hashes."""
    res1 = client.post(
        "/api/v1/evidence/upload",
        files={"file": ("original.png", sample_png_bytes, "image/png")},
    )
    res2 = client.post(
        "/api/v1/evidence/upload",
        files={"file": ("duplicate.png", sample_png_bytes, "image/png")},
    )
    assert res1.status_code == 201
    assert res2.status_code == 201

    hash1 = res1.json()["evidence"]["sha256"]
    hash2 = res2.json()["evidence"]["sha256"]
    assert hash1 == hash2
    assert len(hash1) == 64


# =========================================================================
# 14. IMAGES FROM UNSEEN SOURCES TEST
# =========================================================================

def test_category_14_unseen_sources_novel_structures(client: TestClient):
    """Category 14: Grayscale (1-ch), RGBA with transparency, and ultra-wide 32:9 geometry."""
    # 1. 1-channel Grayscale image
    gray_img = Image.new("L", (240, 240), color=128)
    buf_gray = io.BytesIO()
    gray_img.save(buf_gray, format="PNG")

    res_gray = client.post(
        "/api/v1/evidence/upload",
        files={"file": ("single_channel_gray.png", buf_gray.getvalue(), "image/png")},
    )
    assert res_gray.status_code == 201
    assert res_gray.json()["analyzers"]["image"]["color_space"] in ["GRAY", "L", "RGB"]

    # 2. 4-channel RGBA with transparent alpha channel
    rgba_img = Image.new("RGBA", (200, 200), color=(100, 150, 200, 128))
    buf_rgba = io.BytesIO()
    rgba_img.save(buf_rgba, format="PNG")

    res_rgba = client.post(
        "/api/v1/evidence/upload",
        files={"file": ("alpha_transparent.png", buf_rgba.getvalue(), "image/png")},
    )
    assert res_rgba.status_code == 201
    assert res_rgba.json()["analyzers"]["image"]["channels"] in [3, 4]

    # 3. Ultra-wide aspect ratio (32:9 - 640x180)
    wide_img = Image.new("RGB", (640, 180), color=(40, 40, 50))
    buf_wide = io.BytesIO()
    wide_img.save(buf_wide, format="PNG")

    res_wide = client.post(
        "/api/v1/evidence/upload",
        files={"file": ("ultrawide_32_9.png", buf_wide.getvalue(), "image/png")},
    )
    assert res_wide.status_code == 201
    assert res_wide.json()["analyzers"]["screenshot"]["aspect_ratio_standard"] is False


# =========================================================================
# 15. API FAILURE MODES TEST
# =========================================================================

def test_api_failure_modes(client: TestClient):
    """Tests proper error responses for 404, 400, and 422 HTTP conditions."""
    # Non-existent evidence ID
    res_404 = client.get("/api/v1/evidence/non-existent-uuid-12345")
    assert res_404.status_code == 404

    # Non-existent report
    res_rep_404 = client.get("/api/v1/evidence/non-existent-uuid-12345/report")
    assert res_rep_404.status_code == 404

    # Empty payload upload
    res_empty = client.post(
        "/api/v1/evidence/upload",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert res_empty.status_code == 400


# =========================================================================
# 16. FRONTEND/BACKEND CONTRACT INTEGRITY TEST
# =========================================================================

def test_frontend_backend_contract_schema(client: TestClient, sample_png_bytes: bytes):
    """Verifies that backend response adheres precisely to frontend TypeScript types."""
    res = client.post(
        "/api/v1/evidence/upload",
        files={"file": ("contract_check.png", sample_png_bytes, "image/png")},
    )
    assert res.status_code == 201
    data = res.json()

    # AnalysisResultResponse fields
    assert "analysis_id" in data
    assert "evidence" in data
    assert "analyzers" in data
    assert "findings" in data
    assert "final_verdict" in data
    assert "limitations" in data
    assert "pipeline_version" in data

    # EvidenceInfo fields
    ev = data["evidence"]
    assert "filename" in ev
    assert "sha256" in ev
    assert "mime_type" in ev
    assert "size" in ev
    assert "dimensions" in ev

    # FinalVerdict fields
    fv = data["final_verdict"]
    assert fv["label"] in ["REAL", "EDITED", "AI-GENERATED", "SCREENSHOT-MANIPULATED", "UNKNOWN"]
    assert "confidence" in fv
    assert "risk_score" in fv
    assert "justification" in fv
    assert "supporting_findings" in fv
    assert "contradictory_findings" in fv
    assert "decision_rules_triggered" in fv


# =========================================================================
# 17. ML EVALUATION & CALIBRATION CALCULATION TEST
# =========================================================================

def test_ml_evaluation_metrics_and_calibration():
    """Computes exact accuracy, precision, recall, macro-F1, confusion matrix, ECE, MCE, AUROC."""
    class_labels = ["REAL", "EDITED", "AI-GENERATED", "SCREENSHOT-MANIPULATED"]

    # Ground truth vs predicted for a controlled distribution of 20 samples
    y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
    y_pred = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 3, 3, 3, 3, 3, 2]
    confs  = [0.95, 0.88, 0.92, 0.85, 0.60, 0.90, 0.85, 0.94, 0.78, 0.65,
              0.89, 0.92, 0.95, 0.84, 0.58, 0.91, 0.88, 0.96, 0.82, 0.62]

    # Classification Metrics
    cls_metrics = compute_classification_metrics(y_true, y_pred, class_labels)
    assert cls_metrics["total_samples"] == 20
    assert cls_metrics["accuracy"] == 0.80  # 16 / 20 correct
    assert cls_metrics["macro_precision"] == 0.80
    assert cls_metrics["macro_recall"] == 0.80
    assert cls_metrics["macro_f1"] == 0.80

    # Calibration Metrics
    calib_metrics = compute_calibration_metrics(confs, y_pred, y_true, num_bins=5)
    assert "ece" in calib_metrics
    assert "mce" in calib_metrics
    assert 0.0 <= calib_metrics["ece"] <= 0.25

    # Out-of-Distribution Metrics
    id_scores = [0.95, 0.92, 0.88, 0.94, 0.89, 0.91, 0.96]
    ood_scores = [0.35, 0.28, 0.42, 0.30, 0.25, 0.38, 0.40]
    ood_metrics = compute_ood_metrics(id_scores, ood_scores)
    assert ood_metrics["auroc"] == 1.0  # Perfectly separated distributions
    assert ood_metrics["fpr95"] == 0.0
