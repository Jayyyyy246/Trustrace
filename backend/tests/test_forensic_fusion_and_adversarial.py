"""
Unit and Adversarial tests for Module 6 — Evidence Fusion.
Includes adversarial test cases with intentionally conflicting indicators.
"""
from forensic.fusion import fusion_engine
from app.schemas.forensic import BaseAnalyzerResult, FindingItem
from app.schemas.common import VerdictLabel, FindingSeverity


def test_fusion_rule_metadata_and_ela_corroboration():
    meta = BaseAnalyzerResult(
        metrics={"editing_software_detected": True, "software": "Adobe Photoshop 2026"},
        findings=[
            FindingItem(
                finding_id="F-1",
                category="METADATA",
                severity=FindingSeverity.HIGH,
                title="Editing tool",
                description="Photoshop detected",
                evidence="Photoshop",
                interpretation="Edited",
                limitation="May be innocent export",
            )
        ],
    )
    img = BaseAnalyzerResult(
        metrics={"ela": {"q90": {"variance": 140.0}}, "copy_move": {"detected": False}},
    )
    screen = BaseAnalyzerResult(metrics={})
    ocr = BaseAnalyzerResult(metrics={})

    verdict = fusion_engine.fuse(meta, img, screen, ocr)
    assert verdict.label == VerdictLabel.EDITED
    assert verdict.confidence > 0.8
    assert "RULE-FUSE-02-METADATA-TOOL-AND-ELA-CORROBORATION" in verdict.rules_triggered
    assert len(verdict.contributing_factors) >= 2


def test_fusion_clean_native_screenshot():
    meta = BaseAnalyzerResult(metrics={"editing_software_detected": False, "exif_present": False})
    img = BaseAnalyzerResult(metrics={"copy_move": {"detected": False}, "ela": {"q90": {"variance": 20.0}}})
    screen = BaseAnalyzerResult(metrics={"is_probable_screenshot": True, "matched_viewport": "iPhone 14 Pro"})
    ocr = BaseAnalyzerResult(metrics={"word_count": 15})
    ml = {"model_status": "AVAILABLE", "predicted_label": "REAL", "confidence": 0.85}

    verdict = fusion_engine.fuse(meta, img, screen, ocr, ml_prediction=ml)
    # Native unmanipulated screenshot corroborated by ML is classified as REAL
    assert verdict.label == VerdictLabel.REAL
    assert "RULE-FUSE-04-NATIVE-SCREENSHOT-STRUCTURE" in verdict.rules_triggered
    assert verdict.conflict_detected is False


# -------------------------------------------------------------------
# ADVERSARIAL TESTS (Intentionally Conflicting Indicators)
# -------------------------------------------------------------------

def test_adversarial_conflict_camera_exif_vs_copy_move():
    """
    Adversarial Case 1:
    Image preserves authentic Nikon D850 EXIF metadata,
    BUT pixels contain a mathematically verified copy-move cloning cluster.
    Engine must flag conflict and prioritize physical pixel evidence.
    """
    meta = BaseAnalyzerResult(
        metrics={
            "exif_present": True,
            "camera": {"make": "Nikon", "model": "D850"},
            "editing_software_detected": False,
        }
    )
    img = BaseAnalyzerResult(
        metrics={
            "copy_move": {"detected": True, "clusters": 2},
            "ela": {"q90": {"variance": 40.0}},
        }
    )
    screen = BaseAnalyzerResult(metrics={"is_probable_screenshot": False})
    ocr = BaseAnalyzerResult(metrics={})

    verdict = fusion_engine.fuse(meta, img, screen, ocr)
    assert verdict.label == VerdictLabel.EDITED
    assert verdict.conflict_detected is True
    assert "RULE-CONFLICT-CAMERA-EXIF-VS-CLONE" in verdict.rules_triggered
    assert "donor" in verdict.conflict_details.lower() or "conflict" in verdict.conflict_details.lower()


def test_adversarial_conflict_heavy_compression_vs_ela():
    """
    Adversarial Case 2:
    Image has elevated ELA variance (typically suspicious),
    BUT estimated JPEG quality is very low (quality 50).
    Heavy compression naturally generates high ELA variance.
    Engine must detect conflict and output UNKNOWN rather than false positive EDITED.
    """
    meta = BaseAnalyzerResult(metrics={"editing_software_detected": False})
    img = BaseAnalyzerResult(
        metrics={
            "copy_move": {"detected": False},
            "ela": {"q90": {"variance": 180.0}},
            "estimated_jpeg_quality": 50,  # Heavy lossy compression
        }
    )
    screen = BaseAnalyzerResult(metrics={})
    ocr = BaseAnalyzerResult(metrics={})

    verdict = fusion_engine.fuse(meta, img, screen, ocr)
    assert verdict.label == VerdictLabel.UNKNOWN
    assert verdict.confidence is None
    assert verdict.conflict_detected is True
    assert "RULE-CONFLICT-HEAVY-COMPRESSION-AMBIGUITY" in verdict.rules_triggered


def test_adversarial_conflict_ml_predicts_real_vs_physical_photoshop():
    """
    Adversarial Case 3:
    Machine learning model outputs REAL with high probability (0.92),
    BUT deterministic metadata detects Adobe Photoshop and severe noise ratio.
    Engine must refuse naive averaging, flag ML conflict, and output UNKNOWN.
    """
    meta = BaseAnalyzerResult(
        metrics={"editing_software_detected": True, "software": "Adobe Photoshop"}
    )
    img = BaseAnalyzerResult(
        metrics={
            "copy_move": {"detected": False},
            "noise_residual": {"tile_variance_ratio": 35.0},
        }
    )
    screen = BaseAnalyzerResult(metrics={})
    ocr = BaseAnalyzerResult(metrics={})
    ml = {
        "model_status": "AVAILABLE",
        "predicted_label": "REAL",
        "confidence": 0.92,
    }

    verdict = fusion_engine.fuse(meta, img, screen, ocr, ml_prediction=ml)
    assert verdict.label == VerdictLabel.UNKNOWN
    assert verdict.conflict_detected is True
    assert "RULE-CONFLICT-ML-VS-PHYSICAL-FORENSICS" in verdict.rules_triggered
    assert "UNKNOWN" in verdict.justification


def test_adversarial_inconclusive_flat_image():
    """
    Adversarial Case 4:
    Image is stripped of EXIF, has uniform texture, no active ML.
    Engine must return UNKNOWN without pretending a prediction exists.
    """
    meta = BaseAnalyzerResult(metrics={"exif_present": False, "editing_software_detected": False})
    img = BaseAnalyzerResult(
        metrics={"copy_move": {"detected": False}, "ela": {"q90": {"variance": 10.0}}}
    )
    screen = BaseAnalyzerResult(metrics={"is_probable_screenshot": False})
    ocr = BaseAnalyzerResult(metrics={})

    verdict = fusion_engine.fuse(meta, img, screen, ocr, ml_prediction={"model_status": "NOT_AVAILABLE"})
    assert verdict.label == VerdictLabel.UNKNOWN
    assert verdict.confidence is None
    assert "RULE-FUSE-07-INCONCLUSIVE-FORENSIC-SIGNAL" in verdict.rules_triggered
