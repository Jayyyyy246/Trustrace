"""Unit and adversarial test suite for the TRUSTTRACE Evidence Decision Engine.

Verifies:
- All 7 input dimensions are integrated into the EvidenceGraph
- Output schema completeness (final_label, confidence, uncertainty, supporting, contradictory, unavailable, explanation, limitations)
- Explicit conflict resolution (ML AI-GENERATED vs hardware camera sensor evidence -> UNKNOWN)
- Every supported label: REAL, EDITED, AI-GENERATED, SCREENSHOT-MANIPULATED, UNKNOWN
- EvidenceSignal structure: source, metric, direction, reliability, explanation
- Traceability of structured explanation ('WHY TRUSTTRACE REACHED THIS ASSESSMENT')
- Non-fabricated explanations derived from actual analyzer outputs
- Internal Evidence Graph scoring and conflict edge inspection
- Input normalization across raw dicts, BaseAnalyzerResult, and FindingItem lists
- Handling of custom analyzer baseline reliability weightings
"""

import pytest

from app.schemas.common import AnalyzerStatus, FindingSeverity, VerdictLabel
from app.schemas.forensic import BaseAnalyzerResult, FindingItem, MetadataAnalyzerResult
from forensic.decision_engine import (
    ConflictEdgeItem,
    EvidenceDecisionEngine,
    EvidenceGraph,
    EvidenceSignalItem,
)


@pytest.fixture
def engine():
    return EvidenceDecisionEngine()


def test_decision_engine_output_schema_completeness(engine):
    """Verify that evaluate() returns a complete FinalVerdict containing all required fields."""
    metadata = {
        "editing_software_detected": True,
        "software": "Adobe Photoshop 2026",
        "exif_present": True,
        "camera": {"make": "Apple", "model": "iPhone 15 Pro"},
    }
    image_forensics = {
        "copy_move": {"detected": False, "clusters": 0},
        "ela": {"q90": {"variance": 145.0}},
        "noise_residual": {"tile_variance_ratio": 2.5},
        "estimated_jpeg_quality": 90,
    }
    screenshot = {"is_probable_screenshot": False}
    ocr = {"status": "COMPLETED", "word_count": 0}
    ml_prediction = {
        "model_status": "AVAILABLE",
        "predicted_label": "EDITED",
        "confidence": 0.88,
    }
    ml_uncertainty = 0.12
    analyzer_status = {
        "metadata": "COMPLETED",
        "image_analysis": "COMPLETED",
        "screenshot": "COMPLETED",
        "ocr": "COMPLETED",
        "ml_inference": "AVAILABLE",
    }

    verdict = engine.evaluate(
        metadata=metadata,
        image_forensics=image_forensics,
        screenshot=screenshot,
        ocr=ocr,
        ml_prediction=ml_prediction,
        ml_uncertainty=ml_uncertainty,
        analyzer_status=analyzer_status,
    )

    # 1. Output label and confidence/uncertainty
    assert verdict.label == VerdictLabel.EDITED
    assert verdict.confidence is not None
    assert 0.0 <= verdict.confidence <= 1.0
    assert verdict.uncertainty is not None
    assert 0.0 <= verdict.uncertainty <= 1.0

    # 2. Supporting and contradictory findings
    assert len(verdict.supporting_findings) > 0
    for s in verdict.supporting_findings:
        assert s.source in ("metadata", "image_analysis", "screenshot", "ocr", "ml_inference")
        assert s.metric
        assert s.direction in (
            VerdictLabel.REAL,
            VerdictLabel.EDITED,
            VerdictLabel.AI_GENERATED,
            VerdictLabel.SCREENSHOT_MANIPULATED,
            VerdictLabel.UNKNOWN,
        )
        assert 0.0 <= s.reliability <= 1.0
        assert s.explanation
        assert s.polarity == "SUPPORTING"

    # 3. Contradictory findings
    assert isinstance(verdict.contradictory_findings, list)

    # 4. Unavailable analyzers
    assert isinstance(verdict.unavailable_analyzers, list)

    # 5. Traceable Explanation block
    assert "WHY TRUSTTRACE REACHED THIS ASSESSMENT" in verdict.explanation
    assert "Evidence supporting assessment:" in verdict.explanation
    assert "Evidence against assessment:" in verdict.explanation
    assert "Limitations:" in verdict.explanation
    assert "✓" in verdict.explanation

    # 6. Limitations
    assert isinstance(verdict.limitations, list)
    assert len(verdict.limitations) > 0


def test_adversarial_conflict_ml_ai_generated_vs_camera_sensor(engine):
    """
    CRITICAL REQUIREMENT:
    If ML says AI-GENERATED with high probability but deterministic forensic evidence
    strongly contradicts it (e.g. authentic camera sensor EXIF, uniform Poisson noise,
    clean single-compression DQT), do NOT blindly return AI-GENERATED.
    Instead evaluate the conflict and return UNKNOWN with clear conflict documentation.
    """
    metadata = {
        "exif_present": True,
        "camera": {"make": "Sony", "model": "Alpha 7R V"},
        "editing_software_detected": False,
    }
    image_forensics = {
        "copy_move": {"detected": False, "clusters": 0},
        "ela": {"q90": {"variance": 35.0}},  # Clean uniform error level
        "noise_residual": {"tile_variance_ratio": 1.8},  # Physically natural Poisson-Gaussian sensor noise
        "estimated_jpeg_quality": 98,
    }
    screenshot = {"is_probable_screenshot": False}
    ocr = {"status": "COMPLETED", "word_count": 0}

    # Model aggressively claims AI-GENERATED
    ml_prediction = {
        "model_status": "AVAILABLE",
        "predicted_label": "AI-GENERATED",
        "confidence": 0.94,
    }
    ml_uncertainty = 0.06

    verdict = engine.evaluate(
        metadata=metadata,
        image_forensics=image_forensics,
        screenshot=screenshot,
        ocr=ocr,
        ml_prediction=ml_prediction,
        ml_uncertainty=ml_uncertainty,
    )

    # MUST NOT blindly adopt AI-GENERATED
    assert verdict.label == VerdictLabel.UNKNOWN
    assert verdict.confidence is None  # Inconclusive due to adversarial conflict
    assert verdict.conflict_detected is True
    assert "RULE-CONFLICT-ML-AI-VS-CAMERA-SENSOR" in verdict.decision_rules_triggered
    assert "Sony" in verdict.conflict_details

    # Explanation must document the conflict and explain why UNKNOWN was reached
    assert "contradict synthetic generation" in verdict.conflict_details
    assert any("Sony Alpha 7R V" in s.explanation for s in verdict.supporting_findings + verdict.contradictory_findings)


def test_adversarial_conflict_camera_exif_vs_clone_donor(engine):
    """
    Conflict: Authentic camera metadata is present, but pixel copy-move cloning is detected.
    Engine must detect metadata donor retention and label EDITED with conflict recorded.
    """
    metadata = {
        "exif_present": True,
        "camera": {"make": "Canon", "model": "EOS R5"},
        "editing_software_detected": False,
    }
    image_forensics = {
        "copy_move": {"detected": True, "clusters": 2},
        "ela": {"q90": {"variance": 45.0}},
    }
    screenshot = {"is_probable_screenshot": False}
    ocr = {"status": "COMPLETED", "word_count": 0}

    verdict = engine.evaluate(
        metadata=metadata,
        image_forensics=image_forensics,
        screenshot=screenshot,
        ocr=ocr,
    )

    assert verdict.label == VerdictLabel.EDITED
    assert verdict.conflict_detected is True
    assert "RULE-CONFLICT-CAMERA-EXIF-VS-CLONE" in verdict.decision_rules_triggered
    assert "donor" in verdict.conflict_details.lower() or "conflict" in verdict.conflict_details.lower()
    # Camera EXIF appears in contradictory findings (evidence opposing editing)
    assert any("Canon EOS R5" in s.explanation for s in verdict.contradictory_findings)


def test_adversarial_conflict_ml_real_vs_photoshop(engine):
    """
    Conflict: ML model predicts REAL with high probability (0.95),
    BUT container records Adobe Photoshop software tag.
    Engine must refuse naive averaging, flag ML conflict, and output UNKNOWN.
    """
    metadata = {
        "editing_software_detected": True,
        "software": "Adobe Photoshop 2026",
    }
    image_forensics = {
        "copy_move": {"detected": False},
        "noise_residual": {"tile_variance_ratio": 32.0},
    }
    screenshot = {"is_probable_screenshot": False}
    ocr = {"status": "COMPLETED", "word_count": 0}
    ml_prediction = {
        "model_status": "AVAILABLE",
        "predicted_label": "REAL",
        "confidence": 0.95,
    }

    verdict = engine.evaluate(
        metadata=metadata,
        image_forensics=image_forensics,
        screenshot=screenshot,
        ocr=ocr,
        ml_prediction=ml_prediction,
    )

    assert verdict.label == VerdictLabel.UNKNOWN
    assert verdict.confidence is None
    assert verdict.conflict_detected is True
    assert "RULE-CONFLICT-ML-VS-PHYSICAL-FORENSICS" in verdict.decision_rules_triggered


def test_adversarial_conflict_heavy_compression_vs_ela(engine):
    """
    Conflict: High ELA variance coincides with heavy lossy compression (Quality 55).
    Engine must recognize compression false-positive risk and output UNKNOWN.
    """
    metadata = {"editing_software_detected": False}
    image_forensics = {
        "copy_move": {"detected": False},
        "ela": {"q90": {"variance": 160.0}},
        "estimated_jpeg_quality": 55,
    }
    screenshot = {"is_probable_screenshot": False}
    ocr = {"status": "COMPLETED", "word_count": 0}

    verdict = engine.evaluate(
        metadata=metadata,
        image_forensics=image_forensics,
        screenshot=screenshot,
        ocr=ocr,
    )

    assert verdict.label == VerdictLabel.UNKNOWN
    assert verdict.confidence is None
    assert verdict.conflict_detected is True
    assert "RULE-CONFLICT-HEAVY-COMPRESSION-AMBIGUITY" in verdict.decision_rules_triggered


def test_label_real_pristine_camera_photo(engine):
    """Verify REAL label when hardware camera metadata and pixel uniformity are intact."""
    metadata = {
        "exif_present": True,
        "camera": {"make": "Nikon", "model": "Z9"},
        "editing_software_detected": False,
    }
    image_forensics = {
        "copy_move": {"detected": False, "clusters": 0},
        "ela": {"q90": {"variance": 22.0}},
        "noise_residual": {"tile_variance_ratio": 1.2},
    }
    screenshot = {"is_probable_screenshot": False}
    ocr = {"status": "COMPLETED", "word_count": 0}
    ml_prediction = {
        "model_status": "AVAILABLE",
        "predicted_label": "REAL",
        "confidence": 0.88,
    }

    verdict = engine.evaluate(
        metadata=metadata,
        image_forensics=image_forensics,
        screenshot=screenshot,
        ocr=ocr,
        ml_prediction=ml_prediction,
    )

    assert verdict.label == VerdictLabel.REAL
    assert verdict.confidence >= 0.80
    assert verdict.conflict_detected is False
    assert "RULE-FUSE-06-PRISTINE-CAMERA-CONTAINER" in verdict.decision_rules_triggered


def test_label_edited_copy_move_cloning(engine):
    """Verify EDITED label when copy-move geometric keypoint matching detects cloned regions."""
    metadata = {"exif_present": False, "editing_software_detected": False}
    image_forensics = {
        "copy_move": {"detected": True, "clusters": 3},
        "ela": {"q90": {"variance": 110.0}},
    }
    screenshot = {"is_probable_screenshot": False}
    ocr = {"status": "COMPLETED", "word_count": 0}

    verdict = engine.evaluate(
        metadata=metadata,
        image_forensics=image_forensics,
        screenshot=screenshot,
        ocr=ocr,
    )

    assert verdict.label == VerdictLabel.EDITED
    assert verdict.confidence >= 0.85
    assert verdict.risk_score >= 0.85
    assert "RULE-IMG-01-COPY-MOVE-CLONING" in verdict.decision_rules_triggered


def test_label_ai_generated_corroborated(engine):
    """Verify AI-GENERATED label when ML model identifies generative artifacts and EXIF is absent."""
    metadata = {"exif_present": False, "editing_software_detected": False}
    image_forensics = {
        "copy_move": {"detected": False, "clusters": 0},
        "ela": {"q90": {"variance": 40.0}},
    }
    screenshot = {"is_probable_screenshot": False}
    ocr = {"status": "COMPLETED", "word_count": 0}
    ml_prediction = {
        "model_status": "AVAILABLE",
        "predicted_label": "AI-GENERATED",
        "confidence": 0.91,
    }

    verdict = engine.evaluate(
        metadata=metadata,
        image_forensics=image_forensics,
        screenshot=screenshot,
        ocr=ocr,
        ml_prediction=ml_prediction,
    )

    assert verdict.label == VerdictLabel.AI_GENERATED
    assert verdict.confidence >= 0.80
    assert "RULE-AI-01-SYNTHETIC-GENERATION-CORROBORATED" in verdict.decision_rules_triggered


def test_label_screenshot_manipulated_typography_anomaly(engine):
    """Verify SCREENSHOT-MANIPULATED when digital screenshot displays inconsistent typography kerning."""
    metadata = {"exif_present": False, "editing_software_detected": False}
    image_forensics = {"copy_move": {"detected": False}, "ela": {"q90": {"variance": 30.0}}}
    screenshot = {"is_probable_screenshot": True, "matched_viewport": "iPhone 14 Pro"}
    ocr = {"status": "COMPLETED", "word_count": 25, "font_anomaly_detected": True}

    verdict = engine.evaluate(
        metadata=metadata,
        image_forensics=image_forensics,
        screenshot=screenshot,
        ocr=ocr,
    )

    assert verdict.label == VerdictLabel.SCREENSHOT_MANIPULATED
    assert "RULE-SCREEN-02-MANIPULATED-TYPOGRAPHY" in verdict.decision_rules_triggered
    assert any(s.direction == VerdictLabel.SCREENSHOT_MANIPULATED for s in verdict.supporting_findings)


def test_unavailable_analyzers_tracking(engine):
    """Verify that unavailable analyzers (e.g. OCR and ML) are explicitly listed in unavailable_analyzers."""
    metadata = {"exif_present": False, "editing_software_detected": False}
    image_forensics = {"copy_move": {"detected": False}, "ela": {"q90": {"variance": 30.0}}}
    screenshot = {"is_probable_screenshot": False}
    ocr = {"status": "NOT_AVAILABLE", "word_count": 0}
    ml_prediction = {"model_status": "NOT_AVAILABLE"}

    verdict = engine.evaluate(
        metadata=metadata,
        image_forensics=image_forensics,
        screenshot=screenshot,
        ocr=ocr,
        ml_prediction=ml_prediction,
        analyzer_status={"ocr": "NOT_AVAILABLE", "ml_inference": "NOT_AVAILABLE"},
    )

    assert verdict.label == VerdictLabel.UNKNOWN
    assert "ocr" in verdict.unavailable_analyzers
    assert "ml_inference" in verdict.unavailable_analyzers
    assert any("NOT_AVAILABLE" in lim for lim in verdict.limitations)


def test_evidence_graph_internals_and_conflict_edges(engine):
    """Verify that EvidenceGraph computes scores, stores signals, and tracks conflict edges."""
    graph = EvidenceGraph()
    graph.register_source("metadata", "COMPLETED", baseline_reliability=0.95)
    graph.register_source("ml_inference", "AVAILABLE", baseline_reliability=0.85)

    sig1 = EvidenceSignalItem(
        source="metadata",
        metric="camera_sensor_exif",
        direction=VerdictLabel.REAL,
        reliability=0.90,
        weight=1.2,
        explanation="Authentic camera sensor EXIF",
    )
    sig2 = EvidenceSignalItem(
        source="ml_inference",
        metric="deep_learning_artifact_classification",
        direction=VerdictLabel.AI_GENERATED,
        reliability=0.88,
        weight=1.3,
        explanation="Model predicted AI-GENERATED",
    )
    graph.add_signal(sig1)
    graph.add_signal(sig2)
    graph.add_conflict(
        sig1=sig1,
        sig2=sig2,
        rule_id="RULE-CONFLICT-ML-AI-VS-CAMERA-SENSOR",
        conflict_type="ADVERSARIAL_CONTRADICTION",
        explanation="EXIF camera hardware contradicts synthetic ML prediction",
    )

    scores = graph.compute_hypothesis_scores()
    assert scores[VerdictLabel.REAL]["support"] > 0
    assert scores[VerdictLabel.AI_GENERATED]["support"] > 0
    # Both hypotheses encounter friction/opposition from each other
    assert scores[VerdictLabel.REAL]["opposition"] > 0
    assert scores[VerdictLabel.AI_GENERATED]["opposition"] > 0

    conflicts = graph.get_active_conflicts()
    assert len(conflicts) == 1
    assert conflicts[0].rule_id == "RULE-CONFLICT-ML-AI-VS-CAMERA-SENSOR"
    assert conflicts[0].source_signal == sig1
    assert conflicts[0].contradicting_signal == sig2


def test_input_flexibility_pydantic_and_findings_lists(engine):
    """Verify that evaluate() seamlessly accepts BaseAnalyzerResult objects and FindingItem lists."""
    meta_result = BaseAnalyzerResult(
        status=AnalyzerStatus.COMPLETED,
        metrics={"editing_software_detected": True, "software": "Adobe Photoshop 2026"},
        findings=[
            FindingItem(
                finding_id="F-META-01",
                analyzer="metadata",
                category="METADATA",
                severity=FindingSeverity.HIGH,
                title="Editing software tag",
                description="Photoshop tag present",
                evidence="Adobe Photoshop 2026",
            )
        ],
    )
    img_findings = [
        FindingItem(
            finding_id="F-IMG-01",
            analyzer="image_analysis",
            category="CLONE",
            severity=FindingSeverity.CRITICAL,
            title="Copy move detected",
            description="2 keypoint clone clusters",
            evidence={"clusters": 2},
        )
    ]
    screenshot = BaseAnalyzerResult(status=AnalyzerStatus.COMPLETED, metrics={"is_probable_screenshot": False})
    ocr = BaseAnalyzerResult(status=AnalyzerStatus.COMPLETED, metrics={"word_count": 0})

    verdict = engine.evaluate(
        metadata=meta_result,
        image_forensics=img_findings,
        screenshot=screenshot,
        ocr=ocr,
    )

    assert verdict.label == VerdictLabel.EDITED
    assert verdict.confidence is not None
    assert any("copy_move" in s.metric or "cloning" in s.explanation.lower() for s in verdict.supporting_findings)


def test_custom_analyzer_reliability_weighting(engine):
    """Verify that analyzer_status can pass custom reliability weights for each analyzer."""
    metadata = {
        "editing_software_detected": True,
        "software": "Canva",
    }
    image_forensics = {"ela": {"q90": {"variance": 20.0}}}
    screenshot = {"is_probable_screenshot": False}
    ocr = {"word_count": 0}

    # Pass dict with custom reliability
    analyzer_status = {
        "metadata": {"status": "COMPLETED", "reliability": 0.98},
        "image_analysis": {"status": "COMPLETED", "reliability": 0.95},
        "screenshot": 0.90,
        "ocr": 0.85,
        "ml_inference": "NOT_AVAILABLE",
    }

    verdict = engine.evaluate(
        metadata=metadata,
        image_forensics=image_forensics,
        screenshot=screenshot,
        ocr=ocr,
        analyzer_status=analyzer_status,
    )

    assert verdict.label == VerdictLabel.EDITED
    assert "ml_inference" in verdict.unavailable_analyzers
    assert "metadata" not in verdict.unavailable_analyzers


def test_traceable_explanation_format_and_non_fabrication(engine):
    """
    Verify that the explanation follows the exact format:
    WHY TRUSTTRACE REACHED THIS ASSESSMENT
    Evidence supporting assessment:
    ✓ ...
    Evidence against assessment:
    ⚠ ...
    Limitations:
    • ...
    And contains no fabricated text.
    """
    metadata = {
        "editing_software_detected": True,
        "software": "Adobe Photoshop 2026",
        "exif_present": False,
    }
    image_forensics = {
        "ela": {"q90": {"variance": 140.0}},
        "estimated_jpeg_quality": 60,
    }
    screenshot = {"is_probable_screenshot": False}
    ocr = {"word_count": 0}

    verdict = engine.evaluate(
        metadata=metadata,
        image_forensics=image_forensics,
        screenshot=screenshot,
        ocr=ocr,
    )

    exp = verdict.explanation
    assert exp.startswith("WHY TRUSTTRACE REACHED THIS ASSESSMENT")
    assert "Evidence supporting assessment:" in exp
    assert "Evidence against assessment:" in exp
    assert "Limitations:" in exp

    # Check actual bullet markers
    assert "✓ Container software tag explicitly records digital editing tool ('Adobe Photoshop 2026')." in exp
    assert any("⚠" in line for line in exp.split("\n"))
    assert any("•" in line for line in exp.split("\n"))
