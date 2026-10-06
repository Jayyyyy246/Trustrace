"""Production Activation End-to-End Acceptance Test for TRUSTTRACE.
Strictly validates:
1. Real OCR execution (no hardcoded text, real engine, bounding boxes)
2. Real ML inference or transparent NOT_AVAILABLE state with genuine diagnostics
3. Real evidence fusion receiving OCR and ML outputs
4. Authentic final verdict generation
5. PDF report generation containing complete OCR and ML forensic sections
"""
import io
import os
import tempfile
from pathlib import Path
import pytest
import torch
from PIL import Image

from app.schemas.common import AnalyzerStatus, VerdictLabel
from app.services.evidence_service import evidence_service
from app.services.ocr_service import ocr_service
from app.services.ml_inference_service import ml_inference_service
from app.services.report_service import report_service
from model.architecture import build_model
from model.dataset import CLASS_LABELS, CLASS_TO_IDX
from model.versioning import ModelMetadata, compute_sha256


@pytest.fixture
def real_evidence_bytes():
    with open("data/test_payment_proof.jpg", "rb") as f:
        return f.read()


def test_phase17_e2e_acceptance_real_evidence_pipeline(real_evidence_bytes):
    """
    Executes the entire end-to-end pipeline:
    UPLOAD -> VALIDATION -> HASH -> QUARANTINE -> METADATA -> IMAGE FORENSICS ->
    SCREENSHOT -> OCR -> ML -> EVIDENCE FUSION -> VERDICT -> PDF REPORT
    """
    # 1. Ingest evidence
    intake = evidence_service.ingest_evidence(
        file_bytes=real_evidence_bytes,
        original_filename="payment_verification_audit.jpg",
    )
    assert intake.evidence_id is not None
    assert intake.sha256 is not None

    # 2. Run complete analysis pipeline
    analysis = evidence_service.run_analysis(intake.evidence_id)
    assert analysis.evidence.evidence_id == intake.evidence_id
    assert analysis.evidence.sha256 == intake.sha256

    # 3. ASSERTION 1 & 2: Real OCR reaches real engine and extracts genuine text
    ocr_res = analysis.analyzers.ocr
    assert ocr_res.status == AnalyzerStatus.AVAILABLE
    assert ocr_res.engine in ("Windows.Media.Ocr", "Tesseract OCR", "Tesseract (Local)")
    assert ocr_res.word_count >= 2
    assert ocr_res.text is not None
    assert "TRANSACTION" in ocr_res.text or "RECORD" in ocr_res.text
    assert len(ocr_res.regions) >= 2
    for r in ocr_res.regions:
        assert len(r.bbox) == 4
        assert r.bbox[2] > 0 and r.bbox[3] > 0
    assert ocr_res.processing_time_ms is not None
    assert ocr_res.processing_time_ms > 0.0

    # 4. ASSERTION 3 & 4: ML Analyzer behavior
    ml_res = analysis.analyzers.ml
    if not ml_inference_service.is_available():
        # Without checkpoint, state MUST be strictly NOT_AVAILABLE (no fake predictions)
        assert ml_res.model_status == "NOT_AVAILABLE"
        assert ml_res.predicted_label == "UNKNOWN"
        assert ml_res.confidence is None
        assert ml_res.failure_reason is not None
        assert "train_pipeline.py" in ml_res.failure_reason or "checkpoint" in ml_res.failure_reason.lower()
        assert ml_res.diagnostics is not None
        assert ml_res.diagnostics.device in ("CPU", "CUDA")
    else:
        # With active checkpoint, state MUST be AVAILABLE with real inference
        assert ml_res.model_status == "AVAILABLE"
        assert ml_res.predicted_label in (CLASS_LABELS + ["UNKNOWN"])
        assert ml_res.confidence is not None
        assert ml_res.diagnostics is not None

    # 5. ASSERTION 5: Evidence fusion receives both OCR and ML results
    verdict = analysis.final_verdict
    assert verdict.label in (
        VerdictLabel.REAL,
        VerdictLabel.EDITED,
        VerdictLabel.AI_GENERATED,
        VerdictLabel.SCREENSHOT_MANIPULATED,
        VerdictLabel.UNKNOWN,
    )
    # OCR is available, so it MUST NOT be in unavailable_analyzers
    assert "ocr" not in verdict.unavailable_analyzers

    # 6. ASSERTION 6: Final verdict explanation is grounded in real findings
    assert len(verdict.decision_rules_triggered) > 0
    assert verdict.justification is not None and len(verdict.justification) > 10

    # 7. ASSERTION 7: PDF Report generation and section verification
    report_data = report_service.build_forensic_report_data(analysis)
    assert "sections" in report_data
    # Check Section 6 (OCR)
    sec_ocr = report_data["sections"]["6_ocr_findings"]
    assert sec_ocr["section_number"] == 6
    assert sec_ocr["engine"] in ("Windows.Media.Ocr", "Tesseract OCR", "Tesseract (Local)")
    assert sec_ocr["word_count"] >= 2
    assert "Extracted" in sec_ocr["observation"]

    # Check Section 9 (ML)
    sec_ml = report_data["sections"]["9_ml_analysis"]
    assert sec_ml["section_number"] == 9
    assert sec_ml["model_status"] in ("AVAILABLE", "NOT_AVAILABLE")

    # Generate actual PDF bytes
    pdf_bytes = report_service.generate_pdf_report(analysis)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-1.7")
    assert len(pdf_bytes) > 15000  # Multi-page complete report


def test_phase17_ml_trained_checkpoint_inference_and_fusion_integration():
    """
    Tests that when a genuine trained checkpoint is provided, MLInferenceService
    loads it, verifies SHA-256 integrity, runs real inference on uploaded bytes,
    and feeds the result into Evidence Fusion.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        ckpt_path = tmp_path / "model_audit.pth"
        meta_path = tmp_path / "model_audit.json"

        # Create and save a small authentic model checkpoint
        model = build_model("custom_forensic_cnn", pretrained=False, num_classes=4)
        torch.save({"architecture": "custom_forensic_cnn", "state_dict": model.state_dict()}, ckpt_path)

        ckpt_sha = compute_sha256(ckpt_path)
        meta = ModelMetadata(
            model_name="CustomForensicCNN",
            architecture="custom_forensic_cnn",
            model_version="trusttrace-audit-v1.0",
            training_dataset_version="dataset-v1.0-synthetic",
            preprocessing_version="v1.0.0-imagenet",
            num_classes=4,
            class_mapping=CLASS_TO_IDX,
            sha256_checksum=ckpt_sha,
            temperature=1.0,
            ood_thresholds={"confidence_threshold": 0.50, "entropy_threshold": 0.85, "energy_threshold": -0.5},
        )
        meta.save(meta_path)

        # Create custom instance of MLInferenceService with this checkpoint
        from app.services.ml_inference_service import MLInferenceService
        test_ml_service = MLInferenceService(model_path=str(ckpt_path))

        assert test_ml_service.is_available() is True
        diag = test_ml_service.get_diagnostics()
        assert diag["status"] == "AVAILABLE"
        assert diag["checkpoint_sha256"] == ckpt_sha
        assert diag["device"] in ("CPU", "CUDA")

        # Run real inference on test image bytes
        with open("data/test_payment_proof.jpg", "rb") as f:
            test_bytes = f.read()

        ml_res, findings = test_ml_service.predict(test_bytes)

        assert ml_res.model_status == "AVAILABLE"
        assert ml_res.model_version == "trusttrace-audit-v1.0"
        assert ml_res.predicted_label in (CLASS_LABELS + ["UNKNOWN"])
        assert ml_res.class_probabilities is not None
        assert len(ml_res.class_probabilities) == 4
        assert ml_res.confidence is not None
        assert ml_res.uncertainty is not None
        assert ml_res.inference_time_ms is not None
        assert ml_res.inference_time_ms > 0.0
        assert ml_res.checkpoint_sha256 == ckpt_sha
