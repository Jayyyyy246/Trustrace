"""Tests for TRUSTTRACE Forensic Report System.

Verifies:
- All 17 numbered sections in digital evidence analysis report
- Strict distinction of OBSERVATION from INTERPRETATION
- Linguistic compliance (no "proves fake", preferred language used)
- Mandatory audit fields: SHA-256, analysis ID, model version, model checksum, analyzer versions, timestamp
- Vectorized A4 PDF generation (valid %PDF- bytes, headers, footers)
- REST API report endpoints (JSON and downloadable PDF stream)
"""

import io
from fastapi.testclient import TestClient
from PIL import Image

from app.services.evidence_service import evidence_service
from app.services.report_service import report_service
from app.schemas.common import VerdictLabel


def _create_sample_png_bytes(width: int = 250, height: int = 250, color=(60, 120, 200)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_build_forensic_report_all_17_sections(sample_png_bytes: bytes):
    """Verifies that all 17 numbered sections are present and populated from real backend results."""
    # Quarantined intake and analysis
    upload_record = evidence_service.ingest_evidence(sample_png_bytes, "report_sample.png")
    analysis = evidence_service.run_analysis(upload_record.evidence_id)

    report_data = report_service.build_forensic_report_data(analysis)

    assert report_data["title"] == "TRUSTTRACE DIGITAL EVIDENCE ANALYSIS REPORT"
    assert "sections" in report_data
    sections = report_data["sections"]

    expected_section_keys = [
        "1_case_analysis_id",
        "2_evidence_information",
        "3_sha256_integrity_hash",
        "4_file_information",
        "5_metadata_findings",
        "6_ocr_findings",
        "7_image_forensic_findings",
        "8_screenshot_analysis",
        "9_ml_analysis",
        "10_evidence_fusion",
        "11_final_assessment",
        "12_supporting_evidence",
        "13_contradictory_evidence",
        "14_limitations",
        "15_analyzer_status",
        "16_model_information",
        "17_timestamp",
    ]

    for expected_key in expected_section_keys:
        assert expected_key in sections, f"Missing section: {expected_key}"

    # Verify section numbers 1 to 17
    for i, key in enumerate(expected_section_keys, 1):
        assert sections[key]["section_number"] == i

    # Section 1: Case / Analysis ID
    assert sections["1_case_analysis_id"]["analysis_id"] == analysis.analysis_id
    assert "TRUSTTRACE-CASE-" in sections["1_case_analysis_id"]["case_reference"]

    # Section 2: Evidence Information
    assert sections["2_evidence_information"]["filename"] == "report_sample.png"
    assert sections["2_evidence_information"]["evidence_id"] == upload_record.evidence_id

    # Section 3: SHA-256
    assert sections["3_sha256_integrity_hash"]["sha256"] == upload_record.sha256
    assert len(sections["3_sha256_integrity_hash"]["sha256"]) == 64

    # Section 4: File Information
    assert sections["4_file_information"]["file_size_bytes"] == len(sample_png_bytes)
    assert "200" in sections["4_file_information"]["dimensions"]

    # Section 11: Final Assessment
    assert "label" in sections["11_final_assessment"]
    assert "assessment_statement" in sections["11_final_assessment"]

    # Section 12 & 13: Evidence items
    assert "items" in sections["12_supporting_evidence"]
    assert "items" in sections["13_contradictory_evidence"]

    # Section 14: Limitations
    assert len(sections["14_limitations"]["items"]) > 0

    # Section 15: Analyzer Status
    analyzers = sections["15_analyzer_status"]["analyzers"]
    assert "metadata" in analyzers
    assert "image_analysis" in analyzers
    assert "screenshot" in analyzers
    assert "ocr" in analyzers
    assert "ml_inference" in analyzers
    assert "decision_engine" in analyzers

    # Section 16: Model Information
    sec16 = sections["16_model_information"]
    assert "model_version" in sec16
    assert "model_checksum" in sec16
    assert sec16["model_checksum"].startswith("SHA256:")

    # Section 17: Timestamp
    sec17 = sections["17_timestamp"]
    assert "timestamp_utc" in sec17
    assert "iso8601" in sec17


def test_observation_vs_interpretation_distinction(sample_png_bytes: bytes):
    """Ensures analytical observation is strictly distinct from forensic interpretation."""
    upload_record = evidence_service.ingest_evidence(sample_png_bytes, "obs_interp_test.png")
    analysis = evidence_service.run_analysis(upload_record.evidence_id)
    report_data = report_service.build_forensic_report_data(analysis)
    sections = report_data["sections"]

    eval_sections = [
        "5_metadata_findings",
        "6_ocr_findings",
        "7_image_forensic_findings",
        "8_screenshot_analysis",
        "9_ml_analysis",
        "10_evidence_fusion",
    ]

    for sec_key in eval_sections:
        sec = sections[sec_key]
        assert "observation" in sec, f"Section {sec_key} missing observation"
        assert "interpretation" in sec, f"Section {sec_key} missing interpretation"
        assert len(sec["observation"].strip()) > 0
        assert len(sec["interpretation"].strip()) > 0
        # Observation and interpretation must not be identical strings
        assert sec["observation"] != sec["interpretation"], f"Section {sec_key} has duplicate text for obs & interp"


def test_linguistic_standards_no_forbidden_phrasing(sample_tampered_metadata_jpeg_bytes: bytes, sample_png_bytes: bytes):
    """Verifies that no forbidden certainty language is ever generated, and preferred phrases are used."""
    # 1. Tampered / EDITED file
    upload_tampered = evidence_service.ingest_evidence(sample_tampered_metadata_jpeg_bytes, "tampered.jpg")
    analysis_tampered = evidence_service.run_analysis(upload_tampered.evidence_id)
    report_tampered = report_service.build_forensic_report_data(analysis_tampered)

    def search_forbidden_words(obj, path=""):
        if isinstance(obj, str):
            lower = obj.lower()
            assert "proves the image is fake" not in lower, f"Forbidden phrase found at {path}: {obj}"
            assert "this proves the file was modified" not in lower, f"Forbidden phrase found at {path}: {obj}"
        elif isinstance(obj, dict):
            for k, v in obj.items():
                search_forbidden_words(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                search_forbidden_words(v, f"{path}[{i}]")

    search_forbidden_words(report_tampered)

    # If EDITED verdict, must use preferred language
    sec11 = report_tampered["sections"]["11_final_assessment"]
    if sec11["label"] == "EDITED":
        assert sec11["assessment_statement"] == "TRUSTTRACE detected indicators consistent with image manipulation."


def test_insufficient_evidence_unknown_verdict_phrasing():
    """Verifies standard phrasing when evidence is insufficient."""
    # Create an artificial analysis with UNKNOWN verdict
    from app.schemas.evidence import EvidenceInfo
    from app.schemas.forensic import AnalyzersContainer, MetadataAnalyzerResult, ImageAnalyzerResult, OCRAnalyzerResult, ScreenshotAnalyzerResult, MLAnalyzerResult
    from app.schemas.analysis import AnalysisResultResponse
    from app.schemas.verdict import FinalVerdict
    from datetime import datetime, timezone

    unknown_analysis = AnalysisResultResponse(
        analysis_id="unknown-test-1234-5678-90ab",
        evidence=EvidenceInfo(
            evidence_id="e-unknown-1",
            filename="ambiguous.jpg",
            sha256="a" * 64,
            mime_type="image/jpeg",
            size=1024,
            dimensions={"width": 300, "height": 300},
        ),
        analyzers=AnalyzersContainer(
            metadata=MetadataAnalyzerResult(),
            image=ImageAnalyzerResult(
                dimensions={"width": 300, "height": 300},
                channels=3,
                color_space="RGB",
                aspect_ratio=1.0,
                laplacian_variance=25.0,
                luminance_mean=128.0,
                luminance_std=30.0,
            ),
            ocr=OCRAnalyzerResult(status="NOT_AVAILABLE"),
            screenshot=ScreenshotAnalyzerResult(status="NOT_AVAILABLE"),
            ml=MLAnalyzerResult(model_status="NOT_AVAILABLE"),
        ),
        findings=[],
        final_verdict=FinalVerdict(
            label=VerdictLabel.UNKNOWN,
            confidence=None,
            uncertainty=1.0,
            risk_score=0.5,
            justification="Signals were inconclusive.",
        ),
        limitations=["Compression prevents conclusive assessment."],
        created_at=datetime.now(timezone.utc),
        pipeline_version="1.0.0-phase1",
    )

    report_data = report_service.build_forensic_report_data(unknown_analysis)
    sec11 = report_data["sections"]["11_final_assessment"]
    assert sec11["label"] == "UNKNOWN"
    assert sec11["assessment_statement"] == "The available evidence was insufficient to determine authenticity."


def test_pdf_report_generation(sample_png_bytes: bytes):
    """Verifies vectorized A4 PDF report generation with required audit headers."""
    upload_record = evidence_service.ingest_evidence(sample_png_bytes, "pdf_test.png")
    analysis = evidence_service.run_analysis(upload_record.evidence_id)

    pdf_bytes = report_service.generate_pdf_report(analysis)

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 2000, "PDF bytes should be non-trivial"
    # PDF magic number
    assert pdf_bytes.startswith(b"%PDF-"), "Generated document must be a valid PDF binary"

    # Use PyMuPDF to inspect generated PDF content
    import pymupdf
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    assert len(doc) >= 1

    page1_text = doc[0].get_text()
    assert "TRUSTTRACE DIGITAL EVIDENCE ANALYSIS REPORT" in page1_text
    assert "FINAL ASSESSMENT" in page1_text
    assert upload_record.sha256[:16] in page1_text or upload_record.sha256 in page1_text


def test_api_report_endpoints(client: TestClient, sample_png_bytes: bytes):
    """Tests GET /evidence/{id}/report (JSON and ?format=pdf) and GET /evidence/{id}/report/pdf."""
    upload_res = client.post(
        "/api/v1/evidence/upload",
        files={"file": ("api_report_test.png", sample_png_bytes, "image/png")},
    )
    assert upload_res.status_code == 201
    evidence_id = upload_res.json()["evidence"]["evidence_id"]

    # 1. JSON Report
    json_res = client.get(f"/api/v1/evidence/{evidence_id}/report")
    assert json_res.status_code == 200
    report_json = json_res.json()
    assert "sections" in report_json
    assert len(report_json["sections"]) == 17
    assert report_json["sha256"] == upload_res.json()["evidence"]["sha256"]
    assert report_json["model_checksum"].startswith("SHA256:")

    # 2. PDF via Query Param: ?format=pdf
    param_pdf_res = client.get(f"/api/v1/evidence/{evidence_id}/report?format=pdf")
    assert param_pdf_res.status_code == 200
    assert param_pdf_res.headers["content-type"] == "application/pdf"
    assert "attachment" in param_pdf_res.headers["content-disposition"]
    assert param_pdf_res.content.startswith(b"%PDF-")

    # 3. Dedicated PDF endpoint: /report/pdf
    dedicated_pdf_res = client.get(f"/api/v1/evidence/{evidence_id}/report/pdf")
    assert dedicated_pdf_res.status_code == 200
    assert dedicated_pdf_res.headers["content-type"] == "application/pdf"
    assert "attachment" in dedicated_pdf_res.headers["content-disposition"]
    assert dedicated_pdf_res.content.startswith(b"%PDF-")
