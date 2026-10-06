"""
End-to-end integration tests for Evidence API endpoints.
Validates the complete contract, schemas, error codes, and verdict behavior.
"""
from fastapi.testclient import TestClient


def test_upload_and_auto_analyze_valid_png(client: TestClient, sample_png_bytes: bytes):
    response = client.post(
        "/api/v1/evidence/upload",
        files={"file": ("test_capture.png", sample_png_bytes, "image/png")},
    )
    assert response.status_code == 201
    data = response.json()

    # 1. Verify top-level structure exactly as specified
    assert "analysis_id" in data
    assert "evidence" in data
    assert "analyzers" in data
    assert "findings" in data
    assert "final_verdict" in data
    assert "limitations" in data

    # 2. Verify evidence metadata
    evidence = data["evidence"]
    assert evidence["filename"].endswith(".png")
    assert len(evidence["sha256"]) == 64
    assert evidence["mime_type"] == "image/png"
    assert evidence["size"] == len(sample_png_bytes)
    assert evidence["dimensions"]["width"] == 200
    assert evidence["dimensions"]["height"] == 200

    # 3. Verify analyzers dictionary
    analyzers = data["analyzers"]
    assert "metadata" in analyzers
    assert "image" in analyzers
    assert "ocr" in analyzers
    assert "screenshot" in analyzers
    assert "ml" in analyzers

    # ML model status must be valid
    assert analyzers["ml"]["model_status"] in ("AVAILABLE", "NOT_AVAILABLE")

    # 4. Verify Final Verdict
    # For clean image with ML unavailable, verdict must be UNKNOWN with null confidence!
    verdict = data["final_verdict"]
    assert verdict["label"] == "UNKNOWN"
    assert verdict["confidence"] is None
    assert "risk_score" in verdict
    assert "justification" in verdict

    # 5. Verify Limitations
    assert len(data["limitations"]) > 0


def test_upload_tampered_metadata_verdict_edited(
    client: TestClient, sample_tampered_metadata_jpeg_bytes: bytes
):
    response = client.post(
        "/api/v1/evidence/upload",
        files={"file": ("photoshop_doc.jpg", sample_tampered_metadata_jpeg_bytes, "image/jpeg")},
    )
    assert response.status_code == 201
    data = response.json()

    # Must detect Photoshop tampering and yield EDITED verdict
    assert data["final_verdict"]["label"] == "EDITED"
    assert data["final_verdict"]["confidence"] is not None
    assert data["final_verdict"]["confidence"] > 0.8
    assert data["analyzers"]["metadata"]["editing_software_detected"] is True


def test_reject_invalid_spoofed_file(client: TestClient, invalid_text_file_bytes: bytes):
    response = client.post(
        "/api/v1/evidence/upload",
        files={"file": ("malicious.png", invalid_text_file_bytes, "image/png")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "supported" in data["detail"].lower() or "validation" in data["detail"].lower()


def test_get_evidence_and_analysis_by_id(client: TestClient, sample_png_bytes: bytes):
    # Upload with auto_analyze=False to get intake record
    upload_res = client.post(
        "/api/v1/evidence/upload?auto_analyze=false",
        files={"file": ("custody_item.png", sample_png_bytes, "image/png")},
    )
    assert upload_res.status_code == 201
    evidence_id = upload_res.json()["evidence_id"]

    # Query GET /evidence/{id}
    get_res = client.get(f"/api/v1/evidence/{evidence_id}")
    assert get_res.status_code == 200
    assert get_res.json()["evidence_id"] == evidence_id
    assert get_res.json()["sha256"] == upload_res.json()["sha256"]

    # Query GET /evidence/{id}/analysis
    analysis_res = client.get(f"/api/v1/evidence/{evidence_id}/analysis")
    assert analysis_res.status_code == 200
    analysis_data = analysis_res.json()
    assert analysis_data["evidence"]["sha256"] == upload_res.json()["sha256"]

    # Query GET /evidence/{id}/report
    report_res = client.get(f"/api/v1/evidence/{evidence_id}/report")
    assert report_res.status_code == 200
    report_data = report_res.json()
    assert "chain_of_custody" in report_data
    assert "verdict_assessment" in report_data


def test_get_non_existent_evidence_returns_404(client: TestClient):
    response = client.get("/api/v1/evidence/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
