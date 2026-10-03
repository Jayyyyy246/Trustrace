"""
Unit tests for Module 1 — Forensic Hashing.
"""
from forensic.hashing import hashing_analyzer


def test_hashing_analyzer_structure(sample_png_bytes: bytes):
    res = hashing_analyzer.analyze(sample_png_bytes)
    assert res.status == "completed"
    assert "sha256" in res.metrics
    assert "sha512" in res.metrics
    assert "md5" in res.metrics
    assert len(res.metrics["sha256"]) == 64
    assert len(res.metrics["sha512"]) == 128
    assert len(res.metrics["md5"]) == 32
    assert len(res.findings) >= 1
    assert len(res.limitations) >= 1

    # Verify algorithm status reporting
    alg_names = [a["algorithm"] for a in res.algorithms]
    assert "SHA-256" in alg_names
    assert "SHA-512" in alg_names
    for a in res.algorithms:
        assert a["status"] == "COMPLETED"


def test_hashing_finding_structure(sample_jpeg_bytes: bytes):
    res = hashing_analyzer.analyze(sample_jpeg_bytes)
    f = res.findings[0]
    assert f.finding_id == "FIND-HASH-001"
    assert f.category == "HASHING"
    assert f.severity.value == "INFO"
    assert f.evidence is not None
    assert f.interpretation != ""
    assert "chain of custody" in f.interpretation.lower() or "custody" in f.interpretation.lower()
    assert "pre-ingestion" in f.limitation.lower() or "before" in f.limitation.lower()
