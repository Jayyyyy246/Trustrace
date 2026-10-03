"""
Tests for system health and root endpoints.
"""
from fastapi.testclient import TestClient


def test_root_endpoint(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "TRUSTTRACE"
    assert "version" in data
    assert data["documentation"] == "/docs"


def test_health_endpoint(client: TestClient):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["system"] == "TRUSTTRACE"
    assert data["storage"]["status"] == "online"
    # Ensure analyzers status reports properly
    assert data["analyzers"]["hashing"] == "ACTIVE"
    assert data["analyzers"]["metadata_exif"] == "ACTIVE"
    assert data["analyzers"]["image_cv"] == "ACTIVE"
    assert data["analyzers"]["compression_ela"] == "ACTIVE"
    assert data["analyzers"]["screenshot_geometry"] == "ACTIVE"
    # In Phase 1, ML must be NOT_AVAILABLE
    assert data["analyzers"]["ml_inference"] == "NOT_AVAILABLE"
