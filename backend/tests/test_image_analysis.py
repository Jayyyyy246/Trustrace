"""
Tests for OpenCV and computer vision image analysis.
"""
from app.services.image_analysis_service import image_analysis_service
from app.schemas.common import AnalyzerStatus


def test_image_dimensions_and_channels(sample_png_bytes: bytes):
    result, findings = image_analysis_service.analyze_image(sample_png_bytes)
    assert result.status == AnalyzerStatus.COMPLETED
    assert result.dimensions["width"] == 200
    assert result.dimensions["height"] == 200
    assert result.channels == 3
    assert result.aspect_ratio == 1.0
    # Flat color has zero edge gradients
    assert result.laplacian_variance >= 0.0
    assert result.is_blurry is True
    assert result.ela_computed is True


def test_image_edge_gradient_variance(sample_jpeg_bytes: bytes):
    result, findings = image_analysis_service.analyze_image(sample_jpeg_bytes)
    assert result.status == AnalyzerStatus.COMPLETED
    assert result.laplacian_variance >= 0.0


def test_image_ela_computation(sample_jpeg_bytes: bytes):
    result, findings = image_analysis_service.analyze_image(sample_jpeg_bytes)
    assert result.status == AnalyzerStatus.COMPLETED
    assert result.dimensions["width"] == 300
    assert result.dimensions["height"] == 400
    assert result.ela_computed is True
    assert result.ela_mean_delta is not None
    assert result.ela_variance is not None
