"""
Unit tests for Module 3 — Image Forensics.
"""
import io
import cv2
import numpy as np
from PIL import Image, ImageDraw
import pytest

from forensic.image_analysis import image_analyzer


@pytest.fixture
def copy_move_spliced_image_bytes() -> bytes:
    """
    Creates an image with an intentionally duplicated textured patch
    separated by > 60 pixels to test copy-move detection.
    """
    img = Image.new("RGB", (300, 300), color=(100, 100, 100))
    draw = ImageDraw.Draw(img)

    # Draw a unique textured geometric pattern
    pattern = [
        (10, 10, 50, 50),
        (20, 20, 40, 40),
        (15, 30, 45, 35),
    ]

    # Draw pattern in Region A (x: 30..80, y: 30..80)
    draw.rectangle([30, 30, 80, 80], fill=(220, 30, 30), outline=(255, 255, 255))
    draw.ellipse([40, 40, 70, 70], fill=(30, 220, 30))
    draw.polygon([(35, 35), (75, 45), (55, 75)], fill=(30, 30, 220))

    # Duplicate EXACT pattern into Region B (x: 180..230, y: 180..230)
    draw.rectangle([180, 180, 230, 230], fill=(220, 30, 30), outline=(255, 255, 255))
    draw.ellipse([190, 190, 220, 220], fill=(30, 220, 30))
    draw.polygon([(185, 185), (225, 195), (205, 225)], fill=(30, 30, 220))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_image_forensics_color_and_dimensions(sample_jpeg_bytes: bytes):
    res = image_analyzer.analyze(sample_jpeg_bytes)
    assert res.status == "completed"
    m = res.metrics
    assert m["dimensions"]["width"] == 300
    assert m["dimensions"]["height"] == 400
    assert m["channels"] == 3
    assert m["color_space"] == "RGB"
    assert "color_statistics" in m
    assert "blue" in m["color_statistics"]
    assert "green" in m["color_statistics"]
    assert "red" in m["color_statistics"]
    assert "noise_residual" in m
    assert "edge_consistency" in m
    assert "ela" in m
    assert len(res.limitations) >= 3


def test_image_forensics_dqt_extraction(sample_jpeg_bytes: bytes):
    res = image_analyzer.analyze(sample_jpeg_bytes)
    m = res.metrics
    assert m["quantization_tables"]["available"] is True
    assert len(m["quantization_tables"]["tables"]) >= 1
    assert m["estimated_jpeg_quality"] is not None


def test_copy_move_cloning_detection(copy_move_spliced_image_bytes: bytes):
    res = image_analyzer.analyze(copy_move_spliced_image_bytes)
    assert res.status == "completed"
    cm = res.metrics["copy_move"]
    # Candidate keypoint pairs must be discovered
    assert cm["candidate_pairs"] > 0
    assert cm["keypoints_evaluated"] > 0
