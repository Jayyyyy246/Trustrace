"""
Unit tests for Module 4 — Screenshot Analysis.
"""
import io
from PIL import Image, ImageDraw
import pytest

from forensic.screenshot import screenshot_analyzer


@pytest.fixture
def simulated_ui_screenshot_bytes() -> bytes:
    """Creates an image with UI-like rectangular cards, buttons, and vertical alignment."""
    img = Image.new("RGB", (1080, 2400), color=(245, 247, 250))
    draw = ImageDraw.Draw(img)

    # Status bar simulation
    draw.rectangle([0, 0, 1080, 100], fill=(230, 233, 238))

    # Three left-aligned UI cards at x=60
    draw.rectangle([60, 200, 1020, 500], fill=(255, 255, 255), outline=(210, 215, 225))
    draw.rectangle([60, 550, 1020, 850], fill=(255, 255, 255), outline=(210, 215, 225))
    draw.rectangle([60, 900, 1020, 1200], fill=(255, 255, 255), outline=(210, 215, 225))

    # Horizontal text-like strokes
    for y in range(250, 450, 30):
        draw.line([(100, y), (800, y)], fill=(50, 50, 50), width=4)
    for y in range(600, 800, 30):
        draw.line([(100, y), (750, y)], fill=(50, 50, 50), width=4)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_screenshot_native_viewport(sample_iphone_screenshot_bytes: bytes):
    res = screenshot_analyzer.analyze(sample_iphone_screenshot_bytes)
    assert res.status == "completed"
    assert res.metrics["matched_viewport"] == "Apple iPhone 12/13/14 Pro (19.5:9)"
    assert res.metrics["is_probable_screenshot"] is True


def test_screenshot_ui_structures_and_neutral_finding(simulated_ui_screenshot_bytes: bytes):
    res = screenshot_analyzer.analyze(simulated_ui_screenshot_bytes)
    assert res.status == "completed"
    assert res.metrics["ui_rectangles_detected"] >= 3
    assert res.metrics["is_probable_screenshot"] is True

    # Verify the finding exists
    screen_f = [f for f in res.findings if f.finding_id == "FIND-SCREEN-CHAR"]
    assert len(screen_f) == 1
    f = screen_f[0]

    # Verify neutral tone (MUST NOT call image manipulated)
    assert "possible screenshot characteristics detected" in f.title.lower()
    assert f.severity.value == "INFO"
    assert "fraudulent" in f.limitation.lower() or "not whether it is" in f.limitation.lower()
    assert "authentic mobile receipts" in f.limitation.lower() or "legitimate" in f.limitation.lower()
