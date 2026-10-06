"""
MODULE 4 — SCREENSHOT ANALYSIS
Dedicated analyzer detecting measurable characteristics of digital screen captures:
- Rectangular UI structures (cards, buttons, banners, chat bubbles)
- Text-heavy horizontal bands
- Repeated alignment patterns (vertical left-alignments)
- Sharp image boundaries
- Native display viewport dimension matching
Strict rule: The result must report "possible screenshot characteristics detected"
rather than automatically labeling evidence as manipulated.
"""
import io
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image

from app.schemas.common import FindingSeverity
from app.schemas.forensic import FindingItem, BaseAnalyzerResult

COMMON_SCREENSHOT_RESOLUTIONS = [
    # Mobile (Portrait)
    (1170, 2532, "Apple iPhone 12/13/14 Pro (19.5:9)"),
    (1179, 2556, "Apple iPhone 14/15/16 Pro (19.5:9)"),
    (1284, 2778, "Apple iPhone 12/13/14 Pro Max (19.5:9)"),
    (1290, 2796, "Apple iPhone 14/15/16 Pro Max (19.5:9)"),
    (1080, 2400, "Android Flagship FHD+ (20:9)"),
    (1080, 2340, "Android Flagship FHD+ (19.5:9)"),
    (1440, 3088, "Samsung Galaxy Ultra QHD+ (19.3:9)"),
    (1440, 3120, "Google Pixel Pro (19.5:9)"),
    (1080, 1920, "Standard Mobile FHD (16:9)"),
    (750, 1334, "Apple iPhone 6/7/8/SE (16:9)"),
    # Tablets
    (2048, 2732, "Apple iPad Pro 12.9 (4:3)"),
    (1668, 2388, "Apple iPad Pro 11 (4.3:3)"),
    # Desktop Displays
    (1920, 1080, "Desktop Full HD (16:9)"),
    (2560, 1440, "Desktop QHD (16:9)"),
    (3840, 2160, "Desktop 4K UHD (16:9)"),
    (2560, 1600, "MacBook Retina (16:10)"),
    (3024, 1964, "MacBook Pro 14 (15.4:10)"),
]


class ForensicScreenshotAnalyzer:
    """Measurable UI structure, layout, and display viewport analyzer."""

    @staticmethod
    def _is_standard_aspect_ratio(aspect_ratio: float) -> bool:
        """Checks if aspect ratio approximates standard display ratios (within 1.5% tolerance)."""
        standard_ratios = [
            16 / 9,      # 1.777
            9 / 16,      # 0.5625
            19.5 / 9,    # 2.166
            9 / 19.5,    # 0.4615
            20 / 9,      # 2.222
            9 / 20,      # 0.450
            16 / 10,     # 1.60
            10 / 16,     # 0.625
            4 / 3,       # 1.333
            3 / 4,       # 0.75
        ]
        return any(abs(aspect_ratio - target) / target < 0.015 for target in standard_ratios)

    def _match_viewport(self, width: int, height: int) -> Optional[str]:
        # 1. Exact orientation match
        for vw, vh, label in COMMON_SCREENSHOT_RESOLUTIONS:
            if width == vw and height == vh:
                return label
        # 2. Rotated / inverted orientation match
        for vw, vh, label in COMMON_SCREENSHOT_RESOLUTIONS:
            if width == vh and height == vw:
                return label
        return None

    def _detect_ui_rectangles(self, gray: np.ndarray) -> Tuple[int, List[Dict[str, int]], int]:
        """
        Uses edge detection and contour approximation to identify rectangular UI containers
        and measures repeated vertical alignment patterns.
        """
        h, w = gray.shape
        # Canny edge detection with sensible thresholds for UI borders
        edges = cv2.Canny(gray, 20, 80)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ui_rects = []
        left_alignments = []

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            x, y, rw, rh = cv2.boundingRect(cnt)
            # Filter reasonable UI component sizes (cards, bubbles, buttons, banners)
            if rw > 40 and rh > 20 and (rw < w * 0.99 or rh < h * 0.99):
                # Check if roughly rectangular (4 corners or high bounding box fill)
                area = cv2.contourArea(cnt)
                rect_area = rw * rh
                if rect_area > 0 and (len(approx) == 4 or area / rect_area > 0.6):
                    ui_rects.append({"x": int(x), "y": int(y), "w": int(rw), "h": int(rh)})
                    left_alignments.append(int(x))

        # Check for repeated vertical alignments (left-aligned cards or chat bubbles)
        repeated_alignments = 0
        if len(left_alignments) >= 3:
            # Bin into 8-pixel tolerance
            binned = [round(x / 8.0) * 8 for x in left_alignments]
            counts = {b: binned.count(b) for b in set(binned)}
            repeated_alignments = sum(1 for c in counts.values() if c >= 2)

        return len(ui_rects), ui_rects[:10], repeated_alignments

    def _estimate_text_density(self, gray: np.ndarray) -> Dict[str, Any]:
        """Estimates high-frequency edge density across horizontal scanlines."""
        h, w = gray.shape
        sobelx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
        row_density = np.mean(sobelx, axis=1)
        text_like_rows = np.sum(row_density > 10.0)
        text_density_ratio = round(float(text_like_rows) / max(h, 1), 3)

        return {
            "text_density_ratio": text_density_ratio,
            "text_like_rows_count": int(text_like_rows),
            "is_text_heavy": text_density_ratio > 0.20,
        }

    def analyze(self, data: bytes) -> BaseAnalyzerResult:
        findings: List[FindingItem] = []
        metrics: Dict[str, Any] = {}
        limitations: List[str] = [
            "Detecting screenshot characteristics indicates format provenance, NOT malicious manipulation.",
            "Cropped screenshots or partial screen captures may not match canonical device viewports.",
            "Documents, slide decks, and clean digital PDFs often exhibit UI-like rectangular structures without being device screenshots.",
        ]

        try:
            with Image.open(io.BytesIO(data)) as pil_img:
                width, height = pil_img.size
                aspect_ratio = round(width / max(height, 1), 4)

            nparr = np.frombuffer(data, np.uint8)
            cv_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if cv_img is None:
                raise ValueError("Could not decode image for screenshot inspection.")

            if len(cv_img.shape) == 2:
                gray = cv_img
            elif cv_img.shape[2] == 4:
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

            viewport_match = self._match_viewport(width, height)
            ui_count, sample_rects, repeated_aligns = self._detect_ui_rectangles(gray)
            text_stats = self._estimate_text_density(gray)

            # Determine composite screenshot score
            screenshot_signals = 0
            if viewport_match:
                screenshot_signals += 3
            if ui_count >= 2:
                screenshot_signals += 2
            if repeated_aligns > 0:
                screenshot_signals += 2
            if text_stats["is_text_heavy"]:
                screenshot_signals += 2

            is_probable_screenshot = (viewport_match is not None) or (screenshot_signals >= 3)

            metrics = {
                "dimensions": {"width": width, "height": height},
                "aspect_ratio": aspect_ratio,
                "aspect_ratio_standard": self._is_standard_aspect_ratio(aspect_ratio),
                "matched_viewport": viewport_match,
                "ui_rectangles_detected": ui_count,
                "repeated_alignments_count": repeated_aligns,
                "text_density": text_stats,
                "screenshot_indicator_score": screenshot_signals,
                "is_probable_screenshot": is_probable_screenshot,
            }

            if is_probable_screenshot:
                findings.append(
                    FindingItem(
                        finding_id="FIND-SCREEN-CHAR",
                        category="SCREENSHOT",
                        severity=FindingSeverity.INFO,
                        title="Possible Screenshot Characteristics Detected",
                        description=(
                            "Evidence exhibits measurable layout characteristics consistent with digital screen capture "
                            f"(UI containers: {ui_count}, text density: {text_stats['text_density_ratio']}, "
                            f"viewport: {viewport_match or 'Non-standard'})."
                        ),
                        evidence=metrics,
                        interpretation=(
                            "The structural composition (rectangular containers, alignment grids, and font density) "
                            "indicates this file was generated as a screenshot or digital UI export."
                        ),
                        limitation=(
                            "Screenshot characteristics describe how the image was captured, NOT whether it is fraudulent. "
                            "Authentic mobile receipts and legitimate chats are also screenshots."
                        ),
                        confidence=0.85,
                        is_anomaly=False,
                    )
                )

            return BaseAnalyzerResult(
                status="completed",
                findings=findings,
                metrics=metrics,
                limitations=limitations,
            )

        except Exception as e:
            return BaseAnalyzerResult(
                status="failed",
                findings=[],
                metrics={"error": str(e)},
                limitations=limitations,
            )


screenshot_analyzer = ForensicScreenshotAnalyzer()
