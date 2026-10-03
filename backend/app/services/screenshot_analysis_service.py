"""
Screenshot and display geometry validation service.
Compares evidence dimensions and aspect ratios against canonical viewport profiles.
"""
from typing import Tuple, List, Optional, Dict
from app.schemas.common import AnalyzerStatus, FindingSeverity
from app.schemas.forensic import ScreenshotAnalyzerResult, FindingItem


# Canonical display resolutions: (width, height, label)
COMMON_VIEWPORTS = [
    # Mobile (Portrait)
    (1170, 2532, "Apple iPhone 12/13/14 Pro (19.5:9)"),
    (1179, 2556, "Apple iPhone 14/15/16 Pro (19.5:9)"),
    (1284, 2778, "Apple iPhone 12/13/14 Pro Max (19.5:9)"),
    (1290, 2796, "Apple iPhone 14/15/16 Pro Max (19.5:9)"),
    (1080, 2400, "Android Flagship FHD+ (20:9)"),
    (1080, 2340, "Android Flagship FHD+ (19.5:9)"),
    (1440, 3088, "Samsung Galaxy Ultra QHD+ (19.3:9)"),
    (1440, 3120, "Google Pixel Pro (19.5:9)"),
    (1080, 1920, "Standard Mobile FHD Portrait (16:9)"),
    (750, 1334, "Apple iPhone 6/7/8/SE (16:9)"),
    # Tablets
    (2048, 2732, "Apple iPad Pro 12.9 (4:3)"),
    (1668, 2388, "Apple iPad Pro 11 (4.3:3)"),
    # Desktop Displays (Landscape)
    (1920, 1080, "Desktop Full HD (16:9)"),
    (2560, 1440, "Desktop QHD (16:9)"),
    (3840, 2160, "Desktop 4K UHD (16:9)"),
    (2560, 1600, "MacBook Retina (16:10)"),
    (3024, 1964, "MacBook Pro 14 (15.4:10)"),
]


class ScreenshotAnalysisService:
    @staticmethod
    def _find_matching_viewport(width: int, height: int) -> Optional[str]:
        """Matches pixel dimensions against known device display profiles."""
        # 1. Exact orientation match
        for vw, vh, label in COMMON_VIEWPORTS:
            if width == vw and height == vh:
                return label
        # 2. Rotated / inverted orientation match
        for vw, vh, label in COMMON_VIEWPORTS:
            if width == vh and height == vw:
                return label
        return None

    @staticmethod
    def _is_standard_aspect_ratio(aspect_ratio: float) -> bool:
        """Checks if aspect ratio approximates standard display ratios (within 1% tolerance)."""
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

    def analyze_screenshot(
        self, width: int, height: int, aspect_ratio: float
    ) -> Tuple[ScreenshotAnalyzerResult, List[FindingItem]]:
        """
        Evaluates structural viewport consistency.
        Returns ScreenshotAnalyzerResult and forensic findings.
        """
        findings: List[FindingItem] = []
        indicators: List[str] = []

        matched_viewport = self._find_matching_viewport(width, height)
        is_common = matched_viewport is not None
        is_standard_ratio = self._is_standard_aspect_ratio(aspect_ratio)

        if is_common:
            indicators.append(f"Dimensions ({width}x{height}) exactly match profile: {matched_viewport}")
            findings.append(
                FindingItem(
                    finding_id="FIND-SCREEN-001",
                    analyzer="ScreenshotAnalyzer",
                    title="Exact Native Viewport Match",
                    description=f"Evidence resolution matches standard digital device display: {matched_viewport}.",
                    severity=FindingSeverity.INFO,
                    confidence=0.90,
                    technical_details={
                        "width": width,
                        "height": height,
                        "matched_profile": matched_viewport,
                    },
                    is_anomaly=False,
                )
            )
        elif is_standard_ratio:
            indicators.append(f"Aspect ratio ({aspect_ratio}) corresponds to a standard display viewport.")
        else:
            indicators.append(f"Non-standard aspect ratio ({aspect_ratio}); image may be cropped or scaled.")
            findings.append(
                FindingItem(
                    finding_id="FIND-SCREEN-002",
                    analyzer="ScreenshotAnalyzer",
                    title="Non-Standard Display Aspect Ratio",
                    description=(
                        f"Image aspect ratio {aspect_ratio} deviates from standard mobile/desktop display ratios. "
                        "Indicates manual cropping, resizing, or partial UI capture."
                    ),
                    severity=FindingSeverity.LOW,
                    confidence=0.65,
                    technical_details={"aspect_ratio": aspect_ratio, "dimensions": f"{width}x{height}"},
                    is_anomaly=True,
                )
            )

        result = ScreenshotAnalyzerResult(
            status=AnalyzerStatus.COMPLETED,
            is_common_viewport=is_common,
            matched_viewport=matched_viewport,
            aspect_ratio_standard=is_standard_ratio,
            indicators=indicators,
        )
        return result, findings


screenshot_analysis_service = ScreenshotAnalysisService()
