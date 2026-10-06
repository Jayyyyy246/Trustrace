"""
Screenshot and display geometry validation service.
Compares evidence dimensions and aspect ratios against canonical viewport profiles.
"""
from typing import Tuple, List, Optional
from app.schemas.common import AnalyzerStatus, FindingSeverity
from app.schemas.forensic import ScreenshotAnalyzerResult, FindingItem
from forensic.screenshot import screenshot_analyzer, COMMON_SCREENSHOT_RESOLUTIONS

# Backward compatibility alias
COMMON_VIEWPORTS = COMMON_SCREENSHOT_RESOLUTIONS


class ScreenshotAnalysisService:
    @staticmethod
    def _find_matching_viewport(width: int, height: int) -> Optional[str]:
        """Matches pixel dimensions against known device display profiles."""
        return screenshot_analyzer._match_viewport(width, height)

    @staticmethod
    def _is_standard_aspect_ratio(aspect_ratio: float) -> bool:
        """Checks if aspect ratio approximates standard display ratios (within 1.5% tolerance)."""
        return screenshot_analyzer._is_standard_aspect_ratio(aspect_ratio)

    def analyze_screenshot(
        self,
        data: Optional[bytes] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        aspect_ratio: Optional[float] = None,
    ) -> Tuple[ScreenshotAnalyzerResult, List[FindingItem]]:
        """
        Evaluates structural viewport consistency and pixel-level screenshot characteristics
        (UI rectangle detection, horizontal text scanline bands, vertical alignment grids)
        via canonical ForensicScreenshotAnalyzer.
        """
        # If real image bytes are passed, execute full pixel and layout analysis
        if data is not None and len(data) > 0:
            base_res = screenshot_analyzer.analyze(data)
            if base_res.status == "failed":
                err = base_res.metrics.get("error", "Screenshot decoding failure")
                return (
                    ScreenshotAnalyzerResult(
                        status=AnalyzerStatus.FAILED,
                        is_common_viewport=False,
                        matched_viewport=None,
                        aspect_ratio_standard=False,
                        indicators=[f"Analysis failed: {err}"],
                    ),
                    base_res.findings,
                )

            m = base_res.metrics
            matched_viewport = m.get("matched_viewport")
            is_common = matched_viewport is not None
            aspect_ratio_val = m.get("aspect_ratio", aspect_ratio or 1.0)
            is_standard_ratio = m.get("aspect_ratio_standard", self._is_standard_aspect_ratio(aspect_ratio_val))
            ui_count = m.get("ui_rectangles_detected", 0)
            repeated_aligns = m.get("repeated_alignments_count", 0)
            text_density = m.get("text_density", {})
            text_ratio = text_density.get("text_density_ratio", 0.0)

            indicators: List[str] = []
            if is_common:
                indicators.append(f"Dimensions ({m['dimensions']['width']}x{m['dimensions']['height']}) exactly match profile: {matched_viewport}")
            elif is_standard_ratio:
                indicators.append(f"Aspect ratio ({aspect_ratio_val}) corresponds to a standard display viewport.")
            else:
                indicators.append(f"Non-standard aspect ratio ({aspect_ratio_val}); image may be cropped or scaled.")

            if ui_count > 0:
                indicators.append(f"Detected {ui_count} rectangular UI structures.")
            if repeated_aligns > 0:
                indicators.append(f"Detected {repeated_aligns} repeated vertical alignment grids.")
            if text_ratio > 0.20:
                indicators.append(f"Elevated horizontal text scanline density ({text_ratio:.3f}).")

            # Map findings for schema compatibility
            findings: List[FindingItem] = list(base_res.findings)
            # If viewport matched, also ensure FIND-SCREEN-001 is included for legacy tests
            if is_common and not any(f.finding_id == "FIND-SCREEN-001" for f in findings):
                findings.append(
                    FindingItem(
                        finding_id="FIND-SCREEN-001",
                        analyzer="ScreenshotAnalyzer",
                        category="SCREENSHOT",
                        title="Exact Native Viewport Match",
                        description=f"Evidence resolution matches standard digital device display: {matched_viewport}.",
                        severity=FindingSeverity.INFO,
                        confidence=0.90,
                        technical_details={
                            "width": m["dimensions"]["width"],
                            "height": m["dimensions"]["height"],
                            "matched_profile": matched_viewport,
                        },
                        is_anomaly=False,
                    )
                )

            result = ScreenshotAnalyzerResult(
                status=AnalyzerStatus.COMPLETED,
                is_common_viewport=is_common,
                matched_viewport=matched_viewport,
                aspect_ratio_standard=is_standard_ratio,
                indicators=indicators,
                ui_rectangles_detected=ui_count,
                repeated_alignments_count=repeated_aligns,
                text_density_ratio=text_ratio,
            )
            return result, findings

        # Fallback: Dimension-only evaluation if image bytes are not provided
        w = width or 0
        h = height or 0
        ar = aspect_ratio or (round(w / max(h, 1), 4) if w and h else 1.0)
        matched_viewport = self._find_matching_viewport(w, h)
        is_common = matched_viewport is not None
        is_standard_ratio = self._is_standard_aspect_ratio(ar)

        findings: List[FindingItem] = []
        indicators: List[str] = []

        if is_common:
            indicators.append(f"Dimensions ({w}x{h}) exactly match profile: {matched_viewport}")
            findings.append(
                FindingItem(
                    finding_id="FIND-SCREEN-001",
                    analyzer="ScreenshotAnalyzer",
                    category="SCREENSHOT",
                    title="Exact Native Viewport Match",
                    description=f"Evidence resolution matches standard digital device display: {matched_viewport}.",
                    severity=FindingSeverity.INFO,
                    confidence=0.90,
                    technical_details={"width": w, "height": h, "matched_profile": matched_viewport},
                    is_anomaly=False,
                )
            )
        elif is_standard_ratio:
            indicators.append(f"Aspect ratio ({ar}) corresponds to a standard display viewport.")
        else:
            indicators.append(f"Non-standard aspect ratio ({ar}); image may be cropped or scaled.")
            findings.append(
                FindingItem(
                    finding_id="FIND-SCREEN-002",
                    analyzer="ScreenshotAnalyzer",
                    category="SCREENSHOT",
                    title="Non-Standard Display Aspect Ratio",
                    description=(
                        f"Image aspect ratio {ar} deviates from standard mobile/desktop display ratios. "
                        "Indicates manual cropping, resizing, or partial UI capture."
                    ),
                    severity=FindingSeverity.LOW,
                    confidence=0.65,
                    technical_details={"aspect_ratio": ar, "dimensions": f"{w}x{h}"},
                    is_anomaly=True,
                )
            )

        result = ScreenshotAnalyzerResult(
            status=AnalyzerStatus.COMPLETED,
            is_common_viewport=is_common,
            matched_viewport=matched_viewport,
            aspect_ratio_standard=is_standard_ratio,
            indicators=indicators,
            ui_rectangles_detected=0,
            repeated_alignments_count=0,
            text_density_ratio=0.0,
        )
        return result, findings


screenshot_analysis_service = ScreenshotAnalysisService()

