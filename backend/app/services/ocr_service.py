"""
Optical Character Recognition (OCR) and text verification service.
Evaluates local OCR availability honestly without fabricating extracted text.
"""
from typing import Tuple, List, Optional
from app.schemas.common import AnalyzerStatus, FindingSeverity
from app.schemas.forensic import OCRAnalyzerResult, OCRRegion, FindingItem
from forensic.ocr import ocr_analyzer


class OCRService:
    def is_engine_available(self) -> bool:
        """Checks if any supported OCR engine (Tesseract or Windows Media OCR) is available."""
        return ocr_analyzer.is_available()

    def get_engine_name(self) -> str:
        """Returns the active OCR engine name."""
        return ocr_analyzer.get_engine_name()

    def analyze_text(self, data: bytes) -> Tuple[OCRAnalyzerResult, List[FindingItem]]:
        """
        Executes OCR extraction and typography analysis via canonical ForensicOCRAnalyzer.
        Differentiates AVAILABLE, NOT_AVAILABLE, and FAILED states.
        """
        base_res = ocr_analyzer.analyze(data)

        if base_res.status == "not_available":
            avail_note = (
                base_res.limitations[-1]
                if base_res.limitations
                else "No operational OCR engine found on the host system. Text extraction unavailable."
            )
            result = OCRAnalyzerResult(
                status=AnalyzerStatus.NOT_AVAILABLE,
                engine=base_res.metrics.get("engine", "None (Offline)"),
                text=None,
                confidence=None,
                regions=[],
                language=None,
                processing_time_ms=None,
                word_count=0,
                character_count=0,
                availability_note=avail_note,
                failure_reason=None,
                font_anomaly_detected=False,
            )
            return result, base_res.findings

        if base_res.status == "failed":
            err = base_res.metrics.get("error", "OCR execution failed")
            return (
                OCRAnalyzerResult(
                    status=AnalyzerStatus.FAILED,
                    engine=base_res.metrics.get("engine", "OCR Engine"),
                    text=None,
                    confidence=None,
                    regions=[],
                    language=None,
                    processing_time_ms=None,
                    word_count=0,
                    character_count=0,
                    availability_note=None,
                    failure_reason=err,
                    font_anomaly_detected=False,
                ),
                base_res.findings,
            )

        m = base_res.metrics
        word_count = m.get("word_count", 0)
        char_count = m.get("character_count", 0)
        text = m.get("extracted_text", "")
        font_anomaly = m.get("font_anomaly_detected", False)
        avg_conf = m.get("average_confidence")
        engine_name = m.get("engine", "OCR Engine")
        lang = m.get("language")
        proc_time = m.get("processing_time_ms")

        # Map raw regions to OCRRegion schema
        regions: List[OCRRegion] = []
        for r in m.get("regions", []):
            if "text" in r and "bbox" in r:
                regions.append(OCRRegion(
                    text=str(r["text"]),
                    bbox=list(r["bbox"]),
                    confidence=r.get("confidence")
                ))

        # Map findings for schema compatibility
        findings: List[FindingItem] = []
        for f in base_res.findings:
            if f.finding_id == "FIND-OCR-TEXT":
                findings.append(
                    FindingItem(
                        finding_id="FIND-OCR-001",
                        analyzer="OCRAnalyzer",
                        category="OCR",
                        title=f"Embedded Text Glyphs Detected ({engine_name})",
                        description=f"OCR extracted {word_count} words across the evidence image.",
                        severity=FindingSeverity.INFO,
                        confidence=avg_conf if avg_conf is not None else 0.85,
                        technical_details={
                            "word_count": word_count,
                            "char_count": char_count,
                            "engine": engine_name,
                            "language": lang,
                            "processing_time_ms": proc_time,
                        },
                        is_anomaly=False,
                    )
                )
            else:
                findings.append(f)

        result = OCRAnalyzerResult(
            status=AnalyzerStatus.AVAILABLE,
            engine=engine_name,
            text=text if text else "",
            confidence=avg_conf,
            regions=regions,
            language=lang,
            processing_time_ms=proc_time,
            word_count=word_count,
            character_count=char_count,
            availability_note=None,
            failure_reason=None,
            font_anomaly_detected=font_anomaly,
        )
        return result, findings


ocr_service = OCRService()

