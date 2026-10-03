"""
Optical Character Recognition (OCR) and text verification service.
Evaluates local OCR availability honestly without fabricating extracted text.
"""
import io
import shutil
from typing import Tuple, List, Optional
from PIL import Image

from app.schemas.common import AnalyzerStatus, FindingSeverity
from app.schemas.forensic import OCRAnalyzerResult, FindingItem


class OCRService:
    def __init__(self):
        self._tesseract_available: Optional[bool] = None

    def is_engine_available(self) -> bool:
        """Checks if local Tesseract executable exists on the host system."""
        if self._tesseract_available is None:
            # Check PATH or common installation locations
            binary = shutil.which("tesseract")
            if binary:
                self._tesseract_available = True
            else:
                # Common Windows install path fallback
                win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                if shutil.which(win_path):
                    self._tesseract_available = True
                else:
                    self._tesseract_available = False
        return self._tesseract_available

    def analyze_text(self, data: bytes) -> Tuple[OCRAnalyzerResult, List[FindingItem]]:
        """
        Executes OCR extraction if a local engine is installed.
        Explicitly marks as NOT_AVAILABLE if the binary is absent.
        """
        findings: List[FindingItem] = []

        if not self.is_engine_available():
            result = OCRAnalyzerResult(
                status=AnalyzerStatus.NOT_AVAILABLE,
                engine="Tesseract (Offline)",
                text=None,
                word_count=0,
                character_count=0,
                availability_note="Local Tesseract OCR engine binary not found in system PATH. Text extraction unavailable.",
            )
            return result, findings

        # If available, execute pytesseract
        try:
            import pytesseract
            with Image.open(io.BytesIO(data)) as img:
                extracted = pytesseract.image_to_string(img)
                clean_text = extracted.strip()
                words = clean_text.split()
                word_count = len(words)
                char_count = len(clean_text)

                if word_count > 0:
                    findings.append(
                        FindingItem(
                            finding_id="FIND-OCR-001",
                            analyzer="OCRAnalyzer",
                            title="Embedded Text Glyphs Detected",
                            description=f"OCR extracted {word_count} words across the evidence image.",
                            severity=FindingSeverity.INFO,
                            confidence=0.85,
                            technical_details={"word_count": word_count, "char_count": char_count},
                            is_anomaly=False,
                        )
                    )

                result = OCRAnalyzerResult(
                    status=AnalyzerStatus.COMPLETED,
                    engine="Tesseract (Local)",
                    text=clean_text if word_count > 0 else None,
                    word_count=word_count,
                    character_count=char_count,
                    availability_note=None,
                )
                return result, findings
        except Exception as e:
            return (
                OCRAnalyzerResult(
                    status=AnalyzerStatus.FAILED,
                    engine="Tesseract",
                    text=None,
                    word_count=0,
                    character_count=0,
                    availability_note=f"OCR execution failed: {str(e)}",
                ),
                findings,
            )


ocr_service = OCRService()
