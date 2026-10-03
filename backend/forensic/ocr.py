"""
MODULE 5 — OCR
Optical Character Recognition and typography extraction analyzer.
Returns extracted text, word bounding boxes, mean confidence, and language tags.
Honestly handles availability of local OCR binary without generating simulated text.
"""
import io
import shutil
from typing import Dict, Any, List, Optional
from PIL import Image

from app.schemas.common import FindingSeverity
from app.schemas.forensic import FindingItem, BaseAnalyzerResult


class ForensicOCRAnalyzer:
    """Extracts text glyphs, word positions, and layout typography metrics."""

    def __init__(self):
        self._engine_checked = False
        self._tesseract_path: Optional[str] = None

    def _locate_tesseract(self) -> Optional[str]:
        if not self._engine_checked:
            binary = shutil.which("tesseract")
            if binary:
                self._tesseract_path = binary
            else:
                for candidate in [
                    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                    "/usr/bin/tesseract",
                    "/usr/local/bin/tesseract",
                ]:
                    if shutil.which(candidate):
                        self._tesseract_path = candidate
                        break
            self._engine_checked = True
        return self._tesseract_path

    def analyze(self, data: bytes, lang: str = "eng") -> BaseAnalyzerResult:
        findings: List[FindingItem] = []
        metrics: Dict[str, Any] = {
            "engine": "Tesseract OCR",
            "language": lang,
            "extracted_text": None,
            "word_count": 0,
            "character_count": 0,
            "average_confidence": None,
            "bounding_boxes": [],
        }
        limitations: List[str] = [
            "OCR accuracy degrades significantly on low-resolution, blurred, or heavily compressed text.",
            "Detecting text does not verify the authenticity of the information stated within the text.",
            "Tesseract does not detect subtle subpixel font forgery without dedicated typography geometry analysis.",
        ]

        tess_bin = self._locate_tesseract()
        if not tess_bin:
            return BaseAnalyzerResult(
                status="not_available",
                findings=[],
                metrics=metrics,
                limitations=limitations + [
                    "Local Tesseract OCR binary not found in system PATH. Install Tesseract to enable text extraction."
                ],
            )

        try:
            import pytesseract
            # Configure path if non-standard
            if self._tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = self._tesseract_path

            with Image.open(io.BytesIO(data)) as img:
                # 1. Full text string
                full_text = pytesseract.image_to_string(img, lang=lang).strip()

                # 2. Detailed word-level data (boxes & confidence)
                data_dict = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)

                boxes = []
                confidences = []
                words = []

                n_boxes = len(data_dict["text"])
                for i in range(n_boxes):
                    word = data_dict["text"][i].strip()
                    conf = float(data_dict["conf"][i])
                    if word and conf > 0:
                        words.append(word)
                        confidences.append(conf)
                        boxes.append({
                            "word": word,
                            "x": data_dict["left"][i],
                            "y": data_dict["top"][i],
                            "w": data_dict["width"][i],
                            "h": data_dict["height"][i],
                            "confidence": round(conf, 1),
                        })

                avg_conf = round(float(np.mean(confidences)), 2) if confidences else None

                metrics["extracted_text"] = full_text if full_text else None
                metrics["word_count"] = len(words)
                metrics["character_count"] = len(full_text)
                metrics["average_confidence"] = avg_conf
                metrics["bounding_boxes"] = boxes[:50]  # Cap payload size for safety

                if len(words) > 0:
                    findings.append(
                        FindingItem(
                            finding_id="FIND-OCR-TEXT",
                            category="OCR",
                            severity=FindingSeverity.INFO,
                            title="Embedded Text Glyphs Extracted",
                            description=f"OCR extracted {len(words)} word(s) across the document/image with mean confidence {avg_conf}%.",
                            evidence={
                                "word_count": len(words),
                                "character_count": len(full_text),
                                "average_confidence": avg_conf,
                            },
                            interpretation="Text content is present and was machine-transcribed.",
                            limitation="OCR measures glyph readability, not factual truth or typographic authenticity.",
                            confidence=0.90,
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


ocr_analyzer = ForensicOCRAnalyzer()
