"""
MODULE 5 — OCR
Optical Character Recognition and typography extraction analyzer.
Returns extracted text, word bounding boxes, mean confidence, and language tags.
Honestly handles availability of local OCR binary without generating simulated text.
Performs real mathematical typography anomaly detection (baseline drift, kerning irregularity).
"""
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
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
            # 1. Environment variable override
            env_cmd = os.getenv("TESSERACT_CMD")
            if env_cmd and (shutil.which(env_cmd) or os.path.isfile(env_cmd)):
                self._tesseract_path = env_cmd
                self._engine_checked = True
                return self._tesseract_path

            # 2. System PATH
            binary = shutil.which("tesseract")
            if binary:
                self._tesseract_path = binary
            else:
                # 3. Known standard install paths across platforms
                candidates = [
                    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
                    "/usr/bin/tesseract",
                    "/usr/local/bin/tesseract",
                    "/opt/homebrew/bin/tesseract",
                ]
                for candidate in candidates:
                    if shutil.which(candidate) or os.path.isfile(candidate):
                        self._tesseract_path = candidate
                        break
            self._engine_checked = True
        return self._tesseract_path

    def _has_windows_media_ocr(self) -> bool:
        if sys.platform != "win32":
            return False
        script = Path(__file__).parent / "win_ocr.ps1"
        return script.is_file() and shutil.which("powershell") is not None

    def is_available(self) -> bool:
        """Determines if any supported OCR executable or native runtime is available on the host."""
        return self._locate_tesseract() is not None or self._has_windows_media_ocr()

    def get_engine_name(self) -> str:
        """Returns the primary active OCR engine name."""
        if self._locate_tesseract() is not None:
            return "Tesseract OCR"
        if self._has_windows_media_ocr():
            return "Windows.Media.Ocr"
        return "None (Unavailable)"

    def detect_typography_anomalies(
        self,
        boxes: List[Dict[str, Any]],
        image_height: int = 1000,
    ) -> Tuple[bool, Dict[str, Any], List[FindingItem]]:
        """
        Analyzes geometric word bounding boxes for digital typography manipulation:
        - Baseline drift / vertical misalignment within common text lines
        - Inter-word spacing / kerning irregularities
        - Non-uniform font box height scaling
        """
        if not boxes or len(boxes) < 2:
            return False, {"font_anomaly_detected": False, "word_count_analyzed": len(boxes)}, []

        # Group words by approximate vertical line (bin by top coordinate / median height)
        valid_boxes = [b for b in boxes if b.get("w", 0) > 3 and b.get("h", 0) > 3]
        if len(valid_boxes) < 2:
            return False, {"font_anomaly_detected": False, "word_count_analyzed": len(valid_boxes)}, []

        median_h = float(np.median([b["h"] for b in valid_boxes]))
        # Cluster boxes into lines based on vertical span overlap and proximity
        sorted_boxes = sorted(valid_boxes, key=lambda b: (b["y"], b["x"]))
        lines: List[List[Dict[str, Any]]] = []
        for box in sorted_boxes:
            b_top = box["y"]
            b_bot = box["y"] + box["h"]
            box_center_y = (b_top + b_bot) / 2.0

            matched_line = None
            for line in lines:
                line_tops = [b["y"] for b in line]
                line_bots = [b["y"] + b["h"] for b in line]
                line_avg_y = np.mean([(t + b) / 2.0 for t, b in zip(line_tops, line_bots)])
                line_min_top = min(line_tops)
                line_max_bot = max(line_bots)

                # Check if vertical interval overlaps or center is within line tolerance
                has_vertical_overlap = (min(b_bot, line_max_bot) - max(b_top, line_min_top)) > 0
                is_close_vertically = abs(box_center_y - line_avg_y) <= max(14.0, median_h * 0.9)

                if has_vertical_overlap or is_close_vertically:
                    matched_line = line
                    break

            if matched_line is not None:
                matched_line.append(box)
            else:
                lines.append([box])

        # Analyze each line with >= 2 words for baseline and kerning anomalies
        anomalous_lines = 0
        max_baseline_drift = 0.0
        kerning_irregularities = 0

        for line in lines:
            if len(line) < 2:
                continue
            line_sorted = sorted(line, key=lambda b: b["x"])
            baselines = [b["y"] + b["h"] for b in line_sorted]
            heights = [b["h"] for b in line_sorted]
            baseline_std = float(np.std(baselines))
            line_median_h = float(np.median(heights))

            if baseline_std > max_baseline_drift:
                max_baseline_drift = baseline_std

            # Significant vertical baseline drift: inserted words shifted from true baseline
            has_baseline_drift = (
                baseline_std > 5.0 and (baseline_std / max(line_median_h, 1.0)) > 0.25
            )

            # Check horizontal spacing gaps
            gaps = []
            for i in range(len(line_sorted) - 1):
                gap = line_sorted[i + 1]["x"] - (line_sorted[i]["x"] + line_sorted[i]["w"])
                gaps.append(gap)

            # Negative gap means glyph collision / overlap (tampering trace)
            has_kerning_collision = any(g < -3 for g in gaps)
            if has_baseline_drift or has_kerning_collision:
                anomalous_lines += 1
                if has_kerning_collision:
                    kerning_irregularities += sum(1 for g in gaps if g < -3)

        font_anomaly_detected = (anomalous_lines >= 1 and max_baseline_drift > 6.0) or (kerning_irregularities >= 2)

        metrics = {
            "font_anomaly_detected": font_anomaly_detected,
            "max_baseline_drift": round(max_baseline_drift, 2),
            "anomalous_lines_count": anomalous_lines,
            "kerning_irregularities_count": kerning_irregularities,
            "lines_analyzed": len(lines),
            "total_words_analyzed": len(valid_boxes),
        }

        findings: List[FindingItem] = []
        if font_anomaly_detected:
            findings.append(
                FindingItem(
                    finding_id="FIND-OCR-TYPOGRAPHY-ANOMALY",
                    category="OCR",
                    severity=FindingSeverity.HIGH,
                    title="Typography Kerning and Baseline Drift Anomaly Detected",
                    description=(
                        f"Text layout exhibits irregular baseline displacement ({max_baseline_drift:.1f}px drift) "
                        "and inconsistent glyph alignments characteristic of synthetic text overlay tampering."
                    ),
                    evidence=metrics,
                    interpretation="Text layout anomalies indicate potential synthetic text alteration or overlay tampering.",
                    limitation="Handwritten text, curved banners, or stylized artistic fonts can produce irregular baselines without tampering.",
                    confidence=0.82,
                    is_anomaly=True,
                )
            )

        return font_anomaly_detected, metrics, findings

    def analyze(self, data: bytes, lang: str = "eng") -> BaseAnalyzerResult:
        t0 = time.perf_counter()
        findings: List[FindingItem] = []
        metrics: Dict[str, Any] = {
            "engine": "Unknown",
            "language": lang,
            "extracted_text": None,
            "word_count": 0,
            "character_count": 0,
            "average_confidence": None,
            "bounding_boxes": [],
            "regions": [],
            "font_anomaly_detected": False,
            "processing_time_ms": 0.0,
        }
        limitations: List[str] = [
            "OCR accuracy degrades significantly on low-resolution, blurred, or heavily compressed text.",
            "Detecting text does not verify the authenticity of the information stated within the text.",
            "OCR engines do not detect subtle subpixel font forgery without dedicated typography geometry analysis.",
        ]

        # Verify that data represents readable image bytes
        try:
            with Image.open(io.BytesIO(data)) as img:
                img.verify()
        except Exception as e:
            return BaseAnalyzerResult(
                status="failed",
                findings=[],
                metrics={"engine": "ImageDecoder", "error": f"Invalid or unreadable image bytes: {e}", "font_anomaly_detected": False},
                limitations=limitations,
            )

        tess_bin = self._locate_tesseract()
        has_win_ocr = self._has_windows_media_ocr()

        # Engine 1: Tesseract OCR (if installed)
        if tess_bin:
            try:
                import pytesseract
                if self._tesseract_path:
                    pytesseract.pytesseract.tesseract_cmd = self._tesseract_path

                with Image.open(io.BytesIO(data)) as img:
                    img_w, img_h = img.size

                    full_text = pytesseract.image_to_string(img, lang=lang).strip()
                    data_dict = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)

                    boxes = []
                    regions = []
                    confidences = []
                    words = []

                    n_boxes = len(data_dict["text"])
                    for i in range(n_boxes):
                        word = data_dict["text"][i].strip()
                        conf = float(data_dict["conf"][i])
                        if word and conf > 0:
                            words.append(word)
                            confidences.append(conf)
                            bx = int(data_dict["left"][i])
                            by = int(data_dict["top"][i])
                            bw = int(data_dict["width"][i])
                            bh = int(data_dict["height"][i])
                            boxes.append({"word": word, "x": bx, "y": by, "w": bw, "h": bh, "confidence": round(conf, 1)})
                            regions.append({
                                "text": word,
                                "bbox": [bx, by, bw, bh],
                                "confidence": round(conf / 100.0, 3),
                            })

                    avg_conf = round(float(np.mean(confidences)) / 100.0, 3) if confidences else None
                    font_anomaly, typo_metrics, typo_findings = self.detect_typography_anomalies(boxes, image_height=img_h)

                    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                    metrics.update({
                        "engine": "Tesseract OCR",
                        "extracted_text": full_text if full_text else "",
                        "word_count": len(words),
                        "character_count": len(full_text),
                        "average_confidence": avg_conf,
                        "bounding_boxes": boxes[:50],
                        "regions": regions[:50],
                        "font_anomaly_detected": font_anomaly,
                        "typography": typo_metrics,
                        "processing_time_ms": elapsed_ms,
                    })

                    if len(words) > 0:
                        findings.append(
                            FindingItem(
                                finding_id="FIND-OCR-TEXT",
                                category="OCR",
                                severity=FindingSeverity.INFO,
                                title="Embedded Text Glyphs Extracted",
                                description=f"OCR extracted {len(words)} word(s) across the document/image with mean confidence {avg_conf if avg_conf else 'N/A'}.",
                                evidence={"word_count": len(words), "character_count": len(full_text), "average_confidence": avg_conf},
                                interpretation="Text content is present and was machine-transcribed.",
                                limitation="OCR measures glyph readability, not factual truth or typographic authenticity.",
                                confidence=0.90,
                                is_anomaly=False,
                            )
                        )
                    findings.extend(typo_findings)

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
                    metrics={"engine": "Tesseract OCR", "error": str(e), "font_anomaly_detected": False},
                    limitations=limitations,
                )

        # Engine 2: Windows Native Media OCR (Windows 10/11 fallback)
        elif has_win_ocr:
            tmp_path = None
            try:
                # Write payload to temporary file for Windows Runtime BitmapDecoder
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name

                ps_script = str(Path(__file__).parent / "win_ocr.ps1")
                cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps_script, "-ImagePath", tmp_path]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

                if proc.returncode != 0:
                    err_msg = proc.stderr.strip() or f"PowerShell process exited with code {proc.returncode}"
                    return BaseAnalyzerResult(
                        status="failed",
                        findings=[],
                        metrics={"engine": "Windows.Media.Ocr", "error": err_msg, "font_anomaly_detected": False},
                        limitations=limitations,
                    )

                out = json.loads(proc.stdout.strip())
                if out.get("status") == "FAILED":
                    return BaseAnalyzerResult(
                        status="failed",
                        findings=[],
                        metrics={"engine": "Windows.Media.Ocr", "error": out.get("error", "WinRT OCR error"), "font_anomaly_detected": False},
                        limitations=limitations,
                    )

                full_text = out.get("text", "").strip()
                raw_regions = out.get("regions", [])
                words = [r["text"] for r in raw_regions if r.get("text")]
                boxes = [
                    {"word": r["text"], "x": r["bbox"][0], "y": r["bbox"][1], "w": r["bbox"][2], "h": r["bbox"][3]}
                    for r in raw_regions if "bbox" in r and len(r["bbox"]) == 4
                ]

                # Typography anomaly detection on bounding boxes
                font_anomaly, typo_metrics, typo_findings = self.detect_typography_anomalies(boxes)

                elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                win_lang = out.get("language", "en")
                normalized_lang = "eng" if (win_lang.lower().startswith("en") or lang == "eng") else win_lang
                metrics.update({
                    "engine": "Windows.Media.Ocr",
                    "language": normalized_lang,
                    "extracted_text": full_text,
                    "word_count": len(words),
                    "character_count": len(full_text),
                    "average_confidence": None,  # WinRT does not fabricate probabilistic confidence
                    "bounding_boxes": boxes[:50],
                    "regions": raw_regions[:50],
                    "font_anomaly_detected": font_anomaly,
                    "typography": typo_metrics,
                    "processing_time_ms": elapsed_ms,
                })

                if len(words) > 0:
                    findings.append(
                        FindingItem(
                            finding_id="FIND-OCR-TEXT",
                            category="OCR",
                            severity=FindingSeverity.INFO,
                            title="Embedded Text Glyphs Extracted",
                            description=f"Windows Native OCR extracted {len(words)} word(s) across the document/image.",
                            evidence={"word_count": len(words), "character_count": len(full_text), "engine": "Windows.Media.Ocr"},
                            interpretation="Text content is present and was machine-transcribed via Windows.Media.Ocr.",
                            limitation="OCR measures glyph readability, not factual truth or typographic authenticity.",
                            confidence=0.90,
                            is_anomaly=False,
                        )
                    )
                findings.extend(typo_findings)

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
                    metrics={"engine": "Windows.Media.Ocr", "error": str(e), "font_anomaly_detected": False},
                    limitations=limitations,
                )
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

        # Neither engine is installed/available
        else:
            metrics["engine"] = "None (Offline)"
            metrics["dependency_status"] = "NOT_INSTALLED"
            return BaseAnalyzerResult(
                status="not_available",
                findings=[],
                metrics=metrics,
                limitations=limitations + [
                    "No operational OCR engine was found on the host system. "
                    "Install Tesseract (e.g. 'winget install UB-Mannheim.TesseractOCR' or 'apt-get install tesseract-ocr') "
                    "or enable Windows.Media.Ocr to activate optical character recognition."
                ],
            )


ocr_analyzer = ForensicOCRAnalyzer()

