"""
MODULE 2 — METADATA
Extracts and analyzes container, EXIF, TIFF, JFIF, and color profile metadata.
Strictly adheres to forensic principle: Never assume metadata means authenticity.
"""
import io
from typing import Dict, Any, List, Optional
from PIL import Image, ExifTags

from app.schemas.common import FindingSeverity
from app.schemas.forensic import FindingItem, BaseAnalyzerResult

KNOWN_EDITING_SOFTWARE_KEYWORDS = [
    "photoshop",
    "gimp",
    "canva",
    "procreate",
    "lightroom",
    "pixlr",
    "paint.net",
    "snapseed",
    "midjourney",
    "stable diffusion",
    "dall-e",
    "firefly",
    "illustrator",
    "affinity",
    "vsco",
    "facetune",
]


class ForensicMetadataAnalyzer:
    """Extracts, normalizes, and inspects container and camera metadata."""

    def analyze(self, data: bytes) -> BaseAnalyzerResult:
        findings: List[FindingItem] = []
        metrics: Dict[str, Any] = {}
        limitations: List[str] = [
            "Metadata can be easily stripped, edited, or forged using standard command-line tools (e.g. exiftool).",
            "The presence of pristine camera metadata is not conclusive proof of an unmanipulated image.",
            "The absence of metadata is common in social media uploads (WhatsApp, Twitter, Instagram) and does not inherently indicate fraud.",
        ]

        try:
            with Image.open(io.BytesIO(data)) as img:
                format_name = img.format or "UNKNOWN"
                width, height = img.size
                mode = img.mode

                metrics["format"] = format_name
                metrics["dimensions"] = {"width": width, "height": height}
                metrics["color_mode"] = mode
                metrics["info_tags"] = {}

                # 1. Inspect image.info (ICC profile, JFIF, PNG chunks)
                icc_profile_name = None
                if hasattr(img, "info") and img.info:
                    for k, v in img.info.items():
                        if k == "icc_profile" and isinstance(v, bytes):
                            # Try to extract ICC profile description
                            try:
                                # Look for description tag 'desc' in ICC bytes
                                desc_idx = v.find(b"desc")
                                if desc_idx != -1 and desc_idx + 12 < len(v):
                                    str_len = int.from_bytes(v[desc_idx+8:desc_idx+12], "big")
                                    raw_name = v[desc_idx+12:desc_idx+12+str_len]
                                    icc_profile_name = raw_name.decode("utf-8", errors="ignore").strip("\x00")
                            except Exception:
                                icc_profile_name = "Embedded ICC Profile (Binary)"
                            metrics["color_profile"] = icc_profile_name or "Embedded Profile"
                        elif isinstance(v, (str, int, float, bool)):
                            metrics["info_tags"][str(k)] = v

                metrics["dpi"] = img.info.get("dpi")

                # 2. Extract EXIF
                exif_present = False
                exif_dict: Dict[str, Any] = {}
                camera_make = None
                camera_model = None
                software_tag = None
                orientation = None
                create_date = None
                modify_date = None
                original_date = None
                compression_tag = None

                raw_exif = img.getexif() if hasattr(img, "getexif") else None
                if raw_exif and len(raw_exif) > 0:
                    exif_present = True
                    for tag_id, val in raw_exif.items():
                        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                        if isinstance(val, bytes):
                            try:
                                val = val.decode("utf-8", errors="ignore").strip("\x00")
                            except Exception:
                                val = f"<bytes:{len(val)}>"
                        exif_dict[tag_name] = str(val)

                        # Extract specific standard keys
                        if tag_name == "Make":
                            camera_make = str(val).strip()
                        elif tag_name == "Model":
                            camera_model = str(val).strip()
                        elif tag_name == "Software":
                            software_tag = str(val).strip()
                        elif tag_name == "Orientation":
                            orientation = str(val)
                        elif tag_name == "DateTime":
                            modify_date = str(val).strip()
                        elif tag_name == "DateTimeOriginal":
                            original_date = str(val).strip()
                        elif tag_name == "DateTimeDigitized":
                            create_date = str(val).strip()
                        elif tag_name == "Compression":
                            compression_tag = str(val)

                metrics["exif_present"] = exif_present
                metrics["camera"] = {"make": camera_make, "model": camera_model}
                metrics["software"] = software_tag
                metrics["orientation"] = orientation
                metrics["timestamps"] = {
                    "modify_date": modify_date,
                    "original_date": original_date,
                    "digitized_date": create_date,
                }
                metrics["compression_metadata"] = {
                    "tag": compression_tag,
                    "format": format_name,
                }
                metrics["exif_tags"] = exif_dict

                # 3. Detect Editing Software Signatures
                detected_software = []
                # Check Software tag
                if software_tag:
                    for kw in KNOWN_EDITING_SOFTWARE_KEYWORDS:
                        if kw in software_tag.lower():
                            detected_software.append(f"EXIF Software: {software_tag}")
                            break

                # Check info tags (e.g. Adobe XMP, PNG software chunks)
                for k, v in metrics["info_tags"].items():
                    for kw in KNOWN_EDITING_SOFTWARE_KEYWORDS:
                        if kw in str(v).lower():
                            detected_software.append(f"{k}: {v}")
                            break

                metrics["editing_software_detected"] = len(detected_software) > 0

                if detected_software:
                    findings.append(
                        FindingItem(
                            finding_id="FIND-META-SOFTWARE",
                            category="METADATA",
                            severity=FindingSeverity.HIGH,
                            title="Editing Software Signature Present in Metadata",
                            description=(
                                f"Digital processing tool signature identified: {', '.join(detected_software)}"
                            ),
                            evidence={"detected_software_tags": detected_software},
                            interpretation=(
                                "The image file was saved or exported by digital manipulation software. "
                                "TRUSTTRACE detected indicators consistent with image manipulation or re-export on a computer workstation."
                            ),
                            limitation=(
                                "Software signatures do not specify the extent of editing. An export tool may have "
                                "simply re-encoded or resized an authentic image without altering scene content."
                            ),
                            confidence=0.90,
                            is_anomaly=True,
                        )
                    )

                # 4. Check Timestamp Discrepancies
                if original_date and modify_date and original_date != modify_date:
                    findings.append(
                        FindingItem(
                            finding_id="FIND-META-TIMESTAMP",
                            category="METADATA",
                            severity=FindingSeverity.LOW,
                            title="Metadata Timestamp Inconsistency",
                            description=(
                                f"Original capture timestamp ({original_date}) differs from modification timestamp ({modify_date})."
                            ),
                            evidence={"original": original_date, "modified": modify_date},
                            interpretation="File was modified at a different time than initial capture.",
                            limitation="Timestamp differences commonly occur during non-fraudulent operations like rotating or tagging.",
                            confidence=0.75,
                            is_anomaly=True,
                        )
                    )

                # 5. Check Absence of EXIF
                if not exif_present:
                    findings.append(
                        FindingItem(
                            finding_id="FIND-META-STRIPPED",
                            category="METADATA",
                            severity=FindingSeverity.INFO,
                            title="No Camera Sensor EXIF Metadata",
                            description="Container lacks camera sensor and capture parameter tags.",
                            evidence={"format": format_name, "exif_tags_count": 0},
                            interpretation="Standard for digital screenshots, graphics, or images stripped by messaging platforms.",
                            limitation="Absence of EXIF is neutral; messaging apps strip metadata for privacy reasons.",
                            confidence=0.50,
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


metadata_analyzer = ForensicMetadataAnalyzer()
