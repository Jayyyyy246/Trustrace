"""
Metadata extraction and container analysis service for digital evidence.
Analyzes EXIF, TIFF, JFIF, and container tags without fabricating findings.
"""
import io
from typing import Tuple, List, Dict, Any, Optional
from PIL import Image, ExifTags

from app.schemas.common import AnalyzerStatus, FindingSeverity
from app.schemas.forensic import MetadataAnalyzerResult, FindingItem


KNOWN_EDITING_SIGNATURES = [
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
]


class MetadataService:
    def extract_metadata(self, data: bytes) -> Tuple[MetadataAnalyzerResult, List[FindingItem]]:
        """
        Parses EXIF and container metadata from image bytes.
        Returns the parsed MetadataAnalyzerResult and any detected forensic findings.
        """
        findings: List[FindingItem] = []
        anomalies: List[str] = []
        all_tags: Dict[str, Any] = {}
        
        exif_present = False
        camera_make: Optional[str] = None
        camera_model: Optional[str] = None
        software: Optional[str] = None
        create_date: Optional[str] = None
        modify_date: Optional[str] = None
        editing_detected = False

        try:
            with Image.open(io.BytesIO(data)) as img:
                # 1. Inspect image.info dictionary (PNG text chunks, JPEG comments)
                if hasattr(img, "info") and img.info:
                    for key, val in img.info.items():
                        if isinstance(val, (str, int, float, bool)):
                            all_tags[f"info:{key}"] = str(val)

                # 2. Inspect EXIF
                exif_data = img.getexif() if hasattr(img, "getexif") else None
                if exif_data and len(exif_data) > 0:
                    exif_present = True
                    for tag_id, value in exif_data.items():
                        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                        # Filter raw bytes
                        if isinstance(value, bytes):
                            try:
                                value = value.decode("utf-8", errors="ignore").strip("\x00")
                            except Exception:
                                value = f"<binary {len(value)} bytes>"
                        
                        all_tags[tag_name] = str(value)

                        # Extract standard fields
                        if tag_name == "Make" and not camera_make:
                            camera_make = str(value).strip()
                        elif tag_name == "Model" and not camera_model:
                            camera_model = str(value).strip()
                        elif tag_name == "Software" and not software:
                            software = str(value).strip()
                        elif tag_name == "DateTime" and not modify_date:
                            modify_date = str(value).strip()
                        elif tag_name == "DateTimeOriginal" and not create_date:
                            create_date = str(value).strip()

                # 3. Detect Editing Software Signatures in Software or container tags
                software_to_check = [software] if software else []
                # Also check all tags values
                for k, v in all_tags.items():
                    if any(sig in str(v).lower() for sig in KNOWN_EDITING_SIGNATURES):
                        editing_detected = True
                        sig_msg = f"Editing software artifact detected in tag '{k}': {v}"
                        if sig_msg not in anomalies:
                            anomalies.append(sig_msg)

                if editing_detected:
                    findings.append(
                        FindingItem(
                            finding_id="FIND-META-001",
                            analyzer="MetadataAnalyzer",
                            title="Editing Software Signature Detected",
                            description=(
                                f"Metadata contains traces of digital manipulation tools: "
                                f"{', '.join(anomalies)}"
                            ),
                            severity=FindingSeverity.HIGH,
                            confidence=0.92,
                            technical_details={"detected_tags": anomalies, "software": software},
                            is_anomaly=True,
                        )
                    )

                # 4. Check Date Inconsistencies
                if create_date and modify_date and create_date != modify_date:
                    anomaly_msg = f"Original capture date ({create_date}) differs from modification date ({modify_date})."
                    anomalies.append(anomaly_msg)
                    findings.append(
                        FindingItem(
                            finding_id="FIND-META-002",
                            analyzer="MetadataAnalyzer",
                            title="Metadata Timestamp Discrepancy",
                            description=anomaly_msg,
                            severity=FindingSeverity.LOW,
                            confidence=0.75,
                            technical_details={"create_date": create_date, "modify_date": modify_date},
                            is_anomaly=True,
                        )
                    )

                # 5. Check if uncompressed format lacks expected EXIF
                if not exif_present:
                    findings.append(
                        FindingItem(
                            finding_id="FIND-META-003",
                            analyzer="MetadataAnalyzer",
                            title="Absence of Camera Sensor Metadata",
                            description="Image container does not contain hardware EXIF tags. Common for screenshots, web exports, or stripped evidence.",
                            severity=FindingSeverity.INFO,
                            confidence=0.60,
                            technical_details={"container_format": img.format},
                            is_anomaly=False,
                        )
                    )

        except Exception as e:
            return (
                MetadataAnalyzerResult(
                    status=AnalyzerStatus.FAILED,
                    anomalies=[f"Metadata extraction error: {str(e)}"],
                ),
                findings,
            )

        result = MetadataAnalyzerResult(
            status=AnalyzerStatus.COMPLETED,
            exif_present=exif_present,
            camera_make=camera_make,
            camera_model=camera_model,
            software=software,
            create_date=create_date,
            modify_date=modify_date,
            all_tags=all_tags,
            anomalies=anomalies,
            editing_software_detected=editing_detected,
        )
        return result, findings


metadata_service = MetadataService()
