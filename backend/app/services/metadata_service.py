"""
Metadata extraction and container analysis service for digital evidence.
Analyzes EXIF, TIFF, JFIF, and container tags without fabricating findings.
"""
from typing import Tuple, List, Dict, Any, Optional
from app.schemas.common import AnalyzerStatus, FindingSeverity
from app.schemas.forensic import MetadataAnalyzerResult, FindingItem
from forensic.metadata import metadata_analyzer, KNOWN_EDITING_SOFTWARE_KEYWORDS

# Export for backward compatibility
KNOWN_EDITING_SIGNATURES = KNOWN_EDITING_SOFTWARE_KEYWORDS


class MetadataService:
    def extract_metadata(self, data: bytes) -> Tuple[MetadataAnalyzerResult, List[FindingItem]]:
        """
        Parses EXIF and container metadata from image bytes via canonical ForensicMetadataAnalyzer.
        Returns the parsed MetadataAnalyzerResult and itemized forensic findings.
        """
        base_res = metadata_analyzer.analyze(data)
        if base_res.status == "failed":
            err = base_res.metrics.get("error", "Metadata extraction error")
            return (
                MetadataAnalyzerResult(
                    status=AnalyzerStatus.FAILED,
                    anomalies=[f"Metadata extraction error: {err}"],
                ),
                base_res.findings,
            )

        m = base_res.metrics
        cam = m.get("camera", {}) or {}
        timestamps = m.get("timestamps", {}) or {}
        software = m.get("software")
        editing_detected = m.get("editing_software_detected", False)
        anomalies: List[str] = []

        all_tags: Dict[str, Any] = {}
        for k, v in m.get("info_tags", {}).items():
            all_tags[f"info:{k}"] = str(v)
        for k, v in m.get("exif_tags", {}).items():
            all_tags[k] = str(v)

        mapped_findings: List[FindingItem] = []
        for f in base_res.findings:
            if f.is_anomaly:
                anomalies.append(f.description)
            # Map canonical finding IDs to legacy contract IDs where expected by tests
            if f.finding_id == "FIND-META-SOFTWARE":
                mapped_findings.append(
                    FindingItem(
                        finding_id="FIND-META-001",
                        analyzer="MetadataAnalyzer",
                        category="METADATA",
                        title=f.title,
                        description=f.description,
                        severity=f.severity,
                        confidence=f.confidence,
                        technical_details={"detected_tags": anomalies, "software": software},
                        is_anomaly=True,
                    )
                )
            elif f.finding_id == "FIND-META-TIMESTAMP":
                mapped_findings.append(
                    FindingItem(
                        finding_id="FIND-META-002",
                        analyzer="MetadataAnalyzer",
                        category="METADATA",
                        title=f.title,
                        description=f.description,
                        severity=f.severity,
                        confidence=f.confidence,
                        technical_details=f.evidence,
                        is_anomaly=True,
                    )
                )
            elif f.finding_id == "FIND-META-STRIPPED":
                mapped_findings.append(
                    FindingItem(
                        finding_id="FIND-META-003",
                        analyzer="MetadataAnalyzer",
                        category="METADATA",
                        title=f.title,
                        description=f.description,
                        severity=f.severity,
                        confidence=f.confidence,
                        technical_details=f.evidence,
                        is_anomaly=False,
                    )
                )
            else:
                mapped_findings.append(f)

        result = MetadataAnalyzerResult(
            status=AnalyzerStatus.COMPLETED,
            exif_present=m.get("exif_present", False),
            camera_make=cam.get("make"),
            camera_model=cam.get("model"),
            software=software,
            create_date=timestamps.get("digitized_date") or timestamps.get("original_date"),
            modify_date=timestamps.get("modify_date"),
            all_tags=all_tags,
            anomalies=anomalies,
            editing_software_detected=editing_detected,
        )
        return result, mapped_findings


metadata_service = MetadataService()

