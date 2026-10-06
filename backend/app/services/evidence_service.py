"""
Evidence Service orchestrating secure intake, hashing, quarantine storage,
and multi-service analysis execution.
"""
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, List
from pathlib import Path

from app.core.security import validate_evidence_payload, generate_evidence_id
from app.core.logging import logger
from app.core.exceptions import EvidenceNotFoundError
from app.storage.base import EvidenceStorage
from app.storage.local import local_storage
from app.schemas.evidence import EvidenceUploadResponse, EvidenceInfo
from app.schemas.forensic import AnalyzersContainer, FindingItem
from app.schemas.analysis import AnalysisResultResponse
from app.services.hashing_service import hashing_service, HashingService
from app.services.metadata_service import metadata_service, MetadataService
from app.services.image_analysis_service import image_analysis_service, ImageAnalysisService
from app.services.ocr_service import ocr_service, OCRService
from app.services.screenshot_analysis_service import screenshot_analysis_service, ScreenshotAnalysisService
from app.services.ml_inference_service import ml_inference_service, MLInferenceService
from app.services.evidence_fusion_service import evidence_fusion_service, EvidenceFusionService


class EvidenceService:
    def __init__(
        self,
        storage: Optional[EvidenceStorage] = None,
        hasher: Optional[HashingService] = None,
        metadata_svc: Optional[MetadataService] = None,
        image_svc: Optional[ImageAnalysisService] = None,
        ocr_svc: Optional[OCRService] = None,
        screenshot_svc: Optional[ScreenshotAnalysisService] = None,
        ml_svc: Optional[MLInferenceService] = None,
        fusion_svc: Optional[EvidenceFusionService] = None,
    ):
        self.storage = storage or local_storage
        self.hasher = hasher or hashing_service
        self.metadata_svc = metadata_svc or metadata_service
        self.image_svc = image_svc or image_analysis_service
        self.ocr_svc = ocr_svc or ocr_service
        self.screenshot_svc = screenshot_svc or screenshot_analysis_service
        self.ml_svc = ml_svc or ml_inference_service
        self.fusion_svc = fusion_svc or evidence_fusion_service
        
        # In-memory registry for fast job/analysis retrieval
        self._analysis_cache: Dict[str, AnalysisResultResponse] = {}
        self._evidence_registry: Dict[str, EvidenceUploadResponse] = {}

        # Index pre-existing quarantine records on startup
        try:
            if hasattr(self.storage, "base_dir") and self.storage.base_dir.exists():
                for p in self.storage.base_dir.iterdir():
                    if p.is_file() and not p.name.startswith("."):
                        eid = p.stem
                        if eid not in self._evidence_registry:
                            fsize = p.stat().st_size
                            mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
                            mime = "image/jpeg" if p.suffix.lower() in [".jpg", ".jpeg"] else ("image/png" if p.suffix.lower() == ".png" else "application/octet-stream")
                            self._evidence_registry[eid] = EvidenceUploadResponse(
                                evidence_id=eid,
                                filename=p.name,
                                sha256="COMPUTED_ON_INTAKE",
                                mime_type=mime,
                                size=fsize,
                                uploaded_at=mtime,
                                status="STORED",
                            )
        except Exception as exc:
            logger.warning(f"Could not index existing quarantine files: {exc}")

    def ingest_evidence(self, file_bytes: bytes, original_filename: str) -> EvidenceUploadResponse:
        """
        Validates, hashes, and quarantines an uploaded evidence file.
        """
        logger.info(f"Ingesting raw evidence: '{original_filename}' ({len(file_bytes)} bytes)")
        
        # 1. Validation & sanitization
        safe_filename, detected_mime, extension = validate_evidence_payload(
            file_bytes=file_bytes,
            original_filename=original_filename,
        )

        # 2. Cryptographic hashing
        hashes = self.hasher.compute_all_hashes(file_bytes)

        # 3. Generate Evidence ID and persist to quarantine
        evidence_id = generate_evidence_id()
        saved_path = self.storage.save(
            evidence_id=evidence_id,
            filename=safe_filename,
            data=file_bytes,
        )
        logger.info(f"Evidence {evidence_id} quarantined at: {saved_path.name}")

        response = EvidenceUploadResponse(
            evidence_id=evidence_id,
            filename=safe_filename,
            sha256=hashes.sha256,
            mime_type=detected_mime,
            size=len(file_bytes),
            uploaded_at=datetime.now(timezone.utc),
            status="QUEUED",
        )
        self._evidence_registry[evidence_id] = response
        return response

    def run_analysis(self, evidence_id: str) -> AnalysisResultResponse:
        """
        Executes the complete multi-analyzer forensic pipeline on stored evidence.
        """
        # Check cache first
        if evidence_id in self._analysis_cache:
            return self._analysis_cache[evidence_id]

        # Check if ID is an analysis_id
        for cached in self._analysis_cache.values():
            if cached.analysis_id == evidence_id:
                return cached

        if not self.storage.exists(evidence_id):
            raise EvidenceNotFoundError(evidence_id)

        data = self.storage.read_bytes(evidence_id)
        evidence_info_record = self._evidence_registry.get(evidence_id)
        
        filename = evidence_info_record.filename if evidence_info_record else f"{evidence_id}.bin"
        mime_type = evidence_info_record.mime_type if evidence_info_record else "application/octet-stream"
        sha256_hash = self.hasher.compute_sha256(data)

        # Verify integrity
        if evidence_info_record and not self.hasher.verify_integrity(data, evidence_info_record.sha256):
            logger.critical(f"INTEGRITY VIOLATION DETECTED FOR EVIDENCE {evidence_id}!")
            raise ValueError("Evidence file mutated in quarantine! Hash mismatch.")

        logger.info(f"Executing forensic pipeline for evidence {evidence_id} (SHA256: {sha256_hash[:12]}...)")

        all_findings: List[FindingItem] = []

        # 1. Metadata Analysis
        meta_res, meta_findings = self.metadata_svc.extract_metadata(data)
        all_findings.extend(meta_findings)

        # 2. Image & Low-level Computer Vision Analysis
        img_res, img_findings = self.image_svc.analyze_image(data)
        all_findings.extend(img_findings)

        # 3. OCR & Typography Analysis
        ocr_res, ocr_findings = self.ocr_svc.analyze_text(data)
        all_findings.extend(ocr_findings)

        # 4. Screenshot & Viewport Analysis (Executes real pixel and UI layout analysis)
        screen_res, screen_findings = self.screenshot_svc.analyze_screenshot(
            data=data,
            width=img_res.dimensions.get("width", 0),
            height=img_res.dimensions.get("height", 0),
            aspect_ratio=img_res.aspect_ratio,
        )
        all_findings.extend(screen_findings)

        # 5. ML Inference (Explicitly reports NOT_AVAILABLE if models not loaded)
        ml_res, ml_findings = self.ml_svc.predict(data)
        all_findings.extend(ml_findings)

        # 6. Evidence Fusion & Decision Engine
        verdict, limitations = self.fusion_svc.synthesize_verdict(
            metadata_res=meta_res,
            image_res=img_res,
            ocr_res=ocr_res,
            screenshot_res=screen_res,
            ml_res=ml_res,
            findings=all_findings,
        )

        analysis_id = generate_evidence_id()
        
        evidence_summary = EvidenceInfo(
            evidence_id=evidence_id,
            filename=filename,
            sha256=sha256_hash,
            mime_type=mime_type,
            size=len(data),
            dimensions=img_res.dimensions if img_res.dimensions["width"] > 0 else None,
        )

        analyzers_container = AnalyzersContainer(
            metadata=meta_res,
            image=img_res,
            ocr=ocr_res,
            screenshot=screen_res,
            ml=ml_res,
        )

        result = AnalysisResultResponse(
            analysis_id=analysis_id,
            evidence=evidence_summary,
            analyzers=analyzers_container,
            findings=all_findings,
            final_verdict=verdict,
            limitations=limitations,
            created_at=datetime.now(timezone.utc),
            pipeline_version="1.0.0-phase1",
        )

        self._analysis_cache[evidence_id] = result
        return result

    def get_evidence_record(self, evidence_id: str) -> Optional[EvidenceUploadResponse]:
        """Retrieves stored evidence record by ID."""
        return self._evidence_registry.get(evidence_id)

    def list_all_evidence(self) -> List[EvidenceUploadResponse]:
        """Returns all ingested evidence records."""
        return list(self._evidence_registry.values())

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Calculates authentic operational metrics from actual backend data."""
        total_evidence = len(self._evidence_registry)
        total_analyzed = len(self._analysis_cache)
        success_rate = (total_analyzed / total_evidence * 100.0) if total_evidence > 0 else 100.0

        recent_investigations = []
        for ev_id, analysis in sorted(self._analysis_cache.items(), key=lambda x: x[1].created_at, reverse=True)[:10]:
            recent_investigations.append({
                "evidence_id": ev_id,
                "analysis_id": analysis.analysis_id,
                "filename": analysis.evidence.filename,
                "final_verdict": analysis.final_verdict.label,
                "confidence": analysis.final_verdict.confidence,
                "uncertainty": analysis.final_verdict.uncertainty,
                "created_at": analysis.created_at,
                "sha256": analysis.evidence.sha256,
            })

        latest_evidence = []
        for ev in sorted(self._evidence_registry.values(), key=lambda x: x.uploaded_at, reverse=True)[:10]:
            latest_evidence.append({
                "evidence_id": ev.evidence_id,
                "filename": ev.filename,
                "sha256": ev.sha256,
                "size": ev.size,
                "mime_type": ev.mime_type,
                "uploaded_at": ev.uploaded_at,
                "status": ev.status,
            })

        ocr_available = self.ocr_svc.is_engine_available()
        ml_available = self.ml_svc.is_available()

        return {
            "active_investigations": total_evidence,
            "total_evidence_analyzed": total_analyzed,
            "analysis_success_rate": round(success_rate, 1),
            "recent_investigations": recent_investigations,
            "latest_evidence": latest_evidence,
            "system_analyzer_status": {
                "hashing": "ACTIVE",
                "metadata_exif": "ACTIVE",
                "image_cv": "ACTIVE",
                "compression_ela": "ACTIVE",
                "screenshot_geometry": "ACTIVE",
                "ocr_text": "ACTIVE" if ocr_available else "NOT_AVAILABLE",
                "ml_inference": "ACTIVE" if ml_available else "NOT_AVAILABLE",
            },
            "model_status": {
                "is_loaded": ml_available,
                "model_version": self.ml_svc._model_version if ml_available else None,
                "architecture": self.ml_svc._predictor.architecture if ml_available and self.ml_svc._predictor else None,
            }
        }


evidence_service = EvidenceService()
