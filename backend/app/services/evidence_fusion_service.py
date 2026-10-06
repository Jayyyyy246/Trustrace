"""
Evidence Fusion and Decision Engine service for TRUSTTRACE.
Combines deterministic forensic findings, computer vision metrics,
and ML inference status into an explainable assessment.
Never hard-codes verdicts; calculates risk and applies transparent rules.
"""
from typing import List, Tuple
from app.schemas.common import VerdictLabel, FindingSeverity
from app.schemas.forensic import (
    MetadataAnalyzerResult,
    ImageAnalyzerResult,
    OCRAnalyzerResult,
    ScreenshotAnalyzerResult,
    MLAnalyzerResult,
    FindingItem,
)
from app.schemas.verdict import FinalVerdict


from forensic.fusion import fusion_engine
from app.schemas.forensic import BaseAnalyzerResult


class EvidenceFusionService:
    def synthesize_verdict(
        self,
        metadata_res: MetadataAnalyzerResult,
        image_res: ImageAnalyzerResult,
        ocr_res: OCRAnalyzerResult,
        screenshot_res: ScreenshotAnalyzerResult,
        ml_res: MLAnalyzerResult,
        findings: List[FindingItem],
    ) -> Tuple[FinalVerdict, List[str]]:
        """
        Synthesizes all analyzer inputs into a FinalVerdict and a list of operational limitations
        using the ForensicEvidenceFusionEngine.
        """
        # Convert schemas to BaseAnalyzerResult representations
        meta_base = BaseAnalyzerResult(
            status=metadata_res.status.value,
            findings=[f for f in findings if f.category == "METADATA" or "META" in f.finding_id],
            metrics={
                "editing_software_detected": metadata_res.editing_software_detected,
                "exif_present": metadata_res.exif_present,
                "camera": {"make": metadata_res.camera_make, "model": metadata_res.camera_model},
                "software": metadata_res.software,
                "timestamps": {"modify_date": metadata_res.modify_date, "create_date": metadata_res.create_date},
            },
            limitations=[
                "Metadata can be stripped or altered; presence of camera metadata does not prove authenticity."
            ] if not metadata_res.exif_present else [],
        )

        img_base = BaseAnalyzerResult(
            status=image_res.status.value,
            findings=[f for f in findings if f.category in ("COMPRESSION", "NOISE", "CLONE") or "IMG" in f.finding_id],
            metrics={
                "dimensions": image_res.dimensions,
                "laplacian_variance": image_res.laplacian_variance,
                "ela": {"q90": {"variance": image_res.ela_variance or 0.0, "mean_delta": image_res.ela_mean_delta or 0.0}},
                "copy_move": {
                    "detected": any(f.category == "CLONE" or "COPYMOVE" in f.finding_id for f in findings),
                    "clusters": 1 if any(f.category == "CLONE" or "COPYMOVE" in f.finding_id for f in findings) else 0,
                },
                "estimated_jpeg_quality": image_res.estimated_jpeg_quality,
                "noise_residual": image_res.noise_residual,
            },
            limitations=[],
        )

        screen_base = BaseAnalyzerResult(
            status=screenshot_res.status.value,
            findings=[f for f in findings if f.category == "SCREENSHOT" or "SCREEN" in f.finding_id],
            metrics={
                "is_probable_screenshot": screenshot_res.is_common_viewport or screenshot_res.aspect_ratio_standard,
                "matched_viewport": screenshot_res.matched_viewport,
                "aspect_ratio_standard": screenshot_res.aspect_ratio_standard,
                "ui_rectangles_detected": screenshot_res.ui_rectangles_detected,
                "repeated_alignments_count": screenshot_res.repeated_alignments_count,
                "text_density_ratio": screenshot_res.text_density_ratio,
            },
            limitations=[],
        )

        ocr_base = BaseAnalyzerResult(
            status=ocr_res.status.value,
            findings=[f for f in findings if f.category == "OCR" or "OCR" in f.finding_id],
            metrics={
                "word_count": ocr_res.word_count,
                "character_count": ocr_res.character_count,
                "font_anomaly_detected": ocr_res.font_anomaly_detected,
            },
            limitations=[ocr_res.availability_note] if ocr_res.availability_note else [],
        )

        ml_dict = {
            "model_status": ml_res.model_status,
            "predicted_label": ml_res.predicted_label,
            "confidence": ml_res.confidence,
            "uncertainty": getattr(ml_res, "uncertainty", None),
            "probabilities": getattr(ml_res, "class_probabilities", None),
        }

        # Run rule-based fusion
        fusion_result = fusion_engine.fuse(
            metadata_res=meta_base,
            image_res=img_base,
            screenshot_res=screen_base,
            ocr_res=ocr_base,
            ml_prediction=ml_dict,
        )

        verdict = FinalVerdict(
            label=fusion_result.label,
            confidence=fusion_result.confidence,
            uncertainty=fusion_result.uncertainty,
            risk_score=fusion_result.risk_score,
            justification=fusion_result.justification,
            explanation=fusion_result.explanation,
            supporting_findings=fusion_result.supporting_findings,
            contradictory_findings=fusion_result.contradictory_findings,
            unavailable_analyzers=fusion_result.unavailable_analyzers,
            limitations=fusion_result.limitations,
            conflict_detected=fusion_result.conflict_detected,
            conflict_details=fusion_result.conflict_details,
            decision_rules_triggered=fusion_result.rules_triggered,
        )

        return verdict, fusion_result.limitations


evidence_fusion_service = EvidenceFusionService()
