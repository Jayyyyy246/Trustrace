"""MODULE 6 — EVIDENCE FUSION & DECISION ENGINE.

Transparent, graph-based multi-criteria evidence fusion engine.
Combines deterministic forensic findings, container metadata, screenshot layout,
OCR typography, and ML inference status into an explainable assessment.
Does NOT average arbitrary numbers; executes verifiable logic and resolves conflicts.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from app.schemas.common import VerdictLabel, FindingSeverity
from app.schemas.forensic import FindingItem, BaseAnalyzerResult
from app.schemas.verdict import EvidenceSignal
from forensic.decision_engine import evidence_decision_engine, EvidenceDecisionEngine


class FusionVerdict(BaseModel):
    label: VerdictLabel = VerdictLabel.UNKNOWN
    confidence: Optional[float] = None
    uncertainty: Optional[float] = None
    risk_score: float = 0.0
    justification: str
    explanation: Optional[str] = None
    contributing_factors: List[str] = Field(default_factory=list)
    supporting_findings: List[EvidenceSignal] = Field(default_factory=list)
    contradictory_findings: List[EvidenceSignal] = Field(default_factory=list)
    unavailable_analyzers: List[str] = Field(default_factory=list)
    rules_triggered: List[str] = Field(default_factory=list)
    conflict_detected: bool = False
    conflict_details: Optional[str] = None
    limitations: List[str] = Field(default_factory=list)


class ForensicEvidenceFusionEngine:
    """Explainable rule-based decision engine for digital forensics."""

    def __init__(self, engine: Optional[EvidenceDecisionEngine] = None):
        self.engine = engine or evidence_decision_engine

    def fuse(
        self,
        metadata_res: BaseAnalyzerResult,
        image_res: BaseAnalyzerResult,
        screenshot_res: BaseAnalyzerResult,
        ocr_res: BaseAnalyzerResult,
        ml_prediction: Optional[Dict[str, Any]] = None,
    ) -> FusionVerdict:
        # Collect input limitations
        all_limitations: List[str] = []
        for r in [metadata_res, image_res, screenshot_res, ocr_res]:
            all_limitations.extend(r.limitations)

        meta_metrics = dict(metadata_res.metrics)
        img_metrics = dict(image_res.metrics)
        screen_metrics = dict(screenshot_res.metrics)
        ocr_metrics = dict(ocr_res.metrics)

        # Status tracking
        statuses = {
            "metadata": metadata_res.status.value if hasattr(metadata_res.status, "value") else str(metadata_res.status),
            "image_analysis": image_res.status.value if hasattr(image_res.status, "value") else str(image_res.status),
            "screenshot": screenshot_res.status.value if hasattr(screenshot_res.status, "value") else str(screenshot_res.status),
            "ocr": ocr_res.status.value if hasattr(ocr_res.status, "value") else str(ocr_res.status),
            "ml_inference": (ml_prediction.get("model_status") if ml_prediction else "NOT_AVAILABLE"),
        }

        # Invoke core decision engine
        verdict = self.engine.evaluate(
            metadata=meta_metrics,
            image_forensics=img_metrics,
            screenshot=screen_metrics,
            ocr=ocr_metrics,
            ml_prediction=ml_prediction,
            analyzer_status=statuses,
        )

        # Merge limitations
        merged_limitations = list(dict.fromkeys(all_limitations + verdict.limitations))

        # Contributing factors derived from supporting findings
        factors = [s.explanation for s in verdict.supporting_findings]
        if verdict.conflict_details:
            factors.append(verdict.conflict_details)

        return FusionVerdict(
            label=verdict.label,
            confidence=verdict.confidence,
            uncertainty=verdict.uncertainty,
            risk_score=verdict.risk_score,
            justification=verdict.justification,
            explanation=verdict.explanation,
            contributing_factors=factors,
            supporting_findings=verdict.supporting_findings,
            contradictory_findings=verdict.contradictory_findings,
            unavailable_analyzers=verdict.unavailable_analyzers,
            rules_triggered=verdict.decision_rules_triggered,
            conflict_detected=verdict.conflict_detected,
            conflict_details=verdict.conflict_details,
            limitations=merged_limitations,
        )


fusion_engine = ForensicEvidenceFusionEngine()
