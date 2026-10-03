"""Verdict schemas representing the final evidence-based assessment."""

from typing import List, Optional
from pydantic import BaseModel, Field

from app.schemas.common import VerdictLabel


class EvidenceSignal(BaseModel):
    """Structured evidential signal extracted from an independent analyzer."""
    source: str = Field(..., description="Originating analyzer module (e.g., 'metadata', 'image_analysis', 'screenshot', 'ocr', 'ml_inference')")
    metric: str = Field(..., description="Technical metric identifier")
    direction: VerdictLabel = Field(..., description="Forensic hypothesis supported by this signal")
    reliability: float = Field(..., ge=0.0, le=1.0, description="Scientific/empirical reliability weighting in [0, 1]")
    explanation: str = Field(..., description="Factual, verifiable natural-language explanation of this evidence")
    polarity: str = Field(default="SUPPORTING", description="Relationship to candidate verdict: 'SUPPORTING' or 'CONTRADICTORY'")


class FinalVerdict(BaseModel):
    label: VerdictLabel = Field(
        default=VerdictLabel.UNKNOWN,
        description="Primary classification verdict: REAL, EDITED, AI-GENERATED, SCREENSHOT-MANIPULATED, or UNKNOWN."
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Calibrated confidence score. Must be null if evidence is inconclusive or model unavailable."
    )
    uncertainty: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Quantified uncertainty score in [0, 1] accounting for analyzer coverage and signal conflicts."
    )
    risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Evidence-based risk index (0.0 to 1.0) derived from forensic anomalies."
    )
    justification: str = Field(
        description="Transparent natural language explanation of how the verdict was derived."
    )
    explanation: Optional[str] = Field(
        default=None,
        description="Structured multi-section breakdown answering 'WHY TRUSTTRACE REACHED THIS ASSESSMENT'."
    )
    supporting_findings: List[EvidenceSignal] = Field(
        default_factory=list,
        description="Itemized evidence signals directly supporting the assessment."
    )
    contradictory_findings: List[EvidenceSignal] = Field(
        default_factory=list,
        description="Itemized evidence signals opposing or creating friction against the assessment."
    )
    unavailable_analyzers: List[str] = Field(
        default_factory=list,
        description="List of analyzers that were inactive, unavailable, or failed during this run."
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Operational, physical, or cryptographic limitations applicable to this assessment."
    )
    conflict_detected: bool = Field(
        default=False,
        description="Flag indicating whether adversarial or irreconcilable evidence conflict was encountered."
    )
    conflict_details: Optional[str] = Field(
        default=None,
        description="Detailed description of conflicting evidence signals if detected."
    )
    decision_rules_triggered: List[str] = Field(
        default_factory=list,
        description="Itemized list of decision engine rules that evaluated this evidence."
    )
