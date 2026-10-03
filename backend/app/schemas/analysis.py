"""
Consolidated Analysis Result Schemas for TRUSTTRACE.
Conforms to the primary API specification.
"""
from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field

from app.schemas.evidence import EvidenceInfo
from app.schemas.forensic import AnalyzersContainer, FindingItem
from app.schemas.verdict import FinalVerdict


class AnalysisResultResponse(BaseModel):
    analysis_id: str = Field(description="Unique UUID for this analysis execution")
    evidence: EvidenceInfo
    analyzers: AnalyzersContainer
    findings: List[FindingItem] = Field(default_factory=list)
    final_verdict: FinalVerdict
    limitations: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    pipeline_version: str = "1.0.0-phase1"
