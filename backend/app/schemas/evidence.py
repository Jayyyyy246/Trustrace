"""
Evidence schemas for ingestion, hashing, and evidence tracking.
"""
from datetime import datetime
from typing import Optional, Dict
from pydantic import BaseModel, Field


class EvidenceHashes(BaseModel):
    sha256: str = Field(description="SHA-256 cryptographic hash (mandatory for chain-of-custody)")
    md5: str = Field(description="MD5 hash for legacy checksum cross-referencing")
    sha1: str = Field(description="SHA-1 cryptographic hash")
    phash: Optional[str] = Field(default=None, description="Perceptual hash (DCT based) for duplicate detection")


class EvidenceInfo(BaseModel):
    evidence_id: Optional[str] = None
    filename: str
    sha256: str
    mime_type: str
    size: int
    dimensions: Optional[Dict[str, int]] = None


class EvidenceUploadResponse(BaseModel):
    evidence_id: str
    filename: str
    sha256: str
    mime_type: str
    size: int
    uploaded_at: datetime
    status: str = "QUEUED"
