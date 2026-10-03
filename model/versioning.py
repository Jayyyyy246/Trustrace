"""Model versioning, checksum verification, and forensic metadata schema."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


def compute_sha256(file_path: str | Path) -> str:
    """Compute the cryptographic SHA-256 digest of a model checkpoint or file.
    
    Reads in 64KB chunks to maintain a minimal memory footprint.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Target file does not exist for SHA-256 calculation: {path}")

    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_sha256(file_path: str | Path, expected_hash: str) -> bool:
    """Verify that a file matches its expected SHA-256 checksum."""
    if not expected_hash:
        return False
    actual_hash = compute_sha256(file_path)
    return actual_hash.lower() == expected_hash.strip().lower()


class ModelMetadata(BaseModel):
    """Forensic metadata schema for all TRUSTTRACE trained models."""
    model_version: str = Field(..., description="Semantic version of model weights (e.g. 'trusttrace-v1.0.0')")
    training_dataset_version: str = Field(..., description="Dataset identifier and version used for training")
    preprocessing_version: str = Field(..., description="Preprocessing pipeline version and resolution")
    training_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp of model training completion"
    )
    architecture: str = Field(..., description="Backbone architecture (e.g. 'efficientnet_b0', 'mobilenet_v3_small')")
    class_mapping: Dict[str, int] = Field(..., description="Mapping from forensic class name to numeric index")
    evaluation_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Measured validation/test metrics (accuracy, macro_f1, per_class, ece, etc.)"
    )
    sha256_checksum: str = Field(..., description="SHA-256 digest of the model checkpoint weights")
    ood_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "confidence_threshold": 0.55,
            "entropy_threshold": 0.82,
            "energy_threshold": -0.50,
        },
        description="Calibrated thresholds for triggering UNKNOWN classification"
    )
    temperature: float = Field(1.0, description="Post-hoc temperature scaling parameter")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Training hyperparameters")

    def save(self, output_path: str | Path) -> Path:
        """Save metadata to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)
        return path

    @classmethod
    def load(cls, file_path: str | Path) -> ModelMetadata:
        """Load and validate metadata from a JSON file."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Metadata file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
