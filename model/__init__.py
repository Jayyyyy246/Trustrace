"""TRUSTTRACE Machine Learning Forensic Pipeline Package."""

from model.dataset import (
    CLASS_LABELS,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    UNKNOWN_LABEL,
    DatasetConfig,
    ForensicDataset,
    build_transforms,
    create_leak_free_splits,
    validate_dataset_directory,
)
from model.architecture import ForensicClassifier, build_model
from model.predictor import ForensicPredictor, InferenceResult
from model.versioning import ModelMetadata, compute_sha256

__all__ = [
    "CLASS_LABELS",
    "CLASS_TO_IDX",
    "IDX_TO_CLASS",
    "UNKNOWN_LABEL",
    "DatasetConfig",
    "ForensicDataset",
    "build_transforms",
    "create_leak_free_splits",
    "validate_dataset_directory",
    "ForensicClassifier",
    "build_model",
    "ForensicPredictor",
    "InferenceResult",
    "ModelMetadata",
    "compute_sha256",
]
