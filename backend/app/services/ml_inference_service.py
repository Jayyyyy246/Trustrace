import os
from pathlib import Path
from typing import Tuple, List, Optional
from app.schemas.forensic import MLAnalyzerResult, FindingItem
from app.schemas.common import FindingSeverity


class MLInferenceService:
    def __init__(self, model_path: Optional[str] = None):
        self._is_model_loaded = False
        self._predictor = None
        self._model_version = None
        self._model_path = model_path or os.getenv("TRUSTTRACE_MODEL_PATH")
        if self._model_path and Path(self._model_path).is_file():
            self.load_model(self._model_path)

    def load_model(self, model_path: str, metadata_path: Optional[str] = None) -> bool:
        """Load and verify a trained TRUSTTRACE model checkpoint."""
        try:
            from model.predictor import ForensicPredictor
            self._predictor = ForensicPredictor(
                model_path=model_path,
                metadata_path=metadata_path,
                verify_checksum=True
            )
            self._is_model_loaded = True
            self._model_version = self._predictor.metadata.model_version if self._predictor.metadata else "custom"
            return True
        except Exception:
            self._is_model_loaded = False
            self._predictor = None
            return False

    def is_available(self) -> bool:
        """Indicates whether active inference weights are loaded."""
        return self._is_model_loaded

    def predict(self, image_data: bytes) -> Tuple[MLAnalyzerResult, List[FindingItem]]:
        """
        Executes ML model inference if loaded.
        Explicitly reports NOT_AVAILABLE and UNKNOWN when weights are not present.
        """
        if not self._is_model_loaded or self._predictor is None:
            result = MLAnalyzerResult(
                model_status="NOT_AVAILABLE",
                model_name=None,
                model_version=None,
                predicted_label="UNKNOWN",
                confidence=None,
                class_probabilities=None,
                uncertainty=None,
                note=(
                    "Deep learning models not active in runtime. "
                    "Skipping ML inference to avoid fabricated probabilistic outputs."
                ),
            )
            return result, []

        try:
            inference = self._predictor.predict(image_data)
            pred_label = inference["prediction"]
            probabilities = inference.get("probabilities", {})
            uncertainty = inference.get("uncertainty", 0.0)
            max_conf = max(probabilities.values()) if probabilities else None

            result = MLAnalyzerResult(
                model_status="AVAILABLE",
                model_name=self._predictor.architecture,
                model_version=inference.get("model_version", self._model_version),
                predicted_label=pred_label,
                confidence=max_conf,
                class_probabilities=probabilities,
                uncertainty=uncertainty,
                note=f"Prediction generated via {self._predictor.architecture} model with uncertainty {uncertainty:.4f}."
            )

            findings: List[FindingItem] = []
            if pred_label != "REAL" and pred_label != "UNKNOWN":
                findings.append(FindingItem(
                    analyzer="ml_inference",
                    code="ML-ARTIFACT-CLASSIFICATION",
                    title="Machine Learning Artifact Classification",
                    description=f"Neural model classified image as '{pred_label}' (Uncertainty: {uncertainty:.4f})",
                    severity=FindingSeverity.MEDIUM if uncertainty > 0.4 else FindingSeverity.HIGH,
                    evidence={"prediction": pred_label, "probabilities": probabilities, "uncertainty": uncertainty},
                    category="model_inference",
                    interpretation=f"Pattern characteristics align with '{pred_label}' class distribution.",
                    limitation="ML classification provides advisory signal and must be corroborated by forensic evidence."
                ))

            return result, findings

        except Exception as e:
            result = MLAnalyzerResult(
                model_status="ERROR",
                model_name=self._predictor.architecture if self._predictor else None,
                model_version=self._model_version,
                predicted_label="UNKNOWN",
                confidence=None,
                class_probabilities=None,
                uncertainty=1.0,
                note=f"Inference execution error: {e}"
            )
            return result, []


ml_inference_service = MLInferenceService()
