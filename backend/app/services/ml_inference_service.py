import os
import time
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import torch

from app.schemas.forensic import MLAnalyzerResult, ModelDiagnostics, FindingItem
from app.schemas.common import FindingSeverity
from model.versioning import compute_sha256


class MLInferenceService:
    def __init__(self, model_path: Optional[str] = None):
        self._is_model_loaded = False
        self._predictor = None
        self._model_version = None
        self._checkpoint_path: Optional[Path] = None
        self._checkpoint_sha256: Optional[str] = None
        self._unavailability_reason: Optional[str] = None
        self._device = "CUDA" if torch.cuda.is_available() else "CPU"

        # Check explicit path or environment variable
        target_path = model_path or os.getenv("TRUSTTRACE_MODEL_PATH")
        if target_path is not None:
            if Path(target_path).is_file():
                self.load_model(target_path)
            else:
                self._is_model_loaded = False
                self._unavailability_reason = f"Specified model checkpoint does not exist: {target_path}"
        else:
            # Auto-discover checkpoint in model/checkpoints/
            self._auto_discover_checkpoint()

    def _auto_discover_checkpoint(self) -> None:
        """Looks for any valid trained .pth or .pt file in model/checkpoints/."""
        checkpoints_dir = Path(__file__).resolve().parent.parent.parent.parent / "model" / "checkpoints"
        if checkpoints_dir.is_dir():
            candidates = sorted(
                list(checkpoints_dir.glob("*.pth")) + list(checkpoints_dir.glob("*.pt")),
                key=os.path.getmtime,
                reverse=True,
            )
            for cand in candidates:
                if cand.is_file() and cand.stat().st_size > 0:
                    if self.load_model(str(cand)):
                        return

        self._unavailability_reason = (
            "No trained model checkpoint found in model/checkpoints/. "
            "To activate Deep ML analysis, run 'python train_pipeline.py' with a verified dataset "
            "or place a trained checkpoint in model/checkpoints/."
        )

    def load_model(self, model_path: str, metadata_path: Optional[str] = None) -> bool:
        """Load and verify a trained TRUSTTRACE model checkpoint."""
        try:
            from model.predictor import ForensicPredictor
            p_path = Path(model_path)
            self._predictor = ForensicPredictor(
                model_path=p_path,
                metadata_path=metadata_path,
                verify_checksum=True
            )
            self._is_model_loaded = True
            self._checkpoint_path = p_path
            self._model_version = self._predictor.metadata.model_version if self._predictor.metadata else "trusttrace-v1.0.0"
            self._checkpoint_sha256 = (
                self._predictor.metadata.sha256_checksum
                if (self._predictor.metadata and self._predictor.metadata.sha256_checksum)
                else compute_sha256(p_path)
            )
            self._unavailability_reason = None
            return True
        except Exception as e:
            self._is_model_loaded = False
            self._predictor = None
            self._checkpoint_path = None
            self._checkpoint_sha256 = None
            self._unavailability_reason = f"Checkpoint loading failed: {str(e)}"
            return False

    def is_available(self) -> bool:
        """Indicates whether active inference weights are loaded."""
        return self._is_model_loaded

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns complete runtime diagnostic status for system inspection."""
        status = "AVAILABLE" if self._is_model_loaded else "NOT_AVAILABLE"
        arch = self._predictor.architecture if self._predictor else "EfficientNet-B0 (Configured)"
        calib_active = (self._predictor.metadata.temperature != 1.0) if (self._predictor and self._predictor.metadata) else False
        return {
            "status": status,
            "model_name": arch,
            "model_version": self._model_version,
            "checkpoint_path": str(self._checkpoint_path) if self._checkpoint_path else None,
            "checkpoint_sha256": self._checkpoint_sha256,
            "device": self._device,
            "calibration": "ACTIVE" if calib_active else "INACTIVE",
            "ood_detection": "ACTIVE",
            "mc_dropout": "ACTIVE",
            "reason": self._unavailability_reason,
        }

    def predict(self, image_data: bytes) -> Tuple[MLAnalyzerResult, List[FindingItem]]:
        """
        Executes ML model inference if loaded.
        Explicitly reports NOT_AVAILABLE and UNKNOWN when weights are not present.
        """
        if not self._is_model_loaded or self._predictor is None:
            diag = ModelDiagnostics(
                name="EfficientNet-B0 / Forensic Deep Classifier",
                version=None,
                checkpoint_path=None,
                checkpoint_sha256=None,
                architecture="efficientnet_b0",
                device=self._device,
                inference_time_ms=None,
                preprocessing_version="v1.0.0-imagenet",
                calibration_active=False,
                ood_active=True,
                mc_dropout_active=True,
            )
            result = MLAnalyzerResult(
                status="NOT_AVAILABLE",
                model_status="NOT_AVAILABLE",
                model_name="EfficientNet-B0 (Configured)",
                model_version=None,
                predicted_label="UNKNOWN",
                confidence=None,
                class_probabilities=None,
                uncertainty=None,
                ood_detected=None,
                calibrated=False,
                device=self._device,
                inference_time_ms=None,
                checkpoint_sha256=None,
                failure_reason=self._unavailability_reason,
                note=(
                    "Deep learning models not active in runtime. "
                    "Skipping ML inference to avoid fabricated probabilistic outputs."
                ),
                diagnostics=diag,
            )
            return result, []

        try:
            t0 = time.perf_counter()
            inference = self._predictor.predict(image_data)
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            pred_label = inference["prediction"]
            probabilities = inference.get("probabilities", {})
            uncertainty = inference.get("uncertainty", 0.0)
            max_conf = max(probabilities.values()) if probabilities else None
            is_ood = (pred_label == "UNKNOWN")
            is_calibrated = (
                self._predictor.metadata is not None
                and self._predictor.metadata.temperature is not None
                and self._predictor.metadata.temperature != 1.0
            )

            diag = ModelDiagnostics(
                name=self._predictor.architecture,
                version=inference.get("model_version", self._model_version),
                checkpoint_path=str(self._checkpoint_path) if self._checkpoint_path else None,
                checkpoint_sha256=self._checkpoint_sha256,
                architecture=self._predictor.architecture,
                device=self._device,
                inference_time_ms=elapsed_ms,
                preprocessing_version="v1.0.0-imagenet",
                calibration_active=is_calibrated,
                ood_active=True,
                mc_dropout_active=True,
            )

            result = MLAnalyzerResult(
                status="AVAILABLE",
                model_status="AVAILABLE",
                model_name=self._predictor.architecture,
                model_version=inference.get("model_version", self._model_version),
                predicted_label=pred_label,
                confidence=max_conf,
                class_probabilities=probabilities,
                uncertainty=uncertainty,
                ood_detected=is_ood,
                calibrated=is_calibrated,
                device=self._device,
                inference_time_ms=elapsed_ms,
                checkpoint_sha256=self._checkpoint_sha256,
                failure_reason=None,
                note=f"Prediction generated via {self._predictor.architecture} model ({self._device}) with uncertainty {uncertainty:.4f}.",
                diagnostics=diag,
            )

            findings: List[FindingItem] = []
            if pred_label not in ("REAL", "UNKNOWN"):
                findings.append(FindingItem(
                    finding_id="FIND-ML-ARTIFACT",
                    analyzer="MLInferenceEngine",
                    category="MODEL_INFERENCE",
                    title="Machine Learning Artifact Classification",
                    description=f"Neural model classified image as '{pred_label}' (Uncertainty: {uncertainty:.4f})",
                    severity=FindingSeverity.MEDIUM if uncertainty > 0.4 else FindingSeverity.HIGH,
                    evidence={"prediction": pred_label, "probabilities": probabilities, "uncertainty": uncertainty, "ood_detected": is_ood},
                    interpretation=f"Pattern characteristics align with '{pred_label}' class distribution.",
                    limitation="ML classification provides advisory signal and must be corroborated by forensic evidence.",
                    confidence=max_conf if max_conf is not None else 0.70,
                    is_anomaly=True,
                ))

            return result, findings

        except Exception as e:
            diag = ModelDiagnostics(
                name=self._predictor.architecture if self._predictor else "EfficientNet-B0",
                version=self._model_version,
                checkpoint_path=str(self._checkpoint_path) if self._checkpoint_path else None,
                checkpoint_sha256=self._checkpoint_sha256,
                architecture=self._predictor.architecture if self._predictor else "efficientnet_b0",
                device=self._device,
                inference_time_ms=None,
                preprocessing_version="v1.0.0-imagenet",
                calibration_active=False,
                ood_active=True,
                mc_dropout_active=True,
            )
            result = MLAnalyzerResult(
                status="FAILED",
                model_status="FAILED",
                model_name=self._predictor.architecture if self._predictor else None,
                model_version=self._model_version,
                predicted_label="UNKNOWN",
                confidence=None,
                class_probabilities=None,
                uncertainty=1.0,
                ood_detected=True,
                calibrated=False,
                device=self._device,
                inference_time_ms=None,
                checkpoint_sha256=self._checkpoint_sha256,
                failure_reason=str(e),
                note=f"Inference execution failed: {e}",
                diagnostics=diag,
            )
            return result, []


ml_inference_service = MLInferenceService()
