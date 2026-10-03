"""TRUSTTRACE Model Predictor: production inference engine with OOD uncertainty logic and checksum verification."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional, Union

from PIL import Image
from pydantic import BaseModel, Field
import torch

from model.architecture import build_model, compute_uncertainty_metrics
from model.dataset import (
    CLASS_LABELS,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    UNKNOWN_LABEL,
    CorruptImageError,
    build_transforms,
    load_image_safely,
)
from model.versioning import ModelMetadata, compute_sha256, verify_sha256


class ChecksumMismatchError(ValueError):
    """Raised when model file SHA-256 does not match recorded metadata."""
    pass


class UnsupportedModelError(ValueError):
    """Raised when model architecture or weights are unrecognized/unsupported."""
    pass


class ModelNotFoundError(FileNotFoundError):
    """Raised when model checkpoint cannot be located."""
    pass


class InferenceResult(BaseModel):
    """Strict schema contract for TRUSTTRACE ML predictions."""
    model_version: str = Field(..., description="Version of the model that generated this inference")
    prediction: str = Field(..., description="Predicted forensic class, or UNKNOWN if uncertainty/OOD threshold triggered")
    probabilities: Dict[str, float] = Field(..., description="Calibrated class probability distribution")
    uncertainty: float = Field(..., description="Quantified epistemic/aleatoric uncertainty in [0, 1]")
    status: str = Field(default="completed", description="Status of the inference operation ('completed' or 'error')")
    entropy: Optional[float] = Field(None, description="Normalized Shannon entropy")
    energy: Optional[float] = Field(None, description="Free energy score")
    raw_class: Optional[str] = Field(None, description="Top predicted class prior to uncertainty thresholding")


class ForensicPredictor:
    """Production inference engine for digital evidence visual verification."""

    def __init__(
        self,
        model_path: str | Path,
        metadata_path: Optional[str | Path] = None,
        device: Optional[str] = None,
        verify_checksum: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise ModelNotFoundError(f"Model checkpoint file not found at: {self.model_path}")

        # Device selection
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Resolve metadata path if not explicitly provided
        self.metadata: Optional[ModelMetadata] = None
        m_path = self._resolve_metadata_path(metadata_path)
        if m_path and m_path.is_file():
            self.metadata = ModelMetadata.load(m_path)

        # Checksum integrity verification
        if verify_checksum:
            self._verify_model_integrity()

        # Architecture resolution
        architecture = "efficientnet_b0"
        if self.metadata:
            architecture = self.metadata.architecture
        self.architecture = architecture

        # Load weights and build model
        self.model = self._load_model_weights()
        self.model.to(self.device)
        self.model.eval()

        # Transforms
        self.transform = build_transforms(is_training=False)

    def _resolve_metadata_path(self, metadata_path: Optional[str | Path]) -> Optional[Path]:
        if metadata_path:
            p = Path(metadata_path)
            return p if p.is_file() else None

        # Try adjacent metadata folder or sibling .json
        stem = self.model_path.stem
        sibling = self.model_path.with_suffix(".json")
        if sibling.is_file():
            return sibling

        meta_dir = self.model_path.parent.parent / "metadata"
        candidate = meta_dir / f"{stem}.json"
        if candidate.is_file():
            return candidate

        return None

    def _verify_model_integrity(self) -> None:
        if self.metadata and self.metadata.sha256_checksum:
            actual_sha = compute_sha256(self.model_path)
            expected_sha = self.metadata.sha256_checksum.strip().lower()
            if actual_sha != expected_sha:
                raise ChecksumMismatchError(
                    f"Model checksum verification failed for {self.model_path}!\n"
                    f"Expected: {expected_sha}\n"
                    f"Actual:   {actual_sha}"
                )

    def _load_model_weights(self) -> torch.nn.Module:
        try:
            # Safe checkpoint loading (weights_only=True where possible in modern PyTorch)
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
            except Exception:
                checkpoint = torch.load(self.model_path, map_location=self.device)

            if isinstance(checkpoint, dict) and "architecture" in checkpoint:
                arch = checkpoint["architecture"]
                if arch != self.architecture:
                    self.architecture = arch

            model = build_model(
                architecture=self.architecture,
                pretrained=False,
                num_classes=len(CLASS_LABELS),
            )

            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint
            elif isinstance(checkpoint, torch.nn.Module):
                return checkpoint
            else:
                raise UnsupportedModelError(f"Unrecognized checkpoint format in: {self.model_path}")

            # Strip possible DataParallel 'module.' prefix
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                new_key = k[7:] if k.startswith("module.") else k
                cleaned_state_dict[new_key] = v

            model.load_state_dict(cleaned_state_dict, strict=True)
            return model

        except (ValueError, KeyError) as e:
            raise UnsupportedModelError(f"Unsupported or corrupted model architecture: {e}") from e
        except Exception as e:
            if "architecture" in str(e).lower() or "unexpected key" in str(e).lower():
                raise UnsupportedModelError(f"Model checkpoint architecture incompatible: {e}") from e
            raise

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, bytes],
        confidence_threshold: Optional[float] = None,
        entropy_threshold: Optional[float] = None,
        energy_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run forensic inference on an image and return structured assessment."""
        # 1. Image preprocessing
        pil_img = self._load_image_input(image_input)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # 2. Model forward pass
        with torch.no_grad():
            logits = self.model(tensor)

        # 3. Calibration parameters
        temp = self.metadata.temperature if self.metadata else 1.0
        ood_cfg = self.metadata.ood_thresholds if self.metadata else {}

        c_thresh = confidence_threshold if confidence_threshold is not None else ood_cfg.get("confidence_threshold", 0.55)
        e_thresh = entropy_threshold if entropy_threshold is not None else ood_cfg.get("entropy_threshold", 0.82)
        n_thresh = energy_threshold if energy_threshold is not None else ood_cfg.get("energy_threshold", -0.50)

        # 4. Uncertainty & OOD calculation
        metrics = compute_uncertainty_metrics(
            logits=logits[0],
            temperature=temp,
            confidence_threshold=c_thresh,
            entropy_threshold=e_thresh,
            energy_threshold=n_thresh,
        )

        version = self.metadata.model_version if self.metadata else f"trusttrace-{self.architecture}-v1.0"

        # 5. Build strict output schema
        result = InferenceResult(
            model_version=version,
            prediction=metrics["prediction"],
            probabilities=metrics["probabilities"],
            uncertainty=metrics["uncertainty"],
            status="completed",
            entropy=metrics["entropy"],
            energy=metrics["energy"],
            raw_class=CLASS_LABELS[int(torch.argmax(logits[0]).item())],
        )

        return result.model_dump(exclude={"entropy", "energy", "raw_class"} if not True else set())

    def _load_image_input(self, image_input: Union[str, Path, Image.Image, bytes]) -> Image.Image:
        """Safely parse input into PIL Image, handling byte buffers and paths."""
        if isinstance(image_input, Image.Image):
            try:
                image_input.verify()
            except Exception:
                pass
            return image_input.convert("RGB")

        if isinstance(image_input, (str, Path)):
            return load_image_safely(image_input)

        if isinstance(image_input, (bytes, bytearray)):
            if len(image_input) == 0:
                raise CorruptImageError("Empty byte stream provided as image input")
            try:
                stream = io.BytesIO(image_input)
                with Image.open(stream) as img:
                    img.verify()
                stream.seek(0)
                with Image.open(stream) as img:
                    return img.convert("RGB")
            except Exception as e:
                raise CorruptImageError(f"Cannot decode image from bytes: {e}") from e

        raise TypeError(f"Unsupported image input type: {type(image_input)}")
