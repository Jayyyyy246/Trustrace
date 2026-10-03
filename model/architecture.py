"""TRUSTTRACE Neural Architecture for Forensic Artifact Classification."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from model.dataset import CLASS_LABELS, UNKNOWN_LABEL


class ForensicClassifier(nn.Module):
    """Deep neural network for digital forensic classification with OOD/uncertainty scoring."""

    def __init__(
        self,
        architecture: str = "efficientnet_b0",
        pretrained: bool = False,
        num_classes: int = len(CLASS_LABELS),
        dropout_rate: float = 0.3,
        feature_dim: int = 256,
    ) -> None:
        super().__init__()
        self.architecture_name = architecture.lower()
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        if self.architecture_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
            in_features = backbone.classifier[1].in_features
            # Retain feature extractor up to pool
            self.backbone = backbone.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        elif self.architecture_name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            backbone = models.mobilenet_v3_small(weights=weights)
            in_features = backbone.classifier[0].in_features
            self.backbone = backbone.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        elif self.architecture_name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            backbone = models.convnext_tiny(weights=weights)
            in_features = backbone.classifier[2].in_features
            self.backbone = backbone.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        elif self.architecture_name == "custom_forensic_cnn":
            # Compact lightweight CNN for testing or constrained execution
            in_features = 128
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.SiLU(),
                nn.Conv2d(64, in_features, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(in_features),
                nn.SiLU(),
            )
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        else:
            raise ValueError(
                f"Unsupported architecture '{architecture}'. "
                f"Choose from: 'efficientnet_b0', 'mobilenet_v3_small', 'convnext_tiny', 'custom_forensic_cnn'."
            )

        # Forensic classification head
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.SiLU(),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

        # Initialize head layers
        nn.init.kaiming_normal_(self.projection[2].weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.projection[2].bias, 0.0)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0.0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent forensic feature embedding vector."""
        feat_map = self.backbone(x)
        pooled = self.pool(feat_map)
        embedding = self.projection(pooled)
        return embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unscaled class logits."""
        embedding = self.extract_features(x)
        logits = self.classifier(embedding)
        return logits


def compute_uncertainty_metrics(
    logits: torch.Tensor,
    temperature: float = 1.0,
    confidence_threshold: float = 0.55,
    entropy_threshold: float = 0.82,
    energy_threshold: float = -0.50,
) -> Dict[str, Union[torch.Tensor, float, str, Dict[str, float]]]:
    """Compute temperature-scaled probabilities, entropy, energy score, and OOD assessment.
    
    Returns:
    - probabilities: Dict of class -> probability
    - max_prob: float (MSP)
    - entropy: float (Normalized Shannon entropy in [0, 1])
    - energy: float (Free energy score)
    - is_ood: bool (whether sample is judged out-of-distribution)
    - uncertainty_score: float (composite uncertainty in [0, 1])
    - final_class: str (one of REAL, EDITED, AI-GENERATED, SCREENSHOT-MANIPULATED, or UNKNOWN)
    """
    temp = max(float(temperature), 1e-4)
    scaled_logits = logits / temp
    probs = F.softmax(scaled_logits, dim=-1)

    # 1. Maximum Softmax Probability (MSP)
    max_prob, argmax_idx = torch.max(probs, dim=-1)
    max_prob_val = float(max_prob.item())
    pred_idx_val = int(argmax_idx.item())

    # 2. Normalized Shannon Entropy (H in [0, 1])
    num_classes = logits.shape[-1]
    eps = 1e-10
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1) / math.log(max(num_classes, 2))
    entropy_val = float(entropy.item())

    # 3. Energy score: -T * logsumexp(logits / T)
    energy = -temp * torch.logsumexp(scaled_logits, dim=-1)
    energy_val = float(energy.item())

    # Map class probabilities
    prob_dict: Dict[str, float] = {}
    for i, label in enumerate(CLASS_LABELS[:num_classes]):
        prob_dict[label] = round(float(probs[..., i].item()), 4)

    # Composite uncertainty score combining normalized entropy and margin
    composite_uncertainty = max(0.0, min(1.0, 0.6 * entropy_val + 0.4 * (1.0 - max_prob_val)))

    # OOD / Uncertainty Trigger Logic:
    # If confidence is below threshold, or entropy is abnormally elevated, or energy exceeds threshold:
    # Class is strictly labeled UNKNOWN.
    is_uncertain = (
        max_prob_val < confidence_threshold
        or entropy_val > entropy_threshold
        or energy_val > energy_threshold
    )

    if is_uncertain:
        final_prediction = UNKNOWN_LABEL
    else:
        final_prediction = CLASS_LABELS[pred_idx_val]

    return {
        "probabilities": prob_dict,
        "max_prob": max_prob_val,
        "entropy": entropy_val,
        "energy": energy_val,
        "is_ood": is_uncertain,
        "uncertainty": round(composite_uncertainty, 4),
        "prediction": final_prediction,
    }


def build_model(
    architecture: str = "efficientnet_b0",
    pretrained: bool = False,
    num_classes: int = len(CLASS_LABELS),
) -> ForensicClassifier:
    """Factory function for creating TRUSTTRACE forensic classifiers."""
    return ForensicClassifier(
        architecture=architecture,
        pretrained=pretrained,
        num_classes=num_classes,
    )
