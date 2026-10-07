#!/usr/bin/env python3
"""
TRUSTTRACE: Phase 5 Evidence Fusion Model Training Script.

Extracts multimodal forensic features and trains a lightweight, interpretable
evidence-fusion model (Regularized Logistic Regression):
1. Signal A: Global visual authenticity (P_EDITED_global, logit_EDITED_global)
   from frozen Phase 4 model (models/phase4_authenticity_best.pt)
2. Signal B: Fine-grained OCR layout, typography, and token consistency features
   from data/manifests/phase5_ocr_features.csv
3. Combines into data/manifests/phase5_fusion_features.csv
4. Fits StandardScaler strictly on training partition
5. Trains balanced Logistic Regression with cross-validated C on validation partition
6. Calibrates optimal probability threshold on validation partition
7. Saves model bundle to models/phase5_fusion_best.joblib
"""

import sys
import os
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_DIR / "data"
MANIFESTS_DIR = DATA_DIR / "manifests"
MODELS_DIR = WORKSPACE_DIR / "models"
REPORTS_DIR = WORKSPACE_DIR / "reports"

MASTER_MANIFEST_PATH = MANIFESTS_DIR / "phase4_dataset_master.csv"
OCR_FEATURES_PATH = MANIFESTS_DIR / "phase5_ocr_features.csv"
FUSION_FEATURES_PATH = MANIFESTS_DIR / "phase5_fusion_features.csv"
PHASE4_MODEL_PATH = MODELS_DIR / "phase4_authenticity_best.pt"
FUSION_MODEL_PATH = MODELS_DIR / "phase5_fusion_best.joblib"

RANDOM_SEED = 42

FEATURE_COLS = [
    "p_edited_global",
    "logit_edited_global",
    "word_count",
    "line_count",
    "text_density",
    "max_baseline_drift",
    "mean_baseline_drift",
    "anomalous_lines_count",
    "kerning_irregularities_count",
    "font_height_variance",
    "font_anomaly_detected",
    "numeric_token_count",
    "numeric_token_ratio",
    "repeated_token_count",
    "repeated_token_ratio",
    "price_pattern_count",
    "duplicate_amounts_count",
    "line_spacing_variance",
    "left_margin_alignment_std",
]


class MobileNetV3AuthenticityClassifier(nn.Module):
    def __init__(self, pretrained: bool = False, dropout_rate: float = 0.2):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights=None)
        in_features = backbone.classifier[0].in_features
        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        pooled = self.pool(feat)
        logits = self.classifier(pooled)
        return logits


def extract_global_probabilities(device: torch.device) -> Dict[str, Tuple[float, float, float]]:
    print(f"Loading frozen Phase 4 model from {PHASE4_MODEL_PATH}...", flush=True)
    checkpoint = torch.load(PHASE4_MODEL_PATH, map_location=device)
    model = MobileNetV3AuthenticityClassifier(pretrained=False, dropout_rate=checkpoint.get("dropout_rate", 0.2)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    with open(MASTER_MANIFEST_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Extracting global visual probabilities for {len(rows)} samples...", flush=True)
    prob_dict: Dict[str, Tuple[float, float, float]] = {}

    with torch.no_grad():
        for r in rows:
            sid = r["sample_id"]
            img_p = Path(r["image_path"])
            with Image.open(img_p) as img:
                rgb_img = img.convert("RGB")
                tensor = test_transform(rgb_img).unsqueeze(0).to(device)
                logits = model(tensor).squeeze(0)
                probs = torch.softmax(logits, dim=0)

                p_real = float(probs[0].item())
                p_edited = float(probs[1].item())
                logit_diff = float((logits[1] - logits[0]).item())

                prob_dict[sid] = (p_real, p_edited, logit_diff)

    return prob_dict


def build_unified_features(device: torch.device) -> List[Dict[str, Any]]:
    prob_dict = extract_global_probabilities(device)

    with open(OCR_FEATURES_PATH, "r", encoding="utf-8") as f:
        ocr_rows = list(csv.DictReader(f))

    print(f"Combining global visual features with OCR typography features ({len(ocr_rows)} rows)...", flush=True)
    combined: List[Dict[str, Any]] = []

    for r in ocr_rows:
        sid = r["sample_id"]
        p_real, p_edited, logit_diff = prob_dict.get(sid, (0.5, 0.5, 0.0))

        rec = {
            "sample_id": sid,
            "image_path": r["image_path"],
            "split": r["split"],
            "label": r["label"],
            "group_id": r["group_id"],
            "p_real_global": round(p_real, 5),
            "p_edited_global": round(p_edited, 5),
            "logit_edited_global": round(logit_diff, 5),
            "word_count": int(r["word_count"]),
            "line_count": int(r["line_count"]),
            "char_count": int(r["char_count"]),
            "text_density": float(r["text_density"]),
            "max_baseline_drift": float(r["max_baseline_drift"]),
            "mean_baseline_drift": float(r["mean_baseline_drift"]),
            "anomalous_lines_count": int(r["anomalous_lines_count"]),
            "kerning_irregularities_count": int(r["kerning_irregularities_count"]),
            "font_height_variance": float(r["font_height_variance"]),
            "font_anomaly_detected": int(r["font_anomaly_detected"]),
            "numeric_token_count": int(r["numeric_token_count"]),
            "numeric_token_ratio": float(r["numeric_token_ratio"]),
            "repeated_token_count": int(r["repeated_token_count"]),
            "repeated_token_ratio": float(r["repeated_token_ratio"]),
            "price_pattern_count": int(r["price_pattern_count"]),
            "duplicate_amounts_count": int(r["duplicate_amounts_count"]),
            "line_spacing_variance": float(r["line_spacing_variance"]),
            "left_margin_alignment_std": float(r["left_margin_alignment_std"]),
        }
        combined.append(rec)

    # Save to CSV
    fieldnames = list(combined[0].keys())
    with open(FUSION_FEATURES_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined)

    print(f"Saved unified fusion features manifest to: {FUSION_FEATURES_PATH}")
    return combined


def train_fusion_model() -> None:
    print("=" * 70)
    print("TRUSTTRACE: Training Phase 5 Evidence Fusion Model")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Step 1: Build or load unified features
    if not FUSION_FEATURES_PATH.is_file():
        dataset_rows = build_unified_features(device)
    else:
        with open(FUSION_FEATURES_PATH, "r", encoding="utf-8") as f:
            dataset_rows = list(csv.DictReader(f))
        print(f"Loaded existing unified features from {FUSION_FEATURES_PATH.name} ({len(dataset_rows)} rows)")

    # Strict partition isolation
    train_rows = [r for r in dataset_rows if r["split"] == "train"]
    val_rows = [r for r in dataset_rows if r["split"] == "val"]
    test_rows = [r for r in dataset_rows if r["split"] == "test"]

    print(f"Partitions: Train={len(train_rows)}, Val={len(val_rows)}, Test={len(test_rows)} (Held-out)")

    # Construct matrices
    def to_matrix(rows_list: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        X = np.array([[float(r[col]) for col in FEATURE_COLS] for r in rows_list], dtype=np.float32)
        y = np.array([1 if r["label"] == "EDITED" else 0 for r in rows_list], dtype=np.int64)
        return X, y

    X_train, y_train = to_matrix(train_rows)
    X_val, y_val = to_matrix(val_rows)
    X_test, y_test = to_matrix(test_rows)

    # Normalization fitted STRICTLY on train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Grid search regularization parameter C on validation set
    candidate_c = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    best_c = 0.1
    best_val_macro_f1 = -1.0
    best_clf = None

    print("\nTuning regularization parameter C on validation partition...")
    for c_val in candidate_c:
        clf = LogisticRegression(
            C=c_val,
            class_weight="balanced",
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_SEED,
        )
        clf.fit(X_train_scaled, y_train)

        val_probs = clf.predict_proba(X_val_scaled)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_val, val_preds, labels=[0, 1], zero_division=0)
        macro_f1 = float(np.mean(f1))

        print(f"  C={c_val:6.3f} | Val Macro-F1: {macro_f1:.4f} | EDITED Rec: {f1[1]:.4f}")
        if macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = macro_f1
            best_c = c_val
            best_clf = clf

    print(f"\nOptimal Regularization C = {best_c} (Validation Macro-F1: {best_val_macro_f1:.4f})")

    # Calibrate decision threshold on validation partition
    val_probs = best_clf.predict_proba(X_val_scaled)[:, 1]
    candidate_thresholds = np.arange(0.10, 0.90, 0.05)
    best_threshold = 0.50
    best_thresh_macro_f1 = -1.0
    best_thresh_metrics = {}

    print("Calibrating decision threshold on validation partition...")
    for th in candidate_thresholds:
        preds = (val_probs >= th).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_val, preds, labels=[0, 1], zero_division=0)
        macro_f1 = float(np.mean(f1))

        if macro_f1 > best_thresh_macro_f1:
            best_thresh_macro_f1 = macro_f1
            best_threshold = float(th)
            best_thresh_metrics = {
                "threshold": round(float(th), 2),
                "macro_f1": round(macro_f1, 4),
                "edited_recall": round(float(rec[1]), 4),
                "edited_precision": round(float(prec[1]), 4),
                "real_recall": round(float(rec[0]), 4),
            }

    print(f"Optimal Validation Threshold: {best_threshold:.2f}")
    print(f"  Validation Metrics @ threshold {best_threshold:.2f}: {best_thresh_metrics}")

    # Inspect feature importances (coefficients)
    coefs = best_clf.coef_[0]
    sorted_features = sorted(zip(FEATURE_COLS, coefs), key=lambda x: abs(x[1]), reverse=True)
    print("\nFeature Coefficients (Ordered by Absolute Impact):")
    for name, w in sorted_features:
        print(f"  {name:30s}: {w:+.4f}")

    # Save complete model bundle
    bundle = {
        "model": best_clf,
        "scaler": scaler,
        "feature_names": FEATURE_COLS,
        "optimal_c": best_c,
        "calibrated_threshold": best_threshold,
        "val_calibrated_metrics": best_thresh_metrics,
        "random_seed": RANDOM_SEED,
        "feature_coefficients": {k: float(v) for k, v in zip(FEATURE_COLS, coefs)},
    }

    joblib.dump(bundle, FUSION_MODEL_PATH)
    print(f"\nSaved primary Phase 5 fusion model bundle to: {FUSION_MODEL_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    train_fusion_model()
