"""TRUSTTRACE comprehensive evaluation suite: metrics, confusion matrix, calibration, and OOD analysis."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.dataset import CLASS_LABELS, UNKNOWN_LABEL


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_labels: Sequence[str] = CLASS_LABELS
) -> Dict[str, Any]:
    """Compute exact confusion matrix, accuracy, precision, recall, and macro-F1.
    
    No metrics are fabricated; every number is computed from the provided arrays.
    """
    y_t = np.array(y_true, dtype=int)
    y_p = np.array(y_pred, dtype=int)
    n_samples = len(y_t)
    n_classes = len(class_labels)

    if n_samples == 0:
        return {
            "total_samples": 0,
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "per_class": {},
            "confusion_matrix": [],
        }

    # Confusion Matrix: rows = True, columns = Predicted
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_t, y_p):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1

    accuracy = float(np.trace(cm) / max(n_samples, 1))

    per_class: Dict[str, Dict[str, float]] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for i, label in enumerate(class_labels):
        tp = int(cm[i, i])
        fp = int(np.sum(cm[:, i]) - tp)
        fn = int(np.sum(cm[i, :]) - tp)
        tn = int(n_samples - (tp + fp + fn))
        support = int(np.sum(cm[i, :]))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        per_class[label] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "support": support,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
        }

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    return {
        "total_samples": n_samples,
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_labels": list(class_labels),
    }


def compute_calibration_metrics(
    confidences: Sequence[float],
    predictions: Sequence[int],
    targets: Sequence[int],
    num_bins: int = 10
) -> Dict[str, Any]:
    """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)."""
    confs = np.array(confidences, dtype=float)
    preds = np.array(predictions, dtype=int)
    targs = np.array(targets, dtype=int)
    n = len(confs)

    if n == 0:
        return {"ece": 0.0, "mce": 0.0, "bins": []}

    accuracies = (preds == targs).astype(float)
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)

    ece = 0.0
    mce = 0.0
    bin_data: List[Dict[str, Any]] = []

    for i in range(num_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        if i == num_bins - 1:
            in_bin = (confs >= lower) & (confs <= upper)
        else:
            in_bin = (confs >= lower) & (confs < upper)

        count = int(np.sum(in_bin))
        if count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confs[in_bin]))
            error = abs(bin_acc - bin_conf)
            ece += (count / n) * error
            mce = max(mce, error)
        else:
            bin_acc = 0.0
            bin_conf = 0.0
            error = 0.0

        bin_data.append({
            "bin_idx": i,
            "range": [round(lower, 2), round(upper, 2)],
            "count": count,
            "accuracy": round(bin_acc, 4),
            "confidence": round(bin_conf, 4),
            "calibration_error": round(error, 4),
        })

    return {
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "num_bins": num_bins,
        "bins": bin_data,
    }


def compute_ood_metrics(
    id_scores: Sequence[float],
    ood_scores: Sequence[float]
) -> Dict[str, Any]:
    """Compute AUROC and FPR95 for out-of-distribution evaluation.
    
    Higher score indicates ID (e.g. MSP or -Uncertainty).
    """
    id_arr = np.array(id_scores, dtype=float)
    ood_arr = np.array(ood_scores, dtype=float)

    if len(id_arr) == 0 or len(ood_arr) == 0:
        return {"auroc": 0.5, "fpr95": 1.0, "id_count": len(id_arr), "ood_count": len(ood_arr)}

    # Binary labels: 1 for ID, 0 for OOD
    labels = np.concatenate([np.ones_like(id_arr), np.zeros_like(ood_arr)])
    scores = np.concatenate([id_arr, ood_arr])

    # Rank-based AUROC (Mann-Whitney U equivalent)
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]

    n_pos = len(id_arr)
    n_neg = len(ood_arr)

    # TPR and FPR vectors
    tpr = np.cumsum(sorted_labels) / n_pos
    fpr = np.cumsum(1.0 - sorted_labels) / n_neg

    # Trapezoidal integration for AUROC
    if hasattr(np, "trapezoid"):
        auroc = float(np.trapezoid(tpr, fpr)) if len(fpr) > 1 else 0.5
    elif hasattr(np, "trapz"):
        auroc = float(np.trapz(tpr, fpr)) if len(fpr) > 1 else 0.5
    else:
        auroc = float(np.sum(0.5 * (tpr[1:] + tpr[:-1]) * np.diff(fpr))) if len(fpr) > 1 else 0.5

    # FPR at 95% TPR
    idx_95 = np.where(tpr >= 0.95)[0]
    fpr95 = float(fpr[idx_95[0]]) if len(idx_95) > 0 else 1.0

    return {
        "auroc": round(max(0.0, min(1.0, auroc)), 4),
        "fpr95": round(max(0.0, min(1.0, fpr95)), 4),
        "id_mean_score": round(float(np.mean(id_arr)), 4),
        "ood_mean_score": round(float(np.mean(ood_arr)), 4),
        "id_count": len(id_arr),
        "ood_count": len(ood_arr),
    }


def evaluate_dataloader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    temperature: float = 1.0,
    ood_thresholds: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Execute complete forensic evaluation across a DataLoader.
    
    Generates true empirical metrics with zero fabrication.
    """
    model.eval()
    all_targets: List[int] = []
    all_preds: List[int] = []
    all_confs: List[float] = []
    all_entropies: List[float] = []
    all_energies: List[float] = []
    all_logits: List[torch.Tensor] = []

    conf_thresh = ood_thresholds.get("confidence_threshold", 0.55) if ood_thresholds else 0.55
    entropy_thresh = ood_thresholds.get("entropy_threshold", 0.82) if ood_thresholds else 0.82

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch[0], batch[1]
            inputs = inputs.to(device)
            logits = model(inputs)

            all_logits.append(logits.cpu())
            all_targets.extend(targets.tolist())

            # Scaled softmax
            probs = F.softmax(logits / max(temperature, 1e-4), dim=-1)
            confs, preds = torch.max(probs, dim=-1)

            # Entropy
            num_classes = logits.shape[-1]
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1) / math.log(max(num_classes, 2))

            # Energy
            energy = -temperature * torch.logsumexp(logits / max(temperature, 1e-4), dim=-1)

            all_confs.extend(confs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_entropies.extend(entropy.cpu().tolist())
            all_energies.extend(energy.cpu().tolist())

    # Multi-class performance
    class_metrics = compute_classification_metrics(all_targets, all_preds, CLASS_LABELS)

    # Calibration performance
    calib_metrics = compute_calibration_metrics(all_confs, all_preds, all_targets, num_bins=10)

    # Uncertainty statistics
    uncertainty_stats = {
        "mean_confidence": round(float(np.mean(all_confs)), 4) if all_confs else 0.0,
        "mean_entropy": round(float(np.mean(all_entropies)), 4) if all_entropies else 0.0,
        "mean_energy": round(float(np.mean(all_energies)), 4) if all_energies else 0.0,
        "uncertainty_rate": round(
            float(np.mean([(c < conf_thresh or e > entropy_thresh) for c, e in zip(all_confs, all_entropies)])),
            4
        ) if all_confs else 0.0,
    }

    results = {
        "classification": class_metrics,
        "calibration": calib_metrics,
        "uncertainty": uncertainty_stats,
    }
    return results
