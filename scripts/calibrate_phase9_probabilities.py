#!/usr/bin/env python3
"""
TRUSTTRACE: Phase 9 Probability Calibration Suite.

Evaluates post-hoc probability calibration techniques on validation candidate logits:
1. Uncalibrated Baseline (T=1.0)
2. Temperature Scaling (optimal T via NLL minimization)
3. Platt Scaling (logistic calibration: a*z + b)
4. Isotonic Regression

Strict Protocol:
- Calibrators fitted on VALIDATION split ONLY.
- Evaluates ECE (Expected Calibration Error), Brier Score Loss, NLL on validation.
- Selects best calibration method on validation.
- Evaluates locked calibrator on quarantined TEST split.
- Outputs: reports/phase9_calibration.json
"""

import sys
import os
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

DATA_DIR = WORKSPACE_DIR / "data"
MANIFESTS_DIR = DATA_DIR / "manifests"
REPORTS_DIR = WORKSPACE_DIR / "reports"

VAL_MANIFEST = MANIFESTS_DIR / "phase9_candidate_features_val.csv"
TEST_MANIFEST = MANIFESTS_DIR / "phase9_candidate_features_test.csv"
CALIBRATION_JSON = REPORTS_DIR / "phase9_calibration.json"


def compute_ece(targets: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (probs > bin_lower) & (probs <= bin_upper) if i > 0 else (probs >= bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(targets[in_bin])
            conf_in_bin = np.mean(probs[in_bin])
            ece += np.abs(conf_in_bin - acc_in_bin) * prop_in_bin
    return float(ece)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def calibrate_probabilities():
    print("=" * 70)
    print("TRUSTTRACE: Phase 9 Probability Calibration Optimization")
    print("=" * 70)

    # 1. Load Validation Data
    with open(VAL_MANIFEST, "r", encoding="utf-8") as f:
        val_rows = list(csv.DictReader(f))
    val_logits = np.array([float(r["raw_logit"]) for r in val_rows])
    val_probs = np.array([float(r["forged_probability"]) for r in val_rows])
    val_targets = np.array([int(r["binary_label"]) for r in val_rows])

    print(f"Loaded {len(val_rows)} validation candidates ({sum(val_targets)} positive, {len(val_targets) - sum(val_targets)} negative).")

    # 2. Evaluate Baseline Uncalibrated
    base_brier = brier_score_loss(val_targets, val_probs)
    base_ece = compute_ece(val_targets, val_probs)
    base_nll = log_loss(val_targets, val_probs)
    print(f"Baseline Uncalibrated (Val) | ECE: {base_ece:.4f} | Brier: {base_brier:.4f} | NLL: {base_nll:.4f}")

    # 3. Fit Temperature Scaling via NLL minimization
    def nll_obj(t_val):
        t = t_val[0]
        if t <= 0.01:
            return 1e6
        p = sigmoid(val_logits / t)
        p = np.clip(p, 1e-7, 1.0 - 1e-7)
        return log_loss(val_targets, p)

    res = minimize(nll_obj, [1.0], method="Nelder-Mead")
    opt_t = float(res.x[0])
    temp_probs_val = sigmoid(val_logits / opt_t)
    temp_brier = brier_score_loss(val_targets, temp_probs_val)
    temp_ece = compute_ece(val_targets, temp_probs_val)
    temp_nll = log_loss(val_targets, temp_probs_val)
    print(f"Temperature Scaling (T={opt_t:.3f}) (Val) | ECE: {temp_ece:.4f} | Brier: {temp_brier:.4f} | NLL: {temp_nll:.4f}")

    # 4. Fit Platt Scaling (Logistic Regression on raw logits)
    platt_model = LogisticRegression(C=1.0, solver="lbfgs")
    platt_model.fit(val_logits.reshape(-1, 1), val_targets)
    platt_probs_val = platt_model.predict_proba(val_logits.reshape(-1, 1))[:, 1]
    platt_brier = brier_score_loss(val_targets, platt_probs_val)
    platt_ece = compute_ece(val_targets, platt_probs_val)
    platt_nll = log_loss(val_targets, platt_probs_val)
    print(f"Platt Scaling (Val) | ECE: {platt_ece:.4f} | Brier: {platt_brier:.4f} | NLL: {platt_nll:.4f}")

    # 5. Fit Isotonic Regression
    iso_model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso_model.fit(val_probs, val_targets)
    iso_probs_val = iso_model.predict(val_probs)
    iso_brier = brier_score_loss(val_targets, iso_probs_val)
    iso_ece = compute_ece(val_targets, iso_probs_val)
    iso_nll = log_loss(val_targets, np.clip(iso_probs_val, 1e-5, 1.0 - 1e-5))
    print(f"Isotonic Regression (Val) | ECE: {iso_ece:.4f} | Brier: {iso_brier:.4f} | NLL: {iso_nll:.4f}")

    # Select Best Method based on Validation ECE
    val_methods = {
        "Uncalibrated": (base_ece, base_brier, base_nll),
        "Temperature Scaling": (temp_ece, temp_brier, temp_nll),
        "Platt Scaling": (platt_ece, platt_brier, platt_nll),
        "Isotonic Regression": (iso_ece, iso_brier, iso_nll),
    }

    best_method = min(val_methods.keys(), key=lambda k: val_methods[k][0])
    print(f"\nLocked Best Calibration Method on Validation: {best_method}")

    # 6. Evaluate on Held-Out Test Set
    with open(TEST_MANIFEST, "r", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))
    test_logits = np.array([float(r["raw_logit"]) for r in test_rows])
    test_probs = np.array([float(r["forged_probability"]) for r in test_rows])
    test_targets = np.array([int(r["binary_label"]) for r in test_rows])

    # Test baseline
    test_base_ece = compute_ece(test_targets, test_probs)
    test_base_brier = brier_score_loss(test_targets, test_probs)

    # Test Temperature
    test_temp_probs = sigmoid(test_logits / opt_t)
    test_temp_ece = compute_ece(test_targets, test_temp_probs)
    test_temp_brier = brier_score_loss(test_targets, test_temp_probs)

    # Test Platt
    test_platt_probs = platt_model.predict_proba(test_logits.reshape(-1, 1))[:, 1]
    test_platt_ece = compute_ece(test_targets, test_platt_probs)
    test_platt_brier = brier_score_loss(test_targets, test_platt_probs)

    # Test Isotonic
    test_iso_probs = iso_model.predict(test_probs)
    test_iso_ece = compute_ece(test_targets, test_iso_probs)
    test_iso_brier = brier_score_loss(test_targets, test_iso_probs)

    calibration_results = {
        "validation_comparison": {
            "Uncalibrated": {
                "ece": round(base_ece, 4),
                "brier": round(base_brier, 4),
                "nll": round(base_nll, 4),
            },
            "Temperature_Scaling": {
                "optimal_T": round(opt_t, 4),
                "ece": round(temp_ece, 4),
                "brier": round(temp_brier, 4),
                "nll": round(temp_nll, 4),
            },
            "Platt_Scaling": {
                "coef": round(float(platt_model.coef_[0][0]), 4),
                "intercept": round(float(platt_model.intercept_[0]), 4),
                "ece": round(platt_ece, 4),
                "brier": round(platt_brier, 4),
                "nll": round(platt_nll, 4),
            },
            "Isotonic_Regression": {
                "ece": round(iso_ece, 4),
                "brier": round(iso_brier, 4),
                "nll": round(iso_nll, 4),
            },
        },
        "selected_best_method": best_method,
        "optimal_temperature_T": round(opt_t, 4),
        "test_evaluation": {
            "Uncalibrated": {
                "ece": round(test_base_ece, 4),
                "brier": round(test_base_brier, 4),
            },
            "Temperature_Scaling": {
                "ece": round(test_temp_ece, 4),
                "brier": round(test_temp_brier, 4),
            },
            "Platt_Scaling": {
                "ece": round(test_platt_ece, 4),
                "brier": round(test_platt_brier, 4),
            },
            "Isotonic_Regression": {
                "ece": round(test_iso_ece, 4),
                "brier": round(test_iso_brier, 4),
            },
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_JSON, "w", encoding="utf-8") as f:
        json.dump(calibration_results, f, indent=2)
    print(f"\nSaved calibration results to: {CALIBRATION_JSON}")

    return calibration_results


if __name__ == "__main__":
    calibrate_probabilities()
