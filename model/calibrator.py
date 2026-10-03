"""TRUSTTRACE post-hoc temperature scaling calibrator."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.evaluator import compute_calibration_metrics


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling layer for probability calibration."""

    def __init__(self, initial_temp: float = 1.5) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / torch.clamp(self.temperature, min=1e-3)


def calibrate_model(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_iter: int = 50,
    lr: float = 0.01,
) -> Tuple[float, Dict[str, Any]]:
    """Find optimal temperature T minimizing negative log likelihood on validation logits.
    
    Returns:
    - optimal_temperature: float
    - calibration_report: Dict with before/after ECE and MCE
    """
    model.eval()
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            logits = model(inputs)
            logits_list.append(logits)
            labels_list.append(labels)

    if not logits_list:
        return 1.0, {"status": "empty_validation_set", "temperature": 1.0}

    all_logits = torch.cat(logits_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    # Pre-calibration metrics
    pre_probs = torch.softmax(all_logits, dim=-1)
    pre_confs, pre_preds = torch.max(pre_probs, dim=-1)
    pre_calib = compute_calibration_metrics(
        pre_confs.cpu().numpy(),
        pre_preds.cpu().numpy(),
        all_labels.cpu().numpy()
    )

    # Optimization
    scaler = TemperatureScaler().to(device)
    nll_criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iter)

    def eval_loss():
        optimizer.zero_grad()
        loss = nll_criterion(scaler(all_logits), all_labels)
        loss.backward()
        return loss

    try:
        optimizer.step(eval_loss)
    except Exception:
        # Fallback to Adam if LBFGS encounters conditioning issues
        adam_opt = optim.Adam([scaler.temperature], lr=lr)
        for _ in range(max_iter):
            adam_opt.zero_grad()
            loss = nll_criterion(scaler(all_logits), all_labels)
            loss.backward()
            adam_opt.step()

    optimal_t = float(torch.clamp(scaler.temperature, min=0.01, max=10.0).item())

    # Post-calibration metrics
    post_logits = all_logits / optimal_t
    post_probs = torch.softmax(post_logits, dim=-1)
    post_confs, post_preds = torch.max(post_probs, dim=-1)
    post_calib = compute_calibration_metrics(
        post_confs.cpu().numpy(),
        post_preds.cpu().numpy(),
        all_labels.cpu().numpy()
    )

    report = {
        "optimal_temperature": round(optimal_t, 4),
        "pre_calibration": {
            "ece": pre_calib["ece"],
            "mce": pre_calib["mce"],
        },
        "post_calibration": {
            "ece": post_calib["ece"],
            "mce": post_calib["mce"],
        },
        "ece_reduction": round(pre_calib["ece"] - post_calib["ece"], 4),
    }

    return optimal_t, report
