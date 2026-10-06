#!/usr/bin/env python3
"""
TRUSTTRACE: STFD Tamper Localization Experiment Reproducibility Script (Phase 2).

Documents and reproduces the exact experiment configuration for:
- Dataset manifests (clustered Train, Val, Test)
- Ground-truth binary mask handling
- Random seed (42)
- Model architecture (ForensicUNet)
- Loss formulation (BCEWithLogits + Soft Dice Loss)
- Training hyperparameters & resolution (256x256)
- Software environment versions

Saves:
- reports/stfd_localization_experiment_config.json
"""

import sys
import json
import platform
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = WORKSPACE_DIR / "data" / "manifests"
REPORTS_DIR = WORKSPACE_DIR / "reports"
CONFIG_JSON_PATH = REPORTS_DIR / "stfd_localization_experiment_config.json"


def generate_localization_config():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    import torch
    import torchvision
    import sklearn
    import numpy as np
    import PIL

    config = {
        "experiment_name": "stfd_tamper_localization_supervised",
        "phase": "PHASE_2_TAMPER_LOCALIZATION_MODEL",
        "random_seed": 42,
        "dataset": {
            "name": "Smartphone Tampering Forensic Dataset (STFD - ICASSP 2023)",
            "isolation_method": "PHASH_LE4_CONNECTED_COMPONENTS",
            "train_manifest": str(MANIFESTS_DIR / "trusttrace_stfd_clustered_train.csv"),
            "val_manifest": str(MANIFESTS_DIR / "trusttrace_stfd_clustered_val.csv"),
            "test_manifest": str(MANIFESTS_DIR / "trusttrace_stfd_clustered_test.csv"),
            "samples": {
                "train": 2752,
                "val": 590,
                "test": 590,
                "total": 3932,
            },
            "mask_target": "Official STFD binary ground-truth masks (0=background, 1=tampered)",
            "mask_interpolation": "NEAREST (Strictly nearest-neighbor, zero bilinear/bicubic distortion)",
        },
        "model": {
            "architecture": "ForensicUNet",
            "base_filters": 16,
            "total_parameters": 482737,
            "encoder_stages": 3,
            "skip_connections": "Multi-scale concatenation at 1/1, 1/2, 1/4, 1/8 spatial scales",
            "output_channels": 1,
            "activation": "Sigmoid (applied during inference and soft dice evaluation)",
        },
        "hyperparameters": {
            "input_resolution": [256, 256],
            "batch_size": 32,
            "epochs": 4,
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR (T_max=4, eta_min=1e-6)",
            "loss_function": "Combined Weighted BCEWithLogits (pos_weight=5.0) + Soft Dice Loss",
            "model_selection_metric": "Validation Foreground Dice Score",
            "decision_threshold": 0.5,
        },
        "preprocessing_and_augmentation": {
            "train": [
                "Resize((256, 256), Image.Resampling.BILINEAR for Image, NEAREST for Mask)",
                "Synchronized RandomHorizontalFlip(p=0.5) applied identically to Image and Mask",
                "ToTensor()",
                "Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) for Image",
                "Binary thresholding (>128 -> 1.0, <=128 -> 0.0) for Mask",
            ],
            "val_and_test": [
                "Resize((256, 256), Image.Resampling.BILINEAR for Image, NEAREST for Mask)",
                "ToTensor()",
                "Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) for Image",
                "Binary thresholding (>128 -> 1.0, <=128 -> 0.0) for Mask",
            ],
        },
        "software_environment": {
            "os": platform.platform(),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "scikit_learn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "pillow_version": PIL.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }

    with open(CONFIG_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved reproducible localization experiment configuration to: {CONFIG_JSON_PATH.name}")
    return config


if __name__ == "__main__":
    generate_localization_config()
