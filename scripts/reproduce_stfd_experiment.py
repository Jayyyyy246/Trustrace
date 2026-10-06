#!/usr/bin/env python3
"""
TRUSTTRACE: STFD Manipulation Classifier Experiment Reproducibility Script.

Documents and reproduces the exact experiment configuration for:
- Dataset manifests (clustered Train, Val, Test)
- Random seed (42)
- Model architecture (MobileNetV3-Small)
- Training hyperparameters
- Preprocessing & normalization pipeline
- Software environment versions

Saves:
- reports/stfd_experiment_config.json
"""

import sys
import json
import platform
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = WORKSPACE_DIR / "data" / "manifests"
REPORTS_DIR = WORKSPACE_DIR / "reports"
CONFIG_JSON_PATH = REPORTS_DIR / "stfd_experiment_config.json"


def generate_experiment_config():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    import torch
    import torchvision
    import sklearn
    import numpy as np
    import PIL

    config = {
        "experiment_name": "stfd_5class_manipulation_classification",
        "phase": "PHASE_1_MANIPULATION_CLASSIFIER_BASELINE",
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
            "classes": [
                "COPY_MOVE",
                "SPLICING",
                "REMOVAL",
                "INSERTION",
                "REPLACEMENT",
            ],
            "num_classes": 5,
        },
        "model": {
            "backbone": "mobilenet_v3_small",
            "pretrained": True,
            "pretrained_weights": "MobileNet_V3_Small_Weights.DEFAULT",
            "feature_dim": 256,
            "dropout_rate": 0.3,
            "projection_head": "Flatten -> Dropout(0.3) -> Linear(576, 256) -> BatchNorm1d(256) -> SiLU -> Linear(256, 5)",
        },
        "hyperparameters": {
            "image_size": [224, 224],
            "batch_size": 32,
            "epochs": 6,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR (T_max=6, eta_min=1e-6)",
            "loss_function": "CrossEntropyLoss with Inverse Class Frequency Weighting",
            "gradient_clipping_max_norm": 5.0,
            "model_selection_metric": "Validation Macro-F1",
        },
        "preprocessing": {
            "train": [
                "Resize((224, 224))",
                "ColorJitter(brightness=0.05, contrast=0.05)",
                "ToTensor()",
                "Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
            ],
            "val_and_test": [
                "Resize((224, 224))",
                "ToTensor()",
                "Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
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

    print(f"Saved reproducible experiment configuration to: {CONFIG_JSON_PATH.name}")
    return config


if __name__ == "__main__":
    generate_experiment_config()
