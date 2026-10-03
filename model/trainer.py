"""TRUSTTRACE Model Trainer: reproducible PyTorch training pipeline, checkpointing, and versioning."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from model.architecture import build_model
from model.calibrator import calibrate_model
from model.dataset import (
    CLASS_LABELS,
    CLASS_TO_IDX,
    DatasetConfig,
    ForensicDataset,
    build_transforms,
    create_leak_free_splits,
    collect_dataset_samples,
)
from model.evaluator import evaluate_dataloader
from model.versioning import ModelMetadata, compute_sha256


class ForensicTrainer:
    """End-to-end reproducible training orchestrator for TRUSTTRACE forensic models."""

    def __init__(
        self,
        dataset_dir: str | Path,
        output_dir: str | Path = "model",
        architecture: str = "efficientnet_b0",
        pretrained: bool = False,
        config: Optional[DatasetConfig] = None,
        model_version: str = "trusttrace-v1.0.0",
        device: Optional[str] = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.metadata_dir = self.output_dir / "metadata"
        self.eval_dir = self.output_dir / "evaluation"

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        self.architecture = architecture
        self.pretrained = pretrained
        self.config = config or DatasetConfig()
        self.model_version = model_version

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Collect samples, generate leak-free splits, and return DataLoaders."""
        samples = collect_dataset_samples(
            self.dataset_dir,
            expected_classes=CLASS_LABELS,
            group_separator=self.config.group_separator
        )

        if len(samples) == 0:
            raise ValueError(f"No valid image samples found in dataset directory: {self.dataset_dir}")

        train_s, val_s, test_s = create_leak_free_splits(samples, self.config)

        train_tf = build_transforms(is_training=True, config=self.config)
        eval_tf = build_transforms(is_training=False, config=self.config)

        train_ds = ForensicDataset(train_s, transform=train_tf)
        val_ds = ForensicDataset(val_s, transform=eval_tf)
        test_ds = ForensicDataset(test_s, transform=eval_tf)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=len(train_ds) > self.config.batch_size
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )

        return train_loader, val_loader, test_loader

    def train(
        self,
        epochs: int = 10,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        early_stopping_patience: int = 5,
    ) -> Dict[str, Any]:
        """Execute complete reproducible training with real evaluation."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        train_loader, val_loader, test_loader = self.prepare_data()

        # Build model
        model = build_model(
            architecture=self.architecture,
            pretrained=self.pretrained,
            num_classes=len(CLASS_LABELS)
        ).to(self.device)

        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        best_val_loss = float("inf")
        best_model_weights = copy.deepcopy(model.state_dict())
        patience_counter = 0

        train_history = []

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch in train_loader:
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == targets.data).item()
                total += targets.size(0)

            scheduler.step()

            epoch_train_loss = running_loss / max(total, 1)
            epoch_train_acc = correct / max(total, 1)

            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += torch.sum(preds == targets.data).item()
                    val_total += targets.size(0)

            epoch_val_loss = val_loss / max(val_total, 1)
            epoch_val_acc = val_correct / max(val_total, 1)

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(epoch_train_loss, 4),
                "train_accuracy": round(epoch_train_acc, 4),
                "val_loss": round(epoch_val_loss, 4),
                "val_accuracy": round(epoch_val_acc, 4),
            }
            train_history.append(epoch_record)

            # Checkpoint tracking
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    break

        # Load best weights
        model.load_state_dict(best_model_weights)

        # Save checkpoint weights
        checkpoint_filename = f"{self.model_version}.pt"
        checkpoint_path = self.checkpoints_dir / checkpoint_filename
        torch.save({
            "model_version": self.model_version,
            "architecture": self.architecture,
            "state_dict": model.state_dict(),
            "class_mapping": CLASS_TO_IDX,
            "config": self.config.__dict__,
        }, checkpoint_path)

        # Compute SHA-256 Checksum
        sha256 = compute_sha256(checkpoint_path)

        # Post-hoc Temperature Calibration
        optimal_temp, calib_report = calibrate_model(model, val_loader, self.device)

        # Comprehensive Evaluation on Test Set
        eval_metrics = evaluate_dataloader(
            model=model,
            dataloader=test_loader,
            device=self.device,
            temperature=optimal_temp
        )
        eval_metrics["calibration_tuning"] = calib_report
        eval_metrics["training_history"] = train_history

        # Save Evaluation Report
        eval_file = self.eval_dir / f"{self.model_version}_eval.json"
        with open(eval_file, "w", encoding="utf-8") as f:
            import json
            json.dump(eval_metrics, f, indent=2)

        # Construct and Save Metadata
        metadata = ModelMetadata(
            model_version=self.model_version,
            training_dataset_version=self.config.dataset_version,
            preprocessing_version=self.config.preprocessing_version,
            architecture=self.architecture,
            class_mapping=CLASS_TO_IDX,
            evaluation_metrics=eval_metrics,
            sha256_checksum=sha256,
            temperature=optimal_temp,
            hyperparameters={
                "epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": self.config.batch_size,
                "seed": self.config.seed,
            }
        )
        metadata_file = self.metadata_dir / f"{self.model_version}.json"
        metadata.save(metadata_file)

        return {
            "checkpoint_path": str(checkpoint_path),
            "metadata_path": str(metadata_file),
            "evaluation_path": str(eval_file),
            "sha256": sha256,
            "metrics": eval_metrics,
        }
