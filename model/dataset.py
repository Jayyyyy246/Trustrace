"""TRUSTTRACE Dataset configuration, validation, leak-free splitting, and preprocessing."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# Primary technical forensic classes
CLASS_LABELS: List[str] = [
    "REAL",
    "EDITED",
    "AI-GENERATED",
    "SCREENSHOT-MANIPULATED",
]

UNKNOWN_LABEL: str = "UNKNOWN"

CLASS_TO_IDX: Dict[str, int] = {label: idx for idx, label in enumerate(CLASS_LABELS)}
IDX_TO_CLASS: Dict[int, str] = {idx: label for label, idx in CLASS_TO_IDX.items()}

# Standard image extensions supported in digital forensics
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


@dataclass
class DatasetConfig:
    """Dataset configuration parameters for reproducible training & evaluation."""
    image_size: Tuple[int, int] = (224, 224)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    batch_size: int = 16
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    dataset_version: str = "trusttrace-ds-v1.0"
    preprocessing_version: str = "prep-standard-224x224-v1"
    group_separator: Optional[str] = None  # e.g., "__" to group variations of same source image


class CorruptImageError(ValueError):
    """Raised when an image file cannot be parsed or decoded."""
    pass


def compute_image_content_hash(file_path: Path) -> str:
    """Compute SHA-256 of raw image file bytes for strict deduplication."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_image_safely(file_path: str | Path) -> Image.Image:
    """Load an image file safely, convert to RGB, and handle EXIF orientation."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    try:
        with Image.open(path) as img:
            # Force decode of image data to verify integrity
            img.verify()

        # Reopen for actual processing (verify closes stream in PIL)
        with Image.open(path) as img:
            # Handle orientation if EXIF present
            try:
                img = ImageOps.exif_transpose(img)
            except Exception:
                pass
            rgb_img = img.convert("RGB")
            # Load pixel data into memory
            rgb_img.load()
            return rgb_img
    except Exception as exc:
        raise CorruptImageError(f"Corrupt or unreadable image '{path}': {exc}") from exc


def build_transforms(is_training: bool, config: Optional[DatasetConfig] = None) -> T.Compose:
    """Build standardized PyTorch vision transforms.
    
    Training: includes subtle forensic-safe augmentations (slight affine, horizontal flip, jitter).
    Evaluation / Inference: strictly deterministic resizing and normalization.
    """
    cfg = config or DatasetConfig()
    size = cfg.image_size
    mean = cfg.mean
    std = cfg.std

    if is_training:
        return T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08),
            T.RandomAffine(degrees=4, translate=(0.02, 0.02)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        return T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


def validate_dataset_directory(
    dataset_dir: str | Path,
    expected_classes: Sequence[str] = CLASS_LABELS
) -> Dict[str, any]:
    """Inspect and validate a dataset directory structure.
    
    Checks:
    - Root and class directories exist
    - Files are readable and uncorrupted
    - Duplicate content hashes across files
    - Per-class counts and distribution
    """
    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root directory does not exist: {root}")

    results = {
        "root": str(root),
        "classes_found": [],
        "classes_missing": [],
        "total_images": 0,
        "corrupt_images": [],
        "duplicate_hashes": {},
        "class_counts": {},
        "is_valid": True,
    }

    seen_hashes: Dict[str, str] = {}  # hash -> first_path

    for label in expected_classes:
        class_dir = root / label
        if not class_dir.is_dir():
            results["classes_missing"].append(label)
            results["class_counts"][label] = 0
            continue

        results["classes_found"].append(label)
        image_files = [
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ]
        results["class_counts"][label] = len(image_files)

        for img_path in image_files:
            results["total_images"] += 1
            # Check file readability
            try:
                img_hash = compute_image_content_hash(img_path)
                if img_hash in seen_hashes:
                    results["duplicate_hashes"].setdefault(img_hash, []).append(str(img_path))
                else:
                    seen_hashes[img_hash] = str(img_path)

                _ = load_image_safely(img_path)
            except CorruptImageError:
                results["corrupt_images"].append(str(img_path))
            except Exception:
                results["corrupt_images"].append(str(img_path))

    if results["classes_missing"] or len(results["corrupt_images"]) > 0:
        results["is_valid"] = False

    return results


@dataclass
class DatasetSample:
    """Individual dataset sample item."""
    path: Path
    label_name: str
    label_idx: int
    content_hash: str
    group_id: str  # Used for grouping to prevent data leakage


def collect_dataset_samples(
    dataset_dir: str | Path,
    expected_classes: Sequence[str] = CLASS_LABELS,
    group_separator: Optional[str] = None
) -> List[DatasetSample]:
    """Collect and deduplicate dataset samples from class subdirectories."""
    root = Path(dataset_dir)
    samples: List[DatasetSample] = []
    seen_hashes = set()

    for label in expected_classes:
        class_dir = root / label
        if not class_dir.is_dir():
            continue
        idx = CLASS_TO_IDX[label]

        for p in sorted(class_dir.iterdir()):
            if not p.is_file() or p.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
                continue

            try:
                chash = compute_image_content_hash(p)
                # Leakage prevention: exclude exact duplicates
                if chash in seen_hashes:
                    continue
                seen_hashes.add(chash)

                # Grouping ID (e.g. source image prefix if variations exist)
                stem = p.stem
                group_id = stem.split(group_separator)[0] if group_separator and group_separator in stem else stem

                samples.append(DatasetSample(
                    path=p,
                    label_name=label,
                    label_idx=idx,
                    content_hash=chash,
                    group_id=group_id
                ))
            except Exception:
                continue

    return samples


def create_leak_free_splits(
    samples: Sequence[DatasetSample],
    config: Optional[DatasetConfig] = None
) -> Tuple[List[DatasetSample], List[DatasetSample], List[DatasetSample]]:
    """Partition samples into train, validation, and test splits with zero data leakage.
    
    Guarantees:
    - Group-aware: samples with the same group_id are NEVER split across partitions.
    - Hash-distinct: identical image content hashes never cross boundaries.
    - Stratified: maintains balanced class distribution where possible.
    - Deterministic: reproducible via config.seed.
    """
    cfg = config or DatasetConfig()
    rng = np.random.RandomState(cfg.seed)

    # Group samples by group_id
    groups: Dict[str, List[DatasetSample]] = {}
    for s in samples:
        groups.setdefault(s.group_id, []).append(s)

    # Class-aware group assignment
    class_groups: Dict[int, List[str]] = {idx: [] for idx in range(len(CLASS_LABELS))}
    for gid, group_samples in groups.items():
        # Representative class of the group
        primary_class = group_samples[0].label_idx
        class_groups[primary_class].append(gid)

    train_samples: List[DatasetSample] = []
    val_samples: List[DatasetSample] = []
    test_samples: List[DatasetSample] = []

    for _, gids in class_groups.items():
        if not gids:
            continue
        gids_shuffled = list(gids)
        rng.shuffle(gids_shuffled)

        n = len(gids_shuffled)
        n_train = int(round(n * cfg.train_ratio))
        n_val = int(round(n * cfg.val_ratio))
        # Ensure at least 1 in train if n >= 1
        n_train = max(1, n_train) if n >= 1 else 0
        if n > 1 and n_train >= n:
            n_train = n - 1

        train_gids = set(gids_shuffled[:n_train])
        val_gids = set(gids_shuffled[n_train:n_train + n_val])
        test_gids = set(gids_shuffled[n_train + n_val:])

        for gid in train_gids:
            train_samples.extend(groups[gid])
        for gid in val_gids:
            val_samples.extend(groups[gid])
        for gid in test_gids:
            test_samples.extend(groups[gid])

    # Final check: strictly assert disjoint hashes across splits
    train_hashes = {s.content_hash for s in train_samples}
    val_hashes = {s.content_hash for s in val_samples}
    test_hashes = {s.content_hash for s in test_samples}

    assert len(train_hashes.intersection(val_hashes)) == 0, "Data leakage detected: train and val share image hashes"
    assert len(train_hashes.intersection(test_hashes)) == 0, "Data leakage detected: train and test share image hashes"
    assert len(val_hashes.intersection(test_hashes)) == 0, "Data leakage detected: val and test share image hashes"

    return train_samples, val_samples, test_samples


class ForensicDataset(Dataset):
    """PyTorch Dataset for forensic evidence images."""

    def __init__(
        self,
        samples: Sequence[DatasetSample],
        transform: Optional[Callable] = None
    ) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        sample = self.samples[idx]
        image = load_image_safely(sample.path)

        if self.transform is not None:
            tensor = self.transform(image)
        else:
            tensor = T.ToTensor()(image)

        return tensor, sample.label_idx, str(sample.path)
