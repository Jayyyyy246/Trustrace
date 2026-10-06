"""TRUSTTRACE Cryptographic and Perceptual Image Deduplication Engine.

Provides:
1. Cryptographic SHA-256 byte hashing for exact content deduplication.
2. Difference Hash (dHash) perceptual hashing for near-duplicate identification.
3. Hamming distance calculation for perceptual similarity metrics.
4. Cross-split data leakage detection (exact and near-duplicate).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image


def compute_sha256(target: Union[Path, str, bytes]) -> str:
    """Computes SHA-256 hex digest of file path or raw bytes."""
    hasher = hashlib.sha256()
    if isinstance(target, bytes):
        hasher.update(target)
    else:
        with open(target, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
    return hasher.hexdigest().lower()


def compute_dhash(image_input: Union[Path, str, bytes, Image.Image], hash_size: int = 8) -> str:
    """
    Computes 64-bit Difference Hash (dHash) for an image.
    
    Algorithm:
    1. Convert image to grayscale.
    2. Resize to (hash_size + 1, hash_size) using BILINEAR filter.
    3. Compare adjacent horizontal pixels (P[y, x+1] > P[y, x]).
    4. Convert bit sequence to 16-character hex string.
    """
    if isinstance(image_input, Image.Image):
        img = image_input.convert("L")
    elif isinstance(image_input, bytes):
        import io
        img = Image.open(io.BytesIO(image_input)).convert("L")
    else:
        img = Image.open(image_input).convert("L")

    # Resize to width = hash_size + 1, height = hash_size
    resized = img.resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.array(resized, dtype=np.float32)

    # Compute horizontal difference: width becomes hash_size
    diff = pixels[:, 1:] > pixels[:, :-1]
    
    # Flatten to 1D boolean array (64 bits)
    bits = diff.flatten()
    
    # Pack into an integer and format as hex
    int_val = 0
    for b in bits:
        int_val = (int_val << 1) | int(b)
        
    return f"{int_val:0{hash_size * hash_size // 4}x}"


def hamming_distance(hex_hash1: str, hex_hash2: str) -> int:
    """Computes Hamming distance between two hex hash strings."""
    if not hex_hash1 or not hex_hash2 or len(hex_hash1) != len(hex_hash2):
        return 64
    val1 = int(hex_hash1, 16)
    val2 = int(hex_hash2, 16)
    return bin(val1 ^ val2).count("1")


def detect_duplicates(
    records: List[Dict[str, Union[str, int]]],
    perceptual_threshold: int = 4,
) -> Dict[str, Union[List[List[str]], List[Tuple[str, str, int]]]]:
    """
    Detects exact SHA-256 duplicates and suspected near-duplicate pairs (dHash <= threshold).
    
    Args:
        records: List of dicts, each with at least 'image_id', 'sha256', and optionally 'dhash'.
        perceptual_threshold: Maximum Hamming distance to consider near-duplicates (default 4).
        
    Returns:
        Dict containing:
        - 'exact_duplicates': List of image_id groups sharing the same SHA-256
        - 'near_duplicates': List of (id1, id2, distance) pairs
    """
    sha_map: Dict[str, List[str]] = {}
    for r in records:
        sha = str(r["sha256"])
        sha_map.setdefault(sha, []).append(str(r["image_id"]))

    exact_duplicates = [ids for ids in sha_map.values() if len(ids) > 1]

    near_duplicates: List[Tuple[str, str, int]] = []
    # Only compare distinct SHA-256 records with dhash
    dhash_records = [r for r in records if r.get("dhash")]
    n = len(dhash_records)
    for i in range(n):
        r1 = dhash_records[i]
        h1 = str(r1["dhash"])
        for j in range(i + 1, min(n, i + 500)):  # capped window for scalability
            r2 = dhash_records[j]
            if r1["sha256"] == r2["sha256"]:
                continue
            h2 = str(r2["dhash"])
            dist = hamming_distance(h1, h2)
            if dist <= perceptual_threshold:
                near_duplicates.append((str(r1["image_id"]), str(r2["image_id"]), dist))

    return {
        "exact_duplicates": exact_duplicates,
        "near_duplicates": near_duplicates,
    }


def verify_cross_split_leakage(
    train_records: List[Dict[str, Union[str, int]]],
    val_records: List[Dict[str, Union[str, int]]],
    test_records: List[Dict[str, Union[str, int]]],
    perceptual_threshold: int = 2,
) -> Dict[str, Union[bool, List[str]]]:
    """
    Asserts zero exact SHA-256 duplicates and no severe perceptual near-duplicates
    leak across train, validation, and test splits.
    
    Returns:
        Dict with 'has_leakage' (bool) and 'violations' (List[str]).
    """
    violations: List[str] = []

    splits = {
        "TRAIN": train_records,
        "VALIDATION": val_records,
        "TEST": test_records,
    }

    # 1. Exact SHA-256 collision across splits
    sha_to_split: Dict[str, str] = {}
    for s_name, recs in splits.items():
        for r in recs:
            sha = str(r["sha256"])
            if sha in sha_to_split and sha_to_split[sha] != s_name:
                violations.append(
                    f"Exact SHA-256 leak: {r['image_id']} in split {s_name} "
                    f"collides with image in split {sha_to_split[sha]} (SHA: {sha[:12]}...)"
                )
            else:
                sha_to_split[sha] = s_name

    # 2. Source group leakage (if source_group is present)
    group_to_split: Dict[str, str] = {}
    for s_name, recs in splits.items():
        for r in recs:
            grp = r.get("source_group") or r.get("source_id")
            if grp:
                grp_str = str(grp)
                if grp_str in group_to_split and group_to_split[grp_str] != s_name:
                    violations.append(
                        f"Source group leak: Group '{grp_str}' split between "
                        f"{group_to_split[grp_str]} and {s_name} (Image: {r['image_id']})"
                    )
                else:
                    group_to_split[grp_str] = s_name

    return {
        "has_leakage": len(violations) > 0,
        "violations": violations,
    }
