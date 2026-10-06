"""Tests for TRUSTTRACE Dataset Ingestion, Manifest Generation, Deduplication, and Quality Gate Validation."""

import csv
import io
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict

import pytest
from PIL import Image

from data.deduplication import compute_sha256, compute_dhash, hamming_distance, detect_duplicates, verify_cross_split_leakage
from data.ingest_dataset import validate_image_file, map_source_to_class, ingest_dataset, TARGET_CLASSES, MANIFEST_FIELDNAMES
from data.generate_screenshot_manipulations import generate_manipulation, TRANSFORMATION_TYPES
from data.validate_dataset import validate_dataset_quality
from train_pipeline import run_pipeline


@pytest.fixture
def temp_workspace():
    """Provides an isolated temporary directory for dataset tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


def create_dummy_image(path: Path, width: int = 100, height: int = 100, color: tuple = (200, 200, 200), fmt: str = "PNG"):
    """Helper to create a decodable dummy image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), color=color)
    img.save(path, format=fmt)


# =====================================================================
# 1. Deduplication and Hashing Tests
# =====================================================================

def test_sha256_and_dhash(temp_workspace: Path):
    img_path = temp_workspace / "test.png"
    create_dummy_image(img_path, width=64, height=64, color=(100, 150, 200))

    sha = compute_sha256(img_path)
    assert len(sha) == 64
    assert all(c in "0123456789abcdef" for c in sha)

    dh = compute_dhash(img_path)
    assert len(dh) == 16
    assert all(c in "0123456789abcdef" for c in dh)

    # Identical image yields 0 hamming distance
    assert hamming_distance(dh, dh) == 0


def test_leakage_detection_finds_collisions():
    train = [{"image_id": "img1", "sha256": "hash_a", "source_id": "src1"}]
    val = [{"image_id": "img2", "sha256": "hash_b", "source_id": "src2"}]
    test = [{"image_id": "img3", "sha256": "hash_a", "source_id": "src3"}]  # Collides with train

    res = verify_cross_split_leakage(train, val, test)
    assert res["has_leakage"] is True
    assert any("Exact SHA-256 leak" in v for v in res["violations"])


# =====================================================================
# 2. Validation & Rejection Rules Tests
# =====================================================================

def test_validate_image_file_rejections(temp_workspace: Path):
    # Non-existent
    ok, reason, _ = validate_image_file(temp_workspace / "non_existent.png")
    assert ok is False
    assert "not exist" in reason.lower()

    # Unsupported extension
    txt_path = temp_workspace / "bad.txt"
    txt_path.write_text("not an image")
    ok, reason, _ = validate_image_file(txt_path)
    assert ok is False
    assert "unsupported extension" in reason.lower()

    # Empty file
    empty_png = temp_workspace / "empty.png"
    empty_png.touch()
    ok, reason, _ = validate_image_file(empty_png)
    assert ok is False
    assert "empty" in reason.lower()

    # Corrupt header
    corrupt_png = temp_workspace / "corrupt.png"
    corrupt_png.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\x00CORRUPT")
    ok, reason, sha = validate_image_file(corrupt_png)
    assert ok is False
    assert "corrupt" in reason.lower() or "indecodable" in reason.lower()
    assert sha is not None  # SHA-256 still calculable for audit trail


# =====================================================================
# 3. Source Provenance & Class Mapping Tests
# =====================================================================

def test_source_mapping_verified_and_unmapped():
    # Verified sources
    cls, err = map_source_to_class("dresden")
    assert cls == "REAL"
    assert err is None

    cls, err = map_source_to_class("casia_v2")
    assert cls == "EDITED"
    assert err is None

    cls, err = map_source_to_class("cifake")
    assert cls == "AI-GENERATED"
    assert err is None

    cls, err = map_source_to_class("trusttrace_screenshot_manipulation")
    assert cls == "SCREENSHOT-MANIPULATED"
    assert err is None

    # Unmapped / unverified source
    cls, err = map_source_to_class("random_internet_scrape_2026")
    assert cls is None
    assert "UNMAPPED" in err


# =====================================================================
# 4. Controlled Screenshot Manipulation Generator Tests
# =====================================================================

def test_screenshot_manipulation_generator(temp_workspace: Path):
    src_img = temp_workspace / "base_receipt.png"
    create_dummy_image(src_img, width=400, height=800, color=(250, 250, 250))
    out_dir = temp_workspace / "output_tampered"

    for tf in TRANSFORMATION_TYPES:
        prov = generate_manipulation(src_img, out_dir, tf)
        assert prov["transformation_type"] == tf
        assert prov["target_class"] == "SCREENSHOT-MANIPULATED"
        assert prov["source_image_id"] == src_img.name
        assert prov["source_image_sha256"] != prov["generated_image_sha256"]
        assert Path(prov["generated_image_path"]).is_file()

        # Check output decodability
        with Image.open(prov["generated_image_path"]) as img:
            assert img.size == (400, 800)


# =====================================================================
# 5. Ingestion Pipeline & Manifest Generation Tests
# =====================================================================

def test_ingest_dataset_full_flow(temp_workspace: Path):
    src_dir = temp_workspace / "source_casia"
    # Create 5 sample images
    for i in range(5):
        create_dummy_image(src_dir / f"casia_splice_{i}.png", width=64, height=64, color=(i * 40, 100, 100))

    ds_dir = temp_workspace / "dataset"
    man_dir = temp_workspace / "manifests"

    report = ingest_dataset(
        source_dir=src_dir,
        dataset_dir=ds_dir,
        manifests_dir=man_dir,
        source_name="casia_v2",
        copy_to_dataset=True,
    )

    assert report["total_images"] == 5
    assert report["classes"]["EDITED"] == 5

    # Check manifest files exist and are populated
    manifest_csv = man_dir / "dataset_manifest.csv"
    train_csv = man_dir / "train.csv"
    val_csv = man_dir / "validation.csv"
    test_csv = man_dir / "test.csv"
    rejected_csv = man_dir / "rejected.csv"

    for p in (manifest_csv, train_csv, val_csv, test_csv, rejected_csv):
        assert p.is_file()

    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == MANIFEST_FIELDNAMES
        rows = list(reader)
        assert len(rows) == 5
        assert all(r["target_class"] == "EDITED" for r in rows)


# =====================================================================
# 6. Quality Gate & Training Refusal Tests
# =====================================================================

def test_validate_dataset_quality_fails_on_empty(temp_workspace: Path):
    ds_dir = temp_workspace / "empty_dataset"
    for c in TARGET_CLASSES:
        (ds_dir / c).mkdir(parents=True, exist_ok=True)

    res = validate_dataset_quality(ds_dir, min_samples_per_class=5)
    assert res["is_valid"] is False
    assert any("minimum sample requirement" in err for err in res["errors"])


def test_validate_dataset_rejects_prohibited_unknown_dir(temp_workspace: Path):
    ds_dir = temp_workspace / "bad_dataset"
    for c in TARGET_CLASSES + ["UNKNOWN"]:
        (ds_dir / c).mkdir(parents=True, exist_ok=True)
        create_dummy_image(ds_dir / c / "sample.png", width=64, height=64)

    res = validate_dataset_quality(ds_dir, min_samples_per_class=1)
    assert res["is_valid"] is False
    assert any("UNKNOWN" in err for err in res["errors"])


def test_train_pipeline_stops_on_incomplete_dataset(temp_workspace: Path):
    ds_dir = temp_workspace / "dataset"
    for c in TARGET_CLASSES:
        (ds_dir / c).mkdir(parents=True, exist_ok=True)

    out_dir = temp_workspace / "model_out"

    # Must exit with non-zero status and refuse training
    exit_code = run_pipeline(data_dir=ds_dir, output_dir=out_dir)
    assert exit_code == 1
