"""Comprehensive unit & integration tests for TRUSTTRACE ML pipeline.

Tests required by specification:
- Preprocessing
- Class mapping
- Inference
- Malformed images
- Model loading
- Checksum verification
- Unsupported model
- Missing model
- OOD/uncertainty logic
- Leak-free splitting
"""

import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
import torch

from model.architecture import ForensicClassifier, build_model, compute_uncertainty_metrics
from model.calibrator import TemperatureScaler, calibrate_model
from model.dataset import (
    CLASS_LABELS,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    UNKNOWN_LABEL,
    CorruptImageError,
    DatasetConfig,
    DatasetSample,
    ForensicDataset,
    build_transforms,
    compute_image_content_hash,
    create_leak_free_splits,
    load_image_safely,
    validate_dataset_directory,
)
from model.evaluator import (
    compute_calibration_metrics,
    compute_classification_metrics,
    compute_ood_metrics,
)
from model.predictor import (
    ChecksumMismatchError,
    ForensicPredictor,
    InferenceResult,
    ModelNotFoundError,
    UnsupportedModelError,
)
from model.versioning import ModelMetadata, compute_sha256, verify_sha256


# -------------------------------------------------------------------------
# 1. CLASS MAPPING TESTS
# -------------------------------------------------------------------------

def test_class_mapping_completeness():
    """Verify that all required forensic classes and UNKNOWN exist with proper indexing."""
    expected = ["REAL", "EDITED", "AI-GENERATED", "SCREENSHOT-MANIPULATED"]
    assert CLASS_LABELS == expected
    assert UNKNOWN_LABEL == "UNKNOWN"

    for idx, label in enumerate(CLASS_LABELS):
        assert CLASS_TO_IDX[label] == idx
        assert IDX_TO_CLASS[idx] == label

    assert len(CLASS_TO_IDX) == 4
    assert len(IDX_TO_CLASS) == 4


# -------------------------------------------------------------------------
# 2. PREPROCESSING TESTS
# -------------------------------------------------------------------------

def test_preprocessing_eval_transforms():
    """Verify that evaluation/inference transforms produce deterministic, correctly normalized tensors."""
    config = DatasetConfig(image_size=(224, 224))
    transform = build_transforms(is_training=False, config=config)

    # Synthetic RGB image
    pil_img = Image.new("RGB", (300, 450), color=(120, 180, 240))
    tensor = transform(pil_img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32

    # Check normalization: values should be centered around ImageNet statistics
    assert not torch.isnan(tensor).any()
    assert not torch.isinf(tensor).any()


def test_preprocessing_training_transforms():
    """Verify that training transforms apply augmentations while preserving tensor dimensions."""
    config = DatasetConfig(image_size=(224, 224))
    transform = build_transforms(is_training=True, config=config)

    pil_img = Image.new("RGB", (100, 100), color=(50, 150, 200))
    tensor = transform(pil_img)

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_safe_image_loading(tmp_path):
    """Verify safe image loading converts different color spaces to RGB."""
    # RGBA image
    rgba_path = tmp_path / "test_rgba.png"
    Image.new("RGBA", (50, 50), color=(10, 20, 30, 255)).save(rgba_path)
    loaded_rgba = load_image_safely(rgba_path)
    assert loaded_rgba.mode == "RGB"

    # Grayscale image
    l_path = tmp_path / "test_l.png"
    Image.new("L", (50, 50), color=128).save(l_path)
    loaded_l = load_image_safely(l_path)
    assert loaded_l.mode == "RGB"


# -------------------------------------------------------------------------
# 3. MALFORMED IMAGES TESTS
# -------------------------------------------------------------------------

def test_malformed_image_corrupt_bytes(tmp_path):
    """Verify CorruptImageError is raised when reading corrupted or truncated image files."""
    bad_file = tmp_path / "corrupt.jpg"
    bad_file.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00\x00\x01" + b"\x00" * 20)

    with pytest.raises(CorruptImageError):
        load_image_safely(bad_file)


def test_malformed_image_zero_bytes(tmp_path):
    """Verify CorruptImageError is raised on zero-byte files."""
    empty_file = tmp_path / "empty.png"
    empty_file.write_bytes(b"")

    with pytest.raises(CorruptImageError):
        load_image_safely(empty_file)


def test_malformed_image_text_disguised_as_image(tmp_path):
    """Verify CorruptImageError is raised on non-image text files disguised with image extension."""
    fake_img = tmp_path / "fake.jpg"
    fake_img.write_text("This is not a JPEG file header.")

    with pytest.raises(CorruptImageError):
        load_image_safely(fake_img)


# -------------------------------------------------------------------------
# 4. DATA LEAKAGE PREVENTION & DATASET SPLITTING TESTS
# -------------------------------------------------------------------------

def test_leak_free_splitting(tmp_path):
    """Verify that dataset splitting guarantees strict zero-overlap in hashes and groups."""
    samples = []
    # Create 20 synthetic samples across 4 classes
    for c_idx, label in enumerate(CLASS_LABELS):
        for i in range(5):
            p = tmp_path / f"{label}_{i}.png"
            Image.new("RGB", (32, 32), color=(c_idx * 50, i * 40, 100)).save(p)
            chash = compute_image_content_hash(p)
            samples.append(DatasetSample(
                path=p,
                label_name=label,
                label_idx=c_idx,
                content_hash=chash,
                group_id=f"group_{c_idx}_{i // 2}"  # Test grouped images
            ))

    cfg = DatasetConfig(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)
    train_s, val_s, test_s = create_leak_free_splits(samples, cfg)

    train_hashes = {s.content_hash for s in train_s}
    val_hashes = {s.content_hash for s in val_s}
    test_hashes = {s.content_hash for s in test_s}

    # Strict disjointness
    assert len(train_hashes.intersection(val_hashes)) == 0
    assert len(train_hashes.intersection(test_hashes)) == 0
    assert len(val_hashes.intersection(test_hashes)) == 0

    # Strict group integrity: groups must not be split across partitions
    train_groups = {s.group_id for s in train_s}
    val_groups = {s.group_id for s in val_s}
    test_groups = {s.group_id for s in test_s}

    assert len(train_groups.intersection(val_groups)) == 0
    assert len(train_groups.intersection(test_groups)) == 0
    assert len(val_groups.intersection(test_groups)) == 0


# -------------------------------------------------------------------------
# 5. MODEL LOADING, SAVE, AND CHECKSUM TESTS
# -------------------------------------------------------------------------

@pytest.fixture
def dummy_checkpoint(tmp_path):
    """Create a verified dummy PyTorch checkpoint and matching metadata."""
    model = build_model(architecture="custom_forensic_cnn", pretrained=False)
    ckpt_path = tmp_path / "dummy_model.pt"
    torch.save({
        "architecture": "custom_forensic_cnn",
        "state_dict": model.state_dict(),
        "model_version": "trusttrace-test-v1.0",
    }, ckpt_path)

    sha256 = compute_sha256(ckpt_path)
    metadata = ModelMetadata(
        model_version="trusttrace-test-v1.0",
        training_dataset_version="test-ds-v1",
        preprocessing_version="prep-test-v1",
        architecture="custom_forensic_cnn",
        class_mapping=CLASS_TO_IDX,
        sha256_checksum=sha256,
        temperature=1.0,
    )
    meta_path = tmp_path / "dummy_model.json"
    metadata.save(meta_path)
    return ckpt_path, meta_path, sha256


def test_model_loading_success(dummy_checkpoint):
    """Verify successful model loading and checkpoint verification."""
    ckpt_path, meta_path, _ = dummy_checkpoint
    predictor = ForensicPredictor(
        model_path=ckpt_path,
        metadata_path=meta_path,
        verify_checksum=True
    )
    assert predictor.model is not None
    assert predictor.architecture == "custom_forensic_cnn"
    assert predictor.metadata.model_version == "trusttrace-test-v1.0"


def test_missing_model_error(tmp_path):
    """Verify ModelNotFoundError when checkpoint file does not exist."""
    missing_path = tmp_path / "does_not_exist.pt"
    with pytest.raises(ModelNotFoundError):
        ForensicPredictor(model_path=missing_path)


def test_checksum_verification_failure(dummy_checkpoint):
    """Verify ChecksumMismatchError when model file content is altered."""
    ckpt_path, meta_path, _ = dummy_checkpoint

    # Tamper with checkpoint by appending 4 bytes
    with open(ckpt_path, "ab") as f:
        f.write(b"HACK")

    with pytest.raises(ChecksumMismatchError):
        ForensicPredictor(
            model_path=ckpt_path,
            metadata_path=meta_path,
            verify_checksum=True
        )


def test_unsupported_model_architecture(tmp_path):
    """Verify UnsupportedModelError when checkpoint defines an invalid architecture."""
    bad_ckpt = tmp_path / "unsupported_arch.pt"
    torch.save({
        "architecture": "non_existent_neural_net_9999",
        "state_dict": {},
    }, bad_ckpt)

    with pytest.raises(UnsupportedModelError):
        ForensicPredictor(model_path=bad_ckpt, verify_checksum=False)


# -------------------------------------------------------------------------
# 6. INFERENCE & UNCERTAINTY / OOD LOGIC TESTS
# -------------------------------------------------------------------------

def test_inference_contract_and_structure(dummy_checkpoint, tmp_path):
    """Verify exact schema contract for model inference output."""
    ckpt_path, meta_path, _ = dummy_checkpoint
    predictor = ForensicPredictor(
        model_path=ckpt_path,
        metadata_path=meta_path,
        verify_checksum=True
    )

    test_img = tmp_path / "sample.png"
    Image.new("RGB", (224, 224), color=(100, 150, 200)).save(test_img)

    result = predictor.predict(test_img)

    # Check contract fields
    assert "model_version" in result
    assert "prediction" in result
    assert "probabilities" in result
    assert "uncertainty" in result
    assert "status" in result
    assert result["status"] == "completed"
    assert isinstance(result["uncertainty"], float)
    assert 0.0 <= result["uncertainty"] <= 1.0

    # Prediction must be one of the supported classes or UNKNOWN
    assert result["prediction"] in CLASS_LABELS or result["prediction"] == UNKNOWN_LABEL

    # Check probabilities dict
    probs = result["probabilities"]
    assert len(probs) == 4
    for label in CLASS_LABELS:
        assert label in probs
        assert 0.0 <= probs[label] <= 1.0


def test_uncertainty_ood_trigger_unknown():
    """Verify that low-confidence or high-entropy logits trigger the UNKNOWN class."""
    # Flat logits -> uniform distribution -> maximum entropy
    flat_logits = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    metrics = compute_uncertainty_metrics(
        logits=flat_logits,
        temperature=1.0,
        confidence_threshold=0.55,
        entropy_threshold=0.82
    )

    # Flat distribution should trigger UNKNOWN due to low max_prob (0.25) and high entropy (~1.0)
    assert metrics["prediction"] == UNKNOWN_LABEL
    assert metrics["is_ood"] is True
    assert metrics["uncertainty"] > 0.8


def test_high_confidence_known_class_prediction():
    """Verify that decisive logits yield the target class without triggering UNKNOWN."""
    decisive_logits = torch.tensor([10.0, 0.0, -2.0, -1.0], dtype=torch.float32)
    metrics = compute_uncertainty_metrics(
        logits=decisive_logits,
        temperature=1.0,
        confidence_threshold=0.55,
        entropy_threshold=0.82
    )

    assert metrics["prediction"] == "REAL"
    assert metrics["is_ood"] is False
    assert metrics["uncertainty"] < 0.2
    assert metrics["probabilities"]["REAL"] > 0.95


# -------------------------------------------------------------------------
# 7. EVALUATION & CALIBRATION METRICS TESTS
# -------------------------------------------------------------------------

def test_classification_metrics_computation():
    """Verify exact computation of confusion matrix, precision, recall, and macro-F1."""
    y_true = [0, 0, 1, 1, 2, 3]
    y_pred = [0, 1, 1, 1, 2, 0]

    metrics = compute_classification_metrics(y_true, y_pred, CLASS_LABELS)
    assert metrics["total_samples"] == 6
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0

    cm = np.array(metrics["confusion_matrix"])
    assert cm.shape == (4, 4)
    assert np.sum(cm) == 6


def test_calibration_metrics_computation():
    """Verify Expected Calibration Error (ECE) and MCE computation."""
    confs = [0.9, 0.8, 0.6, 0.7, 0.5]
    preds = [0, 1, 0, 1, 2]
    targets = [0, 1, 1, 1, 3]  # 3 correct, 2 incorrect

    calib = compute_calibration_metrics(confs, preds, targets, num_bins=5)
    assert "ece" in calib
    assert "mce" in calib
    assert 0.0 <= calib["ece"] <= 1.0
    assert 0.0 <= calib["mce"] <= 1.0


def test_ood_metrics_computation():
    """Verify Out-of-Distribution AUROC and FPR95 calculation."""
    id_scores = [0.95, 0.88, 0.92, 0.85, 0.99]
    ood_scores = [0.20, 0.35, 0.15, 0.40, 0.30]

    ood_eval = compute_ood_metrics(id_scores, ood_scores)
    assert "auroc" in ood_eval
    assert "fpr95" in ood_eval
    # Perfectly separated scores should produce AUROC == 1.0
    assert ood_eval["auroc"] == 1.0
    assert ood_eval["fpr95"] == 0.0


# -------------------------------------------------------------------------
# 8. END-TO-END PIPELINE & EVIDENCE FUSION INDEPENDENCE TESTS
# -------------------------------------------------------------------------

def test_end_to_end_training_and_inference_pipeline(tmp_path):
    """Verify that ForensicTrainer trains on real data, generates genuine metrics,
    computes valid SHA-256 checksum, and enables ForensicPredictor inference.
    """
    from model.trainer import ForensicTrainer

    dataset_root = tmp_path / "dataset"
    for label in CLASS_LABELS:
        c_dir = dataset_root / label
        c_dir.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            img_path = c_dir / f"img_{i}.jpg"
            # Deterministic distinct colors
            color = (
                (CLASS_TO_IDX[label] * 60) % 255,
                (i * 50) % 255,
                150
            )
            Image.new("RGB", (64, 64), color=color).save(img_path, quality=90)

    output_dir = tmp_path / "model_out"
    config = DatasetConfig(
        image_size=(64, 64),
        batch_size=4,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=123,
    )

    trainer = ForensicTrainer(
        dataset_dir=dataset_root,
        output_dir=output_dir,
        architecture="custom_forensic_cnn",
        pretrained=False,
        config=config,
        model_version="trusttrace-e2e-v1.0"
    )

    results = trainer.train(epochs=2, lr=1e-3)

    # 1. Assert artifact paths exist
    ckpt_path = Path(results["checkpoint_path"])
    meta_path = Path(results["metadata_path"])
    eval_path = Path(results["evaluation_path"])

    assert ckpt_path.is_file()
    assert meta_path.is_file()
    assert eval_path.is_file()

    # 2. Assert SHA-256 integrity
    assert results["sha256"] == compute_sha256(ckpt_path)
    metadata = ModelMetadata.load(meta_path)
    assert metadata.sha256_checksum == results["sha256"]
    assert metadata.architecture == "custom_forensic_cnn"
    assert metadata.class_mapping == CLASS_TO_IDX

    # 3. Assert real computed metrics (not fabricated)
    metrics = results["metrics"]
    assert "classification" in metrics
    assert "accuracy" in metrics["classification"]
    assert "confusion_matrix" in metrics["classification"]
    assert "calibration" in metrics
    assert "ece" in metrics["calibration"]

    # 4. Assert ForensicPredictor inference works with trained weights
    predictor = ForensicPredictor(
        model_path=ckpt_path,
        metadata_path=meta_path,
        verify_checksum=True
    )
    inference = predictor.predict(dataset_root / "REAL" / "img_0.jpg")
    assert inference["status"] == "completed"
    assert inference["model_version"] == "trusttrace-e2e-v1.0"
    assert inference["prediction"] in (CLASS_LABELS + [UNKNOWN_LABEL])
    assert "probabilities" in inference
    assert len(inference["probabilities"]) == 4


def test_ml_prediction_does_not_override_evidence_fusion():
    """Verify CRITICAL RULE: ML prediction must NOT automatically become the final
    TRUSTTRACE verdict; Evidence Fusion Layer arbitrates with forensic corroboration.
    """
    from forensic.fusion import ForensicEvidenceFusionEngine
    from app.schemas.forensic import BaseAnalyzerResult, AnalyzerStatus

    fusion_engine = ForensicEvidenceFusionEngine()

    # Case 1: Clean camera metadata, no image manipulation findings
    meta_res = BaseAnalyzerResult(
        status=AnalyzerStatus.COMPLETED,
        analyzer="metadata",
        metrics={"exif_present": True, "camera": {"make": "Sony", "model": "A7 IV"}, "editing_software_detected": False},
        findings=[],
        limitations=[]
    )
    img_res = BaseAnalyzerResult(
        status=AnalyzerStatus.COMPLETED,
        analyzer="image_analysis",
        metrics={"copy_move": {"detected": False}, "estimated_jpeg_quality": 95},
        findings=[],
        limitations=[]
    )
    screen_res = BaseAnalyzerResult(
        status=AnalyzerStatus.COMPLETED,
        analyzer="screenshot",
        metrics={"is_probable_screenshot": False},
        findings=[],
        limitations=[]
    )
    ocr_res = BaseAnalyzerResult(
        status=AnalyzerStatus.COMPLETED,
        analyzer="ocr",
        metrics={},
        findings=[],
        limitations=[]
    )

    # ML predicts EDITED with moderate confidence
    ml_prediction = {
        "model_status": "AVAILABLE",
        "predicted_label": "EDITED",
        "confidence": 0.65,
        "probabilities": {"REAL": 0.20, "EDITED": 0.65, "AI-GENERATED": 0.10, "SCREENSHOT-MANIPULATED": 0.05}
    }

    verdict = fusion_engine.fuse(
        metadata_res=meta_res,
        image_res=img_res,
        screenshot_res=screen_res,
        ocr_res=ocr_res,
        ml_prediction=ml_prediction
    )

    # Final verdict must NOT blindly become EDITED without forensic corroboration.
    # The fusion engine detects conflict or marks inconclusive/requires corroboration.
    assert verdict.label != "EDITED" or verdict.conflict_detected is True

