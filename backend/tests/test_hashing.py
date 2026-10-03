"""
Tests for cryptographic and perceptual hashing.
"""
from app.services.hashing_service import hashing_service


def test_sha256_reproducibility(sample_png_bytes: bytes):
    hash1 = hashing_service.compute_sha256(sample_png_bytes)
    hash2 = hashing_service.compute_sha256(sample_png_bytes)
    assert hash1 == hash2
    assert len(hash1) == 64
    assert all(c in "0123456789abcdef" for c in hash1)


def test_hash_integrity_verification(sample_jpeg_bytes: bytes):
    correct_hash = hashing_service.compute_sha256(sample_jpeg_bytes)
    assert hashing_service.verify_integrity(sample_jpeg_bytes, correct_hash) is True
    assert hashing_service.verify_integrity(sample_jpeg_bytes, "0" * 64) is False


def test_compute_all_hashes(sample_png_bytes: bytes):
    hashes = hashing_service.compute_all_hashes(sample_png_bytes)
    assert len(hashes.sha256) == 64
    assert len(hashes.md5) == 32
    assert len(hashes.sha1) == 40
    assert hashes.phash is not None
