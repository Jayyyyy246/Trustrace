"""
Cryptographic and perceptual hashing service for digital evidence verification.
Guarantees evidence integrity and produces cryptographic digests.
"""
import hashlib
import io
from typing import Optional
from PIL import Image
import numpy as np

from app.schemas.evidence import EvidenceHashes


class HashingService:
    @staticmethod
    def compute_sha256(data: bytes) -> str:
        """Computes NIST-standard SHA-256 hash."""
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def compute_md5(data: bytes) -> str:
        """Computes MD5 hash."""
        return hashlib.md5(data).hexdigest()

    @staticmethod
    def compute_sha1(data: bytes) -> str:
        """Computes SHA-1 hash."""
        return hashlib.sha1(data).hexdigest()

    @staticmethod
    def compute_phash(data: bytes) -> Optional[str]:
        """
        Computes a perceptual hash (average perceptual DCT-style hash)
        using Pillow and NumPy. Resilient against minor re-compression.
        """
        try:
            with Image.open(io.BytesIO(data)) as img:
                # Convert to grayscale and resize to 8x8
                gray = img.convert("L").resize((8, 8), Image.Resampling.BILINEAR)
                pixels = np.array(gray, dtype=np.float32).flatten()
                avg = pixels.mean()
                # Compute bitmask where pixel > average
                bits = "".join("1" if p > avg else "0" for p in pixels)
                # Convert 64 bits to 16-character hex string
                hex_str = f"{int(bits, 2):016x}"
                return hex_str
        except Exception:
            return None

    def compute_all_hashes(self, data: bytes) -> EvidenceHashes:
        """Computes SHA-256, MD5, SHA-1, and pHash for an evidence buffer."""
        return EvidenceHashes(
            sha256=self.compute_sha256(data),
            md5=self.compute_md5(data),
            sha1=self.compute_sha1(data),
            phash=self.compute_phash(data),
        )

    def verify_integrity(self, data: bytes, expected_sha256: str) -> bool:
        """Validates that evidence has not mutated against expected SHA-256."""
        actual = self.compute_sha256(data)
        return actual.lower() == expected_sha256.lower()


hashing_service = HashingService()
