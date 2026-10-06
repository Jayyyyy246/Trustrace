"""
MODULE 1 — HASHING
Cryptographic and integrity verification analyzer.
Computes SHA-256, SHA-512, MD5, and perceptual hash digests.
"""
import hashlib
import io
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image

from app.schemas.common import FindingSeverity
from app.schemas.forensic import FindingItem, BaseAnalyzerResult


class HashResult(BaseAnalyzerResult):
    algorithms: List[Dict[str, str]] = []


class ForensicHashingAnalyzer:
    """Deterministic hashing engine for evidence chain-of-custody."""

    def compute_sha256(self, data: bytes) -> str:
        """Computes NIST FIPS 180-4 SHA-256 digest."""
        return hashlib.sha256(data).hexdigest()

    def compute_sha512(self, data: bytes) -> str:
        """Computes NIST FIPS 180-4 SHA-512 digest."""
        return hashlib.sha512(data).hexdigest()

    def compute_sha1(self, data: bytes) -> str:
        """Computes SHA-1 digest."""
        return hashlib.sha1(data).hexdigest()

    def compute_md5(self, data: bytes) -> str:
        """Computes legacy MD5 digest for cross-verification."""
        return hashlib.md5(data).hexdigest()

    def compute_phash(self, data: bytes) -> Optional[str]:
        """Computes average 64-bit perceptual hash."""
        try:
            with Image.open(io.BytesIO(data)) as img:
                gray = img.convert("L").resize((8, 8), Image.Resampling.BILINEAR)
                pixels = np.array(gray, dtype=np.float32).flatten()
                avg = pixels.mean()
                bits = "".join("1" if p > avg else "0" for p in pixels)
                return f"{int(bits, 2):016x}"
        except Exception:
            return None

    def analyze(self, data: bytes) -> HashResult:
        """
        Executes full cryptographic and perceptual hash suite.
        Returns standardized analyzer response.
        """
        sha256_val = self.compute_sha256(data)
        sha512_val = self.compute_sha512(data)
        md5_val = self.compute_md5(data)
        phash_val = self.compute_phash(data)

        algorithms = [
            {"algorithm": "SHA-256", "hash": sha256_val, "status": "COMPLETED"},
            {"algorithm": "SHA-512", "hash": sha512_val, "status": "COMPLETED"},
            {"algorithm": "MD5", "hash": md5_val, "status": "COMPLETED"},
        ]
        if phash_val:
            algorithms.append({"algorithm": "pHash", "hash": phash_val, "status": "COMPLETED"})

        metrics = {
            "sha256": sha256_val,
            "sha512": sha512_val,
            "md5": md5_val,
            "phash": phash_val,
            "byte_size": len(data),
        }

        findings = [
            FindingItem(
                finding_id="FIND-HASH-001",
                category="HASHING",
                severity=FindingSeverity.INFO,
                title="Cryptographic Chain-of-Custody Established",
                description=f"Evidence fixed with immutable SHA-256: {sha256_val[:16]}...",
                evidence={"sha256": sha256_val, "sha512": sha512_val},
                interpretation="Cryptographic digest uniquely identifies this exact byte sequence for custody tracking.",
                limitation="A cryptographic hash verifies subsequent integrity; it cannot detect modifications made before ingestion.",
                confidence=1.0,
                is_anomaly=False,
            )
        ]

        limitations = [
            "Hashing validates that data has not changed since ingestion. It does not measure pre-ingestion authenticity.",
            "Perceptual hashing (pHash) is heuristic and may fail on major cropping or extreme aspect ratio distortions.",
        ]

        return HashResult(
            status="completed",
            findings=findings,
            metrics=metrics,
            limitations=limitations,
            algorithms=algorithms,
        )


hashing_analyzer = ForensicHashingAnalyzer()
