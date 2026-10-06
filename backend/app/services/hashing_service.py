"""
Cryptographic and perceptual hashing service for digital evidence verification.
Guarantees evidence integrity and produces cryptographic digests.
"""
from typing import Optional
from app.schemas.evidence import EvidenceHashes
from forensic.hashing import hashing_analyzer


class HashingService:
    @staticmethod
    def compute_sha256(data: bytes) -> str:
        """Computes NIST-standard SHA-256 hash via canonical forensic hasher."""
        return hashing_analyzer.compute_sha256(data)

    @staticmethod
    def compute_md5(data: bytes) -> str:
        """Computes MD5 hash via canonical forensic hasher."""
        return hashing_analyzer.compute_md5(data)

    @staticmethod
    def compute_sha1(data: bytes) -> str:
        """Computes SHA-1 hash via canonical forensic hasher."""
        return hashing_analyzer.compute_sha1(data)

    @staticmethod
    def compute_phash(data: bytes) -> Optional[str]:
        """Computes perceptual hash via canonical forensic hasher."""
        return hashing_analyzer.compute_phash(data)

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

