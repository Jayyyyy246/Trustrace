"""
Storage abstraction protocol for evidence files.
Allows backend to switch seamlessly between local POSIX storage,
in-memory storage for unit tests, or future cloud storage.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, BinaryIO


class EvidenceStorage(ABC):
    @abstractmethod
    def save(self, evidence_id: str, filename: str, data: bytes) -> Path:
        """Saves raw evidence bytes to quarantined storage and returns stored file path."""
        pass

    @abstractmethod
    def get_path(self, evidence_id: str) -> Optional[Path]:
        """Returns the physical Path of stored evidence if it exists."""
        pass

    @abstractmethod
    def read_bytes(self, evidence_id: str) -> bytes:
        """Reads raw bytes for stored evidence."""
        pass

    @abstractmethod
    def exists(self, evidence_id: str) -> bool:
        """Checks if evidence file exists."""
        pass

    @abstractmethod
    def delete(self, evidence_id: str) -> bool:
        """Deletes evidence from quarantine (used primarily in test cleanup)."""
        pass
