"""
Local secure storage provider for TRUSTTRACE.
Enforces quarantine isolation, path validation, and read-only protection.
"""
import os
import stat
from pathlib import Path
from typing import Optional, Dict

from app.core.config import settings
from app.core.exceptions import EvidenceNotFoundError
from app.storage.base import EvidenceStorage


class LocalEvidenceStorage(EvidenceStorage):
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = (base_dir or settings.QUARANTINE_DIR).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        # In-memory mapping of evidence_id to file path for fast resolution
        self._path_registry: Dict[str, Path] = {}
        # Index existing quarantine files on startup
        try:
            for item in self.base_dir.iterdir():
                if item.is_file():
                    self._path_registry[item.stem] = item
        except OSError:
            pass

    def _resolve_evidence_path(self, evidence_id: str, suffix: str = ".bin") -> Path:
        """Deterministically computes isolated file location with strict boundary protection."""
        import re
        safe_id = re.sub(r"[^a-zA-Z0-9_\-]", "", str(evidence_id))
        safe_suffix = re.sub(r"[^a-zA-Z0-9\.]", "", str(suffix))
        if not safe_suffix.startswith("."):
            safe_suffix = f".{safe_suffix}"
        
        target_path = (self.base_dir / f"{safe_id}{safe_suffix}").resolve()
        if not str(target_path).startswith(str(self.base_dir)):
            raise ValueError(f"Path traversal detected for evidence ID: {evidence_id}")
        return target_path

    def save(self, evidence_id: str, filename: str, data: bytes) -> Path:
        """
        Saves raw bytes into quarantine.
        Applies read-only permission where supported to preserve chain-of-custody.
        """
        suffix = Path(filename).suffix.lower()
        if not suffix:
            suffix = ".bin"
        
        target_path = self._resolve_evidence_path(evidence_id, suffix=suffix)
        
        # Write bytes
        with open(target_path, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

        # Set read-only permissions to prevent in-place tampering
        try:
            os.chmod(target_path, stat.S_IREAD | stat.S_IRGRP)
        except OSError:
            pass

        self._path_registry[evidence_id] = target_path
        return target_path

    def get_path(self, evidence_id: str) -> Optional[Path]:
        """Resolves file path for evidence ID safely."""
        import re
        safe_id = re.sub(r"[^a-zA-Z0-9_\-]", "", str(evidence_id))
        if not safe_id:
            return None

        if safe_id in self._path_registry:
            path = self._path_registry[safe_id]
            if path.exists():
                return path

        # Scan directory for matching prefix safely within base_dir
        matches = list(self.base_dir.glob(f"{safe_id}.*"))
        if matches:
            self._path_registry[safe_id] = matches[0]
            return matches[0]
            
        return None

    def read_bytes(self, evidence_id: str) -> bytes:
        """Reads raw bytes for stored evidence."""
        path = self.get_path(evidence_id)
        if not path or not path.exists():
            raise EvidenceNotFoundError(evidence_id)
        with open(path, "rb") as f:
            return f.read()

    def exists(self, evidence_id: str) -> bool:
        """Checks if evidence file exists."""
        return self.get_path(evidence_id) is not None

    def delete(self, evidence_id: str) -> bool:
        """Removes evidence file from quarantine (unlocks read-only if set)."""
        path = self.get_path(evidence_id)
        if not path or not path.exists():
            return False
        try:
            # Grant write permission before deleting
            os.chmod(path, stat.S_IWRITE)
            path.unlink()
            self._path_registry.pop(evidence_id, None)
            return True
        except OSError:
            return False


local_storage = LocalEvidenceStorage()
