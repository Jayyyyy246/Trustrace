"""
Common enumerations and types for TRUSTTRACE.
"""
from enum import Enum


class AnalyzerStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    NOT_AVAILABLE = "NOT_AVAILABLE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class FindingSeverity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class VerdictLabel(str, Enum):
    REAL = "REAL"
    EDITED = "EDITED"
    AI_GENERATED = "AI-GENERATED"
    SCREENSHOT_MANIPULATED = "SCREENSHOT-MANIPULATED"
    UNKNOWN = "UNKNOWN"
