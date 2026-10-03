"""
Custom domain exceptions for TRUSTTRACE evidence processing.
"""


class TrustTraceException(Exception):
    """Base exception for all TRUSTTRACE errors."""
    def __init__(self, message: str, code: str = "INTERNAL_ERROR"):
        super().__init__(message)
        self.message = message
        self.code = code


class EvidenceValidationError(TrustTraceException):
    """Raised when uploaded evidence fails structural or security validation."""
    def __init__(self, message: str):
        super().__init__(message, code="EVIDENCE_VALIDATION_ERROR")


class FileTooLargeError(EvidenceValidationError):
    """Raised when uploaded evidence exceeds maximum allowed byte limit."""
    def __init__(self, max_bytes: int, actual_bytes: int):
        super().__init__(
            f"File size {actual_bytes} bytes exceeds maximum allowed {max_bytes} bytes."
        )
        self.code = "FILE_TOO_LARGE"


class UnsupportedMimeTypeError(EvidenceValidationError):
    """Raised when file MIME type or magic signature is not allowed."""
    def __init__(self, detected_mime: str):
        super().__init__(
            f"Detected MIME type '{detected_mime}' is not supported for forensic analysis."
        )
        self.code = "UNSUPPORTED_MIME_TYPE"


class EvidenceNotFoundError(TrustTraceException):
    """Raised when an evidence ID is not found in the storage or database."""
    def __init__(self, evidence_id: str):
        super().__init__(
            f"Evidence record with ID '{evidence_id}' was not found.",
            code="EVIDENCE_NOT_FOUND"
        )


class AnalysisExecutionError(TrustTraceException):
    """Raised when a forensic analyzer encounters an unrecoverable runtime error."""
    def __init__(self, analyzer_name: str, reason: str):
        super().__init__(
            f"Analyzer '{analyzer_name}' failed execution: {reason}",
            code="ANALYSIS_EXECUTION_ERROR"
        )
