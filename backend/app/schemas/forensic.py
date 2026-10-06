"""
Forensic analyzer result schemas.
Every field must be populated from actual technical analysis or explicitly
designated as unavailable. No fabricated or simulated data.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from app.schemas.common import AnalyzerStatus, FindingSeverity


class FindingItem(BaseModel):
    finding_id: str
    category: str = Field(default="GENERAL", description="Forensic domain category (e.g., METADATA, COMPRESSION, NOISE, CLONE, SCREENSHOT, OCR)")
    severity: FindingSeverity = Field(description="Severity: INFO, LOW, MEDIUM, HIGH, CRITICAL")
    description: str = Field(description="Summary of the forensic observation")
    evidence: Any = Field(default=None, description="Measurable technical proof/metric value")
    interpretation: str = Field(default="Observed technical metric or container characteristic.", description="Scientific interpretation of the observed metric")
    limitation: str = Field(default="Metric is contextual and not standalone proof of authenticity or fraud.", description="Forensic caveats, failure modes, or why this is not standalone proof")
    # Compatibility fields
    title: Optional[str] = None
    analyzer: Optional[str] = None
    confidence: float = Field(default=0.70, ge=0.0, le=1.0, description="Calibrated confidence score")
    technical_details: Dict[str, Any] = Field(default_factory=dict)
    is_anomaly: bool = False

    def model_post_init(self, __context: Any) -> None:
        if self.title is None:
            self.title = self.description[:60]
        if not self.technical_details and self.evidence is not None:
            self.technical_details = {"evidence": self.evidence}


class BaseAnalyzerResult(BaseModel):
    status: str = Field(default="completed", description="Execution status: completed, failed, not_available")
    findings: List[FindingItem] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    limitations: List[str] = Field(default_factory=list)


class MetadataAnalyzerResult(BaseModel):
    status: AnalyzerStatus = AnalyzerStatus.COMPLETED
    exif_present: bool = False
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    software: Optional[str] = None
    create_date: Optional[str] = None
    modify_date: Optional[str] = None
    all_tags: Dict[str, Any] = Field(default_factory=dict)
    anomalies: List[str] = Field(default_factory=list)
    editing_software_detected: bool = False


class ImageAnalyzerResult(BaseModel):
    status: AnalyzerStatus = AnalyzerStatus.COMPLETED
    dimensions: Dict[str, int] = Field(description="Width and height in pixels")
    channels: int
    color_space: str
    aspect_ratio: float
    laplacian_variance: float = Field(description="Focus/blur metric via Laplacian operator variance")
    is_blurry: bool = False
    luminance_mean: float
    luminance_std: float
    ela_computed: bool = False
    ela_mean_delta: Optional[float] = None
    ela_variance: Optional[float] = None
    estimated_jpeg_quality: Optional[int] = None
    noise_residual: Optional[Dict[str, Any]] = None


class OCRRegion(BaseModel):
    text: str
    bbox: List[int] = Field(description="Bounding box coordinates [x, y, width, height]")
    confidence: Optional[float] = Field(default=None, description="Detection confidence in [0, 1] if reported by engine")


class OCRAnalyzerResult(BaseModel):
    status: AnalyzerStatus = AnalyzerStatus.NOT_AVAILABLE
    engine: str = "Tesseract"
    text: Optional[str] = None
    confidence: Optional[float] = None
    regions: List[OCRRegion] = Field(default_factory=list)
    language: Optional[str] = None
    processing_time_ms: Optional[float] = None
    word_count: int = 0
    character_count: int = 0
    availability_note: Optional[str] = None
    failure_reason: Optional[str] = None
    font_anomaly_detected: bool = False


class ScreenshotAnalyzerResult(BaseModel):
    status: AnalyzerStatus = AnalyzerStatus.COMPLETED
    is_common_viewport: bool = False
    matched_viewport: Optional[str] = None
    aspect_ratio_standard: bool = False
    indicators: List[str] = Field(default_factory=list)
    ui_rectangles_detected: Optional[int] = None
    repeated_alignments_count: Optional[int] = None
    text_density_ratio: Optional[float] = None


class ModelDiagnostics(BaseModel):
    name: Optional[str] = None
    version: Optional[str] = None
    checkpoint_path: Optional[str] = None
    checkpoint_sha256: Optional[str] = None
    architecture: Optional[str] = None
    device: str = "CPU"
    inference_time_ms: Optional[float] = None
    preprocessing_version: str = "v1.0.0-imagenet"
    calibration_active: bool = False
    ood_active: bool = False
    mc_dropout_active: bool = False


class MLAnalyzerResult(BaseModel):
    status: str = Field(
        default="NOT_AVAILABLE",
        description="Explicit status indicator: AVAILABLE, NOT_AVAILABLE, or FAILED."
    )
    model_status: str = Field(
        default="NOT_AVAILABLE",
        description="Explicit status indicator: AVAILABLE, NOT_AVAILABLE, or FAILED."
    )
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    predicted_label: Optional[str] = None
    confidence: Optional[float] = None
    class_probabilities: Optional[Dict[str, float]] = None
    uncertainty: Optional[float] = Field(
        default=None,
        description="Quantified uncertainty score in [0, 1] derived from entropy and OOD energy logic."
    )
    ood_detected: Optional[bool] = None
    calibrated: bool = False
    device: Optional[str] = None
    inference_time_ms: Optional[float] = None
    checkpoint_sha256: Optional[str] = None
    failure_reason: Optional[str] = None
    note: str = Field(
        default="Deep learning model weights not loaded in Phase 1 runtime. Skipping ML inference."
    )
    diagnostics: Optional[ModelDiagnostics] = None


class AnalyzersContainer(BaseModel):
    metadata: MetadataAnalyzerResult
    image: ImageAnalyzerResult
    ocr: OCRAnalyzerResult
    screenshot: ScreenshotAnalyzerResult
    ml: MLAnalyzerResult
