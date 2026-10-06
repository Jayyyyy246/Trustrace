export type VerdictLabel = 
  | 'REAL'
  | 'EDITED'
  | 'AI-GENERATED'
  | 'SCREENSHOT-MANIPULATED'
  | 'UNKNOWN';

export type FindingSeverity = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

export interface EvidenceSignal {
  source: string;
  metric: string;
  direction: VerdictLabel;
  reliability: number;
  explanation: string;
  polarity: 'SUPPORTING' | 'CONTRADICTORY';
}

export interface FinalVerdict {
  label: VerdictLabel;
  confidence: number | null;
  uncertainty?: number | null;
  risk_score: number;
  justification: string;
  explanation?: string | null;
  supporting_findings: EvidenceSignal[];
  contradictory_findings: EvidenceSignal[];
  unavailable_analyzers: string[];
  limitations: string[];
  conflict_detected: boolean;
  conflict_details?: string | null;
  decision_rules_triggered: string[];
}

export interface EvidenceInfo {
  evidence_id?: string;
  filename: string;
  sha256: string;
  mime_type: string;
  size: number;
  dimensions?: {
    width: number;
    height: number;
  } | null;
}

export interface FindingItem {
  finding_id: string;
  analyzer: string;
  category: string;
  severity: FindingSeverity;
  title: string;
  description: string;
  evidence: any;
  interpretation?: string;
  limitation?: string;
}

export interface MetadataAnalyzerResult {
  status: string;
  exif_present: boolean;
  camera_make?: string | null;
  camera_model?: string | null;
  software?: string | null;
  modify_date?: string | null;
  create_date?: string | null;
  editing_software_detected: boolean;
  color_space?: string | null;
  raw_tags?: Record<string, any>;
  anomalies: string[];
}

export interface ImageAnalyzerResult {
  status: string;
  dimensions: { width: number; height: number };
  channels: number;
  color_space: string;
  aspect_ratio: number;
  estimated_jpeg_quality?: number | null;
  quantization_tables?: Record<string, number[][]>;
  laplacian_variance: number;
  is_blurry: boolean;
  luminance_mean: number;
  luminance_std: number;
  ela_computed: boolean;
  ela_mean_delta?: number | null;
  ela_variance?: number | null;
}

export interface OCRRegion {
  text: string;
  bbox: [number, number, number, number];
  confidence?: number | null;
}

export interface OCRAnalyzerResult {
  status: string;
  engine: string;
  text?: string | null;
  confidence?: number | null;
  regions?: OCRRegion[];
  language?: string | null;
  processing_time_ms?: number | null;
  word_count: number;
  character_count: number;
  availability_note?: string | null;
  failure_reason?: string | null;
  font_anomaly_detected?: boolean;
}

export interface ScreenshotAnalyzerResult {
  status: string;
  is_common_viewport: boolean;
  matched_viewport?: string | null;
  aspect_ratio_standard: boolean;
  indicators: string[];
}

export interface ModelDiagnostics {
  name?: string | null;
  version?: string | null;
  checkpoint_path?: string | null;
  checkpoint_sha256?: string | null;
  architecture?: string | null;
  device: string;
  inference_time_ms?: number | null;
  preprocessing_version: string;
  calibration_active: boolean;
  ood_active: boolean;
  mc_dropout_active: boolean;
}

export interface MLAnalyzerResult {
  model_status: string;
  model_name?: string | null;
  model_version?: string | null;
  predicted_label?: string | null;
  confidence?: number | null;
  uncertainty?: number | null;
  class_probabilities?: Record<string, number> | null;
  ood_detected?: boolean | null;
  calibrated?: boolean;
  device?: string | null;
  inference_time_ms?: number | null;
  checkpoint_sha256?: string | null;
  failure_reason?: string | null;
  note: string;
  diagnostics?: ModelDiagnostics | null;
}

export interface AnalyzersContainer {
  metadata: MetadataAnalyzerResult;
  image: ImageAnalyzerResult;
  ocr: OCRAnalyzerResult;
  screenshot: ScreenshotAnalyzerResult;
  ml: MLAnalyzerResult;
}

export interface AnalysisResultResponse {
  analysis_id: string;
  evidence: EvidenceInfo;
  analyzers: AnalyzersContainer;
  findings: FindingItem[];
  final_verdict: FinalVerdict;
  limitations: string[];
  created_at: string;
  pipeline_version: string;
}

export interface EvidenceUploadResponse {
  evidence_id: string;
  filename: string;
  sha256: string;
  mime_type: string;
  size: number;
  uploaded_at: string;
  status: string;
}

export interface DashboardStats {
  active_investigations: number;
  total_evidence_analyzed: number;
  analysis_success_rate: number;
  recent_investigations: Array<{
    evidence_id: string;
    analysis_id: string;
    filename: string;
    final_verdict: VerdictLabel;
    confidence: number | null;
    uncertainty?: number | null;
    created_at: string;
    sha256: string;
  }>;
  latest_evidence: Array<{
    evidence_id: string;
    filename: string;
    sha256: string;
    size: number;
    mime_type: string;
    uploaded_at: string;
    status: string;
  }>;
  system_analyzer_status: Record<string, string>;
  model_status: {
    is_loaded: boolean;
    model_version?: string | null;
    architecture?: string | null;
    checksum?: string | null;
  };
}

export interface SystemHealth {
  status: string;
  system: string;
  version: string;
  environment: string;
  timestamp_utc: string;
  storage: {
    status: string;
    provider: string;
  };
  analyzers: Record<string, string>;
}
