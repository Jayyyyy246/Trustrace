import React, { useState, useRef } from 'react';
import { api, ApiError } from '../services/api';
import type { AnalysisResultResponse, EvidenceUploadResponse } from '../types/forensic';
import {
  UploadCloud,
  FileCheck,
  ShieldCheck,
  Lock,
  Hash,
  AlertCircle,
  Copy,
  Check,
  ArrowRight,
  RefreshCw,
} from 'lucide-react';

interface UploadPageProps {
  onNavigate: (tab: any, evidenceId?: string) => void;
}

export const UploadPage: React.FC<UploadPageProps> = ({ onNavigate }) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [autoAnalyze, setAutoAnalyze] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadResult, setUploadResult] = useState<AnalysisResultResponse | EvidenceUploadResponse | null>(null);
  const [copiedField, setCopiedField] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const supportedFormats = ['image/jpeg', 'image/png', 'image/webp', 'image/tiff', 'image/bmp'];
  const maxSizeBytes = 25 * 1024 * 1024; // 25 MB

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const validateAndSetFile = (file: File) => {
    setUploadError(null);
    if (!supportedFormats.includes(file.type) && !/\.(jpe?g|png|webp|tiff?|bmp)$/i.test(file.name)) {
      setUploadError(`Unsupported format '${file.type || file.name}'. Allowed: JPEG, PNG, WebP, TIFF, BMP.`);
      return;
    }
    if (file.size > maxSizeBytes) {
      setUploadError(`File size ${(file.size / 1024 / 1024).toFixed(1)} MB exceeds maximum allowed limit of 25 MB.`);
      return;
    }
    setSelectedFile(file);
    setUploadResult(null);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setUploading(true);
    setUploadError(null);
    setUploadProgress(20);

    try {
      setUploadProgress(50);
      const res = await api.uploadEvidence(selectedFile, autoAnalyze);
      setUploadProgress(100);
      setUploadResult(res);
    } catch (err) {
      setUploadError((err as ApiError).message || 'Failed to intake evidence file into quarantine store.');
    } finally {
      setUploading(false);
    }
  };

  const copyToClipboard = (text: string, field: string) => {
    navigator.clipboard.writeText(text);
    setCopiedField(field);
    setTimeout(() => setCopiedField(null), 2000);
  };

  const evidenceId = uploadResult
    ? 'analysis_id' in uploadResult
      ? uploadResult.analysis_id
      : uploadResult.evidence_id
    : null;

  const sha256 = uploadResult
    ? 'evidence' in uploadResult
      ? uploadResult.evidence.sha256
      : uploadResult.sha256
    : null;

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Page Title */}
      <div className="pb-2 border-b border-lab-border">
        <h1 className="text-xl font-bold font-mono tracking-wide text-slate-100 flex items-center gap-2.5">
          <UploadCloud className="w-5 h-5 text-forensic-blueLight" />
          DIGITAL EVIDENCE INTAKE & QUARANTINE
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Cryptographically verify, quarantine, and submit digital assets for multi-criteria technical examination.
        </p>
      </div>

      {/* Main Upload Container */}
      {!uploadResult ? (
        <div className="space-y-6">
          {/* Dropzone Card */}
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-all duration-200 ${
              dragActive
                ? 'border-forensic-blue bg-lab-900/90 shadow-forensic-glow'
                : selectedFile
                ? 'border-emerald-500/60 bg-lab-900/60'
                : 'border-lab-border hover:border-slate-600 bg-lab-900/40'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".jpg,.jpeg,.png,.webp,.tiff,.tif,.bmp"
              onChange={handleChange}
              className="hidden"
            />

            <div className="flex flex-col items-center justify-center space-y-3">
              <div className="w-14 h-14 rounded-full bg-lab-850 border border-lab-border flex items-center justify-center text-slate-300">
                {selectedFile ? (
                  <FileCheck className="w-7 h-7 text-emerald-400" />
                ) : (
                  <UploadCloud className="w-7 h-7 text-forensic-blueLight" />
                )}
              </div>

              {selectedFile ? (
                <div className="space-y-1">
                  <div className="text-sm font-mono font-bold text-slate-100">{selectedFile.name}</div>
                  <div className="text-xs font-mono text-slate-400">
                    {(selectedFile.size / 1024).toFixed(1)} KB • {selectedFile.type || 'image'}
                  </div>
                  <div className="text-xs text-emerald-400 font-mono pt-1">
                    ✓ Ready for quarantined intake
                  </div>
                </div>
              ) : (
                <div className="space-y-1">
                  <div className="text-sm font-mono font-medium text-slate-200">
                    Drag and drop evidence file here, or <span className="text-forensic-blueLight underline">browse</span>
                  </div>
                  <div className="text-xs text-slate-400 font-mono">
                    Binary image files: JPEG, PNG, WebP, TIFF, BMP (Max 25 MB)
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Options & Action Bar */}
          <div className="forensic-panel p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <label className="flex items-center gap-3 cursor-pointer text-xs font-mono text-slate-300">
              <input
                type="checkbox"
                checked={autoAnalyze}
                onChange={(e) => setAutoAnalyze(e.target.checked)}
                className="w-4 h-4 rounded bg-lab-950 border-lab-border text-forensic-blue focus:ring-0"
              />
              <span>Execute comprehensive analysis immediately upon intake</span>
            </label>

            <button
              onClick={handleUpload}
              disabled={!selectedFile || uploading}
              className={`px-5 py-2 rounded text-xs font-mono font-bold tracking-wider transition-all flex items-center justify-center gap-2 ${
                !selectedFile || uploading
                  ? 'bg-lab-800 text-slate-500 cursor-not-allowed border border-lab-border'
                  : 'bg-forensic-blue hover:bg-sky-600 text-white shadow-forensic-glow'
              }`}
            >
              {uploading ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  Quarantining & Processing...
                </>
              ) : (
                <>
                  <Lock className="w-3.5 h-3.5" />
                  Submit to Quarantine
                </>
              )}
            </button>
          </div>

          {/* Upload Progress Bar if active */}
          {uploading && (
            <div className="space-y-1">
              <div className="flex justify-between text-xs font-mono text-slate-400">
                <span>Running intake verification & hash calculation...</span>
                <span>{uploadProgress}%</span>
              </div>
              <div className="w-full h-1.5 bg-lab-900 rounded-full overflow-hidden">
                <div
                  className="h-full bg-forensic-blue transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          )}

          {/* Error Banner */}
          {uploadError && (
            <div className="p-4 rounded-lg bg-rose-950/40 border border-rose-600/50 flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
              <div className="text-xs font-mono text-rose-200">
                <div className="font-bold">Intake Validation Failed:</div>
                <div className="text-slate-300 mt-0.5">{uploadError}</div>
              </div>
            </div>
          )}

          {/* Forensic Standards & Legal Notice Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
            <div className="forensic-panel p-4 space-y-2">
              <div className="flex items-center gap-2 text-xs font-mono font-bold text-forensic-blueLight uppercase">
                <Hash className="w-4 h-4" />
                Chain-of-Custody SHA-256 Guarantee
              </div>
              <p className="text-xs text-slate-400 leading-relaxed font-mono">
                Upon stream ingestion, non-destructive bitwise SHA-256 and MD5 digests are computed. Quarantine files are locked in read-only mode to prevent in-place mutation.
              </p>
            </div>

            <div className="forensic-panel p-4 space-y-2">
              <div className="flex items-center gap-2 text-xs font-mono font-bold text-forensic-tealLight uppercase">
                <ShieldCheck className="w-4 h-4" />
                Air-Gapped Privacy Architecture
              </div>
              <p className="text-xs text-slate-400 leading-relaxed font-mono">
                All evidence processing occurs strictly within the local host environment. Visual data is never dispatched to external LLMs, public training pipelines, or telemetry collectors.
              </p>
            </div>
          </div>
        </div>
      ) : (
        /* Intake Confirmation / Receipt Card */
        <div className="space-y-6">
          <div className="forensic-panel p-6 border-emerald-500/40 space-y-6">
            <div className="flex items-center justify-between pb-4 border-b border-lab-border">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-emerald-950/80 border border-emerald-500/40 flex items-center justify-center text-emerald-400">
                  <Check className="w-5 h-5" />
                </div>
                <div>
                  <h2 className="text-sm font-mono font-bold text-slate-100 uppercase">
                    Evidence Ingested & Verified Successfully
                  </h2>
                  <p className="text-xs font-mono text-slate-400">
                    Chain-of-custody recorded. File secured in quarantine storage.
                  </p>
                </div>
              </div>
              <span className="px-2.5 py-1 rounded text-xs font-mono font-bold bg-emerald-950/80 border border-emerald-500/40 text-emerald-300">
                STATUS: SECURED
              </span>
            </div>

            {/* Field Breakdown */}
            <div className="space-y-3 text-xs font-mono">
              {/* Evidence ID */}
              <div className="p-3 rounded bg-lab-950/80 border border-lab-border flex items-center justify-between gap-4">
                <div>
                  <span className="text-slate-400 block text-[10px] uppercase">Assigned Evidence ID:</span>
                  <span className="text-slate-200 font-bold select-all">{evidenceId}</span>
                </div>
                {evidenceId && (
                  <button
                    onClick={() => copyToClipboard(evidenceId, 'id')}
                    className="p-1.5 rounded hover:bg-lab-800 text-slate-400 hover:text-white"
                    title="Copy Evidence ID"
                  >
                    {copiedField === 'id' ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                  </button>
                )}
              </div>

              {/* SHA-256 */}
              <div className="p-3 rounded bg-lab-950/80 border border-lab-border flex items-center justify-between gap-4">
                <div className="min-w-0">
                  <span className="text-slate-400 block text-[10px] uppercase">SHA-256 Digest:</span>
                  <span className="text-forensic-blueLight truncate block font-mono select-all">
                    {sha256}
                  </span>
                </div>
                {sha256 && (
                  <button
                    onClick={() => copyToClipboard(sha256, 'hash')}
                    className="p-1.5 rounded hover:bg-lab-800 text-slate-400 hover:text-white shrink-0"
                    title="Copy SHA-256"
                  >
                    {copiedField === 'hash' ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                  </button>
                )}
              </div>

              {/* File Info Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-1">
                <div className="p-2.5 rounded bg-lab-950/60 border border-lab-border/50">
                  <span className="text-slate-500 block text-[10px]">Filename:</span>
                  <span className="text-slate-200 truncate block font-medium" title={selectedFile?.name}>
                    {selectedFile?.name}
                  </span>
                </div>
                <div className="p-2.5 rounded bg-lab-950/60 border border-lab-border/50">
                  <span className="text-slate-500 block text-[10px]">Size:</span>
                  <span className="text-slate-200">
                    {selectedFile ? (selectedFile.size / 1024).toFixed(1) : 0} KB
                  </span>
                </div>
                <div className="p-2.5 rounded bg-lab-950/60 border border-lab-border/50">
                  <span className="text-slate-500 block text-[10px]">MIME Type:</span>
                  <span className="text-slate-200">{selectedFile?.type || 'image/jpeg'}</span>
                </div>
                <div className="p-2.5 rounded bg-lab-950/60 border border-lab-border/50">
                  <span className="text-slate-500 block text-[10px]">Quarantine:</span>
                  <span className="text-emerald-400 font-bold">READ-ONLY</span>
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="pt-2 flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-t border-lab-border">
              <button
                onClick={() => {
                  setSelectedFile(null);
                  setUploadResult(null);
                }}
                className="px-4 py-2 rounded bg-lab-900 hover:bg-lab-850 border border-lab-border text-xs font-mono text-slate-300"
              >
                + Ingest Another Item
              </button>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => onNavigate('analysis', evidenceId || undefined)}
                  className="px-4 py-2 rounded bg-lab-800 hover:bg-lab-750 border border-lab-border text-xs font-mono text-slate-200"
                >
                  View Timeline
                </button>
                <button
                  onClick={() => onNavigate('analysis', evidenceId || undefined)}
                  className="px-5 py-2 rounded bg-forensic-blue hover:bg-sky-600 text-white text-xs font-mono font-bold flex items-center gap-1.5 shadow-forensic-card"
                >
                  Inspect Full Forensic Assessment <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
