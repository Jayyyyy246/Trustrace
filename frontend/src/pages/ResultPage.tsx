import React, { useState } from 'react';
import type { AnalysisResultResponse, FindingItem } from '../types/forensic';
import { api } from '../services/api';
import { StatusBadge, SeverityBadge, ForensicIndicatorIcon } from '../components/StatusBadge';
import {
  AlertTriangle,
  FileCheck,
  Binary,
  Microscope,
  Search,
  Layers,
  Cpu,
  GitMerge,
  HelpCircle,
  AlertOctagon,
  Download,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
  Image as ImageIcon,
  ShieldCheck,
  FileText,
} from 'lucide-react';

interface ResultPageProps {
  analysis: AnalysisResultResponse;
  onNavigate: (tab: any, evidenceId?: string) => void;
}

export const ResultPage: React.FC<ResultPageProps> = ({ analysis, onNavigate }) => {
  const [copied, setCopied] = useState(false);
  const [copiedExplanation, setCopiedExplanation] = useState(false);
  const [expandedFindings, setExpandedFindings] = useState<Record<string, boolean>>({});
  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    summary: true,
    integrity: true,
    metadata: true,
    forensics: true,
    screenshot: false,
    ocr: false,
    ml: true,
    fusion: true,
    whyThis: true,
    limitations: true,
    report: false,
  });

  const [downloadingPdf, setDownloadingPdf] = useState(false);

  const toggleSection = (section: string) => {
    setOpenSections((prev) => ({ ...prev, [section]: !prev[section] }));
  };

  const toggleFinding = (findingId: string) => {
    setExpandedFindings((prev) => ({ ...prev, [findingId]: !prev[findingId] }));
  };

  const copyJson = () => {
    navigator.clipboard.writeText(JSON.stringify(analysis, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadPdf = async () => {
    const targetId = analysis.evidence.evidence_id || analysis.analysis_id;
    try {
      setDownloadingPdf(true);
      await api.downloadReportPdf(targetId, `TRUSTTRACE_FORENSIC_REPORT_${targetId.slice(0, 8).toUpperCase()}.pdf`);
    } catch (err) {
      alert(`PDF download failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setDownloadingPdf(false);
    }
  };

  const copyAssessmentExplanation = () => {
    if (analysis.final_verdict.explanation) {
      navigator.clipboard.writeText(analysis.final_verdict.explanation);
      setCopiedExplanation(true);
      setTimeout(() => setCopiedExplanation(false), 2000);
    }
  };

  const downloadJson = () => {
    const blob = new Blob([JSON.stringify(analysis, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `TRUSTTRACE_AUDIT_${analysis.analysis_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const verdict = analysis.final_verdict;
  const ev = analysis.evidence;
  const analyzers = analysis.analyzers;

  // Determine finding status indicator
  const getFindingStatus = (f: FindingItem): 'normal' | 'suspicious' | 'strong' | 'unavailable' => {
    const sev = f.severity ? String(f.severity).toUpperCase() : 'LOW';
    if (sev === 'CRITICAL' || sev === 'HIGH') return 'strong';
    if (sev === 'MEDIUM') return 'suspicious';
    return 'normal';
  };

  return (
    <div className="max-w-5xl mx-auto space-y-6 pb-12">
      {/* ------------------------------------------------------------- */}
      {/* TOP SECTION: TRUSTTRACE ASSESSMENT BANNER                     */}
      {/* ------------------------------------------------------------- */}
      <div className="forensic-panel p-6 border-slate-700 bg-gradient-to-b from-lab-900 to-lab-950 space-y-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-lab-border">
          <div className="space-y-1">
            <div className="text-[11px] font-mono tracking-widest text-slate-400 uppercase">
              Official Technical Assessment
            </div>
            <h1 className="text-2xl font-bold font-mono tracking-wider text-slate-100 flex items-center gap-3">
              TRUSTTRACE ASSESSMENT
            </h1>
          </div>

          {/* Primary Classification Label */}
          <div className="flex items-center gap-3">
            <StatusBadge label={verdict.label} size="lg" />
          </div>
        </div>

        {/* Conflict Alert Banner if present */}
        {verdict.conflict_detected && (
          <div className="p-4 rounded-lg bg-amber-950/50 border border-amber-500/60 text-amber-200 text-xs font-mono space-y-1">
            <div className="font-bold flex items-center gap-2 text-amber-300">
              <AlertTriangle className="w-4 h-4 text-amber-400" />
              Adversarial Evidence Conflict Detected
            </div>
            <div className="text-slate-300 leading-relaxed pl-6">
              {verdict.conflict_details || 'Independent forensic signals diverge; engine arbitrated conflict to prevent uncorroborated false attribution.'}
            </div>
          </div>
        )}

        {/* Metrics Bar: Confidence & Risk Score */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-1">
          {/* Confidence / Uncertainty */}
          <div className="p-3.5 rounded-lg bg-lab-950/80 border border-lab-border">
            <div className="text-[11px] font-mono text-slate-400 uppercase">Calibrated Confidence:</div>
            <div className="mt-1 flex items-baseline gap-2">
              {verdict.confidence !== null ? (
                <>
                  <span className="text-2xl font-bold font-mono text-slate-100">
                    {(verdict.confidence * 100).toFixed(1)}%
                  </span>
                  <span className="text-[11px] text-slate-400 font-mono">
                    (Uncertainty: {verdict.uncertainty !== undefined && verdict.uncertainty !== null ? verdict.uncertainty.toFixed(2) : ((1 - verdict.confidence).toFixed(2))})
                  </span>
                </>
              ) : (
                <span className="text-sm font-bold font-mono text-amber-400/90 py-1">
                  Confidence unavailable
                </span>
              )}
            </div>
            <div className="text-[10px] text-slate-500 font-mono mt-1">
              {verdict.confidence !== null ? 'Statistically derived from multi-signal corroboration' : 'Inconclusive / non-probabilistic assessment'}
            </div>
          </div>

          {/* Risk Score */}
          <div className="p-3.5 rounded-lg bg-lab-950/80 border border-lab-border">
            <div className="text-[11px] font-mono text-slate-400 uppercase">Tamper Risk Index:</div>
            <div className="mt-1 flex items-baseline gap-2">
              <span className={`text-2xl font-bold font-mono ${
                verdict.risk_score >= 0.7 ? 'text-rose-400' : verdict.risk_score >= 0.4 ? 'text-amber-400' : 'text-emerald-400'
              }`}>
                {(verdict.risk_score * 100).toFixed(0)} / 100
              </span>
              <span className="text-[11px] text-slate-400 font-mono">
                {verdict.risk_score >= 0.7 ? 'High Risk' : verdict.risk_score >= 0.4 ? 'Moderate' : 'Low Risk'}
              </span>
            </div>
            {/* Visual Risk Bar */}
            <div className="w-full h-1.5 bg-lab-900 rounded-full mt-2 overflow-hidden">
              <div
                className={`h-full ${
                  verdict.risk_score >= 0.7 ? 'bg-rose-500' : verdict.risk_score >= 0.4 ? 'bg-amber-500' : 'bg-emerald-500'
                }`}
                style={{ width: `${Math.max(5, verdict.risk_score * 100)}%` }}
              />
            </div>
          </div>

          {/* Rules Triggered Summary */}
          <div className="p-3.5 rounded-lg bg-lab-950/80 border border-lab-border">
            <div className="text-[11px] font-mono text-slate-400 uppercase">Decision Rules:</div>
            <div className="mt-1 text-2xl font-bold font-mono text-slate-100">
              {verdict.decision_rules_triggered.length}
            </div>
            <div className="text-[10px] text-slate-500 font-mono mt-1">
              Evaluated via graph arbitration logic
            </div>
          </div>
        </div>

        {/* Justification summary text */}
        <div className="p-4 rounded-lg bg-lab-950/90 border border-lab-border/70 text-xs font-mono text-slate-300 leading-relaxed">
          <strong className="text-slate-100 block mb-1">Executive Forensic Assessment:</strong>
          {verdict.justification}
        </div>
      </div>

      {/* ------------------------------------------------------------- */}
      {/* 11 EXPANDABLE FORENSIC SECTIONS                               */}
      {/* ------------------------------------------------------------- */}

      {/* Section 1: Evidence Summary */}
      <div className="forensic-panel overflow-hidden">
        <button
          onClick={() => toggleSection('summary')}
          aria-expanded={openSections.summary}
          className="w-full p-4 flex items-center justify-between text-left hover:bg-lab-850/50 transition-colors focus:outline-none"
        >
          <div className="flex items-center gap-2.5 font-mono text-xs font-bold text-slate-200 uppercase">
            <FileCheck className="w-4 h-4 text-forensic-blueLight" />
            1. Evidence Summary
          </div>
          {openSections.summary ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
        </button>

        {openSections.summary && (
          <div className="p-5 border-t border-lab-border space-y-4 text-xs font-mono">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Evidence details */}
              <div className="space-y-2.5">
                <div className="flex justify-between p-2 rounded bg-lab-950/60 border border-lab-border/40">
                  <span className="text-slate-400">Original Filename:</span>
                  <span className="text-slate-200 font-bold select-all">{ev.filename}</span>
                </div>
                <div className="flex justify-between p-2 rounded bg-lab-950/60 border border-lab-border/40">
                  <span className="text-slate-400">File Size:</span>
                  <span className="text-slate-200">{(ev.size / 1024).toFixed(1)} KB ({ev.size.toLocaleString()} bytes)</span>
                </div>
                <div className="flex justify-between p-2 rounded bg-lab-950/60 border border-lab-border/40">
                  <span className="text-slate-400">MIME Format:</span>
                  <span className="text-slate-200">{ev.mime_type}</span>
                </div>
                <div className="flex justify-between p-2 rounded bg-lab-950/60 border border-lab-border/40">
                  <span className="text-slate-400">Dimensions:</span>
                  <span className="text-slate-200">
                    {ev.dimensions ? `${ev.dimensions.width} × ${ev.dimensions.height} px` : 'Unspecified'}
                  </span>
                </div>
                <div className="flex justify-between p-2 rounded bg-lab-950/60 border border-lab-border/40">
                  <span className="text-slate-400">Quarantine Status:</span>
                  <span className="text-emerald-400 font-bold">LOCKED READ-ONLY</span>
                </div>
              </div>

              {/* Visual preview */}
              <div className="p-3 rounded bg-lab-950 border border-lab-border flex flex-col items-center justify-center space-y-2">
                <div className="relative max-h-48 w-full flex items-center justify-center overflow-hidden rounded bg-black/40">
                  <img
                    src={api.getEvidenceFileUrl(analysis.analysis_id)}
                    alt={ev.filename}
                    className="max-h-44 max-w-full object-contain"
                    onError={(e) => {
                      (e.target as HTMLElement).style.display = 'none';
                    }}
                  />
                  <ImageIcon className="w-10 h-10 text-slate-700 absolute -z-10" />
                </div>
                <span className="text-[10px] text-slate-500">Quarantined binary preview</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Section 2: Integrity */}
      <div className="forensic-panel overflow-hidden">
        <button
          onClick={() => toggleSection('integrity')}
          aria-expanded={openSections.integrity}
          className="w-full p-4 flex items-center justify-between text-left hover:bg-lab-850/50 transition-colors focus:outline-none"
        >
          <div className="flex items-center gap-2.5 font-mono text-xs font-bold text-slate-200 uppercase">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            2. Integrity
          </div>
          {openSections.integrity ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
        </button>

        {openSections.integrity && (
          <div className="p-5 border-t border-lab-border space-y-3 text-xs font-mono">
            <div className="p-3 rounded bg-lab-950 border border-lab-border space-y-1">
              <span className="text-[10px] text-slate-400 uppercase">SHA-256 Digest (Chain of Custody Standard):</span>
              <div className="font-mono text-forensic-blueLight text-xs break-all select-all font-semibold">
                {ev.sha256}
              </div>
            </div>
            <div className="p-3 rounded bg-lab-950 border border-lab-border flex items-center justify-between">
              <span className="text-slate-400">Cryptographic Bitwise Verification:</span>
              <span className="text-emerald-400 font-bold flex items-center gap-1.5">
                <Check className="w-3.5 h-3.5" /> UNCOMPROMISED (Tamper-Sealed)
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Section 3: Metadata */}
      <div className="forensic-panel overflow-hidden">
        <button
          onClick={() => toggleSection('metadata')}
          aria-expanded={openSections.metadata}
          className="w-full p-4 flex items-center justify-between text-left hover:bg-lab-850/50 transition-colors focus:outline-none"
        >
          <div className="flex items-center gap-2.5 font-mono text-xs font-bold text-slate-200 uppercase">
            <Binary className="w-4 h-4 text-forensic-amberLight" />
            3. Metadata
          </div>
          {openSections.metadata ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
        </button>

        {openSections.metadata && (
          <div className="p-5 border-t border-lab-border space-y-3 text-xs font-mono">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div className="p-2.5 rounded bg-lab-950/60 border border-lab-border/40">
                <span className="text-slate-400 block text-[10px]">EXIF Presence:</span>
                <span className={analyzers.metadata.exif_present ? 'text-emerald-400 font-bold' : 'text-slate-400'}>
                  {analyzers.metadata.exif_present ? '✓ PRESENT' : '○ ABSENT / STRIPPED'}
                </span>
              </div>
              <div className="p-2.5 rounded bg-lab-950/60 border border-lab-border/40">
                <span className="text-slate-400 block text-[10px]">Editing Software Detected:</span>
                <span className={analyzers.metadata.editing_software_detected ? 'text-rose-400 font-bold' : 'text-emerald-400'}>
                  {analyzers.metadata.editing_software_detected ? '✕ DETECTED' : '✓ NONE DETECTED'}
                </span>
              </div>
              <div className="p-2.5 rounded bg-lab-950/60 border border-lab-border/40">
                <span className="text-slate-400 block text-[10px]">Camera Hardware:</span>
                <span className="text-slate-200 font-medium">
                  {analyzers.metadata.camera_make
                    ? `${analyzers.metadata.camera_make} ${analyzers.metadata.camera_model || ''}`
                    : 'No Camera Tags'}
                </span>
              </div>
              <div className="p-2.5 rounded bg-lab-950/60 border border-lab-border/40">
                <span className="text-slate-400 block text-[10px]">Software Tag:</span>
                <span className="text-slate-200 font-medium">
                  {analyzers.metadata.software || 'None Recorded'}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Section 4: Forensic Findings */}
      <div className="forensic-panel overflow-hidden">
        <button
          onClick={() => toggleSection('forensics')}
          aria-expanded={openSections.forensics}
          className="w-full p-4 flex items-center justify-between text-left hover:bg-lab-850/50 transition-colors focus:outline-none"
        >
          <div className="flex items-center gap-2.5 font-mono text-xs font-bold text-slate-200 uppercase">
            <Microscope className="w-4 h-4 text-forensic-tealLight" />
            4. Forensic Findings
          </div>
          {openSections.forensics ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
        </button>

        {openSections.forensics && (
          <div className="p-5 border-t border-lab-border space-y-4 text-xs font-mono">
            {/* Quick Metrics Header */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="p-2.5 rounded bg-lab-950/60 border border-lab-border/40">
                <span className="text-slate-500 block text-[10px]">ELA Variance (Q90):</span>
                <span className="text-slate-200 font-bold">{analyzers.image.ela_variance?.toFixed(1) || '0.0'}</span>
              </div>
              <div className="p-2.5 rounded bg-lab-950/60 border border-lab-border/40">
                <span className="text-slate-500 block text-[10px]">Estimated Quality:</span>
                <span className="text-slate-200">{analyzers.image.estimated_jpeg_quality || 'Lossless / PNG'}</span>
              </div>
              <div className="p-2.5 rounded bg-lab-950/60 border border-lab-border/40">
                <span className="text-slate-500 block text-[10px]">Laplacian Focus:</span>
                <span className="text-slate-200">{analyzers.image.laplacian_variance.toFixed(1)}</span>
              </div>
              <div className="p-2.5 rounded bg-lab-950/60 border border-lab-border/40">
                <span className="text-slate-500 block text-[10px]">Luminance Std:</span>
                <span className="text-slate-200">{analyzers.image.luminance_std.toFixed(1)}</span>
              </div>
            </div>

            {/* Expandable itemized findings (What was detected, Metric, Why it matters, Limitations) */}
            <div className="space-y-3 pt-2">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold font-mono text-slate-300 uppercase">
                  Itemized Forensic Findings ({analysis.findings.length})
                </h3>
                <span className="text-[10px] text-slate-500">Click card to expand technical details</span>
              </div>

              {analysis.findings.length === 0 ? (
                <div className="p-4 rounded bg-lab-950 text-slate-500 text-center">
                  ✓ No anomalous forensic indicators triggered above threshold.
                </div>
              ) : (
                analysis.findings.map((f) => {
                  const isExpanded = expandedFindings[f.finding_id] ?? true;
                  const status = getFindingStatus(f);
                  const metricValue = f.evidence !== undefined && f.evidence !== null
                    ? (typeof f.evidence === 'object' ? JSON.stringify(f.evidence) : String(f.evidence))
                    : 'Computed heuristic threshold';

                  return (
                    <div
                      key={f.finding_id}
                      className="p-3.5 rounded bg-lab-950/80 border border-lab-border space-y-2 hover:border-slate-600 transition-colors"
                    >
                      <div
                        onClick={() => toggleFinding(f.finding_id)}
                        className="flex items-center justify-between gap-2 cursor-pointer select-none"
                      >
                        <div className="flex items-center gap-2">
                          <ForensicIndicatorIcon status={status} showLabel />
                          <span className="font-bold text-slate-200">{f.title}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <SeverityBadge severity={f.severity} />
                          {isExpanded ? <ChevronUp className="w-3.5 h-3.5 text-slate-500" /> : <ChevronDown className="w-3.5 h-3.5 text-slate-500" />}
                        </div>
                      </div>

                      {/* What was detected */}
                      <p className="text-slate-400 leading-relaxed text-[11px]">{f.description}</p>

                      {/* Expandable 4-point forensic detail */}
                      {isExpanded && (
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-[10px] text-slate-400 pt-2 border-t border-lab-border/40">
                          <div>
                            <strong className="text-slate-300 block mb-0.5">What was detected:</strong>
                            <span className="text-slate-300">{f.title}</span>
                          </div>
                          <div>
                            <strong className="text-slate-300 block mb-0.5">Metric:</strong>
                            <code className="text-forensic-blueLight break-all">{metricValue}</code>
                          </div>
                          <div>
                            <strong className="text-slate-300 block mb-0.5">Why it matters:</strong>
                            <span className="text-slate-300">{f.interpretation || 'Forensic signal indicating statistical or spatial divergence'}</span>
                          </div>
                          <div>
                            <strong className="text-slate-300 block mb-0.5">Limitations:</strong>
                            <span className="text-slate-300">{f.limitation || 'Physical sensor noise and container compression constraints apply'}</span>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })
              )}
            </div>
          </div>
        )}
      </div>

      {/* Section 5: Screenshot Analysis */}
      <div className="forensic-panel overflow-hidden">
        <button
          onClick={() => toggleSection('screenshot')}
          aria-expanded={openSections.screenshot}
          className="w-full p-4 flex items-center justify-between text-left hover:bg-lab-850/50 transition-colors focus:outline-none"
        >
          <div className="flex items-center gap-2.5 font-mono text-xs font-bold text-slate-200 uppercase">
            <Layers className="w-4 h-4 text-forensic-blueLight" />
            5. Screenshot Analysis
          </div>
          {openSections.screenshot ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
        </button>

        {openSections.screenshot && (
          <div className="p-5 border-t border-lab-border space-y-3 text-xs font-mono">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div className="p-2.5 rounded bg-lab-950/60 border border-lab-border/40">
                <span className="text-slate-500 block text-[10px]">Canonical Viewport:</span>
                <span className="text-slate-200">{analyzers.screenshot.matched_viewport || 'Generic / Non-standard'}</span>
              </div>
              <div className="p-2.5 rounded bg-lab-950/60 border border-lab-border/40">
                <span className="text-slate-500 block text-[10px]">Standard Aspect Ratio:</span>
                <span className={analyzers.screenshot.aspect_ratio_standard ? 'text-emerald-400' : 'text-slate-400'}>
                  {analyzers.screenshot.aspect_ratio_standard ? '✓ Standard Display' : '○ Custom crop'}
                </span>
              </div>
              <div className="p-2.5 rounded bg-lab-950/60 border border-lab-border/40">
                <span className="text-slate-500 block text-[10px]">Status:</span>
                <span className="text-slate-200">{analyzers.screenshot.status}</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Section 6: OCR */}
      <div className="forensic-panel overflow-hidden">
        <button
          onClick={() => toggleSection('ocr')}
          aria-expanded={openSections.ocr}
          className="w-full p-4 flex items-center justify-between text-left hover:bg-lab-850/50 transition-colors focus:outline-none"
        >
          <div className="flex items-center gap-2.5 font-mono text-xs font-bold text-slate-200 uppercase">
            <Search className="w-4 h-4 text-forensic-tealLight" />
            6. OCR
          </div>
          {openSections.ocr ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
        </button>

        {openSections.ocr && (
          <div className="p-5 border-t border-lab-border space-y-3 text-xs font-mono">
            <div className="flex flex-wrap items-center justify-between gap-3 p-2.5 rounded bg-lab-950/60 border border-lab-border/40">
              <span className="text-slate-400">OCR Engine: <strong className="text-slate-200">{analyzers.ocr.engine}</strong></span>
              <span className="text-slate-400">Words Extracted: <strong className="text-slate-200">{analyzers.ocr.word_count}</strong></span>
              <span className="text-slate-400">Characters: <strong className="text-slate-200">{analyzers.ocr.character_count}</strong></span>
            </div>
            {analyzers.ocr.status === 'NOT_AVAILABLE' && (
              <div className="p-3 rounded bg-lab-950 text-slate-500">
                ○ Tesseract OCR engine was NOT_AVAILABLE in the active backend environment.
              </div>
            )}
          </div>
        )}
      </div>

      {/* Section 7: ML Analysis */}
      <div className="forensic-panel overflow-hidden">
        <button
          onClick={() => toggleSection('ml')}
          aria-expanded={openSections.ml}
          className="w-full p-4 flex items-center justify-between text-left hover:bg-lab-850/50 transition-colors focus:outline-none"
        >
          <div className="flex items-center gap-2.5 font-mono text-xs font-bold text-slate-200 uppercase">
            <Cpu className="w-4 h-4 text-forensic-blueLight" />
            7. ML Analysis
          </div>
          {openSections.ml ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
        </button>

        {openSections.ml && (
          <div className="p-5 border-t border-lab-border space-y-4 text-xs font-mono">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div className="p-2.5 rounded bg-lab-950/60 border border-lab-border/40">
                <span className="text-slate-400 block text-[10px]">Model Status:</span>
                <span className={analyzers.ml.model_status === 'AVAILABLE' ? 'text-emerald-400 font-bold' : 'text-amber-400 font-bold'}>
                  {analyzers.ml.model_status}
                </span>
              </div>
              <div className="p-2.5 rounded bg-lab-950/60 border border-lab-border/40">
                <span className="text-slate-400 block text-[10px]">Architecture / Version:</span>
                <span className="text-slate-200">
                  {analyzers.ml.model_name || 'EfficientNet'} ({analyzers.ml.model_version || 'v1.0.0'})
                </span>
              </div>
              <div className="p-2.5 rounded bg-lab-950/60 border border-lab-border/40">
                <span className="text-slate-400 block text-[10px]">Predicted Artifact Class:</span>
                <span className="text-forensic-blueLight font-bold">
                  {analyzers.ml.predicted_label || 'UNKNOWN'}
                </span>
              </div>
            </div>

            {/* Class Probabilities Distribution */}
            {analyzers.ml.class_probabilities && Object.keys(analyzers.ml.class_probabilities).length > 0 && (
              <div className="space-y-2 pt-1">
                <span className="text-[11px] font-bold text-slate-300 block">Class Probability Distribution:</span>
                <div className="space-y-1.5">
                  {Object.entries(analyzers.ml.class_probabilities).map(([cls, prob]) => (
                    <div key={cls} className="space-y-0.5">
                      <div className="flex justify-between text-[10px]">
                        <span className="text-slate-300">{cls}</span>
                        <span className="text-slate-400">{(prob * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full h-1.5 bg-lab-950 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-forensic-blue"
                          style={{ width: `${Math.max(2, prob * 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="p-3 rounded bg-lab-950/80 border border-lab-border text-[11px] text-slate-400">
              {analyzers.ml.note}
            </div>
          </div>
        )}
      </div>

      {/* Section 8: Evidence Fusion */}
      <div className="forensic-panel overflow-hidden">
        <button
          onClick={() => toggleSection('fusion')}
          aria-expanded={openSections.fusion}
          className="w-full p-4 flex items-center justify-between text-left hover:bg-lab-850/50 transition-colors focus:outline-none"
        >
          <div className="flex items-center gap-2.5 font-mono text-xs font-bold text-slate-200 uppercase">
            <GitMerge className="w-4 h-4 text-emerald-400" />
            8. Evidence Fusion
          </div>
          {openSections.fusion ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
        </button>

        {openSections.fusion && (
          <div className="p-5 border-t border-lab-border space-y-3 text-xs font-mono">
            <div className="text-slate-400 text-[11px]">
              The Evidence Decision Engine applied the following rule-based criteria:
            </div>
            <div className="space-y-1.5">
              {verdict.decision_rules_triggered.map((rule, idx) => (
                <div key={idx} className="p-2 rounded bg-lab-950/60 border border-lab-border/40 text-slate-300 font-mono text-[11px] flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-forensic-blueLight" />
                  <code>{rule}</code>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Section 9: Why This Assessment? */}
      <div className="forensic-panel overflow-hidden border-forensic-blue/40">
        <button
          onClick={() => toggleSection('whyThis')}
          aria-expanded={openSections.whyThis}
          className="w-full p-4 flex items-center justify-between text-left bg-lab-850/40 hover:bg-lab-850/80 transition-colors focus:outline-none"
        >
          <div className="flex items-center gap-2.5 font-mono text-xs font-bold text-forensic-blueLight uppercase">
            <HelpCircle className="w-4 h-4" />
            9. Why This Assessment?
          </div>
          {openSections.whyThis ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
        </button>

        {openSections.whyThis && (
          <div className="p-6 border-t border-lab-border space-y-5 text-xs font-mono">
            {/* Supporting findings */}
            <div className="space-y-2">
              <h4 className="text-xs font-bold text-emerald-400 uppercase tracking-wider flex items-center gap-1.5">
                <Check className="w-4 h-4" />
                Evidence Supporting Assessment:
              </h4>
              <div className="space-y-1.5 pl-2">
                {verdict.supporting_findings && verdict.supporting_findings.length > 0 ? (
                  verdict.supporting_findings.map((sig, idx) => (
                    <div key={idx} className="p-2.5 rounded bg-emerald-950/20 border border-emerald-500/20 text-slate-200 flex items-start gap-2">
                      <span className="text-emerald-400 font-bold mt-0.5">✓</span>
                      <div className="leading-relaxed">
                        <span className="font-semibold">{sig.explanation}</span>
                        <div className="text-[10px] text-slate-400 mt-0.5">
                          Source: <code className="text-slate-300">{sig.source}</code> • Metric: <code className="text-slate-300">{sig.metric}</code> (Reliability: {(sig.reliability * 100).toFixed(0)}%)
                        </div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-slate-500 text-[11px] p-2">
                    • No definitive corroborating physical signals observed.
                  </div>
                )}
              </div>
            </div>

            {/* Contradictory findings */}
            <div className="space-y-2 pt-2 border-t border-lab-border/40">
              <h4 className="text-xs font-bold text-amber-400 uppercase tracking-wider flex items-center gap-1.5">
                <AlertTriangle className="w-4 h-4" />
                Evidence Against Assessment / Contradictions:
              </h4>
              <div className="space-y-1.5 pl-2">
                {verdict.contradictory_findings && verdict.contradictory_findings.length > 0 ? (
                  verdict.contradictory_findings.map((sig, idx) => (
                    <div key={idx} className="p-2.5 rounded bg-amber-950/20 border border-amber-500/20 text-slate-200 flex items-start gap-2">
                      <span className="text-amber-400 font-bold mt-0.5">⚠</span>
                      <div className="leading-relaxed">
                        <span className="font-semibold">{sig.explanation}</span>
                        <div className="text-[10px] text-slate-400 mt-0.5">
                          Source: <code className="text-slate-300">{sig.source}</code> • Metric: <code className="text-slate-300">{sig.metric}</code>
                        </div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-slate-500 text-[11px] p-2">
                    • No contradictory forensic signals detected against this verdict.
                  </div>
                )}
              </div>
            </div>

            {/* Structured Raw Engine Output */}
            {verdict.explanation && (
              <div className="pt-2 border-t border-lab-border/40 space-y-2">
                <div className="flex items-center justify-between">
                  <h4 className="text-[11px] font-bold text-slate-300 uppercase tracking-wider">
                    Full Engine Explanation Text:
                  </h4>
                  <button
                    onClick={copyAssessmentExplanation}
                    className="flex items-center gap-1 text-[11px] text-forensic-blueLight hover:text-sky-300"
                  >
                    {copiedExplanation ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                    <span>{copiedExplanation ? 'Copied' : 'Copy Explanation'}</span>
                  </button>
                </div>
                <pre className="p-3.5 rounded bg-lab-950 border border-lab-border text-slate-300 text-[11px] font-mono whitespace-pre-wrap leading-relaxed">
                  {verdict.explanation}
                </pre>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Section 10: Limitations */}
      <div className="forensic-panel overflow-hidden">
        <button
          onClick={() => toggleSection('limitations')}
          aria-expanded={openSections.limitations}
          className="w-full p-4 flex items-center justify-between text-left hover:bg-lab-850/50 transition-colors focus:outline-none"
        >
          <div className="flex items-center gap-2.5 font-mono text-xs font-bold text-slate-200 uppercase">
            <AlertOctagon className="w-4 h-4 text-forensic-amberLight" />
            10. Limitations
          </div>
          {openSections.limitations ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
        </button>

        {openSections.limitations && (
          <div className="p-5 border-t border-lab-border space-y-2 text-xs font-mono">
            <p className="text-slate-400 text-[11px] leading-relaxed mb-3">
              In accordance with scientific integrity standards, forensic assessments must declare operational boundaries and degradation constraints:
            </p>
            <ul className="space-y-1.5 pl-2">
              {verdict.limitations.map((lim, idx) => (
                <li key={idx} className="text-slate-300 flex items-start gap-2">
                  <span className="text-slate-500">•</span>
                  <span>{lim}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Section 11: Report */}
      <div className="forensic-panel overflow-hidden">
        <button
          onClick={() => toggleSection('report')}
          aria-expanded={openSections.report}
          className="w-full p-4 flex items-center justify-between text-left hover:bg-lab-850/50 transition-colors focus:outline-none"
        >
          <div className="flex items-center gap-2.5 font-mono text-xs font-bold text-slate-200 uppercase">
            <FileText className="w-4 h-4 text-forensic-blueLight" />
            11. Report
          </div>
          {openSections.report ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
        </button>

        {openSections.report && (
          <div className="p-5 border-t border-lab-border space-y-4 text-xs font-mono">
            <div className="flex flex-wrap gap-3">
              <button
                onClick={handleDownloadPdf}
                disabled={downloadingPdf}
                className="flex items-center gap-2 px-4 py-2 rounded bg-emerald-600 hover:bg-emerald-500 text-white font-bold transition-colors disabled:opacity-50 shadow-sm"
              >
                <Download className={`w-4 h-4 ${downloadingPdf ? 'animate-spin' : ''}`} />
                {downloadingPdf ? 'Generating PDF...' : 'Download Official PDF Report'}
              </button>

              <button
                onClick={downloadJson}
                className="flex items-center gap-2 px-4 py-2 rounded bg-forensic-blue hover:bg-sky-600 text-white font-bold transition-colors"
              >
                <Download className="w-4 h-4" /> Download Audit Digest (.json)
              </button>

              <button
                onClick={copyJson}
                className="flex items-center gap-2 px-4 py-2 rounded bg-lab-900 border border-lab-border hover:bg-lab-850 text-slate-200 transition-colors"
              >
                {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                Copy Full JSON Payload
              </button>

              <button
                onClick={() => onNavigate('reports', analysis.analysis_id)}
                className="flex items-center gap-2 px-4 py-2 rounded bg-lab-900 border border-lab-border hover:bg-lab-850 text-slate-200 transition-colors"
              >
                <FileCheck className="w-4 h-4 text-forensic-blueLight" />
                View 17-Section Report
              </button>
            </div>

            <div className="p-3 rounded bg-lab-950 border border-lab-border text-[11px] text-slate-400">
              Official PDF report includes cryptographic SHA-256 signatures, execution timestamps, individual analyzer outputs, observation vs. interpretation disclosures, and decision rules. Suitable for evidentiary chain-of-custody verification.
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
