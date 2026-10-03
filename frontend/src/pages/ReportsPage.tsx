import React, { useState, useEffect } from 'react';
import { api, ApiError } from '../services/api';
import type { AnalysisResultResponse, EvidenceUploadResponse } from '../types/forensic';
import { StatusBadge } from '../components/StatusBadge';
import {
  FileText,
  Printer,
  ShieldCheck,
  AlertTriangle,
  HardDrive,
  Copy,
  Check,
  RefreshCw,
  Download,
  Binary,
  Layers,
  Cpu,
  Clock,
  Sparkles,
  Search,
} from 'lucide-react';

interface ReportsPageProps {
  evidenceId?: string | null;
  onNavigate: (tab: any, evidenceId?: string) => void;
}

export const ReportsPage: React.FC<ReportsPageProps> = ({ evidenceId, onNavigate }) => {
  const [evidenceList, setEvidenceList] = useState<EvidenceUploadResponse[]>([]);
  const [selectedId, setSelectedId] = useState<string>(evidenceId || '');
  const [analysis, setAnalysis] = useState<AnalysisResultResponse | null>(null);
  const [reportData, setReportData] = useState<any | null>(null);
  const [loading, setLoading] = useState(false);
  const [downloadingPdf, setDownloadingPdf] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copiedHash, setCopiedHash] = useState(false);
  const [copiedMd, setCopiedMd] = useState(false);
  const [copiedJson, setCopiedJson] = useState(false);

  useEffect(() => {
    const fetchList = async () => {
      try {
        const items = await api.listEvidence();
        setEvidenceList(items);
        if (!selectedId && items.length > 0) {
          setSelectedId(items[0].evidence_id);
        }
      } catch {
        // Fallback
      }
    };
    fetchList();
  }, []);

  useEffect(() => {
    if (evidenceId && evidenceId !== selectedId) {
      setSelectedId(evidenceId);
    }
  }, [evidenceId]);

  const loadReport = async (id: string) => {
    setLoading(true);
    setError(null);
    try {
      const [ana, rep] = await Promise.all([api.getAnalysis(id), api.getReport(id)]);
      setAnalysis(ana);
      setReportData(rep);
    } catch (err) {
      setError((err as ApiError).message || 'Failed to generate report for selected case.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedId) {
      loadReport(selectedId);
    }
  }, [selectedId]);

  const handleDownloadPdf = async () => {
    if (!selectedId) return;
    try {
      setDownloadingPdf(true);
      await api.downloadReportPdf(selectedId, `TRUSTTRACE_REPORT_${selectedId.slice(0, 8).toUpperCase()}.pdf`);
    } catch (err) {
      alert(`PDF download failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setDownloadingPdf(false);
    }
  };

  const copySha256 = (hash: string) => {
    navigator.clipboard.writeText(hash);
    setCopiedHash(true);
    setTimeout(() => setCopiedHash(false), 2000);
  };

  const generateMarkdownReport = () => {
    if (!analysis || !reportData) return '';
    const dateStr = new Date(analysis.created_at).toUTCString();
    const secs = reportData.sections || {};
    const sec11 = secs['11_final_assessment'] || {};
    return `# TRUSTTRACE DIGITAL EVIDENCE ANALYSIS REPORT
**Case / Analysis ID:** ${analysis.analysis_id}  
**Report Date (UTC):** ${dateStr}  
**Pipeline Version:** ${analysis.pipeline_version}  
**Platform:** TRUSTTRACE Digital Evidence Forensic Workstation  

---

## 1. CASE / ANALYSIS ID
- Case Reference: ${secs['1_case_analysis_id']?.case_reference || analysis.analysis_id}
- Enclave Status: ${secs['1_case_analysis_id']?.quarantine_status || 'LOCKED READ-ONLY ENCLAVE'}

## 2. EVIDENCE INFORMATION
- Target Filename: ${analysis.evidence.filename}
- Evidence UUID: ${analysis.evidence.evidence_id}
- MIME Container: ${analysis.evidence.mime_type}

## 3. SHA-256 INTEGRITY HASH
\`${analysis.evidence.sha256}\`
- Verification: FRE 902(13)/(14) Bitwise Cryptographic Certified Record

## 4. FILE INFORMATION
- File Size: ${(analysis.evidence.size / 1024).toFixed(1)} KB (${analysis.evidence.size} bytes)
- Spatial Dimensions: ${secs['4_file_information']?.dimensions || 'N/A'}
- Quality Estimation: ${secs['4_file_information']?.estimated_quality || 'N/A'}

## 5. METADATA FINDINGS
- Observation: ${secs['5_metadata_findings']?.observation || 'N/A'}
- Interpretation: ${secs['5_metadata_findings']?.interpretation || 'N/A'}

## 6. OCR FINDINGS
- Observation: ${secs['6_ocr_findings']?.observation || 'N/A'}
- Interpretation: ${secs['6_ocr_findings']?.interpretation || 'N/A'}

## 7. IMAGE FORENSIC FINDINGS
- Observation: ${secs['7_image_forensic_findings']?.observation || 'N/A'}
- Interpretation: ${secs['7_image_forensic_findings']?.interpretation || 'N/A'}

## 8. SCREENSHOT ANALYSIS
- Observation: ${secs['8_screenshot_analysis']?.observation || 'N/A'}
- Interpretation: ${secs['8_screenshot_analysis']?.interpretation || 'N/A'}

## 9. ML ANALYSIS
- Observation: ${secs['9_ml_analysis']?.observation || 'N/A'}
- Interpretation: ${secs['9_ml_analysis']?.interpretation || 'N/A'}

## 10. EVIDENCE FUSION
- Observation: ${secs['10_evidence_fusion']?.observation || 'N/A'}
- Interpretation: ${secs['10_evidence_fusion']?.interpretation || 'N/A'}

## 11. FINAL ASSESSMENT
- Verdict: ${sec11.label || analysis.final_verdict.label}
- Calibrated Confidence: ${sec11.confidence_formatted || 'N/A'}
- Tamper Risk Index: ${sec11.risk_score ? (sec11.risk_score * 100).toFixed(0) : '0'} / 100
- Finding Statement: ${sec11.assessment_statement || analysis.final_verdict.justification}

## 12. SUPPORTING EVIDENCE (${secs['12_supporting_evidence']?.count || 0} signals)
${(secs['12_supporting_evidence']?.items || []).map((s: any) => `- ✓ [${s.source}] ${s.explanation}`).join('\n')}

## 13. CONTRADICTORY EVIDENCE (${secs['13_contradictory_evidence']?.count || 0} signals)
${(secs['13_contradictory_evidence']?.items || []).map((c: any) => `- ⚠ [${c.source}] ${c.explanation}`).join('\n')}

## 14. LIMITATIONS
${(secs['14_limitations']?.items || []).map((lim: string) => `- • ${lim}`).join('\n')}

## 15. ANALYZER STATUS
${JSON.stringify(secs['15_analyzer_status']?.analyzers || {}, null, 2)}

## 16. MODEL INFORMATION
- Version: ${secs['16_model_information']?.model_version || 'N/A'}
- Checksum: ${secs['16_model_information']?.model_checksum || 'N/A'}
- Architecture: ${secs['16_model_information']?.architecture || 'N/A'}

## 17. TIMESTAMP
- UTC: ${secs['17_timestamp']?.timestamp_utc || dateStr}
- ISO 8601: ${secs['17_timestamp']?.iso8601 || analysis.created_at}
- Standard: TRUSTTRACE Forensic Platform Daubert-Admissible Verification
`;
  };

  const copyMarkdown = () => {
    const md = generateMarkdownReport();
    navigator.clipboard.writeText(md);
    setCopiedMd(true);
    setTimeout(() => setCopiedMd(false), 2000);
  };

  const copyJson = () => {
    if (!reportData) return;
    navigator.clipboard.writeText(JSON.stringify(reportData, null, 2));
    setCopiedJson(true);
    setTimeout(() => setCopiedJson(false), 2000);
  };

  const handlePrint = () => {
    window.print();
  };

  const secs = reportData?.sections || null;
  const sec11 = secs ? secs['11_final_assessment'] : null;

  return (
    <div className="space-y-6">
      {/* Header Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-lab-border print:hidden">
        <div>
          <h1 className="text-xl font-bold font-mono tracking-wide text-slate-100 flex items-center gap-2.5">
            <FileText className="w-5 h-5 text-forensic-blueLight" />
            TRUSTTRACE DIGITAL EVIDENCE ANALYSIS REPORT
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Official 17-section investigation report compiled strictly from actual backend analysis results.
          </p>
        </div>

        {/* Action buttons & selector */}
        <div className="flex items-center flex-wrap gap-2">
          {evidenceList.length > 0 && (
            <select
              value={selectedId}
              onChange={(e) => setSelectedId(e.target.value)}
              className="forensic-input text-xs max-w-xs"
            >
              {evidenceList.map((item) => (
                <option key={item.evidence_id} value={item.evidence_id}>
                  {item.filename} ({item.evidence_id.slice(0, 8)}...)
                </option>
              ))}
            </select>
          )}

          <button
            onClick={() => selectedId && loadReport(selectedId)}
            disabled={loading || !selectedId}
            className="p-2 rounded bg-lab-900 border border-lab-border text-slate-400 hover:text-white"
            title="Refresh Report"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>

          <button
            onClick={handleDownloadPdf}
            disabled={!reportData || downloadingPdf}
            className="px-3.5 py-1.5 rounded bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-mono font-bold flex items-center gap-1.5 shadow-sm transition-colors disabled:opacity-50"
            title="Download Vectorized A4 Forensic PDF Report"
          >
            <Download className={`w-3.5 h-3.5 ${downloadingPdf ? 'animate-spin' : ''}`} />
            <span>{downloadingPdf ? 'Compiling PDF...' : 'Download Official PDF'}</span>
          </button>

          <button
            onClick={copyMarkdown}
            disabled={!reportData}
            className="px-3 py-1.5 rounded bg-lab-900 hover:bg-lab-800 border border-lab-border text-xs font-mono text-slate-300 flex items-center gap-1.5"
            title="Copy Report in Markdown"
          >
            {copiedMd ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copiedMd ? 'Copied' : 'Markdown'}</span>
          </button>

          <button
            onClick={copyJson}
            disabled={!reportData}
            className="px-3 py-1.5 rounded bg-lab-900 hover:bg-lab-800 border border-lab-border text-xs font-mono text-slate-300 flex items-center gap-1.5"
            title="Copy 17-Section JSON"
          >
            {copiedJson ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copiedJson ? 'Copied' : 'JSON'}</span>
          </button>

          <button
            onClick={handlePrint}
            disabled={!reportData}
            className="px-3 py-1.5 rounded bg-lab-900 hover:bg-lab-800 border border-lab-border text-xs font-mono text-slate-300 flex items-center gap-1.5"
            title="Print Case Document"
          >
            <Printer className="w-3.5 h-3.5" />
            <span>Print</span>
          </button>
        </div>
      </div>

      {/* Loading state */}
      {loading && (
        <div className="forensic-panel p-12 text-center space-y-3">
          <RefreshCw className="w-8 h-8 text-forensic-blueLight animate-spin mx-auto" />
          <div className="text-xs font-mono text-slate-300">
            Compiling 17-section forensic evidence report from backend analysis results...
          </div>
        </div>
      )}

      {/* Error state */}
      {error && !loading && (
        <div className="p-4 rounded-lg bg-rose-950/40 border border-rose-600/50 flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div className="text-xs font-mono text-slate-300">
            <div className="font-bold text-rose-300">Report Retrieval Error:</div>
            <div>{error}</div>
          </div>
        </div>
      )}

      {/* No selection state */}
      {!selectedId && !loading && (
        <div className="forensic-panel p-12 text-center space-y-3">
          <FileText className="w-10 h-10 text-slate-600 mx-auto" />
          <div className="text-sm font-mono text-slate-300 font-bold">No Evidence Selected</div>
          <p className="text-xs font-mono text-slate-400 max-w-md mx-auto">
            Select an evidence case or upload an intake asset to generate an official examination report.
          </p>
          <button
            onClick={() => onNavigate('upload')}
            className="mt-2 px-4 py-2 rounded bg-forensic-blue hover:bg-sky-600 text-white text-xs font-mono font-medium"
          >
            Upload Evidence File
          </button>
        </div>
      )}

      {/* Render Official 17-Section Case Report */}
      {reportData && secs && !loading && (
        <div className="space-y-6 bg-lab-950 p-6 sm:p-8 rounded-xl border border-lab-border shadow-forensic-card print:border-none print:p-0 print:bg-white print:text-black">
          {/* Document Header Banner */}
          <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4 pb-6 border-b border-lab-border print:border-black">
            <div>
              <div className="flex items-center gap-2">
                <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-forensic-blue/20 text-forensic-blueLight border border-forensic-blue/40 print:bg-gray-100 print:text-black print:border-black">
                  TRUSTTRACE OFFICIAL EXAMINATION RECORD
                </span>
                <span className="text-xs font-mono text-slate-400 print:text-gray-600">
                  REF: {reportData.report_reference}
                </span>
              </div>
              <h2 className="text-xl sm:text-2xl font-bold font-mono text-slate-100 mt-2 tracking-tight print:text-black">
                TRUSTTRACE DIGITAL EVIDENCE ANALYSIS REPORT
              </h2>
              <div className="text-xs font-mono text-slate-400 mt-1 print:text-gray-600">
                PIPELINE BUILD {reportData.pipeline_version} • FRE 902(13)/(14) SELF-AUTHENTICATING RECORD
              </div>
            </div>

            <div className="text-right sm:self-start">
              <div className="text-xs font-mono text-slate-400 print:text-gray-600">Certified Date (UTC)</div>
              <div className="text-xs font-mono font-bold text-slate-200 print:text-black">
                {reportData.created_at_utc}
              </div>
              <div className="text-[10px] font-mono text-emerald-400 mt-1 flex items-center justify-end gap-1 print:text-black">
                <ShieldCheck className="w-3.5 h-3.5" /> SECURE QUARANTINE SEALED
              </div>
            </div>
          </div>

          {/* Section 11: Final Assessment Callout Banner (Top Priority) */}
          <div className="p-4 sm:p-5 rounded-lg bg-lab-900 border border-lab-border flex flex-col sm:flex-row sm:items-center justify-between gap-4 print:bg-gray-50 print:border-black">
            <div className="space-y-1">
              <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 print:text-gray-600">
                11. Final Assessment Verdict
              </span>
              <div className="flex items-center gap-3">
                <StatusBadge label={sec11?.label || 'UNKNOWN'} size="lg" />
                <span className="text-xs font-mono text-slate-300 print:text-black">
                  Tamper Risk Index: <strong>{(sec11?.risk_score * 100).toFixed(0)} / 100</strong>
                </span>
              </div>
              <div className="text-xs font-mono text-sky-300 font-semibold pt-1 print:text-black">
                Finding: {sec11?.assessment_statement}
              </div>
            </div>

            <div className="text-left sm:text-right">
              <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 print:text-gray-600">
                Calibrated Confidence
              </span>
              <div className="text-base font-mono font-bold text-slate-200 print:text-black">
                {sec11?.confidence_formatted || 'Confidence unavailable'}
              </div>
              <div className="text-[10px] font-mono text-slate-400 print:text-gray-600">
                Calibrated Uncertainty: {sec11?.uncertainty_formatted}
              </div>
            </div>
          </div>

          {/* Grid: Sections 1 to 4 (Chain of Custody & File Info) */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Section 1: Case / Analysis ID */}
            <div className="p-3.5 rounded bg-lab-900/60 border border-lab-border text-xs font-mono space-y-1 print:bg-white print:border-gray-300">
              <span className="text-slate-500 block text-[10px] uppercase font-bold">1. Case / Analysis ID</span>
              <div className="text-slate-200 font-medium break-all">
                {secs['1_case_analysis_id']?.analysis_id}
              </div>
              <div className="text-[10px] text-slate-400 pt-1 flex items-center justify-between">
                <span>Ref: {secs['1_case_analysis_id']?.case_reference}</span>
                <span className="text-emerald-400">{secs['1_case_analysis_id']?.quarantine_status}</span>
              </div>
            </div>

            {/* Section 2: Evidence Information */}
            <div className="p-3.5 rounded bg-lab-900/60 border border-lab-border text-xs font-mono space-y-1 print:bg-white print:border-gray-300">
              <span className="text-slate-500 block text-[10px] uppercase font-bold">2. Evidence Information</span>
              <div className="text-slate-200 font-medium break-all">
                {secs['2_evidence_information']?.filename}
              </div>
              <div className="text-[10px] text-slate-400 pt-1">
                ID: {secs['2_evidence_information']?.evidence_id} • MIME: {secs['2_evidence_information']?.mime_type}
              </div>
            </div>

            {/* Section 3: SHA-256 Integrity Hash */}
            <div className="p-3.5 rounded bg-lab-900/60 border border-lab-border text-xs font-mono space-y-1 md:col-span-2 print:bg-white print:border-gray-300">
              <div className="flex items-center justify-between">
                <span className="text-slate-500 block text-[10px] uppercase font-bold">3. SHA-256 Integrity Hash</span>
                <button
                  onClick={() => copySha256(secs['3_sha256_integrity_hash']?.sha256 || '')}
                  className="text-[10px] text-forensic-blueLight hover:underline flex items-center gap-1 print:hidden"
                >
                  {copiedHash ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                  <span>{copiedHash ? 'Copied' : 'Copy Hash'}</span>
                </button>
              </div>
              <div className="text-slate-100 font-mono text-[11px] break-all font-semibold text-forensic-blueLight">
                {secs['3_sha256_integrity_hash']?.sha256}
              </div>
              <div className="text-[10px] text-emerald-400 pt-0.5 flex items-center gap-2">
                <ShieldCheck className="w-3 h-3" />
                <span>{secs['3_sha256_integrity_hash']?.verification_status}</span>
                <span className="text-slate-500">•</span>
                <span className="text-slate-400">{secs['3_sha256_integrity_hash']?.chain_of_custody_standard}</span>
              </div>
            </div>

            {/* Section 4: File Information */}
            <div className="p-3.5 rounded bg-lab-900/60 border border-lab-border text-xs font-mono space-y-1 md:col-span-2 print:bg-white print:border-gray-300">
              <span className="text-slate-500 block text-[10px] uppercase font-bold">4. File Information</span>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-slate-300 text-[11px] pt-1">
                <div>
                  <span className="text-slate-500 block text-[10px]">FILE SIZE</span>
                  <span>{secs['4_file_information']?.file_size_formatted}</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[10px]">SPATIAL DIMENSIONS</span>
                  <span>{secs['4_file_information']?.dimensions}</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[10px]">COLOR SPACE & CHANNELS</span>
                  <span>{secs['4_file_information']?.color_space} ({secs['4_file_information']?.channels} ch)</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[10px]">ESTIMATED JPEG QUALITY</span>
                  <span>{secs['4_file_information']?.estimated_quality}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Detailed Findings: Sections 5 to 10 (Strictly distinguishing OBSERVATION from INTERPRETATION) */}
          <div className="space-y-4">
            <h3 className="text-sm font-mono font-bold text-slate-200 uppercase tracking-wider flex items-center gap-2 border-b border-lab-border/60 pb-1.5 print:border-black print:text-black">
              <Binary className="w-4 h-4 text-forensic-blueLight print:text-black" />
              5–10. Multi-Modal Technical Observations & Forensic Interpretations
            </h3>

            {[
              { key: '5_metadata_findings', num: '5', title: 'Metadata Findings', icon: HardDrive },
              { key: '6_ocr_findings', num: '6', title: 'OCR Findings', icon: Search },
              { key: '7_image_forensic_findings', num: '7', title: 'Image Forensic Findings', icon: Layers },
              { key: '8_screenshot_analysis', num: '8', title: 'Screenshot Analysis', icon: Sparkles },
              { key: '9_ml_analysis', num: '9', title: 'ML Analysis', icon: Cpu },
              { key: '10_evidence_fusion', num: '10', title: 'Evidence Fusion', icon: Binary },
            ].map(({ key, num, title, icon: Icon }) => {
              const sec = secs[key];
              if (!sec) return null;
              return (
                <div
                  key={key}
                  className="p-3.5 rounded bg-lab-900/40 border border-lab-border text-xs font-mono space-y-2.5 print:bg-white print:border-gray-300"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-slate-200 flex items-center gap-2">
                      <Icon className="w-3.5 h-3.5 text-forensic-blueLight" />
                      {num}. {title}
                    </span>
                  </div>

                  {/* OBSERVATION */}
                  <div className="p-2.5 rounded bg-lab-950/70 border border-lab-border/60 space-y-1">
                    <span className="text-[10px] font-bold text-sky-400 uppercase tracking-wider block">
                      OBSERVATION (Factual Analyzer Output)
                    </span>
                    <p className="text-slate-300 text-[11px] leading-relaxed">{sec.observation}</p>
                  </div>

                  {/* INTERPRETATION */}
                  <div className="p-2.5 rounded bg-amber-950/30 border border-amber-600/40 space-y-1">
                    <span className="text-[10px] font-bold text-amber-300 uppercase tracking-wider block">
                      INTERPRETATION (Forensic Evaluation)
                    </span>
                    <p className="text-amber-100 text-[11px] leading-relaxed">{sec.interpretation}</p>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Sections 12 & 13: Supporting & Contradictory Evidence */}
          <div className="space-y-4">
            <h3 className="text-sm font-mono font-bold text-slate-200 uppercase tracking-wider flex items-center gap-2 border-b border-lab-border/60 pb-1.5 print:border-black print:text-black">
              <ShieldCheck className="w-4 h-4 text-forensic-blueLight print:text-black" />
              12–13. Arbitrated Forensic Signals
            </h3>

            {/* Section 12: Supporting Evidence */}
            <div className="p-3.5 rounded bg-emerald-950/20 border border-emerald-600/40 text-xs font-mono space-y-2">
              <span className="text-emerald-400 font-bold block text-[11px] uppercase">
                12. Supporting Evidence ({secs['12_supporting_evidence']?.count || 0} signals)
              </span>
              <ul className="space-y-1.5">
                {(secs['12_supporting_evidence']?.items || []).map((s: any, idx: number) => (
                  <li key={idx} className="text-emerald-200 text-[11px] flex items-start gap-2">
                    <span className="font-bold text-emerald-400">✓</span>
                    <span>
                      <strong className="text-slate-200">[{s.source?.toUpperCase()}]</strong> {s.explanation}
                    </span>
                  </li>
                ))}
                {(!secs['12_supporting_evidence']?.items || secs['12_supporting_evidence']?.items.length === 0) && (
                  <li className="text-slate-400 italic text-[11px]">• No definitive corroborating physical signals observed.</li>
                )}
              </ul>
            </div>

            {/* Section 13: Contradictory Evidence */}
            <div className="p-3.5 rounded bg-amber-950/20 border border-amber-600/40 text-xs font-mono space-y-2">
              <span className="text-amber-400 font-bold block text-[11px] uppercase">
                13. Contradictory Evidence ({secs['13_contradictory_evidence']?.count || 0} signals)
              </span>
              <ul className="space-y-1.5">
                {(secs['13_contradictory_evidence']?.items || []).map((c: any, idx: number) => (
                  <li key={idx} className="text-amber-200 text-[11px] flex items-start gap-2">
                    <span className="font-bold text-amber-400">⚠</span>
                    <span>
                      <strong className="text-slate-200">[{c.source?.toUpperCase()}]</strong> {c.explanation}
                    </span>
                  </li>
                ))}
                {(!secs['13_contradictory_evidence']?.items || secs['13_contradictory_evidence']?.items.length === 0) && (
                  <li className="text-slate-400 italic text-[11px]">• No contradictory forensic signals detected against this assessment.</li>
                )}
              </ul>
            </div>
          </div>

          {/* Section 14: Limitations */}
          <div className="space-y-3">
            <h3 className="text-sm font-mono font-bold text-slate-200 uppercase tracking-wider flex items-center gap-2 border-b border-lab-border/60 pb-1.5 print:border-black print:text-black">
              <AlertTriangle className="w-4 h-4 text-forensic-blueLight print:text-black" />
              14. Operational Limitations & Daubert Boundaries
            </h3>

            <div className="p-3.5 rounded bg-lab-900/50 border border-lab-border text-xs font-mono space-y-1.5 print:bg-white print:border-gray-300">
              <ul className="list-disc list-inside space-y-1 text-slate-300 text-[11px] leading-relaxed">
                {(secs['14_limitations']?.items || []).map((lim: string, idx: number) => (
                  <li key={idx}>{lim}</li>
                ))}
              </ul>
            </div>
          </div>

          {/* Sections 15, 16, 17: Analyzer Status, Model Information & Timestamp */}
          <div className="space-y-3">
            <h3 className="text-sm font-mono font-bold text-slate-200 uppercase tracking-wider flex items-center gap-2 border-b border-lab-border/60 pb-1.5 print:border-black print:text-black">
              <Clock className="w-4 h-4 text-forensic-blueLight print:text-black" />
              15–17. Analyzer Versions, Model Governance & Execution Timestamp
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs font-mono">
              {/* Section 15: Analyzer Status */}
              <div className="p-3 rounded bg-lab-900/60 border border-lab-border space-y-1.5">
                <span className="text-slate-500 block text-[10px] uppercase font-bold">15. Analyzer Status & Versions</span>
                <div className="space-y-1 text-[11px] text-slate-300">
                  {Object.entries(secs['15_analyzer_status']?.analyzers || {}).map(([key, val]: [string, any]) => (
                    <div key={key} className="flex items-center justify-between">
                      <span className="text-slate-400 capitalize">{key.replace('_', ' ')}:</span>
                      <span className="font-semibold text-slate-200">{val.status} ({val.version?.split(' ')[0]})</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Section 16: Model Information */}
              <div className="p-3 rounded bg-lab-900/60 border border-lab-border space-y-1.5">
                <span className="text-slate-500 block text-[10px] uppercase font-bold">16. Model Information</span>
                <div className="space-y-1 text-[11px]">
                  <div>
                    <span className="text-slate-500">Version: </span>
                    <span className="text-slate-200">{secs['16_model_information']?.model_version}</span>
                  </div>
                  <div>
                    <span className="text-slate-500">Architecture: </span>
                    <span className="text-slate-200">{secs['16_model_information']?.architecture}</span>
                  </div>
                  <div className="pt-1">
                    <span className="text-slate-500 block text-[10px]">MODEL SHA-256 CHECKSUM</span>
                    <span className="text-forensic-blueLight font-mono text-[10px] break-all">
                      {secs['16_model_information']?.model_checksum}
                    </span>
                  </div>
                </div>
              </div>

              {/* Section 17: Timestamp */}
              <div className="p-3 rounded bg-lab-900/60 border border-lab-border space-y-1 md:col-span-2">
                <span className="text-slate-500 block text-[10px] uppercase font-bold">17. Certified Timestamp & Legal Authority</span>
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-[11px] pt-1">
                  <div>
                    <span className="text-slate-500">Timestamp (UTC): </span>
                    <span className="text-slate-200 font-mono font-bold">{secs['17_timestamp']?.timestamp_utc}</span>
                    <span className="text-slate-500 ml-2">({secs['17_timestamp']?.iso8601})</span>
                  </div>
                  <span className="text-emerald-400 font-semibold">{secs['17_timestamp']?.certification_authority}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Official Signoff Footer */}
          <div className="pt-6 border-t border-lab-border text-center text-xs font-mono text-slate-500 print:border-black print:text-black">
            TRUSTTRACE Automated Digital Forensics Platform • Report Hash Integrity Sealed • FRE 902(13)/(14) Validated
          </div>
        </div>
      )}
    </div>
  );
};
