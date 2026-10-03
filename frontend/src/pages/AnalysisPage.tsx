import React, { useEffect, useState } from 'react';
import type { AnalysisResultResponse } from '../types/forensic';
import { api, ApiError } from '../services/api';
import { StatusBadge } from '../components/StatusBadge';
import { ResultPage } from './ResultPage';
import {
  Microscope,
  CheckCircle2,
  Clock,
  AlertCircle,
  FileCheck,
  ShieldCheck,
  Binary,
  Layers,
  Search,
  Cpu,
  GitMerge,
  Eye,
  RefreshCw,
  ArrowDown,
} from 'lucide-react';

interface AnalysisPageProps {
  evidenceId?: string | null;
  initialMode?: 'timeline' | 'assessment';
  onNavigate: (tab: any, evidenceId?: string) => void;
}

export const AnalysisPage: React.FC<AnalysisPageProps> = ({
  evidenceId,
  initialMode = 'timeline',
  onNavigate,
}) => {
  const [analysis, setAnalysis] = useState<AnalysisResultResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [evidenceList, setEvidenceList] = useState<Array<{ id: string; name: string }>>([]);
  const [selectedId, setSelectedId] = useState<string>(evidenceId || '');
  const [viewMode, setViewMode] = useState<'timeline' | 'assessment'>(initialMode);

  useEffect(() => {
    // Load available evidence list for selection
    const loadList = async () => {
      try {
        const items = await api.listEvidence();
        const mapped = items.map((i) => ({ id: i.evidence_id, name: i.filename }));
        setEvidenceList(mapped);
        if (!selectedId && mapped.length > 0) {
          setSelectedId(mapped[0].id);
        }
      } catch {
        // Fallback
      }
    };
    loadList();
  }, []);

  useEffect(() => {
    if (evidenceId && evidenceId !== selectedId) {
      setSelectedId(evidenceId);
    }
  }, [evidenceId]);

  const loadAnalysis = async (id: string) => {
    if (!id) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.getAnalysis(id);
      setAnalysis(res);
    } catch (err) {
      setError((err as ApiError).message || `Failed to retrieve analysis for evidence ${id}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedId) {
      loadAnalysis(selectedId);
    }
  }, [selectedId]);

  // Investigation Timeline Stages (Strict 9-stage sequence matching prompt requirements)
  const getTimelineStages = () => {
    if (!analysis) return [];

    const meta = analysis.analyzers.metadata;
    const img = analysis.analyzers.image;
    const ocr = analysis.analyzers.ocr;
    const screen = analysis.analyzers.screenshot;
    const ml = analysis.analyzers.ml;
    const fusion = analysis.final_verdict;

    return [
      {
        step: 1,
        title: 'Evidence Received',
        status: 'COMPLETED',
        details: `Quarantined into vault as '${analysis.evidence.filename}' (${(analysis.evidence.size / 1024).toFixed(1)} KB, MIME: ${analysis.evidence.mime_type})`,
        icon: FileCheck,
      },
      {
        step: 2,
        title: 'Integrity Verification',
        status: 'COMPLETED',
        details: `SHA-256: ${analysis.evidence.sha256.slice(0, 16)}... (Cryptographically Sealed & Verified)`,
        icon: ShieldCheck,
      },
      {
        step: 3,
        title: 'Metadata Extraction',
        status: meta.status || 'COMPLETED',
        details: meta.exif_present
          ? `EXIF Present (${meta.camera_make || 'Unknown Make'} ${meta.camera_model || ''}) • Software: ${meta.software || 'None Recorded'}`
          : 'No EXIF metadata present (Container stripped or synthetic)',
        icon: Binary,
      },
      {
        step: 4,
        title: 'Forensic Analysis',
        status: img.status || 'COMPLETED',
        details: `ELA Variance: ${img.ela_variance?.toFixed(1) || '0.0'} • JPEG Quality: ${img.estimated_jpeg_quality || 'Lossless'} • Spatial Noise Ratio: ${img.luminance_std.toFixed(1)}`,
        icon: Microscope,
      },
      {
        step: 5,
        title: 'OCR',
        status: ocr.status || 'COMPLETED',
        details: ocr.status === 'NOT_AVAILABLE'
          ? 'Tesseract OCR engine NOT_AVAILABLE in current runtime'
          : `Extracted ${ocr.word_count} words • ${ocr.character_count} characters`,
        icon: Search,
      },
      {
        step: 6,
        title: 'Screenshot Analysis',
        status: screen.status || 'COMPLETED',
        details: screen.is_common_viewport
          ? `Matches canonical viewport '${screen.matched_viewport}'`
          : 'Non-standard capture geometry or camera photo',
        icon: Layers,
      },
      {
        step: 7,
        title: 'ML Analysis',
        status: ml.model_status || 'NOT_AVAILABLE',
        details: ml.model_status === 'AVAILABLE'
          ? `Predicted ${ml.predicted_label} (Conf: ${((ml.confidence || 0) * 100).toFixed(0)}%, Uncertainty: ${ml.uncertainty?.toFixed(3) || 'N/A'})`
          : 'Deep learning weights NOT_AVAILABLE in active runtime (Skipping inference to avoid fake results)',
        icon: Cpu,
      },
      {
        step: 8,
        title: 'Evidence Fusion',
        status: 'COMPLETED',
        details: fusion.conflict_detected
          ? `Conflict Arbitrated: ${fusion.conflict_details || 'Cross-modal discrepancy resolved'}`
          : `Synthesized ${fusion.decision_rules_triggered.length} forensic rule(s) without contradiction`,
        icon: GitMerge,
      },
      {
        step: 9,
        title: 'Final Assessment',
        status: 'COMPLETED',
        details: `Classification: ${fusion.label} • Calibrated Confidence: ${fusion.confidence !== null ? `${(fusion.confidence * 100).toFixed(0)}%` : 'Confidence unavailable'}`,
        icon: CheckCircle2,
      },
    ];
  };

  const stages = getTimelineStages();

  return (
    <div className="space-y-6">
      {/* Top Header & View Mode Switcher */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-lab-border">
        <div>
          <h1 className="text-xl font-bold font-mono tracking-wide text-slate-100 flex items-center gap-2.5">
            <Microscope className="w-5 h-5 text-forensic-blueLight" />
            FORENSIC ANALYSIS & TIMELINE EXAMINATION
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Nine-stage investigation verification timeline and comprehensive evidentiary assessment.
          </p>
        </div>

        {/* Controls: Mode Switcher + Case Selector */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Mode Switch Tabs */}
          <div className="flex items-center p-1 rounded-lg bg-lab-950 border border-lab-border">
            <button
              onClick={() => setViewMode('timeline')}
              className={`px-3 py-1 rounded text-xs font-mono transition-colors flex items-center gap-1.5 ${
                viewMode === 'timeline'
                  ? 'bg-lab-800 text-forensic-blueLight font-bold border border-forensic-blue/40 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Clock className="w-3.5 h-3.5" />
              <span>Timeline (9 Stages)</span>
            </button>
            <button
              onClick={() => setViewMode('assessment')}
              className={`px-3 py-1 rounded text-xs font-mono transition-colors flex items-center gap-1.5 ${
                viewMode === 'assessment'
                  ? 'bg-lab-800 text-forensic-blueLight font-bold border border-forensic-blue/40 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Eye className="w-3.5 h-3.5" />
              <span>Full Assessment (11 Sections)</span>
            </button>
          </div>

          {/* Case dropdown */}
          {evidenceList.length > 0 && (
            <select
              value={selectedId}
              onChange={(e) => setSelectedId(e.target.value)}
              className="forensic-input text-xs max-w-xs"
            >
              {evidenceList.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name} ({item.id.slice(0, 8)}...)
                </option>
              ))}
            </select>
          )}

          <button
            onClick={() => loadAnalysis(selectedId)}
            disabled={loading || !selectedId}
            className="p-2 rounded bg-lab-900 border border-lab-border text-slate-400 hover:text-white"
            title="Reload Timeline"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Error state */}
      {error && (
        <div className="p-4 rounded-lg bg-rose-950/40 border border-rose-600/50 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div className="text-xs font-mono text-slate-300">
            <div className="font-bold text-rose-300">Analysis Retrieval Error:</div>
            <div>{error}</div>
          </div>
        </div>
      )}

      {/* No selection state */}
      {!selectedId && !loading && (
        <div className="forensic-panel p-12 text-center space-y-3">
          <Microscope className="w-10 h-10 text-slate-600 mx-auto" />
          <div className="text-sm font-mono text-slate-300 font-bold">No Evidence Selected</div>
          <p className="text-xs font-mono text-slate-400 max-w-md mx-auto">
            Upload an evidence file or choose an existing quarantined case from the registry to view the verification timeline.
          </p>
          <button
            onClick={() => onNavigate('upload')}
            className="mt-2 px-4 py-2 rounded bg-forensic-blue hover:bg-sky-600 text-white text-xs font-mono font-medium"
          >
            Upload Evidence File
          </button>
        </div>
      )}

      {/* Loading state */}
      {loading ? (
        <div className="forensic-panel p-12 text-center space-y-3">
          <RefreshCw className="w-8 h-8 text-forensic-blueLight animate-spin mx-auto" />
          <div className="text-xs font-mono text-slate-300">
            Executing multi-stage forensic analysis pipeline...
          </div>
        </div>
      ) : analysis && (
        viewMode === 'assessment' ? (
          /* Render Flagship Result Page */
          <ResultPage analysis={analysis} onNavigate={onNavigate} />
        ) : (
          /* Render 9-Stage Investigation Timeline */
          <div className="space-y-6">
            {/* Quick Header Summary */}
            <div className="forensic-panel p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-lab-900/90 border-forensic-blue/30">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono text-slate-400">Current Target:</span>
                  <span className="text-sm font-mono font-bold text-slate-100">{analysis.evidence.filename}</span>
                </div>
                <div className="text-[11px] font-mono text-slate-400">
                  Evidence ID: <code className="text-slate-300">{analysis.analysis_id}</code>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <StatusBadge label={analysis.final_verdict.label} size="md" />
                <button
                  onClick={() => setViewMode('assessment')}
                  className="px-4 py-1.5 rounded bg-forensic-blue hover:bg-sky-600 text-white text-xs font-mono font-bold flex items-center gap-1.5 shadow-sm"
                >
                  <Eye className="w-3.5 h-3.5" /> Inspect Full Assessment (11 Sections)
                </button>
              </div>
            </div>

            {/* Timeline Flow */}
            <div className="forensic-panel p-6 space-y-2">
              <div className="text-[11px] font-mono text-slate-400 uppercase tracking-wider mb-4 pb-2 border-b border-lab-border flex items-center justify-between">
                <span>INVESTIGATION TIMELINE PIPELINE</span>
                <span className="text-emerald-400 font-bold">9 VERIFICATION STAGES</span>
              </div>

              <div className="space-y-3">
                {stages.map((stage, idx) => {
                  const Icon = stage.icon;
                  const isCompleted = stage.status === 'COMPLETED' || stage.status === 'ACTIVE' || stage.status === 'AVAILABLE';
                  const isUnavailable = stage.status === 'NOT_AVAILABLE';
                  const isLast = idx === stages.length - 1;

                  return (
                    <div key={stage.step} className="space-y-3">
                      {/* Stage Card */}
                      <div className="p-4 rounded-lg bg-lab-950/80 border border-lab-border/80 hover:border-slate-600 transition-colors">
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-lab-border/40">
                          <div className="flex items-center gap-3">
                            <span className="w-6 h-6 rounded-full bg-lab-900 border border-lab-border flex items-center justify-center font-mono text-[11px] font-bold text-slate-300">
                              {stage.step}
                            </span>
                            <div className="flex items-center gap-2">
                              <Icon className={`w-4 h-4 ${isCompleted ? 'text-emerald-400' : isUnavailable ? 'text-slate-500' : 'text-amber-400'}`} />
                              <h3 className="text-xs font-mono font-bold text-slate-100 uppercase tracking-wider">
                                {stage.title}
                              </h3>
                            </div>
                          </div>

                          <span
                            className={`px-2.5 py-0.5 rounded text-[10px] font-mono font-bold uppercase self-start sm:self-auto ${
                              isCompleted
                                ? 'bg-emerald-950/80 text-emerald-300 border border-emerald-500/30'
                                : isUnavailable
                                ? 'bg-slate-900 text-slate-400 border border-slate-700'
                                : 'bg-amber-950/80 text-amber-300 border border-amber-500/30'
                            }`}
                          >
                            {stage.status}
                          </span>
                        </div>

                        <div className="text-xs font-mono text-slate-400 pt-2 leading-relaxed">
                          {stage.details}
                        </div>
                      </div>

                      {/* Directional Arrow between stages */}
                      {!isLast && (
                        <div className="flex items-center justify-center text-slate-600 py-0.5">
                          <div className="flex items-center gap-2 font-mono text-xs text-forensic-blueLight/60">
                            <span className="w-8 h-[1px] bg-lab-border"></span>
                            <ArrowDown className="w-4 h-4 text-forensic-blueLight" />
                            <span className="w-8 h-[1px] bg-lab-border"></span>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )
      )}
    </div>
  );
};
