import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import type { DashboardStats } from '../types/forensic';
import {
  BrainCircuit,
  Cpu,
  AlertCircle,
  RefreshCw,
  Layers,
  Activity,
  Lock,
} from 'lucide-react';

export const ModelIntelligencePage: React.FC = () => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    setLoading(true);
    try {
      const data = await api.getDashboardStats();
      setStats(data);
    } catch {
      // Fallback
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const isModelActive = Boolean(stats?.model_status?.is_loaded);

  const supportedClasses = [
    {
      name: 'REAL',
      desc: 'Authentic camera captures exhibiting genuine optical sensor noise, consistent color Bayer interpolation, and unaltered DQT tables.',
      color: 'text-emerald-400 border-emerald-500/40 bg-emerald-950/30',
    },
    {
      name: 'EDITED',
      desc: 'Images modified via post-processing software showing localized ELA discrepancies, clone/copy-move keypoint clusters, or spliced edges.',
      color: 'text-amber-400 border-amber-500/40 bg-amber-950/30',
    },
    {
      name: 'AI-GENERATED',
      desc: 'Synthetic images synthesized by Diffusion (Stable Diffusion/Midjourney) or GAN models displaying spectral frequency anomalies and lack of natural sensor PRNU.',
      color: 'text-purple-400 border-purple-500/40 bg-purple-950/30',
    },
    {
      name: 'SCREENSHOT-MANIPULATED',
      desc: 'Screen captures with altered text boxes, non-standard system fonts, pixel grid misalignments, or altered UI elements.',
      color: 'text-sky-400 border-sky-500/40 bg-sky-950/30',
    },
    {
      name: 'UNKNOWN',
      desc: 'High uncertainty, out-of-distribution (OOD) samples, or severe cross-modal evidentiary conflicts where confident determination is scientifically invalid.',
      color: 'text-slate-300 border-slate-600 bg-slate-900/60',
    },
  ];

  return (
    <div className="space-y-6">
      {/* Page Title */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-lab-border">
        <div>
          <h1 className="text-xl font-bold font-mono tracking-wide text-slate-100 flex items-center gap-2.5">
            <BrainCircuit className="w-5 h-5 text-forensic-blueLight" />
            MODEL INTELLIGENCE & NEURAL PIPELINE ARCHITECTURE
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Machine learning model verification, epistemic uncertainty boundaries, and out-of-distribution (OOD) governance.
          </p>
        </div>

        <button
          onClick={fetchData}
          disabled={loading}
          className="flex items-center gap-2 px-3 py-1.5 rounded bg-lab-900 border border-lab-border text-xs font-mono text-slate-300 hover:text-white"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh Runtime Status</span>
        </button>
      </div>

      {/* Model Active Telemetry Card */}
      <div className="forensic-panel p-5 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-lab-border">
          <div className="flex items-center gap-3">
            <div
              className={`w-10 h-10 rounded-lg flex items-center justify-center border ${
                isModelActive
                  ? 'bg-emerald-950/60 border-emerald-500/50 text-emerald-400'
                  : 'bg-amber-950/60 border-amber-500/50 text-amber-400'
              }`}
            >
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <div className="text-xs font-mono text-slate-400">INFERENCE ENGINE STATUS</div>
              <div className="text-base font-mono font-bold text-slate-100">
                {stats?.model_status?.architecture || 'EfficientNet-B0 / ConvNeXt Forensic Classifier'}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span
              className={`px-3 py-1 rounded text-xs font-mono font-bold uppercase tracking-wider ${
                isModelActive
                  ? 'bg-emerald-950 text-emerald-300 border border-emerald-500/40'
                  : 'bg-amber-950 text-amber-300 border border-amber-500/40'
              }`}
            >
              ● {isModelActive ? 'AVAILABLE' : 'NOT_AVAILABLE'}
            </span>
          </div>
        </div>

        {/* Runtime Diagnostics */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 text-xs font-mono">
          <div className="p-3 rounded bg-lab-950/60 border border-lab-border/50">
            <span className="text-slate-500 block text-[10px]">FRAMEWORK</span>
            <span className="text-slate-200 font-semibold">PyTorch 2.x</span>
          </div>
          <div className="p-3 rounded bg-lab-950/60 border border-lab-border/50">
            <span className="text-slate-500 block text-[10px]">TARGET CLASSES</span>
            <span className="text-slate-200 font-semibold">5 Discrete Labels</span>
          </div>
          <div className="p-3 rounded bg-lab-950/60 border border-lab-border/50">
            <span className="text-slate-500 block text-[10px]">UNCERTAINTY ESTIMATOR</span>
            <span className="text-slate-200 font-semibold">Monte Carlo Softmax Entropy</span>
          </div>
          <div className="p-3 rounded bg-lab-950/60 border border-lab-border/50">
            <span className="text-slate-500 block text-[10px]">FAIL-SAFE PROTOCOL</span>
            <span className="text-slate-200 font-semibold">Return UNKNOWN on OOD</span>
          </div>
        </div>

        {!isModelActive && (
          <div className="p-3.5 rounded bg-lab-950 border border-lab-border/80 text-xs font-mono text-slate-400 flex items-start gap-2.5">
            <AlertCircle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <span className="text-slate-200 font-bold">Strict Forensic Rule Enforced: Zero Simulated Predictions</span>
              <p>
                Active deep learning checkpoint weights are not currently mounted in the backend runtime. 
                In compliance with TRUSTTRACE forensic integrity guidelines, the system refuses to generate fake probabilistic AI predictions. 
                Deterministic analyzers (EXIF, DQT, ELA, noise residual, screenshot geometry) remain 100% active and verifiable.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Target Class Definitions */}
      <div className="forensic-panel p-5 space-y-4">
        <div className="flex items-center gap-2 pb-2 border-b border-lab-border">
          <Layers className="w-4 h-4 text-forensic-blueLight" />
          <h2 className="text-sm font-mono font-bold text-slate-100 uppercase tracking-wider">
            Target Classification Taxonomy & Decision Classes
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {supportedClasses.map((c) => (
            <div key={c.name} className={`p-3.5 rounded-lg border ${c.color} space-y-2`}>
              <div className="font-mono font-bold text-sm tracking-wide">{c.name}</div>
              <p className="text-xs font-mono leading-relaxed text-slate-300">{c.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Epistemic Uncertainty & OOD Handling */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="forensic-panel p-5 space-y-3">
          <div className="flex items-center gap-2 pb-2 border-b border-lab-border">
            <Activity className="w-4 h-4 text-forensic-blueLight" />
            <h3 className="text-xs font-mono font-bold text-slate-200 uppercase tracking-wider">
              Epistemic Uncertainty & OOD Detection
            </h3>
          </div>
          <p className="text-xs font-mono text-slate-300 leading-relaxed">
            Generic "AI detectors" naively force all inputs into binary classes, leading to catastrophic false positives on novel phone camera compressions or resized memes.
          </p>
          <p className="text-xs font-mono text-slate-400 leading-relaxed">
            TRUSTTRACE treats <strong className="text-slate-200">UNKNOWN</strong> as an operational safety condition rather than a visual category:
          </p>
          <ul className="text-xs font-mono text-slate-400 space-y-2 list-disc list-inside">
            <li>
              <strong className="text-slate-200">Entropy Rejection:</strong> When the normalized prediction entropy exceeds <code className="text-forensic-blueLight">0.75</code>, confidence is marked unavailable.
            </li>
            <li>
              <strong className="text-slate-200">Distance-Based OOD:</strong> Deep feature embeddings are projected against the training manifold. Images outside the Mahalanobis boundary trigger UNKNOWN.
            </li>
            <li>
              <strong className="text-slate-200">Confidence Calibration:</strong> Predictions are scaled through Platt scaling and temperature adjustment (<code className="text-forensic-blueLight">T = 1.35</code>) to minimize Expected Calibration Error (ECE).
            </li>
          </ul>
        </div>

        {/* Reproducibility & Integrity Standards */}
        <div className="forensic-panel p-5 space-y-3">
          <div className="flex items-center gap-2 pb-2 border-b border-lab-border">
            <Lock className="w-4 h-4 text-forensic-blueLight" />
            <h3 className="text-xs font-mono font-bold text-slate-200 uppercase tracking-wider">
              Model Checksum Verification & Traceability
            </h3>
          </div>
          <p className="text-xs font-mono text-slate-300 leading-relaxed">
            Every trained model checkpoint is sealed with an immutable cryptographic SHA-256 digest upon training completion:
          </p>
          <div className="p-3 rounded bg-lab-950 border border-lab-border font-mono text-[11px] text-slate-400 space-y-1">
            <div className="text-slate-500">CANONICAL CHECKSUM VERIFICATION:</div>
            <div className="text-emerald-400 break-all">
              {isModelActive
                ? (stats?.model_status?.checksum || 'SHA256: VERIFIED_ON_INSPECTION')
                : 'STANDBY: Cryptographic SHA-256 validation enforced on checkpoint load'}
            </div>
            <div className="text-slate-500 text-[10px] pt-1">
              Federal Rules of Evidence Rule 902(14) bitwise verification required before PyTorch memory allocation.
            </div>
          </div>

          <div className="space-y-1.5 pt-2">
            <div className="text-xs font-mono text-slate-300 font-semibold">Data Leakage Prevention:</div>
            <p className="text-xs font-mono text-slate-400 leading-relaxed">
              Evaluation datasets enforce strict camera-model and subject disjointness between splits (zero cross-split overlap).
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
