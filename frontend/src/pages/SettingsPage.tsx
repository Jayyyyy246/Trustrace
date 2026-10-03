import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import type { SystemHealth } from '../types/forensic';
import {
  Settings as SettingsIcon,
  RefreshCw,
  Server,
  Sliders,
  Lock,
} from 'lucide-react';

export const SettingsPage: React.FC = () => {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [pingLatency, setPingLatency] = useState<number | null>(null);
  const [pinging, setPinging] = useState(false);
  const [autoAnalyze, setAutoAnalyze] = useState<boolean>(() => {
    return localStorage.getItem('trusttrace_auto_analyze') !== 'false';
  });
  const [highContrast, setHighContrast] = useState<boolean>(() => {
    return localStorage.getItem('trusttrace_high_contrast') === 'true';
  });

  const fetchHealth = async () => {
    setLoading(true);
    try {
      const data = await api.getHealth();
      setHealth(data);
    } catch {
      // Handled
    } finally {
      setLoading(false);
    }
  };

  const measurePing = async () => {
    setPinging(true);
    const start = performance.now();
    try {
      await api.getHealth();
      const end = performance.now();
      setPingLatency(Math.round(end - start));
    } catch {
      setPingLatency(-1);
    } finally {
      setPinging(false);
    }
  };

  useEffect(() => {
    fetchHealth();
    measurePing();
  }, []);

  const handleToggleAutoAnalyze = (val: boolean) => {
    setAutoAnalyze(val);
    localStorage.setItem('trusttrace_auto_analyze', val.toString());
  };

  const handleToggleHighContrast = (val: boolean) => {
    setHighContrast(val);
    localStorage.setItem('trusttrace_high_contrast', val.toString());
    if (val) {
      document.documentElement.classList.add('high-contrast');
    } else {
      document.documentElement.classList.remove('high-contrast');
    }
  };

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-lab-border">
        <div>
          <h1 className="text-xl font-bold font-mono tracking-wide text-slate-100 flex items-center gap-2.5">
            <SettingsIcon className="w-5 h-5 text-forensic-blueLight" />
            FORENSIC WORKSTATION SYSTEM CONFIGURATION
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            System diagnostics, backend connectivity parameters, quarantine enclave settings, and workstation preferences.
          </p>
        </div>

        <button
          onClick={() => {
            fetchHealth();
            measurePing();
          }}
          disabled={loading || pinging}
          className="flex items-center gap-2 px-3 py-1.5 rounded bg-lab-900 border border-lab-border text-xs font-mono text-slate-300 hover:text-white"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading || pinging ? 'animate-spin' : ''}`} />
          <span>Run Diagnostic Ping</span>
        </button>
      </div>

      {/* Backend Operational Health */}
      <div className="forensic-panel p-5 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-lab-border">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-emerald-950/60 border border-emerald-500/40 text-emerald-400 flex items-center justify-center">
              <Server className="w-5 h-5" />
            </div>
            <div>
              <div className="text-xs font-mono text-slate-400">FASTAPI BACKEND SERVICE</div>
              <div className="text-base font-mono font-bold text-slate-100 flex items-center gap-2">
                <span>{health?.system || 'TRUSTTRACE Forensic Engine'}</span>
                <span className="text-xs font-mono text-slate-400 font-normal">v{health?.version || '1.0.0'}</span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="text-right">
              <div className="text-[10px] font-mono text-slate-500 uppercase">Live Latency</div>
              <div className="text-xs font-mono font-bold text-emerald-400">
                {pingLatency !== null ? (pingLatency >= 0 ? `${pingLatency} ms` : 'Disconnected') : 'Measuring...'}
              </div>
            </div>
            <span className="px-3 py-1 rounded text-xs font-mono font-bold uppercase bg-emerald-950 text-emerald-300 border border-emerald-500/40">
              ● {health?.status?.toUpperCase() || 'ONLINE'}
            </span>
          </div>
        </div>

        {/* System parameters grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 text-xs font-mono">
          <div className="p-3 rounded bg-lab-950/60 border border-lab-border/50">
            <span className="text-slate-500 block text-[10px]">ENVIRONMENT</span>
            <span className="text-slate-200 font-semibold">{health?.environment || 'development'}</span>
          </div>
          <div className="p-3 rounded bg-lab-950/60 border border-lab-border/50">
            <span className="text-slate-500 block text-[10px]">QUARANTINE ENCLAVE</span>
            <span className="text-emerald-400 font-semibold">{health?.storage?.status?.toUpperCase() || 'ONLINE'}</span>
          </div>
          <div className="p-3 rounded bg-lab-950/60 border border-lab-border/50">
            <span className="text-slate-500 block text-[10px]">STORAGE ENGINE</span>
            <span className="text-slate-200 font-semibold">{health?.storage?.provider || 'LocalEvidenceStorage'}</span>
          </div>
          <div className="p-3 rounded bg-lab-950/60 border border-lab-border/50">
            <span className="text-slate-500 block text-[10px]">UTC CLOCK SYNC</span>
            <span className="text-slate-200 font-semibold">
              {health?.timestamp_utc ? new Date(health.timestamp_utc).toLocaleTimeString() : 'Synchronized'}
            </span>
          </div>
        </div>

        {/* Analyzer Status Breakdown */}
        {health?.analyzers && (
          <div className="space-y-2 pt-2 border-t border-lab-border/60">
            <div className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">
              Installed Analyzers & Capability Matrix
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
              {Object.entries(health.analyzers).map(([key, val]) => {
                const isActive = val === 'ACTIVE';
                return (
                  <div
                    key={key}
                    className="p-2 rounded bg-lab-950/80 border border-lab-border/50 flex items-center justify-between text-xs font-mono"
                  >
                    <span className="text-slate-300 truncate text-[11px]">{key}</span>
                    <span
                      className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                        isActive
                          ? 'bg-emerald-950 text-emerald-400 border border-emerald-500/30'
                          : 'bg-amber-950 text-amber-400 border border-amber-500/30'
                      }`}
                    >
                      {val}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Evidence Enclave & Intake Policy */}
      <div className="forensic-panel p-5 space-y-4">
        <div className="flex items-center gap-2 pb-2 border-b border-lab-border">
          <Lock className="w-4 h-4 text-forensic-blueLight" />
          <h2 className="text-sm font-mono font-bold text-slate-100 uppercase tracking-wider">
            Quarantine Enclave Intake Policies & Constraints
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
          <div className="p-3.5 rounded bg-lab-950/70 border border-lab-border space-y-2">
            <div className="font-bold text-slate-200">Cryptographic Integrity Rule</div>
            <p className="text-slate-400 leading-relaxed text-[11px]">
              Every ingested piece of evidence is immediately isolated in a read-only quarantine folder. 
              The initial SHA-256 hash is permanently stamped in the evidence intake record to guarantee non-repudiation.
            </p>
          </div>

          <div className="p-3.5 rounded bg-lab-950/70 border border-lab-border space-y-2">
            <div className="font-bold text-slate-200">File Payload Boundaries</div>
            <p className="text-slate-400 leading-relaxed text-[11px]">
              Maximum file size capped at <strong>25 MB</strong> per asset. 
              MIME types verified via magic byte inspection: JPEG (<code className="text-slate-300">FF D8 FF</code>), PNG (<code className="text-slate-300">89 50 4E 47</code>), WebP (<code className="text-slate-300">52 49 46 46</code>), TIFF, BMP.
            </p>
          </div>
        </div>
      </div>

      {/* Workstation UI Preferences */}
      <div className="forensic-panel p-5 space-y-4">
        <div className="flex items-center gap-2 pb-2 border-b border-lab-border">
          <Sliders className="w-4 h-4 text-forensic-blueLight" />
          <h2 className="text-sm font-mono font-bold text-slate-100 uppercase tracking-wider">
            Workstation UI & Workflow Preferences
          </h2>
        </div>

        <div className="space-y-3">
          {/* Toggle Auto-Analyze */}
          <div className="flex items-center justify-between p-3 rounded bg-lab-950/70 border border-lab-border">
            <div className="space-y-0.5">
              <div className="text-xs font-mono font-bold text-slate-200">
                Automatic Forensic Execution on Evidence Upload
              </div>
              <div className="text-[11px] font-mono text-slate-400">
                When enabled, uploading an evidence asset immediately triggers the 9-stage verification pipeline.
              </div>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={autoAnalyze}
                onChange={(e) => handleToggleAutoAnalyze(e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-lab-800 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-forensic-blue"></div>
            </label>
          </div>

          {/* High Contrast Mode */}
          <div className="flex items-center justify-between p-3 rounded bg-lab-950/70 border border-lab-border">
            <div className="space-y-0.5">
              <div className="text-xs font-mono font-bold text-slate-200">
                Enhanced Contrast Mode (WCAG AAA Compliance)
              </div>
              <div className="text-[11px] font-mono text-slate-400">
                Increases border contrast and text luminance for laboratory monitor environments.
              </div>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={highContrast}
                onChange={(e) => handleToggleHighContrast(e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-lab-800 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-forensic-blue"></div>
            </label>
          </div>
        </div>
      </div>
    </div>
  );
};
