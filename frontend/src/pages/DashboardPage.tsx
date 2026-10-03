import React, { useEffect, useState } from 'react';
import type { DashboardStats } from '../types/forensic';
import { api, ApiError } from '../services/api';
import { StatusBadge } from '../components/StatusBadge';
import {
  ShieldAlert,
  FileCheck,
  CheckCircle2,
  Clock,
  HardDrive,
  Cpu,
  ArrowRight,
  RefreshCw,
  AlertCircle,
  FileDigit,
  Fingerprint,
} from 'lucide-react';

interface DashboardPageProps {
  onNavigate: (tab: any, evidenceId?: string) => void;
}

export const DashboardPage: React.FC<DashboardPageProps> = ({ onNavigate }) => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStats = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getDashboardStats();
      setStats(data);
    } catch (err) {
      setError((err as ApiError).message || 'Failed to connect to forensic backend API');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
  }, []);

  return (
    <div className="space-y-6">
      {/* Top Banner & Refresh */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-lab-border">
        <div>
          <h1 className="text-xl font-bold font-mono tracking-wide text-slate-100 flex items-center gap-2.5">
            <HardDrive className="w-5 h-5 text-forensic-blueLight" />
            EVIDENCE INVESTIGATION DASHBOARD
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Real-time platform telemetry and forensic intake analytics.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={fetchStats}
            disabled={loading}
            className="flex items-center gap-2 px-3 py-1.5 rounded bg-lab-900 border border-lab-border text-xs font-mono text-slate-300 hover:text-white hover:border-slate-600 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            Refresh Data
          </button>
          <button
            onClick={() => onNavigate('upload')}
            className="flex items-center gap-2 px-3.5 py-1.5 rounded bg-forensic-blue hover:bg-sky-600 text-white text-xs font-mono font-medium shadow-sm transition-colors"
          >
            + Ingest Evidence
          </button>
        </div>
      </div>

      {/* Error state if backend offline */}
      {error && (
        <div className="p-4 rounded-lg bg-rose-950/40 border border-rose-600/50 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div className="text-xs space-y-1">
            <div className="font-mono font-bold text-rose-300">Backend Connection Error</div>
            <div className="text-slate-300">{error}</div>
            <div className="text-slate-400 text-[11px] pt-1">
              Ensure FastAPI backend server is active at <code className="text-slate-300 bg-lab-950 px-1 py-0.5 rounded">http://127.0.0.1:8001</code>.
            </div>
          </div>
        </div>
      )}

      {/* Primary KPI Cards (Actual backend telemetry only) */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Active Investigations */}
        <div className="forensic-panel p-5">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-mono uppercase tracking-wider">Active Investigations</span>
            <FileDigit className="w-4 h-4 text-forensic-blueLight" />
          </div>
          <div className="mt-3 flex items-baseline gap-2">
            <span className="text-3xl font-bold font-mono text-slate-100">
              {loading ? '—' : stats ? stats.active_investigations : 0}
            </span>
            <span className="text-xs text-slate-500 font-mono">quarantined cases</span>
          </div>
          <div className="mt-3 text-[11px] text-slate-400 font-mono flex items-center gap-1.5">
            <Fingerprint className="w-3.5 h-3.5 text-slate-500" />
            Verified via SHA-256 integrity
          </div>
        </div>

        {/* Total Evidence Analyzed */}
        <div className="forensic-panel p-5">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-mono uppercase tracking-wider">Total Evidence Analyzed</span>
            <FileCheck className="w-4 h-4 text-forensic-tealLight" />
          </div>
          <div className="mt-3 flex items-baseline gap-2">
            <span className="text-3xl font-bold font-mono text-slate-100">
              {loading ? '—' : stats ? stats.total_evidence_analyzed : 0}
            </span>
            <span className="text-xs text-slate-500 font-mono">evidence files processed</span>
          </div>
          <div className="mt-3 text-[11px] text-slate-400 font-mono flex items-center gap-1.5">
            <CheckCircle2 className="w-3.5 h-3.5 text-forensic-tealLight" />
            Deterministic & ML fused pipeline
          </div>
        </div>

        {/* Analysis Success Rate */}
        <div className="forensic-panel p-5">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-mono uppercase tracking-wider">Analysis Success Rate</span>
            <Cpu className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="mt-3 flex items-baseline gap-2">
            <span className="text-3xl font-bold font-mono text-slate-100">
              {loading ? '—' : stats ? `${stats.analysis_success_rate}%` : '100%'}
            </span>
            <span className="text-xs text-slate-500 font-mono">pipeline reliability</span>
          </div>
          <div className="mt-3 text-[11px] text-slate-400 font-mono flex items-center gap-1.5">
            <Clock className="w-3.5 h-3.5 text-slate-500" />
            Non-destructive audit trail
          </div>
        </div>
      </div>

      {/* Main Grid: Recent Investigations + Status Panels */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Recent Investigations (2 spans) */}
        <div className="lg:col-span-2 space-y-6">
          <div className="forensic-panel p-5">
            <div className="flex items-center justify-between pb-3 border-b border-lab-border">
              <h2 className="text-sm font-bold font-mono tracking-wider text-slate-200 uppercase flex items-center gap-2">
                <Clock className="w-4 h-4 text-forensic-blueLight" />
                Recent Forensic Investigations
              </h2>
              <button
                onClick={() => onNavigate('investigations')}
                className="text-xs font-mono text-forensic-blueLight hover:text-sky-300 flex items-center gap-1"
              >
                View all <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>

            {loading ? (
              <div className="py-12 text-center text-xs font-mono text-slate-500">
                Loading investigation logs...
              </div>
            ) : !stats || stats.recent_investigations.length === 0 ? (
              <div className="py-12 text-center space-y-3">
                <FileDigit className="w-8 h-8 text-slate-600 mx-auto" />
                <div className="text-xs font-mono text-slate-400">No forensic investigations completed yet.</div>
                <button
                  onClick={() => onNavigate('upload')}
                  className="px-3 py-1.5 rounded bg-lab-800 hover:bg-lab-750 border border-lab-border text-xs font-mono text-slate-200"
                >
                  Upload First Evidence Item
                </button>
              </div>
            ) : (
              <div className="divide-y divide-lab-border mt-2">
                {stats.recent_investigations.map((item) => (
                  <div
                    key={item.analysis_id}
                    onClick={() => onNavigate('analysis', item.evidence_id)}
                    className="py-3 flex flex-col sm:flex-row sm:items-center justify-between gap-3 hover:bg-lab-850/60 px-2 rounded cursor-pointer transition-colors"
                  >
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-xs font-medium text-slate-200">
                          {item.filename}
                        </span>
                        <StatusBadge label={item.final_verdict} size="sm" />
                      </div>
                      <div className="text-[11px] font-mono text-slate-400 flex items-center gap-3">
                        <span>ID: {item.evidence_id.slice(0, 8)}...</span>
                        <span>SHA: {item.sha256.slice(0, 10)}...</span>
                      </div>
                    </div>

                    <div className="flex items-center gap-3 sm:text-right">
                      <div className="text-right">
                        <div className="text-xs font-mono text-slate-300">
                          {item.confidence !== null ? `Conf: ${(item.confidence * 100).toFixed(0)}%` : 'Inconclusive'}
                        </div>
                        <div className="text-[10px] text-slate-400 font-mono">
                          {new Date(item.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </div>
                      </div>
                      <ArrowRight className="w-4 h-4 text-slate-500" />
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Latest Quarantined Evidence Intake */}
          <div className="forensic-panel p-5">
            <div className="flex items-center justify-between pb-3 border-b border-lab-border">
              <h2 className="text-sm font-bold font-mono tracking-wider text-slate-200 uppercase flex items-center gap-2">
                <Fingerprint className="w-4 h-4 text-forensic-tealLight" />
                Latest Ingested Evidence in Quarantine
              </h2>
              <button
                onClick={() => onNavigate('upload')}
                className="text-xs font-mono text-forensic-tealLight hover:text-teal-300 flex items-center gap-1"
              >
                + New Intake
              </button>
            </div>

            {loading ? (
              <div className="py-8 text-center text-xs font-mono text-slate-500">
                Checking quarantine storage...
              </div>
            ) : !stats || stats.latest_evidence.length === 0 ? (
              <div className="py-8 text-center text-xs font-mono text-slate-500">
                Quarantine store is currently empty.
              </div>
            ) : (
              <div className="overflow-x-auto mt-2">
                <table className="w-full text-left text-xs font-mono">
                  <thead>
                    <tr className="text-slate-400 border-b border-lab-border">
                      <th className="pb-2">Evidence ID</th>
                      <th className="pb-2">Filename</th>
                      <th className="pb-2">Size</th>
                      <th className="pb-2">MIME</th>
                      <th className="pb-2">Intake Time</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-lab-border/50 text-slate-300">
                    {stats.latest_evidence.map((ev) => (
                      <tr
                        key={ev.evidence_id}
                        onClick={() => onNavigate('analysis', ev.evidence_id)}
                        className="hover:bg-lab-850/60 cursor-pointer"
                      >
                        <td className="py-2.5 text-forensic-blueLight font-medium">
                          {ev.evidence_id.slice(0, 8)}...
                        </td>
                        <td className="py-2.5 max-w-[150px] truncate" title={ev.filename}>
                          {ev.filename}
                        </td>
                        <td className="py-2.5 text-slate-400">
                          {(ev.size / 1024).toFixed(1)} KB
                        </td>
                        <td className="py-2.5 text-slate-400">{ev.mime_type}</td>
                        <td className="py-2.5 text-slate-400">
                          {new Date(ev.uploaded_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: System Analyzers & Model Intelligence Status */}
        <div className="space-y-6">
          {/* System Analyzer Status */}
          <div className="forensic-panel p-5">
            <h2 className="text-sm font-bold font-mono tracking-wider text-slate-200 uppercase pb-3 border-b border-lab-border flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 text-forensic-amberLight" />
              System Analyzer Status
            </h2>
            <div className="mt-4 space-y-2.5 text-xs font-mono">
              {stats?.system_analyzer_status ? (
                Object.entries(stats.system_analyzer_status).map(([name, status]) => (
                  <div key={name} className="flex items-center justify-between p-2 rounded bg-lab-950/60 border border-lab-border/50">
                    <span className="text-slate-300 capitalize">
                      {name.replace('_', ' ')}
                    </span>
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        status === 'ACTIVE'
                          ? 'bg-emerald-950/80 text-emerald-400 border border-emerald-500/30'
                          : 'bg-slate-800 text-slate-400 border border-slate-700'
                      }`}
                    >
                      {status}
                    </span>
                  </div>
                ))
              ) : (
                <div className="text-slate-500 text-xs">Querying analyzer engines...</div>
              )}
            </div>
          </div>

          {/* Machine Learning Model Status */}
          <div className="forensic-panel p-5">
            <h2 className="text-sm font-bold font-mono tracking-wider text-slate-200 uppercase pb-3 border-b border-lab-border flex items-center gap-2">
              <Cpu className="w-4 h-4 text-forensic-blueLight" />
              Model Intelligence Status
            </h2>
            <div className="mt-4 space-y-3 text-xs font-mono">
              <div className="flex items-center justify-between p-2 rounded bg-lab-950/60 border border-lab-border/50">
                <span className="text-slate-400">Runtime Status:</span>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                  stats?.model_status?.is_loaded
                    ? 'bg-emerald-950/80 text-emerald-400 border border-emerald-500/30'
                    : 'bg-amber-950/80 text-amber-400 border border-amber-500/30'
                }`}>
                  {stats?.model_status?.is_loaded ? 'ACTIVE WEIGHTS' : 'NOT_AVAILABLE'}
                </span>
              </div>

              <div className="flex items-center justify-between p-2 rounded bg-lab-950/60 border border-lab-border/50">
                <span className="text-slate-400">Architecture:</span>
                <span className="text-slate-200 font-semibold">
                  {stats?.model_status?.architecture || 'EfficientNet / MobileNet'}
                </span>
              </div>

              <div className="flex items-center justify-between p-2 rounded bg-lab-950/60 border border-lab-border/50">
                <span className="text-slate-400">Version:</span>
                <span className="text-slate-200">
                  {stats?.model_status?.model_version || 'trusttrace-v1.0.0'}
                </span>
              </div>

              <div className="p-3 rounded bg-lab-950/80 border border-lab-border text-[11px] text-slate-400 leading-relaxed">
                <strong className="text-slate-200 block mb-1">Scientific Integrity Rule:</strong>
                ML output provides an advisory probability signal and is never solely promoted to a final verdict without Evidence Fusion corroboration.
              </div>

              <button
                onClick={() => onNavigate('models')}
                className="w-full mt-2 py-2 rounded bg-lab-800 hover:bg-lab-750 border border-lab-border text-xs font-mono text-slate-300 hover:text-white transition-colors text-center block"
              >
                Inspect Model Intelligence →
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
