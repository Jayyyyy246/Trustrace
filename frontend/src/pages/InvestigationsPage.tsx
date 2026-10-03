import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import type { EvidenceUploadResponse } from '../types/forensic';
import {
  FolderSearch,
  Search,
  Filter,
  RefreshCw,
  ArrowRight,
  HardDrive,
} from 'lucide-react';

interface InvestigationsPageProps {
  onNavigate: (tab: any, evidenceId?: string) => void;
}

export const InvestigationsPage: React.FC<InvestigationsPageProps> = ({ onNavigate }) => {
  const [evidenceList, setEvidenceList] = useState<EvidenceUploadResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterVerdict, setFilterVerdict] = useState<string>('ALL');

  const loadData = async () => {
    setLoading(true);
    try {
      const data = await api.listEvidence();
      setEvidenceList(data);
    } catch {
      // Fallback
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const filtered = evidenceList.filter((item) => {
    const query = searchQuery.toLowerCase();
    const matchesSearch =
      item.filename.toLowerCase().includes(query) ||
      item.evidence_id.toLowerCase().includes(query) ||
      item.sha256.toLowerCase().includes(query);
    const matchesStatus = filterVerdict === 'ALL' || item.status === filterVerdict;
    return matchesSearch && matchesStatus;
  });

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-lab-border">
        <div>
          <h1 className="text-xl font-bold font-mono tracking-wide text-slate-100 flex items-center gap-2.5">
            <FolderSearch className="w-5 h-5 text-forensic-blueLight" />
            FORENSIC CASE INVESTIGATION REGISTRY
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Browse and inspect all digital assets stored in the quarantined forensic vault.
          </p>
        </div>

        <button
          onClick={loadData}
          disabled={loading}
          className="flex items-center gap-2 px-3 py-1.5 rounded bg-lab-900 border border-lab-border text-xs font-mono text-slate-300 hover:text-white"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          Refresh Registry
        </button>
      </div>

      {/* Filter and Search Bar */}
      <div className="forensic-panel p-4 flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="w-4 h-4 text-slate-500 absolute left-3 top-2.5" />
          <input
            type="text"
            placeholder="Search by filename, Evidence ID, or SHA-256..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-3 py-1.5 rounded bg-lab-950 border border-lab-border text-xs font-mono text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-forensic-blue"
          />
        </div>

        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-slate-500" />
          <select
            value={filterVerdict}
            onChange={(e) => setFilterVerdict(e.target.value)}
            className="forensic-input text-xs py-1.5"
          >
            <option value="ALL">All Statuses</option>
            <option value="QUEUED">QUEUED</option>
            <option value="COMPLETED">COMPLETED</option>
          </select>
        </div>
      </div>

      {/* Results Table */}
      <div className="forensic-panel p-5 overflow-hidden">
        {loading ? (
          <div className="py-12 text-center text-xs font-mono text-slate-500">
            Scanning forensic evidence registry...
          </div>
        ) : filtered.length === 0 ? (
          <div className="py-12 text-center space-y-3">
            <HardDrive className="w-8 h-8 text-slate-600 mx-auto" />
            <div className="text-xs font-mono text-slate-400">No matching evidence records found.</div>
            <button
              onClick={() => onNavigate('upload')}
              className="px-3.5 py-1.5 rounded bg-forensic-blue hover:bg-sky-600 text-white text-xs font-mono font-medium"
            >
              Upload New Evidence
            </button>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="text-slate-400 border-b border-lab-border">
                  <th className="pb-3">Evidence ID</th>
                  <th className="pb-3">Filename</th>
                  <th className="pb-3">SHA-256 Hash</th>
                  <th className="pb-3">Size</th>
                  <th className="pb-3">Intake Time</th>
                  <th className="pb-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-lab-border/60 text-slate-300">
                {filtered.map((item) => (
                  <tr
                    key={item.evidence_id}
                    className="hover:bg-lab-850/60 transition-colors"
                  >
                    <td className="py-3 font-medium text-forensic-blueLight">
                      {item.evidence_id.slice(0, 10)}...
                    </td>
                    <td className="py-3 max-w-[200px] truncate font-medium text-slate-200" title={item.filename}>
                      {item.filename}
                    </td>
                    <td className="py-3 max-w-[160px] truncate text-slate-400" title={item.sha256}>
                      {item.sha256.slice(0, 16)}...
                    </td>
                    <td className="py-3 text-slate-400">
                      {(item.size / 1024).toFixed(1)} KB
                    </td>
                    <td className="py-3 text-slate-400">
                      {new Date(item.uploaded_at).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' })}
                    </td>
                    <td className="py-3 text-right">
                      <button
                        onClick={() => onNavigate('analysis', item.evidence_id)}
                        className="px-3 py-1 rounded bg-lab-800 hover:bg-forensic-blue hover:text-white border border-lab-border text-slate-300 font-mono text-[11px] transition-colors inline-flex items-center gap-1.5"
                      >
                        Inspect <ArrowRight className="w-3 h-3" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
