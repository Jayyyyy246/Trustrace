import React from 'react';
import type { VerdictLabel, FindingSeverity } from '../types/forensic';
import { CheckCircle2, AlertTriangle, HelpCircle, ShieldAlert, Cpu } from 'lucide-react';

interface StatusBadgeProps {
  label: VerdictLabel | string;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ label, size = 'md', className = '' }) => {
  const norm = String(label).toUpperCase();

  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5',
    md: 'text-xs px-2.5 py-1',
    lg: 'text-sm px-3.5 py-1.5 font-semibold',
  };

  switch (norm) {
    case 'REAL':
      return (
        <span className={`inline-flex items-center gap-1.5 rounded font-mono font-medium border bg-emerald-950/70 border-emerald-500/40 text-emerald-300 ${sizeClasses[size]} ${className}`}>
          <CheckCircle2 className={size === 'lg' ? 'w-4 h-4 text-emerald-400' : 'w-3.5 h-3.5 text-emerald-400'} />
          REAL / AUTHENTIC
        </span>
      );

    case 'EDITED':
      return (
        <span className={`inline-flex items-center gap-1.5 rounded font-mono font-medium border bg-rose-950/70 border-rose-500/40 text-rose-300 ${sizeClasses[size]} ${className}`}>
          <ShieldAlert className={size === 'lg' ? 'w-4 h-4 text-rose-400' : 'w-3.5 h-3.5 text-rose-400'} />
          EDITED / MANIPULATED
        </span>
      );

    case 'AI-GENERATED':
      return (
        <span className={`inline-flex items-center gap-1.5 rounded font-mono font-medium border bg-purple-950/70 border-purple-500/40 text-purple-300 ${sizeClasses[size]} ${className}`}>
          <Cpu className={size === 'lg' ? 'w-4 h-4 text-purple-400' : 'w-3.5 h-3.5 text-purple-400'} />
          AI-GENERATED SYNTHETIC
        </span>
      );

    case 'SCREENSHOT-MANIPULATED':
      return (
        <span className={`inline-flex items-center gap-1.5 rounded font-mono font-medium border bg-amber-950/70 border-amber-500/40 text-amber-300 ${sizeClasses[size]} ${className}`}>
          <AlertTriangle className={size === 'lg' ? 'w-4 h-4 text-amber-400' : 'w-3.5 h-3.5 text-amber-400'} />
          SCREENSHOT-MANIPULATED
        </span>
      );

    case 'UNKNOWN':
    default:
      return (
        <span className={`inline-flex items-center gap-1.5 rounded font-mono font-medium border bg-slate-900/90 border-slate-600/50 text-slate-300 ${sizeClasses[size]} ${className}`}>
          <HelpCircle className={size === 'lg' ? 'w-4 h-4 text-slate-400' : 'w-3.5 h-3.5 text-slate-400'} />
          UNKNOWN / INCONCLUSIVE
        </span>
      );
  }
};

export const SeverityBadge: React.FC<{ severity: FindingSeverity }> = ({ severity }) => {
  const norm = severity ? String(severity).toUpperCase() : 'LOW';
  switch (norm) {
    case 'CRITICAL':
      return <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-rose-950/90 border border-rose-600 text-rose-300 uppercase">Critical</span>;
    case 'HIGH':
      return <span className="px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-rose-950/60 border border-rose-500/40 text-rose-300 uppercase">High</span>;
    case 'MEDIUM':
      return <span className="px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-amber-950/60 border border-amber-500/40 text-amber-300 uppercase">Medium</span>;
    case 'LOW':
    default:
      return <span className="px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-slate-800 border border-slate-700 text-slate-300 uppercase">Low</span>;
  }
};

export const ForensicIndicatorIcon: React.FC<{
  status: 'normal' | 'suspicious' | 'strong' | 'unavailable';
  showLabel?: boolean;
}> = ({ status, showLabel = false }) => {
  switch (status) {
    case 'normal':
      return (
        <span className="inline-flex items-center gap-1.5 text-emerald-400 font-mono font-bold text-xs" title="Normal / Clean">
          <span className="w-4 h-4 rounded-full bg-emerald-950/80 border border-emerald-500/50 flex items-center justify-center text-[10px]">✓</span>
          {showLabel && <span className="text-[10px] font-semibold text-emerald-400">Normal</span>}
        </span>
      );
    case 'suspicious':
      return (
        <span className="inline-flex items-center gap-1.5 text-amber-400 font-mono font-bold text-xs" title="Suspicious Indicator">
          <span className="w-4 h-4 rounded-full bg-amber-950/80 border border-amber-500/50 flex items-center justify-center text-[10px]">⚠</span>
          {showLabel && <span className="text-[10px] font-semibold text-amber-400">Suspicious</span>}
        </span>
      );
    case 'strong':
      return (
        <span className="inline-flex items-center gap-1.5 text-rose-400 font-mono font-bold text-xs" title="Strong Manipulation Indicator">
          <span className="w-4 h-4 rounded-full bg-rose-950/80 border border-rose-500/50 flex items-center justify-center text-[10px]">✕</span>
          {showLabel && <span className="text-[10px] font-semibold text-rose-400">Strong indicator</span>}
        </span>
      );
    case 'unavailable':
    default:
      return (
        <span className="inline-flex items-center gap-1.5 text-slate-500 font-mono font-bold text-xs" title="Unavailable">
          <span className="w-4 h-4 rounded-full bg-slate-900 border border-slate-700 flex items-center justify-center text-[10px]">○</span>
          {showLabel && <span className="text-[10px] font-normal text-slate-400">Unavailable</span>}
        </span>
      );
  }
};
