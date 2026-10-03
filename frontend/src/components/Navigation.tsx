import React, { useState } from 'react';
import {
  LayoutDashboard,
  FolderSearch,
  UploadCloud,
  Microscope,
  FileText,
  BrainCircuit,
  BookOpen,
  Settings,
  ShieldCheck,
  Menu,
  X,
} from 'lucide-react';

export type NavTab = 
  | 'dashboard'
  | 'investigations'
  | 'upload'
  | 'analysis'
  | 'reports'
  | 'models'
  | 'docs'
  | 'settings';

interface NavigationProps {
  currentTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
  systemOnline?: boolean;
}

export const Navigation: React.FC<NavigationProps> = ({
  currentTab,
  onSelectTab,
  systemOnline = true,
}) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems: Array<{ id: NavTab; label: string; icon: React.ElementType }> = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'investigations', label: 'Investigations', icon: FolderSearch },
    { id: 'upload', label: 'Upload Evidence', icon: UploadCloud },
    { id: 'analysis', label: 'Analysis', icon: Microscope },
    { id: 'reports', label: 'Reports', icon: FileText },
    { id: 'models', label: 'Model Intelligence', icon: BrainCircuit },
    { id: 'docs', label: 'Documentation', icon: BookOpen },
    { id: 'settings', label: 'Settings', icon: Settings },
  ];

  const handleSelect = (tab: NavTab) => {
    onSelectTab(tab);
    setMobileMenuOpen(false);
  };

  return (
    <header className="sticky top-0 z-50 bg-lab-950/95 backdrop-blur-md border-b border-lab-border">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Platform Title */}
          <div className="flex items-center gap-3">
            <button
              onClick={() => handleSelect('dashboard')}
              className="flex items-center gap-2.5 text-left focus:outline-none group"
            >
              <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-lab-800 to-lab-900 border border-forensic-blue/40 flex items-center justify-center shadow-forensic-card group-hover:border-forensic-blue transition-colors">
                <ShieldCheck className="w-5 h-5 text-forensic-blueLight" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-mono text-base font-bold tracking-wider text-slate-100">
                    TRUST<span className="text-forensic-blueLight">TRACE</span>
                  </span>
                  <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-lab-800 border border-lab-border text-slate-400">
                    v1.0.0
                  </span>
                </div>
                <div className="text-[10px] text-slate-400 tracking-wider uppercase font-mono">
                  Forensic Evidence Workstation
                </div>
              </div>
            </button>
          </div>

          {/* Desktop Navigation Links */}
          <nav className="hidden md:flex items-center gap-1" aria-label="Main Navigation">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = currentTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => handleSelect(item.id)}
                  aria-current={isActive ? 'page' : undefined}
                  className={`flex items-center gap-2 px-3 py-2 rounded-md text-xs font-mono transition-all ${
                    isActive
                      ? 'bg-lab-800/90 text-forensic-blueLight border border-forensic-blue/30 font-semibold shadow-sm'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-lab-900/60'
                  }`}
                >
                  <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-forensic-blueLight' : 'text-slate-400'}`} />
                  {item.label}
                </button>
              );
            })}
          </nav>

          {/* Status & Mobile Menu Toggle */}
          <div className="flex items-center gap-3">
            <div className="hidden lg:flex items-center gap-2 px-2.5 py-1 rounded bg-lab-900 border border-lab-border text-xs font-mono">
              <span className="relative flex h-2 w-2">
                <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${systemOnline ? 'bg-emerald-400' : 'bg-rose-400'}`}></span>
                <span className={`relative inline-flex rounded-full h-2 w-2 ${systemOnline ? 'bg-emerald-500' : 'bg-rose-500'}`}></span>
              </span>
              <span className="text-slate-300 text-[11px]">
                {systemOnline ? 'API ONLINE' : 'OFFLINE'}
              </span>
            </div>

            {/* Mobile Hamburger Button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-2 rounded-md bg-lab-900 border border-lab-border text-slate-400 hover:text-slate-200 focus:outline-none"
              aria-label="Toggle navigation menu"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu Dropdown */}
      {mobileMenuOpen && (
        <div className="md:hidden border-t border-lab-border bg-lab-950/98 px-4 pt-2 pb-4 space-y-1 shadow-2xl">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = currentTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => handleSelect(item.id)}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-mono transition-colors ${
                  isActive
                    ? 'bg-lab-800 text-forensic-blueLight border border-forensic-blue/40 font-semibold'
                    : 'text-slate-400 hover:bg-lab-900 hover:text-slate-200'
                }`}
              >
                <Icon className="w-4 h-4" />
                {item.label}
              </button>
            );
          })}
        </div>
      )}
    </header>
  );
};
