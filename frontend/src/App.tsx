import { useState, useEffect } from 'react';
import { Navigation, type NavTab } from './components/Navigation';
import { DashboardPage } from './pages/DashboardPage';
import { InvestigationsPage } from './pages/InvestigationsPage';
import { UploadPage } from './pages/UploadPage';
import { AnalysisPage } from './pages/AnalysisPage';
import { ReportsPage } from './pages/ReportsPage';
import { ModelIntelligencePage } from './pages/ModelIntelligencePage';
import { DocumentationPage } from './pages/DocumentationPage';
import { SettingsPage } from './pages/SettingsPage';
import { api } from './services/api';
import { ShieldCheck } from 'lucide-react';

export function App() {
  const [currentTab, setCurrentTab] = useState<NavTab>('dashboard');
  const [activeEvidenceId, setActiveEvidenceId] = useState<string | null>(null);
  const [analysisMode, setAnalysisMode] = useState<'timeline' | 'assessment'>('timeline');
  const [systemOnline, setSystemOnline] = useState<boolean>(true);

  // Periodically verify backend connectivity
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const h = await api.getHealth();
        setSystemOnline(h.status === 'healthy');
      } catch {
        setSystemOnline(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleNavigate = (
    tab: NavTab | 'result',
    evidenceId?: string,
    mode?: 'timeline' | 'assessment'
  ) => {
    if (evidenceId) {
      setActiveEvidenceId(evidenceId);
    }

    if (tab === 'result') {
      setCurrentTab('analysis');
      setAnalysisMode('assessment');
    } else {
      setCurrentTab(tab as NavTab);
      if (mode) {
        setAnalysisMode(mode);
      }
    }

    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen bg-lab-950 text-slate-100 flex flex-col font-sans selection:bg-forensic-blue/30 selection:text-white">
      {/* Accessibility Skip Link */}
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:px-4 focus:py-2 focus:bg-forensic-blue focus:text-white focus:rounded focus:outline-none font-mono text-xs"
      >
        Skip to main content
      </a>

      {/* Main Forensic Header Navigation */}
      <Navigation
        currentTab={currentTab}
        onSelectTab={(tab) => handleNavigate(tab)}
        systemOnline={systemOnline}
      />

      {/* Main Investigation Workspace */}
      <main id="main-content" className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {currentTab === 'dashboard' && (
          <DashboardPage
            onNavigate={(tab, id) => {
              if (tab === 'analysis') {
                handleNavigate('analysis', id, 'assessment');
              } else {
                handleNavigate(tab, id);
              }
            }}
          />
        )}

        {currentTab === 'investigations' && (
          <InvestigationsPage
            onNavigate={(tab, id) => {
              if (tab === 'analysis') {
                handleNavigate('analysis', id, 'assessment');
              } else {
                handleNavigate(tab, id);
              }
            }}
          />
        )}

        {currentTab === 'upload' && (
          <UploadPage
            onNavigate={(tab, id) => {
              if (tab === 'analysis') {
                handleNavigate('analysis', id, 'timeline');
              } else {
                handleNavigate(tab, id);
              }
            }}
          />
        )}

        {currentTab === 'analysis' && (
          <AnalysisPage
            evidenceId={activeEvidenceId}
            initialMode={analysisMode}
            onNavigate={(tab, id) => handleNavigate(tab, id)}
          />
        )}

        {currentTab === 'reports' && (
          <ReportsPage
            evidenceId={activeEvidenceId}
            onNavigate={(tab, id) => handleNavigate(tab, id)}
          />
        )}

        {currentTab === 'models' && <ModelIntelligencePage />}

        {currentTab === 'docs' && <DocumentationPage />}

        {currentTab === 'settings' && <SettingsPage />}
      </main>

      {/* Technical Lab Station Footer */}
      <footer className="mt-auto border-t border-lab-border bg-lab-950/80 py-4 text-xs font-mono text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1.5 text-slate-400">
              <ShieldCheck className="w-4 h-4 text-forensic-blueLight" />
              TRUSTTRACE DIGITAL EVIDENCE LABORATORY
            </span>
            <span className="hidden md:inline">•</span>
            <span className="hidden md:inline">Daubert Standard Compliant</span>
          </div>

          <div className="flex items-center gap-4 text-[11px]">
            <span>SHA-256 Custody Ledger</span>
            <span>•</span>
            <button
              onClick={() => handleNavigate('docs')}
              className="hover:text-slate-300 transition-colors"
            >
              Methodology Spec
            </button>
            <span>•</span>
            <span className="text-slate-400">Station v1.0.0</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
