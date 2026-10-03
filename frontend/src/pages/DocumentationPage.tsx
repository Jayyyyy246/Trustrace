import React, { useState } from 'react';
import {
  BookOpen,
  Binary,
  Microscope,
  Layers,
  Search,
  GitMerge,
  ShieldCheck,
  FileText,
} from 'lucide-react';

export const DocumentationPage: React.FC = () => {
  const [activeSection, setActiveSection] = useState<string>('architecture');

  const navLinks = [
    { id: 'architecture', title: 'Pipeline Architecture' },
    { id: 'ela', title: 'Error Level Analysis (ELA)' },
    { id: 'dqt', title: 'JPEG Quantization Tables (DQT)' },
    { id: 'noise', title: 'Sensor Noise & Residuals' },
    { id: 'copymove', title: 'Copy-Move Forgery Detection' },
    { id: 'metadata', title: 'Container & EXIF Metadata' },
    { id: 'screenshot', title: 'Screenshot & UI Geometry' },
    { id: 'fusion', title: 'Evidence Decision Engine' },
    { id: 'daubert', title: 'Legal & Forensic Standards' },
  ];

  return (
    <div className="space-y-6">
      {/* Page Title */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-lab-border">
        <div>
          <h1 className="text-xl font-bold font-mono tracking-wide text-slate-100 flex items-center gap-2.5">
            <BookOpen className="w-5 h-5 text-forensic-blueLight" />
            FORENSIC SYSTEM SPECIFICATION & METHODOLOGY MANUAL
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Algorithmic formulations, mathematical models, decision rules, and legal admissibility standards.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Navigation Sidebar */}
        <div className="lg:col-span-1 space-y-1">
          <div className="text-[11px] font-mono text-slate-400 uppercase tracking-wider mb-2 px-2">
            TABLE OF CONTENTS
          </div>
          <div className="space-y-1 bg-lab-950 p-2 rounded-lg border border-lab-border">
            {navLinks.map((link) => (
              <button
                key={link.id}
                onClick={() => setActiveSection(link.id)}
                className={`w-full text-left px-3 py-2 rounded text-xs font-mono transition-colors ${
                  activeSection === link.id
                    ? 'bg-forensic-blue/20 text-forensic-blueLight font-bold border border-forensic-blue/40'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-lab-900/60'
                }`}
              >
                {link.title}
              </button>
            ))}
          </div>
        </div>

        {/* Content Area */}
        <div className="lg:col-span-3 space-y-6">
          {activeSection === 'architecture' && (
            <div className="forensic-panel p-6 space-y-4">
              <div className="flex items-center gap-2 pb-2 border-b border-lab-border">
                <Layers className="w-5 h-5 text-forensic-blueLight" />
                <h2 className="text-base font-mono font-bold text-slate-100 uppercase tracking-wide">
                  1. Multi-Stage Forensic Verification Architecture
                </h2>
              </div>
              <p className="text-xs font-mono text-slate-300 leading-relaxed">
                TRUSTTRACE operates as a strictly decoupled forensic verification pipeline. Incoming assets pass through nine consecutive stages of cryptographic validation, container parsing, pixel-level computer vision analysis, and deterministic evidence fusion.
              </p>
              <div className="p-4 rounded bg-lab-950 border border-lab-border font-mono text-xs space-y-2">
                <div className="text-emerald-400 font-bold">THE NINE-STAGE PIPELINE:</div>
                <ol className="list-decimal list-inside space-y-1 text-slate-300 text-[11px]">
                  <li><strong>Evidence Intake:</strong> Magic-byte MIME verification, file size cap enforcement, and immutable quarantine isolation.</li>
                  <li><strong>Integrity Verification:</strong> Cryptographic SHA-256 and MD5 hashing recorded into the custody ledger.</li>
                  <li><strong>Metadata Extraction:</strong> EXIF tag parsing, camera hardware signatures, and timestamp cross-correlation.</li>
                  <li><strong>Forensic CV:</strong> Error Level Analysis, Discrete Quantization Table inspection, Laplacian noise residual extraction.</li>
                  <li><strong>OCR & Typography:</strong> Text extraction, bounding box collinearity, character pitch, and font baseline consistency.</li>
                  <li><strong>Screenshot Geometry:</strong> Viewport aspect ratio matching against known device specifications (iOS, Android, Windows, macOS).</li>
                  <li><strong>ML Neural Analysis:</strong> Deep visual artifact classification with calibrated epistemic uncertainty.</li>
                  <li><strong>Evidence Fusion:</strong> Multi-source arbitration matrix evaluating supporting vs. contradictory signals.</li>
                  <li><strong>Final Assessment:</strong> Calibrated assessment with itemized forensic justifications and limitation disclosures.</li>
                </ol>
              </div>
            </div>
          )}

          {activeSection === 'ela' && (
            <div className="forensic-panel p-6 space-y-4">
              <div className="flex items-center gap-2 pb-2 border-b border-lab-border">
                <Microscope className="w-5 h-5 text-forensic-blueLight" />
                <h2 className="text-base font-mono font-bold text-slate-100 uppercase tracking-wide">
                  2. Error Level Analysis (ELA)
                </h2>
              </div>
              <p className="text-xs font-mono text-slate-300 leading-relaxed">
                Error Level Analysis evaluates intentional post-processing modifications in lossy compressed images (specifically JPEG).
              </p>
              <div className="p-3.5 rounded bg-lab-950 border border-lab-border font-mono text-xs space-y-2">
                <div className="text-forensic-blueLight font-bold">MATHEMATICAL FORMULATION:</div>
                <div className="text-slate-300">
                  <code className="text-emerald-300">ΔE(x, y) = | I(x, y) - JPEG_Q95(I(x, y)) | × α</code>
                </div>
                <p className="text-[11px] text-slate-400">
                  Where <code className="text-slate-300">I</code> is the input image array, <code className="text-slate-300">JPEG_Q95</code> is a uniform 95% quality recompression, and <code className="text-slate-300">α = 10</code> is the enhancement scaling factor.
                </p>
              </div>
              <div className="space-y-1.5 text-xs font-mono text-slate-300">
                <div className="font-bold text-slate-200">Interpretation Guidelines:</div>
                <ul className="list-disc list-inside space-y-1 text-slate-400 text-[11px]">
                  <li>Uniform error distribution across textures indicates uniform generational compression (consistent with original sensor capture).</li>
                  <li>Discontinuous high-variance localized patches indicate digital splicing or pasted foreign elements saved at a different compression level.</li>
                  <li><strong>Limitation:</strong> Multiple historical resavings or heavy social media compression can homogenize error levels, leading to false negatives.</li>
                </ul>
              </div>
            </div>
          )}

          {activeSection === 'dqt' && (
            <div className="forensic-panel p-6 space-y-4">
              <div className="flex items-center gap-2 pb-2 border-b border-lab-border">
                <Binary className="w-5 h-5 text-forensic-blueLight" />
                <h2 className="text-base font-mono font-bold text-slate-100 uppercase tracking-wide">
                  3. Discrete Quantization Table (DQT) Analysis
                </h2>
              </div>
              <p className="text-xs font-mono text-slate-300 leading-relaxed">
                JPEG compression utilizes 8x8 luminance and chrominance quantization matrices. Camera manufacturers (Apple, Samsung, Canon, Nikon) and editing software (Photoshop, GIMP) utilize distinct proprietary tables.
              </p>
              <div className="p-3.5 rounded bg-lab-950 border border-lab-border font-mono text-xs space-y-2">
                <div className="text-forensic-blueLight font-bold">ANOMALY DETECTION RULES:</div>
                <ul className="list-disc list-inside space-y-1 text-slate-300 text-[11px]">
                  <li><strong>Standard IJG Matching:</strong> Detection of standard Independent JPEG Group default tables indicates generic software encoding rather than direct camera sensor firmware.</li>
                  <li><strong>Double Quantization Artifacts:</strong> Histograms of rounded DCT coefficients showing periodic zero-clustering indicate repeated lossy compression cycles.</li>
                </ul>
              </div>
            </div>
          )}

          {activeSection === 'noise' && (
            <div className="forensic-panel p-6 space-y-4">
              <div className="flex items-center gap-2 pb-2 border-b border-lab-border">
                <Search className="w-5 h-5 text-forensic-blueLight" />
                <h2 className="text-base font-mono font-bold text-slate-100 uppercase tracking-wide">
                  4. Sensor Noise Residuals & PRNU Analysis
                </h2>
              </div>
              <p className="text-xs font-mono text-slate-300 leading-relaxed">
                Physical camera sensors exhibit Photo-Response Non-Uniformity (PRNU) caused by microscopic semiconductor variations. Synthetic AI images and clean digital graphics lack natural sensor PRNU.
              </p>
              <div className="p-3.5 rounded bg-lab-950 border border-lab-border font-mono text-xs space-y-2">
                <div className="text-forensic-blueLight font-bold">LAPLACIAN HIGH-PASS FILTERING:</div>
                <p className="text-[11px] text-slate-300">
                  TRUSTTRACE isolates the high-frequency residual noise component using discrete 3x3 Laplacian spatial convolution:
                </p>
                <div className="text-slate-400 font-mono text-[11px]">
                  K_Laplacian = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
                </div>
                <p className="text-[11px] text-slate-400">
                  The variance <code className="text-slate-200">σ²_noise</code> across uniform patches is compared against empirical optical thresholds. Abnormally flat surfaces (&lt; 2.5) suggest synthetic generation or artificial vector rendering.
                </p>
              </div>
            </div>
          )}

          {activeSection === 'copymove' && (
            <div className="forensic-panel p-6 space-y-4">
              <div className="flex items-center gap-2 pb-2 border-b border-lab-border">
                <Binary className="w-5 h-5 text-forensic-blueLight" />
                <h2 className="text-base font-mono font-bold text-slate-100 uppercase tracking-wide">
                  5. Copy-Move Forgery Detection
                </h2>
              </div>
              <p className="text-xs font-mono text-slate-300 leading-relaxed">
                Copy-Move forgery occurs when an actor clones an image region (e.g., duplicating cash in a payment proof or cloning crowd faces) to obscure or duplicate evidence.
              </p>
              <div className="p-3.5 rounded bg-lab-950 border border-lab-border font-mono text-xs space-y-2">
                <div className="text-forensic-blueLight font-bold">ALGORITHMIC WORKFLOW:</div>
                <ol className="list-decimal list-inside space-y-1 text-slate-300 text-[11px]">
                  <li>Extract invariant feature keypoints and descriptors using ORB (Oriented FAST and Rotated BRIEF).</li>
                  <li>Compute mutual nearest neighbor matches across spatial coordinates.</li>
                  <li>Filter out proximate keypoints (spatial distance &lt; 20 pixels) to avoid naturally repeating texture false positives.</li>
                  <li>Cluster parallel displacement vectors using RANSAC affine consistency checks.</li>
                </ol>
              </div>
            </div>
          )}

          {activeSection === 'metadata' && (
            <div className="forensic-panel p-6 space-y-4">
              <div className="flex items-center gap-2 pb-2 border-b border-lab-border">
                <FileText className="w-5 h-5 text-forensic-blueLight" />
                <h2 className="text-base font-mono font-bold text-slate-100 uppercase tracking-wide">
                  6. Container & EXIF Metadata Verification
                </h2>
              </div>
              <p className="text-xs font-mono text-slate-300 leading-relaxed">
                Metadata provides investigative leads but must NEVER be treated as conclusive proof of authenticity due to trivial forging capabilities.
              </p>
              <div className="space-y-2 text-xs font-mono text-slate-300">
                <div className="font-bold text-slate-200">Critical Checkpoint Indicators:</div>
                <ul className="list-disc list-inside space-y-1 text-slate-400 text-[11px]">
                  <li><strong>Software Tags:</strong> Presence of tags such as "Adobe Photoshop", "GIMP", or "Canva" immediately refutes claim of unaltered direct sensor capture.</li>
                  <li><strong>Timestamp Inversion:</strong> When <code className="text-slate-300">DateTimeDigitized</code> occurs after <code className="text-slate-300">DateTimeModified</code>, temporal tampering is flagged.</li>
                  <li><strong>Stripped Metadata:</strong> Most social messaging apps (WhatsApp, Telegram, Signal) deliberately strip EXIF metadata for privacy; stripped metadata alone is NOT proof of forgery.</li>
                </ul>
              </div>
            </div>
          )}

          {activeSection === 'screenshot' && (
            <div className="forensic-panel p-6 space-y-4">
              <div className="flex items-center gap-2 pb-2 border-b border-lab-border">
                <Layers className="w-5 h-5 text-forensic-blueLight" />
                <h2 className="text-base font-mono font-bold text-slate-100 uppercase tracking-wide">
                  7. Screenshot Geometry & UI Validation
                </h2>
              </div>
              <p className="text-xs font-mono text-slate-300 leading-relaxed">
                Financial fraud and fake conversation evidence (e.g. WhatsApp chats, bank transfers) frequently rely on fake screenshot generator templates.
              </p>
              <div className="p-3.5 rounded bg-lab-950 border border-lab-border font-mono text-xs space-y-2">
                <div className="text-forensic-blueLight font-bold">GEOMETRIC CRITERIA:</div>
                <ul className="list-disc list-inside space-y-1 text-slate-300 text-[11px]">
                  <li><strong>Canonical Viewport Ratios:</strong> Dimensions are compared against iOS (e.g. 1170x2532, 1284x2778, 1290x2796), Android (1080x2400, 1440x3120), and Desktop resolutions.</li>
                  <li><strong>Aspect Ratio Concordance:</strong> Screenshots exhibiting non-integer display aspect ratios suggest cropping or web-based mockup generators.</li>
                </ul>
              </div>
            </div>
          )}

          {activeSection === 'fusion' && (
            <div className="forensic-panel p-6 space-y-4">
              <div className="flex items-center gap-2 pb-2 border-b border-lab-border">
                <GitMerge className="w-5 h-5 text-forensic-blueLight" />
                <h2 className="text-base font-mono font-bold text-slate-100 uppercase tracking-wide">
                  8. Evidence Decision Engine & Arbitration
                </h2>
              </div>
              <p className="text-xs font-mono text-slate-300 leading-relaxed">
                The TRUSTTRACE Decision Engine is a transparent, deterministic expert system, NOT an unexplainable generative LLM.
              </p>
              <div className="p-3.5 rounded bg-lab-950 border border-lab-border font-mono text-xs space-y-2">
                <div className="text-emerald-400 font-bold">CROSS-MODAL CONFLICT RESOLUTION:</div>
                <p className="text-[11px] text-slate-300">
                  If a neural classifier predicts <code className="text-purple-300">AI-GENERATED</code> with high confidence, but hardware EXIF tags confirm Canon EOS sensor serialization, matching Bayer CFA interpolation, and consistent DQT tables, TRUSTTRACE DOES NOT blindly follow the neural network.
                </p>
                <p className="text-[11px] text-amber-300">
                  Instead, the conflict is flagged, confidence is suppressed, and the system arbitrates to <strong className="text-white">UNKNOWN (CONFLICT DETECTED)</strong> with explicit justifications provided to the human investigator.
                </p>
              </div>
            </div>
          )}

          {activeSection === 'daubert' && (
            <div className="forensic-panel p-6 space-y-4">
              <div className="flex items-center gap-2 pb-2 border-b border-lab-border">
                <ShieldCheck className="w-5 h-5 text-forensic-blueLight" />
                <h2 className="text-base font-mono font-bold text-slate-100 uppercase tracking-wide">
                  9. Legal Admissibility & Evidentiary Standards
                </h2>
              </div>
              <p className="text-xs font-mono text-slate-300 leading-relaxed">
                TRUSTTRACE is engineered to adhere to the evidentiary requirements of modern judicial systems:
              </p>
              <div className="space-y-3 text-xs font-mono text-slate-300">
                <div className="p-3 rounded bg-lab-950 border border-lab-border/70 space-y-1">
                  <div className="font-bold text-slate-200">Federal Rules of Evidence 901 & 902 (FRE):</div>
                  <p className="text-slate-400 text-[11px]">
                    Preservation of immutable SHA-256 cryptographic hashes at ingestion provides self-authenticating digital chain-of-custody verification.
                  </p>
                </div>

                <div className="p-3 rounded bg-lab-950 border border-lab-border/70 space-y-1">
                  <div className="font-bold text-slate-200">The Daubert Standard (Repeatability & Peer Review):</div>
                  <p className="text-slate-400 text-[11px]">
                    All deterministic algorithms (ELA, DQT, ORB, Laplacian residuals) rely on established, peer-reviewed computer science literature with known error rate bounds. No black-box generative models are utilized to fabricate findings.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
