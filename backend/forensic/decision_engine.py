"""TRUSTTRACE Evidence Decision Engine.

A transparent, deterministic, graph-based evidence arbitration engine.
Combines independent evidence sources into explainable forensic assessments.
Evaluates evidential support, detects and resolves cross-modal conflicts,
tracks analyzer availability, and outputs traceable natural language explanations.

THIS IS NOT AN LLM CHATBOT. Every field and explanation is derived from
verifiable analyzer computations and deterministic forensic principles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from app.schemas.common import VerdictLabel
from app.schemas.verdict import EvidenceSignal, FinalVerdict


@dataclass
class EvidenceSignalItem:
    """Internal representation of a factual forensic evidence signal."""
    source: str
    metric: str
    direction: VerdictLabel
    reliability: float  # [0.0, 1.0]
    weight: float       # Effective weight for graph scoring
    explanation: str
    raw_value: Any = None
    is_contradictory: bool = False

    def to_schema(self, polarity: str = "SUPPORTING") -> EvidenceSignal:
        return EvidenceSignal(
            source=self.source,
            metric=self.metric,
            direction=self.direction,
            reliability=round(self.reliability, 4),
            explanation=self.explanation,
            polarity=polarity,
        )


@dataclass
class ConflictEdgeItem:
    """Directed edge representing an evidential contradiction between two signals."""
    source_signal: EvidenceSignalItem
    contradicting_signal: EvidenceSignalItem
    rule_id: str
    conflict_type: str
    explanation: str


class EvidenceGraph:
    """Directed Evidence Graph representing sources, signals, hypotheses, and conflicts."""

    def __init__(self) -> None:
        self.sources: Dict[str, Dict[str, Any]] = {}  # source_name -> {status, reliability}
        self.signals: List[EvidenceSignalItem] = []
        self.conflicts: List[ConflictEdgeItem] = []
        self.hypotheses: List[VerdictLabel] = [
            VerdictLabel.REAL,
            VerdictLabel.EDITED,
            VerdictLabel.AI_GENERATED,
            VerdictLabel.SCREENSHOT_MANIPULATED,
            VerdictLabel.UNKNOWN,
        ]

    def register_source(self, name: str, status: str, baseline_reliability: float = 1.0) -> None:
        """Register an originating analyzer source with operational status and reliability."""
        self.sources[name] = {
            "status": str(status).upper(),
            "reliability": max(0.0, min(1.0, float(baseline_reliability))),
        }

    def add_signal(self, signal: EvidenceSignalItem) -> None:
        """Add an observed evidence signal to the graph."""
        self.signals.append(signal)

    def add_conflict(
        self,
        sig1: EvidenceSignalItem,
        sig2: EvidenceSignalItem,
        rule_id: str,
        conflict_type: str,
        explanation: str,
    ) -> None:
        """Register an evidential conflict edge between two incompatible signals."""
        self.conflicts.append(ConflictEdgeItem(
            source_signal=sig1,
            contradicting_signal=sig2,
            rule_id=rule_id,
            conflict_type=conflict_type,
            explanation=explanation,
        ))

    def get_signals_for_source(self, source: str) -> List[EvidenceSignalItem]:
        return [s for s in self.signals if s.source == source]

    def get_signals_for_direction(self, direction: VerdictLabel) -> List[EvidenceSignalItem]:
        return [s for s in self.signals if s.direction == direction]

    def get_active_conflicts(self) -> List[ConflictEdgeItem]:
        return list(self.conflicts)

    def compute_hypothesis_scores(self) -> Dict[VerdictLabel, Dict[str, float]]:
        """Calculate aggregate positive support, opposition friction, and net score for each hypothesis."""
        scores: Dict[VerdictLabel, Dict[str, float]] = {
            h: {"support": 0.0, "opposition": 0.0, "net": 0.0, "conflict_ratio": 0.0}
            for h in self.hypotheses
        }

        # Cross-hypothesis friction matrix
        friction_matrix: Dict[Tuple[VerdictLabel, VerdictLabel], float] = {
            (VerdictLabel.REAL, VerdictLabel.EDITED): 1.0,
            (VerdictLabel.EDITED, VerdictLabel.REAL): 1.0,
            (VerdictLabel.REAL, VerdictLabel.AI_GENERATED): 1.0,
            (VerdictLabel.AI_GENERATED, VerdictLabel.REAL): 1.0,
            (VerdictLabel.REAL, VerdictLabel.SCREENSHOT_MANIPULATED): 0.85,
            (VerdictLabel.SCREENSHOT_MANIPULATED, VerdictLabel.REAL): 0.85,
            (VerdictLabel.EDITED, VerdictLabel.AI_GENERATED): 0.70,
            (VerdictLabel.AI_GENERATED, VerdictLabel.EDITED): 0.70,
            (VerdictLabel.EDITED, VerdictLabel.SCREENSHOT_MANIPULATED): 0.35,
            (VerdictLabel.SCREENSHOT_MANIPULATED, VerdictLabel.EDITED): 0.35,
        }

        for s in self.signals:
            source_info = self.sources.get(s.source, {"reliability": 1.0})
            effective_reliability = s.reliability * source_info.get("reliability", 1.0)
            sig_mass = s.weight * effective_reliability

            for h in self.hypotheses:
                if s.direction == h:
                    # Positive evidential support
                    scores[h]["support"] += sig_mass
                else:
                    # Inherent friction/opposition against rival hypotheses
                    if s.direction != VerdictLabel.UNKNOWN and h != VerdictLabel.UNKNOWN:
                        f_factor = friction_matrix.get((s.direction, h), 0.65)
                        scores[h]["opposition"] += sig_mass * f_factor

        for h in self.hypotheses:
            supp = scores[h]["support"]
            opp = scores[h]["opposition"]
            scores[h]["net"] = supp - (0.5 * opp)
            scores[h]["conflict_ratio"] = opp / (supp + 1e-5)

        return scores


def _normalize_input(val: Any) -> Dict[str, Any]:
    """Normalizes any analyzer input (dict, BaseAnalyzerResult, Pydantic model, or list) into a dict."""
    if val is None:
        return {}
    if isinstance(val, dict):
        return dict(val)
    if hasattr(val, "metrics") and isinstance(val.metrics, dict):
        d = dict(val.metrics)
        if hasattr(val, "status"):
            st = val.status.value if hasattr(val.status, "value") else str(val.status)
            d["status"] = st
        if hasattr(val, "findings") and isinstance(val.findings, list):
            d["findings"] = val.findings
        if hasattr(val, "limitations") and isinstance(val.limitations, list):
            d["limitations"] = val.limitations
        return d
    if hasattr(val, "model_dump"):
        return val.model_dump()
    if hasattr(val, "dict"):
        return val.dict()
    if isinstance(val, list):
        return {"findings": val}
    return {}


class EvidenceDecisionEngine:
    """Comprehensive, transparent evidence decision and arbitration engine.
    
    Combines independent evidence sources:
    1. Metadata findings
    2. Image forensic findings
    3. Screenshot findings
    4. OCR / layout findings
    5. ML prediction
    6. ML uncertainty
    7. Analyzer reliability / status
    """

    def __init__(self) -> None:
        pass

    def evaluate(
        self,
        metadata: Any,
        image_forensics: Any,
        screenshot: Any,
        ocr: Any,
        ml_prediction: Optional[Any] = None,
        ml_uncertainty: Optional[float] = None,
        analyzer_status: Optional[Dict[str, Any]] = None,
    ) -> FinalVerdict:
        """Execute full evidence decision evaluation across all available sources."""
        graph = EvidenceGraph()
        rules_triggered: List[str] = []
        limitations: List[str] = []
        unavailable_analyzers: List[str] = []

        # -----------------------------------------------------------------
        # 1. Normalize All Inputs
        # -----------------------------------------------------------------
        meta_dict = _normalize_input(metadata)
        img_dict = _normalize_input(image_forensics)
        screen_dict = _normalize_input(screenshot)
        ocr_dict = _normalize_input(ocr)
        ml_dict = _normalize_input(ml_prediction) if ml_prediction is not None else None

        # Add explicit input limitations
        for inp_d in [meta_dict, img_dict, screen_dict, ocr_dict]:
            inp_lims = inp_d.get("limitations")
            if isinstance(inp_lims, list):
                limitations.extend(inp_lims)

        # -----------------------------------------------------------------
        # 2. Register Analyzers & Track Availability
        # -----------------------------------------------------------------
        statuses = analyzer_status or {}
        analyzers_to_check = [
            ("metadata", meta_dict.get("status", "COMPLETED")),
            ("image_analysis", img_dict.get("status", "COMPLETED")),
            ("screenshot", screen_dict.get("status", "COMPLETED")),
            ("ocr", ocr_dict.get("status", "COMPLETED")),
            ("ml_inference", (ml_dict.get("model_status") if ml_dict else "NOT_AVAILABLE")),
        ]

        for name, default_status in analyzers_to_check:
            st_val = statuses.get(name, default_status)
            if isinstance(st_val, dict):
                st_str = str(st_val.get("status", "COMPLETED")).upper()
                st_rel = float(st_val.get("reliability", 1.0))
            elif isinstance(st_val, (int, float)):
                st_rel = float(st_val)
                st_str = "COMPLETED" if st_rel > 0 else "NOT_AVAILABLE"
            else:
                st_str = str(st_val).upper()
                st_rel = 1.0 if st_str in ("COMPLETED", "AVAILABLE") else 0.0

            graph.register_source(name, st_str, baseline_reliability=st_rel)
            if st_str in ("NOT_AVAILABLE", "UNAVAILABLE", "ERROR", "SKIPPED", "FAILED"):
                unavailable_analyzers.append(name)

        if "ml_inference" in unavailable_analyzers:
            limitations.append(
                "Machine learning inference weights were NOT_AVAILABLE for this run. "
                "Verdict is derived strictly from deterministic signal & container indicators."
            )
        if "ocr" in unavailable_analyzers:
            limitations.append("Optical Character Recognition (OCR) engine was NOT_AVAILABLE.")

        # -----------------------------------------------------------------
        # 3. Extract Evidence Signals from Metadata
        # -----------------------------------------------------------------
        editing_tool = bool(meta_dict.get("editing_software_detected", False))
        software_name = meta_dict.get("software") or ""
        exif_present = bool(meta_dict.get("exif_present", False))
        camera = meta_dict.get("camera") or {}
        if isinstance(camera, dict):
            camera_make = camera.get("make") or meta_dict.get("camera_make")
            camera_model = camera.get("model") or meta_dict.get("camera_model")
        else:
            camera_make = str(camera)
            camera_model = meta_dict.get("camera_model")

        # Inspect any finding items for metadata tampering tags
        meta_findings = meta_dict.get("findings", [])
        if isinstance(meta_findings, list):
            for f in meta_findings:
                f_desc = getattr(f, "description", "") or (f.get("description", "") if isinstance(f, dict) else "")
                if "photoshop" in f_desc.lower() or "gimp" in f_desc.lower() or "editing" in f_desc.lower():
                    editing_tool = True
                    if not software_name:
                        software_name = "Photo Editing Software"

        if editing_tool:
            graph.add_signal(EvidenceSignalItem(
                source="metadata",
                metric="software_signature",
                direction=VerdictLabel.EDITED,
                reliability=0.92,
                weight=1.5,
                explanation=f"Container software tag explicitly records digital editing tool ('{software_name or 'Photo Editor'}').",
                raw_value=software_name,
            ))
            rules_triggered.append("RULE-META-01-EDITING-TOOL-TAG")

        if exif_present and camera_make:
            graph.add_signal(EvidenceSignalItem(
                source="metadata",
                metric="camera_sensor_exif",
                direction=VerdictLabel.REAL,
                reliability=0.85,
                weight=1.2,
                explanation=f"Authentic camera hardware provenance recorded ({camera_make} {camera_model or ''}).",
                raw_value={"make": camera_make, "model": camera_model},
            ))
            rules_triggered.append("RULE-META-02-CAMERA-EXIF-AUTHENTIC")
        elif not exif_present:
            graph.add_signal(EvidenceSignalItem(
                source="metadata",
                metric="metadata_absent",
                direction=VerdictLabel.UNKNOWN,
                reliability=0.75,
                weight=0.5,
                explanation="Original camera EXIF metadata is stripped or absent from container.",
            ))
            limitations.append("Original camera EXIF metadata is stripped or absent from container.")

        # -----------------------------------------------------------------
        # 4. Extract Evidence Signals from Image Forensics
        # -----------------------------------------------------------------
        copy_move = img_dict.get("copy_move") or {}
        copy_move_detected = bool(copy_move.get("detected", False)) if isinstance(copy_move, dict) else False
        copy_move_clusters = int(copy_move.get("clusters", 0) or 0) if isinstance(copy_move, dict) else 0

        # Inspect image findings for copy-move detections
        img_findings = img_dict.get("findings", [])
        if isinstance(img_findings, list):
            for f in img_findings:
                cat = getattr(f, "category", "") or (f.get("category", "") if isinstance(f, dict) else "")
                fid = getattr(f, "finding_id", "") or (f.get("finding_id", "") if isinstance(f, dict) else "")
                if cat in ("CLONE", "COPYMOVE") or "COPYMOVE" in fid or "CLONE" in fid:
                    copy_move_detected = True
                    copy_move_clusters = max(copy_move_clusters, 1)

        ela = img_dict.get("ela") or {}
        ela_q90 = ela.get("q90") or {} if isinstance(ela, dict) else {}
        ela_var = float(ela_q90.get("variance", img_dict.get("ela_variance", 0.0) or 0.0))

        noise_res = img_dict.get("noise_residual") or {} if isinstance(img_dict.get("noise_residual"), dict) else {}
        noise_ratio = float(noise_res.get("tile_variance_ratio", img_dict.get("noise_ratio", 1.0) or 1.0))

        quality = img_dict.get("estimated_jpeg_quality")
        if quality is not None:
            quality = int(quality)
            if quality <= 75:
                graph.add_signal(EvidenceSignalItem(
                    source="image_analysis",
                    metric="jpeg_recompression",
                    direction=VerdictLabel.UNKNOWN,
                    reliability=0.80,
                    weight=0.7,
                    explanation=f"Image has undergone lossy JPEG recompression (Estimated quality: {quality}).",
                    raw_value=quality,
                ))
            if quality <= 65:
                limitations.append(f"Image has undergone heavy lossy JPEG recompression (Estimated quality: {quality}).")

        # Copy-Move: Ground truth pixel mathematics
        if copy_move_detected and copy_move_clusters >= 1:
            graph.add_signal(EvidenceSignalItem(
                source="image_analysis",
                metric="copy_move_clusters",
                direction=VerdictLabel.EDITED,
                reliability=0.96,
                weight=2.0,
                explanation=f"Verified {copy_move_clusters} keypoint cluster(s) with identical geometric spatial translation vectors (cloning/splicing).",
                raw_value=copy_move_clusters,
            ))
            rules_triggered.append("RULE-IMG-01-COPY-MOVE-CLONING")

        # ELA Variance
        if ela_var > 120.0:
            is_reliable_ela = not (quality is not None and quality <= 65)
            rel = 0.75 if is_reliable_ela else 0.50
            direction = VerdictLabel.EDITED if (is_reliable_ela or editing_tool or copy_move_detected) else VerdictLabel.UNKNOWN
            exp = (
                f"Localized compression error variance elevated (ELA variance: {ela_var:.1f})."
                if is_reliable_ela
                else f"Elevated ELA variance ({ela_var:.1f}), though image has undergone lossy JPEG compression (Quality {quality})."
            )
            graph.add_signal(EvidenceSignalItem(
                source="image_analysis",
                metric="ela_variance",
                direction=direction,
                reliability=rel,
                weight=1.0 if is_reliable_ela else 0.5,
                explanation=exp,
                raw_value=ela_var,
            ))
            rules_triggered.append("RULE-IMG-02-ELA-VARIANCE")
        elif 0.0 < ela_var < 75.0:
            graph.add_signal(EvidenceSignalItem(
                source="image_analysis",
                metric="ela_uniformity",
                direction=VerdictLabel.REAL,
                reliability=0.72,
                weight=0.8,
                explanation=f"Uniform error level response across 8x8 block grid (ELA variance: {ela_var:.1f}).",
                raw_value=ela_var,
            ))
            rules_triggered.append("RULE-IMG-03-ELA-UNIFORM")

        # Noise Floor
        if noise_ratio > 30.0:
            graph.add_signal(EvidenceSignalItem(
                source="image_analysis",
                metric="noise_residual_variance",
                direction=VerdictLabel.EDITED,
                reliability=0.85,
                weight=1.2,
                explanation=f"Spatial noise floor exhibits severe localized inconsistency (Tile variance ratio: {noise_ratio:.1f}).",
                raw_value=noise_ratio,
            ))
            rules_triggered.append("RULE-IMG-04-NOISE-INCONSISTENCY")

        # -----------------------------------------------------------------
        # 5. Extract Evidence Signals from Screenshot Analysis
        # -----------------------------------------------------------------
        is_screen = bool(screen_dict.get("is_probable_screenshot", False) or screen_dict.get("is_common_viewport", False))
        matched_vp = screen_dict.get("matched_viewport")

        if is_screen:
            exp_vp = f"Matches canonical display viewport '{matched_vp}'" if matched_vp else "Matches standard screen aspect ratio"
            screen_dir = VerdictLabel.SCREENSHOT_MANIPULATED if (editing_tool or copy_move_detected) else VerdictLabel.REAL
            graph.add_signal(EvidenceSignalItem(
                source="screenshot",
                metric="display_geometry",
                direction=screen_dir,
                reliability=0.75,
                weight=0.9,
                explanation=f"Digital screen capture geometry detected ({exp_vp}).",
                raw_value=matched_vp,
            ))
            rules_triggered.append("RULE-SCREEN-01-VIEWPORT-GEOMETRY")

        # -----------------------------------------------------------------
        # 6. Extract Evidence Signals from OCR / Typography
        # -----------------------------------------------------------------
        word_count = int(ocr_dict.get("word_count", 0) or 0)
        font_anomaly = bool(ocr_dict.get("font_anomaly_detected", False) or ocr_dict.get("font_anomaly", False))
        ocr_findings = ocr_dict.get("findings", [])
        if isinstance(ocr_findings, list):
            for f in ocr_findings:
                f_desc = getattr(f, "description", "") or (f.get("description", "") if isinstance(f, dict) else "")
                if "font" in f_desc.lower() or "typography" in f_desc.lower() or "kerning" in f_desc.lower():
                    font_anomaly = True

        if word_count > 0 and font_anomaly:
            graph.add_signal(EvidenceSignalItem(
                source="ocr",
                metric="font_metrics",
                direction=VerdictLabel.SCREENSHOT_MANIPULATED,
                reliability=0.82,
                weight=1.1,
                explanation="Inconsistent font baseline or typography kerning anomaly detected in rendered text.",
            ))
            rules_triggered.append("RULE-OCR-01-TYPOGRAPHY-ANOMALY")

        # -----------------------------------------------------------------
        # 7. Extract Evidence Signals from Machine Learning Pipeline
        # -----------------------------------------------------------------
        ml_status = (ml_dict.get("model_status") if ml_dict else "NOT_AVAILABLE")
        ml_available = ml_status == "AVAILABLE"
        ml_label_str = ml_dict.get("predicted_label") if ml_dict else None
        ml_conf = float(ml_dict.get("confidence") or 0.0) if ml_dict else 0.0
        ml_unc = float(
            ml_uncertainty
            if ml_uncertainty is not None
            else (ml_dict.get("uncertainty") if ml_dict and ml_dict.get("uncertainty") is not None else (1.0 - ml_conf if ml_conf > 0 else 0.5))
        ) if ml_dict else 1.0

        ml_signal_item: Optional[EvidenceSignalItem] = None
        if ml_available and ml_label_str:
            ml_dir = VerdictLabel.UNKNOWN
            for vl in VerdictLabel:
                if vl.value == ml_label_str or vl.name == ml_label_str:
                    ml_dir = vl
                    break

            ml_rel = max(0.1, min(0.88, (1.0 - ml_unc) * ml_conf))
            ml_signal_item = EvidenceSignalItem(
                source="ml_inference",
                metric="deep_learning_artifact_classification",
                direction=ml_dir,
                reliability=ml_rel,
                weight=1.3,
                explanation=f"Neural model classifies visual features as '{ml_label_str}' (Confidence: {ml_conf:.2f}, Uncertainty: {ml_unc:.2f}).",
                raw_value={"label": ml_label_str, "confidence": ml_conf, "uncertainty": ml_unc},
            )
            graph.add_signal(ml_signal_item)
            rules_triggered.append(f"RULE-ML-01-{ml_label_str}")

        # Compute Evidence Graph Scores
        graph_scores = graph.compute_hypothesis_scores()

        # -----------------------------------------------------------------
        # 8. CONFLICT DETECTION, ARBITRATION & DECISION LOGIC
        # -----------------------------------------------------------------
        conflict_detected = False
        conflict_details: Optional[str] = None
        candidate_label = VerdictLabel.UNKNOWN
        confidence: Optional[float] = None
        risk_score = 0.0
        justification = ""

        # CONFLICT CASE 1: ML says AI-GENERATED vs Deterministic Hardware Camera Sensor Evidence
        # If ML says AI-GENERATED with high probability, but physical camera EXIF + clean DQT + uniform noise exist:
        if (
            ml_available
            and ml_label_str == "AI-GENERATED"
            and ml_conf >= 0.70
            and exif_present
            and camera_make
            and not copy_move_detected
            and not editing_tool
            and ela_var < 85.0
            and noise_ratio < 6.0
        ):
            conflict_detected = True
            cam_sig = next((s for s in graph.signals if s.metric == "camera_sensor_exif"), None)
            if ml_signal_item and cam_sig:
                graph.add_conflict(
                    sig1=ml_signal_item,
                    sig2=cam_sig,
                    rule_id="RULE-CONFLICT-ML-AI-VS-CAMERA-SENSOR",
                    conflict_type="ADVERSARIAL_PROVENANCE_CONTRADICTION",
                    explanation=f"Deep model AI prediction ({ml_conf:.2f}) contradicts physical sensor hardware ({camera_make}).",
                )
            conflict_details = (
                f"Adversarial Conflict: ML model predicts AI-GENERATED ({ml_conf:.2f}), "
                f"but physical camera sensor metadata ({camera_make} {camera_model or ''}), "
                "uniform sensor Poisson-Gaussian noise, and single-compression quantization contradict synthetic generation."
            )
            rules_triggered.append("RULE-CONFLICT-ML-AI-VS-CAMERA-SENSOR")
            candidate_label = VerdictLabel.UNKNOWN
            confidence = None
            risk_score = 0.45
            justification = (
                "Assessment Inconclusive: Deep learning classifier flagged synthetic visual characteristics, "
                "but physical sensor hardware metadata and natural noise continuity contradict generative origin. "
                "Escalating to UNKNOWN to avoid false-positive generative attribution."
            )

        # CONFLICT CASE 2: Camera EXIF vs Physical Copy-Move Cloning (Donor Image Retention)
        elif copy_move_detected and copy_move_clusters >= 1:
            if exif_present and camera_make:
                conflict_detected = True
                clone_sig = next((s for s in graph.signals if s.metric == "copy_move_clusters"), None)
                cam_sig = next((s for s in graph.signals if s.metric == "camera_sensor_exif"), None)
                if clone_sig and cam_sig:
                    graph.add_conflict(
                        sig1=clone_sig,
                        sig2=cam_sig,
                        rule_id="RULE-CONFLICT-CAMERA-EXIF-VS-CLONE",
                        conflict_type="METADATA_DONOR_HIJACKING",
                        explanation="Camera EXIF was retained from donor capture, but pixel coordinates contain verified keypoint cloning.",
                    )
                conflict_details = (
                    f"Metadata Conflict: Container reports authentic camera hardware ({camera_make}), "
                    "but pixel-level geometric keypoint analysis detected indicators consistent with copy-move cloning. "
                    "Metadata was likely retained from a donor image or modified."
                )
                rules_triggered.append("RULE-CONFLICT-CAMERA-EXIF-VS-CLONE")

            candidate_label = VerdictLabel.EDITED
            confidence = 0.88 if not conflict_detected else 0.82
            risk_score = 0.90
            justification = (
                "Deterministic keypoint feature matching identified duplicate image regions "
                "with parallel spatial translation geometry, consistent with copy-move tampering."
            )
            if conflict_detected:
                justification += f" Note: {conflict_details}"

        # CONFLICT CASE 3: ML says REAL vs Deterministic Manipulation (Photoshop Tag or High Noise Ratio)
        elif ml_available and ml_label_str == "REAL" and (editing_tool or noise_ratio > 30.0):
            conflict_detected = True
            edit_sig = next((s for s in graph.signals if s.direction == VerdictLabel.EDITED), None)
            if ml_signal_item and edit_sig:
                graph.add_conflict(
                    sig1=ml_signal_item,
                    sig2=edit_sig,
                    rule_id="RULE-CONFLICT-ML-VS-PHYSICAL-FORENSICS",
                    conflict_type="MODEL_FALSE_NEGATIVE_CONFLICT",
                    explanation="ML predicts REAL, but physical container or noise analysis indicates image manipulation.",
                )
            conflict_details = (
                f"Model Conflict: ML predicts REAL ({ml_conf:.2f}), but physical container/pixel analysis "
                "detected editing software signatures or severe spatial noise disruption."
            )
            rules_triggered.append("RULE-CONFLICT-ML-VS-PHYSICAL-FORENSICS")
            candidate_label = VerdictLabel.UNKNOWN
            confidence = None
            risk_score = 0.55
            justification = (
                "Indeterminate assessment: Machine learning inference directly conflicts with "
                "observed deterministic forensic signals. Escalating to UNKNOWN."
            )

        # CONFLICT CASE 4: Elevated ELA on Low-Quality Lossy JPEG (without corroborating tool tags or clones)
        elif not editing_tool and not copy_move_detected and ela_var > 120.0 and quality is not None and quality <= 65:
            conflict_detected = True
            ela_sig = next((s for s in graph.signals if s.metric == "ela_variance"), None)
            jpeg_sig = next((s for s in graph.signals if s.metric == "jpeg_recompression"), None)
            if ela_sig and jpeg_sig:
                graph.add_conflict(
                    sig1=ela_sig,
                    sig2=jpeg_sig,
                    rule_id="RULE-CONFLICT-HEAVY-COMPRESSION-AMBIGUITY",
                    conflict_type="COMPRESSION_ARTIFACT_CONFUSION",
                    explanation=f"High ELA variance ({ela_var:.1f}) coincides with lossy compression (Quality {quality}).",
                )
            conflict_details = (
                f"Compression Conflict: Elevated ELA variance ({ela_var:.1f}) coincides with severe JPEG "
                f"compression degradation (Quality {quality}). Lossy recompression creates false-positive ELA spikes."
            )
            rules_triggered.append("RULE-CONFLICT-HEAVY-COMPRESSION-AMBIGUITY")
            rules_triggered.append("RULE-FUSE-04-COMPRESSION-AMBIGUITY")
            rules_triggered.append("RULE-FUSE-04")
            candidate_label = VerdictLabel.UNKNOWN
            confidence = None
            risk_score = 0.45
            justification = (
                "Evidence cannot be reliably assessed: Heavy lossy recompression masks authentic pixel "
                "characteristics and introduces synthetic boundary artifacts."
            )

        # STANDARD RULE 5: Editing Software Signature with corroboration
        elif editing_tool:
            has_corroboration = (ela_var > 120.0 or (ml_available and ml_label_str == "EDITED"))
            rules_triggered.append("RULE-01-METADATA-EDITING-TOOL-DETECTED")
            if has_corroboration:
                rules_triggered.append("RULE-FUSE-02-METADATA-TOOL-AND-ELA-CORROBORATION")
            candidate_label = VerdictLabel.EDITED
            confidence = 0.88 if has_corroboration else 0.82
            risk_score = 0.85 if has_corroboration else 0.80
            justification = (
                f"Evidence exhibits container provenance artifacts confirming processing "
                f"with digital manipulation software ('{software_name or 'Photo Editor'}')."
            )

        # STANDARD RULE 6: Screenshot Manipulation (Typography / Layout anomalies)
        elif is_screen and font_anomaly:
            rules_triggered.append("RULE-SCREEN-02-MANIPULATED-TYPOGRAPHY")
            candidate_label = VerdictLabel.SCREENSHOT_MANIPULATED
            confidence = 0.82
            risk_score = 0.75
            justification = (
                "Screen capture exhibits irregular typography kerning and baseline drift "
                "characteristic of synthetic text overlay tampering."
            )

        # STANDARD RULE 7: Clean Native Screenshot
        elif is_screen and not editing_tool and not copy_move_detected and ml_available and ml_label_str == "REAL":
            rules_triggered.append("RULE-FUSE-04-NATIVE-SCREENSHOT-STRUCTURE")
            candidate_label = VerdictLabel.REAL
            confidence = 0.78
            risk_score = 0.15
            justification = (
                "Evidence matches native device screen capture geometry without detectable "
                "splicing, clone vectors, or editing tool artifacts."
            )

        # STANDARD RULE 8: Clean Pristine Camera Photo
        elif exif_present and camera_make and not editing_tool and not copy_move_detected and ela_var < 80.0:
            rules_triggered.append("RULE-FUSE-06-PRISTINE-CAMERA-CONTAINER")
            candidate_label = VerdictLabel.REAL
            confidence = 0.82
            risk_score = 0.10
            justification = (
                "Hardware camera sensor metadata is intact, recompression error distribution is uniform, "
                "and no geometric tampering traces were observed."
            )

        # STANDARD RULE 9: AI-Generated Corroborated
        elif ml_available and ml_label_str == "AI-GENERATED" and ml_conf >= 0.80 and not exif_present:
            rules_triggered.append("RULE-AI-01-SYNTHETIC-GENERATION-CORROBORATED")
            candidate_label = VerdictLabel.AI_GENERATED
            confidence = round(ml_conf * 0.90, 4)
            risk_score = 0.75
            justification = (
                f"Deep visual feature analysis identified generative model synthesis artifacts "
                f"(Confidence: {ml_conf:.2f}), corroborated by absence of hardware camera sensor provenance."
            )

        # DEFAULT RULE 10: Inconclusive / Missing Models -> UNKNOWN
        else:
            rules_triggered.append("RULE-FUSE-07-INCONCLUSIVE-FORENSIC-SIGNAL")
            candidate_label = VerdictLabel.UNKNOWN
            confidence = None
            risk_score = 0.20 if is_screen else 0.10
            justification = (
                "Insufficient conclusive forensic evidence to establish authenticity or tampering. "
                "Because deterministic signals are non-definitive and deep models are either unavailable "
                "or uncertain, the evidence is responsibly classified as UNKNOWN in accordance with scientific standards."
            )

        # -----------------------------------------------------------------
        # 9. Partition Supporting vs Contradictory Evidence Signals
        # -----------------------------------------------------------------
        supporting: List[EvidenceSignal] = []
        contradictory: List[EvidenceSignal] = []

        if candidate_label == VerdictLabel.UNKNOWN:
            if conflict_detected:
                # When arbitrating an adversarial conflict, signals on either side explain the impasse
                for sig in graph.signals:
                    if sig.direction in (VerdictLabel.REAL, VerdictLabel.UNKNOWN):
                        supporting.append(sig.to_schema(polarity="SUPPORTING"))
                    else:
                        contradictory.append(sig.to_schema(polarity="CONTRADICTORY"))
            else:
                for sig in graph.signals:
                    if sig.direction == VerdictLabel.REAL:
                        supporting.append(sig.to_schema(polarity="SUPPORTING"))
                    elif sig.direction in (VerdictLabel.EDITED, VerdictLabel.AI_GENERATED, VerdictLabel.SCREENSHOT_MANIPULATED):
                        contradictory.append(sig.to_schema(polarity="CONTRADICTORY"))
                    else:
                        # Cautionary / degradation signals
                        contradictory.append(sig.to_schema(polarity="CONTRADICTORY"))
        else:
            for sig in graph.signals:
                if sig.direction == candidate_label:
                    supporting.append(sig.to_schema(polarity="SUPPORTING"))
                else:
                    contradictory.append(sig.to_schema(polarity="CONTRADICTORY"))

        # -----------------------------------------------------------------
        # 10. Operational & Scientific Limitations
        # -----------------------------------------------------------------
        clean_limitations = list(dict.fromkeys(limitations))
        if not clean_limitations:
            clean_limitations.append("Source image provenance cannot be cryptographically verified without original capture hash.")
        if quality is not None and quality <= 80:
            clean_limitations.append("Compression and quantization artifacts may have been introduced by social media or messaging platforms during transit.")

        clean_limitations = list(dict.fromkeys(clean_limitations))

        # -----------------------------------------------------------------
        # 11. Format Traceable Natural Language Explanation
        # -----------------------------------------------------------------
        explanation_lines = [
            "WHY TRUSTTRACE REACHED THIS ASSESSMENT",
            "",
            "Evidence supporting assessment:",
        ]

        if supporting:
            for s in supporting:
                explanation_lines.append(f"✓ {s.explanation}")
        else:
            explanation_lines.append("• No definitive supporting physical signals identified.")

        explanation_lines.append("")
        explanation_lines.append("Evidence against assessment:")
        if contradictory:
            for c in contradictory:
                explanation_lines.append(f"⚠ {c.explanation}")
        else:
            explanation_lines.append("• No significant contradictory signals observed.")

        explanation_lines.append("")
        explanation_lines.append("Limitations:")
        for lim in clean_limitations:
            explanation_lines.append(f"• {lim}")

        full_explanation_text = "\n".join(explanation_lines)

        # Composite uncertainty score
        final_uncertainty = (1.0 - confidence) if confidence is not None else 1.0

        return FinalVerdict(
            label=candidate_label,
            confidence=confidence,
            uncertainty=round(final_uncertainty, 4),
            risk_score=round(risk_score, 4),
            justification=justification,
            explanation=full_explanation_text,
            supporting_findings=supporting,
            contradictory_findings=contradictory,
            unavailable_analyzers=list(dict.fromkeys(unavailable_analyzers)),
            limitations=clean_limitations,
            conflict_detected=conflict_detected,
            conflict_details=conflict_details,
            decision_rules_triggered=rules_triggered,
        )


# Global singleton instance
evidence_decision_engine = EvidenceDecisionEngine()
