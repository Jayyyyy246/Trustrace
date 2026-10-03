"""
MODULE 3 — IMAGE FORENSICS
Comprehensive deterministic computer vision and physical/compression forensic suite:
- Dimensions, channels, color statistics
- JPEG quantization table (DQT) extraction and quality estimation
- High-pass noise residual & spatial variance analysis
- Edge consistency and gradient distribution
- Local block variance & anomaly detection
- Calibrated Error Level Analysis (ELA)
- SIFT/ORB keypoint copy-move similarity detection
"""
import io
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image

from app.schemas.common import FindingSeverity
from app.schemas.forensic import FindingItem, BaseAnalyzerResult


class ForensicImageAnalyzer:
    """Deterministic image and signal processing forensic engine."""

    def _extract_quantization_tables(self, img_pil: Image.Image) -> Tuple[Dict[str, Any], Optional[int]]:
        """Extracts JPEG Discrete Quantization Tables (DQT) and estimates quality factor."""
        qt_info: Dict[str, Any] = {"available": False, "tables": {}}
        estimated_quality = None

        if hasattr(img_pil, "quantization") and img_pil.quantization:
            qt_info["available"] = True
            for qk, qtable in img_pil.quantization.items():
                qt_info["tables"][f"table_{qk}"] = list(qtable)
            
            # IJG standard 50% luminance table baseline
            # Estimate quality from Table 0 (Luminance)
            if 0 in img_pil.quantization:
                lum_table = img_pil.quantization[0]
                # Average of lower frequency AC values
                if len(lum_table) >= 16:
                    avg_lum = float(np.mean(lum_table[:16]))
                    # Heuristic mapping between IJG table mean and quality factor
                    if avg_lum <= 2:
                        estimated_quality = 98
                    elif avg_lum <= 5:
                        estimated_quality = 95
                    elif avg_lum <= 10:
                        estimated_quality = 90
                    elif avg_lum <= 18:
                        estimated_quality = 80
                    elif avg_lum <= 28:
                        estimated_quality = 70
                    elif avg_lum <= 40:
                        estimated_quality = 60
                    else:
                        estimated_quality = 50

        return qt_info, estimated_quality

    def _compute_color_statistics(self, cv_bgr: np.ndarray) -> Dict[str, Any]:
        """Calculates multi-channel statistical moments (mean, std, skewness)."""
        color_stats: Dict[str, Any] = {}
        channels = cv2.split(cv_bgr)
        names = ["blue", "green", "red"]

        for idx, ch in enumerate(channels[:3]):
            ch_float = ch.astype(np.float64)
            mean = float(np.mean(ch_float))
            std = float(np.std(ch_float))
            # Pearson's moment coefficient of skewness
            if std > 1e-4:
                skew = float(np.mean(((ch_float - mean) / std) ** 3))
            else:
                skew = 0.0

            color_stats[names[idx]] = {
                "mean": round(mean, 2),
                "std": round(std, 2),
                "skewness": round(skew, 3),
            }

        return color_stats

    def _compute_noise_residual(self, gray: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculates Median Filter Residual (MFR) and tile noise variance consistency.
        Returns: (global_noise_std, tile_variance_ratio, tile_variance_std)
        """
        # 1. Median filter residual: R = |I - medfilt(I)|
        if gray.shape[0] < 8 or gray.shape[1] < 8:
            return 0.0, 1.0, 0.0

        med = cv2.medianBlur(gray, 3)
        residual = np.abs(gray.astype(np.float32) - med.astype(np.float32))
        global_noise_std = float(np.std(residual))

        # 2. Divide into 32x32 tiles to inspect local noise floor consistency
        h, w = gray.shape
        tile_size = 32
        tile_variances = []

        for y in range(0, h - tile_size + 1, tile_size):
            for x in range(0, w - tile_size + 1, tile_size):
                tile = residual[y:y+tile_size, x:x+tile_size]
                tile_variances.append(float(np.var(tile)))

        if len(tile_variances) > 4:
            min_v = max(float(np.min(tile_variances)), 0.01)
            max_v = float(np.max(tile_variances))
            tile_variance_ratio = max_v / min_v
            tile_variance_std = float(np.std(tile_variances))
        else:
            tile_variance_ratio = 1.0
            tile_variance_std = 0.0

        return round(global_noise_std, 3), round(tile_variance_ratio, 2), round(tile_variance_std, 3)

    def _compute_edge_consistency(self, gray: np.ndarray) -> Dict[str, Any]:
        """Evaluates Sobel edge gradient distribution across 4 quadrants."""
        h, w = gray.shape
        if h < 8 or w < 8:
            return {
                "mean_edge_gradient": 0.0,
                "quadrant_means": [0.0, 0.0, 0.0, 0.0],
                "quadrant_disparity": 1.0,
            }

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobelx**2 + sobely**2)

        mid_y, mid_x = h // 2, w // 2
        q1 = float(np.mean(mag[:mid_y, :mid_x])) if mid_y > 0 and mid_x > 0 else 0.0
        q2 = float(np.mean(mag[:mid_y, mid_x:])) if mid_y > 0 and mid_x < w else 0.0
        q3 = float(np.mean(mag[mid_y:, :mid_x])) if mid_y < h and mid_x > 0 else 0.0
        q4 = float(np.mean(mag[mid_y:, mid_x:])) if mid_y < h and mid_x < w else 0.0

        quad_means = [q1, q2, q3, q4]
        edge_disparity = round(max(quad_means) / max(min(quad_means), 0.01), 2)

        return {
            "mean_edge_gradient": round(float(np.mean(mag)), 2),
            "quadrant_means": [round(q, 2) for q in quad_means],
            "quadrant_disparity": edge_disparity,
        }

    def _compute_ela(self, pil_img: Image.Image) -> Dict[str, Any]:
        """Computes Error Level Analysis (ELA) metrics at Q=90 and Q=95."""
        if pil_img.width < 8 or pil_img.height < 8:
            return {
                "q90": {"mean_delta": 0.0, "variance": 0.0, "max_delta": 0.0},
                "q95": {"mean_delta": 0.0, "variance": 0.0, "max_delta": 0.0},
            }

        rgb = pil_img.convert("RGB")
        ela_results = {}

        for quality in [90, 95]:
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            recomp = Image.open(buf)

            arr_orig = np.array(rgb, dtype=np.float32)
            arr_recomp = np.array(recomp, dtype=np.float32)

            diff = np.abs(arr_orig - arr_recomp)
            mean_delta = float(np.mean(diff))
            var_delta = float(np.var(diff))
            max_delta = float(np.max(diff))

            ela_results[f"q{quality}"] = {
                "mean_delta": round(mean_delta, 3),
                "variance": round(var_delta, 3),
                "max_delta": round(max_delta, 2),
            }

        return ela_results

    def _detect_copy_move_clusters(self, gray: np.ndarray) -> Dict[str, Any]:
        """
        Detects copy-move cloning artifacts using ORB keypoints and vector displacement clustering.
        Excludes self-matches and finds parallel shift vectors.
        """
        if gray.shape[0] < 32 or gray.shape[1] < 32:
            return {"detected": False, "candidate_pairs": 0, "clusters": 0}

        orb = cv2.ORB_create(nfeatures=600)
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        if descriptors is None or len(keypoints) < 15:
            return {"detected": False, "candidate_pairs": 0, "clusters": 0}

        # Match using Brute Force with Hamming distance (3 nearest neighbors to handle self-match)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(descriptors, descriptors, k=3)

        candidate_pairs = []
        min_spatial_dist = 25.0  # Must be separated by at least 25 pixels

        for m_group in matches:
            if len(m_group) >= 3:
                # m0 is typically self (dist 0), m1 is best non-self, m2 is runner-up
                m_self, m1, m2 = m_group[0], m_group[1], m_group[2]
                if m1.trainIdx == m1.queryIdx:
                    # In rare tie, swap
                    m1 = m_group[2]
                    m2 = m_group[1]

                # Lowe's ratio test on the non-self candidates
                if m1.distance < 0.85 * m2.distance or m1.distance < 20.0:
                    pt1 = np.array(keypoints[m1.queryIdx].pt)
                    pt2 = np.array(keypoints[m1.trainIdx].pt)
                    spatial_dist = float(np.linalg.norm(pt1 - pt2))
                    if spatial_dist >= min_spatial_dist:
                        disp_vector = pt2 - pt1
                        candidate_pairs.append({
                            "pt1": [float(pt1[0]), float(pt1[1])],
                            "pt2": [float(pt2[0]), float(pt2[1])],
                            "vector": [float(disp_vector[0]), float(disp_vector[1])],
                        })

        # Cluster displacement vectors (parallel translation check)
        cluster_count = 0
        if len(candidate_pairs) >= 6:
            vectors = np.array([p["vector"] for p in candidate_pairs])
            # Quantize vectors to 20-pixel bins
            binned = np.round(vectors / 20.0) * 20.0
            unique_bins, counts = np.unique(binned, axis=0, return_counts=True)
            # A cluster has at least 5 keypoint pairs sharing identical translation vector
            cluster_count = int(np.sum(counts >= 5))

        return {
            "detected": cluster_count > 0,
            "candidate_pairs": len(candidate_pairs),
            "clusters": cluster_count,
            "keypoints_evaluated": len(keypoints),
        }

    def analyze(self, data: bytes) -> BaseAnalyzerResult:
        findings: List[FindingItem] = []
        metrics: Dict[str, Any] = {}
        limitations: List[str] = [
            "Error Level Analysis (ELA) is sensitive to overall image compression and texture density; high ELA error in high-contrast or text areas is normal physical behavior.",
            "Copy-move keypoint matching may produce false positives in repetitive textures such as grass, brick walls, carpet patterns, or water waves.",
            "Noise analysis is non-conclusive for images that have undergone repeated resizing or re-encoding across social networks.",
            "A suspicious forensic metric is an investigative lead, not standalone proof of fraud.",
        ]

        try:
            # 1. Pillow load
            with Image.open(io.BytesIO(data)) as pil_img:
                width, height = pil_img.size
                channels = len(pil_img.getbands())
                color_space = pil_img.mode
                aspect_ratio = round(width / max(height, 1), 4)

                # Quantization tables & estimated quality
                dqt_info, estimated_quality = self._extract_quantization_tables(pil_img)
                ela_metrics = self._compute_ela(pil_img)

            # 2. OpenCV decode
            nparr = np.frombuffer(data, np.uint8)
            cv_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if cv_img is None:
                raise ValueError("Image bytes could not be decoded into pixel matrix.")

            # Grayscale conversion
            if len(cv_img.shape) == 2:
                gray = cv_img
                cv_bgr = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
            elif cv_img.shape[2] == 4:
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2GRAY)
                cv_bgr = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2BGR)
            else:
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                cv_bgr = cv_img

            # 3. Compute metrics
            lap_var = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)
            color_stats = self._compute_color_statistics(cv_bgr)
            noise_std, tile_ratio, tile_std = self._compute_noise_residual(gray)
            edge_stats = self._compute_edge_consistency(gray)
            copy_move_stats = self._detect_copy_move_clusters(gray)

            metrics = {
                "dimensions": {"width": width, "height": height},
                "channels": channels,
                "color_space": color_space,
                "aspect_ratio": aspect_ratio,
                "laplacian_variance": lap_var,
                "color_statistics": color_stats,
                "quantization_tables": dqt_info,
                "estimated_jpeg_quality": estimated_quality,
                "noise_residual": {
                    "global_noise_std": noise_std,
                    "tile_variance_ratio": tile_ratio,
                    "tile_variance_std": tile_std,
                },
                "edge_consistency": edge_stats,
                "ela": ela_metrics,
                "copy_move": copy_move_stats,
            }

            # 4. Formulate Findings

            # Finding: Copy-Move cloning match
            if copy_move_stats["detected"]:
                findings.append(
                    FindingItem(
                        finding_id="FIND-IMG-COPYMOVE",
                        category="CLONE",
                        severity=FindingSeverity.HIGH,
                        title="Candidate Copy-Move Duplication Cluster Detected",
                        description=(
                            f"Identified {copy_move_stats['clusters']} cluster(s) with identical "
                            "parallel spatial displacement vectors across image keypoints."
                        ),
                        evidence=copy_move_stats,
                        interpretation=(
                            "Multiple distinct image patches share identical high-dimensional feature descriptors "
                            "and parallel displacement geometry. Strongly suggests cloning or duplicated elements."
                        ),
                        limitation=(
                            "Natural repetitive patterns (e.g. foliage, tiled pavements, window grids) "
                            "can generate geometric keypoint false positives."
                        ),
                        confidence=0.82,
                        is_anomaly=True,
                    )
                )

            # Finding: Inconsistent noise floor
            if tile_ratio > 25.0 and tile_std > 8.0:
                findings.append(
                    FindingItem(
                        finding_id="FIND-IMG-NOISE-INCONSISTENCY",
                        category="NOISE",
                        severity=FindingSeverity.MEDIUM,
                        title="Spatial Noise Floor Inconsistency",
                        description=(
                            f"Tile noise variance disparity ratio is {tile_ratio:.1f}, "
                            "exceeding typical single-camera capture variance."
                        ),
                        evidence={"tile_variance_ratio": tile_ratio, "tile_variance_std": tile_std},
                        interpretation=(
                            "Local high-frequency residuals vary significantly across regions. "
                            "May indicate a foreign patch spliced from an image with a different noise profile."
                        ),
                        limitation=(
                            "Images with mixed flat sky areas and textured surfaces naturally produce "
                            "unequal local noise residuals without manipulation."
                        ),
                        confidence=0.70,
                        is_anomaly=True,
                    )
                )

            # Finding: ELA variance divergence
            q90_var = ela_metrics.get("q90", {}).get("variance", 0.0)
            if q90_var > 150.0:
                findings.append(
                    FindingItem(
                        finding_id="FIND-IMG-ELA-ANOMALY",
                        category="COMPRESSION",
                        severity=FindingSeverity.MEDIUM,
                        title="Elevated Error Level Analysis (ELA) Divergence",
                        description=f"Error level variance at 90% recompression is elevated ({q90_var}).",
                        evidence=ela_metrics["q90"],
                        interpretation=(
                            "Different regions of the image exhibit divergent error deltas upon recompression. "
                            "Commonly observed when inserting elements saved at different JPEG compression levels."
                        ),
                        limitation=(
                            "High-frequency edges, text, and saturated colors naturally compress with higher "
                            "error deltas than uniform backgrounds. ELA must not be used as sole proof."
                        ),
                        confidence=0.68,
                        is_anomaly=True,
                    )
                )

            # Finding: Quantization Table Assessment
            if dqt_info["available"]:
                findings.append(
                    FindingItem(
                        finding_id="FIND-IMG-DQT-EXTRACTED",
                        category="COMPRESSION",
                        severity=FindingSeverity.INFO,
                        title="JPEG Quantization Matrices Extracted",
                        description=f"Extracted {len(dqt_info['tables'])} Discrete Quantization Table(s). Estimated Quality: {estimated_quality or 'Custom'}.",
                        evidence={"tables_count": len(dqt_info["tables"]), "estimated_quality": estimated_quality},
                        interpretation="Quantization matrices define the lossy frequency compression profile applied by the encoder.",
                        limitation="Standard photo editors can apply custom quantization matrices without altering image scene content.",
                        confidence=0.85,
                        is_anomaly=False,
                    )
                )

            return BaseAnalyzerResult(
                status="completed",
                findings=findings,
                metrics=metrics,
                limitations=limitations,
            )

        except Exception as e:
            return BaseAnalyzerResult(
                status="failed",
                findings=[],
                metrics={"error": str(e)},
                limitations=limitations,
            )


image_analyzer = ForensicImageAnalyzer()
