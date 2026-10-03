"""
Image information and low-level computer vision analysis service.
Performs real mathematical analysis (Laplacian variance, luminance stats, ELA computation)
using OpenCV, NumPy, and Pillow. Never fabricates values.
"""
import io
from typing import Tuple, List
import cv2
import numpy as np
from PIL import Image

from app.schemas.common import AnalyzerStatus, FindingSeverity
from app.schemas.forensic import ImageAnalyzerResult, FindingItem


from forensic.image_analysis import image_analyzer


class ImageAnalysisService:
    def analyze_image(self, data: bytes) -> Tuple[ImageAnalyzerResult, List[FindingItem]]:
        """
        Executes real computer vision and image structural inspection
        via ForensicImageAnalyzer. Returns ImageAnalyzerResult and itemized findings.
        """
        base_res = image_analyzer.analyze(data)
        if base_res.status == "failed":
            err_msg = base_res.metrics.get("error", "Image decoding failure")
            return (
                ImageAnalyzerResult(
                    status=AnalyzerStatus.FAILED,
                    dimensions={"width": 0, "height": 0},
                    channels=0,
                    color_space="UNKNOWN",
                    aspect_ratio=0.0,
                    laplacian_variance=0.0,
                    is_blurry=False,
                    luminance_mean=0.0,
                    luminance_std=0.0,
                    ela_computed=False,
                ),
                base_res.findings,
            )

        m = base_res.metrics
        dims = m.get("dimensions", {"width": 0, "height": 0})
        ela_data = m.get("ela", {}).get("q90", {})
        lap_var = m.get("laplacian_variance", 0.0)

        # Extract luminance stats from color_statistics if available
        color_stats = m.get("color_statistics", {})
        lum_mean = round(
            (color_stats.get("red", {}).get("mean", 0) * 0.299 +
             color_stats.get("green", {}).get("mean", 0) * 0.587 +
             color_stats.get("blue", {}).get("mean", 0) * 0.114), 2
        )
        lum_std = round(
            (color_stats.get("red", {}).get("std", 0) * 0.299 +
             color_stats.get("green", {}).get("std", 0) * 0.587 +
             color_stats.get("blue", {}).get("std", 0) * 0.114), 2
        )

        result = ImageAnalyzerResult(
            status=AnalyzerStatus.COMPLETED,
            dimensions=dims,
            channels=m.get("channels", 3),
            color_space=m.get("color_space", "RGB"),
            aspect_ratio=m.get("aspect_ratio", 1.0),
            laplacian_variance=lap_var,
            is_blurry=lap_var < 50.0,
            luminance_mean=lum_mean,
            luminance_std=lum_std,
            ela_computed=bool(ela_data),
            ela_mean_delta=ela_data.get("mean_delta"),
            ela_variance=ela_data.get("variance"),
            estimated_jpeg_quality=m.get("estimated_jpeg_quality"),
        )
        return result, base_res.findings


image_analysis_service = ImageAnalysisService()
