import cv2
import numpy as np
import asyncio
from typing import Tuple, Optional
from logger import get_logger
from config import config

logger = get_logger(__name__)

async def evaluate_quality(frame: Optional[np.ndarray]) -> Tuple[float, str]:
    def _evaluate():
        if frame is None or frame.size == 0:
            return 0.0, "Invalid frame."

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
                
            # Blur Detection
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_range = config.QUALITY_BLUR_SHARP_THRESHOLD - config.QUALITY_BLUR_POOR_THRESHOLD
            blur_score = ((laplacian_var - config.QUALITY_BLUR_POOR_THRESHOLD) / blur_range) * 15.0
            blur_score = float(np.clip(blur_score, 0.0, 15.0))
            
            if blur_score > 12: blur_reason = "Very sharp"
            elif blur_score > 7: blur_reason = "Acceptable sharpness"
            else: blur_reason = "Blurry"
                
            # Exposure Detection
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            total_pixels = gray.shape[0] * gray.shape[1]
            underexposed_ratio = float(np.sum(hist[0:20]) / total_pixels)
            overexposed_ratio = float(np.sum(hist[235:256]) / total_pixels)
            
            bad_exposure_total = underexposed_ratio + overexposed_ratio
            exposure_penalty = (bad_exposure_total / 0.30) * 15.0
            exposure_score = float(np.clip(15.0 - exposure_penalty, 0.0, 15.0))
            
            if exposure_score > 12: exposure_reason = "Good exposure"
            elif exposure_score > 7: exposure_reason = "Slightly poor exposure"
            else: exposure_reason = "Harsh exposure"
                
            return round(blur_score + exposure_score, 1), f"Blur: {blur_reason} ({blur_score:.1f}/15). Exposure: {exposure_reason} ({exposure_score:.1f}/15)."
        except Exception as e:
            logger.error(f"Error evaluating quality: {e}")
            return 0.0, f"Error calculating quality: {str(e)}"
            
    return await asyncio.to_thread(_evaluate)