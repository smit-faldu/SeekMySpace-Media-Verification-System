import cv2
import numpy as np
from typing import Tuple, Optional

from logger import get_logger
from config import config

logger = get_logger(__name__)

def evaluate_quality(frame: Optional[np.ndarray]) -> Tuple[float, str]:
    """
    Evaluates image quality based on blur (Laplacian variance) and brightness.

    Args:
        frame (Optional[np.ndarray]): RGB image array.

    Returns:
        Tuple[float, str]: A tuple containing the total quality score (out of 30) 
                           and a human-readable reasoning string.
    """
    if frame is None or frame.size == 0:
        logger.warning("evaluate_quality called with empty or None frame.")
        return 0.0, "Invalid frame."

    try:
        # Convert to grayscale if necessary
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
            
        # 1. Blur Detection (Variance of Laplacian)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = 0.0
        blur_reason = "Very blurry"
        
        if laplacian_var > config.QUALITY_BLUR_SHARP_THRESHOLD:
            blur_score = 15.0
            blur_reason = "Sharp"
        elif laplacian_var > config.QUALITY_BLUR_ACCEPTABLE_THRESHOLD:
            blur_score = 10.0
            blur_reason = "Slightly blurry but acceptable"
        elif laplacian_var > config.QUALITY_BLUR_POOR_THRESHOLD:
            blur_score = 5.0
            blur_reason = "Blurry"
            
        # 2. Brightness Detection (Mean Pixel Value)
        mean_brightness = np.mean(gray)
        brightness_score = 0.0
        brightness_reason = "Too dark or too bright"
        
        if config.QUALITY_BRIGHTNESS_MIN_GOOD <= mean_brightness <= config.QUALITY_BRIGHTNESS_MAX_GOOD:
            brightness_score = 15.0
            brightness_reason = "Good exposure"
        elif config.QUALITY_BRIGHTNESS_MIN_ACCEPTABLE <= mean_brightness <= config.QUALITY_BRIGHTNESS_MAX_ACCEPTABLE:
            brightness_score = 10.0
            brightness_reason = "Slightly under or over exposed but acceptable"
        elif config.QUALITY_BRIGHTNESS_MIN_POOR <= mean_brightness <= config.QUALITY_BRIGHTNESS_MAX_POOR:
            brightness_score = 5.0
            brightness_reason = "Poor exposure"
            
        total_quality_score = blur_score + brightness_score
        reasoning = f"Blur: {blur_reason} (var: {laplacian_var:.1f}). Brightness: {brightness_reason} (mean: {mean_brightness:.1f})."
        
        logger.debug(f"Quality evaluated: score={total_quality_score}, reasons=[{reasoning}]")
        return total_quality_score, reasoning

    except Exception as e:
        logger.error(f"Error evaluating image quality: {e}", exc_info=True)
        return 0.0, f"Error calculating quality: {str(e)}"
