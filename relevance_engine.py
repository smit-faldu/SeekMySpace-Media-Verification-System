import numpy as np
from typing import Tuple, Optional
from ultralytics import YOLO

from logger import get_logger
from config import config

logger = get_logger(__name__)

class RelevanceEngine:
    """
    Engine for evaluating the relevance of media to commercial/office spaces
    using an object detection model (YOLO).
    """
    def __init__(self):
        try:
            # Load YOLO model. Using nano for speed.
            logger.info(f"Loading YOLO model: {config.YOLO_MODEL_NAME}")
            self.model = YOLO(config.YOLO_MODEL_NAME)
            self.target_classes = config.TARGET_CLASSES
        except Exception as e:
            logger.error(f"Failed to load YOLO model {config.YOLO_MODEL_NAME}: {e}", exc_info=True)
            self.model = None

    def evaluate_relevance(self, frame: Optional[np.ndarray]) -> Tuple[int, str]:
        """
        Evaluates the relevance of a frame to a commercial/office space.
        
        Args:
            frame (Optional[np.ndarray]): RGB image array.

        Returns:
            Tuple[int, str]: A relevance score (0 to 50) and a reasoning string.
        """
        if frame is None or frame.size == 0:
            logger.warning("evaluate_relevance called with empty or None frame.")
            return 0, "Invalid or empty frame."
            
        if self.model is None:
            logger.error("YOLO model is not loaded. Returning 0 relevance.")
            return 0, "Relevance model failed to load."

        try:
            # Inference (verbose=False to keep stdout clean)
            results = self.model(frame, verbose=False)
            
            detected_targets = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    
                    if cls_id in self.target_classes and confidence > config.RELEVANCE_CONFIDENCE_THRESHOLD:
                        detected_targets.append(result.names[cls_id])
                        
            # Score calculation based on number of unique target classes detected
            score = 0
            reasoning = "No relevant commercial space objects detected."
            
            if len(detected_targets) > 0:
                unique_targets = set(detected_targets)
                if len(unique_targets) >= 3:
                    score = 50
                    reasoning = f"Highly relevant space. Detected: {', '.join(unique_targets)}."
                elif len(unique_targets) == 2:
                    score = 35
                    reasoning = f"Moderately relevant space. Detected: {', '.join(unique_targets)}."
                else:
                    score = 20
                    reasoning = f"Slightly relevant space. Detected: {', '.join(unique_targets)}."
                    
            logger.debug(f"Relevance evaluated: score={score}, targets detected={detected_targets}")
            return score, reasoning
            
        except Exception as e:
            logger.error(f"Error evaluating relevance: {e}", exc_info=True)
            return 0, f"Error calculating relevance: {str(e)}"
