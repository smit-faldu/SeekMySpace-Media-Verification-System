from dataclasses import dataclass
from typing import Set

@dataclass
class AppConfig:
    """Centralized configuration for the Media Verification System."""
    
    # Model parameters
    YOLO_MODEL_NAME: str = "yolov8n.pt"
    
    # COCO target classes indicative of an office/commercial space
    # 56: chair, 57: couch, 58: potted plant, 60: dining table, 62: tv/monitor,
    # 63: laptop, 64: mouse, 65: remote, 66: keyboard, 67: cell phone
    TARGET_CLASSES: Set[int] = None
    
    # Scoring Weights & Thresholds
    METADATA_SCORE_GPS: int = 5
    METADATA_SCORE_TIMESTAMP: int = 5
    
    QUALITY_MAX_SCORE: int = 30
    QUALITY_BLUR_SHARP_THRESHOLD: float = 400.0
    QUALITY_BLUR_ACCEPTABLE_THRESHOLD: float = 150.0
    QUALITY_BLUR_POOR_THRESHOLD: float = 80.0
    
    QUALITY_BRIGHTNESS_MIN_GOOD: float = 80.0
    QUALITY_BRIGHTNESS_MAX_GOOD: float = 200.0
    QUALITY_BRIGHTNESS_MIN_ACCEPTABLE: float = 40.0
    QUALITY_BRIGHTNESS_MAX_ACCEPTABLE: float = 220.0
    QUALITY_BRIGHTNESS_MIN_POOR: float = 20.0
    QUALITY_BRIGHTNESS_MAX_POOR: float = 240.0
    
    RELEVANCE_CONFIDENCE_THRESHOLD: float = 0.3
    
    DECISION_AUTO_APPROVE_THRESHOLD: int = 70
    STRICT_STOCK_PHOTO_CHECK: bool = False
    
    # Video Extraction Parameters
    VIDEO_MAX_FRAMES: int = 3
    
    def __post_init__(self):
        if self.TARGET_CLASSES is None:
            self.TARGET_CLASSES = {56, 57, 58, 60, 62, 63, 64, 66, 67}

config = AppConfig()
