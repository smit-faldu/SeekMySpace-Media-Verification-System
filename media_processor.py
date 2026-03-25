import cv2
import exifread
import numpy as np
from PIL import Image, UnidentifiedImageError
from typing import Dict, Any, List, Optional
import os

from logger import get_logger
from config import config

logger = get_logger(__name__)

def extract_metadata_from_image(image_path: str) -> Dict[str, Any]:
    """
    Extract EXIF metadata from an image safely.

    Args:
        image_path (str): The file path to the image.

    Returns:
        Dict[str, Any]: A dictionary containing metadata presence flags and raw data.
    """
    metadata = {
        'has_gps': False,
        'has_timestamp': False,
        'raw_data': {}
    }
    
    if not os.path.exists(image_path):
        logger.error(f"Image path does not exist for metadata extraction: {image_path}")
        return metadata

    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            
            # Check for Timestamp
            if 'EXIF DateTimeOriginal' in tags or 'Image DateTime' in tags:
                metadata['has_timestamp'] = True
                
            # Check for GPS
            gps_keys = [k for k in tags.keys() if k.startswith('GPS')]
            if len(gps_keys) > 0:
                metadata['has_gps'] = True
                
            # Limit raw data string lengths to avoid massive payloads
            metadata['raw_data'] = {k: str(v) for k, v in tags.items() if len(str(v)) < 100}
            
        logger.info(f"Metadata extracted for {image_path}: GPS={metadata['has_gps']}, Timestamp={metadata['has_timestamp']}")
        
    except Exception as e:
        logger.error(f"Error reading EXIF from {image_path}: {e}", exc_info=True)
        
    return metadata

def extract_frames_from_video(video_path: str, num_frames: int = config.VIDEO_MAX_FRAMES) -> List[np.ndarray]:
    """
    Extract a specified number of frames evenly spaced from a video file.

    Args:
        video_path (str): The file path to the video.
        num_frames (int): Number of frames to extract.

    Returns:
        List[np.ndarray]: A list of extracted frames in RGB format.
    """
    frames: List[np.ndarray] = []
    
    if not os.path.exists(video_path):
        logger.error(f"Video path does not exist: {video_path}")
        return frames

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Could not open video file: {video_path}")
            return frames
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            logger.warning(f"Video file has no frames or is corrupted: {video_path}")
            cap.release()
            return frames
            
        # Get evenly spaced frame indices
        intervals = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        for frame_idx in intervals:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR (OpenCV default) to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
        cap.release()
        logger.info(f"Successfully extracted {len(frames)} frames from {video_path}")
        
    except Exception as e:
        logger.error(f"Error extracting frames from video {video_path}: {e}", exc_info=True)
        
    return frames

def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from disk safely as a numpy array.

    Args:
        image_path (str): The file path to the image.

    Returns:
        Optional[np.ndarray]: RGB image array, or None if loading fails.
    """
    if not os.path.exists(image_path):
        logger.error(f"Image path does not exist: {image_path}")
        return None

    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        return np.array(img)
    except UnidentifiedImageError:
        logger.error(f"Failed to identify image file (corrupted or unsupported format): {image_path}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading image {image_path}: {e}", exc_info=True)
        return None
