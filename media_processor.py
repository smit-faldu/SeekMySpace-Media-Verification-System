import cv2
import exifread
import numpy as np
from PIL import Image, UnidentifiedImageError
from typing import Dict, Any, List, Optional
import os
from pymediainfo import MediaInfo
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

    return frames
def extract_metadata_from_video(video_path: str) -> Dict[str, Any]:
    """Extract metadata from video files using pymediainfo."""
    metadata = {'has_gps': False, 'has_timestamp': False, 'raw_data': {}}
    if not os.path.exists(video_path): return metadata

    try:
        media_info = MediaInfo.parse(video_path)
        for track in media_info.tracks:
            if track.track_type == "General":
                if track.encoded_date or track.file_last_modification_date:
                    metadata['has_timestamp'] = True
                if track.com_apple_quicktime_location_iso6709 or track.xyz:
                    metadata['has_gps'] = True
        logger.info(f"Video Metadata: GPS={metadata['has_gps']}, Timestamp={metadata['has_timestamp']}")
    except Exception as e:
        logger.error(f"Error reading video metadata: {e}", exc_info=True)
    return metadata

def extract_frames_from_video(video_path: str, interval_seconds: int = 1, max_frames: int = 5) -> List[np.ndarray]:
    """Extract frames dynamically based on video length."""
    frames: List[np.ndarray] = []
    if not os.path.exists(video_path): return frames

    try:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0: fps = 30 # Fallback
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = fps * interval_seconds
        
        for i in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(frames) >= max_frames: # Prevent memory overload on long videos
                break
                
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
    except Exception as e:
        logger.error(f"Error extracting frames: {e}", exc_info=True)
        
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
