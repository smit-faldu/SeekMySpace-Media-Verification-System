import cv2
import exifread
import numpy as np
import asyncio
from PIL import Image, UnidentifiedImageError
from typing import Dict, Any, List, Optional
import os
from pymediainfo import MediaInfo
from logger import get_logger

logger = get_logger(__name__)

async def extract_metadata_from_image(image_path: str) -> Dict[str, Any]:
    def _extract():
        metadata = {'has_gps': False, 'has_timestamp': False, 'raw_data': {}}
        if not os.path.exists(image_path): return metadata
        try:
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)
                if 'EXIF DateTimeOriginal' in tags or 'Image DateTime' in tags:
                    metadata['has_timestamp'] = True
                gps_keys = [k for k in tags.keys() if k.startswith('GPS')]
                if len(gps_keys) > 0:
                    metadata['has_gps'] = True
                metadata['raw_data'] = {k: str(v) for k, v in tags.items() if len(str(v)) < 100}
        except Exception as e:
            logger.error(f"Error reading EXIF from {image_path}: {e}")
        return metadata
    
    return await asyncio.to_thread(_extract)

async def extract_metadata_from_video(video_path: str) -> Dict[str, Any]:
    def _extract():
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
        except Exception as e:
            logger.error(f"Error reading video metadata: {e}")
        return metadata
        
    return await asyncio.to_thread(_extract)

async def extract_frames_from_video(video_path: str, interval_seconds: int = 1, max_frames: int = 5) -> List[np.ndarray]:
    def _extract():
        frames: List[np.ndarray] = []
        if not os.path.exists(video_path): return frames
        try:
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0: fps = 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = fps * interval_seconds
            
            for i in range(0, total_frames, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if len(frames) >= max_frames:
                    break
            cap.release()
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
        return frames
        
    return await asyncio.to_thread(_extract)

async def load_image(image_path: str) -> Optional[np.ndarray]:
    def _load():
        if not os.path.exists(image_path): return None
        try:
            img = Image.open(image_path).convert('RGB')
            return np.array(img)
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None
            
    return await asyncio.to_thread(_load)