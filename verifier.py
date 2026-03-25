import os
import asyncio
from typing import TypedDict, List

from media_processor import (
    extract_metadata_from_image, extract_metadata_from_video, 
    extract_frames_from_video, load_image
)
from quality_evaluator import evaluate_quality
from relevance_engine import RelevanceEngine
from logger import get_logger
from config import config

logger = get_logger(__name__)

class VerificationDetails(TypedDict):
    metadata_score: int
    quality_score: float
    relevance_score: float
    gps: bool
    timestamp: bool

class VerificationResult(TypedDict):
    score: float
    decision: str
    reasoning: str
    details: VerificationDetails

class MediaVerifier:
    def __init__(self):
        logger.info("Initializing MediaVerifier instance.")
        self.relevance_engine = None
        
    def _init_relevance_engine(self):
        if self.relevance_engine is None:
            self.relevance_engine = RelevanceEngine()

    async def verify_media(self, file_path: str) -> VerificationResult:
        error_result: VerificationResult = {
            "score": 0.0, "decision": "Error", "reasoning": "Could not read or process media.",
            "details": {"metadata_score": 0, "quality_score": 0.0, "relevance_score": 0.0, "gps": False, "timestamp": False}
        }
        
        if not os.path.exists(file_path):
            error_result["reasoning"] = f"File not found: {file_path}"
            return error_result
            
        ext = os.path.splitext(file_path)[1].lower()
        is_video = ext in ['.mp4', '.avi', '.mov']
        
        metadata_score = 0
        metadata_details = {"gps": False, "timestamp": False}
        
        try:
            # 1. & 2. Concurrent Metadata and Frame Extraction
            if not is_video:
                metadata_task = extract_metadata_from_image(file_path)
                frames_task = load_image(file_path)
                metadata, frame = await asyncio.gather(metadata_task, frames_task)
                frames = [frame] if frame is not None else []
            else:
                metadata_task = extract_metadata_from_video(file_path)
                frames_task = extract_frames_from_video(file_path, interval_seconds=1, max_frames=config.VIDEO_MAX_FRAMES)
                metadata, frames = await asyncio.gather(metadata_task, frames_task)

            if metadata.get('has_gps'):
                metadata_score += config.METADATA_SCORE_GPS
                metadata_details['gps'] = True
            if metadata.get('has_timestamp'):
                metadata_score += config.METADATA_SCORE_TIMESTAMP
                metadata_details['timestamp'] = True
                    
            if not frames:
                error_result["reasoning"] = "Failed to extract any readable frames."
                return error_result
                
            # 3. & 4. Concurrent Quality and Relevance Assessment across all frames
            self._init_relevance_engine()
            
            quality_tasks = [evaluate_quality(f) for f in frames]
            relevance_tasks = [self.relevance_engine.evaluate_relevance(f) for f in frames]
            
            # Await all model inference and OpenCV processing concurrently
            quality_results = await asyncio.gather(*quality_tasks)
            relevance_results = await asyncio.gather(*relevance_tasks)
            
            # Aggregate Quality
            total_q_score = sum(res[0] for res in quality_results)
            quality_score = total_q_score / len(frames)
            quality_reason = quality_results[0][1] if len(frames) == 1 else f"Average quality ({quality_score:.1f}) across {len(frames)} frames."
            
            # Aggregate Relevance (Max score across frames)
            relevance_score = max(res[0] for res in relevance_results)
            max_idx = [res[0] for res in relevance_results].index(relevance_score)
            relevance_reason = relevance_results[max_idx][1]

            # 5. Final Score Calculation
            total_score = float(metadata_score + quality_score + relevance_score)
            
            # 6. Decision Logic
            decision = "Pending"
            if config.STRICT_STOCK_PHOTO_CHECK and metadata_score == 0 and quality_score >= 25.0 and total_score >= config.DECISION_AUTO_APPROVE_THRESHOLD:
                decision = "Flag for Manual Review"
                relevance_reason += " (WARNING: High quality but missing metadata suggests stock photo.)"
                total_score -= 10.0
            
            if decision == "Pending":
                decision = "Auto Approved" if total_score >= config.DECISION_AUTO_APPROVE_THRESHOLD else "Flag for Manual Review"
                
            return {
                "score": total_score, "decision": decision,
                "reasoning": f"Metadata: {metadata_score}/10. Quality: {quality_score:.1f}/30 ({quality_reason}). Relevance: {relevance_score}/60 ({relevance_reason}).",
                "details": {
                    "metadata_score": metadata_score, "quality_score": quality_score,
                    "relevance_score": relevance_score, "gps": metadata_details["gps"], "timestamp": metadata_details["timestamp"]
                }
            }
        except Exception as e:
            logger.error(f"Uncaught exception: {e}")
            error_result["reasoning"] = f"Internal processing error: {str(e)}"
            return error_result