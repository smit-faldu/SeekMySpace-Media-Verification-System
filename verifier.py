import os
from typing import TypedDict, List

from media_processor import (
    extract_metadata_from_image, 
    extract_metadata_from_video, 
    extract_frames_from_video, 
    load_image
)
from quality_evaluator import evaluate_quality
from relevance_engine import RelevanceEngine
from logger import get_logger
from config import config

logger = get_logger(__name__)

class VerificationDetails(TypedDict):
    metadata_score: int
    quality_score: float
    relevance_score: int
    gps: bool
    timestamp: bool

class VerificationResult(TypedDict):
    score: float
    decision: str
    reasoning: str
    details: VerificationDetails

class MediaVerifier:
    """
    Main aggregator that verifies media by orchestrating
    metadata extraction, quality assessment, and relevance evaluation.
    """
    def __init__(self):
        logger.info("Initializing MediaVerifier instance.")
        self.relevance_engine = None  # Lazy loading to save memory until needed
        
    def _init_relevance_engine(self):
        if self.relevance_engine is None:
            self.relevance_engine = RelevanceEngine()

    def verify_media(self, file_path: str) -> VerificationResult:
        """
        Verify an uploaded media file against all metrics.

        Args:
            file_path (str): The file path to the uploaded media.

        Returns:
            VerificationResult: A strongly typed dict with scores, decision, reasoning, and details.
        """
        # Default error response
        error_result: VerificationResult = {
            "score": 0.0,
            "decision": "Error",
            "reasoning": "Could not read or process media.",
            "details": {
                "metadata_score": 0, "quality_score": 0.0, "relevance_score": 0,
                "gps": False, "timestamp": False
            }
        }
        
        if not os.path.exists(file_path):
            error_result["reasoning"] = f"File not found: {file_path}"
            logger.error(error_result["reasoning"])
            return error_result
            
        logger.info(f"Starting verification for: {file_path}")
        ext = os.path.splitext(file_path)[1].lower()
        is_video = ext in ['.mp4', '.avi', '.mov']
        
        # Breakdown trackers
        metadata_score = 0
        metadata_details = {"gps": False, "timestamp": False}
        quality_score = 0.0
        quality_reason = ""
        relevance_score = 0
        relevance_reason = "No relevance assessed."
        
        try:
            # 1. Metadata Extraction
            if not is_video:
                metadata = extract_metadata_from_image(file_path)
            else:
                # Upgraded to process video metadata using pymediainfo
                metadata = extract_metadata_from_video(file_path)

            if metadata.get('has_gps'):
                metadata_score += config.METADATA_SCORE_GPS
                metadata_details['gps'] = True
            if metadata.get('has_timestamp'):
                metadata_score += config.METADATA_SCORE_TIMESTAMP
                metadata_details['timestamp'] = True
                
            # 2. Extract frames
            frames = []
            if is_video:
                # Using the upgraded dynamic frame extraction
                frames = extract_frames_from_video(file_path, interval_seconds=1, max_frames=5)
            else:
                frame = load_image(file_path)
                if frame is not None:
                    frames.append(frame)
                    
            if not frames:
                error_result["reasoning"] = "Failed to extract any readable frames from the media."
                logger.error(error_result["reasoning"])
                return error_result
                
            # 3. Assess Quality (Average across frames)
            total_q_score = 0.0
            q_reasons: List[str] = []
            
            for f in frames:
                sc, rsn = evaluate_quality(f)
                total_q_score += sc
                q_reasons.append(rsn)
                
            quality_score = total_q_score / len(frames)
            quality_reason = q_reasons[0] if len(frames) == 1 else f"Average quality ({quality_score:.1f}) across {len(frames)} frames."
            
            # 4. Assess Relevance
            self._init_relevance_engine()
            r_scores: List[int] = []
            r_reasons: List[str] = []
            
            for f in frames:
                sc, rsn = self.relevance_engine.evaluate_relevance(f)
                r_scores.append(sc)
                r_reasons.append(rsn)
                
            # Use the max relevance score across frames
            if r_scores:
                relevance_score = max(r_scores)
                max_idx = r_scores.index(relevance_score)
                relevance_reason = r_reasons[max_idx]

            # 5. Final Score Calculation
            total_score = float(metadata_score + quality_score + relevance_score)
            
            # 6. Decision Logic & Heuristics
            decision = "Pending"
            
            # Heuristic 1: The "Stock Photo" Check
            # Now controlled by config.py so it doesn't ruin testing
            if config.STRICT_STOCK_PHOTO_CHECK:
                if metadata_score == 0 and quality_score >= 25.0 and total_score >= config.DECISION_AUTO_APPROVE_THRESHOLD:
                    decision = "Flag for Manual Review"
                    relevance_reason += " (WARNING: Extremely high quality but missing metadata suggests potential stock photo.)"
                    total_score -= 10.0
            
            # Standard threshold fallback if no heuristics triggered
            if decision == "Pending":
                if total_score >= config.DECISION_AUTO_APPROVE_THRESHOLD:
                    decision = "Auto Approved"
                else:
                    decision = "Flag for Manual Review"
                
            final_reasoning = (
                f"Metadata provided {metadata_score}/20 points. "
                f"Quality scored {quality_score:.1f}/30 points ({quality_reason}). "
                f"Relevance scored {relevance_score}/60 points ({relevance_reason})."
            )
            
            logger.info(f"Verification complete for {file_path}. Score: {total_score}, Decision: {decision}")
            
            return {
                "score": total_score,
                "decision": decision,
                "reasoning": final_reasoning,
                "details": {
                    "metadata_score": metadata_score,
                    "quality_score": quality_score,
                    "relevance_score": relevance_score,
                    "gps": metadata_details["gps"],
                    "timestamp": metadata_details["timestamp"]
                }
            }
            
        except Exception as e:
            logger.error(f"Uncaught exception during media verification: {e}", exc_info=True)
            error_result["reasoning"] = f"Internal processing error: {str(e)}"
            return error_result