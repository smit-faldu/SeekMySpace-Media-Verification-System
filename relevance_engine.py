import asyncio
from typing import Tuple, Optional
import numpy as np
from PIL import Image
from transformers import pipeline

from logger import get_logger

logger = get_logger(__name__)

class RelevanceEngine:
    def __init__(self):
        try:
            logger.info("Loading local CLIP zero-shot classifier...")
            self.classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")
            
            self.high_relevance_labels = [
                "professional corporate office interior", "modern co-working space",
                "conference room with table and chairs", "commercial retail store interior",
                "industrial warehouse interior"
            ]
            self.moderate_relevance_labels = [
                "empty commercial real estate", "building lobby or reception area",
                "office cubicles", "restaurant or cafe interior"
            ]
            self.negative_labels = [
                "residential living room", "messy bedroom", "home kitchen or bathroom",
                "outdoor nature or park", "street view with cars", 
                "close-up portrait of a person", "screenshot of text or document", "selfie photograph"
            ]
            self.target_labels = self.high_relevance_labels + self.moderate_relevance_labels + self.negative_labels
        except Exception as e:
            logger.error(f"Failed to load CLIP classifier: {e}")
            self.classifier = None

    async def evaluate_relevance(self, frame: Optional[np.ndarray]) -> Tuple[float, str]:
        def _evaluate():
            if frame is None or frame.size == 0: return 0.0, "Invalid or empty frame."
            if self.classifier is None: return 0.0, "Relevance model failed to load."

            try:
                pil_img = Image.fromarray(frame)
                results = self.classifier(pil_img, candidate_labels=self.target_labels)
                
                top_label = results[0]['label']
                top_score = results[0]['score']
                
                if top_label in self.high_relevance_labels:
                    score = min(60.0, 35.0 + (top_score * 25.0))
                    reasoning = f"High relevance: '{top_label}' ({top_score:.2f} conf)."
                elif top_label in self.moderate_relevance_labels:
                    score = min(40.0, 15.0 + (top_score * 25.0))
                    reasoning = f"Moderate relevance: '{top_label}' ({top_score:.2f} conf)."
                elif top_label in self.negative_labels:
                    score = 0.0 if top_score > 0.4 else min(10.0, 10.0 - (top_score * 20.0))
                    reasoning = f"Low relevance (Rejected): '{top_label}' ({top_score:.2f} conf)."
                else:
                    score = 10.0
                    reasoning = f"Unknown classification status for '{top_label}'."
                
                return round(score, 1), reasoning
            except Exception as e:
                return 0.0, f"Error calculating relevance: {str(e)}"
                
        return await asyncio.to_thread(_evaluate)