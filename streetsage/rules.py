"""
Rule engine for priority scoring and instruction generation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from config import (
    WEIGHT_TTC,
    WEIGHT_DISTANCE,
    WEIGHT_CONFIDENCE,
    SIDE_NAMES,
    DISTANCE_PHRASES,
    ACTION_VERBS,
    TTC_EMERGENCY_THRESHOLD
)

logger = logging.getLogger(__name__)


class RuleEngine:
    """Prioritizes hazards and generates actionable instructions."""

    def __init__(self):
        """Initialize rule engine."""
        pass

    def score_dynamic_object(self, obj: Dict, depth_bucket: str) -> float:
        """
        Score a dynamic object (person, vehicle, etc.).

        Args:
            obj: Object dict with ttc, confidence, etc.
            depth_bucket: Distance bucket

        Returns:
            Risk score (0..1+)
        """
        score = 0.0

        # TTC component
        if obj.get("ttc") is not None:
            ttc = obj["ttc"]
            if ttc > 0:
                score += WEIGHT_TTC * (1.0 / ttc)

        # Distance component
        dist_weight = WEIGHT_DISTANCE.get(depth_bucket, 0.0)
        score += dist_weight

        # Confidence component
        conf = obj.get("confidence", 0.0)
        score += WEIGHT_CONFIDENCE * conf

        return score

    def score_ground_hazard(self, hazard_scores: Dict[str, float]) -> float:
        """
        Score ground hazards.

        Args:
            hazard_scores: Dict of {hazard_type: score}

        Returns:
            Combined risk score
        """
        # Weighted sum of ground hazards
        weights = {
            "puddle": 0.4,
            "uneven": 0.4,
            "cone": 0.3
        }

        score = 0.0
        for htype, hscore in hazard_scores.items():
            weight = weights.get(htype, 0.3)
            score += weight * hscore

        return score

    def select_top_hazard(
        self,
        tracked_objects: List[Dict],
        ground_hazards: Dict[str, float],
        depth_map: Optional[np.ndarray] = None
    ) -> Tuple[Optional[Dict], str]:
        """
        Select the highest priority hazard.

        Args:
            tracked_objects: List of tracked object summaries
            ground_hazards: Dict of ground hazard scores
            depth_map: Optional depth map for additional context

        Returns:
            Tuple of (top_hazard_dict, hazard_type)
            hazard_type is "dynamic" or "ground"
        """
        top_score = -1.0
        top_hazard = None
        hazard_type = None

        # Score dynamic objects
        for obj in tracked_objects:
            # Get depth bucket (use from obj or estimate)
            depth_bucket = obj.get("distance_bucket", "unknown")

            score = self.score_dynamic_object(obj, depth_bucket)

            if score > top_score:
                top_score = score
                top_hazard = obj.copy()
                top_hazard["risk_score"] = score
                hazard_type = "dynamic"

        # Score ground hazards
        ground_score = self.score_ground_hazard(ground_hazards)

        if ground_score > top_score:
            top_score = ground_score
            # Find dominant ground hazard
            dominant = max(ground_hazards.items(), key=lambda x: x[1])
            top_hazard = {
                "hazard_type": dominant[0],
                "score": dominant[1],
                "risk_score": ground_score
            }
            hazard_type = "ground"

        return top_hazard, hazard_type

    def generate_instruction(self, hazard: Dict, hazard_type: str) -> str:
        """
        Generate spoken instruction from hazard.

        Args:
            hazard: Hazard dict
            hazard_type: "dynamic" or "ground"

        Returns:
            Spoken instruction string
        """
        if hazard is None:
            return ""

        if hazard_type == "dynamic":
            return self._generate_dynamic_instruction(hazard)
        elif hazard_type == "ground":
            return self._generate_ground_instruction(hazard)
        else:
            return ""

    def _generate_dynamic_instruction(self, obj: Dict) -> str:
        """
        Generate instruction for dynamic object.

        Format: "<Object> <approaching> from your <side>, <distance>. <Action>."

        Example: "Bicycle approaching from your left, about four meters. Pause."
        """
        class_name = obj.get("class_name", "object").capitalize()
        side = obj.get("side", "center")
        distance_bucket = obj.get("distance_bucket", "unknown")
        ttc = obj.get("ttc")

        # Build instruction
        parts = []

        # Object + side
        side_phrase = SIDE_NAMES.get(side, side)
        if side == "center":
            parts.append(f"{class_name} ahead")
        else:
            parts.append(f"{class_name} approaching from your {side_phrase}")

        # Distance
        dist_phrase = DISTANCE_PHRASES.get(distance_bucket, "ahead")
        parts.append(dist_phrase)

        # Action
        action = self._decide_action(obj, distance_bucket, ttc)
        action_phrase = ACTION_VERBS.get(action, "Proceed with caution")

        instruction = f"{', '.join(parts)}. {action_phrase}."

        return instruction

    def _generate_ground_instruction(self, hazard: Dict) -> str:
        """
        Generate instruction for ground hazard.

        Format: "<Hazard> ahead <distance>. <Action>."

        Example: "Puddle ahead two meters, slightly right. Step left one pace."
        """
        hazard_type = hazard.get("hazard_type", "obstacle")
        score = hazard.get("score", 0.0)

        # Map hazard type to friendly name
        hazard_names = {
            "puddle": "Puddle",
            "uneven": "Uneven surface",
            "cone": "Caution marker"
        }
        hazard_name = hazard_names.get(hazard_type, "Obstacle")

        # Estimate distance (ground hazards are typically in lower frame = near)
        if score > 0.5:
            distance = "less than two meters"
        elif score > 0.3:
            distance = "about two meters"
        else:
            distance = "ahead for three meters"

        # Action
        if hazard_type == "puddle":
            action = "Step left one pace"
        elif hazard_type == "uneven":
            action = "Slow down"
        elif hazard_type == "cone":
            action = "Veer slightly right"
        else:
            action = "Proceed with caution"

        instruction = f"{hazard_name} {distance}. {action}."

        return instruction

    def _decide_action(self, obj: Dict, distance_bucket: str, ttc: Optional[float]) -> str:
        """
        Decide action verb based on object properties.

        Args:
            obj: Object dict
            distance_bucket: Distance bucket
            ttc: Time to collision

        Returns:
            Action key
        """
        side = obj.get("side", "center")

        # Emergency stop
        if ttc is not None and ttc < TTC_EMERGENCY_THRESHOLD:
            return "stop"

        # Very near
        if distance_bucket == "very_near":
            if side == "center":
                return "pause"
            elif side == "left":
                return "step_right"
            else:
                return "step_left"

        # Near
        if distance_bucket == "near":
            if side == "center":
                return "slow"
            elif side == "left":
                return "veer_right"
            else:
                return "veer_left"

        # Mid/far
        return "caution"


def create_scene_dict(
    tracked_objects: List[Dict],
    ground_hazards: Dict[str, float],
    depth_map: Optional[np.ndarray] = None
) -> Dict:
    """
    Create a scene dictionary for the rule engine.

    Args:
        tracked_objects: List of tracked object summaries
        ground_hazards: Ground hazard scores
        depth_map: Optional depth map

    Returns:
        Scene dict
    """
    return {
        "tracked_objects": tracked_objects,
        "ground_hazards": ground_hazards,
        "depth_map": depth_map
    }


if __name__ == "__main__":
    # Test rule engine
    logging.basicConfig(level=logging.INFO)

    engine = RuleEngine()

    # Test dynamic object
    test_obj = {
        "class_name": "bicycle",
        "side": "left",
        "distance_bucket": "near",
        "confidence": 0.9,
        "ttc": 2.5
    }

    score = engine.score_dynamic_object(test_obj, "near")
    instruction = engine.generate_instruction(test_obj, "dynamic")

    print(f"Dynamic object score: {score:.2f}")
    print(f"Instruction: {instruction}")

    # Test ground hazard
    test_hazard = {
        "hazard_type": "puddle",
        "score": 0.6,
        "risk_score": 0.5
    }

    instruction = engine.generate_instruction(test_hazard, "ground")
    print(f"\nGround hazard instruction: {instruction}")

    # Test selection
    tracked = [test_obj]
    ground = {"puddle": 0.3, "uneven": 0.2, "cone": 0.0}

    top, htype = engine.select_top_hazard(tracked, ground)
    print(f"\nTop hazard type: {htype}")
    print(f"Top hazard: {top}")

    final_instruction = engine.generate_instruction(top, htype)
    print(f"Final instruction: {final_instruction}")
