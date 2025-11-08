"""
Scene memory ring buffer for maintaining short-term context.
Stores recent detections, hazards, and instructions for Q&A.
"""

import time
from typing import List, Dict, Optional
from collections import deque
import logging

from config import MEMORY_DURATION_SEC

logger = logging.getLogger(__name__)


class SceneMemory:
    """Ring buffer for recent scene information."""

    def __init__(self, duration_sec: float = MEMORY_DURATION_SEC):
        """
        Initialize scene memory.

        Args:
            duration_sec: How long to keep items in memory
        """
        self.duration_sec = duration_sec
        self.items = deque()

    def add_frame(self, timestamp: float, data: Dict):
        """
        Add a frame's data to memory.

        Args:
            timestamp: Unix timestamp
            data: Dict containing frame analysis results
        """
        item = {
            "timestamp": timestamp,
            "data": data
        }
        self.items.append(item)
        self._cleanup(timestamp)

    def _cleanup(self, current_time: float):
        """Remove items older than duration."""
        cutoff = current_time - self.duration_sec
        while self.items and self.items[0]["timestamp"] < cutoff:
            self.items.popleft()

    def get_recent(self, max_age_sec: Optional[float] = None) -> List[Dict]:
        """
        Get recent items.

        Args:
            max_age_sec: Optional max age filter

        Returns:
            List of recent data items
        """
        if max_age_sec is None:
            return [item["data"] for item in self.items]

        current_time = time.time()
        cutoff = current_time - max_age_sec
        return [item["data"] for item in self.items if item["timestamp"] >= cutoff]

    def get_most_recent(self) -> Optional[Dict]:
        """Get the most recent frame data."""
        if self.items:
            return self.items[-1]["data"]
        return None

    def query_objects(self, class_name: Optional[str] = None, side: Optional[str] = None) -> List[Dict]:
        """
        Query objects in memory.

        Args:
            class_name: Filter by class name
            side: Filter by side ("left", "center", "right")

        Returns:
            List of matching objects
        """
        results = []

        for item in self.items:
            data = item["data"]

            # Check tracked objects
            if "tracked_objects" in data:
                for obj in data["tracked_objects"]:
                    match = True
                    if class_name and obj.get("class_name") != class_name:
                        match = False
                    if side and obj.get("side") != side:
                        match = False

                    if match:
                        results.append(obj)

        return results

    def query_hazards(self, hazard_type: Optional[str] = None) -> List[Dict]:
        """
        Query ground hazards in memory.

        Args:
            hazard_type: Filter by type ("puddle", "uneven", "cone")

        Returns:
            List of matching hazards
        """
        results = []

        for item in self.items:
            data = item["data"]

            if "ground_hazards" in data:
                hazards = data["ground_hazards"]
                if hazard_type:
                    if hazard_type in hazards:
                        results.append({
                            "type": hazard_type,
                            "score": hazards[hazard_type],
                            "timestamp": item["timestamp"]
                        })
                else:
                    for htype, score in hazards.items():
                        results.append({
                            "type": htype,
                            "score": score,
                            "timestamp": item["timestamp"]
                        })

        return results

    def get_last_instruction(self) -> Optional[str]:
        """Get the most recent instruction that was spoken."""
        for item in reversed(self.items):
            data = item["data"]
            if "instruction" in data and data["instruction"]:
                return data["instruction"]
        return None

    def clear(self):
        """Clear all memory."""
        self.items.clear()

    def size(self) -> int:
        """Get number of items in memory."""
        return len(self.items)


class InstructionRateLimiter:
    """Rate limiter for spoken instructions with object-based deduplication."""

    def __init__(self, min_interval_sec: float = 3.0, emergency_threshold_ttc: float = 1.5, emergency_min_interval_sec: float = 0.5):
        """
        Initialize rate limiter.

        Args:
            min_interval_sec: Minimum time between instructions
            emergency_threshold_ttc: TTC threshold for emergency override
            emergency_min_interval_sec: Minimum time between emergency instructions (prevents every-frame alerts)
        """
        self.min_interval_sec = min_interval_sec
        self.emergency_threshold_ttc = emergency_threshold_ttc
        self.emergency_min_interval_sec = emergency_min_interval_sec
        self.last_instruction_time = 0.0
        self.last_alerted_hazard = None  # Track what we last alerted about
        self.last_hazard_type = None  # Track whether it was dynamic or ground

    def should_alert(self, current_time: float, hazard: Optional[Dict], hazard_type: str, ttc: Optional[float] = None) -> bool:
        """
        Check if we should speak about this hazard.

        Args:
            current_time: Current timestamp
            hazard: The hazard dict (object or ground hazard)
            hazard_type: "dynamic" or "ground"
            ttc: Optional TTC for emergency override

        Returns:
            True if we should alert about this hazard
        """
        if hazard is None:
            # No hazard - reset tracking
            if self.last_alerted_hazard is not None:
                logger.debug("No hazard detected, resetting alert tracking")
                self.last_alerted_hazard = None
                self.last_hazard_type = None
            return False

        # Check if this is the same hazard we already alerted about
        if self._is_same_hazard(hazard, hazard_type):
            logger.debug("Same hazard as before, suppressing repeated alert")
            return False

        # Different hazard - check time-based rate limiting
        elapsed = current_time - self.last_instruction_time

        # Emergency override - use shorter interval but still throttle
        if ttc is not None and ttc < self.emergency_threshold_ttc:
            return elapsed >= self.emergency_min_interval_sec

        # Normal rate limiting
        return elapsed >= self.min_interval_sec

    def _is_same_hazard(self, hazard: Dict, hazard_type: str) -> bool:
        """
        Check if this is the same hazard we alerted about before.

        Args:
            hazard: Current hazard dict
            hazard_type: "dynamic" or "ground"

        Returns:
            True if same hazard
        """
        if self.last_alerted_hazard is None or self.last_hazard_type != hazard_type:
            return False

        if hazard_type == "dynamic":
            # For dynamic objects, compare class name and approximate position
            return (
                hazard.get("class_name") == self.last_alerted_hazard.get("class_name") and
                hazard.get("side") == self.last_alerted_hazard.get("side") and
                hazard.get("distance_bucket") == self.last_alerted_hazard.get("distance_bucket")
            )
        else:
            # For ground hazards, just check if we've already alerted (ground hazards don't move)
            return True

    def mark_spoken(self, current_time: float, hazard: Optional[Dict] = None, hazard_type: Optional[str] = None):
        """
        Mark that an instruction was spoken.

        Args:
            current_time: Current timestamp
            hazard: The hazard we alerted about
            hazard_type: "dynamic" or "ground"
        """
        self.last_instruction_time = current_time
        self.last_alerted_hazard = hazard
        self.last_hazard_type = hazard_type
        logger.debug(f"Marked alert for {hazard_type} hazard: {hazard.get('class_name') if hazard else 'none'}")

    def reset(self):
        """Reset the limiter."""
        self.last_instruction_time = 0.0
        self.last_alerted_hazard = None
        self.last_hazard_type = None


if __name__ == "__main__":
    # Test scene memory
    logging.basicConfig(level=logging.INFO)

    memory = SceneMemory(duration_sec=3.0)

    # Add some test data
    current_time = time.time()

    for i in range(5):
        data = {
            "tracked_objects": [
                {"class_name": "person", "side": "left", "distance_bucket": "near"},
                {"class_name": "car", "side": "right", "distance_bucket": "far"},
            ],
            "ground_hazards": {
                "puddle": 0.3,
                "uneven": 0.1,
                "cone": 0.0
            },
            "instruction": f"Instruction {i}"
        }
        memory.add_frame(current_time + i * 0.5, data)

    print(f"Memory size: {memory.size()}")
    print(f"Recent objects: {len(memory.query_objects())}")
    print(f"Recent hazards: {len(memory.query_hazards())}")
    print(f"Last instruction: {memory.get_last_instruction()}")

    # Test rate limiter
    limiter = InstructionRateLimiter(min_interval_sec=2.0)

    t = time.time()
    print(f"\nRate limiter test:")
    print(f"Can speak now? {limiter.can_speak(t)}")
    limiter.mark_spoken(t)
    print(f"Can speak after 1s? {limiter.can_speak(t + 1.0)}")
    print(f"Can speak after 2s? {limiter.can_speak(t + 2.0)}")
    print(f"Can speak with emergency (TTC=1s)? {limiter.can_speak(t + 0.5, ttc=1.0)}")
