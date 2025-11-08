"""
Object tracking with Time-To-Collision (TTC) estimation.
Uses simple centroid-based tracking with velocity estimation.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.spatial import distance
import logging

from config import (
    MAX_TRACKING_AGE,
    IOU_THRESHOLD,
    MIN_VELOCITY_THRESHOLD,
    TTC_BASE,
    TTC_VELOCITY_FACTOR,
    TTC_MIN
)

logger = logging.getLogger(__name__)


class TrackedObject:
    """Represents a tracked object with history."""

    def __init__(self, object_id: int, detection: Dict, frame_num: int):
        """
        Initialize tracked object.

        Args:
            object_id: Unique ID
            detection: Detection dict from detector
            frame_num: Current frame number
        """
        self.id = object_id
        self.class_name = detection["class_name"]
        self.is_dynamic = detection["is_dynamic"]

        # Position history (centers)
        self.centers = [detection["center"]]
        self.bboxes = [detection["bbox"]]
        self.confidences = [detection["confidence"]]
        self.frame_nums = [frame_num]

        # Tracking metadata
        self.age = 0
        self.hits = 1
        self.time_since_update = 0

    def update(self, detection: Dict, frame_num: int):
        """Update with new detection."""
        self.centers.append(detection["center"])
        self.bboxes.append(detection["bbox"])
        self.confidences.append(detection["confidence"])
        self.frame_nums.append(frame_num)

        # Keep only recent history (last 10 frames)
        if len(self.centers) > 10:
            self.centers.pop(0)
            self.bboxes.pop(0)
            self.confidences.pop(0)
            self.frame_nums.pop(0)

        self.time_since_update = 0
        self.hits += 1

    def predict(self):
        """Predict next position (simple: assume constant velocity)."""
        self.age += 1
        self.time_since_update += 1

    def get_current_center(self) -> Tuple[float, float]:
        """Get most recent center."""
        return self.centers[-1]

    def get_current_bbox(self) -> List[float]:
        """Get most recent bbox."""
        return self.bboxes[-1]

    def get_velocity(self) -> Tuple[float, float]:
        """
        Estimate velocity in pixels/frame.

        Returns:
            (vx, vy) velocity vector
        """
        if len(self.centers) < 2:
            return 0.0, 0.0

        # Use last 3 points for smoothing if available
        n = min(3, len(self.centers))
        recent_centers = self.centers[-n:]

        # Linear regression for velocity
        cx_values = [c[0] for c in recent_centers]
        cy_values = [c[1] for c in recent_centers]

        if n == 2:
            vx = cx_values[1] - cx_values[0]
            vy = cy_values[1] - cy_values[0]
        else:
            # Average velocity over recent frames
            vx = (cx_values[-1] - cx_values[0]) / (n - 1)
            vy = (cy_values[-1] - cy_values[0]) / (n - 1)

        return vx, vy

    def is_moving(self) -> bool:
        """Check if object is moving."""
        vx, vy = self.get_velocity()
        speed = np.sqrt(vx**2 + vy**2)
        return speed > MIN_VELOCITY_THRESHOLD


class ObjectTracker:
    """Multi-object tracker."""

    def __init__(self):
        """Initialize tracker."""
        self.next_id = 0
        self.tracked_objects: List[TrackedObject] = []
        self.frame_num = 0

    def update(self, detections: List[Dict]) -> List[TrackedObject]:
        """
        Update tracker with new detections.

        Args:
            detections: List of detection dicts

        Returns:
            List of active tracked objects
        """
        self.frame_num += 1

        # Predict all tracked objects
        for obj in self.tracked_objects:
            obj.predict()

        # Match detections to tracked objects
        if len(self.tracked_objects) == 0:
            # No existing tracks, create new ones
            for det in detections:
                self._create_track(det)
        elif len(detections) == 0:
            # No detections, just age existing tracks
            pass
        else:
            # Match using Hungarian algorithm (simplified: greedy matching by distance)
            matched, unmatched_dets, unmatched_tracks = self._match_detections(detections)

            # Update matched tracks
            for det_idx, track_idx in matched:
                self.tracked_objects[track_idx].update(detections[det_idx], self.frame_num)

            # Create new tracks for unmatched detections
            for det_idx in unmatched_dets:
                self._create_track(detections[det_idx])

        # Remove old tracks
        self.tracked_objects = [
            obj for obj in self.tracked_objects
            if obj.time_since_update < MAX_TRACKING_AGE
        ]

        return self.tracked_objects

    def _create_track(self, detection: Dict):
        """Create new track from detection."""
        new_obj = TrackedObject(self.next_id, detection, self.frame_num)
        self.tracked_objects.append(new_obj)
        self.next_id += 1

    def _match_detections(self, detections: List[Dict]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to tracks using distance metric.

        Returns:
            Tuple of (matched_pairs, unmatched_det_indices, unmatched_track_indices)
        """
        if len(self.tracked_objects) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(range(len(self.tracked_objects)))

        # Compute distance matrix
        track_centers = np.array([obj.get_current_center() for obj in self.tracked_objects])
        det_centers = np.array([det["center"] for det in detections])

        dist_matrix = distance.cdist(track_centers, det_centers, metric='euclidean')

        # Greedy matching (simple approach)
        matched_pairs = []
        unmatched_dets = set(range(len(detections)))
        unmatched_tracks = set(range(len(self.tracked_objects)))

        # Sort by distance
        rows, cols = np.where(dist_matrix < 100)  # Max distance threshold
        distances = dist_matrix[rows, cols]
        sorted_indices = np.argsort(distances)

        for idx in sorted_indices:
            track_idx = rows[idx]
            det_idx = cols[idx]

            if track_idx in unmatched_tracks and det_idx in unmatched_dets:
                matched_pairs.append((det_idx, track_idx))
                unmatched_tracks.remove(track_idx)
                unmatched_dets.remove(det_idx)

        return matched_pairs, list(unmatched_dets), list(unmatched_tracks)


def estimate_ttc(tracked_obj: TrackedObject, frame_height: int, fps: float = 3.0) -> Optional[float]:
    """
    Estimate Time-To-Collision for a tracked object.

    Args:
        tracked_obj: TrackedObject instance
        frame_height: Frame height in pixels
        fps: Frames per second

    Returns:
        TTC in seconds, or None if not applicable
    """
    if not tracked_obj.is_dynamic or not tracked_obj.is_moving():
        return None

    vx, vy = tracked_obj.get_velocity()

    # Check if moving toward camera (positive vy in image coords = moving down = approaching)
    if vy <= 0:
        return None  # Moving away or sideways

    # Simple heuristic: TTC = base - velocity_factor * vy
    # Higher velocity → lower TTC
    ttc = TTC_BASE - TTC_VELOCITY_FACTOR * abs(vy)
    ttc = max(TTC_MIN, ttc)

    return ttc


def get_tracked_objects_summary(tracked_objects: List[TrackedObject], frame_width: int, frame_height: int) -> List[Dict]:
    """
    Create summary of tracked objects for decision making.

    Args:
        tracked_objects: List of TrackedObject instances
        frame_width: Frame width
        frame_height: Frame height

    Returns:
        List of summary dicts
    """
    summaries = []

    for obj in tracked_objects:
        cx, cy = obj.get_current_center()

        # Determine side
        third = frame_width / 3
        if cx < third:
            side = "left"
        elif cx < 2 * third:
            side = "center"
        else:
            side = "right"

        # Get velocity and TTC
        vx, vy = obj.get_velocity()
        ttc = estimate_ttc(obj, frame_height)

        summary = {
            "id": obj.id,
            "class_name": obj.class_name,
            "is_dynamic": obj.is_dynamic,
            "center": (cx, cy),
            "bbox": obj.get_current_bbox(),
            "confidence": obj.confidences[-1],
            "side": side,
            "velocity": (vx, vy),
            "is_moving": obj.is_moving(),
            "ttc": ttc
        }

        summaries.append(summary)

    return summaries


if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)

    tracker = ObjectTracker()

    # Simulate some detections
    test_detections = [
        [
            {"class_name": "person", "confidence": 0.9, "bbox": [100, 100, 200, 300],
             "center": (150, 200), "is_dynamic": True},
        ],
        [
            {"class_name": "person", "confidence": 0.9, "bbox": [105, 110, 205, 310],
             "center": (155, 210), "is_dynamic": True},
        ],
        [
            {"class_name": "person", "confidence": 0.9, "bbox": [110, 120, 210, 320],
             "center": (160, 220), "is_dynamic": True},
        ],
    ]

    for frame_num, dets in enumerate(test_detections):
        tracked = tracker.update(dets)
        print(f"\nFrame {frame_num}:")
        for obj in tracked:
            vx, vy = obj.get_velocity()
            print(f"  ID {obj.id}: {obj.class_name} at {obj.get_current_center()}, "
                  f"velocity=({vx:.1f}, {vy:.1f}), moving={obj.is_moving()}")
