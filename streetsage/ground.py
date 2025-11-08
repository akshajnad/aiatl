"""
Ground hazard detection using OpenCV heuristics.
Detects puddles, uneven surfaces, and traffic cones.
"""

import cv2
import numpy as np
from typing import Dict, Tuple
import logging

from config import (
    GROUND_ROI_RATIO,
    PUDDLE_THRESHOLD,
    UNEVEN_THRESHOLD,
    CONE_THRESHOLD
)

logger = logging.getLogger(__name__)


class GroundHazardDetector:
    """Detects ground-level hazards using OpenCV."""

    def __init__(self):
        """Initialize detector."""
        self.ground_roi_ratio = GROUND_ROI_RATIO

    def get_ground_roi(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract ground ROI (bottom portion of frame).

        Args:
            frame: Input image (BGR)

        Returns:
            Ground ROI
        """
        h, w = frame.shape[:2]
        ground_y = int(h * (1 - self.ground_roi_ratio))
        return frame[ground_y:, :]

    def detect_puddles(self, frame: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Detect puddles/ice using specular reflection heuristic.

        Args:
            frame: Input image (BGR)

        Returns:
            Tuple of (puddle_score, mask)
        """
        # Get ground ROI
        ground_roi = self.get_ground_roi(frame)
        h, w = ground_roi.shape[:2]

        if h == 0 or w == 0:
            return 0.0, np.zeros((1, 1), dtype=np.uint8)

        # Check if frame is mostly dark (invalid/black frame)
        mean_brightness = np.mean(ground_roi)
        if mean_brightness < 10:
            logger.debug(f"Skipping puddle detection on dark frame (brightness: {mean_brightness:.1f})")
            return 0.0, np.zeros(ground_roi.shape[:2], dtype=np.uint8)

        # Convert to HSV
        hsv = cv2.cvtColor(ground_roi, cv2.COLOR_BGR2HSV)

        # Detect specular highlights (high V, low S)
        # Puddles/ice reflect light → bright spots with low saturation
        mask = cv2.inRange(hsv, (0, 0, 220), (180, 40, 255))

        # Calculate ratio of specular pixels
        specular_ratio = np.count_nonzero(mask) / (h * w)

        # Only trigger if we have a reasonable amount of specular pixels
        # Avoid false positives from a few bright pixels in dark frames
        if specular_ratio < PUDDLE_THRESHOLD:
            return 0.0, mask

        # Score: clamp(1.8 * ratio)
        puddle_score = min(1.0, 1.8 * specular_ratio)

        return puddle_score, mask

    def detect_uneven(self, frame: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Detect uneven surfaces using edge variance.

        Args:
            frame: Input image (BGR)

        Returns:
            Tuple of (uneven_score, edges)
        """
        # Get ground ROI
        ground_roi = self.get_ground_roi(frame)

        if ground_roi.shape[0] == 0 or ground_roi.shape[1] == 0:
            return 0.0, np.zeros((1, 1), dtype=np.uint8)

        # Convert to grayscale
        gray = cv2.cvtColor(ground_roi, cv2.COLOR_BGR2GRAY)

        # Compute Laplacian (edge detection)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Calculate variance (uneven surfaces → high edge variance)
        variance = laplacian.var()

        # Score: clamp(variance / 1200)
        uneven_score = min(1.0, variance / 1200.0)

        # Convert for visualization
        edges = np.abs(laplacian).astype(np.uint8)

        return uneven_score, edges

    def detect_cones(self, frame: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Detect traffic cones/caution tape using orange color.

        Args:
            frame: Input image (BGR)

        Returns:
            Tuple of (cone_score, mask)
        """
        # Get ground ROI
        ground_roi = self.get_ground_roi(frame)
        h, w = ground_roi.shape[:2]

        if h == 0 or w == 0:
            return 0.0, np.zeros((1, 1), dtype=np.uint8)

        # Convert to HSV
        hsv = cv2.cvtColor(ground_roi, cv2.COLOR_BGR2HSV)

        # Orange color range (for traffic cones/tape)
        # Orange hue is around 10-25 in OpenCV HSV
        lower_orange = np.array([5, 100, 100])
        upper_orange = np.array([25, 255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # Calculate ratio of orange pixels
        orange_ratio = np.count_nonzero(mask) / (h * w)

        # Score: clamp(1.5 * ratio)
        cone_score = min(1.0, 1.5 * orange_ratio)

        return cone_score, mask

    def detect_all(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Detect all ground hazards.

        Args:
            frame: Input image (BGR)

        Returns:
            Dict with scores for each hazard type
        """
        puddle_score, _ = self.detect_puddles(frame)
        uneven_score, _ = self.detect_uneven(frame)
        cone_score, _ = self.detect_cones(frame)

        return {
            "puddle": puddle_score,
            "uneven": uneven_score,
            "cone": cone_score
        }

    def detect_with_visualization(self, frame: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Detect hazards and create visualization.

        Args:
            frame: Input image (BGR)

        Returns:
            Tuple of (scores_dict, visualization)
        """
        ground_roi = self.get_ground_roi(frame)

        puddle_score, puddle_mask = self.detect_puddles(frame)
        uneven_score, uneven_edges = self.detect_uneven(frame)
        cone_score, cone_mask = self.detect_cones(frame)

        # Create visualization
        h, w = ground_roi.shape[:2]

        # Color the masks
        puddle_vis = cv2.cvtColor(puddle_mask, cv2.COLOR_GRAY2BGR)
        puddle_vis[:, :, 0] = 0  # Remove blue
        puddle_vis[:, :, 1] = puddle_mask  # Green channel

        uneven_vis = cv2.cvtColor(uneven_edges, cv2.COLOR_GRAY2BGR)
        uneven_vis[:, :, 2] = uneven_edges  # Red channel

        cone_vis = cv2.cvtColor(cone_mask, cv2.COLOR_GRAY2BGR)
        cone_vis[:, :, 0] = cone_mask  # Blue channel
        cone_vis[:, :, 1] = cone_mask // 2  # Some green

        # Combine
        combined = cv2.addWeighted(puddle_vis, 0.5, uneven_vis, 0.5, 0)
        combined = cv2.addWeighted(combined, 1.0, cone_vis, 0.5, 0)

        # Overlay on ground ROI
        overlay = ground_roi.copy()
        overlay = cv2.addWeighted(overlay, 0.7, combined, 0.3, 0)

        # Add text
        y_offset = 30
        cv2.putText(overlay, f"Puddle: {puddle_score:.2f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 30
        cv2.putText(overlay, f"Uneven: {uneven_score:.2f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_offset += 30
        cv2.putText(overlay, f"Cone: {cone_score:.2f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)

        scores = {
            "puddle": puddle_score,
            "uneven": uneven_score,
            "cone": cone_score
        }

        return scores, overlay


if __name__ == "__main__":
    # Test with webcam
    logging.basicConfig(level=logging.INFO)

    detector = GroundHazardDetector()
    cap = cv2.VideoCapture(0)

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        scores, vis = detector.detect_with_visualization(frame)

        print(f"Puddle: {scores['puddle']:.2f}, "
              f"Uneven: {scores['uneven']:.2f}, "
              f"Cone: {scores['cone']:.2f}")

        cv2.imshow("Ground Hazards", vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
