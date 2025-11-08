"""
Monocular depth estimation using MiDaS small.
"""

import cv2
import numpy as np
import torch
from typing import Tuple, List
import logging

from config import DEPTH_BUCKETS

logger = logging.getLogger(__name__)


class DepthEstimator:
    """MiDaS small depth estimation wrapper."""

    def __init__(self, model_type: str = "DPT_Hybrid"):
        """
        Initialize depth estimator.

        Args:
            model_type: MiDaS model type (DPT_Hybrid, DPT_Large, MiDaS_small)
        """
        logger.info(f"Loading MiDaS {model_type} model...")

        # Load MiDaS model
        self.model = torch.hub.load("intel-isl/MiDaS", model_type)

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

        logger.info(f"MiDaS loaded on {self.device}")

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth map from frame.

        Args:
            frame: Input image (BGR)

        Returns:
            Normalized depth map (0..1, where higher = closer)
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Transform
        input_batch = self.transform(img_rgb).to(self.device)

        # Predict
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # Normalize to 0..1 (invert so closer = higher value)
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 0:
            depth_normalized = 1.0 - (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth_map)

        return depth_normalized

    def get_roi_depth(self, depth_map: np.ndarray, bbox: List[float]) -> float:
        """
        Get median depth value for a bounding box ROI.

        Args:
            depth_map: Normalized depth map
            bbox: [x1, y1, x2, y2]

        Returns:
            Median depth value in ROI
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = depth_map.shape

        # Clamp to image bounds
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return 0.0

        roi = depth_map[y1:y2, x1:x2]
        return float(np.median(roi))

    def depth_to_bucket(self, depth_value: float) -> str:
        """
        Convert depth value to distance bucket.

        Args:
            depth_value: Normalized depth (0..1)

        Returns:
            Distance bucket name
        """
        if depth_value > DEPTH_BUCKETS["very_near"]:
            return "very_near"
        elif depth_value > DEPTH_BUCKETS["near"]:
            return "near"
        elif depth_value > DEPTH_BUCKETS["mid"]:
            return "mid"
        else:
            return "far"

    def visualize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Create colorized depth visualization.

        Args:
            depth_map: Normalized depth map

        Returns:
            Colorized depth map (BGR)
        """
        depth_colored = cv2.applyColorMap(
            (depth_map * 255).astype(np.uint8),
            cv2.COLORMAP_MAGMA
        )
        return depth_colored


def get_ground_plane_depth(depth_map: np.ndarray, roi_ratio: float = 0.35) -> float:
    """
    Get median depth of ground plane (bottom portion of frame).

    Args:
        depth_map: Normalized depth map
        roi_ratio: Ratio of frame height to consider as ground

    Returns:
        Median depth of ground plane
    """
    h, w = depth_map.shape
    ground_y = int(h * (1 - roi_ratio))
    ground_roi = depth_map[ground_y:, :]
    return float(np.median(ground_roi))


if __name__ == "__main__":
    # Test with webcam
    logging.basicConfig(level=logging.INFO)

    estimator = DepthEstimator(model_type="MiDaS_small")
    cap = cv2.VideoCapture(0)

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Estimate depth
        depth = estimator.estimate(frame)
        depth_vis = estimator.visualize_depth(depth)

        # Get center depth
        h, w = frame.shape[:2]
        center_depth = estimator.get_roi_depth(depth, [w//2 - 50, h//2 - 50, w//2 + 50, h//2 + 50])
        bucket = estimator.depth_to_bucket(center_depth)

        # Display
        combined = np.hstack([frame, depth_vis])
        cv2.putText(combined, f"Center: {bucket} ({center_depth:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Depth", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
