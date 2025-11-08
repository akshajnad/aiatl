"""
Object detection using YOLOv8n.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple
import logging

from config import DETECTION_CONFIDENCE, DYNAMIC_CLASSES, STATIC_OBSTACLES

logger = logging.getLogger(__name__)


class ObjectDetector:
    """YOLOv8n object detection wrapper."""

    def __init__(self, model_size: str = "yolov8n.pt", confidence: float = DETECTION_CONFIDENCE):
        """
        Initialize detector.

        Args:
            model_size: YOLOv8 model size (yolov8n, yolov8s, etc.)
            confidence: Confidence threshold
        """
        self.confidence = confidence
        logger.info(f"Loading {model_size} model...")
        self.model = YOLO(model_size)
        logger.info("Model loaded successfully")

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in frame.

        Args:
            frame: Input image (BGR)

        Returns:
            List of detections, each dict containing:
                - class_name: str
                - confidence: float
                - bbox: [x1, y1, x2, y2]
                - center: (cx, cy)
                - is_dynamic: bool
        """
        results = self.model(frame, conf=self.confidence, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract box data
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[cls_id]

                # Calculate center
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # Determine if dynamic
                is_dynamic = class_name in DYNAMIC_CLASSES

                detection = {
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "center": (float(cx), float(cy)),
                    "is_dynamic": is_dynamic
                }
                detections.append(detection)

        return detections

    def detect_with_visualization(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        Detect and draw bounding boxes on frame.

        Args:
            frame: Input image (BGR)

        Returns:
            Tuple of (detections, annotated_frame)
        """
        detections = self.detect(frame)
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = (0, 255, 0) if det["is_dynamic"] else (255, 0, 0)
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(annotated, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return detections, annotated


def get_detection_side(center_x: float, frame_width: int) -> str:
    """
    Determine which side of the frame an object is on.

    Args:
        center_x: X coordinate of object center
        frame_width: Width of frame

    Returns:
        "left", "center", or "right"
    """
    third = frame_width / 3
    if center_x < third:
        return "left"
    elif center_x < 2 * third:
        return "center"
    else:
        return "right"


def get_detection_roi(bbox: List[float], padding: int = 10) -> Tuple[int, int, int, int]:
    """
    Get padded ROI for a detection.

    Args:
        bbox: [x1, y1, x2, y2]
        padding: Padding in pixels

    Returns:
        (x1, y1, x2, y2) with padding
    """
    x1, y1, x2, y2 = bbox
    return (
        max(0, int(x1) - padding),
        max(0, int(y1) - padding),
        int(x2) + padding,
        int(y2) + padding
    )


if __name__ == "__main__":
    # Test with webcam
    logging.basicConfig(level=logging.INFO)

    detector = ObjectDetector()
    cap = cv2.VideoCapture(0)

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections, annotated = detector.detect_with_visualization(frame)

        # Print detections
        for det in detections:
            side = get_detection_side(det["center"][0], frame.shape[1])
            print(f"{det['class_name']} ({det['confidence']:.2f}) - {side}")

        cv2.imshow("Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
