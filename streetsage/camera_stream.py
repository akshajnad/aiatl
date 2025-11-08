"""
Camera stream manager for StreetSage.
Supports webcam, RTSP, and MJPEG streams.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CameraStream:
    """Manages video input from various sources."""

    def __init__(self, source):
        """
        Initialize camera stream.

        Args:
            source: int (webcam index), str (RTSP/MJPEG URL), or path to video file
        """
        self.source = source
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count = 0

    def open(self) -> bool:
        """
        Open the video stream.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.source)

            # Set buffer size to 1 for real-time streaming
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {self.source}")
                return False

            # Test read
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.error(f"Failed to read from video source: {self.source}")
                return False

            height, width = frame.shape[:2]
            logger.info(f"Camera opened: {self.source} ({width}x{height})")
            return True

        except Exception as e:
            logger.error(f"Error opening camera: {e}")
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the stream.

        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None

        try:
            ret, frame = self.cap.read()
            if ret:
                self.frame_count += 1
            return ret, frame
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return False, None

    def release(self):
        """Release the video stream."""
        if self.cap is not None:
            self.cap.release()
            logger.info(f"Camera released after {self.frame_count} frames")

    def get_fps(self) -> float:
        """Get the FPS of the video source."""
        if self.cap is not None:
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 0.0

    def get_resolution(self) -> Tuple[int, int]:
        """Get the resolution (width, height) of the video source."""
        if self.cap is not None:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return width, height
        return 0, 0

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


def test_camera_sources():
    """
    Test various camera sources and print which ones work.
    Useful for debugging camera setup.
    """
    print("Testing camera sources...")

    # Test webcam indices 0-2
    for i in range(3):
        print(f"\nTesting webcam {i}...")
        stream = CameraStream(i)
        if stream.open():
            ret, frame = stream.read()
            if ret:
                h, w = frame.shape[:2]
                print(f"  ✓ Webcam {i} works: {w}x{h}")
            stream.release()
        else:
            print(f"  ✗ Webcam {i} not available")

    # Example RTSP/MJPEG URLs (won't work without actual server)
    example_urls = [
        "rtsp://192.168.1.100:8080/h264_ulaw.sdp",
        "http://192.168.1.100:8080/video"
    ]

    print("\n\nExample URLs to try with your phone:")
    print("  Android: Install 'IP Webcam' app")
    print("    - RTSP: rtsp://<phone-ip>:8080/h264_ulaw.sdp")
    print("    - MJPEG: http://<phone-ip>:8080/video")
    print("  iOS: Install 'Iriun Webcam' or similar")
    print("    - Follow app instructions to get URL")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_camera_sources()
