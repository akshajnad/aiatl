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
        # If source is an integer (webcam), try to auto-detect working camera
        if isinstance(self.source, int):
            return self._open_webcam_with_fallback(self.source)
        else:
            return self._open_source(self.source)

    def _open_source(self, source) -> bool:
        """
        Open a specific video source.

        Args:
            source: Video source to open

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Attempting to open camera source: {source}")
            self.cap = cv2.VideoCapture(source)

            # Set buffer size to 1 for real-time streaming
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {source}")
                if self.cap:
                    self.cap.release()
                self.cap = None
                return False

            # Test read multiple frames to ensure camera is working
            for attempt in range(3):
                ret, frame = self.cap.read()
                if ret and frame is not None and self._is_valid_frame(frame):
                    height, width = frame.shape[:2]
                    logger.info(f"✓ Camera opened successfully: {source} ({width}x{height})")
                    self.source = source  # Update source to the working one
                    return True
                if attempt < 2:
                    import time
                    time.sleep(0.1)

            logger.error(f"Failed to read valid frames from video source: {source}")
            if self.cap:
                self.cap.release()
            self.cap = None
            return False

        except Exception as e:
            logger.error(f"Error opening camera: {e}")
            if self.cap:
                self.cap.release()
            self.cap = None
            return False

    def _open_webcam_with_fallback(self, preferred_index: int) -> bool:
        """
        Try to open webcam with fallback to other indices.

        Args:
            preferred_index: Preferred camera index

        Returns:
            True if any camera opened successfully
        """
        # Try preferred index first
        if self._open_source(preferred_index):
            return True

        # If preferred fails, try other common indices
        logger.warning(f"Camera {preferred_index} failed, trying other indices...")
        for index in range(0, 5):
            if index == preferred_index:
                continue
            logger.info(f"Trying camera index {index}...")
            if self._open_source(index):
                logger.info(f"Using camera {index} instead of {preferred_index}")
                return True

        logger.error("No working camera found!")
        return False

    def _is_valid_frame(self, frame: np.ndarray) -> bool:
        """
        Check if a frame is valid (not black/empty).

        Args:
            frame: Frame to validate

        Returns:
            True if frame appears valid
        """
        if frame is None or frame.size == 0:
            return False

        # Check if frame is mostly black (mean brightness very low)
        mean_brightness = np.mean(frame)
        if mean_brightness < 5:  # Almost completely black
            logger.warning(f"Frame appears invalid (mean brightness: {mean_brightness:.1f})")
            return False

        # Check if frame has any variation (not a solid color)
        std_dev = np.std(frame)
        if std_dev < 1:  # No variation
            logger.warning(f"Frame appears invalid (std dev: {std_dev:.1f})")
            return False

        return True

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
            if ret and frame is not None:
                # Validate frame quality
                if not self._is_valid_frame(frame):
                    logger.warning("Read invalid frame from camera")
                    return False, None
                self.frame_count += 1
                return True, frame
            return False, None
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

    print("\n\nExample phone camera options:")
    print("  Easiest: Use DroidCam (USB mode)")
    print("    - Works on both Android & iOS")
    print("    - Install app + desktop client from dev47apps.com")
    print("    - Connect via USB, appears as webcam device")
    print("\n  Wi-Fi Streaming:")
    print("    Android: Install 'IP Webcam' app")
    print("      - RTSP: rtsp://<phone-ip>:8080/h264_ulaw.sdp")
    print("      - MJPEG: http://<phone-ip>:8080/video")
    print("    iOS: Install 'EpocCam' app")
    print("      - Install drivers, phone appears as webcam")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_camera_sources()
