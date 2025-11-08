#!/usr/bin/env python3
"""
Simple camera test script to diagnose webcam issues.
"""

import cv2
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_camera_index(index):
    """Test a specific camera index."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing camera index {index}...")
    logger.info(f"{'='*60}")

    cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        logger.error(f"✗ Camera {index} could not be opened")
        return False

    # Try to read a frame
    ret, frame = cap.read()

    if not ret or frame is None:
        logger.error(f"✗ Camera {index} opened but cannot read frames")
        cap.release()
        return False

    # Check frame properties
    height, width = frame.shape[:2]
    mean_brightness = frame.mean()

    logger.info(f"✓ Camera {index} is working!")
    logger.info(f"  Resolution: {width}x{height}")
    logger.info(f"  Mean brightness: {mean_brightness:.1f}")
    logger.info(f"  Frame shape: {frame.shape}")

    # Show the frame
    cv2.imshow(f"Camera {index} Test", frame)
    logger.info(f"  Press any key to close preview...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cap.release()
    return True


def main():
    """Test all common camera indices."""
    print("\n" + "="*60)
    print("StreetSage Camera Diagnostic Tool")
    print("="*60)
    print("\nThis will test camera indices 0-4 to find working cameras.")
    print("If you see a preview window, the camera is working.\n")

    working_cameras = []

    for i in range(5):
        if test_camera_index(i):
            working_cameras.append(i)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if working_cameras:
        print(f"\n✓ Found {len(working_cameras)} working camera(s): {working_cameras}")
        print(f"\nTo use camera {working_cameras[0]}, run:")
        print(f"  python app.py --source {working_cameras[0]} --viz")
    else:
        print("\n✗ No working cameras found!")
        print("\nTroubleshooting tips:")
        print("  1. Check if another app is using the camera")
        print("  2. Check camera permissions")
        print("  3. Try connecting an external webcam")
        print("  4. For Iriun/DroidCam: Make sure the app is running and connected")

    print()


if __name__ == "__main__":
    main()
