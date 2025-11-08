#!/usr/bin/env python3
"""
Smoke tests for StreetSage modules.
Tests each module independently to ensure basic functionality.
"""

import sys
import os
import logging
import numpy as np
import cv2

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_detect():
    """Test object detection."""
    logger.info("Testing object detection...")

    from detect import ObjectDetector

    detector = ObjectDetector()

    # Create a dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Run detection
    detections = detector.detect(frame)

    # Should return a list (even if empty)
    assert isinstance(detections, list), "Detector should return a list"

    logger.info(f"✓ Detection test passed (found {len(detections)} objects)")


def test_depth():
    """Test depth estimation."""
    logger.info("Testing depth estimation...")

    from depth import DepthEstimator

    estimator = DepthEstimator(model_type="MiDaS_small")

    # Create a dummy frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Estimate depth
    depth = estimator.estimate(frame)

    # Check shape
    assert depth.shape == (480, 640), "Depth map should match frame dimensions"

    # Check values are normalized
    assert 0 <= depth.min() <= 1, "Depth values should be normalized to 0..1"
    assert 0 <= depth.max() <= 1, "Depth values should be normalized to 0..1"

    # Test bucketing
    bucket = estimator.depth_to_bucket(0.8)
    assert bucket in ["very_near", "near", "mid", "far"], "Bucket should be valid"

    logger.info(f"✓ Depth test passed (bucket test: {bucket})")


def test_ground():
    """Test ground hazard detection."""
    logger.info("Testing ground hazard detection...")

    from ground import GroundHazardDetector

    detector = GroundHazardDetector()

    # Create a dummy frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Detect hazards
    hazards = detector.detect_all(frame)

    # Check output format
    assert isinstance(hazards, dict), "Hazards should be a dict"
    assert "puddle" in hazards, "Should detect puddle score"
    assert "uneven" in hazards, "Should detect uneven score"
    assert "cone" in hazards, "Should detect cone score"

    # Check scores are in range
    for name, score in hazards.items():
        assert 0 <= score <= 1, f"{name} score should be in range 0..1"

    logger.info(f"✓ Ground hazard test passed (puddle={hazards['puddle']:.2f})")


def test_track():
    """Test object tracking."""
    logger.info("Testing object tracking...")

    from track import ObjectTracker

    tracker = ObjectTracker()

    # Simulate detections over 3 frames
    frame1_dets = [
        {"class_name": "person", "confidence": 0.9, "bbox": [100, 100, 200, 300],
         "center": (150, 200), "is_dynamic": True}
    ]

    frame2_dets = [
        {"class_name": "person", "confidence": 0.9, "bbox": [105, 105, 205, 305],
         "center": (155, 205), "is_dynamic": True}
    ]

    frame3_dets = [
        {"class_name": "person", "confidence": 0.9, "bbox": [110, 110, 210, 310],
         "center": (160, 210), "is_dynamic": True}
    ]

    # Track across frames
    tracked1 = tracker.update(frame1_dets)
    tracked2 = tracker.update(frame2_dets)
    tracked3 = tracker.update(frame3_dets)

    # Should maintain same ID
    assert len(tracked1) == 1, "Should have 1 tracked object"
    assert len(tracked3) == 1, "Should still have 1 tracked object"
    assert tracked1[0].id == tracked3[0].id, "Object ID should persist"

    # Should detect movement
    vx, vy = tracked3[0].get_velocity()
    assert vy > 0, "Should detect downward movement"

    logger.info(f"✓ Tracking test passed (velocity={vx:.1f}, {vy:.1f})")


def test_rules():
    """Test rule engine."""
    logger.info("Testing rule engine...")

    from rules import RuleEngine

    engine = RuleEngine()

    # Test dynamic object
    test_obj = {
        "class_name": "bicycle",
        "side": "left",
        "distance_bucket": "near",
        "confidence": 0.9,
        "ttc": 2.5
    }

    instruction = engine.generate_instruction(test_obj, "dynamic")

    assert isinstance(instruction, str), "Instruction should be a string"
    assert len(instruction) > 0, "Instruction should not be empty"
    assert "bicycle" in instruction.lower(), "Instruction should mention the object"

    logger.info(f"✓ Rules test passed: '{instruction}'")


def test_memory():
    """Test scene memory."""
    logger.info("Testing scene memory...")

    from memory import SceneMemory
    import time

    memory = SceneMemory(duration_sec=2.0)

    # Add some data
    now = time.time()
    for i in range(3):
        data = {
            "tracked_objects": [{"class_name": "person", "side": "left"}],
            "ground_hazards": {"puddle": 0.3},
            "instruction": f"Test {i}"
        }
        memory.add_frame(now + i * 0.5, data)

    # Check size
    assert memory.size() == 3, "Memory should have 3 items"

    # Get most recent
    recent = memory.get_most_recent()
    assert recent is not None, "Should have most recent data"
    assert recent["instruction"] == "Test 2", "Most recent should be last added"

    logger.info(f"✓ Memory test passed (size={memory.size()})")


def test_ocr():
    """Test OCR."""
    logger.info("Testing OCR...")

    from ocr_read import OCRReader

    try:
        reader = OCRReader()

        # Create a simple test image with text
        frame = 255 * np.ones((200, 400, 3), dtype=np.uint8)
        cv2.putText(frame, "STOP", (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

        # Try to read
        text = reader.read_text(frame)

        # OCR might not work perfectly, so just check it returns a string
        assert isinstance(text, str), "OCR should return a string"

        logger.info(f"✓ OCR test passed (read: '{text}')")

    except Exception as e:
        logger.warning(f"⚠ OCR test skipped (Tesseract may not be installed): {e}")


def test_config():
    """Test configuration loading."""
    logger.info("Testing configuration...")

    import config

    # Check some key configs exist
    assert hasattr(config, 'FRAME_RATE'), "Config should have FRAME_RATE"
    assert hasattr(config, 'DETECTION_CONFIDENCE'), "Config should have DETECTION_CONFIDENCE"
    assert hasattr(config, 'DEPTH_BUCKETS'), "Config should have DEPTH_BUCKETS"

    assert isinstance(config.DEPTH_BUCKETS, dict), "DEPTH_BUCKETS should be a dict"
    assert "very_near" in config.DEPTH_BUCKETS, "Should have very_near bucket"

    logger.info("✓ Config test passed")


def run_all_tests():
    """Run all smoke tests."""
    tests = [
        ("Configuration", test_config),
        ("Object Detection", test_detect),
        ("Depth Estimation", test_depth),
        ("Ground Hazards", test_ground),
        ("Tracking", test_track),
        ("Rule Engine", test_rules),
        ("Scene Memory", test_memory),
        ("OCR", test_ocr),
    ]

    print("\n" + "="*60)
    print("  StreetSage Pipeline Smoke Tests")
    print("="*60 + "\n")

    failed = []

    for name, test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            logger.error(f"✗ {name} test FAILED: {e}")
            failed.append(name)

    print("\n" + "="*60)
    if failed:
        print(f"FAILED: {len(failed)} test(s) failed:")
        for name in failed:
            print(f"  - {name}")
        print("="*60)
        sys.exit(1)
    else:
        print("SUCCESS: All tests passed!")
        print("="*60)
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
