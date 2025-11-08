#!/usr/bin/env python3
"""
StreetSage - Assistive Audio Navigation App

Main application loop:
  Camera → Perception → Tracking → Rules → Speech → Logging
"""

import cv2
import numpy as np
import time
import logging
import argparse
from pathlib import Path

# Import our modules
from camera_stream import CameraStream
from detect import ObjectDetector, get_detection_side
from depth import DepthEstimator
from ground import GroundHazardDetector
from track import ObjectTracker, get_tracked_objects_summary
from rules import RuleEngine
from memory import SceneMemory, InstructionRateLimiter
from ocr_read import OCRReader
from audio_tts import TTSEngine
from audio_stt import STTEngine
from intents import IntentHandler
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class StreetSageApp:
    """Main application class."""

    def __init__(self, video_source=None, enable_viz: bool = False, enable_voice_qa: bool = False):
        """
        Initialize StreetSage app.

        Args:
            video_source: Video source (int or URL)
            enable_viz: Enable visualization window
            enable_voice_qa: Enable voice Q&A
        """
        logger.info("Initializing StreetSage...")

        # Configuration
        self.video_source = video_source if video_source is not None else config.VIDEO_SOURCE
        self.enable_viz = enable_viz
        self.enable_voice_qa = enable_voice_qa
        self.frame_rate = config.FRAME_RATE
        self.frame_interval = 1.0 / self.frame_rate

        # Initialize components
        logger.info("Loading perception models...")
        self.detector = ObjectDetector()
        self.depth_estimator = DepthEstimator(model_type="MiDaS_small")
        self.ground_detector = GroundHazardDetector()
        self.ocr_reader = OCRReader()

        logger.info("Initializing tracking and decision making...")
        self.tracker = ObjectTracker()
        self.rule_engine = RuleEngine()
        self.memory = SceneMemory()
        self.rate_limiter = InstructionRateLimiter(
            min_interval_sec=config.INSTRUCTION_CADENCE_SEC,
            emergency_threshold_ttc=config.TTC_EMERGENCY_THRESHOLD,
            emergency_min_interval_sec=config.EMERGENCY_INSTRUCTION_CADENCE_SEC
        )

        logger.info("Initializing audio I/O...")
        self.tts = TTSEngine()
        self.stt = STTEngine(model_size="base") if enable_voice_qa else None
        self.intent_handler = IntentHandler(self.memory) if enable_voice_qa else None

        # State
        self.camera = None
        self.current_frame = None
        self.running = False

        logger.info("StreetSage initialized successfully")

    def start(self):
        """Start the application."""
        logger.info("Starting StreetSage...")

        # Speak disclaimer
        self.tts.speak(config.STARTUP_DISCLAIMER)

        # Open camera
        self.camera = CameraStream(self.video_source)
        if not self.camera.open():
            logger.error("Failed to open camera")
            self.tts.speak("Failed to open camera. Please check your camera connection.")
            logger.error("\n" + "="*60)
            logger.error("CAMERA TROUBLESHOOTING")
            logger.error("="*60)
            logger.error("1. Run 'python test_camera.py' to diagnose camera issues")
            logger.error("2. Try specifying a different camera: python app.py --source 1 --viz")
            logger.error("3. Check if another application is using the camera")
            logger.error("4. For Iriun/DroidCam, ensure the app is running and connected")
            logger.error("="*60 + "\n")
            return

        frame_width, frame_height = self.camera.get_resolution()
        logger.info(f"Camera resolution: {frame_width}x{frame_height}")

        # Start voice Q&A if enabled
        if self.enable_voice_qa and self.stt:
            logger.info("Voice Q&A enabled - listening for commands")
            self.stt.start_continuous_listening(self._handle_voice_command, duration_sec=3.0)

        # Main loop
        self.running = True
        self._main_loop()

    def _main_loop(self):
        """Main processing loop."""
        last_frame_time = 0.0
        consecutive_failures = 0
        max_consecutive_failures = 30  # Exit after 30 failed reads

        try:
            while self.running:
                current_time = time.time()

                # Frame rate limiting
                if current_time - last_frame_time < self.frame_interval:
                    time.sleep(0.01)
                    continue

                last_frame_time = current_time

                # Read frame
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    logger.warning(f"Failed to read frame ({consecutive_failures}/{max_consecutive_failures})")

                    if consecutive_failures >= max_consecutive_failures:
                        logger.error("Too many consecutive frame read failures. Camera may be disconnected.")
                        self.tts.speak("Camera connection lost. Exiting.")
                        self.running = False
                        break

                    time.sleep(0.1)
                    continue

                # Reset failure counter on successful read
                consecutive_failures = 0
                self.current_frame = frame

                # Process frame
                self._process_frame(frame, current_time)

                # Visualization
                if self.enable_viz:
                    cv2.imshow("StreetSage", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("User requested quit")
                        self.running = False
                    elif key == ord('r'):
                        # Manual OCR read
                        self._read_sign()

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self._cleanup()

    def _process_frame(self, frame: np.ndarray, timestamp: float):
        """
        Process a single frame.

        Args:
            frame: Input frame (BGR)
            timestamp: Current timestamp
        """
        h, w = frame.shape[:2]

        # 1. Object detection
        detections = self.detector.detect(frame)

        # 2. Depth estimation
        depth_map = self.depth_estimator.estimate(frame)

        # Add depth bucket to detections
        for det in detections:
            roi_depth = self.depth_estimator.get_roi_depth(depth_map, det["bbox"])
            det["distance_bucket"] = self.depth_estimator.depth_to_bucket(roi_depth)
            det["depth_value"] = roi_depth

        # 3. Ground hazard detection
        ground_hazards = self.ground_detector.detect_all(frame)

        # 4. Tracking + TTC
        tracked_objects = self.tracker.update(detections)
        tracked_summaries = get_tracked_objects_summary(tracked_objects, w, h)

        # Add depth buckets to tracked summaries
        for summary in tracked_summaries:
            # Find matching detection for depth
            for det in detections:
                if abs(summary["center"][0] - det["center"][0]) < 10:
                    summary["distance_bucket"] = det.get("distance_bucket", "unknown")
                    break

        # 5. Rule engine - select top hazard
        top_hazard, hazard_type = self.rule_engine.select_top_hazard(
            tracked_summaries,
            ground_hazards,
            depth_map
        )

        # 6. Generate instruction
        instruction = ""
        if top_hazard is not None:
            ttc = top_hazard.get("ttc") if hazard_type == "dynamic" else None

            if self.rate_limiter.can_speak(timestamp, ttc):
                instruction = self.rule_engine.generate_instruction(top_hazard, hazard_type)

                if instruction:
                    logger.info(f"Instruction: {instruction}")
                    self.tts.speak_async(instruction)
                    self.rate_limiter.mark_spoken(timestamp)

        # 7. Update scene memory
        scene_data = {
            "tracked_objects": tracked_summaries,
            "ground_hazards": ground_hazards,
            "instruction": instruction,
            "timestamp": timestamp
        }
        self.memory.add_frame(timestamp, scene_data)

    def _handle_voice_command(self, text: str):
        """
        Handle voice command from STT.

        Args:
            text: Transcribed text
        """
        if not text or not self.intent_handler:
            return

        logger.info(f"Voice command: {text}")

        # Handle intent
        response = self.intent_handler.handle(
            text,
            current_frame=self.current_frame,
            ocr_reader=self.ocr_reader
        )

        if response:
            logger.info(f"Response: {response}")
            self.tts.speak(response)

    def _read_sign(self):
        """Manually trigger sign reading."""
        if self.current_frame is not None:
            from ocr_read import read_sign_on_demand
            text = read_sign_on_demand(self.current_frame, self.ocr_reader)

            if text:
                response = f"It says: {text}"
                logger.info(response)
                self.tts.speak(response)
            else:
                self.tts.speak("I couldn't read any text.")

    def _cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")

        self.running = False

        if self.stt and self.enable_voice_qa:
            self.stt.stop_continuous_listening()

        if self.camera:
            self.camera.release()

        if self.enable_viz:
            cv2.destroyAllWindows()

        logger.info("Shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="StreetSage Assistive Audio Navigation")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video source (0 for webcam, or RTSP/MJPEG URL)"
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Enable visualization window"
    )
    parser.add_argument(
        "--voice-qa",
        action="store_true",
        help="Enable voice Q&A (requires microphone)"
    )

    args = parser.parse_args()

    # Convert source to int if it's a number
    source = args.source
    if source is not None:
        try:
            source = int(source)
        except ValueError:
            pass  # Keep as string URL

    # Create and run app
    app = StreetSageApp(
        video_source=source,
        enable_viz=args.viz,
        enable_voice_qa=args.voice_qa
    )

    app.start()


if __name__ == "__main__":
    main()
