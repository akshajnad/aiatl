"""
Configuration module for StreetSage.
Loads settings from .env and provides constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# ElevenLabs
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")

# Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCSP0hOyo1nl4F3Zs7yQEVLNaPer-WjR2A")

# Video Source
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")
# Convert to int if it's a number
try:
    VIDEO_SOURCE = int(VIDEO_SOURCE)
except ValueError:
    pass  # Keep as string (RTSP/MJPEG URL)

# Processing Settings
FRAME_RATE = float(os.getenv("FRAME_RATE", "3"))
DETECTION_CONFIDENCE = float(os.getenv("DETECTION_CONFIDENCE", "0.35"))
INSTRUCTION_CADENCE_SEC = float(os.getenv("INSTRUCTION_CADENCE_SEC", "3.0"))

# Audio Settings
ENABLE_TTS = os.getenv("ENABLE_TTS", "true").lower() == "true"
ENABLE_STT = os.getenv("ENABLE_STT", "true").lower() == "true"

# Detection Classes
# COCO classes we care about for navigation
DYNAMIC_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "dog", "cat"}
STATIC_OBSTACLES = {"chair", "bench", "potted plant", "fire hydrant", "stop sign"}

# Depth Bucketing Thresholds (normalized MiDaS output 0..1)
DEPTH_BUCKETS = {
    "very_near": 0.75,  # >0.75
    "near": 0.55,       # >0.55
    "mid": 0.35,        # >0.35
    "far": 0.0          # <=0.35
}

# Ground Hazard Thresholds
GROUND_ROI_RATIO = 0.35  # Bottom 35% of frame
PUDDLE_THRESHOLD = 0.02  # Specular pixel ratio
UNEVEN_THRESHOLD = 800   # Laplacian variance
CONE_THRESHOLD = 0.01    # Orange pixel ratio

# Tracking
MAX_TRACKING_AGE = 5       # frames
IOU_THRESHOLD = 0.3        # for matching
MIN_VELOCITY_THRESHOLD = 2.0  # pixels/frame to consider "moving"

# TTC (Time To Collision) Constants
TTC_BASE = 5.0             # base seconds
TTC_VELOCITY_FACTOR = 0.2  # scaling factor
TTC_MIN = 0.5              # minimum TTC in seconds
TTC_EMERGENCY_THRESHOLD = 1.5  # emergency if TTC < this

# Priority Scoring Weights
WEIGHT_TTC = 0.6
WEIGHT_DISTANCE = {
    "very_near": 0.3,
    "near": 0.2,
    "mid": 0.1,
    "far": 0.05
}
WEIGHT_CONFIDENCE = 0.2

# Instruction Templates
SIDE_NAMES = {
    "left": "left",
    "center": "center",
    "right": "right"
}

DISTANCE_PHRASES = {
    "very_near": "less than one meter",
    "near": "about two meters",
    "mid": "about four meters",
    "far": "far ahead"
}

ACTION_VERBS = {
    "stop": "Stop immediately",
    "pause": "Pause",
    "slow": "Slow down",
    "veer_left": "Veer left",
    "veer_right": "Veer right",
    "step_left": "Step left one pace",
    "step_right": "Step right one pace",
    "caution": "Proceed with caution"
}

# Scene Memory
MEMORY_DURATION_SEC = 5.0  # Keep last 5 seconds

# Disclaimer
STARTUP_DISCLAIMER = (
    "StreetSage assistive navigation starting. "
    "This is an aid, not a medical device. "
    "Always use additional navigation tools and your own judgment."
)
