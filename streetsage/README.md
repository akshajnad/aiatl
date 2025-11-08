# StreetSage - Assistive Audio Navigation

**Local, audio-first assistive app for blind/low-vision users**

StreetSage provides real-time navigation assistance using:
- Local computer vision (YOLOv8n + MiDaS)
- Phone/laptop camera as video source
- ElevenLabs text-to-speech for audio guidance
- Local Whisper speech-to-text for voice Q&A
- Snowflake for event logging and analytics

⚠️ **DISCLAIMER**: This is an assistive aid, not a medical device. Always use additional navigation tools and your own judgment.

---

## Features

### Perception
- **Object Detection**: YOLOv8n detects people, vehicles, and obstacles
- **Depth Estimation**: MiDaS provides distance buckets (very near, near, mid, far)
- **Ground Hazards**: OpenCV heuristics detect puddles, uneven surfaces, and cones
- **OCR**: On-demand text reading for signs

### Intelligence
- **Object Tracking**: Tracks objects across frames with velocity estimation
- **Time-to-Collision (TTC)**: Estimates collision risk for moving objects
- **Priority Scoring**: Rule-based system selects most critical hazard
- **Scene Memory**: Maintains 5-second context window for Q&A

### Audio Interface
- **Spoken Instructions**: Concise, actionable guidance (e.g., "Bicycle approaching from your left, about four meters. Pause.")
- **Rate Limiting**: Maximum one instruction every 3 seconds (unless emergency)
- **Voice Q&A**: Ask "how far?", "where?", "what do I do?", "read that sign"

### Data & Analytics
- **Snowflake Logging**: Event summaries for post-hoc analysis
- **Privacy-First**: No raw video leaves the device

---

## Quick Start

### 1. System Dependencies

#### macOS
```bash
# Install Tesseract for OCR
brew install tesseract

# Install portaudio for audio (if needed)
brew install portaudio
```

#### Ubuntu/Linux
```bash
# Install Tesseract
sudo apt-get update
sudo apt-get install -y tesseract-ocr

# Install portaudio
sudo apt-get install -y portaudio19-dev python3-pyaudio
```

#### Windows
```bash
# Download Tesseract installer from:
# https://github.com/UB-Mannheim/tesseract/wiki

# Install and add to PATH
```

### 2. Python Environment

**Requires Python 3.11+**

```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your credentials
nano .env  # or your favorite editor
```

Required credentials:
- **ELEVENLABS_API_KEY**: Get from https://elevenlabs.io
- **SNOWFLAKE_***: Your Snowflake connection details
- **VIDEO_SOURCE**: Camera source (0 for laptop webcam)

### 4. Initialize Database

```bash
# Create Snowflake tables and views
python app.py --init-db
```

### 5. Run the App

```bash
# Basic mode (laptop webcam, audio output only)
python app.py

# With visualization window
python app.py --viz

# With voice Q&A enabled
python app.py --voice-qa

# Custom video source (phone camera)
python app.py --source rtsp://192.168.1.100:8080/h264_ulaw.sdp

# Or use the convenience script
./scripts/run_local.sh --viz --voice-qa
```

---

## Camera Setup

### Option 1: Laptop Webcam (Simplest)

Just use your laptop's built-in camera - no phone needed!

```bash
# Uses default webcam (index 0)
python app.py

# Or explicitly specify
python app.py --source 0
```

**Pros**: No setup, works immediately
**Cons**: Limited mobility compared to phone on lanyard/chest mount

---

### Option 2: Phone Camera - USB Mode (Recommended)

Easiest and most reliable phone camera option:

#### DroidCam (Android & iOS)
1. **Install DroidCam**:
   - Android: [Google Play Store](https://play.google.com/store/apps/details?id=com.dev47apps.droidcam)
   - iOS: [App Store](https://apps.apple.com/app/droidcam-webcam-obs-camera/id1510258102)
2. **Install DroidCam Client** on your computer:
   - Download from [dev47apps.com](https://www.dev47apps.com/)
3. **Connect via USB**:
   - Plug in phone via USB cable
   - Enable "USB Debugging" on Android (Settings → Developer Options)
   - Launch DroidCam on both phone and computer
   - Click "Connect" in DroidCam Client
4. **Find video device**:
   ```bash
   ls /dev/video*  # Linux
   # Or check DroidCam Client for device number
   ```
5. **Use in StreetSage**:
   ```bash
   python app.py --source 1  # Or whichever index DroidCam uses
   ```

**Pros**: Stable connection, works on both Android/iOS, no Wi-Fi needed
**Cons**: Requires USB cable, desktop client installation

---

### Option 3: Phone Camera - Wi-Fi Streaming

If you prefer wireless:

#### Android - IP Webcam (Free)
1. Install **IP Webcam** from [Play Store](https://play.google.com/store/apps/details?id=com.pas.webcam)
2. Open app → **Start Server**
3. Note the IP address shown (e.g., `192.168.1.100:8080`)
4. Use in StreetSage:
   ```bash
   # RTSP (recommended)
   python app.py --source rtsp://192.168.1.100:8080/h264_ulaw.sdp

   # Or MJPEG
   python app.py --source http://192.168.1.100:8080/video
   ```

**Pros**: No cable, free, feature-rich
**Cons**: Requires same Wi-Fi network, can lag

#### iOS - EpocCam (Freemium)
1. Install **EpocCam** from [App Store](https://apps.apple.com/app/epoccam-webcam-for-computer/id435355256)
2. Install **EpocCam Drivers** on your computer:
   - Download from [kinoni.com](https://www.kinoni.com/)
3. Connect via Wi-Fi:
   - Ensure phone and computer on same network
   - Launch EpocCam on phone
   - Phone auto-appears as webcam
4. Find device index:
   ```bash
   ls /dev/video*  # Linux/macOS
   # Or check System Settings → Camera
   ```
5. Use in StreetSage:
   ```bash
   python app.py --source 2  # Or whichever index EpocCam uses
   ```

**Pros**: Easy setup, works wirelessly
**Cons**: Free version has watermark, requires driver installation

---

## Voice Commands

When voice Q&A is enabled (`--voice-qa`), you can ask:

| Command | Example | Response |
|---------|---------|----------|
| **How far?** | "How far is it?" | "The bicycle is about four meters." |
| **Where?** | "Where is the person?" | "The person is on your left." |
| **What to do?** | "What should I do?" | Current instruction or "Continue forward." |
| **Read sign** | "Read that sign" | Reads text from current frame |
| **What is that?** | "What is that?" | Lists detected objects |
| **Repeat** | "Repeat that" | Repeats last instruction |

---

## Testing

### Smoke Tests
```bash
# Run all module tests
python scripts/test_pipeline.py
```

Tests verify:
- ✓ Object detection returns valid detections
- ✓ Depth estimation produces normalized depth maps
- ✓ Ground hazards return scores in 0..1 range
- ✓ Tracking maintains object IDs across frames
- ✓ Rule engine generates valid instructions
- ✓ Scene memory stores and retrieves data
- ✓ OCR can read text (if Tesseract installed)

### Acceptance Test

**Setup**: Laptop webcam pointed at staged scene with:
- Person or object moving left→right
- Shiny surface (puddle proxy) on floor
- Printed text sheet

**Expected Results**:
1. ✓ Hear dynamic warning with side + distance + action
2. ✓ Hear ground warning (puddle/uneven)
3. ✓ Ask "how far is it?" → get distance answer
4. ✓ Say "read the sign" → hear first line from text
5. ✓ Confirm event row in Snowflake with instruction

```sql
-- Check recent events
SELECT * FROM events
ORDER BY ts DESC
LIMIT 10;

-- View top hazards
SELECT * FROM top_hazards
ORDER BY risk_score DESC;
```

---

## Architecture

```
┌─────────────────┐
│ Phone/Laptop    │
│ Camera          │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  PERCEPTION PIPELINE (Local)            │
│  ┌──────────┬──────────┬──────────┐    │
│  │ YOLOv8n  │  MiDaS   │ OpenCV   │    │
│  │ Objects  │  Depth   │ Ground   │    │
│  └──────────┴──────────┴──────────┘    │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  TRACKING & MEMORY                      │
│  • Object tracking (ID persistence)     │
│  • Velocity & TTC estimation            │
│  • 5-second scene memory                │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  RULE ENGINE                            │
│  • Priority scoring                     │
│  • Hazard selection                     │
│  • Instruction generation               │
│  • Rate limiting (3s cadence)           │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  AUDIO OUTPUT (ElevenLabs TTS)          │
│  "Bicycle approaching from your left,   │
│   about four meters. Pause."            │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  SNOWFLAKE (Event Logging)              │
│  • Numeric features only                │
│  • Analytics views                      │
│  • Post-hoc tuning data                 │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  VOICE Q&A (Optional)                   │
│  Whisper STT → Intent Parser            │
│           → Answer from Memory          │
└─────────────────────────────────────────┘
```

---

## Configuration Reference

### Detection Settings
- `FRAME_RATE`: Processing FPS (default: 3)
- `DETECTION_CONFIDENCE`: YOLO confidence threshold (default: 0.35)
- `INSTRUCTION_CADENCE_SEC`: Min time between instructions (default: 3.0)

### Depth Buckets
- `very_near`: >0.75 normalized depth (~0-1m)
- `near`: >0.55 (~1-2m)
- `mid`: >0.35 (~2-4m)
- `far`: ≤0.35 (>4m)

### Ground Hazard Thresholds
- `PUDDLE_THRESHOLD`: Specular pixel ratio (0.02)
- `UNEVEN_THRESHOLD`: Laplacian variance (800)
- `CONE_THRESHOLD`: Orange pixel ratio (0.01)

### Priority Scoring Weights
- `WEIGHT_TTC`: 0.6 (time-to-collision)
- `WEIGHT_DISTANCE`: 0.3 (very near), 0.2 (near), 0.1 (mid)
- `WEIGHT_CONFIDENCE`: 0.2

---

## File Structure

```
streetsage/
├── app.py                    # Main application loop
├── camera_stream.py          # Video source manager
├── detect.py                 # YOLOv8n detection
├── depth.py                  # MiDaS depth estimation
├── ground.py                 # Ground hazard detection
├── track.py                  # Object tracking + TTC
├── rules.py                  # Priority scoring + instruction gen
├── memory.py                 # Scene memory + rate limiting
├── ocr_read.py               # OCR on demand
├── audio_tts.py              # ElevenLabs TTS
├── audio_stt.py              # Whisper STT
├── intents.py                # Voice Q&A intent parsing
├── snowflake_io.py           # Snowflake connection + logging
├── config.py                 # Configuration
├── requirements.txt          # Dependencies
├── .env.example              # Environment template
├── README.md                 # This file
├── sql/
│   ├── create_tables.sql     # Schema
│   └── create_views.sql      # Analytics views
└── scripts/
    ├── run_local.sh          # Convenience launcher
    └── test_pipeline.py      # Smoke tests
```

---

## Algorithms

### Object Detection
- **Model**: YOLOv8n (nano, CPU-friendly)
- **Confidence**: 0.35
- **Classes**: Person, bicycle, car, motorcycle, bus, truck, dog, cat (dynamic)

### Depth Estimation
- **Model**: MiDaS small (monocular depth)
- **Output**: Normalized 0..1 (higher = closer)
- **Bucketing**: ROI median mapped to distance categories

### Ground Hazards

**Puddle/Ice Detection**:
```python
# Specular reflection in HSV
mask = (V > 220) & (S < 40)
puddle_score = clamp(1.8 * pixel_ratio)
```

**Uneven Surface**:
```python
# Edge variance
laplacian = cv2.Laplacian(gray)
uneven_score = clamp(variance / 1200)
```

**Cone/Tape Detection**:
```python
# Orange color in HSV
mask = (5 < H < 25) & (S > 100)
cone_score = clamp(1.5 * pixel_ratio)
```

### Tracking + TTC
- **Tracker**: Centroid-based with velocity estimation
- **TTC**: `max(0.5, 5 - 0.2 * velocity_y)` seconds
- **Emergency**: TTC < 1.5s overrides rate limiting

### Priority Scoring
```python
risk_dynamic = 0.6/TTC + distance_weight + 0.2*confidence
risk_ground = 0.4*puddle + 0.4*uneven + 0.3*cone
```

---

## Snowflake Schema

### Events Table
```sql
CREATE TABLE events (
  event_id STRING,
  ts TIMESTAMP_TZ,
  session_id STRING,
  object_class STRING,       -- 'person','bicycle','puddle',etc.
  side STRING,               -- 'left','center','right'
  distance_bucket STRING,    -- 'very_near','near','mid','far'
  ttc_sec FLOAT,             -- Time to collision (NULL if static)
  confidence FLOAT,
  instruction STRING,        -- Spoken text
  source STRING              -- 'cv','ocr','rule'
);
```

### OCR Text Table
```sql
CREATE TABLE ocr_text (
  event_id STRING,
  ts TIMESTAMP_TZ,
  snippet STRING,
  side STRING,
  distance_bucket STRING
);
```

### Top Hazards View
```sql
CREATE VIEW top_hazards AS
SELECT
  DATE_TRUNC('second', ts) AS bucket_ts,
  object_class, side, distance_bucket,
  MAX(risk_score) AS risk_score,
  ANY_VALUE(instruction) AS example_instruction
FROM events
WHERE ts > DATEADD('minute', -2, CURRENT_TIMESTAMP())
GROUP BY 1,2,3,4
ORDER BY risk_score DESC;
```

---

## Troubleshooting

### Camera Issues
```bash
# Test available cameras
python camera_stream.py

# List video devices (Linux)
ls /dev/video*

# Test with OpenCV
python -c "import cv2; print(cv2.VideoCapture(0).read())"
```

### Audio Issues
```bash
# Test TTS
python audio_tts.py

# Test STT (requires microphone)
python audio_stt.py

# Check audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### Snowflake Connection
```bash
# Test connection
python snowflake_io.py

# Check credentials in .env
cat .env | grep SNOWFLAKE
```

### Model Download
First run will download models:
- YOLOv8n (~6 MB)
- MiDaS small (~100 MB)
- Whisper base (~140 MB)

Models are cached in `~/.cache/` for future runs.

---

## Performance Tuning

### CPU-Friendly Defaults
- YOLOv8n (fastest YOLO variant)
- MiDaS small (vs. DPT_Large)
- 3 FPS processing rate
- Whisper base (vs. large)

### For Better Performance
```python
# In config.py or .env:
FRAME_RATE=1  # Process 1 FPS instead of 3
```

### For Better Accuracy
```python
# Use larger models (slower):
detector = ObjectDetector("yolov8s.pt")  # small instead of nano
estimator = DepthEstimator("DPT_Hybrid")  # hybrid instead of small
stt = STTEngine("small")  # small instead of base
```

---

## Privacy & Security

### Data Minimization
- **No raw video** leaves the device
- **No video recording** to disk
- **Only numeric features** sent to Snowflake:
  - Object class, side, distance bucket
  - TTC, confidence scores
  - Spoken instruction text
  - OCR text snippets (on demand only)

### Network Traffic
- **ElevenLabs**: Text → audio (HTTPS)
- **Snowflake**: Event rows (encrypted connection)
- **No external APIs** for vision (fully local)

### Recommendations
- Use strong Snowflake credentials
- Rotate API keys regularly
- Review Snowflake access logs
- Consider VPN for network streams

---

## Future Enhancements

### Planned (Post-MVP)
- [ ] **Vultr Cloud Relay**: Multi-device event aggregation
- [ ] **Live Dashboard**: Web view of recent hazards
- [ ] **GPS Integration**: Location-aware warnings
- [ ] **Obstacle Depth Fusion**: Combine detection + depth for better distance
- [ ] **Custom Voice Training**: Personalized TTS voices
- [ ] **Offline Mode**: Queue events when network unavailable

### Advanced Features
- [ ] **Semantic Segmentation**: Better ground/obstacle distinction
- [ ] **SLAM**: Build local map for navigation
- [ ] **Multi-Camera**: Stereo vision for true depth
- [ ] **Haptic Feedback**: Vibration patterns for direction
- [ ] **Route Planning**: Integration with navigation apps

---

## Contributing

This is a 1-day hackathon project. Contributions welcome!

**Areas for improvement**:
- Better TTC estimation (currently heuristic-based)
- More robust ground hazard detection
- Multi-language support for TTS/STT
- Performance optimization for mobile deployment
- Clinical validation with blind/low-vision users

---

## License

MIT License - See LICENSE file

---

## Acknowledgments

- **Ultralytics** for YOLOv8
- **Intel ISL** for MiDaS
- **ElevenLabs** for TTS API
- **Guillermo Cámbara** for faster-whisper
- **Snowflake** for analytics platform

---

## Support

For issues and questions:
- GitHub Issues: [Report a bug]
- Documentation: This README
- Examples: See `scripts/test_pipeline.py`

---

**Remember**: StreetSage is an assistive aid, not a medical device. Always use additional navigation tools and your own judgment when navigating.

Stay safe! 🚶‍♂️🦯
