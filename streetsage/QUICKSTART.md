# StreetSage - Quick Start Guide

**Get running in 5 minutes!**

This guide will walk you through everything you need to install and run StreetSage.

---

## What is StreetSage?

StreetSage is a **local assistive audio navigation app** for blind and low-vision users. It uses your camera to detect obstacles, estimates distances, and speaks warnings through your speakers.

**Example output**: *"Bicycle approaching from your left, about four meters. Pause."*

---

## Prerequisites

### 1. Check Python Version

You need **Python 3.11 or higher**.

```bash
python3 --version
```

If you don't have Python 3.11+, download it from [python.org](https://www.python.org/downloads/).

### 2. Install System Dependencies

#### macOS
```bash
brew install tesseract portaudio
```

#### Ubuntu/Linux
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr portaudio19-dev python3-pyaudio
```

#### Windows
1. Download Tesseract installer from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install and add to PATH

---

## Installation Steps

### Step 1: Navigate to Project Directory

```bash
cd streetsage
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv .venv

# Activate it
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

You should see `(.venv)` in your terminal prompt.

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

This will take a few minutes. It installs:
- Computer vision libraries (OpenCV, YOLOv8, MiDaS)
- Audio libraries (ElevenLabs, Whisper, sounddevice)
- AI libraries (Google Gemini)
- Other utilities

**First run note**: When you run the app for the first time, it will download AI models (~250 MB total):
- YOLOv8n (~6 MB) - object detection
- MiDaS small (~100 MB) - depth estimation
- Whisper base (~140 MB) - speech recognition

These are cached locally and won't need to be downloaded again.

### Step 4: Get API Keys

You need **two free API keys**:

#### ElevenLabs (for Text-to-Speech)
1. Go to [elevenlabs.io](https://elevenlabs.io)
2. Sign up for a free account
3. Go to [Profile → API Keys](https://elevenlabs.io/app/settings/api-keys)
4. Copy your API key

#### Google Gemini (for Voice Q&A)
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with Google account
3. Click **"Get API Key"** → **"Create API key"**
4. Copy your API key

### Step 5: Configure Environment

```bash
# Copy the example config file
cp .env.example .env

# Edit the file
nano .env
```

Add your API keys:
```
ELEVENLABS_API_KEY=your_elevenlabs_key_here
GEMINI_API_KEY=your_gemini_key_here
VIDEO_SOURCE=0
```

**Save and exit** (Ctrl+O, Enter, Ctrl+X in nano).

---

## Running the App

### Basic Mode (Recommended for First Run)

```bash
python app.py --viz
```

This runs with:
- Your laptop's built-in webcam
- Visual window showing detections (so you can see what's happening)
- Audio warnings spoken through speakers

### What You Should See/Hear

1. **Window opens** showing your webcam feed
2. **Colored boxes** appear around detected objects (people, vehicles, etc.)
3. **Audio warnings** play when objects are detected
   - Example: *"Person on your left, very near. Stop."*

### Other Running Options

```bash
# Without visualization (audio only)
python app.py

# With voice Q&A enabled (ask questions)
python app.py --viz --voice-qa

# Use phone camera (see Camera Setup below)
python app.py --source rtsp://192.168.1.100:8080/h264_ulaw.sdp

# Using convenience script
./scripts/run_local.sh --viz --voice-qa
```

### Keyboard Controls

When running:
- **Press 'q'** to quit
- **Press 'r'** to read text (OCR) when `--viz` is enabled
- Speak questions when `--voice-qa` is enabled

---

## Camera Setup Options

### Option 1: Laptop Webcam (Easiest)

**No setup needed!** Just run:
```bash
python app.py --viz
```

### Option 2: Phone Camera via USB

**Best for mobility** - wear phone on lanyard pointing forward.

#### Using DroidCam (Android/iOS)
1. Install **DroidCam** app on phone:
   - [Android - Play Store](https://play.google.com/store/apps/details?id=com.dev47apps.droidcam)
   - [iOS - App Store](https://apps.apple.com/app/droidcam-webcam-obs-camera/id1510258102)

2. Install **DroidCam Client** on computer:
   - Download from [dev47apps.com](https://www.dev47apps.com/)

3. Connect phone via USB cable

4. Enable USB debugging on Android:
   - Go to Settings → About Phone → Tap "Build Number" 7 times
   - Go to Settings → Developer Options → Enable "USB Debugging"

5. Launch DroidCam on phone and computer → Click **Connect**

6. Find the video device number:
   ```bash
   ls /dev/video*
   # Note the highest number (e.g., /dev/video2)
   ```

7. Run StreetSage:
   ```bash
   python app.py --source 2 --viz
   ```

### Option 3: Phone Camera via Wi-Fi

#### Android - IP Webcam
1. Install [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam)
2. Open app → **Start Server**
3. Note the IP address (e.g., `192.168.1.100:8080`)
4. Run StreetSage:
   ```bash
   python app.py --source rtsp://192.168.1.100:8080/h264_ulaw.sdp --viz
   ```

#### iOS - EpocCam
1. Install [EpocCam](https://apps.apple.com/app/epoccam-webcam-for-computer/id435355256)
2. Install EpocCam Drivers on computer from [kinoni.com](https://www.kinoni.com/)
3. Connect phone to same Wi-Fi as computer
4. Launch EpocCam on phone
5. Find device number:
   ```bash
   ls /dev/video*
   ```
6. Run StreetSage:
   ```bash
   python app.py --source 2 --viz
   ```

---

## Voice Q&A Commands

When running with `--voice-qa`, you can ask:

| You Say | StreetSage Says |
|---------|----------------|
| "How far is it?" | "The bicycle is about four meters." |
| "Where is the person?" | "The person is on your left." |
| "What should I do?" | Repeats current instruction or "Continue forward." |
| "Read that sign" | Reads text from current view |
| "What is that?" | Lists all detected objects |
| "Is it safe?" | Assesses current hazards |

**Powered by Google Gemini** - understands natural language questions!

---

## Testing

Verify everything works:

```bash
# Run smoke tests
python scripts/test_pipeline.py
```

Expected output:
```
✓ Object detection working
✓ Depth estimation working
✓ Ground hazard detection working
✓ Object tracking working
✓ Rule engine working
✓ Scene memory working
✓ OCR working
```

---

## Troubleshooting

### Problem: "No module named 'cv2'"

**Solution**: Activate your virtual environment:
```bash
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

### Problem: "Camera not found" or black screen

**Solution**: Test camera access:
```bash
# List available cameras (Linux/macOS)
ls /dev/video*

# Test camera 0
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.read())"
```

Try different source numbers:
```bash
python app.py --source 0 --viz
python app.py --source 1 --viz
python app.py --source 2 --viz
```

### Problem: "ElevenLabs API key invalid"

**Solution**: Check your `.env` file:
```bash
cat .env | grep ELEVENLABS
```

Make sure:
- No extra spaces around the `=`
- No quotes around the key
- Key is copied correctly from ElevenLabs dashboard

### Problem: Audio not playing

**Solution**: Test audio:
```bash
# Check available audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Test TTS directly
python audio_tts.py
```

Make sure:
- Speakers/headphones are connected
- Volume is not muted
- Default audio device is set correctly

### Problem: "pytesseract not found"

**Solution**: Install Tesseract:
```bash
# macOS
brew install tesseract

# Ubuntu/Linux
sudo apt-get install tesseract-ocr

# Windows - download installer from:
# https://github.com/UB-Mannheim/tesseract/wiki
```

### Problem: Slow performance / lag

**Solution**: Reduce frame rate in `.env`:
```
FRAME_RATE=1  # Process 1 frame per second instead of 3
```

Or run without visualization:
```bash
python app.py  # No --viz flag
```

### Problem: "CUDA not available" warning

**This is normal!** The app is designed to run on CPU. The warning can be ignored.

To suppress it, you can set:
```bash
export CUDA_VISIBLE_DEVICES=""
```

---

## What Happens on First Run

1. **Model downloads** (~250 MB):
   - YOLOv8n for object detection
   - MiDaS for depth estimation
   - Whisper for speech recognition

   These are cached in `~/.cache/` and won't download again.

2. **Camera initialization**: May take 5-10 seconds

3. **First detection**: May take a few seconds to warm up

After first run, startup is much faster!

---

## Performance Tips

### For Faster Processing
- Use laptop webcam instead of phone (less latency)
- Lower frame rate: `FRAME_RATE=1` in `.env`
- Disable visualization: run without `--viz`

### For Better Accuracy
- Use phone camera with better lens
- Ensure good lighting
- Point camera at chest height, slightly downward
- Keep camera stable (mount on lanyard/chest strap)

---

## Configuration Quick Reference

Edit `.env` to customize:

```bash
# Required
ELEVENLABS_API_KEY=your_key
GEMINI_API_KEY=your_key

# Video source
VIDEO_SOURCE=0                    # 0=webcam, 1+=other cameras

# Performance
FRAME_RATE=3                      # Frames per second (1-5)
DETECTION_CONFIDENCE=0.35         # YOLO confidence (0.0-1.0)

# Audio
INSTRUCTION_CADENCE_SEC=3.0       # Min seconds between warnings
```

---

## Next Steps

1. **Try different cameras**: Experiment with laptop webcam vs phone
2. **Test voice Q&A**: Run with `--voice-qa` and ask questions
3. **Read the full README**: See `README.md` for architecture details
4. **Customize settings**: Adjust `.env` for your preferences

---

## File Structure Reference

```
streetsage/
├── app.py                    # ← Main app (start here)
├── requirements.txt          # ← Dependencies
├── .env.example              # ← Config template
├── .env                      # ← Your config (create this)
├── README.md                 # ← Full documentation
├── QUICKSTART.md             # ← This file
│
├── detect.py                 # Object detection (YOLOv8)
├── depth.py                  # Depth estimation (MiDaS)
├── ground.py                 # Ground hazard detection
├── track.py                  # Object tracking
├── rules.py                  # Decision engine
├── memory.py                 # Scene memory
│
├── audio_tts.py              # Text-to-speech (ElevenLabs)
├── audio_stt.py              # Speech-to-text (Whisper)
├── gemini_ai.py              # AI Q&A (Gemini)
├── intents.py                # Voice command handling
├── ocr_read.py               # Text reading
│
└── scripts/
    ├── run_local.sh          # Convenience launcher
    └── test_pipeline.py      # Smoke tests
```

---

## Privacy & Safety

**What StreetSage does:**
- ✓ Processes video **locally** on your device
- ✓ Only sends **text** to external APIs (ElevenLabs, Gemini)
- ✓ Does **not record** or store video
- ✓ Does **not send** raw video anywhere

**External API usage:**
- **ElevenLabs**: Text → Speech (HTTPS)
- **Google Gemini**: Scene descriptions + questions → Answers (HTTPS)

**Important disclaimer**: StreetSage is an **assistive aid**, not a medical device. Always use additional navigation tools and your own judgment.

---

## Getting Help

### Check the logs
When running, StreetSage prints status messages. If something goes wrong, read the error messages carefully.

### Common commands for debugging
```bash
# Test camera
python camera_stream.py

# Test audio
python audio_tts.py

# Test Gemini
python gemini_ai.py

# Run full tests
python scripts/test_pipeline.py
```

### Still stuck?
1. Check you activated the virtual environment (`(.venv)` in prompt)
2. Verify API keys in `.env`
3. Try a different camera source number
4. Check the detailed README.md

---

## Summary Checklist

- [ ] Python 3.11+ installed
- [ ] System dependencies installed (Tesseract, PortAudio)
- [ ] Virtual environment created and activated
- [ ] Requirements installed (`pip install -r requirements.txt`)
- [ ] API keys obtained (ElevenLabs, Gemini)
- [ ] `.env` file configured
- [ ] Camera tested (`python app.py --viz`)
- [ ] Audio working (hearing spoken warnings)

**All checked?** You're ready to use StreetSage! 🎉

---

**Happy navigating! Stay safe!** 🚶‍♂️🦯
