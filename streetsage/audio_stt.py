"""
Speech-to-Text using faster-whisper (local).
Provides voice input for Q&A.
"""

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from typing import Optional
import logging
import queue
import threading

from config import ENABLE_STT

logger = logging.getLogger(__name__)


class STTEngine:
    """Local STT using faster-whisper."""

    def __init__(self, model_size: str = "base", enabled: bool = ENABLE_STT):
        """
        Initialize STT engine.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            enabled: Whether STT is enabled
        """
        self.enabled = enabled
        self.model = None
        self.is_listening = False

        if self.enabled:
            logger.info(f"Loading Whisper {model_size} model...")
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            logger.info("Whisper model loaded")
        else:
            logger.warning("STT is disabled")

        # Audio recording settings
        self.sample_rate = 16000
        self.channels = 1

    def transcribe_audio(self, audio_data: np.ndarray) -> str:
        """
        Transcribe audio data.

        Args:
            audio_data: Audio samples (float32, -1..1 range)

        Returns:
            Transcribed text
        """
        if not self.enabled or self.model is None:
            logger.warning("STT not available")
            return ""

        try:
            # Whisper expects float32 audio
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Transcribe
            segments, info = self.model.transcribe(audio_data, beam_size=5)

            # Collect text
            text = " ".join([segment.text for segment in segments])
            text = text.strip()

            logger.info(f"Transcribed: {text}")
            return text

        except Exception as e:
            logger.error(f"STT error: {e}")
            return ""

    def record_audio(self, duration_sec: float = 3.0) -> np.ndarray:
        """
        Record audio from microphone.

        Args:
            duration_sec: Recording duration in seconds

        Returns:
            Audio data (float32, mono)
        """
        logger.info(f"Recording for {duration_sec} seconds...")

        try:
            audio = sd.rec(
                int(duration_sec * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32'
            )
            sd.wait()  # Wait for recording to finish

            # Flatten to mono if needed
            if audio.ndim > 1:
                audio = audio[:, 0]

            return audio

        except Exception as e:
            logger.error(f"Recording error: {e}")
            return np.array([], dtype=np.float32)

    def listen_once(self, duration_sec: float = 3.0) -> str:
        """
        Record and transcribe once.

        Args:
            duration_sec: Recording duration

        Returns:
            Transcribed text
        """
        if not self.enabled:
            return ""

        audio = self.record_audio(duration_sec)
        if len(audio) == 0:
            return ""

        return self.transcribe_audio(audio)

    def start_continuous_listening(self, callback, duration_sec: float = 3.0):
        """
        Start continuous listening in background thread.

        Args:
            callback: Function to call with transcribed text
            duration_sec: Duration of each recording chunk
        """
        if not self.enabled:
            logger.warning("STT not enabled, cannot start listening")
            return

        self.is_listening = True

        def listen_loop():
            while self.is_listening:
                text = self.listen_once(duration_sec)
                if text:
                    callback(text)

        thread = threading.Thread(target=listen_loop, daemon=True)
        thread.start()
        logger.info("Started continuous listening")

    def stop_continuous_listening(self):
        """Stop continuous listening."""
        self.is_listening = False
        logger.info("Stopped continuous listening")


class VoiceActivatedRecorder:
    """Voice-activated recording with simple energy-based VAD."""

    def __init__(self, stt_engine: STTEngine, energy_threshold: float = 0.01):
        """
        Initialize voice-activated recorder.

        Args:
            stt_engine: STT engine instance
            energy_threshold: Energy threshold for voice detection
        """
        self.stt_engine = stt_engine
        self.energy_threshold = energy_threshold
        self.is_recording = False
        self.audio_queue = queue.Queue()

    def energy(self, audio: np.ndarray) -> float:
        """Calculate audio energy."""
        return float(np.sqrt(np.mean(audio**2)))

    def record_until_silence(self, max_duration: float = 10.0, silence_duration: float = 1.5) -> str:
        """
        Record until silence is detected.

        Args:
            max_duration: Maximum recording duration
            silence_duration: Duration of silence to stop recording

        Returns:
            Transcribed text
        """
        logger.info("Listening for speech...")

        chunk_duration = 0.5  # 500ms chunks
        chunk_samples = int(chunk_duration * self.stt_engine.sample_rate)

        recorded_chunks = []
        silence_chunks = 0
        max_silence_chunks = int(silence_duration / chunk_duration)
        max_chunks = int(max_duration / chunk_duration)

        try:
            with sd.InputStream(
                samplerate=self.stt_engine.sample_rate,
                channels=1,
                dtype='float32'
            ) as stream:

                for _ in range(max_chunks):
                    chunk, _ = stream.read(chunk_samples)
                    chunk = chunk[:, 0]  # Mono

                    energy = self.energy(chunk)

                    if energy > self.energy_threshold:
                        # Voice detected
                        recorded_chunks.append(chunk)
                        silence_chunks = 0
                    elif len(recorded_chunks) > 0:
                        # Silence after speech
                        recorded_chunks.append(chunk)
                        silence_chunks += 1

                        if silence_chunks >= max_silence_chunks:
                            logger.info("Silence detected, stopping recording")
                            break

            # Concatenate and transcribe
            if len(recorded_chunks) > 0:
                audio = np.concatenate(recorded_chunks)
                return self.stt_engine.transcribe_audio(audio)
            else:
                logger.info("No speech detected")
                return ""

        except Exception as e:
            logger.error(f"Recording error: {e}")
            return ""


def test_stt():
    """Test STT functionality."""
    engine = STTEngine(model_size="base")

    print("\nSTT Test: Say something in 3 seconds...")
    text = engine.listen_once(duration_sec=3.0)
    print(f"You said: {text}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_stt()
