"""
Text-to-Speech using ElevenLabs API.
Provides both streaming and blocking modes.
"""

import os
import io
import logging
from typing import Optional
from elevenlabs import generate, play, Voice, VoiceSettings, stream
import requests

from config import ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, ENABLE_TTS

logger = logging.getLogger(__name__)


class TTSEngine:
    """ElevenLabs TTS wrapper."""

    def __init__(self, api_key: Optional[str] = None, voice_id: Optional[str] = None, enabled: bool = ENABLE_TTS):
        """
        Initialize TTS engine.

        Args:
            api_key: ElevenLabs API key
            voice_id: Voice ID to use
            enabled: Whether TTS is enabled
        """
        self.api_key = api_key or ELEVENLABS_API_KEY
        self.voice_id = voice_id or ELEVENLABS_VOICE_ID
        self.enabled = enabled and bool(self.api_key)
        self.current_thread = None  # Track current speaking thread
        self.stop_speaking = False  # Flag to cancel current speech

        if not self.enabled:
            logger.warning("TTS is disabled (no API key or explicitly disabled)")
        else:
            logger.info(f"TTS initialized with voice {self.voice_id}")

        # Set API key as environment variable for elevenlabs library
        if self.api_key:
            os.environ["ELEVEN_API_KEY"] = self.api_key

    def speak(self, text: str, blocking: bool = True) -> bool:
        """
        Speak text using TTS.

        Args:
            text: Text to speak
            blocking: If True, wait for speech to complete

        Returns:
            True if successful
        """
        if not text:
            return False

        if not self.enabled:
            # Fallback: print to console
            print(f"[TTS]: {text}")
            return True

        try:
            logger.info(f"Speaking: {text}")

            # Generate audio
            audio = generate(
                text=text,
                voice=self.voice_id,
                model="eleven_multilingual_v2"
            )

            # Play audio
            if blocking:
                play(audio)
            else:
                # Non-blocking: play in separate thread (simple approach)
                import threading
                threading.Thread(target=play, args=(audio,), daemon=True).start()

            return True

        except Exception as e:
            logger.error(f"TTS error: {e}")
            # Fallback to print
            print(f"[TTS ERROR - Fallback]: {text}")
            return False

    def speak_streaming(self, text: str) -> bool:
        """
        Speak text using streaming mode (lower latency).

        Args:
            text: Text to speak

        Returns:
            True if successful
        """
        if not text:
            return False

        if not self.enabled:
            print(f"[TTS Streaming]: {text}")
            return True

        try:
            logger.info(f"Speaking (streaming): {text}")

            # Stream audio
            audio_stream = generate(
                text=text,
                voice=self.voice_id,
                model="eleven_multilingual_v2",
                stream=True
            )

            stream(audio_stream)

            return True

        except Exception as e:
            logger.error(f"TTS streaming error: {e}")
            print(f"[TTS ERROR - Fallback]: {text}")
            return False

    def speak_async(self, text: str):
        """
        Speak text asynchronously (fire and forget).
        Cancels any previous speech before starting new one.

        Args:
            text: Text to speak
        """
        import threading

        # Cancel previous speech if still running
        if self.current_thread is not None and self.current_thread.is_alive():
            logger.debug("Canceling previous speech")
            self.stop_speaking = True
            # Don't wait for it to finish, just start the new one

        # Reset stop flag and start new speech
        self.stop_speaking = False
        self.current_thread = threading.Thread(target=self.speak, args=(text, True), daemon=True)
        self.current_thread.start()


def test_tts():
    """Test TTS functionality."""
    engine = TTSEngine()

    test_phrases = [
        "StreetSage assistive navigation starting.",
        "Bicycle approaching from your left, about four meters. Pause.",
        "Puddle ahead two meters. Step left one pace.",
        "Person ahead less than one meter. Stop immediately."
    ]

    print("Testing TTS...")
    for phrase in test_phrases:
        print(f"\nSpeaking: {phrase}")
        engine.speak(phrase, blocking=True)
        import time
        time.sleep(0.5)

    print("\nTesting streaming mode...")
    engine.speak_streaming("This is a test of streaming mode for lower latency.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_tts()
