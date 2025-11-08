"""
Simple intent parser for voice Q&A.
No LLM dependency - uses keyword matching.
"""

import re
from typing import Optional, Dict
import logging

from config import DISTANCE_PHRASES, ACTION_VERBS

logger = logging.getLogger(__name__)


class IntentParser:
    """Parse voice commands and questions."""

    def __init__(self):
        """Initialize parser."""
        # Intent patterns (simple keyword matching)
        self.patterns = {
            "how_far": [
                r"\bhow far\b",
                r"\bhow close\b",
                r"\bdistance\b",
                r"\bhow many (meters|feet)\b"
            ],
            "where": [
                r"\bwhere\b",
                r"\bwhich side\b",
                r"\bwhat side\b",
                r"\blocation\b"
            ],
            "what_do": [
                r"\bwhat (do|should) (i|we) do\b",
                r"\bwhat action\b",
                r"\bwhat now\b",
                r"\bwhat next\b"
            ],
            "read_sign": [
                r"\bread\b.*\b(sign|text|label)\b",
                r"\bwhat (does|do) (it|that|the sign) say\b",
                r"\bwhat('s| is) written\b"
            ],
            "what_is": [
                r"\bwhat (is|are)\b",
                r"\bwhat('s| is) that\b",
                r"\bidentify\b"
            ],
            "repeat": [
                r"\brepeat\b",
                r"\bsay (that )?again\b",
                r"\bwhat (did you say|was that)\b"
            ]
        }

    def parse(self, text: str) -> Optional[str]:
        """
        Parse text to determine intent.

        Args:
            text: User's spoken text

        Returns:
            Intent name or None
        """
        if not text:
            return None

        text_lower = text.lower()

        # Check each pattern
        for intent, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    logger.info(f"Matched intent: {intent} from '{text}'")
                    return intent

        logger.info(f"No intent matched for: '{text}'")
        return None


class IntentHandler:
    """Handle intents using scene memory."""

    def __init__(self, scene_memory):
        """
        Initialize handler.

        Args:
            scene_memory: SceneMemory instance
        """
        self.memory = scene_memory
        self.parser = IntentParser()

    def handle(self, text: str, current_frame=None, ocr_reader=None) -> str:
        """
        Handle a voice command.

        Args:
            text: User's spoken text
            current_frame: Optional current frame for OCR
            ocr_reader: Optional OCR reader instance

        Returns:
            Response text to speak
        """
        intent = self.parser.parse(text)

        if intent is None:
            return "I didn't understand that. Try asking how far, where, what to do, or read the sign."

        if intent == "how_far":
            return self._handle_how_far()
        elif intent == "where":
            return self._handle_where()
        elif intent == "what_do":
            return self._handle_what_do()
        elif intent == "read_sign":
            return self._handle_read_sign(current_frame, ocr_reader)
        elif intent == "what_is":
            return self._handle_what_is()
        elif intent == "repeat":
            return self._handle_repeat()
        else:
            return "I can't help with that yet."

    def _handle_how_far(self) -> str:
        """Handle 'how far' question."""
        # Get most recent scene
        recent = self.memory.get_most_recent()
        if not recent:
            return "I haven't detected anything yet."

        # Look for tracked objects
        objects = recent.get("tracked_objects", [])
        if not objects:
            return "Nothing detected nearby right now."

        # Get closest object
        closest = min(objects, key=lambda x: self._distance_rank(x.get("distance_bucket", "far")))

        distance_bucket = closest.get("distance_bucket", "unknown")
        distance_phrase = DISTANCE_PHRASES.get(distance_bucket, "some distance ahead")

        class_name = closest.get("class_name", "object")

        return f"The {class_name} is {distance_phrase}."

    def _handle_where(self) -> str:
        """Handle 'where' question."""
        recent = self.memory.get_most_recent()
        if not recent:
            return "I haven't detected anything yet."

        objects = recent.get("tracked_objects", [])
        if not objects:
            return "Nothing detected nearby right now."

        # Get closest or most relevant object
        closest = min(objects, key=lambda x: self._distance_rank(x.get("distance_bucket", "far")))

        side = closest.get("side", "center")
        class_name = closest.get("class_name", "object")

        if side == "center":
            return f"The {class_name} is directly ahead."
        else:
            return f"The {class_name} is on your {side}."

    def _handle_what_do(self) -> str:
        """Handle 'what do I do' question."""
        recent = self.memory.get_most_recent()
        if not recent:
            return "Continue forward. I'll let you know if I detect anything."

        # Get last instruction if available
        instruction = recent.get("instruction")
        if instruction:
            return instruction

        # Otherwise, analyze current situation
        objects = recent.get("tracked_objects", [])
        hazards = recent.get("ground_hazards", {})

        if not objects and max(hazards.values(), default=0.0) < 0.3:
            return "Path looks clear. Continue forward."
        else:
            return "Wait for my next instruction."

    def _handle_read_sign(self, current_frame, ocr_reader) -> str:
        """Handle 'read the sign' request."""
        if current_frame is None or ocr_reader is None:
            return "I can't read signs right now."

        from ocr_read import read_sign_on_demand

        logger.info("Reading sign...")
        text = read_sign_on_demand(current_frame, ocr_reader)

        if text:
            return f"It says: {text}"
        else:
            return "I couldn't read any text."

    def _handle_what_is(self) -> str:
        """Handle 'what is that' question."""
        recent = self.memory.get_most_recent()
        if not recent:
            return "I haven't detected anything yet."

        objects = recent.get("tracked_objects", [])
        if not objects:
            return "I don't see anything specific."

        # List nearby objects
        obj_names = [obj.get("class_name", "object") for obj in objects[:3]]
        unique_names = list(dict.fromkeys(obj_names))  # Remove duplicates, preserve order

        if len(unique_names) == 1:
            return f"I see a {unique_names[0]}."
        elif len(unique_names) == 2:
            return f"I see a {unique_names[0]} and a {unique_names[1]}."
        else:
            return f"I see {', '.join(unique_names[:-1])}, and a {unique_names[-1]}."

    def _handle_repeat(self) -> str:
        """Handle 'repeat' request."""
        last_instruction = self.memory.get_last_instruction()
        if last_instruction:
            return last_instruction
        else:
            return "I haven't said anything yet."

    def _distance_rank(self, bucket: str) -> int:
        """Rank distance bucket (lower = closer)."""
        ranks = {
            "very_near": 0,
            "near": 1,
            "mid": 2,
            "far": 3,
            "unknown": 4
        }
        return ranks.get(bucket, 4)


if __name__ == "__main__":
    # Test intent parsing
    logging.basicConfig(level=logging.INFO)

    parser = IntentParser()

    test_phrases = [
        "How far is it?",
        "Where is the person?",
        "What should I do?",
        "Read the sign",
        "What is that?",
        "Repeat that please",
        "This is random text"
    ]

    for phrase in test_phrases:
        intent = parser.parse(phrase)
        print(f"'{phrase}' → {intent}")
