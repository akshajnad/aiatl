"""
Anthropic Claude integration for intelligent Q&A and scene understanding.
"""

import anthropic
import logging
from typing import Optional, Dict, List
import json

from config import ANTHROPIC_API_KEY

logger = logging.getLogger(__name__)


class ClaudeAssistant:
    """Claude-powered assistant for natural language Q&A."""

    def __init__(self, enabled: bool = True):
        """
        Initialize Claude assistant.

        Args:
            enabled: Whether Claude is enabled
        """
        self.enabled = enabled and bool(ANTHROPIC_API_KEY)
        self.client = None

        if self.enabled:
            try:
                self._initialize()
            except Exception as e:
                logger.error(f"Failed to initialize Claude: {e}")
                self.enabled = False
        else:
            logger.warning("Claude is disabled or API key not provided")

    def _initialize(self):
        """Initialize Claude API."""
        logger.info("Initializing Claude...")
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        logger.info("Claude initialized successfully")

    def answer_question(
        self,
        question: str,
        scene_context: Optional[Dict] = None
    ) -> str:
        """
        Answer a question about the current scene using Claude.

        Args:
            question: User's question
            scene_context: Current scene data (tracked objects, hazards, etc.)

        Returns:
            Answer text
        """
        if not self.enabled:
            return "AI assistant is not available right now."

        try:
            # Build context-aware prompt
            system_prompt, user_prompt = self._build_prompt(question, scene_context)

            # Generate response using Claude
            message = self.client.messages.create(
                model="claude-3-5-haiku-20241022",  # Fast and cost-effective
                max_tokens=150,
                temperature=0.7,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            answer = message.content[0].text.strip()
            logger.debug(f"Question: {question} → Answer: {answer}")
            return answer

        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            return "I'm having trouble answering that right now."

    def _build_prompt(self, question: str, scene_context: Optional[Dict]) -> tuple[str, str]:
        """
        Build a context-aware prompt for Claude.

        Args:
            question: User's question
            scene_context: Scene data

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # System context
        system_prompt = (
            "You are a helpful assistant for a blind/low-vision navigation app called StreetSage. "
            "Answer questions concisely and clearly in 1-2 sentences. "
            "Focus on being direct and actionable. "
            "Only use the scene information provided to answer questions."
        )

        # Build user prompt with scene context
        prompt_parts = []

        # Add scene context if available
        if scene_context:
            prompt_parts.append("Current scene information:")

            # Tracked objects
            objects = scene_context.get("tracked_objects", [])
            if objects:
                obj_descriptions = []
                for obj in objects[:5]:  # Top 5 objects
                    class_name = obj.get("class_name", "object")
                    side = obj.get("side", "center")
                    distance = obj.get("distance_bucket", "unknown distance")
                    ttc = obj.get("ttc")

                    obj_desc = f"- {class_name} on {side}, {distance}"
                    if ttc:
                        obj_desc += f", TTC: {ttc:.1f}s"
                    obj_descriptions.append(obj_desc)

                prompt_parts.append("\nDetected objects:")
                prompt_parts.append("\n".join(obj_descriptions))

            # Ground hazards
            hazards = scene_context.get("ground_hazards", {})
            if hazards and any(score > 0.3 for score in hazards.values()):
                hazard_list = []
                if hazards.get("puddle", 0) > 0.3:
                    hazard_list.append("puddle/wet surface")
                if hazards.get("uneven", 0) > 0.3:
                    hazard_list.append("uneven surface")
                if hazards.get("cone", 0) > 0.3:
                    hazard_list.append("traffic cone/tape")

                if hazard_list:
                    prompt_parts.append("\nGround hazards detected:")
                    prompt_parts.append("- " + ", ".join(hazard_list))

            # Last instruction
            last_instruction = scene_context.get("instruction")
            if last_instruction:
                prompt_parts.append(f"\nLast instruction given: \"{last_instruction}\"")

        # User question
        prompt_parts.append(f"\nUser question: {question}")

        user_prompt = "\n".join(prompt_parts)

        return system_prompt, user_prompt

    def analyze_scene(
        self,
        tracked_objects: List[Dict],
        ground_hazards: Dict[str, float]
    ) -> Optional[str]:
        """
        Get AI-powered scene analysis.

        Args:
            tracked_objects: List of tracked object dicts
            ground_hazards: Ground hazard scores

        Returns:
            Scene summary or None
        """
        if not self.enabled:
            return None

        try:
            # Build scene description
            scene_parts = ["Analyze this navigation scene:"]

            if tracked_objects:
                scene_parts.append("\nObjects detected:")
                for obj in tracked_objects[:5]:
                    class_name = obj.get("class_name", "object")
                    side = obj.get("side", "center")
                    distance = obj.get("distance_bucket", "unknown")
                    ttc = obj.get("ttc")

                    obj_desc = f"- {class_name} on {side}, {distance}"
                    if ttc:
                        obj_desc += f", TTC: {ttc:.1f}s"
                    scene_parts.append(obj_desc)

            if ground_hazards:
                active_hazards = [h for h, score in ground_hazards.items() if score > 0.3]
                if active_hazards:
                    scene_parts.append(f"\nGround hazards: {', '.join(active_hazards)}")

            scene_parts.append(
                "\nProvide a brief (1 sentence) assessment of the most important thing "
                "for navigation safety:"
            )

            user_prompt = "\n".join(scene_parts)

            message = self.client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=100,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            return message.content[0].text.strip()

        except Exception as e:
            logger.error(f"Failed to analyze scene: {e}")
            return None


# Global instance
_claude_assistant: Optional[ClaudeAssistant] = None


def get_claude_assistant() -> ClaudeAssistant:
    """Get or create global Claude assistant."""
    global _claude_assistant
    if _claude_assistant is None:
        _claude_assistant = ClaudeAssistant()
    return _claude_assistant


if __name__ == "__main__":
    # Test Claude connection
    logging.basicConfig(level=logging.INFO)

    assistant = ClaudeAssistant()

    if assistant.enabled:
        print("Testing Claude assistant...")

        # Test scene context
        test_scene = {
            "tracked_objects": [
                {
                    "class_name": "person",
                    "side": "left",
                    "distance_bucket": "near",
                    "ttc": 2.5
                },
                {
                    "class_name": "bicycle",
                    "side": "right",
                    "distance_bucket": "mid",
                    "ttc": 4.0
                }
            ],
            "ground_hazards": {
                "puddle": 0.5,
                "uneven": 0.2,
                "cone": 0.0
            },
            "instruction": "Person approaching from your left. Pause."
        }

        # Test questions
        questions = [
            "How far is the person?",
            "Where is the bicycle?",
            "What should I do?",
            "Is there anything on the ground?",
            "What's the biggest danger right now?"
        ]

        for q in questions:
            answer = assistant.answer_question(q, test_scene)
            print(f"\nQ: {q}")
            print(f"A: {answer}")

        # Test scene analysis
        print("\n" + "="*50)
        analysis = assistant.analyze_scene(
            test_scene["tracked_objects"],
            test_scene["ground_hazards"]
        )
        print(f"Scene analysis: {analysis}")

    else:
        print("Claude not configured. Check your ANTHROPIC_API_KEY.")
