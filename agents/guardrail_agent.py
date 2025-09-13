from crewai import Agent
from utils.logger_utils import log_message

class GuardrailAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Safety and Compliance Guardian",
            goal="Ensure all interactions comply with policies and are safe for all parties.",
            backstory="I monitor for inappropriate content, ensure professional conduct, and enforce conversation limits.",
            verbose=True
        )

    def check_content(self, text: str) -> bool:
        """Check if content is safe and appropriate"""
        forbidden_words = ["offensive", "harmful"]  # Add real list
        for word in forbidden_words:
            if word in text.lower():
                log_message(f"Content violation: {word}")
                return False
        return True

    def enforce_limits(self, conversation_length: int) -> str:
        """Enforce conversation length limits"""
        if conversation_length > 3:  # Reduced limit for testing
            print(f"🛑 Conversation limit reached: {conversation_length} turns")
            return "END_CONVERSATION"
        return "CONTINUE"

    def review_decision(self, decision: str):
        """Log review of decisions"""
        log_message(f"Guardrail decision: {decision}")
