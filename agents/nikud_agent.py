from crewai import Agent
from utils.nikud_utils import add_nikud_to_text
from utils.logger_utils import log_message

class NikudAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Hebrew Pronunciation Expert",
            goal="Add nikud (vowel marks) to Hebrew text for better pronunciation guidance.",
            backstory="I am an expert in Hebrew linguistics, specializing in vocalization to help non-native speakers pronounce words correctly.",
            verbose=True
        )

    def add_pronunciation(self, text: str) -> str:
        """Add pronunciation nikud to the text"""
        nikud_text = add_nikud_to_text(text)
        log_message(f"Nikud added: {nikud_text}")
        return nikud_text
