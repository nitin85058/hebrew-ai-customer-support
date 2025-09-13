from crewai import Agent
from utils.audio_utils import text_to_speech, convert_mp3_to_wav
from utils.logger_utils import log_message
import asyncio

class TTSAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Text-to-Speech Specialist",
            goal="Convert Hebrew text to high-quality audio speech files.",
            backstory="I specialize in generating natural and clear voice overs from text, particularly for Hebrew language.",
            verbose=True
        )

    async def synthesize_speech(self, text: str, output_file: str = "temp_speech.mp3") -> str:
        """Synthesize speech from text"""
        audio_file = await text_to_speech(text, output_file)
        log_message(f"Speech synthesized: {audio_file}")
        # Skipping mp3 to wav conversion
        return audio_file
