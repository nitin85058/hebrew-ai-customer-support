from crewai import Agent
from utils.stt_utils import transcribe_audio
from utils.logger_utils import log_message

class STTAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Speech-to-Text Specialist",
            goal="Accurately transcribe Hebrew speech from audio files to text.",
            backstory="I excel at understanding and transcribing spoken Hebrew, handling accents and dialects effectively.",
            verbose=True
        )

    def transcribe_to_text(self, audio_file: str) -> str:
        """Transcribe audio to text"""
        transcript = transcribe_audio(audio_file)
        log_message(f"Transcribed: {transcript}")
        return transcript
