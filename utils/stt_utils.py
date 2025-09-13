import whisper

model = whisper.load_model("base")  # Load once

def transcribe_audio(audio_file: str) -> str:
    """Transcribe audio file to Hebrew text"""
    result = model.transcribe(audio_file, language='he')
    return result["text"]

# Whisper supports Hebrew.
