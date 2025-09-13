import edge_tts
import asyncio
import pydub
from pydub import AudioSegment
from playsound import playsound
import io

async def text_to_speech(text: str, output_file: str = "output.mp3"):
    """Convert text to speech using edge-tts"""
    communicate = edge_tts.Communicate(text, "he-IL-AvriNeural")  # Hebrew female voice
    await communicate.save(output_file)
    return output_file

def play_audio(file_path: str):
    """Play audio file"""
    playsound(file_path)

def convert_mp3_to_wav(mp3_file: str, wav_file: str):
    """Convert mp3 to wav for whisper"""
    sound = AudioSegment.from_mp3(mp3_file)
    sound.export(wav_file, format="wav")

def get_audio_duration(file_path: str) -> float:
    """Get audio duration in seconds"""
    audio = AudioSegment.from_file(file_path)
    return audio.duration_seconds

# For STT, will use whisper later
