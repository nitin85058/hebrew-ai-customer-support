import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(filename='conversation.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_message(message: str):
    """Log a message"""
    logging.info(message)

def log_token_usage(tokens: int):
    """Log token usage"""
    log_message(f"Tokens used: {tokens}")

def create_transcript_entry(speaker: str, message: str, timestamp: datetime = None):
    """Add to transcript"""
    if timestamp is None:
        timestamp = datetime.now()
    log_message(f"[{timestamp}] {speaker}: {message}")

# Additional tracking
