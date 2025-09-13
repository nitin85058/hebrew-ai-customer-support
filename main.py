from dotenv import load_dotenv
from flows.conversation_flow import ConversationFlow
import os
import asyncio

# Load environment variables
load_dotenv()

# Ensure API key is set
if not os.getenv("GROQ_API_KEY"):
    print("Please set GROQ_API_KEY in .env file")
    exit(1)

def main():
    flow = ConversationFlow()
    
    # Run the flow
    flow.kickoff()
    
    print("Conversation simulation complete.")
    print("Check transcript.txt and conversation.log for logs.")
    print("Audio files saved as .wav")

if __name__ == "__main__":
    main()
