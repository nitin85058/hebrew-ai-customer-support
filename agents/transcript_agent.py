from crewai import Agent
from litellm import acompletion as completion
from utils.logger_utils import log_message
import os

class TranscriptAgent(Agent):
    def __init__(self, llm_model="groq/llama-3.1-8b-instant"):
        super().__init__(
            role="Transcript Recorder",
            goal="Record, save, and summarize the conversation transcript for review.",
            backstory="I ensure all communications are logged accurately for compliance and improvement.",
            llm=llm_model,
            verbose=True
        )
        # Store model for local use in an instance variable dictionary
        self._model_config = {"llm_model": llm_model}
        self.llm = llm_model  # Ensure llm attribute exists

    def save_transcript(self, transcript_data: list, file_name: str = "transcript.txt"):
        """Save full transcript to file"""
        try:
            # Ensure transcript_data is a list
            if not isinstance(transcript_data, list):
                print(f"Warning: transcript_data is not a list, got {type(transcript_data)}")
                return

            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_name) if os.path.dirname(file_name) else ".", exist_ok=True)

            with open(file_name, "w", encoding="utf-8") as f:
                if not transcript_data:
                    f.write("No conversation data available.\n")
                else:
                    for entry in transcript_data:
                        if isinstance(entry, dict) and 'timestamp' in entry and 'speaker' in entry and 'message' in entry:
                            f.write(f"{entry['timestamp']} - {entry['speaker']}: {entry['message']}\n")
                        else:
                            f.write(f"Invalid entry format: {entry}\n")

            print(f"✅ Transcript saved to {file_name}")
            log_message(f"Transcript saved to {file_name}")

        except Exception as e:
            print(f"❌ Error saving transcript: {e}")
            log_message(f"Error saving transcript: {e}")

    async def generate_summary(self, transcript_data: list) -> str:
        """Generate summary of the conversation"""
        try:
            if not transcript_data:
                return "No conversation data available for summary."

            # Handle async completion correctly
            llm_model = getattr(self, '_model_config', {}).get('llm_model', "groq/llama-3.1-8b-instant") if hasattr(self, '_model_config') else "groq/llama-3.1-8b-instant"

            full_transcript = "\n".join(
                [f"{entry.get('speaker', 'Unknown')}: {entry.get('message', '')}" for entry in transcript_data if isinstance(entry, dict)]
            )
            prompt = f"""
            Please provide a concise summary of the following customer service conversation transcript:

            ---
            {full_transcript}
            ---
            """
            response = await completion(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            summary_text = response.choices[0].message.content if response.choices else "Summary generation failed."

            log_message(f"Generated summary: {summary_text}")
            return summary_text

        except Exception as e:
            error_msg = f"Error generating summary: {e}"
            print(error_msg)
            log_message(error_msg)
            return "Summary generation failed due to an error."
