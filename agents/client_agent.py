from crewai import Agent
from litellm import acompletion as completion
from utils.logger_utils import log_message, create_transcript_entry
from utils.audio_utils import play_audio
import os

class ClientAgent(Agent):
    def __init__(self, llm_model="groq/llama-3.1-8b-instant"):
        super().__init__(
            role="Frustrated Customer",
            goal="Expect perfect customer service while trying to cancel TV subscription.",
            backstory="I am a customer who is unhappy with the service and wants to cancel, but can be convinced otherwise with good offers.",
            llm=llm_model,  # Pass the llm model directly to CrewAI Agent
            verbose=True
        )
        # Store model for local use in an instance variable dictionary
        self._model_config = {"llm_model": llm_model}

    async def express_concern(self, initial_message: bool = True) -> str:
        """Express concern or respond to agent"""
        prompt = ""
        if initial_message:
            prompt = """
            You are a frustrated Israeli customer calling customer service to cancel your TV subscription.
            Express your dissatisfaction and request cancellation in Hebrew.

            Be polite but firm. Start the call.
            """
        else:
            prompt = """
            Continue the conversation as the customer.
            Respond to the agent's latest message in Hebrew.
            Remain frustrated but open to persuasion.
            """
        response = await completion(
            model=self._model_config["llm_model"],
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = response.choices[0].message.content
        create_transcript_entry("Customer", response_text)
        return response_text

    def listen_to_response(self, audio_file: str):
        """'Listen' to agent's response - play audio (skipped in simulation)"""
        # play_audio(audio_file)  # Skipped for simulation to avoid audio playback
        log_message("Customer listened to response audio: " + os.path.basename(audio_file))
