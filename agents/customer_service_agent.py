from crewai import Agent
from litellm import acompletion as completion
from utils.logger_utils import log_message, create_transcript_entry
from utils.audio_utils import play_audio
import os

class CustomerServiceAgent(Agent):
    def __init__(self, llm_model="groq/llama-3.1-8b-instant"):
        super().__init__(
            role="Customer Service Manager",
            goal="Handle customer inquiries professionally, especially cancellation requests in Hebrew.",
            backstory="I am trained to manage customer relationships, respond empathetically, and use all available tools to provide the best service.",
            llm=llm_model,  # Pass the llm model directly to CrewAI Agent
            verbose=True
        )
        # Store model for local use in an instance variable dictionary
        self._model_config = {"llm_model": llm_model}

    async def respond_to_customer(self, customer_input: str) -> str:
        """Generate response to customer"""
        prompt = f"""
        You are a customer service agent for a TV subscription service in Israel.
        The customer said (in Hebrew): "{customer_input}"
        They are trying to cancel their subscription.

        Respond empathetically, professionally, and in Hebrew.
        Try to retain the customer if possible, offer alternatives.

        Keep response concise.
        """
        response = await completion(
            model=self._model_config["llm_model"],
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = response.choices[0].message.content
        create_transcript_entry("Agent", response_text)
        return response_text

    def play_response_audio(self, audio_file: str):
        """Play the agent's response"""
        play_audio(audio_file)
        log_message(f"Played audio: {audio_file}")
