from crewai.flow.flow import Flow, listen, start
from agents.nikud_agent import NikudAgent
from agents.tts_agent import TTSAgent
from agents.stt_agent import STTAgent
from agents.customer_service_agent import CustomerServiceAgent
from agents.client_agent import ClientAgent
from agents.transcript_agent import TranscriptAgent
from agents.token_tracker_agent import TokenTrackerAgent
from agents.guardrail_agent import GuardrailAgent
from utils.logger_utils import create_transcript_entry
import asyncio
import datetime

class ConversationFlow(Flow):
    def __init__(self):
        super().__init__()  # Initialize the parent Flow class
        self.nikud_agent = NikudAgent()
        self.tts_agent = TTSAgent()
        self.stt_agent = STTAgent()
        self.cs_agent = CustomerServiceAgent()
        self.client_agent = ClientAgent()
        self.transcript_agent = TranscriptAgent()
        self.token_agent = TokenTrackerAgent()
        self.guardrail_agent = GuardrailAgent()
        self.conversation = []
        self.turn = 0

    @start()
    async def initiate_call(self):
        """Start the conversation"""
        self.token_agent.reset_tracker()
        customer_text = await self.client_agent.express_concern(initial_message=True)
        self.conversation.append({"speaker": "Customer", "message": customer_text, "timestamp": datetime.datetime.now().isoformat()})
        return customer_text

    @listen("initiate_call")
    async def process_customer_input(self, text: str):
        """Process customer input"""
        print(f"Customer: {text}")
        
        # Guardrail check
        safe = self.guardrail_agent.check_content(text)
        if not safe:
            print("Content violation!")
            return
        
        # Optional Nikud
        nikud_text = self.nikud_agent.add_pronunciation(text)
        print(f"Nikud: {nikud_text}")
        
        # TTS for customer input? Or skip for simulation
        # audio_file = self.tts_agent.synthesize_speech(nikud_text)  # Save voice
        
        # STT simulation: since text is already there
        transcript = text  # In real: transcript = self.stt_agent.transcribe_to_text(audio_file)
        print(f"Transcript: {transcript}")
        
        return transcript

    @listen("process_customer_input")
    async def agent_respond(self, customer_transcript: str):
        """Agent generates response"""
        response_text = await self.cs_agent.respond_to_customer(customer_transcript)
        self.conversation.append({"speaker": "Agent", "message": response_text, "timestamp": datetime.datetime.now().isoformat()})
        print(f"Agent: {response_text}")
        
        # TTS the response
        response_audio = await self.tts_agent.synthesize_speech(response_text, f"agent_{self.turn}.mp3")
        print(f"Response audio: {response_audio}")
        
        # Track tokens (estimate)
        self.token_agent.track_usage(len(customer_transcript), len(response_text))
        
        return response_audio

    @listen("agent_respond")
    async def client_respond(self, audio_file: str):
        """Client 'hears' and responds"""
        self.client_agent.listen_to_response(audio_file)
        new_response = await self.client_agent.express_concern(initial_message=False)
        self.conversation.append({"speaker": "Customer", "message": new_response, "timestamp": datetime.datetime.now().isoformat()})
        self.turn += 1

        # Force end conversation after 1 turns for testing (1 full conversation round)
        if self.turn >= 1:
            print(f"🛑 Force ending conversation after Turn {self.turn} (debug)")
            return "END"

        print(f"Customer: Turn {self.turn}: {new_response[:50]}...")
        return new_response

    @listen("client_respond")
    async def repeat_conversation(self, response: str):
        """Repeat the loop"""
        if response == "END":
            return "END_CONVERSATION"
        else:
            return response  # This will trigger process_customer_input again

    @listen("repeat_conversation")
    async def handle_end_conversation(self, response: str):
        """Handle end of conversation"""
        print(f"🔍 handle_end_conversation called with response: '{response[:100]}...'")

        if response == "END_CONVERSATION":
            print("🚀 Starting transcript saving process...")

            # Debug: Check if conversation data exists
            print(f"📊 Conversation data length: {len(self.conversation)}")
            if self.conversation:
                print(f"📋 First conversation entry: {self.conversation[0]}")

            try:
                # Save transcript and generate summary
                print("💾 Calling save_transcript...")
                self.transcript_agent.save_transcript(self.conversation)
                print("✅ save_transcript completed")

                print("📝 Generating summary...")
                summary = await self.transcript_agent.generate_summary(self.conversation)

                print("🏁 Conversation Summary:")
                print("=" * 50)
                print(summary)
                total_tokens = self.token_agent.get_usage()
                print(f"Total tokens used: {total_tokens}")
                print("=" * 50)
                return "CONVERSATION_COMPLETED"

            except Exception as e:
                print(f"❌ Error in handle_end_conversation: {e}")
                import traceback
                traceback.print_exc()
                return "ERROR_COMPLETED"
        else:
            # This shouldn't happen in normal flow - response should always be "END_CONVERSATION" or customer message
            print(f"⚠️ Unexpected response in handle_end_conversation: {response}")
            return response

# Since it's agent-to-agent, the flow loops between them.

# Note: This is a simplified version. In practice, add more logic.
