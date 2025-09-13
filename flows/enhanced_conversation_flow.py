import asyncio
import websockets
import json
import logging
from typing import Dict, List, Optional, Any
import datetime
import os
from dotenv import load_dotenv

from crewai.flow.flow import Flow, listen, start

# Import agents
from agents.sentiment_agent import SentimentAgent
from agents.quality_agent import QualityAgent
from agents.enhanced_guardrail_agent import EnhancedGuardrailAgent
from agents.customer_service_agent import CustomerServiceAgent
from agents.client_agent import ClientAgent
from agents.transcript_agent import TranscriptAgent
from agents.token_tracker_agent import TokenTrackerAgent
from agents.tts_agent import TTSAgent
from agents.stt_agent import STTAgent
from agents.nikud_agent import NikudAgent

# Import utilities
from utils.language_utils import LanguageSupport
from utils.logger_utils import log_message, create_transcript_entry
from utils.audio_utils import play_audio

logger = logging.getLogger(__name__)

class WebSocketConnection:
    """WebSocket connection manager"""
    def __init__(self, websocket, path):
        self.websocket = websocket
        self.path = path
        self.conversation_id = f"conv_{int(asyncio.get_event_loop().time())}"
        self.is_active = True

    async def send(self, message_type: str, data: Dict[str, Any]):
        """Send message to client"""
        message = {
            "type": message_type,
            "timestamp": datetime.datetime.now().isoformat(),
            "data": data
        }
        try:
            await self.websocket.send(json.dumps(message, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")

    async def receive(self) -> Dict[str, Any]:
        """Receive message from client"""
        try:
            message = await self.websocket.recv()
            return json.loads(message)
        except Exception as e:
            logger.error(f"Failed to receive WebSocket message: {e}")
            return {"type": "error", "data": str(e)}

class EnhancedConversationFlow(Flow):
    """Enhanced conversation flow with real-time AI capabilities"""

    def __init__(self):
        super().__init__()
        self._initialize_agents()
        self.language_support = LanguageSupport()
        self.websocket_connections: Dict[str, WebSocketConnection] = {}
        self.active_conversations: Dict[str, List[Dict]] = {}
        self.customer_profiles: Dict[str, Dict] = {}

        # Performance counters
        self.total_connections = 0
        self.total_conversations = 0

    def _initialize_agents(self):
        """Initialize all agents"""
        self.sentiment_agent = SentimentAgent()
        self.quality_agent = QualityAgent()
        self.guardrail_agent = EnhancedGuardrailAgent()
        self.cs_agent = CustomerServiceAgent()
        self.client_agent = ClientAgent()
        self.transcript_agent = TranscriptAgent()
        self.token_tracker = TokenTrackerAgent()
        self.tts_agent = TTSAgent()
        self.stt_agent = STTAgent()
        self.nikud_agent = NikudAgent()

    @start()
    async def initialize_system(self):
        """Initialize the enhanced conversation system"""
        self.token_tracker.reset_tracker()
        logger.info("Enhanced conversation system initialized")

        # Start WebSocket server
        server = await websockets.serve(
            self.handle_connection,
            "localhost",
            8080,
            ping_interval=30,
            ping_timeout=10
        )
        logger.info("WebSocket server started on ws://localhost:8080")

        return {"status": "initialized", "websocket_port": 8080}

    async def handle_connection(self, websocket, path):
        """Handle WebSocket connection"""
        connection = WebSocketConnection(websocket, path)
        self.websocket_connections[connection.conversation_id] = connection
        self.total_connections += 1

        logger.info(f"New connection: {connection.conversation_id}")

        try:
            await connection.send("connected", {
                "conversation_id": connection.conversation_id,
                "supported_languages": list(self.language_support.supported_languages.keys()),
                "features": ["real_time_analysis", "multilingual", "sentiment_tracking", "quality_scoring"]
            })

            await self.conversation_loop(connection)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed: {connection.conversation_id}")
        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            if connection.conversation_id in self.websocket_connections:
                del self.websocket_connections[connection.conversation_id]

    async def conversation_loop(self, connection: WebSocketConnection):
        """Main conversation loop"""
        conversation_history = []
        self.active_conversations[connection.conversation_id] = conversation_history

        # Initialize customer profile
        self.customer_profiles[connection.conversation_id] = {
            "language": "he",
            "sentiment_trend": [],
            "quality_scores": [],
            "vip_status": False,
            "churn_risk": "low"
        }

        try:
            while connection.is_active:
                message = await connection.receive()

                if message["type"] == "user_message":
                    response = await self.process_user_message(message["data"], connection, conversation_history)
                    await connection.send("agent_response", response)

                elif message["type"] == "voice_data":
                    transcript = await self.process_voice_input(message["data"], connection)
                    await connection.send("transcript", {"text": transcript})

                elif message["type"] == "language_change":
                    await self.handle_language_change(message["data"], connection)

                elif message["type"] == "end_conversation":
                    await self.end_conversation(connection.conversation_id)
                    break

                elif message["type"] == "error":
                    logger.error(f"Received error message: {message['data']}")
                    break

        except Exception as e:
            logger.error(f"Conversation loop error: {e}")
        finally:
            if connection.conversation_id in self.active_conversations:
                del self.active_conversations[connection.conversation_id]

    async def process_user_message(self, message_data: Dict, connection: WebSocketConnection,
                                 conversation_history: List) -> Dict:
        """Process incoming user message with comprehensive analysis"""

        user_text = message_data.get("text", "")
        user_language = message_data.get("language", "auto")
        message_type = message_data.get("type", "text")

        # Detect language if not specified
        if user_language == "auto":
            detected_lang, confidence = self.language_support.detect_language(user_text)
            user_language = detected_lang
            self.customer_profiles[connection.conversation_id]["language"] = detected_lang

        # Translate if necessary (assume agents respond in Hebrew)
        if user_language != "he" and user_text:
            translation_result = await self.language_support.translate_text(user_text, user_language, "he")
            if translation_result["success"]:
                processed_text = translation_result["translated_text"]
            else:
                processed_text = user_text  # Fallback to original text
        else:
            processed_text = user_text

        # Enhanced content safety check
        safety_analysis = await self.guardrail_agent.advanced_content_check(
            processed_text,
            context={
                "conversation_history": conversation_history,
                "customer_profile": self.customer_profiles[connection.conversation_id]
            }
        )

        if not safety_analysis["safe"]:
            await connection.send("safety_alert", safety_analysis)
            # Continue but flag the message
            response_text = "אני מבין את החשש שלך. בואו נעבור לנהל את השיחה בצורה בטוחה ותקינה."
            emergency_detected = safety_analysis.get("emergency_detected", False)
            if emergency_detected:
                response_text = "זוהתה מצב חירום. אני מעביר אותך מיד לקבלה אנושית. אנא המתן."
        else:
            # Generate agent response using AI insights
            sentiment_analysis = self.sentiment_agent.analyze_sentiment(processed_text)
            await connection.send("sentiment_analysis", sentiment_analysis)

            # Get AI-powered response
            response_text = await self.generate_enhanced_response(
                processed_text,
                conversation_history,
                sentiment_analysis,
                safety_analysis
            )

        # Add to conversation history
        conversation_history.append({
            "speaker": "Customer",
            "text": processed_text,
            "original_text": user_text,
            "language": user_language,
            "sentiment": sentiment_analysis if 'sentiment_analysis' in locals() else None,
            "timestamp": datetime.datetime.now().isoformat()
        })

        # Generate audio response
        audio_file = await self.tts_agent.synthesize_speech(response_text, f"response_{connection.conversation_id}_{len(conversation_history)}.mp3")

        # Track tokens and quality
        self.token_tracker.track_usage(len(processed_text), len(response_text))

        # Quality evaluation
        if len(conversation_history) >= 3:
            quality_result = await self.quality_agent.evaluate_conversation_quality(conversation_history)
            await connection.send("quality_feedback", quality_result)
            self.customer_profiles[connection.conversation_id]["quality_scores"].append(quality_result["quality_score"])

        # Create transcript entry
        create_transcript_entry("Agent", response_text)

        return {
            "text": response_text,
            "audio_file": audio_file,
            "language": "he",
            "sentiment_recommendations": sentiment_analysis.get("recommendations", []) if 'sentiment_analysis' in locals() else []
        }

    async def generate_enhanced_response(self, user_text: str, conversation_history: List,
                                      sentiment_analysis: Dict, safety_analysis: Dict) -> str:
        """Generate AI-enhanced response with multiple insights"""

        # Build context prompt
        context = f"""
        Customer message: {user_text}

        Sentiment Analysis:
        - Overall: {sentiment_analysis.get('overall_sentiment', 'neutral')}
        - Score: {sentiment_analysis.get('sentiment_score', 0):.2f}
        - Frustration Level: {sentiment_analysis.get('frustration_level', 0)}
        - Is Emergency: {sentiment_analysis.get('is_emergency', False)}
        - Recommendations: {', '.join(sentiment_analysis.get('recommendations', []))}

        Safety Status: {'SAFE' if safety_analysis['safe'] else 'REQUIRES_ATTENTION'}

        Conversation History Length: {len(conversation_history)}
        """

        # Enhanced agent response
        prompt = f"""
        You are an expert customer service agent for TV subscription services.

        {context}

        Generate a response that:
        1. Addresses the customer's sentiment appropriately
        2. Uses empathy when needed (frustration level: {sentiment_analysis.get('frustration_level', 0)})
        3. Provides solutions for TV subscription issues
        4. Stays professional and helpful
        5. Considers retention when appropriate
        6. Responds in Hebrew

        Keep response concise but comprehensive.
        """

        response = await self.cs_agent.respond_to_customer(prompt)
        return response.choices[0].message.content if hasattr(response, 'choices') else response

    async def process_voice_input(self, voice_data: Dict, connection: WebSocketConnection) -> str:
        """Process voice input data"""
        try:
            # This would integrate with actual voice processing
            # For now, simulate voice-to-text conversion
            audio_file = voice_data.get("audio_file")

            if audio_file and os.path.exists(audio_file):
                # Use STT agent (would need to be enhanced for real-time)
                transcript = await self.stt_agent.transcribe_to_text(audio_file)

                # Apply nikud (Hebrew pronunciation)
                nikud_text = self.nikud_agent.add_pronunciation(transcript)

                return nikud_text
            else:
                return "שמעתי את ההקלטה שלך, אנא נסה שוב."

        except Exception as e:
            logger.error(f"Voice processing error: {e}")
            return "מצטער, לא הצלחתי לעבד את הקול. אנא כתוב את ההודעה."

    async def handle_language_change(self, language_data: Dict, connection: WebSocketConnection):
        """Handle language preference change"""
        new_language = language_data.get("language")
        if new_language in self.language_support.supported_languages:
            self.customer_profiles[connection.conversation_id]["language"] = new_language
            await connection.send("language_updated", {
                "language": new_language,
                "info": self.language_support.get_language_info(new_language)
            })

    async def generate_realtime_insights(self, connection_id: str) -> Dict:
        """Generate real-time insights for active conversation"""
        if connection_id not in self.active_conversations:
            return {"error": "Conversation not found"}

        conversation = self.active_conversations[connection_id]
        profile = self.customer_profiles.get(connection_id, {})

        # Sentiment trends
        sentiment_trend = self.sentiment_agent.track_sentiment_trends(conversation)

        # Quality assessment
        if len(conversation) >= 3:
            quality = await self.quality_agent.evaluate_conversation_quality(conversation)
        else:
            quality = {"quality_score": 0, "message": "Insufficient data"}

        return {
            "conversation_stats": {
                "total_exchanges": len(conversation),
                "duration": "active",  # Would calculate from timestamps
                "current_language": profile.get("language", "he")
            },
            "sentiment_trend": sentiment_trend,
            "quality_score": quality.get("quality_score", 0),
            "risk_assessment": {
                "churn_risk": profile.get("churn_risk", "low"),
                "recommendations": quality.get("recommendations", [])
            }
        }

    async def end_conversation(self, conversation_id: str):
        """End conversation and generate final reports"""
        try:
            conversation = self.active_conversations.get(conversation_id, [])

            # Generate final analysis
            if conversation:
                # Final quality evaluation
                final_quality = await self.quality_agent.evaluate_conversation_quality(conversation)

                # Sentiment summary
                sentiment_summary = self.sentiment_agent.track_sentiment_trends(conversation)

                # Save transcript
                await self.transcript_agent.save_transcript(conversation)
                summary = await self.transcript_agent.generate_summary(conversation)

                # Generate performance report
                performance_report = self.quality_agent.generate_performance_report()

                final_report = {
                    "conversation_id": conversation_id,
                    "summary": summary,
                    "final_quality": final_quality,
                    "sentiment_summary": sentiment_summary,
                    "performance_report": performance_report,
                    "total_tokens": self.token_tracker.get_usage(),
                    "total_exchanges": len(conversation)
                }

                # Send final report if connection exists
                if conversation_id in self.websocket_connections:
                    await self.websocket_connections[conversation_id].send("conversation_ended", final_report)

                logger.info(f"Conversation {conversation_id} ended. Final quality: {final_quality.get('grade', 'N/A')}")

                self.total_conversations += 1

        except Exception as e:
            logger.error(f"Error ending conversation {conversation_id}: {e}")

    def get_system_stats(self) -> Dict:
        """Get system-wide statistics"""
        return {
            "active_connections": len(self.websocket_connections),
            "total_connections": self.total_connections,
            "total_conversations": self.total_conversations,
            "performance_metrics": self.quality_agent.generate_performance_report() if self.quality_agent else {},
            "system_uptime": "running",
            "supported_languages": len(self.language_support.supported_languages)
        }
