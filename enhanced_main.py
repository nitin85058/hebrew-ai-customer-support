#!/usr/bin/env python3
"""
Enhanced Hebrew Customer Support CrewAI Project
==========================================

This enhanced version includes:
1. Real-Time Multi-Modal Conversation Support (WebSocket integration)
2. Advanced AI-Powered Features (Sentiment Analysis, Quality Scoring)
3. Enhanced Safety & Compliance (ML-powered guardrails)
4. Global & Multilingual Expansion (Real-time translation)

Run this script to start the enhanced conversation system.
"""

import asyncio
import logging
import signal
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure API key is set
if not os.getenv("GROQ_API_KEY"):
    print("Please set GROQ_API_KEY in .env file")
    print("Get your free API key from https://console.groq.com/")
    sys.exit(1)

# Import enhanced system
from flows.enhanced_conversation_flow import EnhancedConversationFlow

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_conversation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """Main function to run the enhanced conversation system"""
    try:
        logger.info("Starting Enhanced Hebrew Customer Support System...")

        # Initialize the enhanced flow
        flow = EnhancedConversationFlow()

        # Start the system
        init_result = await flow.initialize_system()

        print("🚀 Enhanced Customer Support System Started!")
        print("=" * 50)
        print(f"📡 WebSocket Server: ws://localhost:{init_result['websocket_port']}")
        print("🌍 Supported Languages: Multi-lingual support enabled")
        print("🧠 AI Features: Sentiment analysis, quality scoring, enhanced safety")
        print("🎯 Real-time: WebSocket integration for live conversations")
        print("=" * 50)
        print("📋 System is ready to accept connections...")
        print("🛑 Use Ctrl+C to stop the server")
        print()

        # Keep the server running
        loop = asyncio.get_running_loop()

        def signal_handler(sig, frame):
            logger.info("Receiving shutdown signal...")
            print("\n🛑 Shutting down the enhanced customer support system...")
            tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
            logger.info(f"Cancelling {len(tasks)} running tasks...")
            for task in tasks:
                task.cancel()

        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            # Keep running indefinitely
            await asyncio.Future()
        except asyncio.CancelledError:
            logger.info("System shutdown initiated")
        finally:
            # Generate final system report
            stats = flow.get_system_stats()
            print("\n📊 Final System Statistics:")
            print(f"• Total Connections: {stats['total_connections']}")
            print(f"• Total Conversations: {stats['total_conversations']}")
            print(f"• Supported Languages: {stats['supported_languages']}")
            print(f"• System Status: {stats['system_uptime']}")

            if stats['total_conversations'] > 0:
                perf_report = stats.get('performance_metrics', {})
                avg_quality = perf_report.get('period_metrics', {}).get('average_quality_score', 0)
                resolution_rate = perf_report.get('period_metrics', {}).get('resolution_rate', 0) * 100

                print(f"• Average Quality Score: {avg_quality:.2f}/10")
                print(f"• Resolution Rate: {resolution_rate:.1f}%")
                print(f"• Top Issues: {len(perf_report.get('top_issues', []))}")

            print("\n✅ System shutdown complete!")

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        sys.exit(1)

def print_feature_summary():
    """Print summary of enhanced features"""
    print("\n⭐ Enhanced Features Available:")
    print("1. 🔄 Real-Time Conversation Support")
    print("   • WebSocket integration for live communication")
    print("   • Multi-modal inputs (text, voice, future: video)")
    print("   • Real-time sentiment and quality analysis")
    print()
    print("2. 🧠 Advanced AI-Powered Features")
    print("   • Sentiment Analysis Agent with emotion detection")
    print("   • Context-aware conversation memory")
    print("   • Personalization engine for tailored responses")
    print("   • Intent classification for better understanding")
    print()
    print("3. 📊 Quality & Analytics Enhancement")
    print("   • Real-time conversation quality scoring")
    print("   • Performance analytics dashboard (future: web interface)")
    print("   • A/B testing framework for response optimization")
    print("   • Voice quality analysis (future: enhanced audio processing)")
    print()
    print("4. 🛡️ Enhanced Safety & Compliance")
    print("   • ML-powered advanced guard rails")
    print("   • Multi-language regulatory compliance")
    print("   • Real-time bias detection")
    print("   • Emergency protocol agent")
    print()
    print("5. 🌍 Global & Multilingual Expansion")
    print("   • Support for 9+ languages with auto-detection")
    print("   • Real-time translation capabilities")
    print("   • Cultural adaptation for different regions")
    print("   • Hebrew dialect recognition")
    print()

if __name__ == "__main__":
    # Print feature summary
    print_feature_summary()

    # Run the enhanced system
    asyncio.run(main())
