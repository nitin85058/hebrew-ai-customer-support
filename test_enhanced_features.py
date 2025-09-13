#!/usr/bin/env python3
"""
Test script for enhanced features implementation
Run this script to verify all enhanced agents and utilities can be imported and initialized
"""

import sys
import os
import asyncio
from dotenv import load_dotenv

# Load environment
load_dotenv()

def test_imports():
    """Test that all enhanced components can be imported"""
    print("🔧 Testing enhanced feature imports...")

    try:
        # Test new agents
        from agents.sentiment_agent import SentimentAgent
        print("✅ SentimentAgent imported successfully")

        from agents.quality_agent import QualityAgent
        print("✅ QualityAgent imported successfully")

        from agents.enhanced_guardrail_agent import EnhancedGuardrailAgent
        print("✅ EnhancedGuardrailAgent imported successfully")

        # Test utilities
        from utils.language_utils import LanguageSupport
        print("✅ LanguageSupport imported successfully")

        # Test enhanced flow
        from flows.enhanced_conversation_flow import EnhancedConversationFlow
        print("✅ EnhancedConversationFlow imported successfully")

        # Test enhanced main
        from enhanced_main import print_feature_summary
        print("✅ Enhanced main module imported successfully")

        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during import: {e}")
        return False

async def test_initializations():
    """Test that all enhanced agents can be initialized"""
    print("\n🏗️ Testing enhanced feature initializations...")

    try:
        # Skip if no API key
        if not os.getenv("GROQ_API_KEY"):
            print("⚠️ Skipping agent initializations (no GROQ_API_KEY)")
            return True

        # Test agent initializations
        from agents.sentiment_agent import SentimentAgent
        sentiment_agent = SentimentAgent()
        print("✅ SentimentAgent initialized successfully")

        from agents.quality_agent import QualityAgent
        quality_agent = QualityAgent()
        print("✅ QualityAgent initialized successfully")

        from utils.language_utils import LanguageSupport
        lang_support = LanguageSupport()
        print("✅ LanguageSupport initialized successfully")

        # Test basic functionality
        print("\n🧪 Testing basic functionality...")

        # Test language detection
        detected_lang, confidence = lang_support.detect_language("ביטול המנוי בבקשה")
        print(f"✅ Language detection: {detected_lang} (confidence: {confidence:.2f})")

        # Test sentiment analysis
        sample_text = "ביטול המנוי שלי בבקשה, זה לא עובד טוב"
        sentiment_result = sentiment_agent.analyze_sentiment(sample_text)
        print(f"✅ Sentiment analysis: {sentiment_result['overall_sentiment']} (score: {sentiment_result['sentiment_score']:.2f})")

        # Test guardrail
        from agents.enhanced_guardrail_agent import EnhancedGuardrailAgent
        guardrail_agent = EnhancedGuardrailAgent()
        safety_result = await guardrail_agent.advanced_content_check(sample_text)
        print(f"✅ Advanced safety check: {'SAFE' if safety_result['safe'] else 'FLAGGED'}")

        return True

    except Exception as e:
        print(f"❌ Initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configurations():
    """Test configuration and setup"""
    print("\n⚙️ Testing configurations...")

    try:
        # Check requirements.txt exists
        if os.path.exists("requirements.txt"):
            with open("requirements.txt", "r") as f:
                requirements = f.read()
                if "transformers" in requirements and "websockets" in requirements:
                    print("✅ Enhanced requirements.txt contains necessary packages")
                else:
                    print("⚠️ Some enhanced packages might be missing from requirements.txt")
        else:
            print("❌ requirements.txt not found")
            return False

        # Check if enhanced_main.py exists
        if os.path.exists("enhanced_main.py"):
            print("✅ enhanced_main.py created successfully")
        else:
            print("❌ enhanced_main.py not found")
            return False

        # Check README updated
        if os.path.exists("README.md"):
            with open("README.md", "r") as f:
                readme_content = f.read()
                if "Real-Time Multi-Modal Conversation Support" in readme_content:
                    print("✅ README.md updated with enhanced features")
                else:
                    print("⚠️ README.md might not contain enhanced feature documentation")

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing Enhanced Hebrew Customer Support Implementation")
    print("=" * 60)

    # Run tests
    tests = [
        ("Import Tests", test_imports),
        ("Configuration Tests", test_configurations),
    ]

    async_tests = [
        ("Initialization Tests", test_initializations),
    ]

    results = []

    # Synchronous tests
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))

    # Asynchronous tests
    for test_name, test_func in async_tests:
        print(f"\n{test_name}")
        try:
            success = asyncio.run(test_func())
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if not success:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! Enhanced features are ready to use.")
        print("🚀 Run 'python enhanced_main.py' to start the enhanced system.")
    else:
        print("⚠️ Some tests failed. Please check the output above for details.")
        print("💡 Make sure all dependencies are installed and GROQ_API_KEY is set.")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
