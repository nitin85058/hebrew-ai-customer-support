# 🚀 Enhanced Hebrew Customer Support CrewAI Project

This advanced project provides a state-of-the-art multilingual customer service platform with real-time AI capabilities, built using CrewAI framework. It simulates and handles TV subscription cancellation scenarios in Hebrew and multiple other languages.

## ✨ Enhanced Features

### 1. 🔄 Real-Time Multi-Modal Conversation Support
- **WebSocket Integration**: Live bidirectional communication
- **Multi-Modal Inputs**: Support for text, voice, and future video inputs
- **Real-Time Analysis**: Instant sentiment and quality evaluation
- **Live Transcription**: Real-time speech-to-text with incremental updates

### 2. 🧠 Advanced AI-Powered Features
- **Sentiment Analysis Agent**: Emotion detection and emotional state analysis
- **Context-Aware Responses**: Conversation memory with summarization
- **Personalization Engine**: Customer profile-based tailored responses
- **Intent Classification**: Advanced NLP for request categorization

### 3. 📊 Quality & Analytics Enhancement
- **Conversation Quality Scoring**: Automated scoring of agent responses and customer satisfaction
- **Performance Analytics**: Real-time metrics, success rates, and conversation insights
- **A/B Testing Framework**: Test different response strategies and measure effectiveness
- **Voice Quality Analysis**: Evaluate audio clarity, accent detection, and pronunciation

### 4. 🛡️ Enhanced Safety & Compliance
- **Advanced Guardrails with ML**: Machine learning-powered content moderation
- **Regulatory Compliance**: GDPR, PCI, HIPAA pattern detection
- **Bias Detection**: Real-time identification of biased language
- **Emergency Protocol Agent**: Crisis situation detection and handling

### 5. 🌍 Global & Multilingual Expansion
- **9+ Supported Languages**: Hebrew, Arabic, English, French, German, Spanish, Russian, Chinese, Yiddish
- **Real-Time Translation**: Automatic translation between languages
- **Cultural Adaptation**: Region-specific conversation norms and etiquette
- **Dialect Recognition**: Hebrew and Arabic dialect detection (Modern Israeli, Ashkenazi, Mizrahi, etc.)

## 🏗️ Architecture

The enhanced system uses a **multi-agent orchestrator** with the following components:

### Core Agents
- **SentimentAgent**: Analyzes customer emotion and provides strategy recommendations
- **QualityAgent**: Evaluates conversation quality and generates improvement recommendations
- **EnhancedGuardrailAgent**: Advanced safety monitoring with ML capabilities
- **CustomerServiceAgent**: Enhanced customer service responses with AI insights
- **TranscriptAgent**: Comprehensive conversation logging and summarization
- **LanguageSupport**: Multi-lingual translation and cultural adaptation

### Utility Modules
- **language_utils.py**: Language detection, translation, and cultural adaptation
- **Enhanced Conversation Flow**: WebSocket-based real-time orchestration

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- GROQ API Key
- Required dependencies

### Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set API keys in `.env`**:
   ```bash
   GROQ_API_KEY=your_groq_key_here
   ```
   Get your free API key from [Groq Console](https://console.groq.com/)

3. **Run the enhanced system**:
   ```bash
   python enhanced_main.py
   ```

The system will start a WebSocket server on `ws://localhost:8080`

## 📡 WebSocket API

### Connection
Connect to `ws://localhost:8080` to start a conversation.

### Message Format
```json
{
  "type": "user_message",
  "data": {
    "text": "ביטול המנוי שלי בבקשה",
    "language": "he"
  }
}
```

### Response Types
- `connected`: Connection confirmation with supported features
- `agent_response`: AI-powered response with sentiment analysis
- `sentiment_analysis`: Real-time emotion analysis
- `quality_feedback`: Conversation quality scoring
- `safety_alert`: Content safety notifications
- `language_updated`: Language preference changes
- `conversation_ended`: Final conversation report

## 🎯 Usage Examples

### Text-based Conversation
```python
# Client sends
{
  "type": "user_message",
  "data": {
    "text": "אני רוצה לבטל את המנוי",
    "language": "he"
  }
}

# Server responds with analysis + response
{
  "type": "sentiment_analysis",
  "data": {
    "overall_sentiment": "negative",
    "frustration_level": 2,
    "recommendations": ["Show empathy", "Offer alternatives"]
  }
}
```

### Language Change
```python
{
  "type": "language_change",
  "data": {
    "language": "ar"
  }
}
```

## 🛠️ Development Features

### Training and Fine-tuning
```python
from agents.enhanced_guardrail_agent import EnhancedGuardrailAgent

agent = EnhancedGuardrailAgent()
# Train threat detection model
agent.train_threat_model(training_data)

# Update safety patterns
agent.update_patterns({"custom": ["pattern1", "pattern2"]})
```

### System Monitoring
```python
from flows.enhanced_conversation_flow import EnhancedConversationFlow

flow = EnhancedConversationFlow()
stats = flow.get_system_stats()

# Real-time insights
insights = await flow.generate_realtime_insights(conversation_id)
```

## 📈 Performance Metrics

The system tracks comprehensive analytics:
- **Conversation Quality Score**: 0-10 scale with detailed breakdowns
- **Sentiment Trends**: Real-time emotion monitoring
- **Resolution Rates**: Success metrics for customer service interactions
- **Safety Compliance**: Violation tracking and reporting
- **Multi-lingual Success**: Translation accuracy and cultural adaptation effectiveness

## 🔧 Configuration

### Environment Variables
```bash
GROQ_API_KEY=your_key_here
WEBHOOK_URL=https://your-webhook.com  # Optional
ENABLE_VOICE_PROCESSING=true        # Optional
LOG_LEVEL=INFO                       # Optional
```

### Supported Languages Configuration
Languages are configured in `utils/language_utils.py` with automatic detection and cultural adaptation patterns.

## 📝 Scripts

- `main.py`: Original simulation system
- `enhanced_main.py`: New real-time WebSocket system
- `flows/conversation_flow.py`: Original flow orchestration
- `flows/enhanced_conversation_flow.py`: Advanced real-time flow

## 🔐 Security Features

- **Content Moderation**: Multi-layered safety checks
- **Data Privacy**: Compliance with GDPR, CCPA standards
- **Emergency Detection**: Crisis situation handling protocols
- **Bias Monitoring**: Real-time bias detection and alerts
- **Regulatory Compliance**: Built-in compliance patterns for multiple regions

## 🌟 Future Enhancements

- **Voice Quality Analysis**: Advanced audio processing with librosa and speechbrain
- **Video Support**: Facial expression recognition and body language analysis
- **Dashboard Interface**: Web-based monitoring and analytics interface
- **Integration APIs**: CRM, billing, and external service integration
- **Advanced ML Models**: Reinforcem ent learning for response optimization

---

## 📄 Legacy Notes

The original simulation system is preserved in `main.py` and `flows/conversation_flow.py` for backward compatibility.

---

*Built with ❤️ using CrewAI framework for advanced AI orchestration*
