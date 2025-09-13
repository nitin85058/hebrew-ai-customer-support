from crewai import Agent
from textblob import TextBlob
from transformers import pipeline
import torch
import logging

logger = logging.getLogger(__name__)

class SentimentAgent(Agent):
    def __init__(self, llm_model="groq/llama-3.1-8b-instant"):
        super().__init__(
            role="Sentiment Analysis Specialist",
            goal="Analyze customer sentiment and emotional state to guide response strategy",
            backstory="I analyze language patterns, tone, and emotional indicators to provide actionable insights for customer service interactions.",
            llm=llm_model,
            verbose=True
        )
        # Initialize sentiment analysis models
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
        except:
            logger.warning("Advanced sentiment model not available, using basic TextBlob")
            self.sentiment_analyzer = None

        self._model_config = {"llm_model": llm_model}

    def analyze_sentiment(self, text: str) -> dict:
        """Analyze sentiment of text using multiple methods"""
        try:
            # Basic sentiment analysis with TextBlob
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            # Advanced sentiment if model is available
            advanced_sentiment = None
            if self.sentiment_analyzer:
                try:
                    results = self.sentiment_analyzer(text)
                    if results:
                        # results is a list of dicts with scores for each label
                        advanced_sentiment = results[0] if results else None
                except Exception as e:
                    logger.warning(f"Advanced sentiment analysis failed: {e}")

            # Determine overall sentiment
            sentiment_score = polarity  # -1 to 1
            if sentiment_score > 0.1:
                overall_sentiment = "positive"
            elif sentiment_score < -0.1:
                overall_sentiment = "negative"
            else:
                overall_sentiment = "neutral"

            # Emergency detection keywords
            emergency_keywords = [
                "emergency", "urgent", "crisis", "immediately", "help",
                "danger", "threat", "emergency", "פיקוח נפש", "סכנה"
            ]

            is_emergency = any(keyword in text.lower() for keyword in emergency_keywords)

            # Frustration indicators
            frustration_indicators = ["frustrated", "angry", "upset", "irritated", "annoyed", "fed up", "כועס", "מתוסכל"]
            frustration_level = sum(1 for indicator in frustration_indicators if indicator in text.lower())

            result = {
                "overall_sentiment": overall_sentiment,
                "sentiment_score": sentiment_score,
                "subjectivity": subjectivity,
                "is_emergency": is_emergency,
                "frustration_level": min(frustration_level, 5),  # Cap at 5
                "confidence": advanced_sentiment.get("score", 0.5) if advanced_sentiment else 0.8,
                "recommendations": self._generate_recommendations(sentiment_score, is_emergency, frustration_level)
            }

            return result

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {
                "overall_sentiment": "neutral",
                "sentiment_score": 0.0,
                "subjectivity": 0.5,
                "is_emergency": False,
                "frustration_level": 0,
                "confidence": 0.0,
                "recommendations": ["Continue with standard protocol"]
            }

    def _generate_recommendations(self, sentiment_score: float, is_emergency: bool, frustration_level: int) -> list:
        """Generate actionable recommendations based on sentiment analysis"""
        recommendations = []

        if is_emergency:
            recommendations.append("PRIORITY: Escalate immediately to emergency protocol")
            recommendations.append("Notify supervisor immediately")
            return recommendations

        if sentiment_score < -0.6:
            recommendations.append("HIGH PRIORITY: Customer is very dissatisfied - offer immediate compensation")
            recommendations.append("Empathize strongly and apologize profusely")
            recommendations.append("Escalate to supervisor if issue persists")

        elif sentiment_score < -0.3:
            recommendations.append("Escalate to next level support")
            recommendations.append("Offer goodwill gesture or discount")
            recommendations.append("Document for quality review")

        elif sentiment_score > 0.3:
            recommendations.append("Customer is satisfied - reinforce positive experience")
            recommendations.append("Use opportunity to promote additional services")

        if frustration_level > 3:
            recommendations.append("Customer shows high frustration - remain calm and patient")
            recommendations.append("Avoid technical jargon")
            recommendations.append("Offer to transfer to different representative")

        return recommendations if recommendations else ["Continue with standard protocol"]

    def track_sentiment_trends(self, conversation_history: list) -> dict:
        """Track sentiment changes throughout conversation"""
        if not conversation_history:
            return {"trend": "no_data", "change_rate": 0.0}

        sentiments = []
        for entry in conversation_history:
            if "speaker" in entry and "message" in entry:
                analysis = self.analyze_sentiment(entry["message"])
                sentiments.append({
                    "speaker": entry["speaker"],
                    "sentiment": analysis["sentiment_score"],
                    "timestamp": entry.get("timestamp")
                })

        if len(sentiments) < 2:
            return {"trend": "insufficient_data", "change_rate": 0.0}

        initial_sentiment = sentiments[0]["sentiment"]
        final_sentiment = sentiments[-1]["sentiment"]
        change_rate = final_sentiment - initial_sentiment

        if change_rate > 0.2:
            trend = "improving"
        elif change_rate < -0.2:
            trend = "worsening"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "change_rate": change_rate,
            "initial_sentiment": initial_sentiment,
            "final_sentiment": final_sentiment
        }
