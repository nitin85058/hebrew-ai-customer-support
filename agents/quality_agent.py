from crewai import Agent
from litellm import acompletion as completion
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

class QualityAgent(Agent):
    def __init__(self, llm_model="groq/llama-3.1-8b-instant"):
        super().__init__(
            role="Quality Assurance and Analytics Specialist",
            goal="Evaluate conversation quality, track performance metrics, and provide actionable insights",
            backstory="I analyze customer service interactions to ensure high standards, measure performance, and identify areas for improvement.",
            llm=llm_model,
            verbose=True
        )

        self._model_config = {"llm_model": llm_model}
        self.quality_history = []
        self.performance_metrics = {
            "total_conversations": 0,
            "average_quality_score": 0.0,
            "resolution_rate": 0.0,
            "average_response_time": 0.0,
            "customer_satisfaction": 0.0
        }

        # Initialize regression model for quality prediction
        self.quality_predictor = LinearRegression()

    async def evaluate_conversation_quality(self, conversation_history: list) -> dict:
        """Evaluate overall conversation quality using multiple criteria"""
        try:
            if not conversation_history:
                return {"quality_score": 0.0, "issues": ["No conversation data"]}

            # Extract conversation metrics
            agent_responses = [entry for entry in conversation_history if entry.get("speaker") == "Agent"]
            customer_responses = [entry for entry in conversation_history if entry.get("speaker") == "Customer"]

            # Calculate various quality metrics
            metrics = {
                "response_count": len(agent_responses),
                "conversation_length": len(conversation_history),
                "avg_response_length": np.mean([len(r.get("message", "")) for r in agent_responses]) if agent_responses else 0,
                "empathy_indicators": self._count_empathy_indicators(agent_responses),
                "resolution_indicators": self._count_resolution_indicators(agent_responses),
                "professionalism_score": await self._evaluate_professionalism(agent_responses),
                "clarity_score": await self._evaluate_clarity(agent_responses),
                "efficiency_score": self._calculate_efficiency_score(conversation_history)
            }

            # Calculate weighted quality score
            quality_score = self._calculate_overall_quality_score(metrics)

            # Generate recommendations and issues
            issues = self._identify_quality_issues(metrics, quality_score)
            recommendations = await self._generate_quality_recommendations(metrics, quality_score)

            # Update performance metrics
            self._update_performance_metrics(quality_score, conversation_history)

            result = {
                "quality_score": round(quality_score, 2),  # 0-10 scale
                "metrics": metrics,
                "issues": issues,
                "recommendations": recommendations,
                "grade": self._convert_score_to_grade(quality_score),
                "timestamp": datetime.now().isoformat()
            }

            # Store in history
            self.quality_history.append(result)

            return result

        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")
            return {
                "quality_score": 5.0,  # Neutral default
                "issues": ["Quality evaluation error"],
                "recommendations": ["Review manually"],
                "grade": "C"
            }

    def _count_empathy_indicators(self, responses: list) -> int:
        """Count empathy-related phrases in responses"""
        empathy_words = [
            "understand", "sorry", "apologize", "empathy", "frustrating",
            "מבין", "סליחה", "מצטער", "מתנצל", "מבין את התסכול"
        ]
        count = 0
        for response in responses:
            message = response.get("message", "").lower()
            count += sum(1 for word in empathy_words if word in message)
        return count

    def _count_resolution_indicators(self, responses: list) -> int:
        """Count resolution-focused phrases in responses"""
        resolution_words = [
            "solve", "fix", "resolve", "address", "handle", "assistance",
            "פיתרון", "לפתור", "לטפל", "עזרה"
        ]
        count = 0
        for response in responses:
            message = response.get("message", "").lower()
            count += sum(1 for word in resolution_words if word in message)
        return count

    async def _evaluate_professionalism(self, responses: list) -> float:
        """Evaluate professionalism of responses using LLM"""
        if not responses:
            return 5.0

        sample_responses = responses[:3]  # Evaluate first 3 responses

        prompt = f"""
        Evaluate the professionalism of these customer service responses on a scale of 1-10.
        Consider: courtesy, appropriate language, tone, and adherence to professional standards.
        Provide only the numerical score.

        Responses to evaluate:
        {json.dumps([r.get("message", "") for r in sample_responses], indent=2)}
        """

        try:
            response = await completion(
                model=self._model_config["llm_model"],
                messages=[{"role": "user", "content": prompt}]
            )
            score = float(response.choices[0].message.content.strip())
            return min(max(score, 1.0), 10.0)
        except:
            return 7.0  # Default professional score

    async def _evaluate_clarity(self, responses: list) -> float:
        """Evaluate clarity and comprehensibility of responses"""
        if not responses:
            return 5.0

        prompt = f"""
        Evaluate the clarity and comprehensibility of these customer service responses on a scale of 1-10.
        Consider: clear language, structure, avoidance of jargon, and easy understanding.
        Provide only the numerical score.

        Sample responses:
        {json.dumps([r.get("message", "") for r in responses[:2]], indent=2)}
        """

        try:
            response = await completion(
                model=self._model_config["llm_model"],
                messages=[{"role": "user", "content": prompt}]
            )
            score = float(response.choices[0].message.content.strip())
            return min(max(score, 1.0), 10.0)
        except:
            return 7.0

    def _calculate_efficiency_score(self, conversation_history: list) -> float:
        """Calculate conversation efficiency"""
        total_turns = len(conversation_history)
        agent_turns = len([r for r in conversation_history if r.get("speaker") == "Agent"])

        if total_turns == 0:
            return 5.0

        # Efficiency formula: favor concise resolutions
        efficiency_ratio = agent_turns / total_turns
        base_score = 10 - (total_turns - 5) * 0.5  # Penalty for long conversations

        return min(max(base_score * efficiency_ratio, 1.0), 10.0)

    def _calculate_overall_quality_score(self, metrics: dict) -> float:
        """Calculate weighted overall quality score"""
        weights = {
            "empathy_indicators": 0.25,
            "resolution_indicators": 0.25,
            "professionalism_score": 0.20,
            "clarity_score": 0.15,
            "efficiency_score": 0.15
        }

        # Normalize indicators to 1-10 scale
        normalized_empathy = min(metrics["empathy_indicators"] * 2, 10.0)
        normalized_resolution = min(metrics["resolution_indicators"] * 2, 10.0)

        weighted_score = (
            weights["empathy_indicators"] * normalized_empathy +
            weights["resolution_indicators"] * normalized_resolution +
            weights["professionalism_score"] * metrics["professionalism_score"] +
            weights["clarity_score"] * metrics["clarity_score"] +
            weights["efficiency_score"] * metrics["efficiency_score"]
        )

        return weighted_score

    def _identify_quality_issues(self, metrics: dict, quality_score: float) -> list:
        """Identify specific quality issues"""
        issues = []

        if metrics["empathy_indicators"] < 2:
            issues.append("Low empathy indicators - responses lack emotional intelligence")
        if metrics["resolution_indicators"] < 2:
            issues.append("Few resolution-focused statements - needs better problem-solving focus")
        if metrics["professionalism_score"] < 6.0:
            issues.append("Professionalism below standard")
        if metrics["clarity_score"] < 6.0:
            issues.append("Response clarity could be improved")
        if metrics["efficiency_score"] < 6.0:
            issues.append("Conversation efficiency needs improvement")
        if quality_score < 6.0:
            issues.append("Overall quality score below acceptable threshold")

        return issues if issues else ["No significant quality issues identified"]

    async def _generate_quality_recommendations(self, metrics: dict, quality_score: float) -> list:
        """Generate improvements recommendations"""
        recommendations = []

        if metrics["empathy_indicators"] < 2:
            recommendations.append("Incorporate more empathy phrases: 'I understand your frustration'")
        if metrics["resolution_indicators"] < 2:
            recommendations.append("Focus on providing concrete solutions and next steps")
        if metrics["professionalism_score"] < 7.0:
            recommendations.append("Maintain consistently professional tone throughout conversation")
        if metrics["clarity_score"] < 7.0:
            recommendations.append("Use clearer language and avoid technical jargon")
        if quality_score < 7.0:
            recommendations.append("Consider additional training in customer service best practices")

        return recommendations if recommendations else ["Continue current practices - quality is acceptable"]

    def _convert_score_to_grade(self, score: float) -> str:
        """Convert numerical score to letter grade"""
        if score >= 9.0:
            return "A+"
        elif score >= 8.0:
            return "A"
        elif score >= 7.0:
            return "B"
        elif score >= 6.0:
            return "C"
        elif score >= 5.0:
            return "D"
        else:
            return "F"

    def _update_performance_metrics(self, quality_score: float, conversation_history: list):
        """Update overall performance metrics"""
        self.performance_metrics["total_conversations"] += 1

        # Rolling average for quality
        current_avg = self.performance_metrics["average_quality_score"]
        new_avg = (current_avg * (self.performance_metrics["total_conversations"] - 1) + quality_score) / self.performance_metrics["total_conversations"]
        self.performance_metrics["average_quality_score"] = new_avg

        # Estimate resolution rate (simplified)
        if len(conversation_history) < 8:  # Assume < 8 turns means resolved
            self.performance_metrics["resolution_rate"] = (self.performance_metrics["resolution_rate"] * (self.performance_metrics["total_conversations"] - 1) + 1) / self.performance_metrics["total_conversations"]
        else:
            self.performance_metrics["resolution_rate"] = (self.performance_metrics["resolution_rate"] * (self.performance_metrics["total_conversations"] - 1)) / self.performance_metrics["total_conversations"]

    def generate_performance_report(self) -> dict:
        """Generate comprehensive performance report"""
        return {
            "period_metrics": self.performance_metrics,
            "quality_distribution": self._get_quality_distribution(),
            "top_issues": self._get_most_common_issues(),
            "trends": self._analyze_quality_trends()
        }

    def _get_quality_distribution(self) -> dict:
        """Get distribution of quality scores"""
        if not self.quality_history:
            return {}

        scores = [entry["quality_score"] for entry in self.quality_history]
        return {
            "excellent": len([s for s in scores if s >= 9.0]),
            "good": len([s for s in scores if 7.0 <= s < 9.0]),
            "average": len([s for s in scores if 6.0 <= s < 7.0]),
            "needs_improvement": len([s for s in scores if s < 6.0])
        }

    def _get_most_common_issues(self) -> list:
        """Get most common quality issues"""
        if not self.quality_history:
            return []

        all_issues = []
        for entry in self.quality_history:
            all_issues.extend(entry["issues"])

        # Count frequency
        from collections import Counter
        issue_counts = Counter(all_issues)
        return issue_counts.most_common(5)

    def _analyze_quality_trends(self) -> dict:
        """Analyze quality trends over time"""
        if len(self.quality_history) < 3:
            return {"trend": "insufficient_data"}

        recent_scores = [entry["quality_score"] for entry in self.quality_history[-10:]]
        if len(recent_scores) < 3:
            return {"trend": "insufficient_data"}

        trend_slope = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]

        if trend_slope > 0.1:
            trend = "improving"
        elif trend_slope < -0.1:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "slope": trend_slope,
            "recent_average": np.mean(recent_scores)
        }
