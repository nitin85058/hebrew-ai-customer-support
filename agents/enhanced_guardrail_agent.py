from crewai import Agent
from litellm import acompletion as completion
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import re
import json
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class EnhancedGuardrailAgent(Agent):
    def __init__(self, llm_model="groq/llama-3.1-8b-instant"):
        super().__init__(
            role="Advanced Safety and Compliance Guardian",
            goal="Ensure all interactions comply with policies, detect sophisticated threats, and adapt to emerging risks",
            backstory="I use machine learning and advanced AI to protect against harmful content, ensure regulatory compliance, and maintain safe interactions across multiple languages.",
            llm=llm_model,
            verbose=True
        )

        self._model_config = {"llm_model": llm_model}

        # Initialize ML models for threat detection
        self.threat_detector = None
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
        self.bias_detector = None

        # Regulatory compliance patterns
        self.regulatory_patterns = self._initialize_regulatory_patterns()

        # Emergency detection keywords in multiple languages
        self.emergency_keywords = self._initialize_emergency_keywords()

        # Safety violation database
        self.violation_history = []

        # Bias detection categories
        self.bias_categories = [
            "gender", "age", "religion", "ethnicity", "disability",
            "political", "sexual_orientation", "income", "geographic"
        ]

    def _initialize_regulatory_patterns(self) -> dict:
        """Initialize patterns for regulatory compliance"""
        return {
            "gdpr": [
                r"delete.*data", r"access.*information", r"personal.*data",
                "מחק.*מידע", "גישה.*למידע", "מידע.*אישי"
            ],
            "pci": [
                r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # Credit card pattern
                r"cvv", r"security.*code", "מספר.*אשראי", "קוד.*בטיחות"
            ],
            "hipaa": [
                r"\b\d{3}-\d{2}-\d{4}\b",  # SSN pattern
                r"medical.*record", r"health.*information", "תיק.*רפואי", "מידע.*בריאות"
            ],
            "anti_money_laundering": [
                r"transfer.*money", r"wire.*transfer", r"bitcoin", r"crypto",
                "העברת.*כספים", "העברה.*בנקאית", "ביטקוין"
            ]
        }

    def _initialize_emergency_keywords(self) -> dict:
        """Initialize emergency keywords in multiple languages"""
        return {
            "en": [
                "emergency", "urgent", "crisis", "immediately", "life threatening",
                "danger", "threat", "attack", "fire", "accident", "medical emergency"
            ],
            "he": [
                "חירום", "דחוף", "משבר", "מיידית", "סכנת חיים",
                "סכנה", "איום", "התקפה", "שריפה", "תאונה", "חירום רפואי"
            ],
            "ar": [
                "طوارئ", "عاجل", "أزمة", "فورا", "تهديد للحياة",
                "خطر", "تهديد", "هجوم", "حريق", "حادث", "طوارئ طبية"
            ]
        }

    async def advanced_content_check(self, text: str, context: dict = None) -> dict:
        """Perform comprehensive content safety analysis"""
        try:
            results = {
                "safe": True,
                "violations": [],
                "severity": "low",
                "recommendations": [],
                "blocked_categories": [],
                "regulatory_flags": [],
                "bias_flags": [],
                "emergency_detected": False
            }

            # Basic safety check (existing functionality)
            basic_safe = self._basic_content_check(text)
            if not basic_safe["safe"]:
                results["safe"] = False
                results["violations"].extend(basic_safe["violations"])
                results["severity"] = basic_safe["severity"]

            # Advanced threat detection
            threat_analysis = await self._advanced_threat_detection(text)
            if not threat_analysis["safe"]:
                results["safe"] = False
                results["violations"].extend(threat_analysis["violations"])
                results["severity"] = max(results["severity"], threat_analysis["severity"],
                                        key=lambda x: ["low", "medium", "high", "critical"].index(x))
                results["blocked_categories"].extend(threat_analysis["categories"])

            # Regulatory compliance check
            regulatory_flags = self._check_regulatory_compliance(text)
            if regulatory_flags:
                results["regulatory_flags"] = regulatory_flags
                if any(flag["severity"] == "high" for flag in regulatory_flags):
                    results["severity"] = "high"

            # Bias detection
            bias_analysis = await self._detect_bias(text)
            if bias_analysis["biased_content"]:
                results["bias_flags"] = bias_analysis["flags"]
                if not results["safe"]:
                    results["severity"] = "medium"

            # Emergency detection
            emergency_status = self._detect_emergency(text)
            if emergency_status["is_emergency"]:
                results["emergency_detected"] = True
                results["recommendations"].append("Emergency protocol activated - prioritize response")

            # Context-aware analysis
            if context:
                contextual_analysis = await self._contextual_risk_assessment(text, context)
                if contextual_analysis["risk_level"] == "high":
                    results["severity"] = "high"
                    results["recommendations"].extend(contextual_analysis["recommendations"])

            # Generate final recommendations
            results["recommendations"].extend(self._generate_safety_recommendations(results))

            # Log violation if unsafe
            if not results["safe"]:
                self._log_violation(text, results)

            return results

        except Exception as e:
            logger.error(f"Advanced content check failed: {e}")
            return {
                "safe": False,
                "violations": ["Safety check error"],
                "severity": "high",
                "recommendations": ["Manual review required"]
            }

    def _basic_content_check(self, text: str) -> dict:
        """Basic content safety check (enhanced version of original)"""
        result = {"safe": True, "violations": [], "severity": "low"}

        # Enhanced forbidden content patterns
        forbidden_patterns = {
            "harmful": [
                r"kill|murder|suicide|self-harm|harm|die|death",
                r"violence|assault|attack|weapon|bomb|explosive",
                "הריגה|התאבדות|פגיעה|מוות|אלימות|התקפה|נשק|פצצה"
            ],
            "hate_speech": [
                r"racist|hate|hateful|bigot|supremacist|nazis",
                r"slur|racism|sexism|homophobia|transphobia",
                "גזען|שנאה|גזענות|סקסיזם|הומופוביה"
            ],
            "harassment": [
                r"stalk|harass|threaten|intimidate|bully",
                r"abuse|misconduct|inappropriate|offensive",
                "רדיפה|הטרדה|איום|הפחדה|התעללות"
            ],
            "spam": [
                r"buy.*now|click.*here|limited.*time|urgent.*offer",
                r"marketing|advertisement|promotion|sales",
                "קנה עכשיו|לחץ כאן|הצעה מוגבלת"
            ]
        }

        severity_scores = {"harmful": "critical", "hate_speech": "high", "harassment": "high", "spam": "low"}

        for category, patterns in forbidden_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text.lower()):
                    result["safe"] = False
                    result["violations"].append(f"{category}: {pattern}")
                    current_severity = severity_scores[category]
                    result["severity"] = max(result["severity"], current_severity,
                                           key=lambda x: ["low", "medium", "high", "critical"].index(x))

        return result

    async def _advanced_threat_detection(self, text: str) -> dict:
        """Advanced threat detection using AI and ML"""
        result = {"safe": True, "violations": [], "severity": "low", "categories": []}

        try:
            # Use LLM for nuanced threat detection
            prompt = f"""
            Analyze the following text for potential threats, manipulation, or harmful intent.
            Consider: social engineering, phishing attempts, harassment, coercion, manipulation, fraud.

            Text: "{text}"

            Respond with JSON only:
            {{
                "threat_detected": boolean,
                "threat_type": "string or array of strings",
                "severity": "low/medium/high/critical",
                "confidence": 0-1
            }}
            """

            response = await completion(
                model=self._model_config["llm_model"],
                messages=[{"role": "user", "content": prompt}]
            )

            analysis = json.loads(response.choices[0].message.content)

            if analysis.get("threat_detected"):
                result["safe"] = False
                result["violations"].append(f"Advanced threat: {analysis.get('threat_type', 'unknown')}")
                result["severity"] = analysis.get("severity", "medium")
                result["categories"] = [analysis.get("threat_type")] if isinstance(analysis.get("threat_type"), str) else analysis.get("threat_type", [])

        except Exception as e:
            logger.warning(f"Advanced threat detection failed: {e}")
            # Fall back to basic checks
            pass

        return result

    def _check_regulatory_compliance(self, text: str) -> list:
        """Check for regulatory compliance issues"""
        flags = []

        for regulation, patterns in self.regulatory_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    flags.append({
                        "regulation": regulation,
                        "pattern": pattern,
                        "severity": "medium" if regulation in ["gdpr", "hipaa"] else "low"
                    })

        return flags

    async def _detect_bias(self, text: str) -> dict:
        """Detect potential bias in communication"""
        result = {"biased_content": False, "flags": []}

        try:
            prompt = f"""
            Analyze the text for potential bias or discriminatory content.
            Look for stereotypes, generalizations, or exclusionary language.

            Text: "{text}"

            Check for bias in: gender, age, religion, ethnicity, disability,
            political views, sexual orientation, income, geographic location.

            Respond with JSON:
            {{
                "biased": boolean,
                "bias_types": ["array of detected bias types"],
                "explanation": "brief explanation"
            }}
            """

            response = await completion(
                model=self._model_config["llm_model"],
                messages=[{"role": "user", "content": prompt}]
            )

            analysis = json.loads(response.choices[0].message.content)

            if analysis.get("biased"):
                result["biased_content"] = True
                result["flags"] = analysis.get("bias_types", [])

        except Exception as e:
            logger.warning(f"Bias detection failed: {e}")

        return result

    def _detect_emergency(self, text: str) -> dict:
        """Detect emergency situations"""
        text_lower = text.lower()

        for language, keywords in self.emergency_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return {
                    "is_emergency": True,
                    "language": language,
                    "keywords_found": [kw for kw in keywords if kw in text_lower]
                }

        return {"is_emergency": False}

    async def _contextual_risk_assessment(self, text: str, context: dict) -> dict:
        """Assess risk based on conversation context"""
        result = {"risk_level": "low", "recommendations": []}

        try:
            conversation_history = context.get("conversation_history", [])
            customer_profile = context.get("customer_profile", {})

            # Analyze interaction patterns
            if len(conversation_history) > 10:
                result["recommendations"].append("Long conversation - consider supervisor escalation")
                result["risk_level"] = "medium"

            # Check for repeated complaints
            customer_messages = [msg for msg in conversation_history if msg.get("speaker") == "Customer"]
            if len(customer_messages) > 5:
                complaint_patterns = sum(1 for msg in customer_messages
                                       if any(word in msg.get("message", "").lower()
                                             for word in ["problem", "issue", "wrong", "bad", "terrible"]))
                if complaint_patterns > 3:
                    result["risk_level"] = "high"
                    result["recommendations"].append("High complaint frequency - activate retention protocol")

            # VIP or at-risk customer detection
            if customer_profile.get("vip_status") or customer_profile.get("churn_risk") == "high":
                result["recommendations"].append("VIP/at-risk customer - prioritize and offer premium support")

        except Exception as e:
            logger.warning(f"Contextual risk assessment failed: {e}")

        return result

    def _generate_safety_recommendations(self, results: dict) -> list:
        """Generate recommendations based on safety analysis"""
        recommendations = []

        if not results["safe"]:
            recommendations.append("Block content and flag for review")

        if results["severity"] == "critical":
            recommendations.extend([
                "Escalate to security team immediately",
                "Log incident for compliance reporting",
                "Consider legal notification requirements"
            ])

        if results["regulatory_flags"]:
            recommendations.append("Regulatory compliance review required")

        if results["emergency_detected"]:
            recommendations.append("Activate emergency response protocol")

        if results["bias_flags"]:
            recommendations.append("Review for bias and cultural sensitivity")

        return recommendations

    def _log_violation(self, text: str, results: dict):
        """Log safety violations for monitoring"""
        violation = {
            "timestamp": datetime.now().isoformat(),
            "text_snippet": text[:100] + "..." if len(text) > 100 else text,
            "violations": results["violations"],
            "severity": results["severity"],
            "categories": results["blocked_categories"]
        }

        self.violation_history.append(violation)

        # Keep only last 1000 violations for memory efficiency
        if len(self.violation_history) > 1000:
            self.violation_history = self.violation_history[-1000:]

    def get_violation_report(self) -> dict:
        """Generate violation statistics report"""
        if not self.violation_history:
            return {"message": "No violations recorded"}

        total_violations = len(self.violation_history)
        severity_counts = {}
        category_counts = {}

        for violation in self.violation_history:
            severity = violation["severity"]
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

            for category in violation.get("categories", []):
                category_counts[category] = category_counts.get(category, 0) + 1

        return {
            "total_violations": total_violations,
            "severity_distribution": severity_counts,
            "category_distribution": category_counts,
            "recent_violations": self.violation_history[-10:]  # Last 10 violations
        }

    def train_threat_model(self, training_data: pd.DataFrame):
        """Train ML model for threat detection"""
        try:
            # Simple training data structure expected: columns=['text', 'is_threat', 'threat_type']
            X = self.vectorizer.fit_transform(training_data['text'])
            y = training_data['is_threat'].astype(int)

            self.threat_detector = RandomForestClassifier(n_estimators=100, random_state=42)
            self.threat_detector.fit(X, y)

            logger.info("Threat detection model trained successfully")
        except Exception as e:
            logger.error(f"Model training failed: {e}")

    def update_patterns(self, new_patterns: dict):
        """Update detection patterns dynamically"""
        for category, patterns in new_patterns.items():
            if category in self.regulatory_patterns:
                self.regulatory_patterns[category].extend(patterns)
            else:
                self.regulatory_patterns[category] = patterns

        logger.info(f"Patterns updated: {list(new_patterns.keys())}")
