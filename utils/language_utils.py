from googletrans import Translator
from langdetect import detect, LangDetectError
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class LanguageSupport:
    """Advanced language support utilities for multi-lingual customer service"""

    def __init__(self):
        self.translator = Translator()
        self.supported_languages = {
            "en": {"name": "English", "native": "English", "rtl": False},
            "he": {"name": "Hebrew", "native": "עברית", "rtl": True},
            "ar": {"name": "Arabic", "native": "العربية", "rtl": True},
            "fr": {"name": "French", "native": "Français", "rtl": False},
            "de": {"name": "German", "native": "Deutsch", "rtl": False},
            "es": {"name": "Spanish", "native": "Español", "rtl": False},
            "ru": {"name": "Russian", "native": "Русский", "rtl": False},
            "zh": {"name": "Chinese", "native": "中文", "rtl": False},
            "yi": {"name": "Yiddish", "native": "ייִדיש", "rtl": True}
        }

        # Cultural adaptation patterns
        self.cultural_patterns = self._initialize_cultural_patterns()

        # Region-specific terminology
        self.regional_terms = self._initialize_regional_terms()

    def _initialize_cultural_patterns(self) -> Dict[str, Dict]:
        """Initialize cultural communication patterns"""
        return {
            "he": {
                "greeting_formal": ["שלום", "שלום רב"],
                "greeting_casual": ["שלום", "היי"],
                "politeness_particles": ["בבקשה", "תודה רבה", "אנא"],
                "gratitude_expressions": ["תודה רבה", "תודה מאוד", "אני מודה לך"],
                "apology_forms": ["סליחה", "מצטער", "אני מתנצל"],
                "urgency_markers": ["מיידית", "דחוף", "בהקדם האפשרי"]
            },
            "ar": {
                "greeting_formal": ["السلام عليكم", "مرحباً"],
                "greeting_casual": ["مرحبا", "أهلاً وسهلاً"],
                "politeness_particles": ["من فضلك", "شكراً جزيلاً", "رجاءً"],
                "gratitude_expressions": ["شكراً لك", "أشكرك"],
                "apology_forms": ["عذراً", "آسف"],
                "urgency_markers": ["عاجل", "فوراً", "حالاً"]
            }
        }

    def _initialize_regional_terms(self) -> Dict[str, Dict]:
        """Initialize region-specific terminology"""
        return {
            "he": {
                "TV": "טלוויזיה",
                "subscription": "מנוי",
                "cancel": "ביטול",
                "service": "שירות",
                "customer service": "שירות לקוחות",
                "billing": "חיוב",
                "complaint": "תלונה",
                "refund": "החזר",
                "technical support": "תמיכה טכנית"
            },
            "ar": {
                "TV": "التلفاز",
                "subscription": "الاشتراك",
                "cancel": "إلغاء",
                "service": "الخدمة",
                "customer service": "خدمة العملاء",
                "billing": "الفواتير",
                "complaint": "الشكوى",
                "refund": "الاسترداد",
                "technical support": "الدعم الفني"
            }
        }

    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect the language of input text"""
        try:
            # Clean text for better detection
            clean_text = self._clean_text_for_detection(text)

            if not clean_text:
                return "en", 0.0

            lang = detect(clean_text)
            confidence = 0.8  # Default confidence for langdetect

            # Normalize language codes
            normalized_lang = self._normalize_language_code(lang)

            return normalized_lang, confidence

        except LangDetectError:
            logger.warning(f"Could not detect language for text: {text[:50]}...")
            return "en", 0.0
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return "en", 0.0

    def _clean_text_for_detection(self, text: str) -> str:
        """Clean text for better language detection"""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

        # Remove special characters but keep language-specific characters
        text = re.sub(r'[^\w\s\u0590-\u05FF\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text.strip()

    def _normalize_language_code(self, lang_code: str) -> str:
        """Normalize language codes to our supported formats"""
        normalization_map = {
            "iw": "he",  # Hebrew
            "heb": "he",
            "ara": "ar",
            "arabic": "ar",
            "chi": "zh",
            "zho": "zh"
        }

        return normalization_map.get(lang_code.lower(), lang_code.lower())

    async def translate_text(self, text: str, source_lang: str = None, target_lang: str = "he") -> Dict:
        """Translate text between languages"""
        try:
            result = {
                "translated_text": "",
                "source_language": source_lang or "auto",
                "target_language": target_lang,
                "confidence": 0.0,
                "success": False
            }

            if not text.strip():
                return result

            # Detect source language if not provided
            if not source_lang or source_lang == "auto":
                detected_lang, confidence = self.detect_language(text)
                result["source_language"] = detected_lang

                # Don't translate if already in target language
                if detected_lang == target_lang:
                    result["translated_text"] = text
                    result["confidence"] = confidence
                    result["success"] = True
                    return result

            # Perform translation
            translation = self.translator.translate(text, src=source_lang, dest=target_lang)

            result["translated_text"] = translation.text
            result["confidence"] = 0.9  # Translation confidence
            result["success"] = True

            return result

        except Exception as e:
            logger.error(f"Translation error: {e}")
            return {
                "translated_text": text,  # Return original text on failure
                "source_language": source_lang or "unknown",
                "target_language": target_lang,
                "confidence": 0.0,
                "success": False
            }

    def adapt_cultural_context(self, text: str, target_lang: str, context: str = "customer_service") -> str:
        """Adapt text for cultural and regional context"""
        if target_lang not in self.cultural_patterns:
            return text

        patterns = self.cultural_patterns[target_lang]

        # Adapt politeness and formality based on context
        if context == "customer_service":
            # Add appropriate politeness particles
            if target_lang == "he":
                text = self._add_hebrew_courtesy(text)
            elif target_lang == "ar":
                text = self._add_arabic_courtesy(text)

        return text

    def _add_hebrew_courtesy(self, text: str) -> str:
        """Add Hebrew courtesy expressions"""
        # Add "בבקשה" (please) if offering help
        if any(word in text.lower() for word in ["help", "assist", "can", "able", "will"]):
            if "בבקשה" not in text:
                text = "בבקשה " + text.lower()

        # Add "תודה" (thank you) at appropriate points
        if any(word in text.lower() for word in ["appreciate", "grateful", "thanks"]):
            if "תודה" not in text:
                text = text.replace(".", " תודה.")

        return text

    def _add_arabic_courtesy(self, text: str) -> str:
        """Add Arabic courtesy expressions"""
        # This would implement Arabic-specific courtesy adaptations
        return text

    def detect_dialect(self, text: str, language: str) -> Dict:
        """Detect specific dialects within a language"""
        result = {
            "dialect": "standard",
            "confidence": 0.0,
            "region": "unknown"
        }

        if language == "he":
            return self._detect_hebrew_dialect(text)
        elif language == "ar":
            return self._detect_arabic_dialect(text)
        elif language == "zh":
            return self._detect_chinese_dialect(text)

        return result

    def _detect_hebrew_dialect(self, text: str) -> Dict:
        """Detect Hebrew dialect variations"""
        # Simple dialect detection patterns
        ashkenazi_patterns = ["ייִדיש", "אַשכּנז", "פולין"]
        mizrahi_patterns = ["מזרחי", "ספרד", "מרוקו"]
        modern_patterns = ["טכנולוגיה", "מחשב", "אינטרנט", "סמארטפון"]

        ashkenazi_score = sum(1 for pattern in ashkenazi_patterns if pattern in text)
        mizrahi_score = sum(1 for pattern in mizrahi_patterns if pattern in text)
        modern_score = sum(1 for pattern in modern_patterns if pattern in text)

        if modern_score > max(ashkenazi_score, mizrahi_score):
            return {"dialect": "modern_israeli", "confidence": 0.8, "region": "israel"}
        elif ashkenazi_score > mizrahi_score:
            return {"dialect": "ashkenazi", "confidence": 0.7, "region": "eastern_europe"}
        elif mizrahi_score > ashkenazi_score:
            return {"dialect": "mizrahi", "confidence": 0.7, "region": "middle_east"}
        else:
            return {"dialect": "standard_modern_hebrew", "confidence": 0.6, "region": "israel"}

    def _detect_arabic_dialect(self, text: str) -> Dict:
        """Detect Arabic dialect variations"""
        # Basic Arabic dialect detection
        dialects = {
            "modern_standard_arabic": ["MSA", "فصحى"],
            "egyptian_arabic": ["مصر", "القاهرة", "إحنا"],
            "gulf_arabic": ["خليج", "سعودية", "إمارات"],
            "levantine_arabic": ["شام", "لبنان", "سوريا", "أردن"],
            "maghrebi_arabic": ["مغرب", "تونس", "الجزائر"]
        }

        # Simplified detection - would need more sophisticated analysis
        return {"dialect": "modern_standard_arabic", "confidence": 0.5, "region": "general"}

    def _detect_chinese_dialect(self, text: str) -> Dict:
        """Detect Chinese dialect variations"""
        # Basic Chinese dialect detection
        return {"dialect": "mandarin", "confidence": 0.5, "region": "china"}

    def get_language_info(self, lang_code: str) -> Dict:
        """Get detailed information about a language"""
        return self.supported_languages.get(lang_code, {
            "name": "Unknown",
            "native": lang_code,
            "rtl": False
        })

    def is_rtl_language(self, lang_code: str) -> bool:
        """Check if language uses right-to-left script"""
        return self.supported_languages.get(lang_code, {}).get("rtl", False)

    def get_regional_term(self, term: str, language: str) -> str:
        """Get regional equivalent of a term"""
        if language in self.regional_terms:
            return self.regional_terms[language].get(term, term)
        return term
