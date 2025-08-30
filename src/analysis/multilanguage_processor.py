#!/usr/bin/env python3
"""Multi-Language Content Processing Engine
Advanced language detection, translation, and cross-linguistic analysis

Author: Advanced IT Specialist
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LanguageDetectionResult:
    """Result of language detection analysis"""

    document_id: str
    primary_language: str
    language_confidence: float
    detected_languages: dict[str, float]  # language_code: confidence
    script_type: str  # latin, cyrillic, arabic, chinese, etc.
    mixed_language_segments: list[dict[str, Any]]
    translation_complexity: str  # simple, moderate, complex, highly_complex


@dataclass
class TranslationResult:
    """Result of document translation"""

    document_id: str
    source_language: str
    target_language: str
    translated_content: str
    translation_quality_score: float
    translation_method: str  # api, rule_based, neural, hybrid
    cultural_adaptations: list[str]
    untranslatable_terms: list[str]
    translation_notes: list[str]


@dataclass
class CrossLinguisticMatch:
    """Cross-linguistic content match"""

    source_doc_id: str
    target_doc_id: str
    source_language: str
    target_language: str
    semantic_similarity: float
    structural_similarity: float
    cultural_context_alignment: float
    match_type: str  # translation, parallel_content, cultural_variant


@dataclass
class MultilingualCorpusAnalysis:
    """Analysis of multilingual document corpus"""

    total_documents: int
    language_distribution: dict[str, int]
    script_distribution: dict[str, int]
    cross_linguistic_matches: list[CrossLinguisticMatch]
    cultural_themes: dict[str, list[str]]
    translation_recommendations: list[str]
    linguistic_insights: list[str]


class MultiLanguageProcessor:
    """Advanced multi-language content processing engine"""

    def __init__(self):
        # Language detection patterns and features
        self.language_patterns = {
            "english": {
                "common_words": ["the", "and", "of", "to", "a", "in", "is", "it", "you", "that"],
                "char_patterns": r"[a-zA-Z\s\.,;:!?\-\'\"]+",
                "script": "latin",
            },
            "spanish": {
                "common_words": ["el", "la", "de", "que", "y", "a", "en", "un", "es", "se"],
                "char_patterns": r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ\s\.,;:!?\-\'\"]+",
                "script": "latin",
            },
            "french": {
                "common_words": ["le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir"],
                "char_patterns": r"[a-zA-ZàâäçéèêëïîôùûüÿÀÂÄÇÉÈÊËÏÎÔÙÛÜŸ\s\.,;:!?\-\'\"]+",
                "script": "latin",
            },
            "german": {
                "common_words": [
                    "der",
                    "die",
                    "und",
                    "in",
                    "den",
                    "von",
                    "zu",
                    "das",
                    "mit",
                    "sich",
                ],
                "char_patterns": r"[a-zA-ZäöüßÄÖÜ\s\.,;:!?\-\'\"]+",
                "script": "latin",
            },
            "russian": {
                "common_words": ["в", "и", "не", "на", "я", "быть", "то", "он", "с", "а"],
                "char_patterns": r"[а-яёА-ЯЁ\s\.,;:!?\-\'\"]+",
                "script": "cyrillic",
            },
            "chinese": {
                "common_words": ["的", "一", "是", "了", "我", "不", "人", "在", "他", "有"],
                "char_patterns": r"[\u4e00-\u9fff\s\.,;:!?\-\'\"]+",
                "script": "chinese",
            },
            "arabic": {
                "common_words": ["في", "من", "إلى", "على", "أن", "هذا", "كان", "لا", "ما", "هو"],
                "char_patterns": r"[\u0600-\u06ff\s\.,;:!?\-\'\"]+",
                "script": "arabic",
            },
            "japanese": {
                "common_words": ["の", "に", "は", "を", "た", "が", "で", "て", "と", "し"],
                "char_patterns": r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\s\.,;:!?\-\'\"]+",
                "script": "japanese",
            },
        }

        # Translation services configuration
        self.translation_services = {
            "google_translate": {
                "api_endpoint": "https://translate.googleapis.com/translate_a/single",
                "supported_languages": list(self.language_patterns.keys()),
                "quality_rating": 0.8,
                "cost_per_char": 0.00002,
            },
            "deepl": {
                "api_endpoint": "https://api-free.deepl.com/v2/translate",
                "supported_languages": ["english", "german", "french", "spanish", "italian"],
                "quality_rating": 0.9,
                "cost_per_char": 0.00002,
            },
        }

        # Cultural context mappings
        self.cultural_contexts = {
            "english": {
                "regions": ["US", "UK", "AU", "CA"],
                "cultural_themes": ["individualism", "democracy", "capitalism", "common_law"],
                "date_format": "MM/DD/YYYY",
                "measurement_system": "imperial",
            },
            "spanish": {
                "regions": ["ES", "MX", "AR", "CO", "PE"],
                "cultural_themes": ["family", "catholicism", "collectivism", "civil_law"],
                "date_format": "DD/MM/YYYY",
                "measurement_system": "metric",
            },
            "chinese": {
                "regions": ["CN", "TW", "HK", "SG"],
                "cultural_themes": ["confucianism", "collectivism", "hierarchy", "harmony"],
                "date_format": "YYYY/MM/DD",
                "measurement_system": "metric",
            },
            "arabic": {
                "regions": ["SA", "EG", "AE", "MA", "IQ"],
                "cultural_themes": ["islam", "family", "honor", "hospitality"],
                "date_format": "DD/MM/YYYY",
                "measurement_system": "metric",
            },
        }

        # Complex translation indicators
        self.complexity_indicators = {
            "idioms": r"\b(?:piece of cake|break a leg|hit the nail|spill the beans)\b",
            "cultural_references": r"\b(?:hollywood|wall street|silicon valley|broadway)\b",
            "technical_terms": r"\b[A-Z]{2,}(?:[a-z]*[A-Z]*)*\b",
            "proper_nouns": r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",
            "domain_specific": r"\b(?:algorithm|methodology|paradigm|framework)\b",
        }

    async def detect_language(self, document: Any) -> LanguageDetectionResult:
        """Detect language(s) in document content"""
        content = self._extract_document_text(document)
        document_id = getattr(document, "document_id", "unknown")

        logger.info(f"Detecting language for document: {document_id}")

        # Language detection analysis
        language_scores = {}

        for lang_code, lang_data in self.language_patterns.items():
            score = self._calculate_language_score(content, lang_data)
            if score > 0.1:  # Minimum threshold
                language_scores[lang_code] = score

        if not language_scores:
            # Fallback detection
            primary_language = "english"
            language_confidence = 0.5
        else:
            # Determine primary language
            primary_language = max(language_scores, key=language_scores.get)
            language_confidence = language_scores[primary_language]

        # Detect script type
        script_type = self.language_patterns[primary_language]["script"]

        # Detect mixed language segments
        mixed_segments = self._detect_mixed_language_segments(content, language_scores)

        # Assess translation complexity
        translation_complexity = self._assess_translation_complexity(content, primary_language)

        result = LanguageDetectionResult(
            document_id=document_id,
            primary_language=primary_language,
            language_confidence=language_confidence,
            detected_languages=language_scores,
            script_type=script_type,
            mixed_language_segments=mixed_segments,
            translation_complexity=translation_complexity,
        )

        logger.info(
            f"Language detection completed: {primary_language} (confidence: {language_confidence:.2f})"
        )
        return result

    async def translate_document(
        self, document: Any, target_language: str, translation_service: str = "google_translate"
    ) -> TranslationResult:
        """Translate document to target language"""
        content = self._extract_document_text(document)
        document_id = getattr(document, "document_id", "unknown")

        # Detect source language first
        lang_detection = await self.detect_language(document)
        source_language = lang_detection.primary_language

        logger.info(
            f"Translating document {document_id} from {source_language} to {target_language}"
        )

        if source_language == target_language:
            return TranslationResult(
                document_id=document_id,
                source_language=source_language,
                target_language=target_language,
                translated_content=content,
                translation_quality_score=1.0,
                translation_method="no_translation_needed",
                cultural_adaptations=[],
                untranslatable_terms=[],
                translation_notes=["Source and target languages are identical"],
            )

        # Perform translation
        translated_content = await self._perform_translation(
            content, source_language, target_language, translation_service
        )

        # Assess translation quality
        quality_score = self._assess_translation_quality(
            content, translated_content, source_language, target_language
        )

        # Identify cultural adaptations needed
        cultural_adaptations = self._identify_cultural_adaptations(
            content, source_language, target_language
        )

        # Identify untranslatable terms
        untranslatable_terms = self._identify_untranslatable_terms(content, source_language)

        # Generate translation notes
        translation_notes = self._generate_translation_notes(
            content, source_language, target_language, lang_detection.translation_complexity
        )

        result = TranslationResult(
            document_id=document_id,
            source_language=source_language,
            target_language=target_language,
            translated_content=translated_content,
            translation_quality_score=quality_score,
            translation_method=translation_service,
            cultural_adaptations=cultural_adaptations,
            untranslatable_terms=untranslatable_terms,
            translation_notes=translation_notes,
        )

        logger.info(f"Translation completed with quality score: {quality_score:.2f}")
        return result

    async def find_cross_linguistic_matches(
        self, documents: list[Any], similarity_threshold: float = 0.7
    ) -> list[CrossLinguisticMatch]:
        """Find cross-linguistic matches between documents"""
        logger.info(f"Finding cross-linguistic matches among {len(documents)} documents")

        matches = []

        # Detect languages for all documents
        language_detections = {}
        for doc in documents:
            detection = await self.detect_language(doc)
            language_detections[doc] = detection

        # Compare documents across different languages
        for i, doc1 in enumerate(documents):
            for doc2 in documents[i + 1 :]:
                detection1 = language_detections[doc1]
                detection2 = language_detections[doc2]

                # Only compare documents in different languages
                if detection1.primary_language != detection2.primary_language:
                    match = await self._analyze_cross_linguistic_match(
                        doc1, doc2, detection1, detection2, similarity_threshold
                    )
                    if match:
                        matches.append(match)

        logger.info(f"Found {len(matches)} cross-linguistic matches")
        return matches

    async def analyze_multilingual_corpus(self, documents: list[Any]) -> MultilingualCorpusAnalysis:
        """Analyze multilingual document corpus"""
        logger.info(f"Analyzing multilingual corpus of {len(documents)} documents")

        # Detect languages for all documents
        language_detections = []
        for doc in documents:
            detection = await self.detect_language(doc)
            language_detections.append(detection)

        # Analyze language distribution
        language_distribution = Counter(d.primary_language for d in language_detections)
        script_distribution = Counter(d.script_type for d in language_detections)

        # Find cross-linguistic matches
        cross_linguistic_matches = await self.find_cross_linguistic_matches(documents)

        # Analyze cultural themes by language
        cultural_themes = self._analyze_cultural_themes_by_language(documents, language_detections)

        # Generate translation recommendations
        translation_recommendations = self._generate_translation_recommendations(
            language_distribution, cross_linguistic_matches
        )

        # Generate linguistic insights
        linguistic_insights = self._generate_linguistic_insights(
            language_detections, cross_linguistic_matches
        )

        analysis = MultilingualCorpusAnalysis(
            total_documents=len(documents),
            language_distribution=dict(language_distribution),
            script_distribution=dict(script_distribution),
            cross_linguistic_matches=cross_linguistic_matches,
            cultural_themes=cultural_themes,
            translation_recommendations=translation_recommendations,
            linguistic_insights=linguistic_insights,
        )

        logger.info("Multilingual corpus analysis completed")
        return analysis

    # Language detection helper methods

    def _calculate_language_score(self, content: str, lang_data: dict[str, Any]) -> float:
        """Calculate language score for content"""
        score = 0.0
        content_lower = content.lower()

        # Check common words
        common_words = lang_data["common_words"]
        word_matches = sum(1 for word in common_words if f" {word} " in f" {content_lower} ")
        word_score = word_matches / len(common_words)
        score += word_score * 0.6

        # Check character patterns
        char_pattern = lang_data["char_patterns"]
        char_matches = len(re.findall(char_pattern, content))
        total_chars = len(content)
        if total_chars > 0:
            char_score = char_matches / total_chars
            score += char_score * 0.4

        return min(1.0, score)

    def _detect_mixed_language_segments(
        self, content: str, language_scores: dict[str, float]
    ) -> list[dict[str, Any]]:
        """Detect segments with mixed languages"""
        mixed_segments = []

        # Split content into paragraphs
        paragraphs = content.split("\n\n")

        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.strip()) < 50:  # Skip short paragraphs
                continue

            # Calculate language scores for paragraph
            para_scores = {}
            for lang_code, lang_data in self.language_patterns.items():
                if lang_code in language_scores:  # Only check detected languages
                    para_score = self._calculate_language_score(paragraph, lang_data)
                    if para_score > 0.2:
                        para_scores[lang_code] = para_score

            # Check if multiple languages detected
            if len(para_scores) > 1:
                mixed_segments.append(
                    {
                        "segment_index": i,
                        "content_preview": paragraph[:100] + "...",
                        "detected_languages": para_scores,
                        "segment_type": "mixed_paragraph",
                    }
                )

        return mixed_segments

    def _assess_translation_complexity(self, content: str, source_language: str) -> str:
        """Assess translation complexity of content"""
        complexity_score = 0

        # Check for complexity indicators
        for indicator_type, pattern in self.complexity_indicators.items():
            matches = len(re.findall(pattern, content, re.IGNORECASE))
            if matches > 0:
                complexity_score += matches * 0.1

        # Language-specific complexity factors
        if source_language in ["chinese", "arabic", "japanese"]:
            complexity_score += 0.3  # Script differences add complexity

        if source_language in ["german", "finnish"]:
            complexity_score += 0.2  # Complex grammar

        # Determine complexity level
        if complexity_score < 0.3:
            return "simple"
        if complexity_score < 0.6:
            return "moderate"
        if complexity_score < 1.0:
            return "complex"
        return "highly_complex"

    # Translation helper methods

    async def _perform_translation(
        self, content: str, source_lang: str, target_lang: str, service: str
    ) -> str:
        """Perform actual translation using specified service"""
        if service not in self.translation_services:
            logger.error(f"Translation service {service} not available")
            return content  # Return original if service unavailable

        try:
            if service == "google_translate":
                return await self._google_translate(content, source_lang, target_lang)
            if service == "deepl":
                return await self._deepl_translate(content, source_lang, target_lang)
            # Fallback: simple rule-based translation (mock)
            return self._rule_based_translation(content, source_lang, target_lang)

        except Exception as e:
            logger.error(f"Translation failed: {e!s}")
            return content  # Return original on error

    async def _google_translate(self, content: str, source_lang: str, target_lang: str) -> str:
        """Translate using Google Translate API (mock implementation)"""
        # Note: This is a simplified mock implementation
        # In production, you would use the actual Google Translate API

        # Mock translation for demonstration
        if source_lang == "spanish" and target_lang == "english":
            # Simple Spanish to English translations
            translations = {
                "el": "the",
                "la": "the",
                "de": "of",
                "que": "that",
                "y": "and",
                "a": "to",
                "en": "in",
                "un": "a",
                "es": "is",
                "se": "it",
            }

            words = content.split()
            translated_words = [translations.get(word.lower(), word) for word in words]
            return " ".join(translated_words)

        # For other language pairs, return annotated original
        return f"[TRANSLATED FROM {source_lang.upper()}] {content}"

    async def _deepl_translate(self, content: str, source_lang: str, target_lang: str) -> str:
        """Translate using DeepL API (mock implementation)"""
        # Mock implementation
        return f"[DEEPL TRANSLATION FROM {source_lang.upper()}] {content}"

    def _rule_based_translation(self, content: str, source_lang: str, target_lang: str) -> str:
        """Simple rule-based translation fallback"""
        return f"[RULE-BASED TRANSLATION FROM {source_lang.upper()}] {content}"

    def _assess_translation_quality(
        self, original: str, translated: str, source_lang: str, target_lang: str
    ) -> float:
        """Assess quality of translation"""
        quality_score = 0.7  # Base score

        # Length comparison (significant changes might indicate issues)
        length_ratio = len(translated) / max(1, len(original))
        if 0.5 <= length_ratio <= 2.0:  # Reasonable length change
            quality_score += 0.1
        else:
            quality_score -= 0.2

        # Check for untranslated segments
        if "[TRANSLATED FROM" in translated:
            quality_score -= 0.2  # Mock translation indicator

        # Language pair difficulty adjustment
        difficult_pairs = [
            ("chinese", "english"),
            ("arabic", "english"),
            ("japanese", "english"),
            ("english", "chinese"),
        ]

        if (source_lang, target_lang) in difficult_pairs:
            quality_score -= 0.1

        return max(0.0, min(1.0, quality_score))

    def _identify_cultural_adaptations(
        self, content: str, source_lang: str, target_lang: str
    ) -> list[str]:
        """Identify needed cultural adaptations"""
        adaptations = []

        source_context = self.cultural_contexts.get(source_lang, {})
        target_context = self.cultural_contexts.get(target_lang, {})

        # Date format adaptations
        if source_context.get("date_format") != target_context.get("date_format"):
            if re.search(r"\d{1,2}/\d{1,2}/\d{4}", content):
                adaptations.append("Date format conversion needed")

        # Measurement system adaptations
        if source_context.get("measurement_system") != target_context.get("measurement_system"):
            if re.search(r"\d+\s*(?:feet|inches|miles|pounds)", content, re.IGNORECASE):
                adaptations.append("Imperial to metric conversion needed")

        # Cultural reference adaptations
        source_themes = set(source_context.get("cultural_themes", []))
        target_themes = set(target_context.get("cultural_themes", []))

        if not source_themes.intersection(target_themes):
            adaptations.append("Cultural context explanation needed")

        return adaptations

    def _identify_untranslatable_terms(self, content: str, source_lang: str) -> list[str]:
        """Identify terms that may be difficult to translate"""
        untranslatable = []

        # Language-specific untranslatable patterns
        if source_lang == "german":
            # German compound words
            compound_pattern = r"\b[A-Z][a-z]*(?:[A-Z][a-z]*){2,}\b"
            compounds = re.findall(compound_pattern, content)
            untranslatable.extend(compounds[:5])  # Limit to first 5

        elif source_lang == "japanese":
            # Honorifics and cultural terms
            honorifics = ["san", "sama", "kun", "chan", "sensei", "senpai"]
            for honorific in honorifics:
                if honorific in content.lower():
                    untranslatable.append(honorific)

        elif source_lang == "chinese":
            # Cultural concepts
            cultural_terms = ["guanxi", "mianzi", "feng shui", "qi"]
            for term in cultural_terms:
                if term in content.lower():
                    untranslatable.append(term)

        # General untranslatable patterns
        # Idioms and expressions
        idiom_patterns = [
            r"\b(?:break a leg|piece of cake|hit the nail)\b",
            r"\b(?:cuando las ranas críen pelo|estar en las nubes)\b",  # Spanish idioms
        ]

        for pattern in idiom_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            untranslatable.extend(matches)

        return list(set(untranslatable))  # Remove duplicates

    def _generate_translation_notes(
        self, content: str, source_lang: str, target_lang: str, complexity: str
    ) -> list[str]:
        """Generate helpful translation notes"""
        notes = []

        notes.append(f"Translation complexity assessed as: {complexity}")

        if complexity in ["complex", "highly_complex"]:
            notes.append("Professional human review recommended for accuracy")

        # Language-specific notes
        if source_lang == "chinese" and target_lang == "english":
            notes.append(
                "Chinese characters may have multiple meanings - context verification needed"
            )

        if source_lang in ["arabic", "hebrew"]:
            notes.append("Right-to-left script - formatting considerations required")

        if len(content) > 5000:
            notes.append(
                "Large document - consider section-by-section translation for better accuracy"
            )

        return notes

    # Cross-linguistic analysis helper methods

    async def _analyze_cross_linguistic_match(
        self,
        doc1: Any,
        doc2: Any,
        detection1: LanguageDetectionResult,
        detection2: LanguageDetectionResult,
        similarity_threshold: float,
    ) -> CrossLinguisticMatch | None:
        """Analyze potential cross-linguistic match between two documents"""
        content1 = self._extract_document_text(doc1)
        content2 = self._extract_document_text(doc2)

        # Calculate semantic similarity (simplified)
        semantic_similarity = self._calculate_semantic_similarity(content1, content2)

        if semantic_similarity < similarity_threshold:
            return None

        # Calculate structural similarity
        structural_similarity = self._calculate_structural_similarity(doc1, doc2)

        # Calculate cultural context alignment
        cultural_alignment = self._calculate_cultural_alignment(
            detection1.primary_language, detection2.primary_language
        )

        # Determine match type
        match_type = self._determine_match_type(
            semantic_similarity, structural_similarity, cultural_alignment
        )

        return CrossLinguisticMatch(
            source_doc_id=getattr(doc1, "document_id", "unknown"),
            target_doc_id=getattr(doc2, "document_id", "unknown"),
            source_language=detection1.primary_language,
            target_language=detection2.primary_language,
            semantic_similarity=semantic_similarity,
            structural_similarity=structural_similarity,
            cultural_context_alignment=cultural_alignment,
            match_type=match_type,
        )

    def _calculate_semantic_similarity(self, content1: str, content2: str) -> float:
        """Calculate semantic similarity between texts (simplified)"""
        # Simplified implementation using word overlap
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _calculate_structural_similarity(self, doc1: Any, doc2: Any) -> float:
        """Calculate structural similarity between documents"""
        # Compare document metadata structure
        similarity_score = 0.0

        # Compare presence of similar fields
        fields_to_compare = ["title", "authors", "abstract", "keywords"]
        common_fields = 0
        total_fields = 0

        for field in fields_to_compare:
            has_field1 = hasattr(doc1, field) and getattr(doc1, field)
            has_field2 = hasattr(doc2, field) and getattr(doc2, field)

            if has_field1 and has_field2:
                common_fields += 1
            total_fields += 1 if has_field1 or has_field2 else 0

        if total_fields > 0:
            similarity_score = common_fields / total_fields

        return similarity_score

    def _calculate_cultural_alignment(self, lang1: str, lang2: str) -> float:
        """Calculate cultural context alignment between languages"""
        context1 = self.cultural_contexts.get(lang1, {})
        context2 = self.cultural_contexts.get(lang2, {})

        if not context1 or not context2:
            return 0.5  # Neutral if no context data

        # Compare cultural themes
        themes1 = set(context1.get("cultural_themes", []))
        themes2 = set(context2.get("cultural_themes", []))

        if themes1 and themes2:
            theme_overlap = len(themes1.intersection(themes2)) / len(themes1.union(themes2))
        else:
            theme_overlap = 0.0

        # Compare regions (for language variants)
        regions1 = set(context1.get("regions", []))
        regions2 = set(context2.get("regions", []))

        if regions1 and regions2:
            region_overlap = len(regions1.intersection(regions2)) / len(regions1.union(regions2))
        else:
            region_overlap = 0.0

        return (theme_overlap + region_overlap) / 2

    def _determine_match_type(
        self, semantic_sim: float, structural_sim: float, cultural_align: float
    ) -> str:
        """Determine type of cross-linguistic match"""
        if semantic_sim > 0.9 and structural_sim > 0.8:
            return "translation"
        if semantic_sim > 0.8 and cultural_align > 0.6:
            return "cultural_variant"
        if semantic_sim > 0.7:
            return "parallel_content"
        return "related_content"

    # Corpus analysis helper methods

    def _analyze_cultural_themes_by_language(
        self, documents: list[Any], language_detections: list[LanguageDetectionResult]
    ) -> dict[str, list[str]]:
        """Analyze cultural themes by language"""
        cultural_themes = defaultdict(list)

        for doc, detection in zip(documents, language_detections, strict=False):
            lang = detection.primary_language
            content = self._extract_document_text(doc)

            # Extract themes based on cultural context
            context = self.cultural_contexts.get(lang, {})
            themes = context.get("cultural_themes", [])

            for theme in themes:
                if theme.lower() in content.lower():
                    cultural_themes[lang].append(theme)

        # Remove duplicates and get top themes per language
        for lang in cultural_themes:
            theme_counts = Counter(cultural_themes[lang])
            cultural_themes[lang] = [theme for theme, count in theme_counts.most_common(5)]

        return dict(cultural_themes)

    def _generate_translation_recommendations(
        self, language_distribution: Counter, cross_linguistic_matches: list[CrossLinguisticMatch]
    ) -> list[str]:
        """Generate translation recommendations"""
        recommendations = []

        # Recommend translating from less common to more common languages
        if len(language_distribution) > 1:
            most_common_lang = language_distribution.most_common(1)[0][0]
            other_languages = [
                lang for lang, count in language_distribution.items() if lang != most_common_lang
            ]

            if other_languages:
                recommendations.append(
                    f"Consider translating documents from {', '.join(other_languages)} to {most_common_lang} for broader accessibility"
                )

        # Recommend translations for high-quality matches
        high_quality_matches = [m for m in cross_linguistic_matches if m.semantic_similarity > 0.8]
        if high_quality_matches:
            recommendations.append(
                f"Found {len(high_quality_matches)} high-quality cross-linguistic matches - consider creating parallel translations"
            )

        # Language-specific recommendations
        if (
            "english" in language_distribution
            and language_distribution["english"] < language_distribution.total() * 0.3
        ):
            recommendations.append(
                "Consider translating key documents to English for international accessibility"
            )

        return recommendations

    def _generate_linguistic_insights(
        self,
        language_detections: list[LanguageDetectionResult],
        cross_linguistic_matches: list[CrossLinguisticMatch],
    ) -> list[str]:
        """Generate linguistic insights from analysis"""
        insights = []

        # Language diversity insights
        unique_languages = set(d.primary_language for d in language_detections)
        if len(unique_languages) > 5:
            insights.append(
                f"High linguistic diversity detected: {len(unique_languages)} languages represented"
            )

        # Script diversity insights
        unique_scripts = set(d.script_type for d in language_detections)
        if len(unique_scripts) > 2:
            insights.append(f"Multiple writing systems present: {', '.join(unique_scripts)}")

        # Mixed language content insights
        mixed_content_docs = [d for d in language_detections if d.mixed_language_segments]
        if mixed_content_docs:
            insights.append(f"{len(mixed_content_docs)} documents contain mixed-language content")

        # Cross-linguistic pattern insights
        if cross_linguistic_matches:
            translation_matches = [
                m for m in cross_linguistic_matches if m.match_type == "translation"
            ]
            if translation_matches:
                insights.append(f"Detected {len(translation_matches)} potential translation pairs")

        # Translation complexity insights
        complex_docs = [
            d
            for d in language_detections
            if d.translation_complexity in ["complex", "highly_complex"]
        ]
        if complex_docs:
            insights.append(f"{len(complex_docs)} documents require complex translation approaches")

        return insights

    def _extract_document_text(self, document: Any) -> str:
        """Extract text content from document"""
        text_fields = ["content", "abstract", "description", "title"]

        text_parts = []
        for field in text_fields:
            if hasattr(document, field):
                field_text = getattr(document, field, "")
                if field_text:
                    text_parts.append(str(field_text))

        return " ".join(text_parts)


class MultiLanguageOrchestrator:
    """Orchestrator for multi-language processing workflows"""

    def __init__(self):
        self.processor = MultiLanguageProcessor()

    async def process_multilingual_collection(
        self, documents: list[Any], target_language: str | None = None
    ) -> dict[str, Any]:
        """Process multilingual document collection"""
        logger.info(f"Processing multilingual collection of {len(documents)} documents")

        # Analyze corpus
        corpus_analysis = await self.processor.analyze_multilingual_corpus(documents)

        # Perform translations if target language specified
        translations = []
        if target_language:
            for doc in documents:
                detection = await self.processor.detect_language(doc)
                if detection.primary_language != target_language:
                    translation = await self.processor.translate_document(doc, target_language)
                    translations.append(translation)

        # Generate comprehensive results
        return {
            "corpus_analysis": corpus_analysis,
            "translations": translations,
            "processing_metadata": {
                "total_documents_processed": len(documents),
                "languages_detected": len(corpus_analysis.language_distribution),
                "translations_performed": len(translations),
                "processing_timestamp": datetime.now(),
            },
        }
