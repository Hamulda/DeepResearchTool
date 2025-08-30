#!/usr/bin/env python3
"""Information Priority Scorer for Deep Research Tool
Multi-factor scoring system for information relevance and credibility

Author: Advanced IT Specialist
"""

from collections import Counter
from datetime import datetime
import math
import re
import statistics
from typing import Any


class InformationPriorityScorer:
    """Advanced multi-factor scoring system for information prioritization"""

    def __init__(self):
        """Initialize the priority scorer with weights and patterns"""
        # Enhanced keyword weights for different categories with more sophisticated scoring
        self.keyword_weights = {
            "conspiracy_keywords": {
                "declassified": 0.9,
                "classified": 0.8,
                "secret": 0.7,
                "cover-up": 0.85,
                "conspiracy": 0.6,
                "hidden": 0.5,
                "suppressed": 0.8,
                "leaked": 0.9,
                "whistleblower": 0.95,
                "revelation": 0.7,
                "exposed": 0.8,
                "confidential": 0.85,
            },
            "research_keywords": {
                "study": 0.7,
                "research": 0.8,
                "analysis": 0.6,
                "investigation": 0.85,
                "evidence": 0.9,
                "data": 0.5,
                "findings": 0.8,
                "results": 0.6,
                "conclusion": 0.7,
                "methodology": 0.6,
                "peer-reviewed": 0.95,
                "clinical trial": 0.9,
            },
            "authority_keywords": {
                "government": 0.6,
                "official": 0.7,
                "agency": 0.6,
                "department": 0.5,
                "military": 0.7,
                "intelligence": 0.8,
                "document": 0.6,
                "report": 0.5,
                "fbi": 0.8,
                "cia": 0.8,
                "nsa": 0.8,
                "pentagon": 0.7,
            },
            "temporal_keywords": {
                "historical": 0.6,
                "archive": 0.7,
                "vintage": 0.5,
                "original": 0.8,
                "primary source": 0.9,
                "contemporary": 0.6,
                "period": 0.4,
                "era": 0.4,
                "decade": 0.4,
                "recently declassified": 0.95,
                "newly discovered": 0.9,
            },
            "medical_keywords": {
                "peptide": 0.9,
                "hormone": 0.8,
                "clinical": 0.8,
                "therapeutic": 0.8,
                "mechanism": 0.7,
                "pharmacokinetics": 0.9,
                "bioavailability": 0.8,
                "receptor": 0.7,
                "pathway": 0.7,
                "absorption": 0.6,
                "metabolism": 0.7,
                "half-life": 0.6,
                "dosage": 0.5,
                "side effects": 0.6,
                "contraindication": 0.7,
            },
            "quality_indicators": {
                "peer-reviewed": 0.95,
                "meta-analysis": 0.9,
                "systematic review": 0.85,
                "randomized controlled trial": 0.9,
                "double-blind": 0.8,
                "placebo-controlled": 0.8,
                "longitudinal study": 0.7,
                "cohort study": 0.7,
                "case study": 0.5,
            },
            "breakthrough_indicators": {
                "breakthrough": 0.95,
                "revolutionary": 0.9,
                "novel": 0.8,
                "first-of-its-kind": 0.95,
                "unprecedented": 0.9,
                "paradigm shift": 0.95,
                "game-changing": 0.85,
                "groundbreaking": 0.9,
                "innovative": 0.7,
                "cutting-edge": 0.8,
            },
            "controversy_indicators": {
                "controversial": 0.7,
                "disputed": 0.8,
                "debated": 0.6,
                "contradictory": 0.75,
                "conflicting": 0.7,
                "questioned": 0.6,
                "challenged": 0.65,
                "refuted": 0.8,
            },
        }

        # Enhanced source credibility with more granular scoring
        self.source_credibility = {
            "arxiv.org": 0.9,
            "pubmed.ncbi.nlm.nih.gov": 0.95,
            "nature.com": 0.95,
            "science.org": 0.95,
            "cell.com": 0.9,
            "nejm.org": 0.95,
            "thelancet.com": 0.9,
            "archive.org": 0.85,
            "government": 0.8,
            "academic": 0.85,
            ".edu": 0.85,
            "clinicaltrials.gov": 0.9,
            "cochrane.org": 0.95,
            "who.int": 0.9,
            "nih.gov": 0.9,
            "fda.gov": 0.85,
            "cdc.gov": 0.85,
            "news_mainstream": 0.7,
            "news_alternative": 0.5,
            "blog": 0.3,
            "forum": 0.2,
            "social_media": 0.1,
            "unknown": 0.4,
            "peptide_database": 0.85,
            "medical_database": 0.9,
        }

        # Enhanced temporal decay with content-specific factors
        self.temporal_decay = {
            "conspiracy": 0.015,  # Conspiracy theories may gain relevance over time
            "research": 0.08,  # Medical research becomes obsolete faster
            "news": 0.15,  # News becomes obsolete very quickly
            "archive": 0.005,  # Archives maintain high relevance
            "medical": 0.06,  # Medical info has moderate decay
            "peptide": 0.04,  # Peptide research has slower decay
        }

        # Advanced citation and reference patterns
        self.citation_patterns = [
            r"\bcited?\s+by\s+(\d+)\b",  # "cited by 150"
            r"\b(\d+)\s+citations?\b",  # "150 citations"
            r"\bimpact\s+factor[:=]\s*(\d+\.?\d*)\b",  # "impact factor: 8.5"
            r"\bh-index[:=]\s*(\d+)\b",  # "h-index: 25"
            r"\bdoi[:=]\s*10\.\d+/\S+",  # DOI patterns
            r"\bpmid[:=]\s*(\d+)\b",  # PubMed ID
        ]

        # Content quality indicators
        self.quality_patterns = {
            "high_quality": [
                r"\bpeer.?reviewed?\b",
                r"\bmeta.?analysis\b",
                r"\bsystematic\s+review\b",
                r"\brandomized\s+controlled\s+trial\b",
                r"\bdouble.?blind\b",
                r"\bplacebo.?controlled\b",
                r"\bmulticenter\b",
                r"\bprospective\b",
            ],
            "medium_quality": [
                r"\bcohort\s+study\b",
                r"\bcase.?control\b",
                r"\bobservational\s+study\b",
                r"\bcross.?sectional\b",
                r"\bpilot\s+study\b",
            ],
            "low_quality": [
                r"\bcase\s+report\b",
                r"\banecdotal\b",
                r"\bopinion\b",
                r"\beditorial\b",
                r"\bcommentary\b",
            ],
        }

        # Domain expertise indicators
        self.expertise_indicators = {
            "high_expertise": [
                r"\bprofessor\b",
                r"\bphd\b",
                r"\bmd\b",
                r"\bresearch\s+scientist\b",
                r"\bprincipal\s+investigator\b",
                r"\bdepartment\s+of\b",
                r"\buniversity\s+of\b",
            ],
            "institutional": [
                r"\bharvard\b",
                r"\bmit\b",
                r"\bstanford\b",
                r"\bmayo\s+clinic\b",
                r"\bjohns\s+hopkins\b",
                r"\bnih\b",
                r"\bwho\b",
                r"\bfda\b",
            ],
        }

        # Statistical indicators for reliability
        self.statistical_indicators = {
            "sample_size_patterns": [
                r"\bn\s*=\s*(\d+)\b",  # "n = 100"
                r"\bsample\s+size\s*[:=]\s*(\d+)\b",  # "sample size: 100"
                r"\b(\d+)\s+participants?\b",  # "100 participants"
                r"\b(\d+)\s+subjects?\b",  # "100 subjects"
            ],
            "statistical_tests": [
                r"\bp\s*<\s*0\.05\b",
                r"\bp\s*<\s*0\.01\b",
                r"\bp\s*<\s*0\.001\b",
                r"\bconfidence\s+interval\b",
                r"\bci\s*[:=]\s*\d+%\b",
                r"\bt-test\b",
                r"\banova\b",
                r"\bregression\b",
            ],
        }

    def score_information(self, content: str, metadata: dict[str, Any]) -> float:
        """Calculate comprehensive priority score for information with enhanced analysis

        Args:
            content: Text content to score
            metadata: Metadata about the content

        Returns:
            Priority score between 0.0 and 1.0

        """
        scores = {}

        # Core scoring components
        scores["keyword"] = self._score_keyword_relevance(content)
        scores["credibility"] = self._score_source_credibility(metadata)
        scores["temporal"] = self._score_temporal_relevance(content, metadata)
        scores["cross_ref"] = self._score_cross_references(content)
        scores["uniqueness"] = self._score_uniqueness(content, metadata)
        scores["quality"] = self._score_content_quality(content)

        # Enhanced scoring components
        scores["semantic_coherence"] = self._score_semantic_coherence(content)
        scores["statistical_rigor"] = self._score_statistical_rigor(content)
        scores["controversy_value"] = self._score_controversy_indicators(content)
        scores["breakthrough_potential"] = self._score_breakthrough_indicators(content)

        # Multi-level confidence scoring
        confidence_level = self._calculate_confidence_level(scores, content, metadata)

        # Weighted combination with confidence adjustment
        base_weights = {
            "keyword": 0.20,
            "credibility": 0.18,
            "temporal": 0.12,
            "cross_ref": 0.12,
            "uniqueness": 0.10,
            "quality": 0.10,
            "semantic_coherence": 0.08,
            "statistical_rigor": 0.05,
            "controversy_value": 0.03,
            "breakthrough_potential": 0.02,
        }

        # Adjust weights based on confidence level
        adjusted_weights = self._adjust_weights_by_confidence(base_weights, confidence_level)

        final_score = sum(scores[key] * adjusted_weights[key] for key in scores)

        # Apply confidence multiplier
        final_score *= confidence_level

        return min(1.0, max(0.0, final_score))

    def _score_keyword_relevance(self, content: str) -> float:
        """Score based on presence of relevant keywords"""
        content_lower = content.lower()
        total_score = 0.0
        total_weight = 0.0

        for category, keywords in self.keyword_weights.items():
            category_score = 0.0
            category_weight = 1.0

            for keyword, weight in keywords.items():
                # Count occurrences with diminishing returns
                count = len(re.findall(r"\b" + re.escape(keyword) + r"\b", content_lower))
                if count > 0:
                    # Logarithmic scaling for multiple occurrences
                    keyword_score = min(1.0, math.log(count + 1) / math.log(5))
                    category_score = max(category_score, keyword_score * weight)

            total_score += category_score * category_weight
            total_weight += category_weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _score_source_credibility(self, metadata: dict[str, Any]) -> float:
        """Score based on source credibility"""
        source_url = metadata.get("source_url", "")
        source_type = metadata.get("source_type", "unknown")

        # Check URL-based credibility
        for domain, credibility in self.source_credibility.items():
            if domain in source_url.lower():
                return credibility

        # Check type-based credibility
        return self.source_credibility.get(source_type, 0.4)

    def _score_temporal_relevance(self, content: str, metadata: dict[str, Any]) -> float:
        """Score based on temporal relevance"""
        # Extract date from metadata or content
        pub_date = self._extract_date(content, metadata)
        if not pub_date:
            return 0.5  # Neutral score for unknown dates

        # Calculate age in years
        age_years = (datetime.now() - pub_date).days / 365.25

        # Determine content type for decay calculation
        content_type = self._classify_content_type(content, metadata)
        decay_rate = self.temporal_decay.get(content_type, 0.05)

        # Calculate temporal relevance with exponential decay
        relevance = math.exp(-decay_rate * age_years)

        # Bonus for historical significance (very old documents)
        if age_years > 30:
            historical_bonus = min(0.3, age_years / 100)
            relevance += historical_bonus

        return min(1.0, relevance)

    def _score_cross_references(self, content: str) -> float:
        """Score based on cross-reference indicators"""
        cross_ref_count = 0

        for pattern in self.citation_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            cross_ref_count += len(matches)

        # URL references
        url_count = len(re.findall(r"https?://\S+", content))
        cross_ref_count += url_count

        # Citation patterns
        citation_count = len(re.findall(r"\[\d+\]|\(\d{4}\)", content))
        cross_ref_count += citation_count

        # Logarithmic scaling
        if cross_ref_count > 0:
            return min(1.0, math.log(cross_ref_count + 1) / math.log(10))
        return 0.0

    def _score_uniqueness(self, content: str, metadata: dict[str, Any]) -> float:
        """Score based on content uniqueness"""
        # Simple uniqueness based on rare words and phrases
        words = re.findall(r"\b\w{4,}\b", content.lower())
        word_freq = Counter(words)

        # Count rare words (appearing only once)
        rare_words = sum(1 for count in word_freq.values() if count == 1)
        total_words = len(words)

        if total_words == 0:
            return 0.0

        uniqueness_ratio = rare_words / total_words

        # Bonus for specific indicators of unique content
        unique_indicators = [
            "exclusive",
            "unprecedented",
            "never before",
            "first time",
            "newly discovered",
            "recently declassified",
            "breaking",
        ]

        bonus = 0.0
        for indicator in unique_indicators:
            if indicator in content.lower():
                bonus += 0.1

        return min(1.0, uniqueness_ratio + bonus)

    def _score_content_quality(self, content: str) -> float:
        """Score based on content quality indicators"""
        # Length score (optimal around 500-2000 characters)
        length = len(content)
        if length < 100:
            length_score = length / 100
        elif length > 5000:
            length_score = 5000 / length
        else:
            length_score = 1.0

        # Sentence structure score
        sentences = re.split(r"[.!?]+", content)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if not valid_sentences:
            structure_score = 0.0
        else:
            avg_sentence_length = sum(len(s.split()) for s in valid_sentences) / len(
                valid_sentences
            )
            # Optimal sentence length around 15-25 words
            if 10 <= avg_sentence_length <= 30:
                structure_score = 1.0
            else:
                structure_score = max(0.3, 1.0 - abs(avg_sentence_length - 20) / 50)

        # Readability indicators
        readability_score = 0.5  # Base score

        # Check for structured content
        if re.search(r"\b(abstract|introduction|methodology|conclusion)\b", content, re.IGNORECASE):
            readability_score += 0.2

        # Check for data/evidence
        if re.search(r"\b(data|evidence|study|research|analysis)\b", content, re.IGNORECASE):
            readability_score += 0.2

        # Penalty for excessive caps or poor formatting
        caps_ratio = sum(1 for c in content if c.isupper()) / len(content) if content else 0
        if caps_ratio > 0.3:
            readability_score -= 0.3

        # Combine quality scores
        quality_scores = [length_score, structure_score, readability_score]
        return sum(quality_scores) / len(quality_scores)

    def _extract_date(self, content: str, metadata: dict[str, Any]) -> datetime:
        """Extract publication date from content or metadata"""
        # Try metadata first
        if "date" in metadata:
            try:
                if isinstance(metadata["date"], datetime):
                    return metadata["date"]
                return datetime.fromisoformat(str(metadata["date"]))
            except:
                pass

        # Try to extract from content
        date_patterns = [
            r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b",  # YYYY-MM-DD
            r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b",  # MM/DD/YYYY
            r"\b(\d{1,2})-(\d{1,2})-(\d{4})\b",  # MM-DD-YYYY
            r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+(\d{1,2}),?\s+(\d{4})\b",
        ]

        for pattern in date_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                try:
                    match = matches[0]
                    if len(match) == 3 and match[0].isalpha():
                        # Month name format
                        month_map = {
                            "jan": 1,
                            "feb": 2,
                            "mar": 3,
                            "apr": 4,
                            "may": 5,
                            "jun": 6,
                            "jul": 7,
                            "aug": 8,
                            "sep": 9,
                            "oct": 10,
                            "nov": 11,
                            "dec": 12,
                        }
                        month = month_map.get(match[0][:3].lower(), 1)
                        return datetime(int(match[2]), month, int(match[1]))
                    if "-" in content[content.find(match[0]) : content.find(match[0]) + 20]:
                        # YYYY-MM-DD format
                        return datetime(int(match[0]), int(match[1]), int(match[2]))
                    # MM/DD/YYYY format
                    return datetime(int(match[2]), int(match[0]), int(match[1]))
                except:
                    continue

        return None

    def _classify_content_type(self, content: str, metadata: dict[str, Any]) -> str:
        """Classify content type for temporal scoring"""
        content_lower = content.lower()

        # Check metadata first
        content_type = metadata.get("content_type", "")
        if content_type in self.temporal_decay:
            return content_type

        # Classify based on content
        if any(word in content_lower for word in ["conspiracy", "theory", "cover-up", "secret"]):
            return "conspiracy"
        if any(word in content_lower for word in ["study", "research", "analysis", "findings"]):
            return "research"
        if any(word in content_lower for word in ["breaking", "news", "report", "update"]):
            return "news"
        if any(
            word in content_lower for word in ["archive", "historical", "vintage", "original"]
        ):
            return "archive"

        return "research"  # Default classification

    def update_keyword_weights(self, new_keywords: dict[str, set[str]]):
        """Update keyword weights with new categories"""
        for category, keywords in new_keywords.items():
            if category in self.keyword_weights:
                self.keyword_weights[category].update(keywords)
            else:
                self.keyword_weights[category] = keywords

    def update_source_credibility(self, new_sources: dict[str, float]):
        """Update source credibility scores"""
        self.source_credibility.update(new_sources)

    def get_score_breakdown(self, content: str, metadata: dict[str, Any]) -> dict[str, float]:
        """Get detailed breakdown of scoring components"""
        return {
            "keyword_relevance": self._score_keyword_relevance(content),
            "source_credibility": self._score_source_credibility(metadata),
            "temporal_relevance": self._score_temporal_relevance(content, metadata),
            "cross_references": self._score_cross_references(content),
            "uniqueness": self._score_uniqueness(content, metadata),
            "content_quality": self._score_content_quality(content),
            "semantic_coherence": self._score_semantic_coherence(content),
            "statistical_rigor": self._score_statistical_rigor(content),
            "controversy_value": self._score_controversy_indicators(content),
            "breakthrough_potential": self._score_breakthrough_indicators(content),
        }

    def _score_semantic_coherence(self, content: str) -> float:
        """Score based on semantic coherence and logical flow"""
        sentences = re.split(r"[.!?]+", content)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if len(valid_sentences) < 2:
            return 0.5

        # Check for logical connectors
        connectors = [
            "therefore",
            "however",
            "furthermore",
            "moreover",
            "consequently",
            "nevertheless",
            "in addition",
            "as a result",
            "on the other hand",
        ]

        connector_count = sum(
            1
            for sentence in valid_sentences
            for connector in connectors
            if connector in sentence.lower()
        )

        connector_ratio = connector_count / len(valid_sentences)

        # Check for consistent terminology
        technical_terms = re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b", content)
        term_consistency = len(set(technical_terms)) / max(len(technical_terms), 1)

        # Check for balanced sentence structure
        sentence_lengths = [len(s.split()) for s in valid_sentences]
        length_variance = statistics.variance(sentence_lengths) if len(sentence_lengths) > 1 else 0
        structure_score = max(0, 1 - length_variance / 100)

        return connector_ratio * 0.4 + term_consistency * 0.3 + structure_score * 0.3

    def _score_statistical_rigor(self, content: str) -> float:
        """Score based on statistical indicators and methodology"""
        rigor_score = 0.0

        # Check for sample size
        for pattern in self.statistical_indicators["sample_size_patterns"]:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                sample_size = max(int(match) for match in matches if match.isdigit())
                if sample_size >= 1000:
                    rigor_score += 0.3
                elif sample_size >= 100:
                    rigor_score += 0.2
                elif sample_size >= 30:
                    rigor_score += 0.1
                break

        # Check for statistical tests
        for pattern in self.statistical_indicators["statistical_tests"]:
            if re.search(pattern, content, re.IGNORECASE):
                rigor_score += 0.15

        # Check for effect size reporting
        effect_size_patterns = [
            r"\bcohen\'?s\s+d\b",
            r"\beffect\s+size\b",
            r"\bodds\s+ratio\b",
            r"\bhazard\s+ratio\b",
            r"\bretative\s+risk\b",
        ]

        for pattern in effect_size_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                rigor_score += 0.1
                break

        return min(1.0, rigor_score)

    def _score_controversy_indicators(self, content: str) -> float:
        """Score content for controversial aspects that may indicate importance"""
        controversy_score = 0.0
        content_lower = content.lower()

        for keyword, weight in self.keyword_weights["controversy_indicators"].items():
            if keyword in content_lower:
                controversy_score = max(controversy_score, weight)

        # Check for opposing viewpoints
        opposition_patterns = [
            r"\bhowever\b",
            r"\bbut\b",
            r"\bcontrary\s+to\b",
            r"\bon\s+the\s+other\s+hand\b",
            r"\bnevertheless\b",
            r"\bdespite\b",
            r"\balthough\b",
        ]

        opposition_count = sum(
            1 for pattern in opposition_patterns if re.search(pattern, content, re.IGNORECASE)
        )

        if opposition_count > 0:
            controversy_score += min(0.3, opposition_count * 0.1)

        return min(1.0, controversy_score)

    def _score_breakthrough_indicators(self, content: str) -> float:
        """Score content for breakthrough or novel findings"""
        breakthrough_score = 0.0
        content_lower = content.lower()

        for keyword, weight in self.keyword_weights["breakthrough_indicators"].items():
            if keyword in content_lower:
                breakthrough_score = max(breakthrough_score, weight)

        # Check for novelty indicators
        novelty_patterns = [
            r"\bfirst\s+time\b",
            r"\bnever\s+before\b",
            r"\bpreviously\s+unknown\b",
            r"\bnew\s+discovery\b",
            r"\brecent\s+breakthrough\b",
        ]

        for pattern in novelty_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                breakthrough_score += 0.1

        return min(1.0, breakthrough_score)

    def _calculate_confidence_level(
        self, scores: dict[str, float], content: str, metadata: dict[str, Any]
    ) -> float:
        """Calculate confidence level based on multiple factors"""
        confidence_factors = []

        # Source reliability factor
        source_credibility = scores.get("credibility", 0.5)
        confidence_factors.append(source_credibility)

        # Content length factor (optimal range)
        content_length = len(content)
        if 500 <= content_length <= 3000:
            length_confidence = 1.0
        elif content_length < 100:
            length_confidence = content_length / 100
        else:
            length_confidence = max(0.3, 3000 / content_length)
        confidence_factors.append(length_confidence)

        # Citation factor
        citation_score = scores.get("cross_ref", 0.0)
        confidence_factors.append(min(1.0, citation_score * 2))

        # Quality factor
        quality_score = scores.get("quality", 0.0)
        confidence_factors.append(quality_score)

        # Metadata completeness factor
        metadata_completeness = len([v for v in metadata.values() if v]) / max(len(metadata), 1)
        confidence_factors.append(metadata_completeness)

        # Calculate weighted average with emphasis on source credibility
        weights = [0.3, 0.2, 0.2, 0.2, 0.1]
        confidence = sum(factor * weight for factor, weight in zip(confidence_factors, weights, strict=False))

        return max(0.1, min(1.0, confidence))

    def _adjust_weights_by_confidence(
        self, base_weights: dict[str, float], confidence: float
    ) -> dict[str, float]:
        """Adjust scoring weights based on confidence level"""
        adjusted_weights = base_weights.copy()

        if confidence < 0.5:
            # Lower confidence: reduce impact of secondary factors
            adjusted_weights["credibility"] += 0.05
            adjusted_weights["quality"] += 0.05
            adjusted_weights["semantic_coherence"] -= 0.03
            adjusted_weights["controversy_value"] -= 0.02
        elif confidence > 0.8:
            # Higher confidence: allow secondary factors more influence
            adjusted_weights["breakthrough_potential"] += 0.02
            adjusted_weights["controversy_value"] += 0.02
            adjusted_weights["statistical_rigor"] += 0.01

        # Normalize weights to sum to 1.0
        total_weight = sum(adjusted_weights.values())
        adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}

        return adjusted_weights

    def get_detailed_analysis(self, content: str, metadata: dict[str, Any]) -> dict[str, Any]:
        """Get comprehensive analysis breakdown"""
        scores = {
            "keyword_relevance": self._score_keyword_relevance(content),
            "source_credibility": self._score_source_credibility(metadata),
            "temporal_relevance": self._score_temporal_relevance(content, metadata),
            "cross_references": self._score_cross_references(content),
            "uniqueness": self._score_uniqueness(content, metadata),
            "content_quality": self._score_content_quality(content),
            "semantic_coherence": self._score_semantic_coherence(content),
            "statistical_rigor": self._score_statistical_rigor(content),
            "controversy_value": self._score_controversy_indicators(content),
            "breakthrough_potential": self._score_breakthrough_indicators(content),
        }

        confidence_level = self._calculate_confidence_level(scores, content, metadata)
        final_score = self.score_information(content, metadata)

        # Extract key insights
        insights = self._extract_key_insights(content, scores)

        return {
            "final_score": final_score,
            "confidence_level": confidence_level,
            "component_scores": scores,
            "key_insights": insights,
            "content_classification": self._classify_content_type(content, metadata),
            "reliability_assessment": self._assess_reliability(scores, metadata),
        }

    def _extract_key_insights(self, content: str, scores: dict[str, float]) -> list[str]:
        """Extract key insights about the content"""
        insights = []

        if scores["breakthrough_potential"] > 0.7:
            insights.append("Contains potential breakthrough or novel findings")

        if scores["controversy_value"] > 0.6:
            insights.append("Content addresses controversial or disputed topics")

        if scores["statistical_rigor"] > 0.7:
            insights.append("Demonstrates strong statistical methodology")

        if scores["source_credibility"] > 0.9:
            insights.append("From highly credible and authoritative source")

        if scores["cross_references"] > 0.8:
            insights.append("Well-referenced with multiple citations")

        if scores["semantic_coherence"] > 0.8:
            insights.append("Demonstrates logical flow and coherent argumentation")

        return insights

    def _assess_reliability(self, scores: dict[str, float], metadata: dict[str, Any]) -> str:
        """Assess overall reliability of the information"""
        reliability_score = (
            scores["source_credibility"] * 0.4
            + scores["content_quality"] * 0.3
            + scores["statistical_rigor"] * 0.2
            + scores["cross_references"] * 0.1
        )

        if reliability_score >= 0.85:
            return "Very High"
        if reliability_score >= 0.7:
            return "High"
        if reliability_score >= 0.5:
            return "Medium"
        if reliability_score >= 0.3:
            return "Low"
        return "Very Low"
