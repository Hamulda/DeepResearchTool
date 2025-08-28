#!/usr/bin/env python3
"""
Contradiction Detection and Sets
Advanced claim verification with pro/contra evidence grouping

Author: Senior Python/MLOps Agent
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging
from datetime import datetime
import hashlib
import json

logger = logging.getLogger(__name__)


class ContradictionType(Enum):
    """Types of contradictions detected"""
    DIRECT = "direct"  # Direct negation
    SEMANTIC = "semantic"  # Contradictory meaning
    TEMPORAL = "temporal"  # Conflicting timelines
    QUANTITATIVE = "quantitative"  # Conflicting numbers/stats
    SOURCE = "source"  # Same source contradicting itself
    METHODOLOGICAL = "methodological"  # Different methods, different results


class ConfidenceLevel(Enum):
    """Confidence levels for contradiction detection"""
    HIGH = "high"      # >0.8
    MEDIUM = "medium"  # 0.5-0.8
    LOW = "low"        # 0.2-0.5
    UNCERTAIN = "uncertain"  # <0.2


@dataclass
class Claim:
    """Represents a research claim with evidence"""
    id: str
    text: str
    evidence: List[Dict[str, Any]]
    confidence: float
    source_urls: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)


@dataclass
class Contradiction:
    """Represents a detected contradiction between claims"""
    id: str
    claim_a: Claim
    claim_b: Claim
    contradiction_type: ContradictionType
    confidence: float
    evidence: Dict[str, Any]
    resolution_suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "claim_a_id": self.claim_a.id,
            "claim_b_id": self.claim_b.id,
            "type": self.contradiction_type.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "resolution_suggestion": self.resolution_suggestion
        }


@dataclass
class ContradictionSet:
    """Group of related contradictions with pro/contra evidence"""
    topic: str
    supporting_claims: List[Claim]
    contradicting_claims: List[Claim]
    contradictions: List[Contradiction]
    confidence_score: float
    calibration_hint: str

    @property
    def pro_evidence_count(self) -> int:
        return sum(len(claim.evidence) for claim in self.supporting_claims)

    @property
    def contra_evidence_count(self) -> int:
        return sum(len(claim.evidence) for claim in self.contradicting_claims)

    @property
    def total_evidence_count(self) -> int:
        return self.pro_evidence_count + self.contra_evidence_count


class ContradictionDetector:
    """Advanced contradiction detection engine"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.verification_config = config.get("verification", {})

        # Detection thresholds
        self.semantic_threshold = self.verification_config.get("semantic_threshold", 0.7)
        self.temporal_threshold = self.verification_config.get("temporal_threshold", 0.8)
        self.quantitative_threshold = self.verification_config.get("quantitative_threshold", 0.6)

        # Contradiction patterns
        self.negation_patterns = [
            ("not", "is not", "does not", "cannot", "never"),
            ("effective", "ineffective", "counterproductive"),
            ("safe", "unsafe", "dangerous", "harmful"),
            ("increase", "decrease", "reduce", "lower"),
            ("positive", "negative", "adverse"),
            ("supports", "contradicts", "opposes", "refutes")
        ]

        # Calibration mappings
        self.calibration_messages = {
            ConfidenceLevel.HIGH: "Strong contradiction detected. Recommend careful source verification.",
            ConfidenceLevel.MEDIUM: "Moderate contradiction. Consider additional evidence.",
            ConfidenceLevel.LOW: "Potential contradiction. May require expert assessment.",
            ConfidenceLevel.UNCERTAIN: "Unclear relationship. Insufficient evidence for determination."
        }

        logger.info("Contradiction detector initialized")

    def detect_contradictions(self, claims: List[Claim]) -> List[Contradiction]:
        """Detect contradictions between claims"""

        contradictions = []

        # Compare each pair of claims
        for i, claim_a in enumerate(claims):
            for j, claim_b in enumerate(claims[i+1:], i+1):
                contradiction = self._analyze_claim_pair(claim_a, claim_b)
                if contradiction:
                    contradictions.append(contradiction)

        logger.info(f"Detected {len(contradictions)} contradictions among {len(claims)} claims")
        return contradictions

    def _analyze_claim_pair(self, claim_a: Claim, claim_b: Claim) -> Optional[Contradiction]:
        """Analyze a pair of claims for contradictions"""

        # Skip if same claim
        if claim_a.id == claim_b.id:
            return None

        # Check different types of contradictions
        contradiction_checks = [
            self._check_direct_contradiction,
            self._check_semantic_contradiction,
            self._check_temporal_contradiction,
            self._check_quantitative_contradiction,
            self._check_source_contradiction
        ]

        best_contradiction = None
        best_confidence = 0.0

        for check_func in contradiction_checks:
            result = check_func(claim_a, claim_b)
            if result and result["confidence"] > best_confidence:
                best_confidence = result["confidence"]
                best_contradiction = result

        # Only return if confidence above threshold
        if best_contradiction and best_confidence > 0.3:
            contradiction_id = self._generate_contradiction_id(claim_a, claim_b)

            return Contradiction(
                id=contradiction_id,
                claim_a=claim_a,
                claim_b=claim_b,
                contradiction_type=ContradictionType(best_contradiction["type"]),
                confidence=best_confidence,
                evidence=best_contradiction["evidence"],
                resolution_suggestion=best_contradiction.get("resolution")
            )

        return None

    def _check_direct_contradiction(self, claim_a: Claim, claim_b: Claim) -> Optional[Dict[str, Any]]:
        """Check for direct negation patterns"""

        text_a = claim_a.text.lower()
        text_b = claim_b.text.lower()

        contradiction_score = 0.0
        detected_patterns = []

        for pattern_group in self.negation_patterns:
            positive_terms = [term for term in pattern_group if not term.startswith("not") and not term.startswith("in") and not term.startswith("un")]
            negative_terms = [term for term in pattern_group if term.startswith("not") or term.startswith("in") or term.startswith("un") or "not" in term]

            # Check if one claim has positive terms and other has negative
            pos_in_a = any(term in text_a for term in positive_terms)
            neg_in_a = any(term in text_a for term in negative_terms)
            pos_in_b = any(term in text_b for term in positive_terms)
            neg_in_b = any(term in text_b for term in negative_terms)

            if (pos_in_a and neg_in_b) or (neg_in_a and pos_in_b):
                contradiction_score += 0.3
                detected_patterns.append(pattern_group)

        if contradiction_score > 0.5:
            return {
                "type": "direct",
                "confidence": min(contradiction_score, 1.0),
                "evidence": {
                    "patterns_detected": detected_patterns,
                    "claim_a_text": claim_a.text,
                    "claim_b_text": claim_b.text
                },
                "resolution": "Direct contradiction detected. Verify source credibility and temporal context."
            }

        return None

    def _check_semantic_contradiction(self, claim_a: Claim, claim_b: Claim) -> Optional[Dict[str, Any]]:
        """Check for semantic contradictions using embeddings (mock implementation)"""

        # Mock semantic similarity (in production, use actual embeddings)
        similarity = self._calculate_mock_semantic_similarity(claim_a.text, claim_b.text)

        # High similarity with opposing sentiment indicates semantic contradiction
        if similarity > self.semantic_threshold:
            sentiment_a = self._analyze_sentiment(claim_a.text)
            sentiment_b = self._analyze_sentiment(claim_b.text)

            if sentiment_a * sentiment_b < -0.3:  # Opposite sentiments
                confidence = similarity * abs(sentiment_a - sentiment_b)

                return {
                    "type": "semantic",
                    "confidence": confidence,
                    "evidence": {
                        "semantic_similarity": similarity,
                        "sentiment_a": sentiment_a,
                        "sentiment_b": sentiment_b,
                        "claim_a_text": claim_a.text,
                        "claim_b_text": claim_b.text
                    },
                    "resolution": "Semantic contradiction detected. Claims discuss same topic with opposite conclusions."
                }

        return None

    def _check_temporal_contradiction(self, claim_a: Claim, claim_b: Claim) -> Optional[Dict[str, Any]]:
        """Check for temporal contradictions"""

        # Extract temporal information
        time_a = self._extract_temporal_info(claim_a)
        time_b = self._extract_temporal_info(claim_b)

        if time_a and time_b and time_a.get("period") == time_b.get("period"):
            # Same time period, check for contradictory events
            events_a = time_a.get("events", [])
            events_b = time_b.get("events", [])

            contradictory_events = self._find_contradictory_events(events_a, events_b)

            if contradictory_events:
                confidence = 0.7 * len(contradictory_events) / max(len(events_a), len(events_b))

                return {
                    "type": "temporal",
                    "confidence": confidence,
                    "evidence": {
                        "time_period": time_a["period"],
                        "contradictory_events": contradictory_events,
                        "claim_a_events": events_a,
                        "claim_b_events": events_b
                    },
                    "resolution": "Temporal contradiction detected. Verify timeline and sequence of events."
                }

        return None

    def _check_quantitative_contradiction(self, claim_a: Claim, claim_b: Claim) -> Optional[Dict[str, Any]]:
        """Check for contradictory numbers/statistics"""

        numbers_a = self._extract_numbers(claim_a.text)
        numbers_b = self._extract_numbers(claim_b.text)

        if numbers_a and numbers_b:
            contradictions = []

            for num_a in numbers_a:
                for num_b in numbers_b:
                    if self._are_numbers_contradictory(num_a, num_b):
                        contradictions.append({
                            "value_a": num_a,
                            "value_b": num_b,
                            "difference": abs(num_a["value"] - num_b["value"]),
                            "relative_difference": abs(num_a["value"] - num_b["value"]) / max(num_a["value"], num_b["value"])
                        })

            if contradictions:
                avg_rel_diff = np.mean([c["relative_difference"] for c in contradictions])
                confidence = min(avg_rel_diff, 1.0)

                return {
                    "type": "quantitative",
                    "confidence": confidence,
                    "evidence": {
                        "contradictory_numbers": contradictions,
                        "numbers_a": numbers_a,
                        "numbers_b": numbers_b
                    },
                    "resolution": "Quantitative contradiction detected. Verify data sources and measurement methods."
                }

        return None

    def _check_source_contradiction(self, claim_a: Claim, claim_b: Claim) -> Optional[Dict[str, Any]]:
        """Check if same source contradicts itself"""

        sources_a = set(claim_a.source_urls)
        sources_b = set(claim_b.source_urls)

        common_sources = sources_a.intersection(sources_b)

        if common_sources:
            # Same source making contradictory claims
            confidence = 0.8  # High confidence for self-contradiction

            return {
                "type": "source",
                "confidence": confidence,
                "evidence": {
                    "common_sources": list(common_sources),
                    "claim_a_sources": list(sources_a),
                    "claim_b_sources": list(sources_b)
                },
                "resolution": "Source contradiction detected. Same source provides conflicting information."
            }

        return None

    def _calculate_mock_semantic_similarity(self, text_a: str, text_b: str) -> float:
        """Mock semantic similarity calculation"""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())

        intersection = len(words_a.intersection(words_b))
        union = len(words_a.union(words_b))

        return intersection / union if union > 0 else 0.0

    def _analyze_sentiment(self, text: str) -> float:
        """Mock sentiment analysis (-1 to 1)"""
        positive_words = ["effective", "safe", "good", "positive", "beneficial", "successful"]
        negative_words = ["ineffective", "unsafe", "bad", "negative", "harmful", "failed"]

        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count + neg_count == 0:
            return 0.0

        return (pos_count - neg_count) / (pos_count + neg_count)

    def _extract_temporal_info(self, claim: Claim) -> Optional[Dict[str, Any]]:
        """Extract temporal information from claim"""
        # Simplified temporal extraction
        import re

        text = claim.text.lower()

        # Look for year patterns
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, claim.text)

        if years:
            return {
                "period": years[0],
                "events": [claim.text]  # Simplified
            }

        return None

    def _find_contradictory_events(self, events_a: List[str], events_b: List[str]) -> List[Dict[str, str]]:
        """Find contradictory events in same time period"""
        contradictions = []

        for event_a in events_a:
            for event_b in events_b:
                if self._are_events_contradictory(event_a, event_b):
                    contradictions.append({
                        "event_a": event_a,
                        "event_b": event_b
                    })

        return contradictions

    def _are_events_contradictory(self, event_a: str, event_b: str) -> bool:
        """Check if two events are contradictory"""
        # Simplified contradiction check
        sentiment_a = self._analyze_sentiment(event_a)
        sentiment_b = self._analyze_sentiment(event_b)

        return sentiment_a * sentiment_b < -0.5

    def _extract_numbers(self, text: str) -> List[Dict[str, Any]]:
        """Extract numbers and their context from text"""
        import re

        # Pattern for numbers with units/context
        number_pattern = r'(\d+(?:\.\d+)?)\s*(%|percent|cases|deaths|people|patients|mg|kg|ml|years?|months?|days?|hours?|minutes?)'

        matches = re.findall(number_pattern, text.lower())

        numbers = []
        for value_str, unit in matches:
            try:
                value = float(value_str)
                numbers.append({
                    "value": value,
                    "unit": unit,
                    "context": text
                })
            except ValueError:
                continue

        return numbers

    def _are_numbers_contradictory(self, num_a: Dict[str, Any], num_b: Dict[str, Any]) -> bool:
        """Check if two numbers are contradictory"""

        # Same unit, significantly different values
        if num_a["unit"] == num_b["unit"]:
            relative_diff = abs(num_a["value"] - num_b["value"]) / max(num_a["value"], num_b["value"])
            return relative_diff > 0.5  # 50% difference threshold

        return False

    def _generate_contradiction_id(self, claim_a: Claim, claim_b: Claim) -> str:
        """Generate unique ID for contradiction"""
        combined = f"{claim_a.id}_{claim_b.id}"
        return hashlib.md5(combined.encode()).hexdigest()[:8]

    def create_contradiction_sets(self, claims: List[Claim], contradictions: List[Contradiction]) -> List[ContradictionSet]:
        """Group claims and contradictions into coherent sets"""

        # Group by topic/theme
        topic_groups = self._group_claims_by_topic(claims)

        contradiction_sets = []

        for topic, topic_claims in topic_groups.items():
            # Find contradictions within this topic
            topic_contradictions = [
                c for c in contradictions
                if c.claim_a in topic_claims and c.claim_b in topic_claims
            ]

            if topic_contradictions:
                # Separate supporting vs contradicting claims
                supporting, contradicting = self._separate_pro_contra_claims(topic_claims, topic_contradictions)

                # Calculate confidence and calibration
                confidence = self._calculate_set_confidence(supporting, contradicting, topic_contradictions)
                calibration = self._get_calibration_hint(confidence, len(supporting), len(contradicting))

                contradiction_set = ContradictionSet(
                    topic=topic,
                    supporting_claims=supporting,
                    contradicting_claims=contradicting,
                    contradictions=topic_contradictions,
                    confidence_score=confidence,
                    calibration_hint=calibration
                )

                contradiction_sets.append(contradiction_set)

        logger.info(f"Created {len(contradiction_sets)} contradiction sets")
        return contradiction_sets

    def _group_claims_by_topic(self, claims: List[Claim]) -> Dict[str, List[Claim]]:
        """Group claims by topic/theme"""

        # Simple topic extraction based on keywords
        topic_keywords = {
            "vaccine_effectiveness": ["vaccine", "effectiveness", "efficacy", "protection"],
            "vaccine_safety": ["vaccine", "safety", "side effects", "adverse"],
            "covid_transmission": ["transmission", "spread", "contagious", "infection"],
            "treatment": ["treatment", "therapy", "drug", "medication"],
            "policy": ["policy", "mandate", "restriction", "regulation"]
        }

        groups = {}

        for claim in claims:
            text_lower = claim.text.lower()

            # Find best matching topic
            best_topic = "general"
            best_score = 0

            for topic, keywords in topic_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                if score > best_score:
                    best_score = score
                    best_topic = topic

            if best_topic not in groups:
                groups[best_topic] = []
            groups[best_topic].append(claim)

        return groups

    def _separate_pro_contra_claims(
        self,
        claims: List[Claim],
        contradictions: List[Contradiction]
    ) -> Tuple[List[Claim], List[Claim]]:
        """Separate claims into supporting vs contradicting groups"""

        # Build contradiction graph
        contra_pairs = set()
        for contradiction in contradictions:
            contra_pairs.add((contradiction.claim_a.id, contradiction.claim_b.id))
            contra_pairs.add((contradiction.claim_b.id, contradiction.claim_a.id))

        # Simple grouping: majority sentiment
        positive_claims = []
        negative_claims = []

        for claim in claims:
            sentiment = self._analyze_sentiment(claim.text)
            if sentiment > 0:
                positive_claims.append(claim)
            else:
                negative_claims.append(claim)

        return positive_claims, negative_claims

    def _calculate_set_confidence(
        self,
        supporting: List[Claim],
        contradicting: List[Claim],
        contradictions: List[Contradiction]
    ) -> float:
        """Calculate confidence score for contradiction set"""

        if not contradictions:
            return 0.0

        # Average contradiction confidence
        avg_contradiction_confidence = np.mean([c.confidence for c in contradictions])

        # Evidence balance factor
        total_supporting_evidence = sum(len(claim.evidence) for claim in supporting)
        total_contradicting_evidence = sum(len(claim.evidence) for claim in contradicting)

        if total_supporting_evidence + total_contradicting_evidence == 0:
            evidence_balance = 0.5
        else:
            evidence_balance = min(total_supporting_evidence, total_contradicting_evidence) / (total_supporting_evidence + total_contradicting_evidence)

        # Combine factors
        final_confidence = avg_contradiction_confidence * (0.7 + 0.3 * evidence_balance)

        return min(final_confidence, 1.0)

    def _get_calibration_hint(self, confidence: float, pro_count: int, contra_count: int) -> str:
        """Get calibration hint based on confidence and evidence"""

        if confidence > 0.8:
            level = ConfidenceLevel.HIGH
        elif confidence > 0.5:
            level = ConfidenceLevel.MEDIUM
        elif confidence > 0.2:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.UNCERTAIN

        base_message = self.calibration_messages[level]

        # Add evidence balance information
        total_evidence = pro_count + contra_count
        if total_evidence > 0:
            pro_ratio = pro_count / total_evidence
            if pro_ratio > 0.8:
                balance_note = " Evidence heavily favors one side."
            elif pro_ratio < 0.2:
                balance_note = " Evidence heavily contradicts main position."
            else:
                balance_note = " Evidence is relatively balanced."
        else:
            balance_note = " Limited evidence available."

        return base_message + balance_note


def create_contradiction_detector(config: Dict[str, Any]) -> ContradictionDetector:
    """Factory function for contradiction detector"""
    return ContradictionDetector(config)


# Usage example
if __name__ == "__main__":
    config = {
        "verification": {
            "semantic_threshold": 0.7,
            "temporal_threshold": 0.8,
            "quantitative_threshold": 0.6
        }
    }

    detector = ContradictionDetector(config)

    # Test claims
    claims = [
        Claim(
            id="claim_1",
            text="COVID-19 vaccines are 95% effective at preventing infection",
            evidence=[{"source": "study_1", "confidence": 0.9}],
            confidence=0.9,
            source_urls=["https://example.com/study1"]
        ),
        Claim(
            id="claim_2",
            text="COVID-19 vaccines are not effective at preventing infection",
            evidence=[{"source": "study_2", "confidence": 0.8}],
            confidence=0.8,
            source_urls=["https://example.com/study2"]
        )
    ]

    contradictions = detector.detect_contradictions(claims)
    print(f"Detected {len(contradictions)} contradictions")

    if contradictions:
        contradiction_sets = detector.create_contradiction_sets(claims, contradictions)
        print(f"Created {len(contradiction_sets)} contradiction sets")

        for cs in contradiction_sets:
            print(f"Topic: {cs.topic}")
            print(f"Supporting: {len(cs.supporting_claims)}, Contradicting: {len(cs.contradicting_claims)}")
            print(f"Confidence: {cs.confidence_score:.2f}")
            print(f"Calibration: {cs.calibration_hint}")
