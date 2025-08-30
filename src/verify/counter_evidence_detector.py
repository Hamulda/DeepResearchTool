#!/usr/bin/env python3
"""Counter-Evidence Detection System
Specializovaný systém pro detekci a analýzu protikladných důkazů

Author: Senior Python/MLOps Agent
"""

from collections import defaultdict
from dataclasses import dataclass
import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CounterEvidence:
    """Protikladný důkaz"""

    evidence_id: str
    original_claim: str
    counter_claim: str
    source_doc_id: str
    confidence: float
    contradiction_type: str  # direct, partial, contextual, methodological
    evidence_text: str
    char_start: int
    char_end: int
    credibility_score: float


@dataclass
class ContradictionAnalysis:
    """Analýza rozporů mezi důkazy"""

    claim_id: str
    supporting_evidence: list[str]
    counter_evidence: list[CounterEvidence]
    contradiction_strength: float
    resolution_strategy: str
    confidence_adjustment: float
    analysis_notes: str


@dataclass
class DisagreementCoverage:
    """Metriky pokrytí disagreement"""

    total_claims: int
    claims_with_counter_evidence: int
    disagreement_ratio: float
    quality_counter_evidence: int
    coverage_score: float
    detailed_analysis: list[ContradictionAnalysis]


class CounterEvidenceDetector:
    """Detektor protikladných důkazů s pokročilou analýzou"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.counter_evidence_config = config.get("counter_evidence", {})

        # Detection settings
        self.enable_counter_search = self.counter_evidence_config.get("enabled", True)
        self.min_contradiction_confidence = self.counter_evidence_config.get("min_confidence", 0.6)
        self.max_counter_evidence_per_claim = self.counter_evidence_config.get("max_per_claim", 5)

        # Contradiction patterns
        self.contradiction_patterns = self._load_contradiction_patterns()

        # Quality thresholds
        self.quality_thresholds = self.counter_evidence_config.get("quality_thresholds", {})
        self.min_disagreement_coverage = self.quality_thresholds.get(
            "min_disagreement_coverage", 0.3
        )

        # Analysis cache
        self.analysis_cache = {}

    def _load_contradiction_patterns(self) -> dict[str, list[str]]:
        """Načtení vzorů pro detekci rozporů"""
        patterns = {
            "direct_negation": [
                r"\b(not|no|never|none|neither|nor)\s+",
                r"\b(incorrect|false|wrong|untrue|inaccurate)\b",
                r"\b(refutes?|disproves?|contradicts?|disputes?)\b",
                r"\b(fails to|unable to|cannot|does not)\b",
            ],
            "contrast_indicators": [
                r"\b(however|but|although|though|despite|nevertheless)\b",
                r"\b(on the contrary|in contrast|conversely|opposite)\b",
                r"\b(while|whereas|instead|rather than)\b",
                r"\b(alternatively|different|contrary to)\b",
            ],
            "questioning_validity": [
                r"\b(questionable|doubtful|uncertain|unclear)\b",
                r"\b(limited evidence|insufficient data|weak support)\b",
                r"\b(controversy|debate|disagreement|dispute)\b",
                r"\b(challenges?|criticizes?|questions?)\b",
            ],
            "methodological_criticism": [
                r"\b(flawed methodology|biased sample|inadequate controls)\b",
                r"\b(correlation not causation|confounding factors)\b",
                r"\b(replication crisis|irreproducible|unreliable)\b",
                r"\b(statistical significance|p-hacking|publication bias)\b",
            ],
        }

        return patterns

    async def detect_counter_evidence(
        self, claims: list[dict[str, Any]], evidence_passages: list[dict[str, Any]]
    ) -> DisagreementCoverage:
        """Hlavní detekce counter-evidence pro všechny claims

        Args:
            claims: Seznam claims s evidence bindings
            evidence_passages: Všechny dostupné evidence passages

        Returns:
            DisagreementCoverage s complete analysis

        """
        logger.info(f"Starting counter-evidence detection for {len(claims)} claims")

        if not self.enable_counter_search:
            logger.info("Counter-evidence search disabled")
            return self._empty_disagreement_coverage(len(claims))

        start_time = time.time()
        detailed_analysis = []

        for claim in claims:
            analysis = await self._analyze_claim_contradictions(claim, evidence_passages)
            detailed_analysis.append(analysis)

        # Calculate disagreement coverage metrics
        coverage = self._calculate_disagreement_coverage(detailed_analysis)

        processing_time = time.time() - start_time
        logger.info(f"Counter-evidence detection completed in {processing_time:.2f}s")
        logger.info(f"Disagreement coverage: {coverage.disagreement_ratio:.1%}")

        return coverage

    async def _analyze_claim_contradictions(
        self, claim: dict[str, Any], evidence_passages: list[dict[str, Any]]
    ) -> ContradictionAnalysis:
        """Analýza rozporů pro konkrétní claim"""
        claim_text = claim.get("text", "")
        claim_id = claim.get("id", "unknown")

        # Find potential counter-evidence
        counter_evidence = await self._find_counter_evidence_for_claim(
            claim_text, evidence_passages
        )

        # Analyze contradiction strength
        contradiction_strength = self._calculate_contradiction_strength(
            claim_text, counter_evidence
        )

        # Determine resolution strategy
        resolution_strategy = self._determine_resolution_strategy(claim, counter_evidence)

        # Calculate confidence adjustment
        confidence_adjustment = self._calculate_confidence_adjustment(
            counter_evidence, contradiction_strength
        )

        # Generate analysis notes
        analysis_notes = self._generate_analysis_notes(
            claim, counter_evidence, contradiction_strength
        )

        analysis = ContradictionAnalysis(
            claim_id=claim_id,
            supporting_evidence=claim.get("supporting_evidence", []),
            counter_evidence=counter_evidence,
            contradiction_strength=contradiction_strength,
            resolution_strategy=resolution_strategy,
            confidence_adjustment=confidence_adjustment,
            analysis_notes=analysis_notes,
        )

        return analysis

    async def _find_counter_evidence_for_claim(
        self, claim_text: str, evidence_passages: list[dict[str, Any]]
    ) -> list[CounterEvidence]:
        """Nalezení counter-evidence pro konkrétní claim"""
        counter_evidence = []
        claim_terms = set(claim_text.lower().split())

        for passage in evidence_passages:
            content = passage.get("content", "")
            doc_id = passage.get("doc_id", "unknown")

            # Check for contradiction patterns
            contradictions = self._detect_contradictions_in_passage(claim_text, content, doc_id)

            for contradiction in contradictions:
                if contradiction.confidence >= self.min_contradiction_confidence:
                    counter_evidence.append(contradiction)

        # Sort by confidence and limit count
        counter_evidence.sort(key=lambda x: x.confidence, reverse=True)
        return counter_evidence[: self.max_counter_evidence_per_claim]

    def _detect_contradictions_in_passage(
        self, claim_text: str, passage_content: str, doc_id: str
    ) -> list[CounterEvidence]:
        """Detekce rozporů v konkrétní pasáži"""
        contradictions = []
        claim_terms = set(claim_text.lower().split())

        # Split passage into sentences
        sentences = re.split(r"[.!?]+\s+", passage_content)
        char_offset = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                char_offset += len(sentence) + 1
                continue

            sentence_terms = set(sentence.lower().split())

            # Check for term overlap (relevance to claim)
            term_overlap = len(claim_terms.intersection(sentence_terms))
            if term_overlap < 2:
                char_offset += len(sentence) + 1
                continue

            # Check each contradiction pattern type
            for contradiction_type, patterns in self.contradiction_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        # Found potential contradiction
                        confidence = self._calculate_contradiction_confidence(
                            claim_text, sentence, contradiction_type, term_overlap
                        )

                        if confidence >= self.min_contradiction_confidence:
                            start_pos = passage_content.find(sentence, char_offset)

                            contradiction = CounterEvidence(
                                evidence_id=f"counter_{doc_id}_{len(contradictions)}",
                                original_claim=claim_text,
                                counter_claim=sentence,
                                source_doc_id=doc_id,
                                confidence=confidence,
                                contradiction_type=contradiction_type,
                                evidence_text=sentence,
                                char_start=start_pos,
                                char_end=start_pos + len(sentence),
                                credibility_score=self._assess_source_credibility(doc_id),
                            )

                            contradictions.append(contradiction)
                        break  # Found pattern, no need to check others for this sentence

            char_offset += len(sentence) + 1

        return contradictions

    def _calculate_contradiction_confidence(
        self, claim_text: str, counter_sentence: str, contradiction_type: str, term_overlap: int
    ) -> float:
        """Výpočet confidence pro contradiction"""
        base_confidence = 0.5

        # Adjust by contradiction type strength
        type_weights = {
            "direct_negation": 0.9,
            "methodological_criticism": 0.8,
            "contrast_indicators": 0.7,
            "questioning_validity": 0.6,
        }

        base_confidence *= type_weights.get(contradiction_type, 0.5)

        # Boost for higher term overlap
        overlap_boost = min(term_overlap / 5.0, 0.3)
        base_confidence += overlap_boost

        # Boost for strong contradiction indicators
        strong_indicators = ["not", "false", "incorrect", "refutes", "disproves"]
        if any(indicator in counter_sentence.lower() for indicator in strong_indicators):
            base_confidence += 0.2

        # Penalty for weak language
        weak_indicators = ["might", "could", "possibly", "perhaps", "maybe"]
        if any(indicator in counter_sentence.lower() for indicator in weak_indicators):
            base_confidence -= 0.2

        return max(0.1, min(1.0, base_confidence))

    def _assess_source_credibility(self, doc_id: str) -> float:
        """Odhad credibility zdroje"""
        # Simple heuristic based on source type (would be enhanced with real credibility data)
        if any(indicator in doc_id.lower() for indicator in ["academic", "journal", "research"]):
            return 0.9
        if any(indicator in doc_id.lower() for indicator in ["government", "official"]):
            return 0.8
        if any(indicator in doc_id.lower() for indicator in ["news", "media"]):
            return 0.6
        return 0.5

    def _calculate_contradiction_strength(
        self, claim_text: str, counter_evidence: list[CounterEvidence]
    ) -> float:
        """Výpočet celkové síly contradictions"""
        if not counter_evidence:
            return 0.0

        # Weight by confidence and credibility
        weighted_contradictions = []
        for evidence in counter_evidence:
            weight = evidence.confidence * evidence.credibility_score
            weighted_contradictions.append(weight)

        # Calculate average weighted contradiction strength
        avg_strength = sum(weighted_contradictions) / len(weighted_contradictions)

        # Boost for multiple independent contradictions
        independence_boost = min(len(counter_evidence) / 3.0, 0.3)

        total_strength = avg_strength + independence_boost

        return min(total_strength, 1.0)

    def _determine_resolution_strategy(
        self, claim: dict[str, Any], counter_evidence: list[CounterEvidence]
    ) -> str:
        """Určení strategie pro řešení rozporů"""
        if not counter_evidence:
            return "no_contradiction"

        # Count contradiction types
        type_counts = defaultdict(int)
        for evidence in counter_evidence:
            type_counts[evidence.contradiction_type] += 1

        total_contradictions = len(counter_evidence)
        avg_confidence = sum(e.confidence for e in counter_evidence) / total_contradictions

        # Determine strategy based on contradiction characteristics
        if avg_confidence >= 0.8 and total_contradictions >= 3:
            return "major_dispute_flag"
        if "direct_negation" in type_counts and type_counts["direct_negation"] >= 2:
            return "direct_contradiction_warning"
        if "methodological_criticism" in type_counts:
            return "methodological_concerns_note"
        if avg_confidence >= 0.6:
            return "moderate_disagreement_flag"
        return "minor_contradiction_note"

    def _calculate_confidence_adjustment(
        self, counter_evidence: list[CounterEvidence], contradiction_strength: float
    ) -> float:
        """Výpočet úpravy confidence na základě contradictions"""
        if not counter_evidence:
            return 0.0

        # Base adjustment from contradiction strength
        base_adjustment = contradiction_strength * 0.5

        # Additional penalty for high-credibility contradictions
        high_credibility_count = sum(1 for e in counter_evidence if e.credibility_score >= 0.8)
        credibility_penalty = high_credibility_count * 0.1

        # Additional penalty for direct contradictions
        direct_contradictions = sum(
            1 for e in counter_evidence if e.contradiction_type == "direct_negation"
        )
        direct_penalty = direct_contradictions * 0.15

        total_adjustment = base_adjustment + credibility_penalty + direct_penalty

        return min(total_adjustment, 0.8)  # Maximum 80% confidence reduction

    def _generate_analysis_notes(
        self,
        claim: dict[str, Any],
        counter_evidence: list[CounterEvidence],
        contradiction_strength: float,
    ) -> str:
        """Generování analysis notes"""
        if not counter_evidence:
            return "No contradictory evidence found."

        notes = []
        notes.append(
            f"Found {len(counter_evidence)} contradictory evidence(s) with strength {contradiction_strength:.2f}"
        )

        # Group by contradiction type
        type_groups = defaultdict(list)
        for evidence in counter_evidence:
            type_groups[evidence.contradiction_type].append(evidence)

        for contradiction_type, evidences in type_groups.items():
            avg_confidence = sum(e.confidence for e in evidences) / len(evidences)
            notes.append(
                f"- {contradiction_type}: {len(evidences)} instance(s), avg confidence {avg_confidence:.2f}"
            )

        # Highlight high-confidence contradictions
        high_conf_contradictions = [e for e in counter_evidence if e.confidence >= 0.8]
        if high_conf_contradictions:
            notes.append(f"High-confidence contradictions ({len(high_conf_contradictions)}):")
            for evidence in high_conf_contradictions[:2]:  # Show top 2
                notes.append(f"  - {evidence.source_doc_id}: {evidence.counter_claim[:100]}...")

        return " ".join(notes)

    def _calculate_disagreement_coverage(
        self, detailed_analysis: list[ContradictionAnalysis]
    ) -> DisagreementCoverage:
        """Výpočet disagreement coverage metrics"""
        total_claims = len(detailed_analysis)
        claims_with_counter_evidence = sum(
            1 for analysis in detailed_analysis if analysis.counter_evidence
        )

        disagreement_ratio = claims_with_counter_evidence / total_claims if total_claims > 0 else 0

        # Count quality counter-evidence (high confidence + credibility)
        quality_counter_evidence = 0
        for analysis in detailed_analysis:
            quality_evidence = [
                e
                for e in analysis.counter_evidence
                if e.confidence >= 0.7 and e.credibility_score >= 0.7
            ]
            quality_counter_evidence += len(quality_evidence)

        # Calculate coverage score
        coverage_score = self._calculate_coverage_score(
            disagreement_ratio, quality_counter_evidence, total_claims
        )

        coverage = DisagreementCoverage(
            total_claims=total_claims,
            claims_with_counter_evidence=claims_with_counter_evidence,
            disagreement_ratio=disagreement_ratio,
            quality_counter_evidence=quality_counter_evidence,
            coverage_score=coverage_score,
            detailed_analysis=detailed_analysis,
        )

        return coverage

    def _calculate_coverage_score(
        self, disagreement_ratio: float, quality_counter_evidence: int, total_claims: int
    ) -> float:
        """Výpočet overall coverage score"""
        # Base score from disagreement ratio
        base_score = disagreement_ratio

        # Boost for quality counter-evidence
        if total_claims > 0:
            quality_ratio = quality_counter_evidence / total_claims
            quality_boost = min(quality_ratio, 0.3)
            base_score += quality_boost

        # Normalization
        coverage_score = min(base_score, 1.0)

        return coverage_score

    def _empty_disagreement_coverage(self, total_claims: int) -> DisagreementCoverage:
        """Prázdná disagreement coverage když je detekce vypnuta"""
        return DisagreementCoverage(
            total_claims=total_claims,
            claims_with_counter_evidence=0,
            disagreement_ratio=0.0,
            quality_counter_evidence=0,
            coverage_score=0.0,
            detailed_analysis=[],
        )

    async def generate_disagreement_report(self, coverage: DisagreementCoverage) -> dict[str, Any]:
        """Generování disagreement report"""
        report = {
            "disagreement_summary": {
                "total_claims": coverage.total_claims,
                "claims_with_counter_evidence": coverage.claims_with_counter_evidence,
                "disagreement_ratio": f"{coverage.disagreement_ratio:.1%}",
                "quality_counter_evidence": coverage.quality_counter_evidence,
                "coverage_score": f"{coverage.coverage_score:.3f}",
            },
            "detailed_contradictions": [],
            "pattern_analysis": self._analyze_contradiction_patterns(coverage.detailed_analysis),
            "recommendations": self._generate_recommendations(coverage),
        }

        # Add detailed contradiction analysis
        for analysis in coverage.detailed_analysis:
            if analysis.counter_evidence:
                detail = {
                    "claim_id": analysis.claim_id,
                    "contradiction_strength": analysis.contradiction_strength,
                    "resolution_strategy": analysis.resolution_strategy,
                    "confidence_adjustment": analysis.confidence_adjustment,
                    "counter_evidence_count": len(analysis.counter_evidence),
                    "analysis_notes": analysis.analysis_notes,
                    "counter_evidence_details": [
                        {
                            "evidence_id": evidence.evidence_id,
                            "contradiction_type": evidence.contradiction_type,
                            "confidence": evidence.confidence,
                            "credibility_score": evidence.credibility_score,
                            "source_doc_id": evidence.source_doc_id,
                        }
                        for evidence in analysis.counter_evidence
                    ],
                }
                report["detailed_contradictions"].append(detail)

        return report

    def _analyze_contradiction_patterns(
        self, detailed_analysis: list[ContradictionAnalysis]
    ) -> dict[str, Any]:
        """Analýza vzorů v contradictions"""
        pattern_counts = defaultdict(int)
        total_contradictions = 0

        for analysis in detailed_analysis:
            for evidence in analysis.counter_evidence:
                pattern_counts[evidence.contradiction_type] += 1
                total_contradictions += 1

        pattern_analysis = {
            "total_contradictions": total_contradictions,
            "pattern_distribution": dict(pattern_counts),
            "most_common_pattern": (
                max(pattern_counts.items(), key=lambda x: x[1])[0] if pattern_counts else "none"
            ),
        }

        return pattern_analysis

    def _generate_recommendations(self, coverage: DisagreementCoverage) -> list[str]:
        """Generování doporučení na základě disagreement analysis"""
        recommendations = []

        if coverage.disagreement_ratio < 0.2:
            recommendations.append(
                "Low disagreement coverage detected. Consider expanding counter-evidence search."
            )

        if coverage.quality_counter_evidence == 0:
            recommendations.append(
                "No high-quality counter-evidence found. Verify search comprehensiveness."
            )

        if coverage.coverage_score < self.min_disagreement_coverage:
            recommendations.append(
                f"Coverage score ({coverage.coverage_score:.2f}) below threshold ({self.min_disagreement_coverage}). Enhance disagreement detection."
            )

        # Analysis-specific recommendations
        high_contradiction_claims = [
            analysis
            for analysis in coverage.detailed_analysis
            if analysis.contradiction_strength >= 0.7
        ]

        if high_contradiction_claims:
            recommendations.append(
                f"Found {len(high_contradiction_claims)} claim(s) with high contradiction. Consider additional verification."
            )

        if not recommendations:
            recommendations.append("Disagreement coverage analysis completed successfully.")

        return recommendations
