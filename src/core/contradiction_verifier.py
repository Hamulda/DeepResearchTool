#!/usr/bin/env python3
"""Contradiction Verification Engine
Advanced verification with conflict detection and evidence cross-checking

Author: Senior IT Specialist
"""

from dataclasses import dataclass, field
from datetime import datetime
import logging
import re
from typing import Any


@dataclass
class ConflictEvidence:
    """Evidence that contradicts a claim"""

    source_id: str
    snippet: str
    reasoning: str
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ClaimVerification:
    """Verification result for a single claim"""

    claim_id: str
    claim_text: str
    confidence_score: float
    is_disputed: bool
    supporting_evidence: list[dict[str, Any]]
    conflicting_evidence: list[ConflictEvidence]
    verification_method: str
    cross_check_results: dict[str, Any]


class ContraQueryGenerator:
    """Generates counter-queries to find contradicting evidence"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Templates for generating counter-queries
        self.negation_templates = [
            "evidence against {claim}",
            "studies that contradict {claim}",
            "research refuting {claim}",
            "criticism of {claim}",
            "alternative views on {claim}",
            "debunking {claim}",
            "false claims about {topic}",
            "misconceptions about {topic}",
        ]

        # Patterns to extract key topics from claims
        self.topic_patterns = [
            r"(\w+(?:\s+\w+){0,2})\s+(?:is|are|has|have|shows|demonstrates)",
            r"(?:the|a)\s+(\w+(?:\s+\w+){0,2})\s+(?:effect|impact|benefit|risk)",
            r"(\w+(?:\s+\w+){0,2})\s+(?:increases|decreases|improves|reduces)",
        ]

    def extract_key_topics(self, claim: str) -> list[str]:
        """Extract key topics from claim text"""
        topics = []

        # Try pattern matching
        for pattern in self.topic_patterns:
            matches = re.findall(pattern, claim.lower())
            topics.extend(matches)

        # Fallback: extract important nouns (simplified)
        words = claim.split()
        for word in words:
            if len(word) > 4 and word.lower() not in [
                "that",
                "this",
                "with",
                "from",
                "they",
                "were",
                "been",
                "have",
                "said",
            ]:
                topics.append(word)

        # Remove duplicates and return top 3
        unique_topics = list(dict.fromkeys(topics))[:3]
        self.logger.debug(f"Extracted topics from '{claim[:50]}...': {unique_topics}")

        return unique_topics

    def generate_counter_queries(self, claim: str) -> list[str]:
        """Generate queries to find contradictory evidence"""
        topics = self.extract_key_topics(claim)
        counter_queries = []

        # Generate queries for each template
        for template in self.negation_templates:
            # Use full claim
            if "{claim}" in template:
                query = template.format(claim=claim)
                counter_queries.append(query)

            # Use extracted topics
            elif "{topic}" in template:
                for topic in topics[:2]:  # Use top 2 topics
                    query = template.format(topic=topic)
                    counter_queries.append(query)

        # Add specific contradiction patterns
        if topics:
            primary_topic = topics[0]
            specific_queries = [
                f"evidence that {primary_topic} is ineffective",
                f"studies showing {primary_topic} does not work",
                f"research questioning {primary_topic}",
                f"limitations of {primary_topic}",
            ]
            counter_queries.extend(specific_queries)

        self.logger.info(f"Generated {len(counter_queries)} counter-queries for claim")
        return counter_queries[:8]  # Limit to prevent too many queries


class ConflictDetector:
    """Detects conflicts between supporting and contradicting evidence"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)

        # Conflict indicators
        self.conflict_keywords = [
            "however",
            "but",
            "although",
            "contrary",
            "opposite",
            "contradicts",
            "refutes",
            "disproves",
            "challenges",
            "questions",
            "disputes",
            "false",
            "incorrect",
            "wrong",
            "misleading",
            "myth",
            "debunked",
        ]

    def _check_lexical_conflicts(self, support_text: str, contra_text: str) -> float:
        """Check for lexical conflict indicators"""
        contra_lower = contra_text.lower()
        conflict_score = 0.0

        # Count conflict keywords
        for keyword in self.conflict_keywords:
            if keyword in contra_lower:
                conflict_score += 0.1

        # Check for direct negations
        support_words = set(support_text.lower().split())
        contra_words = set(contra_text.lower().split())

        # Look for "not + word" patterns where word is in support
        contra_text_words = contra_text.lower().split()
        for i, word in enumerate(contra_text_words):
            if word == "not" and i + 1 < len(contra_text_words):
                next_word = contra_text_words[i + 1]
                if next_word in support_words:
                    conflict_score += 0.2

        return min(conflict_score, 1.0)

    async def detect_conflict(
        self, claim: str, supporting_evidence: list[str], contradicting_evidence: list[str]
    ) -> tuple[bool, float, str]:
        """Detect if there's a conflict between supporting and contradicting evidence

        Returns:
            - is_conflict: Whether conflict was detected
            - confidence: Confidence in conflict detection (0-1)
            - reasoning: Explanation of the conflict

        """
        if not contradicting_evidence:
            return False, 0.0, "No contradicting evidence found"

        max_conflict_score = 0.0
        conflict_reasoning = ""

        # Check each piece of contradicting evidence against supporting evidence
        for contra_text in contradicting_evidence[:3]:  # Check top 3
            for support_text in supporting_evidence[:3]:
                lexical_score = self._check_lexical_conflicts(support_text, contra_text)

                if lexical_score > max_conflict_score:
                    max_conflict_score = lexical_score
                    conflict_reasoning = (
                        "Potential conflict detected between supporting and contradicting evidence"
                    )

        # Use LLM for deeper conflict analysis if available
        if self.llm_client and max_conflict_score > 0.3:
            try:
                llm_conflict_score, llm_reasoning = await self._llm_conflict_analysis(
                    claim,
                    supporting_evidence[0] if supporting_evidence else "",
                    contradicting_evidence[0],
                )

                # Combine lexical and LLM scores
                combined_score = (max_conflict_score + llm_conflict_score) / 2
                combined_reasoning = f"{conflict_reasoning}. LLM analysis: {llm_reasoning}"

                return combined_score > 0.5, combined_score, combined_reasoning

            except Exception as e:
                self.logger.warning(f"LLM conflict analysis failed: {e}")

        is_conflict = max_conflict_score > 0.4
        return is_conflict, max_conflict_score, conflict_reasoning

    async def _llm_conflict_analysis(
        self, claim: str, support: str, contra: str
    ) -> tuple[float, str]:
        """Use LLM to analyze potential conflicts"""
        prompt = f"""Analyze if there is a contradiction between the following:

CLAIM: {claim}

SUPPORTING EVIDENCE: {support}

CONTRADICTING EVIDENCE: {contra}

Is there a real contradiction? Rate from 0 (no conflict) to 1 (strong conflict) and explain briefly.

Response format:
CONFLICT_SCORE: 0.X
REASONING: Brief explanation"""

        try:
            response = await self.llm_client.generate(prompt, max_tokens=200)

            # Parse response
            score_match = re.search(r"CONFLICT_SCORE:\s*(0\.\d+|1\.0)", response)
            reasoning_match = re.search(r"REASONING:\s*(.+)", response, re.DOTALL)

            score = float(score_match.group(1)) if score_match else 0.5
            reasoning = (
                reasoning_match.group(1).strip() if reasoning_match else "LLM analysis completed"
            )

            return score, reasoning

        except Exception as e:
            self.logger.error(f"LLM conflict analysis error: {e}")
            return 0.5, "LLM analysis failed"


class ContradictionVerifier:
    """Main contradiction verification engine"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.max_counter_queries = config.get("max_counter_queries", 5)
        self.enable_hitl_checkpoint = config.get("enable_hitl_checkpoint", True)

        # Components
        self.query_generator = ContraQueryGenerator()
        self.conflict_detector = ConflictDetector()

        # State
        self.verification_cache = {}

    async def verify_claim(self, claim: dict[str, Any], retriever=None) -> ClaimVerification:
        """Comprehensive verification of a single claim with contradiction checking

        Args:
            claim: Claim object with text and supporting evidence
            retriever: Retrieval engine for finding contradicting evidence

        Returns:
            ClaimVerification object with detailed results

        """
        claim_id = claim.get("id", f"claim_{hash(claim['text'])}")
        claim_text = claim["text"]
        supporting_evidence = claim.get("evidence", [])

        self.logger.info(f"Verifying claim: {claim_text[:100]}...")

        # Extract supporting evidence text
        support_texts = [ev.get("snippet", "") for ev in supporting_evidence]

        # Generate counter-queries
        counter_queries = self.query_generator.generate_counter_queries(claim_text)

        # Search for contradicting evidence
        conflicting_evidence = []
        if retriever:
            for query in counter_queries[: self.max_counter_queries]:
                try:
                    contra_results = await retriever.retrieve(query, top_k=3)

                    for result in contra_results:
                        conflict_ev = ConflictEvidence(
                            source_id=result.get("doc_id", "unknown"),
                            snippet=result.get("content", "")[:500],
                            reasoning=f"Found via counter-query: '{query}'",
                            confidence=result.get("score", 0.5),
                        )
                        conflicting_evidence.append(conflict_ev)

                except Exception as e:
                    self.logger.warning(f"Counter-query failed: {query} - {e}")

        # Detect conflicts
        contra_texts = [ce.snippet for ce in conflicting_evidence]
        is_conflict, conflict_confidence, conflict_reasoning = (
            await self.conflict_detector.detect_conflict(claim_text, support_texts, contra_texts)
        )

        # Calculate overall confidence score
        base_confidence = self._calculate_base_confidence(supporting_evidence)

        if is_conflict:
            # Reduce confidence based on conflict strength
            confidence_penalty = conflict_confidence * 0.5
            final_confidence = max(0.1, base_confidence - confidence_penalty)
        else:
            final_confidence = base_confidence

        # Determine if claim is disputed
        is_disputed = is_conflict and conflict_confidence > 0.5 and len(conflicting_evidence) >= 2

        # Cross-check results
        cross_check = {
            "counter_queries_generated": len(counter_queries),
            "contradicting_sources_found": len(conflicting_evidence),
            "conflict_detected": is_conflict,
            "conflict_confidence": conflict_confidence,
            "conflict_reasoning": conflict_reasoning,
        }

        verification = ClaimVerification(
            claim_id=claim_id,
            claim_text=claim_text,
            confidence_score=final_confidence,
            is_disputed=is_disputed,
            supporting_evidence=supporting_evidence,
            conflicting_evidence=conflicting_evidence,
            verification_method="contradiction_verification",
            cross_check_results=cross_check,
        )

        # Log results
        self.logger.info("Claim verification complete:")
        self.logger.info(f"  Confidence: {final_confidence:.3f}")
        self.logger.info(f"  Disputed: {is_disputed}")
        self.logger.info(f"  Conflicts found: {len(conflicting_evidence)}")

        return verification

    def _calculate_base_confidence(self, supporting_evidence: list[dict[str, Any]]) -> float:
        """Calculate base confidence from supporting evidence quality"""
        if not supporting_evidence:
            return 0.1

        # Factors for confidence calculation
        evidence_count = len(supporting_evidence)

        # Source diversity (different domains)
        sources = set()
        for ev in supporting_evidence:
            source_url = ev.get("source_url", "")
            if source_url:
                # Extract domain
                domain = source_url.split("//")[-1].split("/")[0]
                sources.add(domain)

        source_diversity = min(len(sources) / 3.0, 1.0)  # Max at 3 different sources

        # Evidence quality (based on snippet length and score)
        quality_scores = []
        for ev in supporting_evidence:
            snippet_length = len(ev.get("snippet", ""))
            score = ev.get("score", 0.5)
            quality = min(snippet_length / 200.0, 1.0) * score
            quality_scores.append(quality)

        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5

        # Combine factors
        evidence_factor = min(evidence_count / 3.0, 1.0)  # Max at 3 pieces of evidence
        base_confidence = evidence_factor * 0.4 + source_diversity * 0.3 + avg_quality * 0.3

        return max(0.1, min(base_confidence, 0.9))

    async def verify_claims_batch(
        self, claims: list[dict[str, Any]], retriever=None
    ) -> list[ClaimVerification]:
        """Verify multiple claims with contradiction checking"""
        self.logger.info(f"Starting batch verification of {len(claims)} claims")

        verifications = []
        disputed_count = 0

        for i, claim in enumerate(claims, 1):
            self.logger.info(f"Verifying claim {i}/{len(claims)}")

            verification = await self.verify_claim(claim, retriever)
            verifications.append(verification)

            if verification.is_disputed:
                disputed_count += 1
                self.logger.warning(f"Disputed claim detected: {verification.claim_text[:100]}...")

        self.logger.info(f"Batch verification complete: {disputed_count} disputed claims found")

        # HiTL checkpoint for disputed claims
        if disputed_count > 0 and self.enable_hitl_checkpoint:
            self.logger.info("HiTL checkpoint triggered due to disputed claims")
            # In real implementation, this would trigger human review workflow

        return verifications


def create_contradiction_verifier(config: dict[str, Any]) -> ContradictionVerifier:
    """Factory function for contradiction verifier"""
    return ContradictionVerifier(config)
