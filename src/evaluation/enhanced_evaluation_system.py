#!/usr/bin/env python3
"""Enhanced RAG Evaluation System
Comprehensive metrics for retrieval, citation, groundedness and hallucination detection

Author: Senior IT Specialist
"""

from dataclasses import dataclass, field
from datetime import datetime
import logging
import re
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics"""

    # Retrieval metrics
    retrieval_recall_at_k: dict[int, float] = field(default_factory=dict)
    retrieval_precision_at_k: dict[int, float] = field(default_factory=dict)
    retrieval_ndcg_at_k: dict[int, float] = field(default_factory=dict)
    retrieval_mrr: float = 0.0

    # Citation metrics
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    citation_f1: float = 0.0

    # Groundedness metrics
    groundedness_rate: float = 0.0
    claim_support_ratio: float = 0.0
    evidence_coverage: float = 0.0

    # Hallucination metrics
    hallucination_rate: float = 0.0
    unsupported_claims_ratio: float = 0.0

    # Overall quality
    answer_relevance: float = 0.0
    answer_completeness: float = 0.0

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    profile: str = "unknown"
    total_claims: int = 0
    total_evidence: int = 0


class RetrievalEvaluator:
    """Evaluates retrieval system performance"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def evaluate_retrieval(
        self,
        retrieved_docs: list[dict[str, Any]],
        ground_truth_docs: list[str],
        k_values: list[int] = None,
    ) -> dict[str, float]:
        """Evaluate retrieval metrics at different k values

        Args:
            retrieved_docs: List of retrieved documents with scores
            ground_truth_docs: List of relevant document IDs
            k_values: K values to evaluate (default: [1, 3, 5, 10, 20])

        Returns:
            Dictionary with retrieval metrics

        """
        if k_values is None:
            k_values = [1, 3, 5, 10, 20]

        if not retrieved_docs or not ground_truth_docs:
            return (
                {f"recall_at_{k}": 0.0 for k in k_values}
                | {f"precision_at_{k}": 0.0 for k in k_values}
                | {f"ndcg_at_{k}": 0.0 for k in k_values}
                | {"mrr": 0.0}
            )

        retrieved_ids = [doc.get("doc_id", "") for doc in retrieved_docs]
        ground_truth_set = set(ground_truth_docs)

        metrics = {}

        # Calculate metrics for each k
        for k in k_values:
            top_k_ids = retrieved_ids[:k]
            relevant_retrieved = [doc_id for doc_id in top_k_ids if doc_id in ground_truth_set]

            # Recall@k
            recall_k = len(relevant_retrieved) / len(ground_truth_set) if ground_truth_set else 0
            metrics[f"recall_at_{k}"] = recall_k

            # Precision@k
            precision_k = len(relevant_retrieved) / k if k > 0 else 0
            metrics[f"precision_at_{k}"] = precision_k

            # nDCG@k
            dcg = 0
            for i, doc_id in enumerate(top_k_ids):
                if doc_id in ground_truth_set:
                    dcg += 1 / np.log2(i + 2)  # i+2 because log2(1) = 0

            # Ideal DCG (assuming all relevant docs are at the top)
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(ground_truth_set), k)))

            ndcg_k = dcg / idcg if idcg > 0 else 0
            metrics[f"ndcg_at_{k}"] = ndcg_k

        # Mean Reciprocal Rank (MRR)
        mrr = 0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in ground_truth_set:
                mrr = 1 / (i + 1)
                break
        metrics["mrr"] = mrr

        self.logger.info(
            f"Retrieval evaluation: R@10={metrics.get('recall_at_10', 0):.3f}, "
            f"P@10={metrics.get('precision_at_10', 0):.3f}, "
            f"nDCG@10={metrics.get('ndcg_at_10', 0):.3f}"
        )

        return metrics


class CitationEvaluator:
    """Evaluates citation quality and accuracy"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sentence_encoder = None

    def _load_encoder(self):
        """Lazy load sentence encoder"""
        if self.sentence_encoder is None:
            try:
                self.sentence_encoder = SentenceTransformer("all-MiniLM-L6-v2")
                self.logger.info("Sentence encoder loaded for citation evaluation")
            except Exception as e:
                self.logger.warning(f"Failed to load sentence encoder: {e}")

    def evaluate_citations(
        self, claims: list[dict[str, Any]], all_evidence: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Evaluate citation precision and recall

        Args:
            claims: List of claims with their evidence
            all_evidence: All available evidence from retrieval

        Returns:
            Citation metrics

        """
        if not claims:
            return {"citation_precision": 0.0, "citation_recall": 0.0, "citation_f1": 0.0}

        total_cited_evidence = 0
        total_relevant_citations = 0
        total_available_evidence = len(all_evidence)
        total_claims_with_citations = 0

        for claim in claims:
            claim_text = claim.get("text", "")
            claim_evidence = claim.get("evidence", [])

            if claim_evidence:
                total_claims_with_citations += 1

            cited_evidence_ids = set()
            relevant_citations = 0

            for evidence in claim_evidence:
                evidence_id = evidence.get("source_id", evidence.get("doc_id", ""))
                cited_evidence_ids.add(evidence_id)

                # Check if citation is relevant to claim
                if self._is_citation_relevant(claim_text, evidence):
                    relevant_citations += 1

            total_cited_evidence += len(cited_evidence_ids)
            total_relevant_citations += relevant_citations

        # Citation Precision: relevant citations / total citations
        citation_precision = (
            total_relevant_citations / total_cited_evidence if total_cited_evidence > 0 else 0
        )

        # Citation Recall: claims with citations / total claims
        citation_recall = total_claims_with_citations / len(claims) if len(claims) > 0 else 0

        # Citation F1
        citation_f1 = (
            2 * citation_precision * citation_recall / (citation_precision + citation_recall)
            if (citation_precision + citation_recall) > 0
            else 0
        )

        self.logger.info(
            f"Citation evaluation: Precision={citation_precision:.3f}, "
            f"Recall={citation_recall:.3f}, F1={citation_f1:.3f}"
        )

        return {
            "citation_precision": citation_precision,
            "citation_recall": citation_recall,
            "citation_f1": citation_f1,
        }

    def _is_citation_relevant(self, claim_text: str, evidence: dict[str, Any]) -> bool:
        """Check if citation is relevant to claim using semantic similarity"""
        evidence_text = evidence.get("snippet", evidence.get("content", ""))

        if not claim_text or not evidence_text:
            return False

        # Simple keyword overlap as fallback
        claim_words = set(claim_text.lower().split())
        evidence_words = set(evidence_text.lower().split())

        overlap = len(claim_words & evidence_words)
        overlap_ratio = (
            overlap / min(len(claim_words), len(evidence_words))
            if claim_words and evidence_words
            else 0
        )

        # Use semantic similarity if available
        if self.sentence_encoder is None:
            self._load_encoder()

        if self.sentence_encoder:
            try:
                claim_embedding = self.sentence_encoder.encode([claim_text])
                evidence_embedding = self.sentence_encoder.encode([evidence_text])
                similarity = cosine_similarity(claim_embedding, evidence_embedding)[0][0]

                # Combine keyword overlap and semantic similarity
                relevance_score = 0.6 * similarity + 0.4 * overlap_ratio
                return relevance_score > 0.5
            except Exception as e:
                self.logger.warning(f"Semantic similarity failed: {e}")

        # Fallback to keyword overlap
        return overlap_ratio > 0.3


class GroundednessEvaluator:
    """Evaluates how well claims are grounded in evidence"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def evaluate_groundedness(
        self, claims: list[dict[str, Any]], min_evidence_per_claim: int = 2
    ) -> dict[str, float]:
        """Evaluate groundedness metrics

        Args:
            claims: List of claims with evidence
            min_evidence_per_claim: Minimum evidence required per claim

        Returns:
            Groundedness metrics

        """
        if not claims:
            return {"groundedness_rate": 0.0, "claim_support_ratio": 0.0, "evidence_coverage": 0.0}

        grounded_claims = 0
        total_evidence_count = 0
        claims_with_min_evidence = 0

        for claim in claims:
            evidence_list = claim.get("evidence", [])
            evidence_count = len(evidence_list)
            total_evidence_count += evidence_count

            # Check if claim is grounded (has evidence)
            if evidence_count > 0:
                grounded_claims += 1

            # Check if claim has minimum required evidence
            if evidence_count >= min_evidence_per_claim:
                claims_with_min_evidence += 1

        # Groundedness rate: claims with evidence / total claims
        groundedness_rate = grounded_claims / len(claims)

        # Claim support ratio: claims with min evidence / total claims
        claim_support_ratio = claims_with_min_evidence / len(claims)

        # Evidence coverage: average evidence per claim
        evidence_coverage = total_evidence_count / len(claims)

        self.logger.info(
            f"Groundedness evaluation: Rate={groundedness_rate:.3f}, "
            f"Support ratio={claim_support_ratio:.3f}, "
            f"Coverage={evidence_coverage:.1f}"
        )

        return {
            "groundedness_rate": groundedness_rate,
            "claim_support_ratio": claim_support_ratio,
            "evidence_coverage": evidence_coverage,
        }


class HallucinationDetector:
    """Detects hallucinations and unsupported claims"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Hallucination indicators
        self.hallucination_patterns = [
            r"\b(definitely|certainly|absolutely|undoubtedly)\b",  # Overconfident language
            r"\b(all|every|never|always|none)\b",  # Absolute statements
            r"\b(\d+\.\d+%|\d+%)\b",  # Specific percentages without source
            r"\b(studies show|research proves|scientists say)\b",  # Vague attributions
        ]

    def detect_hallucinations(
        self, claims: list[dict[str, Any]], all_evidence: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Detect potential hallucinations in claims

        Args:
            claims: List of claims to check
            all_evidence: All available evidence

        Returns:
            Hallucination metrics

        """
        if not claims:
            return {"hallucination_rate": 0.0, "unsupported_claims_ratio": 0.0}

        hallucination_count = 0
        unsupported_claims = 0

        # Create evidence lookup
        evidence_texts = set()
        for evidence in all_evidence:
            evidence_text = evidence.get("snippet", evidence.get("content", "")).lower()
            evidence_texts.add(evidence_text)

        for claim in claims:
            claim_text = claim.get("text", "")
            claim_evidence = claim.get("evidence", [])

            # Check for unsupported claims
            if not claim_evidence:
                unsupported_claims += 1
                continue

            # Check for hallucination patterns
            hallucination_score = self._calculate_hallucination_score(claim_text)

            # Check if claim content is supported by evidence
            is_supported = self._is_claim_supported(claim_text, claim_evidence, evidence_texts)

            if hallucination_score > 0.5 or not is_supported:
                hallucination_count += 1
                self.logger.debug(f"Potential hallucination detected: {claim_text[:100]}...")

        hallucination_rate = hallucination_count / len(claims)
        unsupported_claims_ratio = unsupported_claims / len(claims)

        self.logger.info(
            f"Hallucination detection: Rate={hallucination_rate:.3f}, "
            f"Unsupported ratio={unsupported_claims_ratio:.3f}"
        )

        return {
            "hallucination_rate": hallucination_rate,
            "unsupported_claims_ratio": unsupported_claims_ratio,
        }

    def _calculate_hallucination_score(self, claim_text: str) -> float:
        """Calculate hallucination score based on linguistic patterns"""
        score = 0.0
        text_lower = claim_text.lower()

        for pattern in self.hallucination_patterns:
            matches = re.findall(pattern, text_lower)
            score += len(matches) * 0.2

        return min(score, 1.0)

    def _is_claim_supported(
        self, claim_text: str, claim_evidence: list[dict[str, Any]], all_evidence_texts: set
    ) -> bool:
        """Check if claim is supported by evidence"""
        claim_words = set(claim_text.lower().split())

        for evidence in claim_evidence:
            evidence_text = evidence.get("snippet", evidence.get("content", "")).lower()
            evidence_words = set(evidence_text.split())

            # Check word overlap
            overlap = len(claim_words & evidence_words)
            overlap_ratio = overlap / len(claim_words) if claim_words else 0

            if overlap_ratio > 0.3:  # 30% word overlap threshold
                return True

        return False


class ComprehensiveRAGEvaluator:
    """Main RAG evaluation system combining all metrics"""

    def __init__(self, config: dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize evaluators
        self.retrieval_evaluator = RetrievalEvaluator()
        self.citation_evaluator = CitationEvaluator()
        self.groundedness_evaluator = GroundednessEvaluator()
        self.hallucination_detector = HallucinationDetector()

        # Quality gate thresholds (profile-specific)
        self.quality_gates = {
            "quick": {
                "groundedness_rate": 0.75,
                "citation_precision": 0.65,
                "retrieval_recall_at_10": 0.60,
                "hallucination_rate": 0.25,
            },
            "thorough": {
                "groundedness_rate": 0.85,
                "citation_precision": 0.75,
                "retrieval_recall_at_10": 0.70,
                "hallucination_rate": 0.15,
            },
        }

    async def evaluate_response(self, evaluation_data: dict[str, Any]) -> EvaluationMetrics:
        """Comprehensive evaluation of RAG response

        Args:
            evaluation_data: Contains query, claims, evidence, ground_truth, etc.

        Returns:
            Complete evaluation metrics

        """
        self.logger.info("Starting comprehensive RAG evaluation...")

        claims = evaluation_data.get("claims", [])
        evidence = evaluation_data.get("evidence", [])
        ground_truth_docs = evaluation_data.get("ground_truth_docs", [])
        retrieved_docs = evaluation_data.get("retrieved_docs", [])
        profile = evaluation_data.get("profile", "quick")

        metrics = EvaluationMetrics(profile=profile)

        # Retrieval evaluation
        if retrieved_docs and ground_truth_docs:
            retrieval_metrics = self.retrieval_evaluator.evaluate_retrieval(
                retrieved_docs, ground_truth_docs
            )

            metrics.retrieval_recall_at_k = {
                k: retrieval_metrics.get(f"recall_at_{k}", 0.0) for k in [1, 3, 5, 10, 20]
            }
            metrics.retrieval_precision_at_k = {
                k: retrieval_metrics.get(f"precision_at_{k}", 0.0) for k in [1, 3, 5, 10, 20]
            }
            metrics.retrieval_ndcg_at_k = {
                k: retrieval_metrics.get(f"ndcg_at_{k}", 0.0) for k in [1, 3, 5, 10, 20]
            }
            metrics.retrieval_mrr = retrieval_metrics.get("mrr", 0.0)

        # Citation evaluation
        if claims:
            citation_metrics = self.citation_evaluator.evaluate_citations(claims, evidence)
            metrics.citation_precision = citation_metrics["citation_precision"]
            metrics.citation_recall = citation_metrics["citation_recall"]
            metrics.citation_f1 = citation_metrics["citation_f1"]

        # Groundedness evaluation
        if claims:
            groundedness_metrics = self.groundedness_evaluator.evaluate_groundedness(claims)
            metrics.groundedness_rate = groundedness_metrics["groundedness_rate"]
            metrics.claim_support_ratio = groundedness_metrics["claim_support_ratio"]
            metrics.evidence_coverage = groundedness_metrics["evidence_coverage"]

        # Hallucination detection
        if claims:
            hallucination_metrics = self.hallucination_detector.detect_hallucinations(
                claims, evidence
            )
            metrics.hallucination_rate = hallucination_metrics["hallucination_rate"]
            metrics.unsupported_claims_ratio = hallucination_metrics["unsupported_claims_ratio"]

        # Overall quality metrics
        metrics.answer_relevance = self._calculate_answer_relevance(
            evaluation_data.get("query", ""), claims
        )
        metrics.answer_completeness = self._calculate_answer_completeness(
            claims, evaluation_data.get("expected_claims", [])
        )

        # Metadata
        metrics.total_claims = len(claims)
        metrics.total_evidence = len(evidence)

        # Log summary
        self.logger.info("RAG evaluation complete:")
        self.logger.info(f"  Groundedness: {metrics.groundedness_rate:.3f}")
        self.logger.info(f"  Citation precision: {metrics.citation_precision:.3f}")
        self.logger.info(f"  Hallucination rate: {metrics.hallucination_rate:.3f}")
        self.logger.info(f"  Retrieval R@10: {metrics.retrieval_recall_at_k.get(10, 0):.3f}")

        return metrics

    def _calculate_answer_relevance(self, query: str, claims: list[dict[str, Any]]) -> float:
        """Calculate how relevant the answer is to the query"""
        if not query or not claims:
            return 0.0

        query_words = set(query.lower().split())

        relevance_scores = []
        for claim in claims:
            claim_text = claim.get("text", "")
            claim_words = set(claim_text.lower().split())

            overlap = len(query_words & claim_words)
            relevance = overlap / len(query_words) if query_words else 0
            relevance_scores.append(relevance)

        return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0

    def _calculate_answer_completeness(
        self, claims: list[dict[str, Any]], expected_claims: list[str]
    ) -> float:
        """Calculate how complete the answer is compared to expected claims"""
        if not expected_claims:
            return 1.0  # No expectations, assume complete

        if not claims:
            return 0.0

        claim_texts = [claim.get("text", "") for claim in claims]
        covered_expectations = 0

        for expected in expected_claims:
            expected_words = set(expected.lower().split())

            for claim_text in claim_texts:
                claim_words = set(claim_text.lower().split())
                overlap = len(expected_words & claim_words)
                overlap_ratio = overlap / len(expected_words) if expected_words else 0

                if overlap_ratio > 0.5:  # 50% word overlap threshold
                    covered_expectations += 1
                    break

        return covered_expectations / len(expected_claims)

    def check_quality_gates(self, metrics: EvaluationMetrics) -> tuple[bool, list[str]]:
        """Check if metrics pass quality gates for the profile

        Returns:
            - passed: Whether all gates passed
            - failures: List of failed gate descriptions

        """
        profile = metrics.profile
        gates = self.quality_gates.get(profile, self.quality_gates["quick"])

        failures = []

        # Check each gate
        if metrics.groundedness_rate < gates["groundedness_rate"]:
            failures.append(
                f"Groundedness rate {metrics.groundedness_rate:.3f} < {gates['groundedness_rate']}"
            )

        if metrics.citation_precision < gates["citation_precision"]:
            failures.append(
                f"Citation precision {metrics.citation_precision:.3f} < {gates['citation_precision']}"
            )

        recall_10 = metrics.retrieval_recall_at_k.get(10, 0)
        if recall_10 < gates["retrieval_recall_at_10"]:
            failures.append(
                f"Retrieval recall@10 {recall_10:.3f} < {gates['retrieval_recall_at_10']}"
            )

        if metrics.hallucination_rate > gates["hallucination_rate"]:
            failures.append(
                f"Hallucination rate {metrics.hallucination_rate:.3f} > {gates['hallucination_rate']}"
            )

        passed = len(failures) == 0

        if passed:
            self.logger.info(f"✅ All quality gates passed for {profile} profile")
        else:
            self.logger.warning(
                f"❌ Quality gates failed for {profile} profile: {len(failures)} issues"
            )
            for failure in failures:
                self.logger.warning(f"  - {failure}")

        return passed, failures


def create_rag_evaluator(config: dict[str, Any] = None) -> ComprehensiveRAGEvaluator:
    """Factory function for RAG evaluator"""
    return ComprehensiveRAGEvaluator(config)
