#!/usr/bin/env python3
"""
FÁZE 5: Enhanced Metrics Framework
Rozšířený systém metrik pro evaluaci evidence-based research

Author: Senior Python/MLOps Agent
"""

import json
import time
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import asyncio
from pathlib import Path


@dataclass
class EvidenceItem:
    """Jednotka evidence s kompletními metadaty"""

    doc_id: str
    content: str
    char_offset_start: int
    char_offset_end: int
    source_type: str  # primary_literature, secondary_source, aggregator
    authority_score: float
    recency_score: float
    relevance_score: float
    citation_context: str
    quality_indicators: Dict[str, float] = field(default_factory=dict)


@dataclass
class ClaimWithEvidence:
    """Tvrzení s vázanou evidencí"""

    claim_text: str
    claim_id: str
    evidence_items: List[EvidenceItem]
    confidence_score: float
    supporting_citations: List[str]
    contradicting_citations: List[str] = field(default_factory=list)
    consensus_strength: float = 0.0


@dataclass
class EvaluationResult:
    """Výsledek evaluace s kompletními metrikami"""

    query: str
    timestamp: datetime

    # Retrieval metrics
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)

    # Evidence quality metrics
    evidence_coverage: float = 0.0
    citation_precision: float = 0.0
    groundedness: float = 0.0
    hallucination_rate: float = 0.0
    disagreement_coverage: float = 0.0
    context_usage_efficiency: float = 0.0

    # Performance metrics
    latency_ms: float = 0.0
    token_budget_used: int = 0
    token_budget_total: int = 0

    # Quality breakdown
    primary_source_ratio: float = 0.0
    citation_diversity: float = 0.0
    temporal_currency: float = 0.0

    # Detailed breakdown
    claims_with_evidence: List[ClaimWithEvidence] = field(default_factory=list)
    processing_log: List[Dict[str, Any]] = field(default_factory=list)


class EnhancedMetricsCalculator:
    """Pokročilý kalkulátor metrik pro FÁZE 5"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.k_values = config.get("evaluation", {}).get("k_values", [1, 3, 5, 10, 20])
        self.min_evidence_per_claim = config.get("evaluation", {}).get("min_evidence_per_claim", 2)
        self.hallucination_threshold = config.get("evaluation", {}).get(
            "hallucination_threshold", 0.1
        )

    def calculate_recall_at_k(
        self, retrieved_docs: List[str], relevant_docs: List[str], k: int
    ) -> float:
        """Vypočítá recall@k metriku"""
        if not relevant_docs:
            return 0.0

        retrieved_k = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)

        intersection = retrieved_k.intersection(relevant_set)
        return len(intersection) / len(relevant_set)

    def calculate_ndcg_at_k(
        self, retrieved_docs: List[str], relevance_scores: Dict[str, float], k: int
    ) -> float:
        """Vypočítá nDCG@k metriku"""
        if not retrieved_docs or not relevance_scores:
            return 0.0

        # DCG calculation
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_docs[:k]):
            if doc_id in relevance_scores:
                rel = relevance_scores[doc_id]
                dcg += (2**rel - 1) / math.log2(i + 2)

        # IDCG calculation (ideal ranking)
        sorted_scores = sorted(relevance_scores.values(), reverse=True)
        idcg = 0.0
        for i, rel in enumerate(sorted_scores[:k]):
            idcg += (2**rel - 1) / math.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    def calculate_evidence_coverage(self, claims: List[ClaimWithEvidence]) -> float:
        """Vypočítá evidence coverage - kolik claims má dostatečnou evidenci"""
        if not claims:
            return 0.0

        well_supported_claims = 0
        for claim in claims:
            independent_sources = len(set(evidence.doc_id for evidence in claim.evidence_items))
            if independent_sources >= self.min_evidence_per_claim:
                well_supported_claims += 1

        return well_supported_claims / len(claims)

    def calculate_citation_precision(self, claims: List[ClaimWithEvidence]) -> float:
        """Vypočítá precision citací - kolik citací je skutečně relevantních"""
        total_citations = 0
        accurate_citations = 0

        for claim in claims:
            for evidence in claim.evidence_items:
                total_citations += 1
                # Check if citation context supports the claim
                if self._citation_supports_claim(claim.claim_text, evidence.citation_context):
                    accurate_citations += 1

        return accurate_citations / total_citations if total_citations > 0 else 0.0

    def calculate_groundedness(self, claims: List[ClaimWithEvidence]) -> float:
        """Vypočítá groundedness - míru zakotvenosti claims v evidenci"""
        if not claims:
            return 0.0

        total_groundedness = 0.0
        for claim in claims:
            claim_groundedness = 0.0
            for evidence in claim.evidence_items:
                # Combine relevance and authority scores
                evidence_strength = evidence.relevance_score * 0.6 + evidence.authority_score * 0.4
                claim_groundedness += evidence_strength

            # Normalize by number of evidence items
            if claim.evidence_items:
                claim_groundedness /= len(claim.evidence_items)

            total_groundedness += claim_groundedness

        return total_groundedness / len(claims)

    def calculate_hallucination_rate(self, claims: List[ClaimWithEvidence]) -> float:
        """Vypočítá hallucination rate - míru nepodložených tvrzení"""
        if not claims:
            return 0.0

        hallucinated_claims = 0
        for claim in claims:
            # Claim is considered hallucinated if it has very weak evidence support
            evidence_strength = sum(evidence.relevance_score for evidence in claim.evidence_items)
            avg_evidence_strength = (
                evidence_strength / len(claim.evidence_items) if claim.evidence_items else 0
            )

            if avg_evidence_strength < self.hallucination_threshold:
                hallucinated_claims += 1

        return hallucinated_claims / len(claims)

    def calculate_disagreement_coverage(self, claims: List[ClaimWithEvidence]) -> float:
        """Vypočítá disagreement coverage - kolik kvalitních protinázorů bylo nalezeno"""
        if not claims:
            return 0.0

        claims_with_counterevidence = 0
        for claim in claims:
            if claim.contradicting_citations:
                # Check quality of contradicting evidence
                high_quality_contradictions = sum(
                    1
                    for citation in claim.contradicting_citations
                    if self._is_high_quality_contradiction(citation)
                )
                if high_quality_contradictions > 0:
                    claims_with_counterevidence += 1

        return claims_with_counterevidence / len(claims)

    def calculate_context_usage_efficiency(
        self, claims: List[ClaimWithEvidence], token_budget_used: int, token_budget_total: int
    ) -> float:
        """Vypočítá context usage efficiency"""
        if token_budget_total == 0:
            return 0.0

        # Calculate information density
        total_evidence_quality = sum(
            sum(evidence.relevance_score for evidence in claim.evidence_items) for claim in claims
        )

        budget_utilization = token_budget_used / token_budget_total

        # Efficiency = information gained per token used
        if token_budget_used > 0:
            return total_evidence_quality / (token_budget_used / 1000)  # per 1k tokens
        return 0.0

    def calculate_primary_source_ratio(self, claims: List[ClaimWithEvidence]) -> float:
        """Vypočítá poměr primárních zdrojů"""
        total_evidence = 0
        primary_evidence = 0

        for claim in claims:
            for evidence in claim.evidence_items:
                total_evidence += 1
                if evidence.source_type == "primary_literature":
                    primary_evidence += 1

        return primary_evidence / total_evidence if total_evidence > 0 else 0.0

    def calculate_citation_diversity(self, claims: List[ClaimWithEvidence]) -> float:
        """Vypočítá diverzitu citací"""
        all_doc_ids = set()
        for claim in claims:
            for evidence in claim.evidence_items:
                all_doc_ids.add(evidence.doc_id)

        total_citations = sum(len(claim.evidence_items) for claim in claims)
        unique_sources = len(all_doc_ids)

        return unique_sources / total_citations if total_citations > 0 else 0.0

    def calculate_temporal_currency(self, claims: List[ClaimWithEvidence]) -> float:
        """Vypočítá temporal currency - aktuálnost zdrojů"""
        total_recency = 0.0
        total_evidence = 0

        for claim in claims:
            for evidence in claim.evidence_items:
                total_recency += evidence.recency_score
                total_evidence += 1

        return total_recency / total_evidence if total_evidence > 0 else 0.0

    def _citation_supports_claim(self, claim_text: str, citation_context: str) -> bool:
        """Jednoduchá heuristika pro ověření podpory citace"""
        # Implementace by měla být pokročilejší - semantic similarity, NLI, etc.
        claim_words = set(claim_text.lower().split())
        context_words = set(citation_context.lower().split())

        overlap = len(claim_words.intersection(context_words))
        return overlap / len(claim_words) > 0.3 if claim_words else False

    def _is_high_quality_contradiction(self, contradiction_citation: str) -> bool:
        """Ověří kvalitu protiargumentu"""
        # Placeholder - implementace by měla ověřovat autoritu zdroje,
        # explicitnost rozporu, metodologickou kvalitu
        return len(contradiction_citation) > 50  # Základní heuristika


class ComprehensiveEvaluator:
    """Komprehenzivní evaluátor pro FÁZE 5"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_calculator = EnhancedMetricsCalculator(config)
        self.k_values = config.get("evaluation", {}).get("k_values", [1, 3, 5, 10, 20])

    async def evaluate_research_session(
        self,
        query: str,
        retrieved_documents: List[Dict[str, Any]],
        generated_claims: List[ClaimWithEvidence],
        ground_truth: Optional[Dict[str, Any]] = None,
        processing_metrics: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """Provede komprehenzivní evaluaci research session"""

        start_time = time.time()

        result = EvaluationResult(
            query=query, timestamp=datetime.now(), claims_with_evidence=generated_claims
        )

        # Calculate retrieval metrics if ground truth available
        if ground_truth:
            retrieved_doc_ids = [doc.get("doc_id", "") for doc in retrieved_documents]
            relevant_doc_ids = ground_truth.get("relevant_documents", [])
            relevance_scores = ground_truth.get("relevance_scores", {})

            for k in self.k_values:
                result.recall_at_k[k] = self.metrics_calculator.calculate_recall_at_k(
                    retrieved_doc_ids, relevant_doc_ids, k
                )
                result.ndcg_at_k[k] = self.metrics_calculator.calculate_ndcg_at_k(
                    retrieved_doc_ids, relevance_scores, k
                )

        # Calculate evidence quality metrics
        result.evidence_coverage = self.metrics_calculator.calculate_evidence_coverage(
            generated_claims
        )
        result.citation_precision = self.metrics_calculator.calculate_citation_precision(
            generated_claims
        )
        result.groundedness = self.metrics_calculator.calculate_groundedness(generated_claims)
        result.hallucination_rate = self.metrics_calculator.calculate_hallucination_rate(
            generated_claims
        )
        result.disagreement_coverage = self.metrics_calculator.calculate_disagreement_coverage(
            generated_claims
        )

        # Calculate efficiency metrics
        if processing_metrics:
            token_used = processing_metrics.get("token_budget_used", 0)
            token_total = processing_metrics.get("token_budget_total", 0)
            result.token_budget_used = token_used
            result.token_budget_total = token_total
            result.context_usage_efficiency = (
                self.metrics_calculator.calculate_context_usage_efficiency(
                    generated_claims, token_used, token_total
                )
            )
            result.latency_ms = processing_metrics.get("total_latency_ms", 0)

        # Calculate quality breakdown metrics
        result.primary_source_ratio = self.metrics_calculator.calculate_primary_source_ratio(
            generated_claims
        )
        result.citation_diversity = self.metrics_calculator.calculate_citation_diversity(
            generated_claims
        )
        result.temporal_currency = self.metrics_calculator.calculate_temporal_currency(
            generated_claims
        )

        # Log processing details
        result.processing_log.append(
            {
                "step": "comprehensive_evaluation",
                "duration_ms": (time.time() - start_time) * 1000,
                "metrics_calculated": len(
                    [attr for attr in dir(result) if not attr.startswith("_")]
                ),
                "timestamp": datetime.now().isoformat(),
            }
        )

        return result

    def generate_evaluation_report(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Generuje souhrnný evaluační report"""
        if not results:
            return {"error": "No evaluation results provided"}

        report = {
            "evaluation_summary": {
                "total_queries_evaluated": len(results),
                "evaluation_period": {
                    "start": min(r.timestamp for r in results).isoformat(),
                    "end": max(r.timestamp for r in results).isoformat(),
                },
            },
            "aggregate_metrics": {},
            "performance_analysis": {},
            "quality_trends": {},
            "recommendations": [],
        }

        # Aggregate metrics across all results
        metrics_to_aggregate = [
            "evidence_coverage",
            "citation_precision",
            "groundedness",
            "hallucination_rate",
            "disagreement_coverage",
            "context_usage_efficiency",
            "primary_source_ratio",
            "citation_diversity",
            "temporal_currency",
        ]

        for metric in metrics_to_aggregate:
            values = [getattr(r, metric) for r in results if hasattr(r, metric)]
            if values:
                report["aggregate_metrics"][metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values),
                }

        # Aggregate recall@k and nDCG@k
        for k in self.k_values:
            recall_values = [r.recall_at_k.get(k, 0) for r in results if r.recall_at_k]
            ndcg_values = [r.ndcg_at_k.get(k, 0) for r in results if r.ndcg_at_k]

            if recall_values:
                report["aggregate_metrics"][f"recall_at_{k}"] = {
                    "mean": np.mean(recall_values),
                    "std": np.std(recall_values),
                }

            if ndcg_values:
                report["aggregate_metrics"][f"ndcg_at_{k}"] = {
                    "mean": np.mean(ndcg_values),
                    "std": np.std(ndcg_values),
                }

        # Performance analysis
        latencies = [r.latency_ms for r in results if r.latency_ms > 0]
        if latencies:
            report["performance_analysis"]["latency_ms"] = {
                "mean": np.mean(latencies),
                "p50": np.percentile(latencies, 50),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99),
            }

        # Generate recommendations
        recommendations = self._generate_recommendations(report["aggregate_metrics"])
        report["recommendations"] = recommendations

        return report

    def _generate_recommendations(self, aggregate_metrics: Dict[str, Any]) -> List[str]:
        """Generuje doporučení na základě metrik"""
        recommendations = []

        # Check evidence coverage
        if "evidence_coverage" in aggregate_metrics:
            coverage = aggregate_metrics["evidence_coverage"]["mean"]
            if coverage < 0.8:
                recommendations.append(
                    f"Evidence coverage je nízká ({coverage:.2f}). "
                    f"Zvyšte požadavky na min_evidence_per_claim nebo zlepšete retrieval."
                )

        # Check hallucination rate
        if "hallucination_rate" in aggregate_metrics:
            hallucination = aggregate_metrics["hallucination_rate"]["mean"]
            if hallucination > 0.1:
                recommendations.append(
                    f"Hallucination rate je vysoká ({hallucination:.2f}). "
                    f"Posílete verification gates a relevance filtering."
                )

        # Check primary source ratio
        if "primary_source_ratio" in aggregate_metrics:
            primary_ratio = aggregate_metrics["primary_source_ratio"]["mean"]
            if primary_ratio < 0.6:
                recommendations.append(
                    f"Poměr primárních zdrojů je nízký ({primary_ratio:.2f}). "
                    f"Prioritizujte academic papers a original research."
                )

        # Check context efficiency
        if "context_usage_efficiency" in aggregate_metrics:
            efficiency = aggregate_metrics["context_usage_efficiency"]["mean"]
            if efficiency < 2.0:  # Arbitrary threshold
                recommendations.append(
                    f"Context usage efficiency je nízká ({efficiency:.2f}). "
                    f"Optimalizujte contextual compression a token budgeting."
                )

        return recommendations

    async def export_detailed_results(
        self, results: List[EvaluationResult], output_path: str
    ) -> None:
        """Exportuje detailní výsledky do JSON"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert results to serializable format
        serializable_results = []
        for result in results:
            result_dict = {
                "query": result.query,
                "timestamp": result.timestamp.isoformat(),
                "recall_at_k": result.recall_at_k,
                "ndcg_at_k": result.ndcg_at_k,
                "evidence_coverage": result.evidence_coverage,
                "citation_precision": result.citation_precision,
                "groundedness": result.groundedness,
                "hallucination_rate": result.hallucination_rate,
                "disagreement_coverage": result.disagreement_coverage,
                "context_usage_efficiency": result.context_usage_efficiency,
                "latency_ms": result.latency_ms,
                "token_budget_used": result.token_budget_used,
                "token_budget_total": result.token_budget_total,
                "primary_source_ratio": result.primary_source_ratio,
                "citation_diversity": result.citation_diversity,
                "temporal_currency": result.temporal_currency,
                "claims_count": len(result.claims_with_evidence),
                "processing_log": result.processing_log,
            }
            serializable_results.append(result_dict)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "evaluation_results": serializable_results,
                    "summary_report": self.generate_evaluation_report(results),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )


# Factory function for creating evaluator
def create_comprehensive_evaluator(config: Dict[str, Any]) -> ComprehensiveEvaluator:
    """Factory function pro vytvoření comprehensive evaluator"""
    return ComprehensiveEvaluator(config)


if __name__ == "__main__":
    # Test the enhanced metrics framework
    config = {
        "evaluation": {
            "k_values": [1, 3, 5, 10],
            "min_evidence_per_claim": 2,
            "hallucination_threshold": 0.1,
        }
    }

    evaluator = create_comprehensive_evaluator(config)
    print("✅ Enhanced Metrics Framework initialized successfully!")
