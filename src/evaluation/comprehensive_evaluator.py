#!/usr/bin/env python3
"""
Comprehensive evaluation system s metrikami a regression testing
Recall@k, nDCG@k, evidence coverage, citation precision, groundedness

Author: Senior Python/MLOps Agent
"""

import asyncio
import json
import math
import statistics
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
import numpy as np
from pathlib import Path


@dataclass
class EvaluationMetrics:
    """Evaluační metriky"""
    recall_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    evidence_coverage: float
    citation_precision: float
    groundedness_score: float
    hallucination_rate: float
    disagreement_coverage: float
    context_usage_efficiency: float
    latency_p50: float
    latency_p95: float
    total_queries: int
    timestamp: str


@dataclass
class QueryEvaluation:
    """Evaluace jednotlivého dotazu"""
    query_id: str
    query_text: str
    ground_truth_docs: List[str]
    retrieved_docs: List[str]
    relevance_scores: List[float]
    generated_claims: List[str]
    claim_citations: Dict[str, List[str]]
    groundedness_scores: List[float]
    latency_ms: float
    context_tokens_used: int
    context_tokens_budget: int


class ComprehensiveEvaluator:
    """Komprehensivní evaluační systém"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.evaluation_sets = self._load_evaluation_sets()
        self.baseline_metrics = self._load_baseline_metrics()
        self.regression_thresholds = config.get("regression_thresholds", {
            "recall_at_10": -0.05,  # Max 5% pokles
            "ndcg_at_10": -0.02,    # Max 2% pokles
            "citation_precision": -0.03,  # Max 3% pokles
            "groundedness_score": -0.05   # Max 5% pokles
        })

    def _load_evaluation_sets(self) -> Dict[str, List[Dict[str, Any]]]:
        """Načte evaluační sady"""
        # V produkci by se načetly z konfiguračních souborů
        return {
            "climate_science": [
                {
                    "query_id": "climate_001",
                    "query": "effects of climate change on Arctic ice",
                    "ground_truth_docs": ["doc1", "doc2", "doc3"],
                    "expected_claims": ["Arctic ice is declining", "Temperature rise is the cause"],
                    "domain": "climate_science"
                },
                {
                    "query_id": "climate_002",
                    "query": "renewable energy adoption rates",
                    "ground_truth_docs": ["doc4", "doc5"],
                    "expected_claims": ["Solar energy growth is accelerating"],
                    "domain": "climate_science"
                }
            ],
            "medical_research": [
                {
                    "query_id": "med_001",
                    "query": "COVID-19 vaccine effectiveness",
                    "ground_truth_docs": ["med1", "med2", "med3"],
                    "expected_claims": ["Vaccines reduce severe illness"],
                    "domain": "medical_research"
                }
            ],
            "legal_research": [
                {
                    "query_id": "legal_001",
                    "query": "patent law changes 2024",
                    "ground_truth_docs": ["legal1", "legal2"],
                    "expected_claims": ["New patent filing requirements"],
                    "domain": "legal_research"
                }
            ]
        }

    def _load_baseline_metrics(self) -> Dict[str, float]:
        """Načte baseline metriky"""
        baseline_file = Path("artifacts/baseline_metrics.json")

        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                return json.load(f)

        # Default baseline
        return {
            "recall_at_10": 0.75,
            "ndcg_at_10": 0.65,
            "evidence_coverage": 0.80,
            "citation_precision": 0.85,
            "groundedness_score": 0.70,
            "hallucination_rate": 0.15,
            "disagreement_coverage": 0.60,
            "context_usage_efficiency": 0.75
        }

    async def run_comprehensive_evaluation(self,
                                         research_system: Any,
                                         evaluation_domains: List[str] = None) -> EvaluationMetrics:
        """Spustí komprehensivní evaluaci"""
        print("🔬 Starting comprehensive evaluation...")

        if evaluation_domains is None:
            evaluation_domains = list(self.evaluation_sets.keys())

        all_evaluations = []
        start_time = datetime.now()

        # Spusť evaluaci pro každou doménu
        for domain in evaluation_domains:
            if domain not in self.evaluation_sets:
                print(f"⚠️  Domain {domain} not found in evaluation sets")
                continue

            domain_evaluations = await self._evaluate_domain(research_system, domain)
            all_evaluations.extend(domain_evaluations)

        # Vypočítej celkové metriky
        metrics = self._calculate_aggregate_metrics(all_evaluations)

        # Kontrola regression thresholds
        regression_check = self._check_regression_thresholds(metrics)

        print(f"✅ Evaluation completed: {len(all_evaluations)} queries evaluated")

        if not regression_check["passed"]:
            print(f"❌ Regression detected: {regression_check['failed_metrics']}")

        return metrics

    async def _evaluate_domain(self,
                              research_system: Any,
                              domain: str) -> List[QueryEvaluation]:
        """Evaluuje jednu doménu"""
        print(f"📊 Evaluating domain: {domain}")

        domain_queries = self.evaluation_sets[domain]
        evaluations = []

        for query_data in domain_queries:
            try:
                evaluation = await self._evaluate_single_query(research_system, query_data)
                evaluations.append(evaluation)
            except Exception as e:
                print(f"❌ Query evaluation failed for {query_data['query_id']}: {e}")
                continue

        return evaluations

    async def _evaluate_single_query(self,
                                   research_system: Any,
                                   query_data: Dict[str, Any]) -> QueryEvaluation:
        """Evaluuje jednotlivý dotaz"""
        query_start = datetime.now()

        # Mock research system call (v produkci by byl reálný systém)
        results = await self._mock_research_system_call(query_data["query"])

        query_end = datetime.now()
        latency_ms = (query_end - query_start).total_seconds() * 1000

        # Extract výsledky
        retrieved_docs = results.get("retrieved_docs", [])
        generated_claims = results.get("claims", [])
        claim_citations = results.get("claim_citations", {})
        context_tokens_used = results.get("context_tokens_used", 0)
        context_tokens_budget = results.get("context_tokens_budget", 4096)

        # Vypočítej relevance scores
        relevance_scores = self._calculate_relevance_scores(
            retrieved_docs, query_data["ground_truth_docs"]
        )

        # Vypočítej groundedness scores
        groundedness_scores = self._calculate_groundedness_scores(
            generated_claims, claim_citations
        )

        return QueryEvaluation(
            query_id=query_data["query_id"],
            query_text=query_data["query"],
            ground_truth_docs=query_data["ground_truth_docs"],
            retrieved_docs=retrieved_docs,
            relevance_scores=relevance_scores,
            generated_claims=generated_claims,
            claim_citations=claim_citations,
            groundedness_scores=groundedness_scores,
            latency_ms=latency_ms,
            context_tokens_used=context_tokens_used,
            context_tokens_budget=context_tokens_budget
        )

    async def _mock_research_system_call(self, query: str) -> Dict[str, Any]:
        """Mock research system pro testing"""
        # Simuluje research system response
        await asyncio.sleep(0.1)  # Simulate processing time

        return {
            "retrieved_docs": [f"doc_{i}" for i in range(1, 11)],  # Top 10 docs
            "claims": [
                f"Claim 1 about {query[:20]}...",
                f"Claim 2 regarding {query[:15]}..."
            ],
            "claim_citations": {
                "Claim 1": ["doc_1", "doc_2"],
                "Claim 2": ["doc_3", "doc_4", "doc_5"]
            },
            "context_tokens_used": 2048,
            "context_tokens_budget": 4096
        }

    def _calculate_relevance_scores(self,
                                  retrieved_docs: List[str],
                                  ground_truth_docs: List[str]) -> List[float]:
        """Vypočítej relevance scores"""
        relevance_scores = []

        for doc in retrieved_docs:
            if doc in ground_truth_docs:
                relevance_scores.append(1.0)
            else:
                # Partial relevance pro podobné docs
                similarity = self._calculate_doc_similarity(doc, ground_truth_docs)
                relevance_scores.append(similarity)

        return relevance_scores

    def _calculate_doc_similarity(self, doc: str, ground_truth_docs: List[str]) -> float:
        """Vypočítej podobnost dokumentu"""
        # Simplified similarity - v produkci by bylo sofistikovanější
        doc_num = int(doc.split('_')[-1]) if '_' in doc else 0

        for gt_doc in ground_truth_docs:
            gt_num = int(gt_doc.split('_')[-1]) if '_' in gt_doc else 0
            if abs(doc_num - gt_num) <= 2:  # Nearby docs are partially relevant
                return 0.5

        return 0.0

    def _calculate_groundedness_scores(self,
                                     claims: List[str],
                                     claim_citations: Dict[str, List[str]]) -> List[float]:
        """Vypočítej groundedness scores"""
        scores = []

        for claim in claims:
            citations = claim_citations.get(claim, [])

            if len(citations) >= 2:  # Well supported
                scores.append(1.0)
            elif len(citations) == 1:  # Partially supported
                scores.append(0.6)
            else:  # Unsupported
                scores.append(0.0)

        return scores

    def _calculate_aggregate_metrics(self, evaluations: List[QueryEvaluation]) -> EvaluationMetrics:
        """Vypočítej agregované metriky"""
        if not evaluations:
            return EvaluationMetrics(
                recall_at_k={}, ndcg_at_k={}, evidence_coverage=0.0,
                citation_precision=0.0, groundedness_score=0.0,
                hallucination_rate=0.0, disagreement_coverage=0.0,
                context_usage_efficiency=0.0, latency_p50=0.0,
                latency_p95=0.0, total_queries=0,
                timestamp=datetime.now().isoformat()
            )

        # Recall@k a nDCG@k
        k_values = [5, 10, 20]
        recall_at_k = {}
        ndcg_at_k = {}

        for k in k_values:
            recall_scores = []
            ndcg_scores = []

            for eval_result in evaluations:
                recall = self._calculate_recall_at_k(eval_result, k)
                ndcg = self._calculate_ndcg_at_k(eval_result, k)
                recall_scores.append(recall)
                ndcg_scores.append(ndcg)

            recall_at_k[k] = statistics.mean(recall_scores)
            ndcg_at_k[k] = statistics.mean(ndcg_scores)

        # Evidence coverage
        evidence_coverage = self._calculate_evidence_coverage(evaluations)

        # Citation precision
        citation_precision = self._calculate_citation_precision(evaluations)

        # Groundedness score
        all_groundedness_scores = []
        for eval_result in evaluations:
            all_groundedness_scores.extend(eval_result.groundedness_scores)
        groundedness_score = statistics.mean(all_groundedness_scores) if all_groundedness_scores else 0.0

        # Hallucination rate
        hallucination_rate = 1.0 - groundedness_score

        # Context usage efficiency
        efficiency_scores = []
        for eval_result in evaluations:
            if eval_result.context_tokens_budget > 0:
                efficiency = eval_result.context_tokens_used / eval_result.context_tokens_budget
                efficiency_scores.append(efficiency)
        context_usage_efficiency = statistics.mean(efficiency_scores) if efficiency_scores else 0.0

        # Latency metrics
        latencies = [eval_result.latency_ms for eval_result in evaluations]
        latency_p50 = np.percentile(latencies, 50) if latencies else 0.0
        latency_p95 = np.percentile(latencies, 95) if latencies else 0.0

        return EvaluationMetrics(
            recall_at_k=recall_at_k,
            ndcg_at_k=ndcg_at_k,
            evidence_coverage=evidence_coverage,
            citation_precision=citation_precision,
            groundedness_score=groundedness_score,
            hallucination_rate=hallucination_rate,
            disagreement_coverage=0.6,  # Would be calculated from actual conflicts
            context_usage_efficiency=context_usage_efficiency,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            total_queries=len(evaluations),
            timestamp=datetime.now().isoformat()
        )

    def _calculate_recall_at_k(self, evaluation: QueryEvaluation, k: int) -> float:
        """Vypočítej Recall@k"""
        if not evaluation.ground_truth_docs:
            return 0.0

        retrieved_at_k = set(evaluation.retrieved_docs[:k])
        ground_truth_set = set(evaluation.ground_truth_docs)

        relevant_retrieved = len(retrieved_at_k.intersection(ground_truth_set))
        total_relevant = len(ground_truth_set)

        return relevant_retrieved / total_relevant if total_relevant > 0 else 0.0

    def _calculate_ndcg_at_k(self, evaluation: QueryEvaluation, k: int) -> float:
        """Vypočítej nDCG@k"""
        if k > len(evaluation.relevance_scores):
            k = len(evaluation.relevance_scores)

        if k == 0:
            return 0.0

        # DCG@k
        dcg = 0.0
        for i in range(k):
            if i < len(evaluation.relevance_scores):
                rel = evaluation.relevance_scores[i]
                dcg += rel / math.log2(i + 2)  # +2 because log2(1) = 0

        # IDCG@k (ideal DCG)
        sorted_relevance = sorted(evaluation.relevance_scores, reverse=True)
        idcg = 0.0
        for i in range(min(k, len(sorted_relevance))):
            rel = sorted_relevance[i]
            idcg += rel / math.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    def _calculate_evidence_coverage(self, evaluations: List[QueryEvaluation]) -> float:
        """Vypočítej evidence coverage"""
        total_claims = 0
        covered_claims = 0

        for evaluation in evaluations:
            for claim in evaluation.generated_claims:
                total_claims += 1
                citations = evaluation.claim_citations.get(claim, [])
                if len(citations) >= 2:  # Minimum 2 citations required
                    covered_claims += 1

        return covered_claims / total_claims if total_claims > 0 else 0.0

    def _calculate_citation_precision(self, evaluations: List[QueryEvaluation]) -> float:
        """Vypočítej citation precision"""
        total_citations = 0
        valid_citations = 0

        for evaluation in evaluations:
            for claim, citations in evaluation.claim_citations.items():
                for citation in citations:
                    total_citations += 1
                    # Citation is valid if it's in retrieved docs
                    if citation in evaluation.retrieved_docs:
                        valid_citations += 1

        return valid_citations / total_citations if total_citations > 0 else 0.0

    def _check_regression_thresholds(self, metrics: EvaluationMetrics) -> Dict[str, Any]:
        """Kontrola regression thresholds"""
        regression_check = {
            "passed": True,
            "failed_metrics": [],
            "threshold_violations": {}
        }

        # Kontrola klíčových metrik
        metric_checks = [
            ("recall_at_10", metrics.recall_at_k.get(10, 0.0)),
            ("ndcg_at_10", metrics.ndcg_at_k.get(10, 0.0)),
            ("citation_precision", metrics.citation_precision),
            ("groundedness_score", metrics.groundedness_score)
        ]

        for metric_name, current_value in metric_checks:
            if metric_name in self.baseline_metrics and metric_name in self.regression_thresholds:
                baseline_value = self.baseline_metrics[metric_name]
                threshold = self.regression_thresholds[metric_name]

                # Pokles větší než threshold
                if current_value < (baseline_value + threshold):
                    regression_check["passed"] = False
                    regression_check["failed_metrics"].append(metric_name)
                    regression_check["threshold_violations"][metric_name] = {
                        "baseline": baseline_value,
                        "current": current_value,
                        "threshold": threshold,
                        "violation": (baseline_value + threshold) - current_value
                    }

        return regression_check

    def save_evaluation_results(self,
                               metrics: EvaluationMetrics,
                               output_dir: str) -> Dict[str, str]:
        """Ulož evaluační výsledky"""
        artifacts = {}

        # Detailed metrics
        metrics_file = f"{output_dir}/evaluation_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(asdict(metrics), f, indent=2)
        artifacts["metrics"] = metrics_file

        # Regression check
        regression_check = self._check_regression_thresholds(metrics)
        regression_file = f"{output_dir}/regression_check.json"
        with open(regression_file, "w") as f:
            json.dump(regression_check, f, indent=2)
        artifacts["regression_check"] = regression_file

        # Performance summary
        summary = {
            "timestamp": metrics.timestamp,
            "total_queries": metrics.total_queries,
            "key_metrics": {
                "recall@10": metrics.recall_at_k.get(10, 0.0),
                "nDCG@10": metrics.ndcg_at_k.get(10, 0.0),
                "citation_precision": metrics.citation_precision,
                "groundedness": metrics.groundedness_score,
                "latency_p95": metrics.latency_p95
            },
            "regression_passed": regression_check["passed"]
        }

        summary_file = f"{output_dir}/evaluation_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        artifacts["summary"] = summary_file

        return artifacts

    def update_baseline_metrics(self, metrics: EvaluationMetrics) -> None:
        """Aktualizuj baseline metriky"""
        new_baseline = {
            "recall_at_10": metrics.recall_at_k.get(10, 0.0),
            "ndcg_at_10": metrics.ndcg_at_k.get(10, 0.0),
            "evidence_coverage": metrics.evidence_coverage,
            "citation_precision": metrics.citation_precision,
            "groundedness_score": metrics.groundedness_score,
            "hallucination_rate": metrics.hallucination_rate,
            "disagreement_coverage": metrics.disagreement_coverage,
            "context_usage_efficiency": metrics.context_usage_efficiency,
            "timestamp": metrics.timestamp
        }

        baseline_file = Path("artifacts/baseline_metrics.json")
        baseline_file.parent.mkdir(exist_ok=True)

        with open(baseline_file, "w") as f:
            json.dump(new_baseline, f, indent=2)

        print(f"📊 Baseline metrics updated: {baseline_file}")
