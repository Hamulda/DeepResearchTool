#!/usr/bin/env python3
"""
FÁZE 5: Regression Test Suite
Regression testy napříč 10+ doménami s tracking latency/recall trade-offs

Author: Senior Python/MLOps Agent
"""

import json
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import numpy as np

from src.evaluation.enhanced_metrics import (
    ComprehensiveEvaluator,
    EvaluationResult,
    ClaimWithEvidence,
    EvidenceItem
)


@dataclass
class DomainTestCase:
    """Test case pro specifickou doménu"""
    domain: str
    query: str
    expected_claim_count: int
    expected_evidence_per_claim: int
    quality_thresholds: Dict[str, float]
    ground_truth_docs: List[str] = field(default_factory=list)
    relevance_scores: Dict[str, float] = field(default_factory=dict)
    complexity_level: str = "medium"  # easy, medium, hard


@dataclass
class RegressionTestResult:
    """Výsledek regression testu"""
    domain: str
    test_case: DomainTestCase
    evaluation_result: EvaluationResult
    baseline_comparison: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    passed_thresholds: Dict[str, bool] = field(default_factory=dict)
    overall_status: str = "unknown"  # passed, failed, degraded


class RegressionTestSuite:
    """Regression test suite pro FÁZE 5"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.evaluator = ComprehensiveEvaluator(config)
        self.baseline_path = Path(config.get('regression', {}).get('baseline_path', 'data/baselines'))
        self.results_path = Path(config.get('regression', {}).get('results_path', 'docs/regression_results'))
        self.domain_test_cases = self._create_domain_test_cases()

    def _create_domain_test_cases(self) -> List[DomainTestCase]:
        """Vytvoří test cases pro různé domény"""
        test_cases = [
            # 1. Climate Science
            DomainTestCase(
                domain="climate_science",
                query="What is the current scientific consensus on anthropogenic climate change?",
                expected_claim_count=3,
                expected_evidence_per_claim=3,
                quality_thresholds={
                    "evidence_coverage": 0.85,
                    "citation_precision": 0.80,
                    "groundedness": 0.85,
                    "hallucination_rate": 0.05,
                    "primary_source_ratio": 0.70
                },
                complexity_level="medium"
            ),

            # 2. Medical Research
            DomainTestCase(
                domain="medical_research",
                query="What are the efficacy rates of mRNA COVID-19 vaccines?",
                expected_claim_count=4,
                expected_evidence_per_claim=4,
                quality_thresholds={
                    "evidence_coverage": 0.90,
                    "citation_precision": 0.85,
                    "groundedness": 0.90,
                    "hallucination_rate": 0.03,
                    "primary_source_ratio": 0.80
                },
                complexity_level="hard"
            ),

            # 3. Technology & AI
            DomainTestCase(
                domain="ai_technology",
                query="What are the current capabilities and limitations of large language models?",
                expected_claim_count=5,
                expected_evidence_per_claim=3,
                quality_thresholds={
                    "evidence_coverage": 0.75,
                    "citation_precision": 0.75,
                    "groundedness": 0.80,
                    "hallucination_rate": 0.08,
                    "primary_source_ratio": 0.60
                },
                complexity_level="medium"
            ),

            # 4. Economics & Finance
            DomainTestCase(
                domain="economics",
                query="What are the economic impacts of quantitative easing on inflation?",
                expected_claim_count=4,
                expected_evidence_per_claim=3,
                quality_thresholds={
                    "evidence_coverage": 0.80,
                    "citation_precision": 0.78,
                    "groundedness": 0.82,
                    "hallucination_rate": 0.06,
                    "primary_source_ratio": 0.65
                },
                complexity_level="hard"
            ),

            # 5. Legal & Policy
            DomainTestCase(
                domain="legal_policy",
                query="What are the key provisions of GDPR regarding data processing consent?",
                expected_claim_count=4,
                expected_evidence_per_claim=2,
                quality_thresholds={
                    "evidence_coverage": 0.85,
                    "citation_precision": 0.90,
                    "groundedness": 0.88,
                    "hallucination_rate": 0.02,
                    "primary_source_ratio": 0.75
                },
                complexity_level="medium"
            ),

            # 6. Historical Research
            DomainTestCase(
                domain="history",
                query="What were the primary causes of the fall of the Roman Empire?",
                expected_claim_count=4,
                expected_evidence_per_claim=3,
                quality_thresholds={
                    "evidence_coverage": 0.75,
                    "citation_precision": 0.70,
                    "groundedness": 0.75,
                    "hallucination_rate": 0.10,
                    "primary_source_ratio": 0.50
                },
                complexity_level="hard"
            ),

            # 7. Psychology & Neuroscience
            DomainTestCase(
                domain="psychology",
                query="What is the current understanding of the neural basis of consciousness?",
                expected_claim_count=3,
                expected_evidence_per_claim=3,
                quality_thresholds={
                    "evidence_coverage": 0.70,
                    "citation_precision": 0.75,
                    "groundedness": 0.78,
                    "hallucination_rate": 0.08,
                    "primary_source_ratio": 0.70
                },
                complexity_level="hard"
            ),

            # 8. Environmental Science
            DomainTestCase(
                domain="environmental_science",
                query="What are the main drivers of biodiversity loss in tropical rainforests?",
                expected_claim_count=4,
                expected_evidence_per_claim=3,
                quality_thresholds={
                    "evidence_coverage": 0.80,
                    "citation_precision": 0.78,
                    "groundedness": 0.82,
                    "hallucination_rate": 0.06,
                    "primary_source_ratio": 0.68
                },
                complexity_level="medium"
            ),

            # 9. Physics & Astronomy
            DomainTestCase(
                domain="physics",
                query="What is the current evidence for dark matter and dark energy?",
                expected_claim_count=3,
                expected_evidence_per_claim=4,
                quality_thresholds={
                    "evidence_coverage": 0.85,
                    "citation_precision": 0.82,
                    "groundedness": 0.85,
                    "hallucination_rate": 0.04,
                    "primary_source_ratio": 0.75
                },
                complexity_level="hard"
            ),

            # 10. Social Science
            DomainTestCase(
                domain="social_science",
                query="What are the effects of social media on mental health in adolescents?",
                expected_claim_count=4,
                expected_evidence_per_claim=3,
                quality_thresholds={
                    "evidence_coverage": 0.78,
                    "citation_precision": 0.76,
                    "groundedness": 0.80,
                    "hallucination_rate": 0.07,
                    "primary_source_ratio": 0.65
                },
                complexity_level="medium"
            ),

            # 11. Computer Science
            DomainTestCase(
                domain="computer_science",
                query="What are the key advances in quantum computing error correction?",
                expected_claim_count=3,
                expected_evidence_per_claim=3,
                quality_thresholds={
                    "evidence_coverage": 0.80,
                    "citation_precision": 0.80,
                    "groundedness": 0.83,
                    "hallucination_rate": 0.05,
                    "primary_source_ratio": 0.72
                },
                complexity_level="hard"
            ),

            # 12. Materials Science
            DomainTestCase(
                domain="materials_science",
                query="What are the recent breakthroughs in perovskite solar cell efficiency?",
                expected_claim_count=3,
                expected_evidence_per_claim=3,
                quality_thresholds={
                    "evidence_coverage": 0.82,
                    "citation_precision": 0.78,
                    "groundedness": 0.80,
                    "hallucination_rate": 0.06,
                    "primary_source_ratio": 0.70
                },
                complexity_level="medium"
            )
        ]

        return test_cases

    async def run_regression_tests(self,
                                 test_subset: Optional[List[str]] = None) -> List[RegressionTestResult]:
        """Spustí regression testy pro všechny nebo vybrané domény"""

        results = []
        test_cases_to_run = self.domain_test_cases

        if test_subset:
            test_cases_to_run = [tc for tc in self.domain_test_cases if tc.domain in test_subset]

        print(f"🚀 Spouštím regression testy pro {len(test_cases_to_run)} domén...")

        for test_case in test_cases_to_run:
            print(f"📊 Testování domény: {test_case.domain}")

            try:
                result = await self._run_single_domain_test(test_case)
                results.append(result)

                status_emoji = "✅" if result.overall_status == "passed" else "❌" if result.overall_status == "failed" else "⚠️"
                print(f"{status_emoji} {test_case.domain}: {result.overall_status}")

            except Exception as e:
                print(f"💥 Chyba při testování {test_case.domain}: {str(e)}")
                # Create failed result
                failed_result = RegressionTestResult(
                    domain=test_case.domain,
                    test_case=test_case,
                    evaluation_result=EvaluationResult(query=test_case.query, timestamp=datetime.now()),
                    overall_status="failed"
                )
                results.append(failed_result)

        return results

    async def _run_single_domain_test(self, test_case: DomainTestCase) -> RegressionTestResult:
        """Spustí test pro jednu doménu"""
        start_time = time.time()

        # Mock research execution - v produkci by se volal skutečný research agent
        mock_claims, mock_documents = await self._mock_research_execution(test_case)

        processing_metrics = {
            "total_latency_ms": (time.time() - start_time) * 1000,
            "token_budget_used": np.random.randint(2000, 6000),  # Mock
            "token_budget_total": 8000
        }

        # Evaluate the results
        evaluation_result = await self.evaluator.evaluate_research_session(
            query=test_case.query,
            retrieved_documents=mock_documents,
            generated_claims=mock_claims,
            ground_truth={
                "relevant_documents": test_case.ground_truth_docs,
                "relevance_scores": test_case.relevance_scores
            },
            processing_metrics=processing_metrics
        )

        # Compare against baseline
        baseline_comparison = await self._compare_against_baseline(test_case.domain, evaluation_result)

        # Check threshold compliance
        passed_thresholds = self._check_threshold_compliance(test_case, evaluation_result)

        # Determine overall status
        overall_status = self._determine_test_status(passed_thresholds, baseline_comparison)

        return RegressionTestResult(
            domain=test_case.domain,
            test_case=test_case,
            evaluation_result=evaluation_result,
            baseline_comparison=baseline_comparison,
            performance_metrics={
                "latency_ms": processing_metrics["total_latency_ms"],
                "tokens_per_second": processing_metrics["token_budget_used"] / (processing_metrics["total_latency_ms"] / 1000),
                "efficiency_score": evaluation_result.context_usage_efficiency
            },
            passed_thresholds=passed_thresholds,
            overall_status=overall_status
        )

    async def _mock_research_execution(self, test_case: DomainTestCase) -> Tuple[List[ClaimWithEvidence], List[Dict[str, Any]]]:
        """Mock research execution pro testování"""
        # Simulate realistic but deterministic results based on domain
        np.random.seed(hash(test_case.domain) % 2**32)

        claims = []
        documents = []

        for i in range(test_case.expected_claim_count):
            evidence_items = []
            for j in range(test_case.expected_evidence_per_claim):
                evidence = EvidenceItem(
                    doc_id=f"{test_case.domain}_doc_{i}_{j}",
                    content=f"Mock evidence content for {test_case.domain}",
                    char_offset_start=0,
                    char_offset_end=100,
                    source_type="primary_literature" if np.random.random() > 0.3 else "secondary_source",
                    authority_score=np.random.uniform(0.6, 0.95),
                    recency_score=np.random.uniform(0.5, 0.9),
                    relevance_score=np.random.uniform(0.7, 0.95),
                    citation_context=f"Context for claim {i} in {test_case.domain}"
                )
                evidence_items.append(evidence)

                # Add corresponding document
                documents.append({
                    "doc_id": evidence.doc_id,
                    "title": f"Research Paper {i}_{j}",
                    "content": evidence.content,
                    "source_type": evidence.source_type
                })

            claim = ClaimWithEvidence(
                claim_text=f"Claim {i} about {test_case.domain}",
                claim_id=f"{test_case.domain}_claim_{i}",
                evidence_items=evidence_items,
                confidence_score=np.random.uniform(0.7, 0.9),
                supporting_citations=[e.doc_id for e in evidence_items]
            )
            claims.append(claim)

        return claims, documents

    async def _compare_against_baseline(self, domain: str,
                                      evaluation_result: EvaluationResult) -> Dict[str, float]:
        """Porovná výsledky s baseline"""
        baseline_file = self.baseline_path / f"{domain}_baseline.json"

        if not baseline_file.exists():
            # Create baseline if it doesn't exist
            await self._create_baseline(domain, evaluation_result)
            return {}

        try:
            with open(baseline_file, 'r') as f:
                baseline_data = json.load(f)

            comparison = {}
            metrics_to_compare = [
                'evidence_coverage', 'citation_precision', 'groundedness',
                'hallucination_rate', 'disagreement_coverage', 'context_usage_efficiency'
            ]

            for metric in metrics_to_compare:
                current_value = getattr(evaluation_result, metric, 0)
                baseline_value = baseline_data.get(metric, 0)

                if baseline_value > 0:
                    percentage_change = ((current_value - baseline_value) / baseline_value) * 100
                    comparison[f"{metric}_change_percent"] = percentage_change
                else:
                    comparison[f"{metric}_change_percent"] = 0

            return comparison

        except Exception as e:
            print(f"⚠️ Chyba při čtení baseline pro {domain}: {str(e)}")
            return {}

    async def _create_baseline(self, domain: str, evaluation_result: EvaluationResult):
        """Vytvoří baseline pro doménu"""
        self.baseline_path.mkdir(parents=True, exist_ok=True)

        baseline_data = {
            "domain": domain,
            "created_at": datetime.now().isoformat(),
            "evidence_coverage": evaluation_result.evidence_coverage,
            "citation_precision": evaluation_result.citation_precision,
            "groundedness": evaluation_result.groundedness,
            "hallucination_rate": evaluation_result.hallucination_rate,
            "disagreement_coverage": evaluation_result.disagreement_coverage,
            "context_usage_efficiency": evaluation_result.context_usage_efficiency,
            "primary_source_ratio": evaluation_result.primary_source_ratio,
            "citation_diversity": evaluation_result.citation_diversity,
            "temporal_currency": evaluation_result.temporal_currency
        }

        baseline_file = self.baseline_path / f"{domain}_baseline.json"
        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)

        print(f"📝 Vytvořen baseline pro {domain}")

    def _check_threshold_compliance(self, test_case: DomainTestCase,
                                  evaluation_result: EvaluationResult) -> Dict[str, bool]:
        """Zkontroluje splnění threshold pro daný test case"""
        passed = {}

        for metric, threshold in test_case.quality_thresholds.items():
            current_value = getattr(evaluation_result, metric, 0)

            if metric == "hallucination_rate":
                # Lower is better for hallucination rate
                passed[metric] = current_value <= threshold
            else:
                # Higher is better for other metrics
                passed[metric] = current_value >= threshold

        return passed

    def _determine_test_status(self, passed_thresholds: Dict[str, bool],
                             baseline_comparison: Dict[str, float]) -> str:
        """Určí celkový status testu"""

        # Check if all thresholds passed
        threshold_pass_rate = sum(passed_thresholds.values()) / len(passed_thresholds)

        if threshold_pass_rate >= 0.8:  # 80% thresholds must pass
            # Check baseline degradation
            significant_degradations = sum(
                1 for change in baseline_comparison.values()
                if change < -5.0  # More than 5% degradation
            )

            if significant_degradations == 0:
                return "passed"
            elif significant_degradations <= 1:
                return "degraded"
            else:
                return "failed"
        else:
            return "failed"

    def generate_regression_report(self, results: List[RegressionTestResult]) -> Dict[str, Any]:
        """Generuje regression report"""
        if not results:
            return {"error": "No regression test results"}

        passed_tests = [r for r in results if r.overall_status == "passed"]
        failed_tests = [r for r in results if r.overall_status == "failed"]
        degraded_tests = [r for r in results if r.overall_status == "degraded"]

        report = {
            "regression_test_summary": {
                "total_domains_tested": len(results),
                "passed_tests": len(passed_tests),
                "failed_tests": len(failed_tests),
                "degraded_tests": len(degraded_tests),
                "overall_pass_rate": len(passed_tests) / len(results),
                "test_execution_time": datetime.now().isoformat()
            },
            "domain_results": {},
            "performance_analysis": {},
            "quality_trends": {},
            "critical_failures": [],
            "recommendations": []
        }

        # Domain-specific results
        for result in results:
            domain_report = {
                "status": result.overall_status,
                "thresholds_passed": result.passed_thresholds,
                "baseline_comparison": result.baseline_comparison,
                "performance_metrics": result.performance_metrics,
                "evaluation_metrics": {
                    "evidence_coverage": result.evaluation_result.evidence_coverage,
                    "citation_precision": result.evaluation_result.citation_precision,
                    "groundedness": result.evaluation_result.groundedness,
                    "hallucination_rate": result.evaluation_result.hallucination_rate,
                    "disagreement_coverage": result.evaluation_result.disagreement_coverage,
                    "context_usage_efficiency": result.evaluation_result.context_usage_efficiency
                }
            }
            report["domain_results"][result.domain] = domain_report

        # Performance analysis
        latencies = [r.performance_metrics.get("latency_ms", 0) for r in results]
        efficiencies = [r.evaluation_result.context_usage_efficiency for r in results]

        report["performance_analysis"] = {
            "latency_stats": {
                "mean_ms": np.mean(latencies),
                "p95_ms": np.percentile(latencies, 95),
                "max_ms": np.max(latencies)
            },
            "efficiency_stats": {
                "mean": np.mean(efficiencies),
                "min": np.min(efficiencies),
                "max": np.max(efficiencies)
            }
        }

        # Identify critical failures
        for result in failed_tests:
            critical_failure = {
                "domain": result.domain,
                "failed_thresholds": [k for k, v in result.passed_thresholds.items() if not v],
                "baseline_degradations": [
                    k for k, v in result.baseline_comparison.items()
                    if v < -10.0  # More than 10% degradation
                ]
            }
            report["critical_failures"].append(critical_failure)

        # Generate recommendations
        if failed_tests:
            report["recommendations"].append(
                f"❌ {len(failed_tests)} domén selhalo. Prioritně opravit před nasazením."
            )

        if degraded_tests:
            report["recommendations"].append(
                f"⚠️ {len(degraded_tests)} domén má degradovaný výkon. Zkontrolovat a optimalizovat."
            )

        if report["performance_analysis"]["latency_stats"]["p95_ms"] > 120000:  # 2 minutes
            report["recommendations"].append(
                "🐌 Vysoká latence (P95 > 2min). Optimalizovat rychlost retrieval a synthesis."
            )

        return report

    async def export_regression_results(self, results: List[RegressionTestResult],
                                      output_path: Optional[str] = None):
        """Exportuje regression results"""
        if output_path is None:
            output_path = self.results_path / f"regression_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        export_data = {
            "regression_test_results": {
                "execution_timestamp": datetime.now().isoformat(),
                "total_domains": len(results),
                "summary_report": self.generate_regression_report(results),
                "detailed_results": []
            }
        }

        # Add detailed results
        for result in results:
            detailed_result = {
                "domain": result.domain,
                "query": result.test_case.query,
                "overall_status": result.overall_status,
                "thresholds": {
                    "expected": result.test_case.quality_thresholds,
                    "actual": {
                        "evidence_coverage": result.evaluation_result.evidence_coverage,
                        "citation_precision": result.evaluation_result.citation_precision,
                        "groundedness": result.evaluation_result.groundedness,
                        "hallucination_rate": result.evaluation_result.hallucination_rate,
                        "primary_source_ratio": result.evaluation_result.primary_source_ratio
                    },
                    "passed": result.passed_thresholds
                },
                "baseline_comparison": result.baseline_comparison,
                "performance": result.performance_metrics
            }
            export_data["regression_test_results"]["detailed_results"].append(detailed_result)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"📊 Regression results exportovány: {output_file}")


# Factory function
def create_regression_test_suite(config: Dict[str, Any]) -> RegressionTestSuite:
    """Factory function pro vytvoření regression test suite"""
    return RegressionTestSuite(config)


if __name__ == "__main__":
    # Test regression suite
    config = {
        "evaluation": {
            "k_values": [1, 3, 5, 10],
            "min_evidence_per_claim": 2,
            "hallucination_threshold": 0.1
        },
        "regression": {
            "baseline_path": "data/baselines",
            "results_path": "docs/regression_results"
        }
    }

    test_suite = create_regression_test_suite(config)
    print(f"✅ Regression Test Suite vytvořena s {len(test_suite.domain_test_cases)} doménami!")
