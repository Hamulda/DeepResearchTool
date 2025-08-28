#!/usr/bin/env python3
"""
Domain-Specific Evaluation Suites
Comprehensive evaluation framework for different research domains

Author: Senior Python/MLOps Agent
"""

from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import json
import csv
import logging
import statistics
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class EvaluationDomain(Enum):
    """Research domains for evaluation"""
    MEDICAL = "medical"
    LEGAL = "legal"
    FINANCIAL = "financial"
    SCIENTIFIC = "scientific"
    NEWS = "news"
    ACADEMIC = "academic"
    POLICY = "policy"


class MetricType(Enum):
    """Types of evaluation metrics"""
    RECALL = "recall"
    PRECISION = "precision"
    NDCG = "ndcg"
    GROUNDEDNESS = "groundedness"
    CITATION_PRECISION = "citation_precision"
    CONTRADICTION_RATE = "contradiction_rate"
    CALIBRATION = "calibration"
    LATENCY = "latency"
    COST_EFFICIENCY = "cost_efficiency"


@dataclass
class EvaluationQuery:
    """Single evaluation query with ground truth"""
    query_id: str
    query_text: str
    domain: EvaluationDomain
    expected_claims: List[str]
    expected_sources: List[str]
    ground_truth_labels: Dict[str, Any]
    difficulty: str = "medium"  # easy, medium, hard
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of running an evaluation"""
    query_id: str
    domain: EvaluationDomain
    metrics: Dict[str, float]
    claims_generated: List[Dict[str, Any]]
    execution_time_ms: float
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainEvaluationSuite:
    """Evaluation suite for a specific domain"""
    domain: EvaluationDomain
    name: str
    description: str
    queries: List[EvaluationQuery]
    expected_metrics: Dict[str, Dict[str, float]]  # metric_name -> {min, target, max}

    def get_query_count(self) -> int:
        return len(self.queries)

    def get_difficulty_distribution(self) -> Dict[str, int]:
        from collections import Counter
        return dict(Counter(q.difficulty for q in self.queries))


class BaseDomainEvaluator(ABC):
    """Base class for domain-specific evaluators"""

    def __init__(self, domain: EvaluationDomain, config: Dict[str, Any]):
        self.domain = domain
        self.config = config
        self.domain_config = config.get("evaluation", {}).get("domains", {}).get(domain.value, {})

    @abstractmethod
    def load_evaluation_suite(self) -> DomainEvaluationSuite:
        """Load domain-specific evaluation queries and ground truth"""
        pass

    @abstractmethod
    def calculate_domain_specific_metrics(
        self,
        query: EvaluationQuery,
        result: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate domain-specific evaluation metrics"""
        pass

    def evaluate_groundedness(self, claims: List[Dict[str, Any]], query: EvaluationQuery) -> float:
        """Evaluate groundedness of claims against expected sources"""

        if not claims:
            return 0.0

        grounded_claims = 0
        expected_sources = set(query.expected_sources)

        for claim in claims:
            claim_sources = set()
            for evidence in claim.get("evidence", []):
                source_url = evidence.get("source_url", "")
                if source_url:
                    claim_sources.add(source_url)

            # Check if claim has sources from expected set
            if claim_sources.intersection(expected_sources):
                grounded_claims += 1

        return grounded_claims / len(claims)

    def evaluate_citation_precision(self, claims: List[Dict[str, Any]], query: EvaluationQuery) -> float:
        """Evaluate precision of citations"""

        total_citations = 0
        valid_citations = 0
        expected_sources = set(query.expected_sources)

        for claim in claims:
            for evidence in claim.get("evidence", []):
                total_citations += 1
                source_url = evidence.get("source_url", "")

                # Citation is valid if from expected sources
                if source_url in expected_sources:
                    valid_citations += 1

        return valid_citations / max(total_citations, 1)

    def evaluate_recall_at_k(self, claims: List[Dict[str, Any]], query: EvaluationQuery, k: int = 10) -> float:
        """Evaluate recall@k for claims"""

        expected_claims = set(query.expected_claims)
        generated_claims = set()

        for i, claim in enumerate(claims[:k]):
            claim_text = claim.get("text", "").lower()
            generated_claims.add(claim_text)

        # Simple text matching for recall (could be enhanced with semantic similarity)
        matched_claims = 0
        for expected in expected_claims:
            for generated in generated_claims:
                if self._claims_match(expected.lower(), generated):
                    matched_claims += 1
                    break

        return matched_claims / max(len(expected_claims), 1)

    def _claims_match(self, expected: str, generated: str, threshold: float = 0.7) -> bool:
        """Check if two claims match (simple text overlap)"""

        expected_words = set(expected.split())
        generated_words = set(generated.split())

        if not expected_words:
            return False

        overlap = len(expected_words.intersection(generated_words))
        return overlap / len(expected_words) >= threshold


class MedicalEvaluator(BaseDomainEvaluator):
    """Evaluator for medical/healthcare domain"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(EvaluationDomain.MEDICAL, config)

    def load_evaluation_suite(self) -> DomainEvaluationSuite:
        """Load medical domain evaluation suite"""

        # Sample medical evaluation queries
        queries = [
            EvaluationQuery(
                query_id="med_001",
                query_text="COVID-19 vaccine effectiveness against Delta variant",
                domain=EvaluationDomain.MEDICAL,
                expected_claims=[
                    "COVID-19 vaccines show reduced effectiveness against Delta variant",
                    "Pfizer vaccine shows 64% effectiveness against Delta",
                    "Booster doses restore effectiveness against variants"
                ],
                expected_sources=[
                    "https://pubmed.ncbi.nlm.nih.gov/study1",
                    "https://nejm.org/article1",
                    "https://cdc.gov/report1"
                ],
                ground_truth_labels={"vaccine_effectiveness": 0.64, "variant": "delta"},
                difficulty="medium"
            ),
            EvaluationQuery(
                query_id="med_002",
                query_text="Side effects of mRNA vaccines in elderly population",
                domain=EvaluationDomain.MEDICAL,
                expected_claims=[
                    "mRNA vaccines show mild side effects in elderly",
                    "Most common side effect is injection site pain",
                    "Serious adverse events are rare in elderly"
                ],
                expected_sources=[
                    "https://pubmed.ncbi.nlm.nih.gov/study2",
                    "https://fda.gov/safety-report"
                ],
                ground_truth_labels={"population": "elderly", "vaccine_type": "mRNA"},
                difficulty="hard"
            )
        ]

        expected_metrics = {
            "groundedness": {"min": 0.7, "target": 0.85, "max": 1.0},
            "citation_precision": {"min": 0.8, "target": 0.9, "max": 1.0},
            "recall@10": {"min": 0.6, "target": 0.75, "max": 1.0},
            "contradiction_rate": {"min": 0.0, "target": 0.05, "max": 0.2}
        }

        return DomainEvaluationSuite(
            domain=EvaluationDomain.MEDICAL,
            name="Medical Research Evaluation",
            description="Evaluation suite for medical and healthcare research",
            queries=queries,
            expected_metrics=expected_metrics
        )

    def calculate_domain_specific_metrics(
        self,
        query: EvaluationQuery,
        result: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate medical-specific metrics"""

        claims = result.get("claims", [])

        metrics = {
            "groundedness": self.evaluate_groundedness(claims, query),
            "citation_precision": self.evaluate_citation_precision(claims, query),
            "recall@10": self.evaluate_recall_at_k(claims, query, 10),
            "medical_source_ratio": self._evaluate_medical_source_ratio(claims),
            "clinical_evidence_ratio": self._evaluate_clinical_evidence_ratio(claims)
        }

        return metrics

    def _evaluate_medical_source_ratio(self, claims: List[Dict[str, Any]]) -> float:
        """Evaluate ratio of medical sources used"""

        total_sources = 0
        medical_sources = 0

        medical_domains = {
            "pubmed.ncbi.nlm.nih.gov", "nejm.org", "thelancet.com",
            "jama.jamanetwork.com", "bmj.com", "cdc.gov", "who.int"
        }

        for claim in claims:
            for evidence in claim.get("evidence", []):
                source_url = evidence.get("source_url", "")
                if source_url:
                    total_sources += 1
                    for domain in medical_domains:
                        if domain in source_url:
                            medical_sources += 1
                            break

        return medical_sources / max(total_sources, 1)

    def _evaluate_clinical_evidence_ratio(self, claims: List[Dict[str, Any]]) -> float:
        """Evaluate ratio of claims with clinical evidence"""

        if not claims:
            return 0.0

        clinical_keywords = ["clinical trial", "randomized", "placebo", "study", "patients"]
        clinical_claims = 0

        for claim in claims:
            claim_text = claim.get("text", "").lower()
            if any(keyword in claim_text for keyword in clinical_keywords):
                clinical_claims += 1

        return clinical_claims / len(claims)


class LegalEvaluator(BaseDomainEvaluator):
    """Evaluator for legal domain"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(EvaluationDomain.LEGAL, config)

    def load_evaluation_suite(self) -> DomainEvaluationSuite:
        """Load legal domain evaluation suite"""

        queries = [
            EvaluationQuery(
                query_id="legal_001",
                query_text="GDPR compliance requirements for AI systems",
                domain=EvaluationDomain.LEGAL,
                expected_claims=[
                    "AI systems must ensure data subject rights under GDPR",
                    "Automated decision making requires explicit consent",
                    "Data protection impact assessments required for high-risk AI"
                ],
                expected_sources=[
                    "https://eur-lex.europa.eu/legal-content/gdpr",
                    "https://edpb.europa.eu/guidelines"
                ],
                ground_truth_labels={"regulation": "GDPR", "domain": "AI"},
                difficulty="hard"
            )
        ]

        expected_metrics = {
            "groundedness": {"min": 0.8, "target": 0.9, "max": 1.0},
            "citation_precision": {"min": 0.85, "target": 0.95, "max": 1.0},
            "legal_authority_ratio": {"min": 0.7, "target": 0.85, "max": 1.0}
        }

        return DomainEvaluationSuite(
            domain=EvaluationDomain.LEGAL,
            name="Legal Research Evaluation",
            description="Evaluation suite for legal research and compliance",
            queries=queries,
            expected_metrics=expected_metrics
        )

    def calculate_domain_specific_metrics(
        self,
        query: EvaluationQuery,
        result: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate legal-specific metrics"""

        claims = result.get("claims", [])

        metrics = {
            "groundedness": self.evaluate_groundedness(claims, query),
            "citation_precision": self.evaluate_citation_precision(claims, query),
            "legal_authority_ratio": self._evaluate_legal_authority_ratio(claims),
            "statutory_reference_ratio": self._evaluate_statutory_reference_ratio(claims)
        }

        return metrics

    def _evaluate_legal_authority_ratio(self, claims: List[Dict[str, Any]]) -> float:
        """Evaluate ratio of authoritative legal sources"""

        total_sources = 0
        authoritative_sources = 0

        legal_authorities = {
            "eur-lex.europa.eu", "law.gov", "supremecourt.gov",
            "courtlistener.com", "westlaw.com", "lexisnexis.com"
        }

        for claim in claims:
            for evidence in claim.get("evidence", []):
                source_url = evidence.get("source_url", "")
                if source_url:
                    total_sources += 1
                    for authority in legal_authorities:
                        if authority in source_url:
                            authoritative_sources += 1
                            break

        return authoritative_sources / max(total_sources, 1)

    def _evaluate_statutory_reference_ratio(self, claims: List[Dict[str, Any]]) -> float:
        """Evaluate ratio of claims with statutory references"""

        if not claims:
            return 0.0

        statutory_patterns = ["article", "section", "paragraph", "regulation", "directive", "§"]
        claims_with_references = 0

        for claim in claims:
            claim_text = claim.get("text", "").lower()
            if any(pattern in claim_text for pattern in statutory_patterns):
                claims_with_references += 1

        return claims_with_references / len(claims)


class FinancialEvaluator(BaseDomainEvaluator):
    """Evaluator for financial domain"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(EvaluationDomain.FINANCIAL, config)

    def load_evaluation_suite(self) -> DomainEvaluationSuite:
        """Load financial domain evaluation suite"""

        queries = [
            EvaluationQuery(
                query_id="fin_001",
                query_text="Federal Reserve interest rate policy impact on inflation",
                domain=EvaluationDomain.FINANCIAL,
                expected_claims=[
                    "Fed raises interest rates to combat inflation",
                    "Higher rates reduce money supply and spending",
                    "Rate increases show effectiveness after 12-18 months"
                ],
                expected_sources=[
                    "https://federalreserve.gov/fomc",
                    "https://sec.gov/filings"
                ],
                ground_truth_labels={"topic": "monetary_policy", "timeframe": "2023-2024"},
                difficulty="medium"
            )
        ]

        expected_metrics = {
            "groundedness": {"min": 0.75, "target": 0.85, "max": 1.0},
            "citation_precision": {"min": 0.8, "target": 0.9, "max": 1.0},
            "quantitative_accuracy": {"min": 0.7, "target": 0.85, "max": 1.0}
        }

        return DomainEvaluationSuite(
            domain=EvaluationDomain.FINANCIAL,
            name="Financial Research Evaluation",
            description="Evaluation suite for financial and economic research",
            queries=queries,
            expected_metrics=expected_metrics
        )

    def calculate_domain_specific_metrics(
        self,
        query: EvaluationQuery,
        result: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate financial-specific metrics"""

        claims = result.get("claims", [])

        metrics = {
            "groundedness": self.evaluate_groundedness(claims, query),
            "citation_precision": self.evaluate_citation_precision(claims, query),
            "financial_source_ratio": self._evaluate_financial_source_ratio(claims),
            "quantitative_claim_ratio": self._evaluate_quantitative_claim_ratio(claims)
        }

        return metrics

    def _evaluate_financial_source_ratio(self, claims: List[Dict[str, Any]]) -> float:
        """Evaluate ratio of authoritative financial sources"""

        total_sources = 0
        financial_sources = 0

        financial_authorities = {
            "federalreserve.gov", "sec.gov", "treasury.gov", "imf.org",
            "worldbank.org", "bis.org", "bloomberg.com", "reuters.com"
        }

        for claim in claims:
            for evidence in claim.get("evidence", []):
                source_url = evidence.get("source_url", "")
                if source_url:
                    total_sources += 1
                    for authority in financial_authorities:
                        if authority in source_url:
                            financial_sources += 1
                            break

        return financial_sources / max(total_sources, 1)

    def _evaluate_quantitative_claim_ratio(self, claims: List[Dict[str, Any]]) -> float:
        """Evaluate ratio of claims with quantitative data"""

        if not claims:
            return 0.0

        import re
        quantitative_claims = 0

        # Pattern for numbers with financial context
        number_pattern = r'\d+(?:\.\d+)?%|\$\d+|\d+\s*basis\s*points|\d+(?:\.\d+)?\s*trillion|\d+(?:\.\d+)?\s*billion'

        for claim in claims:
            claim_text = claim.get("text", "")
            if re.search(number_pattern, claim_text, re.IGNORECASE):
                quantitative_claims += 1

        return quantitative_claims / len(claims)


class DomainEvaluationRunner:
    """Main evaluation runner for all domains"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.evaluation_config = config.get("evaluation", {})

        # Initialize domain evaluators
        self.evaluators = {
            EvaluationDomain.MEDICAL: MedicalEvaluator(config),
            EvaluationDomain.LEGAL: LegalEvaluator(config),
            EvaluationDomain.FINANCIAL: FinancialEvaluator(config)
        }

        # Results storage
        self.results_dir = Path(config.get("evaluation_results", "evaluation_results"))
        self.results_dir.mkdir(exist_ok=True)

        logger.info(f"Domain evaluation runner initialized with {len(self.evaluators)} domains")

    def run_domain_evaluation(
        self,
        domain: EvaluationDomain,
        research_system: Any,  # The actual research system to evaluate
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Run evaluation for a specific domain"""

        if domain not in self.evaluators:
            raise ValueError(f"No evaluator for domain: {domain}")

        evaluator = self.evaluators[domain]
        suite = evaluator.load_evaluation_suite()

        logger.info(f"Running {domain.value} evaluation with {suite.get_query_count()} queries")

        results = []
        start_time = datetime.now(timezone.utc)

        for query in suite.queries:
            try:
                # Run query through research system
                query_start = datetime.now(timezone.utc)

                # This would call the actual research system
                # For now, we'll simulate the call
                system_result = self._simulate_research_system_call(research_system, query)

                query_end = datetime.now(timezone.utc)
                execution_time_ms = (query_end - query_start).total_seconds() * 1000

                # Calculate metrics
                metrics = evaluator.calculate_domain_specific_metrics(query, system_result)

                # Add standard metrics
                metrics.update({
                    "execution_time_ms": execution_time_ms,
                    "claims_count": len(system_result.get("claims", [])),
                    "evidence_count": sum(len(c.get("evidence", [])) for c in system_result.get("claims", []))
                })

                result = EvaluationResult(
                    query_id=query.query_id,
                    domain=domain,
                    metrics=metrics,
                    claims_generated=system_result.get("claims", []),
                    execution_time_ms=execution_time_ms,
                    metadata={"query_difficulty": query.difficulty}
                )

                results.append(result)
                logger.info(f"Completed query {query.query_id}: {len(metrics)} metrics calculated")

            except Exception as e:
                error_result = EvaluationResult(
                    query_id=query.query_id,
                    domain=domain,
                    metrics={},
                    claims_generated=[],
                    execution_time_ms=0.0,
                    errors=[str(e)]
                )
                results.append(error_result)
                logger.error(f"Failed to evaluate query {query.query_id}: {e}")

        # Calculate aggregate metrics
        evaluation_summary = self._calculate_evaluation_summary(suite, results)

        evaluation_data = {
            "domain": domain.value,
            "suite_name": suite.name,
            "evaluation_timestamp": start_time.isoformat(),
            "total_queries": len(suite.queries),
            "successful_queries": len([r for r in results if not r.errors]),
            "failed_queries": len([r for r in results if r.errors]),
            "summary_metrics": evaluation_summary,
            "expected_metrics": suite.expected_metrics,
            "individual_results": [self._result_to_dict(r) for r in results]
        }

        if save_results:
            self._save_evaluation_results(domain, evaluation_data)

        return evaluation_data

    def _simulate_research_system_call(self, research_system: Any, query: EvaluationQuery) -> Dict[str, Any]:
        """Simulate calling the research system (placeholder)"""

        # In actual implementation, this would call:
        # return research_system.execute_research_workflow(query.query_text)

        # For now, return mock data
        return {
            "claims": [
                {
                    "text": f"Mock claim for {query.query_text}",
                    "evidence": [
                        {"source_url": "https://example.com/source1", "confidence": 0.8}
                    ],
                    "confidence": 0.7
                }
            ],
            "metadata": {"execution_time": 1500}
        }

    def _calculate_evaluation_summary(
        self,
        suite: DomainEvaluationSuite,
        results: List[EvaluationResult]
    ) -> Dict[str, float]:
        """Calculate summary metrics across all queries"""

        successful_results = [r for r in results if not r.errors]

        if not successful_results:
            return {}

        # Aggregate metrics
        metric_names = set()
        for result in successful_results:
            metric_names.update(result.metrics.keys())

        summary = {}
        for metric_name in metric_names:
            values = [r.metrics[metric_name] for r in successful_results if metric_name in r.metrics]
            if values:
                summary[f"{metric_name}_avg"] = statistics.mean(values)
                summary[f"{metric_name}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
                summary[f"{metric_name}_min"] = min(values)
                summary[f"{metric_name}_max"] = max(values)

        # Calculate pass/fail rates
        for metric_name, thresholds in suite.expected_metrics.items():
            if f"{metric_name}_avg" in summary:
                avg_value = summary[f"{metric_name}_avg"]
                target = thresholds.get("target", thresholds.get("min", 0.5))

                summary[f"{metric_name}_target_met"] = 1.0 if avg_value >= target else 0.0
                summary[f"{metric_name}_target_ratio"] = avg_value / target if target > 0 else 0.0

        return summary

    def _result_to_dict(self, result: EvaluationResult) -> Dict[str, Any]:
        """Convert evaluation result to dictionary"""

        return {
            "query_id": result.query_id,
            "domain": result.domain.value,
            "metrics": result.metrics,
            "claims_count": len(result.claims_generated),
            "execution_time_ms": result.execution_time_ms,
            "errors": result.errors,
            "metadata": result.metadata
        }

    def _save_evaluation_results(self, domain: EvaluationDomain, evaluation_data: Dict[str, Any]):
        """Save evaluation results to files"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        json_file = self.results_dir / f"{domain.value}_evaluation_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)

        # Save CSV summary
        csv_file = self.results_dir / f"{domain.value}_summary_{timestamp}.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(["Metric", "Value", "Target", "Status"])

            # Write summary metrics
            expected_metrics = evaluation_data.get("expected_metrics", {})
            summary_metrics = evaluation_data.get("summary_metrics", {})

            for metric_name in expected_metrics:
                avg_key = f"{metric_name}_avg"
                if avg_key in summary_metrics:
                    value = summary_metrics[avg_key]
                    target = expected_metrics[metric_name].get("target", 0.0)
                    status = "PASS" if value >= target else "FAIL"
                    writer.writerow([metric_name, f"{value:.4f}", f"{target:.4f}", status])

        logger.info(f"Evaluation results saved: {json_file}, {csv_file}")

    def run_all_domains(self, research_system: Any) -> Dict[str, Any]:
        """Run evaluation across all domains"""

        all_results = {}

        for domain in self.evaluators.keys():
            try:
                domain_results = self.run_domain_evaluation(domain, research_system)
                all_results[domain.value] = domain_results
            except Exception as e:
                logger.error(f"Failed to evaluate domain {domain.value}: {e}")
                all_results[domain.value] = {"error": str(e)}

        # Create combined summary
        combined_summary = self._create_combined_summary(all_results)

        # Save combined results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_file = self.results_dir / f"combined_evaluation_{timestamp}.json"

        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump({
                "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
                "domains_evaluated": list(all_results.keys()),
                "combined_summary": combined_summary,
                "domain_results": all_results
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"Combined evaluation results saved: {combined_file}")
        return all_results

    def _create_combined_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary across all domains"""

        total_queries = 0
        total_successful = 0
        total_failed = 0

        domain_scores = {}

        for domain, results in all_results.items():
            if "error" in results:
                continue

            total_queries += results.get("total_queries", 0)
            total_successful += results.get("successful_queries", 0)
            total_failed += results.get("failed_queries", 0)

            # Calculate domain score (average of key metrics)
            summary_metrics = results.get("summary_metrics", {})
            key_metrics = ["groundedness_avg", "citation_precision_avg", "recall@10_avg"]

            domain_metric_values = []
            for metric in key_metrics:
                if metric in summary_metrics:
                    domain_metric_values.append(summary_metrics[metric])

            if domain_metric_values:
                domain_scores[domain] = statistics.mean(domain_metric_values)

        return {
            "total_queries": total_queries,
            "success_rate": total_successful / max(total_queries, 1),
            "domain_scores": domain_scores,
            "overall_score": statistics.mean(domain_scores.values()) if domain_scores else 0.0
        }


def create_domain_evaluation_runner(config: Dict[str, Any]) -> DomainEvaluationRunner:
    """Factory function for domain evaluation runner"""
    return DomainEvaluationRunner(config)


# Usage example
if __name__ == "__main__":
    config = {
        "evaluation": {
            "domains": {
                "medical": {"enabled": True},
                "legal": {"enabled": True},
                "financial": {"enabled": True}
            }
        },
        "evaluation_results": "test_evaluation_results"
    }

    runner = DomainEvaluationRunner(config)

    # Run medical domain evaluation
    medical_results = runner.run_domain_evaluation(
        EvaluationDomain.MEDICAL,
        None  # Mock research system
    )

    print(f"Medical evaluation completed:")
    print(f"- Total queries: {medical_results['total_queries']}")
    print(f"- Successful: {medical_results['successful_queries']}")
    print(f"- Key metrics: {list(medical_results['summary_metrics'].keys())[:5]}")

    # Run all domains
    all_results = runner.run_all_domains(None)
    print(f"\nAll domains evaluation completed: {len(all_results)} domains")
