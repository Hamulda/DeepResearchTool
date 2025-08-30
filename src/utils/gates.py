#!/usr/bin/env python3
"""Centrální gatekeeper pro automatické validační brány
Nahrazuje všechny HITL checkpointy fail-hard pravidly

Author: Senior Python/MLOps Agent
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Any

logger = logging.getLogger(__name__)


class GateError(Exception):
    """Základní třída pro gate errors"""

    def __init__(self, message: str, suggestions: list[str] = None):
        super().__init__(message)
        self.suggestions = suggestions or []


class EvidenceGateError(GateError):
    """Chyba při validaci evidence - nedostatek citací, nekvalitní zdroje"""



class ComplianceGateError(GateError):
    """Chyba při dodržování compliance pravidel - robots.txt, rate limiting"""



class MetricsGateError(GateError):
    """Chyba při nesplnění metrik kvality - recall, precision, groundedness"""



class QualityGateError(GateError):
    """Chyba při kontrole kvality výstupu - formát, struktura, konzistence"""



@dataclass
class GateResult:
    """Výsledek validační brány"""

    passed: bool
    gate_name: str
    score: float
    threshold: float
    message: str
    suggestions: list[str]
    metadata: dict[str, Any]


class ValidationGate(ABC):
    """Abstraktní třída pro validační brány"""

    def __init__(self, name: str, threshold: float):
        self.name = name
        self.threshold = threshold

    @abstractmethod
    async def validate(self, data: Any) -> GateResult:
        """Validace dat"""


class EvidenceGate(ValidationGate):
    """Validace evidence - minimální počet citací per claim"""

    def __init__(self, min_citations_per_claim: int = 2, min_source_diversity: float = 0.7):
        super().__init__("evidence_gate", min_citations_per_claim)
        self.min_citations_per_claim = min_citations_per_claim
        self.min_source_diversity = min_source_diversity

    async def validate(self, synthesis_result: Any) -> GateResult:
        """Validace evidence binding pro každý claim"""
        if not hasattr(synthesis_result, "claims"):
            return GateResult(
                passed=False,
                gate_name=self.name,
                score=0.0,
                threshold=self.threshold,
                message="No claims found in synthesis result",
                suggestions=["Check synthesis pipeline", "Verify input documents"],
                metadata={},
            )

        claims = synthesis_result.claims
        if not claims:
            return GateResult(
                passed=False,
                gate_name=self.name,
                score=0.0,
                threshold=self.threshold,
                message="No claims generated",
                suggestions=[
                    "Check input document quality",
                    "Verify synthesis template",
                    "Review claim extraction logic",
                ],
                metadata={"claim_count": 0},
            )

        failed_claims = []
        total_citations = 0

        for i, claim in enumerate(claims):
            citations = getattr(claim, "citations", [])
            citation_count = len(citations)
            total_citations += citation_count

            if citation_count < self.min_citations_per_claim:
                failed_claims.append(
                    {
                        "claim_id": i,
                        "claim_text": str(claim)[:100] + "...",
                        "citation_count": citation_count,
                        "required": self.min_citations_per_claim,
                    }
                )

        avg_citations = total_citations / len(claims) if claims else 0
        passed = len(failed_claims) == 0

        if not passed:
            suggestions = [
                "Increase retrieval recall parameters",
                "Expand source diversity",
                "Review claim granularity",
                f"Target {self.min_citations_per_claim} citations per claim minimum",
            ]
            message = f"Evidence gate failed: {len(failed_claims)}/{len(claims)} claims lack sufficient citations"
        else:
            suggestions = []
            message = f"Evidence gate passed: All {len(claims)} claims have ≥{self.min_citations_per_claim} citations"

        return GateResult(
            passed=passed,
            gate_name=self.name,
            score=avg_citations,
            threshold=self.threshold,
            message=message,
            suggestions=suggestions,
            metadata={
                "total_claims": len(claims),
                "failed_claims": failed_claims,
                "avg_citations_per_claim": avg_citations,
                "total_citations": total_citations,
            },
        )


class ComplianceGate(ValidationGate):
    """Validace compliance - robots.txt, rate limiting, legal constraints"""

    def __init__(self, max_rate_violations: int = 0, max_robots_violations: int = 0):
        super().__init__("compliance_gate", 0.0)  # Zero tolerance
        self.max_rate_violations = max_rate_violations
        self.max_robots_violations = max_robots_violations

    async def validate(self, retrieval_log: dict[str, Any]) -> GateResult:
        """Validace compliance během retrieval procesu"""
        rate_violations = retrieval_log.get("rate_limit_violations", [])
        robots_violations = retrieval_log.get("robots_txt_violations", [])
        blocked_domains = retrieval_log.get("blocked_domains", [])

        total_violations = len(rate_violations) + len(robots_violations)
        passed = (
            len(rate_violations) <= self.max_rate_violations
            and len(robots_violations) <= self.max_robots_violations
        )

        if not passed:
            suggestions = [
                "Review and increase rate limiting delays",
                "Check robots.txt compliance logic",
                "Add more domains to whitelist",
                "Implement exponential backoff",
                "Review user-agent configuration",
            ]
            message = f"Compliance violations detected: {len(rate_violations)} rate limit, {len(robots_violations)} robots.txt"
        else:
            suggestions = []
            message = "All compliance checks passed"

        return GateResult(
            passed=passed,
            gate_name=self.name,
            score=1.0 - (total_violations / max(1, len(retrieval_log.get("requests", [])))),
            threshold=self.threshold,
            message=message,
            suggestions=suggestions,
            metadata={
                "rate_violations": rate_violations,
                "robots_violations": robots_violations,
                "blocked_domains": blocked_domains,
                "total_requests": len(retrieval_log.get("requests", [])),
            },
        )


class MetricsGate(ValidationGate):
    """Validace metrik kvality - recall, precision, groundedness"""

    def __init__(
        self, min_recall: float = 0.7, min_precision: float = 0.8, min_groundedness: float = 0.85
    ):
        super().__init__("metrics_gate", min_recall)
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.min_groundedness = min_groundedness

    async def validate(self, evaluation_result: dict[str, float]) -> GateResult:
        """Validace evaluation metrik"""
        recall = evaluation_result.get("recall_at_10", 0.0)
        precision = evaluation_result.get("citation_precision", 0.0)
        groundedness = evaluation_result.get("groundedness", 0.0)

        failed_metrics = []

        if recall < self.min_recall:
            failed_metrics.append(f"recall@10: {recall:.3f} < {self.min_recall}")

        if precision < self.min_precision:
            failed_metrics.append(f"citation_precision: {precision:.3f} < {self.min_precision}")

        if groundedness < self.min_groundedness:
            failed_metrics.append(f"groundedness: {groundedness:.3f} < {self.min_groundedness}")

        passed = len(failed_metrics) == 0
        composite_score = (recall + precision + groundedness) / 3

        if not passed:
            suggestions = [
                "Increase retrieval parameters (ef_search, top_k)",
                "Improve re-ranking model quality",
                "Review synthesis template accuracy",
                "Add more diverse training data",
                "Tune RRF parameters",
            ]
            message = f"Metrics gate failed: {', '.join(failed_metrics)}"
        else:
            suggestions = []
            message = "All quality metrics passed thresholds"

        return GateResult(
            passed=passed,
            gate_name=self.name,
            score=composite_score,
            threshold=self.threshold,
            message=message,
            suggestions=suggestions,
            metadata={
                "recall_at_10": recall,
                "citation_precision": precision,
                "groundedness": groundedness,
                "failed_metrics": failed_metrics,
            },
        )


class QualityGate(ValidationGate):
    """Validace kvality výstupu - formát, struktura, token budget"""

    def __init__(self, max_token_budget: int = 8000, min_claim_quality: float = 0.7):
        super().__init__("quality_gate", min_claim_quality)
        self.max_token_budget = max_token_budget
        self.min_claim_quality = min_claim_quality

    async def validate(self, output_data: dict[str, Any]) -> GateResult:
        """Validace kvality finálního výstupu"""
        token_count = output_data.get("token_count", 0)
        claims = output_data.get("claims", [])
        citations = output_data.get("citations", [])

        issues = []

        # Token budget check
        if token_count > self.max_token_budget:
            issues.append(f"Token budget exceeded: {token_count} > {self.max_token_budget}")

        # Format validation
        if not isinstance(claims, list) or len(claims) == 0:
            issues.append("No valid claims found in output")

        if not isinstance(citations, list) or len(citations) == 0:
            issues.append("No citations found in output")

        # Structure validation
        for i, claim in enumerate(claims):
            if not hasattr(claim, "text") or not hasattr(claim, "citations"):
                issues.append(f"Claim {i} missing required fields")

        passed = len(issues) == 0
        quality_score = max(0.0, 1.0 - len(issues) * 0.1)

        if not passed:
            suggestions = [
                "Review output format validation",
                "Check synthesis template structure",
                "Implement token budget compression",
                "Validate claim extraction logic",
            ]
            message = f"Quality issues detected: {'; '.join(issues)}"
        else:
            suggestions = []
            message = "Output quality validation passed"

        return GateResult(
            passed=passed,
            gate_name=self.name,
            score=quality_score,
            threshold=self.threshold,
            message=message,
            suggestions=suggestions,
            metadata={
                "token_count": token_count,
                "token_budget": self.max_token_budget,
                "claim_count": len(claims),
                "citation_count": len(citations),
                "issues": issues,
            },
        )


class GateKeeper:
    """Centrální gatekeeper pro všechny validační brány"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.gates = []
        self._initialize_gates()

    def _initialize_gates(self):
        """Inicializace všech bran podle konfigurace"""
        # Evidence gate
        evidence_config = self.config.get("gates", {}).get("evidence", {})
        self.gates.append(
            EvidenceGate(
                min_citations_per_claim=evidence_config.get("min_citations_per_claim", 2),
                min_source_diversity=evidence_config.get("min_source_diversity", 0.7),
            )
        )

        # Compliance gate
        compliance_config = self.config.get("gates", {}).get("compliance", {})
        self.gates.append(
            ComplianceGate(
                max_rate_violations=compliance_config.get("max_rate_violations", 0),
                max_robots_violations=compliance_config.get("max_robots_violations", 0),
            )
        )

        # Metrics gate
        metrics_config = self.config.get("gates", {}).get("metrics", {})
        self.gates.append(
            MetricsGate(
                min_recall=metrics_config.get("min_recall", 0.7),
                min_precision=metrics_config.get("min_precision", 0.8),
                min_groundedness=metrics_config.get("min_groundedness", 0.85),
            )
        )

        # Quality gate
        quality_config = self.config.get("gates", {}).get("quality", {})
        self.gates.append(
            QualityGate(
                max_token_budget=quality_config.get("max_token_budget", 8000),
                min_claim_quality=quality_config.get("min_claim_quality", 0.7),
            )
        )

    async def validate_all(self, data: dict[str, Any]) -> list[GateResult]:
        """Spuštění všech validačních bran"""
        results = []

        for gate in self.gates:
            try:
                if gate.name == "evidence_gate":
                    result = await gate.validate(data.get("synthesis_result"))
                elif gate.name == "compliance_gate":
                    result = await gate.validate(data.get("retrieval_log", {}))
                elif gate.name == "metrics_gate":
                    result = await gate.validate(data.get("evaluation_result", {}))
                elif gate.name == "quality_gate":
                    result = await gate.validate(data.get("output_data", {}))
                else:
                    result = await gate.validate(data)

                results.append(result)

                # Fail-hard na první neprojitou bránu
                if not result.passed:
                    error_class = self._get_error_class(gate.name)
                    raise error_class(result.message, result.suggestions)

            except Exception as e:
                logger.error(f"Gate {gate.name} failed with error: {e}")
                results.append(
                    GateResult(
                        passed=False,
                        gate_name=gate.name,
                        score=0.0,
                        threshold=gate.threshold,
                        message=f"Gate execution failed: {e!s}",
                        suggestions=["Check gate implementation", "Review input data format"],
                        metadata={"error": str(e)},
                    )
                )
                raise

        return results

    def _get_error_class(self, gate_name: str) -> type:
        """Vrací odpovídající error třídu pro bránu"""
        error_mapping = {
            "evidence_gate": EvidenceGateError,
            "compliance_gate": ComplianceGateError,
            "metrics_gate": MetricsGateError,
            "quality_gate": QualityGateError,
        }
        return error_mapping.get(gate_name, GateError)

    def get_gate_summary(self, results: list[GateResult]) -> dict[str, Any]:
        """Vytvoří souhrn všech bran"""
        passed_gates = [r for r in results if r.passed]
        failed_gates = [r for r in results if not r.passed]

        return {
            "total_gates": len(results),
            "passed_gates": len(passed_gates),
            "failed_gates": len(failed_gates),
            "success_rate": len(passed_gates) / len(results) if results else 0.0,
            "gate_results": [
                {
                    "name": r.gate_name,
                    "passed": r.passed,
                    "score": r.score,
                    "threshold": r.threshold,
                    "message": r.message,
                }
                for r in results
            ],
            "suggestions": [suggestion for r in failed_gates for suggestion in r.suggestions],
        }
