#!/usr/bin/env python3
"""Validation Gates System
Implementuje fail-fast/fail-hard validační brány pro odstranění HITL checkpointů

Author: Senior Python/MLOps Agent
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Úrovně validace"""

    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationResult(Enum):
    """Výsledky validace"""

    PASS = "pass"
    FAIL_WARN = "fail_warn"
    FAIL_HARD = "fail_hard"


@dataclass
class ValidationGateResult:
    """Výsledek validační brány"""

    gate_name: str
    result: ValidationResult
    level: ValidationLevel
    message: str
    details: dict[str, Any]
    remediation_suggestion: str


class ValidationGate(ABC):
    """Abstraktní třída pro validační brány"""

    def __init__(self, name: str, level: ValidationLevel = ValidationLevel.ERROR):
        self.name = name
        self.level = level

    @abstractmethod
    async def validate(self, context: dict[str, Any]) -> ValidationGateResult:
        """Provedení validace"""


class QueryValidationGate(ValidationGate):
    """Validace kvality dotazu"""

    def __init__(self):
        super().__init__("query_validation", ValidationLevel.ERROR)

    async def validate(self, context: dict[str, Any]) -> ValidationGateResult:
        query = context.get("query", "")

        if not query or len(query.strip()) < 10:
            return ValidationGateResult(
                gate_name=self.name,
                result=ValidationResult.FAIL_HARD,
                level=self.level,
                message="Dotaz je příliš krátký nebo prázdný",
                details={"query_length": len(query)},
                remediation_suggestion="Zadejte dotaz s alespoň 10 znaky popisující výzkumnou otázku",
            )

        # Kontrola nebezpečných vzorů
        dangerous_patterns = ["DELETE", "DROP", "TRUNCATE", "<script>", "javascript:"]
        if any(pattern.lower() in query.lower() for pattern in dangerous_patterns):
            return ValidationGateResult(
                gate_name=self.name,
                result=ValidationResult.FAIL_HARD,
                level=ValidationLevel.CRITICAL,
                message="Dotaz obsahuje potenciálně nebezpečné vzory",
                details={"query": query[:100]},
                remediation_suggestion="Odstraňte nebezpečné vzory z dotazu",
            )

        return ValidationGateResult(
            gate_name=self.name,
            result=ValidationResult.PASS,
            level=self.level,
            message="Dotaz je validní",
            details={"query_length": len(query)},
            remediation_suggestion="",
        )


class RetrievalValidationGate(ValidationGate):
    """Validace výsledků retrievalu"""

    def __init__(self, min_documents: int = 5):
        super().__init__("retrieval_validation", ValidationLevel.ERROR)
        self.min_documents = min_documents

    async def validate(self, context: dict[str, Any]) -> ValidationGateResult:
        documents = context.get("retrieved_documents", [])

        if len(documents) < self.min_documents:
            return ValidationGateResult(
                gate_name=self.name,
                result=ValidationResult.FAIL_HARD,
                level=self.level,
                message=f"Nedostatek dokumentů pro analýzu (nalezeno {len(documents)}, minimum {self.min_documents})",
                details={"document_count": len(documents), "minimum_required": self.min_documents},
                remediation_suggestion="Rozšiřte dotaz nebo snižte filtry pro získání více dokumentů",
            )

        # Kontrola kvality dokumentů
        quality_score = sum(doc.get("relevance_score", 0) for doc in documents) / len(documents)
        if quality_score < 0.3:
            return ValidationGateResult(
                gate_name=self.name,
                result=ValidationResult.FAIL_WARN,
                level=ValidationLevel.WARNING,
                message=f"Nízká průměrná kvalita dokumentů (skóre: {quality_score:.2f})",
                details={"average_quality": quality_score, "document_count": len(documents)},
                remediation_suggestion="Zvážte reformulaci dotazu pro lepší relevanci",
            )

        return ValidationGateResult(
            gate_name=self.name,
            result=ValidationResult.PASS,
            level=self.level,
            message="Retrieval úspěšný",
            details={"document_count": len(documents), "average_quality": quality_score},
            remediation_suggestion="",
        )


class EvidenceValidationGate(ValidationGate):
    """Validace evidence binding"""

    def __init__(self, min_citations_per_claim: int = 2):
        super().__init__("evidence_validation", ValidationLevel.ERROR)
        self.min_citations_per_claim = min_citations_per_claim

    async def validate(self, context: dict[str, Any]) -> ValidationGateResult:
        claims = context.get("verified_claims", [])

        if not claims:
            return ValidationGateResult(
                gate_name=self.name,
                result=ValidationResult.FAIL_HARD,
                level=self.level,
                message="Nebyla nalezena žádná ověřitelná tvrzení",
                details={"claims_count": 0},
                remediation_suggestion="Upravte dotaz nebo snižte kritéria pro generování tvrzení",
            )

        # Kontrola citací pro každé tvrzení
        undercited_claims = []
        for claim in claims:
            citation_count = len(claim.get("citations", []))
            if citation_count < self.min_citations_per_claim:
                undercited_claims.append(
                    {"claim": claim.get("text", "")[:100], "citation_count": citation_count}
                )

        if undercited_claims:
            return ValidationGateResult(
                gate_name=self.name,
                result=ValidationResult.FAIL_HARD,
                level=self.level,
                message=f"Některá tvrzení nemají dostatek citací (minimum {self.min_citations_per_claim})",
                details={"undercited_claims": undercited_claims},
                remediation_suggestion="Zvyšte hloubku vyhledávání nebo snižte citační požadavky",
            )

        return ValidationGateResult(
            gate_name=self.name,
            result=ValidationResult.PASS,
            level=self.level,
            message="Evidence binding úspěšný",
            details={"claims_count": len(claims), "min_citations": self.min_citations_per_claim},
            remediation_suggestion="",
        )


class QualityValidationGate(ValidationGate):
    """Validace kvality finálního výstupu"""

    def __init__(self, min_confidence: float = 0.6):
        super().__init__("quality_validation", ValidationLevel.ERROR)
        self.min_confidence = min_confidence

    async def validate(self, context: dict[str, Any]) -> ValidationGateResult:
        overall_confidence = context.get("overall_confidence", 0.0)

        if overall_confidence < self.min_confidence:
            return ValidationGateResult(
                gate_name=self.name,
                result=ValidationResult.FAIL_WARN,
                level=ValidationLevel.WARNING,
                message=f"Nízká celková důvěryhodnost ({overall_confidence:.2f} < {self.min_confidence})",
                details={"confidence": overall_confidence, "threshold": self.min_confidence},
                remediation_suggestion="Zvyšte hloubku výzkumu nebo zpřesněte dotaz",
            )

        # Kontrola flag claims
        flagged_claims = context.get("flagged_claims", [])
        if len(flagged_claims) > 3:
            return ValidationGateResult(
                gate_name=self.name,
                result=ValidationResult.FAIL_WARN,
                level=ValidationLevel.WARNING,
                message=f"Vysoký počet problematických tvrzení ({len(flagged_claims)})",
                details={"flagged_count": len(flagged_claims)},
                remediation_suggestion="Zvyšte standard evidence nebo zpřesněte dotaz",
            )

        return ValidationGateResult(
            gate_name=self.name,
            result=ValidationResult.PASS,
            level=self.level,
            message="Kvalita výstupu je přijatelná",
            details={"confidence": overall_confidence, "flagged_claims": len(flagged_claims)},
            remediation_suggestion="",
        )


class ValidationGateManager:
    """Správce validačních bran"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.gates: list[ValidationGate] = []
        self._setup_default_gates()

    def _setup_default_gates(self):
        """Nastavení výchozích validačních bran"""
        validation_config = self.config.get("validation_gates", {})

        # Query validation
        self.gates.append(QueryValidationGate())

        # Retrieval validation
        min_docs = validation_config.get("min_documents", 5)
        self.gates.append(RetrievalValidationGate(min_docs))

        # Evidence validation
        min_citations = validation_config.get("min_citations_per_claim", 2)
        self.gates.append(EvidenceValidationGate(min_citations))

        # Quality validation
        min_confidence = validation_config.get("min_confidence", 0.6)
        self.gates.append(QualityValidationGate(min_confidence))

    def add_gate(self, gate: ValidationGate):
        """Přidání vlastní validační brány"""
        self.gates.append(gate)

    async def validate_all(
        self, context: dict[str, Any]
    ) -> tuple[bool, list[ValidationGateResult]]:
        """Spuštění všech validačních bran"""
        results = []
        has_critical_failure = False

        for gate in self.gates:
            try:
                result = await gate.validate(context)
                results.append(result)

                if result.result == ValidationResult.FAIL_HARD:
                    has_critical_failure = True
                    logger.error(f"Validation gate {gate.name} failed: {result.message}")
                elif result.result == ValidationResult.FAIL_WARN:
                    logger.warning(f"Validation gate {gate.name} warning: {result.message}")
                else:
                    logger.info(f"Validation gate {gate.name} passed")

            except Exception as e:
                logger.error(f"Error in validation gate {gate.name}: {e}")
                results.append(
                    ValidationGateResult(
                        gate_name=gate.name,
                        result=ValidationResult.FAIL_HARD,
                        level=ValidationLevel.CRITICAL,
                        message=f"Chyba při validaci: {e!s}",
                        details={"error": str(e)},
                        remediation_suggestion="Opravte konfiguraci nebo kontaktujte podporu",
                    )
                )
                has_critical_failure = True

        return not has_critical_failure, results

    def get_remediation_plan(self, results: list[ValidationGateResult]) -> list[str]:
        """Vytvoření plánu nápravy na základě výsledků validace"""
        remediation_steps = []

        for result in results:
            if result.result in [ValidationResult.FAIL_HARD, ValidationResult.FAIL_WARN]:
                if result.remediation_suggestion:
                    remediation_steps.append(f"{result.gate_name}: {result.remediation_suggestion}")

        return remediation_steps
