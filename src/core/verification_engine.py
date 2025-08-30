#!/usr/bin/env python3
"""Verification Engine pro Deep Research Tool
Nezávislá verifikace claims jiným modelem/promptem

Author: Senior IT Specialist
"""

from dataclasses import dataclass
from datetime import datetime
import json
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class VerificationResult:
    """Výsledek verifikace pro claim"""

    claim_id: str
    original_text: str
    verification_status: str  # "verified", "flagged", "uncertain"
    confidence: float
    reasoning: str
    evidence_assessment: dict[str, Any]
    recommendations: list[str]


class VerificationEngine:
    """Engine pro nezávislou verifikaci claims"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.verification_config = (
            config.get("workflow", {}).get("phases", {}).get("verification", {})
        )
        self.m1_config = config.get("m1_optimization", {})

        # Verification parametry
        self.verification_threshold = self.verification_config.get("verification_threshold", 0.8)
        self.independent_model = self.verification_config.get("independent_model", "qwen2.5:7b")

        self.logger = structlog.get_logger(__name__)

    async def initialize(self):
        """Inicializace verification enginu"""
        self.logger.info("Inicializace verification enginu")

        # Import Ollama agenta pro verifikaci
        from ..core.ollama_agent import OllamaResearchAgent

        self.ollama_agent = OllamaResearchAgent(self.config)

        # Nezávislý model pro verifikaci
        self.verification_model = (
            self.m1_config.get("ollama", {}).get("models", {}).get("verification", "qwen2.5:7b")
        )

        self.logger.info(
            "Verification engine inicializován",
            model=self.verification_model,
            threshold=self.verification_threshold,
        )

    async def verify_claims(
        self,
        claims: list[dict[str, Any]],
        evidence_bindings: dict[str, list[dict[str, Any]]],
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Hlavní metoda pro verifikaci claims"""
        if threshold is None:
            threshold = self.verification_threshold

        self.logger.info(
            "Spouštím verifikaci claims", claims_count=len(claims), threshold=threshold
        )

        verification_results = []

        # Verifikace každého claim samostatně
        for claim in claims:
            claim_id = claim.get("id", "")
            evidence_list = evidence_bindings.get(claim_id, [])

            verification_result = await self._verify_single_claim(claim, evidence_list, threshold)
            verification_results.append(verification_result)

        # Klasifikace výsledků
        verified_claims = []
        flagged_claims = []
        uncertain_claims = []

        for result in verification_results:
            if result.verification_status == "verified":
                verified_claims.append(self._verification_result_to_dict(result))
            elif result.verification_status == "flagged":
                flagged_claims.append(self._verification_result_to_dict(result))
            else:
                uncertain_claims.append(self._verification_result_to_dict(result))

        # Celková confidence
        overall_confidence = self._calculate_verification_confidence(verification_results)

        # Reasoning summary
        reasoning_summary = self._create_reasoning_summary(verification_results)

        return {
            "verified_claims": verified_claims,
            "flagged_claims": flagged_claims,
            "uncertain_claims": uncertain_claims,
            "confidence": overall_confidence,
            "reasoning": reasoning_summary,
            "verification_metadata": {
                "total_claims": len(claims),
                "verified_count": len(verified_claims),
                "flagged_count": len(flagged_claims),
                "uncertain_count": len(uncertain_claims),
                "threshold_used": threshold,
                "model_used": self.verification_model,
            },
        }

    async def _verify_single_claim(
        self, claim: dict[str, Any], evidence_list: list[dict[str, Any]], threshold: float
    ) -> VerificationResult:
        """Verifikace jednotlivého claim"""
        claim_id = claim.get("id", "")
        claim_text = claim.get("text", "")
        original_confidence = claim.get("confidence", 0.0)

        self.logger.info("Verifikuji claim", claim_id=claim_id)

        # Příprava evidence pro analýzu
        evidence_text = ""
        for i, evidence in enumerate(evidence_list):
            evidence_text += f"""
Evidence {i+1}:
Source: {evidence.get('citation', '')}
Passage: {evidence.get('passage', '')}
Confidence: {evidence.get('confidence', 0.0)}
---
"""

        # Verification prompt
        verification_prompt = f"""
Jako nezávislý expert na ověřování faktických tvrzení, proveď kritickou analýzu následujícího claim a jeho důkazů.

CLAIM K OVĚŘENÍ: {claim_text}
PŮVODNÍ CONFIDENCE: {original_confidence}

DOSTUPNÉ DŮKAZY:
{evidence_text}

ÚKOLY:
1. Ověř faktickou správnost claim
2. Zhodnoť kvalitu a spolehlivost důkazů
3. Identifikuj případné nesrovnalosti nebo slabá místa
4. Poskytni nezávislé hodnocení confidence
5. Doporuč případné další kroky

KRITÉRIA HODNOCENÍ:
- Faktická přesnost claim
- Kvalita a relevance důkazů
- Nezávislost zdrojů
- Konzistence informací
- Dostatečnost evidence

Odpověz ve formátu JSON:
{{
  "verification_status": "verified|flagged|uncertain",
  "confidence": 0.85,
  "reasoning": "Detailní odůvodnění hodnocení...",
  "evidence_assessment": {{
    "quality_score": 0.8,
    "independence_score": 0.9,
    "relevance_score": 0.85,
    "consistency_score": 0.9,
    "coverage_score": 0.7
  }},
  "identified_issues": ["seznam případných problémů"],
  "recommendations": ["doporučení pro zlepšení"]
}}
"""

        try:
            response = await self.ollama_agent.generate_response(
                verification_prompt, model=self.verification_model, max_tokens=800
            )

            # Parsování JSON odpovědi
            verification_data = json.loads(response)

            # Určení verification status na základě confidence a threshold
            verification_confidence = verification_data.get("confidence", 0.0)

            if verification_confidence >= threshold:
                status = "verified"
            elif verification_confidence >= threshold * 0.6:
                status = "uncertain"
            else:
                status = "flagged"

            # Override pokud jsou explicitní issues
            identified_issues = verification_data.get("identified_issues", [])
            if identified_issues:
                if verification_confidence < threshold * 0.8:
                    status = "flagged"

            return VerificationResult(
                claim_id=claim_id,
                original_text=claim_text,
                verification_status=status,
                confidence=verification_confidence,
                reasoning=verification_data.get("reasoning", ""),
                evidence_assessment=verification_data.get("evidence_assessment", {}),
                recommendations=verification_data.get("recommendations", []),
            )

        except Exception as e:
            self.logger.error("Chyba při verifikaci claim", claim_id=claim_id, error=str(e))

            # Fallback verifikace
            return self._create_fallback_verification(claim_id, claim_text, threshold)

    def _create_fallback_verification(
        self, claim_id: str, claim_text: str, threshold: float
    ) -> VerificationResult:
        """Fallback verifikace při chybě"""
        return VerificationResult(
            claim_id=claim_id,
            original_text=claim_text,
            verification_status="uncertain",
            confidence=0.5,
            reasoning="Verifikace selhala, požaduje manuální kontrolu",
            evidence_assessment={
                "quality_score": 0.5,
                "independence_score": 0.5,
                "relevance_score": 0.5,
                "consistency_score": 0.5,
                "coverage_score": 0.5,
            },
            recommendations=["Manuální verifikace požadována"],
        )

    def _calculate_verification_confidence(
        self, verification_results: list[VerificationResult]
    ) -> float:
        """Výpočet celkové verification confidence"""
        if not verification_results:
            return 0.0

        # Vážený průměr podle verification status
        total_weighted_confidence = 0.0
        total_weight = 0.0

        for result in verification_results:
            # Váha podle verification status
            if result.verification_status == "verified":
                weight = 1.0
            elif result.verification_status == "uncertain":
                weight = 0.5
            else:  # flagged
                weight = 0.1

            total_weighted_confidence += result.confidence * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return total_weighted_confidence / total_weight

    def _create_reasoning_summary(
        self, verification_results: list[VerificationResult]
    ) -> list[dict[str, Any]]:
        """Vytvoření shrnutí reasoning"""
        reasoning_summary = []

        for result in verification_results:
            summary_item = {
                "claim_id": result.claim_id,
                "verification_status": result.verification_status,
                "confidence": result.confidence,
                "key_reasoning": (
                    result.reasoning[:200] + "..."
                    if len(result.reasoning) > 200
                    else result.reasoning
                ),
                "evidence_quality": result.evidence_assessment.get("quality_score", 0.0),
                "main_recommendations": result.recommendations[:2],  # Top 2 recommendations
            }
            reasoning_summary.append(summary_item)

        return reasoning_summary

    def _verification_result_to_dict(self, result: VerificationResult) -> dict[str, Any]:
        """Konverze VerificationResult na dictionary"""
        return {
            "id": result.claim_id,
            "text": result.original_text,
            "verification_status": result.verification_status,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "evidence_assessment": result.evidence_assessment,
            "recommendations": result.recommendations,
            "verified_at": datetime.now().isoformat(),
        }

    async def cross_validate_claims(
        self, claims: list[dict[str, Any]], evidence_bindings: dict[str, list[dict[str, Any]]]
    ) -> dict[str, Any]:
        """Cross-validace claims mezi sebou"""
        self.logger.info("Spouštím cross-validaci claims")

        # Hledání konfliktních nebo podporujících claims
        claim_relationships = await self._analyze_claim_relationships(claims)

        # Identifikace problematických claims
        consistency_issues = await self._identify_consistency_issues(claims, claim_relationships)

        # Doporučení pro rozřešení konfliktů
        resolution_recommendations = await self._generate_resolution_recommendations(
            claims, consistency_issues
        )

        return {
            "claim_relationships": claim_relationships,
            "consistency_issues": consistency_issues,
            "resolution_recommendations": resolution_recommendations,
            "cross_validation_summary": {
                "total_claims": len(claims),
                "conflicting_pairs": len(
                    [r for r in claim_relationships if r["relationship"] == "conflicting"]
                ),
                "supporting_pairs": len(
                    [r for r in claim_relationships if r["relationship"] == "supporting"]
                ),
                "consistency_score": self._calculate_consistency_score(
                    consistency_issues, len(claims)
                ),
            },
        }

    async def _analyze_claim_relationships(
        self, claims: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Analýza vztahů mezi claims"""
        relationships = []

        # Porovnání každého páru claims
        for i, claim1 in enumerate(claims):
            for j, claim2 in enumerate(claims[i + 1 :], i + 1):

                relationship_prompt = f"""
Analyzuj vztah mezi následujícími dvěma claims:

CLAIM 1: {claim1.get('text', '')}
CLAIM 2: {claim2.get('text', '')}

Určí typ vztahu:
- supporting: claims se vzájemně podporují
- conflicting: claims si odporují
- independent: claims jsou nezávislé
- related: claims jsou příbuzné ale nekonfliktní

Odpověz ve formátu JSON:
{{
  "relationship": "supporting|conflicting|independent|related",
  "confidence": 0.85,
  "explanation": "Stručné vysvětlení vztahu"
}}
"""

                try:
                    response = await self.ollama_agent.generate_response(
                        relationship_prompt, model=self.verification_model, max_tokens=200
                    )

                    relationship_data = json.loads(response)

                    # Přidání pouze významných vztahů
                    if relationship_data.get("relationship") != "independent":
                        relationships.append(
                            {
                                "claim1_id": claim1.get("id"),
                                "claim2_id": claim2.get("id"),
                                "relationship": relationship_data.get("relationship"),
                                "confidence": relationship_data.get("confidence", 0.0),
                                "explanation": relationship_data.get("explanation", ""),
                            }
                        )

                except Exception as e:
                    self.logger.error("Chyba při analýze vztahu claims", error=str(e))
                    continue

        return relationships

    async def _identify_consistency_issues(
        self, claims: list[dict[str, Any]], relationships: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Identifikace problémů s konzistencí"""
        issues = []

        # Hledání konfliktních claims
        conflicting_pairs = [r for r in relationships if r["relationship"] == "conflicting"]

        for conflict in conflicting_pairs:
            claim1 = next((c for c in claims if c.get("id") == conflict["claim1_id"]), None)
            claim2 = next((c for c in claims if c.get("id") == conflict["claim2_id"]), None)

            if claim1 and claim2:
                issue = {
                    "type": "conflicting_claims",
                    "severity": "high" if conflict["confidence"] > 0.8 else "medium",
                    "claim1": {
                        "id": claim1.get("id"),
                        "text": claim1.get("text"),
                        "confidence": claim1.get("confidence", 0.0),
                    },
                    "claim2": {
                        "id": claim2.get("id"),
                        "text": claim2.get("text"),
                        "confidence": claim2.get("confidence", 0.0),
                    },
                    "conflict_explanation": conflict["explanation"],
                    "recommended_action": self._recommend_conflict_resolution(claim1, claim2),
                }
                issues.append(issue)

        return issues

    def _recommend_conflict_resolution(self, claim1: dict[str, Any], claim2: dict[str, Any]) -> str:
        """Doporučení pro vyřešení konfliktu"""
        conf1 = claim1.get("confidence", 0.0)
        conf2 = claim2.get("confidence", 0.0)

        if abs(conf1 - conf2) > 0.3:
            if conf1 > conf2:
                return f"Upřednostnit claim 1 (vyšší confidence: {conf1} vs {conf2})"
            return f"Upřednostnit claim 2 (vyšší confidence: {conf2} vs {conf1})"
        return "Vyžaduje manuální rozhodnutí - podobná confidence"

    async def _generate_resolution_recommendations(
        self, claims: list[dict[str, Any]], issues: list[dict[str, Any]]
    ) -> list[str]:
        """Generování doporučení pro rozřešení problémů"""
        recommendations = []

        if not issues:
            recommendations.append("Žádné významné konzistenční problémy nebyly identifikovány")
            return recommendations

        high_severity_issues = [i for i in issues if i.get("severity") == "high"]

        if high_severity_issues:
            recommendations.append(
                f"Kritické: {len(high_severity_issues)} konfliktních claims vyžaduje okamžité řešení"
            )

            for issue in high_severity_issues[:3]:  # Top 3 kritické issues
                recommendations.append(
                    f"- {issue.get('recommended_action', 'Vyžaduje manuální kontrolu')}"
                )

        medium_severity_issues = [i for i in issues if i.get("severity") == "medium"]
        if medium_severity_issues:
            recommendations.append(
                f"Střední priorita: {len(medium_severity_issues)} claims vyžaduje dodatečnou verifikaci"
            )

        return recommendations

    def _calculate_consistency_score(
        self, issues: list[dict[str, Any]], total_claims: int
    ) -> float:
        """Výpočet skóre konzistence"""
        if total_claims == 0:
            return 1.0

        high_issues = len([i for i in issues if i.get("severity") == "high"])
        medium_issues = len([i for i in issues if i.get("severity") == "medium"])

        # Penalty za issues
        penalty = (high_issues * 0.2 + medium_issues * 0.1) / total_claims

        return max(1.0 - penalty, 0.0)
