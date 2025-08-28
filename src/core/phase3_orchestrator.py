#!/usr/bin/env python3
"""
FÁZE 3 Integration System
Integruje template synthesis, adversarial verification a specialized connectors

Author: Senior Python/MLOps Agent
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import time
import json
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# Import FÁZE 3 components
from src.synthesis.template_synthesis import (
    create_template_synthesizer, Claim, CitationInfo
)
from src.verify.adversarial_verification import (
    create_adversarial_verification_system, RelationshipType, ConflictSet
)
from src.connectors.enhanced_specialized_connectors import (
    create_specialized_connector_orchestrator, SourceResult
)


@dataclass
class Phase3Result:
    """Výsledek FÁZE 3 processing"""
    synthesized_claims: List[Claim]
    verified_claims: List[Dict[str, Any]]
    claim_relationships: List[Dict[str, Any]]
    conflict_sets: List[ConflictSet]
    disagreement_coverage: List[Dict[str, Any]]
    specialized_sources: Dict[str, List[SourceResult]]
    temporal_diffs: List[Dict[str, Any]]
    processing_metadata: Dict[str, Any]


class Phase3Orchestrator:
    """Orchestrátor pro celou FÁZI 3"""

    def __init__(self, config: Dict[str, Any], llm_client=None):
        self.config = config
        self.llm_client = llm_client

        # Initialize components
        self.template_synthesizer = create_template_synthesizer(config, llm_client)
        self.adversarial_verifier = create_adversarial_verification_system(
            config, None, llm_client  # No retrieval system for mock
        )
        self.specialized_connectors = create_specialized_connector_orchestrator(
            config.get("specialized_connectors", {})
        )

        self.orchestration_stats = {
            "phase3_runs": 0,
            "avg_processing_time": 0.0,
            "total_claims_processed": 0,
            "total_conflicts_detected": 0
        }

    async def initialize(self):
        """Inicializace orchestrátoru"""
        try:
            await self.specialized_connectors.initialize()
            logger.info("Phase 3 orchestrator initialized successfully")
        except Exception as e:
            logger.warning(f"Specialized connectors initialization failed: {e}")

    async def process_research_query(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        template_name: Optional[str] = None,
        include_specialized_sources: bool = True,
        include_temporal_analysis: bool = False
    ) -> Phase3Result:
        """
        Kompletní FÁZE 3 processing
        """
        start_time = time.time()

        logger.info(f"Starting Phase 3 processing for query: {query}")

        processing_metadata = {
            "query": query,
            "input_documents": len(documents),
            "template_name": template_name,
            "specialized_sources_enabled": include_specialized_sources,
            "temporal_analysis_enabled": include_temporal_analysis,
            "start_time": datetime.now().isoformat()
        }

        # 1. Template-driven synthesis
        print("🔄 Phase 3.1: Template-driven synthesis...")
        synthesis_start = time.time()

        synthesized_claims, synthesis_metadata = await self.template_synthesizer.synthesize_with_template(
            query=query,
            documents=documents,
            template_name=template_name,
            min_claims=self.config.get("synthesis", {}).get("min_claims", 3),
            max_claims=self.config.get("synthesis", {}).get("max_claims", 10)
        )

        synthesis_time = time.time() - synthesis_start
        processing_metadata["synthesis"] = synthesis_metadata
        processing_metadata["synthesis_time"] = synthesis_time

        print(f"✅ Synthesis completed: {len(synthesized_claims)} claims in {synthesis_time:.2f}s")

        # 2. Specialized source expansion (optional)
        specialized_sources = {}
        if include_specialized_sources:
            print("🔄 Phase 3.3: Specialized source expansion...")
            expansion_start = time.time()

            try:
                specialized_sources = await self.specialized_connectors.search_all_sources(
                    query, max_results_per_source=5
                )
                expansion_time = time.time() - expansion_start
                processing_metadata["source_expansion_time"] = expansion_time

                total_specialized = sum(len(results) for results in specialized_sources.values())
                print(f"✅ Source expansion completed: {total_specialized} documents in {expansion_time:.2f}s")

            except Exception as e:
                logger.warning(f"Specialized source expansion failed: {e}")
                processing_metadata["source_expansion_error"] = str(e)

        # 3. Adversarial verification
        print("🔄 Phase 3.2: Adversarial verification...")
        verification_start = time.time()

        # Convert claims to dict format for verification
        claims_for_verification = [
            {
                "claim_id": claim.claim_id,
                "text": claim.text,
                "confidence": claim.confidence,
                "citations": [asdict(citation) for citation in claim.citations]
            }
            for claim in synthesized_claims
        ]

        verification_result = await self.adversarial_verifier.perform_adversarial_verification(
            claims_for_verification, documents
        )

        verification_time = time.time() - verification_start
        processing_metadata["verification"] = verification_result.get("verification_report", {})
        processing_metadata["verification_time"] = verification_time

        verified_claims = verification_result.get("verified_claims", [])
        claim_relationships = verification_result.get("claim_relationships", [])
        conflict_sets = verification_result.get("conflict_sets", [])
        disagreement_coverage = verification_result.get("disagreement_coverage", [])

        print(f"✅ Verification completed: {len(conflict_sets)} conflicts detected in {verification_time:.2f}s")

        # 4. Temporal analysis (optional)
        temporal_diffs = []
        if include_temporal_analysis and specialized_sources:
            print("🔄 Phase 3.3: Temporal diff analysis...")
            temporal_start = time.time()

            temporal_diffs = await self._perform_temporal_analysis(specialized_sources)

            temporal_time = time.time() - temporal_start
            processing_metadata["temporal_analysis_time"] = temporal_time

            print(f"✅ Temporal analysis completed: {len(temporal_diffs)} diffs in {temporal_time:.2f}s")

        # 5. Final integration and quality assessment
        total_time = time.time() - start_time
        processing_metadata.update({
            "total_processing_time": total_time,
            "end_time": datetime.now().isoformat(),
            "quality_metrics": self._calculate_quality_metrics(
                synthesized_claims, verified_claims, conflict_sets
            )
        })

        # Update orchestration stats
        self._update_orchestration_stats(len(synthesized_claims), len(conflict_sets), total_time)

        # Create result
        result = Phase3Result(
            synthesized_claims=synthesized_claims,
            verified_claims=verified_claims,
            claim_relationships=[asdict(rel) for rel in claim_relationships],
            conflict_sets=conflict_sets,
            disagreement_coverage=disagreement_coverage,
            specialized_sources=specialized_sources,
            temporal_diffs=temporal_diffs,
            processing_metadata=processing_metadata
        )

        logger.info(f"Phase 3 processing completed in {total_time:.2f}s: "
                   f"{len(synthesized_claims)} claims, {len(conflict_sets)} conflicts")

        return result

    async def _perform_temporal_analysis(
        self,
        specialized_sources: Dict[str, List[SourceResult]]
    ) -> List[Dict[str, Any]]:
        """Provede temporal analysis na specialized sources"""
        temporal_diffs = []

        # Look for Memento results with temporal diffs
        memento_results = specialized_sources.get("memento", [])

        for result in memento_results:
            diffs = result.metadata.get("temporal_diffs", [])
            for diff in diffs:
                temporal_diffs.append({
                    "url": diff.url if hasattr(diff, 'url') else result.url,
                    "diff_type": diff.diff_type if hasattr(diff, 'diff_type') else "unknown",
                    "changes": diff.changes if hasattr(diff, 'changes') else [],
                    "impact_assessment": diff.impact_assessment if hasattr(diff, 'impact_assessment') else "unknown",
                    "source_result_id": result.id
                })

        return temporal_diffs

    def _calculate_quality_metrics(
        self,
        synthesized_claims: List[Claim],
        verified_claims: List[Dict[str, Any]],
        conflict_sets: List[ConflictSet]
    ) -> Dict[str, Any]:
        """Vypočítá quality metrics pro FÁZI 3"""
        if not synthesized_claims:
            return {"error": "no_claims_synthesized"}

        # Evidence binding quality
        total_citations = sum(len(claim.citations) for claim in synthesized_claims)
        avg_citations_per_claim = total_citations / len(synthesized_claims)

        # Independent sources
        claims_with_independent_sources = sum(
            1 for claim in synthesized_claims if claim.has_independent_sources()
        )
        independent_sources_ratio = claims_with_independent_sources / len(synthesized_claims)

        # Confidence penalties applied
        confidence_penalties = sum(
            claim.get("confidence_penalty", 0) for claim in verified_claims
        )
        avg_confidence_penalty = confidence_penalties / len(verified_claims) if verified_claims else 0

        # Conflict detection effectiveness
        conflict_ratio = len(conflict_sets) / len(synthesized_claims) if synthesized_claims else 0

        return {
            "total_claims": len(synthesized_claims),
            "avg_citations_per_claim": avg_citations_per_claim,
            "independent_sources_ratio": independent_sources_ratio,
            "avg_confidence_penalty": avg_confidence_penalty,
            "conflict_ratio": conflict_ratio,
            "verification_coverage": len(verified_claims) / len(synthesized_claims) if synthesized_claims else 0
        }

    def _update_orchestration_stats(
        self,
        claims_count: int,
        conflicts_count: int,
        elapsed_time: float
    ):
        """Aktualizuje orchestration statistiky"""
        self.orchestration_stats["phase3_runs"] += 1
        self.orchestration_stats["total_claims_processed"] += claims_count
        self.orchestration_stats["total_conflicts_detected"] += conflicts_count

        # Exponential moving average
        alpha = 0.1
        if self.orchestration_stats["avg_processing_time"] == 0:
            self.orchestration_stats["avg_processing_time"] = elapsed_time
        else:
            self.orchestration_stats["avg_processing_time"] = (
                alpha * elapsed_time +
                (1 - alpha) * self.orchestration_stats["avg_processing_time"]
            )

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Získá statistiky všech komponent FÁZE 3"""
        return {
            "orchestration_stats": self.orchestration_stats,
            "synthesis_stats": self.template_synthesizer.get_stats(),
            "verification_stats": self.adversarial_verifier.get_stats(),
            "connector_stats": self.specialized_connectors.get_connector_stats()
        }

    async def close(self):
        """Zavře všechny komponenty"""
        await self.specialized_connectors.close_all()


# Factory funkce
def create_phase3_orchestrator(config: Dict[str, Any], llm_client=None) -> Phase3Orchestrator:
    """Factory funkce pro Phase 3 orchestrator"""
    return Phase3Orchestrator(config, llm_client)


# Export hlavních tříd
__all__ = [
    "Phase3Result",
    "Phase3Orchestrator",
    "create_phase3_orchestrator"
]
