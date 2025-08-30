#!/usr/bin/env python3
"""
FÁZE 3 Integration Module
Integrace Enhanced Synthesis + Counter-Evidence Detection + Adversarial Verification

Author: Senior Python/MLOps Agent
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import json
import time

from ..synthesis.enhanced_synthesis_engine import EnhancedSynthesisEngine, SynthesisResult
from ..verify.counter_evidence_detector import CounterEvidenceDetector, DisagreementCoverage
from ..verify.adversarial_verification import AdversarialVerificationEngine, AdversarialVerificationResult

logger = logging.getLogger(__name__)


@dataclass
class Phase3ProcessingResult:
    """Výsledek kompletního FÁZE 3 zpracování"""

    # Input data
    query: str
    compressed_content: str
    evidence_passages: List[Dict[str, Any]]

    # Processing results
    synthesis_result: SynthesisResult
    disagreement_coverage: DisagreementCoverage
    adversarial_verification: List[AdversarialVerificationResult]

    # Integration metrics
    processing_time: float
    final_quality_metrics: Dict[str, float]
    evidence_binding_quality: Dict[str, float]

    # Audit trail
    processing_log: List[Dict[str, Any]]


class Phase3Integrator:
    """Integrátor pro všechny FÁZE 3 komponenty"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.phase3_config = config.get("phase3", {})

        # Initialize components
        self.synthesis_engine = EnhancedSynthesisEngine(config)
        self.counter_evidence_detector = CounterEvidenceDetector(config)
        self.adversarial_verifier = AdversarialVerificationEngine(config)

        # Integration settings
        self.integration_config = self.phase3_config.get("integration", {})
        self.enable_parallel_processing = self.integration_config.get("parallel_processing", False)  # Sequential for dependency

        # Quality gates
        self.quality_gates = self.phase3_config.get("quality_gates", {})
        self.min_groundedness = self.quality_gates.get("min_groundedness", 0.8)
        self.max_hallucination_rate = self.quality_gates.get("max_hallucination_rate", 0.1)
        self.min_disagreement_coverage = self.quality_gates.get("min_disagreement_coverage", 0.3)

        # Fail-hard settings
        self.fail_hard_enabled = self.phase3_config.get("fail_hard_on_quality", True)

        # Audit
        self.processing_log = []

    async def initialize(self):
        """Inicializace všech FÁZE 3 komponent"""

        logger.info("Initializing Phase 3 Integration Pipeline...")

        try:
            # Initialize all components
            await asyncio.gather(
                self.synthesis_engine.initialize() if hasattr(self.synthesis_engine, 'initialize') else asyncio.sleep(0),
                # Counter-evidence detector and adversarial verifier don't need async init
                asyncio.sleep(0),
                asyncio.sleep(0)
            )

            logger.info("✅ Phase 3 Integration Pipeline initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Phase 3 pipeline: {e}")
            raise

    async def process_synthesis_and_verification(self,
                                               query: str,
                                               compressed_content: str,
                                               evidence_passages: List[Dict[str, Any]]) -> Phase3ProcessingResult:
        """
        Kompletní FÁZE 3 zpracování: Synthesis → Counter-Evidence → Adversarial Verification

        Args:
            query: Výzkumný dotaz
            compressed_content: Komprimovaný obsah z FÁZE 2
            evidence_passages: Evidence passages s metadata

        Returns:
            Phase3ProcessingResult s complete evidence binding a verification
        """

        start_time = time.time()
        self.processing_log = []

        logger.info(f"Starting Phase 3 processing for query: {query}")

        try:
            # STEP 1: Enhanced Synthesis with Evidence Binding
            synthesis_start = time.time()
            synthesis_result = await self._step1_enhanced_synthesis(query, compressed_content, evidence_passages)
            synthesis_time = time.time() - synthesis_start

            self._log_processing_step("enhanced_synthesis", {
                "citations_created": synthesis_result.citation_count,
                "independent_sources": synthesis_result.independent_sources,
                "verification_score": synthesis_result.verification_score,
                "processing_time": synthesis_time
            })

            # STEP 2: Counter-Evidence Detection
            counter_evidence_start = time.time()
            disagreement_coverage = await self._step2_counter_evidence_detection(synthesis_result, evidence_passages)
            counter_evidence_time = time.time() - counter_evidence_start

            self._log_processing_step("counter_evidence_detection", {
                "claims_with_counter_evidence": disagreement_coverage.claims_with_counter_evidence,
                "disagreement_ratio": disagreement_coverage.disagreement_ratio,
                "quality_counter_evidence": disagreement_coverage.quality_counter_evidence,
                "processing_time": counter_evidence_time
            })

            # STEP 3: Adversarial Verification
            adversarial_start = time.time()
            adversarial_verification = await self._step3_adversarial_verification(synthesis_result, evidence_passages)
            adversarial_time = time.time() - adversarial_start

            avg_robustness = sum(r.overall_robustness for r in adversarial_verification) / len(adversarial_verification) if adversarial_verification else 0

            self._log_processing_step("adversarial_verification", {
                "claims_verified": len(adversarial_verification),
                "average_robustness": avg_robustness,
                "processing_time": adversarial_time
            })

            # STEP 4: Update synthesis with verification results
            updated_synthesis = await self._step4_update_synthesis_with_verification(
                synthesis_result, disagreement_coverage, adversarial_verification
            )

            # STEP 5: Calculate final quality metrics
            final_quality_metrics = self._calculate_final_quality_metrics(
                updated_synthesis, disagreement_coverage, adversarial_verification
            )

            evidence_binding_quality = self._calculate_evidence_binding_quality(updated_synthesis)

            # STEP 6: Validate final quality gates
            await self._validate_final_quality_gates(final_quality_metrics, evidence_binding_quality)

            total_time = time.time() - start_time

            # Create final result
            result = Phase3ProcessingResult(
                query=query,
                compressed_content=compressed_content,
                evidence_passages=evidence_passages,
                synthesis_result=updated_synthesis,
                disagreement_coverage=disagreement_coverage,
                adversarial_verification=adversarial_verification,
                processing_time=total_time,
                final_quality_metrics=final_quality_metrics,
                evidence_binding_quality=evidence_binding_quality,
                processing_log=self.processing_log.copy()
            )

            logger.info(f"Phase 3 processing completed in {total_time:.2f}s")
            logger.info(f"Final groundedness: {final_quality_metrics.get('groundedness', 0):.3f}")
            logger.info(f"Disagreement coverage: {disagreement_coverage.disagreement_ratio:.1%}")

            return result

        except Exception as e:
            logger.error(f"Phase 3 processing failed: {e}")
            raise

    async def _step1_enhanced_synthesis(self,
                                      query: str,
                                      compressed_content: str,
                                      evidence_passages: List[Dict[str, Any]]) -> SynthesisResult:
        """STEP 1: Enhanced synthesis with evidence binding"""

        logger.info("Phase 3 Step 1: Enhanced synthesis with evidence binding")

        return await self.synthesis_engine.synthesize_with_evidence(
            query, compressed_content, evidence_passages
        )

    async def _step2_counter_evidence_detection(self,
                                              synthesis_result: SynthesisResult,
                                              evidence_passages: List[Dict[str, Any]]) -> DisagreementCoverage:
        """STEP 2: Counter-evidence detection"""

        logger.info("Phase 3 Step 2: Counter-evidence detection")

        # Convert evidence bindings to claims format for counter-evidence detection
        claims = []
        for binding in synthesis_result.evidence_bindings:
            claim = {
                "id": binding.claim_id,
                "text": binding.claim_text,
                "supporting_evidence": [slot.doc_id for slot in binding.citation_slots]
            }
            claims.append(claim)

        return await self.counter_evidence_detector.detect_counter_evidence(claims, evidence_passages)

    async def _step3_adversarial_verification(self,
                                            synthesis_result: SynthesisResult,
                                            evidence_passages: List[Dict[str, Any]]) -> List[AdversarialVerificationResult]:
        """STEP 3: Adversarial verification"""

        logger.info("Phase 3 Step 3: Adversarial verification")

        # Convert evidence bindings to claims format for adversarial verification
        claims_with_evidence = []
        for binding in synthesis_result.evidence_bindings:
            claim_data = {
                "id": binding.claim_id,
                "text": binding.claim_text,
                "evidence_strength": binding.evidence_strength,
                "citation_slots": binding.citation_slots
            }
            claims_with_evidence.append(claim_data)

        return await self.adversarial_verifier.verify_claims_adversarially(claims_with_evidence, evidence_passages)

    async def _step4_update_synthesis_with_verification(self,
                                                      synthesis_result: SynthesisResult,
                                                      disagreement_coverage: DisagreementCoverage,
                                                      adversarial_verification: List[AdversarialVerificationResult]) -> SynthesisResult:
        """STEP 4: Update synthesis s verification results"""

        logger.info("Phase 3 Step 4: Updating synthesis with verification results")

        # Create mapping for quick lookup
        disagreement_map = {analysis.claim_id: analysis for analysis in disagreement_coverage.detailed_analysis}
        adversarial_map = {result.claim_id: result for result in adversarial_verification}

        # Update evidence bindings with verification results
        for binding in synthesis_result.evidence_bindings:
            claim_id = binding.claim_id

            # Apply counter-evidence adjustments
            if claim_id in disagreement_map:
                disagreement_analysis = disagreement_map[claim_id]

                # Add contradiction flags
                if disagreement_analysis.counter_evidence:
                    for counter_evidence in disagreement_analysis.counter_evidence:
                        flag = f"Counter-evidence from {counter_evidence.source_doc_id}: {counter_evidence.contradiction_type}"
                        if flag not in binding.contradiction_flags:
                            binding.contradiction_flags.append(flag)

                # Adjust confidence
                binding.confidence_score = max(0.1, binding.confidence_score - disagreement_analysis.confidence_adjustment)

                # Update verification notes
                binding.verification_notes += f" Disagreement analysis: {disagreement_analysis.analysis_notes}"

            # Apply adversarial verification adjustments
            if claim_id in adversarial_map:
                adversarial_result = adversarial_map[claim_id]

                # Adjust confidence based on robustness
                binding.confidence_score = max(0.1, binding.confidence_score - adversarial_result.confidence_adjustment)

                # Update verification notes
                binding.verification_notes += f" Adversarial verification: {adversarial_result.verification_notes}"

                # Update citation slot verification status based on robustness
                for slot in binding.citation_slots:
                    if adversarial_result.overall_robustness >= 0.8:
                        slot.verification_status = "verified"
                    elif adversarial_result.overall_robustness >= 0.6:
                        slot.verification_status = "partially_verified"
                    elif len(adversarial_result.robustness_gaps) > 0:
                        slot.verification_status = "questioned"
                    else:
                        slot.verification_status = "unverified"

        # Recalculate verification score
        total_citations = sum(len(b.citation_slots) for b in synthesis_result.evidence_bindings)
        verified_citations = sum(
            len([s for s in b.citation_slots if s.verification_status in ["verified", "partially_verified"]])
            for b in synthesis_result.evidence_bindings
        )

        synthesis_result.verification_score = verified_citations / total_citations if total_citations > 0 else 0

        return synthesis_result

    def _calculate_final_quality_metrics(self,
                                       synthesis_result: SynthesisResult,
                                       disagreement_coverage: DisagreementCoverage,
                                       adversarial_verification: List[AdversarialVerificationResult]) -> Dict[str, float]:
        """Výpočet finálních quality metrics"""

        metrics = {}

        # Groundedness (citation coverage + verification)
        citation_coverage = 1.0 if synthesis_result.citation_count >= len(synthesis_result.evidence_bindings) * 2 else 0.5
        verification_quality = synthesis_result.verification_score
        metrics["groundedness"] = (citation_coverage + verification_quality) / 2

        # Hallucination rate (based on unverified claims)
        total_claims = len(synthesis_result.evidence_bindings)
        unverified_claims = sum(
            1 for binding in synthesis_result.evidence_bindings
            if binding.confidence_score < 0.5 or len(binding.citation_slots) == 0
        )
        metrics["hallucination_rate"] = unverified_claims / total_claims if total_claims > 0 else 0

        # Disagreement coverage from counter-evidence
        metrics["disagreement_coverage"] = disagreement_coverage.disagreement_ratio

        # Adversarial robustness
        if adversarial_verification:
            avg_robustness = sum(r.overall_robustness for r in adversarial_verification) / len(adversarial_verification)
            metrics["adversarial_robustness"] = avg_robustness
        else:
            metrics["adversarial_robustness"] = 0.0

        # Citation precision (independent sources)
        required_independence = len(synthesis_result.evidence_bindings) * 2  # 2 independent sources per claim
        actual_independence = synthesis_result.independent_sources
        metrics["citation_precision"] = min(actual_independence / required_independence, 1.0) if required_independence > 0 else 0

        # Evidence binding strength
        if synthesis_result.evidence_bindings:
            avg_evidence_strength = sum(b.evidence_strength for b in synthesis_result.evidence_bindings) / len(synthesis_result.evidence_bindings)
            metrics["evidence_binding_strength"] = avg_evidence_strength
        else:
            metrics["evidence_binding_strength"] = 0.0

        # Overall quality score
        metrics["overall_quality_score"] = (
            metrics["groundedness"] * 0.3 +
            (1 - metrics["hallucination_rate"]) * 0.2 +
            metrics["disagreement_coverage"] * 0.2 +
            metrics["adversarial_robustness"] * 0.15 +
            metrics["citation_precision"] * 0.15
        )

        return metrics

    def _calculate_evidence_binding_quality(self, synthesis_result: SynthesisResult) -> Dict[str, float]:
        """Výpočet evidence binding quality metrics"""

        metrics = {}

        if not synthesis_result.evidence_bindings:
            return {"error": "No evidence bindings found"}

        # Per-claim evidence binding metrics
        total_claims = len(synthesis_result.evidence_bindings)

        # Citation completeness (každý claim má ≥2 citace)
        well_cited_claims = sum(1 for b in synthesis_result.evidence_bindings if len(b.citation_slots) >= 2)
        metrics["citation_completeness"] = well_cited_claims / total_claims

        # Verification completeness (citations mají verification status)
        verified_bindings = sum(
            1 for b in synthesis_result.evidence_bindings
            if any(slot.verification_status in ["verified", "partially_verified"] for slot in b.citation_slots)
        )
        metrics["verification_completeness"] = verified_bindings / total_claims

        # Source diversity (independent sources per claim)
        avg_source_diversity = 0
        for binding in synthesis_result.evidence_bindings:
            unique_sources = len(set(slot.doc_id for slot in binding.citation_slots))
            required_sources = max(2, len(binding.citation_slots))
            diversity = min(unique_sources / required_sources, 1.0)
            avg_source_diversity += diversity

        metrics["source_diversity"] = avg_source_diversity / total_claims

        # Char-offset precision (citations mají přesné char offsety)
        precise_citations = sum(
            len([slot for slot in binding.citation_slots if slot.char_start >= 0 and slot.char_end > slot.char_start])
            for binding in synthesis_result.evidence_bindings
        )
        total_citations = sum(len(binding.citation_slots) for binding in synthesis_result.evidence_bindings)
        metrics["char_offset_precision"] = precise_citations / total_citations if total_citations > 0 else 0

        # Evidence strength distribution
        strength_scores = [binding.evidence_strength for binding in synthesis_result.evidence_bindings]
        metrics["avg_evidence_strength"] = sum(strength_scores) / len(strength_scores)
        metrics["min_evidence_strength"] = min(strength_scores)

        return metrics

    async def _validate_final_quality_gates(self,
                                          final_quality_metrics: Dict[str, float],
                                          evidence_binding_quality: Dict[str, float]):
        """Validace finálních quality gates - fail-hard při nesplnění"""

        failed_validations = []

        # Check groundedness threshold
        groundedness = final_quality_metrics.get("groundedness", 0)
        if groundedness < self.min_groundedness:
            failed_validations.append({
                "metric": "groundedness",
                "required": self.min_groundedness,
                "actual": groundedness
            })

        # Check hallucination rate threshold
        hallucination_rate = final_quality_metrics.get("hallucination_rate", 1.0)
        if hallucination_rate > self.max_hallucination_rate:
            failed_validations.append({
                "metric": "hallucination_rate",
                "required": f"<= {self.max_hallucination_rate}",
                "actual": hallucination_rate
            })

        # Check disagreement coverage threshold
        disagreement_coverage = final_quality_metrics.get("disagreement_coverage", 0)
        if disagreement_coverage < self.min_disagreement_coverage:
            failed_validations.append({
                "metric": "disagreement_coverage",
                "required": self.min_disagreement_coverage,
                "actual": disagreement_coverage
            })

        # Check evidence binding quality
        citation_completeness = evidence_binding_quality.get("citation_completeness", 0)
        if citation_completeness < 0.8:  # 80% of claims should have ≥2 citations
            failed_validations.append({
                "metric": "citation_completeness",
                "required": 0.8,
                "actual": citation_completeness
            })

        verification_completeness = evidence_binding_quality.get("verification_completeness", 0)
        if verification_completeness < 0.7:  # 70% of claims should have verified citations
            failed_validations.append({
                "metric": "verification_completeness",
                "required": 0.7,
                "actual": verification_completeness
            })

        # Fail-hard if validations failed
        if failed_validations and self.fail_hard_enabled:
            error_msg = "Phase 3 final quality validation failed:\n"
            for failure in failed_validations:
                error_msg += f"  - {failure['metric']}: {failure['actual']:.3f} (required: {failure['required']})\n"

            logger.error(error_msg)
            raise ValueError(f"Phase 3 quality validation failed: {error_msg}")

        elif failed_validations:
            logger.warning("Phase 3 quality validation failed but fail-hard disabled")
            for failure in failed_validations:
                logger.warning(f"Quality issue: {failure['metric']} = {failure['actual']:.3f} (required: {failure['required']})")

    def _log_processing_step(self, step: str, data: Dict[str, Any]):
        """Logování processing step pro audit"""

        log_entry = {
            "step": step,
            "timestamp": time.time(),
            "data": data
        }

        self.processing_log.append(log_entry)
        logger.info(f"Phase 3 Step {step}: {data}")

    def get_integration_report(self, result: Phase3ProcessingResult) -> Dict[str, Any]:
        """Generování integration report pro audit"""

        report = {
            "phase3_integration_summary": {
                "query": result.query,
                "processing_time": f"{result.processing_time:.2f}s",
                "final_quality_score": f"{result.final_quality_metrics.get('overall_quality_score', 0):.3f}",
                "evidence_bindings": len(result.synthesis_result.evidence_bindings),
                "total_citations": result.synthesis_result.citation_count,
                "independent_sources": result.synthesis_result.independent_sources
            },

            "synthesis_results": {
                "verification_score": result.synthesis_result.verification_score,
                "quality_metrics": result.synthesis_result.quality_metrics,
                "evidence_bindings_count": len(result.synthesis_result.evidence_bindings)
            },

            "counter_evidence_analysis": {
                "total_claims": result.disagreement_coverage.total_claims,
                "claims_with_counter_evidence": result.disagreement_coverage.claims_with_counter_evidence,
                "disagreement_ratio": f"{result.disagreement_coverage.disagreement_ratio:.1%}",
                "quality_counter_evidence": result.disagreement_coverage.quality_counter_evidence,
                "coverage_score": result.disagreement_coverage.coverage_score
            },

            "adversarial_verification_summary": {
                "claims_verified": len(result.adversarial_verification),
                "average_robustness": sum(r.overall_robustness for r in result.adversarial_verification) / len(result.adversarial_verification) if result.adversarial_verification else 0,
                "total_challenges": sum(len(r.challenges_generated) for r in result.adversarial_verification)
            },

            "final_quality_metrics": result.final_quality_metrics,
            "evidence_binding_quality": result.evidence_binding_quality,
            "processing_log": result.processing_log
        }

        return report

    def get_final_synthesis_text(self, result: Phase3ProcessingResult) -> str:
        """Získání finálního synthesis textu"""

        return result.synthesis_result.synthesis_text

    def export_complete_audit_trail(self, result: Phase3ProcessingResult, output_path: str):
        """Export kompletního audit trail do JSON"""

        # Get individual component reports
        synthesis_report = self.synthesis_engine.get_synthesis_report(result.synthesis_result)
        disagreement_report = await self.counter_evidence_detector.generate_disagreement_report(result.disagreement_coverage)
        adversarial_report = await self.adversarial_verifier.generate_verification_report(result.adversarial_verification)

        complete_audit = {
            "phase3_complete_audit": {
                "metadata": {
                    "query": result.query,
                    "timestamp": time.time(),
                    "processing_time": result.processing_time,
                    "evidence_passages_count": len(result.evidence_passages)
                },
                "integration_report": self.get_integration_report(result),
                "component_reports": {
                    "synthesis": synthesis_report,
                    "counter_evidence": disagreement_report,
                    "adversarial_verification": adversarial_report
                },
                "final_synthesis_text": result.synthesis_result.synthesis_text,
                "evidence_binding_details": [
                    {
                        "claim_id": binding.claim_id,
                        "claim_text": binding.claim_text,
                        "citations": [
                            {
                                "slot_id": slot.slot_id,
                                "doc_id": slot.doc_id,
                                "char_range": f"{slot.char_start}-{slot.char_end}",
                                "verification_status": slot.verification_status,
                                "confidence": slot.confidence,
                                "source_text": slot.source_text[:200] + "..." if len(slot.source_text) > 200 else slot.source_text
                            }
                            for slot in binding.citation_slots
                        ],
                        "evidence_strength": binding.evidence_strength,
                        "confidence_score": binding.confidence_score,
                        "contradiction_flags": binding.contradiction_flags,
                        "verification_notes": binding.verification_notes
                    }
                    for binding in result.synthesis_result.evidence_bindings
                ]
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(complete_audit, f, indent=2, ensure_ascii=False)

        logger.info(f"Complete Phase 3 audit trail exported to {output_path}")
