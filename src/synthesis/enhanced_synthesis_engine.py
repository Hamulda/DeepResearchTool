#!/usr/bin/env python3
"""
Enhanced Synthesis Engine
Pokročilá syntéza s explicitními citačními sloty a přesnou referencí (doc ID + char-offset)

Author: Senior Python/MLOps Agent
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import json
import re
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CitationSlot:
    """Explicitní citační slot s přesnou referencí"""
    slot_id: str
    claim_text: str
    doc_id: str
    char_start: int
    char_end: int
    source_text: str
    confidence: float
    evidence_type: str  # primary, supporting, contextual
    verification_status: str  # verified, disputed, unverified


@dataclass
class EvidenceBinding:
    """Vazba mezi tvrzením a důkazy"""
    claim_id: str
    claim_text: str
    citation_slots: List[CitationSlot]
    evidence_strength: float
    contradiction_flags: List[str]
    confidence_score: float
    verification_notes: str


@dataclass
class SynthesisResult:
    """Výsledek syntézy s evidence binding"""
    synthesis_text: str
    evidence_bindings: List[EvidenceBinding]
    citation_count: int
    independent_sources: int
    verification_score: float
    quality_metrics: Dict[str, float]
    audit_trail: List[Dict[str, Any]]


class EnhancedSynthesisEngine:
    """Enhanced synthesis engine s explicitními citacemi a verifikací"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.synthesis_config = config.get("synthesis", {})

        # Citation requirements
        self.min_citations_per_claim = self.synthesis_config.get("min_citations_per_claim", 2)
        self.require_independent_sources = self.synthesis_config.get("require_independent_sources", True)

        # Synthesis templates
        self.citation_templates = self._load_citation_templates()

        # Verification settings
        self.verification_config = self.synthesis_config.get("verification", {})
        self.enable_counter_evidence = self.verification_config.get("enable_counter_evidence", True)
        self.adversarial_verification = self.verification_config.get("adversarial_verification", True)

        # Quality thresholds
        self.quality_thresholds = self.synthesis_config.get("quality_thresholds", {})
        self.min_verification_score = self.quality_thresholds.get("min_verification_score", 0.7)
        self.min_evidence_strength = self.quality_thresholds.get("min_evidence_strength", 0.6)

        # Audit
        self.audit_trail = []

    def _load_citation_templates(self) -> Dict[str, str]:
        """Načtení šablon pro citace"""

        templates = self.synthesis_config.get("citation_templates", {})

        # Default templates
        defaults = {
            "academic": "According to {source_title} ({doc_id}), {claim_text} [Citation: {slot_id}]",
            "evidence_based": "Research evidence shows that {claim_text}, as documented in {source_title} [Ref: {slot_id}]",
            "multi_source": "Multiple sources confirm that {claim_text} ({source_list}) [Citations: {slot_ids}]",
            "disputed": "While {claim_text} is supported by {supporting_sources}, this finding is disputed by {counter_sources} [Mixed evidence: {slot_ids}]"
        }

        return {**defaults, **templates}

    async def synthesize_with_evidence(self,
                                     query: str,
                                     compressed_content: str,
                                     evidence_passages: List[Dict[str, Any]]) -> SynthesisResult:
        """
        Hlavní syntéza s explicitním evidence binding

        Args:
            query: Výzkumný dotaz
            compressed_content: Komprimovaný obsah z FÁZE 2
            evidence_passages: Seznam evidenčních pasáží s metadata

        Returns:
            SynthesisResult s complete evidence binding
        """

        logger.info(f"Starting enhanced synthesis for query: {query}")

        self.audit_trail = []
        start_time = time.time()

        try:
            # STEP 1: Extract claims from compressed content
            claims = await self._extract_claims(compressed_content, query)
            self._log_audit("claim_extraction", {
                "extracted_claims": len(claims),
                "query": query
            })

            # STEP 2: Create evidence bindings for each claim
            evidence_bindings = []
            for claim in claims:
                binding = await self._create_evidence_binding(claim, evidence_passages)
                evidence_bindings.append(binding)

            self._log_audit("evidence_binding", {
                "bindings_created": len(evidence_bindings),
                "total_citations": sum(len(b.citation_slots) for b in evidence_bindings)
            })

            # STEP 3: Counter-evidence sweep (if enabled)
            if self.enable_counter_evidence:
                evidence_bindings = await self._counter_evidence_sweep(evidence_bindings, evidence_passages)
                self._log_audit("counter_evidence_sweep", {
                    "bindings_updated": len(evidence_bindings)
                })

            # STEP 4: Adversarial verification (if enabled)
            if self.adversarial_verification:
                evidence_bindings = await self._adversarial_verification(evidence_bindings)
                self._log_audit("adversarial_verification", {
                    "verifications_completed": len(evidence_bindings)
                })

            # STEP 5: Generate synthesis text with citations
            synthesis_text = await self._generate_synthesis_text(evidence_bindings, query)

            # STEP 6: Calculate quality metrics
            quality_metrics = self._calculate_synthesis_quality(evidence_bindings, synthesis_text)

            # STEP 7: Validate synthesis quality
            await self._validate_synthesis_quality(evidence_bindings, quality_metrics)

            processing_time = time.time() - start_time

            # Create result
            result = SynthesisResult(
                synthesis_text=synthesis_text,
                evidence_bindings=evidence_bindings,
                citation_count=sum(len(b.citation_slots) for b in evidence_bindings),
                independent_sources=self._count_independent_sources(evidence_bindings),
                verification_score=quality_metrics.get("verification_score", 0.0),
                quality_metrics=quality_metrics,
                audit_trail=self.audit_trail.copy()
            )

            logger.info(f"Enhanced synthesis completed in {processing_time:.2f}s")
            logger.info(f"Generated {result.citation_count} citations from {result.independent_sources} independent sources")

            return result

        except Exception as e:
            logger.error(f"Enhanced synthesis failed: {e}")
            raise

    async def _extract_claims(self, content: str, query: str) -> List[Dict[str, Any]]:
        """Extrakce tvrzení z komprimovaného obsahu"""

        # Split content into sentences for claim analysis
        sentences = self._split_into_sentences(content)

        claims = []
        for i, sentence in enumerate(sentences):
            if self._is_claim_sentence(sentence, query):
                claim = {
                    "id": f"claim_{i}",
                    "text": sentence.strip(),
                    "position": i,
                    "confidence": self._estimate_claim_confidence(sentence, query),
                    "evidence_required": self._estimate_evidence_required(sentence)
                }
                claims.append(claim)

        return claims

    def _split_into_sentences(self, text: str) -> List[str]:
        """Rozdělení textu na věty"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _is_claim_sentence(self, sentence: str, query: str) -> bool:
        """Detekce, zda věta obsahuje tvrzení"""

        # Claim indicators
        claim_patterns = [
            r'\b(shows?|indicates?|suggests?|demonstrates?|proves?|reveals?)\b',
            r'\b(evidence|research|study|analysis|data)\s+(shows?|indicates?|suggests?)\b',
            r'\b(according to|based on|research shows)\b',
            r'\b(conclude|find|observe|report)\b'
        ]

        for pattern in claim_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True

        # Query relevance check
        query_terms = set(query.lower().split())
        sentence_terms = set(sentence.lower().split())

        if len(query_terms.intersection(sentence_terms)) >= 2:
            return True

        return False

    def _estimate_claim_confidence(self, sentence: str, query: str) -> float:
        """Odhad confidence pro tvrzení"""

        confidence = 0.5  # Base confidence

        # Strong evidence indicators
        strong_indicators = ['proves', 'demonstrates', 'shows', 'indicates', 'evidence']
        weak_indicators = ['might', 'could', 'possibly', 'perhaps', 'maybe']

        for indicator in strong_indicators:
            if indicator in sentence.lower():
                confidence += 0.2

        for indicator in weak_indicators:
            if indicator in sentence.lower():
                confidence -= 0.2

        # Query relevance boost
        query_terms = set(query.lower().split())
        sentence_terms = set(sentence.lower().split())
        relevance = len(query_terms.intersection(sentence_terms)) / len(query_terms)
        confidence += relevance * 0.3

        return max(0.1, min(1.0, confidence))

    def _estimate_evidence_required(self, sentence: str) -> int:
        """Odhad počtu vyžadovaných důkazů"""

        # Strong claims need more evidence
        if any(word in sentence.lower() for word in ['proves', 'demonstrates', 'conclusively']):
            return max(self.min_citations_per_claim, 3)

        return self.min_citations_per_claim

    async def _create_evidence_binding(self,
                                     claim: Dict[str, Any],
                                     evidence_passages: List[Dict[str, Any]]) -> EvidenceBinding:
        """Vytvoření evidence binding pro tvrzení"""

        claim_text = claim["text"]
        evidence_required = claim["evidence_required"]

        # Find relevant evidence passages
        relevant_passages = self._find_relevant_passages(claim_text, evidence_passages)

        # Create citation slots
        citation_slots = []
        for i, passage in enumerate(relevant_passages[:evidence_required]):
            # Find best text segment in passage
            best_segment = self._find_best_evidence_segment(claim_text, passage["content"])

            if best_segment:
                slot = CitationSlot(
                    slot_id=f"{claim['id']}_cite_{i}",
                    claim_text=claim_text,
                    doc_id=passage.get("doc_id", f"doc_{i}"),
                    char_start=best_segment["start"],
                    char_end=best_segment["end"],
                    source_text=best_segment["text"],
                    confidence=best_segment["confidence"],
                    evidence_type="primary" if i == 0 else "supporting",
                    verification_status="unverified"
                )
                citation_slots.append(slot)

        # Calculate evidence strength
        evidence_strength = self._calculate_evidence_strength(citation_slots, claim)

        binding = EvidenceBinding(
            claim_id=claim["id"],
            claim_text=claim_text,
            citation_slots=citation_slots,
            evidence_strength=evidence_strength,
            contradiction_flags=[],
            confidence_score=claim["confidence"],
            verification_notes=""
        )

        return binding

    def _find_relevant_passages(self,
                              claim_text: str,
                              evidence_passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Nalezení relevantních pasáží pro tvrzení"""

        claim_terms = set(claim_text.lower().split())

        scored_passages = []
        for passage in evidence_passages:
            content = passage.get("content", "")
            content_terms = set(content.lower().split())

            # Calculate relevance score
            intersection = len(claim_terms.intersection(content_terms))
            union = len(claim_terms.union(content_terms))
            jaccard_score = intersection / union if union > 0 else 0

            # Boost for evidence indicators
            evidence_boost = 0
            evidence_indicators = ['evidence', 'research', 'study', 'data', 'analysis']
            for indicator in evidence_indicators:
                if indicator in content.lower():
                    evidence_boost += 0.1

            total_score = jaccard_score + evidence_boost

            if total_score > 0.1:  # Minimum relevance threshold
                scored_passages.append((total_score, passage))

        # Sort by relevance score (highest first)
        scored_passages.sort(key=lambda x: x[0], reverse=True)

        return [passage for score, passage in scored_passages]

    def _find_best_evidence_segment(self,
                                  claim_text: str,
                                  passage_content: str) -> Optional[Dict[str, Any]]:
        """Nalezení nejlepšího evidence segmentu v pasáži"""

        # Split passage into sentences
        sentences = self._split_into_sentences(passage_content)

        best_segment = None
        best_score = 0
        char_offset = 0

        claim_terms = set(claim_text.lower().split())

        for sentence in sentences:
            sentence_terms = set(sentence.lower().split())

            # Calculate similarity to claim
            intersection = len(claim_terms.intersection(sentence_terms))
            similarity = intersection / len(claim_terms) if claim_terms else 0

            # Boost for evidence language
            evidence_boost = 0
            if any(word in sentence.lower() for word in ['shows', 'indicates', 'demonstrates', 'evidence']):
                evidence_boost = 0.3

            total_score = similarity + evidence_boost

            if total_score > best_score:
                best_score = total_score
                start_pos = passage_content.find(sentence, char_offset)

                best_segment = {
                    "text": sentence,
                    "start": start_pos,
                    "end": start_pos + len(sentence),
                    "confidence": total_score
                }

            char_offset += len(sentence) + 1

        return best_segment if best_score > 0.2 else None

    def _calculate_evidence_strength(self,
                                   citation_slots: List[CitationSlot],
                                   claim: Dict[str, Any]) -> float:
        """Výpočet síly důkazů pro tvrzení"""

        if not citation_slots:
            return 0.0

        # Base strength from citation confidence
        avg_confidence = sum(slot.confidence for slot in citation_slots) / len(citation_slots)

        # Boost for multiple independent sources
        unique_docs = len(set(slot.doc_id for slot in citation_slots))
        independence_boost = min(unique_docs / self.min_citations_per_claim, 1.0) * 0.3

        # Boost for primary evidence
        primary_count = sum(1 for slot in citation_slots if slot.evidence_type == "primary")
        primary_boost = min(primary_count / 2, 1.0) * 0.2

        total_strength = avg_confidence + independence_boost + primary_boost

        return min(total_strength, 1.0)

    async def _counter_evidence_sweep(self,
                                    evidence_bindings: List[EvidenceBinding],
                                    evidence_passages: List[Dict[str, Any]]) -> List[EvidenceBinding]:
        """Counter-evidence sweep pro nalezení vyvracejících důkazů"""

        logger.info("Performing counter-evidence sweep...")

        for binding in evidence_bindings:
            claim_text = binding.claim_text

            # Look for contradictory evidence
            counter_evidence = await self._find_counter_evidence(claim_text, evidence_passages)

            if counter_evidence:
                # Add contradiction flags
                for evidence in counter_evidence:
                    flag = f"Contradicted by: {evidence['doc_id']} - {evidence['contradiction_text'][:100]}..."
                    binding.contradiction_flags.append(flag)

                # Adjust confidence based on contradictions
                contradiction_penalty = len(counter_evidence) * 0.2
                binding.confidence_score = max(0.1, binding.confidence_score - contradiction_penalty)

                binding.verification_notes += f" Found {len(counter_evidence)} contradictory evidence(s)."

        return evidence_bindings

    async def _find_counter_evidence(self,
                                   claim_text: str,
                                   evidence_passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Nalezení protikladných důkazů"""

        counter_evidence = []

        # Contradiction indicators
        negation_patterns = [
            r'\b(not|no|never|none|neither|nor)\b',
            r'\b(however|but|although|despite|nevertheless)\b',
            r'\b(contrary|opposite|against|refutes?|disputes?)\b',
            r'\b(disproves?|contradicts?|challenges?)\b'
        ]

        claim_terms = set(claim_text.lower().split())

        for passage in evidence_passages:
            content = passage.get("content", "")

            # Check for negation patterns
            has_negation = any(re.search(pattern, content, re.IGNORECASE) for pattern in negation_patterns)

            if has_negation:
                # Check if content relates to the claim
                content_terms = set(content.lower().split())
                overlap = len(claim_terms.intersection(content_terms))

                if overlap >= 2:  # Minimum term overlap for relevance
                    sentences = self._split_into_sentences(content)

                    for sentence in sentences:
                        if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in negation_patterns):
                            if any(term in sentence.lower() for term in claim_terms):
                                counter_evidence.append({
                                    "doc_id": passage.get("doc_id", "unknown"),
                                    "contradiction_text": sentence,
                                    "confidence": 0.7  # Fixed confidence for counter-evidence
                                })

        return counter_evidence

    async def _adversarial_verification(self,
                                      evidence_bindings: List[EvidenceBinding]) -> List[EvidenceBinding]:
        """Adversarial verification loop"""

        logger.info("Performing adversarial verification...")

        for binding in evidence_bindings:
            # Generate adversarial questions
            adversarial_questions = self._generate_adversarial_questions(binding.claim_text)

            # Check if existing evidence answers adversarial questions
            for question in adversarial_questions:
                answered = self._check_adversarial_question(question, binding.citation_slots)

                if not answered:
                    # Reduce confidence for unanswered adversarial questions
                    binding.confidence_score *= 0.9
                    binding.verification_notes += f" Adversarial question unanswered: {question[:50]}... "

            # Update verification status based on checks
            for slot in binding.citation_slots:
                if binding.confidence_score >= 0.7:
                    slot.verification_status = "verified"
                elif binding.contradiction_flags:
                    slot.verification_status = "disputed"
                else:
                    slot.verification_status = "unverified"

        return evidence_bindings

    def _generate_adversarial_questions(self, claim_text: str) -> List[str]:
        """Generování adversarial otázek pro tvrzení"""

        questions = []

        # Template-based adversarial questions
        templates = [
            f"What evidence contradicts the claim that {claim_text}?",
            f"Under what conditions might {claim_text} not be true?",
            f"What are the limitations of the evidence supporting {claim_text}?",
            f"How reliable are the sources claiming that {claim_text}?",
            f"What alternative explanations exist for {claim_text}?"
        ]

        # Select subset based on claim complexity
        num_questions = min(len(templates), 3)
        questions = templates[:num_questions]

        return questions

    def _check_adversarial_question(self,
                                  question: str,
                                  citation_slots: List[CitationSlot]) -> bool:
        """Kontrola, zda existující evidence odpovídá na adversarial otázku"""

        question_terms = set(question.lower().split())

        for slot in citation_slots:
            source_terms = set(slot.source_text.lower().split())

            # Check for term overlap
            overlap = len(question_terms.intersection(source_terms))
            if overlap >= 3:  # Minimum overlap for relevance
                return True

        return False

    async def _generate_synthesis_text(self,
                                     evidence_bindings: List[EvidenceBinding],
                                     query: str) -> str:
        """Generování synthesis textu s citacemi"""

        if not evidence_bindings:
            return "No evidence found to synthesize response."

        synthesis_parts = []

        # Introduction
        synthesis_parts.append(f"Based on the available evidence, the following findings address the query: {query}")
        synthesis_parts.append("")

        # Process each claim with evidence
        for i, binding in enumerate(evidence_bindings, 1):
            if not binding.citation_slots:
                continue

            # Choose template based on evidence characteristics
            template_key = self._choose_citation_template(binding)
            template = self.citation_templates.get(template_key, self.citation_templates["academic"])

            # Prepare template variables
            source_titles = [f"Source {slot.doc_id}" for slot in binding.citation_slots]
            slot_ids = [slot.slot_id for slot in binding.citation_slots]

            # Format claim with citations
            if binding.contradiction_flags:
                # Use disputed template
                supporting_sources = ", ".join(source_titles[:2])
                counter_sources = "contradictory evidence"
                claim_text = template.format(
                    claim_text=binding.claim_text,
                    supporting_sources=supporting_sources,
                    counter_sources=counter_sources,
                    slot_ids=", ".join(slot_ids)
                )
            else:
                # Use standard template
                claim_text = template.format(
                    source_title=source_titles[0] if source_titles else "Unknown",
                    doc_id=binding.citation_slots[0].doc_id if binding.citation_slots else "unknown",
                    claim_text=binding.claim_text,
                    slot_id=slot_ids[0] if slot_ids else "unknown",
                    source_list=", ".join(source_titles),
                    slot_ids=", ".join(slot_ids)
                )

            synthesis_parts.append(f"{i}. {claim_text}")

            # Add evidence strength note
            strength_note = f"   Evidence strength: {binding.evidence_strength:.2f}"
            if binding.contradiction_flags:
                strength_note += f" (disputed by {len(binding.contradiction_flags)} source(s))"
            synthesis_parts.append(strength_note)
            synthesis_parts.append("")

        return "\n".join(synthesis_parts)

    def _choose_citation_template(self, binding: EvidenceBinding) -> str:
        """Výběr vhodné šablony pro citace"""

        if binding.contradiction_flags:
            return "disputed"
        elif len(binding.citation_slots) > 2:
            return "multi_source"
        elif binding.evidence_strength >= 0.8:
            return "evidence_based"
        else:
            return "academic"

    def _calculate_synthesis_quality(self,
                                   evidence_bindings: List[EvidenceBinding],
                                   synthesis_text: str) -> Dict[str, float]:
        """Výpočet quality metrics pro syntézu"""

        metrics = {}

        # Citation coverage
        total_claims = len(evidence_bindings)
        cited_claims = sum(1 for b in evidence_bindings if b.citation_slots)
        metrics["citation_coverage"] = cited_claims / total_claims if total_claims > 0 else 0

        # Citation density
        total_citations = sum(len(b.citation_slots) for b in evidence_bindings)
        metrics["citation_density"] = total_citations / total_claims if total_claims > 0 else 0

        # Evidence strength
        if evidence_bindings:
            avg_evidence_strength = sum(b.evidence_strength for b in evidence_bindings) / len(evidence_bindings)
            metrics["evidence_strength"] = avg_evidence_strength
        else:
            metrics["evidence_strength"] = 0.0

        # Verification score
        verified_citations = sum(
            len([s for s in b.citation_slots if s.verification_status == "verified"])
            for b in evidence_bindings
        )
        metrics["verification_score"] = verified_citations / total_citations if total_citations > 0 else 0

        # Independence score
        independent_sources = self._count_independent_sources(evidence_bindings)
        metrics["source_independence"] = min(independent_sources / len(evidence_bindings), 1.0) if evidence_bindings else 0

        # Contradiction penalty
        total_contradictions = sum(len(b.contradiction_flags) for b in evidence_bindings)
        contradiction_penalty = min(total_contradictions * 0.1, 0.5)
        metrics["contradiction_penalty"] = contradiction_penalty

        # Overall verification score
        base_score = (
            metrics["citation_coverage"] * 0.3 +
            metrics["evidence_strength"] * 0.3 +
            metrics["verification_score"] * 0.2 +
            metrics["source_independence"] * 0.2
        )
        metrics["overall_verification_score"] = max(0, base_score - contradiction_penalty)

        return metrics

    def _count_independent_sources(self, evidence_bindings: List[EvidenceBinding]) -> int:
        """Počítání nezávislých zdrojů"""

        all_doc_ids = set()
        for binding in evidence_bindings:
            for slot in binding.citation_slots:
                all_doc_ids.add(slot.doc_id)

        return len(all_doc_ids)

    async def _validate_synthesis_quality(self,
                                        evidence_bindings: List[EvidenceBinding],
                                        quality_metrics: Dict[str, float]):
        """Validace kvality syntézy - fail-hard při nesplnění"""

        failed_validations = []

        # Check minimum verification score
        verification_score = quality_metrics.get("overall_verification_score", 0)
        if verification_score < self.min_verification_score:
            failed_validations.append({
                "metric": "verification_score",
                "required": self.min_verification_score,
                "actual": verification_score
            })

        # Check minimum evidence strength
        evidence_strength = quality_metrics.get("evidence_strength", 0)
        if evidence_strength < self.min_evidence_strength:
            failed_validations.append({
                "metric": "evidence_strength",
                "required": self.min_evidence_strength,
                "actual": evidence_strength
            })

        # Check minimum citations per claim
        under_cited_claims = []
        for binding in evidence_bindings:
            if len(binding.citation_slots) < self.min_citations_per_claim:
                under_cited_claims.append(binding.claim_id)

        if under_cited_claims:
            failed_validations.append({
                "metric": "min_citations_per_claim",
                "required": self.min_citations_per_claim,
                "under_cited_claims": under_cited_claims
            })

        # Check source independence (if required)
        if self.require_independent_sources:
            independent_sources = self._count_independent_sources(evidence_bindings)
            required_independence = len(evidence_bindings)

            if independent_sources < required_independence:
                failed_validations.append({
                    "metric": "source_independence",
                    "required": required_independence,
                    "actual": independent_sources
                })

        # Fail-hard if validations failed
        if failed_validations:
            error_msg = "Synthesis quality validation failed:\n"
            for failure in failed_validations:
                if "under_cited_claims" in failure:
                    error_msg += f"  - Claims with insufficient citations: {failure['under_cited_claims']}\n"
                else:
                    error_msg += f"  - {failure['metric']}: {failure['actual']:.3f} < {failure['required']:.3f}\n"

            logger.error(error_msg)

            fail_hard = self.synthesis_config.get("fail_hard_on_quality", True)
            if fail_hard:
                raise ValueError(f"Synthesis quality validation failed: {error_msg}")
            else:
                logger.warning("Synthesis quality validation failed but fail-hard disabled")

    def _log_audit(self, step: str, data: Dict[str, Any]):
        """Logování audit trail"""

        log_entry = {
            "step": step,
            "timestamp": time.time(),
            "data": data
        }

        self.audit_trail.append(log_entry)
        logger.info(f"Synthesis step {step}: {data}")

    def get_synthesis_report(self, result: SynthesisResult) -> Dict[str, Any]:
        """Generování synthesis report pro audit"""

        report = {
            "synthesis_summary": {
                "citation_count": result.citation_count,
                "independent_sources": result.independent_sources,
                "verification_score": f"{result.verification_score:.3f}",
                "evidence_bindings": len(result.evidence_bindings)
            },
            "quality_metrics": result.quality_metrics,
            "evidence_details": [],
            "audit_trail": result.audit_trail
        }

        # Add evidence binding details
        for binding in result.evidence_bindings:
            detail = {
                "claim_id": binding.claim_id,
                "claim_text": binding.claim_text,
                "citation_count": len(binding.citation_slots),
                "evidence_strength": binding.evidence_strength,
                "contradiction_flags": len(binding.contradiction_flags),
                "confidence_score": binding.confidence_score,
                "citations": [
                    {
                        "slot_id": slot.slot_id,
                        "doc_id": slot.doc_id,
                        "char_range": f"{slot.char_start}-{slot.char_end}",
                        "verification_status": slot.verification_status,
                        "confidence": slot.confidence
                    }
                    for slot in binding.citation_slots
                ]
            }
            report["evidence_details"].append(detail)

        return report
