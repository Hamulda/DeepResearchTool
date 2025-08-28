#!/usr/bin/env python3
"""
Adversarial verifikace a konflikt detekce
Counter-evidence sweep a claim graph s support/contradict vztahy

Author: Senior Python/MLOps Agent
"""

import json
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
import asyncio
from collections import defaultdict

from ..synthesis.template_synthesis import Claim, Citation


@dataclass
class ConflictSet:
    """Sada konfliktních claims"""
    conflict_id: str
    claims: List[str]  # claim_ids
    evidence_for: List[Citation]
    evidence_against: List[Citation]
    confidence_penalty: float
    resolution_status: str  # "unresolved", "resolved", "flagged"


@dataclass
class ClaimRelation:
    """Vztah mezi claims"""
    source_claim_id: str
    target_claim_id: str
    relation_type: str  # "supports", "contradicts", "neutral"
    confidence: float
    evidence: List[Citation]


class AdversarialVerifier:
    """Adversarial verifikace s counter-evidence sweep"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.contradiction_threshold = config.get("contradiction_threshold", 0.7)
        self.min_counter_evidence = config.get("min_counter_evidence", 1)
        self.confidence_penalty = config.get("confidence_penalty", 0.3)

    async def verify_claims(self,
                          claims: List[Claim],
                          evidence_contexts: List[Dict[str, Any]]) -> Tuple[List[Claim], List[ConflictSet]]:
        """Hlavní verifikační pipeline"""
        print("🔍 Starting adversarial verification...")

        # 1. Counter-evidence sweep
        updated_claims = await self._counter_evidence_sweep(claims, evidence_contexts)

        # 2. Detect contradictions
        conflict_sets = self._detect_contradictions(updated_claims)

        # 3. Apply confidence penalties
        penalized_claims = self._apply_confidence_penalties(updated_claims, conflict_sets)

        # 4. Update claim graph relations
        claim_relations = self._build_claim_relations(penalized_claims)

        print(f"✅ Verification complete: {len(conflict_sets)} conflicts detected")

        return penalized_claims, conflict_sets

    async def _counter_evidence_sweep(self,
                                    claims: List[Claim],
                                    evidence_contexts: List[Dict[str, Any]]) -> List[Claim]:
        """Hledá counter-evidence pro každý claim"""
        updated_claims = []

        for claim in claims:
            print(f"🔎 Counter-evidence sweep for claim: {claim.claim_id[:8]}...")

            # Hledej kontradikující evidenci
            counter_evidence = self._find_counter_evidence(claim, evidence_contexts)

            # Aktualizuj claim s counter evidence
            updated_claim = Claim(
                claim_id=claim.claim_id,
                text=claim.text,
                citations=claim.citations + counter_evidence,
                confidence=claim.confidence,
                support_count=claim.support_count,
                contradict_count=len(counter_evidence),
                conflict_sets=claim.conflict_sets
            )

            updated_claims.append(updated_claim)

        return updated_claims

    def _find_counter_evidence(self,
                              claim: Claim,
                              evidence_contexts: List[Dict[str, Any]]) -> List[Citation]:
        """Najde kontradikující evidenci pro claim"""
        counter_evidence = []
        claim_words = set(claim.text.lower().split())

        # Klíčová slova pro negaci
        negation_patterns = [
            r'\b(not|no|never|cannot|does not|did not|will not)\b',
            r'\b(however|but|although|despite|contrary)\b',
            r'\b(refutes|disproves|contradicts|opposes)\b',
            r'\b(unlikely|doubtful|questionable|disputed)\b'
        ]

        for context in evidence_contexts:
            content = context.get("content", "")
            doc_id = context.get("doc_id", "unknown")

            # Hledej věty s negačními vzory
            sentences = re.split(r'[.!?]+', content)

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:
                    continue

                # Kontrola negačních vzorů
                has_negation = any(re.search(pattern, sentence.lower())
                                 for pattern in negation_patterns)

                if has_negation:
                    # Kontrola relevance k claim
                    sentence_words = set(sentence.lower().split())
                    overlap = len(claim_words.intersection(sentence_words))
                    relevance = overlap / max(len(claim_words), len(sentence_words))

                    if relevance > 0.3:  # Nižší threshold pro counter-evidence
                        # Najdi char offsety
                        char_start = content.find(sentence)
                        char_end = char_start + len(sentence)

                        if char_start >= 0:
                            citation = Citation(
                                doc_id=doc_id,
                                char_start=char_start,
                                char_end=char_end,
                                source_text=sentence,
                                confidence=relevance,
                                metadata={
                                    "source_type": context.get("source_type", "unknown"),
                                    "timestamp": datetime.now().isoformat(),
                                    "relevance_score": relevance,
                                    "evidence_type": "contradictory",
                                    "negation_patterns": [
                                        pattern for pattern in negation_patterns
                                        if re.search(pattern, sentence.lower())
                                    ]
                                }
                            )
                            counter_evidence.append(citation)

        # Seřaď podle relevance
        counter_evidence.sort(key=lambda x: x.confidence, reverse=True)
        return counter_evidence[:5]  # Max 5 counter-evidence per claim

    def _detect_contradictions(self, claims: List[Claim]) -> List[ConflictSet]:
        """Detekuje kontradikce mezi claims"""
        conflict_sets = []
        processed_pairs = set()

        for i, claim1 in enumerate(claims):
            for j, claim2 in enumerate(claims[i+1:], i+1):
                pair_key = tuple(sorted([claim1.claim_id, claim2.claim_id]))
                if pair_key in processed_pairs:
                    continue

                processed_pairs.add(pair_key)

                # Detekce kontradikce
                contradiction_score = self._calculate_contradiction_score(claim1, claim2)

                if contradiction_score > self.contradiction_threshold:
                    conflict_id = f"conflict_{len(conflict_sets)}"

                    # Sbírej evidenci pro a proti
                    evidence_for_1 = [c for c in claim1.citations
                                    if c.metadata.get("evidence_type") != "contradictory"]
                    evidence_against_1 = [c for c in claim1.citations
                                        if c.metadata.get("evidence_type") == "contradictory"]

                    evidence_for_2 = [c for c in claim2.citations
                                    if c.metadata.get("evidence_type") != "contradictory"]
                    evidence_against_2 = [c for c in claim2.citations
                                        if c.metadata.get("evidence_type") == "contradictory"]

                    conflict_set = ConflictSet(
                        conflict_id=conflict_id,
                        claims=[claim1.claim_id, claim2.claim_id],
                        evidence_for=evidence_for_1 + evidence_for_2,
                        evidence_against=evidence_against_1 + evidence_against_2,
                        confidence_penalty=self.confidence_penalty * contradiction_score,
                        resolution_status="unresolved"
                    )

                    conflict_sets.append(conflict_set)

                    # Aktualizuj claims s conflict set IDs
                    claim1.conflict_sets.append(conflict_id)
                    claim2.conflict_sets.append(conflict_id)

        return conflict_sets

    def _calculate_contradiction_score(self, claim1: Claim, claim2: Claim) -> float:
        """Vypočítá skóre kontradikce mezi dvěma claims"""
        # Semantic overlap
        words1 = set(claim1.text.lower().split())
        words2 = set(claim2.text.lower().split())
        overlap = len(words1.intersection(words2))
        semantic_similarity = overlap / max(len(words1), len(words2))

        # Counter-evidence overlap
        counter1 = claim1.contradict_count
        counter2 = claim2.contradict_count
        counter_factor = min(counter1, counter2) / max(counter1 + counter2, 1)

        # Negation detection
        negation_score = 0
        if semantic_similarity > 0.5:  # Pouze pro sémanticky podobné claims
            # Hledej negační vzory mezi claims
            text1_lower = claim1.text.lower()
            text2_lower = claim2.text.lower()

            negation_indicators = [
                ("increase", "decrease"), ("rise", "fall"), ("improve", "worsen"),
                ("positive", "negative"), ("support", "oppose"), ("confirm", "deny"),
                ("true", "false"), ("valid", "invalid")
            ]

            for pos, neg in negation_indicators:
                if ((pos in text1_lower and neg in text2_lower) or
                    (neg in text1_lower and pos in text2_lower)):
                    negation_score += 0.3

        # Kombinované skóre
        contradiction_score = (semantic_similarity * 0.4 +
                             counter_factor * 0.3 +
                             min(negation_score, 1.0) * 0.3)

        return contradiction_score

    def _apply_confidence_penalties(self,
                                   claims: List[Claim],
                                   conflict_sets: List[ConflictSet]) -> List[Claim]:
        """Aplikuje confidence penalizace na sporné claims"""
        penalty_map = defaultdict(float)

        # Vypočítaj penalties z conflict sets
        for conflict_set in conflict_sets:
            penalty = conflict_set.confidence_penalty
            for claim_id in conflict_set.claims:
                penalty_map[claim_id] += penalty

        # Aplikuj penalties
        penalized_claims = []
        for claim in claims:
            penalty = min(penalty_map[claim.claim_id], 0.8)  # Max 80% penalty
            new_confidence = max(claim.confidence * (1 - penalty), 0.1)  # Min 10% confidence

            penalized_claim = Claim(
                claim_id=claim.claim_id,
                text=claim.text,
                citations=claim.citations,
                confidence=new_confidence,
                support_count=claim.support_count,
                contradict_count=claim.contradict_count,
                conflict_sets=claim.conflict_sets
            )

            penalized_claims.append(penalized_claim)

        return penalized_claims

    def _build_claim_relations(self, claims: List[Claim]) -> List[ClaimRelation]:
        """Buduje vztahy mezi claims"""
        relations = []

        for i, claim1 in enumerate(claims):
            for j, claim2 in enumerate(claims[i+1:], i+1):
                # Určí typ vztahu
                if any(conflict_id in claim2.conflict_sets for conflict_id in claim1.conflict_sets):
                    relation_type = "contradicts"
                    confidence = self._calculate_contradiction_score(claim1, claim2)
                else:
                    # Kontrola support vztahu
                    support_score = self._calculate_support_score(claim1, claim2)
                    if support_score > 0.6:
                        relation_type = "supports"
                        confidence = support_score
                    else:
                        relation_type = "neutral"
                        confidence = 0.5

                if relation_type != "neutral":
                    relation = ClaimRelation(
                        source_claim_id=claim1.claim_id,
                        target_claim_id=claim2.claim_id,
                        relation_type=relation_type,
                        confidence=confidence,
                        evidence=claim1.citations + claim2.citations
                    )
                    relations.append(relation)

        return relations

    def _calculate_support_score(self, claim1: Claim, claim2: Claim) -> float:
        """Vypočítá support skóre mezi claims"""
        # Sdílené zdroje
        sources1 = set(c.doc_id for c in claim1.citations)
        sources2 = set(c.doc_id for c in claim2.citations)
        shared_sources = len(sources1.intersection(sources2))
        source_support = shared_sources / max(len(sources1), len(sources2))

        # Sémantická podobnost (bez negace)
        words1 = set(claim1.text.lower().split())
        words2 = set(claim2.text.lower().split())
        semantic_similarity = len(words1.intersection(words2)) / max(len(words1), len(words2))

        return (source_support * 0.6 + semantic_similarity * 0.4)

    def calculate_disagreement_coverage(self,
                                      claims: List[Claim],
                                      conflict_sets: List[ConflictSet]) -> Dict[str, float]:
        """Vypočítá disagreement coverage metriky"""
        if not claims:
            return {"total_coverage": 0.0, "avg_counter_evidence": 0.0, "conflict_rate": 0.0}

        # Celková disagreement coverage
        total_claims = len(claims)
        claims_with_counter = sum(1 for c in claims if c.contradict_count > 0)
        total_coverage = claims_with_counter / total_claims

        # Průměrný počet counter-evidence
        avg_counter_evidence = sum(c.contradict_count for c in claims) / total_claims

        # Conflict rate
        claims_in_conflicts = len(set(claim_id for cs in conflict_sets for claim_id in cs.claims))
        conflict_rate = claims_in_conflicts / total_claims

        return {
            "total_coverage": total_coverage,
            "avg_counter_evidence": avg_counter_evidence,
            "conflict_rate": conflict_rate,
            "total_conflicts": len(conflict_sets)
        }

    def save_verification_artifacts(self,
                                  claims: List[Claim],
                                  conflict_sets: List[ConflictSet],
                                  output_dir: str) -> Dict[str, str]:
        """Uloží verifikační artefakty"""
        artifacts = {}

        # Conflict sets
        conflict_data = [asdict(cs) for cs in conflict_sets]
        conflict_file = f"{output_dir}/conflict_sets.json"
        with open(conflict_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "conflict_sets": conflict_data
            }, f, indent=2)
        artifacts["conflict_sets"] = conflict_file

        # Disagreement coverage
        coverage_metrics = self.calculate_disagreement_coverage(claims, conflict_sets)
        coverage_file = f"{output_dir}/disagreement_coverage.json"
        with open(coverage_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "metrics": coverage_metrics
            }, f, indent=2)
        artifacts["disagreement_coverage"] = coverage_file

        return artifacts
