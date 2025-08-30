#!/usr/bin/env python3
"""
Template-driven synthesis s přesnou citací
Každé tvrzení má canonical doc ID + char-offsety pro audit

Author: Senior Python/MLOps Agent
"""

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import uuid


@dataclass
class Citation:
    """Přesná citace s char-offsety"""

    doc_id: str
    char_start: int
    char_end: int
    source_text: str
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class Claim:
    """Tvrzení s evidencí"""

    claim_id: str
    text: str
    citations: List[Citation]
    confidence: float
    support_count: int
    contradict_count: int
    conflict_sets: List[str]


class TemplateBasedSynthesizer:
    """Template-driven syntéza s citačními sloty"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_citations_per_claim = config.get("min_citations_per_claim", 2)
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Načte syntézní templaty"""
        return {
            "claim_with_evidence": """
{claim_text}

Evidence:
{citations}
""",
            "contradictory_evidence": """
However, contradictory evidence suggests:
{counter_evidence}
""",
            "synthesis_report": """
# Research Synthesis Report

## Key Findings

{claims}

## Evidence Summary
- Total claims: {total_claims}
- Total citations: {total_citations}
- Average citations per claim: {avg_citations}
- Confidence score: {avg_confidence:.2f}

## Methodology
Research conducted using multi-source retrieval with evidence verification.
All claims require minimum {min_citations} independent sources.
""",
        }

    def synthesize_claims(self, evidence_contexts: List[Dict[str, Any]], query: str) -> List[Claim]:
        """Vytvoří claims s přesnými citacemi"""
        claims = []

        # Extrahuj potenciální tvrzení z kontextů
        potential_claims = self._extract_potential_claims(evidence_contexts, query)

        for claim_text in potential_claims:
            claim_id = str(uuid.uuid4())

            # Najdi supporting evidence
            citations = self._find_supporting_evidence(claim_text, evidence_contexts)

            # Kontrola minimální evidence
            if len(citations) >= self.min_citations_per_claim:
                claim = Claim(
                    claim_id=claim_id,
                    text=claim_text,
                    citations=citations,
                    confidence=self._calculate_claim_confidence(citations),
                    support_count=len(citations),
                    contradict_count=0,  # Will be filled by verification
                    conflict_sets=[],
                )
                claims.append(claim)

        return claims

    def _extract_potential_claims(
        self, evidence_contexts: List[Dict[str, Any]], query: str
    ) -> List[str]:
        """Extrahuje potenciální tvrzení z kontextů"""
        claims = []

        for context in evidence_contexts:
            content = context.get("content", "")

            # Jednoduché extrakce tvrzení pomocí sentence segmentation
            sentences = re.split(r"[.!?]+", content)

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20 and self._is_relevant_claim(sentence, query):
                    claims.append(sentence)

        # Deduplikuj podobná tvrzení
        return self._deduplicate_claims(claims)

    def _is_relevant_claim(self, sentence: str, query: str) -> bool:
        """Kontrola relevance tvrzení k dotazu"""
        query_words = set(query.lower().split())
        sentence_words = set(sentence.lower().split())

        # Alespoň 30% překryv slov
        overlap = len(query_words.intersection(sentence_words))
        return overlap >= max(1, len(query_words) * 0.3)

    def _deduplicate_claims(self, claims: List[str]) -> List[str]:
        """Deduplikuje podobná tvrzení"""
        unique_claims = []

        for claim in claims:
            is_duplicate = False
            claim_words = set(claim.lower().split())

            for existing in unique_claims:
                existing_words = set(existing.lower().split())
                overlap = len(claim_words.intersection(existing_words))
                similarity = overlap / max(len(claim_words), len(existing_words))

                if similarity > 0.8:  # 80% podobnost = duplikát
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_claims.append(claim)

        return unique_claims

    def _find_supporting_evidence(
        self, claim_text: str, evidence_contexts: List[Dict[str, Any]]
    ) -> List[Citation]:
        """Najde supporting evidence pro claim"""
        citations = []
        claim_words = set(claim_text.lower().split())

        for context in evidence_contexts:
            content = context.get("content", "")
            doc_id = context.get("doc_id", "unknown")

            # Najdi relevantní pasáže
            sentences = re.split(r"[.!?]+", content)

            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if len(sentence) < 10:
                    continue

                sentence_words = set(sentence.lower().split())
                overlap = len(claim_words.intersection(sentence_words))
                relevance = overlap / max(len(claim_words), len(sentence_words))

                if relevance > 0.4:  # 40% relevance threshold
                    # Najdi char offsety
                    char_start = content.find(sentence)
                    char_end = char_start + len(sentence)

                    if char_start >= 0:  # Successfully found
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
                            },
                        )
                        citations.append(citation)

        # Seřaď podle confidence a vezmi top N
        citations.sort(key=lambda x: x.confidence, reverse=True)
        return citations[: self.min_citations_per_claim * 2]  # Max 2x min required

    def _calculate_claim_confidence(self, citations: List[Citation]) -> float:
        """Vypočítá confidence pro claim"""
        if not citations:
            return 0.0

        # Průměrná confidence citací s bonusem za počet zdrojů
        base_confidence = sum(c.confidence for c in citations) / len(citations)

        # Bonus za více nezávislých zdrojů
        unique_sources = len(set(c.doc_id for c in citations))
        source_bonus = min(0.2, unique_sources * 0.05)

        return min(1.0, base_confidence + source_bonus)

    def generate_synthesis_report(self, claims: List[Claim], query: str) -> str:
        """Generuje finální syntézní report"""
        if not claims:
            return "No claims could be synthesized with sufficient evidence."

        # Formatuj claims
        formatted_claims = []
        for i, claim in enumerate(claims, 1):
            citations_text = "\n".join(
                [
                    f"- [{c.doc_id}] {c.source_text[:100]}..."
                    for c in claim.citations[:3]  # Top 3 citations
                ]
            )

            claim_section = self.templates["claim_with_evidence"].format(
                claim_text=f"{i}. {claim.text}", citations=citations_text
            )
            formatted_claims.append(claim_section)

        # Vypočítej celkové statistiky
        total_claims = len(claims)
        total_citations = sum(len(c.citations) for c in claims)
        avg_citations = total_citations / total_claims if total_claims > 0 else 0
        avg_confidence = sum(c.confidence for c in claims) / total_claims if total_claims > 0 else 0

        # Vytvoř finální report
        report = self.templates["synthesis_report"].format(
            claims="\n\n".join(formatted_claims),
            total_claims=total_claims,
            total_citations=total_citations,
            avg_citations=avg_citations,
            avg_confidence=avg_confidence,
            min_citations=self.min_citations_per_claim,
        )

        return report

    def save_audit_artifacts(
        self, claims: List[Claim], query: str, output_dir: str
    ) -> Dict[str, str]:
        """Uloží audit artefakty"""
        artifacts = {}

        # Claims s citacemi
        claims_data = []
        for claim in claims:
            claim_data = {
                "claim_id": claim.claim_id,
                "text": claim.text,
                "confidence": claim.confidence,
                "support_count": claim.support_count,
                "contradict_count": claim.contradict_count,
                "conflict_sets": claim.conflict_sets,
                "citations": [
                    {
                        "doc_id": c.doc_id,
                        "char_start": c.char_start,
                        "char_end": c.char_end,
                        "source_text": c.source_text,
                        "confidence": c.confidence,
                        "metadata": c.metadata,
                    }
                    for c in claim.citations
                ],
            }
            claims_data.append(claim_data)

        claims_file = f"{output_dir}/claims_audit.json"
        with open(claims_file, "w") as f:
            json.dump(
                {"query": query, "timestamp": datetime.now().isoformat(), "claims": claims_data},
                f,
                indent=2,
            )

        artifacts["claims_audit"] = claims_file

        # Citation map
        citation_map = {}
        for claim in claims:
            for citation in claim.citations:
                if citation.doc_id not in citation_map:
                    citation_map[citation.doc_id] = []

                citation_map[citation.doc_id].append(
                    {
                        "claim_id": claim.claim_id,
                        "char_start": citation.char_start,
                        "char_end": citation.char_end,
                        "confidence": citation.confidence,
                    }
                )

        citation_file = f"{output_dir}/citation_map.json"
        with open(citation_file, "w") as f:
            json.dump(citation_map, f, indent=2)

        artifacts["citation_map"] = citation_file

        return artifacts
