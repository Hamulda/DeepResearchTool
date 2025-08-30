#!/usr/bin/env python3
"""Synthesis Engine s Evidence Binding pro Deep Research Tool
Implementuje syntézu s povinným per-claim evidence binding

Author: Senior IT Specialist
"""

from dataclasses import dataclass, field
import json
import re
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Evidence:
    """Evidence pro claim"""

    source_id: str
    source_url: str
    passage: str
    confidence: float
    citation: str  # WARC:{filename}:{offset}, DOI:{doi}, etc.
    timestamp: str
    offset_start: int | None = None
    offset_end: int | None = None
    verification_status: str = "pending"


@dataclass
class Claim:
    """Jednotlivý claim s evidence"""

    id: str
    text: str
    confidence: float
    evidence_list: list[Evidence] = field(default_factory=list)
    topic: str = ""
    claim_type: str = "factual"  # factual, opinion, prediction

    def has_sufficient_evidence(self, min_evidence: int = 2) -> bool:
        """Kontrola dostatečných důkazů"""
        return len(self.evidence_list) >= min_evidence

    def get_independent_sources(self) -> int:
        """Počet nezávislých zdrojů"""
        sources = set(evidence.source_id for evidence in self.evidence_list)
        return len(sources)


class SynthesisEngine:
    """Engine pro syntézu s evidence binding"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.synthesis_config = config.get("workflow", {}).get("phases", {}).get("synthesis", {})
        self.evidence_config = config.get("evidence_binding", {})
        self.m1_config = config.get("m1_optimization", {})

        # Konfigurace evidence binding
        self.min_evidence_per_claim = self.synthesis_config.get("min_evidence_per_claim", 2)
        self.confidence_threshold = self.synthesis_config.get("confidence_threshold", 0.7)
        self.max_confidence_without_evidence = self.evidence_config.get(
            "max_confidence_without_evidence", 0.3
        )

        # Citation formáty
        self.citation_formats = self.evidence_config.get("citation_formats", {})

        self.logger = structlog.get_logger(__name__)

    async def initialize(self):
        """Inicializace synthesis enginu"""
        self.logger.info("Inicializace synthesis enginu")

        # Import Ollama agenta pro syntézu
        from ..core.ollama_agent import OllamaResearchAgent

        self.ollama_agent = OllamaResearchAgent(self.config)

        # Model pro syntézu
        self.synthesis_model = (
            self.m1_config.get("ollama", {}).get("models", {}).get("synthesis", "llama3.2:8b")
        )

        self.logger.info("Synthesis engine inicializován")

    async def synthesize_with_evidence(
        self,
        query: str,
        documents: list[dict[str, Any]],
        min_evidence_per_claim: int | None = None,
    ) -> dict[str, Any]:
        """Hlavní metoda pro syntézu s evidence binding"""
        if min_evidence_per_claim is None:
            min_evidence_per_claim = self.min_evidence_per_claim

        self.logger.info(
            "Spouštím syntézu s evidence binding",
            query=query,
            documents=len(documents),
            min_evidence=min_evidence_per_claim,
        )

        # Krok 1: Extrakce claims z dokumentů
        extracted_claims = await self._extract_claims_from_documents(query, documents)

        # Krok 2: Evidence binding pro každý claim
        claims_with_evidence = await self._bind_evidence_to_claims(extracted_claims, documents)

        # Krok 3: Filtrování claims bez dostatečných důkazů
        verified_claims = self._filter_claims_by_evidence(
            claims_with_evidence, min_evidence_per_claim
        )

        # Krok 4: Syntéza finálního odpovědi
        final_synthesis = await self._generate_final_synthesis(query, verified_claims)

        # Krok 5: Výpočet celkové confidence
        overall_confidence = self._calculate_overall_confidence(verified_claims)

        # Evidence bindings pro export
        evidence_bindings = self._create_evidence_bindings(verified_claims)

        return {
            "claims": [self._claim_to_dict(claim) for claim in verified_claims],
            "evidence_bindings": evidence_bindings,
            "confidence": overall_confidence,
            "synthesis_text": final_synthesis,
            "metadata": {
                "total_extracted_claims": len(extracted_claims),
                "verified_claims": len(verified_claims),
                "filtered_claims": len(extracted_claims) - len(verified_claims),
                "evidence_requirement": min_evidence_per_claim,
            },
        }

    async def _extract_claims_from_documents(
        self, query: str, documents: list[dict[str, Any]]
    ) -> list[Claim]:
        """Extrakce claims z dokumentů"""
        # Připrava dokumentů pro analýzu
        documents_text = ""
        for i, doc in enumerate(documents[:10]):  # Limit pro kontext
            title = doc.get("title", "")
            content = doc.get("content", "")[:500]  # Truncate
            source = doc.get("source", "")

            documents_text += f"""
Document {i+1} ({source}):
Title: {title}
Content: {content}
---
"""

        extraction_prompt = f"""
Jako expert na analýzu výzkumných dokumentů, extrahuj konkrétní faktické claims (tvrzení) z následujících dokumentů, které se vztahují k dotazu.

Dotaz: {query}

Dokumenty:
{documents_text}

Pro každý claim:
1. Formuluj jej jasně a konkrétně
2. Klasifikuj typ (factual/opinion/prediction)  
3. Identifikuj téma/kategorii
4. Odhodni důvěryhodnost (0.0-1.0)

Odpověz ve formátu JSON:
{{
  "claims": [
    {{
      "text": "Konkrétní factické tvrzení",
      "type": "factual", 
      "topic": "kategorie",
      "confidence": 0.8
    }},
    ...
  ]
}}

Extrauj maximálně 15 nejdůležitějších claims.
"""

        try:
            response = await self.ollama_agent.generate_response(
                extraction_prompt, model=self.synthesis_model, max_tokens=1500
            )

            # Parsování JSON odpovědi
            claims_data = json.loads(response)
            claims_list = claims_data.get("claims", [])

            # Vytvoření Claim objektů
            extracted_claims = []
            for i, claim_data in enumerate(claims_list):
                claim = Claim(
                    id=f"claim_{i+1}",
                    text=claim_data.get("text", ""),
                    confidence=claim_data.get("confidence", 0.5),
                    topic=claim_data.get("topic", ""),
                    claim_type=claim_data.get("type", "factual"),
                )
                extracted_claims.append(claim)

            self.logger.info("Claims extrahovány", count=len(extracted_claims))
            return extracted_claims

        except Exception as e:
            self.logger.error("Chyba při extrakci claims", error=str(e))
            return []

    async def _bind_evidence_to_claims(
        self, claims: list[Claim], documents: list[dict[str, Any]]
    ) -> list[Claim]:
        """Binding evidence k claims"""
        self.logger.info("Provádím evidence binding", claims=len(claims))

        for claim in claims:
            # Hledání evidence pro každý claim
            claim.evidence_list = await self._find_evidence_for_claim(claim, documents)

            # Aktualizace confidence na základě evidence
            if not claim.has_sufficient_evidence(self.min_evidence_per_claim):
                claim.confidence = min(claim.confidence, self.max_confidence_without_evidence)

        return claims

    async def _find_evidence_for_claim(
        self, claim: Claim, documents: list[dict[str, Any]]
    ) -> list[Evidence]:
        """Hledání evidence pro konkrétní claim"""
        evidence_list = []
        claim_keywords = self._extract_keywords(claim.text)

        for doc in documents:
            # Hledání relevantních pasáží
            relevant_passages = self._find_relevant_passages(
                claim.text, doc.get("content", ""), claim_keywords
            )

            for passage, confidence in relevant_passages:
                # Vytvoření citace podle typu zdroje
                citation = self._create_citation(doc)

                evidence = Evidence(
                    source_id=doc.get("id", ""),
                    source_url=doc.get("url", ""),
                    passage=passage,
                    confidence=confidence,
                    citation=citation,
                    timestamp=doc.get("timestamp", ""),
                    offset_start=doc.get("content", "").find(passage),
                    offset_end=doc.get("content", "").find(passage) + len(passage),
                )

                evidence_list.append(evidence)

                # Limit evidence na dokument
                if len([e for e in evidence_list if e.source_id == doc.get("id", "")]) >= 2:
                    break

        # Řazení podle confidence a výběr nejlepších
        evidence_list.sort(key=lambda e: e.confidence, reverse=True)

        # Zajištění nezávislých zdrojů
        independent_evidence = self._ensure_independent_sources(evidence_list)

        return independent_evidence[:4]  # Maximum 4 evidence na claim

    def _find_relevant_passages(
        self, claim_text: str, document_content: str, claim_keywords: list[str]
    ) -> list[tuple[str, float]]:
        """Hledání relevantních pasáží v dokumentu"""
        if not document_content:
            return []

        passages = []
        sentences = re.split(r"[.!?]+", document_content)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Příliš krátké věty
                continue

            # Výpočet relevance na základě keyword overlap
            sentence_lower = sentence.lower()
            claim_lower = claim_text.lower()

            # Keyword matching
            keyword_matches = sum(
                1 for keyword in claim_keywords if keyword.lower() in sentence_lower
            )
            keyword_score = keyword_matches / len(claim_keywords) if claim_keywords else 0

            # Semantic similarity (jednoduchá heuristika)
            common_words = set(claim_lower.split()) & set(sentence_lower.split())
            semantic_score = len(common_words) / max(len(claim_lower.split()), 1)

            # Kombinované skóre
            relevance_score = 0.6 * keyword_score + 0.4 * semantic_score

            if relevance_score > 0.3:  # Threshold pro relevanci
                passages.append((sentence, relevance_score))

        # Řazení podle relevance
        passages.sort(key=lambda x: x[1], reverse=True)

        return passages[:3]  # Top 3 pasáže na dokument

    def _extract_keywords(self, text: str) -> list[str]:
        """Extrakce klíčových slov z textu"""
        # Jednoduché odstranění stop words a extrakce klíčových slov
        stop_words = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "will",
            "with",
            "že",
            "je",
            "se",
            "na",
            "v",
            "do",
            "za",
            "s",
            "o",
            "nebo",
            "ale",
            "když",
            "jak",
        }

        words = re.findall(r"\b\w+\b", text.lower())
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]

        return list(set(keywords))  # Unique keywords

    def _create_citation(self, document: dict[str, Any]) -> str:
        """Vytvoření citace podle typu zdroje"""
        source = document.get("source", "").lower()
        url = document.get("url", "")

        # WARC format
        if "warc" in document.get("metadata", {}):
            warc_info = document["metadata"]["warc"]
            return self.citation_formats.get("warc", "WARC:{filename}:{offset}").format(
                filename=warc_info.get("filename", ""), offset=warc_info.get("offset", "")
            )

        # DOI format
        if "doi" in document.get("metadata", {}):
            doi = document["metadata"]["doi"]
            return self.citation_formats.get("doi", "DOI:{doi}").format(doi=doi)

        # arXiv format
        if "arxiv.org" in url:
            arxiv_id = re.search(r"arxiv\.org/abs/([^/]+)", url)
            if arxiv_id:
                return self.citation_formats.get("arxiv", "arXiv:{id}").format(id=arxiv_id.group(1))

        # SEC format
        if "sec.gov" in url and "edgar" in url:
            return self.citation_formats.get("sec", "SEC:{cik}:{form}:{accession}").format(
                cik=document.get("metadata", {}).get("cik", ""),
                form=document.get("metadata", {}).get("form", ""),
                accession=document.get("metadata", {}).get("accession", ""),
            )

        # Court format
        if "courtlistener" in source:
            return self.citation_formats.get("court", "Court:{docket_id}:{document_id}").format(
                docket_id=document.get("metadata", {}).get("docket_id", ""),
                document_id=document.get("metadata", {}).get("document_id", ""),
            )

        # Memento format
        if "memento" in document.get("metadata", {}):
            memento_info = document["metadata"]["memento"]
            return self.citation_formats.get("memento", "Memento:{datetime}:{url}").format(
                datetime=memento_info.get("datetime", ""), url=memento_info.get("url", "")
            )

        # Default URL citation
        return f"URL:{url}"

    def _ensure_independent_sources(self, evidence_list: list[Evidence]) -> list[Evidence]:
        """Zajištění nezávislých zdrojů pro evidence"""
        independent_evidence = []
        seen_sources = set()

        for evidence in evidence_list:
            source_domain = self._extract_domain(evidence.source_url)

            if source_domain not in seen_sources:
                independent_evidence.append(evidence)
                seen_sources.add(source_domain)

                # Limit na nezávislé zdroje
                if len(independent_evidence) >= 4:
                    break

        return independent_evidence

    def _extract_domain(self, url: str) -> str:
        """Extrakce domény z URL"""
        import urllib.parse

        try:
            parsed = urllib.parse.urlparse(url)
            return parsed.netloc.lower()
        except:
            return url

    def _filter_claims_by_evidence(self, claims: list[Claim], min_evidence: int) -> list[Claim]:
        """Filtrování claims podle dostatečných důkazů"""
        verified_claims = []

        for claim in claims:
            # Kontrola minimálního počtu evidence
            if claim.has_sufficient_evidence(min_evidence):
                # Kontrola nezávislých zdrojů
                independent_sources = claim.get_independent_sources()
                if independent_sources >= min_evidence:
                    verified_claims.append(claim)
                else:
                    self.logger.warning(
                        "Claim nemá dostatečně nezávislé zdroje",
                        claim_id=claim.id,
                        sources=independent_sources,
                        required=min_evidence,
                    )
            else:
                self.logger.warning(
                    "Claim nemá dostatečné evidence",
                    claim_id=claim.id,
                    evidence_count=len(claim.evidence_list),
                    required=min_evidence,
                )

        self.logger.info(
            "Claims filtrovány podle evidence", original=len(claims), verified=len(verified_claims)
        )

        return verified_claims

    async def _generate_final_synthesis(self, query: str, verified_claims: list[Claim]) -> str:
        """Generování finální syntézy"""
        if not verified_claims:
            return "Na základě dostupných zdrojů nebylo možné najít dostatečně podložená tvrzení pro odpověď na dotaz."

        # Příprava claims pro syntézu
        claims_text = ""
        for claim in verified_claims:
            evidence_summary = f" (podloženo {len(claim.evidence_list)} zdroji)"
            claims_text += f"- {claim.text}{evidence_summary}\n"

        synthesis_prompt = f"""
Na základě následujících ověřených tvrzení vytvoř koherentní a strukturovanou odpověď na dotaz.

Dotaz: {query}

Ověřená tvrzení s důkazy:
{claims_text}

Požadavky na odpověď:
1. Strukturovaná a logická odpověď
2. Každé tvrzení musí být podložené citací
3. Uveď celkovou míru důvěryhodnosti
4. Zdůrazni omezení a nejistoty

Odpověz ve strukturovaném formátu s jasným označením citací.
"""

        try:
            synthesis = await self.ollama_agent.generate_response(
                synthesis_prompt, model=self.synthesis_model, max_tokens=1000
            )

            return synthesis

        except Exception as e:
            self.logger.error("Chyba při generování syntézy", error=str(e))
            return "Chyba při generování finální syntézy."

    def _calculate_overall_confidence(self, claims: list[Claim]) -> float:
        """Výpočet celkové confidence"""
        if not claims:
            return 0.0

        # Vážený průměr podle počtu evidence
        total_weighted_confidence = 0.0
        total_weight = 0.0

        for claim in claims:
            # Váha na základě počtu evidence a nezávislých zdrojů
            evidence_weight = min(len(claim.evidence_list) / 3.0, 1.0)
            source_weight = min(claim.get_independent_sources() / 2.0, 1.0)

            weight = evidence_weight * source_weight
            total_weighted_confidence += claim.confidence * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return total_weighted_confidence / total_weight

    def _create_evidence_bindings(self, claims: list[Claim]) -> dict[str, list[dict[str, Any]]]:
        """Vytvoření evidence bindings pro export"""
        bindings = {}

        for claim in claims:
            evidence_data = []
            for evidence in claim.evidence_list:
                evidence_data.append(
                    {
                        "source": evidence.source_id,
                        "url": evidence.source_url,
                        "citation": evidence.citation,
                        "passage": evidence.passage,
                        "confidence": evidence.confidence,
                        "timestamp": evidence.timestamp,
                        "offset_start": evidence.offset_start,
                        "offset_end": evidence.offset_end,
                    }
                )

            bindings[claim.id] = evidence_data

        return bindings

    def _claim_to_dict(self, claim: Claim) -> dict[str, Any]:
        """Konverze Claim objektu na dictionary"""
        return {
            "id": claim.id,
            "text": claim.text,
            "confidence": claim.confidence,
            "topic": claim.topic,
            "type": claim.claim_type,
            "evidence_count": len(claim.evidence_list),
            "independent_sources": claim.get_independent_sources(),
            "has_sufficient_evidence": claim.has_sufficient_evidence(self.min_evidence_per_claim),
        }
