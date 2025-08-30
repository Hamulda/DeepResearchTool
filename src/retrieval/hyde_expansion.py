#!/usr/bin/env python3
"""HyDE (Hypothetical Document Embeddings) Query Expansion
Generuje hypotetickou odpověď na dotaz a používá ji pro lepší embedding retrieval

Author: Senior Python/MLOps Agent
"""

import asyncio
from dataclasses import dataclass
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HyDEResult:
    """Výsledek HyDE expanze"""

    original_query: str
    hypothetical_document: str
    expanded_query: str
    expansion_method: str
    generation_time: float
    confidence_score: float
    fallback_used: bool = False


class HyDEQueryExpander:
    """HyDE Query Expansion engine"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.hyde_config = config.get("retrieval", {}).get("hyde", {})
        self.enabled = self.hyde_config.get("enabled", False)

        # HyDE generation parameters
        self.max_length = self.hyde_config.get("max_length", 200)
        self.temperature = self.hyde_config.get("temperature", 0.7)
        self.fallback_enabled = self.hyde_config.get("fallback_enabled", True)

        # Generation prompts for different domains
        self.generation_prompts = {
            "academic": "Write a comprehensive academic paragraph that would answer this research question: {query}",
            "factual": "Provide a detailed factual response to this question: {query}",
            "technical": "Explain the technical aspects and details related to: {query}",
            "general": "Write an informative passage that addresses: {query}",
        }

        # LLM client (will be injected)
        self.llm_client = None

    async def initialize(self, llm_client):
        """Inicializace s LLM klientem"""
        self.llm_client = llm_client
        logger.info("HyDE Query Expander initialized")

    async def expand_query(self, query: str, domain: str = "general") -> HyDEResult:
        """Expanze dotazu pomocí HyDE

        Args:
            query: Původní dotaz
            domain: Doména pro výběr promptu (academic, factual, technical, general)

        Returns:
            HyDEResult s expandovaným dotazem

        """
        start_time = time.time()

        if not self.enabled:
            return HyDEResult(
                original_query=query,
                hypothetical_document="",
                expanded_query=query,
                expansion_method="disabled",
                generation_time=0.0,
                confidence_score=1.0,
                fallback_used=True,
            )

        try:
            # Generování hypotetického dokumentu
            hypothetical_doc, confidence = await self._generate_hypothetical_document(query, domain)

            # Kombinace původního dotazu s hypotetickým dokumentem
            expanded_query = self._combine_query_and_document(query, hypothetical_doc)

            generation_time = time.time() - start_time

            return HyDEResult(
                original_query=query,
                hypothetical_document=hypothetical_doc,
                expanded_query=expanded_query,
                expansion_method=f"hyde_{domain}",
                generation_time=generation_time,
                confidence_score=confidence,
                fallback_used=False,
            )

        except Exception as e:
            logger.warning(f"HyDE expansion failed: {e}")

            if self.fallback_enabled:
                # Fallback na původní dotaz
                return HyDEResult(
                    original_query=query,
                    hypothetical_document="",
                    expanded_query=query,
                    expansion_method="fallback_original",
                    generation_time=time.time() - start_time,
                    confidence_score=0.5,
                    fallback_used=True,
                )
            raise

    async def _generate_hypothetical_document(self, query: str, domain: str) -> tuple[str, float]:
        """Generování hypotetického dokumentu"""
        if not self.llm_client:
            raise RuntimeError("LLM client not initialized")

        # Výběr promptu podle domény
        prompt_template = self.generation_prompts.get(domain, self.generation_prompts["general"])
        prompt = prompt_template.format(query=query)

        try:
            # Volání LLM pro generování hypotetického dokumentu
            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=self.max_length,
                temperature=self.temperature,
                stop_sequences=["\n\n", "Question:", "Q:"],
            )

            hypothetical_doc = response.get("text", "").strip()

            # Hodnocení kvality generovaného dokumentu
            confidence = self._assess_generation_quality(query, hypothetical_doc)

            # Minimální kontrola kvality
            if len(hypothetical_doc) < 20:
                raise ValueError("Generated document too short")

            return hypothetical_doc, confidence

        except Exception as e:
            logger.error(f"Failed to generate hypothetical document: {e}")
            raise

    def _assess_generation_quality(self, query: str, generated_doc: str) -> float:
        """Hodnocení kvality generovaného dokumentu"""
        # Základní heuristiky pro hodnocení kvality
        confidence = 0.5  # Baseline

        # Délka dokumentu
        if 50 <= len(generated_doc) <= 300:
            confidence += 0.2
        elif len(generated_doc) > 300:
            confidence += 0.1

        # Přítomnost klíčových slov z dotazu
        query_words = set(query.lower().split())
        doc_words = set(generated_doc.lower().split())
        overlap = len(query_words.intersection(doc_words)) / len(query_words)
        confidence += overlap * 0.2

        # Kontrola struktury (věty, interpunkce)
        if "." in generated_doc and len(generated_doc.split(".")) >= 2:
            confidence += 0.1

        # Kontrola na repetitivní text
        words = generated_doc.split()
        unique_words = len(set(words))
        if len(words) > 0 and unique_words / len(words) > 0.7:
            confidence += 0.1

        return min(1.0, confidence)

    def _combine_query_and_document(self, query: str, hypothetical_doc: str) -> str:
        """Kombinace původního dotazu s hypotetickým dokumentem"""
        if not hypothetical_doc:
            return query

        # Strategie kombinace podle konfigurace
        combination_strategy = self.hyde_config.get("combination_strategy", "append")

        if combination_strategy == "append":
            return f"{query} {hypothetical_doc}"
        if combination_strategy == "weighted":
            # Váženě preferuje původní dotaz
            return f"{query} {query} {hypothetical_doc}"
        if combination_strategy == "interleave":
            # Prokládá klíčová slova
            query_words = query.split()
            doc_words = hypothetical_doc.split()[:20]  # Limit pro performance
            combined = []

            # Začni s původním dotazem
            combined.extend(query_words)
            # Přidej slova z hypotetického dokumentu
            combined.extend(doc_words)

            return " ".join(combined)
        return f"{query} {hypothetical_doc}"

    async def batch_expand_queries(
        self, queries: list[str], domain: str = "general"
    ) -> list[HyDEResult]:
        """Batch expanze více dotazů"""
        tasks = [self.expand_query(query, domain) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Zpracování výsledků a chyb
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"HyDE expansion failed for query {i}: {result}")
                # Fallback výsledek
                processed_results.append(
                    HyDEResult(
                        original_query=queries[i],
                        hypothetical_document="",
                        expanded_query=queries[i],
                        expansion_method="error_fallback",
                        generation_time=0.0,
                        confidence_score=0.0,
                        fallback_used=True,
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    def get_expansion_stats(self, results: list[HyDEResult]) -> dict[str, Any]:
        """Statistiky HyDE expanze"""
        if not results:
            return {}

        total_results = len(results)
        fallback_count = sum(1 for r in results if r.fallback_used)

        successful_results = [r for r in results if not r.fallback_used]

        stats = {
            "total_queries": total_results,
            "successful_expansions": len(successful_results),
            "fallback_rate": fallback_count / total_results,
            "avg_generation_time": sum(r.generation_time for r in results) / total_results,
            "avg_confidence": sum(r.confidence_score for r in results) / total_results,
        }

        if successful_results:
            stats.update(
                {
                    "avg_doc_length": sum(len(r.hypothetical_document) for r in successful_results)
                    / len(successful_results),
                    "avg_expansion_ratio": sum(
                        len(r.expanded_query) / len(r.original_query) for r in successful_results
                    )
                    / len(successful_results),
                }
            )

        return stats


# Factory funkce pro snadnou integraci
async def create_hyde_expander(config: dict[str, Any], llm_client=None) -> HyDEQueryExpander:
    """Factory funkce pro vytvoření HyDE expanderu"""
    expander = HyDEQueryExpander(config)
    if llm_client:
        await expander.initialize(llm_client)
    return expander
