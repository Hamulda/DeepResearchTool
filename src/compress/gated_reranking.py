#!/usr/bin/env python3
"""Gated Reranking Engine - OPTIMALIZOVÁNO pro M1
Dvoufázový re-ranking: BM25 (rychlá pre-filtrace) → LLM (precizní hodnocení)
Dramaticky snižuje počet dokumentů pro pomalý LLM (až 85% redukce)

Author: Senior Python/MLOps Agent
"""

from dataclasses import dataclass
from enum import Enum
import logging
import re
import time
from typing import Any

# NOVÉ IMPORTY pro optimalizovaný re-ranking
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class RerankingStage(Enum):
    """Reranking stages in the optimized pipeline"""

    BM25_PREFILTER = "bm25_prefilter"          # FÁZE 1: Rychlý BM25 filter
    LLM_PRECISION = "llm_precision"            # FÁZE 2: Precizní LLM hodnocení
    UNCERTAINTY_GATE = "uncertainty_gate"
    FINAL_RANKING = "final_ranking"


@dataclass
class OptimizedRerankingConfig:
    """Konfigurace pro optimalizovaný dvoufázový re-ranking"""

    # KLÍČOVÉ OPTIMALIZACE: Dvoufázový systém
    bm25_candidates: int = 100              # Kolik dokumentů z retrievalu
    llm_candidates: int = 15                # Kolik top BM25 kandidátů pro LLM (85% redukce!)

    # BM25 konfigurace (Fáze 1)
    bm25_enabled: bool = True
    bm25_k1: float = 1.2                   # BM25 parameter k1
    bm25_b: float = 0.75                   # BM25 parameter b

    # LLM konfigurace (Fáze 2)
    llm_enabled: bool = True
    confidence_threshold: float = 0.7
    relevance_threshold: float = 0.3

    # Uncertainty gating
    uncertainty_threshold: float = 0.1
    uncertainty_enabled: bool = True

    # Performance optimalizace
    parallel_processing: bool = True
    max_workers: int = 4
    timeout_seconds: int = 30


@dataclass
class DocumentCandidate:
    """Reprezentace dokumentu v re-ranking pipeline"""

    id: str
    content: str
    title: str
    source: str
    metadata: dict[str, Any]

    # Skóre z různých fází
    initial_score: float = 0.0
    bm25_score: float = 0.0
    llm_score: float = 0.0
    final_score: float = 0.0

    # Metriky hodnocení
    confidence: float = 0.0
    uncertainty: float = 0.0
    relevance_reasons: list[str] = None

    # Pipeline tracking
    processed_stages: list[str] = None
    processing_time_ms: float = 0.0

    def __post_init__(self):
        if self.relevance_reasons is None:
            self.relevance_reasons = []
        if self.processed_stages is None:
            self.processed_stages = []


class OptimizedGatedReranker:
    """OPTIMALIZOVANÝ Gated Reranker s dvoufázovým systemem

    KLÍČOVÁ OPTIMALIZACE:
    1. BM25 pre-filtrace (rychlá, na CPU) - redukuje kandidáty o 85%
    2. LLM precision ranking (pomalý, na GPU) - pouze na top kandidátech

    Výsledek: Dramatické zrychlení při zachování kvality
    """

    def __init__(self, config: OptimizedRerankingConfig, llm_client=None):
        self.config = config
        self.llm_client = llm_client

        # BM25 komponenty
        self.bm25_index: BM25Okapi | None = None
        self.corpus_processed: list[list[str]] = []
        self.document_mapping: dict[int, str] = {}

        # Performance tracking
        self.performance_stats = {
            "total_documents_processed": 0,
            "bm25_filtering_time_ms": 0.0,
            "llm_ranking_time_ms": 0.0,
            "total_processing_time_ms": 0.0,
            "reduction_ratio": 0.0
        }

        logger.info(f"✅ Optimalizovaný Reranker inicializován: "
                   f"BM25({config.bm25_candidates}) → LLM({config.llm_candidates}) "
                   f"= {((config.bm25_candidates - config.llm_candidates) / config.bm25_candidates * 100):.1f}% redukce")

    async def rerank_documents(self,
                             query: str,
                             documents: list[dict[str, Any]]) -> list[DocumentCandidate]:
        """HLAVNÍ OPTIMALIZOVANÁ METODA: Dvoufázový re-ranking

        Args:
            query: Uživatelský dotaz
            documents: Seznam dokumentů k re-rankingu

        Returns:
            Seřazené dokumenty s optimalizovaným skóre

        """
        start_time = time.time()
        logger.info(f"🚀 Spouštím optimalizovaný re-ranking: {len(documents)} dokumentů")

        # Konverze na DocumentCandidate objekty
        candidates = [self._create_candidate(doc, i) for i, doc in enumerate(documents)]

        # FÁZE 1: BM25 Pre-filtrace (KLÍČOVÁ OPTIMALIZACE!)
        if self.config.bm25_enabled and len(candidates) > self.config.llm_candidates:
            candidates = await self._bm25_prefilter_phase(query, candidates)
            logger.info(f"✅ BM25 pre-filtrace: {len(candidates)} kandidátů (z {len(documents)})")

        # FÁZE 2: LLM Precision Ranking (pouze na top kandidátech)
        if self.config.llm_enabled and self.llm_client:
            candidates = await self._llm_precision_phase(query, candidates)
            logger.info("✅ LLM precision ranking dokončen")

        # FÁZE 3: Uncertainty Gating a finální ranking
        if self.config.uncertainty_enabled:
            candidates = await self._uncertainty_gating_phase(candidates)

        # Finální seřazení podle kombinovaného skóre
        candidates = self._final_ranking_phase(candidates)

        # Performance statistiky
        total_time = (time.time() - start_time) * 1000
        self._update_performance_stats(total_time, len(documents), len(candidates))

        logger.info(f"🎉 Optimalizovaný re-ranking dokončen za {total_time:.1f}ms, "
                   f"redukce: {self.performance_stats['reduction_ratio']:.1f}%")

        return candidates

    async def _bm25_prefilter_phase(self,
                                   query: str,
                                   candidates: list[DocumentCandidate]) -> list[DocumentCandidate]:
        """FÁZE 1: BM25 Pre-filtrace - rychlé odstranění irelevantních dokumentů

        Tato fáze běží na CPU a je extrémně rychlá
        """
        start_time = time.time()
        logger.debug(f"🔍 BM25 pre-filtrace: {len(candidates)} → {self.config.bm25_candidates}")

        # Příprava korpusu pro BM25
        corpus = []
        for candidate in candidates:
            # Kombinace title + content pro lepší matching
            text = f"{candidate.title} {candidate.content}"
            # Tokenizace (jednoduchá ale efektivní)
            tokens = self._tokenize_for_bm25(text)
            corpus.append(tokens)

        # Inicializace BM25 indexu
        self.bm25_index = BM25Okapi(corpus, k1=self.config.bm25_k1, b=self.config.bm25_b)

        # Tokenizace dotazu
        query_tokens = self._tokenize_for_bm25(query)

        # BM25 skórování
        bm25_scores = self.bm25_index.get_scores(query_tokens)

        # Přiřazení BM25 skóre kandidátům
        for i, candidate in enumerate(candidates):
            candidate.bm25_score = float(bm25_scores[i])
            candidate.processed_stages.append(RerankingStage.BM25_PREFILTER.value)

        # Seřazení podle BM25 skóre a výběr top kandidátů
        candidates_sorted = sorted(candidates, key=lambda x: x.bm25_score, reverse=True)
        top_candidates = candidates_sorted[:self.config.bm25_candidates]

        # Performance tracking
        bm25_time = (time.time() - start_time) * 1000
        self.performance_stats["bm25_filtering_time_ms"] = bm25_time

        logger.debug(f"⚡ BM25 pre-filtrace dokončena za {bm25_time:.1f}ms")

        return top_candidates

    async def _llm_precision_phase(self,
                                  query: str,
                                  candidates: list[DocumentCandidate]) -> list[DocumentCandidate]:
        """FÁZE 2: LLM Precision Ranking - precizní hodnocení pouze top kandidátů

        Tato fáze je pomalá ale velmi přesná, běží pouze na pre-filtrovaných kandidátech
        """
        start_time = time.time()
        logger.debug(f"🧠 LLM precision ranking: {len(candidates)} kandidátů")

        # Omezení na LLM kandidáty (další redukce)
        llm_candidates = candidates[:self.config.llm_candidates]
        logger.debug(f"🎯 LLM hodnotí pouze top {len(llm_candidates)} kandidátů "
                    f"(85% redukce z původního počtu)")

        # Paralelní zpracování LLM hodnocení
        if self.config.parallel_processing:
            llm_candidates = await self._parallel_llm_evaluation(query, llm_candidates)
        else:
            llm_candidates = await self._sequential_llm_evaluation(query, llm_candidates)

        # Kandidáti, kteří nebyli hodnoceni LLM, dostanou skóre založené na BM25
        remaining_candidates = candidates[len(llm_candidates):]
        for candidate in remaining_candidates:
            candidate.llm_score = candidate.bm25_score * 0.5  # Penalizace za nehodnocení
            candidate.confidence = 0.3  # Nízká confidence

        # Sloučení hodnocených a nehodnocených kandidátů
        all_candidates = llm_candidates + remaining_candidates

        # Performance tracking
        llm_time = (time.time() - start_time) * 1000
        self.performance_stats["llm_ranking_time_ms"] = llm_time

        logger.debug(f"🧠 LLM precision ranking dokončen za {llm_time:.1f}ms")

        return all_candidates

    async def _parallel_llm_evaluation(self,
                                      query: str,
                                      candidates: list[DocumentCandidate]) -> list[DocumentCandidate]:
        """Paralelní LLM hodnocení pro rychlejší zpracování"""
        import asyncio

        async def evaluate_candidate(candidate: DocumentCandidate) -> DocumentCandidate:
            """Hodnocení jednoho kandidáta"""
            try:
                # LLM prompt pro hodnocení relevance
                evaluation_prompt = self._create_evaluation_prompt(query, candidate)

                # Volání LLM (asynchronní)
                response = await self.llm_client.generate_async(
                    evaluation_prompt,
                    max_tokens=100,
                    temperature=0.1  # Nízká teplota pro konzistentní hodnocení
                )

                # Parsování LLM odpovědi
                score, confidence, reasons = self._parse_llm_evaluation(response)

                candidate.llm_score = score
                candidate.confidence = confidence
                candidate.relevance_reasons = reasons
                candidate.processed_stages.append(RerankingStage.LLM_PRECISION.value)

                return candidate

            except Exception as e:
                logger.warning(f"LLM hodnocení selhalo pro kandidáta {candidate.id}: {e}")
                candidate.llm_score = candidate.bm25_score * 0.5
                candidate.confidence = 0.1
                return candidate

        # Paralelní zpracování s limitem workerů
        semaphore = asyncio.Semaphore(self.config.max_workers)

        async def evaluate_with_semaphore(candidate):
            async with semaphore:
                return await evaluate_candidate(candidate)

        # Spuštění paralelního hodnocení
        tasks = [evaluate_with_semaphore(candidate) for candidate in candidates]

        try:
            evaluated_candidates = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=self.config.timeout_seconds
            )
            return evaluated_candidates
        except TimeoutError:
            logger.warning(f"LLM hodnocení timeout po {self.config.timeout_seconds}s")
            return candidates

    async def _sequential_llm_evaluation(self,
                                        query: str,
                                        candidates: list[DocumentCandidate]) -> list[DocumentCandidate]:
        """Sekvenční LLM hodnocení (fallback)"""
        for candidate in candidates:
            try:
                evaluation_prompt = self._create_evaluation_prompt(query, candidate)
                response = await self.llm_client.generate_async(evaluation_prompt, max_tokens=100)

                score, confidence, reasons = self._parse_llm_evaluation(response)
                candidate.llm_score = score
                candidate.confidence = confidence
                candidate.relevance_reasons = reasons
                candidate.processed_stages.append(RerankingStage.LLM_PRECISION.value)

            except Exception as e:
                logger.warning(f"LLM hodnocení selhalo pro kandidáta {candidate.id}: {e}")
                candidate.llm_score = candidate.bm25_score * 0.5
                candidate.confidence = 0.1

        return candidates

    def _create_evaluation_prompt(self, query: str, candidate: DocumentCandidate) -> str:
        """Vytvoření prompt pro LLM hodnocení relevance"""
        prompt = f"""Evaluate the relevance of this document to the given query.

Query: {query}

Document Title: {candidate.title}
Document Content: {candidate.content[:500]}...

Rate the relevance on a scale of 0.0 to 1.0 and provide your confidence level.

Response format:
Score: [0.0-1.0]
Confidence: [0.0-1.0]
Reasons: [brief explanation]

Response:"""

        return prompt

    def _parse_llm_evaluation(self, response: str) -> tuple[float, float, list[str]]:
        """Parsování LLM odpovědi na skóre, confidence a důvody"""
        try:
            lines = response.strip().split('\n')
            score = 0.5
            confidence = 0.5
            reasons = []

            for line in lines:
                line = line.strip()
                if line.startswith('Score:'):
                    score = float(re.search(r'(\d+\.?\d*)', line).group(1))
                    score = max(0.0, min(1.0, score))
                elif line.startswith('Confidence:'):
                    confidence = float(re.search(r'(\d+\.?\d*)', line).group(1))
                    confidence = max(0.0, min(1.0, confidence))
                elif line.startswith('Reasons:'):
                    reason_text = line.replace('Reasons:', '').strip()
                    reasons = [reason_text] if reason_text else []

            return score, confidence, reasons

        except Exception as e:
            logger.warning(f"Chyba při parsování LLM odpovědi: {e}")
            return 0.5, 0.3, []

    async def _uncertainty_gating_phase(self,
                                       candidates: list[DocumentCandidate]) -> list[DocumentCandidate]:
        """FÁZE 3: Uncertainty Gating - filtrování nejistých výsledků
        """
        logger.debug(f"🚪 Uncertainty gating: {len(candidates)} kandidátů")

        for candidate in candidates:
            # Výpočet uncertainty na základě confidence a konzistence skóre
            score_variance = abs(candidate.bm25_score - candidate.llm_score)
            uncertainty = 1.0 - candidate.confidence + (score_variance * 0.1)
            candidate.uncertainty = max(0.0, min(1.0, uncertainty))

            candidate.processed_stages.append(RerankingStage.UNCERTAINTY_GATE.value)

        # Filtrování kandidátů s vysokou uncertainty
        filtered_candidates = [
            candidate for candidate in candidates
            if candidate.uncertainty <= self.config.uncertainty_threshold
        ]

        if len(filtered_candidates) < len(candidates):
            logger.debug(f"🚪 Uncertainty gate odfiltroval "
                        f"{len(candidates) - len(filtered_candidates)} nejistých kandidátů")

        return filtered_candidates if filtered_candidates else candidates[:5]  # Minimálně 5 výsledků

    def _final_ranking_phase(self, candidates: list[DocumentCandidate]) -> list[DocumentCandidate]:
        """FÁZE 4: Finální ranking - kombinace všech skóre
        """
        logger.debug(f"🏁 Finální ranking: {len(candidates)} kandidátů")

        for candidate in candidates:
            # Vážená kombinace BM25 a LLM skóre
            bm25_weight = 0.3
            llm_weight = 0.7

            # Základní kombinované skóre
            combined_score = (candidate.bm25_score * bm25_weight +
                            candidate.llm_score * llm_weight)

            # Confidence boost
            confidence_boost = candidate.confidence * 0.1

            # Finální skóre
            candidate.final_score = combined_score + confidence_boost
            candidate.processed_stages.append(RerankingStage.FINAL_RANKING.value)

        # Finální seřazení
        final_candidates = sorted(candidates, key=lambda x: x.final_score, reverse=True)

        logger.debug("🏁 Finální ranking dokončen")

        return final_candidates

    def _tokenize_for_bm25(self, text: str) -> list[str]:
        """Rychlá tokenizace pro BM25"""
        # Normalizace textu
        text = text.lower()
        # Odebrání speciálních znaků a rozdělení na slova
        tokens = re.findall(r'\b[a-z]{2,}\b', text)
        # Filtrování velmi častých slov (stop words)
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had'}
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]

        return tokens

    def _create_candidate(self, document: dict[str, Any], index: int) -> DocumentCandidate:
        """Vytvoření DocumentCandidate z dokumentu"""
        return DocumentCandidate(
            id=document.get('id', f'doc_{index}'),
            content=document.get('content', ''),
            title=document.get('title', ''),
            source=document.get('source', ''),
            metadata=document.get('metadata', {}),
            initial_score=document.get('score', 0.0)
        )

    def _update_performance_stats(self, total_time_ms: float, initial_count: int, final_count: int):
        """Aktualizace performance statistik"""
        self.performance_stats.update({
            "total_documents_processed": initial_count,
            "total_processing_time_ms": total_time_ms,
            "reduction_ratio": ((initial_count - final_count) / initial_count * 100) if initial_count > 0 else 0.0
        })

    def get_performance_report(self) -> dict[str, Any]:
        """Získání performance reportu"""
        return {
            "optimization_summary": {
                "bm25_candidates": self.config.bm25_candidates,
                "llm_candidates": self.config.llm_candidates,
                "theoretical_reduction_percent": ((self.config.bm25_candidates - self.config.llm_candidates) / self.config.bm25_candidates * 100),
                "actual_reduction_percent": self.performance_stats["reduction_ratio"]
            },
            "timing_breakdown": {
                "bm25_filtering_ms": self.performance_stats["bm25_filtering_time_ms"],
                "llm_ranking_ms": self.performance_stats["llm_ranking_time_ms"],
                "total_processing_ms": self.performance_stats["total_processing_time_ms"],
                "speedup_factor": self._calculate_speedup_factor()
            },
            "efficiency_metrics": {
                "documents_per_second": self._calculate_throughput(),
                "llm_calls_saved": self.config.bm25_candidates - self.config.llm_candidates,
                "estimated_cost_savings_percent": self._estimate_cost_savings()
            }
        }

    def _calculate_speedup_factor(self) -> float:
        """Výpočet faktoru zrychlení oproti full LLM ranking"""
        # Odhad času, který by trval full LLM ranking
        estimated_full_llm_time = self.performance_stats["llm_ranking_time_ms"] * (
            self.config.bm25_candidates / max(1, self.config.llm_candidates)
        )

        actual_time = self.performance_stats["total_processing_time_ms"]

        return estimated_full_llm_time / max(1, actual_time)

    def _calculate_throughput(self) -> float:
        """Výpočet propustnosti dokumentů za sekundu"""
        if self.performance_stats["total_processing_time_ms"] > 0:
            return (self.performance_stats["total_documents_processed"] * 1000.0 /
                   self.performance_stats["total_processing_time_ms"])
        return 0.0

    def _estimate_cost_savings(self) -> float:
        """Odhad úspory nákladů (méně LLM volání)"""
        llm_calls_saved = self.config.bm25_candidates - self.config.llm_candidates
        total_potential_calls = self.config.bm25_candidates

        return (llm_calls_saved / max(1, total_potential_calls)) * 100.0


# Convenience funkce pro snadné použití
async def create_optimized_reranker(config: dict[str, Any], llm_client=None) -> OptimizedGatedReranker:
    """Factory funkce pro vytvoření optimalizovaného rerankeru

    Args:
        config: Konfigurační slovník
        llm_client: LLM klient pro precision ranking

    Returns:
        Nakonfigurovaný OptimizedGatedReranker

    """
    reranking_config = config.get("reranking", {})

    optimized_config = OptimizedRerankingConfig(
        bm25_candidates=reranking_config.get("bm25_candidates", 100),
        llm_candidates=reranking_config.get("llm_candidates", 15),
        bm25_k1=reranking_config.get("bm25_k1", 1.2),
        bm25_b=reranking_config.get("bm25_b", 0.75),
        confidence_threshold=reranking_config.get("confidence_threshold", 0.7),
        relevance_threshold=reranking_config.get("relevance_threshold", 0.3),
        uncertainty_threshold=reranking_config.get("uncertainty_threshold", 0.1)
    )

    reranker = OptimizedGatedReranker(optimized_config, llm_client)

    logger.info(f"✅ Optimalizovaný reranker vytvořen: "
               f"{optimized_config.bm25_candidates} → {optimized_config.llm_candidates} dokumentů "
               f"({((optimized_config.bm25_candidates - optimized_config.llm_candidates) / optimized_config.bm25_candidates * 100):.1f}% redukce LLM volání)")

    return reranker
