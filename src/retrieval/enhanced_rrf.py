#!/usr/bin/env python3
"""Enhanced RRF (Reciprocal Rank Fusion) s priory a MMR diversifikace
Authority/recency priory + diversifikační MMR algoritmus

Author: Senior Python/MLOps Agent
"""

from dataclasses import dataclass
from datetime import datetime
import logging
import math
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RRFConfig:
    """Konfigurace RRF systému"""

    k_parameter: int = 60  # RRF konstantní parameter
    authority_weight: float = 0.3  # Váha authority prioru
    recency_weight: float = 0.2  # Váha recency prioru
    diversity_weight: float = 0.1  # Váha diversity bonusu

    # Source authority mapping
    source_authorities: dict[str, float] = None  # domain -> authority score

    # Recency parameters
    recency_decay_days: int = 365  # Dny pro exponential decay
    max_recency_bonus: float = 1.5  # Maximální recency bonus

    # MMR parameters
    mmr_enabled: bool = True
    diversity_lambda: float = 0.7  # Relevance vs diversity trade-off
    diversity_k: int = 20  # Počet výsledků pro MMR diversifikaci
    similarity_threshold: float = 0.8  # Práh pro podobnost


@dataclass
class RankedDocument:
    """Dokument s ranking metadata"""

    doc_id: str
    content: str
    original_score: float
    rank_position: int
    source_domain: str
    timestamp: datetime | None = None

    # RRF komponenty
    rrf_score: float = 0.0
    authority_bonus: float = 0.0
    recency_bonus: float = 0.0
    diversity_penalty: float = 0.0
    final_score: float = 0.0

    # Metadata
    fusion_sources: list[str] = None  # Které retrieval systémy našly tento dokument
    similarity_vector: np.ndarray | None = None


class SourceAuthorityManager:
    """Manager pro source authority scoring"""

    def __init__(self, config: RRFConfig):
        self.config = config
        self.authority_cache = {}

        # Default authority scores
        self.default_authorities = {
            "arxiv.org": 0.9,
            "nature.com": 0.95,
            "science.org": 0.95,
            "acm.org": 0.9,
            "ieee.org": 0.9,
            "pubmed.ncbi.nlm.nih.gov": 0.85,
            "scholar.google.com": 0.7,
            "wikipedia.org": 0.6,
            "reddit.com": 0.3,
            "twitter.com": 0.2,
            "unknown": 0.5,
        }

        # Merge s user-defined authorities
        if config.source_authorities:
            self.default_authorities.update(config.source_authorities)

    def get_authority_score(self, domain: str) -> float:
        """Získá authority score pro doménu"""
        if domain in self.authority_cache:
            return self.authority_cache[domain]

        # Exact match
        if domain in self.default_authorities:
            score = self.default_authorities[domain]
        else:
            # Partial match (subdomena)
            score = self.default_authorities["unknown"]
            for auth_domain, auth_score in self.default_authorities.items():
                if domain.endswith(auth_domain) or auth_domain in domain:
                    score = max(score, auth_score * 0.9)  # Slight penalty for subdomains
                    break

        self.authority_cache[domain] = score
        return score

    def update_authorities(self, new_authorities: dict[str, float]):
        """Aktualizuje authority scores"""
        self.default_authorities.update(new_authorities)
        self.authority_cache.clear()  # Clear cache


class RecencyScorer:
    """Scorer pro recency bonusy"""

    def __init__(self, config: RRFConfig):
        self.config = config

    def calculate_recency_bonus(self, timestamp: datetime | None) -> float:
        """Vypočítá recency bonus na základě timestamp"""
        if not timestamp:
            return 0.0

        now = datetime.now()
        days_old = (now - timestamp).days

        if days_old < 0:  # Future date
            return 0.0

        # Exponential decay
        decay_factor = math.exp(-days_old / self.config.recency_decay_days)
        recency_bonus = decay_factor * self.config.max_recency_bonus

        return min(recency_bonus, self.config.max_recency_bonus)


class MMRDiversifier:
    """MMR (Maximal Marginal Relevance) diversifikace"""

    def __init__(self, config: RRFConfig, embedding_model=None):
        self.config = config
        self.embedding_model = embedding_model

    async def diversify_results(
        self, ranked_docs: list[RankedDocument], query_embedding: np.ndarray | None = None
    ) -> list[RankedDocument]:
        """Aplikuje MMR diversifikaci na ranked dokumenty
        """
        if not self.config.mmr_enabled or len(ranked_docs) <= 1:
            return ranked_docs

        # Vezmi top diversity_k dokumentů pro MMR
        candidates = ranked_docs[: self.config.diversity_k]
        remaining = ranked_docs[self.config.diversity_k :]

        # Získej embeddings pro diversifikaci
        await self._ensure_embeddings(candidates)

        # MMR algoritmus
        selected = []
        remaining_candidates = candidates.copy()

        # První dokument je vždy nejvíce relevantní
        if remaining_candidates:
            first_doc = remaining_candidates.pop(0)
            selected.append(first_doc)

        # Iterativně vybírej dokumenty s nejvyšším MMR score
        while remaining_candidates and len(selected) < self.config.diversity_k:
            best_doc = None
            best_mmr_score = -float("inf")

            for doc in remaining_candidates:
                mmr_score = self._calculate_mmr_score(doc, selected, query_embedding)
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_doc = doc

            if best_doc:
                remaining_candidates.remove(best_doc)
                selected.append(best_doc)

        # Kombinuj selected s remaining dokumenty
        final_results = selected + remaining

        # Přepočítej final scores s diversity penalties
        self._apply_diversity_penalties(final_results)

        return final_results

    async def _ensure_embeddings(self, docs: list[RankedDocument]):
        """Zajistí, že všechny dokumenty mají embeddings"""
        for doc in docs:
            if doc.similarity_vector is None:
                if self.embedding_model:
                    try:
                        embedding = await self.embedding_model.encode(doc.content)
                        doc.similarity_vector = np.array(embedding)
                    except Exception as e:
                        logger.warning(f"Failed to generate embedding for doc {doc.doc_id}: {e}")
                        # Fallback na random embedding
                        doc.similarity_vector = np.random.normal(
                            0, 1, 384
                        )  # Default embedding size
                else:
                    # Mock embedding
                    doc.similarity_vector = np.random.normal(0, 1, 384)

    def _calculate_mmr_score(
        self,
        doc: RankedDocument,
        selected: list[RankedDocument],
        query_embedding: np.ndarray | None,
    ) -> float:
        """Vypočítá MMR score pro dokument"""
        relevance_score = doc.final_score  # Use current final score as relevance

        # Vypočítaj maximální similaritu s již vybranými dokumenty
        max_similarity = 0.0
        if selected and doc.similarity_vector is not None:
            similarities = []
            for selected_doc in selected:
                if selected_doc.similarity_vector is not None:
                    sim = self._cosine_similarity(
                        doc.similarity_vector, selected_doc.similarity_vector
                    )
                    similarities.append(sim)

            if similarities:
                max_similarity = max(similarities)

        # MMR formula: λ * relevance - (1-λ) * max_similarity
        mmr_score = (
            self.config.diversity_lambda * relevance_score
            - (1 - self.config.diversity_lambda) * max_similarity
        )

        return mmr_score

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Vypočítá cosine similaritu mezi dvěma vektory"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)
        except Exception:
            return 0.0

    def _apply_diversity_penalties(self, docs: list[RankedDocument]):
        """Aplikuje diversity penalties na final scores"""
        for i, doc in enumerate(docs):
            # Penalty roste s pozicí v diversifikovaném seznamu
            diversity_penalty = i * 0.01  # Malá penalty pro každou pozici
            doc.diversity_penalty = diversity_penalty
            doc.final_score = max(0, doc.final_score - diversity_penalty)


class EnhancedRRF:
    """Enhanced RRF s priory a MMR diversifikací"""

    def __init__(self, config: RRFConfig, embedding_model=None):
        self.config = config
        self.authority_manager = SourceAuthorityManager(config)
        self.recency_scorer = RecencyScorer(config)
        self.mmr_diversifier = MMRDiversifier(config, embedding_model)

        self.fusion_stats = {
            "total_fusions": 0,
            "avg_input_sources": 0.0,
            "avg_authority_bonus": 0.0,
            "avg_recency_bonus": 0.0,
            "mmr_diversifications": 0,
        }

    async def fuse_results(
        self,
        retrieval_results: dict[str, list[dict[str, Any]]],
        query_embedding: np.ndarray | None = None,
    ) -> tuple[list[RankedDocument], dict[str, Any]]:
        """Hlavní RRF fusion s priory a MMR

        Args:
            retrieval_results: {"source_name": [documents]}
            query_embedding: Query embedding pro MMR

        """
        start_time = time.time()
        self.fusion_stats["total_fusions"] += 1

        fusion_metadata = {
            "input_sources": list(retrieval_results.keys()),
            "input_counts": {source: len(docs) for source, docs in retrieval_results.items()},
            "rrf_k_parameter": self.config.k_parameter,
            "authority_weight": self.config.authority_weight,
            "recency_weight": self.config.recency_weight,
        }

        # 1. Vytvoř ranking pro každý source
        source_rankings = {}
        for source_name, docs in retrieval_results.items():
            source_rankings[source_name] = self._create_document_ranking(docs, source_name)

        # 2. RRF fusion
        fused_docs = self._perform_rrf_fusion(source_rankings)

        # 3. Aplikuj priory (authority + recency)
        self._apply_priory(fused_docs)

        # 4. Přepočítaj final scores
        self._calculate_final_scores(fused_docs)

        # 5. Seřaď podle final score
        fused_docs.sort(key=lambda x: x.final_score, reverse=True)

        # 6. MMR diversifikace
        if self.config.mmr_enabled:
            fused_docs = await self.mmr_diversifier.diversify_results(fused_docs, query_embedding)
            self.fusion_stats["mmr_diversifications"] += 1

        # 7. Aktualizuj statistiky
        elapsed_time = time.time() - start_time
        fusion_metadata.update(
            {
                "fusion_time_seconds": elapsed_time,
                "output_documents": len(fused_docs),
                "mmr_enabled": self.config.mmr_enabled,
                "avg_authority_bonus": (
                    np.mean([doc.authority_bonus for doc in fused_docs]) if fused_docs else 0
                ),
                "avg_recency_bonus": (
                    np.mean([doc.recency_bonus for doc in fused_docs]) if fused_docs else 0
                ),
            }
        )

        self._update_fusion_stats(len(retrieval_results), fusion_metadata)

        logger.info(f"RRF fusion completed in {elapsed_time:.2f}s: {len(fused_docs)} documents")

        return fused_docs, fusion_metadata

    def _create_document_ranking(
        self, docs: list[dict[str, Any]], source_name: str
    ) -> list[RankedDocument]:
        """Vytvoří ranking dokumentů z jednoho source"""
        ranked_docs = []

        for i, doc in enumerate(docs):
            # Extrahuj domain z URL nebo source
            domain = self._extract_domain(doc.get("url", ""), doc.get("source", source_name))

            # Parsuj timestamp
            timestamp = self._parse_timestamp(doc.get("timestamp"), doc.get("date"))

            ranked_doc = RankedDocument(
                doc_id=doc.get("id", f"{source_name}_{i}"),
                content=doc.get("content", doc.get("text", "")),
                original_score=doc.get("score", 1.0 / (i + 1)),  # Fallback na reciprocal rank
                rank_position=i,
                source_domain=domain,
                timestamp=timestamp,
                fusion_sources=[source_name],
            )

            ranked_docs.append(ranked_doc)

        return ranked_docs

    def _perform_rrf_fusion(
        self, source_rankings: dict[str, list[RankedDocument]]
    ) -> list[RankedDocument]:
        """Provede základní RRF fusion"""
        doc_scores = {}  # doc_id -> RankedDocument

        for source_name, ranking in source_rankings.items():
            for rank, doc in enumerate(ranking):
                rrf_score = 1.0 / (self.config.k_parameter + rank)

                if doc.doc_id in doc_scores:
                    # Kombinuj s existujícím dokumentem
                    existing_doc = doc_scores[doc.doc_id]
                    existing_doc.rrf_score += rrf_score
                    existing_doc.fusion_sources.append(source_name)
                    # Použij lepší original score
                    existing_doc.original_score = max(existing_doc.original_score, doc.original_score)
                else:
                    # Nový dokument
                    doc.rrf_score = rrf_score
                    doc_scores[doc.doc_id] = doc

        return list(doc_scores.values())

    def _apply_priory(self, docs: list[RankedDocument]):
        """Aplikuje authority a recency priory"""
        for doc in docs:
            # Authority bonus
            authority_score = self.authority_manager.get_authority_score(doc.source_domain)
            doc.authority_bonus = (authority_score - 0.5) * self.config.authority_weight

            # Recency bonus
            recency_bonus = self.recency_scorer.calculate_recency_bonus(doc.timestamp)
            doc.recency_bonus = recency_bonus * self.config.recency_weight

    def _calculate_final_scores(self, docs: list[RankedDocument]):
        """Vypočítá final scores pro všechny dokumenty"""
        for doc in docs:
            doc.final_score = doc.rrf_score + doc.authority_bonus + doc.recency_bonus

    def _extract_domain(self, url: str, source: str) -> str:
        """Extrahuje domain z URL nebo source"""
        if url:
            try:
                from urllib.parse import urlparse

                parsed = urlparse(url)
                return parsed.netloc.lower()
            except:
                pass

        # Fallback na source name
        if "." in source:
            return source.lower()

        return "unknown"

    def _parse_timestamp(self, timestamp_str: Any, date_str: Any) -> datetime | None:
        """Parsuje timestamp z různých formátů"""
        for ts in [timestamp_str, date_str]:
            if not ts:
                continue

            try:
                if isinstance(ts, datetime):
                    return ts
                if isinstance(ts, str):
                    # Zkus různé formáty
                    for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
                        try:
                            return datetime.strptime(ts, fmt)
                        except ValueError:
                            continue
            except:
                continue

        return None

    def _update_fusion_stats(self, num_sources: int, metadata: dict[str, Any]):
        """Aktualizuje fusion statistiky"""
        alpha = 0.1  # Exponential moving average

        # Avg input sources
        if self.fusion_stats["avg_input_sources"] == 0:
            self.fusion_stats["avg_input_sources"] = num_sources
        else:
            self.fusion_stats["avg_input_sources"] = (
                alpha * num_sources + (1 - alpha) * self.fusion_stats["avg_input_sources"]
            )

        # Avg bonuses
        avg_auth = metadata.get("avg_authority_bonus", 0)
        avg_rec = metadata.get("avg_recency_bonus", 0)

        if self.fusion_stats["avg_authority_bonus"] == 0:
            self.fusion_stats["avg_authority_bonus"] = avg_auth
        else:
            self.fusion_stats["avg_authority_bonus"] = (
                alpha * avg_auth + (1 - alpha) * self.fusion_stats["avg_authority_bonus"]
            )

        if self.fusion_stats["avg_recency_bonus"] == 0:
            self.fusion_stats["avg_recency_bonus"] = avg_rec
        else:
            self.fusion_stats["avg_recency_bonus"] = (
                alpha * avg_rec + (1 - alpha) * self.fusion_stats["avg_recency_bonus"]
            )

    def get_stats(self) -> dict[str, Any]:
        """Získá statistiky RRF systému"""
        return self.fusion_stats.copy()

    def update_config(self, new_config: dict[str, Any]):
        """Aktualizuje konfiguraci za běhu"""
        for key, value in new_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Update authority scores if provided
        if "source_authorities" in new_config:
            self.authority_manager.update_authorities(new_config["source_authorities"])


# Factory funkce
def create_enhanced_rrf(config: dict[str, Any], embedding_model=None) -> EnhancedRRF:
    """Factory funkce pro vytvoření Enhanced RRF systému"""
    rrf_config_dict = config.get("retrieval", {}).get("rrf", {})
    rrf_config = RRFConfig(**rrf_config_dict)

    return EnhancedRRF(rrf_config, embedding_model)


# Export hlavních tříd
__all__ = ["EnhancedRRF", "RRFConfig", "RankedDocument", "create_enhanced_rrf"]
