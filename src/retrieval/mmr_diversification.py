#!/usr/bin/env python3
"""
MMR (Maximal Marginal Relevance) Diversification
Diversifikuje výsledky pro pokrytí různých aspektů dotazu

Author: Senior Python/MLOps Agent
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class MMRResult:
    """Výsledek MMR diversifikace"""
    document_id: str
    original_rank: int
    mmr_rank: int
    relevance_score: float
    diversity_score: float
    mmr_score: float
    selected_for_diversity: bool


class MMRDiversifier:
    """MMR Diversification engine"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mmr_config = config.get("retrieval", {}).get("mmr", {})
        self.enabled = self.mmr_config.get("enabled", False)

        # MMR parameters
        self.lambda_param = self.mmr_config.get("lambda", 0.7)  # Relevance vs diversity trade-off
        self.diversity_threshold = self.mmr_config.get("diversity_threshold", 0.8)
        self.max_iterations = self.mmr_config.get("max_iterations", 100)

        # Embedding model for similarity computation
        self.embedding_model = None

    async def initialize(self, embedding_model):
        """Inicializace s embedding modelem"""
        self.embedding_model = embedding_model
        logger.info("MMR Diversifier initialized")

    async def diversify_results(self,
                               documents: List[Dict[str, Any]],
                               query_embedding: np.ndarray,
                               k: int = 20) -> List[MMRResult]:
        """
        Diversifikace výsledků pomocí MMR

        Args:
            documents: Seznam dokumentů s embeddingy a skóre
            query_embedding: Embedding původního dotazu
            k: Počet výsledků k vrácení

        Returns:
            List MMRResult s diversifikovanými výsledky
        """
        start_time = time.time()

        if not self.enabled or not documents:
            # Fallback - vrátí původní pořadí
            return [
                MMRResult(
                    document_id=doc.get("id", f"doc_{i}"),
                    original_rank=i,
                    mmr_rank=i,
                    relevance_score=doc.get("score", 0.0),
                    diversity_score=0.0,
                    mmr_score=doc.get("score", 0.0),
                    selected_for_diversity=False
                ) for i, doc in enumerate(documents[:k])
            ]

        try:
            # Extrakce embeddingů z dokumentů
            doc_embeddings = self._extract_embeddings(documents)

            if doc_embeddings is None or len(doc_embeddings) == 0:
                logger.warning("No embeddings found, falling back to original ranking")
                return self._fallback_ranking(documents, k)

            # MMR algoritmus
            selected_docs = []
            remaining_indices = list(range(len(documents)))

            # Výpočet relevance scores
            relevance_scores = self._calculate_relevance_scores(doc_embeddings, query_embedding)

            for iteration in range(min(k, len(documents), self.max_iterations)):
                if not remaining_indices:
                    break

                best_idx = None
                best_mmr_score = -float('inf')

                for idx in remaining_indices:
                    # MMR skóre = λ * relevance - (1-λ) * max_similarity_to_selected
                    relevance = relevance_scores[idx]

                    if selected_docs:
                        # Najdi maximální podobnost k již vybraným dokumentům
                        max_sim = self._max_similarity_to_selected(
                            doc_embeddings[idx],
                            [doc_embeddings[s["idx"]] for s in selected_docs]
                        )
                    else:
                        max_sim = 0.0

                    mmr_score = (self.lambda_param * relevance -
                               (1 - self.lambda_param) * max_sim)

                    if mmr_score > best_mmr_score:
                        best_mmr_score = mmr_score
                        best_idx = idx

                if best_idx is not None:
                    doc = documents[best_idx]
                    diversity_score = 1.0 - (0.0 if not selected_docs else
                                           self._max_similarity_to_selected(
                                               doc_embeddings[best_idx],
                                               [doc_embeddings[s["idx"]] for s in selected_docs]
                                           ))

                    selected_docs.append({
                        "idx": best_idx,
                        "doc": doc,
                        "relevance": relevance_scores[best_idx],
                        "diversity": diversity_score,
                        "mmr_score": best_mmr_score
                    })

                    remaining_indices.remove(best_idx)

            # Vytvoření výsledků
            results = []
            for mmr_rank, selected in enumerate(selected_docs):
                doc = selected["doc"]
                original_rank = selected["idx"]

                results.append(MMRResult(
                    document_id=doc.get("id", f"doc_{original_rank}"),
                    original_rank=original_rank,
                    mmr_rank=mmr_rank,
                    relevance_score=selected["relevance"],
                    diversity_score=selected["diversity"],
                    mmr_score=selected["mmr_score"],
                    selected_for_diversity=selected["diversity"] > self.diversity_threshold
                ))

            processing_time = time.time() - start_time
            logger.info(f"MMR diversification completed in {processing_time:.2f}s, "
                       f"processed {len(documents)} docs, selected {len(results)}")

            return results

        except Exception as e:
            logger.error(f"MMR diversification failed: {e}")
            return self._fallback_ranking(documents, k)

    def _extract_embeddings(self, documents: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Extrakce embeddingů z dokumentů"""

        embeddings = []
        for doc in documents:
            # Různé možné umístění embeddingů
            embedding = None

            if "embedding" in doc:
                embedding = doc["embedding"]
            elif "vector" in doc:
                embedding = doc["vector"]
            elif "embeddings" in doc:
                embedding = doc["embeddings"]

            if embedding is not None:
                if isinstance(embedding, list):
                    embeddings.append(np.array(embedding))
                elif isinstance(embedding, np.ndarray):
                    embeddings.append(embedding)
                else:
                    logger.warning(f"Unknown embedding format: {type(embedding)}")
                    return None
            else:
                logger.warning("No embedding found in document")
                return None

        if embeddings:
            return np.vstack(embeddings)
        return None

    def _calculate_relevance_scores(self, doc_embeddings: np.ndarray,
                                   query_embedding: np.ndarray) -> np.ndarray:
        """Výpočet relevance scores mezi dokumenty a dotazem"""

        # Normalizace embeddingů
        doc_embeddings_norm = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)

        # Cosine similarity
        relevance_scores = np.dot(doc_embeddings_norm, query_embedding_norm)

        # Normalizace na [0, 1]
        relevance_scores = (relevance_scores + 1) / 2

        return relevance_scores

    def _max_similarity_to_selected(self, doc_embedding: np.ndarray,
                                   selected_embeddings: List[np.ndarray]) -> float:
        """Maximální podobnost k již vybraným dokumentům"""

        if not selected_embeddings:
            return 0.0

        similarities = []
        for selected_emb in selected_embeddings:
            # Cosine similarity
            sim = np.dot(doc_embedding, selected_emb) / (
                np.linalg.norm(doc_embedding) * np.linalg.norm(selected_emb)
            )
            similarities.append(sim)

        return max(similarities)

    def _fallback_ranking(self, documents: List[Dict[str, Any]], k: int) -> List[MMRResult]:
        """Fallback ranking při selhání MMR"""

        return [
            MMRResult(
                document_id=doc.get("id", f"doc_{i}"),
                original_rank=i,
                mmr_rank=i,
                relevance_score=doc.get("score", 0.0),
                diversity_score=0.0,
                mmr_score=doc.get("score", 0.0),
                selected_for_diversity=False
            ) for i, doc in enumerate(documents[:k])
        ]

    async def analyze_diversity_coverage(self, results: List[MMRResult],
                                        documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analýza pokrytí různorodosti"""

        if not results or not self.enabled:
            return {"diversity_analysis": "disabled"}

        # Základní statistiky
        total_results = len(results)
        diverse_results = sum(1 for r in results if r.selected_for_diversity)

        # Analýza změn v pořadí
        rank_changes = []
        for result in results:
            rank_change = abs(result.mmr_rank - result.original_rank)
            rank_changes.append(rank_change)

        # Skóre distribuce
        relevance_scores = [r.relevance_score for r in results]
        diversity_scores = [r.diversity_score for r in results]
        mmr_scores = [r.mmr_score for r in results]

        return {
            "total_results": total_results,
            "diverse_selections": diverse_results,
            "diversity_rate": diverse_results / total_results if total_results > 0 else 0,
            "avg_rank_change": np.mean(rank_changes) if rank_changes else 0,
            "max_rank_change": max(rank_changes) if rank_changes else 0,
            "relevance_stats": {
                "mean": np.mean(relevance_scores),
                "std": np.std(relevance_scores),
                "min": min(relevance_scores),
                "max": max(relevance_scores)
            },
            "diversity_stats": {
                "mean": np.mean(diversity_scores),
                "std": np.std(diversity_scores),
                "min": min(diversity_scores),
                "max": max(diversity_scores)
            },
            "mmr_stats": {
                "mean": np.mean(mmr_scores),
                "std": np.std(mmr_scores),
                "min": min(mmr_scores),
                "max": max(mmr_scores)
            },
            "lambda_parameter": self.lambda_param,
            "diversity_threshold": self.diversity_threshold
        }


# Factory funkce
async def create_mmr_diversifier(config: Dict[str, Any], embedding_model=None) -> MMRDiversifier:
    """Factory funkce pro vytvoření MMR diversifieru"""
    diversifier = MMRDiversifier(config)
    if embedding_model:
        await diversifier.initialize(embedding_model)
    return diversifier
