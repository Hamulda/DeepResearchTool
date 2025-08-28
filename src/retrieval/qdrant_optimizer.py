#!/usr/bin/env python3
"""
Per-Collection Qdrant ef_search Optimization
Optimalizuje ef_search parametry pro různé kolekce s latency/recall trade-off analýzou

Author: Senior Python/MLOps Agent
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import SearchRequest, SearchParams

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Výsledek vyhledávání s metrikami"""
    collection_name: str
    ef_search: int
    query_id: str
    results_count: int
    latency_ms: float
    recall_estimate: float
    precision_estimate: float
    top_scores: List[float]
    search_quality_score: float


@dataclass
class CollectionOptimization:
    """Optimalizace pro jednu kolekci"""
    collection_name: str
    optimal_ef_search: int
    quality_score: float
    latency_ms: float
    recall_estimate: float
    trade_off_analysis: Dict[str, Any]


class QdrantOptimizer:
    """Per-collection Qdrant ef_search optimizer"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.qdrant_config = config.get("qdrant", {})

        # Qdrant client
        self.client = QdrantClient(
            url=self.qdrant_config.get("url", "http://localhost:6333")
        )

        # Optimization parameters
        self.ef_search_range = self.qdrant_config.get("ef_search_range", [16, 32, 64, 96, 128, 192, 256])
        self.test_queries_count = self.qdrant_config.get("test_queries_count", 10)

        # Trade-off preferences
        self.latency_weight = self.qdrant_config.get("latency_weight", 0.4)
        self.recall_weight = self.qdrant_config.get("recall_weight", 0.6)
        self.target_latency_ms = self.qdrant_config.get("target_latency_ms", 100)

        # Collection-specific configurations
        self.collection_configs = self._load_collection_configs()

        # Optimization history
        self.optimization_logs = []

    def _load_collection_configs(self) -> Dict[str, Dict[str, Any]]:
        """Načtení per-collection konfigurací"""

        configs = self.qdrant_config.get("collection_configs", {})

        # Default configurations by collection type
        default_configs = {
            "documents_metadata": {
                "priority": "precision",
                "target_latency": 50,
                "min_ef_search": 32,
                "max_ef_search": 128
            },
            "documents_sections": {
                "priority": "balanced",
                "target_latency": 100,
                "min_ef_search": 64,
                "max_ef_search": 192
            },
            "documents_passages": {
                "priority": "recall",
                "target_latency": 150,
                "min_ef_search": 96,
                "max_ef_search": 256
            },
            "documents_sentences": {
                "priority": "speed",
                "target_latency": 30,
                "min_ef_search": 16,
                "max_ef_search": 64
            }
        }

        # Merge with custom configs
        for collection, default_config in default_configs.items():
            if collection in configs:
                configs[collection] = {**default_config, **configs[collection]}
            else:
                configs[collection] = default_config

        return configs

    async def optimize_all_collections(self) -> Dict[str, CollectionOptimization]:
        """Optimalizace všech dostupných kolekcí"""

        logger.info("Starting per-collection ef_search optimization...")

        # Získání seznamu kolekcí
        try:
            collections_info = self.client.get_collections()
            collection_names = [c.name for c in collections_info.collections]
        except Exception as e:
            logger.error(f"Failed to get collections: {e}")
            return {}

        logger.info(f"Found {len(collection_names)} collections: {collection_names}")

        optimizations = {}

        for collection_name in collection_names:
            try:
                optimization = await self.optimize_collection(collection_name)
                optimizations[collection_name] = optimization
                logger.info(f"Optimized {collection_name}: ef_search={optimization.optimal_ef_search}, "
                           f"latency={optimization.latency_ms:.1f}ms, recall={optimization.recall_estimate:.3f}")
            except Exception as e:
                logger.error(f"Failed to optimize collection {collection_name}: {e}")

        return optimizations

    async def optimize_collection(self, collection_name: str) -> CollectionOptimization:
        """Optimalizace jedné kolekce"""

        logger.info(f"Optimizing collection: {collection_name}")

        # Získání konfigurace pro kolekci
        collection_config = self.collection_configs.get(collection_name, {})
        priority = collection_config.get("priority", "balanced")

        # Generování test vectors
        test_vectors = await self._generate_test_vectors(collection_name)

        if not test_vectors:
            raise ValueError(f"No test vectors available for collection {collection_name}")

        # Testování různých ef_search hodnot
        search_results = []

        ef_range = self._get_ef_range_for_collection(collection_name)

        for ef_search in ef_range:
            logger.debug(f"Testing ef_search={ef_search} for {collection_name}")

            ef_results = []

            for i, test_vector in enumerate(test_vectors):
                result = await self._test_ef_search(
                    collection_name, ef_search, test_vector, f"test_{i}"
                )
                ef_results.append(result)

            # Agregace výsledků pro tento ef_search
            avg_result = self._aggregate_search_results(ef_results, ef_search)
            search_results.append(avg_result)

        # Nalezení optimálního ef_search
        optimal_result = self._find_optimal_ef_search(search_results, priority)

        # Trade-off analýza
        trade_off_analysis = self._analyze_trade_offs(search_results)

        optimization = CollectionOptimization(
            collection_name=collection_name,
            optimal_ef_search=optimal_result.ef_search,
            quality_score=optimal_result.search_quality_score,
            latency_ms=optimal_result.latency_ms,
            recall_estimate=optimal_result.recall_estimate,
            trade_off_analysis=trade_off_analysis
        )

        # Uložení do logu
        self.optimization_logs.append({
            "timestamp": time.time(),
            "collection_name": collection_name,
            "optimization": optimization,
            "all_results": search_results,
            "priority": priority
        })

        return optimization

    def _get_ef_range_for_collection(self, collection_name: str) -> List[int]:
        """Získání ef_search range pro kolekci"""

        collection_config = self.collection_configs.get(collection_name, {})
        min_ef = collection_config.get("min_ef_search", min(self.ef_search_range))
        max_ef = collection_config.get("max_ef_search", max(self.ef_search_range))

        return [ef for ef in self.ef_search_range if min_ef <= ef <= max_ef]

    async def _generate_test_vectors(self, collection_name: str) -> List[List[float]]:
        """Generování test vectors pro kolekci"""

        try:
            # Získání informací o kolekci
            collection_info = self.client.get_collection(collection_name)
            vector_size = collection_info.config.params.vectors.size

            # Generování random vectors (normalized)
            test_vectors = []
            np.random.seed(42)  # Reproducible results

            for _ in range(self.test_queries_count):
                vector = np.random.randn(vector_size)
                vector = vector / np.linalg.norm(vector)  # Normalize
                test_vectors.append(vector.tolist())

            return test_vectors

        except Exception as e:
            logger.error(f"Failed to generate test vectors for {collection_name}: {e}")
            return []

    async def _test_ef_search(self, collection_name: str, ef_search: int,
                             test_vector: List[float], query_id: str) -> SearchResult:
        """Test jedné ef_search hodnoty"""

        start_time = time.time()

        try:
            # Perform search
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=test_vector,
                limit=20,  # Standard limit for testing
                search_params=SearchParams(ef=ef_search)
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract scores and estimate quality
            scores = [hit.score for hit in search_results]
            results_count = len(search_results)

            # Estimate recall and precision (simplified heuristics)
            recall_estimate = self._estimate_recall(scores, ef_search)
            precision_estimate = self._estimate_precision(scores)

            # Overall quality score
            quality_score = self._calculate_quality_score(
                latency_ms, recall_estimate, precision_estimate, ef_search
            )

            return SearchResult(
                collection_name=collection_name,
                ef_search=ef_search,
                query_id=query_id,
                results_count=results_count,
                latency_ms=latency_ms,
                recall_estimate=recall_estimate,
                precision_estimate=precision_estimate,
                top_scores=scores[:10],  # Top 10 scores
                search_quality_score=quality_score
            )

        except Exception as e:
            logger.error(f"Search failed for {collection_name} with ef_search={ef_search}: {e}")

            # Return failed result
            return SearchResult(
                collection_name=collection_name,
                ef_search=ef_search,
                query_id=query_id,
                results_count=0,
                latency_ms=float('inf'),
                recall_estimate=0.0,
                precision_estimate=0.0,
                top_scores=[],
                search_quality_score=0.0
            )

    def _estimate_recall(self, scores: List[float], ef_search: int) -> float:
        """Odhad recall na základě skóre distribuce"""

        if not scores:
            return 0.0

        # Heuristic: higher ef_search generally means better recall
        # Also consider score distribution

        score_mean = np.mean(scores)
        score_std = np.std(scores) if len(scores) > 1 else 0

        # Base recall estimate from ef_search
        ef_factor = min(1.0, ef_search / 128.0)  # Normalize around 128

        # Score quality factor
        score_factor = min(1.0, score_mean)  # Scores should be 0-1

        # Score diversity factor (higher std = more diverse results)
        diversity_factor = min(1.0, score_std * 2)

        # Combined estimate
        recall_estimate = (ef_factor * 0.5 + score_factor * 0.3 + diversity_factor * 0.2)

        return max(0.0, min(1.0, recall_estimate))

    def _estimate_precision(self, scores: List[float]) -> float:
        """Odhad precision na základě top skóre"""

        if not scores:
            return 0.0

        # Take top 5 scores as precision estimate
        top_scores = sorted(scores, reverse=True)[:5]

        if top_scores:
            # Precision based on how high the top scores are
            precision_estimate = np.mean(top_scores)
            return max(0.0, min(1.0, precision_estimate))

        return 0.0

    def _calculate_quality_score(self, latency_ms: float, recall: float,
                                precision: float, ef_search: int) -> float:
        """Výpočet celkového quality score"""

        # Latency penalty (normalized to target latency)
        latency_penalty = max(0, (latency_ms - self.target_latency_ms) / self.target_latency_ms)
        latency_factor = max(0.1, 1.0 - latency_penalty)

        # Quality factors
        quality_factor = (recall * self.recall_weight + precision * (1 - self.recall_weight))

        # Combined score
        quality_score = quality_factor * latency_factor

        return max(0.0, min(1.0, quality_score))

    def _aggregate_search_results(self, results: List[SearchResult],
                                 ef_search: int) -> SearchResult:
        """Agregace výsledků z multiple test queries"""

        if not results:
            return SearchResult(
                collection_name="unknown",
                ef_search=ef_search,
                query_id="aggregated",
                results_count=0,
                latency_ms=float('inf'),
                recall_estimate=0.0,
                precision_estimate=0.0,
                top_scores=[],
                search_quality_score=0.0
            )

        # Průměrné hodnoty
        avg_latency = np.mean([r.latency_ms for r in results])
        avg_recall = np.mean([r.recall_estimate for r in results])
        avg_precision = np.mean([r.precision_estimate for r in results])
        avg_quality = np.mean([r.search_quality_score for r in results])
        avg_results_count = np.mean([r.results_count for r in results])

        # Kombinace top scores
        all_scores = []
        for r in results:
            all_scores.extend(r.top_scores)
        top_scores = sorted(all_scores, reverse=True)[:10]

        return SearchResult(
            collection_name=results[0].collection_name,
            ef_search=ef_search,
            query_id="aggregated",
            results_count=int(avg_results_count),
            latency_ms=avg_latency,
            recall_estimate=avg_recall,
            precision_estimate=avg_precision,
            top_scores=top_scores,
            search_quality_score=avg_quality
        )

    def _find_optimal_ef_search(self, results: List[SearchResult],
                               priority: str) -> SearchResult:
        """Nalezení optimálního ef_search podle priority"""

        if not results:
            raise ValueError("No search results to optimize")

        if priority == "speed":
            # Minimize latency while maintaining reasonable quality
            valid_results = [r for r in results if r.search_quality_score > 0.3]
            return min(valid_results, key=lambda x: x.latency_ms)

        elif priority == "precision":
            # Maximize precision
            return max(results, key=lambda x: x.precision_estimate)

        elif priority == "recall":
            # Maximize recall
            return max(results, key=lambda x: x.recall_estimate)

        else:  # balanced
            # Maximize overall quality score
            return max(results, key=lambda x: x.search_quality_score)

    def _analyze_trade_offs(self, results: List[SearchResult]) -> Dict[str, Any]:
        """Analýza latency/recall trade-offs"""

        if len(results) < 2:
            return {"error": "Insufficient data for trade-off analysis"}

        # Sort by ef_search for analysis
        sorted_results = sorted(results, key=lambda x: x.ef_search)

        # Extract metrics
        ef_values = [r.ef_search for r in sorted_results]
        latencies = [r.latency_ms for r in sorted_results]
        recalls = [r.recall_estimate for r in sorted_results]
        precisions = [r.precision_estimate for r in sorted_results]

        # Calculate correlations and trends
        latency_trend = np.polyfit(ef_values, latencies, 1)[0]  # Slope
        recall_trend = np.polyfit(ef_values, recalls, 1)[0]

        # Find knee points (best trade-offs)
        efficiency_scores = []
        for r in sorted_results:
            # Efficiency = quality / latency
            efficiency = r.search_quality_score / max(1, r.latency_ms / 100)  # Normalize latency
            efficiency_scores.append(efficiency)

        best_efficiency_idx = np.argmax(efficiency_scores)
        knee_point = sorted_results[best_efficiency_idx]

        return {
            "latency_trend_ms_per_ef": latency_trend,
            "recall_trend_per_ef": recall_trend,
            "knee_point": {
                "ef_search": knee_point.ef_search,
                "latency_ms": knee_point.latency_ms,
                "recall": knee_point.recall_estimate,
                "efficiency_score": efficiency_scores[best_efficiency_idx]
            },
            "latency_range": {
                "min": min(latencies),
                "max": max(latencies),
                "range": max(latencies) - min(latencies)
            },
            "recall_range": {
                "min": min(recalls),
                "max": max(recalls),
                "range": max(recalls) - min(recalls)
            },
            "efficiency_scores": dict(zip(ef_values, efficiency_scores))
        }

    def get_optimization_report(self) -> Dict[str, Any]:
        """Získání kompletního optimization reportu"""

        if not self.optimization_logs:
            return {"message": "No optimization data available"}

        # Agregace výsledků napříč kolekcemi
        collection_summaries = {}

        for log in self.optimization_logs:
            collection_name = log["collection_name"]
            optimization = log["optimization"]

            collection_summaries[collection_name] = {
                "optimal_ef_search": optimization.optimal_ef_search,
                "latency_ms": optimization.latency_ms,
                "recall_estimate": optimization.recall_estimate,
                "quality_score": optimization.quality_score,
                "trade_off_analysis": optimization.trade_off_analysis,
                "priority": log["priority"]
            }

        # Global recommendations
        recommendations = self._generate_global_recommendations(collection_summaries)

        return {
            "timestamp": time.time(),
            "total_collections_optimized": len(collection_summaries),
            "collection_summaries": collection_summaries,
            "global_recommendations": recommendations,
            "optimization_parameters": {
                "ef_search_range": self.ef_search_range,
                "latency_weight": self.latency_weight,
                "recall_weight": self.recall_weight,
                "target_latency_ms": self.target_latency_ms
            }
        }

    def _generate_global_recommendations(self, summaries: Dict[str, Any]) -> List[str]:
        """Generování globálních doporučení"""

        recommendations = []

        if not summaries:
            return ["No optimization data available"]

        # Analyze patterns across collections
        ef_values = [s["optimal_ef_search"] for s in summaries.values()]
        latencies = [s["latency_ms"] for s in summaries.values()]

        avg_ef = np.mean(ef_values)
        avg_latency = np.mean(latencies)

        if avg_latency > self.target_latency_ms * 1.5:
            recommendations.append(f"PERFORMANCE: Average latency ({avg_latency:.1f}ms) exceeds target ({self.target_latency_ms}ms). Consider lower ef_search values.")

        if avg_ef < 64:
            recommendations.append("QUALITY: Low ef_search values detected. Consider increasing for better recall.")
        elif avg_ef > 192:
            recommendations.append("EFFICIENCY: High ef_search values detected. Consider decreasing for better speed.")

        # Collection-specific patterns
        metadata_collections = [name for name in summaries.keys() if "metadata" in name]
        if metadata_collections:
            metadata_ef = np.mean([summaries[name]["optimal_ef_search"] for name in metadata_collections])
            recommendations.append(f"CONFIG: Metadata collections optimal ef_search: {metadata_ef:.0f}")

        passage_collections = [name for name in summaries.keys() if "passage" in name]
        if passage_collections:
            passage_ef = np.mean([summaries[name]["optimal_ef_search"] for name in passage_collections])
            recommendations.append(f"CONFIG: Passage collections optimal ef_search: {passage_ef:.0f}")

        return recommendations


# Factory function
def create_qdrant_optimizer(config: Dict[str, Any]) -> QdrantOptimizer:
    """Factory function pro Qdrant optimizer"""
    return QdrantOptimizer(config)
