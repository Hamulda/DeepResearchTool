#!/usr/bin/env python3
"""
Enhanced Retrieval Engine - FÁZE 1 Integration
Integruje HyDE, MMR, Enhanced RRF a Qdrant optimization

Author: Senior Python/MLOps Agent
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import time
import numpy as np

# FÁZE 1 imports
from .hyde_expansion import HyDEQueryExpander, HyDEResult
from .mmr_diversification import MMRDiversifier, MMRResult
from .enhanced_rrf import EnhancedRRFEngine
from .qdrant_optimizer import QdrantOptimizer

logger = logging.getLogger(__name__)


@dataclass
class EnhancedRetrievalResult:
    """Enhanced výsledek retrievalu s FÁZE 1 metrikami"""

    results: List[Dict[str, Any]]
    hyde_expansion: Optional[HyDEResult]
    mmr_diversification: List[MMRResult]
    rrf_fusion_metrics: Dict[str, Any]
    qdrant_performance: Dict[str, Any]
    total_processing_time: float
    phase1_metrics: Dict[str, Any]


class EnhancedRetrievalEngine:
    """Enhanced Retrieval Engine s FÁZE 1 funkcionalitou"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.retrieval_config = config.get("retrieval", {})

        # FÁZE 1 komponenty
        self.hyde_expander = None
        self.mmr_diversifier = None
        self.enhanced_rrf = None
        self.qdrant_optimizer = None

        # Flags pro povolení funkcí
        self.hyde_enabled = self.retrieval_config.get("hyde", {}).get("enabled", False)
        self.mmr_enabled = self.retrieval_config.get("mmr", {}).get("enabled", False)
        self.enhanced_rrf_enabled = True  # Always enabled in FÁZE 1

        # Performance tracking
        self.performance_logs = []

    async def initialize(self, llm_client=None, embedding_model=None):
        """Inicializace všech FÁZE 1 komponent"""

        logger.info("Initializing Enhanced Retrieval Engine (FÁZE 1)...")
        start_time = time.time()

        try:
            # HyDE Query Expander
            if self.hyde_enabled:
                self.hyde_expander = HyDEQueryExpander(self.config)
                if llm_client:
                    await self.hyde_expander.initialize(llm_client)
                logger.info("✅ HyDE Query Expander initialized")

            # MMR Diversifier
            if self.mmr_enabled:
                self.mmr_diversifier = MMRDiversifier(self.config)
                if embedding_model:
                    await self.mmr_diversifier.initialize(embedding_model)
                logger.info("✅ MMR Diversifier initialized")

            # Enhanced RRF
            self.enhanced_rrf = EnhancedRRFEngine(self.config)
            logger.info("✅ Enhanced RRF Engine initialized")

            # Qdrant Optimizer
            self.qdrant_optimizer = QdrantOptimizer(self.config)
            logger.info("✅ Qdrant Optimizer initialized")

            init_time = time.time() - start_time
            logger.info(f"Enhanced Retrieval Engine initialized in {init_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to initialize Enhanced Retrieval Engine: {e}")
            raise

    async def enhanced_retrieve(
        self,
        query: str,
        domain: str = "general",
        k_results: int = 20,
        retrieval_sources: List[str] = None,
    ) -> EnhancedRetrievalResult:
        """
        Enhanced retrieval s kompletní FÁZE 1 pipeline

        Pipeline:
        1. HyDE query expansion (optional)
        2. Multi-source retrieval
        3. Enhanced RRF fusion s per-source priors
        4. MMR diversification
        5. Qdrant performance monitoring
        """

        start_time = time.time()
        logger.info(f"Starting enhanced retrieval for query: {query[:100]}...")

        # STEP 1: HyDE Query Expansion
        hyde_result = None
        effective_query = query

        if self.hyde_enabled and self.hyde_expander:
            logger.debug("Applying HyDE query expansion...")
            hyde_start = time.time()

            hyde_result = await self.hyde_expander.expand_query(query, domain)
            effective_query = hyde_result.expanded_query if not hyde_result.fallback_used else query

            hyde_time = time.time() - hyde_start
            logger.info(
                f"HyDE expansion completed in {hyde_time:.2f}s, "
                f"fallback: {hyde_result.fallback_used}, "
                f"confidence: {hyde_result.confidence_score:.2f}"
            )

        # STEP 2: Multi-source Retrieval
        logger.debug("Executing multi-source retrieval...")
        retrieval_start = time.time()

        # Zde by se volaly skutečné retrieval zdroje
        # Pro demonstraci použiju mock data
        retrieval_results = await self._execute_multi_source_retrieval(
            effective_query, domain, retrieval_sources
        )

        retrieval_time = time.time() - retrieval_start
        logger.info(
            f"Multi-source retrieval completed in {retrieval_time:.2f}s, "
            f"sources: {len(retrieval_results)}"
        )

        # STEP 3: Enhanced RRF Fusion
        logger.debug("Applying Enhanced RRF fusion...")
        rrf_start = time.time()

        rrf_result = await self.enhanced_rrf.enhanced_rrf_fusion(
            retrieval_results, query_domain=domain
        )

        rrf_time = time.time() - rrf_start
        logger.info(f"Enhanced RRF fusion completed in {rrf_time:.2f}s")

        # STEP 4: MMR Diversification
        mmr_results = []
        if self.mmr_enabled and self.mmr_diversifier:
            logger.debug("Applying MMR diversification...")
            mmr_start = time.time()

            # Extract embeddings for MMR (mock data pro demo)
            query_embedding = np.random.rand(384)  # Mock embedding

            mmr_results = await self.mmr_diversifier.diversify_results(
                rrf_result["results"], query_embedding, k_results
            )

            mmr_time = time.time() - mmr_start
            logger.info(
                f"MMR diversification completed in {mmr_time:.2f}s, "
                f"diverse selections: {sum(1 for r in mmr_results if r.selected_for_diversity)}"
            )

        # STEP 5: Qdrant Performance Monitoring
        qdrant_performance = await self._monitor_qdrant_performance()

        # Final results
        if mmr_results:
            # Use MMR-diversified results
            final_results = [
                {
                    **rrf_result["results"][mmr_r.original_rank],
                    "mmr_rank": mmr_r.mmr_rank,
                    "diversity_score": mmr_r.diversity_score,
                    "mmr_score": mmr_r.mmr_score,
                }
                for mmr_r in mmr_results[:k_results]
            ]
        else:
            # Use RRF results
            final_results = rrf_result["results"][:k_results]

        # STEP 6: Comprehensive Metrics
        total_time = time.time() - start_time

        phase1_metrics = {
            "hyde_enabled": self.hyde_enabled,
            "mmr_enabled": self.mmr_enabled,
            "hyde_processing_time": hyde_time if hyde_result else 0,
            "retrieval_processing_time": retrieval_time,
            "rrf_processing_time": rrf_time,
            "mmr_processing_time": mmr_time if mmr_results else 0,
            "total_processing_time": total_time,
            "pipeline_efficiency": k_results / total_time if total_time > 0 else 0,
            "query_expansion_ratio": len(effective_query) / len(query) if query else 1,
            "final_result_count": len(final_results),
        }

        # Log performance
        self.performance_logs.append(
            {
                "timestamp": time.time(),
                "query": query[:100],
                "domain": domain,
                "metrics": phase1_metrics,
                "hyde_used": hyde_result is not None and not hyde_result.fallback_used,
                "mmr_used": len(mmr_results) > 0,
            }
        )

        logger.info(
            f"Enhanced retrieval completed in {total_time:.2f}s, "
            f"returned {len(final_results)} results"
        )

        return EnhancedRetrievalResult(
            results=final_results,
            hyde_expansion=hyde_result,
            mmr_diversification=mmr_results,
            rrf_fusion_metrics=rrf_result["metrics"],
            qdrant_performance=qdrant_performance,
            total_processing_time=total_time,
            phase1_metrics=phase1_metrics,
        )

    async def _execute_multi_source_retrieval(
        self, query: str, domain: str, sources: List[str] = None
    ) -> List[List[Dict[str, Any]]]:
        """Mock multi-source retrieval pro demonstraci"""

        # V reálné implementaci by se zde volaly skutečné retrieval enginy
        # Pro FÁZE 1 demo vytvořím mock data

        if sources is None:
            sources = ["qdrant_semantic", "bm25_sparse", "academic_db"]

        retrieval_results = []

        for source in sources:
            # Mock results for each source
            source_results = []

            for i in range(20):  # 20 results per source
                mock_result = {
                    "id": f"{source}_doc_{i}",
                    "content": f"Mock content for {query} from {source} - document {i}",
                    "score": np.random.uniform(0.3, 0.9),
                    "source_connector": source,
                    "canonical_url": f"https://example.com/{source}/{i}",
                    "title": f"Document {i} from {source}",
                    "metadata": {
                        "publication_date": "2024-01-01",
                        "domain": domain,
                        "source_type": self._map_source_to_type(source),
                    },
                }
                source_results.append(mock_result)

            # Sort by score
            source_results.sort(key=lambda x: x["score"], reverse=True)
            retrieval_results.append(source_results)

        return retrieval_results

    def _map_source_to_type(self, source: str) -> str:
        """Mapování zdroje na typ pro RRF priors"""

        mapping = {
            "academic_db": "academic",
            "news_api": "news",
            "government_db": "government",
            "social_api": "social_media",
            "wikipedia": "wikipedia",
        }

        for pattern, source_type in mapping.items():
            if pattern in source.lower():
                return source_type

        return "wikipedia"  # Default

    async def _monitor_qdrant_performance(self) -> Dict[str, Any]:
        """Monitoring Qdrant performance"""

        try:
            # V reálné implementaci by se zde měřil skutečný výkon
            # Pro demo vrátím mock metriky

            return {
                "collections_monitored": ["documents_metadata", "documents_passages"],
                "average_latency_ms": np.random.uniform(20, 100),
                "current_ef_search": 64,
                "estimated_recall": np.random.uniform(0.7, 0.9),
                "monitoring_timestamp": time.time(),
            }

        except Exception as e:
            logger.warning(f"Qdrant performance monitoring failed: {e}")
            return {"error": str(e)}

    async def optimize_qdrant_collections(self) -> Dict[str, Any]:
        """Spuštění Qdrant optimalizace"""

        if not self.qdrant_optimizer:
            return {"error": "Qdrant optimizer not initialized"}

        try:
            logger.info("Starting Qdrant collections optimization...")
            optimizations = await self.qdrant_optimizer.optimize_all_collections()

            optimization_report = self.qdrant_optimizer.get_optimization_report()

            logger.info(
                f"Qdrant optimization completed: {len(optimizations)} collections optimized"
            )

            return {
                "optimizations": optimizations,
                "report": optimization_report,
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Qdrant optimization failed: {e}")
            return {"error": str(e), "status": "failed"}

    def get_performance_analysis(self) -> Dict[str, Any]:
        """Analýza výkonu FÁZE 1 funkcí"""

        if not self.performance_logs:
            return {"message": "No performance data available"}

        # Aggregate metrics
        total_queries = len(self.performance_logs)

        hyde_usage = sum(1 for log in self.performance_logs if log["hyde_used"])
        mmr_usage = sum(1 for log in self.performance_logs if log["mmr_used"])

        avg_total_time = np.mean(
            [log["metrics"]["total_processing_time"] for log in self.performance_logs]
        )
        avg_hyde_time = np.mean(
            [
                log["metrics"]["hyde_processing_time"]
                for log in self.performance_logs
                if log["hyde_used"]
            ]
        )
        avg_mmr_time = np.mean(
            [
                log["metrics"]["mmr_processing_time"]
                for log in self.performance_logs
                if log["mmr_used"]
            ]
        )

        # Domain analysis
        domain_performance = {}
        for log in self.performance_logs:
            domain = log["domain"]
            if domain not in domain_performance:
                domain_performance[domain] = []
            domain_performance[domain].append(log["metrics"]["total_processing_time"])

        domain_stats = {}
        for domain, times in domain_performance.items():
            domain_stats[domain] = {"avg_time": np.mean(times), "query_count": len(times)}

        return {
            "total_queries_processed": total_queries,
            "hyde_usage_rate": hyde_usage / total_queries if total_queries > 0 else 0,
            "mmr_usage_rate": mmr_usage / total_queries if total_queries > 0 else 0,
            "performance_metrics": {
                "avg_total_processing_time": avg_total_time,
                "avg_hyde_processing_time": avg_hyde_time if hyde_usage > 0 else 0,
                "avg_mmr_processing_time": avg_mmr_time if mmr_usage > 0 else 0,
            },
            "domain_performance": domain_stats,
            "feature_utilization": {
                "hyde_enabled": self.hyde_enabled,
                "mmr_enabled": self.mmr_enabled,
                "enhanced_rrf_enabled": self.enhanced_rrf_enabled,
            },
        }

    async def run_phase1_benchmark(self) -> Dict[str, Any]:
        """Spuštění kompletního FÁZE 1 benchmarku"""

        logger.info("Starting FÁZE 1 comprehensive benchmark...")

        test_queries = [
            ("quantum computing error correction", "science"),
            ("climate change adaptation strategies", "environment"),
            ("artificial intelligence safety research", "technology"),
            ("renewable energy storage solutions", "technology"),
            ("medical imaging deep learning", "medicine"),
        ]

        benchmark_results = []

        for query, domain in test_queries:
            logger.info(f"Benchmarking query: {query}")

            try:
                result = await self.enhanced_retrieve(query, domain, k_results=10)

                benchmark_results.append(
                    {
                        "query": query,
                        "domain": domain,
                        "success": True,
                        "metrics": result.phase1_metrics,
                        "hyde_used": result.hyde_expansion is not None
                        and not result.hyde_expansion.fallback_used,
                        "mmr_diversification_rate": (
                            len([r for r in result.mmr_diversification if r.selected_for_diversity])
                            / len(result.mmr_diversification)
                            if result.mmr_diversification
                            else 0
                        ),
                    }
                )

            except Exception as e:
                logger.error(f"Benchmark failed for query '{query}': {e}")
                benchmark_results.append(
                    {"query": query, "domain": domain, "success": False, "error": str(e)}
                )

        # Aggregate benchmark metrics
        successful_results = [r for r in benchmark_results if r["success"]]

        if successful_results:
            avg_processing_time = np.mean(
                [r["metrics"]["total_processing_time"] for r in successful_results]
            )
            avg_pipeline_efficiency = np.mean(
                [r["metrics"]["pipeline_efficiency"] for r in successful_results]
            )
            hyde_success_rate = sum(1 for r in successful_results if r["hyde_used"]) / len(
                successful_results
            )

            benchmark_summary = {
                "total_queries": len(test_queries),
                "successful_queries": len(successful_results),
                "success_rate": len(successful_results) / len(test_queries),
                "avg_processing_time": avg_processing_time,
                "avg_pipeline_efficiency": avg_pipeline_efficiency,
                "hyde_success_rate": hyde_success_rate,
                "results": benchmark_results,
            }
        else:
            benchmark_summary = {
                "total_queries": len(test_queries),
                "successful_queries": 0,
                "success_rate": 0,
                "error": "All benchmark queries failed",
                "results": benchmark_results,
            }

        logger.info(
            f"FÁZE 1 benchmark completed: {benchmark_summary['success_rate']*100:.1f}% success rate"
        )

        return benchmark_summary


# Factory function
def create_enhanced_retrieval_engine(config: Dict[str, Any]) -> EnhancedRetrievalEngine:
    """Factory function pro Enhanced Retrieval Engine"""
    return EnhancedRetrievalEngine(config)
