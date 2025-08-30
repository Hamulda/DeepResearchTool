#!/usr/bin/env python3
"""
FÁZE 2 Integration Module
Integrace všech FÁZE 2 komponent: pairwise re-ranking, discourse chunking, enhanced compression

Author: Senior Python/MLOps Agent
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import json
import time

from .gated_reranking import GatedReranker, RerankingResult
from .discourse_chunking import DiscourseChunker, ChunkingResult
from .enhanced_contextual_compression import EnhancedContextualCompressor, CompressionResult

logger = logging.getLogger(__name__)


@dataclass
class Phase2ProcessingResult:
    """Výsledek kompletního FÁZE 2 zpracování"""

    # Input data
    original_documents: List[Dict[str, Any]]
    query: str

    # Processing results
    chunking_result: ChunkingResult
    reranking_result: RerankingResult
    compression_result: CompressionResult

    # Integration metrics
    processing_time: float
    quality_metrics: Dict[str, float]
    pipeline_efficiency: Dict[str, float]

    # Audit trail
    processing_log: List[Dict[str, Any]]


class Phase2Integrator:
    """Integrátor pro všechny FÁZE 2 komponenty"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.phase2_config = config.get("phase2", {})

        # Initialize components
        self.discourse_chunker = DiscourseChunker(config)
        self.gated_reranker = GatedReranker(config)
        self.enhanced_compressor = EnhancedContextualCompressor(config)

        # Pipeline settings
        self.pipeline_config = self.phase2_config.get("pipeline", {})
        self.enable_parallel_processing = self.pipeline_config.get("parallel_processing", True)
        self.quality_thresholds = self.pipeline_config.get("quality_thresholds", {})

        # Audit settings
        self.enable_audit_logging = self.phase2_config.get("audit_logging", True)
        self.processing_log = []

    async def initialize(self):
        """Inicializace všech komponent"""

        logger.info("Initializing Phase 2 Integration Pipeline...")

        try:
            # Initialize components in parallel for efficiency
            if self.enable_parallel_processing:
                await asyncio.gather(
                    self.discourse_chunker.initialize(),
                    self.gated_reranker.initialize(),
                    self.enhanced_compressor.initialize(),
                )
            else:
                await self.discourse_chunker.initialize()
                await self.gated_reranker.initialize()
                await self.enhanced_compressor.initialize()

            logger.info("✅ Phase 2 Integration Pipeline initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Phase 2 pipeline: {e}")
            raise

    async def process_documents(
        self, documents: List[Dict[str, Any]], query: str, token_budget: Optional[int] = None
    ) -> Phase2ProcessingResult:
        """
        Kompletní FÁZE 2 zpracování dokumentů

        Pipeline: Discourse Chunking → Pairwise Re-ranking → Enhanced Compression

        Args:
            documents: Seznam dokumentů s metadata
            query: Výzkumný dotaz
            token_budget: Token budget pro kompresi

        Returns:
            Phase2ProcessingResult s výsledky všech kroků
        """

        start_time = time.time()
        self.processing_log = []

        logger.info(f"Starting Phase 2 processing for {len(documents)} documents")

        try:
            # STEP 1: Discourse-aware chunking
            chunking_start = time.time()
            chunking_result = await self._step1_discourse_chunking(documents, query)
            chunking_time = time.time() - chunking_start

            self._log_processing_step(
                "discourse_chunking",
                {
                    "input_documents": len(documents),
                    "output_chunks": len(chunking_result.chunks),
                    "processing_time": chunking_time,
                    "quality_metrics": chunking_result.quality_metrics,
                },
            )

            # STEP 2: Pairwise re-ranking
            reranking_start = time.time()
            reranking_result = await self._step2_pairwise_reranking(chunking_result, query)
            reranking_time = time.time() - reranking_start

            self._log_processing_step(
                "pairwise_reranking",
                {
                    "input_chunks": len(chunking_result.chunks),
                    "reranked_chunks": len(reranking_result.ranked_passages),
                    "processing_time": reranking_time,
                    "quality_metrics": reranking_result.quality_metrics,
                },
            )

            # STEP 3: Enhanced contextual compression
            compression_start = time.time()
            compression_result = await self._step3_enhanced_compression(
                reranking_result, query, token_budget
            )
            compression_time = time.time() - compression_start

            self._log_processing_step(
                "enhanced_compression",
                {
                    "input_units": len(compression_result.original_units),
                    "selected_units": len(compression_result.selected_units),
                    "processing_time": compression_time,
                    "quality_metrics": compression_result.quality_metrics,
                },
            )

            # STEP 4: Calculate integration metrics
            total_time = time.time() - start_time
            quality_metrics = self._calculate_integration_quality_metrics(
                chunking_result, reranking_result, compression_result
            )
            pipeline_efficiency = self._calculate_pipeline_efficiency(
                chunking_time, reranking_time, compression_time, total_time
            )

            # STEP 5: Validate quality thresholds
            await self._validate_quality_thresholds(quality_metrics)

            # Create final result
            result = Phase2ProcessingResult(
                original_documents=documents,
                query=query,
                chunking_result=chunking_result,
                reranking_result=reranking_result,
                compression_result=compression_result,
                processing_time=total_time,
                quality_metrics=quality_metrics,
                pipeline_efficiency=pipeline_efficiency,
                processing_log=self.processing_log.copy(),
            )

            logger.info(f"Phase 2 processing completed in {total_time:.2f}s")
            logger.info(f"Pipeline efficiency: {pipeline_efficiency['overall_efficiency']:.1%}")

            return result

        except Exception as e:
            logger.error(f"Phase 2 processing failed: {e}")
            raise

    async def _step1_discourse_chunking(
        self, documents: List[Dict[str, Any]], query: str
    ) -> ChunkingResult:
        """STEP 1: Discourse-aware chunking"""

        logger.info("Phase 2 Step 1: Discourse-aware chunking")

        return await self.discourse_chunker.chunk_documents(documents, query)

    async def _step2_pairwise_reranking(
        self, chunking_result: ChunkingResult, query: str
    ) -> RerankingResult:
        """STEP 2: Pairwise re-ranking"""

        logger.info("Phase 2 Step 2: Pairwise re-ranking")

        # Convert chunks to reranking format
        passages = []
        for chunk in chunking_result.chunks:
            passage = {
                "content": chunk.text,
                "source_type": chunk.source_type,
                "metadata": {
                    "chunk_id": chunk.id,
                    "chunk_type": chunk.chunk_type,
                    "discourse_features": chunk.discourse_features,
                    "structural_info": chunk.structural_info,
                },
                "initial_score": chunk.relevance_score,
            }
            passages.append(passage)

        return await self.gated_reranker.rerank_passages(passages, query)

    async def _step3_enhanced_compression(
        self, reranking_result: RerankingResult, query: str, token_budget: Optional[int]
    ) -> CompressionResult:
        """STEP 3: Enhanced contextual compression"""

        logger.info("Phase 2 Step 3: Enhanced contextual compression")

        # Convert ranked passages to compression format
        texts = []
        for passage in reranking_result.ranked_passages:
            text_data = {
                "content": passage["content"],
                "source_type": passage.get("source_type", "unknown"),
                "metadata": passage.get("metadata", {}),
                "reranking_score": passage["reranking_score"],
            }
            texts.append(text_data)

        return await self.enhanced_compressor.compress_context(texts, query, token_budget)

    def _calculate_integration_quality_metrics(
        self,
        chunking_result: ChunkingResult,
        reranking_result: RerankingResult,
        compression_result: CompressionResult,
    ) -> Dict[str, float]:
        """Výpočet integrovaných quality metrics"""

        metrics = {}

        # Pipeline preservation metrics
        metrics["information_preservation"] = (
            chunking_result.quality_metrics.get("coherence_score", 0.8) * 0.3
            + reranking_result.quality_metrics.get("ranking_quality", 0.8) * 0.3
            + compression_result.quality_metrics.get("salience_preservation", 0.8) * 0.4
        )

        # Content quality metrics
        metrics["content_coherence"] = chunking_result.quality_metrics.get("coherence_score", 0.8)
        metrics["ranking_quality"] = reranking_result.quality_metrics.get("ranking_quality", 0.8)
        metrics["compression_efficiency"] = compression_result.quality_metrics.get(
            "context_usage_efficiency", 0.8
        )

        # Coverage metrics
        metrics["entity_coverage"] = compression_result.quality_metrics.get("entity_coverage", 0.8)
        metrics["claims_coverage"] = compression_result.quality_metrics.get("claims_coverage", 0.8)
        metrics["query_relevance"] = compression_result.quality_metrics.get(
            "query_relevance_preservation", 0.8
        )

        # Integration-specific metrics
        original_count = len(chunking_result.chunks)
        final_count = len(compression_result.selected_units)
        metrics["overall_compression_ratio"] = (
            final_count / original_count if original_count > 0 else 0
        )

        # Citation precision (from reranking)
        metrics["citation_precision"] = reranking_result.quality_metrics.get(
            "confidence_calibration", 0.8
        )

        return metrics

    def _calculate_pipeline_efficiency(
        self,
        chunking_time: float,
        reranking_time: float,
        compression_time: float,
        total_time: float,
    ) -> Dict[str, float]:
        """Výpočet pipeline efficiency metrics"""

        efficiency = {}

        # Time distribution
        efficiency["chunking_time_ratio"] = chunking_time / total_time
        efficiency["reranking_time_ratio"] = reranking_time / total_time
        efficiency["compression_time_ratio"] = compression_time / total_time

        # Efficiency scores
        processing_time = chunking_time + reranking_time + compression_time
        efficiency["processing_overhead"] = (total_time - processing_time) / total_time
        efficiency["overall_efficiency"] = processing_time / total_time

        # Performance metrics
        efficiency["throughput_chunks_per_second"] = 1.0 / chunking_time if chunking_time > 0 else 0
        efficiency["throughput_reranking_per_second"] = (
            1.0 / reranking_time if reranking_time > 0 else 0
        )
        efficiency["throughput_compression_per_second"] = (
            1.0 / compression_time if compression_time > 0 else 0
        )

        return efficiency

    async def _validate_quality_thresholds(self, quality_metrics: Dict[str, float]):
        """Validace quality thresholds - fail-hard při nesplnění"""

        thresholds = {
            "information_preservation": self.quality_thresholds.get(
                "information_preservation", 0.7
            ),
            "citation_precision": self.quality_thresholds.get("citation_precision", 0.8),
            "entity_coverage": self.quality_thresholds.get("entity_coverage", 0.6),
            "query_relevance": self.quality_thresholds.get("query_relevance", 0.7),
        }

        failed_thresholds = []

        for metric, threshold in thresholds.items():
            if metric in quality_metrics:
                actual_value = quality_metrics[metric]
                if actual_value < threshold:
                    failed_thresholds.append(
                        {
                            "metric": metric,
                            "threshold": threshold,
                            "actual": actual_value,
                            "deficit": threshold - actual_value,
                        }
                    )

        if failed_thresholds:
            error_msg = "Quality thresholds validation failed:\n"
            for failure in failed_thresholds:
                error_msg += f"  - {failure['metric']}: {failure['actual']:.3f} < {failure['threshold']:.3f} (deficit: {failure['deficit']:.3f})\n"

            logger.error(error_msg)

            # Fail-hard mode - raise exception
            if self.phase2_config.get("fail_hard_on_quality", True):
                raise ValueError(f"Phase 2 quality validation failed: {error_msg}")
            else:
                logger.warning("Quality thresholds failed but fail-hard disabled")

    def _log_processing_step(self, step_name: str, step_data: Dict[str, Any]):
        """Logování processing step pro audit"""

        if not self.enable_audit_logging:
            return

        log_entry = {"step": step_name, "timestamp": time.time(), "data": step_data}

        self.processing_log.append(log_entry)

        logger.info(f"Phase 2 Step {step_name}: {step_data}")

    def get_integration_report(self, result: Phase2ProcessingResult) -> Dict[str, Any]:
        """Generování integration report pro audit"""

        report = {
            "phase2_integration_summary": {
                "query": result.query,
                "input_documents": len(result.original_documents),
                "processing_time": f"{result.processing_time:.2f}s",
                "pipeline_steps": len(result.processing_log),
            },
            "pipeline_results": {
                "chunking": {
                    "input_documents": len(result.original_documents),
                    "output_chunks": len(result.chunking_result.chunks),
                    "quality_metrics": result.chunking_result.quality_metrics,
                },
                "reranking": {
                    "input_passages": len(result.chunking_result.chunks),
                    "reranked_passages": len(result.reranking_result.ranked_passages),
                    "quality_metrics": result.reranking_result.quality_metrics,
                },
                "compression": {
                    "input_units": len(result.compression_result.original_units),
                    "selected_units": len(result.compression_result.selected_units),
                    "compression_ratio": f"{result.compression_result.compression_ratio:.1%}",
                    "token_usage": f"{result.compression_result.token_budget_used}/{result.compression_result.token_budget_total}",
                    "quality_metrics": result.compression_result.quality_metrics,
                },
            },
            "integration_quality": result.quality_metrics,
            "pipeline_efficiency": result.pipeline_efficiency,
            "processing_log": result.processing_log,
        }

        return report

    def get_compressed_content(self, result: Phase2ProcessingResult) -> str:
        """Získání finálního komprimovaného obsahu"""

        return self.enhanced_compressor.get_compressed_text(result.compression_result)

    def export_audit_trail(self, result: Phase2ProcessingResult, output_path: str):
        """Export audit trail do JSON souboru"""

        audit_data = {
            "phase2_processing_audit": {
                "metadata": {
                    "query": result.query,
                    "timestamp": time.time(),
                    "processing_time": result.processing_time,
                    "input_documents": len(result.original_documents),
                },
                "pipeline_results": self.get_integration_report(result),
                "detailed_logs": result.processing_log,
                "component_reports": {
                    "chunking": self.discourse_chunker.get_chunking_report(result.chunking_result),
                    "reranking": self.gated_reranker.get_reranking_report(result.reranking_result),
                    "compression": self.enhanced_compressor.get_compression_report(
                        result.compression_result
                    ),
                },
            }
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(audit_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Phase 2 audit trail exported to {output_path}")
