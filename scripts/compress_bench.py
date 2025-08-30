#!/usr/bin/env python3
"""
FÁZE 2 Compression Benchmark
Benchmark script pro měření výkonu compression komponent

Author: Senior Python/MLOps Agent
"""

import asyncio
import time
import json
import argparse
import yaml
import logging
from typing import Dict, Any, List
from pathlib import Path
import psutil
import tracemalloc
from dataclasses import asdict

# Import FÁZE 2 components
from src.compress.enhanced_contextual_compression import EnhancedContextualCompressor
from src.compress.discourse_chunking import DiscourseChunker
from src.compress.gated_reranking import GatedReranker
from src.compress.phase2_integration import Phase2Integrator

logger = logging.getLogger(__name__)


class Phase2Benchmark:
    """Benchmark suite pro FÁZE 2 komponenty"""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.results = {}

        # Performance tracking
        self.memory_tracker = None
        self.start_memory = 0
        self.peak_memory = 0

    def _load_config(self) -> Dict[str, Any]:
        """Načtení konfigurace"""
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _generate_test_documents(self, size: str = "medium") -> List[Dict[str, Any]]:
        """Generování test dokumentů různých velikostí"""

        base_content = """
        Climate change represents one of the most significant challenges facing humanity in the 21st century. 
        Scientific research has demonstrated clear evidence of anthropogenic warming, with greenhouse gas 
        concentrations reaching unprecedented levels in the past 800,000 years. The Intergovernmental Panel 
        on Climate Change (IPCC) reports indicate that global temperatures have risen by approximately 1.1°C 
        since pre-industrial times, with the most rapid warming occurring in recent decades.
        
        The impacts of climate change are manifold and interconnected. Rising sea levels threaten coastal 
        communities worldwide, while changing precipitation patterns affect agricultural productivity and 
        water security. Extreme weather events, including heatwaves, droughts, floods, and intense storms, 
        are becoming more frequent and severe. These changes pose significant risks to human health, 
        economic stability, and ecological integrity.
        
        Mitigation strategies focus on reducing greenhouse gas emissions through renewable energy deployment, 
        energy efficiency improvements, and carbon pricing mechanisms. Adaptation measures aim to enhance 
        resilience to climate impacts through infrastructure upgrades, ecosystem restoration, and improved 
        disaster preparedness. International cooperation remains essential for addressing this global challenge.
        """

        sizes = {
            "small": (5, 200),  # 5 docs, 200 words each
            "medium": (20, 500),  # 20 docs, 500 words each
            "large": (50, 1000),  # 50 docs, 1000 words each
            "xlarge": (100, 1500),  # 100 docs, 1500 words each
        }

        doc_count, words_per_doc = sizes.get(size, sizes["medium"])

        documents = []
        source_types = ["academic", "news", "government", "wikipedia"]

        for i in range(doc_count):
            # Vary content to create realistic diversity
            content_multiplier = max(1, words_per_doc // len(base_content.split()))
            content = (base_content + f" Document {i} specific content. ") * content_multiplier
            content = " ".join(content.split()[:words_per_doc])  # Trim to exact word count

            doc = {
                "content": content,
                "source_type": source_types[i % len(source_types)],
                "metadata": {
                    "title": f"Climate Research Document {i}",
                    "year": 2020 + (i % 4),
                    "words": len(content.split()),
                },
            }
            documents.append(doc)

        return documents

    def _start_memory_tracking(self):
        """Spuštění memory tracking"""
        tracemalloc.start()
        process = psutil.Process()
        self.start_memory = process.memory_info().rss / 1024 / 1024  # MB

    def _stop_memory_tracking(self) -> Dict[str, float]:
        """Ukončení memory tracking a získání výsledků"""
        process = psutil.Process()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "start_memory_mb": self.start_memory,
            "end_memory_mb": end_memory,
            "memory_delta_mb": end_memory - self.start_memory,
            "peak_traced_mb": peak / 1024 / 1024,
            "current_traced_mb": current / 1024 / 1024,
        }

    async def benchmark_discourse_chunking(
        self, documents: List[Dict[str, Any]], query: str
    ) -> Dict[str, Any]:
        """Benchmark discourse chunking"""

        logger.info("Benchmarking Discourse Chunking...")

        chunker = DiscourseChunker(self.config)
        await chunker.initialize()

        self._start_memory_tracking()
        start_time = time.time()

        try:
            result = await chunker.chunk_documents(documents, query)

            end_time = time.time()
            memory_stats = self._stop_memory_tracking()

            benchmark_result = {
                "component": "discourse_chunking",
                "input_documents": len(documents),
                "input_total_words": sum(len(doc["content"].split()) for doc in documents),
                "output_chunks": len(result.chunks),
                "processing_time_seconds": end_time - start_time,
                "chunks_per_second": len(result.chunks) / (end_time - start_time),
                "words_per_second": sum(len(doc["content"].split()) for doc in documents)
                / (end_time - start_time),
                "memory_usage": memory_stats,
                "quality_metrics": result.quality_metrics,
                "strategy": result.strategy,
            }

            logger.info(f"Chunking: {len(result.chunks)} chunks in {end_time - start_time:.2f}s")
            return benchmark_result

        except Exception as e:
            logger.error(f"Chunking benchmark failed: {e}")
            return {"component": "discourse_chunking", "error": str(e)}

    async def benchmark_gated_reranking(
        self, documents: List[Dict[str, Any]], query: str
    ) -> Dict[str, Any]:
        """Benchmark gated re-ranking"""

        logger.info("Benchmarking Gated Re-ranking...")

        # First chunk documents to create passages
        chunker = DiscourseChunker(self.config)
        await chunker.initialize()
        chunk_result = await chunker.chunk_documents(documents, query)

        # Convert chunks to passages
        passages = []
        for chunk in chunk_result.chunks:
            passage = {
                "content": chunk.text,
                "source_type": chunk.source_type,
                "metadata": {"chunk_id": chunk.id},
                "initial_score": chunk.relevance_score,
            }
            passages.append(passage)

        reranker = GatedReranker(self.config)

        # Mock LLM comparison for benchmarking (avoid external API calls)
        async def mock_compare(p1, p2, q):
            await asyncio.sleep(0.01)  # Simulate processing time
            # Simple heuristic based on query terms
            query_terms = q.lower().split()
            score_a = sum(1 for term in query_terms if term in p1["content"].lower())
            score_b = sum(1 for term in query_terms if term in p2["content"].lower())

            if score_a > score_b:
                return {"winner": "A", "margin": 0.7, "rationale": "More query terms"}
            elif score_b > score_a:
                return {"winner": "B", "margin": 0.7, "rationale": "More query terms"}
            else:
                return {"winner": "EQUAL", "margin": 0.1, "rationale": "Similar relevance"}

        reranker._compare_passages_llm = mock_compare
        await reranker.initialize()

        self._start_memory_tracking()
        start_time = time.time()

        try:
            result = await reranker.rerank_passages(passages, query)

            end_time = time.time()
            memory_stats = self._stop_memory_tracking()

            benchmark_result = {
                "component": "gated_reranking",
                "input_passages": len(passages),
                "output_passages": len(result.ranked_passages),
                "processing_time_seconds": end_time - start_time,
                "passages_per_second": len(passages) / (end_time - start_time),
                "comparisons_made": result.comparison_count,
                "comparisons_per_second": result.comparison_count / (end_time - start_time),
                "memory_usage": memory_stats,
                "quality_metrics": result.quality_metrics,
                "strategy": result.strategy,
            }

            logger.info(f"Re-ranking: {len(passages)} passages in {end_time - start_time:.2f}s")
            return benchmark_result

        except Exception as e:
            logger.error(f"Re-ranking benchmark failed: {e}")
            return {"component": "gated_reranking", "error": str(e)}

    async def benchmark_enhanced_compression(
        self, documents: List[Dict[str, Any]], query: str
    ) -> Dict[str, Any]:
        """Benchmark enhanced compression"""

        logger.info("Benchmarking Enhanced Compression...")

        compressor = EnhancedContextualCompressor(self.config)
        await compressor.initialize()

        self._start_memory_tracking()
        start_time = time.time()

        try:
            result = await compressor.compress_context(documents, query)

            end_time = time.time()
            memory_stats = self._stop_memory_tracking()

            benchmark_result = {
                "component": "enhanced_compression",
                "input_units": len(result.original_units),
                "output_units": len(result.selected_units),
                "compression_ratio": result.compression_ratio,
                "token_budget_total": result.token_budget_total,
                "token_budget_used": result.token_budget_used,
                "token_efficiency": result.token_budget_used / result.token_budget_total,
                "processing_time_seconds": end_time - start_time,
                "units_per_second": len(result.original_units) / (end_time - start_time),
                "tokens_per_second": result.token_budget_used / (end_time - start_time),
                "memory_usage": memory_stats,
                "quality_metrics": result.quality_metrics,
                "strategy": result.compression_strategy,
            }

            logger.info(
                f"Compression: {result.compression_ratio:.1%} ratio in {end_time - start_time:.2f}s"
            )
            return benchmark_result

        except Exception as e:
            logger.error(f"Compression benchmark failed: {e}")
            return {"component": "enhanced_compression", "error": str(e)}

    async def benchmark_full_pipeline(
        self, documents: List[Dict[str, Any]], query: str
    ) -> Dict[str, Any]:
        """Benchmark celého Phase 2 pipeline"""

        logger.info("Benchmarking Full Phase 2 Pipeline...")

        integrator = Phase2Integrator(self.config)

        # Mock LLM for integration
        async def mock_compare(p1, p2, q):
            await asyncio.sleep(0.005)  # Faster for pipeline benchmark
            query_terms = q.lower().split()
            score_a = sum(1 for term in query_terms if term in p1["content"].lower())
            score_b = sum(1 for term in query_terms if term in p2["content"].lower())

            if score_a > score_b:
                return {"winner": "A", "margin": 0.6, "rationale": "Pipeline test"}
            elif score_b > score_a:
                return {"winner": "B", "margin": 0.6, "rationale": "Pipeline test"}
            else:
                return {"winner": "EQUAL", "margin": 0.1, "rationale": "Equal"}

        integrator.gated_reranker._compare_passages_llm = mock_compare
        await integrator.initialize()

        self._start_memory_tracking()
        start_time = time.time()

        try:
            result = await integrator.process_documents(documents, query)

            end_time = time.time()
            memory_stats = self._stop_memory_tracking()

            benchmark_result = {
                "component": "full_pipeline",
                "input_documents": len(documents),
                "final_compressed_units": len(result.compression_result.selected_units),
                "overall_compression_ratio": result.quality_metrics.get(
                    "overall_compression_ratio", 0
                ),
                "processing_time_seconds": end_time - start_time,
                "documents_per_second": len(documents) / (end_time - start_time),
                "memory_usage": memory_stats,
                "quality_metrics": result.quality_metrics,
                "pipeline_efficiency": result.pipeline_efficiency,
                "pipeline_steps": len(result.processing_log),
            }

            logger.info(
                f"Full pipeline: {len(documents)} docs → {len(result.compression_result.selected_units)} units in {end_time - start_time:.2f}s"
            )
            return benchmark_result

        except Exception as e:
            logger.error(f"Pipeline benchmark failed: {e}")
            return {"component": "full_pipeline", "error": str(e)}

    async def run_benchmark_suite(self, sizes: List[str] = None) -> Dict[str, Any]:
        """Spuštění kompletní benchmark suite"""

        if sizes is None:
            sizes = ["small", "medium", "large"]

        query = "What are the primary impacts of climate change on global ecosystems and human societies?"

        suite_results = {
            "benchmark_metadata": {
                "timestamp": time.time(),
                "config_path": self.config_path,
                "query": query,
                "sizes_tested": sizes,
            },
            "results": {},
        }

        for size in sizes:
            logger.info(f"\n=== Benchmarking size: {size} ===")

            documents = self._generate_test_documents(size)

            size_results = {
                "dataset_info": {
                    "size": size,
                    "document_count": len(documents),
                    "total_words": sum(len(doc["content"].split()) for doc in documents),
                    "avg_words_per_doc": sum(len(doc["content"].split()) for doc in documents)
                    / len(documents),
                },
                "component_benchmarks": {},
            }

            # Benchmark individual components
            chunking_result = await self.benchmark_discourse_chunking(documents, query)
            size_results["component_benchmarks"]["chunking"] = chunking_result

            reranking_result = await self.benchmark_gated_reranking(documents, query)
            size_results["component_benchmarks"]["reranking"] = reranking_result

            compression_result = await self.benchmark_enhanced_compression(documents, query)
            size_results["component_benchmarks"]["compression"] = compression_result

            # Benchmark full pipeline
            pipeline_result = await self.benchmark_full_pipeline(documents, query)
            size_results["component_benchmarks"]["pipeline"] = pipeline_result

            suite_results["results"][size] = size_results

            # Memory cleanup between sizes
            await asyncio.sleep(1)

        return suite_results

    def export_results(self, results: Dict[str, Any], output_path: str):
        """Export benchmark výsledků"""

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Benchmark results exported to {output_path}")

    def print_summary(self, results: Dict[str, Any]):
        """Tisk summary benchmark výsledků"""

        print("\n" + "=" * 80)
        print("FÁZE 2 COMPRESSION BENCHMARK SUMMARY")
        print("=" * 80)

        for size, size_results in results["results"].items():
            print(f"\nDataset Size: {size.upper()}")
            print("-" * 40)

            dataset_info = size_results["dataset_info"]
            print(f"Documents: {dataset_info['document_count']}")
            print(f"Total Words: {dataset_info['total_words']:,}")
            print(f"Avg Words/Doc: {dataset_info['avg_words_per_doc']:.0f}")

            print("\nComponent Performance:")

            for component, benchmark in size_results["component_benchmarks"].items():
                if "error" in benchmark:
                    print(f"  {component}: ERROR - {benchmark['error']}")
                    continue

                time_taken = benchmark.get("processing_time_seconds", 0)

                if component == "chunking":
                    throughput = benchmark.get("chunks_per_second", 0)
                    print(f"  {component}: {time_taken:.2f}s ({throughput:.1f} chunks/s)")
                elif component == "reranking":
                    throughput = benchmark.get("passages_per_second", 0)
                    comparisons = benchmark.get("comparisons_per_second", 0)
                    print(
                        f"  {component}: {time_taken:.2f}s ({throughput:.1f} passages/s, {comparisons:.1f} comparisons/s)"
                    )
                elif component == "compression":
                    ratio = benchmark.get("compression_ratio", 0)
                    efficiency = benchmark.get("token_efficiency", 0)
                    print(
                        f"  {component}: {time_taken:.2f}s ({ratio:.1%} compression, {efficiency:.1%} token efficiency)"
                    )
                elif component == "pipeline":
                    docs_per_sec = benchmark.get("documents_per_second", 0)
                    overall_ratio = benchmark.get("overall_compression_ratio", 0)
                    print(
                        f"  {component}: {time_taken:.2f}s ({docs_per_sec:.1f} docs/s, {overall_ratio:.1%} overall compression)"
                    )

                # Memory usage
                memory = benchmark.get("memory_usage", {})
                if memory:
                    peak_mb = memory.get("peak_traced_mb", 0)
                    print(f"    Memory: {peak_mb:.1f} MB peak")


async def main():
    """Main benchmark runner"""

    parser = argparse.ArgumentParser(description="FÁZE 2 Compression Benchmark")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["small", "medium", "large"],
        choices=["small", "medium", "large", "xlarge"],
        help="Dataset sizes to benchmark",
    )
    parser.add_argument(
        "--output", default="benchmark_results_phase2.json", help="Output file for results"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    # Run benchmark
    benchmark = Phase2Benchmark(args.config)

    logger.info("Starting FÁZE 2 Compression Benchmark Suite...")
    logger.info(f"Sizes to test: {args.sizes}")
    logger.info(f"Output file: {args.output}")

    results = await benchmark.run_benchmark_suite(args.sizes)

    # Export and summarize
    benchmark.export_results(results, args.output)
    benchmark.print_summary(results)

    logger.info("Benchmark completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
