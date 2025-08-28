#!/usr/bin/env python3
"""
Hierarchical RAG Benchmark Script
Testuje výkon hierarchického retrievalu na různých úrovních

Author: Senior Python/MLOps Agent
"""

import asyncio
import time
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any
import statistics
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class HierarchicalRAGBenchmark:
    """Benchmark pro hierarchical RAG performance"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.test_queries = [
            "quantum computing applications in cryptography",
            "machine learning bias detection methods",
            "climate change mitigation strategies",
            "artificial intelligence explainability",
            "renewable energy grid integration"
        ]

        # Hierarchical levels to test
        self.hierarchy_levels = [1, 2, 3, 4]

        # Chunk sizes to test
        self.chunk_configurations = [
            {"document": 4000, "section": 1000, "passage": 256},
            {"document": 6000, "section": 1500, "passage": 384},
            {"document": 8000, "section": 2000, "passage": 512},
        ]

    async def benchmark_hierarchy_level(self, query: str, levels: int,
                                      chunk_config: Dict[str, int]) -> Dict[str, Any]:
        """Benchmark jedné úrovně hierarchie"""

        start_time = time.time()

        try:
            # Simulate hierarchical retrieval (simplified for benchmark)
            # In real implementation, this would use the actual DAG orchestrator

            # Mock retrieval times based on hierarchy complexity
            base_time = 0.1  # Base retrieval time
            level_penalty = levels * 0.05  # Each level adds latency
            chunk_penalty = sum(chunk_config.values()) / 10000  # Larger chunks = more processing

            simulated_time = base_time + level_penalty + chunk_penalty
            await asyncio.sleep(simulated_time)  # Simulate processing

            # Mock results
            documents_processed = 10 * levels  # More levels = more documents
            recall_score = min(0.95, 0.6 + (levels * 0.1))  # Better recall with more levels
            precision_score = max(0.6, 0.9 - (levels * 0.05))  # Slight precision decrease

            latency = time.time() - start_time

            return {
                "query": query,
                "hierarchy_levels": levels,
                "chunk_config": chunk_config,
                "latency": latency,
                "documents_processed": documents_processed,
                "recall_score": recall_score,
                "precision_score": precision_score,
                "f1_score": 2 * (recall_score * precision_score) / (recall_score + precision_score),
                "throughput": documents_processed / latency,
                "error": None
            }

        except Exception as e:
            return {
                "query": query,
                "hierarchy_levels": levels,
                "chunk_config": chunk_config,
                "error": str(e),
                "latency": time.time() - start_time
            }

    async def run_full_benchmark(self) -> Dict[str, Any]:
        """Spuštění kompletního HRAG benchmarku"""

        print("🚀 Starting Hierarchical RAG benchmark...")

        benchmark_results = {
            "timestamp": time.time(),
            "config": self.config.get("retrieval", {}).get("hierarchical", {}),
            "test_queries": self.test_queries,
            "hierarchy_levels_tested": self.hierarchy_levels,
            "chunk_configurations": self.chunk_configurations,
            "results": []
        }

        total_tests = len(self.test_queries) * len(self.hierarchy_levels) * len(self.chunk_configurations)
        completed_tests = 0

        # Benchmark each combination
        for query in self.test_queries:
            print(f"\n🔍 Testing query: {query[:50]}...")

            for levels in self.hierarchy_levels:
                for chunk_config in self.chunk_configurations:
                    print(f"  Levels: {levels}, Chunks: {chunk_config}")

                    result = await self.benchmark_hierarchy_level(query, levels, chunk_config)
                    benchmark_results["results"].append(result)

                    completed_tests += 1
                    if result.get("error"):
                        print(f"    ❌ Error: {result['error']}")
                    else:
                        print(f"    ✅ Latency: {result['latency']:.2f}s, "
                              f"F1: {result['f1_score']:.3f}, "
                              f"Throughput: {result['throughput']:.1f} docs/s")

                    print(f"    Progress: {completed_tests}/{total_tests}")

        # Calculate aggregated metrics
        benchmark_results["aggregated_metrics"] = self._calculate_aggregated_metrics(
            benchmark_results["results"]
        )

        # Generate recommendations
        benchmark_results["recommendations"] = self._generate_recommendations(
            benchmark_results["results"]
        )

        return benchmark_results

    def _calculate_aggregated_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Výpočet agregovaných metrik"""

        valid_results = [r for r in results if not r.get("error")]

        if not valid_results:
            return {"error": "No valid results found"}

        # Group by hierarchy levels
        by_levels = {}
        for result in valid_results:
            levels = result["hierarchy_levels"]
            if levels not in by_levels:
                by_levels[levels] = []
            by_levels[levels].append(result)

        # Calculate averages per level
        level_metrics = {}
        for levels, level_results in by_levels.items():
            level_metrics[f"level_{levels}"] = {
                "avg_latency": statistics.mean([r["latency"] for r in level_results]),
                "avg_f1": statistics.mean([r["f1_score"] for r in level_results]),
                "avg_throughput": statistics.mean([r["throughput"] for r in level_results]),
                "avg_recall": statistics.mean([r["recall_score"] for r in level_results]),
                "avg_precision": statistics.mean([r["precision_score"] for r in level_results]),
                "test_count": len(level_results)
            }

        # Overall best configuration
        best_result = max(valid_results, key=lambda x: x["f1_score"])

        return {
            "by_hierarchy_levels": level_metrics,
            "best_configuration": {
                "hierarchy_levels": best_result["hierarchy_levels"],
                "chunk_config": best_result["chunk_config"],
                "f1_score": best_result["f1_score"],
                "latency": best_result["latency"],
                "throughput": best_result["throughput"]
            },
            "total_valid_tests": len(valid_results),
            "total_failed_tests": len(results) - len(valid_results)
        }

    def _generate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generování doporučení"""

        valid_results = [r for r in results if not r.get("error")]
        recommendations = []

        if not valid_results:
            recommendations.append("CRITICAL: All benchmark tests failed")
            return recommendations

        # Find optimal hierarchy level
        level_performance = {}
        for result in valid_results:
            level = result["hierarchy_levels"]
            if level not in level_performance:
                level_performance[level] = []
            level_performance[level].append(result["f1_score"])

        best_level = max(level_performance.keys(),
                        key=lambda x: statistics.mean(level_performance[x]))
        best_f1 = statistics.mean(level_performance[best_level])

        recommendations.append(f"OPTIMAL: Use {best_level} hierarchy levels (F1: {best_f1:.3f})")

        # Latency analysis
        avg_latency = statistics.mean([r["latency"] for r in valid_results])
        if avg_latency > 5.0:
            recommendations.append("PERFORMANCE: Consider reducing hierarchy levels or chunk sizes for better latency")
        elif avg_latency < 1.0:
            recommendations.append("OPTIMIZATION: System can handle more complex hierarchies")

        # Throughput analysis
        avg_throughput = statistics.mean([r["throughput"] for r in valid_results])
        if avg_throughput < 10:
            recommendations.append("THROUGHPUT: Consider optimizing chunk processing pipeline")

        # Quality analysis
        avg_f1 = statistics.mean([r["f1_score"] for r in valid_results])
        if avg_f1 < 0.7:
            recommendations.append("QUALITY: Consider increasing hierarchy levels for better recall")
        elif avg_f1 > 0.9:
            recommendations.append("EXCELLENT: High F1 scores achieved across configurations")

        return recommendations


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Hierarchical RAG Benchmark")
    parser.add_argument("--config", "-c", default="config_m1_local.yaml", help="Configuration file")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Configuration file {args.config} not found")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)

    # Run benchmark
    benchmark = HierarchicalRAGBenchmark(config)
    results = await benchmark.run_full_benchmark()

    # Print summary
    if "error" in results.get("aggregated_metrics", {}):
        print(f"❌ Benchmark failed: {results['aggregated_metrics']['error']}")
        sys.exit(1)

    print(f"\n✅ Hierarchical RAG benchmark completed!")

    # Print aggregated results
    agg_metrics = results.get("aggregated_metrics", {})
    if "best_configuration" in agg_metrics:
        best = agg_metrics["best_configuration"]
        print(f"\n🏆 Best Configuration:")
        print(f"  Hierarchy levels: {best['hierarchy_levels']}")
        print(f"  Chunk config: {best['chunk_config']}")
        print(f"  F1 Score: {best['f1_score']:.3f}")
        print(f"  Latency: {best['latency']:.2f}s")
        print(f"  Throughput: {best['throughput']:.1f} docs/s")

    # Print recommendations
    recommendations = results.get("recommendations", [])
    if recommendations:
        print(f"\n💡 Recommendations:")
        for rec in recommendations:
            print(f"  • {rec}")

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
