#!/usr/bin/env python3
"""
Qdrant Benchmark Script
Optimalizace ef_search parametrů pro různé kolekce

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

from src.retrieval.adaptive_params import AdaptiveParameterOptimizer
from qdrant_client import QdrantClient
from qdrant_client.models import SearchRequest


class QdrantBenchmark:
    """Benchmark pro Qdrant performance"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.qdrant_config = config.get("qdrant", {})
        self.client = QdrantClient(url=self.qdrant_config.get("url", "http://localhost:6333"))

        # Test queries
        self.test_queries = [
            "quantum computing error correction",
            "machine learning transformers",
            "climate change adaptation",
            "artificial intelligence safety",
            "renewable energy storage",
            "biomedical imaging analysis",
            "blockchain consensus mechanisms",
            "neural network optimization",
            "data privacy techniques",
            "automated reasoning systems",
        ]

        # ef_search values to test
        self.ef_search_values = [16, 32, 64, 96, 128, 192, 256]

    async def benchmark_collection(
        self, collection_name: str, query_vector: List[float], ef_search: int, num_runs: int = 5
    ) -> Dict[str, Any]:
        """Benchmark jedné kolekce s daným ef_search"""

        latencies = []
        recall_scores = []

        for _ in range(num_runs):
            start_time = time.time()

            try:
                # Perform search
                results = self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=10,
                    search_params={"ef": ef_search},
                )

                latency = time.time() - start_time
                latencies.append(latency)

                # Simple recall approximation (based on score distribution)
                if results:
                    scores = [result.score for result in results]
                    recall_approx = len([s for s in scores if s > 0.7]) / len(scores)
                    recall_scores.append(recall_approx)
                else:
                    recall_scores.append(0.0)

            except Exception as e:
                print(f"Error in benchmark: {e}")
                return {"ef_search": ef_search, "collection": collection_name, "error": str(e)}

        return {
            "ef_search": ef_search,
            "collection": collection_name,
            "avg_latency": statistics.mean(latencies),
            "std_latency": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "avg_recall": statistics.mean(recall_scores),
            "std_recall": statistics.stdev(recall_scores) if len(recall_scores) > 1 else 0,
            "latencies": latencies,
            "recall_scores": recall_scores,
        }

    async def run_full_benchmark(self) -> Dict[str, Any]:
        """Spuštění kompletního benchmarku"""

        print("🚀 Starting Qdrant benchmark...")

        # Get available collections
        collections_info = self.client.get_collections()
        collections = [c.name for c in collections_info.collections]

        if not collections:
            return {"error": "No collections found in Qdrant"}

        print(f"📊 Found collections: {collections}")

        # Generate test query vectors (simplified - using random for demo)
        import numpy as np

        np.random.seed(42)

        # Assume 384-dimensional embeddings (typical for all-MiniLM-L6-v2)
        test_vectors = [np.random.rand(384).tolist() for _ in range(len(self.test_queries))]

        benchmark_results = {
            "timestamp": time.time(),
            "config": self.qdrant_config,
            "collections_tested": collections,
            "ef_search_values": self.ef_search_values,
            "results": {},
        }

        # Benchmark each collection
        for collection_name in collections[:3]:  # Limit to first 3 collections
            print(f"\n🔍 Benchmarking collection: {collection_name}")

            collection_results = []

            for ef_search in self.ef_search_values:
                print(f"  Testing ef_search={ef_search}...")

                # Use first test vector for each ef_search value
                result = await self.benchmark_collection(
                    collection_name, test_vectors[0], ef_search
                )

                collection_results.append(result)

                print(
                    f"    Latency: {result.get('avg_latency', 0)*1000:.1f}ms, "
                    f"Recall: {result.get('avg_recall', 0):.3f}"
                )

            benchmark_results["results"][collection_name] = collection_results

        # Calculate optimization recommendations
        recommendations = self._generate_recommendations(benchmark_results)
        benchmark_results["recommendations"] = recommendations

        return benchmark_results

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generování doporučení na základě benchmark výsledků"""

        recommendations = []

        for collection_name, collection_results in results["results"].items():
            if not collection_results:
                continue

            # Find optimal ef_search (best latency/recall trade-off)
            best_trade_off = None
            best_score = 0

            for result in collection_results:
                if "error" in result:
                    continue

                latency = result.get("avg_latency", float("inf"))
                recall = result.get("avg_recall", 0)

                # Trade-off score (higher recall, lower latency is better)
                score = recall / max(latency * 1000, 1)  # Normalize latency to ms

                if score > best_score:
                    best_score = score
                    best_trade_off = result

            if best_trade_off:
                ef_value = best_trade_off["ef_search"]
                latency = best_trade_off["avg_latency"] * 1000
                recall = best_trade_off["avg_recall"]

                recommendations.append(
                    f"{collection_name}: Optimal ef_search={ef_value} "
                    f"(latency: {latency:.1f}ms, recall: {recall:.3f})"
                )

        # Global recommendations
        if results["results"]:
            avg_latencies = []
            for collection_results in results["results"].values():
                for result in collection_results:
                    if "avg_latency" in result:
                        avg_latencies.append(result["avg_latency"])

            if avg_latencies:
                overall_avg = statistics.mean(avg_latencies) * 1000
                if overall_avg > 200:
                    recommendations.append(
                        "PERFORMANCE: Consider lower ef_search values for better latency"
                    )
                elif overall_avg < 50:
                    recommendations.append(
                        "QUALITY: Consider higher ef_search values for better recall"
                    )

        return recommendations


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Qdrant Benchmark Runner")
    parser.add_argument("--config", "-c", default="config_m1_local.yaml", help="Configuration file")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Load configuration
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Configuration file {args.config} not found")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)

    # Run benchmark
    benchmark = QdrantBenchmark(config)
    results = await benchmark.run_full_benchmark()

    # Print summary
    if "error" in results:
        print(f"❌ Benchmark failed: {results['error']}")
        sys.exit(1)

    print(f"\n✅ Benchmark completed!")
    print(f"📊 Collections tested: {len(results['results'])}")

    # Print recommendations
    recommendations = results.get("recommendations", [])
    if recommendations:
        print(f"\n💡 Recommendations:")
        for rec in recommendations:
            print(f"  • {rec}")

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Results saved to {args.output}")

    print(f"\n🎯 Use these settings in your config:")
    for collection_name, collection_results in results["results"].items():
        if collection_results:
            # Find best ef_search for this collection
            best_result = max(
                [r for r in collection_results if "error" not in r],
                key=lambda x: x.get("avg_recall", 0) / max(x.get("avg_latency", 1) * 1000, 1),
                default=None,
            )
            if best_result:
                print(f"  {collection_name}: ef_search: {best_result['ef_search']}")


if __name__ == "__main__":
    asyncio.run(main())
