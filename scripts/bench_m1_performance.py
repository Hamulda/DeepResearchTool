#!/usr/bin/env python3
"""
FÁZE 6: M1 Performance Benchmark Suite
Komprehenzivní benchmark skripty pro M1 optimalizaci

Author: Senior Python/MLOps Agent
"""

import asyncio
import time
import json
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.optimization.m1_performance import create_m1_optimization_engine, M1PerformanceMetrics
from src.optimization.streaming_engine import create_m1_streaming_engine


class M1BenchmarkSuite:
    """Komprehenzivní M1 benchmark suite"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.m1_engine = create_m1_optimization_engine(config)
        self.streaming_engine = create_m1_streaming_engine(config)
        self.benchmark_results = []

    async def run_comprehensive_benchmarks(
        self, profiles: Optional[List[str]] = None, iterations: int = 3
    ) -> Dict[str, Any]:
        """Spustí komprehenzivní M1 benchmarks"""

        print("🏁 Spouštím M1 Performance Benchmark Suite...")
        print("=" * 60)

        if profiles is None:
            profiles = ["quick", "thorough", "balanced"]

        benchmark_start = time.time()
        results = {
            "m1_benchmark_suite": {
                "timestamp": datetime.now().isoformat(),
                "system_info": self.m1_engine.system_info.__dict__,
                "profiles_tested": profiles,
                "iterations_per_profile": iterations,
                "performance_results": {},
                "streaming_results": {},
                "aggregate_analysis": {},
                "recommendations": [],
            }
        }

        # Test každý profil
        for profile in profiles:
            print(f"\n📊 Benchmarking profile: {profile}")
            print("-" * 40)

            profile_metrics = []

            for iteration in range(iterations):
                print(f"🔄 Iteration {iteration + 1}/{iterations}")

                try:
                    # M1 optimization benchmark
                    metrics = await self.m1_engine.optimize_for_profile(profile)
                    profile_metrics.append(metrics)

                    print(f"   Time: {metrics.execution_time_s:.1f}s")
                    print(f"   Memory: {metrics.memory_peak_mb:.0f}MB")
                    print(f"   Tokens/s: {metrics.tokens_per_second:.1f}")

                    # Brief pause between iterations
                    await asyncio.sleep(1)

                except Exception as e:
                    print(f"   ❌ Iteration {iteration + 1} failed: {str(e)}")

            # Streaming benchmark pro tento profil
            streaming_result = await self._benchmark_streaming_for_profile(profile)

            # Aggregate profile results
            if profile_metrics:
                results["m1_benchmark_suite"]["performance_results"][profile] = {
                    "iterations": len(profile_metrics),
                    "metrics": [self._metrics_to_dict(m) for m in profile_metrics],
                    "avg_execution_time": sum(m.execution_time_s for m in profile_metrics)
                    / len(profile_metrics),
                    "avg_memory_peak": sum(m.memory_peak_mb for m in profile_metrics)
                    / len(profile_metrics),
                    "avg_tokens_per_second": sum(m.tokens_per_second for m in profile_metrics)
                    / len(profile_metrics),
                    "avg_memory_efficiency": sum(m.memory_efficiency for m in profile_metrics)
                    / len(profile_metrics),
                    "success_rate": sum(1 for m in profile_metrics if m.error_rate < 0.1)
                    / len(profile_metrics),
                }

            results["m1_benchmark_suite"]["streaming_results"][profile] = streaming_result

        # Aggregate analysis
        results["m1_benchmark_suite"]["aggregate_analysis"] = await self._analyze_benchmark_results(
            results
        )

        # Generate recommendations
        results["m1_benchmark_suite"]["recommendations"] = self._generate_benchmark_recommendations(
            results
        )

        total_benchmark_time = time.time() - benchmark_start
        results["m1_benchmark_suite"]["total_benchmark_time_s"] = total_benchmark_time

        print(f"\n🎉 Benchmark suite completed in {total_benchmark_time:.1f}s")

        return results

    async def _benchmark_streaming_for_profile(self, profile: str) -> Dict[str, Any]:
        """Benchmark streaming pro specific profile"""

        profile_config = self.m1_engine.performance_profiles[profile]

        test_queries = [
            "What are the latest advances in quantum computing?",
            "Explain machine learning optimization techniques",
            "Recent developments in sustainable energy",
        ]

        streaming_config = {
            "model": profile_config.ollama_model,
            "context_window": profile_config.context_window,
            "max_tokens": profile_config.max_tokens // 2,  # Smaller for benchmarking
            "memory_limit_mb": profile_config.memory_limit_mb,
        }

        try:
            streaming_result = await self.streaming_engine.benchmark_streaming_performance(
                test_queries, streaming_config
            )
            return streaming_result["streaming_benchmark"]

        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def _metrics_to_dict(self, metrics: M1PerformanceMetrics) -> Dict[str, Any]:
        """Převede metrics na dictionary"""
        return {
            "profile_name": metrics.profile_name,
            "execution_time_s": metrics.execution_time_s,
            "memory_peak_mb": metrics.memory_peak_mb,
            "memory_efficiency": metrics.memory_efficiency,
            "tokens_per_second": metrics.tokens_per_second,
            "context_utilization": metrics.context_utilization,
            "mps_utilization": metrics.mps_utilization,
            "early_exit_rate": metrics.early_exit_rate,
            "streaming_chunks": metrics.streaming_chunks,
            "error_rate": metrics.error_rate,
            "timestamp": metrics.timestamp.isoformat(),
        }

    async def _analyze_benchmark_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyzuje benchmark výsledky"""

        performance_results = results["m1_benchmark_suite"]["performance_results"]

        analysis = {
            "profile_comparison": {},
            "performance_trends": {},
            "bottleneck_analysis": {},
            "m1_optimization_effectiveness": {},
        }

        # Profile comparison
        for profile, data in performance_results.items():
            target_time = self.m1_engine.performance_profiles[profile].timeout_seconds
            actual_time = data["avg_execution_time"]

            analysis["profile_comparison"][profile] = {
                "target_time_s": target_time,
                "actual_time_s": actual_time,
                "time_efficiency": min(1.0, target_time / actual_time) if actual_time > 0 else 0,
                "memory_efficiency": data["avg_memory_efficiency"],
                "throughput_tokens_per_s": data["avg_tokens_per_second"],
                "meets_target": actual_time <= target_time,
            }

        # Performance trends
        if len(performance_results) >= 2:
            quick_perf = performance_results.get("quick", {})
            thorough_perf = performance_results.get("thorough", {})

            if quick_perf and thorough_perf:
                scaling_factor = (
                    thorough_perf["avg_execution_time"] / quick_perf["avg_execution_time"]
                    if quick_perf["avg_execution_time"] > 0
                    else 0
                )

                analysis["performance_trends"]["scaling_analysis"] = {
                    "quick_to_thorough_time_ratio": scaling_factor,
                    "expected_ratio": 2.0,  # Thorough should be ~2x slower
                    "scaling_efficiency": (
                        min(1.0, 2.0 / scaling_factor) if scaling_factor > 0 else 0
                    ),
                }

        # M1 optimization effectiveness
        mps_available = results["m1_benchmark_suite"]["system_info"]["mps_available"]
        avg_mps_utilization = 0

        for profile_data in performance_results.values():
            if profile_data["metrics"]:
                avg_mps_utilization += sum(
                    m.get("mps_utilization", 0) for m in profile_data["metrics"]
                ) / len(profile_data["metrics"])

        if performance_results:
            avg_mps_utilization /= len(performance_results)

        analysis["m1_optimization_effectiveness"] = {
            "mps_available": mps_available,
            "avg_mps_utilization": avg_mps_utilization,
            "hardware_optimization_score": avg_mps_utilization if mps_available else 0.5,
        }

        return analysis

    def _generate_benchmark_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generuje doporučení na základě benchmark výsledků"""

        recommendations = []
        performance_results = results["m1_benchmark_suite"]["performance_results"]
        analysis = results["m1_benchmark_suite"]["aggregate_analysis"]

        # Time performance recommendations
        for profile, comparison in analysis["profile_comparison"].items():
            if not comparison["meets_target"]:
                recommendations.append(
                    f"⚠️ {profile} profile exceeds target time: "
                    f"{comparison['actual_time_s']:.1f}s > {comparison['target_time_s']}s. "
                    f"Consider reducing context_window or batch_size."
                )
            elif comparison["time_efficiency"] > 0.8:
                recommendations.append(
                    f"✅ {profile} profile performs well within targets "
                    f"({comparison['time_efficiency']:.1%} efficiency)"
                )

        # Memory recommendations
        for profile, data in performance_results.items():
            memory_limit = self.m1_engine.performance_profiles[profile].memory_limit_mb
            if data["avg_memory_peak"] > memory_limit * 0.9:
                recommendations.append(
                    f"⚠️ {profile} profile near memory limit: "
                    f"{data['avg_memory_peak']:.0f}MB / {memory_limit}MB. "
                    f"Consider reducing batch_size or context_window."
                )

        # M1 optimization recommendations
        mps_effectiveness = analysis["m1_optimization_effectiveness"]
        if mps_effectiveness["mps_available"] and mps_effectiveness["avg_mps_utilization"] < 0.5:
            recommendations.append(
                "🔧 Low MPS utilization detected. Consider optimizing tensor operations for Metal Performance Shaders."
            )
        elif not mps_effectiveness["mps_available"]:
            recommendations.append(
                "💡 MPS not available. Consider upgrading to M1/M2 Mac for hardware acceleration."
            )

        # Streaming recommendations
        streaming_results = results["m1_benchmark_suite"]["streaming_results"]
        for profile, streaming_data in streaming_results.items():
            if "aggregate_metrics" in streaming_data:
                early_exit_rate = streaming_data["aggregate_metrics"].get("avg_early_exit_rate", 0)
                if early_exit_rate > 0.5:
                    recommendations.append(
                        f"ℹ️ {profile} profile has high early exit rate ({early_exit_rate:.1%}). "
                        f"This may indicate repetitive content or overly aggressive novelty thresholds."
                    )

        if not recommendations:
            recommendations.append(
                "🎉 All profiles performing optimally! No immediate optimizations needed."
            )

        return recommendations

    async def export_benchmark_results(
        self, results: Dict[str, Any], output_path: str = "docs/m1_benchmark_results.json"
    ):
        """Exportuje benchmark výsledky"""

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        # Create summary report
        summary_path = output_file.parent / "m1_benchmark_summary.md"
        await self._create_summary_report(results, summary_path)

        print(f"📊 Benchmark results exported: {output_file}")
        print(f"📋 Summary report: {summary_path}")

    async def _create_summary_report(self, results: Dict[str, Any], output_path: Path):
        """Vytvoří Markdown summary report"""

        report_lines = [
            "# M1 Performance Benchmark Report",
            "",
            f"**Generated:** {results['m1_benchmark_suite']['timestamp']}",
            f"**System:** {results['m1_benchmark_suite']['system_info']['platform']}",
            f"**Memory:** {results['m1_benchmark_suite']['system_info']['memory_gb']:.1f}GB",
            f"**MPS Available:** {results['m1_benchmark_suite']['system_info']['mps_available']}",
            "",
            "## Performance Summary",
            "",
            "| Profile | Target Time | Actual Time | Efficiency | Memory Peak | Tokens/s |",
            "|---------|-------------|-------------|------------|-------------|----------|",
        ]

        performance_results = results["m1_benchmark_suite"]["performance_results"]
        analysis = results["m1_benchmark_suite"]["aggregate_analysis"]

        for profile, comparison in analysis["profile_comparison"].items():
            data = performance_results[profile]
            status = "✅" if comparison["meets_target"] else "❌"

            report_lines.append(
                f"| {profile} {status} | {comparison['target_time_s']}s | "
                f"{comparison['actual_time_s']:.1f}s | {comparison['time_efficiency']:.1%} | "
                f"{data['avg_memory_peak']:.0f}MB | {data['avg_tokens_per_second']:.1f} |"
            )

        report_lines.extend(["", "## Recommendations", ""])

        for rec in results["m1_benchmark_suite"]["recommendations"]:
            report_lines.append(f"- {rec}")

        report_lines.extend(
            [
                "",
                f"## Benchmark Details",
                "",
                f"**Total Benchmark Time:** {results['m1_benchmark_suite']['total_benchmark_time_s']:.1f}s",
                f"**Profiles Tested:** {', '.join(results['m1_benchmark_suite']['profiles_tested'])}",
                f"**Iterations per Profile:** {results['m1_benchmark_suite']['iterations_per_profile']}",
                "",
                "For detailed results, see `m1_benchmark_results.json`",
            ]
        )

        with open(output_path, "w") as f:
            f.write("\n".join(report_lines))


async def main():
    """Main function pro M1 benchmark"""
    parser = argparse.ArgumentParser(description="M1 Performance Benchmark Suite")
    parser.add_argument("--config", default="config_m1_local.yaml", help="Config file")
    parser.add_argument(
        "--profiles", nargs="+", default=["quick", "thorough"], help="Profiles to benchmark"
    )
    parser.add_argument("--iterations", type=int, default=3, help="Iterations per profile")
    parser.add_argument("--output", default="docs/m1_benchmark_results.json", help="Output file")

    args = parser.parse_args()

    # Load config
    config = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # Add M1 optimization config
    config.setdefault("m1_optimization", {})
    config.setdefault("streaming", {})

    # Run benchmark
    suite = M1BenchmarkSuite(config)
    results = await suite.run_comprehensive_benchmarks(
        profiles=args.profiles, iterations=args.iterations
    )

    await suite.export_benchmark_results(results, args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("M1 BENCHMARK SUMMARY")
    print("=" * 60)

    for profile, comparison in results["m1_benchmark_suite"]["aggregate_analysis"][
        "profile_comparison"
    ].items():
        status = "✅ PASS" if comparison["meets_target"] else "❌ FAIL"
        print(
            f"{profile:10} {status} | {comparison['actual_time_s']:6.1f}s | {comparison['time_efficiency']:6.1%}"
        )

    print("\nRecommendations:")
    for rec in results["m1_benchmark_suite"]["recommendations"]:
        print(f"  {rec}")


if __name__ == "__main__":
    asyncio.run(main())
