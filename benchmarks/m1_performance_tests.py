"""
M1 Performance Benchmarks pro DeepResearchTool
Měření výkonu na M1 architektuře s cílovými metrikami:
- ≥90% úspěšnost obcházení anti-bot ochran
- ≤6GB špičkové využití paměti
"""

import asyncio
import gc
import psutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

from src.optimization.intelligent_memory import IntelligentMemoryManager
from src.optimization.metal_acceleration import MetalAcceleration
from src.evasion.anti_bot_bypass import AntiBotCircumventionSuite
from src.steganography.advanced_steganalysis import AdvancedSteganalysisEngine
from src.extreme_research_orchestrator import ExtremeResearchOrchestrator


@dataclass
class PerformanceMetrics:
    """Metriky výkonu"""
    test_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_usage_percent: float
    success_rate: float
    throughput_ops_per_second: float
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


class M1PerformanceBenchmarks:
    """Benchmark suite pro M1 Mac optimalizace"""

    def __init__(self):
        self.process = psutil.Process()
        self.results: List[PerformanceMetrics] = []

    def _get_memory_usage_mb(self) -> float:
        """Získání aktuálního využití paměti v MB"""
        return self.process.memory_info().rss / 1024 / 1024

    def _get_cpu_usage(self) -> float:
        """Získání CPU usage"""
        return self.process.cpu_percent()

    async def benchmark_memory_manager(self, operations: int = 10000) -> PerformanceMetrics:
        """Benchmark IntelligentMemoryManager"""
        print(f"🧠 Benchmarking Memory Manager ({operations} operations)...")

        start_time = datetime.now()
        start_memory = self._get_memory_usage_mb()

        manager = IntelligentMemoryManager(max_memory_mb=1024)  # 1GB limit
        await manager.initialize()

        peak_memory = start_memory

        # Performance test - Set operations
        set_start = time.time()
        for i in range(operations):
            await manager.set(f"bench_key_{i}", f"value_{i}" * 100, importance_score=np.random.random())

            if i % 1000 == 0:  # Check memory every 1000 ops
                current_memory = self._get_memory_usage_mb()
                peak_memory = max(peak_memory, current_memory)

        set_duration = time.time() - set_start

        # Performance test - Get operations
        get_start = time.time()
        hit_count = 0
        for i in range(operations):
            result = await manager.get(f"bench_key_{i}")
            if result is not None:
                hit_count += 1

        get_duration = time.time() - get_start

        # Memory optimization test
        optimization_start = time.time()
        optimization_result = await manager.optimize_memory()
        optimization_duration = time.time() - optimization_start

        await manager.close()

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        # Force garbage collection
        gc.collect()

        metrics = PerformanceMetrics(
            test_name="memory_manager",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=total_duration,
            memory_usage_mb=self._get_memory_usage_mb(),
            peak_memory_mb=peak_memory,
            cpu_usage_percent=self._get_cpu_usage(),
            success_rate=hit_count / operations,
            throughput_ops_per_second=operations / total_duration,
            additional_metrics={
                "set_duration_seconds": set_duration,
                "get_duration_seconds": get_duration,
                "optimization_duration_seconds": optimization_duration,
                "cache_hit_rate": hit_count / operations,
                "memory_freed_mb": optimization_result.get("memory_freed_bytes", 0) / 1024 / 1024,
                "entries_evicted": optimization_result.get("removed_inactive_entries", 0)
            }
        )

        self.results.append(metrics)
        return metrics

    async def benchmark_metal_acceleration(self, image_count: int = 100) -> PerformanceMetrics:
        """Benchmark Metal acceleration na M1"""
        print(f"⚡ Benchmarking Metal Acceleration ({image_count} images)...")

        start_time = datetime.now()
        start_memory = self._get_memory_usage_mb()

        acceleration = MetalAcceleration()
        peak_memory = start_memory

        # Generate test data
        test_images = []
        for i in range(image_count):
            test_image = np.random.rand(256, 256).astype(np.float32)
            test_images.append(test_image)

        successful_operations = 0

        # Benchmark image processing
        processing_start = time.time()

        for i, image in enumerate(test_images):
            try:
                result = await acceleration.accelerated_image_analysis(image)
                if result and "entropy" in result:
                    successful_operations += 1

                if i % 20 == 0:  # Check memory every 20 operations
                    current_memory = self._get_memory_usage_mb()
                    peak_memory = max(peak_memory, current_memory)

            except Exception as e:
                print(f"Error in image {i}: {e}")

        processing_duration = time.time() - processing_start

        # Benchmark comparison (CPU vs GPU if available)
        benchmark_result = await acceleration.benchmark_acceleration()

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        metrics = PerformanceMetrics(
            test_name="metal_acceleration",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=total_duration,
            memory_usage_mb=self._get_memory_usage_mb(),
            peak_memory_mb=peak_memory,
            cpu_usage_percent=self._get_cpu_usage(),
            success_rate=successful_operations / image_count,
            throughput_ops_per_second=successful_operations / processing_duration,
            additional_metrics={
                "mlx_available": acceleration.mlx_available,
                "processing_duration_seconds": processing_duration,
                "benchmark_results": benchmark_result,
                "acceleration_type": acceleration.get_acceleration_info()["acceleration_type"]
            }
        )

        self.results.append(metrics)
        return metrics

    async def benchmark_antibot_bypass(self, test_urls: List[str] = None) -> PerformanceMetrics:
        """Benchmark Anti-Bot Bypass úspěšnosti"""
        if test_urls is None:
            # Test URLs s různými typy ochrany
            test_urls = [
                "https://httpbin.org/user-agent",  # Basic
                "https://httpbin.org/headers",     # Headers check
                "https://httpbin.org/status/403",  # Forbidden
                "https://example.com",             # Standard site
                "https://www.google.com"           # Google (potential bot detection)
            ]

        print(f"🛡️ Benchmarking Anti-Bot Bypass ({len(test_urls)} URLs)...")

        start_time = datetime.now()
        start_memory = self._get_memory_usage_mb()

        antibot_suite = AntiBotCircumventionSuite(stealth_mode=True)
        peak_memory = start_memory

        successful_bypasses = 0
        total_response_time = 0
        bypass_details = []

        bypass_start = time.time()

        for i, url in enumerate(test_urls):
            try:
                result = await antibot_suite.circumvent_protection(url)

                bypass_details.append({
                    "url": url,
                    "success": result.success,
                    "method": result.method_used,
                    "response_time_ms": result.response_time_ms,
                    "protection_detected": result.protection_detected
                })

                if result.success:
                    successful_bypasses += 1
                    total_response_time += result.response_time_ms

                current_memory = self._get_memory_usage_mb()
                peak_memory = max(peak_memory, current_memory)

            except Exception as e:
                print(f"Error bypassing {url}: {e}")
                bypass_details.append({
                    "url": url,
                    "success": False,
                    "error": str(e)
                })

        bypass_duration = time.time() - bypass_start

        await antibot_suite.close()

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        success_rate = successful_bypasses / len(test_urls)
        avg_response_time = total_response_time / max(successful_bypasses, 1)

        metrics = PerformanceMetrics(
            test_name="antibot_bypass",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=total_duration,
            memory_usage_mb=self._get_memory_usage_mb(),
            peak_memory_mb=peak_memory,
            cpu_usage_percent=self._get_cpu_usage(),
            success_rate=success_rate,
            throughput_ops_per_second=len(test_urls) / bypass_duration,
            additional_metrics={
                "successful_bypasses": successful_bypasses,
                "total_urls_tested": len(test_urls),
                "average_response_time_ms": avg_response_time,
                "bypass_details": bypass_details,
                "meets_target": success_rate >= 0.90  # 90% target
            }
        )

        self.results.append(metrics)
        return metrics

    async def benchmark_steganography_engine(self, file_count: int = 50) -> PerformanceMetrics:
        """Benchmark steganografického enginu"""
        print(f"🔍 Benchmarking Steganography Engine ({file_count} files)...")

        start_time = datetime.now()
        start_memory = self._get_memory_usage_mb()

        engine = AdvancedSteganalysisEngine(enable_gpu_acceleration=True)
        peak_memory = start_memory

        # Simulace souborů pro analýzu
        import tempfile
        import os
        from PIL import Image

        temp_files = []

        # Vytvoření testovacích souborů
        for i in range(file_count):
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                # Různé velikosti obrázků
                size = 100 + (i * 10)  # 100x100 to 600x600
                img = Image.new('RGB', (size, size), color=(i % 255, (i*2) % 255, (i*3) % 255))
                img.save(temp_file.name)
                temp_files.append(temp_file.name)

        successful_analyses = 0

        analysis_start = time.time()

        # Batch analýza
        try:
            results = await engine.batch_analyze(temp_files)
            successful_analyses = len([r for r in results if not r.errors])

            current_memory = self._get_memory_usage_mb()
            peak_memory = max(peak_memory, current_memory)

        except Exception as e:
            print(f"Batch analysis error: {e}")

        analysis_duration = time.time() - analysis_start

        # Cleanup
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        metrics = PerformanceMetrics(
            test_name="steganography_engine",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=total_duration,
            memory_usage_mb=self._get_memory_usage_mb(),
            peak_memory_mb=peak_memory,
            cpu_usage_percent=self._get_cpu_usage(),
            success_rate=successful_analyses / file_count,
            throughput_ops_per_second=successful_analyses / analysis_duration,
            additional_metrics={
                "files_analyzed": file_count,
                "successful_analyses": successful_analyses,
                "analysis_duration_seconds": analysis_duration,
                "gpu_acceleration_used": engine._gpu_available
            }
        )

        self.results.append(metrics)
        return metrics

    async def benchmark_extreme_orchestrator(self) -> PerformanceMetrics:
        """Benchmark hlavního orchestrátoru"""
        print("🎭 Benchmarking Extreme Research Orchestrator...")

        start_time = datetime.now()
        start_memory = self._get_memory_usage_mb()

        try:
            orchestrator = ExtremeResearchOrchestrator(max_concurrent_tasks=3)
            await orchestrator.initialize()

            peak_memory = start_memory

            # Test systémového statusu
            status_start = time.time()
            status = await orchestrator.get_system_status()
            status_duration = time.time() - status_start

            current_memory = self._get_memory_usage_mb()
            peak_memory = max(peak_memory, current_memory)

            await orchestrator.cleanup()

            success = status["orchestrator"] == "active"

        except Exception as e:
            print(f"Orchestrator benchmark error: {e}")
            success = False
            status_duration = 0
            status = {}

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        metrics = PerformanceMetrics(
            test_name="extreme_orchestrator",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=total_duration,
            memory_usage_mb=self._get_memory_usage_mb(),
            peak_memory_mb=peak_memory,
            cpu_usage_percent=self._get_cpu_usage(),
            success_rate=1.0 if success else 0.0,
            throughput_ops_per_second=1.0 / status_duration if status_duration > 0 else 0,
            additional_metrics={
                "status_check_duration_seconds": status_duration,
                "system_status": status,
                "initialization_successful": success
            }
        )

        self.results.append(metrics)
        return metrics

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generování komplexního performance reportu"""
        if not self.results:
            return {"error": "No benchmark results available"}

        # Celkové metriky
        total_peak_memory = max(r.peak_memory_mb for r in self.results)
        avg_success_rate = sum(r.success_rate for r in self.results) / len(self.results)

        # Kontrola cílových metrik
        memory_target_met = total_peak_memory <= 6144  # 6GB = 6144MB
        success_rate_target_met = avg_success_rate >= 0.90  # 90%

        report = {
            "benchmark_summary": {
                "total_tests": len(self.results),
                "benchmark_date": datetime.now().isoformat(),
                "peak_memory_usage_mb": total_peak_memory,
                "average_success_rate": avg_success_rate,
                "targets_met": {
                    "memory_under_6gb": memory_target_met,
                    "success_rate_over_90pct": success_rate_target_met,
                    "overall_target_met": memory_target_met and success_rate_target_met
                }
            },
            "individual_test_results": [],
            "performance_insights": [],
            "recommendations": []
        }

        # Detail jednotlivých testů
        for result in self.results:
            test_detail = {
                "test_name": result.test_name,
                "duration_seconds": result.duration_seconds,
                "memory_usage_mb": result.memory_usage_mb,
                "peak_memory_mb": result.peak_memory_mb,
                "success_rate": result.success_rate,
                "throughput_ops_per_second": result.throughput_ops_per_second,
                "additional_metrics": result.additional_metrics
            }
            report["individual_test_results"].append(test_detail)

        # Performance insights
        if total_peak_memory > 4096:  # > 4GB
            report["performance_insights"].append(
                f"High memory usage detected: {total_peak_memory:.1f}MB peak"
            )

        if avg_success_rate < 0.85:
            report["performance_insights"].append(
                f"Success rate below target: {avg_success_rate:.1%}"
            )

        # Doporučení
        if not memory_target_met:
            report["recommendations"].append(
                "Consider optimizing memory usage or increasing eviction frequency"
            )

        if not success_rate_target_met:
            report["recommendations"].append(
                "Review anti-bot bypass strategies and implement additional evasion techniques"
            )

        # M1-specific optimalizace
        metal_test = next((r for r in self.results if r.test_name == "metal_acceleration"), None)
        if metal_test and not metal_test.additional_metrics.get("mlx_available", False):
            report["recommendations"].append(
                "Install Apple MLX framework for better M1 GPU acceleration"
            )

        return report

    async def run_full_benchmark_suite(self) -> Dict[str, Any]:
        """Spuštění kompletní benchmark suite"""
        print("🚀 Starting M1 Performance Benchmark Suite...")
        print("=" * 60)

        # Benchmark suite
        try:
            await self.benchmark_memory_manager(operations=5000)
            await asyncio.sleep(2)  # Cool down

            await self.benchmark_metal_acceleration(image_count=50)
            await asyncio.sleep(2)

            await self.benchmark_antibot_bypass()
            await asyncio.sleep(2)

            await self.benchmark_steganography_engine(file_count=20)
            await asyncio.sleep(2)

            await self.benchmark_extreme_orchestrator()

        except Exception as e:
            print(f"Benchmark suite error: {e}")

        print("=" * 60)
        print("✅ Benchmark Suite Completed")

        return self.generate_performance_report()


async def main():
    """Hlavní funkce pro spuštění benchmarků"""
    benchmarks = M1PerformanceBenchmarks()

    print("🍎 DeepResearchTool M1 Performance Benchmarks")
    print("Target Metrics:")
    print("  • Anti-bot bypass success rate: ≥90%")
    print("  • Peak memory usage: ≤6GB")
    print()

    report = await benchmarks.run_full_benchmark_suite()

    # Výpis výsledků
    print("\n📊 BENCHMARK RESULTS")
    print("=" * 60)

    summary = report["benchmark_summary"]
    print(f"Peak Memory Usage: {summary['peak_memory_usage_mb']:.1f}MB")
    print(f"Average Success Rate: {summary['average_success_rate']:.1%}")

    targets = summary["targets_met"]
    print(f"\n🎯 TARGET COMPLIANCE:")
    print(f"  Memory ≤6GB: {'✅' if targets['memory_under_6gb'] else '❌'}")
    print(f"  Success ≥90%: {'✅' if targets['success_rate_over_90pct'] else '❌'}")
    print(f"  Overall: {'✅' if targets['overall_target_met'] else '❌'}")

    if report.get("recommendations"):
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")

    # Uložení reportu
    import json
    report_file = f"benchmarks/m1_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    import os
    os.makedirs("benchmarks", exist_ok=True)

    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n📄 Report saved: {report_file}")


if __name__ == "__main__":
    asyncio.run(main())
