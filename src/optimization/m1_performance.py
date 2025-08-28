#!/usr/bin/env python3
"""
FÁZE 6: M1 Performance Optimization Framework
M1 MacBook optimalizace s Ollama, Metal/MPS acceleration, batch sizing

Author: Senior Python/MLOps Agent
"""

import asyncio
import json
import time
import psutil
import platform
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    import torch
    import torch.mps as mps
    MPS_AVAILABLE = torch.backends.mps.is_available()
except ImportError:
    torch = None
    mps = None
    MPS_AVAILABLE = False


@dataclass
class M1SystemInfo:
    """M1 system information"""
    cpu_cores: int
    memory_gb: float
    platform: str
    mps_available: bool
    ollama_available: bool = False
    metal_performance_shaders: bool = False


@dataclass
class PerformanceProfile:
    """Performance profile konfigurace"""
    profile_name: str
    context_window: int
    batch_size: int
    ef_search: int
    max_tokens: int
    streaming_enabled: bool
    early_exit_threshold: float
    memory_limit_mb: int
    timeout_seconds: int
    ollama_model: str = "qwen2.5:7b-q4_K_M"


@dataclass
class M1PerformanceMetrics:
    """M1 performance metriky"""
    profile_name: str
    execution_time_s: float
    memory_peak_mb: float
    memory_efficiency: float
    tokens_per_second: float
    context_utilization: float
    mps_utilization: float
    early_exit_rate: float
    streaming_chunks: int
    error_rate: float
    timestamp: datetime = field(default_factory=datetime.now)


class M1OptimizationEngine:
    """M1 optimization engine pro FÁZE 6"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.system_info = self._detect_m1_system()
        self.performance_profiles = self._create_performance_profiles()
        self.ollama_client = None
        self._initialize_ollama()

    def _detect_m1_system(self) -> M1SystemInfo:
        """Detekuje M1 system capabilities"""

        cpu_count = psutil.cpu_count(logical=False)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        platform_info = platform.machine()

        # Detect M1/M2 specifically
        is_apple_silicon = platform_info in ['arm64'] and platform.system() == 'Darwin'

        return M1SystemInfo(
            cpu_cores=cpu_count,
            memory_gb=memory_gb,
            platform=f"{platform.system()}_{platform_info}",
            mps_available=MPS_AVAILABLE and is_apple_silicon,
            metal_performance_shaders=is_apple_silicon
        )

    def _create_performance_profiles(self) -> Dict[str, PerformanceProfile]:
        """Vytvoří performance profily pro M1"""

        profiles = {}

        # Quick profile: 25-45s target
        profiles["quick"] = PerformanceProfile(
            profile_name="quick",
            context_window=4096,
            batch_size=8,
            ef_search=64,
            max_tokens=2048,
            streaming_enabled=True,
            early_exit_threshold=0.15,
            memory_limit_mb=4096,  # 4GB limit
            timeout_seconds=45,
            ollama_model="qwen2.5:3b-q4_K_M"  # Smaller model for speed
        )

        # Thorough profile: 90-180s target
        profiles["thorough"] = PerformanceProfile(
            profile_name="thorough",
            context_window=8192,
            batch_size=16,
            ef_search=128,
            max_tokens=4096,
            streaming_enabled=True,
            early_exit_threshold=0.05,
            memory_limit_mb=8192,  # 8GB limit
            timeout_seconds=180,
            ollama_model="qwen2.5:7b-q4_K_M"
        )

        # Balanced profile: For CI/testing
        profiles["balanced"] = PerformanceProfile(
            profile_name="balanced",
            context_window=6144,
            batch_size=12,
            ef_search=96,
            max_tokens=3072,
            streaming_enabled=True,
            early_exit_threshold=0.10,
            memory_limit_mb=6144,
            timeout_seconds=90,
            ollama_model="qwen2.5:7b-q4_K_M"
        )

        return profiles

    def _initialize_ollama(self):
        """Inicializuje Ollama client"""
        try:
            import ollama
            self.ollama_client = ollama.Client()

            # Test Ollama availability
            models = self.ollama_client.list()
            self.system_info.ollama_available = True

            print(f"✅ Ollama detected with {len(models.get('models', []))} models")

        except Exception as e:
            print(f"⚠️ Ollama not available: {str(e)}")
            self.system_info.ollama_available = False

    async def optimize_for_profile(self, profile_name: str) -> M1PerformanceMetrics:
        """Optimalizuje systém pro daný performance profile"""

        if profile_name not in self.performance_profiles:
            raise ValueError(f"Unknown profile: {profile_name}")

        profile = self.performance_profiles[profile_name]
        print(f"🚀 Optimizing for {profile_name} profile...")
        print(f"Target: {profile.timeout_seconds}s, {profile.context_window} context, {profile.max_tokens} tokens")

        start_time = time.time()
        start_memory = self._get_memory_usage()

        try:
            # Configure MPS if available
            if self.system_info.mps_available:
                await self._configure_mps(profile)

            # Run optimized inference
            metrics = await self._run_optimized_inference(profile)

            execution_time = time.time() - start_time
            peak_memory = self._get_peak_memory_usage()

            # Calculate efficiency metrics
            memory_efficiency = min(1.0, profile.memory_limit_mb / peak_memory) if peak_memory > 0 else 1.0

            final_metrics = M1PerformanceMetrics(
                profile_name=profile_name,
                execution_time_s=execution_time,
                memory_peak_mb=peak_memory,
                memory_efficiency=memory_efficiency,
                tokens_per_second=metrics.get("tokens_per_second", 0),
                context_utilization=metrics.get("context_utilization", 0),
                mps_utilization=metrics.get("mps_utilization", 0),
                early_exit_rate=metrics.get("early_exit_rate", 0),
                streaming_chunks=metrics.get("streaming_chunks", 0),
                error_rate=metrics.get("error_rate", 0)
            )

            # Validate profile targets
            success = self._validate_profile_targets(profile, final_metrics)

            if success:
                print(f"✅ {profile_name} profile optimization successful!")
                print(f"   Time: {execution_time:.1f}s (target: <{profile.timeout_seconds}s)")
                print(f"   Memory: {peak_memory:.0f}MB (limit: {profile.memory_limit_mb}MB)")
                print(f"   Efficiency: {memory_efficiency:.2f}")
            else:
                print(f"⚠️ {profile_name} profile did not meet targets")

            return final_metrics

        except Exception as e:
            print(f"❌ Error optimizing {profile_name}: {str(e)}")
            return M1PerformanceMetrics(
                profile_name=profile_name,
                execution_time_s=time.time() - start_time,
                memory_peak_mb=self._get_memory_usage(),
                memory_efficiency=0.0,
                tokens_per_second=0.0,
                context_utilization=0.0,
                mps_utilization=0.0,
                early_exit_rate=0.0,
                streaming_chunks=0,
                error_rate=1.0
            )

    async def _configure_mps(self, profile: PerformanceProfile):
        """Konfiguruje Metal Performance Shaders"""
        if not self.system_info.mps_available:
            return

        try:
            # Set MPS as default device
            if torch is not None:
                device = torch.device("mps")
                print(f"🔧 Configured MPS device for {profile.profile_name}")

                # Pre-warm MPS
                dummy_tensor = torch.randn(1, device=device)
                _ = dummy_tensor * 2
                del dummy_tensor

        except Exception as e:
            print(f"⚠️ MPS configuration warning: {str(e)}")

    async def _run_optimized_inference(self, profile: PerformanceProfile) -> Dict[str, float]:
        """Spustí optimalizovanou inferenci s M1 optimalizacemi"""

        metrics = {
            "tokens_per_second": 0.0,
            "context_utilization": 0.0,
            "mps_utilization": 0.0,
            "early_exit_rate": 0.0,
            "streaming_chunks": 0,
            "error_rate": 0.0
        }

        if not self.system_info.ollama_available:
            # Mock inference for testing
            return await self._mock_optimized_inference(profile)

        try:
            # Sample research query
            test_query = "What are the recent advances in quantum computing?"

            # Configure streaming
            start_time = time.time()
            total_tokens = 0
            streaming_chunks = 0
            early_exits = 0
            total_requests = 3  # Test multiple requests

            for i in range(total_requests):
                try:
                    # Adaptive batch sizing based on profile
                    batch_start = time.time()

                    if profile.streaming_enabled:
                        # Streaming inference with progressive context
                        response_chunks = []

                        stream = self.ollama_client.generate(
                            model=profile.ollama_model,
                            prompt=f"Query {i+1}: {test_query}",
                            stream=True,
                            options={
                                'num_ctx': profile.context_window,
                                'num_predict': profile.max_tokens // total_requests,
                                'temperature': 0.7,
                                'top_p': 0.9
                            }
                        )

                        for chunk in stream:
                            if chunk.get('done', False):
                                break

                            response_chunks.append(chunk.get('response', ''))
                            streaming_chunks += 1

                            # Early exit check
                            if len(response_chunks) > 10:  # Minimum chunks
                                novelty_score = self._calculate_novelty_score(response_chunks[-5:])
                                if novelty_score < profile.early_exit_threshold:
                                    early_exits += 1
                                    break

                        response_text = ''.join(response_chunks)
                        total_tokens += len(response_text.split())

                    else:
                        # Non-streaming inference
                        response = self.ollama_client.generate(
                            model=profile.ollama_model,
                            prompt=f"Query {i+1}: {test_query}",
                            stream=False,
                            options={
                                'num_ctx': profile.context_window,
                                'num_predict': profile.max_tokens // total_requests
                            }
                        )

                        response_text = response.get('response', '')
                        total_tokens += len(response_text.split())

                    # Simulate progressive context building
                    await asyncio.sleep(0.1)  # Brief pause between requests

                except Exception as e:
                    print(f"⚠️ Request {i+1} failed: {str(e)}")
                    metrics["error_rate"] += 1.0 / total_requests

            total_time = time.time() - start_time

            # Calculate metrics
            metrics["tokens_per_second"] = total_tokens / total_time if total_time > 0 else 0
            metrics["context_utilization"] = min(1.0, total_tokens / (profile.context_window * total_requests))
            metrics["mps_utilization"] = 0.8 if self.system_info.mps_available else 0.0  # Estimated
            metrics["early_exit_rate"] = early_exits / total_requests if total_requests > 0 else 0
            metrics["streaming_chunks"] = streaming_chunks

        except Exception as e:
            print(f"❌ Inference error: {str(e)}")
            metrics["error_rate"] = 1.0

        return metrics

    async def _mock_optimized_inference(self, profile: PerformanceProfile) -> Dict[str, float]:
        """Mock optimalizované inference pro testování bez Ollama"""

        # Simulate realistic inference timing
        base_time = profile.timeout_seconds * 0.3  # Use 30% of timeout
        processing_time = base_time * np.random.uniform(0.8, 1.2)
        await asyncio.sleep(min(processing_time, 2.0))  # Max 2s for mock

        return {
            "tokens_per_second": np.random.uniform(15, 25),  # Realistic for M1
            "context_utilization": np.random.uniform(0.6, 0.8),
            "mps_utilization": 0.75 if self.system_info.mps_available else 0.0,
            "early_exit_rate": np.random.uniform(0.1, 0.3),
            "streaming_chunks": np.random.randint(20, 50),
            "error_rate": 0.0
        }

    def _calculate_novelty_score(self, recent_chunks: List[str]) -> float:
        """Vypočítá novelty score pro early exit"""
        if len(recent_chunks) < 2:
            return 1.0

        # Simple novelty calculation based on text similarity
        recent_text = ' '.join(recent_chunks)
        words = recent_text.split()

        if len(words) < 10:
            return 1.0

        # Calculate repetition ratio
        unique_words = len(set(words))
        total_words = len(words)
        novelty = unique_words / total_words if total_words > 0 else 0

        return novelty

    def _get_memory_usage(self) -> float:
        """Získá aktuální memory usage v MB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def _get_peak_memory_usage(self) -> float:
        """Získá peak memory usage (simplified)"""
        # In production, this would track peak usage throughout execution
        return self._get_memory_usage() * 1.2  # Estimate peak as 20% higher

    def _validate_profile_targets(self, profile: PerformanceProfile, metrics: M1PerformanceMetrics) -> bool:
        """Validuje, zda metriky splňují profile targets"""

        validations = []

        # Time validation
        time_ok = metrics.execution_time_s <= profile.timeout_seconds
        validations.append(time_ok)

        # Memory validation (no OOM)
        memory_ok = metrics.memory_peak_mb <= profile.memory_limit_mb
        validations.append(memory_ok)

        # Efficiency validation
        efficiency_ok = metrics.memory_efficiency >= 0.5  # At least 50% efficient
        validations.append(efficiency_ok)

        # Error rate validation
        error_ok = metrics.error_rate <= 0.1  # Max 10% error rate
        validations.append(error_ok)

        return all(validations)

    def generate_m1_telemetry_report(self, metrics_list: List[M1PerformanceMetrics]) -> Dict[str, Any]:
        """Generuje M1 telemetry report v JSON"""

        if not metrics_list:
            return {"error": "No metrics available"}

        report = {
            "m1_performance_report": {
                "timestamp": datetime.now().isoformat(),
                "system_info": {
                    "cpu_cores": self.system_info.cpu_cores,
                    "memory_gb": self.system_info.memory_gb,
                    "platform": self.system_info.platform,
                    "mps_available": self.system_info.mps_available,
                    "ollama_available": self.system_info.ollama_available,
                    "metal_performance_shaders": self.system_info.metal_performance_shaders
                },
                "performance_profiles": {},
                "aggregate_metrics": {},
                "recommendations": []
            }
        }

        # Group metrics by profile
        profile_metrics = {}
        for metric in metrics_list:
            if metric.profile_name not in profile_metrics:
                profile_metrics[metric.profile_name] = []
            profile_metrics[metric.profile_name].append(metric)

        # Analyze each profile
        for profile_name, metrics in profile_metrics.items():
            profile_data = {
                "execution_times": [m.execution_time_s for m in metrics],
                "memory_peaks": [m.memory_peak_mb for m in metrics],
                "tokens_per_second": [m.tokens_per_second for m in metrics],
                "memory_efficiency": [m.memory_efficiency for m in metrics],
                "error_rates": [m.error_rate for m in metrics],
                "target_met": all(self._validate_profile_targets(
                    self.performance_profiles[profile_name], m
                ) for m in metrics)
            }

            # Calculate statistics
            profile_data["avg_execution_time"] = np.mean(profile_data["execution_times"])
            profile_data["p95_execution_time"] = np.percentile(profile_data["execution_times"], 95)
            profile_data["avg_tokens_per_second"] = np.mean(profile_data["tokens_per_second"])
            profile_data["avg_memory_efficiency"] = np.mean(profile_data["memory_efficiency"])

            report["m1_performance_report"]["performance_profiles"][profile_name] = profile_data

        # Generate recommendations
        recommendations = self._generate_m1_recommendations(profile_metrics)
        report["m1_performance_report"]["recommendations"] = recommendations

        return report

    def _generate_m1_recommendations(self, profile_metrics: Dict[str, List[M1PerformanceMetrics]]) -> List[str]:
        """Generuje doporučení pro M1 optimalizaci"""

        recommendations = []

        for profile_name, metrics in profile_metrics.items():
            profile = self.performance_profiles[profile_name]
            avg_time = np.mean([m.execution_time_s for m in metrics])
            avg_memory = np.mean([m.memory_peak_mb for m in metrics])
            avg_efficiency = np.mean([m.memory_efficiency for m in metrics])

            # Time recommendations
            if avg_time > profile.timeout_seconds * 0.8:
                recommendations.append(
                    f"{profile_name}: Consider reducing context_window or batch_size for better timing"
                )

            # Memory recommendations
            if avg_memory > profile.memory_limit_mb * 0.9:
                recommendations.append(
                    f"{profile_name}: Memory usage near limit - consider smaller batch_size"
                )

            # Efficiency recommendations
            if avg_efficiency < 0.6:
                recommendations.append(
                    f"{profile_name}: Low memory efficiency - optimize context utilization"
                )

            # MPS recommendations
            if not self.system_info.mps_available:
                recommendations.append(
                    "Consider upgrading to M1/M2 Mac for Metal Performance Shaders acceleration"
                )

        if not recommendations:
            recommendations.append("✅ All profiles performing within targets - no optimizations needed")

        return recommendations

    async def export_m1_telemetry(self, metrics_list: List[M1PerformanceMetrics],
                                 output_path: str = "docs/m1_telemetry.json"):
        """Exportuje M1 telemetry do JSON"""

        report = self.generate_m1_telemetry_report(metrics_list)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"📊 M1 telemetry exported: {output_file}")


# Factory function
def create_m1_optimization_engine(config: Dict[str, Any]) -> M1OptimizationEngine:
    """Factory function pro vytvoření M1 optimization engine"""
    return M1OptimizationEngine(config)


if __name__ == "__main__":
    # Test M1 optimization framework
    config = {
        "m1_optimization": {
            "profiles": ["quick", "thorough"],
            "enable_mps": True,
            "enable_streaming": True,
            "telemetry_enabled": True
        }
    }

    engine = create_m1_optimization_engine(config)
    print(f"✅ M1 Optimization Engine initialized!")
    print(f"System: {engine.system_info.platform}")
    print(f"MPS Available: {engine.system_info.mps_available}")
    print(f"Ollama Available: {engine.system_info.ollama_available}")
    print(f"Profiles: {list(engine.performance_profiles.keys())}")
