#!/usr/bin/env python3
"""
M1 Device Manager s Metal/MPS optimalizacemi
FP16/AMP, batch sizing, streaming, early-exit optimalizace

Author: Senior Python/MLOps Agent
"""

import json
import time
import psutil
import platform
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import asyncio
import logging


@dataclass
class DeviceProfile:
    """Profil zařízení a jeho schopností"""
    device_type: str  # "mps", "cpu", "cuda"
    device_name: str
    total_memory_gb: float
    available_memory_gb: float
    supports_fp16: bool
    supports_metal: bool
    optimal_batch_size: int
    recommended_context_length: int


@dataclass
class PerformanceMetrics:
    """Výkonnostní metriky"""
    tokens_per_second: float
    memory_usage_gb: float
    memory_peak_gb: float
    cpu_usage_percent: float
    gpu_usage_percent: float
    temperature_celsius: Optional[float]
    power_usage_watts: Optional[float]
    inference_latency_ms: float
    throughput_queries_per_minute: float


class M1DeviceManager:
    """M1 optimalizovaný device manager"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Detekce zařízení
        self.device_profile = self._detect_device_capabilities()
        self.current_metrics = None

        # Optimalizační nastavení
        self.fp16_enabled = config.get("fp16_enabled", True)
        self.metal_enabled = config.get("metal_enabled", True)
        self.adaptive_batching = config.get("adaptive_batching", True)
        self.streaming_enabled = config.get("streaming_enabled", True)

        print(f"🚀 M1 Device Manager initialized: {self.device_profile.device_name}")
        print(f"   Device: {self.device_profile.device_type}")
        print(f"   Memory: {self.device_profile.available_memory_gb:.1f}GB available")
        print(f"   FP16: {self.device_profile.supports_fp16}")
        print(f"   Metal: {self.device_profile.supports_metal}")

    def _detect_device_capabilities(self) -> DeviceProfile:
        """Detekuje schopnosti zařízení"""
        system_info = platform.uname()
        is_apple_silicon = system_info.machine in ["arm64", "aarch64"] and system_info.system == "Darwin"

        # Memory detection
        memory_info = psutil.virtual_memory()
        total_memory_gb = memory_info.total / (1024**3)
        available_memory_gb = memory_info.available / (1024**3)

        if is_apple_silicon:
            # Apple Silicon M1/M2/M3
            device_type = "mps"
            device_name = self._get_apple_silicon_model()
            supports_fp16 = True
            supports_metal = True

            # M1/M2 specific optimizations
            if "M1" in device_name:
                optimal_batch_size = 4
                recommended_context_length = 4096
            elif "M2" in device_name:
                optimal_batch_size = 8
                recommended_context_length = 6144
            elif "M3" in device_name:
                optimal_batch_size = 12
                recommended_context_length = 8192
            else:
                optimal_batch_size = 4
                recommended_context_length = 4096

        else:
            # Fallback to CPU
            device_type = "cpu"
            device_name = f"{system_info.processor} ({psutil.cpu_count()} cores)"
            supports_fp16 = False
            supports_metal = False
            optimal_batch_size = 1
            recommended_context_length = 2048

        return DeviceProfile(
            device_type=device_type,
            device_name=device_name,
            total_memory_gb=total_memory_gb,
            available_memory_gb=available_memory_gb,
            supports_fp16=supports_fp16,
            supports_metal=supports_metal,
            optimal_batch_size=optimal_batch_size,
            recommended_context_length=recommended_context_length
        )

    def _get_apple_silicon_model(self) -> str:
        """Detekuje konkrétní model Apple Silicon"""
        try:
            import subprocess
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                capture_output=True, text=True
            )
            output = result.stdout

            if "Apple M1" in output:
                if "Pro" in output:
                    return "Apple M1 Pro"
                elif "Max" in output:
                    return "Apple M1 Max"
                elif "Ultra" in output:
                    return "Apple M1 Ultra"
                else:
                    return "Apple M1"
            elif "Apple M2" in output:
                if "Pro" in output:
                    return "Apple M2 Pro"
                elif "Max" in output:
                    return "Apple M2 Max"
                elif "Ultra" in output:
                    return "Apple M2 Ultra"
                else:
                    return "Apple M2"
            elif "Apple M3" in output:
                if "Pro" in output:
                    return "Apple M3 Pro"
                elif "Max" in output:
                    return "Apple M3 Max"
                else:
                    return "Apple M3"
            else:
                return "Apple Silicon (Unknown)"

        except Exception:
            return "Apple Silicon"

    def get_optimal_model_config(self, model_size: str = "3b") -> Dict[str, Any]:
        """Vrátí optimální konfiguraci pro daný model"""
        base_config = {
            "device": self.device_profile.device_type,
            "fp16": self.fp16_enabled and self.device_profile.supports_fp16,
            "use_metal": self.metal_enabled and self.device_profile.supports_metal,
            "batch_size": self.device_profile.optimal_batch_size,
            "max_context_length": self.device_profile.recommended_context_length
        }

        # Model-specific optimizations
        if model_size == "3b":
            # Llama 3.2 3B optimizations
            base_config.update({
                "num_threads": min(8, psutil.cpu_count()),
                "memory_budget_gb": min(4, self.device_profile.available_memory_gb * 0.6),
                "quantization": "q4_k_m",
                "context_length": 4096,
                "rope_freq_base": 500000  # Extended context
            })
        elif model_size == "8b":
            # Llama 3.1 8B optimizations
            base_config.update({
                "num_threads": min(10, psutil.cpu_count()),
                "memory_budget_gb": min(8, self.device_profile.available_memory_gb * 0.7),
                "quantization": "q4_k_m",
                "context_length": 8192,
                "rope_freq_base": 500000
            })
        else:
            # Conservative defaults
            base_config.update({
                "num_threads": min(6, psutil.cpu_count()),
                "memory_budget_gb": min(3, self.device_profile.available_memory_gb * 0.5),
                "quantization": "q4_k_m",
                "context_length": 2048
            })

        return base_config

    async def adaptive_batch_sizing(self,
                                  initial_batch_size: int,
                                  target_memory_usage: float = 0.8) -> int:
        """Adaptivní batch sizing podle dostupné paměti"""
        if not self.adaptive_batching:
            return initial_batch_size

        current_memory = psutil.virtual_memory()
        memory_usage_ratio = 1 - (current_memory.available / current_memory.total)

        if memory_usage_ratio > target_memory_usage:
            # Snížit batch size
            new_batch_size = max(1, initial_batch_size // 2)
            self.logger.info(f"🔽 Reducing batch size: {initial_batch_size} → {new_batch_size}")
            return new_batch_size
        elif memory_usage_ratio < target_memory_usage * 0.6:
            # Zvýšit batch size (opatrně)
            new_batch_size = min(initial_batch_size * 2, self.device_profile.optimal_batch_size * 2)
            self.logger.info(f"🔼 Increasing batch size: {initial_batch_size} → {new_batch_size}")
            return new_batch_size

        return initial_batch_size

    def monitor_performance(self) -> PerformanceMetrics:
        """Monitoruj výkonnostní metriky"""
        # CPU metrics
        cpu_usage = psutil.cpu_percent(interval=0.1)

        # Memory metrics
        memory_info = psutil.virtual_memory()
        memory_usage_gb = (memory_info.total - memory_info.available) / (1024**3)

        # Temperature (Apple Silicon specific)
        temperature = self._get_cpu_temperature()

        # Power usage estimate
        power_usage = self._estimate_power_usage(cpu_usage)

        metrics = PerformanceMetrics(
            tokens_per_second=0.0,  # Would be calculated during inference
            memory_usage_gb=memory_usage_gb,
            memory_peak_gb=memory_usage_gb,  # Would track peak during session
            cpu_usage_percent=cpu_usage,
            gpu_usage_percent=0.0,  # MPS doesn't expose GPU usage easily
            temperature_celsius=temperature,
            power_usage_watts=power_usage,
            inference_latency_ms=0.0,  # Would be measured during actual inference
            throughput_queries_per_minute=0.0  # Would be calculated from query times
        )

        self.current_metrics = metrics
        return metrics

    def _get_cpu_temperature(self) -> Optional[float]:
        """Získá teplotu CPU (Apple Silicon)"""
        try:
            import subprocess
            result = subprocess.run(
                ["sudo", "powermetrics", "--samplers", "smc", "-n", "1", "-i", "1000"],
                capture_output=True, text=True, timeout=5
            )

            # Parse temperature from powermetrics output
            for line in result.stdout.split('\n'):
                if "CPU die temperature" in line:
                    temp_str = line.split(':')[-1].strip()
                    temp_value = float(temp_str.replace('C', '').strip())
                    return temp_value

        except Exception:
            pass

        return None

    def _estimate_power_usage(self, cpu_usage: float) -> Optional[float]:
        """Odhadne spotřebu energie"""
        if self.device_profile.device_type == "mps":
            # Apple Silicon power estimates
            if "M1" in self.device_profile.device_name:
                base_power = 10  # Watts
                max_power = 40
            elif "M2" in self.device_profile.device_name:
                base_power = 12
                max_power = 45
            elif "M3" in self.device_profile.device_name:
                base_power = 15
                max_power = 50
            else:
                base_power = 10
                max_power = 40

            # Linear interpolation based on CPU usage
            estimated_power = base_power + (cpu_usage / 100.0) * (max_power - base_power)
            return estimated_power

        return None

    def check_thermal_throttling(self) -> Dict[str, Any]:
        """Kontrola thermal throttling"""
        temperature = self._get_cpu_temperature()
        cpu_usage = psutil.cpu_percent(interval=0.5)

        throttling_status = {
            "is_throttling": False,
            "temperature_celsius": temperature,
            "cpu_usage_percent": cpu_usage,
            "recommendation": "normal_operation"
        }

        if temperature and temperature > 80:
            throttling_status["is_throttling"] = True
            if temperature > 90:
                throttling_status["recommendation"] = "reduce_workload_immediately"
            else:
                throttling_status["recommendation"] = "reduce_workload_gradually"

        return throttling_status

    def suggest_performance_profile(self, workload_type: str = "thorough") -> Dict[str, Any]:
        """Navrhne výkonnostní profil"""
        current_metrics = self.monitor_performance()
        thermal_status = self.check_thermal_throttling()

        if workload_type == "quick":
            # Rychlý profil (25-45s)
            profile = {
                "name": "quick",
                "target_time_seconds": 35,
                "context_length": 4096,
                "max_iterations": 3,
                "batch_size": self.device_profile.optimal_batch_size,
                "use_streaming": True,
                "early_exit_threshold": 0.85
            }
        elif workload_type == "thorough":
            # Důkladný profil (90-180s)
            profile = {
                "name": "thorough",
                "target_time_seconds": 120,
                "context_length": 8192,
                "max_iterations": 8,
                "batch_size": self.device_profile.optimal_batch_size,
                "use_streaming": True,
                "early_exit_threshold": 0.95
            }
        else:
            # Balanced profil
            profile = {
                "name": "balanced",
                "target_time_seconds": 60,
                "context_length": 6144,
                "max_iterations": 5,
                "batch_size": self.device_profile.optimal_batch_size,
                "use_streaming": True,
                "early_exit_threshold": 0.90
            }

        # Thermal adjustments
        if thermal_status["is_throttling"]:
            profile["batch_size"] = max(1, profile["batch_size"] // 2)
            profile["context_length"] = min(profile["context_length"], 4096)
            profile["max_iterations"] = max(1, profile["max_iterations"] // 2)

        # Memory adjustments
        if current_metrics.memory_usage_gb > self.device_profile.total_memory_gb * 0.8:
            profile["context_length"] = min(profile["context_length"], 4096)
            profile["batch_size"] = max(1, profile["batch_size"] // 2)

        return profile

    def optimize_for_streaming(self) -> Dict[str, Any]:
        """Optimalizace pro streaming"""
        return {
            "chunk_size": 512,  # Tokens per chunk
            "buffer_size": 2048,  # Token buffer
            "progressive_context": True,
            "early_yield": True,
            "stream_threshold": 0.7  # Confidence threshold for streaming
        }

    def get_device_status_report(self) -> Dict[str, Any]:
        """Kompletní status report zařízení"""
        current_metrics = self.monitor_performance()
        thermal_status = self.check_thermal_throttling()

        return {
            "timestamp": datetime.now().isoformat(),
            "device_profile": {
                "device_type": self.device_profile.device_type,
                "device_name": self.device_profile.device_name,
                "total_memory_gb": self.device_profile.total_memory_gb,
                "available_memory_gb": self.device_profile.available_memory_gb,
                "supports_fp16": self.device_profile.supports_fp16,
                "supports_metal": self.device_profile.supports_metal
            },
            "current_metrics": {
                "memory_usage_gb": current_metrics.memory_usage_gb,
                "cpu_usage_percent": current_metrics.cpu_usage_percent,
                "temperature_celsius": current_metrics.temperature_celsius,
                "power_usage_watts": current_metrics.power_usage_watts
            },
            "thermal_status": thermal_status,
            "optimization_settings": {
                "fp16_enabled": self.fp16_enabled,
                "metal_enabled": self.metal_enabled,
                "adaptive_batching": self.adaptive_batching,
                "streaming_enabled": self.streaming_enabled
            },
            "recommended_profiles": {
                "quick": self.suggest_performance_profile("quick"),
                "balanced": self.suggest_performance_profile("balanced"),
                "thorough": self.suggest_performance_profile("thorough")
            }
        }

    async def benchmark_device(self, duration_seconds: int = 30) -> Dict[str, Any]:
        """Benchmark zařízení"""
        print(f"🏃 Running device benchmark for {duration_seconds}s...")

        start_time = time.time()
        metrics_history = []

        # Simulate workload
        while time.time() - start_time < duration_seconds:
            metrics = self.monitor_performance()
            metrics_history.append({
                "timestamp": time.time() - start_time,
                "cpu_usage": metrics.cpu_usage_percent,
                "memory_usage_gb": metrics.memory_usage_gb,
                "temperature": metrics.temperature_celsius
            })

            # Simulate CPU load
            await asyncio.sleep(0.5)

        # Calculate benchmark results
        if metrics_history:
            avg_cpu = sum(m["cpu_usage"] for m in metrics_history) / len(metrics_history)
            avg_memory = sum(m["memory_usage_gb"] for m in metrics_history) / len(metrics_history)
            max_temp = max((m["temperature"] for m in metrics_history if m["temperature"]), default=0)

            benchmark_results = {
                "duration_seconds": duration_seconds,
                "average_cpu_usage": avg_cpu,
                "average_memory_usage_gb": avg_memory,
                "peak_temperature_celsius": max_temp,
                "thermal_throttling_detected": max_temp > 85 if max_temp else False,
                "stability_score": self._calculate_stability_score(metrics_history),
                "device_rating": self._calculate_device_rating(avg_cpu, avg_memory, max_temp)
            }
        else:
            benchmark_results = {"error": "No metrics collected"}

        print(f"✅ Benchmark completed: stability={benchmark_results.get('stability_score', 0):.2f}")
        return benchmark_results

    def _calculate_stability_score(self, metrics_history: List[Dict]) -> float:
        """Vypočítá skóre stability"""
        if len(metrics_history) < 2:
            return 0.0

        cpu_values = [m["cpu_usage"] for m in metrics_history]
        memory_values = [m["memory_usage_gb"] for m in metrics_history]

        # Calculate coefficient of variation (stability measure)
        import statistics

        cpu_cv = statistics.stdev(cpu_values) / statistics.mean(cpu_values) if statistics.mean(cpu_values) > 0 else 1
        memory_cv = statistics.stdev(memory_values) / statistics.mean(memory_values) if statistics.mean(memory_values) > 0 else 1

        # Lower CV = higher stability
        stability_score = max(0, 1 - (cpu_cv + memory_cv) / 2)
        return stability_score

    def _calculate_device_rating(self, avg_cpu: float, avg_memory: float, max_temp: float) -> str:
        """Vypočítá celkové hodnocení zařízení"""
        score = 0

        # CPU performance
        if avg_cpu < 50:
            score += 3
        elif avg_cpu < 70:
            score += 2
        else:
            score += 1

        # Memory efficiency
        if avg_memory < self.device_profile.total_memory_gb * 0.6:
            score += 3
        elif avg_memory < self.device_profile.total_memory_gb * 0.8:
            score += 2
        else:
            score += 1

        # Thermal performance
        if max_temp and max_temp < 70:
            score += 3
        elif max_temp and max_temp < 85:
            score += 2
        else:
            score += 1

        # Rating scale
        if score >= 8:
            return "excellent"
        elif score >= 6:
            return "good"
        elif score >= 4:
            return "fair"
        else:
            return "poor"
