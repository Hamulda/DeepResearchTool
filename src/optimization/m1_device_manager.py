#!/usr/bin/env python3
"""M1 Device Manager - Dynamická Správa GPU/CPU Zdrojů pro Apple Silicon
Optimální alokace úkolů mezi Metal GPU a ARM CPU podle priority a dostupnosti zdrojů

Author: Senior Python/MLOps Agent
"""

from dataclasses import dataclass
from enum import Enum
import logging
import platform
import subprocess
import threading
import time
from typing import Any

import psutil

logger = logging.getLogger(__name__)


class Priority(Enum):
    """Priority úkolů pro alokaci zdrojů"""

    LOW = "low"          # Běžné úkoly - preferuj CPU
    MEDIUM = "medium"    # Standardní úkoly - vyváženě
    HIGH = "high"        # Kritické úkoly - preferuj GPU
    CRITICAL = "critical" # Nejdůležitější - maximum GPU


class DeviceType(Enum):
    """Typy zařízení"""

    CPU = "cpu"
    MPS = "mps"          # Metal Performance Shaders (Apple GPU)
    CUDA = "cuda"        # NVIDIA GPU (fallback)


@dataclass
class ResourceInfo:
    """Informace o zdrojích zařízení"""

    device_type: DeviceType
    memory_total_mb: float
    memory_used_mb: float
    memory_available_mb: float
    utilization_percent: float
    temperature_celsius: float | None = None
    power_draw_watts: float | None = None


@dataclass
class InferenceParams:
    """Parametry pro inference optimalizované pro M1"""

    device: str
    n_gpu_layers: int
    n_threads: int
    batch_size: int
    context_length: int

    # M1 specifické optimalizace
    use_metal: bool = True
    memory_lock: bool = True
    numa_affinity: bool = True


class ResourceManager:
    """M1 Resource Manager pro optimální využití Apple Silicon architektury
    
    KLÍČOVÉ FUNKCE:
    - Dynamické rozhodování CPU vs GPU na základě priority a zátěže
    - Monitoring unified memory systému M1
    - Optimalizace pro Metal Performance Shaders
    - Tepelný management pro sustained performance
    """

    def __init__(self):
        """Inicializace Resource Manager s M1 optimalizacemi"""
        self.is_m1 = self._detect_m1_architecture()
        self.device_info = self._detect_available_devices()

        # Resource monitoring
        self.monitoring_enabled = True
        self.monitoring_thread: threading.Thread | None = None
        self.resource_history: list[dict[str, Any]] = []

        # Performance tracking
        self.allocation_stats = {
            "total_allocations": 0,
            "gpu_allocations": 0,
            "cpu_allocations": 0,
            "avg_gpu_utilization": 0.0,
            "memory_pressure_events": 0
        }

        # Thermal management
        self.thermal_throttling_threshold = 85.0  # °C
        self.thermal_throttling_active = False

        # Start monitoring
        self._start_resource_monitoring()

        logger.info(f"✅ Resource Manager inicializován pro {'Apple Silicon M1' if self.is_m1 else 'jinou architekturu'}")
        logger.info(f"📱 Dostupná zařízení: {list(self.device_info.keys())}")

    def _detect_m1_architecture(self) -> bool:
        """Detekce Apple Silicon M1 architektury"""
        try:
            # Kontrola procesoru
            processor = platform.processor()
            machine = platform.machine()

            is_arm64 = machine == 'arm64'
            is_apple = processor == 'arm'

            if is_arm64 and is_apple:
                # Dodatečná kontrola na M1 chip
                try:
                    result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                          check=False, capture_output=True, text=True)
                    cpu_brand = result.stdout.strip()
                    is_m_series = 'Apple' in cpu_brand and ('M1' in cpu_brand or 'M2' in cpu_brand or 'M3' in cpu_brand)

                    if is_m_series:
                        logger.info(f"🍎 Detekován Apple Silicon: {cpu_brand}")
                        return True

                except Exception:
                    pass

            return False

        except Exception as e:
            logger.warning(f"Chyba při detekci M1: {e}")
            return False

    def _detect_available_devices(self) -> dict[str, DeviceType]:
        """Detekce dostupných compute zařízení"""
        devices = {}

        # CPU je vždy dostupný
        devices["cpu"] = DeviceType.CPU

        if self.is_m1:
            # M1 má integrovaný GPU přístupný přes Metal Performance Shaders
            devices["mps"] = DeviceType.MPS
            logger.info("🔧 Metal Performance Shaders (MPS) dostupné")

        # Zkus detekovat CUDA (unlikely na M1, ale pro completeness)
        try:
            import torch
            if torch.cuda.is_available():
                devices["cuda"] = DeviceType.CUDA
                logger.info("🟢 CUDA GPU detekováno")
        except ImportError:
            pass

        return devices

    def get_inference_params(self, priority: str = "medium") -> InferenceParams:
        """HLAVNÍ METODA: Získání optimálních inference parametrů podle priority
        
        Args:
            priority: Priorita úkolu (low, medium, high, critical)
            
        Returns:
            Optimalizované inference parametry pro M1

        """
        try:
            priority_enum = Priority(priority)
        except ValueError:
            logger.warning(f"Neznámá priorita: {priority}, použiji medium")
            priority_enum = Priority.MEDIUM

        # Získání aktuálního stavu zdrojů
        cpu_info = self._get_cpu_info()
        gpu_info = self._get_gpu_info() if "mps" in self.device_info else None

        # Rozhodování na základě priority a dostupnosti zdrojů
        if priority_enum == Priority.CRITICAL:
            return self._get_critical_priority_params(cpu_info, gpu_info)
        if priority_enum == Priority.HIGH:
            return self._get_high_priority_params(cpu_info, gpu_info)
        if priority_enum == Priority.MEDIUM:
            return self._get_medium_priority_params(cpu_info, gpu_info)
        # LOW
        return self._get_low_priority_params(cpu_info, gpu_info)

    def _get_critical_priority_params(self, cpu_info: ResourceInfo, gpu_info: ResourceInfo | None) -> InferenceParams:
        """Parametry pro kritickou prioritu - maximum GPU"""
        if gpu_info and gpu_info.memory_available_mb > 1000 and not self.thermal_throttling_active:
            # Plné využití GPU
            params = InferenceParams(
                device="mps",
                n_gpu_layers=-1,           # Všechny vrstvy na GPU
                n_threads=2,               # Minimum CPU threads
                batch_size=8,              # Větší batch pro GPU
                context_length=8192,       # Maximum context
                use_metal=True,
                memory_lock=True,
                numa_affinity=False        # GPU nepotřebuje NUMA
            )
            self.allocation_stats["gpu_allocations"] += 1
        else:
            # Fallback na výkonný CPU setup
            params = InferenceParams(
                device="cpu",
                n_gpu_layers=0,
                n_threads=8,               # Maximum CPU threads pro M1
                batch_size=4,
                context_length=8192,
                use_metal=False,
                memory_lock=True,
                numa_affinity=True
            )
            self.allocation_stats["cpu_allocations"] += 1

        self.allocation_stats["total_allocations"] += 1
        logger.debug(f"🔴 Critical priority: {params.device} (layers: {params.n_gpu_layers})")

        return params

    def _get_high_priority_params(self, cpu_info: ResourceInfo, gpu_info: ResourceInfo | None) -> InferenceParams:
        """Parametry pro vysokou prioritu - preferuj GPU"""
        if gpu_info and gpu_info.memory_available_mb > 2000 and gpu_info.utilization_percent < 80:
            # GPU s většinou vrstev
            params = InferenceParams(
                device="mps",
                n_gpu_layers=32,           # Většina vrstev na GPU
                n_threads=4,
                batch_size=6,
                context_length=6144,
                use_metal=True,
                memory_lock=True,
                numa_affinity=False
            )
            self.allocation_stats["gpu_allocations"] += 1
        # Hybrid CPU/GPU nebo plný CPU
        elif gpu_info and gpu_info.memory_available_mb > 500:
            params = InferenceParams(
                device="mps",
                n_gpu_layers=16,       # Polovina vrstev na GPU
                n_threads=6,
                batch_size=4,
                context_length=4096,
                use_metal=True,
                memory_lock=True,
                numa_affinity=False
            )
            self.allocation_stats["gpu_allocations"] += 1
        else:
            params = InferenceParams(
                device="cpu",
                n_gpu_layers=0,
                n_threads=6,
                batch_size=4,
                context_length=4096,
                use_metal=False,
                memory_lock=True,
                numa_affinity=True
            )
            self.allocation_stats["cpu_allocations"] += 1

        self.allocation_stats["total_allocations"] += 1
        logger.debug(f"🟠 High priority: {params.device} (layers: {params.n_gpu_layers})")

        return params

    def _get_medium_priority_params(self, cpu_info: ResourceInfo, gpu_info: ResourceInfo | None) -> InferenceParams:
        """Parametry pro střední prioritu - vyvážené využití"""
        # Rozhodování na základě aktuální zátěže
        if gpu_info and gpu_info.utilization_percent < 50 and gpu_info.memory_available_mb > 1500:
            # Mírné využití GPU
            params = InferenceParams(
                device="mps",
                n_gpu_layers=8,            # Část vrstev na GPU
                n_threads=4,
                batch_size=4,
                context_length=4096,
                use_metal=True,
                memory_lock=True,
                numa_affinity=False
            )
            self.allocation_stats["gpu_allocations"] += 1
        else:
            # Preferuj CPU pro stability
            params = InferenceParams(
                device="cpu",
                n_gpu_layers=0,
                n_threads=4,
                batch_size=2,
                context_length=4096,
                use_metal=False,
                memory_lock=True,
                numa_affinity=True
            )
            self.allocation_stats["cpu_allocations"] += 1

        self.allocation_stats["total_allocations"] += 1
        logger.debug(f"🟡 Medium priority: {params.device} (layers: {params.n_gpu_layers})")

        return params

    def _get_low_priority_params(self, cpu_info: ResourceInfo, gpu_info: ResourceInfo | None) -> InferenceParams:
        """Parametry pro nízkou prioritu - šetři zdroje"""
        # Vždy CPU pro nízkou prioritu (šetříme GPU pro důležitější úkoly)
        params = InferenceParams(
            device="cpu",
            n_gpu_layers=0,
            n_threads=2,                   # Minimum threads
            batch_size=1,                  # Malý batch
            context_length=2048,           # Kratší context
            use_metal=False,
            memory_lock=False,             # Neblokuj memory
            numa_affinity=True
        )

        self.allocation_stats["cpu_allocations"] += 1
        self.allocation_stats["total_allocations"] += 1
        logger.debug("🟢 Low priority: CPU only")

        return params

    def _get_cpu_info(self) -> ResourceInfo:
        """Získání informací o CPU"""
        try:
            # Memory info
            memory = psutil.virtual_memory()

            # CPU utilization
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Temperature (M1 specific)
            temperature = self._get_cpu_temperature() if self.is_m1 else None

            return ResourceInfo(
                device_type=DeviceType.CPU,
                memory_total_mb=memory.total / (1024 * 1024),
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                utilization_percent=cpu_percent,
                temperature_celsius=temperature
            )

        except Exception as e:
            logger.warning(f"Chyba při získávání CPU info: {e}")
            return ResourceInfo(
                device_type=DeviceType.CPU,
                memory_total_mb=8192,  # Default pro M1
                memory_used_mb=4096,
                memory_available_mb=4096,
                utilization_percent=50.0
            )

    def _get_gpu_info(self) -> ResourceInfo | None:
        """Získání informací o GPU (MPS)"""
        if not self.is_m1 or "mps" not in self.device_info:
            return None

        try:
            # Na M1 je GPU integrovaný, používá unified memory
            memory = psutil.virtual_memory()

            # Odhad GPU utilizace (simplified)
            # V reálné implementaci by se použil Metal API
            gpu_utilization = self._estimate_gpu_utilization()

            # M1 GPU používá část unified memory
            estimated_gpu_memory = memory.total * 0.6  # ~60% pro GPU na M1
            estimated_gpu_used = estimated_gpu_memory * (gpu_utilization / 100.0)

            return ResourceInfo(
                device_type=DeviceType.MPS,
                memory_total_mb=estimated_gpu_memory / (1024 * 1024),
                memory_used_mb=estimated_gpu_used / (1024 * 1024),
                memory_available_mb=(estimated_gpu_memory - estimated_gpu_used) / (1024 * 1024),
                utilization_percent=gpu_utilization,
                temperature_celsius=self._get_gpu_temperature()
            )

        except Exception as e:
            logger.warning(f"Chyba při získávání GPU info: {e}")
            return None

    def _get_cpu_temperature(self) -> float | None:
        """Získání teploty CPU (M1 specific)"""
        try:
            # M1 temperature monitoring
            result = subprocess.run(['sudo', 'powermetrics', '--samplers', 'smc', '-n', '1', '--show-process-coalition'],
                                  check=False, capture_output=True, text=True, timeout=2)

            if result.returncode == 0:
                # Parse temperature from powermetrics output
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'CPU die temperature' in line:
                        # Extract temperature value
                        import re
                        temp_match = re.search(r'(\d+\.?\d*)', line)
                        if temp_match:
                            return float(temp_match.group(1))

        except Exception:
            pass

        return None

    def _get_gpu_temperature(self) -> float | None:
        """Získání teploty GPU (M1 integrated)"""
        # Na M1 je GPU integrovaný, má podobnou teplotu jako CPU
        cpu_temp = self._get_cpu_temperature()
        if cpu_temp:
            return cpu_temp + 5.0  # GPU je typicky o něco teplejší

        return None

    def _estimate_gpu_utilization(self) -> float:
        """Odhad GPU utilizace (simplified pro M1)"""
        try:
            # Použij aktivitu Graphics procesů jako proxy
            result = subprocess.run(['top', '-l', '1', '-n', '0'],
                                  check=False, capture_output=True, text=True, timeout=2)

            if result.returncode == 0:
                # Hledej procesy s vysokou GPU aktivitou
                lines = result.stdout.split('\n')
                gpu_activity = 0.0

                for line in lines:
                    if 'WindowServer' in line or 'Metal' in line:
                        # Extract CPU usage as proxy for GPU usage
                        import re
                        cpu_match = re.search(r'(\d+\.?\d*)%', line)
                        if cpu_match:
                            gpu_activity = max(gpu_activity, float(cpu_match.group(1)))

                return min(gpu_activity * 1.5, 100.0)  # Scale up a bit

        except Exception:
            pass

        return 30.0  # Conservative default

    def _start_resource_monitoring(self):
        """Spuštění background monitoringu zdrojů"""
        if not self.monitoring_enabled:
            return

        def monitor_resources():
            while self.monitoring_enabled:
                try:
                    cpu_info = self._get_cpu_info()
                    gpu_info = self._get_gpu_info()

                    # Check thermal throttling
                    if cpu_info.temperature_celsius and cpu_info.temperature_celsius > self.thermal_throttling_threshold:
                        if not self.thermal_throttling_active:
                            logger.warning(f"🌡️ Thermal throttling aktivní: {cpu_info.temperature_celsius:.1f}°C")
                            self.thermal_throttling_active = True
                    elif self.thermal_throttling_active:
                        logger.info("🌡️ Thermal throttling deaktivován")
                        self.thermal_throttling_active = False

                    # Check memory pressure
                    if cpu_info.memory_available_mb < 1000:  # Less than 1GB available
                        self.allocation_stats["memory_pressure_events"] += 1
                        logger.warning(f"⚠️ Memory pressure: {cpu_info.memory_available_mb:.0f}MB dostupných")

                    # Store history
                    self.resource_history.append({
                        "timestamp": time.time(),
                        "cpu": cpu_info,
                        "gpu": gpu_info,
                        "thermal_throttling": self.thermal_throttling_active
                    })

                    # Keep only last 100 measurements
                    if len(self.resource_history) > 100:
                        self.resource_history.pop(0)

                    # Update average GPU utilization
                    if gpu_info:
                        current_avg = self.allocation_stats["avg_gpu_utilization"]
                        measurements = len([h for h in self.resource_history if h["gpu"]])
                        if measurements > 0:
                            self.allocation_stats["avg_gpu_utilization"] = (
                                (current_avg * (measurements - 1) + gpu_info.utilization_percent) / measurements
                            )

                    time.sleep(5)  # Monitor every 5 seconds

                except Exception as e:
                    logger.error(f"Chyba v resource monitoring: {e}")
                    time.sleep(10)

        self.monitoring_thread = threading.Thread(target=monitor_resources, daemon=True)
        self.monitoring_thread.start()
        logger.info("📊 Resource monitoring spuštěn")

    def stop_monitoring(self):
        """Zastavení resource monitoringu"""
        self.monitoring_enabled = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("📊 Resource monitoring zastaven")

    def get_resource_report(self) -> dict[str, Any]:
        """Získání kompletního reportu o zdrojích"""
        current_cpu = self._get_cpu_info()
        current_gpu = self._get_gpu_info()

        total_allocations = self.allocation_stats["total_allocations"]

        gpu_usage_percent = 0.0
        if total_allocations > 0:
            gpu_usage_percent = (self.allocation_stats["gpu_allocations"] / total_allocations) * 100

        return {
            "system_info": {
                "is_apple_silicon": self.is_m1,
                "available_devices": list(self.device_info.keys()),
                "thermal_throttling_active": self.thermal_throttling_active
            },
            "current_resources": {
                "cpu": {
                    "memory_available_mb": round(current_cpu.memory_available_mb, 0),
                    "utilization_percent": round(current_cpu.utilization_percent, 1),
                    "temperature_celsius": current_cpu.temperature_celsius
                },
                "gpu": {
                    "memory_available_mb": round(current_gpu.memory_available_mb, 0) if current_gpu else None,
                    "utilization_percent": round(current_gpu.utilization_percent, 1) if current_gpu else None,
                    "temperature_celsius": current_gpu.temperature_celsius if current_gpu else None
                } if current_gpu else None
            },
            "allocation_statistics": {
                "total_allocations": total_allocations,
                "gpu_usage_percent": round(gpu_usage_percent, 1),
                "cpu_usage_percent": round(100 - gpu_usage_percent, 1),
                "avg_gpu_utilization": round(self.allocation_stats["avg_gpu_utilization"], 1),
                "memory_pressure_events": self.allocation_stats["memory_pressure_events"]
            },
            "optimization_recommendations": self._generate_optimization_recommendations(current_cpu, current_gpu)
        }

    def _generate_optimization_recommendations(self, cpu_info: ResourceInfo, gpu_info: ResourceInfo | None) -> list[str]:
        """Generování doporučení pro optimalizaci"""
        recommendations = []

        # Memory recommendations
        if cpu_info.memory_available_mb < 2000:
            recommendations.append("⚠️ Nízká dostupná paměť - zvažte ukončení nepotřebných aplikací")

        # CPU utilization
        if cpu_info.utilization_percent > 80:
            recommendations.append("🔥 Vysoká CPU zátěž - snižte prioritu nekriti  ckých úkolů")

        # GPU recommendations
        if gpu_info:
            if gpu_info.utilization_percent < 30 and self.allocation_stats["gpu_allocations"] < self.allocation_stats["cpu_allocations"]:
                recommendations.append("💡 GPU je nedostatečně využíván - zvažte přesun úkolů na GPU")
            elif gpu_info.utilization_percent > 90:
                recommendations.append("🔥 GPU přetížen - přesuňte některé úkoly na CPU")

        # Thermal recommendations
        if self.thermal_throttling_active:
            recommendations.append("🌡️ Aktivní thermal throttling - snižte zátěž nebo vyčkejte na ochlazení")

        if not recommendations:
            recommendations.append("✅ Systém běží optimálně")

        return recommendations


# Globální instance resource manageru
_resource_manager: ResourceManager | None = None


def get_resource_manager() -> ResourceManager:
    """Získání globální instance resource manageru"""
    global _resource_manager

    if _resource_manager is None:
        _resource_manager = ResourceManager()

    return _resource_manager


def reset_resource_manager():
    """Reset globální instance (pro testování)"""
    global _resource_manager
    if _resource_manager:
        _resource_manager.stop_monitoring()
    _resource_manager = None


# Convenience funkce
def get_inference_params_for_priority(priority: str = "medium") -> InferenceParams:
    """Rychlá funkce pro získání inference parametrů
    
    Args:
        priority: Priorita úkolu (low, medium, high, critical)
        
    Returns:
        Optimalizované inference parametry

    """
    manager = get_resource_manager()
    return manager.get_inference_params(priority)


if __name__ == "__main__":
    # Test functionality
    import time

    # Vytvoření resource manageru
    rm = ResourceManager()

    # Test různých priorit
    priorities = ["low", "medium", "high", "critical"]

    print("🧪 Test inference parametrů pro různé priority:\n")

    for priority in priorities:
        params = rm.get_inference_params(priority)
        print(f"Priority: {priority}")
        print(f"  Device: {params.device}")
        print(f"  GPU layers: {params.n_gpu_layers}")
        print(f"  Threads: {params.n_threads}")
        print(f"  Batch size: {params.batch_size}")
        print(f"  Use Metal: {params.use_metal}")
        print()

    # Počkej na několik monitoring cyklů
    print("⏳ Čekám na monitoring data...")
    time.sleep(12)

    # Zobraz report
    report = rm.get_resource_report()
    print("📊 Resource Report:")
    print(f"  Apple Silicon: {report['system_info']['is_apple_silicon']}")
    print(f"  GPU allocations: {report['allocation_statistics']['gpu_usage_percent']:.1f}%")
    print(f"  Memory available: {report['current_resources']['cpu']['memory_available_mb']:.0f}MB")

    if report['current_resources']['gpu']:
        print(f"  GPU utilization: {report['current_resources']['gpu']['utilization_percent']:.1f}%")

    print("\n💡 Recommendations:")
    for rec in report['optimization_recommendations']:
        print(f"  {rec}")

    # Cleanup
    rm.stop_monitoring()
