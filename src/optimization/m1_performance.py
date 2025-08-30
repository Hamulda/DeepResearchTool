"""M1 Performance Optimizer
Optimalizace pro Apple Silicon s Metal Performance Shaders (MPS)

Author: Senior Python/MLOps Agent
"""

from dataclasses import dataclass
import gc
import logging
import platform
from typing import Any

import psutil
import torch

logger = logging.getLogger(__name__)


@dataclass
class M1OptimizationConfig:
    """Konfigurace optimalizace pro M1"""

    max_memory_gb: float = 6.0  # Limit pro 8GB M1
    batch_size: int = 8
    n_threads: int = 8
    use_mps: bool = True
    enable_memory_efficient_attention: bool = True
    gradient_checkpointing: bool = True


class M1PerformanceOptimizer:
    """Optimalizátor pro Apple Silicon M1/M2"""

    def __init__(self, config: M1OptimizationConfig = None):
        self.config = config or M1OptimizationConfig()
        self.device = self._detect_optimal_device()
        self.memory_info = self._get_memory_info()

    def _detect_optimal_device(self) -> str:
        """Detekuje optimální zařízení pro inference"""
        if torch.backends.mps.is_available() and self.config.use_mps:
            logger.info("🚀 Using Apple Metal Performance Shaders (MPS)")
            return "mps"
        if torch.cuda.is_available():
            logger.info("🚀 Using CUDA")
            return "cuda"
        logger.info("🚀 Using CPU")
        return "cpu"

    def _get_memory_info(self) -> dict[str, Any]:
        """Získá informace o dostupné paměti"""
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent": memory.percent
        }

    def get_llama_cpp_config(self) -> dict[str, Any]:
        """Optimalizovaná konfigurace pro llama-cpp-python na M1"""
        return {
            "n_gpu_layers": -1,  # Všechny vrstvy na GPU/Metal
            "n_threads": self.config.n_threads,
            "n_batch": self.config.batch_size,
            "use_mlock": True,  # Zamknutí paměti pro stabilitu
            "use_mmap": True,   # Memory mapping pro efektivitu
            "verbose": False,
            # Metal-specific optimizations
            "n_ctx": min(4096, int(self.memory_info["available_gb"] * 500)),
        }

    def optimize_sentence_transformer(self, model) -> None:
        """Optimalizuje sentence-transformers model pro MPS"""
        try:
            if self.device == "mps":
                # Explicitně přesuň model na MPS
                model = model.to(self.device)

                # Nastav precision pro MPS
                if hasattr(model, 'half'):
                    model = model.half()  # Použij FP16 pro úsporu paměti

                logger.info("✅ SentenceTransformer optimalizován pro MPS")

            return model
        except Exception as e:
            logger.warning(f"⚠️ MPS optimalizace selhala: {e}, fallback na CPU")
            return model.to("cpu")

    def get_intelligent_batch_size(self, model_size_mb: float = 500) -> int:
        """Inteligentní výpočet batch size na základě dostupné RAM"""
        available_gb = self.memory_info["available_gb"]

        # Ponech 2GB pro systém
        usable_gb = max(available_gb - 2.0, 1.0)

        # Odhad batch size na základě modelu a paměti
        model_memory_gb = model_size_mb / 1024
        estimated_batch_size = int(usable_gb / (model_memory_gb * 2))

        # Omez na rozumné rozmezí
        batch_size = max(1, min(estimated_batch_size, 32))

        logger.info(f"🧠 Intelligent batch size: {batch_size} (dostupná RAM: {available_gb:.1f}GB)")
        return batch_size

    def memory_cleanup(self) -> None:
        """Vyčištění paměti"""
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

        gc.collect()
        logger.info("🧹 Memory cleanup completed")

    def log_system_info(self) -> None:
        """Loguje informace o systému"""
        logger.info("=== M1 System Information ===")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Processor: {platform.processor()}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Total RAM: {self.memory_info['total_gb']:.1f}GB")
        logger.info(f"Available RAM: {self.memory_info['available_gb']:.1f}GB")
        logger.info(f"Memory usage: {self.memory_info['percent']:.1f}%")

        if torch.backends.mps.is_available():
            logger.info("✅ MPS Available")
        else:
            logger.info("❌ MPS Not Available")


# Globální instance optimizátoru
m1_optimizer = M1PerformanceOptimizer()


def log_system_info():
    """Convenience funkce pro logování systému"""
    m1_optimizer.log_system_info()


def get_optimal_device() -> str:
    """Získá optimální zařízení"""
    return m1_optimizer.device


def optimize_for_m1(model, model_type: str = "sentence_transformer"):
    """Optimalizuje model pro M1"""
    if model_type == "sentence_transformer":
        return m1_optimizer.optimize_sentence_transformer(model)
    return model.to(m1_optimizer.device)


def cleanup_memory():
    """Vyčistí paměť"""
    m1_optimizer.memory_cleanup()
