"""
Metal Acceleration pro DeepResearchTool
Optimalizace výpočtů na M1 architektuře pomocí Apple MLX a Metal Performance Shaders.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class MetalAcceleration:
    """
    Metal akcelerace pro M1 Mac s využitím Apple MLX framework
    pro rychlejší steganalýzu a zpracování obrazu/textu.
    """

    def __init__(self):
        self.mlx_available = False
        self.metal_available = False
        self._initialize_acceleration()

    def _initialize_acceleration(self):
        """Inicializace Metal/MLX akcelerace"""
        # Pokus o MLX import
        try:
            import mlx.core as mx
            import mlx.nn as nn
            self.mx = mx
            self.nn = nn
            self.mlx_available = True
            logger.info("Apple MLX framework dostupný")
        except ImportError:
            logger.info("Apple MLX framework nedostupný")

        # Pokus o Metal Performance Shaders
        try:
            import Metal
            self.metal_available = True
            logger.info("Metal framework dostupný")
        except ImportError:
            logger.info("Metal framework nedostupný")

    async def accelerated_image_analysis(self, image_data: np.ndarray) -> Dict[str, Any]:
        """Akcelerovaná analýza obrazu na M1 GPU"""
        if not self.mlx_available:
            return await self._fallback_cpu_analysis(image_data)

        try:
            # Převod na MLX array
            mx_image = self.mx.array(image_data)

            # GPU akcelerované operace
            results = {}

            # Rychlý výpočet entropy pomocí MLX
            results["entropy"] = await self._mlx_calculate_entropy(mx_image)

            # Akcelerovaná FFT analýza
            results["frequency_analysis"] = await self._mlx_frequency_analysis(mx_image)

            # Gradient analýza
            results["gradient_analysis"] = await self._mlx_gradient_analysis(mx_image)

            return results

        except Exception as e:
            logger.warning(f"MLX akcelerace selhala, přechod na CPU: {e}")
            return await self._fallback_cpu_analysis(image_data)

    async def _mlx_calculate_entropy(self, mx_image) -> float:
        """Výpočet entropy pomocí MLX"""
        # Flatten obrázek
        flat_image = self.mx.reshape(mx_image, [-1])

        # Histogram
        hist = self.mx.histogram(flat_image, bins=256)

        # Normalizace
        hist_norm = hist / self.mx.sum(hist)

        # Entropy výpočet
        log_hist = self.mx.log2(hist_norm + 1e-10)
        entropy = -self.mx.sum(hist_norm * log_hist)

        return float(entropy)

    async def _mlx_frequency_analysis(self, mx_image) -> Dict[str, Any]:
        """FFT analýza pomocí MLX"""
        try:
            # 2D FFT
            fft_result = self.mx.fft.fft2(mx_image.astype(self.mx.complex64))

            # Magnitude spektrum
            magnitude = self.mx.abs(fft_result)

            # Frekvenční statistiky
            max_freq = float(self.mx.max(magnitude))
            mean_freq = float(self.mx.mean(magnitude))

            return {
                "max_frequency_magnitude": max_freq,
                "mean_frequency_magnitude": mean_freq,
                "frequency_variance": float(self.mx.var(magnitude))
            }

        except Exception as e:
            logger.warning(f"MLX FFT analýza selhala: {e}")
            return {"error": str(e)}

    async def _mlx_gradient_analysis(self, mx_image) -> Dict[str, Any]:
        """Gradient analýza pomocí MLX"""
        try:
            # Sobel operátory
            sobel_x = self.mx.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_y = self.mx.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

            # Konvoluce (zjednodušená implementace)
            grad_x = self._mlx_simple_conv2d(mx_image, sobel_x)
            grad_y = self._mlx_simple_conv2d(mx_image, sobel_y)

            # Gradient magnitude
            grad_magnitude = self.mx.sqrt(grad_x**2 + grad_y**2)

            return {
                "gradient_mean": float(self.mx.mean(grad_magnitude)),
                "gradient_max": float(self.mx.max(grad_magnitude)),
                "gradient_std": float(self.mx.sqrt(self.mx.var(grad_magnitude)))
            }

        except Exception as e:
            logger.warning(f"MLX gradient analýza selhala: {e}")
            return {"error": str(e)}

    def _mlx_simple_conv2d(self, image, kernel):
        """Zjednodušená 2D konvoluce pro MLX"""
        # Toto je zjednodušená implementace - v produkci by se použily MLX nn.Conv2d
        # Pro demonstrační účely
        try:
            # Padding a konvoluce (velmi zjednodušená)
            h, w = image.shape[-2:]
            kh, kw = kernel.shape

            # Zde by byla plná implementace konvoluce
            # Pro nyní vrátíme placeholder
            return self.mx.zeros_like(image)

        except:
            return self.mx.zeros_like(image)

    async def accelerated_text_processing(self, text_embeddings: np.ndarray) -> Dict[str, Any]:
        """Akcelerované zpracování textu na M1"""
        if not self.mlx_available:
            return await self._fallback_text_processing(text_embeddings)

        try:
            mx_embeddings = self.mx.array(text_embeddings)

            results = {}

            # Rychlé similarity výpočty
            results["similarity_matrix"] = await self._mlx_cosine_similarity(mx_embeddings)

            # Clustering features
            results["clustering_features"] = await self._mlx_clustering_features(mx_embeddings)

            return results

        except Exception as e:
            logger.warning(f"MLX text processing selhalo: {e}")
            return await self._fallback_text_processing(text_embeddings)

    async def _mlx_cosine_similarity(self, embeddings) -> np.ndarray:
        """Cosine similarity pomocí MLX"""
        try:
            # Normalizace
            norms = self.mx.sqrt(self.mx.sum(embeddings**2, axis=1, keepdims=True))
            normalized = embeddings / (norms + 1e-10)

            # Dot product pro similarity
            similarity = self.mx.matmul(normalized, normalized.T)

            return np.array(similarity)

        except Exception as e:
            logger.error(f"MLX cosine similarity chyba: {e}")
            return np.eye(embeddings.shape[0])

    async def _mlx_clustering_features(self, embeddings) -> Dict[str, Any]:
        """Clustering features pomocí MLX"""
        try:
            # Centroid
            centroid = self.mx.mean(embeddings, axis=0)

            # Distances from centroid
            distances = self.mx.sqrt(self.mx.sum((embeddings - centroid)**2, axis=1))

            return {
                "centroid": np.array(centroid),
                "mean_distance": float(self.mx.mean(distances)),
                "max_distance": float(self.mx.max(distances)),
                "std_distance": float(self.mx.sqrt(self.mx.var(distances)))
            }

        except Exception as e:
            logger.error(f"MLX clustering features chyba: {e}")
            return {"error": str(e)}

    async def _fallback_cpu_analysis(self, image_data: np.ndarray) -> Dict[str, Any]:
        """Fallback CPU analýza když MLX není dostupný"""
        results = {}

        # CPU entropy
        flat_image = image_data.flatten()
        hist, _ = np.histogram(flat_image, bins=256)
        hist_norm = hist / np.sum(hist)
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
        results["entropy"] = entropy

        # CPU FFT
        fft_result = np.fft.fft2(image_data)
        magnitude = np.abs(fft_result)
        results["frequency_analysis"] = {
            "max_frequency_magnitude": float(np.max(magnitude)),
            "mean_frequency_magnitude": float(np.mean(magnitude))
        }

        return results

    async def _fallback_text_processing(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Fallback CPU text processing"""
        # CPU cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)
        similarity = np.dot(normalized, normalized.T)

        return {
            "similarity_matrix": similarity,
            "clustering_features": {
                "centroid": np.mean(embeddings, axis=0),
                "mean_distance": float(np.mean(np.linalg.norm(embeddings - np.mean(embeddings, axis=0), axis=1)))
            }
        }

    def get_acceleration_info(self) -> Dict[str, Any]:
        """Informace o dostupné akceleraci"""
        return {
            "mlx_available": self.mlx_available,
            "metal_available": self.metal_available,
            "acceleration_type": "MLX" if self.mlx_available else "CPU",
            "recommended_batch_size": 64 if self.mlx_available else 16
        }

    async def benchmark_acceleration(self) -> Dict[str, Any]:
        """Benchmark Metal akcelerace vs CPU"""
        benchmark_results = {}

        # Test data
        test_image = np.random.rand(512, 512).astype(np.float32)
        test_embeddings = np.random.rand(100, 768).astype(np.float32)

        # Image processing benchmark
        if self.mlx_available:
            start_time = time.time()
            mlx_result = await self.accelerated_image_analysis(test_image)
            mlx_time = time.time() - start_time

            start_time = time.time()
            cpu_result = await self._fallback_cpu_analysis(test_image)
            cpu_time = time.time() - start_time

            benchmark_results["image_processing"] = {
                "mlx_time_seconds": mlx_time,
                "cpu_time_seconds": cpu_time,
                "speedup": cpu_time / mlx_time if mlx_time > 0 else 1.0
            }

        # Text processing benchmark
        if self.mlx_available:
            start_time = time.time()
            mlx_text_result = await self.accelerated_text_processing(test_embeddings)
            mlx_text_time = time.time() - start_time

            start_time = time.time()
            cpu_text_result = await self._fallback_text_processing(test_embeddings)
            cpu_text_time = time.time() - start_time

            benchmark_results["text_processing"] = {
                "mlx_time_seconds": mlx_text_time,
                "cpu_time_seconds": cpu_text_time,
                "speedup": cpu_text_time / mlx_text_time if mlx_text_time > 0 else 1.0
            }

        return benchmark_results


# Singleton instance
_metal_acceleration: Optional[MetalAcceleration] = None


def get_metal_acceleration() -> MetalAcceleration:
    """Factory function pro MetalAcceleration singleton"""
    global _metal_acceleration

    if _metal_acceleration is None:
        _metal_acceleration = MetalAcceleration()

    return _metal_acceleration


# Convenience funkce
async def accelerated_image_analysis(image_data: np.ndarray) -> Dict[str, Any]:
    """Convenience funkce pro akcelerovanou analýzu obrazu"""
    metal = get_metal_acceleration()
    return await metal.accelerated_image_analysis(image_data)


async def accelerated_text_processing(embeddings: np.ndarray) -> Dict[str, Any]:
    """Convenience funkce pro akcelerované zpracování textu"""
    metal = get_metal_acceleration()
    return await metal.accelerated_text_processing(embeddings)
