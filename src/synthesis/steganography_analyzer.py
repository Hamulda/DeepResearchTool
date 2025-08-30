#!/usr/bin/env python3
"""Steganography Analyzer - Detekce dat skrytých v mediálních souborech
Analýza nejméně významného bitu (LSB) a dalších statistických metod

Author: GitHub Copilot
Created: August 28, 2025 - Phase 3 Implementation
"""

import base64
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import io
import logging
from typing import Any

import numpy as np

try:
    from PIL import Image, ImageStat

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from scipy import stats
    from scipy.fft import fft2, ifft2

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SteganographyResult:
    """Výsledek steganografické analýzy"""

    file_path: str
    file_type: str
    analysis_methods: list[str]
    suspicion_score: float
    detected_anomalies: list[str]
    statistical_metrics: dict[str, float]
    lsb_analysis: dict[str, Any]
    frequency_analysis: dict[str, Any]
    metadata_analysis: dict[str, Any]
    extraction_attempts: list[dict[str, Any]]
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class LSBAnalysisResult:
    """Výsledek LSB analýzy"""

    bit_plane: int
    entropy: float
    randomness_score: float
    pattern_regularity: float
    chi_square_p_value: float
    suspicious_regions: list[tuple[int, int, int, int]]  # x, y, width, height
    extracted_data: bytes | None = None


@dataclass
class FrequencyAnalysisResult:
    """Výsledek frekvenční analýzy"""

    dct_anomalies: list[dict[str, Any]]
    fourier_anomalies: list[dict[str, Any]]
    spectral_density_score: float
    periodicity_detected: bool
    dominant_frequencies: list[float]


class SteganographyAnalyzer:
    """Pokročilý analyzér steganografie pro zpravodajské účely"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.stego_config = config.get("steganography", {})

        # Threshold hodnoty pro detekci
        self.suspicion_threshold = self.stego_config.get("suspicion_threshold", 0.7)
        self.entropy_threshold = self.stego_config.get("entropy_threshold", 7.8)
        self.chi_square_threshold = self.stego_config.get("chi_square_threshold", 0.05)

        # Podporované formáty
        self.supported_image_formats = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"}
        self.supported_audio_formats = {".wav", ".mp3", ".flac", ".ogg"}

        # Statistiky analýzy
        self.analysis_stats = defaultdict(int)
        self.detection_cache: dict[str, SteganographyResult] = {}

        # Kontrola závislostí
        if not PIL_AVAILABLE:
            logger.warning("PIL/Pillow not available - image analysis limited")
        if not SCIPY_AVAILABLE:
            logger.warning("SciPy not available - frequency analysis limited")

    async def analyze_media_file(self, file_path: str, file_data: bytes) -> SteganographyResult:
        """Hlavní metoda pro analýzu mediálního souboru"""
        start_time = datetime.now()

        # Cache kontrola
        file_hash = hashlib.sha256(file_data).hexdigest()
        if file_hash in self.detection_cache:
            logger.info(f"Returning cached result for {file_path}")
            return self.detection_cache[file_hash]

        # Identifikace typu souboru
        file_type = self._identify_file_type(file_path, file_data)

        # Inicializace výsledku
        result = SteganographyResult(
            file_path=file_path,
            file_type=file_type,
            analysis_methods=[],
            suspicion_score=0.0,
            detected_anomalies=[],
            statistical_metrics={},
            lsb_analysis={},
            frequency_analysis={},
            metadata_analysis={},
            extraction_attempts=[],
            confidence=0.0,
        )

        try:
            # Analýza podle typu souboru
            if file_type in ["image"]:
                await self._analyze_image(file_data, result)
            elif file_type in ["audio"]:
                await self._analyze_audio(file_data, result)
            else:
                logger.warning(f"Unsupported file type: {file_type}")
                result.detected_anomalies.append(f"Unsupported file type: {file_type}")

            # Výpočet celkového skóre podezření
            result.suspicion_score = await self._calculate_suspicion_score(result)
            result.confidence = await self._calculate_confidence(result)

            # Cache výsledek
            self.detection_cache[file_hash] = result

            # Statistiky
            analysis_time = (datetime.now() - start_time).total_seconds()
            self.analysis_stats[f"analysis_time_{file_type}"] = analysis_time
            self.analysis_stats[f"total_analyses_{file_type}"] += 1

            logger.info(f"Steganography analysis completed for {file_path} in {analysis_time:.2f}s")

        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            result.detected_anomalies.append(f"Analysis error: {e!s}")

        return result

    def _identify_file_type(self, file_path: str, file_data: bytes) -> str:
        """Identifikace typu souboru podle přípony a magic bytes"""
        file_path_lower = file_path.lower()

        # Kontrola podle přípony
        for ext in self.supported_image_formats:
            if file_path_lower.endswith(ext):
                return "image"

        for ext in self.supported_audio_formats:
            if file_path_lower.endswith(ext):
                return "audio"

        # Magic bytes kontrola
        if file_data.startswith(b"\x89PNG") or file_data.startswith(b"\xff\xd8\xff"):
            return "image"
        if file_data.startswith(b"RIFF") and b"WAVE" in file_data[:12]:
            return "audio"

        return "unknown"

    async def _analyze_image(self, image_data: bytes, result: SteganographyResult):
        """Komplexní analýza obrazových souborů"""
        if not PIL_AVAILABLE:
            result.detected_anomalies.append("PIL not available for image analysis")
            return

        try:
            # Načtení obrázku
            image = Image.open(io.BytesIO(image_data))
            result.analysis_methods.append("image_analysis")

            # Základní metadata analýza
            await self._analyze_image_metadata(image, result)

            # LSB analýza všech kanálů
            await self._perform_lsb_analysis(image, result)

            # Statistická analýza
            await self._perform_statistical_analysis(image, result)

            # Frekvenční analýza (pokud je dostupná SciPy)
            if SCIPY_AVAILABLE:
                await self._perform_frequency_analysis(image, result)

            # Detekce známých steganografických nástrojů
            await self._detect_known_stego_tools(image, result)

            # Pokus o extrakci dat
            await self._attempt_data_extraction(image, result)

        except Exception as e:
            logger.error(f"Error in image analysis: {e}")
            result.detected_anomalies.append(f"Image analysis error: {e!s}")

    async def _analyze_image_metadata(self, image: Image.Image, result: SteganographyResult):
        """Analýza metadat obrázku"""
        metadata_analysis = {
            "format": image.format,
            "mode": image.mode,
            "size": image.size,
            "has_exif": hasattr(image, "_getexif") and image._getexif() is not None,
            "suspicious_metadata": [],
        }

        # EXIF analýza
        if hasattr(image, "_getexif") and image._getexif():
            exif_data = image._getexif()
            if exif_data:
                # Kontrola podezřelých EXIF polí
                suspicious_fields = [
                    "UserComment",
                    "ImageDescription",
                    "Software",
                    "Artist",
                    "Copyright",
                    "XPComment",
                ]

                for field in suspicious_fields:
                    if field in exif_data:
                        value = exif_data[field]
                        if isinstance(value, (bytes, str)) and len(str(value)) > 100:
                            metadata_analysis["suspicious_metadata"].append(
                                f"Large {field}: {len(str(value))} characters"
                            )

        # Kontrola neobvyklých parametrů
        if image.mode not in ["RGB", "RGBA", "L", "P"]:
            metadata_analysis["suspicious_metadata"].append(f"Unusual color mode: {image.mode}")

        result.metadata_analysis = metadata_analysis
        result.analysis_methods.append("metadata_analysis")

    async def _perform_lsb_analysis(self, image: Image.Image, result: SteganographyResult):
        """Analýza nejméně významných bitů (LSB)"""
        if image.mode not in ["RGB", "RGBA", "L"]:
            result.detected_anomalies.append("Image mode not suitable for LSB analysis")
            return

        # Konverze na numpy array
        img_array = np.array(image)

        lsb_results = {}

        # Analýza pro každý kanál/bit plane
        if len(img_array.shape) == 3:  # Barevný obrázek
            channels = ["R", "G", "B"]
            if img_array.shape[2] == 4:
                channels.append("A")
        else:  # Grayscale
            channels = ["Gray"]

        for i, channel in enumerate(channels):
            if len(img_array.shape) == 3:
                channel_data = img_array[:, :, i]
            else:
                channel_data = img_array

            # LSB extrakce
            lsb_bits = channel_data & 1

            # Analýza LSB vrstvy
            lsb_analysis = await self._analyze_lsb_layer(lsb_bits, channel)
            lsb_results[channel] = lsb_analysis

            # Detekce podezřelých regionů
            if lsb_analysis.entropy > self.entropy_threshold:
                result.detected_anomalies.append(
                    f"High entropy in {channel} channel LSB: {lsb_analysis.entropy:.3f}"
                )

            if lsb_analysis.chi_square_p_value < self.chi_square_threshold:
                result.detected_anomalies.append(
                    f"Chi-square test failed for {channel} channel: p={lsb_analysis.chi_square_p_value:.6f}"
                )

        result.lsb_analysis = lsb_results
        result.analysis_methods.append("lsb_analysis")

    async def _analyze_lsb_layer(self, lsb_bits: np.ndarray, channel: str) -> LSBAnalysisResult:
        """Detailní analýza LSB vrstvy"""
        flat_bits = lsb_bits.flatten()

        # Výpočet entropie
        unique, counts = np.unique(flat_bits, return_counts=True)
        probabilities = counts / len(flat_bits)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

        # Test náhodnosti
        runs = self._count_runs(flat_bits)
        expected_runs = (2 * np.sum(flat_bits) * (len(flat_bits) - np.sum(flat_bits))) / len(
            flat_bits
        ) + 1
        randomness_score = abs(runs - expected_runs) / expected_runs if expected_runs > 0 else 1.0

        # Chi-square test
        observed_0 = np.sum(flat_bits == 0)
        observed_1 = np.sum(flat_bits == 1)
        expected = len(flat_bits) / 2

        if expected > 0:
            chi_square = ((observed_0 - expected) ** 2 + (observed_1 - expected) ** 2) / expected
            chi_square_p_value = 1 - stats.chi2.cdf(chi_square, 1) if SCIPY_AVAILABLE else 0.5
        else:
            chi_square_p_value = 1.0

        # Detekce pravidelných vzorů
        pattern_regularity = self._detect_patterns(lsb_bits)

        # Detekce podezřelých regionů
        suspicious_regions = self._find_suspicious_regions(lsb_bits)

        return LSBAnalysisResult(
            bit_plane=1,
            entropy=entropy,
            randomness_score=randomness_score,
            pattern_regularity=pattern_regularity,
            chi_square_p_value=chi_square_p_value,
            suspicious_regions=suspicious_regions,
        )

    def _count_runs(self, data: np.ndarray) -> int:
        """Počítání běhů pro test náhodnosti"""
        if len(data) <= 1:
            return 0

        runs = 1
        for i in range(1, len(data)):
            if data[i] != data[i - 1]:
                runs += 1

        return runs

    def _detect_patterns(self, lsb_bits: np.ndarray) -> float:
        """Detekce pravidelných vzorů v LSB"""
        height, width = lsb_bits.shape

        # Horizontální vzory
        h_pattern_score = 0
        for row in range(height):
            row_data = lsb_bits[row, :]
            # Autokorelace pro detekci periodicity
            if len(row_data) > 10:
                autocorr = np.correlate(row_data, row_data, mode="full")
                autocorr = autocorr[autocorr.size // 2 :]
                if len(autocorr) > 1:
                    h_pattern_score += np.max(autocorr[1:]) / autocorr[0]

        # Vertikální vzory
        v_pattern_score = 0
        for col in range(width):
            col_data = lsb_bits[:, col]
            if len(col_data) > 10:
                autocorr = np.correlate(col_data, col_data, mode="full")
                autocorr = autocorr[autocorr.size // 2 :]
                if len(autocorr) > 1:
                    v_pattern_score += np.max(autocorr[1:]) / autocorr[0]

        total_score = (h_pattern_score + v_pattern_score) / (height + width)
        return min(1.0, total_score)

    def _find_suspicious_regions(self, lsb_bits: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Nalezení podezřelých regionů v LSB"""
        height, width = lsb_bits.shape
        regions = []

        # Rozdělení na bloky a analýza entropie
        block_size = 32

        for y in range(0, height - block_size, block_size // 2):
            for x in range(0, width - block_size, block_size // 2):
                block = lsb_bits[y : y + block_size, x : x + block_size]

                # Výpočet entropie bloku
                unique, counts = np.unique(block, return_counts=True)
                probabilities = counts / block.size
                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

                # Pokud je entropie příliš vysoká
                if entropy > 0.95:  # Téměř náhodné
                    regions.append((x, y, block_size, block_size))

        return regions

    async def _perform_statistical_analysis(self, image: Image.Image, result: SteganographyResult):
        """Statistická analýza obrazu"""
        img_array = np.array(image)

        statistics_metrics = {}

        # Základní statistiky
        if len(img_array.shape) == 3:
            for i, channel in enumerate(["R", "G", "B"]):
                channel_data = img_array[:, :, i]
                statistics_metrics[f"{channel}_mean"] = float(np.mean(channel_data))
                statistics_metrics[f"{channel}_std"] = float(np.std(channel_data))
                statistics_metrics[f"{channel}_skew"] = (
                    float(stats.skew(channel_data.flatten())) if SCIPY_AVAILABLE else 0.0
                )
                statistics_metrics[f"{channel}_kurtosis"] = (
                    float(stats.kurtosis(channel_data.flatten())) if SCIPY_AVAILABLE else 0.0
                )
        else:
            statistics_metrics["gray_mean"] = float(np.mean(img_array))
            statistics_metrics["gray_std"] = float(np.std(img_array))

        # Detekce neobvyklých statistických vlastností
        if (
            "R_std" in statistics_metrics
            and "G_std" in statistics_metrics
            and "B_std" in statistics_metrics
        ):
            # Porovnání směrodatných odchylek kanálů
            std_diff = abs(statistics_metrics["R_std"] - statistics_metrics["G_std"])
            if std_diff > 20:
                result.detected_anomalies.append(
                    f"Large standard deviation difference between channels: {std_diff:.2f}"
                )

        result.statistical_metrics = statistics_metrics
        result.analysis_methods.append("statistical_analysis")

    async def _perform_frequency_analysis(self, image: Image.Image, result: SteganographyResult):
        """Frekvenční analýza pomocí DCT a FFT"""
        if not SCIPY_AVAILABLE:
            result.detected_anomalies.append("SciPy not available for frequency analysis")
            return

        img_array = np.array(image)
        if len(img_array.shape) == 3:
            # Konverze na grayscale pro FFT analýzu
            gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = img_array

        frequency_results = {
            "dct_anomalies": [],
            "fourier_anomalies": [],
            "spectral_density_score": 0.0,
            "periodicity_detected": False,
            "dominant_frequencies": [],
        }

        try:
            # FFT analýza
            fft_result = fft2(gray)
            magnitude_spectrum = np.abs(fft_result)

            # Analýza spektrální hustoty
            spectral_density = np.mean(magnitude_spectrum)
            frequency_results["spectral_density_score"] = float(spectral_density)

            # Detekce dominantních frekvencí
            sorted_magnitudes = np.sort(magnitude_spectrum.flatten())[::-1]
            threshold = sorted_magnitudes[int(len(sorted_magnitudes) * 0.001)]  # Top 0.1%

            dominant_freq_count = np.sum(magnitude_spectrum > threshold)
            frequency_results["dominant_frequencies"] = [float(dominant_freq_count)]

            # Detekce periodicity
            if (
                dominant_freq_count > len(sorted_magnitudes) * 0.01
            ):  # Více než 1% dominant frekvencí
                frequency_results["periodicity_detected"] = True
                result.detected_anomalies.append("Unusual frequency periodicity detected")

        except Exception as e:
            logger.error(f"Error in frequency analysis: {e}")
            frequency_results["fourier_anomalies"].append(f"FFT error: {e!s}")

        result.frequency_analysis = frequency_results
        result.analysis_methods.append("frequency_analysis")

    async def _detect_known_stego_tools(self, image: Image.Image, result: SteganographyResult):
        """Detekce známých steganografických nástrojů"""
        img_data = image.tobytes()

        # Signatury známých nástrojů
        tool_signatures = {
            "steghide": [b"steghide", b"STEGHIDE"],
            "lsb_steganography": [b"LSB", b"lsb"],
            "outguess": [b"outguess", b"OUTGUESS"],
            "jsteg": [b"jsteg", b"JSTEG"],
            "f5": [b"F5", b"f5"],
            "openstego": [b"openstego", b"OpenStego"],
        }

        detected_tools = []

        for tool_name, signatures in tool_signatures.items():
            for signature in signatures:
                if signature in img_data:
                    detected_tools.append(tool_name)
                    result.detected_anomalies.append(f"Possible {tool_name} signature detected")
                    break

        if detected_tools:
            result.statistical_metrics["detected_stego_tools"] = detected_tools

    async def _attempt_data_extraction(self, image: Image.Image, result: SteganographyResult):
        """Pokus o extrakci skrytých dat"""
        extraction_attempts = []

        # LSB extrakce
        try:
            extracted_lsb = await self._extract_lsb_data(image)
            if extracted_lsb:
                extraction_attempts.append(
                    {
                        "method": "lsb_extraction",
                        "data_length": len(extracted_lsb),
                        "data_preview": base64.b64encode(extracted_lsb[:100]).decode("ascii"),
                        "success": True,
                    }
                )
        except Exception as e:
            extraction_attempts.append(
                {"method": "lsb_extraction", "error": str(e), "success": False}
            )

        # Pokus o detekci různých formátů
        for attempt in extraction_attempts:
            if attempt.get("success") and "data_preview" in attempt:
                try:
                    decoded_data = base64.b64decode(attempt["data_preview"])

                    # Kontrola zda jsou data text
                    if all(32 <= b <= 126 for b in decoded_data[:50]):
                        attempt["likely_text"] = True
                        result.detected_anomalies.append("Extracted data appears to be text")

                    # Kontrola magic bytes
                    if decoded_data.startswith(b"\x89PNG"):
                        attempt["detected_format"] = "PNG image"
                    elif decoded_data.startswith(b"\xff\xd8"):
                        attempt["detected_format"] = "JPEG image"
                    elif decoded_data.startswith(b"PK"):
                        attempt["detected_format"] = "ZIP archive"

                except Exception:
                    pass

        result.extraction_attempts = extraction_attempts

    async def _extract_lsb_data(self, image: Image.Image) -> bytes | None:
        """Extrakce dat z LSB"""
        if image.mode not in ["RGB", "RGBA", "L"]:
            return None

        img_array = np.array(image)

        # Extrakce LSB ze všech kanálů
        if len(img_array.shape) == 3:
            # RGB kanály
            lsb_data = []
            for channel in range(min(3, img_array.shape[2])):
                channel_lsb = img_array[:, :, channel] & 1
                lsb_data.extend(channel_lsb.flatten())
        else:
            # Grayscale
            lsb_data = (img_array & 1).flatten()

        # Konverze bitů na byty
        if len(lsb_data) % 8 != 0:
            lsb_data = lsb_data[: -(len(lsb_data) % 8)]

        bytes_data = []
        for i in range(0, len(lsb_data), 8):
            byte_bits = lsb_data[i : i + 8]
            byte_value = sum(bit * (2 ** (7 - j)) for j, bit in enumerate(byte_bits))
            bytes_data.append(byte_value)

        return bytes(bytes_data)

    async def _analyze_audio(self, audio_data: bytes, result: SteganographyResult):
        """Analýza audio souborů pro steganografii"""
        # Placeholder pro audio analýzu - vyžaduje specifické audio knihovny
        result.detected_anomalies.append("Audio steganography analysis not fully implemented")
        result.analysis_methods.append("basic_audio_analysis")

        # Základní analýza audio hlavičky
        if audio_data.startswith(b"RIFF") and b"WAVE" in audio_data[:12]:
            # WAV formát
            result.metadata_analysis["format"] = "WAV"

            # Kontrola neobvyklé velikosti
            if len(audio_data) > 100 * 1024 * 1024:  # 100MB
                result.detected_anomalies.append("Unusually large audio file")

    async def _calculate_suspicion_score(self, result: SteganographyResult) -> float:
        """Výpočet celkového skóre podezření"""
        score = 0.0

        # Váhy pro různé faktory
        weights = {
            "anomaly_count": 0.3,
            "high_entropy": 0.25,
            "failed_statistical_tests": 0.2,
            "detected_tools": 0.15,
            "extracted_data": 0.1,
        }

        # Počet anomálií
        anomaly_score = min(1.0, len(result.detected_anomalies) / 5.0)
        score += weights["anomaly_count"] * anomaly_score

        # Vysoká entropie v LSB
        if result.lsb_analysis:
            max_entropy = 0
            for channel_result in result.lsb_analysis.values():
                if hasattr(channel_result, "entropy"):
                    max_entropy = max(max_entropy, channel_result.entropy)

            if max_entropy > self.entropy_threshold:
                entropy_score = min(1.0, (max_entropy - 7.0) / 1.0)
                score += weights["high_entropy"] * entropy_score

        # Neúspěšné statistické testy
        failed_tests = sum(
            1
            for anomaly in result.detected_anomalies
            if "chi-square" in anomaly.lower() or "test failed" in anomaly.lower()
        )
        if failed_tests > 0:
            score += weights["failed_statistical_tests"] * min(1.0, failed_tests / 3.0)

        # Detekované nástroje
        if "detected_stego_tools" in result.statistical_metrics:
            tool_count = len(result.statistical_metrics["detected_stego_tools"])
            score += weights["detected_tools"] * min(1.0, tool_count / 2.0)

        # Extrahovaná data
        successful_extractions = sum(
            1 for attempt in result.extraction_attempts if attempt.get("success", False)
        )
        if successful_extractions > 0:
            score += weights["extracted_data"] * min(1.0, successful_extractions / 2.0)

        return min(1.0, score)

    async def _calculate_confidence(self, result: SteganographyResult) -> float:
        """Výpočet konfidence v analýzu"""
        base_confidence = 0.7

        # Boost za dokončené analýzy
        method_boost = len(result.analysis_methods) * 0.05

        # Penalty za chyby
        error_penalty = (
            sum(0.1 for anomaly in result.detected_anomalies if "error" in anomaly.lower()) * 0.1
        )

        confidence = base_confidence + method_boost - error_penalty
        return max(0.1, min(1.0, confidence))

    def get_analysis_statistics(self) -> dict[str, Any]:
        """Získání statistik analýzy"""
        return {
            "total_analyses": sum(
                v for k, v in self.analysis_stats.items() if k.startswith("total_analyses")
            ),
            "cache_size": len(self.detection_cache),
            "analysis_stats": dict(self.analysis_stats),
            "supported_formats": {
                "image": list(self.supported_image_formats),
                "audio": list(self.supported_audio_formats),
            },
            "dependencies": {"PIL_available": PIL_AVAILABLE, "scipy_available": SCIPY_AVAILABLE},
        }

    async def generate_steganography_report(
        self, results: list[SteganographyResult]
    ) -> dict[str, Any]:
        """Generování komplexní zprávy o steganografické analýze"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_files_analyzed": len(results),
            "suspicious_files": [],
            "analysis_summary": defaultdict(int),
            "common_anomalies": defaultdict(int),
            "extraction_summary": defaultdict(int),
            "confidence_distribution": defaultdict(int),
        }

        for result in results:
            # Kategorizace podle skóre podezření
            if result.suspicion_score > self.suspicion_threshold:
                report["suspicious_files"].append(
                    {
                        "file_path": result.file_path,
                        "suspicion_score": result.suspicion_score,
                        "anomaly_count": len(result.detected_anomalies),
                        "confidence": result.confidence,
                    }
                )

            # Statistiky metod analýzy
            for method in result.analysis_methods:
                report["analysis_summary"][method] += 1

            # Časté anomálie
            for anomaly in result.detected_anomalies:
                # Kategorizace anomálií
                if "entropy" in anomaly.lower():
                    report["common_anomalies"]["high_entropy"] += 1
                elif "chi-square" in anomaly.lower():
                    report["common_anomalies"]["statistical_test_failure"] += 1
                elif "tool" in anomaly.lower():
                    report["common_anomalies"]["stego_tool_detected"] += 1
                elif "frequency" in anomaly.lower():
                    report["common_anomalies"]["frequency_anomaly"] += 1

            # Extrakce statistiky
            for attempt in result.extraction_attempts:
                if attempt.get("success"):
                    report["extraction_summary"]["successful_extractions"] += 1
                else:
                    report["extraction_summary"]["failed_extractions"] += 1

            # Confidence distribuce
            confidence_bin = (
                f"{int(result.confidence * 10) * 10}%-{int(result.confidence * 10) * 10 + 10}%"
            )
            report["confidence_distribution"][confidence_bin] += 1

        return report
