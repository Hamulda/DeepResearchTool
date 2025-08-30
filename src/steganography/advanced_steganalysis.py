"""
Advanced Steganalysis Engine pro DeepResearchTool
Pokročilá steganalýza pro detekci skrytého obsahu v obrazech a audio souborech.
"""

import asyncio
import io
import logging
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import librosa
import numpy as np
from PIL import Image
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram

from ..optimization.intelligent_memory import cache_get, cache_set

logger = logging.getLogger(__name__)


@dataclass
class SteganalysisResult:
    """Výsledek steganalytické analýzy"""
    file_path: str
    file_type: str
    steganography_detected: bool
    confidence_score: float
    detection_methods: List[str] = field(default_factory=list)
    hidden_data_location: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    extracted_data: Optional[bytes] = None
    analysis_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LSBAnalysisResult:
    """Výsledek LSB analýzy"""
    chi_square_score: float
    p_value: float
    suspicious_regions: List[Tuple[int, int, int, int]]  # x, y, width, height
    bit_plane_analysis: Dict[int, float]
    entropy_analysis: Dict[str, float]


@dataclass
class AudioSteganalysisResult:
    """Výsledek audio steganalýzy"""
    echo_detection: Dict[str, Any]
    spectral_anomalies: List[Dict[str, Any]]
    phase_anomalies: Dict[str, Any]
    amplitude_irregularities: List[Dict[str, Any]]
    frequency_analysis: Dict[str, Any]


class AdvancedSteganalysisEngine:
    """
    Pokročilý engine pro steganalýzu s podporou GPU akcelerace na M1.
    Detekuje skrytý obsah v obrazech a audio souborech.
    """

    def __init__(self,
                 enable_gpu_acceleration: bool = True,
                 max_file_size_mb: int = 100):
        self.enable_gpu_acceleration = enable_gpu_acceleration
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

        # Inicializace GPU podpory pro M1
        self._gpu_available = False
        if enable_gpu_acceleration:
            self._initialize_gpu_support()

        logger.info(f"AdvancedSteganalysisEngine inicializován (GPU: {self._gpu_available})")

    def _initialize_gpu_support(self):
        """Inicializace GPU podpory pro M1 Mac"""
        try:
            # Pokus o import Apple MLX pro M1 akceleraci
            import mlx.core as mx
            self._gpu_available = True
            logger.info("MLX GPU akcelerace aktivována pro M1")
        except ImportError:
            try:
                # Fallback na OpenCV CUDA pokud je dostupná
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    self._gpu_available = True
                    logger.info("OpenCV CUDA akcelerace aktivována")
            except:
                logger.info("GPU akcelerace nedostupná, používá se CPU")

    async def analyze_file(self, file_path: str) -> SteganalysisResult:
        """
        Hlavní metoda pro steganalytickou analýzu souboru
        """
        cache_key = f"steganalysis:{file_path}"
        cached_result = await cache_get(cache_key)

        if cached_result:
            return SteganalysisResult(**cached_result)

        # Kontrola velikosti souboru
        try:
            import os
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size_bytes:
                return SteganalysisResult(
                    file_path=file_path,
                    file_type="unknown",
                    steganography_detected=False,
                    confidence_score=0.0,
                    metadata={"error": "Soubor příliš velký"}
                )
        except Exception as e:
            return SteganalysisResult(
                file_path=file_path,
                file_type="unknown",
                steganography_detected=False,
                confidence_score=0.0,
                metadata={"error": str(e)}
            )

        # Detekce typu souboru
        file_type = self._detect_file_type(file_path)

        result = SteganalysisResult(
            file_path=file_path,
            file_type=file_type,
            steganography_detected=False,
            confidence_score=0.0
        )

        try:
            if file_type in ["jpg", "jpeg", "png", "bmp", "tiff"]:
                result = await self._analyze_image(file_path, result)
            elif file_type in ["wav", "mp3", "flac", "ogg"]:
                result = await self._analyze_audio(file_path, result)
            else:
                result.metadata["error"] = f"Nepodporovaný typ souboru: {file_type}"

        except Exception as e:
            result.metadata["analysis_error"] = str(e)
            logger.error(f"Chyba při analýze {file_path}: {e}")

        # Cache výsledku
        result_dict = {
            "file_path": result.file_path,
            "file_type": result.file_type,
            "steganography_detected": result.steganography_detected,
            "confidence_score": result.confidence_score,
            "detection_methods": result.detection_methods,
            "hidden_data_location": result.hidden_data_location,
            "metadata": result.metadata,
            "analysis_details": result.analysis_details
        }
        await cache_set(cache_key, result_dict, ttl=3600)

        return result

    def _detect_file_type(self, file_path: str) -> str:
        """Detekce typu souboru podle přípony a magic bytes"""
        import os
        _, ext = os.path.splitext(file_path.lower())

        if ext:
            return ext[1:]  # Odstranění tečky

        # Fallback: detekce podle magic bytes
        try:
            with open(file_path, 'rb') as f:
                header = f.read(16)

            if header.startswith(b'\xff\xd8\xff'):
                return "jpg"
            elif header.startswith(b'\x89PNG'):
                return "png"
            elif header.startswith(b'RIFF') and b'WAVE' in header:
                return "wav"
            elif header.startswith(b'ID3') or header.startswith(b'\xff\xfb'):
                return "mp3"

        except Exception:
            pass

        return "unknown"

    async def _analyze_image(self, file_path: str, result: SteganalysisResult) -> SteganalysisResult:
        """Analýza obrazu pro steganografii"""

        # Načtení obrazu
        try:
            image = cv2.imread(file_path)
            if image is None:
                # Fallback na PIL
                pil_image = Image.open(file_path)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        except Exception as e:
            result.metadata["load_error"] = str(e)
            return result

        # Detekce LSB steganografie
        lsb_result = await self.detect_lsb_steganography(image)

        # Analýza bit planes
        bit_plane_analysis = self._analyze_bit_planes(image)

        # Entropy analýza
        entropy_analysis = self._calculate_entropy_analysis(image)

        # Chi-square test
        chi_square_results = self._chi_square_test(image)

        # Kombinované skóre
        confidence_scores = []
        detection_methods = []

        if lsb_result.p_value < 0.05:  # Statisticky signifikantní
            confidence_scores.append(0.8)
            detection_methods.append("lsb_chi_square")

        if any(score > 0.7 for score in bit_plane_analysis.values()):
            confidence_scores.append(0.6)
            detection_methods.append("bit_plane_analysis")

        if entropy_analysis["suspicious_entropy_ratio"] > 0.3:
            confidence_scores.append(0.5)
            detection_methods.append("entropy_analysis")

        # Finální hodnocení
        if confidence_scores:
            result.steganography_detected = True
            result.confidence_score = max(confidence_scores)
            result.detection_methods = detection_methods

        result.analysis_details = {
            "lsb_analysis": {
                "chi_square_score": lsb_result.chi_square_score,
                "p_value": lsb_result.p_value,
                "suspicious_regions": len(lsb_result.suspicious_regions)
            },
            "bit_plane_analysis": bit_plane_analysis,
            "entropy_analysis": entropy_analysis,
            "image_properties": {
                "width": image.shape[1],
                "height": image.shape[0],
                "channels": image.shape[2] if len(image.shape) > 2 else 1
            }
        }

        return result

    async def detect_lsb_steganography(self, image: np.ndarray) -> LSBAnalysisResult:
        """
        Detekce LSB steganografie pomocí Chi-square testu
        """
        height, width = image.shape[:2]

        # Analýza po blocích pro lepší lokalizaci
        block_size = 64
        suspicious_regions = []
        chi_scores = []
        p_values = []

        for y in range(0, height - block_size, block_size):
            for x in range(0, width - block_size, block_size):
                block = image[y:y+block_size, x:x+block_size]

                # Chi-square test pro tento blok
                chi_score, p_value = self._chi_square_test_block(block)
                chi_scores.append(chi_score)
                p_values.append(p_value)

                # Pokud je p-value malé, blok je podezřelý
                if p_value < 0.01:
                    suspicious_regions.append((x, y, block_size, block_size))

        # Analýza bit planes
        bit_plane_analysis = {}
        for bit in range(8):
            bit_plane = self._extract_bit_plane(image, bit)
            bit_plane_analysis[bit] = self._analyze_bit_plane_randomness(bit_plane)

        # Entropy analýza
        entropy_analysis = self._calculate_entropy_analysis(image)

        return LSBAnalysisResult(
            chi_square_score=np.mean(chi_scores) if chi_scores else 0,
            p_value=np.mean(p_values) if p_values else 1,
            suspicious_regions=suspicious_regions,
            bit_plane_analysis=bit_plane_analysis,
            entropy_analysis=entropy_analysis
        )

    def _chi_square_test_block(self, block: np.ndarray) -> Tuple[float, float]:
        """Chi-square test pro blok obrazu"""
        if len(block.shape) == 3:
            # Převod na grayscale pro jednoduchost
            block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)

        # Extrakce LSB
        lsb_data = block & 1
        flat_lsb = lsb_data.flatten()

        # Očekávaná frekvence pro náhodná data (50% nul, 50% jedniček)
        total_bits = len(flat_lsb)
        expected_freq = total_bits / 2

        # Pozorovaná frekvence
        observed_zeros = np.sum(flat_lsb == 0)
        observed_ones = np.sum(flat_lsb == 1)

        # Chi-square výpočet
        if expected_freq > 0:
            chi_square = ((observed_zeros - expected_freq) ** 2 / expected_freq +
                         (observed_ones - expected_freq) ** 2 / expected_freq)

            # P-value (approximace pro 1 stupeň volnosti)
            p_value = 1 - stats.chi2.cdf(chi_square, df=1)

            return chi_square, p_value

        return 0.0, 1.0

    def _chi_square_test(self, image: np.ndarray) -> Dict[str, float]:
        """Celkový Chi-square test pro obraz"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Test na celém obrazu
        chi_score, p_value = self._chi_square_test_block(image)

        return {
            "chi_square_score": chi_score,
            "p_value": p_value,
            "is_suspicious": p_value < 0.05
        }

    def _extract_bit_plane(self, image: np.ndarray, bit_position: int) -> np.ndarray:
        """Extrakce konkrétní bit plane z obrazu"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return (image >> bit_position) & 1

    def _analyze_bit_plane_randomness(self, bit_plane: np.ndarray) -> float:
        """Analýza náhodnosti bit plane"""
        flat_plane = bit_plane.flatten()

        # Výpočet entropy
        _, counts = np.unique(flat_plane, return_counts=True)
        probs = counts / len(flat_plane)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        # Normalizace entropy (max = 1 pro ideálně náhodná data)
        max_entropy = 1.0  # log2(2) pro binární data
        normalized_entropy = entropy / max_entropy

        return normalized_entropy

    def _analyze_bit_planes(self, image: np.ndarray) -> Dict[int, float]:
        """Analýza všech bit planes"""
        bit_plane_scores = {}

        for bit in range(8):
            bit_plane = self._extract_bit_plane(image, bit)
            randomness_score = self._analyze_bit_plane_randomness(bit_plane)
            bit_plane_scores[bit] = randomness_score

        return bit_plane_scores

    def _calculate_entropy_analysis(self, image: np.ndarray) -> Dict[str, float]:
        """Pokročilá entropy analýza"""
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # Celková entropy
        hist, _ = np.histogram(gray_image, bins=256, range=(0, 256))
        hist_norm = hist / np.sum(hist)
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))

        # LSB entropy
        lsb_plane = gray_image & 1
        lsb_hist, _ = np.histogram(lsb_plane, bins=2, range=(0, 2))
        lsb_hist_norm = lsb_hist / np.sum(lsb_hist)
        lsb_entropy = -np.sum(lsb_hist_norm * np.log2(lsb_hist_norm + 1e-10))

        # Poměr podezřelé entropy
        suspicious_threshold = 0.9
        suspicious_entropy_ratio = lsb_entropy / max(entropy, 1e-10)

        return {
            "overall_entropy": entropy,
            "lsb_entropy": lsb_entropy,
            "suspicious_entropy_ratio": suspicious_entropy_ratio,
            "is_suspicious": suspicious_entropy_ratio > suspicious_threshold
        }

    async def _analyze_audio(self, file_path: str, result: SteganalysisResult) -> SteganalysisResult:
        """Analýza audio souboru pro steganografii"""

        try:
            # Načtení audio souboru
            y, sr = librosa.load(file_path, sr=None)

        except Exception as e:
            result.metadata["load_error"] = str(e)
            return result

        # Echo detection
        echo_result = self._detect_echo_steganography(y, sr)

        # Spektrální analýza
        spectral_anomalies = self._detect_spectral_anomalies(y, sr)

        # Phase analysis
        phase_anomalies = self._analyze_phase_anomalies(y, sr)

        # Amplitude irregularities
        amplitude_irregularities = self._detect_amplitude_irregularities(y)

        # Kombinované hodnocení
        confidence_scores = []
        detection_methods = []

        if echo_result["echo_detected"]:
            confidence_scores.append(echo_result["confidence"])
            detection_methods.append("echo_detection")

        if len(spectral_anomalies) > 0:
            confidence_scores.append(0.6)
            detection_methods.append("spectral_anomalies")

        if phase_anomalies["suspicious_phase_changes"] > 10:
            confidence_scores.append(0.5)
            detection_methods.append("phase_anomalies")

        if len(amplitude_irregularities) > 5:
            confidence_scores.append(0.4)
            detection_methods.append("amplitude_irregularities")

        # Finální hodnocení
        if confidence_scores:
            result.steganography_detected = True
            result.confidence_score = max(confidence_scores)
            result.detection_methods = detection_methods

        result.analysis_details = {
            "echo_detection": echo_result,
            "spectral_anomalies": len(spectral_anomalies),
            "phase_anomalies": phase_anomalies,
            "amplitude_irregularities": len(amplitude_irregularities),
            "audio_properties": {
                "sample_rate": sr,
                "duration_seconds": len(y) / sr,
                "channels": 1  # librosa načítá mono
            }
        }

        return result

    def analyze_audio_steganography(self, audio_data: np.ndarray, sample_rate: int) -> AudioSteganalysisResult:
        """
        Komplexní analýza audio steganografie pomocí librosa
        """
        # Echo detection
        echo_result = self._detect_echo_steganography(audio_data, sample_rate)

        # Spektrální analýza
        spectral_anomalies = self._detect_spectral_anomalies(audio_data, sample_rate)

        # Phase analýza
        phase_anomalies = self._analyze_phase_anomalies(audio_data, sample_rate)

        # Amplitude irregularities
        amplitude_irregularities = self._detect_amplitude_irregularities(audio_data)

        # Frequency domain analýza
        frequency_analysis = self._analyze_frequency_domain(audio_data, sample_rate)

        return AudioSteganalysisResult(
            echo_detection=echo_result,
            spectral_anomalies=spectral_anomalies,
            phase_anomalies=phase_anomalies,
            amplitude_irregularities=amplitude_irregularities,
            frequency_analysis=frequency_analysis
        )

    def _detect_echo_steganography(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Detekce echo-based steganografie"""

        # Výpočet autokorelace pro detekci periodických vzorů
        autocorr = np.correlate(audio_data, audio_data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]

        # Hledání peaků v autokorelaci (možné echo delays)
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(autocorr[1:1000], height=np.max(autocorr) * 0.1)

        # Analýza periodicity
        echo_detected = len(peaks) > 3
        confidence = min(len(peaks) / 10.0, 1.0) if echo_detected else 0.0

        return {
            "echo_detected": echo_detected,
            "confidence": confidence,
            "potential_delays": peaks.tolist() if len(peaks) > 0 else [],
            "autocorrelation_peaks": len(peaks)
        }

    def _detect_spectral_anomalies(self, audio_data: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """Detekce spektrálních anomálií"""

        # STFT pro time-frequency analýzu
        hop_length = 512
        n_fft = 2048

        stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)

        # Hledání neobvyklých frekvencí s konstantní energií
        anomalies = []

        # Analýza každé frekvence
        for freq_bin in range(magnitude.shape[0]):
            freq_energy = magnitude[freq_bin, :]

            # Detekce neobvykle konstantní energie
            energy_std = np.std(freq_energy)
            energy_mean = np.mean(freq_energy)

            if energy_mean > 0 and energy_std / energy_mean < 0.1:  # Velmi malá variabilita
                freq_hz = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)[freq_bin]

                anomalies.append({
                    "frequency_hz": freq_hz,
                    "energy_mean": energy_mean,
                    "energy_std": energy_std,
                    "variability_ratio": energy_std / energy_mean
                })

        return anomalies

    def _analyze_phase_anomalies(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analýza phase anomálií"""

        # STFT pro phase analýzu
        stft = librosa.stft(audio_data)
        phase = np.angle(stft)

        # Výpočet phase differences
        phase_diff = np.diff(phase, axis=1)

        # Detekce náhlých změn ve fázi
        suspicious_changes = 0
        for freq_bin in range(phase_diff.shape[0]):
            freq_phase_diff = phase_diff[freq_bin, :]

            # Hledání velkých skoků ve fázi
            large_jumps = np.abs(freq_phase_diff) > np.pi * 0.8
            suspicious_changes += np.sum(large_jumps)

        return {
            "suspicious_phase_changes": int(suspicious_changes),
            "phase_discontinuities": suspicious_changes / phase_diff.size,
            "analysis": "Vysoký počet phase diskontinuit může indikovat steganografii"
        }

    def _detect_amplitude_irregularities(self, audio_data: np.ndarray) -> List[Dict[str, Any]]:
        """Detekce amplitude irregularities"""

        # Segmentace audio na kratší úseky
        segment_length = len(audio_data) // 100
        if segment_length < 1000:
            segment_length = 1000

        irregularities = []

        for i in range(0, len(audio_data) - segment_length, segment_length):
            segment = audio_data[i:i + segment_length]

            # Analýza dynamického rozsahu
            dynamic_range = np.max(segment) - np.min(segment)

            # Analýza RMS
            rms = np.sqrt(np.mean(segment ** 2))

            # Detekce neobvyklých vzorů
            if dynamic_range < 0.01 and rms > 0.001:  # Nízký dynamický rozsah ale nenulová energie
                irregularities.append({
                    "start_sample": i,
                    "end_sample": i + segment_length,
                    "dynamic_range": dynamic_range,
                    "rms": rms,
                    "irregularity_type": "low_dynamic_range"
                })

        return irregularities

    def _analyze_frequency_domain(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Pokročilá frequency domain analýza"""

        # FFT analýza
        fft_result = fft(audio_data)
        frequencies = fftfreq(len(audio_data), 1/sample_rate)
        magnitude = np.abs(fft_result)

        # Hledání neobvyklých frequency patterns
        # Analýza spektrální flatness
        spectral_flatness = self._calculate_spectral_flatness(magnitude)

        # Detekce hidden carriers
        potential_carriers = []
        freq_threshold = np.max(magnitude) * 0.01

        for i, (freq, mag) in enumerate(zip(frequencies[:len(frequencies)//2], magnitude[:len(magnitude)//2])):
            if mag > freq_threshold and freq > 0:
                # Kontrola, zda frekvence není harmonická
                fundamental_detected = False
                for harmonic in range(2, 10):
                    if abs(freq / harmonic) in frequencies and magnitude[np.argmin(np.abs(frequencies - freq/harmonic))] > mag * 0.5:
                        fundamental_detected = True
                        break

                if not fundamental_detected:
                    potential_carriers.append({
                        "frequency": freq,
                        "magnitude": mag,
                        "suspicion_level": "isolated_frequency"
                    })

        return {
            "spectral_flatness": spectral_flatness,
            "potential_carriers": potential_carriers[:10],  # Top 10
            "frequency_anomalies": len(potential_carriers)
        }

    def _calculate_spectral_flatness(self, magnitude: np.ndarray) -> float:
        """Výpočet spektrální flatness (Wiener entropy)"""
        # Geometric mean / Arithmetic mean
        magnitude = magnitude[magnitude > 0]  # Odstranění nul

        if len(magnitude) == 0:
            return 0.0

        geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
        arithmetic_mean = np.mean(magnitude)

        return geometric_mean / (arithmetic_mean + 1e-10)

    async def batch_analyze(self, file_paths: List[str]) -> List[SteganalysisResult]:
        """Batch analýza více souborů"""

        # Paralelní zpracování s omezením
        semaphore = asyncio.Semaphore(5)  # Max 5 souborů současně

        async def analyze_with_semaphore(file_path: str) -> SteganalysisResult:
            async with semaphore:
                return await self.analyze_file(file_path)

        tasks = [analyze_with_semaphore(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filtrování chyb
        valid_results = []
        for result in results:
            if isinstance(result, SteganalysisResult):
                valid_results.append(result)
            else:
                logger.error(f"Chyba při batch analýze: {result}")

        return valid_results

    def generate_steganalysis_report(self, results: List[SteganalysisResult]) -> Dict[str, Any]:
        """Generování reportu ze steganalytických výsledků"""

        total_files = len(results)
        suspicious_files = [r for r in results if r.steganography_detected]

        report = {
            "summary": {
                "total_files_analyzed": total_files,
                "suspicious_files_found": len(suspicious_files),
                "detection_rate": len(suspicious_files) / max(total_files, 1),
                "average_confidence": np.mean([r.confidence_score for r in suspicious_files]) if suspicious_files else 0
            },
            "detection_methods_used": {},
            "file_type_breakdown": {},
            "suspicious_files": [],
            "recommendations": []
        }

        # Analýza použitých detekčních metod
        all_methods = []
        for result in suspicious_files:
            all_methods.extend(result.detection_methods)

        unique_methods = set(all_methods)
        for method in unique_methods:
            report["detection_methods_used"][method] = all_methods.count(method)

        # Breakdown podle typů souborů
        file_types = [r.file_type for r in results]
        for file_type in set(file_types):
            report["file_type_breakdown"][file_type] = file_types.count(file_type)

        # Detail podezřelých souborů
        for result in suspicious_files:
            report["suspicious_files"].append({
                "file_path": result.file_path,
                "file_type": result.file_type,
                "confidence_score": result.confidence_score,
                "detection_methods": result.detection_methods,
                "key_findings": result.analysis_details
            })

        # Doporučení
        if suspicious_files:
            report["recommendations"].extend([
                "Prověřte podezřelé soubory manuálně",
                "Spusťte detailní forensní analýzu u vysokých confidence scores",
                "Zkontrolujte zdroj podezřelých souborů"
            ])

        if any("lsb" in str(r.detection_methods) for r in suspicious_files):
            report["recommendations"].append("Detekována možná LSB steganografie - použijte specializované nástroje pro extrakci")

        return report
