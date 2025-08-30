"""Steganography Analyzer for Hidden Content Detection
Image LSB detection, EXIF/metadata analysis, frequency heuristics
Audio metadata and spectrogram analysis with suspicion scoring
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from PIL import ExifTags, Image
    from PIL.ExifTags import TAGS

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL/Pillow not available - image analysis limited")

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available - statistical analysis limited")


class StegoType(Enum):
    """Types of steganography detection"""

    LSB_IMAGE = "lsb_image"
    METADATA_HIDDEN = "metadata_hidden"
    FREQUENCY_ANOMALY = "frequency_anomaly"
    AUDIO_SPECTRAL = "audio_spectral"
    FILE_STRUCTURE = "file_structure"


@dataclass
class StegoConfig:
    """Steganography analyzer configuration"""

    # Analysis types to perform
    enable_lsb_analysis: bool = True
    enable_metadata_analysis: bool = True
    enable_frequency_analysis: bool = True
    enable_audio_analysis: bool = False  # Requires additional dependencies

    # Thresholds for detection
    lsb_entropy_threshold: float = 7.8  # High entropy suggests randomness
    metadata_suspicious_keys: list[str] = field(
        default_factory=lambda: ["comment", "description", "user_comment", "software", "artist"]
    )
    frequency_anomaly_threshold: float = 0.15

    # File size limits (MB)
    max_image_size_mb: float = 50.0
    max_audio_size_mb: float = 100.0

    # Processing settings
    batch_size: int = 10
    enable_parallel_processing: bool = True
    max_workers: int = 4

    # Output settings
    save_analysis_images: bool = False
    detailed_reporting: bool = True


@dataclass
class StegoEvidence:
    """Evidence of potential steganography"""

    evidence_type: StegoType
    confidence: float
    description: str
    technical_details: dict[str, Any]
    file_location: str | None = None


@dataclass
class StegoResult:
    """Result from steganography analysis"""

    file_path: str
    file_type: str
    file_size_bytes: int
    analysis_timestamp: datetime

    # Detection results
    suspicion_score: float  # 0.0 = clean, 1.0 = highly suspicious
    evidence_found: list[StegoEvidence]

    # Technical analysis
    entropy_analysis: dict[str, float] | None = None
    metadata_analysis: dict[str, Any] | None = None
    frequency_analysis: dict[str, Any] | None = None

    # Processing info
    analysis_time_seconds: float = 0.0
    error_message: str | None = None


class ImageLSBAnalyzer:
    """Least Significant Bit analysis for images"""

    def __init__(self, config: StegoConfig):
        self.config = config

    async def analyze_image(self, image_path: Path) -> list[StegoEvidence]:
        """Analyze image for LSB steganography"""
        if not PIL_AVAILABLE or not NUMPY_AVAILABLE:
            return []

        evidence = []

        try:
            image = Image.open(image_path)

            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Get image data as numpy array
            img_array = np.array(image)

            # Analyze each color channel
            for channel_idx, channel_name in enumerate(["Red", "Green", "Blue"]):
                channel_data = img_array[:, :, channel_idx]
                channel_evidence = await self._analyze_channel_lsb(
                    channel_data, channel_name, str(image_path)
                )
                evidence.extend(channel_evidence)

            # Overall image entropy analysis
            overall_entropy = self._calculate_entropy(img_array.flatten())
            if overall_entropy > self.config.lsb_entropy_threshold:
                evidence.append(
                    StegoEvidence(
                        evidence_type=StegoType.LSB_IMAGE,
                        confidence=min(
                            (overall_entropy - self.config.lsb_entropy_threshold) / 2, 1.0
                        ),
                        description=f"High entropy detected: {overall_entropy:.2f}",
                        technical_details={
                            "entropy": overall_entropy,
                            "threshold": self.config.lsb_entropy_threshold,
                            "analysis_type": "overall_image",
                        },
                        file_location=str(image_path),
                    )
                )

        except Exception as e:
            logger.warning(f"LSB analysis failed for {image_path}: {e}")

        return evidence

    async def _analyze_channel_lsb(
        self, channel_data: np.ndarray, channel_name: str, file_path: str
    ) -> list[StegoEvidence]:
        """Analyze single color channel for LSB patterns"""
        evidence = []

        # Extract LSBs
        lsb_data = channel_data & 1

        # Calculate entropy of LSB plane
        lsb_entropy = self._calculate_entropy(lsb_data.flatten())

        # High entropy in LSB plane suggests hidden data
        if lsb_entropy > 0.9:  # LSB should be random for hidden data
            confidence = min(lsb_entropy, 1.0)
            evidence.append(
                StegoEvidence(
                    evidence_type=StegoType.LSB_IMAGE,
                    confidence=confidence,
                    description=f"Suspicious LSB entropy in {channel_name} channel: {lsb_entropy:.3f}",
                    technical_details={
                        "channel": channel_name,
                        "lsb_entropy": lsb_entropy,
                        "expected_range": "0.5-0.7 for natural images",
                    },
                    file_location=file_path,
                )
            )

        # Chi-square test for LSB randomness
        chi_square = self._chi_square_test(lsb_data.flatten())
        if chi_square > 100:  # High chi-square suggests non-random pattern
            evidence.append(
                StegoEvidence(
                    evidence_type=StegoType.LSB_IMAGE,
                    confidence=min(chi_square / 1000, 1.0),
                    description=f"Chi-square test indicates LSB manipulation in {channel_name}",
                    technical_details={
                        "channel": channel_name,
                        "chi_square": chi_square,
                        "interpretation": "High value suggests hidden data",
                    },
                    file_location=file_path,
                )
            )

        return evidence

    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate Shannon entropy of data"""
        if len(data) == 0:
            return 0.0

        # Count occurrences of each value
        values, counts = np.unique(data, return_counts=True)
        probabilities = counts / len(data)

        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy

    def _chi_square_test(self, data: np.ndarray) -> float:
        """Simple chi-square test for randomness"""
        if len(data) < 100:
            return 0.0

        # Count 0s and 1s
        ones = np.sum(data)
        zeros = len(data) - ones

        # Expected frequency (should be roughly equal for random data)
        expected = len(data) / 2

        # Chi-square calculation
        chi_square = ((ones - expected) ** 2 / expected) + ((zeros - expected) ** 2 / expected)
        return chi_square


class MetadataAnalyzer:
    """Analyze file metadata for hidden information"""

    def __init__(self, config: StegoConfig):
        self.config = config

    async def analyze_image_metadata(self, image_path: Path) -> list[StegoEvidence]:
        """Analyze image EXIF and metadata"""
        if not PIL_AVAILABLE:
            return []

        evidence = []

        try:
            image = Image.open(image_path)

            # Extract EXIF data
            exif_data = {}
            if hasattr(image, "_getexif") and image._getexif() is not None:
                exif = image._getexif()
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_data[tag] = value

            # Check for suspicious metadata
            suspicious_findings = self._analyze_metadata_content(exif_data)
            evidence.extend(suspicious_findings)

            # Check for unusual metadata size
            metadata_size = len(str(exif_data))
            if metadata_size > 5000:  # Unusually large metadata
                evidence.append(
                    StegoEvidence(
                        evidence_type=StegoType.METADATA_HIDDEN,
                        confidence=min(metadata_size / 20000, 1.0),
                        description=f"Unusually large metadata size: {metadata_size} bytes",
                        technical_details={
                            "metadata_size": metadata_size,
                            "suspicious_threshold": 5000,
                            "total_fields": len(exif_data),
                        },
                        file_location=str(image_path),
                    )
                )

        except Exception as e:
            logger.warning(f"Metadata analysis failed for {image_path}: {e}")

        return evidence

    def _analyze_metadata_content(self, metadata: dict[str, Any]) -> list[StegoEvidence]:
        """Analyze metadata content for suspicious patterns"""
        evidence = []

        for key, value in metadata.items():
            if isinstance(value, str):
                # Check for base64-like strings
                if self._looks_like_base64(value):
                    evidence.append(
                        StegoEvidence(
                            evidence_type=StegoType.METADATA_HIDDEN,
                            confidence=0.7,
                            description=f"Base64-like data in {key} field",
                            technical_details={
                                "field": key,
                                "value_length": len(value),
                                "pattern": "base64_like",
                            },
                        )
                    )

                # Check for suspicious keywords in metadata
                if key.lower() in self.config.metadata_suspicious_keys:
                    if len(value) > 200:  # Unusually long comment/description
                        evidence.append(
                            StegoEvidence(
                                evidence_type=StegoType.METADATA_HIDDEN,
                                confidence=0.6,
                                description=f"Unusually long {key}: {len(value)} characters",
                                technical_details={
                                    "field": key,
                                    "length": len(value),
                                    "first_100_chars": value[:100],
                                },
                            )
                        )

                # Check for hidden binary data patterns
                if self._contains_binary_patterns(value):
                    evidence.append(
                        StegoEvidence(
                            evidence_type=StegoType.METADATA_HIDDEN,
                            confidence=0.8,
                            description=f"Binary data patterns in {key}",
                            technical_details={"field": key, "pattern_type": "binary_like"},
                        )
                    )

        return evidence

    def _looks_like_base64(self, text: str) -> bool:
        """Check if text looks like base64 encoded data"""
        import re

        # Base64 pattern: 4-character groups, ending with = padding
        base64_pattern = re.compile(r"^[A-Za-z0-9+/]*={0,2}$")

        if len(text) < 20:  # Too short to be meaningful
            return False

        if len(text) % 4 != 0:  # Base64 should be multiple of 4
            return False

        return bool(base64_pattern.match(text))

    def _contains_binary_patterns(self, text: str) -> bool:
        """Check for binary data patterns in text"""
        # Look for high ratio of non-printable characters
        non_printable = sum(1 for c in text if ord(c) < 32 or ord(c) > 126)
        ratio = non_printable / len(text) if text else 0

        return ratio > 0.1  # More than 10% non-printable suggests binary


class FrequencyAnalyzer:
    """Analyze frequency domain for hidden patterns"""

    def __init__(self, config: StegoConfig):
        self.config = config

    async def analyze_image_frequency(self, image_path: Path) -> list[StegoEvidence]:
        """Analyze image in frequency domain"""
        if not PIL_AVAILABLE or not NUMPY_AVAILABLE:
            return []

        evidence = []

        try:
            image = Image.open(image_path)
            if image.mode != "L":  # Convert to grayscale
                image = image.convert("L")

            img_array = np.array(image, dtype=np.float32)

            # Perform 2D FFT
            fft_data = np.fft.fft2(img_array)
            fft_magnitude = np.abs(fft_data)

            # Analyze frequency spectrum for anomalies
            anomalies = self._detect_frequency_anomalies(fft_magnitude)
            evidence.extend(anomalies)

        except Exception as e:
            logger.warning(f"Frequency analysis failed for {image_path}: {e}")

        return evidence

    def _detect_frequency_anomalies(self, fft_magnitude: np.ndarray) -> list[StegoEvidence]:
        """Detect anomalies in frequency spectrum"""
        evidence = []

        # Calculate frequency distribution statistics
        mean_magnitude = np.mean(fft_magnitude)
        std_magnitude = np.std(fft_magnitude)

        # Look for unusual spikes in frequency domain
        threshold = mean_magnitude + 3 * std_magnitude
        spikes = np.sum(fft_magnitude > threshold)

        spike_ratio = spikes / fft_magnitude.size
        if spike_ratio > self.config.frequency_anomaly_threshold:
            evidence.append(
                StegoEvidence(
                    evidence_type=StegoType.FREQUENCY_ANOMALY,
                    confidence=min(spike_ratio * 5, 1.0),
                    description=f"Frequency domain anomalies detected: {spike_ratio:.3f} spike ratio",
                    technical_details={
                        "spike_ratio": spike_ratio,
                        "threshold": self.config.frequency_anomaly_threshold,
                        "total_spikes": int(spikes),
                        "mean_magnitude": float(mean_magnitude),
                        "std_magnitude": float(std_magnitude),
                    },
                )
            )

        return evidence


class SteganographyAnalyzer:
    """Main steganography analysis engine"""

    def __init__(self, config: StegoConfig = None):
        self.config = config or StegoConfig()

        # Initialize analyzers
        self.lsb_analyzer = ImageLSBAnalyzer(self.config)
        self.metadata_analyzer = MetadataAnalyzer(self.config)
        self.frequency_analyzer = FrequencyAnalyzer(self.config)

        # Statistics
        self.files_analyzed = 0
        self.suspicious_files_found = 0

    async def analyze_file(self, file_path: Path) -> StegoResult:
        """Analyze single file for steganography"""
        start_time = datetime.now()

        try:
            # Basic file info
            file_size = file_path.stat().st_size
            file_type = self._detect_file_type(file_path)

            # Check file size limits
            if file_type == "image" and file_size > self.config.max_image_size_mb * 1024 * 1024:
                return StegoResult(
                    file_path=str(file_path),
                    file_type=file_type,
                    file_size_bytes=file_size,
                    analysis_timestamp=start_time,
                    suspicion_score=0.0,
                    evidence_found=[],
                    error_message=f"File too large: {file_size / (1024*1024):.1f}MB",
                )

            all_evidence = []
            analysis_data = {}

            # Perform different types of analysis based on file type
            if file_type == "image":
                # LSB Analysis
                if self.config.enable_lsb_analysis:
                    lsb_evidence = await self.lsb_analyzer.analyze_image(file_path)
                    all_evidence.extend(lsb_evidence)

                # Metadata Analysis
                if self.config.enable_metadata_analysis:
                    metadata_evidence = await self.metadata_analyzer.analyze_image_metadata(
                        file_path
                    )
                    all_evidence.extend(metadata_evidence)

                # Frequency Analysis
                if self.config.enable_frequency_analysis:
                    freq_evidence = await self.frequency_analyzer.analyze_image_frequency(file_path)
                    all_evidence.extend(freq_evidence)

            # Calculate overall suspicion score
            suspicion_score = self._calculate_suspicion_score(all_evidence)

            # Create result
            processing_time = (datetime.now() - start_time).total_seconds()

            result = StegoResult(
                file_path=str(file_path),
                file_type=file_type,
                file_size_bytes=file_size,
                analysis_timestamp=start_time,
                suspicion_score=suspicion_score,
                evidence_found=all_evidence,
                analysis_time_seconds=processing_time,
            )

            # Update statistics
            self.files_analyzed += 1
            if suspicion_score > 0.5:
                self.suspicious_files_found += 1

            return result

        except Exception as e:
            logger.error(f"Analysis failed for {file_path}: {e}")
            return StegoResult(
                file_path=str(file_path),
                file_type="unknown",
                file_size_bytes=0,
                analysis_timestamp=start_time,
                suspicion_score=0.0,
                evidence_found=[],
                error_message=str(e),
            )

    async def analyze_batch(self, file_paths: list[Path]) -> list[StegoResult]:
        """Analyze multiple files in batch"""
        if self.config.enable_parallel_processing:
            # Process files in parallel
            semaphore = asyncio.Semaphore(self.config.max_workers)

            async def analyze_with_semaphore(path):
                async with semaphore:
                    return await self.analyze_file(path)

            tasks = [analyze_with_semaphore(path) for path in file_paths]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions
            valid_results = [r for r in results if isinstance(r, StegoResult)]
            return valid_results
        # Sequential processing
        results = []
        for file_path in file_paths:
            result = await self.analyze_file(file_path)
            results.append(result)

        return results

    def _detect_file_type(self, file_path: Path) -> str:
        """Detect file type from extension"""
        extension = file_path.suffix.lower()

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}
        audio_extensions = {".wav", ".mp3", ".flac", ".ogg"}

        if extension in image_extensions:
            return "image"
        if extension in audio_extensions:
            return "audio"
        return "unknown"

    def _calculate_suspicion_score(self, evidence_list: list[StegoEvidence]) -> float:
        """Calculate overall suspicion score from evidence"""
        if not evidence_list:
            return 0.0

        # Weight different types of evidence
        type_weights = {
            StegoType.LSB_IMAGE: 1.0,
            StegoType.METADATA_HIDDEN: 0.8,
            StegoType.FREQUENCY_ANOMALY: 0.9,
            StegoType.AUDIO_SPECTRAL: 1.0,
            StegoType.FILE_STRUCTURE: 0.7,
        }

        # Calculate weighted average of evidence confidences
        total_weight = 0
        weighted_sum = 0

        for evidence in evidence_list:
            weight = type_weights.get(evidence.evidence_type, 0.5)
            weighted_sum += evidence.confidence * weight
            total_weight += weight

        base_score = weighted_sum / total_weight if total_weight > 0 else 0

        # Bonus for multiple types of evidence
        unique_types = len(set(e.evidence_type for e in evidence_list))
        if unique_types > 1:
            base_score = min(base_score * (1 + unique_types * 0.1), 1.0)

        return base_score

    def get_analysis_stats(self) -> dict[str, Any]:
        """Get analysis statistics"""
        return {
            "files_analyzed": self.files_analyzed,
            "suspicious_files_found": self.suspicious_files_found,
            "suspicion_rate": self.suspicious_files_found / max(1, self.files_analyzed),
            "analyzers_enabled": {
                "lsb_analysis": self.config.enable_lsb_analysis,
                "metadata_analysis": self.config.enable_metadata_analysis,
                "frequency_analysis": self.config.enable_frequency_analysis,
                "audio_analysis": self.config.enable_audio_analysis,
            },
        }


# Utility functions
async def analyze_image_for_steganography(image_path: Path) -> StegoResult:
    """Quick steganography analysis for single image"""
    config = StegoConfig(
        enable_lsb_analysis=True, enable_metadata_analysis=True, enable_frequency_analysis=True
    )

    analyzer = SteganographyAnalyzer(config)
    return await analyzer.analyze_file(image_path)


async def batch_steganography_scan(
    directory: Path, file_pattern: str = "*.jpg"
) -> list[StegoResult]:
    """Scan directory for potential steganography"""
    config = StegoConfig(enable_parallel_processing=True, max_workers=4)

    analyzer = SteganographyAnalyzer(config)

    # Find matching files
    file_paths = list(directory.glob(file_pattern))

    if not file_paths:
        logger.warning(f"No files found matching pattern {file_pattern} in {directory}")
        return []

    return await analyzer.analyze_batch(file_paths)


__all__ = [
    "SteganographyAnalyzer",
    "StegoConfig",
    "StegoEvidence",
    "StegoResult",
    "StegoType",
    "analyze_image_for_steganography",
    "batch_steganography_scan",
]
