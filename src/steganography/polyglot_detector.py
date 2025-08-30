"""
Polyglot File Detector pro DeepResearchTool
Detekce souborů s více formáty a extrakce skrytých dat z polyglot struktur.
"""

import asyncio
import io
import logging
import mimetypes
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..optimization.intelligent_memory import cache_get, cache_set

logger = logging.getLogger(__name__)


@dataclass
class FileFormat:
    """Identifikovaný formát souboru"""
    format_name: str
    mime_type: str
    confidence: float
    start_offset: int
    end_offset: Optional[int] = None
    magic_bytes: bytes = b""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolyglotDetectionResult:
    """Výsledek detekce polyglot souboru"""
    file_path: str
    is_polyglot: bool
    detected_formats: List[FileFormat] = field(default_factory=list)
    trailing_data: Optional[bytes] = None
    trailing_data_size: int = 0
    hidden_files: List[Dict[str, Any]] = field(default_factory=list)
    security_implications: List[str] = field(default_factory=list)
    extraction_results: Dict[str, Any] = field(default_factory=dict)


class MagicBytesDatabase:
    """Databáze magic bytes pro detekci formátů"""

    def __init__(self):
        self.signatures = {
            # Obrázky
            "JPEG": [(b'\xff\xd8\xff\xe0', 0), (b'\xff\xd8\xff\xe1', 0), (b'\xff\xd8\xff\xdb', 0)],
            "PNG": [(b'\x89PNG\r\n\x1a\n', 0)],
            "GIF87a": [(b'GIF87a', 0)],
            "GIF89a": [(b'GIF89a', 0)],
            "BMP": [(b'BM', 0)],
            "TIFF_LE": [(b'II*\x00', 0)],
            "TIFF_BE": [(b'MM\x00*', 0)],
            "WEBP": [(b'RIFF', 0, b'WEBP', 8)],

            # Audio/Video
            "WAV": [(b'RIFF', 0, b'WAVE', 8)],
            "AVI": [(b'RIFF', 0, b'AVI ', 8)],
            "MP3": [(b'ID3', 0), (b'\xff\xfb', 0), (b'\xff\xf3', 0), (b'\xff\xf2', 0)],
            "MP4": [(b'ftyp', 4)],
            "OGG": [(b'OggS', 0)],
            "FLAC": [(b'fLaC', 0)],

            # Archivy
            "ZIP": [(b'PK\x03\x04', 0), (b'PK\x05\x06', 0), (b'PK\x07\x08', 0)],
            "RAR": [(b'Rar!\x1a\x07\x00', 0), (b'Rar!\x1a\x07\x01\x00', 0)],
            "7Z": [(b'7z\xbc\xaf\x27\x1c', 0)],
            "TAR": [(b'ustar\x00', 257), (b'ustar  \x00', 257)],
            "GZIP": [(b'\x1f\x8b', 0)],
            "BZIP2": [(b'BZ', 0)],

            # Dokumenty
            "PDF": [(b'%PDF-', 0)],
            "RTF": [(b'{\\rtf1', 0)],
            "DOC": [(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1', 0)],  # OLE2
            "DOCX": [(b'PK\x03\x04', 0)],  # ZIP-based, need further check

            # Spustitelné soubory
            "PE": [(b'MZ', 0)],  # Windows PE
            "ELF": [(b'\x7fELF', 0)],  # Linux ELF
            "MACH_O": [(b'\xfe\xed\xfa\xce', 0), (b'\xfe\xed\xfa\xcf', 0)],  # macOS
            "JAR": [(b'PK\x03\x04', 0)],  # ZIP-based Java

            # Další formáty
            "XML": [(b'<?xml', 0)],
            "HTML": [(b'<!DOCTYPE html', 0), (b'<html', 0), (b'<HTML', 0)],
            "SQLITE": [(b'SQLite format 3\x00', 0)],
            "ISO": [(b'CD001', 32769)],
        }

        # MIME type mapování
        self.mime_types = {
            "JPEG": "image/jpeg",
            "PNG": "image/png",
            "GIF87a": "image/gif",
            "GIF89a": "image/gif",
            "BMP": "image/bmp",
            "TIFF_LE": "image/tiff",
            "TIFF_BE": "image/tiff",
            "WEBP": "image/webp",
            "WAV": "audio/wav",
            "MP3": "audio/mpeg",
            "MP4": "video/mp4",
            "PDF": "application/pdf",
            "ZIP": "application/zip",
            "RAR": "application/x-rar-compressed",
            "PE": "application/x-msdownload",
            "ELF": "application/x-executable"
        }

    def detect_format(self, data: bytes, offset: int = 0) -> Optional[FileFormat]:
        """Detekce formátu podle magic bytes"""
        for format_name, signatures in self.signatures.items():
            for sig_data in signatures:
                if len(sig_data) == 2:  # Jednoduchý pattern na offsetu
                    pattern, pattern_offset = sig_data
                    check_offset = offset + pattern_offset

                    if (check_offset + len(pattern) <= len(data) and
                        data[check_offset:check_offset + len(pattern)] == pattern):

                        return FileFormat(
                            format_name=format_name,
                            mime_type=self.mime_types.get(format_name, "application/octet-stream"),
                            confidence=0.9,
                            start_offset=offset,
                            magic_bytes=pattern
                        )

                elif len(sig_data) == 4:  # Pattern s druhým kontrol místem
                    pattern1, offset1, pattern2, offset2 = sig_data
                    check_offset1 = offset + offset1
                    check_offset2 = offset + offset2

                    if (check_offset1 + len(pattern1) <= len(data) and
                        check_offset2 + len(pattern2) <= len(data) and
                        data[check_offset1:check_offset1 + len(pattern1)] == pattern1 and
                        data[check_offset2:check_offset2 + len(pattern2)] == pattern2):

                        return FileFormat(
                            format_name=format_name,
                            mime_type=self.mime_types.get(format_name, "application/octet-stream"),
                            confidence=0.95,
                            start_offset=offset,
                            magic_bytes=pattern1 + pattern2
                        )

        return None


class PolyglotFileDetector:
    """
    Pokročilý detektor polyglot souborů s extrakcí skrytých dat.
    Detekuje soubory obsahující více formátů a data připojená za konec.
    """

    def __init__(self, max_file_size_mb: int = 500):
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.magic_db = MagicBytesDatabase()

        logger.info("PolyglotFileDetector inicializován")

    async def analyze_file(self, file_path: str) -> PolyglotDetectionResult:
        """
        Hlavní metoda pro analýzu polyglot souboru
        """
        cache_key = f"polyglot_analysis:{file_path}"
        cached_result = await cache_get(cache_key)

        if cached_result:
            return PolyglotDetectionResult(**cached_result)

        result = PolyglotDetectionResult(
            file_path=file_path,
            is_polyglot=False
        )

        try:
            # Načtení souboru
            with open(file_path, 'rb') as f:
                file_data = f.read(self.max_file_size_bytes)

            if not file_data:
                result.extraction_results["error"] = "Prázdný soubor"
                return result

            # Detekce všech formátů v souboru
            detected_formats = await self._scan_for_formats(file_data)
            result.detected_formats = detected_formats

            # Kontrola, zda je to polyglot
            if len(detected_formats) > 1:
                result.is_polyglot = True

            # Extrakce trailing dat
            trailing_data = await self._extract_trailing_data(file_data, detected_formats)
            if trailing_data:
                result.trailing_data = trailing_data
                result.trailing_data_size = len(trailing_data)

                # Analýza trailing dat
                trailing_analysis = await self._analyze_trailing_data(trailing_data)
                result.hidden_files.extend(trailing_analysis.get("hidden_files", []))

            # Bezpečnostní implikace
            result.security_implications = self._assess_security_implications(result)

            # Pokus o extrakci skrytých souborů
            extraction_results = await self._attempt_extraction(file_data, detected_formats)
            result.extraction_results = extraction_results

        except Exception as e:
            result.extraction_results["error"] = str(e)
            logger.error(f"Chyba při analýze polyglot souboru {file_path}: {e}")

        # Cache výsledku
        result_dict = {
            "file_path": result.file_path,
            "is_polyglot": result.is_polyglot,
            "detected_formats": [
                {
                    "format_name": f.format_name,
                    "mime_type": f.mime_type,
                    "confidence": f.confidence,
                    "start_offset": f.start_offset,
                    "end_offset": f.end_offset,
                    "metadata": f.metadata
                }
                for f in result.detected_formats
            ],
            "trailing_data_size": result.trailing_data_size,
            "hidden_files": result.hidden_files,
            "security_implications": result.security_implications,
            "extraction_results": result.extraction_results
        }
        await cache_set(cache_key, result_dict, ttl=3600)

        return result

    async def _scan_for_formats(self, data: bytes) -> List[FileFormat]:
        """Skenování souboru pro všechny detekované formáty"""
        detected_formats = []
        scan_window = 1024  # Velikost okna pro skenování

        # Skenování od začátku souboru
        for offset in range(0, min(len(data), 50000), scan_window):
            chunk = data[offset:offset + scan_window]

            # Zkusíme detekci na tomto offsetu
            detected_format = self.magic_db.detect_format(data, offset)
            if detected_format:
                # Kontrola, zda už není podobný formát detekován
                is_duplicate = False
                for existing in detected_formats:
                    if (existing.format_name == detected_format.format_name and
                        abs(existing.start_offset - detected_format.start_offset) < 100):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    # Pokus o určení konce formátu
                    end_offset = await self._determine_format_end(data, detected_format)
                    detected_format.end_offset = end_offset
                    detected_formats.append(detected_format)

        # Řazení podle offset
        detected_formats.sort(key=lambda x: x.start_offset)

        return detected_formats

    async def _determine_format_end(self, data: bytes, format_info: FileFormat) -> Optional[int]:
        """Pokus o určení konce konkrétního formátu"""

        if format_info.format_name == "JPEG":
            # JPEG končí FF D9
            start = format_info.start_offset
            for i in range(start + 2, len(data) - 1):
                if data[i] == 0xFF and data[i + 1] == 0xD9:
                    return i + 2

        elif format_info.format_name == "PNG":
            # PNG končí IEND chunk
            start = format_info.start_offset
            iend_marker = b'IEND\xae\x42\x60\x82'
            iend_pos = data.find(iend_marker, start)
            if iend_pos != -1:
                return iend_pos + len(iend_marker)

        elif format_info.format_name == "PDF":
            # PDF končí %%EOF
            start = format_info.start_offset
            eof_marker = b'%%EOF'
            eof_pos = data.find(eof_marker, start)
            if eof_pos != -1:
                return eof_pos + len(eof_marker)

        elif format_info.format_name in ["ZIP", "DOCX", "JAR"]:
            # ZIP má central directory na konci
            return await self._find_zip_end(data, format_info.start_offset)

        # Fallback: pokus o heuristiku
        return None

    async def _find_zip_end(self, data: bytes, start_offset: int) -> Optional[int]:
        """Najití konce ZIP archivu"""
        # Hledání End of Central Directory Record
        eocd_signature = b'PK\x05\x06'

        # Začneme hledat od konce souboru
        for i in range(len(data) - 22, start_offset, -1):
            if data[i:i+4] == eocd_signature:
                # Našli jsme EOCD, čteme velikost komentáře
                if i + 20 < len(data):
                    comment_length = struct.unpack('<H', data[i+20:i+22])[0]
                    return i + 22 + comment_length

        return None

    async def _extract_trailing_data(self, data: bytes, formats: List[FileFormat]) -> Optional[bytes]:
        """Extrakce dat za posledním identifikovaným formátem"""
        if not formats:
            return None

        # Najdeme nejdelší formát nebo ten s největším end_offset
        last_end = 0

        for format_info in formats:
            if format_info.end_offset:
                last_end = max(last_end, format_info.end_offset)
            else:
                # Pokud nemáme end_offset, použijeme heuristiku
                estimated_end = format_info.start_offset + 1024  # Minimální odhad
                last_end = max(last_end, estimated_end)

        # Pokud je za posledním formátem více dat
        if last_end < len(data) - 10:  # Alespoň 10 bytů trailing dat
            trailing_data = data[last_end:]

            # Filtrování prázdných nebo padding bytů
            if len(set(trailing_data)) > 1:  # Více než jen jeden typ byte
                return trailing_data

        return None

    async def _analyze_trailing_data(self, trailing_data: bytes) -> Dict[str, Any]:
        """Analýza trailing dat pro možné skryté soubory"""
        analysis = {
            "size": len(trailing_data),
            "entropy": self._calculate_entropy(trailing_data),
            "hidden_files": [],
            "text_content": None,
            "possible_archive": False
        }

        # Pokus o detekci formátů v trailing datech
        hidden_format = self.magic_db.detect_format(trailing_data)
        if hidden_format:
            analysis["hidden_files"].append({
                "format": hidden_format.format_name,
                "mime_type": hidden_format.mime_type,
                "offset_in_trailing": hidden_format.start_offset,
                "confidence": hidden_format.confidence
            })

        # Kontrola, zda obsahují textová data
        try:
            text_content = trailing_data.decode('utf-8', errors='ignore')
            if len(text_content.strip()) > 10 and text_content.isprintable():
                analysis["text_content"] = text_content[:200]  # První 200 znaků
        except:
            pass

        # Kontrola archivních signatur
        archive_signatures = [b'PK\x03\x04', b'Rar!', b'7z\xbc\xaf\x27\x1c']
        for sig in archive_signatures:
            if sig in trailing_data:
                analysis["possible_archive"] = True
                break

        return analysis

    def _calculate_entropy(self, data: bytes) -> float:
        """Výpočet entropy dat"""
        if not data:
            return 0.0

        # Počet výskytů každého byte
        byte_counts = [0] * 256
        for byte in data:
            byte_counts[byte] += 1

        # Výpočet entropy
        entropy = 0.0
        data_len = len(data)

        for count in byte_counts:
            if count > 0:
                probability = count / data_len
                entropy -= probability * math.log2(probability)

        return entropy

    def extract_trailing_data(self, file_path: str, output_path: str) -> bool:
        """
        Extrakce trailing dat do samostatného souboru
        """
        try:
            with open(file_path, 'rb') as f:
                data = f.read()

            # Rychlá detekce formátů pro určení konce
            formats = []
            main_format = self.magic_db.detect_format(data)
            if main_format:
                formats.append(main_format)

            # Extrakce trailing dat
            trailing_data = None
            if main_format and main_format.end_offset:
                if main_format.end_offset < len(data) - 10:
                    trailing_data = data[main_format.end_offset:]

            if trailing_data:
                with open(output_path, 'wb') as f:
                    f.write(trailing_data)
                return True

        except Exception as e:
            logger.error(f"Chyba při extrakci trailing dat: {e}")

        return False

    async def _attempt_extraction(self, data: bytes, formats: List[FileFormat]) -> Dict[str, Any]:
        """Pokus o extrakci jednotlivých formátů ze souboru"""
        extraction_results = {
            "extracted_formats": [],
            "extraction_errors": [],
            "total_formats": len(formats)
        }

        for i, format_info in enumerate(formats):
            try:
                start = format_info.start_offset
                end = format_info.end_offset or len(data)

                extracted_data = data[start:end]

                extraction_info = {
                    "format_name": format_info.format_name,
                    "mime_type": format_info.mime_type,
                    "start_offset": start,
                    "end_offset": end,
                    "extracted_size": len(extracted_data),
                    "valid_format": self._validate_extracted_format(extracted_data, format_info)
                }

                extraction_results["extracted_formats"].append(extraction_info)

            except Exception as e:
                extraction_results["extraction_errors"].append({
                    "format": format_info.format_name,
                    "error": str(e)
                })

        return extraction_results

    def _validate_extracted_format(self, data: bytes, format_info: FileFormat) -> bool:
        """Validace, zda extrahovaná data tvoří validní soubor daného formátu"""

        # Základní kontrola magic bytes
        if not data.startswith(format_info.magic_bytes):
            return False

        # Specifické kontroly pro různé formáty
        if format_info.format_name == "JPEG":
            # JPEG musí končit FF D9
            return data.endswith(b'\xff\xd9')

        elif format_info.format_name == "PNG":
            # PNG musí obsahovat IHDR a končit IEND
            return b'IHDR' in data and data.endswith(b'IEND\xae\x42\x60\x82')

        elif format_info.format_name == "PDF":
            # PDF musí končit %%EOF
            return b'%%EOF' in data

        elif format_info.format_name in ["ZIP", "DOCX", "JAR"]:
            # ZIP musí mít central directory
            return b'PK\x01\x02' in data  # Central directory file header

        # Pro ostatní formáty stačí magic bytes
        return True

    def _assess_security_implications(self, result: PolyglotDetectionResult) -> List[str]:
        """Hodnocení bezpečnostních implikací polyglot souboru"""
        implications = []

        if result.is_polyglot:
            implications.append("Soubor obsahuje více formátů - možný polyglot útok")

        if result.trailing_data_size > 0:
            implications.append(f"Detekována trailing data ({result.trailing_data_size} bytů)")

        # Kontrola nebezpečných kombinací
        format_names = [f.format_name for f in result.detected_formats]

        if "PE" in format_names or "ELF" in format_names:
            implications.append("Obsahuje spustitelný soubor - vysoké bezpečnostní riziko")

        if "ZIP" in format_names and any(f in format_names for f in ["JPEG", "PNG", "PDF"]):
            implications.append("Kombinace archiv + média - možná skrytá payload")

        if len(result.detected_formats) > 3:
            implications.append("Více než 3 formáty - vysoce suspektní polyglot")

        return implications

    async def batch_analyze(self, file_paths: List[str]) -> List[PolyglotDetectionResult]:
        """Batch analýza více souborů"""
        semaphore = asyncio.Semaphore(3)  # Omezení souběžnosti

        async def analyze_with_semaphore(file_path: str) -> PolyglotDetectionResult:
            async with semaphore:
                return await self.analyze_file(file_path)

        tasks = [analyze_with_semaphore(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results = []
        for result in results:
            if isinstance(result, PolyglotDetectionResult):
                valid_results.append(result)

        return valid_results

    def generate_polyglot_report(self, results: List[PolyglotDetectionResult]) -> Dict[str, Any]:
        """Generování reportu z polyglot analýzy"""

        total_files = len(results)
        polyglot_files = [r for r in results if r.is_polyglot]
        files_with_trailing = [r for r in results if r.trailing_data_size > 0]

        report = {
            "summary": {
                "total_files_analyzed": total_files,
                "polyglot_files_found": len(polyglot_files),
                "files_with_trailing_data": len(files_with_trailing),
                "detection_rate": len(polyglot_files) / max(total_files, 1)
            },
            "format_statistics": {},
            "security_analysis": {
                "high_risk_files": [],
                "total_security_implications": 0
            },
            "polyglot_files": [],
            "recommendations": []
        }

        # Statistiky formátů
        all_formats = []
        for result in results:
            all_formats.extend([f.format_name for f in result.detected_formats])

        unique_formats = set(all_formats)
        for format_name in unique_formats:
            report["format_statistics"][format_name] = all_formats.count(format_name)

        # Bezpečnostní analýza
        for result in polyglot_files:
            report["security_analysis"]["total_security_implications"] += len(result.security_implications)

            if len(result.security_implications) > 2:  # Vysoké riziko
                report["security_analysis"]["high_risk_files"].append({
                    "file_path": result.file_path,
                    "formats_count": len(result.detected_formats),
                    "security_implications": result.security_implications
                })

        # Detail polyglot souborů
        for result in polyglot_files:
            report["polyglot_files"].append({
                "file_path": result.file_path,
                "detected_formats": [f.format_name for f in result.detected_formats],
                "trailing_data_size": result.trailing_data_size,
                "security_risk_level": "high" if len(result.security_implications) > 2 else "medium"
            })

        # Doporučení
        if polyglot_files:
            report["recommendations"].extend([
                "Prověřte všechny polyglot soubory manuálně",
                "Spusťte antivirus skenování na podezřelé soubory",
                "Extrahujte a analyzujte trailing data"
            ])

        if files_with_trailing:
            report["recommendations"].append("Zkontrolujte soubory s trailing daty na skrytý obsah")

        return report
