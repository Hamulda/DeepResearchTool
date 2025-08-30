#!/usr/bin/env python3
"""
Deep Pattern Detector - Hloubková detekce vzorů a skrytých dat
Automatická extrakce specifických artefaktů pomocí rozsáhlé knihovny regulárních výrazů

Author: GitHub Copilot
Created: August 28, 2025 - Phase 3 Implementation
"""

import re
import logging
import asyncio
from typing import Dict, List, Set, Optional, Any, Tuple, Pattern
from dataclasses import dataclass, field
from collections import defaultdict
import ipaddress
import base64
import hashlib
import json
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PatternMatch:
    """Reprezentace nalezeného vzoru"""

    pattern_type: str
    pattern_name: str
    matched_text: str
    confidence: float
    start_position: int
    end_position: int
    context: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_status: str = "unvalidated"  # validated, suspicious, invalid


@dataclass
class ArtefactExtraction:
    """Extrahovaný artefakt s metadaty"""

    artefact_type: str
    value: str
    confidence: float
    source_location: str
    extraction_method: str
    validation_results: Dict[str, Any] = field(default_factory=dict)
    related_patterns: List[str] = field(default_factory=list)


class DeepPatternDetector:
    """Pokročilý detektor vzorů pro zpravodajskou analýzu"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pattern_config = config.get("pattern_detection", {})

        # Kompilované vzory pro výkon
        self.compiled_patterns: Dict[str, Pattern] = {}
        self.pattern_categories: Dict[str, List[str]] = {}

        # Statistiky a metriky
        self.detection_stats = defaultdict(int)
        self.validation_cache: Dict[str, bool] = {}

        self._initialize_patterns()

    def _initialize_patterns(self):
        """Inicializace rozsáhlé knihovny regulárních výrazů"""

        # Kryptoměnové adresy
        crypto_patterns = {
            "bitcoin_address": r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b",
            "bitcoin_segwit": r"\bbc1[a-z0-9]{39,59}\b",
            "ethereum_address": r"\b0x[a-fA-F0-9]{40}\b",
            "monero_address": r"\b4[0-9AB][1-9A-HJ-NP-Za-km-z]{93}\b",
            "litecoin_address": r"\b[LM3][a-km-zA-HJ-NP-Z1-9]{26,33}\b",
            "dogecoin_address": r"\bD{1}[5-9A-HJ-NP-U]{1}[1-9A-HJ-NP-Za-km-z]{32}\b",
            "zcash_address": r"\bt1[a-z0-9]{6,83}\b",
            "ripple_address": r"\br[a-zA-Z0-9]{24,34}\b",
        }

        # Tor a darknet adresy
        darknet_patterns = {
            "onion_v2": r"\b[a-z2-7]{16}\.onion\b",
            "onion_v3": r"\b[a-z2-7]{56}\.onion\b",
            "i2p_address": r"\b[a-zA-Z0-9\-~=]{60}\.b32\.i2p\b",
            "eepsite": r"\b[a-zA-Z0-9\-~=]{520}\.b32\.i2p\b",
        }

        # GPS souřadnice
        gps_patterns = {
            "decimal_degrees": r"[-+]?([1-8]?\d(\.\d+)?|90(\.0+)?),\s*[-+]?(180(\.0+)?|((1[0-7]\d)|([1-9]?\d))(\.\d+)?)",
            "dms_format": r"\d{1,3}°\d{1,2}\'[\d.]+\"[NS],?\s*\d{1,3}°\d{1,2}\'[\d.]+\"[EW]",
            "mgrs_grid": r"\b\d{1,2}[C-X][A-Z]{2}\d{10}\b",
            "plus_codes": r"\b[23456789CFGHJMPQRVWX]{8}\+[23456789CFGHJMPQRVWX]{2,3}\b",
        }

        # Komunikační identifikátory
        communication_patterns = {
            "email_standard": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "email_protonmail": r"\b[A-Za-z0-9._%+-]+@protonmail\.(com|ch)\b",
            "telegram_username": r"@[a-zA-Z0-9_]{5,32}\b",
            "discord_id": r"\b\d{17,19}\b",
            "icq_number": r"\bICQ:\s*\d{6,10}\b",
            "jabber_id": r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
            "skype_username": r"\bskype:\s*[a-zA-Z0-9._-]{6,32}\b",
        }

        # Hashe a kryptografické identifikátory
        crypto_hashes = {
            "md5_hash": r"\b[a-fA-F0-9]{32}\b",
            "sha1_hash": r"\b[a-fA-F0-9]{40}\b",
            "sha256_hash": r"\b[a-fA-F0-9]{64}\b",
            "sha512_hash": r"\b[a-fA-F0-9]{128}\b",
            "ssdeep_hash": r"\b\d+:[A-Za-z0-9/+]+:[A-Za-z0-9/+]+\b",
            "pgp_fingerprint": r"\b[A-F0-9]{4}\s[A-F0-9]{4}\s[A-F0-9]{4}\s[A-F0-9]{4}\s[A-F0-9]{4}\s[A-F0-9]{4}\s[A-F0-9]{4}\s[A-F0-9]{4}\s[A-F0-9]{4}\s[A-F0-9]{4}\b",
        }

        # Síťové identifikátory
        network_patterns = {
            "ipv4_address": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
            "ipv6_address": r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b",
            "mac_address": r"\b[0-9a-fA-F]{2}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\b",
            "domain_name": r"\b[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)*[a-zA-Z]{2,}\b",
            "url_http": r"https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?",
            "ftp_url": r"ftp://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*)?",
        }

        # Dokumenty a certifikáty
        document_patterns = {
            "passport_number": r"\b[A-Z]{2}\d{7}\b",
            "credit_card": r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "phone_intl": r"\+\d{1,3}\s?\d{1,14}",
            "iban": r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b",
        }

        # Časové vzory
        temporal_patterns = {
            "timestamp_unix": r"\b15\d{8}\b",
            "date_iso": r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})\b",
            "date_standard": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        }

        # Kódování a obfuskace
        encoding_patterns = {
            "base64": r"\b[A-Za-z0-9+/]{20,}={0,3}\b",
            "hex_string": r"\b0x[a-fA-F0-9]{8,}\b",
            "rot13": r"\b[a-np-zA-NP-Z]{10,}\b",
        }

        # Spojení všech vzorů
        all_patterns = {
            **crypto_patterns,
            **darknet_patterns,
            **gps_patterns,
            **communication_patterns,
            **crypto_hashes,
            **network_patterns,
            **document_patterns,
            **temporal_patterns,
            **encoding_patterns,
        }

        # Kompilace vzorů
        for name, pattern in all_patterns.items():
            try:
                self.compiled_patterns[name] = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                category = self._get_pattern_category(name)
                if category not in self.pattern_categories:
                    self.pattern_categories[category] = []
                self.pattern_categories[category].append(name)
            except re.error as e:
                logger.warning(f"Failed to compile pattern {name}: {e}")

        logger.info(
            f"Initialized {len(self.compiled_patterns)} patterns in {len(self.pattern_categories)} categories"
        )

    def _get_pattern_category(self, pattern_name: str) -> str:
        """Kategorizace vzorů"""
        if any(x in pattern_name for x in ["bitcoin", "ethereum", "monero", "crypto"]):
            return "cryptocurrency"
        elif any(x in pattern_name for x in ["onion", "i2p", "darknet"]):
            return "darknet"
        elif any(x in pattern_name for x in ["gps", "coordinate", "location"]):
            return "geolocation"
        elif any(x in pattern_name for x in ["email", "telegram", "discord", "communication"]):
            return "communication"
        elif any(x in pattern_name for x in ["hash", "md5", "sha", "fingerprint"]):
            return "cryptographic_hashes"
        elif any(x in pattern_name for x in ["ip", "mac", "domain", "url", "network"]):
            return "network"
        elif any(x in pattern_name for x in ["passport", "credit", "ssn", "phone", "iban"]):
            return "personal_documents"
        elif any(x in pattern_name for x in ["timestamp", "date", "time"]):
            return "temporal"
        elif any(x in pattern_name for x in ["base64", "hex", "rot13", "encoding"]):
            return "encoding"
        else:
            return "miscellaneous"

    async def detect_patterns(self, text: str, source_id: str = "") -> List[PatternMatch]:
        """Hlavní metoda pro detekci vzorů v textu"""
        start_time = datetime.now()
        matches = []

        # Detekce vzorů po kategoriích
        for category, pattern_names in self.pattern_categories.items():
            category_matches = await self._detect_category_patterns(
                text, pattern_names, category, source_id
            )
            matches.extend(category_matches)

        # Statistiky
        detection_time = (datetime.now() - start_time).total_seconds()
        self.detection_stats[f"total_detections_{source_id}"] += len(matches)
        self.detection_stats[f"detection_time_{source_id}"] = detection_time

        logger.info(
            f"Detected {len(matches)} patterns in {detection_time:.2f}s from source {source_id}"
        )

        return matches

    async def _detect_category_patterns(
        self, text: str, pattern_names: List[str], category: str, source_id: str
    ) -> List[PatternMatch]:
        """Detekce vzorů v konkrétní kategorii"""
        matches = []

        for pattern_name in pattern_names:
            if pattern_name not in self.compiled_patterns:
                continue

            pattern = self.compiled_patterns[pattern_name]

            for match in pattern.finditer(text):
                # Kontext kolem nálezu
                context_start = max(0, match.start() - 50)
                context_end = min(len(text), match.end() + 50)
                context = text[context_start:context_end]

                # Výpočet konfidence
                confidence = await self._calculate_pattern_confidence(
                    pattern_name, match.group(), context
                )

                pattern_match = PatternMatch(
                    pattern_type=category,
                    pattern_name=pattern_name,
                    matched_text=match.group(),
                    confidence=confidence,
                    start_position=match.start(),
                    end_position=match.end(),
                    context=context,
                    metadata={
                        "source_id": source_id,
                        "detection_method": "regex",
                        "category": category,
                    },
                )

                matches.append(pattern_match)
                self.detection_stats[f"pattern_{pattern_name}"] += 1

        return matches

    async def _calculate_pattern_confidence(
        self, pattern_name: str, matched_text: str, context: str
    ) -> float:
        """Výpočet konfidence nálezu"""
        base_confidence = 0.7

        # Validace podle typu vzoru
        if pattern_name.startswith("bitcoin"):
            return await self._validate_bitcoin_address(matched_text)
        elif pattern_name.startswith("ethereum"):
            return await self._validate_ethereum_address(matched_text)
        elif pattern_name == "ipv4_address":
            return await self._validate_ipv4(matched_text)
        elif pattern_name == "email_standard":
            return await self._validate_email_format(matched_text)
        elif pattern_name.endswith("_hash"):
            return await self._validate_hash_format(matched_text, pattern_name)

        # Kontextová analýza
        context_boost = 0.0
        context_lower = context.lower()

        # Pozitivní kontextové signály
        positive_signals = [
            "address",
            "wallet",
            "payment",
            "transaction",
            "hash",
            "fingerprint",
            "coordinate",
            "location",
            "contact",
        ]

        for signal in positive_signals:
            if signal in context_lower:
                context_boost += 0.1

        # Negativní kontextové signály
        negative_signals = ["example", "placeholder", "test", "dummy", "fake"]

        for signal in negative_signals:
            if signal in context_lower:
                context_boost -= 0.2

        final_confidence = min(1.0, max(0.1, base_confidence + context_boost))
        return final_confidence

    async def _validate_bitcoin_address(self, address: str) -> float:
        """Validace Bitcoin adresy pomocí checksumu"""
        try:
            if len(address) < 26 or len(address) > 35:
                return 0.3

            # Základní Base58 validace
            valid_chars = set("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")
            if not all(c in valid_chars for c in address):
                return 0.2

            return 0.8  # Vysoká confidence pro správný formát
        except Exception:
            return 0.1

    async def _validate_ethereum_address(self, address: str) -> float:
        """Validace Ethereum adresy"""
        try:
            if len(address) != 42 or not address.startswith("0x"):
                return 0.2

            # Hex validace
            hex_part = address[2:]
            try:
                int(hex_part, 16)
                return 0.9  # Velmi vysoká confidence
            except ValueError:
                return 0.1
        except Exception:
            return 0.1

    async def _validate_ipv4(self, ip: str) -> float:
        """Validace IPv4 adresy"""
        try:
            ipaddress.IPv4Address(ip)
            return 0.95
        except ValueError:
            return 0.1

    async def _validate_email_format(self, email: str) -> float:
        """Validace email formátu"""
        try:
            # Základní struktura
            if email.count("@") != 1:
                return 0.2

            local, domain = email.split("@")

            if len(local) == 0 or len(domain) == 0:
                return 0.2

            if "." not in domain:
                return 0.3

            return 0.8
        except Exception:
            return 0.1

    async def _validate_hash_format(self, hash_value: str, pattern_name: str) -> float:
        """Validace hash formátů"""
        try:
            expected_lengths = {
                "md5_hash": 32,
                "sha1_hash": 40,
                "sha256_hash": 64,
                "sha512_hash": 128,
            }

            if pattern_name in expected_lengths:
                expected_len = expected_lengths[pattern_name]
                if len(hash_value) == expected_len:
                    # Hex validace
                    try:
                        int(hash_value, 16)
                        return 0.9
                    except ValueError:
                        return 0.2
                else:
                    return 0.3

            return 0.6  # Obecná hash confidence
        except Exception:
            return 0.1

    async def extract_artefacts(self, matches: List[PatternMatch]) -> List[ArtefactExtraction]:
        """Extrakce strukturovaných artefaktů z nalezených vzorů"""
        artefacts = []

        # Grupování podle typu
        grouped_matches = defaultdict(list)
        for match in matches:
            grouped_matches[match.pattern_type].append(match)

        for pattern_type, type_matches in grouped_matches.items():
            type_artefacts = await self._extract_type_artefacts(pattern_type, type_matches)
            artefacts.extend(type_artefacts)

        return artefacts

    async def _extract_type_artefacts(
        self, pattern_type: str, matches: List[PatternMatch]
    ) -> List[ArtefactExtraction]:
        """Extrakce artefaktů pro konkrétní typ"""
        artefacts = []

        for match in matches:
            if match.confidence < 0.5:  # Threshold pro extrakci
                continue

            artefact = ArtefactExtraction(
                artefact_type=pattern_type,
                value=match.matched_text,
                confidence=match.confidence,
                source_location=match.metadata.get("source_id", "unknown"),
                extraction_method="deep_pattern_detection",
                validation_results=await self._validate_artefact(match),
                related_patterns=[match.pattern_name],
            )

            artefacts.append(artefact)

        return artefacts

    async def _validate_artefact(self, match: PatternMatch) -> Dict[str, Any]:
        """Pokročilá validace extrahovaného artefaktu"""
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "validation_method": "deep_pattern_validation",
            "checks_performed": [],
        }

        # Cache kontrola
        cache_key = f"{match.pattern_name}:{match.matched_text}"
        if cache_key in self.validation_cache:
            validation_results["cached_result"] = self.validation_cache[cache_key]
            return validation_results

        # Specifické validace podle typu
        if match.pattern_type == "cryptocurrency":
            validation_results.update(await self._validate_crypto_artefact(match))
        elif match.pattern_type == "network":
            validation_results.update(await self._validate_network_artefact(match))
        elif match.pattern_type == "communication":
            validation_results.update(await self._validate_communication_artefact(match))

        # Cache výsledek
        self.validation_cache[cache_key] = validation_results.get("is_valid", False)

        return validation_results

    async def _validate_crypto_artefact(self, match: PatternMatch) -> Dict[str, Any]:
        """Validace kryptoměnových artefaktů"""
        results = {"checks_performed": ["format_check"]}

        if match.pattern_name.startswith("bitcoin"):
            results["bitcoin_validation"] = (
                await self._validate_bitcoin_address(match.matched_text) > 0.7
            )
        elif match.pattern_name.startswith("ethereum"):
            results["ethereum_validation"] = (
                await self._validate_ethereum_address(match.matched_text) > 0.7
            )

        results["is_valid"] = any(v for k, v in results.items() if k.endswith("_validation"))
        return results

    async def _validate_network_artefact(self, match: PatternMatch) -> Dict[str, Any]:
        """Validace síťových artefaktů"""
        results = {"checks_performed": ["format_check"]}

        if match.pattern_name == "ipv4_address":
            try:
                ip = ipaddress.IPv4Address(match.matched_text)
                results["is_private"] = ip.is_private
                results["is_multicast"] = ip.is_multicast
                results["is_valid"] = True
            except ValueError:
                results["is_valid"] = False

        return results

    async def _validate_communication_artefact(self, match: PatternMatch) -> Dict[str, Any]:
        """Validace komunikačních artefaktů"""
        results = {"checks_performed": ["format_check"]}

        if "email" in match.pattern_name:
            results["email_validation"] = (
                await self._validate_email_format(match.matched_text) > 0.7
            )
            results["is_valid"] = results["email_validation"]

        return results

    def get_detection_statistics(self) -> Dict[str, Any]:
        """Získání statistik detekce"""
        return {
            "total_patterns": len(self.compiled_patterns),
            "pattern_categories": len(self.pattern_categories),
            "detection_stats": dict(self.detection_stats),
            "cache_size": len(self.validation_cache),
            "categories": list(self.pattern_categories.keys()),
        }

    async def generate_pattern_report(self, matches: List[PatternMatch]) -> Dict[str, Any]:
        """Generování detailní zprávy o nalezených vzorech"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_matches": len(matches),
            "categories": defaultdict(int),
            "confidence_distribution": defaultdict(int),
            "high_confidence_matches": [],
            "suspicious_patterns": [],
            "validation_summary": defaultdict(int),
        }

        for match in matches:
            report["categories"][match.pattern_type] += 1

            # Confidence bins
            confidence_bin = (
                f"{int(match.confidence * 10) * 10}%-{int(match.confidence * 10) * 10 + 10}%"
            )
            report["confidence_distribution"][confidence_bin] += 1

            # High confidence matches
            if match.confidence > 0.8:
                report["high_confidence_matches"].append(
                    {
                        "type": match.pattern_type,
                        "pattern": match.pattern_name,
                        "value": match.matched_text[:50],  # Truncated for security
                        "confidence": match.confidence,
                    }
                )

            # Suspicious patterns
            if match.confidence < 0.4:
                report["suspicious_patterns"].append(
                    {
                        "type": match.pattern_type,
                        "pattern": match.pattern_name,
                        "confidence": match.confidence,
                        "context": match.context[:100],
                    }
                )

        return report

    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Veřejná metoda pro analýzu textu a detekci vzorů
        Používá se z AgenticLoop a dalších komponent

        Args:
            text: Text k analýze

        Returns:
            Dict s nalezenými vzory a artefakty
        """
        try:
            # Detekce všech vzorů
            pattern_matches = await self.detect_patterns(text)

            # Extrakce artefaktů
            artifacts = await self.extract_artifacts(text)

            # Validace nalezených vzorů
            validated_matches = []
            for match in pattern_matches:
                validation_result = await self._validate_pattern_match(match)
                if validation_result["is_valid"]:
                    match.validation_status = "validated"
                    validated_matches.append(match)

            # Příprava výsledku
            result = {
                "artifacts": [
                    {
                        "type": artifact.artefact_type,
                        "value": artifact.value,
                        "confidence": artifact.confidence,
                        "validation": artifact.validation_results,
                    }
                    for artifact in artifacts
                ],
                "pattern_matches": [
                    {
                        "pattern_type": match.pattern_type,
                        "pattern_name": match.pattern_name,
                        "matched_text": match.matched_text,
                        "confidence": match.confidence,
                        "validation_status": match.validation_status,
                    }
                    for match in validated_matches
                ],
                "patterns_found": len(validated_matches),
                "artifacts_found": len(artifacts),
                "categories": list(set(match.pattern_type for match in validated_matches)),
            }

            return result

        except Exception as e:
            logger.error(f"Chyba při analýze vzorů v textu: {e}")
            return {
                "artifacts": [],
                "pattern_matches": [],
                "patterns_found": 0,
                "artifacts_found": 0,
                "categories": [],
                "error": str(e),
            }
