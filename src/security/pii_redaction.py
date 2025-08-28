#!/usr/bin/env python3
"""
PII/GDPR Redaction Engine
Advanced personal data detection and redaction for GDPR compliance

Author: Senior Python/MLOps Agent
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
import json
import logging
from pathlib import Path
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class PIIType(Enum):
    """Types of personally identifiable information"""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IBAN = "iban"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    NATIONAL_ID = "national_id"
    IP_ADDRESS = "ip_address"
    MAC_ADDRESS = "mac_address"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    BIOMETRIC = "biometric"
    FINANCIAL_INFO = "financial_info"


class RedactionMode(Enum):
    """Redaction modes for different compliance levels"""
    MASK = "mask"           # Replace with asterisks
    HASH = "hash"           # Replace with hash
    REMOVE = "remove"       # Complete removal
    PLACEHOLDER = "placeholder"  # Replace with typed placeholder
    ANONYMIZE = "anonymize" # Replace with anonymous equivalent


@dataclass
class PIIMatch:
    """Detected PII match"""
    pii_type: PIIType
    start_pos: int
    end_pos: int
    original_text: str
    confidence: float
    redacted_text: str
    context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RedactionResult:
    """Result of redaction operation"""
    original_text: str
    redacted_text: str
    matches: List[PIIMatch]
    redaction_timestamp: datetime
    redaction_method: str
    compliance_level: str
    hash_salt: Optional[str] = None


class PIIDetector:
    """Advanced PII detection with multiple patterns and heuristics"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.detection_config = config.get("pii_detection", {})

        # Language-specific configurations
        self.languages = self.detection_config.get("languages", ["en", "cs", "de", "fr"])
        self.current_language = "en"

        # Detection patterns by language and type
        self.patterns = self._initialize_patterns()

        # Named entity recognition (mock - in production use spaCy/transformers)
        self.ner_enabled = self.detection_config.get("ner_enabled", False)

        logger.info(f"PII detector initialized for languages: {self.languages}")

    def _initialize_patterns(self) -> Dict[str, Dict[PIIType, List[str]]]:
        """Initialize PII detection patterns by language"""

        patterns = {
            "en": {
                PIIType.EMAIL: [
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                ],
                PIIType.PHONE: [
                    r'\b\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',  # US
                    r'\b\+?[0-9]{1,4}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}\b'  # International
                ],
                PIIType.SSN: [
                    r'\b\d{3}-\d{2}-\d{4}\b',
                    r'\b\d{9}\b'
                ],
                PIIType.CREDIT_CARD: [
                    r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'
                ],
                PIIType.IP_ADDRESS: [
                    r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
                    r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'  # IPv6
                ],
                PIIType.MAC_ADDRESS: [
                    r'\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b'
                ],
                PIIType.DATE_OF_BIRTH: [
                    r'\b(?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01])[-/](?:19|20)\d{2}\b',  # MM/DD/YYYY
                    r'\b(?:0[1-9]|[12]\d|3[01])[-/](?:0[1-9]|1[0-2])[-/](?:19|20)\d{2}\b'   # DD/MM/YYYY
                ]
            },
            "cs": {  # Czech patterns
                PIIType.EMAIL: [
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                ],
                PIIType.PHONE: [
                    r'\b\+?420[-.\s]?[0-9]{3}[-.\s]?[0-9]{3}[-.\s]?[0-9]{3}\b',  # Czech
                    r'\b[0-9]{3}[-.\s]?[0-9]{3}[-.\s]?[0-9]{3}\b'
                ],
                PIIType.NATIONAL_ID: [
                    r'\b[0-9]{6}/[0-9]{3,4}\b'  # Czech birth number
                ],
                PIIType.DATE_OF_BIRTH: [
                    r'\b(?:0[1-9]|[12]\d|3[01])\.(?:0[1-9]|1[0-2])\.(?:19|20)\d{2}\b'  # DD.MM.YYYY
                ]
            }
        }

        # Add IBAN patterns for EU countries
        iban_pattern = r'\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}([A-Z0-9]?){0,16}\b'
        for lang in patterns:
            patterns[lang][PIIType.IBAN] = [iban_pattern]

        return patterns

    def detect_pii(self, text: str, language: str = "en") -> List[PIIMatch]:
        """Detect all PII in text"""

        self.current_language = language if language in self.patterns else "en"
        matches = []

        # Pattern-based detection
        for pii_type, pattern_list in self.patterns[self.current_language].items():
            for pattern in pattern_list:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    confidence = self._calculate_confidence(match.group(), pii_type)

                    if confidence >= 0.7:  # Confidence threshold
                        pii_match = PIIMatch(
                            pii_type=pii_type,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            original_text=match.group(),
                            confidence=confidence,
                            redacted_text="",  # Will be filled by redactor
                            context=self._extract_context(text, match.start(), match.end())
                        )
                        matches.append(pii_match)

        # NER-based detection for names and addresses
        if self.ner_enabled:
            ner_matches = self._detect_with_ner(text)
            matches.extend(ner_matches)

        # Remove overlapping matches (keep highest confidence)
        matches = self._remove_overlaps(matches)

        logger.debug(f"Detected {len(matches)} PII instances in text")
        return matches

    def _calculate_confidence(self, text: str, pii_type: PIIType) -> float:
        """Calculate confidence score for PII detection"""

        confidence = 0.8  # Base confidence for pattern match

        # Adjust based on PII type and text characteristics
        if pii_type == PIIType.EMAIL:
            if "@" in text and "." in text.split("@")[-1]:
                confidence = 0.95
        elif pii_type == PIIType.PHONE:
            # Check for phone number formatting
            if any(sep in text for sep in ["-", ".", " ", "(", ")"]):
                confidence = 0.9
        elif pii_type == PIIType.CREDIT_CARD:
            # Luhn algorithm check for credit cards
            if self._luhn_check(re.sub(r'\D', '', text)):
                confidence = 0.95
        elif pii_type == PIIType.IP_ADDRESS:
            # Enhanced IP validation
            parts = text.split('.')
            if len(parts) == 4 and all(0 <= int(part) <= 255 for part in parts if part.isdigit()):
                confidence = 0.95

        return confidence

    def _luhn_check(self, card_number: str) -> bool:
        """Luhn algorithm for credit card validation"""
        def digits_of(n):
            return [int(d) for d in str(n)]

        digits = digits_of(card_number)
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        checksum = sum(odd_digits)
        for d in even_digits:
            checksum += sum(digits_of(d*2))
        return checksum % 10 == 0

    def _extract_context(self, text: str, start: int, end: int, context_length: int = 50) -> str:
        """Extract context around PII match"""
        context_start = max(0, start - context_length)
        context_end = min(len(text), end + context_length)
        return text[context_start:context_end]

    def _detect_with_ner(self, text: str) -> List[PIIMatch]:
        """Mock NER-based detection (in production use spaCy/transformers)"""
        # Mock implementation - in production would use actual NER
        matches = []

        # Simple name detection patterns
        name_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
            r'\b[A-Z][a-z]+, [A-Z][a-z]+\b'  # Last, First
        ]

        for pattern in name_patterns:
            for match in re.finditer(pattern, text):
                pii_match = PIIMatch(
                    pii_type=PIIType.NAME,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    original_text=match.group(),
                    confidence=0.6,  # Lower confidence for heuristic detection
                    redacted_text="",
                    context=self._extract_context(text, match.start(), match.end())
                )
                matches.append(pii_match)

        return matches

    def _remove_overlaps(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """Remove overlapping matches, keeping highest confidence"""
        if not matches:
            return matches

        # Sort by position
        sorted_matches = sorted(matches, key=lambda x: x.start_pos)

        result = []
        for match in sorted_matches:
            # Check for overlap with existing matches
            overlap = False
            for existing in result:
                if (match.start_pos < existing.end_pos and
                    match.end_pos > existing.start_pos):
                    # Overlap detected - keep higher confidence
                    if match.confidence > existing.confidence:
                        result.remove(existing)
                        result.append(match)
                    overlap = True
                    break

            if not overlap:
                result.append(match)

        return result


class PIIRedactor:
    """
    FÁZE 7: Enhanced PII Redaction Engine
    Advanced redaction with multiple modes and compliance levels
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redaction_config = config.get("pii_redaction", {})

        # Default redaction modes by PII type
        self.default_modes = {
            PIIType.EMAIL: RedactionMode.MASK,
            PIIType.PHONE: RedactionMode.MASK,
            PIIType.SSN: RedactionMode.HASH,
            PIIType.CREDIT_CARD: RedactionMode.HASH,
            PIIType.IBAN: RedactionMode.HASH,
            PIIType.IP_ADDRESS: RedactionMode.PLACEHOLDER,
            PIIType.NAME: RedactionMode.PLACEHOLDER,
            PIIType.DATE_OF_BIRTH: RedactionMode.ANONYMIZE
        }

        # Salt for hashing
        self.hash_salt = self.redaction_config.get("hash_salt", "deepresearch_pii_salt")

        # Compliance levels
        self.compliance_level = self.redaction_config.get("compliance_level", "strict")

        # Initialize detector
        self.detector = PIIDetector(config)

        logger.info(f"PII redactor initialized with compliance level: {self.compliance_level}")

    def redact_text(
        self,
        text: str,
        language: str = "en",
        custom_modes: Optional[Dict[PIIType, RedactionMode]] = None
    ) -> RedactionResult:
        """
        Main redaction function with comprehensive PII handling
        """
        # Detect PII
        matches = self.detector.detect_pii(text, language)

        # Apply redaction modes
        redaction_modes = {**self.default_modes}
        if custom_modes:
            redaction_modes.update(custom_modes)

        # Perform redaction
        redacted_text = text
        processed_matches = []

        # Sort matches by position (reverse order to maintain positions)
        sorted_matches = sorted(matches, key=lambda x: x.start_pos, reverse=True)

        for match in sorted_matches:
            mode = redaction_modes.get(match.pii_type, RedactionMode.MASK)
            redacted_value = self._apply_redaction(match.original_text, match.pii_type, mode)

            # Replace in text
            redacted_text = (
                redacted_text[:match.start_pos] +
                redacted_value +
                redacted_text[match.end_pos:]
            )

            # Update match with redacted text
            match.redacted_text = redacted_value
            processed_matches.append(match)

        return RedactionResult(
            original_text=text,
            redacted_text=redacted_text,
            matches=list(reversed(processed_matches)),  # Restore original order
            redaction_timestamp=datetime.now(timezone.utc),
            redaction_method=f"advanced_pii_redaction_v1.0",
            compliance_level=self.compliance_level,
            hash_salt=self.hash_salt if any(m.pii_type for m in processed_matches
                                           if redaction_modes.get(m.pii_type) == RedactionMode.HASH) else None
        )

    def _apply_redaction(self, text: str, pii_type: PIIType, mode: RedactionMode) -> str:
        """Apply specific redaction mode to text"""

        if mode == RedactionMode.MASK:
            return self._mask_text(text, pii_type)
        elif mode == RedactionMode.HASH:
            return self._hash_text(text)
        elif mode == RedactionMode.REMOVE:
            return ""
        elif mode == RedactionMode.PLACEHOLDER:
            return self._get_placeholder(pii_type)
        elif mode == RedactionMode.ANONYMIZE:
            return self._anonymize_text(text, pii_type)
        else:
            return self._mask_text(text, pii_type)

    def _mask_text(self, text: str, pii_type: PIIType) -> str:
        """Mask text with asterisks, preserving some structure"""

        if pii_type == PIIType.EMAIL:
            # Mask: user@domain.com -> u***@domain.com
            if "@" in text:
                local, domain = text.split("@", 1)
                masked_local = local[0] + "*" * (len(local) - 1) if len(local) > 1 else "*"
                return f"{masked_local}@{domain}"

        elif pii_type == PIIType.PHONE:
            # Mask: +1-555-123-4567 -> +1-***-***-4567
            digits = re.findall(r'\d', text)
            if len(digits) >= 4:
                masked = re.sub(r'\d', '*', text)
                # Keep last 4 digits
                for i, digit in enumerate(digits[-4:]):
                    masked = masked[::-1].replace('*', digit, 1)[::-1]
                return masked

        elif pii_type == PIIType.CREDIT_CARD:
            # Mask: 4111-1111-1111-1111 -> ****-****-****-1111
            digits = re.findall(r'\d', text)
            if len(digits) >= 4:
                masked = re.sub(r'\d', '*', text)
                # Keep last 4 digits
                for i, digit in enumerate(digits[-4:]):
                    masked = masked[::-1].replace('*', digit, 1)[::-1]
                return masked

        # Default masking
        if len(text) <= 3:
            return "*" * len(text)
        else:
            return text[0] + "*" * (len(text) - 2) + text[-1]

    def _hash_text(self, text: str) -> str:
        """Hash text with salt for irreversible anonymization"""
        salted_text = f"{text}{self.hash_salt}"
        hash_object = hashlib.sha256(salted_text.encode())
        hash_hex = hash_object.hexdigest()
        return f"[HASH:{hash_hex[:16]}]"

    def _get_placeholder(self, pii_type: PIIType) -> str:
        """Get typed placeholder for PII type"""
        placeholders = {
            PIIType.EMAIL: "[EMAIL_ADDRESS]",
            PIIType.PHONE: "[PHONE_NUMBER]",
            PIIType.SSN: "[SSN]",
            PIIType.CREDIT_CARD: "[CREDIT_CARD]",
            PIIType.IBAN: "[BANK_ACCOUNT]",
            PIIType.IP_ADDRESS: "[IP_ADDRESS]",
            PIIType.MAC_ADDRESS: "[MAC_ADDRESS]",
            PIIType.NAME: "[PERSON_NAME]",
            PIIType.ADDRESS: "[ADDRESS]",
            PIIType.DATE_OF_BIRTH: "[DATE_OF_BIRTH]",
            PIIType.NATIONAL_ID: "[NATIONAL_ID]",
            PIIType.PASSPORT: "[PASSPORT_NUMBER]",
            PIIType.DRIVER_LICENSE: "[DRIVER_LICENSE]",
            PIIType.BIOMETRIC: "[BIOMETRIC_DATA]",
            PIIType.FINANCIAL_INFO: "[FINANCIAL_INFO]"
        }
        return placeholders.get(pii_type, "[REDACTED]")

    def _anonymize_text(self, text: str, pii_type: PIIType) -> str:
        """Anonymize with realistic replacements"""

        if pii_type == PIIType.DATE_OF_BIRTH:
            # Replace with generic age range
            return "[AGE_25_35]"
        elif pii_type == PIIType.NAME:
            # Replace with generic names
            generic_names = ["John Doe", "Jane Smith", "Person A", "Individual B"]
            hash_index = hash(text) % len(generic_names)
            return generic_names[hash_index]
        elif pii_type == PIIType.ADDRESS:
            return "[City, Country]"
        else:
            return self._get_placeholder(pii_type)

    def redact_json(self, data: Dict[str, Any], language: str = "en") -> Tuple[Dict[str, Any], List[PIIMatch]]:
        """Redact PII from JSON data structure"""
        all_matches = []

        def redact_recursive(obj):
            if isinstance(obj, dict):
                return {k: redact_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [redact_recursive(item) for item in obj]
            elif isinstance(obj, str):
                result = self.redact_text(obj, language)
                all_matches.extend(result.matches)
                return result.redacted_text
            else:
                return obj

        redacted_data = redact_recursive(data)
        return redacted_data, all_matches

    def get_redaction_stats(self, result: RedactionResult) -> Dict[str, Any]:
        """Get statistics about redaction operation"""

        pii_counts = {}
        for match in result.matches:
            pii_type = match.pii_type.value
            pii_counts[pii_type] = pii_counts.get(pii_type, 0) + 1

        return {
            "total_pii_instances": len(result.matches),
            "pii_types_detected": list(pii_counts.keys()),
            "pii_counts": pii_counts,
            "redaction_rate": len(result.matches) / max(len(result.original_text.split()), 1),
            "compliance_level": result.compliance_level,
            "redaction_timestamp": result.redaction_timestamp.isoformat(),
            "average_confidence": sum(m.confidence for m in result.matches) / max(len(result.matches), 1)
        }


class PIIComplianceLogger:
    """
    FÁZE 7: Compliance logging for PII redaction operations
    """

    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file or Path("pii_compliance.log")
        self.redactor = None

        # Setup compliance logger
        self.compliance_logger = logging.getLogger("pii_compliance")
        self.compliance_logger.setLevel(logging.INFO)

        if not self.compliance_logger.handlers:
            handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.compliance_logger.addHandler(handler)

    def log_redaction_operation(
        self,
        result: RedactionResult,
        context: str = "",
        user_id: Optional[str] = None
    ) -> None:
        """Log PII redaction operation for compliance audit"""

        log_entry = {
            "timestamp": result.redaction_timestamp.isoformat(),
            "operation": "pii_redaction",
            "pii_instances_found": len(result.matches),
            "pii_types": list(set(m.pii_type.value for m in result.matches)),
            "compliance_level": result.compliance_level,
            "redaction_method": result.redaction_method,
            "context": context,
            "user_id": user_id,
            "text_length": len(result.original_text),
            "redacted_length": len(result.redacted_text)
        }

        self.compliance_logger.info(json.dumps(log_entry))

    def log_pii_access_attempt(
        self,
        user_id: str,
        pii_types: List[PIIType],
        allowed: bool,
        reason: str = ""
    ) -> None:
        """Log PII data access attempts"""

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": "pii_access_attempt",
            "user_id": user_id,
            "pii_types": [pii_type.value for pii_type in pii_types],
            "access_allowed": allowed,
            "reason": reason
        }

        self.compliance_logger.info(json.dumps(log_entry))


# Factory functions for easy integration
def create_pii_redactor(config: Optional[Dict[str, Any]] = None) -> PIIRedactor:
    """Factory function to create PII redactor"""
    default_config = {
        "pii_redaction": {
            "compliance_level": "strict",
            "hash_salt": "deepresearch_default_salt"
        },
        "pii_detection": {
            "languages": ["en", "cs"],
            "ner_enabled": False
        }
    }

    if config:
        # Merge configs
        for key, value in config.items():
            if key in default_config and isinstance(value, dict):
                default_config[key].update(value)
            else:
                default_config[key] = value

    return PIIRedactor(default_config)


def redact_text_simple(text: str, language: str = "en") -> str:
    """Simple function for quick text redaction"""
    redactor = create_pii_redactor()
    result = redactor.redact_text(text, language)
    return result.redacted_text


# Demo usage
if __name__ == "__main__":
    # Demo PII redaction
    sample_text = """
    Contact John Doe at john.doe@example.com or call +1-555-123-4567.
    His SSN is 123-45-6789 and credit card 4111-1111-1111-1111.
    IP address: 192.168.1.1, DOB: 01/15/1985
    """

    redactor = create_pii_redactor({
        "pii_redaction": {"compliance_level": "strict"}
    })

    result = redactor.redact_text(sample_text)

    print("Original:", sample_text)
    print("\nRedacted:", result.redacted_text)
    print("\nMatches found:", len(result.matches))

    for match in result.matches:
        print(f"  {match.pii_type.value}: {match.original_text} -> {match.redacted_text}")

    # Statistics
    stats = redactor.get_redaction_stats(result)
    print("\nRedaction stats:", json.dumps(stats, indent=2))
