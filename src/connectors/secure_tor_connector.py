#!/usr/bin/env python3
"""
Security Logging and Content Filtering System
Legal compliance logging with why-blocked artifacts for Tor/Ahmia sources

Author: Senior IT Specialist
"""

import logging
import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum


class BlockReason(Enum):
    """Reasons for content blocking"""
    ILLEGAL_CONTENT = "illegal_content"
    HARMFUL_CONTENT = "harmful_content"
    PRIVACY_VIOLATION = "privacy_violation"
    COPYRIGHT_VIOLATION = "copyright_violation"
    SPAM_CONTENT = "spam_content"
    MALWARE_DETECTED = "malware_detected"
    PHISHING_ATTEMPT = "phishing_attempt"
    GEOGRAPHIC_RESTRICTION = "geographic_restriction"
    AGE_RESTRICTION = "age_restriction"
    TERMS_VIOLATION = "terms_violation"
    LEGAL_COMPLIANCE = "legal_compliance"


@dataclass
class SecurityEvent:
    """Security event record"""
    event_id: str
    timestamp: datetime
    event_type: str  # "content_blocked", "source_filtered", "access_denied"
    source_url: str
    content_hash: Optional[str] = None
    block_reason: Optional[BlockReason] = None
    jurisdiction: str = "unknown"

    # Metadata (no sensitive content)
    source_type: str = "unknown"  # "tor", "clearnet", "onion"
    content_type: str = "unknown"  # "text", "image", "document"
    risk_score: float = 0.0

    # Legal/compliance info
    applicable_laws: List[str] = field(default_factory=list)
    regulatory_framework: str = "unknown"

    # Investigation support (hashed/anonymized)
    session_hash: Optional[str] = None
    user_agent_hash: Optional[str] = None


@dataclass
class FilterRule:
    """Content filtering rule"""
    rule_id: str
    rule_name: str
    pattern: str
    rule_type: str  # "keyword", "regex", "domain", "hash"
    severity: str  # "low", "medium", "high", "critical"
    block_reason: BlockReason
    jurisdiction: str = "global"
    enabled: bool = True

    # Legal basis
    legal_reference: Optional[str] = None
    regulatory_source: Optional[str] = None


class ContentHasher:
    """Secure content hashing for evidence purposes"""

    def __init__(self, salt: str = "research_tool_salt_2024"):
        self.salt = salt
        self.logger = logging.getLogger(__name__)

    def hash_content(self, content: str) -> str:
        """Create secure hash of content for evidence"""
        salted_content = f"{self.salt}:{content}"
        return hashlib.sha256(salted_content.encode()).hexdigest()

    def hash_metadata(self, metadata: Dict[str, Any]) -> str:
        """Hash metadata for investigation support"""
        metadata_str = json.dumps(metadata, sort_keys=True)
        salted_metadata = f"{self.salt}:{metadata_str}"
        return hashlib.sha256(salted_metadata.encode()).hexdigest()[:16]


class LegalComplianceFilter:
    """Legal compliance content filtering"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.hasher = ContentHasher()

        # Jurisdiction configuration
        self.jurisdiction = config.get("jurisdiction", "EU")
        self.strict_mode = config.get("strict_mode", True)

        # Load filtering rules
        self.rules = self._load_filtering_rules()

        # Blocked content tracking
        self.blocked_hashes = set()

        # Legal frameworks
        self.legal_frameworks = {
            "EU": ["GDPR", "DSA", "DMA", "NIS2"],
            "US": ["DMCA", "COPPA", "CAN-SPAM"],
            "UK": ["DPA 2018", "Online Safety Act"],
            "Global": ["UN Declaration of Human Rights"]
        }

    def _load_filtering_rules(self) -> List[FilterRule]:
        """Load content filtering rules"""
        rules = []

        # High-risk content patterns
        high_risk_rules = [
            FilterRule(
                rule_id="illegal_001",
                rule_name="Explicit Illegal Activity",
                pattern=r"\b(child\s*porn|cp\s*links|illegal\s*drugs\s*sale)\b",
                rule_type="regex",
                severity="critical",
                block_reason=BlockReason.ILLEGAL_CONTENT,
                legal_reference="Criminal Code - Child Protection Act",
                regulatory_source="National Law Enforcement"
            ),
            FilterRule(
                rule_id="harmful_001",
                rule_name="Harmful Instructions",
                pattern=r"\b(bomb\s*making|suicide\s*methods|self\s*harm)\b",
                rule_type="regex",
                severity="high",
                block_reason=BlockReason.HARMFUL_CONTENT,
                legal_reference="Public Safety Act",
                regulatory_source="Public Health Authority"
            ),
            FilterRule(
                rule_id="privacy_001",
                rule_name="Personal Data Exposure",
                pattern=r"\b(ssn|social\s*security|credit\s*card\s*\d+)\b",
                rule_type="regex",
                severity="high",
                block_reason=BlockReason.PRIVACY_VIOLATION,
                legal_reference="GDPR Article 6",
                regulatory_source="Data Protection Authority"
            )
        ]

        rules.extend(high_risk_rules)

        # Domain-based rules for known problematic sources
        domain_rules = [
            FilterRule(
                rule_id="domain_001",
                rule_name="Known Malware Domains",
                pattern="malware-site.onion|phishing-site.onion",
                rule_type="domain",
                severity="critical",
                block_reason=BlockReason.MALWARE_DETECTED,
                legal_reference="Cybersecurity Act",
                regulatory_source="Cybersecurity Agency"
            )
        ]

        rules.extend(domain_rules)

        self.logger.info(f"Loaded {len(rules)} filtering rules for jurisdiction: {self.jurisdiction}")
        return rules

    def check_content(self, content: str, source_url: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Check content against filtering rules"""
        metadata = metadata or {}

        # Create content hash for tracking
        content_hash = self.hasher.hash_content(content)

        # Check if already blocked
        if content_hash in self.blocked_hashes:
            return {
                "blocked": True,
                "reason": "previously_blocked",
                "content_hash": content_hash
            }

        # Apply filtering rules
        for rule in self.rules:
            if not rule.enabled:
                continue

            violation = self._check_rule(content, source_url, rule)
            if violation:
                # Content violates rule - block it
                self.blocked_hashes.add(content_hash)

                return {
                    "blocked": True,
                    "rule_violated": rule.rule_id,
                    "rule_name": rule.rule_name,
                    "severity": rule.severity,
                    "block_reason": rule.block_reason.value,
                    "legal_reference": rule.legal_reference,
                    "content_hash": content_hash,
                    "why_blocked": self._generate_why_blocked_explanation(rule, violation)
                }

        # Content passed all filters
        return {
            "blocked": False,
            "content_hash": content_hash,
            "rules_checked": len([r for r in self.rules if r.enabled])
        }

    def _check_rule(self, content: str, source_url: str, rule: FilterRule) -> Optional[Dict[str, Any]]:
        """Check content against specific rule"""
        try:
            if rule.rule_type == "keyword":
                if rule.pattern.lower() in content.lower():
                    return {"match_type": "keyword", "pattern": rule.pattern}

            elif rule.rule_type == "regex":
                import re
                if re.search(rule.pattern, content, re.IGNORECASE):
                    return {"match_type": "regex", "pattern": rule.pattern}

            elif rule.rule_type == "domain":
                if rule.pattern in source_url:
                    return {"match_type": "domain", "pattern": rule.pattern}

            elif rule.rule_type == "hash":
                content_hash = self.hasher.hash_content(content)
                if content_hash == rule.pattern:
                    return {"match_type": "hash", "pattern": rule.pattern}

        except Exception as e:
            self.logger.warning(f"Rule check failed for {rule.rule_id}: {e}")

        return None

    def _generate_why_blocked_explanation(self, rule: FilterRule, violation: Dict[str, Any]) -> str:
        """Generate human-readable explanation for blocking"""
        explanations = {
            BlockReason.ILLEGAL_CONTENT: "Content contains references to illegal activities",
            BlockReason.HARMFUL_CONTENT: "Content contains potentially harmful instructions or information",
            BlockReason.PRIVACY_VIOLATION: "Content contains personal or sensitive information",
            BlockReason.COPYRIGHT_VIOLATION: "Content violates copyright or intellectual property rights",
            BlockReason.MALWARE_DETECTED: "Source is known to distribute malware or malicious content",
            BlockReason.PHISHING_ATTEMPT: "Content appears to be a phishing or fraud attempt",
            BlockReason.LEGAL_COMPLIANCE: "Content blocked for legal compliance reasons"
        }

        base_explanation = explanations.get(rule.block_reason, "Content blocked by security policy")

        legal_context = ""
        if rule.legal_reference:
            legal_context = f" (Legal basis: {rule.legal_reference})"

        return f"{base_explanation}{legal_context}"


class SecurityLogger:
    """Secure logging system for blocked content and security events"""

    def __init__(self, log_dir: str = "logs/security"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup secure logger
        self.logger = logging.getLogger("security_logger")
        self.logger.setLevel(logging.INFO)

        # Create secure log handler
        log_file = self.log_dir / f"security_{datetime.now().strftime('%Y%m%d')}.log"
        handler = logging.FileHandler(log_file)

        # Secure format (no sensitive content)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S UTC'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.hasher = ContentHasher()

    def log_blocked_content(self, content_hash: str, source_url: str,
                          block_reason: BlockReason, rule_info: Dict[str, Any]):
        """Log blocked content event (no sensitive data)"""
        event = SecurityEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.now(timezone.utc),
            event_type="content_blocked",
            source_url=self._sanitize_url(source_url),
            content_hash=content_hash,
            block_reason=block_reason,
            source_type=self._detect_source_type(source_url)
        )

        log_entry = {
            "event_id": event.event_id,
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type,
            "source_url_domain": self._extract_domain(source_url),
            "content_hash": content_hash,
            "block_reason": block_reason.value,
            "rule_id": rule_info.get("rule_id"),
            "rule_name": rule_info.get("rule_name"),
            "severity": rule_info.get("severity"),
            "legal_reference": rule_info.get("legal_reference"),
            "source_type": event.source_type
        }

        self.logger.info(f"BLOCKED_CONTENT: {json.dumps(log_entry)}")

    def log_source_filtered(self, source_url: str, reason: str, metadata: Dict[str, Any] = None):
        """Log source filtering event"""
        metadata = metadata or {}

        event = SecurityEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.now(timezone.utc),
            event_type="source_filtered",
            source_url=self._sanitize_url(source_url),
            source_type=self._detect_source_type(source_url)
        )

        log_entry = {
            "event_id": event.event_id,
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type,
            "source_domain": self._extract_domain(source_url),
            "filter_reason": reason,
            "source_type": event.source_type,
            "metadata_hash": self.hasher.hash_metadata(metadata) if metadata else None
        }

        self.logger.info(f"SOURCE_FILTERED: {json.dumps(log_entry)}")

    def log_access_denied(self, source_url: str, reason: str, user_context: Dict[str, Any] = None):
        """Log access denial event"""
        user_context = user_context or {}

        event = SecurityEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.now(timezone.utc),
            event_type="access_denied",
            source_url=self._sanitize_url(source_url),
            session_hash=self.hasher.hash_metadata(user_context) if user_context else None
        )

        log_entry = {
            "event_id": event.event_id,
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type,
            "source_domain": self._extract_domain(source_url),
            "denial_reason": reason,
            "session_hash": event.session_hash
        }

        self.logger.warning(f"ACCESS_DENIED: {json.dumps(log_entry)}")

    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = hashlib.sha256(f"{timestamp}{datetime.now().microsecond}".encode()).hexdigest()[:8]
        return f"SEC_{timestamp}_{random_suffix}"

    def _sanitize_url(self, url: str) -> str:
        """Sanitize URL for logging (remove sensitive parameters)"""
        from urllib.parse import urlparse, urlunparse

        try:
            parsed = urlparse(url)
            # Remove query parameters and fragment
            sanitized = urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))
            return sanitized
        except Exception:
            return "[INVALID_URL]"

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL for logging"""
        from urllib.parse import urlparse

        try:
            return urlparse(url).netloc
        except Exception:
            return "unknown"

    def _detect_source_type(self, url: str) -> str:
        """Detect source type for classification"""
        if ".onion" in url:
            return "tor_onion"
        elif any(x in url for x in ["tor", "ahmia", "duckduckgo.onion"]):
            return "tor_related"
        elif url.startswith("https://"):
            return "clearnet_https"
        elif url.startswith("http://"):
            return "clearnet_http"
        else:
            return "unknown"


class SecureTorConnector:
    """Secure Tor/Ahmia connector with legal filtering"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Security components
        self.compliance_filter = LegalComplianceFilter(config.get("compliance", {}))
        self.security_logger = SecurityLogger(config.get("log_dir", "logs/security"))

        # Tor configuration
        self.enabled = config.get("enabled", False)
        self.ahmia_enabled = config.get("ahmia_enabled", False)
        self.strict_filtering = config.get("strict_filtering", True)

        # Legal compliance
        self.jurisdiction = config.get("jurisdiction", "EU")
        self.legal_research_only = config.get("legal_research_only", True)

        if not self.enabled:
            self.logger.info("Tor connector disabled by configuration")

    async def search_with_legal_filtering(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search Tor sources with comprehensive legal filtering"""
        if not self.enabled:
            self.logger.info("Tor search skipped - connector disabled")
            return []

        self.logger.info(f"Starting secure Tor search for: {query[:50]}...")

        # Pre-filter query for legal compliance
        query_check = self.compliance_filter.check_content(query, "query://user_input")
        if query_check["blocked"]:
            self.security_logger.log_source_filtered(
                "query://user_input",
                f"Query blocked: {query_check['why_blocked']}"
            )
            return []

        # Search results (mock implementation for security)
        raw_results = await self._secure_search(query, max_results * 2)  # Get extra for filtering

        # Filter results through compliance system
        filtered_results = []
        for result in raw_results:
            content_check = self.compliance_filter.check_content(
                result.get("content", ""),
                result.get("url", ""),
                result
            )

            if content_check["blocked"]:
                # Log blocking but don't include sensitive info
                self.security_logger.log_blocked_content(
                    content_check["content_hash"],
                    result.get("url", "unknown"),
                    BlockReason(content_check["block_reason"]),
                    {
                        "rule_id": content_check.get("rule_violated"),
                        "rule_name": content_check.get("rule_name"),
                        "severity": content_check.get("severity"),
                        "legal_reference": content_check.get("legal_reference")
                    }
                )
                continue

            # Content passed filters
            filtered_result = {
                "source_id": f"tor_filtered_{content_check['content_hash'][:16]}",
                "url": result.get("url", ""),
                "title": result.get("title", ""),
                "snippet": result.get("content", "")[:200],  # Limit snippet
                "content_hash": content_check["content_hash"],
                "security_cleared": True,
                "filter_timestamp": datetime.now(timezone.utc).isoformat()
            }

            filtered_results.append(filtered_result)

            if len(filtered_results) >= max_results:
                break

        self.logger.info(f"Tor search filtered: {len(raw_results)} → {len(filtered_results)} results")
        return filtered_results

    async def _secure_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Secure search implementation (mock for demo)"""
        # In real implementation, this would:
        # 1. Use Tor proxy for anonymity
        # 2. Search Ahmia or other legal Tor search engines
        # 3. Apply rate limiting and security measures

        # Mock results for demonstration
        mock_results = [
            {
                "url": "http://example.onion/research/topic1",
                "title": f"Research Document on {query}",
                "content": f"Academic research discussing {query} in the context of privacy and security.",
                "source_type": "academic_onion"
            },
            {
                "url": "http://library.onion/papers/topic2",
                "title": f"Historical Analysis of {query}",
                "content": f"Historical perspective on {query} with primary source materials.",
                "source_type": "library_onion"
            }
        ]

        # Simulate filtering for legal research purposes
        legal_results = []
        for result in mock_results:
            if self.legal_research_only:
                # Only include academic/research content
                if any(keyword in result["content"].lower() for keyword in ["research", "academic", "analysis", "study"]):
                    legal_results.append(result)
            else:
                legal_results.append(result)

        return legal_results[:limit]

    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report for audit purposes"""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "connector_status": {
                "enabled": self.enabled,
                "ahmia_enabled": self.ahmia_enabled,
                "strict_filtering": self.strict_filtering
            },
            "compliance_configuration": {
                "jurisdiction": self.jurisdiction,
                "legal_research_only": self.legal_research_only,
                "filtering_rules_count": len(self.compliance_filter.rules)
            },
            "security_measures": {
                "content_hashing": True,
                "secure_logging": True,
                "why_blocked_artifacts": True,
                "anonymized_tracking": True
            },
            "blocked_content_count": len(self.compliance_filter.blocked_hashes),
            "legal_frameworks": self.compliance_filter.legal_frameworks.get(self.jurisdiction, [])
        }


def create_secure_tor_connector(config: Dict[str, Any]) -> SecureTorConnector:
    """Factory function for secure Tor connector"""
    return SecureTorConnector(config)
