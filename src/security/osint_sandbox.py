#!/usr/bin/env python3
"""OSINT Security Sandbox
Secure execution environment for OSINT operations with whitelists/blacklists

Author: Senior Python/MLOps Agent
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
import json
import logging
from pathlib import Path
import re
import threading
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for OSINT operations"""

    LOW = "low"  # Minimal restrictions
    MEDIUM = "medium"  # Standard security
    HIGH = "high"  # Strict security
    PARANOID = "paranoid"  # Maximum security


class AccessResult(Enum):
    """Results of access control checks"""

    ALLOWED = "allowed"
    BLOCKED = "blocked"
    RATE_LIMITED = "rate_limited"
    QUARANTINED = "quarantined"


@dataclass
class SecurityRule:
    """Individual security rule"""

    rule_id: str
    rule_type: str  # whitelist, blacklist, rate_limit, content_filter
    pattern: str
    action: str  # allow, block, quarantine, rate_limit
    priority: int = 100
    metadata: dict[str, Any] = field(default_factory=dict)

    def matches(self, url: str, content: str = "") -> bool:
        """Check if rule matches the given URL/content"""
        if self.rule_type in ["whitelist", "blacklist"]:
            return self._matches_url_pattern(url)
        if self.rule_type == "content_filter":
            return self._matches_content_pattern(content)
        return False

    def _matches_url_pattern(self, url: str) -> bool:
        """Check URL pattern match"""
        try:
            parsed = urlparse(url.lower())
            domain = parsed.netloc

            # Support different pattern types
            if self.pattern.startswith("regex:"):
                pattern = self.pattern[6:]
                return bool(re.search(pattern, url, re.IGNORECASE))
            if self.pattern.startswith("domain:"):
                target_domain = self.pattern[7:]
                return domain == target_domain or domain.endswith(f".{target_domain}")
            if self.pattern.startswith("subdomain:"):
                target_domain = self.pattern[10:]
                return domain.endswith(f".{target_domain}")
            # Simple substring match
            return self.pattern.lower() in url.lower()
        except Exception:
            return False

    def _matches_content_pattern(self, content: str) -> bool:
        """Check content pattern match"""
        if self.pattern.startswith("regex:"):
            pattern = self.pattern[6:]
            return bool(re.search(pattern, content, re.IGNORECASE))
        return self.pattern.lower() in content.lower()


@dataclass
class AccessAttempt:
    """Record of an access attempt"""

    timestamp: datetime
    url: str
    source_ip: str
    user_agent: str
    result: AccessResult
    rule_matched: str | None = None
    content_hash: str | None = None


class RateLimiter:
    """Rate limiting for OSINT operations"""

    def __init__(self, config: dict[str, Any]):
        self.limits = config.get(
            "rate_limits",
            {
                "requests_per_minute": 60,
                "requests_per_hour": 1000,
                "requests_per_day": 10000,
                "concurrent_requests": 10,
            },
        )

        # Tracking structures
        self.request_times: deque = deque(maxlen=10000)
        self.domain_requests: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.active_requests: set[str] = set()
        self.lock = threading.Lock()

        logger.info(f"Rate limiter initialized: {self.limits}")

    def check_rate_limit(self, url: str, source_ip: str = "unknown") -> tuple[bool, str]:
        """Check if request is within rate limits"""
        with self.lock:
            now = datetime.now(UTC)
            domain = urlparse(url).netloc

            # Clean old entries
            self._cleanup_old_entries(now)

            # Check concurrent requests
            if len(self.active_requests) >= self.limits["concurrent_requests"]:
                return False, "concurrent_limit_exceeded"

            # Check global rate limits
            minute_ago = now - timedelta(minutes=1)
            hour_ago = now - timedelta(hours=1)
            day_ago = now - timedelta(days=1)

            recent_requests = [t for t in self.request_times if t > minute_ago]
            if len(recent_requests) >= self.limits["requests_per_minute"]:
                return False, "per_minute_limit_exceeded"

            recent_requests = [t for t in self.request_times if t > hour_ago]
            if len(recent_requests) >= self.limits["requests_per_hour"]:
                return False, "per_hour_limit_exceeded"

            recent_requests = [t for t in self.request_times if t > day_ago]
            if len(recent_requests) >= self.limits["requests_per_day"]:
                return False, "per_day_limit_exceeded"

            # Check domain-specific limits
            domain_recent = [t for t in self.domain_requests[domain] if t > minute_ago]
            domain_limit = self.limits.get("domain_requests_per_minute", 30)
            if len(domain_recent) >= domain_limit:
                return False, f"domain_limit_exceeded_{domain}"

            return True, "allowed"

    def record_request(self, url: str, source_ip: str = "unknown"):
        """Record a request for rate limiting"""
        with self.lock:
            now = datetime.now(UTC)
            domain = urlparse(url).netloc

            self.request_times.append(now)
            self.domain_requests[domain].append(now)
            self.active_requests.add(f"{url}_{now.timestamp()}")

    def complete_request(self, url: str, timestamp: datetime):
        """Mark request as completed"""
        with self.lock:
            request_id = f"{url}_{timestamp.timestamp()}"
            self.active_requests.discard(request_id)

    def _cleanup_old_entries(self, now: datetime):
        """Clean up old tracking entries"""
        day_ago = now - timedelta(days=1)

        # Clean global request times
        while self.request_times and self.request_times[0] < day_ago:
            self.request_times.popleft()

        # Clean domain requests
        for domain in list(self.domain_requests.keys()):
            domain_queue = self.domain_requests[domain]
            while domain_queue and domain_queue[0] < day_ago:
                domain_queue.popleft()

            # Remove empty queues
            if not domain_queue:
                del self.domain_requests[domain]


class ContentSanitizer:
    """Sanitize and filter content for security"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.sanitization_config = config.get("content_sanitization", {})

        # PII patterns
        self.pii_patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
            "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
        }

        # Dangerous content patterns
        self.dangerous_patterns = {
            "malware_url": r"(malware|virus|trojan|ransomware)",
            "phishing": r"(phishing|scam|fake|suspicious)",
            "explicit_content": r"(explicit|nsfw|adult)",
            "illegal_content": r"(illegal|criminal|terrorism)",
        }

        logger.info("Content sanitizer initialized")

    def sanitize_content(self, content: str, redact_pii: bool = True) -> tuple[str, dict[str, Any]]:
        """Sanitize content and return sanitized version with metadata"""
        sanitized = content
        redactions = {"pii": [], "dangerous": []}

        if redact_pii:
            sanitized, pii_redactions = self._redact_pii(sanitized)
            redactions["pii"] = pii_redactions

        sanitized, dangerous_redactions = self._filter_dangerous_content(sanitized)
        redactions["dangerous"] = dangerous_redactions

        metadata = {
            "original_length": len(content),
            "sanitized_length": len(sanitized),
            "redactions": redactions,
            "sanitization_timestamp": datetime.now(UTC).isoformat(),
        }

        return sanitized, metadata

    def _redact_pii(self, content: str) -> tuple[str, list[dict[str, Any]]]:
        """Redact personally identifiable information"""
        redacted_content = content
        redactions = []

        for pii_type, pattern in self.pii_patterns.items():
            matches = list(re.finditer(pattern, redacted_content))

            for match in reversed(matches):  # Reverse to maintain positions
                redacted_text = f"[REDACTED_{pii_type.upper()}]"
                redacted_content = (
                    redacted_content[: match.start()]
                    + redacted_text
                    + redacted_content[match.end() :]
                )

                redactions.append(
                    {
                        "type": pii_type,
                        "original_text": match.group(),
                        "position": match.start(),
                        "replacement": redacted_text,
                    }
                )

        return redacted_content, redactions

    def _filter_dangerous_content(self, content: str) -> tuple[str, list[dict[str, Any]]]:
        """Filter dangerous content"""
        filtered_content = content
        removals = []

        for danger_type, pattern in self.dangerous_patterns.items():
            matches = list(re.finditer(pattern, filtered_content, re.IGNORECASE))

            for match in reversed(matches):
                # Remove dangerous content sections
                filtered_content = (
                    filtered_content[: match.start()]
                    + f"[FILTERED_{danger_type.upper()}]"
                    + filtered_content[match.end() :]
                )

                removals.append(
                    {
                        "type": danger_type,
                        "original_text": match.group(),
                        "position": match.start(),
                        "reason": f"dangerous_content_{danger_type}",
                    }
                )

        return filtered_content, removals

    def check_content_safety(self, content: str) -> dict[str, Any]:
        """Check content safety without modification"""
        safety_report = {"safe": True, "risk_level": "low", "issues": [], "score": 1.0}

        total_issues = 0

        # Check for PII
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, content)
            if matches:
                safety_report["issues"].append(
                    {
                        "type": "pii",
                        "subtype": pii_type,
                        "count": len(matches),
                        "severity": "medium",
                    }
                )
                total_issues += len(matches)

        # Check for dangerous content
        for danger_type, pattern in self.dangerous_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                safety_report["issues"].append(
                    {
                        "type": "dangerous_content",
                        "subtype": danger_type,
                        "count": len(matches),
                        "severity": "high",
                    }
                )
                total_issues += len(matches) * 3  # Weight dangerous content more

        # Calculate risk level and score
        if total_issues == 0:
            safety_report["risk_level"] = "low"
            safety_report["score"] = 1.0
        elif total_issues <= 3:
            safety_report["risk_level"] = "medium"
            safety_report["score"] = 0.7
            safety_report["safe"] = False
        else:
            safety_report["risk_level"] = "high"
            safety_report["score"] = 0.3
            safety_report["safe"] = False

        return safety_report


class OSINTSecuritySandbox:
    """Main security sandbox for OSINT operations"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.security_config = config.get("security", {})
        self.osint_config = self.security_config.get("osint", {})

        # Security level
        self.security_level = SecurityLevel(self.osint_config.get("security_level", "medium"))

        # Components
        self.rate_limiter = RateLimiter(self.osint_config)
        self.content_sanitizer = ContentSanitizer(config)

        # Security rules
        self.security_rules: list[SecurityRule] = []
        self._load_security_rules()

        # Access logging
        self.access_log: deque = deque(maxlen=10000)
        self.blocked_attempts: dict[str, int] = defaultdict(int)

        # Audit trail
        self.audit_trail_file = Path(config.get("audit_trail", "security_audit.jsonl"))
        self.audit_trail_file.parent.mkdir(exist_ok=True)

        logger.info(f"OSINT security sandbox initialized (level: {self.security_level.value})")

    def _load_security_rules(self):
        """Load security rules from configuration"""
        rules_config = self.osint_config.get("rules", {})

        # Default whitelists
        default_whitelist = [
            "domain:arxiv.org",
            "domain:pubmed.ncbi.nlm.nih.gov",
            "domain:nature.com",
            "domain:science.org",
            "domain:who.int",
            "domain:cdc.gov",
            "domain:europa.eu",
            "domain:gov",
            "domain:edu",
        ]

        whitelist = rules_config.get("whitelist", default_whitelist)
        for i, pattern in enumerate(whitelist):
            rule = SecurityRule(
                rule_id=f"whitelist_{i}",
                rule_type="whitelist",
                pattern=pattern,
                action="allow",
                priority=10,
            )
            self.security_rules.append(rule)

        # Default blacklists
        default_blacklist = [
            "domain:darkweb.com",
            "regex:.*\\.onion$",  # Tor hidden services (unless explicitly enabled)
            "regex:.*malware.*",
            "regex:.*phishing.*",
            "domain:suspicious-site.com",
        ]

        blacklist = rules_config.get("blacklist", default_blacklist)
        for i, pattern in enumerate(blacklist):
            rule = SecurityRule(
                rule_id=f"blacklist_{i}",
                rule_type="blacklist",
                pattern=pattern,
                action="block",
                priority=20,
            )
            self.security_rules.append(rule)

        # Content filters
        content_filters = rules_config.get(
            "content_filters",
            ["regex:(password|secret|api[_-]?key)", "regex:(malware|virus|trojan)"],
        )

        for i, pattern in enumerate(content_filters):
            rule = SecurityRule(
                rule_id=f"content_filter_{i}",
                rule_type="content_filter",
                pattern=pattern,
                action="quarantine",
                priority=30,
            )
            self.security_rules.append(rule)

        # Sort rules by priority
        self.security_rules.sort(key=lambda r: r.priority)

        logger.info(f"Loaded {len(self.security_rules)} security rules")

    def check_access_permission(
        self, url: str, source_ip: str = "unknown", user_agent: str = "DeepResearchTool"
    ) -> tuple[AccessResult, dict[str, Any]]:
        """Check if access to URL is permitted"""
        timestamp = datetime.now(UTC)

        # Rate limiting check
        rate_ok, rate_reason = self.rate_limiter.check_rate_limit(url, source_ip)
        if not rate_ok:
            result = AccessResult.RATE_LIMITED
            metadata = {"reason": rate_reason, "timestamp": timestamp.isoformat()}
            self._log_access_attempt(timestamp, url, source_ip, user_agent, result, rate_reason)
            return result, metadata

        # Security rules check
        for rule in self.security_rules:
            if rule.matches(url):
                if rule.action == "allow":
                    result = AccessResult.ALLOWED
                    metadata = {
                        "rule_matched": rule.rule_id,
                        "rule_type": rule.rule_type,
                        "timestamp": timestamp.isoformat(),
                    }
                    self._log_access_attempt(
                        timestamp, url, source_ip, user_agent, result, rule.rule_id
                    )
                    self.rate_limiter.record_request(url, source_ip)
                    return result, metadata

                if rule.action == "block":
                    result = AccessResult.BLOCKED
                    metadata = {
                        "rule_matched": rule.rule_id,
                        "rule_type": rule.rule_type,
                        "reason": f"blocked_by_{rule.rule_type}",
                        "timestamp": timestamp.isoformat(),
                    }
                    self._log_access_attempt(
                        timestamp, url, source_ip, user_agent, result, rule.rule_id
                    )
                    self.blocked_attempts[url] += 1
                    return result, metadata

        # Default policy based on security level
        if self.security_level in [SecurityLevel.HIGH, SecurityLevel.PARANOID]:
            # Deny by default for high security
            result = AccessResult.BLOCKED
            metadata = {
                "reason": "default_deny_policy",
                "security_level": self.security_level.value,
                "timestamp": timestamp.isoformat(),
            }
            self._log_access_attempt(timestamp, url, source_ip, user_agent, result, "default_deny")
            return result, metadata
        # Allow by default for low/medium security
        result = AccessResult.ALLOWED
        metadata = {
            "reason": "default_allow_policy",
            "security_level": self.security_level.value,
            "timestamp": timestamp.isoformat(),
        }
        self._log_access_attempt(timestamp, url, source_ip, user_agent, result, "default_allow")
        self.rate_limiter.record_request(url, source_ip)
        return result, metadata

    def process_content(
        self, content: str, url: str, sanitize: bool = True
    ) -> tuple[str, dict[str, Any]]:
        """Process and sanitize content according to security rules"""
        # Check content safety
        safety_report = self.content_sanitizer.check_content_safety(content)

        processed_content = content
        processing_metadata = {
            "safety_report": safety_report,
            "processed": False,
            "url": url,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Apply content filters
        for rule in self.security_rules:
            if rule.rule_type == "content_filter" and rule.matches(url, content):
                if rule.action == "quarantine":
                    processing_metadata["quarantined"] = True
                    processing_metadata["quarantine_rule"] = rule.rule_id
                    logger.warning(f"Content quarantined by rule {rule.rule_id}: {url}")

        # Sanitize content if enabled
        if sanitize and (
            not safety_report["safe"]
            or self.security_level in [SecurityLevel.HIGH, SecurityLevel.PARANOID]
        ):
            processed_content, sanitization_metadata = self.content_sanitizer.sanitize_content(
                content
            )
            processing_metadata["sanitization"] = sanitization_metadata
            processing_metadata["processed"] = True

        # Log content processing
        self._log_content_processing(url, len(content), len(processed_content), safety_report)

        return processed_content, processing_metadata

    def _log_access_attempt(
        self,
        timestamp: datetime,
        url: str,
        source_ip: str,
        user_agent: str,
        result: AccessResult,
        rule_matched: str | None = None,
    ):
        """Log access attempt"""
        attempt = AccessAttempt(
            timestamp=timestamp,
            url=url,
            source_ip=source_ip,
            user_agent=user_agent,
            result=result,
            rule_matched=rule_matched,
        )

        self.access_log.append(attempt)

        # Write to audit trail
        audit_entry = {
            "type": "access_attempt",
            "timestamp": timestamp.isoformat(),
            "url": url,
            "source_ip": source_ip,
            "user_agent": user_agent,
            "result": result.value,
            "rule_matched": rule_matched,
        }

        self._write_audit_entry(audit_entry)

    def _log_content_processing(
        self, url: str, original_size: int, processed_size: int, safety_report: dict[str, Any]
    ):
        """Log content processing"""
        audit_entry = {
            "type": "content_processing",
            "timestamp": datetime.now(UTC).isoformat(),
            "url": url,
            "original_size": original_size,
            "processed_size": processed_size,
            "safety_report": safety_report,
        }

        self._write_audit_entry(audit_entry)

    def _write_audit_entry(self, entry: dict[str, Any]):
        """Write entry to audit trail"""
        try:
            with open(self.audit_trail_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit entry: {e}")

    def get_security_statistics(self) -> dict[str, Any]:
        """Get security statistics and metrics"""
        total_attempts = len(self.access_log)

        if total_attempts == 0:
            return {"message": "No access attempts recorded"}

        # Count by result type
        result_counts = defaultdict(int)
        for attempt in self.access_log:
            result_counts[attempt.result.value] += 1

        # Recent activity (last hour)
        hour_ago = datetime.now(UTC) - timedelta(hours=1)
        recent_attempts = [a for a in self.access_log if a.timestamp > hour_ago]

        # Top blocked URLs
        top_blocked = sorted(self.blocked_attempts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Rule effectiveness
        rule_usage = defaultdict(int)
        for attempt in self.access_log:
            if attempt.rule_matched:
                rule_usage[attempt.rule_matched] += 1

        return {
            "total_access_attempts": total_attempts,
            "result_distribution": dict(result_counts),
            "success_rate": result_counts["allowed"] / total_attempts,
            "block_rate": result_counts["blocked"] / total_attempts,
            "recent_activity_count": len(recent_attempts),
            "top_blocked_urls": top_blocked,
            "active_rules": len(self.security_rules),
            "rule_usage": dict(rule_usage),
            "security_level": self.security_level.value,
            "rate_limiter_stats": {
                "active_requests": len(self.rate_limiter.active_requests),
                "tracked_domains": len(self.rate_limiter.domain_requests),
            },
        }

    def export_audit_trail(
        self, since: datetime | None = None, format: str = "json"
    ) -> dict[str, Any]:
        """Export audit trail for analysis"""
        if since is None:
            since = datetime.now(UTC) - timedelta(days=7)

        # Read audit trail file
        audit_entries = []
        try:
            with open(self.audit_trail_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entry_time = datetime.fromisoformat(entry["timestamp"])
                        if entry_time >= since:
                            audit_entries.append(entry)
                    except (json.JSONDecodeError, KeyError):
                        continue
        except FileNotFoundError:
            pass

        return {
            "export_timestamp": datetime.now(UTC).isoformat(),
            "export_range_start": since.isoformat(),
            "format": format,
            "total_entries": len(audit_entries),
            "entries": audit_entries,
            "statistics": self.get_security_statistics(),
        }


def create_osint_security_sandbox(config: dict[str, Any]) -> OSINTSecuritySandbox:
    """Factory function for OSINT security sandbox"""
    return OSINTSecuritySandbox(config)


# Usage example
if __name__ == "__main__":
    config = {
        "security": {
            "osint": {
                "security_level": "medium",
                "rate_limits": {"requests_per_minute": 30, "requests_per_hour": 500},
                "rules": {
                    "whitelist": ["domain:arxiv.org", "domain:pubmed.ncbi.nlm.nih.gov"],
                    "blacklist": ["regex:.*malware.*", "domain:suspicious-site.com"],
                },
            }
        },
        "audit_trail": "test_audit.jsonl",
    }

    sandbox = OSINTSecuritySandbox(config)

    # Test access permissions
    test_urls = [
        "https://arxiv.org/article123",  # Should be allowed (whitelist)
        "https://suspicious-site.com/page",  # Should be blocked (blacklist)
        "https://example.com/research",  # Should depend on security level
    ]

    for url in test_urls:
        result, metadata = sandbox.check_access_permission(url)
        print(f"URL: {url}")
        print(f"Result: {result.value}")
        print(f"Metadata: {metadata}")
        print()

    # Test content processing
    test_content = "This research shows password: secret123 and email: user@example.com"
    processed, metadata = sandbox.process_content(test_content, "https://example.com/test")

    print(f"Original: {test_content}")
    print(f"Processed: {processed}")
    print(f"Safety report: {metadata['safety_report']}")

    # Get statistics
    stats = sandbox.get_security_statistics()
    print(f"Security stats: {stats}")
