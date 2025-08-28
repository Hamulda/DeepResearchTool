#!/usr/bin/env python3
"""
Bezpečnostní a compliance systém
robots.txt, rate-limity, PII redakce, allow/deny listy

Author: Senior Python/MLOps Agent
"""

import asyncio
import json
import re
import hashlib
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Any, Tuple
from datetime import datetime, timedelta
from urllib.parse import urlparse, urljoin
import aiohttp
import aiofiles
from pathlib import Path


@dataclass
class SecurityViolation:
    """Bezpečnostní porušení"""
    violation_type: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    source_url: Optional[str]
    timestamp: datetime
    action_taken: str


@dataclass
class RateLimit:
    """Rate limit pro doménu"""
    domain: str
    requests_per_minute: int
    requests_per_hour: int
    current_requests: int
    window_start: datetime
    backoff_until: Optional[datetime]


class SecurityManager:
    """Centrální bezpečnostní manager"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.security_violations = []

        # Load security policies
        self.allow_domains = set(config.get("allow_domains", []))
        self.deny_domains = set(config.get("deny_domains", []))
        self.rate_limits = self._load_rate_limits()
        self.robots_cache = {}

        # PII patterns
        self.pii_patterns = self._load_pii_patterns()

        # User agent
        self.user_agent = config.get("user_agent", "DeepResearchTool/1.0 (+research@example.com)")

        print("🔒 Security Manager initialized")
        print(f"   Allow domains: {len(self.allow_domains)}")
        print(f"   Deny domains: {len(self.deny_domains)}")
        print(f"   Rate limits: {len(self.rate_limits)} domains")

    def _load_rate_limits(self) -> Dict[str, RateLimit]:
        """Načte rate limity pro domény"""
        default_limits = self.config.get("default_rate_limits", {
            "requests_per_minute": 10,
            "requests_per_hour": 100
        })

        domain_limits = self.config.get("domain_specific_limits", {
            "api.crossref.org": {"requests_per_minute": 50, "requests_per_hour": 1000},
            "api.openalex.org": {"requests_per_minute": 100, "requests_per_hour": 10000},
            "www.sec.gov": {"requests_per_minute": 5, "requests_per_hour": 50},
            "ahmia.fi": {"requests_per_minute": 2, "requests_per_hour": 20}
        })

        rate_limits = {}

        # Default limit pro všechny domény
        for domain, limits in domain_limits.items():
            rate_limits[domain] = RateLimit(
                domain=domain,
                requests_per_minute=limits.get("requests_per_minute", default_limits["requests_per_minute"]),
                requests_per_hour=limits.get("requests_per_hour", default_limits["requests_per_hour"]),
                current_requests=0,
                window_start=datetime.now(),
                backoff_until=None
            )

        return rate_limits

    def _load_pii_patterns(self) -> List[Tuple[str, str]]:
        """Načte PII detection patterns"""
        return [
            (r'\b\d{3}-\d{2}-\d{4}\b', "SSN"),  # Social Security Number
            (r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', "Credit Card"),  # Credit card
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "Email"),  # Email
            (r'\b\d{3}-\d{3}-\d{4}\b', "Phone"),  # Phone number
            (r'\b(?:\d{1,3}\.){3}\d{1,3}\b', "IP Address"),  # IP address
            (r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b', "IBAN"),  # IBAN
        ]

    async def check_domain_access(self, url: str) -> Tuple[bool, Optional[str]]:
        """Kontrola přístupu k doméně"""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()

        # Kontrola deny list
        if domain in self.deny_domains:
            violation = SecurityViolation(
                violation_type="domain_denied",
                severity="high",
                description=f"Access to domain {domain} is explicitly denied",
                source_url=url,
                timestamp=datetime.now(),
                action_taken="blocked_request"
            )
            self.security_violations.append(violation)
            return False, f"Domain {domain} is on deny list"

        # Kontrola allow list (pokud je definován)
        if self.allow_domains and domain not in self.allow_domains:
            violation = SecurityViolation(
                violation_type="domain_not_allowed",
                severity="medium",
                description=f"Domain {domain} is not on allow list",
                source_url=url,
                timestamp=datetime.now(),
                action_taken="blocked_request"
            )
            self.security_violations.append(violation)
            return False, f"Domain {domain} is not on allow list"

        return True, None

    async def check_robots_txt(self, url: str) -> Tuple[bool, Optional[str]]:
        """Kontrola robots.txt"""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        robots_url = f"{parsed_url.scheme}://{domain}/robots.txt"

        # Cache check
        if domain in self.robots_cache:
            cache_entry = self.robots_cache[domain]
            # Cache expiry (24 hours)
            if datetime.now() - cache_entry["timestamp"] < timedelta(hours=24):
                return self._check_robots_rules(url, cache_entry["rules"])

        # Fetch robots.txt
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"User-Agent": self.user_agent}
                async with session.get(robots_url, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        robots_content = await response.text()
                        rules = self._parse_robots_txt(robots_content)
                    else:
                        # No robots.txt nebo error -> allow
                        rules = {"allowed": True, "crawl_delay": 1}

        except Exception as e:
            # Network error -> conservative allow
            rules = {"allowed": True, "crawl_delay": 2}

        # Cache rules
        self.robots_cache[domain] = {
            "rules": rules,
            "timestamp": datetime.now()
        }

        return self._check_robots_rules(url, rules)

    def _parse_robots_txt(self, content: str) -> Dict[str, Any]:
        """Parse robots.txt obsah"""
        rules = {
            "allowed": True,
            "crawl_delay": 1,
            "disallowed_paths": [],
            "allowed_paths": []
        }

        lines = content.split('\n')
        current_user_agent = None
        applies_to_us = False

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.lower().startswith('user-agent:'):
                user_agent = line.split(':', 1)[1].strip()
                applies_to_us = (user_agent == '*' or
                               'deepresearchtool' in user_agent.lower() or
                               user_agent.lower() in self.user_agent.lower())
                current_user_agent = user_agent

            elif applies_to_us:
                if line.lower().startswith('disallow:'):
                    path = line.split(':', 1)[1].strip()
                    if path:
                        rules["disallowed_paths"].append(path)
                    else:
                        # Empty disallow = allow all
                        rules["allowed"] = True

                elif line.lower().startswith('allow:'):
                    path = line.split(':', 1)[1].strip()
                    if path:
                        rules["allowed_paths"].append(path)

                elif line.lower().startswith('crawl-delay:'):
                    try:
                        delay = float(line.split(':', 1)[1].strip())
                        rules["crawl_delay"] = max(delay, 1)  # Min 1 second
                    except ValueError:
                        pass

        return rules

    def _check_robots_rules(self, url: str, rules: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Kontrola robots.txt pravidel"""
        parsed_url = urlparse(url)
        path = parsed_url.path or '/'

        # Check disallowed paths
        for disallowed in rules.get("disallowed_paths", []):
            if path.startswith(disallowed):
                # Check if there's a more specific allow rule
                allowed_by_specific = False
                for allowed in rules.get("allowed_paths", []):
                    if path.startswith(allowed) and len(allowed) > len(disallowed):
                        allowed_by_specific = True
                        break

                if not allowed_by_specific:
                    violation = SecurityViolation(
                        violation_type="robots_txt_violation",
                        severity="medium",
                        description=f"Path {path} disallowed by robots.txt",
                        source_url=url,
                        timestamp=datetime.now(),
                        action_taken="blocked_request"
                    )
                    self.security_violations.append(violation)
                    return False, f"Path disallowed by robots.txt: {disallowed}"

        return True, None

    async def check_rate_limit(self, url: str) -> Tuple[bool, Optional[float]]:
        """Kontrola rate limitu"""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()

        # Get or create rate limit
        if domain not in self.rate_limits:
            default_limits = self.config.get("default_rate_limits", {
                "requests_per_minute": 10,
                "requests_per_hour": 100
            })
            self.rate_limits[domain] = RateLimit(
                domain=domain,
                requests_per_minute=default_limits["requests_per_minute"],
                requests_per_hour=default_limits["requests_per_hour"],
                current_requests=0,
                window_start=datetime.now(),
                backoff_until=None
            )

        rate_limit = self.rate_limits[domain]
        now = datetime.now()

        # Check backoff
        if rate_limit.backoff_until and now < rate_limit.backoff_until:
            remaining = (rate_limit.backoff_until - now).total_seconds()
            return False, remaining

        # Reset window if needed
        if now - rate_limit.window_start > timedelta(minutes=1):
            rate_limit.current_requests = 0
            rate_limit.window_start = now

        # Check limit
        if rate_limit.current_requests >= rate_limit.requests_per_minute:
            # Apply exponential backoff
            backoff_seconds = min(300, 60 * (2 ** (rate_limit.current_requests - rate_limit.requests_per_minute)))
            rate_limit.backoff_until = now + timedelta(seconds=backoff_seconds)

            violation = SecurityViolation(
                violation_type="rate_limit_exceeded",
                severity="medium",
                description=f"Rate limit exceeded for {domain}",
                source_url=url,
                timestamp=datetime.now(),
                action_taken=f"backoff_{backoff_seconds}s"
            )
            self.security_violations.append(violation)

            return False, backoff_seconds

        # Increment counter
        rate_limit.current_requests += 1

        # Return crawl delay
        robots_rules = self.robots_cache.get(domain, {}).get("rules", {})
        crawl_delay = robots_rules.get("crawl_delay", 1)

        return True, crawl_delay

    def redact_pii(self, text: str) -> Tuple[str, List[str]]:
        """Redakce PII z textu"""
        redacted_text = text
        found_pii = []

        for pattern, pii_type in self.pii_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                found_pii.append(f"{pii_type}: {match[:4]}***")
                # Replace with [REDACTED_TYPE]
                redacted_text = re.sub(pattern, f"[REDACTED_{pii_type.upper().replace(' ', '_')}]", redacted_text)

        if found_pii:
            violation = SecurityViolation(
                violation_type="pii_detected",
                severity="high",
                description=f"PII detected and redacted: {', '.join(set(found_pii))}",
                source_url=None,
                timestamp=datetime.now(),
                action_taken="redacted"
            )
            self.security_violations.append(violation)

        return redacted_text, found_pii

    async def validate_request(self, url: str) -> Tuple[bool, Optional[str], Optional[float]]:
        """Kompletní validace requestu"""
        # 1. Domain access check
        domain_allowed, domain_error = await self.check_domain_access(url)
        if not domain_allowed:
            return False, domain_error, None

        # 2. Robots.txt check
        robots_allowed, robots_error = await self.check_robots_txt(url)
        if not robots_allowed:
            return False, robots_error, None

        # 3. Rate limit check
        rate_allowed, delay_or_backoff = await self.check_rate_limit(url)
        if not rate_allowed:
            return False, f"Rate limited, retry in {delay_or_backoff}s", delay_or_backoff

        return True, None, delay_or_backoff

    def get_security_headers(self, url: str) -> Dict[str, str]:
        """Vrátí bezpečné HTTP headers"""
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",  # Do Not Track
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }

        # Domain-specific headers
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()

        if "crossref.org" in domain:
            headers["mailto"] = "research@example.com"
        elif "europepmc" in domain:
            headers["format"] = "json"

        return headers

    def check_content_safety(self, content: str, url: str) -> Tuple[bool, List[str]]:
        """Kontrola bezpečnosti obsahu"""
        safety_issues = []

        # Malware indicators
        malware_patterns = [
            r'<script[^>]*>.*?</script>',  # Scripts
            r'javascript:',  # Javascript URLs
            r'data:text/html',  # Data URLs
            r'eval\(',  # Eval calls
            r'document\.write',  # Document.write
        ]

        for pattern in malware_patterns:
            if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                safety_issues.append(f"Potential malware pattern: {pattern}")

        # Suspicious content
        suspicious_keywords = [
            'phishing', 'malware', 'virus', 'trojan', 'keylogger',
            'password', 'credit card', 'social security', 'bank account'
        ]

        content_lower = content.lower()
        for keyword in suspicious_keywords:
            if keyword in content_lower:
                safety_issues.append(f"Suspicious keyword: {keyword}")

        # Log violations
        if safety_issues:
            violation = SecurityViolation(
                violation_type="unsafe_content",
                severity="high",
                description=f"Unsafe content detected: {', '.join(safety_issues[:3])}",
                source_url=url,
                timestamp=datetime.now(),
                action_taken="content_flagged"
            )
            self.security_violations.append(violation)

        return len(safety_issues) == 0, safety_issues

    def generate_security_report(self) -> Dict[str, Any]:
        """Generuje bezpečnostní report"""
        now = datetime.now()

        # Violation statistics
        violation_counts = {}
        recent_violations = []

        for violation in self.security_violations:
            violation_counts[violation.violation_type] = violation_counts.get(violation.violation_type, 0) + 1

            # Recent violations (last hour)
            if now - violation.timestamp < timedelta(hours=1):
                recent_violations.append({
                    "type": violation.violation_type,
                    "severity": violation.severity,
                    "description": violation.description,
                    "timestamp": violation.timestamp.isoformat(),
                    "action": violation.action_taken
                })

        # Rate limit status
        rate_limit_status = {}
        for domain, rate_limit in self.rate_limits.items():
            rate_limit_status[domain] = {
                "current_requests": rate_limit.current_requests,
                "limit_per_minute": rate_limit.requests_per_minute,
                "backoff_until": rate_limit.backoff_until.isoformat() if rate_limit.backoff_until else None,
                "window_start": rate_limit.window_start.isoformat()
            }

        return {
            "timestamp": now.isoformat(),
            "total_violations": len(self.security_violations),
            "violation_types": violation_counts,
            "recent_violations": recent_violations,
            "rate_limit_status": rate_limit_status,
            "security_settings": {
                "allow_domains_count": len(self.allow_domains),
                "deny_domains_count": len(self.deny_domains),
                "pii_patterns_count": len(self.pii_patterns),
                "robots_cache_size": len(self.robots_cache)
            }
        }

    def save_security_artifacts(self, output_dir: str) -> Dict[str, str]:
        """Uloží bezpečnostní artefakty"""
        artifacts = {}

        # Security report
        report = self.generate_security_report()
        report_file = f"{output_dir}/security_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        artifacts["security_report"] = report_file

        # Violations log
        violations_data = []
        for violation in self.security_violations:
            violations_data.append({
                "violation_type": violation.violation_type,
                "severity": violation.severity,
                "description": violation.description,
                "source_url": violation.source_url,
                "timestamp": violation.timestamp.isoformat(),
                "action_taken": violation.action_taken
            })

        violations_file = f"{output_dir}/security_violations.json"
        with open(violations_file, "w") as f:
            json.dump(violations_data, f, indent=2)
        artifacts["violations_log"] = violations_file

        return artifacts

    async def cleanup_old_data(self, max_age_hours: int = 24) -> int:
        """Vyčistí stará bezpečnostní data"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        # Clean old violations
        original_count = len(self.security_violations)
        self.security_violations = [v for v in self.security_violations if v.timestamp > cutoff_time]
        removed_violations = original_count - len(self.security_violations)

        # Clean old robots cache
        domains_to_remove = []
        for domain, cache_entry in self.robots_cache.items():
            if cache_entry["timestamp"] < cutoff_time:
                domains_to_remove.append(domain)

        for domain in domains_to_remove:
            del self.robots_cache[domain]

        print(f"🧹 Security cleanup: removed {removed_violations} violations, {len(domains_to_remove)} cached robots.txt")
        return removed_violations + len(domains_to_remove)
