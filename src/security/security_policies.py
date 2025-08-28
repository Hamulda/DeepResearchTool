"""
FÁZE 7: Security Policy Engine
Statická bezpečnostní pravidla a policy validation
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class PolicySeverity(Enum):
    """Úrovně závažnosti bezpečnostních pravidel"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PolicyAction(Enum):
    """Akce při porušení pravidel"""
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    QUARANTINE = "quarantine"
    LOG_ONLY = "log_only"


class PolicyType(Enum):
    """Typy bezpečnostních pravidel"""
    URL_FILTERING = "url_filtering"
    CONTENT_SCANNING = "content_scanning"
    RATE_LIMITING = "rate_limiting"
    PII_PROTECTION = "pii_protection"
    MALWARE_DETECTION = "malware_detection"
    DOMAIN_REPUTATION = "domain_reputation"
    FILE_TYPE_VALIDATION = "file_type_validation"
    SIZE_LIMITS = "size_limits"


@dataclass
class SecurityRule:
    """Definice bezpečnostního pravidla"""
    rule_id: str
    name: str
    description: str
    policy_type: PolicyType
    severity: PolicySeverity
    action: PolicyAction
    conditions: Dict[str, Any]
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)


@dataclass
class PolicyViolation:
    """Záznam o porušení pravidla"""
    violation_id: str
    rule_id: str
    rule_name: str
    severity: PolicySeverity
    action_taken: PolicyAction
    violation_details: Dict[str, Any]
    timestamp: float
    context: str = ""
    remediation_suggested: str = ""


@dataclass
class PolicyValidationResult:
    """Výsledek validace proti pravidlům"""
    allowed: bool
    violations: List[PolicyViolation]
    warnings: List[str]
    action_required: PolicyAction
    confidence_score: float
    processing_time_ms: float


class SecurityPolicyEngine:
    """
    FÁZE 7: Advanced Security Policy Engine

    Features:
    - Konfigurovatelná bezpečnostní pravidla
    - Multi-layer validation (URL, content, metadata)
    - Dynamic rule evaluation
    - Audit logging s compliance tracking
    - Risk scoring a threat detection
    """

    def __init__(
        self,
        policy_file: Optional[Path] = None,
        enable_dynamic_rules: bool = True,
        default_action: PolicyAction = PolicyAction.WARN
    ):
        self.policy_file = policy_file or Path("security_policies.json")
        self.enable_dynamic_rules = enable_dynamic_rules
        self.default_action = default_action

        # Bezpečnostní pravidla
        self.rules: Dict[str, SecurityRule] = {}

        # Whitelist/blacklist
        self.whitelisted_domains: Set[str] = set()
        self.blacklisted_domains: Set[str] = set()
        self.whitelisted_ips: Set[str] = set()
        self.blacklisted_ips: Set[str] = set()

        # Policy statistics
        self.policy_stats = {
            "total_evaluations": 0,
            "violations_detected": 0,
            "blocks_enforced": 0,
            "warnings_issued": 0,
            "rules_triggered": {}
        }

        # Violation history
        self.violation_history: List[PolicyViolation] = []

        # Load initial policies
        self._load_default_policies()
        if self.policy_file.exists():
            self._load_policies_from_file()

        logger.info(f"SecurityPolicyEngine initialized with {len(self.rules)} rules")

    def _load_default_policies(self) -> None:
        """Načtení výchozích bezpečnostních pravidel"""

        default_rules = [
            # URL Filtering Rules
            SecurityRule(
                rule_id="url_malicious_domains",
                name="Malicious Domain Detection",
                description="Block known malicious domains",
                policy_type=PolicyType.URL_FILTERING,
                severity=PolicySeverity.CRITICAL,
                action=PolicyAction.BLOCK,
                conditions={
                    "blacklisted_domains": [
                        "malware-site.com",
                        "phishing-example.net",
                        "suspicious-domain.org"
                    ],
                    "domain_reputation_threshold": 0.3
                },
                tags=["malware", "phishing", "critical"]
            ),

            # Content Scanning Rules
            SecurityRule(
                rule_id="content_malware_detection",
                name="Content Malware Scanner",
                description="Scan content for malware signatures",
                policy_type=PolicyType.CONTENT_SCANNING,
                severity=PolicySeverity.HIGH,
                action=PolicyAction.QUARANTINE,
                conditions={
                    "scan_binaries": True,
                    "scan_archives": True,
                    "max_file_size_mb": 100,
                    "blocked_file_types": [".exe", ".scr", ".bat", ".cmd"]
                },
                tags=["malware", "content", "files"]
            ),

            # Rate Limiting Rules
            SecurityRule(
                rule_id="rate_limit_aggressive",
                name="Aggressive Rate Limiting",
                description="Detect and limit aggressive crawling",
                policy_type=PolicyType.RATE_LIMITING,
                severity=PolicySeverity.MEDIUM,
                action=PolicyAction.WARN,
                conditions={
                    "requests_per_minute_threshold": 100,
                    "requests_per_hour_threshold": 2000,
                    "burst_threshold": 20
                },
                tags=["rate-limiting", "ddos", "crawling"]
            ),

            # PII Protection Rules
            SecurityRule(
                rule_id="pii_strict_protection",
                name="Strict PII Protection",
                description="Block content with unredacted PII",
                policy_type=PolicyType.PII_PROTECTION,
                severity=PolicySeverity.HIGH,
                action=PolicyAction.BLOCK,
                conditions={
                    "max_pii_instances": 0,
                    "blocked_pii_types": [
                        "ssn", "credit_card", "passport", "driver_license"
                    ],
                    "require_redaction": True
                },
                tags=["pii", "privacy", "gdpr"]
            ),

            # File Type Validation
            SecurityRule(
                rule_id="file_type_validation",
                name="File Type Validation",
                description="Validate and restrict file types",
                policy_type=PolicyType.FILE_TYPE_VALIDATION,
                severity=PolicySeverity.MEDIUM,
                action=PolicyAction.WARN,
                conditions={
                    "allowed_file_types": [
                        ".txt", ".pdf", ".doc", ".docx",
                        ".html", ".htm", ".json", ".xml", ".csv"
                    ],
                    "scan_file_headers": True,
                    "check_mime_type": True
                },
                tags=["files", "validation", "security"]
            ),

            # Size Limits
            SecurityRule(
                rule_id="content_size_limits",
                name="Content Size Limits",
                description="Enforce content size limits",
                policy_type=PolicyType.SIZE_LIMITS,
                severity=PolicySeverity.LOW,
                action=PolicyAction.WARN,
                conditions={
                    "max_content_size_mb": 50,
                    "max_response_size_mb": 10,
                    "warn_threshold_mb": 5
                },
                tags=["size", "limits", "performance"]
            )
        ]

        for rule in default_rules:
            self.rules[rule.rule_id] = rule

    def _load_policies_from_file(self) -> None:
        """Načtení pravidel ze souboru"""
        try:
            with open(self.policy_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Load rules
            if "rules" in data:
                for rule_data in data["rules"]:
                    rule = SecurityRule(
                        rule_id=rule_data["rule_id"],
                        name=rule_data["name"],
                        description=rule_data["description"],
                        policy_type=PolicyType(rule_data["policy_type"]),
                        severity=PolicySeverity(rule_data["severity"]),
                        action=PolicyAction(rule_data["action"]),
                        conditions=rule_data["conditions"],
                        enabled=rule_data.get("enabled", True),
                        tags=rule_data.get("tags", [])
                    )
                    self.rules[rule.rule_id] = rule

            # Load whitelists/blacklists
            if "whitelisted_domains" in data:
                self.whitelisted_domains.update(data["whitelisted_domains"])
            if "blacklisted_domains" in data:
                self.blacklisted_domains.update(data["blacklisted_domains"])

            logger.info(f"Loaded {len(data.get('rules', []))} rules from {self.policy_file}")

        except Exception as e:
            logger.error(f"Error loading policies from {self.policy_file}: {e}")

    def save_policies_to_file(self) -> None:
        """Uložení pravidel do souboru"""
        try:
            data = {
                "rules": [
                    {
                        "rule_id": rule.rule_id,
                        "name": rule.name,
                        "description": rule.description,
                        "policy_type": rule.policy_type.value,
                        "severity": rule.severity.value,
                        "action": rule.action.value,
                        "conditions": rule.conditions,
                        "enabled": rule.enabled,
                        "tags": rule.tags,
                        "created_at": rule.created_at,
                        "updated_at": rule.updated_at
                    }
                    for rule in self.rules.values()
                ],
                "whitelisted_domains": list(self.whitelisted_domains),
                "blacklisted_domains": list(self.blacklisted_domains),
                "whitelisted_ips": list(self.whitelisted_ips),
                "blacklisted_ips": list(self.blacklisted_ips)
            }

            with open(self.policy_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(self.rules)} rules to {self.policy_file}")

        except Exception as e:
            logger.error(f"Error saving policies to {self.policy_file}: {e}")

    async def validate_url(self, url: str) -> PolicyValidationResult:
        """Validace URL proti bezpečnostním pravidlům"""
        start_time = time.time()
        violations = []
        warnings = []

        domain = urlparse(url).netloc.lower()

        # Global whitelist check
        if domain in self.whitelisted_domains:
            return PolicyValidationResult(
                allowed=True,
                violations=[],
                warnings=[],
                action_required=PolicyAction.ALLOW,
                confidence_score=1.0,
                processing_time_ms=(time.time() - start_time) * 1000
            )

        # Global blacklist check
        if domain in self.blacklisted_domains:
            violation = PolicyViolation(
                violation_id=f"blacklist_{int(time.time())}",
                rule_id="global_blacklist",
                rule_name="Global Domain Blacklist",
                severity=PolicySeverity.CRITICAL,
                action_taken=PolicyAction.BLOCK,
                violation_details={"domain": domain, "url": url},
                timestamp=time.time(),
                context="Domain found in global blacklist",
                remediation_suggested="Remove domain from blacklist or verify legitimacy"
            )
            violations.append(violation)

        # Evaluate URL filtering rules
        for rule in self.rules.values():
            if not rule.enabled or rule.policy_type != PolicyType.URL_FILTERING:
                continue

            violation = await self._evaluate_url_rule(rule, url, domain)
            if violation:
                violations.append(violation)

        # Determine final action
        action_required = self._determine_action(violations)
        allowed = action_required not in [PolicyAction.BLOCK, PolicyAction.QUARANTINE]

        # Update statistics
        self._update_statistics(violations)

        return PolicyValidationResult(
            allowed=allowed,
            violations=violations,
            warnings=warnings,
            action_required=action_required,
            confidence_score=self._calculate_confidence(violations),
            processing_time_ms=(time.time() - start_time) * 1000
        )

    async def validate_content(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PolicyValidationResult:
        """Validace obsahu proti bezpečnostním pravidlům"""
        start_time = time.time()
        violations = []
        warnings = []

        metadata = metadata or {}

        # Evaluate content scanning rules
        for rule in self.rules.values():
            if not rule.enabled:
                continue

            if rule.policy_type == PolicyType.CONTENT_SCANNING:
                violation = await self._evaluate_content_rule(rule, content, metadata)
            elif rule.policy_type == PolicyType.PII_PROTECTION:
                violation = await self._evaluate_pii_rule(rule, content)
            elif rule.policy_type == PolicyType.SIZE_LIMITS:
                violation = await self._evaluate_size_rule(rule, content, metadata)
            elif rule.policy_type == PolicyType.FILE_TYPE_VALIDATION:
                violation = await self._evaluate_file_type_rule(rule, metadata)
            else:
                continue

            if violation:
                violations.append(violation)

        # Determine final action
        action_required = self._determine_action(violations)
        allowed = action_required not in [PolicyAction.BLOCK, PolicyAction.QUARANTINE]

        # Update statistics
        self._update_statistics(violations)

        return PolicyValidationResult(
            allowed=allowed,
            violations=violations,
            warnings=warnings,
            action_required=action_required,
            confidence_score=self._calculate_confidence(violations),
            processing_time_ms=(time.time() - start_time) * 1000
        )

    async def _evaluate_url_rule(
        self,
        rule: SecurityRule,
        url: str,
        domain: str
    ) -> Optional[PolicyViolation]:
        """Evaluace URL pravidla"""
        conditions = rule.conditions

        # Blacklisted domains check
        if "blacklisted_domains" in conditions:
            if domain in conditions["blacklisted_domains"]:
                return PolicyViolation(
                    violation_id=f"{rule.rule_id}_{int(time.time())}",
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    severity=rule.severity,
                    action_taken=rule.action,
                    violation_details={"domain": domain, "url": url},
                    timestamp=time.time(),
                    context=f"Domain {domain} found in rule blacklist",
                    remediation_suggested="Verify domain legitimacy or update blacklist"
                )

        # Domain reputation check (mock implementation)
        if "domain_reputation_threshold" in conditions:
            reputation_score = await self._get_domain_reputation(domain)
            if reputation_score < conditions["domain_reputation_threshold"]:
                return PolicyViolation(
                    violation_id=f"{rule.rule_id}_{int(time.time())}",
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    severity=rule.severity,
                    action_taken=rule.action,
                    violation_details={
                        "domain": domain,
                        "reputation_score": reputation_score,
                        "threshold": conditions["domain_reputation_threshold"]
                    },
                    timestamp=time.time(),
                    context=f"Domain reputation {reputation_score} below threshold",
                    remediation_suggested="Manual review of domain reputation"
                )

        return None

    async def _evaluate_content_rule(
        self,
        rule: SecurityRule,
        content: str,
        metadata: Dict[str, Any]
    ) -> Optional[PolicyViolation]:
        """Evaluace content scanning pravidla"""
        conditions = rule.conditions

        # File type checking
        if "blocked_file_types" in conditions:
            file_type = metadata.get("file_type", "").lower()
            if file_type in conditions["blocked_file_types"]:
                return PolicyViolation(
                    violation_id=f"{rule.rule_id}_{int(time.time())}",
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    severity=rule.severity,
                    action_taken=rule.action,
                    violation_details={"file_type": file_type},
                    timestamp=time.time(),
                    context=f"Blocked file type detected: {file_type}",
                    remediation_suggested="Verify file safety or update allowed file types"
                )

        # Content size checking
        if "max_file_size_mb" in conditions:
            file_size_mb = len(content.encode('utf-8')) / (1024 * 1024)
            if file_size_mb > conditions["max_file_size_mb"]:
                return PolicyViolation(
                    violation_id=f"{rule.rule_id}_{int(time.time())}",
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    severity=rule.severity,
                    action_taken=rule.action,
                    violation_details={
                        "file_size_mb": file_size_mb,
                        "max_allowed_mb": conditions["max_file_size_mb"]
                    },
                    timestamp=time.time(),
                    context=f"File size {file_size_mb:.2f}MB exceeds limit",
                    remediation_suggested="Reduce file size or increase limit"
                )

        return None

    async def _evaluate_pii_rule(
        self,
        rule: SecurityRule,
        content: str
    ) -> Optional[PolicyViolation]:
        """Evaluace PII protection pravidla"""
        conditions = rule.conditions

        # Mock PII detection - v produkci by se použil skutečný PII detector
        pii_patterns = {
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14})\b',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        }

        pii_found = []
        for pii_type, pattern in pii_patterns.items():
            import re
            matches = re.findall(pattern, content)
            if matches:
                pii_found.extend([(pii_type, match) for match in matches])

        # Check against rule conditions
        if "max_pii_instances" in conditions:
            if len(pii_found) > conditions["max_pii_instances"]:
                return PolicyViolation(
                    violation_id=f"{rule.rule_id}_{int(time.time())}",
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    severity=rule.severity,
                    action_taken=rule.action,
                    violation_details={
                        "pii_instances_found": len(pii_found),
                        "max_allowed": conditions["max_pii_instances"],
                        "pii_types": list(set(pii[0] for pii in pii_found))
                    },
                    timestamp=time.time(),
                    context=f"Found {len(pii_found)} PII instances, max allowed: {conditions['max_pii_instances']}",
                    remediation_suggested="Apply PII redaction before processing"
                )

        return None

    async def _evaluate_size_rule(
        self,
        rule: SecurityRule,
        content: str,
        metadata: Dict[str, Any]
    ) -> Optional[PolicyViolation]:
        """Evaluace size limits pravidla"""
        conditions = rule.conditions
        content_size_mb = len(content.encode('utf-8')) / (1024 * 1024)

        if "max_content_size_mb" in conditions:
            if content_size_mb > conditions["max_content_size_mb"]:
                return PolicyViolation(
                    violation_id=f"{rule.rule_id}_{int(time.time())}",
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    severity=rule.severity,
                    action_taken=rule.action,
                    violation_details={
                        "content_size_mb": content_size_mb,
                        "max_allowed_mb": conditions["max_content_size_mb"]
                    },
                    timestamp=time.time(),
                    context=f"Content size {content_size_mb:.2f}MB exceeds limit",
                    remediation_suggested="Reduce content size or increase limit"
                )

        return None

    async def _evaluate_file_type_rule(
        self,
        rule: SecurityRule,
        metadata: Dict[str, Any]
    ) -> Optional[PolicyViolation]:
        """Evaluace file type validation pravidla"""
        conditions = rule.conditions
        file_type = metadata.get("file_type", "").lower()

        if "allowed_file_types" in conditions and file_type:
            if file_type not in conditions["allowed_file_types"]:
                return PolicyViolation(
                    violation_id=f"{rule.rule_id}_{int(time.time())}",
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    severity=rule.severity,
                    action_taken=rule.action,
                    violation_details={
                        "file_type": file_type,
                        "allowed_types": conditions["allowed_file_types"]
                    },
                    timestamp=time.time(),
                    context=f"File type {file_type} not in allowed list",
                    remediation_suggested="Add file type to allowed list or verify safety"
                )

        return None

    async def _get_domain_reputation(self, domain: str) -> float:
        """Mock domain reputation check"""
        # V produkci by se použila skutečná reputation služba
        known_good_domains = {
            "wikipedia.org": 0.95,
            "github.com": 0.90,
            "stackoverflow.com": 0.85,
            "google.com": 0.95
        }

        known_bad_domains = {
            "malware-site.com": 0.1,
            "phishing-example.net": 0.05,
            "suspicious-domain.org": 0.2
        }

        if domain in known_good_domains:
            return known_good_domains[domain]
        elif domain in known_bad_domains:
            return known_bad_domains[domain]
        else:
            # Default reputation for unknown domains
            return 0.7

    def _determine_action(self, violations: List[PolicyViolation]) -> PolicyAction:
        """Určení finální akce na základě violations"""
        if not violations:
            return PolicyAction.ALLOW

        # Najdi nejhorší akci
        action_priority = {
            PolicyAction.ALLOW: 0,
            PolicyAction.LOG_ONLY: 1,
            PolicyAction.WARN: 2,
            PolicyAction.QUARANTINE: 3,
            PolicyAction.BLOCK: 4
        }

        max_action = PolicyAction.ALLOW
        max_priority = 0

        for violation in violations:
            priority = action_priority.get(violation.action_taken, 0)
            if priority > max_priority:
                max_priority = priority
                max_action = violation.action_taken

        return max_action

    def _calculate_confidence(self, violations: List[PolicyViolation]) -> float:
        """Výpočet confidence score"""
        if not violations:
            return 1.0

        # Confidence klesá s počtem a závažností violations
        severity_weights = {
            PolicySeverity.LOW: 0.1,
            PolicySeverity.MEDIUM: 0.3,
            PolicySeverity.HIGH: 0.6,
            PolicySeverity.CRITICAL: 1.0
        }

        total_weight = sum(severity_weights.get(v.severity, 0.5) for v in violations)
        confidence = max(0.0, 1.0 - (total_weight / 10.0))  # Scale to 0-1

        return confidence

    def _update_statistics(self, violations: List[PolicyViolation]) -> None:
        """Aktualizace statistik"""
        self.policy_stats["total_evaluations"] += 1

        if violations:
            self.policy_stats["violations_detected"] += len(violations)

            for violation in violations:
                rule_id = violation.rule_id
                if rule_id not in self.policy_stats["rules_triggered"]:
                    self.policy_stats["rules_triggered"][rule_id] = 0
                self.policy_stats["rules_triggered"][rule_id] += 1

                if violation.action_taken == PolicyAction.BLOCK:
                    self.policy_stats["blocks_enforced"] += 1
                elif violation.action_taken == PolicyAction.WARN:
                    self.policy_stats["warnings_issued"] += 1

        # Udržuj historii violations (posledních 1000)
        self.violation_history.extend(violations)
        if len(self.violation_history) > 1000:
            self.violation_history = self.violation_history[-1000:]

    def add_rule(self, rule: SecurityRule) -> None:
        """Přidání nového pravidla"""
        rule.updated_at = time.time()
        self.rules[rule.rule_id] = rule
        logger.info(f"Added security rule: {rule.name} ({rule.rule_id})")

    def remove_rule(self, rule_id: str) -> bool:
        """Odstranění pravidla"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed security rule: {rule_id}")
            return True
        return False

    def enable_rule(self, rule_id: str) -> bool:
        """Povolení pravidla"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
            self.rules[rule_id].updated_at = time.time()
            return True
        return False

    def disable_rule(self, rule_id: str) -> bool:
        """Zakázání pravidla"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
            self.rules[rule_id].updated_at = time.time()
            return True
        return False

    def get_policy_stats(self) -> Dict[str, Any]:
        """Získání statistik policy engine"""
        recent_violations = [
            v for v in self.violation_history
            if time.time() - v.timestamp < 3600  # Last hour
        ]

        return {
            **self.policy_stats,
            "active_rules": len([r for r in self.rules.values() if r.enabled]),
            "total_rules": len(self.rules),
            "whitelisted_domains": len(self.whitelisted_domains),
            "blacklisted_domains": len(self.blacklisted_domains),
            "recent_violations_1h": len(recent_violations),
            "violation_rate": len(recent_violations) / max(self.policy_stats["total_evaluations"], 1)
        }


# Factory funkce
def create_security_policy_engine(**kwargs) -> SecurityPolicyEngine:
    """Factory pro vytvoření security policy engine"""
    return SecurityPolicyEngine(**kwargs)


# Demo usage
if __name__ == "__main__":
    async def demo():
        engine = SecurityPolicyEngine()

        # Test URL validation
        test_urls = [
            "https://wikipedia.org/page",
            "https://malware-site.com/payload",
            "https://unknown-domain.com/content"
        ]

        for url in test_urls:
            result = await engine.validate_url(url)
            print(f"\nURL: {url}")
            print(f"Allowed: {result.allowed}")
            print(f"Action: {result.action_required.value}")
            print(f"Violations: {len(result.violations)}")

            for violation in result.violations:
                print(f"  - {violation.rule_name}: {violation.severity.value}")

        # Test content validation
        test_content = """
        Contact me at john.doe@example.com
        SSN: 123-45-6789
        Credit Card: 4111-1111-1111-1111
        """

        result = await engine.validate_content(test_content)
        print(f"\nContent validation:")
        print(f"Allowed: {result.allowed}")
        print(f"Violations: {len(result.violations)}")

        # Statistics
        stats = engine.get_policy_stats()
        print(f"\nPolicy stats: {stats}")

    asyncio.run(demo())
