"""FÁZE 7: Security Integration Module
Centrální integrace všech bezpečnostních komponentů
"""

import asyncio
from dataclasses import dataclass, field
import json
import logging
import time
from typing import Any

from .pii_redaction import PIIComplianceLogger, PIIRedactor, create_pii_redactor
from .rate_limiting import RateLimitConfig, RateLimitEngine
from .robots_compliance import DomainPolicy, RobotsComplianceEngine
from .secrets_manager import SecretsManager, get_secrets_manager
from .security_policies import SecurityPolicyEngine, SecurityRule

logger = logging.getLogger(__name__)


@dataclass
class SecurityCheckResult:
    """Výsledek kompletní bezpečnostní kontroly"""

    url: str
    allowed: bool
    overall_confidence: float
    processing_time_ms: float

    # Jednotlivé výsledky
    robots_result: dict[str, Any] | None = None
    rate_limit_result: dict[str, Any] | None = None
    policy_result: dict[str, Any] | None = None
    pii_result: dict[str, Any] | None = None

    # Souhrnné informace
    violations: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    actions_required: list[str] = field(default_factory=list)


@dataclass
class SecurityConfig:
    """Konfigurace bezpečnostního systému"""

    enable_robots_compliance: bool = True
    enable_rate_limiting: bool = True
    enable_policy_enforcement: bool = True
    enable_pii_protection: bool = True
    enable_secrets_management: bool = True

    # Timeouts
    robots_timeout_ms: int = 5000
    policy_timeout_ms: int = 3000

    # Default actions
    default_action_on_timeout: str = "warn"
    default_action_on_error: str = "allow"


class SecurityOrchestrator:
    """FÁZE 7: Centrální bezpečnostní orchestrátor

    Integruje všechny bezpečnostní komponenty do jednotného API:
    - Robots.txt compliance
    - Rate limiting s backoff
    - PII redaction
    - Security policies
    - Secrets management
    """

    def __init__(self, config: SecurityConfig | None = None):
        self.config = config or SecurityConfig()

        # Initialize components
        self.robots_engine: RobotsComplianceEngine | None = None
        self.rate_limiter: RateLimitEngine | None = None
        self.policy_engine: SecurityPolicyEngine | None = None
        self.pii_redactor: PIIRedactor | None = None
        self.secrets_manager: SecretsManager | None = None
        self.pii_logger: PIIComplianceLogger | None = None

        # Initialize enabled components
        self._initialize_components()

        # Statistics
        self.security_stats = {
            "total_checks": 0,
            "blocked_requests": 0,
            "warnings_issued": 0,
            "pii_redactions": 0,
            "rate_limit_hits": 0,
            "policy_violations": 0,
            "robots_blocks": 0,
        }

        logger.info("SecurityOrchestrator initialized with all FÁZE 7 components")

    def _initialize_components(self) -> None:
        """Inicializace bezpečnostních komponentů"""
        if self.config.enable_secrets_management:
            self.secrets_manager = get_secrets_manager()

        if self.config.enable_robots_compliance:
            self.robots_engine = RobotsComplianceEngine(cache_ttl_hours=24, default_crawl_delay=1.0)

        if self.config.enable_rate_limiting:
            self.rate_limiter = RateLimitEngine(
                default_requests_per_minute=30, default_requests_per_hour=1000
            )

        if self.config.enable_policy_enforcement:
            self.policy_engine = SecurityPolicyEngine()

        if self.config.enable_pii_protection:
            pii_config = {}
            if self.secrets_manager:
                # Load PII config from secrets if available
                hash_salt = self.secrets_manager.get_secret("pii_hash_salt")
                if hash_salt:
                    pii_config = {"pii_redaction": {"hash_salt": hash_salt}}

            self.pii_redactor = create_pii_redactor(pii_config)
            self.pii_logger = PIIComplianceLogger()

    async def check_url_security(self, url: str) -> SecurityCheckResult:
        """Kompletní bezpečnostní kontrola URL
        """
        start_time = time.time()
        violations = []
        warnings = []
        actions_required = []
        overall_allowed = True

        result = SecurityCheckResult(
            url=url, allowed=True, overall_confidence=1.0, processing_time_ms=0
        )

        try:
            # 1. Robots.txt compliance check
            if self.config.enable_robots_compliance and self.robots_engine:
                robots_allowed, robots_reason = await asyncio.wait_for(
                    self.robots_engine.is_url_allowed(url),
                    timeout=self.config.robots_timeout_ms / 1000,
                )

                result.robots_result = {"allowed": robots_allowed, "reason": robots_reason}

                if not robots_allowed:
                    overall_allowed = False
                    violations.append(
                        {"type": "robots_violation", "message": robots_reason, "action": "block"}
                    )
                    self.security_stats["robots_blocks"] += 1

            # 2. Rate limiting check
            if self.config.enable_rate_limiting and self.rate_limiter:
                rate_limit_result = await self.rate_limiter.check_rate_limit(url)

                result.rate_limit_result = {
                    "allowed": rate_limit_result.allowed,
                    "wait_time": rate_limit_result.wait_time,
                    "reason": rate_limit_result.reason,
                }

                if not rate_limit_result.allowed:
                    if rate_limit_result.wait_time > 0:
                        warnings.append(
                            f"Rate limit exceeded, wait {rate_limit_result.wait_time:.1f}s"
                        )
                        actions_required.append("apply_backoff")
                    else:
                        overall_allowed = False
                        violations.append(
                            {
                                "type": "rate_limit_violation",
                                "message": rate_limit_result.reason,
                                "action": "block",
                            }
                        )
                    self.security_stats["rate_limit_hits"] += 1

            # 3. Security policy check
            if self.config.enable_policy_enforcement and self.policy_engine:
                policy_result = await asyncio.wait_for(
                    self.policy_engine.validate_url(url),
                    timeout=self.config.policy_timeout_ms / 1000,
                )

                result.policy_result = {
                    "allowed": policy_result.allowed,
                    "violations": [
                        {
                            "rule_name": v.rule_name,
                            "severity": v.severity.value,
                            "action": v.action_taken.value,
                        }
                        for v in policy_result.violations
                    ],
                    "action_required": policy_result.action_required.value,
                }

                if not policy_result.allowed:
                    overall_allowed = False
                    for violation in policy_result.violations:
                        violations.append(
                            {
                                "type": "policy_violation",
                                "rule": violation.rule_name,
                                "severity": violation.severity.value,
                                "action": violation.action_taken.value,
                            }
                        )
                    self.security_stats["policy_violations"] += len(policy_result.violations)

        except TimeoutError:
            warnings.append("Security check timeout - applying default action")
            if self.config.default_action_on_timeout == "block":
                overall_allowed = False

        except Exception as e:
            logger.error(f"Error in security check for {url}: {e}")
            warnings.append(f"Security check error: {e!s}")
            if self.config.default_action_on_error == "block":
                overall_allowed = False

        # Finalize result
        result.allowed = overall_allowed
        result.violations = violations
        result.warnings = warnings
        result.actions_required = actions_required
        result.overall_confidence = self._calculate_overall_confidence(result)
        result.processing_time_ms = (time.time() - start_time) * 1000

        # Update statistics
        self.security_stats["total_checks"] += 1
        if not overall_allowed:
            self.security_stats["blocked_requests"] += 1
        if warnings:
            self.security_stats["warnings_issued"] += 1

        return result

    async def check_content_security(
        self, content: str, metadata: dict[str, Any] | None = None
    ) -> SecurityCheckResult:
        """Bezpečnostní kontrola obsahu
        """
        start_time = time.time()
        violations = []
        warnings = []
        actions_required = []
        overall_allowed = True

        result = SecurityCheckResult(
            url="content_check", allowed=True, overall_confidence=1.0, processing_time_ms=0
        )

        try:
            # 1. PII detection and redaction
            if self.config.enable_pii_protection and self.pii_redactor:
                pii_result = self.pii_redactor.redact_text(content)

                result.pii_result = {
                    "pii_instances_found": len(pii_result.matches),
                    "pii_types": list(set(m.pii_type.value for m in pii_result.matches)),
                    "redaction_applied": len(pii_result.matches) > 0,
                    "redacted_content_length": len(pii_result.redacted_text),
                }

                if pii_result.matches:
                    self.security_stats["pii_redactions"] += len(pii_result.matches)
                    warnings.append(f"Found {len(pii_result.matches)} PII instances")
                    actions_required.append("apply_pii_redaction")

                # Log PII operation
                if self.pii_logger:
                    self.pii_logger.log_redaction_operation(pii_result, "content_security_check")

            # 2. Policy validation
            if self.config.enable_policy_enforcement and self.policy_engine:
                policy_result = await self.policy_engine.validate_content(content, metadata)

                result.policy_result = {
                    "allowed": policy_result.allowed,
                    "violations": [
                        {
                            "rule_name": v.rule_name,
                            "severity": v.severity.value,
                            "action": v.action_taken.value,
                        }
                        for v in policy_result.violations
                    ],
                }

                if not policy_result.allowed:
                    overall_allowed = False
                    for violation in policy_result.violations:
                        violations.append(
                            {
                                "type": "content_policy_violation",
                                "rule": violation.rule_name,
                                "severity": violation.severity.value,
                            }
                        )

        except Exception as e:
            logger.error(f"Error in content security check: {e}")
            warnings.append(f"Content security check error: {e!s}")

        # Finalize result
        result.allowed = overall_allowed
        result.violations = violations
        result.warnings = warnings
        result.actions_required = actions_required
        result.overall_confidence = self._calculate_overall_confidence(result)
        result.processing_time_ms = (time.time() - start_time) * 1000

        return result

    async def apply_rate_limiting(self, url: str) -> bool:
        """Aplikace rate limitingu s čekáním pokud je potřeba
        """
        if not self.rate_limiter:
            return True

        return await self.rate_limiter.wait_if_needed(url)

    def redact_pii_from_text(self, text: str, language: str = "en") -> str:
        """Redakce PII z textu
        """
        if not self.pii_redactor:
            return text

        result = self.pii_redactor.redact_text(text, language)

        if result.matches and self.pii_logger:
            self.pii_logger.log_redaction_operation(result, "text_redaction")

        return result.redacted_text

    def get_secret(self, name: str, default: str | None = None) -> str | None:
        """Získání tajemství ze secrets manageru
        """
        if not self.secrets_manager:
            return default

        return self.secrets_manager.get_secret(name, default)

    def _calculate_overall_confidence(self, result: SecurityCheckResult) -> float:
        """Výpočet celkové confidence"""
        base_confidence = 1.0

        # Snižuj confidence podle počtu violations
        if result.violations:
            confidence_reduction = len(result.violations) * 0.2
            base_confidence -= confidence_reduction

        # Snižuj confidence podle warnings
        if result.warnings:
            confidence_reduction = len(result.warnings) * 0.1
            base_confidence -= confidence_reduction

        return max(0.0, min(1.0, base_confidence))

    async def configure_domain_policies(self, policies: list[DomainPolicy]) -> None:
        """Konfigurace domain policies"""
        if self.robots_engine:
            for policy in policies:
                self.robots_engine.add_domain_policy(policy)

    async def configure_rate_limits(self, configs: list[RateLimitConfig]) -> None:
        """Konfigurace rate limitů"""
        if self.rate_limiter:
            for config in configs:
                self.rate_limiter.add_domain_config(config)

    def add_security_rule(self, rule: SecurityRule) -> None:
        """Přidání bezpečnostního pravidla"""
        if self.policy_engine:
            self.policy_engine.add_rule(rule)

    def get_security_dashboard(self) -> dict[str, Any]:
        """Bezpečnostní dashboard s přehledem všech komponentů
        """
        dashboard = {
            "timestamp": time.time(),
            "overall_stats": self.security_stats,
            "components": {},
        }

        # Robots compliance stats
        if self.robots_engine:
            dashboard["components"]["robots_compliance"] = self.robots_engine.get_compliance_stats()

        # Rate limiting stats
        if self.rate_limiter:
            dashboard["components"]["rate_limiting"] = self.rate_limiter.get_global_stats()

        # Policy engine stats
        if self.policy_engine:
            dashboard["components"]["security_policies"] = self.policy_engine.get_policy_stats()

        # Secrets manager stats
        if self.secrets_manager:
            dashboard["components"]["secrets_management"] = self.secrets_manager.get_secrets_stats()

        return dashboard

    async def start(self) -> None:
        """Spuštění všech bezpečnostních komponentů"""
        if self.rate_limiter:
            await self.rate_limiter.start_cleanup_task()

        if self.robots_engine:
            # Robots engine nemá background tasky v current implementaci
            pass

        logger.info("Security orchestrator started successfully")

    async def stop(self) -> None:
        """Zastavení všech bezpečnostních komponentů"""
        if self.rate_limiter:
            await self.rate_limiter.stop_cleanup_task()

        logger.info("Security orchestrator stopped")


# Factory funkce pro snadné použití
def create_security_orchestrator(config: SecurityConfig | None = None) -> SecurityOrchestrator:
    """Factory pro vytvoření security orchestrator"""
    return SecurityOrchestrator(config)


# Demo usage
if __name__ == "__main__":

    async def demo():
        # Vytvoření security orchestrator
        config = SecurityConfig(
            enable_robots_compliance=True,
            enable_rate_limiting=True,
            enable_policy_enforcement=True,
            enable_pii_protection=True,
            enable_secrets_management=True,
        )

        orchestrator = create_security_orchestrator(config)
        await orchestrator.start()

        try:
            # Test URL security check
            test_urls = [
                "https://wikipedia.org/safe-page",
                "https://malware-site.com/dangerous",
                "https://example.com/api/data",
            ]

            print("=== URL Security Checks ===")
            for url in test_urls:
                result = await orchestrator.check_url_security(url)
                print(f"\nURL: {url}")
                print(f"Allowed: {result.allowed}")
                print(f"Confidence: {result.overall_confidence:.2f}")
                print(f"Processing time: {result.processing_time_ms:.1f}ms")
                print(f"Violations: {len(result.violations)}")

                if result.warnings:
                    print(f"Warnings: {result.warnings}")

            # Test content security check
            test_content = """
            Contact John Doe at john.doe@example.com
            Phone: +1-555-123-4567
            SSN: 123-45-6789
            This is some research content with PII.
            """

            print("\n=== Content Security Check ===")
            content_result = await orchestrator.check_content_security(test_content)
            print(f"Content allowed: {content_result.allowed}")
            print(
                f"PII instances: {content_result.pii_result['pii_instances_found'] if content_result.pii_result else 0}"
            )

            # Test PII redaction
            print("\n=== PII Redaction ===")
            redacted = orchestrator.redact_pii_from_text(test_content)
            print(f"Original length: {len(test_content)}")
            print(f"Redacted length: {len(redacted)}")
            print(f"Redacted content preview: {redacted[:100]}...")

            # Security dashboard
            print("\n=== Security Dashboard ===")
            dashboard = orchestrator.get_security_dashboard()
            print(json.dumps(dashboard, indent=2, default=str))

        finally:
            await orchestrator.stop()

    asyncio.run(demo())
