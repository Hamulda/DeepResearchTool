#!/usr/bin/env python3
"""Enhanced Security Module for Deep Research Tool
Implements comprehensive security measures for sensitive research operations
"""

import base64
from datetime import datetime
import logging
import os
import re
import secrets
from typing import Any

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class SecureSecretsManager:
    """Enhanced secrets management with encryption and rotation"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.master_key = self._derive_master_key()
        self.cipher_suite = Fernet(self.master_key)
        self.secrets_cache = {}
        self.cache_ttl = config.get("secrets_ttl", 3600)  # 1 hour default

    def _derive_master_key(self) -> bytes:
        """Derive master encryption key from environment"""
        password = os.environ.get("DEEPRESEARCH_MASTER_KEY")
        if not password:
            raise ValueError("DEEPRESEARCH_MASTER_KEY environment variable required")

        # Use a fixed salt for consistency (in production, use secure key storage)
        salt = b"deepresearch_salt_2024"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    def encrypt_secret(self, plaintext: str) -> str:
        """Encrypt sensitive data"""
        encrypted = self.cipher_suite.encrypt(plaintext.encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt_secret(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            decoded = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.cipher_suite.decrypt(decoded)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt secret: {e}")
            raise ValueError("Invalid encrypted data")

    def get_secret(self, key: str, default: str | None = None) -> str | None:
        """Get secret with caching and TTL"""
        # Check cache first
        if key in self.secrets_cache:
            cached_data, timestamp = self.secrets_cache[key]
            if datetime.now().timestamp() - timestamp < self.cache_ttl:
                return cached_data

        # Get from environment
        secret_value = os.environ.get(key, default)
        if secret_value:
            # Cache for future use
            self.secrets_cache[key] = (secret_value, datetime.now().timestamp())
            return secret_value

        return None

    def rotate_secrets(self) -> dict[str, str]:
        """Generate new secrets for rotation"""
        new_secrets = {}

        # Generate new API keys
        new_secrets["API_SECRET_KEY"] = secrets.token_urlsafe(32)
        new_secrets["JWT_SECRET"] = secrets.token_urlsafe(64)
        new_secrets["ENCRYPTION_KEY"] = Fernet.generate_key().decode()

        # Generate new database passwords
        new_secrets["DB_PASSWORD"] = self._generate_strong_password(16)
        new_secrets["REDIS_PASSWORD"] = self._generate_strong_password(12)

        return new_secrets

    def _generate_strong_password(self, length: int) -> str:
        """Generate cryptographically strong password"""
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))


class InputValidator:
    """Enhanced input validation and sanitization"""

    def __init__(self):
        # URL validation patterns
        self.url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        # SQL injection patterns to block
        self.sql_injection_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(--|/\*|\*/|;)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
            r"(\'\s*(OR|AND)\s*\')",
        ]

        # XSS patterns to block
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>",
        ]

    def validate_url(self, url: str) -> bool:
        """Validate URL format and security"""
        if not url or not isinstance(url, str):
            return False

        # Basic format validation
        if not self.url_pattern.match(url):
            return False

        # Security checks
        url_lower = url.lower()

        # Block dangerous protocols
        dangerous_protocols = ['file://', 'ftp://', 'gopher://', 'ldap://']
        if any(url_lower.startswith(proto) for proto in dangerous_protocols):
            return False

        # Block internal/private IPs in production
        if self._is_internal_ip(url):
            logger.warning(f"Blocked access to internal IP: {url}")
            return False

        # Block suspicious domains
        suspicious_domains = ['localhost', '127.0.0.1', '0.0.0.0', '10.', '192.168.', '172.']
        if any(domain in url_lower for domain in suspicious_domains):
            logger.warning(f"Blocked suspicious domain: {url}")
            return False

        return True

    def _is_internal_ip(self, url: str) -> bool:
        """Check if URL points to internal/private IP"""
        from ipaddress import IPv4Address, ip_address
        import re

        # Extract IP from URL
        ip_match = re.search(r'://(\d+\.\d+\.\d+\.\d+)', url)
        if not ip_match:
            return False

        try:
            ip = ip_address(ip_match.group(1))
            if isinstance(ip, IPv4Address):
                return ip.is_private or ip.is_loopback or ip.is_link_local
        except ValueError:
            pass

        return False

    def sanitize_sql_input(self, input_str: str) -> str:
        """Sanitize input to prevent SQL injection"""
        if not input_str:
            return ""

        # Remove potentially dangerous SQL patterns
        sanitized = input_str
        for pattern in self.sql_injection_patterns:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)

        # Escape single quotes
        sanitized = sanitized.replace("'", "''")

        return sanitized

    def sanitize_html_input(self, input_str: str) -> str:
        """Sanitize input to prevent XSS"""
        if not input_str:
            return ""

        # Remove XSS patterns
        sanitized = input_str
        for pattern in self.xss_patterns:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)

        # Escape HTML entities
        sanitized = sanitized.replace("&", "&amp;")
        sanitized = sanitized.replace("<", "&lt;")
        sanitized = sanitized.replace(">", "&gt;")
        sanitized = sanitized.replace('"', "&quot;")
        sanitized = sanitized.replace("'", "&#x27;")

        return sanitized

    def validate_research_query(self, query: str) -> dict[str, Any]:
        """Validate research query for security and compliance"""
        if not query or not isinstance(query, str):
            return {"valid": False, "reason": "Empty or invalid query"}

        # Length check
        if len(query) > 1000:
            return {"valid": False, "reason": "Query too long"}

        # Check for sensitive patterns
        sensitive_patterns = [
            r"\b(password|passwd|secret|key|token)\b",
            r"\b(ssn|social.security|credit.card)\b",
            r"\b(hack|exploit|vulnerability|malware)\b",
        ]

        for pattern in sensitive_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                logger.warning(f"Potentially sensitive query blocked: {query[:50]}...")
                return {"valid": False, "reason": "Query contains sensitive terms"}

        # Check for illegal content patterns
        illegal_patterns = [
            r"\b(illegal|piracy|terrorism|weapons)\b",
            r"\b(drugs|narcotics|trafficking)\b",
        ]

        for pattern in illegal_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                logger.warning(f"Illegal content query blocked: {query[:50]}...")
                return {"valid": False, "reason": "Query contains illegal content terms"}

        return {"valid": True, "sanitized_query": self.sanitize_html_input(query)}


class SecurityAuditLogger:
    """Enhanced security event logging"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.log_level = config.get("security_log_level", "INFO")
        self.audit_logger = logging.getLogger("security_audit")

        # Configure separate security log file
        if config.get("security_log_file"):
            handler = logging.FileHandler(config["security_log_file"])
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s - %(extra_data)s'
            )
            handler.setFormatter(formatter)
            self.audit_logger.addHandler(handler)

    def log_access_attempt(self, ip_address: str, user_agent: str, endpoint: str):
        """Log access attempts for monitoring"""
        self.audit_logger.info(
            f"Access attempt: {endpoint}",
            extra={
                "extra_data": {
                    "ip_address": ip_address,
                    "user_agent": user_agent,
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "access_attempt"
                }
            }
        )

    def log_security_violation(self, violation_type: str, details: dict[str, Any]):
        """Log security violations"""
        self.audit_logger.warning(
            f"Security violation: {violation_type}",
            extra={
                "extra_data": {
                    **details,
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "security_violation"
                }
            }
        )

    def log_authentication_event(self, user_id: str, event_type: str, success: bool):
        """Log authentication events"""
        level = "INFO" if success else "WARNING"
        getattr(self.audit_logger, level.lower())(
            f"Authentication {event_type}: {'SUCCESS' if success else 'FAILED'}",
            extra={
                "extra_data": {
                    "user_id": user_id,
                    "event_type": f"auth_{event_type}",
                    "success": success,
                    "timestamp": datetime.now().isoformat()
                }
            }
        )


class TorSecurityManager:
    """Enhanced security for Tor/I2P operations"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.allowed_exit_nodes = config.get("tor", {}).get("allowed_exit_nodes", [])
        self.blocked_exit_nodes = config.get("tor", {}).get("blocked_exit_nodes", [])

    def validate_tor_connection(self, connection_info: dict[str, Any]) -> bool:
        """Validate Tor connection security"""
        exit_node = connection_info.get("exit_node")
        circuit_id = connection_info.get("circuit_id")

        # Check exit node allowlist/blocklist
        if self.allowed_exit_nodes and exit_node not in self.allowed_exit_nodes:
            logger.warning(f"Exit node not in allowlist: {exit_node}")
            return False

        if exit_node in self.blocked_exit_nodes:
            logger.warning(f"Exit node in blocklist: {exit_node}")
            return False

        # Validate circuit freshness
        circuit_age = connection_info.get("circuit_age_seconds", 0)
        max_circuit_age = self.config.get("tor", {}).get("max_circuit_age", 600)

        if circuit_age > max_circuit_age:
            logger.warning(f"Circuit too old: {circuit_age}s > {max_circuit_age}s")
            return False

        return True

    def get_security_headers(self) -> dict[str, str]:
        """Get security headers for Tor requests"""
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:102.0) Gecko/20100101 Firefox/102.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
        }


class ComplianceManager:
    """Legal and regulatory compliance management"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.jurisdictions = config.get("compliance", {}).get("jurisdictions", ["US", "EU"])
        self.data_retention_days = config.get("compliance", {}).get("data_retention_days", 90)

        # Load domain whitelists/blacklists
        self.load_compliance_lists()

    def load_compliance_lists(self):
        """Load compliance-related domain lists"""
        try:
            # Load from config files
            import json

            whitelist_file = self.config.get("compliance", {}).get("whitelist_file")
            if whitelist_file and os.path.exists(whitelist_file):
                with open(whitelist_file) as f:
                    self.domain_whitelist = set(json.load(f))
            else:
                self.domain_whitelist = set()

            blacklist_file = self.config.get("compliance", {}).get("blacklist_file")
            if blacklist_file and os.path.exists(blacklist_file):
                with open(blacklist_file) as f:
                    self.domain_blacklist = set(json.load(f))
            else:
                self.domain_blacklist = set()

        except Exception as e:
            logger.error(f"Failed to load compliance lists: {e}")
            self.domain_whitelist = set()
            self.domain_blacklist = set()

    def check_domain_compliance(self, domain: str) -> dict[str, Any]:
        """Check if domain access is compliant"""
        domain_lower = domain.lower()

        # Check blacklist first
        if domain_lower in self.domain_blacklist:
            return {
                "compliant": False,
                "reason": "Domain in compliance blacklist",
                "action": "block"
            }

        # Check whitelist if configured
        if self.domain_whitelist and domain_lower not in self.domain_whitelist:
            return {
                "compliant": False,
                "reason": "Domain not in compliance whitelist",
                "action": "block"
            }

        # Additional jurisdiction-specific checks
        if "EU" in self.jurisdictions:
            # GDPR compliance checks
            if self._requires_gdpr_consent(domain):
                return {
                    "compliant": True,
                    "reason": "GDPR consent required",
                    "action": "consent_required"
                }

        return {"compliant": True, "action": "allow"}

    def _requires_gdpr_consent(self, domain: str) -> bool:
        """Check if domain requires GDPR consent"""
        eu_domains = [".eu", ".de", ".fr", ".it", ".es", ".nl", ".be"]
        return any(domain.endswith(tld) for tld in eu_domains)

    def get_data_retention_policy(self) -> dict[str, Any]:
        """Get data retention policy"""
        return {
            "retention_days": self.data_retention_days,
            "auto_delete": True,
            "anonymization_days": max(30, self.data_retention_days // 3),
            "backup_retention_days": self.data_retention_days * 2
        }


def create_security_manager(config: dict[str, Any]) -> dict[str, Any]:
    """Factory function to create security components"""
    return {
        "secrets_manager": SecureSecretsManager(config),
        "input_validator": InputValidator(),
        "audit_logger": SecurityAuditLogger(config),
        "tor_security": TorSecurityManager(config),
        "compliance_manager": ComplianceManager(config)
    }
