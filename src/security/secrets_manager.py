"""FÁZE 7: Secrets Management System
Ochrana citlivých informací v konfiguraci s environment-based přístupem
"""

import base64
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import os
from pathlib import Path
import re
from typing import Any
import warnings

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class SecretType(Enum):
    """Typy tajemství"""

    API_KEY = "api_key"
    DATABASE_URL = "database_url"
    PASSWORD = "password"
    TOKEN = "token"
    CERTIFICATE = "certificate"
    PRIVATE_KEY = "private_key"
    ENCRYPTION_KEY = "encryption_key"
    WEBHOOK_SECRET = "webhook_secret"
    CONNECTION_STRING = "connection_string"


class SecretSource(Enum):
    """Zdroje tajemství"""

    ENVIRONMENT = "environment"
    FILE = "file"
    ENCRYPTED_FILE = "encrypted_file"
    EXTERNAL_VAULT = "external_vault"
    CONFIG = "config"


@dataclass
class SecretDefinition:
    """Definice tajemství"""

    name: str
    secret_type: SecretType
    source: SecretSource
    required: bool = True
    description: str = ""
    env_var: str | None = None
    file_path: str | None = None
    default_value: str | None = None
    validation_pattern: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class SecretValue:
    """Hodnota tajemství s metadata"""

    name: str
    value: str
    source: SecretSource
    is_encrypted: bool = False
    last_accessed: float | None = None
    access_count: int = 0


class SecretsManager:
    """FÁZE 7: Advanced Secrets Management System

    Features:
    - Environment-based konfigurace s fallback
    - Šifrování citlivých dat v konfigu
    - Secret rotation a expiration
    - Audit logging přístupů k tajemstvím
    - Validation a sanitization
    - Multiple sources (env, files, external vaults)
    """

    def __init__(
        self,
        master_key: str | None = None,
        secrets_file: Path | None = None,
        enable_encryption: bool = True,
        audit_access: bool = True,
    ):
        self.secrets_file = secrets_file or Path("secrets.enc")
        self.enable_encryption = enable_encryption
        self.audit_access = audit_access

        # Encryption setup
        self.master_key = master_key or os.getenv("DEEPRESEARCH_MASTER_KEY")
        self._cipher_suite = None
        if self.enable_encryption and self.master_key:
            self._setup_encryption()

        # Secret definitions
        self.secret_definitions: dict[str, SecretDefinition] = {}

        # Cached secrets
        self._secret_cache: dict[str, SecretValue] = {}

        # Access tracking
        self.access_log: list[dict[str, Any]] = []

        # Security patterns
        self.sensitive_patterns = [
            r"password",
            r"secret",
            r"key",
            r"token",
            r"credential",
            r"auth",
            r"api[_-]?key",
            r"access[_-]?token",
            r"private[_-]?key",
        ]

        # Load default secrets configuration
        self._load_default_secrets()

        logger.info(
            f"SecretsManager initialized with encryption={'enabled' if self._cipher_suite else 'disabled'}"
        )

    def _setup_encryption(self) -> None:
        """Nastavení šifrování"""
        try:
            # Derive key from master key
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b"deepresearch_salt",  # V produkci by měl být náhodný
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
            self._cipher_suite = Fernet(key)

        except Exception as e:
            logger.error(f"Failed to setup encryption: {e}")
            self._cipher_suite = None

    def _load_default_secrets(self) -> None:
        """Načtení výchozích definic tajemství"""
        default_secrets = [
            SecretDefinition(
                name="ollama_api_key",
                secret_type=SecretType.API_KEY,
                source=SecretSource.ENVIRONMENT,
                env_var="OLLAMA_API_KEY",
                required=False,
                description="Ollama API key for authenticated requests",
                validation_pattern=r"^[a-zA-Z0-9\-_]{20,}$",
            ),
            SecretDefinition(
                name="qdrant_api_key",
                secret_type=SecretType.API_KEY,
                source=SecretSource.ENVIRONMENT,
                env_var="QDRANT_API_KEY",
                required=False,
                description="Qdrant vector database API key",
                validation_pattern=r"^[a-zA-Z0-9\-_]{16,}$",
            ),
            SecretDefinition(
                name="openai_api_key",
                secret_type=SecretType.API_KEY,
                source=SecretSource.ENVIRONMENT,
                env_var="OPENAI_API_KEY",
                required=False,
                description="OpenAI API key for GPT models",
                validation_pattern=r"^sk-[a-zA-Z0-9]{48}$",
            ),
            SecretDefinition(
                name="database_url",
                secret_type=SecretType.DATABASE_URL,
                source=SecretSource.ENVIRONMENT,
                env_var="DATABASE_URL",
                required=False,
                description="Database connection URL",
                validation_pattern=r"^(postgresql|mysql|sqlite)://.+",
            ),
            SecretDefinition(
                name="jwt_secret",
                secret_type=SecretType.ENCRYPTION_KEY,
                source=SecretSource.ENVIRONMENT,
                env_var="JWT_SECRET",
                required=True,
                description="JWT signing secret",
                validation_pattern=r"^[a-zA-Z0-9\-_]{32,}$",
                default_value=self._generate_random_secret(32),
            ),
            SecretDefinition(
                name="webhook_secret",
                secret_type=SecretType.WEBHOOK_SECRET,
                source=SecretSource.ENVIRONMENT,
                env_var="WEBHOOK_SECRET",
                required=False,
                description="Webhook verification secret",
                validation_pattern=r"^[a-zA-Z0-9\-_]{16,}$",
            ),
            SecretDefinition(
                name="admin_password",
                secret_type=SecretType.PASSWORD,
                source=SecretSource.ENVIRONMENT,
                env_var="ADMIN_PASSWORD",
                required=False,
                description="Admin interface password",
                validation_pattern=r"^.{8,}$",  # Min 8 characters
            ),
        ]

        for secret_def in default_secrets:
            self.secret_definitions[secret_def.name] = secret_def

    def _generate_random_secret(self, length: int = 32) -> str:
        """Generování náhodného tajemství"""
        import secrets
        import string

        alphabet = string.ascii_letters + string.digits + "-_"
        return "".join(secrets.choice(alphabet) for _ in range(length))

    def add_secret_definition(self, definition: SecretDefinition) -> None:
        """Přidání definice tajemství"""
        self.secret_definitions[definition.name] = definition
        logger.info(f"Added secret definition: {definition.name}")

    def get_secret(self, name: str, default: str | None = None) -> str | None:
        """Získání hodnoty tajemství s audit loggingem
        """
        import time

        # Check cache first
        if name in self._secret_cache:
            secret_value = self._secret_cache[name]
            secret_value.access_count += 1
            secret_value.last_accessed = time.time()

            if self.audit_access:
                self._log_secret_access(name, secret_value.source, True)

            return secret_value.value

        # Get secret definition
        if name not in self.secret_definitions:
            logger.warning(f"No definition found for secret: {name}")
            return default

        definition = self.secret_definitions[name]

        # Try to load secret from various sources
        secret_value = self._load_secret_from_sources(definition)

        if secret_value:
            # Cache the secret
            self._secret_cache[name] = secret_value

            if self.audit_access:
                self._log_secret_access(name, secret_value.source, True)

            return secret_value.value

        # Use default if provided
        if default is not None:
            return default

        # Use definition default
        if definition.default_value:
            return definition.default_value

        # Required secret not found
        if definition.required:
            error_msg = f"Required secret '{name}' not found in any source"
            logger.error(error_msg)
            if self.audit_access:
                self._log_secret_access(name, SecretSource.ENVIRONMENT, False)
            raise ValueError(error_msg)

        return None

    def _load_secret_from_sources(self, definition: SecretDefinition) -> SecretValue | None:
        """Načtení tajemství z různých zdrojů"""
        import time

        value = None
        source = definition.source

        if definition.source == SecretSource.ENVIRONMENT:
            # Environment variable
            env_var = definition.env_var or definition.name.upper()
            value = os.getenv(env_var)

        elif definition.source == SecretSource.FILE:
            # Plain file
            if definition.file_path and Path(definition.file_path).exists():
                try:
                    with open(definition.file_path) as f:
                        value = f.read().strip()
                except Exception as e:
                    logger.error(f"Error reading secret from file {definition.file_path}: {e}")

        elif definition.source == SecretSource.ENCRYPTED_FILE:
            # Encrypted file
            value = self._load_encrypted_secret(definition.name)

        if value and self._validate_secret(value, definition):
            return SecretValue(
                name=definition.name,
                value=value,
                source=source,
                is_encrypted=(source == SecretSource.ENCRYPTED_FILE),
                last_accessed=time.time(),
                access_count=1,
            )

        return None

    def _validate_secret(self, value: str, definition: SecretDefinition) -> bool:
        """Validace tajemství podle pattern"""
        if not definition.validation_pattern:
            return True

        try:
            return bool(re.match(definition.validation_pattern, value))
        except Exception as e:
            logger.error(f"Error validating secret {definition.name}: {e}")
            return False

    def set_secret(
        self,
        name: str,
        value: str,
        encrypt: bool = True,
        source: SecretSource = SecretSource.ENCRYPTED_FILE,
    ) -> bool:
        """Uložení tajemství"""
        import time

        try:
            if encrypt and source == SecretSource.ENCRYPTED_FILE:
                self._save_encrypted_secret(name, value)

            # Update cache
            secret_value = SecretValue(
                name=name,
                value=value,
                source=source,
                is_encrypted=encrypt,
                last_accessed=time.time(),
                access_count=0,
            )
            self._secret_cache[name] = secret_value

            logger.info(f"Secret '{name}' saved successfully")
            return True

        except Exception as e:
            logger.error(f"Error saving secret '{name}': {e}")
            return False

    def _save_encrypted_secret(self, name: str, value: str) -> None:
        """Uložení šifrovaného tajemství"""
        if not self._cipher_suite:
            raise ValueError("Encryption not available - no master key provided")

        # Load existing secrets
        secrets_data = {}
        if self.secrets_file.exists():
            secrets_data = self._load_encrypted_secrets_file()

        # Encrypt and save
        encrypted_value = self._cipher_suite.encrypt(value.encode()).decode()
        secrets_data[name] = encrypted_value

        # Save back to file
        self._save_encrypted_secrets_file(secrets_data)

    def _load_encrypted_secret(self, name: str) -> str | None:
        """Načtení šifrovaného tajemství"""
        if not self._cipher_suite:
            return None

        try:
            secrets_data = self._load_encrypted_secrets_file()
            if name in secrets_data:
                encrypted_value = secrets_data[name].encode()
                decrypted_value = self._cipher_suite.decrypt(encrypted_value).decode()
                return decrypted_value
        except Exception as e:
            logger.error(f"Error loading encrypted secret '{name}': {e}")

        return None

    def _load_encrypted_secrets_file(self) -> dict[str, str]:
        """Načtení šifrovaného souboru s tajemstvími"""
        if not self.secrets_file.exists():
            return {}

        try:
            with open(self.secrets_file) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading secrets file: {e}")
            return {}

    def _save_encrypted_secrets_file(self, secrets_data: dict[str, str]) -> None:
        """Uložení šifrovaného souboru s tajemstvími"""
        try:
            with open(self.secrets_file, "w") as f:
                json.dump(secrets_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving secrets file: {e}")

    def _log_secret_access(self, name: str, source: SecretSource, success: bool) -> None:
        """Audit logging přístupu k tajemstvím"""
        import time

        log_entry = {
            "timestamp": time.time(),
            "secret_name": name,
            "source": source.value,
            "success": success,
            "caller": self._get_caller_info(),
        }

        self.access_log.append(log_entry)

        # Keep only last 1000 entries
        if len(self.access_log) > 1000:
            self.access_log = self.access_log[-1000:]

        logger.debug(
            f"Secret access: {name} from {source.value} - {'success' if success else 'failed'}"
        )

    def _get_caller_info(self) -> str:
        """Získání informací o volajícím"""
        import inspect

        try:
            frame = inspect.currentframe()
            # Go up the stack to find the actual caller
            for _ in range(3):
                frame = frame.f_back
                if frame is None:
                    break

            if frame:
                return f"{frame.f_code.co_filename}:{frame.f_lineno}"
        except Exception:
            pass

        return "unknown"

    def scan_config_for_secrets(self, config: dict[str, Any]) -> list[str]:
        """Skenování konfigurace pro potenciální tajemství
        """
        found_secrets = []

        def scan_recursive(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key

                    # Check if key matches sensitive patterns
                    key_lower = key.lower()
                    if any(re.search(pattern, key_lower) for pattern in self.sensitive_patterns):
                        found_secrets.append(current_path)

                    scan_recursive(value, current_path)

            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    scan_recursive(item, f"{path}[{i}]")

        scan_recursive(config)
        return found_secrets

    def sanitize_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Sanitizace konfigurace - nahrazení tajemství placeholdery
        """
        import copy

        sanitized = copy.deepcopy(config)

        def sanitize_recursive(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key

                    # Check if key matches sensitive patterns
                    key_lower = key.lower()
                    if any(re.search(pattern, key_lower) for pattern in self.sensitive_patterns):
                        if isinstance(value, str) and len(value) > 0:
                            obj[key] = "[REDACTED]"
                    else:
                        sanitize_recursive(value, current_path)

            elif isinstance(obj, list):
                for item in obj:
                    sanitize_recursive(item, path)

        sanitize_recursive(sanitized)
        return sanitized

    def rotate_secret(self, name: str) -> bool:
        """Rotace tajemství"""
        try:
            if name not in self.secret_definitions:
                logger.error(f"Cannot rotate unknown secret: {name}")
                return False

            definition = self.secret_definitions[name]

            # Generate new secret
            new_value = self._generate_random_secret()

            # Validate new secret
            if not self._validate_secret(new_value, definition):
                logger.error(f"Generated secret for {name} failed validation")
                return False

            # Save new secret
            return self.set_secret(name, new_value, encrypt=True)

        except Exception as e:
            logger.error(f"Error rotating secret {name}: {e}")
            return False

    def delete_secret(self, name: str) -> bool:
        """Smazání tajemství"""
        try:
            # Remove from cache
            if name in self._secret_cache:
                del self._secret_cache[name]

            # Remove from encrypted file
            if self.secrets_file.exists():
                secrets_data = self._load_encrypted_secrets_file()
                if name in secrets_data:
                    del secrets_data[name]
                    self._save_encrypted_secrets_file(secrets_data)

            logger.info(f"Secret '{name}' deleted successfully")
            return True

        except Exception as e:
            logger.error(f"Error deleting secret '{name}': {e}")
            return False

    def list_secrets(self, include_values: bool = False) -> dict[str, Any]:
        """Seznam všech tajemství"""
        result = {}

        for name, definition in self.secret_definitions.items():
            secret_info = {
                "type": definition.secret_type.value,
                "source": definition.source.value,
                "required": definition.required,
                "description": definition.description,
                "has_value": name in self._secret_cache or self.get_secret(name) is not None,
            }

            if include_values and name in self._secret_cache:
                # Only show values if explicitly requested (for debugging)
                warnings.warn("Including secret values in list - use only for debugging!")
                secret_info["value"] = "[REDACTED]"  # Never actually show values

            result[name] = secret_info

        return result

    def get_secrets_stats(self) -> dict[str, Any]:
        """Statistiky secrets manageru"""
        import time

        recent_accesses = [
            entry
            for entry in self.access_log
            if time.time() - entry["timestamp"] < 3600  # Last hour
        ]

        return {
            "total_secrets_defined": len(self.secret_definitions),
            "secrets_in_cache": len(self._secret_cache),
            "encryption_enabled": self._cipher_suite is not None,
            "total_accesses": len(self.access_log),
            "recent_accesses_1h": len(recent_accesses),
            "successful_accesses": len([e for e in recent_accesses if e["success"]]),
            "failed_accesses": len([e for e in recent_accesses if not e["success"]]),
        }


# Global secrets manager instance
_secrets_manager: SecretsManager | None = None


def get_secrets_manager() -> SecretsManager:
    """Získání globální instance secrets manageru"""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager


def get_secret(name: str, default: str | None = None) -> str | None:
    """Convenience funkce pro získání tajemství"""
    return get_secrets_manager().get_secret(name, default)


def set_secret(name: str, value: str, encrypt: bool = True) -> bool:
    """Convenience funkce pro uložení tajemství"""
    return get_secrets_manager().set_secret(name, value, encrypt)


# Demo usage
if __name__ == "__main__":
    # Demo secrets management

    # Set master key for testing
    os.environ["DEEPRESEARCH_MASTER_KEY"] = "test_master_key_123"

    manager = SecretsManager()

    # Test setting and getting secrets
    print("Setting test secret...")
    manager.set_secret("test_api_key", "sk-1234567890abcdef", encrypt=True)

    print("Getting secret...")
    api_key = manager.get_secret("test_api_key")
    print(f"Retrieved: {api_key[:10]}..." if api_key else "Failed to retrieve")

    # Test config scanning
    test_config = {
        "database": {"host": "localhost", "password": "secret123", "api_key": "sensitive_key"},
        "features": {"enabled": True, "webhook_secret": "webhook123"},
    }

    print("\nScanning config for secrets...")
    found_secrets = manager.scan_config_for_secrets(test_config)
    print(f"Found potential secrets: {found_secrets}")

    print("\nSanitized config:")
    sanitized = manager.sanitize_config(test_config)
    print(json.dumps(sanitized, indent=2))

    # Statistics
    stats = manager.get_secrets_stats()
    print(f"\nSecrets stats: {stats}")
