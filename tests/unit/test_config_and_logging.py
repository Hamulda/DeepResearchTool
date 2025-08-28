#!/usr/bin/env python3
"""
Unit testy pro konfiguraci a základní utility

Author: Senior Python/MLOps Agent
"""

import os
import pytest
from unittest.mock import patch, Mock
from pathlib import Path

from src.config.settings import (
    ApplicationSettings,
    DatabaseSettings,
    AIModelSettings,
    SecuritySettings,
    get_settings
)
from src.utils.logging import (
    StructuredLogger,
    AuditLogger,
    PerformanceLogger,
    get_logger,
    configure_logging
)


class TestApplicationSettings:
    """Testy pro ApplicationSettings"""

    def test_default_values(self):
        """Test výchozích hodnot"""
        settings = ApplicationSettings()

        assert settings.app_name == "DeepResearchTool"
        assert settings.app_version == "1.0.0"
        assert settings.environment == "development"
        assert settings.debug is False
        assert settings.host == "0.0.0.0"
        assert settings.port == 8080

    def test_environment_validation(self):
        """Test validace prostředí"""
        with pytest.raises(ValueError, match="Environment must be one of"):
            ApplicationSettings(environment="invalid")

    @patch.dict(os.environ, {
        "APP_NAME": "TestApp",
        "ENVIRONMENT": "production",
        "DEBUG": "true",
        "PORT": "9000"
    })
    def test_env_override(self):
        """Test načítání z environment variables"""
        settings = ApplicationSettings()

        assert settings.app_name == "TestApp"
        assert settings.environment == "production"
        assert settings.debug is True
        assert settings.port == 9000

    def test_directory_creation(self, tmp_path):
        """Test vytváření adresářů"""
        test_data_dir = tmp_path / "test_data"
        settings = ApplicationSettings(data_dir=test_data_dir)

        assert test_data_dir.exists()
        assert test_data_dir.is_dir()


class TestDatabaseSettings:
    """Testy pro DatabaseSettings"""

    def test_default_values(self):
        """Test výchozích hodnot"""
        settings = DatabaseSettings()

        assert settings.redis_host == "localhost"
        assert settings.redis_port == 6379
        assert settings.postgres_host == "localhost"
        assert settings.postgres_port == 5432

    @patch.dict(os.environ, {
        "DB_REDIS_HOST": "redis-server",
        "DB_REDIS_PORT": "6380",
        "DB_POSTGRES_PASSWORD": "secret123"
    })
    def test_env_prefix(self):
        """Test env prefix funktionality"""
        settings = DatabaseSettings()

        assert settings.redis_host == "redis-server"
        assert settings.redis_port == 6380
        assert settings.postgres_password.get_secret_value() == "secret123"


class TestAIModelSettings:
    """Testy pro AIModelSettings"""

    @patch.dict(os.environ, {
        "AI_OPENAI_API_KEY": "test-key-123",
        "AI_OPENAI_MODEL": "gpt-4-turbo",
        "AI_MAX_CONCURRENT_REQUESTS": "20"
    })
    def test_ai_settings(self):
        """Test AI nastavení"""
        settings = AIModelSettings()

        assert settings.openai_api_key.get_secret_value() == "test-key-123"
        assert settings.openai_model == "gpt-4-turbo"
        assert settings.max_concurrent_requests == 20


class TestSecuritySettings:
    """Testy pro SecuritySettings"""

    @patch.dict(os.environ, {
        "SECURITY_SECRET_KEY": "super-secret-key",
        "SECURITY_JWT_SECRET": "jwt-secret",
        "SECURITY_ENABLE_DATA_ENCRYPTION": "false"
    })
    def test_security_settings(self):
        """Test bezpečnostních nastavení"""
        settings = SecuritySettings()

        assert settings.secret_key.get_secret_value() == "super-secret-key"
        assert settings.jwt_secret.get_secret_value() == "jwt-secret"
        assert settings.enable_data_encryption is False


class TestStructuredLogger:
    """Testy pro StructuredLogger"""

    def test_logger_creation(self):
        """Test vytvoření loggeru"""
        logger = StructuredLogger("test")
        assert logger.context == {}

    def test_logger_with_context(self):
        """Test loggeru s kontextem"""
        logger = StructuredLogger("test", component="test_comp", version="1.0")
        assert logger.context["component"] == "test_comp"
        assert logger.context["version"] == "1.0"

    def test_logger_bind(self):
        """Test bind funkcionality"""
        logger = StructuredLogger("test")
        bound_logger = logger.bind(request_id="123", user_id="user1")

        assert bound_logger.context["request_id"] == "123"
        assert bound_logger.context["user_id"] == "user1"

    @patch('structlog.get_logger')
    def test_log_methods(self, mock_get_logger):
        """Test log metod"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        logger = StructuredLogger("test")

        logger.info("Test message", extra_field="value")
        mock_logger.info.assert_called_once_with("Test message", extra_field="value")

        logger.error("Error message", error_code=500)
        mock_logger.error.assert_called_once_with("Error message", error_code=500)


class TestAuditLogger:
    """Testy pro AuditLogger"""

    @patch('structlog.get_logger')
    def test_user_action_logging(self, mock_get_logger):
        """Test logování uživatelských akcí"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        audit_logger = AuditLogger("auth")
        audit_logger.user_action(
            action="login",
            user_id="user123",
            resource="system"
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert call_args[0][0] == "User action performed"
        assert call_args[1]["action"] == "login"
        assert call_args[1]["user_id"] == "user123"

    @patch('structlog.get_logger')
    def test_security_event_logging(self, mock_get_logger):
        """Test logování bezpečnostních událostí"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        audit_logger = AuditLogger("security")
        audit_logger.security_event(
            event_type="failed_login",
            severity="medium",
            description="Multiple failed login attempts"
        )

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert call_args[0][0] == "Security event detected"
        assert call_args[1]["event_type"] == "failed_login"


class TestPerformanceLogger:
    """Testy pro PerformanceLogger"""

    @patch('structlog.get_logger')
    def test_timing_logging(self, mock_get_logger):
        """Test logování časování"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        perf_logger = PerformanceLogger("api")
        perf_logger.timing(
            operation="search_documents",
            duration_ms=150.5
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert call_args[0][0] == "Operation timing"
        assert call_args[1]["operation"] == "search_documents"
        assert call_args[1]["duration_ms"] == 150.5

    @patch('structlog.get_logger')
    def test_resource_usage_logging(self, mock_get_logger):
        """Test logování využití zdrojů"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        perf_logger = PerformanceLogger("worker")
        perf_logger.resource_usage(
            cpu_percent=45.2,
            memory_mb=512.0
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert call_args[0][0] == "Resource usage"
        assert call_args[1]["cpu_percent"] == 45.2
        assert call_args[1]["memory_mb"] == 512.0


class TestLoggerFactories:
    """Testy pro factory funkce"""

    def test_get_logger(self):
        """Test get_logger factory"""
        logger = get_logger("test_component", request_id="123")

        assert isinstance(logger, StructuredLogger)
        assert logger.context["request_id"] == "123"

    def test_get_audit_logger(self):
        """Test get_audit_logger factory"""
        from src.utils.logging import get_audit_logger

        logger = get_audit_logger("auth_service")

        assert isinstance(logger, AuditLogger)
        assert logger.context["component"] == "auth_service"

    def test_get_performance_logger(self):
        """Test get_performance_logger factory"""
        from src.utils.logging import get_performance_logger

        logger = get_performance_logger("search_engine")

        assert isinstance(logger, PerformanceLogger)
        assert logger.context["component"] == "search_engine"


@pytest.mark.integration
class TestLoggingConfiguration:
    """Integrační testy pro konfiguraci logování"""

    def test_configure_logging_json_format(self, tmp_path):
        """Test konfigurace s JSON formátem"""
        log_file = tmp_path / "test.log"

        configure_logging(
            log_level="INFO",
            log_format="json",
            log_file=str(log_file),
            enable_console=False
        )

        logger = get_logger("test")
        logger.info("Test message", field="value")

        # Ověření že se vytvořil log soubor
        assert log_file.exists()

    def test_configure_logging_text_format(self):
        """Test konfigurace s text formátem"""
        configure_logging(
            log_level="DEBUG",
            log_format="text",
            enable_console=True
        )

        logger = get_logger("test")
        logger.debug("Debug message")

        # Test že se logger vytvořil bez chyby
        assert logger is not None
