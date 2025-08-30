#!/usr/bin/env python3
"""
Comprehensive unit tests for core configuration module
Tests config loading, validation, and error handling scenarios
"""

import pytest
import os
import tempfile
from unittest.mock import patch, mock_open
from pathlib import Path

from src.core.config import Config, load_config, validate_config


class TestConfig:
    """Test suite for Config class"""

    def test_config_initialization_default(self):
        """Test Config initialization with default values"""
        config = Config()

        assert config.environment == "development"
        assert config.debug is True
        assert config.log_level == "INFO"
        assert isinstance(config.api, dict)
        assert isinstance(config.database, dict)

    def test_config_initialization_with_data(self):
        """Test Config initialization with provided data"""
        test_data = {
            "environment": "production",
            "debug": False,
            "log_level": "ERROR",
            "api": {"host": "0.0.0.0", "port": 8000},
            "database": {"url": "postgresql://test"}
        }

        config = Config(test_data)

        assert config.environment == "production"
        assert config.debug is False
        assert config.log_level == "ERROR"
        assert config.api["host"] == "0.0.0.0"
        assert config.database["url"] == "postgresql://test"

    def test_config_get_method(self):
        """Test Config.get() method for nested access"""
        config = Config({
            "database": {
                "redis": {
                    "host": "localhost",
                    "port": 6379
                }
            }
        })

        assert config.get("database.redis.host") == "localhost"
        assert config.get("database.redis.port") == 6379
        assert config.get("nonexistent.key", "default") == "default"

    def test_config_validation_success(self):
        """Test successful config validation"""
        valid_config = {
            "environment": "production",
            "debug": False,
            "api": {"host": "0.0.0.0", "port": 8000},
            "database": {"url": "postgresql://valid"}
        }

        # Should not raise any exception
        validate_config(valid_config)

    def test_config_validation_missing_required(self):
        """Test config validation with missing required fields"""
        invalid_config = {
            "environment": "production"
            # Missing other required fields
        }

        with pytest.raises(ValueError, match="Missing required config"):
            validate_config(invalid_config)

    def test_config_validation_invalid_environment(self):
        """Test config validation with invalid environment"""
        invalid_config = {
            "environment": "invalid_env",
            "api": {"host": "0.0.0.0", "port": 8000},
            "database": {"url": "postgresql://test"}
        }

        with pytest.raises(ValueError, match="Invalid environment"):
            validate_config(invalid_config)


class TestLoadConfig:
    """Test suite for load_config function"""

    def test_load_config_from_file(self):
        """Test loading config from YAML file"""
        yaml_content = """
        environment: testing
        debug: true
        log_level: DEBUG
        api:
          host: 127.0.0.1
          port: 8080
        database:
          url: postgresql://test
        """

        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with patch("os.path.exists", return_value=True):
                config = load_config("test_config.yaml")

                assert config.environment == "testing"
                assert config.debug is True
                assert config.api["port"] == 8080

    def test_load_config_file_not_found(self):
        """Test behavior when config file doesn't exist"""
        with patch("os.path.exists", return_value=False):
            config = load_config("nonexistent.yaml")

            # Should return default config
            assert config.environment == "development"
            assert config.debug is True

    def test_load_config_invalid_yaml(self):
        """Test handling of invalid YAML syntax"""
        invalid_yaml = """
        environment: testing
        invalid: [unclosed bracket
        """

        with patch("builtins.open", mock_open(read_data=invalid_yaml)):
            with patch("os.path.exists", return_value=True):
                with pytest.raises(ValueError, match="Invalid YAML"):
                    load_config("invalid.yaml")

    def test_load_config_environment_override(self):
        """Test environment variable override of config values"""
        yaml_content = """
        environment: development
        api:
          port: 8000
        """

        with patch.dict(os.environ, {"API_PORT": "9000"}):
            with patch("builtins.open", mock_open(read_data=yaml_content)):
                with patch("os.path.exists", return_value=True):
                    config = load_config("test.yaml")

                    # Environment variable should override YAML
                    assert config.api["port"] == 9000

    def test_load_config_with_secrets(self):
        """Test config loading with secret values"""
        yaml_content = """
        environment: production
        database:
          password: ${DB_PASSWORD}
        api:
          secret_key: ${API_SECRET}
        """

        with patch.dict(os.environ, {
            "DB_PASSWORD": "secret123",
            "API_SECRET": "api_key_456"
        }):
            with patch("builtins.open", mock_open(read_data=yaml_content)):
                with patch("os.path.exists", return_value=True):
                    config = load_config("test.yaml")

                    assert config.database["password"] == "secret123"
                    assert config.api["secret_key"] == "api_key_456"

    def test_load_config_missing_secrets(self):
        """Test handling of missing secret environment variables"""
        yaml_content = """
        database:
          password: ${MISSING_SECRET}
        """

        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with patch("os.path.exists", return_value=True):
                with pytest.raises(ValueError, match="Missing environment variable"):
                    load_config("test.yaml")


class TestConfigIntegration:
    """Integration tests for config system"""

    def test_full_config_lifecycle(self):
        """Test complete config loading and usage lifecycle"""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
            environment: testing
            debug: false
            log_level: WARNING
            api:
              host: 0.0.0.0
              port: 8000
              workers: 4
            database:
              url: postgresql://user:pass@localhost/testdb
              pool_size: 10
            redis:
              host: localhost
              port: 6379
              db: 0
            security:
              secret_key: test_secret
              jwt_expiration: 3600
            """)
            temp_file = f.name

        try:
            # Load and validate config
            config = load_config(temp_file)

            # Verify all sections loaded correctly
            assert config.environment == "testing"
            assert config.debug is False
            assert config.log_level == "WARNING"

            # Test nested access
            assert config.get("api.workers") == 4
            assert config.get("database.pool_size") == 10
            assert config.get("redis.db") == 0
            assert config.get("security.jwt_expiration") == 3600

            # Test non-existent keys
            assert config.get("nonexistent.key", "default") == "default"

        finally:
            # Cleanup
            os.unlink(temp_file)

    def test_config_environment_specific_loading(self):
        """Test loading environment-specific configurations"""
        base_config = """
        environment: development
        debug: true
        api:
          port: 8000
        """

        prod_config = """
        environment: production
        debug: false
        api:
          port: 80
        """

        # Test development config
        with patch("builtins.open", mock_open(read_data=base_config)):
            with patch("os.path.exists", return_value=True):
                dev_config = load_config("config.yaml")
                assert dev_config.environment == "development"
                assert dev_config.debug is True
                assert dev_config.api["port"] == 8000

        # Test production config
        with patch("builtins.open", mock_open(read_data=prod_config)):
            with patch("os.path.exists", return_value=True):
                prod_config_obj = load_config("config.prod.yaml")
                assert prod_config_obj.environment == "production"
                assert prod_config_obj.debug is False
                assert prod_config_obj.api["port"] == 80

    def test_config_error_handling_edge_cases(self):
        """Test edge cases in config error handling"""
        # Test empty file
        with patch("builtins.open", mock_open(read_data="")):
            with patch("os.path.exists", return_value=True):
                config = load_config("empty.yaml")
                # Should return default config
                assert config.environment == "development"

        # Test file with only comments
        comment_only = """
        # This is a comment
        # Another comment
        """
        with patch("builtins.open", mock_open(read_data=comment_only)):
            with patch("os.path.exists", return_value=True):
                config = load_config("comments.yaml")
                assert config.environment == "development"

        # Test permission denied
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            with patch("os.path.exists", return_value=True):
                with pytest.raises(PermissionError):
                    load_config("protected.yaml")
