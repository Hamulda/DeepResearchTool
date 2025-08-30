"""
Unit testy pro core/config.py - testování načítání a validace konfigurace
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

from src.core.config import (
    AppSettings,
    QdrantConfig,
    DuckDBConfig,
    load_config_from_yaml,
    create_app_settings,
    get_app_settings,
    reset_app_settings,
)


class TestAppSettings:
    """Testy pro AppSettings konfiguraci"""

    def test_default_settings(self):
        """Test výchozích hodnot konfigurace"""
        settings = AppSettings()

        assert settings.debug is False
        assert settings.log_level == "INFO"
        assert settings.apple_silicon_optimizations is True
        assert settings.memory_pressure_monitoring is True

        # Test výchozích komponent
        assert isinstance(settings.qdrant, QdrantConfig)
        assert isinstance(settings.duckdb, DuckDBConfig)

    def test_log_level_validation(self):
        """Test validace log_level"""
        # Validní úrovně
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            settings = AppSettings(log_level=level)
            assert settings.log_level == level

        # Test case insensitive
        settings = AppSettings(log_level="debug")
        assert settings.log_level == "DEBUG"

        # Nevalidní úroveň
        with pytest.raises(ValueError, match="log_level must be one of"):
            AppSettings(log_level="INVALID")

    def test_qdrant_quantization_validation(self):
        """Test validace Qdrant kvantizace"""
        # Validní kvantizace
        qdrant_config = QdrantConfig()
        qdrant_config.quantization.type = "int8"
        settings = AppSettings(qdrant=qdrant_config)
        assert settings.qdrant.quantization.type == "int8"

        # Nevalidní kvantizace
        qdrant_config.quantization.type = "invalid"
        with pytest.raises(ValueError, match="Quantization type must be"):
            AppSettings(qdrant=qdrant_config)


class TestConfigLoading:
    """Testy pro načítání konfigurace ze souborů"""

    def test_load_config_from_nonexistent_file(self):
        """Test načítání neexistujícího konfiguračního souboru"""
        result = load_config_from_yaml("nonexistent.yaml")
        assert result == {}

    def test_load_config_from_yaml_valid(self):
        """Test načítání validního YAML souboru"""
        config_data = {
            "debug": True,
            "log_level": "DEBUG",
            "qdrant": {
                "url": "http://test:6333",
                "collection_name": "test_collection"
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            result = load_config_from_yaml(temp_path)
            assert result["debug"] is True
            assert result["log_level"] == "DEBUG"
            assert result["qdrant"]["url"] == "http://test:6333"
        finally:
            Path(temp_path).unlink()

    def test_load_config_from_yaml_invalid(self):
        """Test načítání nevalidního YAML souboru"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name

        try:
            result = load_config_from_yaml(temp_path)
            assert result == {}
        finally:
            Path(temp_path).unlink()

    def test_create_app_settings_with_config(self):
        """Test vytvoření AppSettings s konfiguračním souborem"""
        config_data = {
            "debug": True,
            "log_level": "DEBUG",
            "apple_silicon_optimizations": False
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            settings = create_app_settings(temp_path)
            assert settings.debug is True
            assert settings.log_level == "DEBUG"
            assert settings.apple_silicon_optimizations is False
        finally:
            Path(temp_path).unlink()


class TestSingletonPattern:
    """Testy pro singleton pattern u get_app_settings"""

    def setup_method(self):
        """Reset singleton před každým testem"""
        reset_app_settings()

    def test_singleton_behavior(self):
        """Test že get_app_settings vrací stejnou instanci"""
        settings1 = get_app_settings()
        settings2 = get_app_settings()

        assert settings1 is settings2

    def test_reset_app_settings(self):
        """Test resetu globální konfigurace"""
        settings1 = get_app_settings()
        reset_app_settings()
        settings2 = get_app_settings()

        assert settings1 is not settings2


class TestQdrantConfig:
    """Testy pro QdrantConfig"""

    def test_default_qdrant_config(self):
        """Test výchozích hodnot Qdrant konfigurace"""
        config = QdrantConfig()

        assert config.url == "http://localhost:6333"
        assert config.collection_name == "research_docs"
        assert config.vector_size == 384
        assert config.distance == "Cosine"
        assert config.quantization.enabled is True
        assert config.quantization.type == "int8"
        assert config.prefer_grpc is False
        assert config.timeout == 60
        assert config.retries == 3


class TestDuckDBConfig:
    """Testy pro DuckDBConfig"""

    def test_default_duckdb_config(self):
        """Test výchozích hodnot DuckDB konfigurace"""
        config = DuckDBConfig()

        assert config.db_path == "data/warehouse.duckdb"
        assert config.memory_limit == "2GB"
        assert config.threads == 4
        assert config.enable_object_cache is True
        assert config.temp_directory == "/tmp/duckdb"


# Integration test s environment variables
class TestEnvironmentIntegration:
    """Testy integrace s environment variables"""

    @patch.dict("os.environ", {"LOG_LEVEL": "ERROR"})
    def test_env_override(self):
        """Test přepsání konfigurace environment variables"""
        settings = AppSettings()
        assert settings.log_level == "ERROR"

    @patch.dict("os.environ", {"DEBUG": "true"})
    def test_debug_env_override(self):
        """Test přepsání debug flag z environment"""
        settings = AppSettings()
        assert settings.debug is True
