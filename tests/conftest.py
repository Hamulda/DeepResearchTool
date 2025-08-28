#!/usr/bin/env python3
"""
Pytest konfigurace a fixtures pro celý projekt

Author: Senior Python/MLOps Agent
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Generator, AsyncGenerator
from unittest.mock import Mock

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock

# Přidání src do path pro testy
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.config.settings import get_settings, ApplicationSettings
from src.utils.logging import get_logger, configure_logging


@pytest.fixture(scope="session")
def event_loop():
    """Vytvoří event loop pro celou test session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings() -> ApplicationSettings:
    """Test konfigurace - izolovaná od produkčního prostředí"""
    os.environ.update({
        "ENVIRONMENT": "testing",
        "DEBUG": "true",
        "AI_OPENAI_API_KEY": "test-key",
        "SECURITY_SECRET_KEY": "test-secret-key",
        "SECURITY_JWT_SECRET": "test-jwt-secret",
        "DB_POSTGRES_PASSWORD": "test-password",
    })

    # Vymazání cache pro settings
    get_settings.cache_clear()

    settings = get_settings()

    # Override pro testování
    settings.database.postgres_host = "localhost"
    settings.database.redis_host = "localhost"
    settings.ai_models.openai_model = "gpt-3.5-turbo"  # Levnější pro testy

    return settings


@pytest.fixture(scope="session")
def test_logger():
    """Nastavení logování pro testy"""
    configure_logging(
        log_level="DEBUG",
        log_format="text",
        enable_console=True
    )
    return get_logger("test")


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI klienta"""
    mock_client = Mock()
    mock_client.chat.completions.create = AsyncMock()
    mock_client.embeddings.create = AsyncMock()

    # Standardní odpovědi
    mock_client.chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content="Test response"))]
    )
    mock_client.embeddings.create.return_value = Mock(
        data=[Mock(embedding=[0.1] * 1536)]
    )

    return mock_client


@pytest.fixture
def mock_redis_client():
    """Mock Redis klienta"""
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.delete.return_value = True
    mock_redis.exists.return_value = False
    return mock_redis


@pytest.fixture
def mock_postgres_client():
    """Mock PostgreSQL klienta"""
    mock_postgres = AsyncMock()
    mock_postgres.execute.return_value = None
    mock_postgres.fetch.return_value = []
    mock_postgres.fetchrow.return_value = None
    return mock_postgres


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant vector DB klienta"""
    mock_qdrant = Mock()
    mock_qdrant.search = Mock(return_value=[])
    mock_qdrant.upsert = Mock(return_value=None)
    mock_qdrant.create_collection = Mock(return_value=None)
    return mock_qdrant


@pytest.fixture
def sample_text_data():
    """Ukázková textová data pro testy"""
    return {
        "short_text": "This is a short test text.",
        "long_text": "This is a much longer test text. " * 100,
        "html_content": "<html><body><h1>Test</h1><p>Content</p></body></html>",
        "json_data": '{"key": "value", "number": 42}',
        "multilingual": {
            "en": "Hello world",
            "cs": "Ahoj světe",
            "de": "Hallo Welt"
        }
    }


@pytest.fixture
def sample_documents():
    """Ukázkové dokumenty pro testování RAG pipeline"""
    return [
        {
            "id": "doc1",
            "title": "Test Document 1",
            "content": "This is the content of test document 1. It contains important information.",
            "metadata": {"source": "test", "timestamp": "2024-01-01T00:00:00Z"}
        },
        {
            "id": "doc2",
            "title": "Test Document 2",
            "content": "This is the content of test document 2. It has different information.",
            "metadata": {"source": "test", "timestamp": "2024-01-02T00:00:00Z"}
        }
    ]


@pytest.fixture
def temp_directory(tmp_path):
    """Dočasný adresář pro testy"""
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()

    # Vytvoření test souborů
    (test_dir / "test.txt").write_text("Test content")
    (test_dir / "test.json").write_text('{"test": true}')

    return test_dir


@pytest.fixture
async def async_temp_directory(tmp_path):
    """Async verze dočasného adresáře"""
    test_dir = tmp_path / "async_test_data"
    test_dir.mkdir()
    return test_dir


# Markery pro různé typy testů
def pytest_configure(config):
    """Konfigurace pytest markerů"""
    config.addinivalue_line(
        "markers", "unit: označuje unit testy"
    )
    config.addinivalue_line(
        "markers", "integration: označuje integrační testy"
    )
    config.addinivalue_line(
        "markers", "slow: označuje pomalé testy"
    )
    config.addinivalue_line(
        "markers", "external: testy vyžadující externí služby"
    )
    config.addinivalue_line(
        "markers", "ai_dependent: testy vyžadující AI API"
    )


# Skipování testů podle podmínek
def pytest_collection_modifyitems(config, items):
    """Modifikace testů podle podmínek"""

    # Skip external testy pokud nejsou nastavené env proměnné
    skip_external = pytest.mark.skip(reason="External services not configured")
    skip_ai = pytest.mark.skip(reason="AI API keys not configured")

    for item in items:
        if "external" in item.keywords:
            # Kontrola dostupnosti externích služeb
            if not all([
                os.getenv("DB_POSTGRES_PASSWORD"),
                os.getenv("DB_REDIS_PASSWORD")
            ]):
                item.add_marker(skip_external)

        if "ai_dependent" in item.keywords:
            # Kontrola AI API klíčů
            if not os.getenv("AI_OPENAI_API_KEY") or os.getenv("AI_OPENAI_API_KEY") == "test-key":
                item.add_marker(skip_ai)
