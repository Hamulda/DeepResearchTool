"""
Unit testy pro scraping/playwright_client.py - testování web scraping funkcionalitě
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from pathlib import Path
import json

from src.scraping.playwright_client import (
    PlaywrightClient,
    DiscoveryMode,
    ExtractionMode,
    ApiEndpoint,
    DiscoveryResult,
    discover_and_extract,
    PLAYWRIGHT_AVAILABLE,
)


class TestApiEndpoint:
    """Testy pro ApiEndpoint dataclass"""

    def test_api_endpoint_creation(self):
        """Test vytvoření ApiEndpoint instance"""
        timestamp = datetime.now()
        endpoint = ApiEndpoint(
            url="https://api.example.com/data",
            method="GET",
            headers={"Accept": "application/json"},
            payload=None,
            response_headers={"Content-Type": "application/json"},
            response_body='{"data": "test"}',
            status_code=200,
            timestamp=timestamp,
            content_type="application/json",
            size_bytes=16
        )

        assert endpoint.url == "https://api.example.com/data"
        assert endpoint.method == "GET"
        assert endpoint.status_code == 200
        assert endpoint.timestamp == timestamp
        assert endpoint.size_bytes == 16


class TestDiscoveryMode:
    """Testy pro DiscoveryMode třídu"""

    def test_discovery_mode_initialization(self):
        """Test inicializace DiscoveryMode"""
        discovery = DiscoveryMode()

        assert discovery.session_id.startswith("discovery_")
        assert discovery.captured_endpoints == []
        assert discovery.static_content == {}

    def test_discovery_mode_with_custom_session_id(self):
        """Test DiscoveryMode s custom session ID"""
        session_id = "test_session_123"
        discovery = DiscoveryMode(session_id)

        assert discovery.session_id == session_id

    def test_is_api_request_patterns(self):
        """Test rozpoznávání API požadavků"""
        discovery = DiscoveryMode()

        # Mock request objects
        api_request = Mock()
        api_request.url = "https://example.com/api/data"
        api_request.headers = {"content-type": "application/json"}

        non_api_request = Mock()
        non_api_request.url = "https://example.com/page.html"
        non_api_request.headers = {"content-type": "text/html"}

        assert discovery._is_api_request(api_request) is True
        assert discovery._is_api_request(non_api_request) is False

    def test_is_api_request_xhr(self):
        """Test rozpoznávání XHR požadavků"""
        discovery = DiscoveryMode()

        xhr_request = Mock()
        xhr_request.url = "https://example.com/data"
        xhr_request.headers = {"x-requested-with": "XMLHttpRequest"}

        assert discovery._is_api_request(xhr_request) is True


class TestExtractionMode:
    """Testy pro ExtractionMode třídu"""

    def test_extraction_mode_initialization(self):
        """Test inicializace ExtractionMode"""
        extraction = ExtractionMode(rate_limit=2.0)

        assert extraction.rate_limit == 2.0
        assert extraction.last_request_time == 0

    @pytest.mark.asyncio
    async def test_enforce_rate_limit(self):
        """Test rate limiting mechanismu"""
        extraction = ExtractionMode(rate_limit=0.1)

        start_time = asyncio.get_event_loop().time()
        await extraction._enforce_rate_limit()
        await extraction._enforce_rate_limit()
        end_time = asyncio.get_event_loop().time()

        # Druhý požadavek by měl být zpožděn
        assert (end_time - start_time) >= 0.1

    @pytest.mark.asyncio
    async def test_replay_single_endpoint_mock(self):
        """Test replay jednoho API endpointu s mock network client"""
        mock_network_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = AsyncMock(return_value='{"success": true}')
        mock_network_client.get.return_value = mock_response

        extraction = ExtractionMode(network_client=mock_network_client)

        endpoint = ApiEndpoint(
            url="https://api.example.com/test",
            method="GET",
            headers={"Accept": "application/json"},
            payload=None,
            response_headers={},
            response_body="",
            status_code=0,
            timestamp=datetime.now(),
            content_type="",
            size_bytes=0
        )

        result = await extraction._replay_single_endpoint(endpoint)

        assert result["url"] == "https://api.example.com/test"
        assert result["method"] == "GET"
        assert result["success"] is True
        mock_network_client.get.assert_called_once()


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not available")
class TestPlaywrightClient:
    """Testy pro PlaywrightClient - pouze pokud je Playwright dostupný"""

    def test_playwright_client_initialization(self):
        """Test inicializace PlaywrightClient"""
        client = PlaywrightClient(headless=True, browser_type="chromium")

        assert client.headless is True
        assert client.browser_type == "chromium"
        assert client.browser is None

    def test_playwright_client_invalid_browser(self):
        """Test nevalidního typu prohlížeče"""
        with pytest.raises(ValueError, match="Unsupported browser type"):
            client = PlaywrightClient(browser_type="invalid")

    @pytest.mark.asyncio
    async def test_playwright_unavailable_import_error(self):
        """Test chyby při nedostupném Playwright"""
        with patch('src.scraping.playwright_client.PLAYWRIGHT_AVAILABLE', False):
            with pytest.raises(ImportError, match="Playwright not available"):
                PlaywrightClient()


class TestDiscoveryResult:
    """Testy pro DiscoveryResult dataclass"""

    def test_discovery_result_creation(self):
        """Test vytvoření DiscoveryResult"""
        timestamp = datetime.now()
        endpoint = ApiEndpoint(
            url="https://api.example.com/test",
            method="GET",
            headers={},
            payload=None,
            response_headers={},
            response_body="{}",
            status_code=200,
            timestamp=timestamp,
            content_type="application/json",
            size_bytes=2
        )

        result = DiscoveryResult(
            page_url="https://example.com",
            page_title="Test Page",
            api_endpoints=[endpoint],
            static_content={"title": "Test Page"},
            discovered_at=timestamp,
            session_id="test_session"
        )

        assert result.page_url == "https://example.com"
        assert result.page_title == "Test Page"
        assert len(result.api_endpoints) == 1
        assert result.session_id == "test_session"


class TestSessionSerialization:
    """Testy pro ukládání a načítání discovery sessions"""

    def test_save_and_load_discovery_session(self, tmp_path):
        """Test uložení a načtení discovery session"""
        # Mock PlaywrightClient instance pro test
        with patch('src.scraping.playwright_client.PLAYWRIGHT_AVAILABLE', True):
            client = PlaywrightClient()

            # Vytvoření test discovery result
            timestamp = datetime.now()
            endpoint = ApiEndpoint(
                url="https://api.example.com/test",
                method="GET",
                headers={"Accept": "application/json"},
                payload=None,
                response_headers={"Content-Type": "application/json"},
                response_body='{"test": "data"}',
                status_code=200,
                timestamp=timestamp,
                content_type="application/json",
                size_bytes=15
            )

            result = DiscoveryResult(
                page_url="https://example.com",
                page_title="Test Page",
                api_endpoints=[endpoint],
                static_content={"title": "Test Page", "text": "Test content"},
                discovered_at=timestamp,
                session_id="test_session"
            )

            # Uložení session
            session_file = client.save_discovery_session(result, tmp_path)

            # Ověření že soubor existuje
            assert session_file.exists()

            # Načtení session
            loaded_result = PlaywrightClient.load_discovery_session(session_file)

            # Ověření načtených dat
            assert loaded_result.page_url == result.page_url
            assert loaded_result.page_title == result.page_title
            assert loaded_result.session_id == result.session_id
            assert len(loaded_result.api_endpoints) == 1
            assert loaded_result.api_endpoints[0].url == endpoint.url


class TestUtilityFunctions:
    """Testy pro utility funkce"""

    @pytest.mark.asyncio
    @patch('src.scraping.playwright_client.PLAYWRIGHT_AVAILABLE', True)
    async def test_discover_and_extract_mock(self):
        """Test discover_and_extract funkce s mock objekty"""
        mock_client = AsyncMock()
        mock_discovery_result = Mock()
        mock_discovery_result.api_endpoints = []
        mock_extraction_result = {
            "total_apis_replayed": 0,
            "successful_replays": 0
        }

        mock_client.discover_page_apis.return_value = mock_discovery_result
        mock_client.extract_with_replay.return_value = mock_extraction_result
        mock_client.save_discovery_session = Mock()

        with patch('src.scraping.playwright_client.PlaywrightClient') as mock_playwright:
            mock_playwright.return_value.__aenter__.return_value = mock_client
            mock_playwright.return_value.__aexit__.return_value = None

            result = await discover_and_extract("https://example.com", headless=True)

            assert "discovery" in result
            assert "extraction" in result
            assert "summary" in result
            assert result["summary"]["url"] == "https://example.com"


# Performance a stress testy
class TestPerformanceScenarios:
    """Testy pro performance scénáře"""

    def test_large_number_of_endpoints(self):
        """Test s velkým počtem API endpointů"""
        discovery = DiscoveryMode()

        # Simulace velkého počtu endpointů
        for i in range(1000):
            endpoint = ApiEndpoint(
                url=f"https://api.example.com/endpoint_{i}",
                method="GET",
                headers={},
                payload=None,
                response_headers={},
                response_body=f'{{"id": {i}}}',
                status_code=200,
                timestamp=datetime.now(),
                content_type="application/json",
                size_bytes=10
            )
            discovery.captured_endpoints.append(endpoint)

        assert len(discovery.captured_endpoints) == 1000

    def test_large_response_body_handling(self):
        """Test zpracování velkých response bodies"""
        large_response = "x" * (10 * 1024 * 1024)  # 10MB response

        endpoint = ApiEndpoint(
            url="https://api.example.com/large",
            method="GET",
            headers={},
            payload=None,
            response_headers={},
            response_body=large_response,
            status_code=200,
            timestamp=datetime.now(),
            content_type="text/plain",
            size_bytes=len(large_response)
        )

        assert endpoint.size_bytes == 10 * 1024 * 1024
        assert len(endpoint.response_body) == 10 * 1024 * 1024
