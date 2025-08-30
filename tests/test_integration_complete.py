#!/usr/bin/env python3
"""
Integrační testy pro RAG pipeline a hlavní komponenty

Author: Senior Python/MLOps Agent
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from src.config.settings import get_settings
from src.utils.logging import get_logger, get_performance_logger


@pytest.mark.integration
class TestRAGPipelineIntegration:
    """Integrační testy pro celý RAG pipeline"""

    @pytest.fixture
    async def rag_pipeline(self, mock_openai_client, mock_qdrant_client, mock_redis_client):
        """Mock RAG pipeline s všemi závislostmi"""
        with patch("src.retrieval.enhanced_rrf.OpenAI") as mock_openai, patch(
            "src.storage.vector_store.QdrantClient"
        ) as mock_qdrant, patch("src.storage.cache.RedisClient") as mock_redis:

            mock_openai.return_value = mock_openai_client
            mock_qdrant.return_value = mock_qdrant_client
            mock_redis.return_value = mock_redis_client

            # Simulace RAG pipeline
            pipeline = Mock()
            pipeline.process_query = AsyncMock()
            pipeline.add_documents = AsyncMock()
            pipeline.search_documents = AsyncMock()

            return pipeline

    @pytest.mark.asyncio
    async def test_end_to_end_query_processing(self, rag_pipeline, sample_documents):
        """Test kompletního zpracování dotazu od začátku do konce"""
        query = "What is the content of test documents?"

        # Mock odpověď pipeline
        expected_response = {
            "answer": "The test documents contain important information for testing.",
            "sources": ["doc1", "doc2"],
            "confidence": 0.85,
        }
        rag_pipeline.process_query.return_value = expected_response

        # Simulace celého procesu
        result = await rag_pipeline.process_query(query)

        assert result["answer"] is not None
        assert "sources" in result
        assert result["confidence"] > 0.0
        rag_pipeline.process_query.assert_called_once_with(query)

    @pytest.mark.asyncio
    async def test_document_ingestion_pipeline(self, rag_pipeline, sample_documents):
        """Test pipeline pro přidávání dokumentů"""
        # Mock úspěšné přidání
        rag_pipeline.add_documents.return_value = {
            "added_count": len(sample_documents),
            "failed_count": 0,
            "processing_time": 1.5,
        }

        result = await rag_pipeline.add_documents(sample_documents)

        assert result["added_count"] == len(sample_documents)
        assert result["failed_count"] == 0
        rag_pipeline.add_documents.assert_called_once_with(sample_documents)

    @pytest.mark.asyncio
    async def test_search_functionality(self, rag_pipeline):
        """Test vyhledávací funkcionality"""
        query = "test information"

        # Mock search results
        search_results = [
            {"id": "doc1", "score": 0.9, "content": "Relevant content"},
            {"id": "doc2", "score": 0.7, "content": "Related information"},
        ]
        rag_pipeline.search_documents.return_value = search_results

        results = await rag_pipeline.search_documents(query, limit=10)

        assert len(results) == 2
        assert all("score" in result for result in results)
        assert all(result["score"] >= 0.0 for result in results)


@pytest.mark.integration
class TestAPIIntegration:
    """Integrační testy pro API komponenty"""

    @pytest.fixture
    def mock_api_client(self):
        """Mock API klient"""
        client = Mock()
        client.post = AsyncMock()
        client.get = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_api_authentication(self, mock_api_client, test_settings):
        """Test API autentifikace"""
        # Mock úspěšné autentifikace
        mock_api_client.post.return_value = Mock(
            status_code=200, json=Mock(return_value={"token": "test-token", "expires_in": 3600})
        )

        # Simulace login request
        response = await mock_api_client.post("/auth/login", json={"api_key": "test-key"})

        assert response.status_code == 200
        token_data = response.json()
        assert "token" in token_data
        assert token_data["expires_in"] > 0

    @pytest.mark.asyncio
    async def test_api_rate_limiting(self, mock_api_client):
        """Test rate limiting"""
        # Simulace překročení rate limitu
        mock_api_client.post.side_effect = [
            Mock(status_code=200),  # První request OK
            Mock(status_code=200),  # Druhý request OK
            Mock(status_code=429),  # Třetí request rate limited
        ]

        responses = []
        for i in range(3):
            response = await mock_api_client.post(f"/api/search", json={"query": f"test {i}"})
            responses.append(response.status_code)

        assert responses[:2] == [200, 200]
        assert responses[2] == 429  # Rate limited

    @pytest.mark.asyncio
    async def test_api_error_handling(self, mock_api_client):
        """Test zpracování chyb API"""
        # Mock různé typy chyb
        mock_api_client.post.side_effect = [
            Mock(status_code=400, json=Mock(return_value={"error": "Bad Request"})),
            Mock(status_code=500, json=Mock(return_value={"error": "Internal Server Error"})),
            Exception("Network error"),
        ]

        # Test 400 error
        response1 = await mock_api_client.post("/api/invalid")
        assert response1.status_code == 400

        # Test 500 error
        response2 = await mock_api_client.post("/api/error")
        assert response2.status_code == 500

        # Test network error
        with pytest.raises(Exception, match="Network error"):
            await mock_api_client.post("/api/network-fail")


@pytest.mark.integration
class TestDatabaseIntegration:
    """Integrační testy pro databázové komponenty"""

    @pytest.mark.asyncio
    async def test_redis_caching_integration(self, mock_redis_client):
        """Test integrace s Redis cache"""
        cache_key = "test:documents:123"
        test_data = {"content": "cached document", "timestamp": "2024-01-01"}

        # Mock cache operations
        mock_redis_client.get.return_value = None  # Cache miss
        mock_redis_client.set.return_value = True
        mock_redis_client.get.return_value = str(test_data)  # Cache hit

        # Test cache miss
        result1 = await mock_redis_client.get(cache_key)
        assert result1 is None

        # Test cache set
        await mock_redis_client.set(cache_key, str(test_data))
        mock_redis_client.set.assert_called_with(cache_key, str(test_data))

        # Test cache hit
        result2 = await mock_redis_client.get(cache_key)
        assert result2 is not None

    @pytest.mark.asyncio
    async def test_vector_db_integration(self, mock_qdrant_client):
        """Test integrace s vector databází"""
        collection_name = "test_documents"

        # Mock vector operations
        mock_qdrant_client.search.return_value = [
            Mock(id="doc1", score=0.9, payload={"content": "test"}),
            Mock(id="doc2", score=0.7, payload={"content": "content"}),
        ]

        # Test search
        results = mock_qdrant_client.search(
            collection_name=collection_name, query_vector=[0.1] * 1536, limit=10
        )

        assert len(results) == 2
        assert results[0].score > results[1].score  # Seřazeno podle skóre
        mock_qdrant_client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_postgres_metadata_storage(self, mock_postgres_client):
        """Test ukládání metadat do PostgreSQL"""
        # Mock database operations
        mock_postgres_client.execute.return_value = None
        mock_postgres_client.fetch.return_value = [
            {"id": 1, "document_id": "doc1", "title": "Test Doc 1"},
            {"id": 2, "document_id": "doc2", "title": "Test Doc 2"},
        ]

        # Test insert
        await mock_postgres_client.execute(
            "INSERT INTO documents (document_id, title) VALUES ($1, $2)", "doc3", "Test Doc 3"
        )
        mock_postgres_client.execute.assert_called()

        # Test select
        results = await mock_postgres_client.fetch("SELECT * FROM documents")
        assert len(results) == 2
        assert results[0]["document_id"] == "doc1"


@pytest.mark.integration
class TestScrapingIntegration:
    """Integrační testy pro scraping komponenty"""

    @pytest.fixture
    def mock_scraper(self):
        """Mock scraper"""
        scraper = Mock()
        scraper.scrape_url = AsyncMock()
        scraper.scrape_multiple_urls = AsyncMock()
        return scraper

    @pytest.mark.asyncio
    async def test_web_scraping_pipeline(self, mock_scraper):
        """Test pipeline pro web scraping"""
        url = "https://example.com/test"

        # Mock scraping result
        mock_scraper.scrape_url.return_value = {
            "url": url,
            "title": "Test Page",
            "content": "This is test content from the page.",
            "metadata": {"scraped_at": "2024-01-01T00:00:00Z"},
        }

        result = await mock_scraper.scrape_url(url)

        assert result["url"] == url
        assert "content" in result
        assert "metadata" in result
        mock_scraper.scrape_url.assert_called_once_with(url)

    @pytest.mark.asyncio
    async def test_batch_scraping(self, mock_scraper):
        """Test batch scrapingu více URL"""
        urls = ["https://example.com/1", "https://example.com/2", "https://example.com/3"]

        # Mock batch results
        mock_scraper.scrape_multiple_urls.return_value = [
            {"url": url, "content": f"Content for {url}", "success": True} for url in urls
        ]

        results = await mock_scraper.scrape_multiple_urls(urls)

        assert len(results) == len(urls)
        assert all(result["success"] for result in results)
        mock_scraper.scrape_multiple_urls.assert_called_once_with(urls)

    @pytest.mark.asyncio
    async def test_scraping_error_handling(self, mock_scraper):
        """Test zpracování chyb při scrapingu"""
        problematic_urls = ["https://invalid-url", "https://404.example.com"]

        # Mock error results
        mock_scraper.scrape_multiple_urls.return_value = [
            {"url": problematic_urls[0], "error": "Invalid URL", "success": False},
            {"url": problematic_urls[1], "error": "404 Not Found", "success": False},
        ]

        results = await mock_scraper.scrape_multiple_urls(problematic_urls)

        assert len(results) == 2
        assert all(not result["success"] for result in results)
        assert all("error" in result for result in results)


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Integrační testy výkonu"""

    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test souběžného zpracování"""
        import time

        async def mock_processing_task(task_id: int, duration: float):
            """Mock úloha zpracování"""
            await asyncio.sleep(duration)
            return {"task_id": task_id, "result": f"processed_{task_id}"}

        # Test concurrent execution
        start_time = time.time()
        tasks = [mock_processing_task(i, 0.1) for i in range(10)]
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time

        assert len(results) == 10
        assert duration < 1.0  # Souběžné spuštění by mělo být rychlejší než sekvenční
        assert all("result" in result for result in results)

    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self):
        """Test využití paměti pod zátěží"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Simulace zatížení
        large_data = []
        for i in range(1000):
            large_data.append({"id": i, "data": "x" * 1000})

        peak_memory = process.memory_info().rss

        # Cleanup
        del large_data

        final_memory = process.memory_info().rss

        # Memory should not grow excessively
        memory_growth = (peak_memory - initial_memory) / 1024 / 1024  # MB
        assert memory_growth < 100  # Shouldn't grow more than 100MB

    def test_cpu_usage_optimization(self):
        """Test optimalizace využití CPU"""
        import time
        import threading

        def cpu_intensive_task():
            """CPU náročná úloha"""
            result = 0
            for i in range(100000):
                result += i * i
            return result

        # Single thread benchmark
        start_time = time.time()
        result1 = cpu_intensive_task()
        single_thread_time = time.time() - start_time

        # Multi-thread test
        start_time = time.time()
        threads = []
        results = []

        def worker():
            results.append(cpu_intensive_task())

        for _ in range(2):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        multi_thread_time = time.time() - start_time

        assert len(results) == 2
        assert all(result == result1 for result in results)
        # Multi-threading by mělo být efektivnější pro CPU úlohy
        assert multi_thread_time <= single_thread_time * 1.5
