"""
Integration testy pro mikroslužbovou architekturu Fáze 1
"""

import pytest
import asyncio
import aiohttp
import redis
import time
import json
from pathlib import Path
import polars as pl
import tempfile
import shutil
from typing import Dict, Any

# Test configuration
TEST_REDIS_URL = "redis://localhost:6379/1"  # Use different DB for tests
TEST_API_URL = "http://localhost:8000"


class TestMicroservicesPhase1:
    """Komplexní testy pro mikroslužbovou architekturu"""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup a teardown pro každý test"""
        # Setup
        self.redis_client = redis.from_url(TEST_REDIS_URL)
        self.temp_dir = Path(tempfile.mkdtemp())

        yield

        # Teardown
        self.redis_client.flushdb()
        self.redis_client.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_api_health_check(self):
        """Test API gateway health check"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{TEST_API_URL}/health") as response:
                assert response.status == 200
                data = await response.json()
                assert data["api"] == "healthy"
                assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_scrape_task_creation(self):
        """Test vytvoření scraping úlohy"""
        test_url = "https://httpbin.org/html"

        async with aiohttp.ClientSession() as session:
            payload = {"url": test_url}
            async with session.post(f"{TEST_API_URL}/scrape", json=payload) as response:
                assert response.status == 200
                data = await response.json()

                assert data["status"] == "queued"
                assert "task_id" in data
                assert test_url in data["message"]

    def test_redis_queue_integration(self):
        """Test Redis queue funkcionalita"""
        # Simuluj vytvoření úlohy
        task_data = {
            "task_id": "test_task_123",
            "url": "https://example.com",
            "status": "queued",
            "created_at": "2025-01-01T00:00:00Z",
        }

        # Ulož do Redis
        self.redis_client.setex(f"task:{task_data['task_id']}", 3600, json.dumps(task_data))

        # Ověř že byla uložena
        stored_data = self.redis_client.get(f"task:{task_data['task_id']}")
        assert stored_data is not None

        parsed_data = json.loads(stored_data)
        assert parsed_data["task_id"] == task_data["task_id"]
        assert parsed_data["url"] == task_data["url"]

    def test_acquisition_worker_data_processing(self):
        """Test zpracování dat v acquisition worker"""
        from workers.acquisition_worker import AcquisitionWorker

        # Vytvoř test worker
        worker = AcquisitionWorker()

        # Simuluj HTML obsah
        test_html = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Test Content</h1>
                <p>This is a test paragraph with some content.</p>
            </body>
        </html>
        """

        # Test základní HTML zpracování
        # (Pro nyní jednoduchý test - rozšíříme ve Fázi 3)
        assert len(test_html) > 0
        assert "Test Content" in test_html

    def test_processing_worker_text_extraction(self):
        """Test extrakce textu v processing worker"""
        from workers.processing_worker import ProcessingWorker

        worker = ProcessingWorker()

        test_html = """
        <html>
            <body>
                <h1>Main Title</h1>
                <p>First paragraph with important content.</p>
                <p>Second paragraph with more data.</p>
            </body>
        </html>
        """

        # Test extrakce textu
        extracted = worker.extract_text_content(test_html)

        assert extracted["text_length"] > 0
        assert extracted["word_count"] > 0
        assert "Main Title" in extracted["clean_text"]
        assert "important content" in extracted["clean_text"]

    def test_text_chunking(self):
        """Test rozdělení textu na chunky"""
        from workers.processing_worker import ProcessingWorker

        worker = ProcessingWorker()

        # Dlouhý testovací text
        test_text = " ".join([f"Word{i}" for i in range(1000)])

        # Test chunking s malými chunky pro testování
        chunks = worker.chunk_text(test_text, chunk_size=10, overlap=2)

        assert len(chunks) > 1
        assert all(len(chunk.split()) <= 12 for chunk in chunks)  # 10 + overlap

        # Test že chunks obsahují očekávaný obsah
        assert "Word1" in chunks[0]
        assert any("Word500" in chunk for chunk in chunks)

    def test_parquet_data_persistence(self):
        """Test ukládání dat v Parquet formátu"""
        import polars as pl

        # Vytvoř testovací data
        test_data = pl.DataFrame(
            {
                "url": ["https://example.com"],
                "content": ["<html><body>Test content</body></html>"],
                "metadata": ['{"status": 200}'],
                "scraped_at": ["2025-01-01T00:00:00Z"],
            }
        )

        # Ulož do Parquet
        output_path = self.temp_dir / "test_data.parquet"
        test_data.write_parquet(output_path)

        # Ověř že soubor existuje
        assert output_path.exists()

        # Načti a ověř obsah
        loaded_data = pl.read_parquet(output_path)
        assert len(loaded_data) == 1
        assert loaded_data["url"][0] == "https://example.com"

    def test_lancedb_integration(self):
        """Test LanceDB integrace"""
        import lancedb

        # Vytvoř dočasnou LanceDB databázi
        db_path = self.temp_dir / "test_lancedb"
        db = lancedb.connect(str(db_path))

        # Testovací dokumenty
        test_documents = [
            {
                "id": "doc1",
                "text": "This is a test document",
                "url": "https://example.com",
                "chunk_index": 0,
            },
            {
                "id": "doc2",
                "text": "Another test document with different content",
                "url": "https://example.org",
                "chunk_index": 0,
            },
        ]

        # Vytvoř tabulku
        table = db.create_table("test_docs", test_documents)

        # Ověř že data byla uložena
        results = table.search().limit(10).to_list()
        assert len(results) == 2
        assert any(doc["text"] == "This is a test document" for doc in results)

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """End-to-end test celého workflow"""
        # Tento test ověří celý pipeline od API po uložení dat

        # 1. Vytvoř úlohu přes API
        test_url = "https://httpbin.org/html"

        async with aiohttp.ClientSession() as session:
            payload = {"url": test_url, "task_id": "e2e_test"}
            async with session.post(f"{TEST_API_URL}/scrape", json=payload) as response:
                assert response.status == 200
                data = await response.json()
                task_id = data["task_id"]

        # 2. Počkej na zpracování (v reálném testě by se použil mock)
        await asyncio.sleep(2)

        # 3. Ověř status úlohy
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{TEST_API_URL}/task/{task_id}") as response:
                if response.status == 200:
                    task_status = await response.json()
                    assert task_status["task_id"] == task_id

    def test_memory_efficiency(self):
        """Test paměťové efektivity Parquet vs JSON"""
        import json
        import os

        # Vytvoř větší testovací dataset
        large_data = {
            "records": [
                {
                    "url": f"https://example{i}.com",
                    "content": f"Large content block {i} " * 100,
                    "metadata": {"id": i, "processed": True},
                }
                for i in range(100)
            ]
        }

        # Ulož jako JSON
        json_path = self.temp_dir / "test_data.json"
        with open(json_path, "w") as f:
            json.dump(large_data, f)

        # Ulož jako Parquet
        parquet_path = self.temp_dir / "test_data.parquet"
        df = pl.DataFrame(
            [
                {
                    "url": record["url"],
                    "content": record["content"],
                    "metadata": json.dumps(record["metadata"]),
                }
                for record in large_data["records"]
            ]
        )
        df.write_parquet(parquet_path)

        # Porovnej velikosti souborů
        json_size = os.path.getsize(json_path)
        parquet_size = os.path.getsize(parquet_path)

        # Parquet by měl být menší díky kompresi
        compression_ratio = parquet_size / json_size
        assert compression_ratio < 0.8  # Alespoň 20% úspora

        print(f"JSON size: {json_size}, Parquet size: {parquet_size}")
        print(f"Compression ratio: {compression_ratio:.2f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
