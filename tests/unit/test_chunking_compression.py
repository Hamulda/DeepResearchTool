#!/usr/bin/env python3
"""
Unit testy pro komprese a chunking komponenty

Author: Senior Python/MLOps Agent
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from src.compress.adaptive_chunking import AdaptiveChunker
from src.compress.enhanced_contextual_compression import EnhancedContextualCompressor
from src.utils.logging import get_logger


class TestAdaptiveChunker:
    """Testy pro AdaptiveChunker"""

    @pytest.fixture
    def chunker(self):
        """Fixture pro AdaptiveChunker"""
        return AdaptiveChunker(chunk_size=500, overlap_size=50, min_chunk_size=100)

    def test_simple_text_chunking(self, chunker, sample_text_data):
        """Test základního chunkingu textu"""
        text = sample_text_data["long_text"]
        chunks = chunker.chunk_text(text)

        assert len(chunks) > 1
        assert all(len(chunk) >= 100 for chunk in chunks)  # min_chunk_size
        assert all(len(chunk) <= 600 for chunk in chunks)  # chunk_size + tolerance

    def test_overlap_functionality(self, chunker):
        """Test překrývání chunků"""
        text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10" * 20
        chunks = chunker.chunk_text(text)

        if len(chunks) > 1:
            # Kontrola překrývání mezi sousedními chunky
            overlap_found = False
            for i in range(len(chunks) - 1):
                if any(word in chunks[i + 1] for word in chunks[i].split()[-10:]):
                    overlap_found = True
                    break
            assert overlap_found

    def test_short_text_no_chunking(self, chunker):
        """Test že krátký text se nechunkuje"""
        short_text = "This is a short text."
        chunks = chunker.chunk_text(short_text)

        assert len(chunks) == 1
        assert chunks[0] == short_text

    def test_empty_text(self, chunker):
        """Test prázdného textu"""
        chunks = chunker.chunk_text("")
        assert len(chunks) == 0

    def test_whitespace_only(self, chunker):
        """Test textu pouze s whitespace"""
        chunks = chunker.chunk_text("   \n\t   ")
        assert len(chunks) == 0

    @pytest.mark.parametrize("chunk_size,overlap", [(100, 10), (1000, 100), (2000, 200)])
    def test_different_sizes(self, chunk_size, overlap, sample_text_data):
        """Test různých velikostí chunků"""
        chunker = AdaptiveChunker(chunk_size=chunk_size, overlap_size=overlap)

        text = sample_text_data["long_text"]
        chunks = chunker.chunk_text(text)

        assert all(len(chunk) <= chunk_size + 100 for chunk in chunks)  # tolerance


class TestEnhancedContextualCompressor:
    """Testy pro EnhancedContextualCompressor"""

    @pytest.fixture
    def compressor(self, mock_openai_client):
        """Fixture pro EnhancedContextualCompressor"""
        with patch("src.compress.enhanced_contextual_compression.OpenAI") as mock_openai:
            mock_openai.return_value = mock_openai_client
            return EnhancedContextualCompressor(model_name="gpt-3.5-turbo", compression_ratio=0.7)

    @pytest.mark.asyncio
    async def test_compress_documents(self, compressor, sample_documents):
        """Test komprese dokumentů"""
        query = "test information"

        # Mock AI odpověď
        compressor.ai_client.chat.completions.create.return_value = Mock(
            choices=[
                Mock(
                    message=Mock(
                        content="Compressed content: This document contains important test information."
                    )
                )
            ]
        )

        compressed = await compressor.compress_documents(sample_documents, query)

        assert len(compressed) <= len(sample_documents)
        assert all("content" in doc for doc in compressed)
        compressor.ai_client.chat.completions.create.assert_called()

    @pytest.mark.asyncio
    async def test_relevance_scoring(self, compressor, sample_documents):
        """Test skórování relevance"""
        query = "important information"

        scores = await compressor._calculate_relevance_scores(sample_documents, query)

        assert len(scores) == len(sample_documents)
        assert all(0 <= score <= 1 for score in scores)

    @pytest.mark.asyncio
    async def test_empty_documents(self, compressor):
        """Test prázdných dokumentů"""
        compressed = await compressor.compress_documents([], "test query")
        assert compressed == []

    @pytest.mark.asyncio
    async def test_single_document(self, compressor, sample_documents):
        """Test jediného dokumentu"""
        single_doc = [sample_documents[0]]

        compressor.ai_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="Compressed single document"))]
        )

        compressed = await compressor.compress_documents(single_doc, "test")

        assert len(compressed) == 1

    @pytest.mark.asyncio
    async def test_compression_ratio(self, compressor, sample_documents):
        """Test kompresního poměru"""
        original_length = sum(len(doc["content"]) for doc in sample_documents)

        compressor.ai_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="Short compressed content"))]
        )

        compressed = await compressor.compress_documents(sample_documents, "test")
        compressed_length = sum(len(doc["content"]) for doc in compressed)

        # Komprese by měla snížit celkovou délku
        assert compressed_length <= original_length


class TestChunkingIntegration:
    """Integrační testy pro chunking komponenty"""

    @pytest.fixture
    def integrated_chunker(self, mock_openai_client):
        """Chunker s AI komponentami"""
        with patch("src.compress.adaptive_chunking.OpenAI") as mock_openai:
            mock_openai.return_value = mock_openai_client
            return AdaptiveChunker(chunk_size=1000, overlap_size=100, use_ai_optimization=True)

    def test_html_content_chunking(self, integrated_chunker, sample_text_data):
        """Test chunkingu HTML obsahu"""
        html = sample_text_data["html_content"]
        chunks = integrated_chunker.chunk_text(html)

        assert len(chunks) >= 1
        # HTML by měl být očištěn
        assert not any("<html>" in chunk for chunk in chunks)

    def test_multilingual_chunking(self, integrated_chunker, sample_text_data):
        """Test chunkingu vícejazyčného obsahu"""
        multilingual_text = " ".join(sample_text_data["multilingual"].values()) * 50
        chunks = integrated_chunker.chunk_text(multilingual_text)

        assert len(chunks) >= 1
        assert all(len(chunk.strip()) > 0 for chunk in chunks)

    @pytest.mark.parametrize("text_type", ["json_data", "long_text"])
    def test_different_content_types(self, integrated_chunker, sample_text_data, text_type):
        """Test různých typů obsahu"""
        content = sample_text_data[text_type]
        chunks = integrated_chunker.chunk_text(content)

        assert isinstance(chunks, list)
        assert all(isinstance(chunk, str) for chunk in chunks)


@pytest.mark.integration
class TestChunkingPerformance:
    """Testy výkonu chunking komponent"""

    def test_large_text_performance(self):
        """Test výkonu s velkým textem"""
        import time

        large_text = "This is a test sentence. " * 10000  # ~250KB textu
        chunker = AdaptiveChunker(chunk_size=1000, overlap_size=100)

        start_time = time.time()
        chunks = chunker.chunk_text(large_text)
        duration = time.time() - start_time

        assert duration < 5.0  # Mělo by být rychlejší než 5 sekund
        assert len(chunks) > 10  # Mělo by vytvořit více chunků

    def test_memory_efficiency(self):
        """Test efektivního využití paměti"""
        import sys

        text = "Memory test content. " * 1000
        chunker = AdaptiveChunker(chunk_size=500)

        # Měření velikosti před a po
        initial_size = sys.getsizeof(text)
        chunks = chunker.chunk_text(text)
        total_chunks_size = sum(sys.getsizeof(chunk) for chunk in chunks)

        # Chunky by neměly zabírat výrazně více paměti
        assert total_chunks_size < initial_size * 1.5

    @pytest.mark.slow
    def test_concurrent_chunking(self):
        """Test souběžného chunkingu"""
        import concurrent.futures
        import threading

        texts = [f"Concurrent test text {i}. " * 100 for i in range(10)]
        chunker = AdaptiveChunker(chunk_size=500)

        def chunk_text(text):
            return chunker.chunk_text(text)

        # Test thread-safety
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(chunk_text, text) for text in texts]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        assert len(results) == len(texts)
        assert all(len(chunks) > 0 for chunks in results)
