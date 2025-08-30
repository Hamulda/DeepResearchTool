"""
Integration testy pro Fázi 2 - M1 Optimalizace
Testování spolupráce všech M1-optimalizovaných komponent
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time

from src.vector_stores.chroma_m1_client import M1OptimizedChromaClient, create_m1_chroma_client
from src.optimization.streaming_data_processor import M1StreamingDataProcessor, StreamingConfig
from src.synthesis.hierarchical_summarizer import M1HierarchicalSummarizer, create_m1_summarizer
from src.optimization.m1_metal_llm import M1MetalLLMClient, create_m1_llm_client


@pytest.mark.integration
class TestM1OptimizedPipeline:
    """Integration testy pro kompletní M1-optimalizovaný pipeline"""

    def setup_method(self):
        """Setup před každým testem"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_file = Path(self.temp_dir) / "test_data.json"

        # Vytvoření test dat
        self.test_documents = [
            {
                "id": f"doc_{i}",
                "title": f"Test Document {i}",
                "content": f"This is test document number {i}. " * 20 +
                          f"It contains information about topic {i % 3}. " * 10,
                "metadata": {"category": f"category_{i % 3}", "importance": i}
            }
            for i in range(100)
        ]

        # Uložení test dat
        with open(self.test_data_file, 'w') as f:
            json.dump(self.test_documents, f)

    def test_complete_m1_pipeline(self):
        """Test kompletního M1 pipeline: Streaming → Sumarizace → Vektorování"""

        # 1. Streaming Data Processor
        processor = M1StreamingDataProcessor(
            StreamingConfig(chunk_size=10, memory_limit_mb=256)
        )

        # 2. Hierarchical Summarizer (bez LLM pro test)
        summarizer = create_m1_summarizer(
            llm_client=None,  # Použije fallback
            compression_ratio=0.5
        )

        # 3. ChromaDB M1 Client
        chroma_client = create_m1_chroma_client(
            collection_name="test_m1_pipeline",
            max_memory_mb=256
        )

        try:
            processed_docs = []
            summarized_docs = []

            # Fáze 1: Streamované načítání a zpracování
            def document_processor(chunk):
                # Extrakce content z dokumentů
                return [doc['content'] for doc in chunk]

            for chunk in processor.process_data_chunks(
                processor.stream_json_file(self.test_data_file),
                document_processor,
                chunk_size=10
            ):
                processed_docs.extend(chunk)

            assert len(processed_docs) == 100

            # Fáze 2: Hierarchická sumarizace
            for i, doc_content in enumerate(processed_docs[:10]):  # Test na 10 dokumentech
                summary_result = summarizer.summarize_document(doc_content)
                summarized_docs.append({
                    'id': f"summary_{i}",
                    'original_content': doc_content,
                    'summary': summary_result['final_summary'],
                    'compression_ratio': summary_result['statistics']['compression_ratio']
                })

            assert len(summarized_docs) == 10

            # Ověření komprese
            avg_compression = sum(doc['compression_ratio'] for doc in summarized_docs) / len(summarized_docs)
            assert avg_compression < 0.8  # Alespoň 20% komprese

            # Fáze 3: Vektorování do ChromaDB
            documents = [doc['summary'] for doc in summarized_docs]
            metadatas = [{'type': 'summary', 'original_id': doc['id']} for doc in summarized_docs]
            ids = [doc['id'] for doc in summarized_docs]

            # Mock embeddings
            embeddings = [[0.1 * i, 0.2 * i, 0.3 * i] for i in range(len(documents))]

            chroma_client.add_documents(documents, embeddings, metadatas, ids)

            # Ověření uložení
            stats = chroma_client.get_collection_stats()
            assert stats['document_count'] == 10
            assert stats['memory_usage_percent'] < 80  # Memory usage pod 80%

            # Test dotazování
            query_embedding = [0.15, 0.25, 0.35]
            results = chroma_client.query_documents(query_embedding, n_results=3)

            assert len(results['documents']) == 3
            assert len(results['metadatas']) == 3

            print("✅ Kompletní M1 pipeline test úspěšný")

        finally:
            # Cleanup
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_memory_efficiency(self):
        """Test paměťové efektivity M1 komponent"""
        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024

        # Test streaming processoru
        processor = M1StreamingDataProcessor(
            StreamingConfig(chunk_size=50, gc_frequency=10)
        )

        # Zpracování velkého množství dat
        large_data = [{"id": i, "text": "x" * 1000} for i in range(1000)]
        large_file = Path(self.temp_dir) / "large_data.json"

        with open(large_file, 'w') as f:
            json.dump(large_data, f)

        def identity_processor(chunk):
            return len(chunk)

        total_processed = 0
        for chunk_size in processor.process_data_chunks(
            processor.stream_json_file(large_file),
            identity_processor
        ):
            total_processed += chunk_size

        assert total_processed == 1000

        # Kontrola memory usage
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory

        # Memory increase by neměl být moc velký (streaming processing)
        assert memory_increase < 100  # Méně než 100MB increase

        print(f"✅ Memory efficiency test: {memory_increase:.1f}MB increase")

    @pytest.mark.asyncio
    async def test_performance_benchmarks(self):
        """Benchmark test pro M1 optimalizace"""

        # Benchmark streaming processoru
        processor = M1StreamingDataProcessor()

        start_time = time.time()

        def simple_processor(chunk):
            # Simulace zpracování
            return [len(item.get('content', '')) for item in chunk]

        total_items = 0
        for result in processor.process_data_chunks(
            processor.stream_json_file(self.test_data_file),
            simple_processor,
            chunk_size=20
        ):
            total_items += len(result)

        processing_time = time.time() - start_time
        items_per_second = total_items / processing_time

        # Performance assertions
        assert items_per_second > 50  # Alespoň 50 items/s
        assert processing_time < 5.0   # Zpracování pod 5 sekund

        print(f"✅ Performance: {items_per_second:.1f} items/s")

    def test_error_handling_and_recovery(self):
        """Test error handling a recovery mechanismů"""

        # Test s chybným JSON souborem
        broken_file = Path(self.temp_dir) / "broken.json"
        with open(broken_file, 'w') as f:
            f.write('{"broken": json}')  # Invalid JSON

        processor = M1StreamingDataProcessor()

        # Streaming procesor by měl zvládnout chybný JSON
        items_processed = 0
        try:
            for item in processor.stream_json_file(broken_file):
                items_processed += 1
        except Exception:
            # Je OK pokud selže, ale neměl by crashnout aplikaci
            pass

        # Test ChromaDB error recovery
        chroma_client = create_m1_chroma_client("test_error_recovery")

        # Pokus o přidání duplicitních ID
        docs = ["doc1", "doc2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        metadatas = [{}, {}]
        ids = ["same_id", "same_id"]  # Duplicitní ID

        try:
            chroma_client.add_documents(docs, embeddings, metadatas, ids)
            assert False, "Mělo by selhat kvůli duplicitním ID"
        except ValueError:
            # Očekávaná chyba
            pass

        print("✅ Error handling test úspěšný")


@pytest.mark.integration
class TestM1LLMIntegration:
    """Integration testy pro M1 LLM optimalizace"""

    @pytest.mark.skipif(True, reason="Vyžaduje běžící Ollama server")
    def test_llm_metal_acceleration(self):
        """Test Metal acceleration pro LLM (pouze pokud je Ollama dostupný)"""

        try:
            llm_client = create_m1_llm_client(
                model_name="llama3:8b-instruct-q4_K_M",
                use_metal=True
            )

            # Test single generation
            response = llm_client.generate(
                "Napiš krátký souhrn o umělé inteligenci.",
                max_tokens=100,
                temperature=0.3
            )

            assert len(response) > 10

            # Test performance stats
            stats = llm_client.get_performance_stats()
            assert stats['average_tokens_per_second'] > 0

            print(f"✅ LLM Metal test: {stats['average_tokens_per_second']:.1f} tok/s")

        except Exception as e:
            pytest.skip(f"Ollama není dostupný: {e}")

    def test_llm_integration_with_summarizer(self):
        """Test integrace LLM s hierarchickou sumarizací"""

        # Mock LLM client
        mock_llm = Mock()
        mock_llm.generate.return_value = "This is a test summary."

        summarizer = create_m1_summarizer(
            llm_client=mock_llm,
            compression_ratio=0.4
        )

        test_text = "This is a very long document. " * 100

        result = summarizer.summarize_document(test_text)

        assert result['final_summary'] == "This is a test summary."
        assert result['statistics']['compression_ratio'] < 0.5

        # Ověření že LLM byl zavolán
        mock_llm.generate.assert_called()

        print("✅ LLM-Summarizer integration test úspěšný")


@pytest.mark.integration
class TestM1ResourceManagement:
    """Testy pro M1 resource management"""

    def test_concurrent_operations(self):
        """Test současného běhu více M1 komponent"""
        import threading
        import concurrent.futures

        results = []

        def streaming_task():
            processor = M1StreamingDataProcessor(
                StreamingConfig(chunk_size=10, memory_limit_mb=128)
            )

            # Vytvoření malého test souboru
            test_data = [{"id": i, "text": f"data {i}"} for i in range(50)]
            test_file = Path(tempfile.mktemp(suffix='.json'))

            with open(test_file, 'w') as f:
                json.dump(test_data, f)

            count = 0
            for item in processor.stream_json_file(test_file):
                count += 1

            test_file.unlink()
            return count

        def chroma_task():
            client = create_m1_chroma_client("concurrent_test", max_memory_mb=128)

            docs = [f"Document {i}" for i in range(20)]
            embeddings = [[0.1 * i, 0.2 * i] for i in range(20)]
            metadatas = [{"index": i} for i in range(20)]
            ids = [f"id_{i}" for i in range(20)]

            client.add_documents(docs, embeddings, metadatas, ids)
            stats = client.get_collection_stats()
            return stats['document_count']

        # Spuštění úkolů současně
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(streaming_task)
            future2 = executor.submit(chroma_task)

            result1 = future1.result(timeout=30)
            result2 = future2.result(timeout=30)

        assert result1 == 50  # Streaming task
        assert result2 == 20  # ChromaDB task

        print("✅ Concurrent operations test úspěšný")

    def test_memory_pressure_handling(self):
        """Test handling memory pressure na M1"""

        # Simulace high memory usage
        chroma_client = create_m1_chroma_client("memory_pressure_test")

        # Přidání většího množství dokumentů pro test memory pressure
        large_docs = [f"Large document content {i}. " * 100 for i in range(100)]
        large_embeddings = [[0.1 * i] * 384 for i in range(100)]  # 384-dim embeddings
        large_metadatas = [{"size": "large", "index": i} for i in range(100)]
        large_ids = [f"large_doc_{i}" for i in range(100)]

        # Batch přidání s memory monitoring
        batch_size = 10
        for i in range(0, len(large_docs), batch_size):
            batch_docs = large_docs[i:i+batch_size]
            batch_embeddings = large_embeddings[i:i+batch_size]
            batch_metadatas = large_metadatas[i:i+batch_size]
            batch_ids = large_ids[i:i+batch_size]

            chroma_client.add_documents(
                batch_docs, batch_embeddings, batch_metadatas, batch_ids,
                batch_size=batch_size
            )

            # Kontrola memory stats
            stats = chroma_client.get_collection_stats()

            # Memory usage by neměl překročit rozumné meze
            assert stats['memory_usage_percent'] < 90

        final_stats = chroma_client.get_collection_stats()
        assert final_stats['document_count'] == 100

        print(f"✅ Memory pressure test: {final_stats['memory_usage_percent']}% memory")


if __name__ == "__main__":
    # Spuštění integration testů
    print("🧪 Spouštím M1 Integration testy...")

    # Lze spustit jednotlivé testy
    test_pipeline = TestM1OptimizedPipeline()
    test_pipeline.setup_method()

    try:
        test_pipeline.test_complete_m1_pipeline()
        test_pipeline.test_memory_efficiency()
        test_pipeline.test_error_handling_and_recovery()

        print("✅ Všechny integration testy prošly!")

    except Exception as e:
        print(f"❌ Integration test selhal: {e}")
        raise
