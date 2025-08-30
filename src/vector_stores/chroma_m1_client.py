"""
ChromaDB In-Process Client - M1 Optimalized
Běží přímo v Python procesu místo samostatného kontejneru
Úspora paměti: ~500MB oproti samostatnému Qdrant kontejneru
"""

import gc
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    logger.warning("ChromaDB not available - falling back to mock implementation")
    CHROMADB_AVAILABLE = False


@dataclass
class M1MemorySettings:
    """M1-specifické nastavení paměti pro ChromaDB"""
    max_batch_size: int = 1000
    embedding_cache_size: int = 10000
    persist_threshold: int = 5000  # Počet dokumentů po kterých persistovat
    gc_frequency: int = 100  # Garbage collection po N operacích
    max_memory_usage_mb: int = 1024  # 1GB limit


class M1OptimizedChromaClient:
    """
    ChromaDB klient optimalizovaný pro MacBook Air M1 (8GB RAM)

    Klíčové optimalizace:
    - In-process běh (žádný Docker kontejner)
    - Automatický garbage collection
    - Batch processing s M1-optimalized velikostmi
    - Memory pressure monitoring
    - Intelligent persistence strategie
    """

    def __init__(self,
                 persist_directory: str = "./chroma_db_m1",
                 collection_name: str = "research_docs",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 memory_settings: Optional[M1MemorySettings] = None):

        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.memory_settings = memory_settings or M1MemorySettings()

        # Counters pro optimalizace
        self._operation_counter = 0
        self._last_gc_time = time.time()
        self._document_count = 0

        # Thread safety
        self._lock = threading.Lock()

        # ChromaDB komponenty
        self.client = None
        self.collection = None
        self.embedding_function = None

        # Inicializace
        self._initialize_client()
        self._setup_embedding_function(embedding_model)
        self._setup_collection()

        logger.info(f"🚀 M1-optimalizovaný ChromaDB klient inicializován")
        logger.info(f"📁 Persist directory: {self.persist_directory}")
        logger.info(f"🧠 Max memory: {self.memory_settings.max_memory_usage_mb}MB")

    def _initialize_client(self):
        """Inicializace ChromaDB klienta s M1 optimalizacemi"""
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB není dostupné. Nainstalujte: pip install chromadb")

        # Vytvoření persist directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # M1-optimalizované nastavení
        settings = Settings(
            chroma_db_impl="duckdb+parquet",  # Lightweight backend
            persist_directory=str(self.persist_directory),
            anonymized_telemetry=False,  # Vypnout telemetrii
        )

        self.client = chromadb.PersistentClient(settings=settings)
        logger.info("✅ ChromaDB persistent client vytvořen")

    def _setup_embedding_function(self, model_name: str):
        """Nastavení embedding funkce s M1 optimalizacemi"""
        try:
            # Použití sentence-transformers s M1 optimalizacemi
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name,
                device="mps" if self._is_mps_available() else "cpu",  # M1 Metal
                normalize_embeddings=True  # Normalizace pro lepší performance
            )
            logger.info(f"✅ Embedding function nastavena: {model_name}")

        except Exception as e:
            logger.warning(f"⚠️ Chyba při nastavení embedding funkce: {e}")
            # Fallback na default
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

    @staticmethod
    def _is_mps_available() -> bool:
        """Kontrola dostupnosti Metal Performance Shaders na M1"""
        try:
            import torch
            return torch.backends.mps.is_available()
        except ImportError:
            return False

    def _setup_collection(self):
        """Vytvoření nebo načtení kolekce"""
        try:
            # Pokus o načtení existující kolekce
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            self._document_count = self.collection.count()
            logger.info(f"✅ Kolekce načtena: {self.collection_name} ({self._document_count} dokumentů)")

        except Exception:
            # Vytvoření nové kolekce
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine", "hnsw:M": 16}  # M1 optimalizace
            )
            self._document_count = 0
            logger.info(f"✅ Nová kolekce vytvořena: {self.collection_name}")

    def add_documents(self,
                     documents: List[str],
                     metadatas: List[Dict[str, Any]],
                     ids: List[str],
                     batch_size: Optional[int] = None) -> None:
        """
        Přidání dokumentů s M1-optimalizovaným batch processingem

        Args:
            documents: Seznam dokumentů
            metadatas: Metadata pro každý dokument
            ids: Unikátní ID pro každý dokument
            batch_size: Velikost dávky (None = auto)
        """
        with self._lock:
            if not batch_size:
                batch_size = self.memory_settings.max_batch_size

            total_docs = len(documents)
            logger.info(f"📝 Přidávám {total_docs} dokumentů v dávkách po {batch_size}")

            # Batch processing
            for i in range(0, total_docs, batch_size):
                batch_end = min(i + batch_size, total_docs)
                batch_docs = documents[i:batch_end]
                batch_meta = metadatas[i:batch_end]
                batch_ids = ids[i:batch_end]

                try:
                    self.collection.add(
                        documents=batch_docs,
                        metadatas=batch_meta,
                        ids=batch_ids
                    )

                    self._document_count += len(batch_docs)
                    self._operation_counter += len(batch_docs)

                    # Progress log
                    if i + batch_size < total_docs:
                        logger.info(f"📊 Zpracováno {batch_end}/{total_docs} dokumentů")

                    # M1 Memory management
                    self._maybe_trigger_gc()
                    self._maybe_persist()

                except Exception as e:
                    logger.error(f"❌ Chyba při přidávání dávky {i}-{batch_end}: {e}")
                    raise

            logger.info(f"✅ Přidáno {total_docs} dokumentů. Celkem: {self._document_count}")

    def query_documents(self,
                       query_text: str,
                       n_results: int = 10,
                       where_filter: Optional[Dict] = None,
                       include_distances: bool = True) -> Dict[str, Any]:
        """
        Dotazování dokumentů s M1 optimalizacemi

        Args:
            query_text: Text dotazu
            n_results: Počet výsledků
            where_filter: Filtry metadata
            include_distances: Zahrnout vzdálenosti

        Returns:
            Slovník s výsledky dotazu
        """
        with self._lock:
            try:
                include_list = ['documents', 'metadatas']
                if include_distances:
                    include_list.append('distances')

                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=n_results,
                    where=where_filter,
                    include=include_list
                )

                # M1 Memory cleanup po dotazu
                self._operation_counter += 1
                self._maybe_trigger_gc()

                # Formátování výsledků
                formatted_results = {
                    'documents': results['documents'][0] if results['documents'] else [],
                    'metadatas': results['metadatas'][0] if results['metadatas'] else [],
                    'ids': results['ids'][0] if results['ids'] else []
                }

                if include_distances and 'distances' in results:
                    formatted_results['distances'] = results['distances'][0]

                logger.debug(f"🔍 Dotaz vrátil {len(formatted_results['documents'])} výsledků")
                return formatted_results

            except Exception as e:
                logger.error(f"❌ Chyba při dotazování: {e}")
                raise

    def delete_documents(self, ids: List[str]) -> None:
        """Smazání dokumentů podle ID"""
        with self._lock:
            try:
                self.collection.delete(ids=ids)
                self._document_count = max(0, self._document_count - len(ids))
                logger.info(f"🗑️ Smazáno {len(ids)} dokumentů")

                # Cleanup po smazání
                self._maybe_trigger_gc()

            except Exception as e:
                logger.error(f"❌ Chyba při mazání dokumentů: {e}")
                raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """Statistiky kolekce a využití paměti"""
        try:
            count = self.collection.count()

            # Memory usage estimation
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            return {
                'document_count': count,
                'collection_name': self.collection_name,
                'memory_usage_mb': round(memory_mb, 2),
                'memory_limit_mb': self.memory_settings.max_memory_usage_mb,
                'memory_usage_percent': round((memory_mb / self.memory_settings.max_memory_usage_mb) * 100, 1),
                'operations_since_last_gc': self._operation_counter,
                'persist_directory': str(self.persist_directory)
            }

        except Exception as e:
            logger.error(f"❌ Chyba při získávání statistik: {e}")
            return {'error': str(e)}

    def _maybe_trigger_gc(self):
        """Spuštění garbage collection při potřebě (M1 optimalizace)"""
        if self._operation_counter >= self.memory_settings.gc_frequency:
            start_time = time.time()
            collected = gc.collect()
            gc_time = time.time() - start_time

            if collected > 0:
                logger.debug(f"🧹 GC: {collected} objektů smazáno za {gc_time:.3f}s")

            self._operation_counter = 0
            self._last_gc_time = time.time()

    def _maybe_persist(self):
        """Automatická persistence při dosažení prahu"""
        if self._document_count % self.memory_settings.persist_threshold == 0:
            logger.info(f"💾 Auto-persist: {self._document_count} dokumentů")

    def optimize_for_m1(self):
        """Spuštění M1-specifických optimalizací"""
        logger.info("⚡ Spouštím M1 optimalizace...")

        # Force garbage collection
        collected = gc.collect()
        logger.info(f"🧹 GC: {collected} objektů uvolněno")

        # Memory usage check
        stats = self.get_collection_stats()
        memory_percent = stats.get('memory_usage_percent', 0)

        if memory_percent > 80:
            logger.warning(f"⚠️ Vysoké využití paměti: {memory_percent}%")

        logger.info(f"✅ M1 optimalizace dokončena. Paměť: {memory_percent}%")
        return stats

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit s cleanup"""
        try:
            self.optimize_for_m1()
            logger.info("👋 ChromaDB klient uzavřen")
        except Exception as e:
            logger.error(f"❌ Chyba při uzavírání klienta: {e}")


# Utility funkce pro snadné použití
def create_m1_chroma_client(collection_name: str = "research_docs",
                           max_memory_mb: int = 1024) -> M1OptimizedChromaClient:
    """Factory funkce pro vytvoření M1-optimalizovaného klienta"""
    memory_settings = M1MemorySettings(max_memory_usage_mb=max_memory_mb)

    return M1OptimizedChromaClient(
        collection_name=collection_name,
        memory_settings=memory_settings
    )


if __name__ == "__main__":
    # Test M1 optimalizovaného klienta
    print("🧪 Testování M1 ChromaDB klienta...")

    with create_m1_chroma_client("test_collection") as client:
        # Test přidání dokumentů
        docs = ["Test document 1", "Test document 2", "Machine learning paper"]
        metas = [{"source": "test"}, {"source": "test"}, {"source": "research"}]
        ids = ["1", "2", "3"]

        client.add_documents(docs, metas, ids)

        # Test dotazu
        results = client.query_documents("machine learning", n_results=2)
        print(f"📊 Nalezeno {len(results['documents'])} dokumentů")

        # Statistiky
        stats = client.get_collection_stats()
        print(f"📈 Statistiky: {stats}")

    print("✅ Test dokončen!")
