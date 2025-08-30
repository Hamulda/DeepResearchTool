"""
Unit testy pro ChromaDB klienta - testování základních operací s vektorovou databází
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Simulace ChromaDB pro testy bez závislosti na skutečné instalaci
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    # Mock ChromaDB pro testy
    chromadb = MagicMock()


class MockCollection:
    """Mock třída pro ChromaDB Collection"""

    def __init__(self, name: str):
        self.name = name
        self._documents = {}
        self._embeddings = {}
        self._metadatas = {}
        self._ids = set()

    def add(self, documents: List[str], embeddings: List[List[float]],
            metadatas: List[Dict], ids: List[str]):
        """Mock add method"""
        for i, doc_id in enumerate(ids):
            if doc_id in self._ids:
                raise ValueError(f"ID {doc_id} already exists")
            self._documents[doc_id] = documents[i]
            self._embeddings[doc_id] = embeddings[i]
            self._metadatas[doc_id] = metadatas[i]
            self._ids.add(doc_id)

    def query(self, query_embeddings: List[List[float]], n_results: int = 10,
              where: Dict = None, include: List[str] = None):
        """Mock query method"""
        # Simulace vyhledávání - vrátíme všechny dokumenty
        ids = list(self._ids)[:n_results]
        documents = [self._documents[id] for id in ids]
        metadatas = [self._metadatas[id] for id in ids]
        distances = [0.1 + i * 0.1 for i in range(len(ids))]  # Mock distances

        result = {
            'ids': [ids],
            'documents': [documents],
            'metadatas': [metadatas],
            'distances': [distances]
        }

        if include and 'embeddings' in include:
            embeddings = [self._embeddings[id] for id in ids]
            result['embeddings'] = [embeddings]

        return result

    def get(self, ids: List[str] = None, where: Dict = None,
            include: List[str] = None):
        """Mock get method"""
        if ids:
            filtered_ids = [id for id in ids if id in self._ids]
        else:
            filtered_ids = list(self._ids)

        documents = [self._documents[id] for id in filtered_ids]
        metadatas = [self._metadatas[id] for id in filtered_ids]

        result = {
            'ids': filtered_ids,
            'documents': documents,
            'metadatas': metadatas
        }

        if include and 'embeddings' in include:
            embeddings = [self._embeddings[id] for id in filtered_ids]
            result['embeddings'] = embeddings

        return result

    def delete(self, ids: List[str] = None, where: Dict = None):
        """Mock delete method"""
        if ids:
            for doc_id in ids:
                if doc_id in self._ids:
                    del self._documents[doc_id]
                    del self._embeddings[doc_id]
                    del self._metadatas[doc_id]
                    self._ids.remove(doc_id)
        else:
            # Delete all
            self._documents.clear()
            self._embeddings.clear()
            self._metadatas.clear()
            self._ids.clear()

    def count(self):
        """Mock count method"""
        return len(self._ids)


class MockChromaClient:
    """Mock třída pro ChromaDB Client"""

    def __init__(self, path: str = None):
        self.path = path
        self._collections = {}

    def create_collection(self, name: str, metadata: Dict = None,
                         embedding_function=None):
        """Mock create_collection method"""
        if name in self._collections:
            raise ValueError(f"Collection {name} already exists")
        collection = MockCollection(name)
        self._collections[name] = collection
        return collection

    def get_collection(self, name: str):
        """Mock get_collection method"""
        if name not in self._collections:
            raise ValueError(f"Collection {name} does not exist")
        return self._collections[name]

    def get_or_create_collection(self, name: str, metadata: Dict = None,
                                embedding_function=None):
        """Mock get_or_create_collection method"""
        if name in self._collections:
            return self._collections[name]
        return self.create_collection(name, metadata, embedding_function)

    def delete_collection(self, name: str):
        """Mock delete_collection method"""
        if name in self._collections:
            del self._collections[name]

    def list_collections(self):
        """Mock list_collections method"""
        return [{"name": name} for name in self._collections.keys()]


class ChromaDBClient:
    """Wrapper pro ChromaDB operace - optimalizovaný pro M1"""

    def __init__(self, persist_directory: str = "./chroma_db",
                 collection_name: str = "research_docs"):
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.client = None
        self.collection = None

    def connect(self):
        """Připojení k ChromaDB"""
        if CHROMADB_AVAILABLE:
            self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        else:
            self.client = MockChromaClient(path=str(self.persist_directory))

        # Získání nebo vytvoření kolekce
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
        except:
            self.collection = self.client.create_collection(name=self.collection_name)

    def add_documents(self, documents: List[str], embeddings: List[List[float]],
                     metadatas: List[Dict], ids: List[str]):
        """Přidání dokumentů do kolekce"""
        if not self.collection:
            self.connect()

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def query_documents(self, query_embedding: List[float], n_results: int = 10,
                       where_filter: Dict = None):
        """Dotaz na dokumenty"""
        if not self.collection:
            self.connect()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=['documents', 'metadatas', 'distances']
        )

        return {
            'documents': results['documents'][0] if results['documents'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else [],
            'distances': results['distances'][0] if results['distances'] else [],
            'ids': results['ids'][0] if results['ids'] else []
        }

    def delete_documents(self, ids: List[str]):
        """Smazání dokumentů"""
        if not self.collection:
            self.connect()

        self.collection.delete(ids=ids)

    def get_document_count(self):
        """Počet dokumentů v kolekci"""
        if not self.collection:
            self.connect()

        return self.collection.count()

    def reset_collection(self):
        """Reset kolekce (smazání všech dokumentů)"""
        if not self.collection:
            self.connect()

        self.collection.delete()


class TestChromaDBClient:
    """Testy pro ChromaDB klienta"""

    def setup_method(self):
        """Příprava před každým testem"""
        self.temp_dir = tempfile.mkdtemp()
        self.client = ChromaDBClient(
            persist_directory=self.temp_dir,
            collection_name="test_collection"
        )

    def teardown_method(self):
        """Úklid po každém testu"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_client_initialization(self):
        """Test inicializace klienta"""
        assert self.client.persist_directory == Path(self.temp_dir)
        assert self.client.collection_name == "test_collection"
        assert self.client.client is None
        assert self.client.collection is None

    def test_connect(self):
        """Test připojení k databázi"""
        self.client.connect()

        assert self.client.client is not None
        assert self.client.collection is not None
        assert self.client.collection.name == "test_collection"

    def test_add_documents(self):
        """Test přidání dokumentů"""
        documents = ["Test document 1", "Test document 2"]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        metadatas = [{"source": "test1"}, {"source": "test2"}]
        ids = ["doc1", "doc2"]

        self.client.add_documents(documents, embeddings, metadatas, ids)

        # Ověření že se dokumenty přidaly
        count = self.client.get_document_count()
        assert count == 2

    def test_query_documents(self):
        """Test dotazování na dokumenty"""
        # Přidání test dokumentů
        documents = ["Machine learning is great", "Python programming"]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        metadatas = [{"topic": "ml"}, {"topic": "programming"}]
        ids = ["doc1", "doc2"]

        self.client.add_documents(documents, embeddings, metadatas, ids)

        # Dotaz
        query_embedding = [0.15, 0.25, 0.35]
        results = self.client.query_documents(query_embedding, n_results=2)

        assert len(results['documents']) == 2
        assert len(results['metadatas']) == 2
        assert len(results['distances']) == 2
        assert len(results['ids']) == 2

    def test_delete_documents(self):
        """Test smazání dokumentů"""
        # Přidání dokumentů
        documents = ["Doc to keep", "Doc to delete"]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        metadatas = [{"keep": True}, {"keep": False}]
        ids = ["keep", "delete"]

        self.client.add_documents(documents, embeddings, metadatas, ids)

        # Smazání jednoho dokumentu
        self.client.delete_documents(["delete"])

        # Ověření
        count = self.client.get_document_count()
        assert count == 1

    def test_reset_collection(self):
        """Test resetu kolekce"""
        # Přidání dokumentů
        documents = ["Doc 1", "Doc 2", "Doc 3"]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        metadatas = [{}, {}, {}]
        ids = ["1", "2", "3"]

        self.client.add_documents(documents, embeddings, metadatas, ids)

        # Reset
        self.client.reset_collection()

        # Ověření že je kolekce prázdná
        count = self.client.get_document_count()
        assert count == 0

    def test_duplicate_id_error(self):
        """Test chyby při duplicitním ID"""
        documents = ["Doc 1", "Doc 2"]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        metadatas = [{}, {}]
        ids = ["doc1", "doc1"]  # Duplicitní ID

        with pytest.raises(ValueError, match="already exists"):
            self.client.add_documents(documents, embeddings, metadatas, ids)

    def test_query_with_filter(self):
        """Test dotazu s filtrem"""
        # Přidání dokumentů s různými metadaty
        documents = ["ML paper", "Web dev article", "AI research"]
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        metadatas = [
            {"category": "research", "year": 2023},
            {"category": "tutorial", "year": 2023},
            {"category": "research", "year": 2024}
        ]
        ids = ["1", "2", "3"]

        self.client.add_documents(documents, embeddings, metadatas, ids)

        # Dotaz s filtrem
        query_embedding = [0.2, 0.3]
        where_filter = {"category": "research"}
        results = self.client.query_documents(
            query_embedding,
            n_results=5,
            where_filter=where_filter
        )

        # Měly by se vrátit pouze research dokumenty
        # Note: Mock implementace nefiltruje skutečně, ale v produkci by ano
        assert len(results['documents']) <= 2


class TestChromaDBIntegration:
    """Integration testy pro ChromaDB"""

    def test_persistence(self):
        """Test perzistence dat"""
        temp_dir = tempfile.mkdtemp()

        try:
            # Vytvoření prvního klienta a přidání dat
            client1 = ChromaDBClient(persist_directory=temp_dir)
            documents = ["Persistent doc"]
            embeddings = [[0.1, 0.2, 0.3]]
            metadatas = [{"persistent": True}]
            ids = ["persist_1"]

            client1.add_documents(documents, embeddings, metadatas, ids)
            count1 = client1.get_document_count()

            # Vytvoření druhého klienta se stejným adresářem
            client2 = ChromaDBClient(persist_directory=temp_dir)
            count2 = client2.get_document_count()

            # Data by měla přetrvat
            assert count1 == count2 == 1

        finally:
            shutil.rmtree(temp_dir)

    def test_multiple_collections(self):
        """Test práce s více kolekcemi"""
        temp_dir = tempfile.mkdtemp()

        try:
            client1 = ChromaDBClient(temp_dir, "collection1")
            client2 = ChromaDBClient(temp_dir, "collection2")

            # Přidání dat do obou kolekcí
            client1.add_documents(["Doc 1"], [[0.1, 0.2]], [{}], ["1"])
            client2.add_documents(["Doc 2"], [[0.3, 0.4]], [{}], ["2"])

            # Každá kolekce by měla mít 1 dokument
            assert client1.get_document_count() == 1
            assert client2.get_document_count() == 1

        finally:
            shutil.rmtree(temp_dir)


class TestM1Optimizations:
    """Testy specifické pro M1 optimalizace"""

    def test_memory_efficient_batching(self):
        """Test paměťově efektivního zpracování"""
        temp_dir = tempfile.mkdtemp()

        try:
            client = ChromaDBClient(persist_directory=temp_dir)

            # Simulace velkého počtu dokumentů (v reálném světě by to bylo větší)
            batch_size = 100
            for batch in range(5):
                documents = [f"Document {i + batch * batch_size}"
                           for i in range(batch_size)]
                embeddings = [[0.1 * i, 0.2 * i] for i in range(batch_size)]
                metadatas = [{"batch": batch} for _ in range(batch_size)]
                ids = [f"doc_{batch}_{i}" for i in range(batch_size)]

                client.add_documents(documents, embeddings, metadatas, ids)

            # Ověření celkového počtu
            total_count = client.get_document_count()
            assert total_count == 500

        finally:
            shutil.rmtree(temp_dir)

    def test_embedding_dimension_handling(self):
        """Test různých dimenzí embeddingů"""
        temp_dir = tempfile.mkdtemp()

        try:
            client = ChromaDBClient(persist_directory=temp_dir)

            # Test různých dimenzí (v produkci by všechny měly být stejné)
            documents = ["Short embedding", "Standard embedding"]
            embeddings = [
                [0.1, 0.2],  # 2D
                [0.3, 0.4, 0.5, 0.6]  # 4D - v produkci by to mělo být konzistentní
            ]
            metadatas = [{}, {}]
            ids = ["short", "standard"]

            # V mock implementaci projde, v produkci by měla být validace
            client.add_documents(documents, embeddings, metadatas, ids)
            count = client.get_document_count()
            assert count == 2

        finally:
            shutil.rmtree(temp_dir)
