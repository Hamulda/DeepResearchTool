"""Modulární architektura pro paměťové úložiště s abstraktní základní třídou
a konkrétní implementací pro ChromaDB

Author: Senior Python/MLOps Agent
"""

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
from typing import Any


@dataclass
class Document:
    """Reprezentace dokumentu v paměťovém úložišti"""

    content: str
    metadata: dict[str, Any]
    embedding: list[float] | None = None
    id: str | None = None


class BaseMemoryStore(ABC):
    """Abstraktní základní třída pro paměťové úložiště"""

    @abstractmethod
    async def initialize(self) -> None:
        """Inicializace úložiště"""

    @abstractmethod
    async def add_document(self, document: Document) -> str:
        """Přidání dokumentu do úložiště

        Args:
            document: Dokument k uložení

        Returns:
            ID uloženého dokumentu

        """

    @abstractmethod
    async def add_documents(self, documents: list[Document]) -> list[str]:
        """Přidání více dokumentů najednou

        Args:
            documents: Seznam dokumentů k uložení

        Returns:
            Seznam ID uložených dokumentů

        """

    @abstractmethod
    async def search(self, query: str, k: int = 5, **kwargs) -> list[Document]:
        """Vyhledání podobných dokumentů

        Args:
            query: Vyhledávací dotaz
            k: Počet výsledků k vrácení
            **kwargs: Dodatečné parametry pro vyhledávání

        Returns:
            Seznam nejpodobnějších dokumentů

        """

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Smazání dokumentu podle ID

        Args:
            document_id: ID dokumentu k smazání

        Returns:
            True pokud byl dokument smazán, False jinak

        """

    @abstractmethod
    async def clear(self) -> None:
        """Vymazání všech dokumentů z úložiště"""


class ChromaMemoryStore(BaseMemoryStore):
    """Konkrétní implementace paměťového úložiště pomocí ChromaDB"""

    def __init__(
        self,
        collection_name: str = "research_documents",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "BAAI/bge-large-en-v1.5",
    ):
        """Inicializace ChromaDB store

        Args:
            collection_name: Název kolekce v ChromaDB
            persist_directory: Adresář pro persistentní uložení
            embedding_model: Název embedding modelu z Hugging Face

        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model

        self.client = None
        self.collection = None
        self.embeddings = None
        self._initialized = False

    async def initialize(self) -> None:
        """Inicializace ChromaDB klienta a kolekce"""
        if self._initialized:
            return

        try:
            import chromadb
            from chromadb.config import Settings
            from sentence_transformers import SentenceTransformer

            # Inicializace ChromaDB klienta
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )

            # Vytvoření nebo získání kolekce
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Research documents collection"},
                )

            # Inicializace embedding modelu
            self.embeddings = SentenceTransformer(self.embedding_model)

            self._initialized = True

        except ImportError as e:
            raise ImportError(f"Chybí závislosti pro ChromaDB: {e}")
        except Exception as e:
            raise RuntimeError(f"Chyba při inicializaci ChromaDB: {e}")

    async def _create_embedding(self, text: str) -> list[float]:
        """Vytvoření embedding pro text

        Args:
            text: Text k zakódování

        Returns:
            Vektor embeddingu

        """
        if not self.embeddings:
            raise RuntimeError("Embedding model není inicializován")

        # Spuštění v thread pool pro neblokující běh
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self.embeddings.encode, text)

        return embedding.tolist()

    async def add_document(self, document: Document) -> str:
        """Přidání jediného dokumentu"""
        if not self._initialized:
            await self.initialize()

        # Vytvoření embeddingu pokud není poskytnut
        if document.embedding is None:
            document.embedding = await self._create_embedding(document.content)

        # Generování ID pokud není poskytnuto
        if document.id is None:
            import uuid

            document.id = str(uuid.uuid4())

        # Přidání do ChromaDB
        self.collection.add(
            embeddings=[document.embedding],
            documents=[document.content],
            metadatas=[document.metadata],
            ids=[document.id],
        )

        return document.id

    async def add_documents(self, documents: list[Document]) -> list[str]:
        """Přidání více dokumentů najednou"""
        if not self._initialized:
            await self.initialize()

        embeddings = []
        contents = []
        metadatas = []
        ids = []

        for doc in documents:
            # Vytvoření embeddingu pokud není poskytnut
            if doc.embedding is None:
                doc.embedding = await self._create_embedding(doc.content)

            # Generování ID pokud není poskytnuto
            if doc.id is None:
                import uuid

                doc.id = str(uuid.uuid4())

            embeddings.append(doc.embedding)
            contents.append(doc.content)
            metadatas.append(doc.metadata)
            ids.append(doc.id)

        # Batch přidání do ChromaDB
        self.collection.add(embeddings=embeddings, documents=contents, metadatas=metadatas, ids=ids)

        return ids

    async def search(self, query: str, k: int = 5, **kwargs) -> list[Document]:
        """Vyhledání podobných dokumentů pomocí sémantického vyhledávání

        Args:
            query: Vyhledávací dotaz
            k: Počet výsledků k vrácení
            **kwargs: Dodatečné filtry (where, where_document)

        Returns:
            Seznam nejpodobnějších dokumentů

        """
        if not self._initialized:
            await self.initialize()

        # Vytvoření embeddingu pro dotaz
        query_embedding = await self._create_embedding(query)

        # Vyhledání v ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=kwargs.get("where"),
            where_document=kwargs.get("where_document"),
        )

        # Konverze výsledků na Document objekty
        documents = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                doc = Document(
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] if results["metadatas"][0] else {},
                    id=results["ids"][0][i],
                )
                # Přidání distance skóre do metadat
                if results.get("distances") and results["distances"][0]:
                    doc.metadata["distance"] = results["distances"][0][i]

                documents.append(doc)

        return documents

    async def delete_document(self, document_id: str) -> bool:
        """Smazání dokumentu podle ID"""
        if not self._initialized:
            await self.initialize()

        try:
            self.collection.delete(ids=[document_id])
            return True
        except Exception:
            return False

    async def clear(self) -> None:
        """Vymazání všech dokumentů z kolekce"""
        if not self._initialized:
            await self.initialize()

        # Smazání kolekce a vytvoření nové
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name, metadata={"description": "Research documents collection"}
        )


# Factory funkce pro získání paměťového úložiště
def get_memory_store(config: dict[str, Any]) -> BaseMemoryStore:
    """Factory funkce pro získání paměťového úložiště podle konfigurace

    Args:
        config: Konfigurační slovník

    Returns:
        Instance paměťového úložiště

    """
    store_type = config.get("memory_store", {}).get("type", "chroma")

    if store_type == "chroma":
        return ChromaMemoryStore(
            collection_name=config.get("memory_store", {}).get(
                "collection_name", "research_documents"
            ),
            persist_directory=config.get("memory_store", {}).get(
                "persist_directory", "./chroma_db"
            ),
            embedding_model=config.get("memory_store", {}).get(
                "embedding_model", "BAAI/bge-large-en-v1.5"
            ),
        )
    raise ValueError(f"Nepodporovaný typ memory store: {store_type}")
