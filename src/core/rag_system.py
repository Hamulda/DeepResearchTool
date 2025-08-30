"""Základní RAG systém s Milvus Lite a sentence-transformers pro Fázi 1.
Implementuje vektorové embeddings a podobnostní vyhledávání.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Optional imports with fallbacks
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False
    duckdb = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
    HAS_MILVUS = True
except ImportError:
    HAS_MILVUS = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Reprezentuje jeden chunk dokumentu pro vektorové vyhledávání."""

    id: str
    content: str
    metadata: dict[str, Any]
    embedding: np.ndarray | None = None


class EmbeddingGenerator:
    """Generátor vektorových embeddingů pomocí sentence-transformers.
    Optimalizován pro Apple Silicon (Metal).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "mps",  # Metal Performance Shaders pro Apple Silicon
    ):
        self.model_name = model_name
        self.device = device

        if not HAS_SENTENCE_TRANSFORMERS:
            logger.warning("SentenceTransformers not available, embedding generation disabled")
            self.model = None
            return

        try:
            # Načtení modelu s Metal optimalizací
            self.model = SentenceTransformer(model_name, device=device)
            logger.info(
                "Embedding model loaded",
                model=model_name,
                device=device,
                embedding_dim=self.model.get_sentence_embedding_dimension(),
            )
        except Exception as e:
            # Fallback na CPU pokud Metal není dostupný
            logger.warning(f"Failed to use Metal, falling back to CPU: {e}")
            self.device = "cpu"
            self.model = SentenceTransformer(model_name, device="cpu")

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Generuje embeddings pro batch textů."""
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            logger.info(
                "Generated embeddings", batch_size=len(texts), embedding_dim=embeddings.shape[1]
            )

            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    def encode_single(self, text: str) -> np.ndarray:
        """Generuje embedding pro jeden text."""
        return self.encode_batch([text])[0]

    @property
    def embedding_dimension(self) -> int:
        """Vrací dimenzi embeddingů."""
        return self.model.get_sentence_embedding_dimension()


class MilvusVectorStore:
    """Vektorová databáze pomocí Milvus Lite pro lokální použití.
    """

    def __init__(
        self,
        collection_name: str = "document_embeddings",
        embedding_dim: int = 384,
        host: str = "localhost",
        port: int = 19530,
    ):
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.host = host
        self.port = port
        self.collection = None

        self._connect_to_milvus()
        self._create_collection()

    def _connect_to_milvus(self) -> None:
        """Připojí se k Milvus serveru."""
        try:
            connections.connect(alias="default", host=self.host, port=self.port)
            logger.info("Connected to Milvus", host=self.host, port=self.port)
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def _create_collection(self) -> None:
        """Vytvoří kolekci pokud neexistuje."""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            logger.info("Using existing collection", name=self.collection_name)
            return

        # Definice schématu kolekce
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255, is_primary=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
        ]

        schema = CollectionSchema(fields=fields, description="Document embeddings for RAG system")

        # Vytvoření kolekce
        self.collection = Collection(name=self.collection_name, schema=schema)

        # Vytvoření indexu pro vektorové vyhledávání
        index_params = {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}}

        self.collection.create_index(field_name="embedding", index_params=index_params)

        logger.info("Created new collection with index", name=self.collection_name)

    async def insert_documents(self, chunks: list[DocumentChunk]) -> None:
        """Vloží dokumenty do vektorové databáze."""
        if not chunks:
            return

        # Příprava dat pro vložení
        entities = [
            [chunk.id for chunk in chunks],  # IDs
            [chunk.content for chunk in chunks],  # Content
            [chunk.metadata.get("source", "") for chunk in chunks],  # Source
            [chunk.metadata.get("timestamp", "") for chunk in chunks],  # Timestamp
            [chunk.embedding.tolist() for chunk in chunks],  # Embeddings
        ]

        try:
            # Vložení dat
            self.collection.insert(entities)
            self.collection.flush()

            logger.info(
                "Inserted documents into Milvus", count=len(chunks), collection=self.collection_name
            )

        except Exception as e:
            logger.error(f"Failed to insert documents: {e}")
            raise

    async def search_similar(
        self, query_embedding: np.ndarray, top_k: int = 5, score_threshold: float = 0.7
    ) -> list[tuple[str, str, float]]:
        """Vyhledá podobné dokumenty."""
        try:
            # Načtení kolekce do paměti pro vyhledávání
            self.collection.load()

            # Parametry vyhledávání
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

            # Vyhledávání
            results = self.collection.search(
                data=[query_embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["id", "content", "source"],
            )

            # Zpracování výsledků
            similar_docs = []
            for hits in results:
                for hit in hits:
                    if hit.score >= score_threshold:
                        similar_docs.append(
                            (hit.entity.get("id"), hit.entity.get("content"), hit.score)
                        )

            logger.info(
                "Found similar documents",
                query_results=len(similar_docs),
                threshold=score_threshold,
            )

            return similar_docs

        except Exception as e:
            logger.error(f"Failed to search similar documents: {e}")
            raise

    def get_collection_stats(self) -> dict[str, Any]:
        """Vrací statistiky kolekce."""
        try:
            self.collection.load()
            num_entities = self.collection.num_entities

            return {
                "collection_name": self.collection_name,
                "num_documents": num_entities,
                "embedding_dimension": self.embedding_dim,
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
            }

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}


class LocalRAGSystem:
    """Kompletní RAG systém kombinující DuckDB, Milvus a sentence-transformers.
    """

    def __init__(
        self,
        data_dir: Path,
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "rag_documents",
    ):
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.collection_name = collection_name

        # Inicializace komponent s fallbacky
        if HAS_SENTENCE_TRANSFORMERS:
            self.embedding_generator = EmbeddingGenerator(model_name)
        else:
            logger.warning("SentenceTransformers not available")
            self.embedding_generator = None

        if HAS_MILVUS and self.embedding_generator:
            self.vector_store = MilvusVectorStore(
                collection_name=collection_name,
                embedding_dim=self.embedding_generator.embedding_dimension,
            )
        else:
            logger.warning("Milvus not available")
            self.vector_store = None

        if HAS_DUCKDB:
            self.duckdb_conn = duckdb.connect(":memory:")
        else:
            logger.warning("DuckDB not available")
            self.duckdb_conn = None

        logger.info(
            "RAG system initialized",
            model=model_name,
            collection=collection_name,
        )

    async def index_documents_from_parquet(
        self, table_name: str, content_column: str = "content", chunk_size: int = 500
    ) -> None:
        """Indexuje dokumenty z Parquet souboru do vektorové databáze."""
        parquet_path = self.data_dir / f"{table_name}.parquet"

        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        try:
            # Načtení dat z Parquet pomocí DuckDB
            query = f"""
            SELECT id, {content_column}, source, timestamp
            FROM read_parquet('{parquet_path}')
            WHERE {content_column} IS NOT NULL 
            AND LENGTH({content_column}) > 10
            """

            result = self.duckdb_conn.execute(query).fetchall()
            columns = [desc[0] for desc in self.duckdb_conn.description]

            # Rozdělení na chunky pro batch processing
            documents = [dict(zip(columns, row, strict=False)) for row in result]

            for i in range(0, len(documents), chunk_size):
                batch = documents[i : i + chunk_size]
                await self._process_document_batch(batch, content_column)

            logger.info("Document indexing completed", total_docs=len(documents), table=table_name)

        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            raise

    async def _process_document_batch(
        self, documents: list[dict[str, Any]], content_column: str
    ) -> None:
        """Zpracuje batch dokumentů - generuje embeddings a ukládá do Milvus."""
        # Extrakce textů pro embedding
        texts = [doc[content_column] for doc in documents]

        # Generování embeddingů
        embeddings = self.embedding_generator.encode_batch(texts)

        # Vytvoření DocumentChunk objektů
        chunks = []
        for i, doc in enumerate(documents):
            chunk = DocumentChunk(
                id=str(doc.get("id", i)),
                content=doc[content_column],
                metadata={
                    "source": doc.get("source", "unknown"),
                    "timestamp": doc.get("timestamp", ""),
                },
                embedding=embeddings[i],
            )
            chunks.append(chunk)

        # Vložení do Milvus
        await self.vector_store.insert_documents(chunks)

    async def search_relevant_context(
        self, query: str, top_k: int = 5, score_threshold: float = 0.7
    ) -> list[dict[str, Any]]:
        """Vyhledá relevantní kontext pro dotaz."""
        try:
            # Generování query embeddingu
            query_embedding = self.embedding_generator.encode_single(query)

            # Vyhledání podobných dokumentů
            similar_docs = await self.vector_store.search_similar(
                query_embedding=query_embedding, top_k=top_k, score_threshold=score_threshold
            )

            # Formátování výsledků
            context_results = []
            for doc_id, content, score in similar_docs:
                context_results.append(
                    {
                        "id": doc_id,
                        "content": content,
                        "similarity_score": score,
                        "relevance": "high" if score > 0.8 else "medium",
                    }
                )

            logger.info(
                "Retrieved relevant context",
                query=query[:50] + "...",
                results_count=len(context_results),
            )

            return context_results

        except Exception as e:
            logger.error(f"Failed to search relevant context: {e}")
            raise

    def get_system_stats(self) -> dict[str, Any]:
        """Vrací statistiky RAG systému."""
        vector_stats = self.vector_store.get_collection_stats()

        return {
            "embedding_model": self.model_name,
            "embedding_dimension": self.embedding_generator.embedding_dimension,
            "device": self.embedding_generator.device,
            "vector_store": vector_stats,
            "data_directory": str(self.data_dir),
        }

    def cleanup(self) -> None:
        """Vyčistí zdroje."""
        self.duckdb_conn.close()
        logger.info("RAG system cleanup completed")


# Příklad použití RAG systému
async def example_rag_usage():
    """Příklad použití RAG systému."""
    # Inicializace RAG systému
    rag_system = LocalRAGSystem(data_dir=Path("./data/parquet"), model_name="all-MiniLM-L6-v2")

    try:
        # Indexování dokumentů (předpokládáme existující Parquet soubor)
        await rag_system.index_documents_from_parquet("scraped_documents")

        # Vyhledání relevantního kontextu
        query = "What are the latest trends in AI research?"
        context = await rag_system.search_relevant_context(query, top_k=3)

        print(f"Query: {query}")
        print(f"Found {len(context)} relevant documents:")

        for i, doc in enumerate(context, 1):
            print(f"\n{i}. Similarity: {doc['similarity_score']:.3f}")
            print(f"Content: {doc['content'][:200]}...")

        # Statistiky systému
        stats = rag_system.get_system_stats()
        print(f"\nRAG System Stats: {stats}")

    finally:
        rag_system.cleanup()


if __name__ == "__main__":
    asyncio.run(example_rag_usage())
