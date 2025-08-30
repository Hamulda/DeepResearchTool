"""Vector Database Integration for M1-Optimized RAG
Local Qdrant/Chroma with scalar quantization and batch optimization
"""

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
import logging
from pathlib import Path
import pickle
from typing import Any, List, Dict

import numpy as np

from .context_manager import DocumentChunk
from .memory_optimizer import MemoryOptimizer
from .async_batch_processor import DatabaseBatchProcessor, BatchConfig

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import Distance, VectorParams

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("Qdrant not available, using fallback vector store")

try:
    import chromadb
    from chromadb.config import Settings

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("ChromaDB not available")


@dataclass
class VectorConfig:
    """Vector database configuration with batch optimization"""

    provider: str = "qdrant"  # qdrant, chroma, local
    collection_name: str = "documents"
    vector_size: int = 384  # MiniLM embedding size
    distance_metric: str = "cosine"
    enable_quantization: bool = True
    quantization_type: str = "int8"  # int8, binary
    enable_mmap: bool = True
    batch_size: int = 200  # Increased for better performance
    index_params: Dict[str, Any] = None

    # Batch optimization settings
    enable_batch_optimization: bool = True
    max_concurrent_batches: int = 3
    batch_timeout: int = 300


@dataclass
class SearchResult:
    """Vector search result"""

    chunk_id: str
    score: float
    chunk: DocumentChunk
    metadata: Dict[str, Any] | None = None


class BaseVectorStore(ABC):
    """Abstract base class for vector stores with batch optimization"""

    def __init__(self, config: VectorConfig, optimizer: MemoryOptimizer):
        self.config = config
        self.optimizer = optimizer

        # Initialize batch processor for database operations
        if config.enable_batch_optimization:
            self.batch_processor = DatabaseBatchProcessor(
                batch_size=config.batch_size
            )
        else:
            self.batch_processor = None

    @abstractmethod
    async def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        """Add chunks with embeddings to the store"""

    @abstractmethod
    async def add_chunks_batch(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> int:
        """Optimized batch addition of chunks with embeddings"""

    @abstractmethod
    async def search(
        self, query_embedding: np.ndarray, limit: int = 10, filter_conditions: Dict | None = None
    ) -> List[SearchResult]:
        """Search for similar chunks"""

    @abstractmethod
    async def search_batch(
        self, query_embeddings: List[np.ndarray], limit: int = 10
    ) -> List[List[SearchResult]]:
        """Batch search for multiple queries"""

    @abstractmethod
    async def delete_collection(self) -> None:
        """Delete the collection"""

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""


class QdrantVectorStore(BaseVectorStore):
    """Qdrant vector store implementation with batch optimization"""

    def __init__(self, config: VectorConfig, optimizer: MemoryOptimizer, data_path: Path):
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant client not available")

        super().__init__(config, optimizer)
        self.data_path = data_path
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Initialize Qdrant client with local storage
        self.client = QdrantClient(path=str(self.data_path))
        self._setup_collection()

    def _setup_collection(self):
        """Setup Qdrant collection with M1 optimization"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_exists = any(
                col.name == self.config.collection_name for col in collections.collections
            )

            if not collection_exists:
                # Create collection with optimized settings
                vector_params = VectorParams(
                    size=self.config.vector_size,
                    distance=(
                        Distance.COSINE if self.config.distance_metric == "cosine" else Distance.DOT
                    ),
                )

                self.client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=vector_params,
                    # Optimized settings for M1
                    optimizers_config=models.OptimizersConfigDiff(
                        memmap_threshold=20000,  # Use memory mapping for large collections
                        indexing_threshold=0,  # Start indexing immediately
                    )
                )

                # Enable quantization for memory efficiency
                if self.config.enable_quantization:
                    if self.config.quantization_type == "int8":
                        quantization_config = models.ScalarQuantization(
                            scalar=models.ScalarQuantizationConfig(
                                type=models.ScalarType.INT8,
                                always_ram=False,  # Use disk for M1 memory constraints
                            )
                        )
                    else:  # binary
                        quantization_config = models.BinaryQuantization(
                            binary=models.BinaryQuantizationConfig(always_ram=False)
                        )

                    self.client.update_collection(
                        collection_name=self.config.collection_name,
                        quantization_config=quantization_config,
                    )

                logger.info(f"Created optimized Qdrant collection: {self.config.collection_name}")

        except Exception as e:
            logger.error(f"Failed to setup Qdrant collection: {e}")
            raise

    async def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        """Legacy single-batch addition (for compatibility)"""
        await self.add_chunks_batch(chunks, embeddings)

    async def add_chunks_batch(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> int:
        """Optimized batch addition with concurrent processing"""
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Number of chunks must match embeddings")

        if not self.batch_processor:
            # Fallback to simple batch processing
            return await self._simple_batch_add(chunks, embeddings)

        # Prepare documents for batch processing
        documents = []
        for i, chunk in enumerate(chunks):
            doc = {
                'chunk': chunk,
                'embedding': embeddings[i],
                'point_id': hash(chunk.chunk_id) % (2**63)
            }
            documents.append(doc)

        # Use DatabaseBatchProcessor for optimized insertion
        return await self.batch_processor.batch_insert_vector_data(self, documents)

    async def _simple_batch_add(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> int:
        """Simple batch addition without batch processor"""
        points = []
        for i, chunk in enumerate(chunks):
            point = models.PointStruct(
                id=hash(chunk.chunk_id) % (2**63),
                vector=embeddings[i].tolist(),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                    "importance_score": chunk.importance_score or 0.0,
                    "section_type": chunk.section_type or "content",
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata or {},
                },
            )
            points.append(point)

        # Batch upload with memory monitoring
        total_inserted = 0
        batch_size = self.config.batch_size

        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]

            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda b=batch: self.client.upsert(
                    collection_name=self.config.collection_name,
                    points=b,
                    wait=True  # Ensure completion for batch consistency
                ),
            )

            total_inserted += len(batch)

            # Memory pressure check and optimization
            if self.optimizer.check_memory_pressure()["pressure"]:
                self.optimizer.force_gc()
                # Small delay to allow GC to complete
                await asyncio.sleep(0.1)

        return total_inserted

    async def batch_upsert_points(self, documents: List[Dict]) -> int:
        """Custom batch upsert for DatabaseBatchProcessor"""
        points = []
        for doc in documents:
            chunk = doc['chunk']
            embedding = doc['embedding']
            point_id = doc['point_id']

            point = models.PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                    "importance_score": chunk.importance_score or 0.0,
                    "section_type": chunk.section_type or "content",
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata or {},
                },
            )
            points.append(point)

        # Single batch upsert
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.client.upsert(
                collection_name=self.config.collection_name,
                points=points,
                wait=True
            ),
        )

        return len(points)

    async def search(
        self, query_embedding: np.ndarray, limit: int = 10, filter_conditions: Dict | None = None
    ) -> List[SearchResult]:
        """Single search with optimized parameters"""
        search_results = await self.search_batch([query_embedding], limit)
        return search_results[0] if search_results else []

    async def search_batch(
        self, query_embeddings: List[np.ndarray], limit: int = 10
    ) -> List[List[SearchResult]]:
        """Optimized batch search"""
        if not query_embeddings:
            return []

        # Prepare batch search requests
        search_requests = []
        for embedding in query_embeddings:
            request = models.SearchRequest(
                vector=embedding.tolist(),
                limit=limit,
                with_payload=True,
                score_threshold=0.5  # Filter low-quality results early
            )
            search_requests.append(request)

        # Execute batch search
        batch_results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.client.search_batch(
                collection_name=self.config.collection_name,
                requests=search_requests
            )
        )

        # Convert results
        all_results = []
        for search_result in batch_results:
            results = []
            for hit in search_result:
                chunk = DocumentChunk(
                    text=hit.payload["text"],
                    chunk_id=hit.payload["chunk_id"],
                    document_id=hit.payload["document_id"],
                    chunk_index=hit.payload["chunk_index"],
                    start_char=0,
                    end_char=len(hit.payload["text"]),
                    token_count=hit.payload["token_count"],
                    importance_score=hit.payload.get("importance_score"),
                    section_type=hit.payload.get("section_type"),
                    metadata=hit.payload.get("metadata"),
                )

                results.append(
                    SearchResult(
                        chunk_id=hit.payload["chunk_id"],
                        score=hit.score,
                        chunk=chunk,
                        metadata={"qdrant_id": hit.id},
                    )
                )
            all_results.append(results)

        return all_results

    async def delete_collection(self) -> None:
        """Delete Qdrant collection"""
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.client.delete_collection(self.config.collection_name)
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive Qdrant collection statistics"""
        try:
            info = self.client.get_collection(self.config.collection_name)

            # Additional performance metrics
            cluster_info = self.client.get_cluster_info()

            return {
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "status": info.status,
                "optimizer_status": info.optimizer_status.status if info.optimizer_status else None,
                "disk_usage_bytes": info.disk_usage if hasattr(info, "disk_usage") else None,
                "ram_usage_bytes": info.ram_usage if hasattr(info, "ram_usage") else None,
                "cluster_status": cluster_info.name if cluster_info else "standalone",
                "quantization_enabled": self.config.enable_quantization,
                "batch_size": self.config.batch_size,
            }
        except Exception as e:
            logger.error(f"Failed to get Qdrant stats: {e}")
            return {"error": str(e)}


# Add missing classes for validation
class VectorStore(BaseVectorStore):
    """Main VectorStore class for backward compatibility"""
    pass


class VectorStoreFactory:
    """Factory for creating vector store instances"""

    @staticmethod
    def create_vector_store(config: VectorConfig, optimizer: MemoryOptimizer, data_path: Path = None):
        """Create appropriate vector store based on configuration"""
        if data_path is None:
            data_path = Path("./vector_store_data")

        if config.provider == "qdrant" and QDRANT_AVAILABLE:
            return QdrantVectorStore(config, optimizer, data_path)
        elif config.provider == "chroma" and CHROMA_AVAILABLE:
            # ChromaDB implementation would go here
            raise NotImplementedError("ChromaDB implementation coming soon")
        else:
            # Fallback to simple in-memory store
            return SimpleVectorStore(config, optimizer)


class SimpleVectorStore(BaseVectorStore):
    """Simple in-memory vector store for testing"""

    def __init__(self, config: VectorConfig, optimizer: MemoryOptimizer):
        super().__init__(config, optimizer)
        self.vectors = []
        self.chunks = []

    async def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        """Add chunks to simple store"""
        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            self.vectors.append(embeddings[i])

    async def add_chunks_batch(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> int:
        """Batch add chunks"""
        await self.add_chunks(chunks, embeddings)
        return len(chunks)

    async def search(self, query_embedding: np.ndarray, limit: int = 10, filter_conditions: Dict = None) -> List[SearchResult]:
        """Simple cosine similarity search"""
        if not self.vectors:
            return []

        # Compute similarities
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = np.dot(query_embedding, vector) / (np.linalg.norm(query_embedding) * np.linalg.norm(vector))
            similarities.append((similarity, i))

        # Sort and get top results
        similarities.sort(reverse=True)
        results = []

        for score, idx in similarities[:limit]:
            results.append(SearchResult(
                chunk_id=self.chunks[idx].chunk_id,
                score=float(score),
                chunk=self.chunks[idx]
            ))

        return results

    async def search_batch(self, query_embeddings: List[np.ndarray], limit: int = 10) -> List[List[SearchResult]]:
        """Batch search"""
        results = []
        for embedding in query_embeddings:
            result = await self.search(embedding, limit)
            results.append(result)
        return results

    async def delete_collection(self) -> None:
        """Clear simple store"""
        self.vectors.clear()
        self.chunks.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get simple store stats"""
        return {
            "vectors_count": len(self.vectors),
            "chunks_count": len(self.chunks),
            "provider": "simple"
        }


# Batch optimization utilities
async def batch_add_documents(
    vector_store: BaseVectorStore,
    chunks: List[DocumentChunk],
    embeddings: np.ndarray,
    batch_size: int = None
) -> int:
    """Utility function for optimized batch document addition"""
    if batch_size is None:
        batch_size = vector_store.config.batch_size

    total_added = 0

    for i in range(0, len(chunks), batch_size):
        end_idx = min(i + batch_size, len(chunks))
        batch_chunks = chunks[i:end_idx]
        batch_embeddings = embeddings[i:end_idx]

        added = await vector_store.add_chunks_batch(batch_chunks, batch_embeddings)
        total_added += added

        # Progress logging
        if (i // batch_size) % 10 == 0:
            logger.info(f"Processed {i + len(batch_chunks)}/{len(chunks)} documents")

    return total_added


async def batch_search_documents(
    vector_store: BaseVectorStore,
    query_embeddings: List[np.ndarray],
    limit: int = 10,
    batch_size: int = 50
) -> List[List[SearchResult]]:
    """Utility function for optimized batch document search"""
    all_results = []

    for i in range(0, len(query_embeddings), batch_size):
        end_idx = min(i + batch_size, len(query_embeddings))
        batch_queries = query_embeddings[i:end_idx]

        if hasattr(vector_store, 'search_batch'):
            batch_results = await vector_store.search_batch(batch_queries, limit)
        else:
            # Fallback for stores without batch search
            batch_results = []
            for query in batch_queries:
                result = await vector_store.search(query, limit)
                batch_results.append(result)

        all_results.extend(batch_results)

    return all_results

