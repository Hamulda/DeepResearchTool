"""RAG Indexer for On-Device Vector Search
Stream Parquet rows, chunking with overlap, small embedding models
Hybrid retrieval with Chroma/Qdrant integration
"""

import asyncio
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np

# Optional import with fallback
try:
    import polars as pl
    HAS_POLARS = True
    LazyFrameType = pl.LazyFrame
    DataFrameType = pl.DataFrame
except ImportError:
    HAS_POLARS = False
    pl = None
    LazyFrameType = Any
    DataFrameType = Any

from ..core.context_manager import ChunkingConfig, HierarchicalChunker
from ..core.memory_optimizer import MemoryOptimizer
from ..core.vector_store import SearchResult, VectorConfig, VectorStoreFactory

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available - using mock embeddings")


@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""

    model_name: str = "all-MiniLM-L6-v2"  # Small, efficient model
    device: str = "cpu"  # Use CPU on M1 for compatibility
    batch_size: int = 32
    max_seq_length: int = 384
    normalize_embeddings: bool = True
    cache_folder: str | None = None


@dataclass
class RAGConfig:
    """RAG indexer configuration"""

    # Embedding settings
    embedding_config: EmbeddingConfig = None

    # Chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100

    # Vector store settings
    vector_store_provider: str = "local"  # local, qdrant, chroma
    vector_store_path: Path | None = None

    # Indexing settings
    batch_size: int = 100
    enable_hybrid_search: bool = True

    # Performance settings
    max_workers: int = 4
    enable_streaming: bool = True
    memory_limit_gb: float = 2.0

    def __post_init__(self):
        if self.embedding_config is None:
            self.embedding_config = EmbeddingConfig()


@dataclass
class IndexResult:
    """Result from indexing operation"""

    total_chunks: int
    indexed_chunks: int
    failed_chunks: int
    embedding_time_seconds: float
    indexing_time_seconds: float
    index_size_mb: float
    memory_usage_gb: float


class EmbeddingGenerator:
    """Generate embeddings using small transformer models"""

    def __init__(self, config: EmbeddingConfig, optimizer: MemoryOptimizer):
        self.config = config
        self.optimizer = optimizer
        self.model = None
        self.model_loaded = False

    async def initialize(self) -> bool:
        """Initialize embedding model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("Using mock embeddings - sentence-transformers not available")
            return True

        try:
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            def load_model():
                return SentenceTransformer(
                    self.config.model_name,
                    device=self.config.device,
                    cache_folder=self.config.cache_folder,
                )

            with ThreadPoolExecutor(max_workers=1) as executor:
                self.model = await loop.run_in_executor(executor, load_model)

            # Configure model
            self.model.max_seq_length = self.config.max_seq_length

            self.model_loaded = True
            logger.info(f"✅ Embedding model loaded: {self.config.model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return False

    async def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for list of texts"""
        if not self.model_loaded:
            if not await self.initialize():
                # Return mock embeddings
                return self._generate_mock_embeddings(texts)

        if not SENTENCE_TRANSFORMERS_AVAILABLE or not self.model:
            return self._generate_mock_embeddings(texts)

        try:
            # Generate embeddings in thread pool
            loop = asyncio.get_event_loop()

            def encode_texts():
                return self.model.encode(
                    texts,
                    batch_size=self.config.batch_size,
                    normalize_embeddings=self.config.normalize_embeddings,
                    show_progress_bar=False,
                )

            with ThreadPoolExecutor(max_workers=1) as executor:
                embeddings = await loop.run_in_executor(executor, encode_texts)

            return embeddings

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return self._generate_mock_embeddings(texts)

    def _generate_mock_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generate mock embeddings for testing"""
        # Use text hash for deterministic embeddings
        embeddings = []
        for text in texts:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            # Convert hash to numbers and normalize
            embedding = np.array([int(text_hash[i : i + 2], 16) for i in range(0, 32, 2)])
            embedding = embedding.astype(np.float32) / 255.0  # Normalize to [0,1]

            # Pad or truncate to 384 dimensions (MiniLM size)
            if len(embedding) < 384:
                embedding = np.pad(embedding, (0, 384 - len(embedding)))
            else:
                embedding = embedding[:384]

            embeddings.append(embedding)

        return np.array(embeddings)

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        if self.model_loaded and self.model:
            return self.model.get_sentence_embedding_dimension()
        return 384  # MiniLM default


class BM25Scorer:
    """Simple BM25 implementation for sparse retrieval"""

    def __init__(self, documents: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.doc_lengths = [len(doc.split()) for doc in documents]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        self.doc_count = len(documents)

        # Build term frequency and document frequency
        self.term_freqs = []
        self.doc_freqs = {}

        for doc in documents:
            terms = doc.lower().split()
            term_freq = {}

            for term in terms:
                term_freq[term] = term_freq.get(term, 0) + 1
                if term not in self.doc_freqs:
                    self.doc_freqs[term] = 0

            self.term_freqs.append(term_freq)

            # Count document frequencies
            for term in set(terms):
                self.doc_freqs[term] += 1

    def score(self, query: str, doc_idx: int) -> float:
        """Calculate BM25 score for query-document pair"""
        query_terms = query.lower().split()
        score = 0.0

        doc_length = self.doc_lengths[doc_idx]
        term_freq = self.term_freqs[doc_idx]

        for term in query_terms:
            if term in term_freq:
                # Term frequency component
                tf = term_freq[term]
                tf_component = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                )

                # Inverse document frequency component
                doc_freq = self.doc_freqs.get(term, 0)
                if doc_freq > 0:
                    idf = np.log((self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5))
                    score += idf * tf_component

        return score

    def get_top_k(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Get top-k documents for query"""
        scores = [(i, self.score(query, i)) for i in range(self.doc_count)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class RAGIndexer:
    """Main RAG indexer with hybrid retrieval capabilities"""

    def __init__(self, config: RAGConfig, optimizer: MemoryOptimizer):
        self.config = config
        self.optimizer = optimizer
        self.embedding_generator = EmbeddingGenerator(config.embedding_config, optimizer)
        self.chunker = HierarchicalChunker(
            ChunkingConfig(
                chunk_size=config.chunk_size,
                overlap_size=config.chunk_overlap,
                min_chunk_size=config.min_chunk_size,
            ),
            optimizer,
        )

        # Vector store
        self.vector_store = None
        self.chunks_metadata = []  # Store chunk metadata

        # BM25 for sparse retrieval
        self.bm25_scorer = None
        self.chunk_texts = []

        # Statistics
        self.total_indexed = 0
        self.last_index_time = None

    async def initialize(self) -> bool:
        """Initialize RAG indexer"""
        logger.info("🔍 Initializing RAG Indexer...")

        # Initialize embedding generator
        if not await self.embedding_generator.initialize():
            logger.warning("Embedding generator initialization failed")

        # Initialize vector store
        vector_config = VectorConfig(
            provider=self.config.vector_store_provider,
            vector_size=self.embedding_generator.get_embedding_dimension(),
            collection_name="rag_chunks",
            enable_quantization=True,
            batch_size=self.config.batch_size,
        )

        store_path = self.config.vector_store_path or Path("rag_vector_store")
        self.vector_store = VectorStoreFactory.create_vector_store(
            vector_config, self.optimizer, store_path
        )

        logger.info(
            f"✅ RAG Indexer initialized with {self.config.vector_store_provider} vector store"
        )
        return True

    async def index_parquet_dataset(
        self, parquet_path: Path, text_column: str = "content", metadata_columns: list[str] = None
    ) -> IndexResult:
        """Index documents from Parquet dataset"""
        start_time = datetime.now()

        logger.info(f"📚 Indexing Parquet dataset: {parquet_path}")

        # Read Parquet with lazy evaluation
        try:
            lazy_df = pl.scan_parquet(parquet_path)

            # Select required columns
            columns = [text_column]
            if metadata_columns:
                columns.extend(metadata_columns)

            lazy_df = lazy_df.select(columns)

            # Process in batches for memory efficiency
            total_chunks = 0
            indexed_chunks = 0
            failed_chunks = 0

            # Stream processing
            if self.config.enable_streaming:
                async for batch_result in self._process_parquet_streaming(
                    lazy_df, text_column, metadata_columns
                ):
                    total_chunks += batch_result["total"]
                    indexed_chunks += batch_result["indexed"]
                    failed_chunks += batch_result["failed"]
            else:
                # Collect all data (for smaller datasets)
                df = lazy_df.collect()
                batch_result = await self._process_dataframe_batch(
                    df, text_column, metadata_columns
                )
                total_chunks = batch_result["total"]
                indexed_chunks = batch_result["indexed"]
                failed_chunks = batch_result["failed"]

            # Build BM25 index for hybrid retrieval
            if self.config.enable_hybrid_search and self.chunk_texts:
                logger.info("🔍 Building BM25 sparse index...")
                self.bm25_scorer = BM25Scorer(self.chunk_texts)

            # Calculate timing and memory usage
            end_time = datetime.now()
            indexing_time = (end_time - start_time).total_seconds()

            result = IndexResult(
                total_chunks=total_chunks,
                indexed_chunks=indexed_chunks,
                failed_chunks=failed_chunks,
                embedding_time_seconds=indexing_time * 0.7,  # Estimate
                indexing_time_seconds=indexing_time,
                index_size_mb=self._estimate_index_size(),
                memory_usage_gb=self.optimizer.check_memory_pressure()["used_percent"]
                / 100
                * 8,  # Estimate for 8GB system
            )

            self.total_indexed = indexed_chunks
            self.last_index_time = end_time

            logger.info(
                f"✅ Indexing completed: {indexed_chunks}/{total_chunks} chunks in {indexing_time:.1f}s"
            )
            return result

        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            raise

    async def _process_parquet_streaming(
        self, lazy_df: LazyFrameType, text_column: str, metadata_columns: list[str] | None
    ) -> Iterator[dict[str, int]]:
        """Process Parquet data in streaming batches"""
        if not HAS_POLARS or lazy_df is None:
            logger.error("Polars not available for streaming processing")
            return iter([])

        # For streaming, we need to implement batching
        # This is a simplified implementation - in practice, you'd use Polars streaming
        try:
            df = lazy_df.collect()  # For now, collect all (would be improved with true streaming)
        except Exception as e:
            logger.error(f"Error collecting lazy frame: {e}")
            return iter([])

        batch_size = self.optimizer.get_optimal_batch_size(record_size_bytes=2048)

        for i in range(0, len(df), batch_size):
            batch_df = df[i : i + batch_size]
            result = await self._process_dataframe_batch(batch_df, text_column, metadata_columns)
            yield result

            # Memory pressure check
            if self.optimizer.check_memory_pressure()["pressure"]:
                self.optimizer.force_gc()

    async def _process_dataframe_batch(
        self, df: DataFrameType, text_column: str, metadata_columns: list[str] | None
    ) -> dict[str, int]:
        """Process single DataFrame batch"""
        if not HAS_POLARS or df is None:
            logger.error("Polars not available for DataFrame processing")
            return {"total": 0, "indexed": 0, "failed": 0}

        total = 0
        indexed = 0
        failed = 0

        batch_chunks = []
        batch_texts = []

        try:
            # Process each row
            for row in df.iter_rows(named=True):
                try:
                    text = row[text_column]
                    if not text or len(text.strip()) < self.config.min_chunk_size:
                        failed += 1
                        continue

                    # Create document ID from row data
                    doc_id = f"doc_{hash(str(row)) % 1000000:06d}"

                    # Chunk the document
                    chunks = self.chunker.chunk_document(text, doc_id)

                    for chunk in chunks:
                        # Add metadata from other columns
                        if metadata_columns:
                            chunk.metadata = chunk.metadata or {}
                            for col in metadata_columns:
                                if col in row:
                                    chunk.metadata[col] = row[col]

                        batch_chunks.append(chunk)
                        batch_texts.append(chunk.text)
                        total += 1

                except Exception as e:
                    logger.warning(f"Failed to process row: {e}")
                    failed += 1

            # Generate embeddings for batch
            if batch_chunks:
                try:
                    embeddings = await self.embedding_generator.generate_embeddings(batch_texts)

                    # Add to vector store
                    await self.vector_store.add_chunks(batch_chunks, embeddings)

                    # Store for BM25
                    self.chunks_metadata.extend(batch_chunks)
                    self.chunk_texts.extend(batch_texts)

                    indexed = len(batch_chunks)

                except Exception as e:
                    logger.error(f"Failed to index batch: {e}")
                    failed += total
                    indexed = 0

        except Exception as e:
            logger.error(f"Error processing DataFrame batch: {e}")
            failed += total

        return {"total": total, "indexed": indexed, "failed": failed}

    async def hybrid_search(
        self, query: str, limit: int = 10, vector_weight: float = 0.7, sparse_weight: float = 0.3
    ) -> list[SearchResult]:
        """Perform hybrid search combining vector and sparse retrieval"""
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized")

        # Vector search
        query_embedding = await self.embedding_generator.generate_embeddings([query])
        vector_results = await self.vector_store.search(query_embedding[0], limit=limit * 2)

        # Sparse search (BM25)
        sparse_results = []
        if self.bm25_scorer:
            bm25_scores = self.bm25_scorer.get_top_k(query, k=limit * 2)

            for doc_idx, score in bm25_scores:
                if doc_idx < len(self.chunks_metadata):
                    chunk = self.chunks_metadata[doc_idx]
                    sparse_results.append(
                        SearchResult(chunk_id=chunk.chunk_id, score=score, chunk=chunk)
                    )

        # Combine and re-rank results
        combined_results = self._combine_search_results(
            vector_results, sparse_results, vector_weight, sparse_weight
        )

        return combined_results[:limit]

    def _combine_search_results(
        self,
        vector_results: list[SearchResult],
        sparse_results: list[SearchResult],
        vector_weight: float,
        sparse_weight: float,
    ) -> list[SearchResult]:
        """Combine vector and sparse search results using RRF"""
        # Reciprocal Rank Fusion (RRF)
        k = 60  # RRF parameter
        combined_scores = {}

        # Add vector scores
        for rank, result in enumerate(vector_results):
            chunk_id = result.chunk_id
            rrf_score = vector_weight / (k + rank + 1)
            combined_scores[chunk_id] = combined_scores.get(chunk_id, 0) + rrf_score

        # Add sparse scores
        for rank, result in enumerate(sparse_results):
            chunk_id = result.chunk_id
            rrf_score = sparse_weight / (k + rank + 1)
            combined_scores[chunk_id] = combined_scores.get(chunk_id, 0) + rrf_score

        # Create combined results
        chunk_map = {}
        for result in vector_results + sparse_results:
            chunk_map[result.chunk_id] = result

        combined_results = []
        for chunk_id, score in sorted(combined_scores.items(), key=lambda x: x[1], reverse=True):
            if chunk_id in chunk_map:
                result = chunk_map[chunk_id]
                result.score = score  # Update with combined score
                combined_results.append(result)

        return combined_results

    def _estimate_index_size(self) -> float:
        """Estimate index size in MB"""
        if not self.chunks_metadata:
            return 0.0

        # Rough estimation: embeddings + metadata
        embedding_dim = self.embedding_generator.get_embedding_dimension()
        vector_size_mb = (
            len(self.chunks_metadata) * embedding_dim * 4 / (1024 * 1024)
        )  # 4 bytes per float

        # Add metadata overhead
        metadata_size_mb = len(self.chunks_metadata) * 0.001  # ~1KB per chunk metadata

        return vector_size_mb + metadata_size_mb

    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Simple vector search"""
        if self.config.enable_hybrid_search and self.bm25_scorer:
            return await self.hybrid_search(query, limit)
        query_embedding = await self.embedding_generator.generate_embeddings([query])
        return await self.vector_store.search(query_embedding[0], limit=limit)

    def get_index_stats(self) -> dict[str, Any]:
        """Get indexing statistics"""
        vector_stats = self.vector_store.get_stats() if self.vector_store else {}

        return {
            "total_indexed": self.total_indexed,
            "last_index_time": self.last_index_time.isoformat() if self.last_index_time else None,
            "vector_store_stats": vector_stats,
            "embedding_model": self.config.embedding_config.model_name,
            "hybrid_search_enabled": self.config.enable_hybrid_search,
            "estimated_index_size_mb": self._estimate_index_size(),
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
        }


# Utility functions
async def quick_index_documents(
    documents: list[str], output_path: Path, chunk_size: int = 512
) -> IndexResult:
    """Quick document indexing for testing"""
    optimizer = MemoryOptimizer(max_memory_gb=4.0)
    config = RAGConfig(chunk_size=chunk_size, vector_store_path=output_path, batch_size=50)

    indexer = RAGIndexer(config, optimizer)
    await indexer.initialize()

    # Convert documents to DataFrame
    df = pl.DataFrame({"content": documents})
    parquet_path = output_path / "temp_docs.parquet"
    df.write_parquet(parquet_path)

    try:
        result = await indexer.index_parquet_dataset(parquet_path)
        return result
    finally:
        if parquet_path.exists():
            parquet_path.unlink()


__all__ = [
    "BM25Scorer",
    "EmbeddingConfig",
    "EmbeddingGenerator",
    "IndexResult",
    "RAGConfig",
    "RAGIndexer",
    "quick_index_documents",
]
