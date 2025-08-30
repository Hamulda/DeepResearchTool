"""Core components for DeepResearchTool v3.0
Enhanced with Memory-Efficient Data Core (Phase 1)
"""

from .async_crawler import (
    AsyncCrawler,
    CrawlConfig,
    CrawlResult,
    PolitenessPolicy,
    crawl_urls_to_parquet,
)
from .config import load_config, get_settings
from .context_manager import (
    ChunkingConfig,
    ContextManager,
    DocumentChunk,
    HierarchicalChunker,
    SemanticCompressor,
)

# Phase 1: Memory-Efficient Data Core
from .memory_optimizer import (
    DuckDBQueryEngine,
    LazyDataPipeline,
    MemoryOptimizer,
    ParquetDatasetManager,
)
from .pipeline import PipelineResult, ResearchPipeline
from .vector_store import BaseVectorStore, SearchResult, VectorConfig, VectorStoreFactory

__all__ = [
    # Existing components
    "load_config",
    "ResearchPipeline",
    "PipelineResult",
    # Phase 1: Memory-Efficient Data Core
    "MemoryOptimizer",
    "LazyDataPipeline",
    "ParquetDatasetManager",
    "DuckDBQueryEngine",
    "AsyncCrawler",
    "CrawlConfig",
    "CrawlResult",
    "PolitenessPolicy",
    "crawl_urls_to_parquet",
    "ContextManager",
    "DocumentChunk",
    "ChunkingConfig",
    "HierarchicalChunker",
    "SemanticCompressor",
    "VectorStoreFactory",
    "VectorConfig",
    "SearchResult",
    "BaseVectorStore",
]
