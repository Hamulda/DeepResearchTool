"""Context Manager for Large Document Processing
Hierarchical chunking and semantic compression for limited context windows
Optimized for MacBook Air M1 8GB RAM constraints
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator, AsyncIterator
from dataclasses import dataclass
import logging
import re
from typing import Any, Callable, Optional

import numpy as np
import polars as pl

from .memory_optimizer import MemoryOptimizer

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Single document chunk with metadata"""

    text: str
    chunk_id: str
    document_id: str
    chunk_index: int
    start_char: int
    end_char: int
    token_count: int
    embedding: Optional[np.ndarray] = None
    importance_score: Optional[float] = None
    section_type: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategy"""

    chunk_size: int = 512
    overlap_size: int = 50
    min_chunk_size: int = 100
    max_chunk_size: int = 1024
    respect_sentence_boundaries: bool = True
    respect_paragraph_boundaries: bool = True
    section_aware: bool = True
    token_counter: str = "simple"  # simple, tiktoken, huggingface


class BaseChunker(ABC):
    """Abstract base class for document chunkers"""

    def __init__(self, config: ChunkingConfig, optimizer: MemoryOptimizer):
        self.config = config
        self.optimizer = optimizer

    @abstractmethod
    def chunk_document(self, text: str, document_id: str) -> list[DocumentChunk]:
        """Chunk a document into smaller pieces"""

    def count_tokens(self, text: str) -> int:
        """Count tokens in text (simple approximation)"""
        if self.config.token_counter == "simple":
            # Rough approximation: 1 token ≈ 4 characters
            return len(text) // 4
        # Could integrate tiktoken or transformers tokenizer here
        return len(text.split())


class HierarchicalChunker(BaseChunker):
    """Hierarchical document chunker with section awareness"""

    def __init__(self, config: ChunkingConfig, optimizer: MemoryOptimizer):
        super().__init__(config, optimizer)

        # Section patterns (markdown-like)
        self.section_patterns = [
            (r"^#{1}\s+(.+)$", "h1"),
            (r"^#{2}\s+(.+)$", "h2"),
            (r"^#{3}\s+(.+)$", "h3"),
            (r"^#{4,6}\s+(.+)$", "h4+"),
            (r"^\s*\n\s*\n", "paragraph_break"),
        ]

    def chunk_document(self, text: str, document_id: str) -> list[DocumentChunk]:
        """Chunk document using hierarchical strategy"""
        # First, identify document structure
        sections = self._identify_sections(text) if self.config.section_aware else []

        if sections:
            return self._chunk_by_sections(text, document_id, sections)
        return self._chunk_by_sliding_window(text, document_id)

    def _identify_sections(self, text: str) -> list[tuple[int, int, str]]:
        """Identify document sections by headers and structure"""
        sections = []
        lines = text.split("\n")
        current_start = 0
        current_type = "content"

        for i, line in enumerate(lines):
            for pattern, section_type in self.section_patterns:
                if re.match(pattern, line, re.MULTILINE):
                    # End previous section
                    if current_start < i:
                        char_start = sum(len(l) + 1 for l in lines[:current_start])
                        char_end = sum(len(l) + 1 for l in lines[:i])
                        sections.append((char_start, char_end, current_type))

                    # Start new section
                    current_start = i
                    current_type = section_type
                    break

        # Add final section
        if current_start < len(lines):
            char_start = sum(len(l) + 1 for l in lines[:current_start])
            sections.append((char_start, len(text), current_type))

        return sections

    def _chunk_by_sections(
        self, text: str, document_id: str, sections: list[tuple[int, int, str]]
    ) -> list[DocumentChunk]:
        """Chunk document respecting section boundaries"""
        chunks = []
        chunk_index = 0

        for start_char, end_char, section_type in sections:
            section_text = text[start_char:end_char].strip()

            if not section_text:
                continue

            # Check if section fits in one chunk
            section_token_count = self.count_tokens(section_text)

            if section_token_count <= self.config.chunk_size:
                # Single chunk for this section
                chunk = DocumentChunk(
                    text=section_text,
                    chunk_id=f"{document_id}_chunk_{chunk_index:04d}",
                    document_id=document_id,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    token_count=section_token_count,
                    section_type=section_type,
                )
                chunks.append(chunk)
                chunk_index += 1
            else:
                # Split large section with sliding window
                section_chunks = self._chunk_by_sliding_window(
                    section_text,
                    document_id,
                    start_char=start_char,
                    chunk_index_offset=chunk_index,
                    section_type=section_type,
                )
                chunks.extend(section_chunks)
                chunk_index += len(section_chunks)

        return chunks

    def _chunk_by_sliding_window(
        self,
        text: str,
        document_id: str,
        start_char: int = 0,
        chunk_index_offset: int = 0,
        section_type: Optional[str] = None,
    ) -> list[DocumentChunk]:
        """Chunk text using sliding window approach"""
        chunks = []

        # Split into sentences if configured
        if self.config.respect_sentence_boundaries:
            sentences = self._split_sentences(text)
        else:
            # Simple word-based splitting
            words = text.split()
            sentences = [" ".join(words[i : i + 50]) for i in range(0, len(words), 50)]

        current_chunk = ""
        current_start = start_char
        chunk_index = chunk_index_offset

        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            test_token_count = self.count_tokens(test_chunk)

            if test_token_count > self.config.chunk_size and current_chunk:
                # Finalize current chunk
                chunk_text = current_chunk.strip()
                if len(chunk_text) >= self.config.min_chunk_size:
                    chunk = DocumentChunk(
                        text=chunk_text,
                        chunk_id=f"{document_id}_chunk_{chunk_index:04d}",
                        document_id=document_id,
                        chunk_index=chunk_index,
                        start_char=current_start,
                        end_char=current_start + len(chunk_text),
                        token_count=self.count_tokens(chunk_text),
                        section_type=section_type,
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                # Start new chunk with overlap
                if self.config.overlap_size > 0:
                    overlap_text = self._get_overlap_text(current_chunk, self.config.overlap_size)
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence

                current_start = (
                    current_start + len(chunk_text) - len(overlap_text)
                    if self.config.overlap_size > 0
                    else current_start + len(chunk_text)
                )
            else:
                current_chunk = test_chunk

        # Add final chunk
        if current_chunk.strip() and len(current_chunk.strip()) >= self.config.min_chunk_size:
            chunk = DocumentChunk(
                text=current_chunk.strip(),
                chunk_id=f"{document_id}_chunk_{chunk_index:04d}",
                document_id=document_id,
                chunk_index=chunk_index,
                start_char=current_start,
                end_char=start_char + len(text),
                token_count=self.count_tokens(current_chunk),
                section_type=section_type,
            )
            chunks.append(chunk)

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences"""
        # Simple sentence splitting (could be improved with spaCy/NLTK)
        sentence_endings = r"[.!?]+\s+"
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Get overlapping text from end of previous chunk"""
        words = text.split()
        if len(words) <= overlap_tokens:
            return text
        return " ".join(words[-overlap_tokens:])


class SemanticCompressor:
    """Semantic compression for large context windows"""

    def __init__(self, optimizer: MemoryOptimizer, target_ratio: float = 0.5):
        self.optimizer = optimizer
        self.target_ratio = target_ratio

    def compress_chunks(
        self,
        chunks: list[DocumentChunk],
        target_token_count: int,
        importance_scorer: Optional[Callable[[DocumentChunk], float]] = None,
    ) -> list[DocumentChunk]:
        """Compress chunks to fit target token count"""
        total_tokens = sum(chunk.token_count for chunk in chunks)

        if total_tokens <= target_token_count:
            return chunks

        # Score chunk importance
        scored_chunks = self._score_chunk_importance(chunks, importance_scorer)

        # Sort by importance and select top chunks
        scored_chunks.sort(key=lambda x: x.importance_score or 0, reverse=True)

        selected_chunks = []
        current_tokens = 0

        for chunk in scored_chunks:
            if current_tokens + chunk.token_count <= target_token_count:
                selected_chunks.append(chunk)
                current_tokens += chunk.token_count
            else:
                # Try to fit partial chunk
                remaining_tokens = target_token_count - current_tokens
                if remaining_tokens > 50:  # Minimum viable chunk
                    truncated_chunk = self._truncate_chunk(chunk, remaining_tokens)
                    selected_chunks.append(truncated_chunk)
                break

        # Re-sort by original order
        selected_chunks.sort(key=lambda x: x.chunk_index)

        logger.info(
            f"Compressed {len(chunks)} chunks ({total_tokens} tokens) to {len(selected_chunks)} chunks ({current_tokens} tokens)"
        )

        return selected_chunks

    def _score_chunk_importance(
        self, chunks: list[DocumentChunk], scorer: Optional[Callable[[DocumentChunk], float]] = None
    ) -> list[DocumentChunk]:
        """Score chunks by importance"""
        if scorer:
            # Use custom scorer
            for chunk in chunks:
                chunk.importance_score = scorer(chunk)
        else:
            # Default importance scoring
            for chunk in chunks:
                score = 0.0

                # Section type importance
                section_weights = {
                    "h1": 1.0,
                    "h2": 0.8,
                    "h3": 0.6,
                    "h4+": 0.4,
                    "content": 0.5,
                    "paragraph_break": 0.2,
                }
                score += section_weights.get(chunk.section_type or "content", 0.5)

                # Length importance (sweet spot around 200-400 tokens)
                length_score = 1.0 - abs(chunk.token_count - 300) / 300
                score += max(0, length_score) * 0.3

                # Position importance (beginning is more important)
                position_score = 1.0 - (chunk.chunk_index / len(chunks))
                score += position_score * 0.2

                chunk.importance_score = score

        return chunks

    def _truncate_chunk(self, chunk: DocumentChunk, target_tokens: int) -> DocumentChunk:
        """Truncate chunk to target token count"""
        words = chunk.text.split()
        target_words = min(len(words), target_tokens * 4)  # Rough token-to-word ratio

        truncated_text = " ".join(words[:target_words])

        return DocumentChunk(
            text=truncated_text,
            chunk_id=chunk.chunk_id + "_truncated",
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            start_char=chunk.start_char,
            end_char=chunk.start_char + len(truncated_text),
            token_count=target_tokens,
            importance_score=chunk.importance_score,
            section_type=chunk.section_type,
            metadata={**(chunk.metadata or {}), "truncated": True},
        )


class ContextManager:
    """High-level context management for document processing"""

    def __init__(
        self, optimizer: MemoryOptimizer, chunking_config: Optional[ChunkingConfig] = None
    ):
        self.optimizer = optimizer
        self.chunking_config = chunking_config or ChunkingConfig()
        self.chunker = HierarchicalChunker(self.chunking_config, optimizer)
        self.compressor = SemanticCompressor(optimizer)

    async def process_documents_streaming(
        self,
        documents: Iterator[tuple[str, str]],  # (document_id, text)
        target_context_tokens: int = 4000,
        batch_size: Optional[int] = None,
    ) -> AsyncIterator[list[DocumentChunk]]:
        """Process documents in streaming fashion"""
        if batch_size is None:
            batch_size = self.optimizer.get_optimal_batch_size(record_size_bytes=1024)

        document_batch = []

        async for doc_id, text in documents:
            document_batch.append((doc_id, text))

            if len(document_batch) >= batch_size:
                processed_chunks = await self._process_document_batch(
                    document_batch, target_context_tokens
                )
                yield processed_chunks

                document_batch = []

                # Memory pressure check
                if self.optimizer.check_memory_pressure()["pressure"]:
                    self.optimizer.force_gc()

        # Process remaining documents
        if document_batch:
            processed_chunks = await self._process_document_batch(
                document_batch, target_context_tokens
            )
            yield processed_chunks

    async def _process_document_batch(
        self, documents: list[tuple[str, str]], target_context_tokens: int
    ) -> list[DocumentChunk]:
        """Process batch of documents"""
        all_chunks = []

        # Chunk all documents
        for doc_id, text in documents:
            chunks = self.chunker.chunk_document(text, doc_id)
            all_chunks.extend(chunks)

        # Compress if needed
        total_tokens = sum(chunk.token_count for chunk in all_chunks)
        if total_tokens > target_context_tokens:
            all_chunks = self.compressor.compress_chunks(all_chunks, target_context_tokens)

        return all_chunks

    def chunks_to_polars(self, chunks: list[DocumentChunk]) -> pl.DataFrame:
        """Convert chunks to Polars DataFrame for analysis"""
        data = []
        for chunk in chunks:
            data.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "importance_score": chunk.importance_score,
                    "section_type": chunk.section_type,
                    "metadata": str(chunk.metadata) if chunk.metadata else None,
                }
            )

        return pl.DataFrame(data)


__all__ = [
    "ChunkingConfig",
    "ContextManager", 
    "DocumentChunk",
    "HierarchicalChunker",
    "SemanticCompressor",
]