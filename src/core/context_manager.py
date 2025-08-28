#!/usr/bin/env python3
"""
Context Manager for Deep Research Tool
Handles intelligent chunking, prioritization and memory management

Author: Advanced IT Specialist
"""

import re
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from .priority_scorer import InformationPriorityScorer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata"""
    content: str
    start_position: int
    end_position: int
    source_document: str
    priority_score: float
    metadata: Dict[str, Any]
    coherence_score: float = 0.0

class ContextManager:
    """Advanced context management system with intelligent chunking and prioritization"""

    def __init__(self, max_context_size: int = 4096, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the context manager

        Args:
            max_context_size: Maximum context size in tokens
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.max_context_size = max_context_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.priority_scorer = InformationPriorityScorer()
        self.conversation_memory = deque(maxlen=10)
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

    def chunk_document(self, text: str, source_document: str,
                      preserve_coherence: bool = True) -> List[DocumentChunk]:
        """
        Intelligent document chunking with coherence preservation

        Args:
            text: Text to chunk
            source_document: Source document identifier
            preserve_coherence: Whether to preserve sentence coherence

        Returns:
            List of DocumentChunk objects
        """
        if preserve_coherence:
            return self._chunk_with_coherence(text, source_document)
        else:
            return self._chunk_simple(text, source_document)

    def _chunk_with_coherence(self, text: str, source_document: str) -> List[DocumentChunk]:
        """Chunk text while preserving sentence coherence"""
        # Split into sentences
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = ""
        start_pos = 0

        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Create chunk
                chunk = DocumentChunk(
                    content=current_chunk.strip(),
                    start_position=start_pos,
                    end_position=start_pos + len(current_chunk),
                    source_document=source_document,
                    priority_score=0.0,  # Will be calculated later
                    metadata={"type": "coherent_chunk"},
                    coherence_score=self._calculate_coherence_score(current_chunk)
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                start_pos = start_pos + len(current_chunk) - len(overlap_text)
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        # Add final chunk
        if current_chunk:
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                start_position=start_pos,
                end_position=start_pos + len(current_chunk),
                source_document=source_document,
                priority_score=0.0,
                metadata={"type": "coherent_chunk"},
                coherence_score=self._calculate_coherence_score(current_chunk)
            )
            chunks.append(chunk)

        return chunks

    def _chunk_simple(self, text: str, source_document: str) -> List[DocumentChunk]:
        """Simple character-based chunking"""
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            chunk = DocumentChunk(
                content=chunk_text,
                start_position=start,
                end_position=end,
                source_document=source_document,
                priority_score=0.0,
                metadata={"type": "simple_chunk"}
            )
            chunks.append(chunk)

            start = end - self.chunk_overlap

        return chunks

    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from the end of a chunk"""
        if len(text) <= overlap_size:
            return text
        return text[-overlap_size:]

    def _calculate_coherence_score(self, text: str) -> float:
        """Calculate coherence score for a text chunk"""
        sentences = nltk.sent_tokenize(text)
        if len(sentences) < 2:
            return 1.0

        # Simple coherence based on sentence length variation
        lengths = [len(s.split()) for s in sentences]
        if not lengths:
            return 0.0

        mean_length = np.mean(lengths)
        std_length = np.std(lengths)

        # Lower variation = higher coherence
        coherence = 1.0 / (1.0 + std_length / (mean_length + 1))
        return min(1.0, max(0.0, coherence))

    def prioritize_chunks(self, chunks: List[DocumentChunk],
                         query_context: str) -> List[DocumentChunk]:
        """
        Prioritize chunks based on relevance and importance

        Args:
            chunks: List of document chunks
            query_context: Current query context

        Returns:
            Sorted list of chunks by priority
        """
        # Calculate priority scores
        for chunk in chunks:
            chunk.priority_score = self.priority_scorer.score_information(
                chunk.content, chunk.metadata
            )

        # Calculate relevance to query
        if query_context and chunks:
            chunk_texts = [chunk.content for chunk in chunks]
            all_texts = chunk_texts + [query_context]

            try:
                tfidf_matrix = self.vectorizer.fit_transform(all_texts)
                query_vector = tfidf_matrix[-1]
                chunk_vectors = tfidf_matrix[:-1]

                # Calculate cosine similarity
                similarities = cosine_similarity(chunk_vectors, query_vector).flatten()

                # Combine priority score with relevance
                for i, chunk in enumerate(chunks):
                    relevance_score = similarities[i] if i < len(similarities) else 0.0
                    chunk.priority_score = (chunk.priority_score * 0.6 + relevance_score * 0.4)

            except ValueError:
                # Handle case where vectorization fails
                pass

        # Sort by priority score (descending)
        return sorted(chunks, key=lambda x: x.priority_score, reverse=True)

    def prepare_context(self, documents: List[Dict[str, Any]],
                       query: str, max_tokens: int = None) -> str:
        """
        Prepare optimized context for AI model

        Args:
            documents: List of document dictionaries
            query: Current query
            max_tokens: Maximum tokens to use (defaults to self.max_context_size)

        Returns:
            Optimized context string
        """
        if max_tokens is None:
            max_tokens = self.max_context_size

        all_chunks = []

        # Chunk all documents
        for doc in documents:
            content = doc.get('content', '')
            source = doc.get('source', 'unknown')
            metadata = doc.get('metadata', {})

            chunks = self.chunk_document(content, source)
            # Add document metadata to chunks
            for chunk in chunks:
                chunk.metadata.update(metadata)
            all_chunks.extend(chunks)

        # Prioritize chunks
        prioritized_chunks = self.prioritize_chunks(all_chunks, query)

        # Build context within token limit
        context_parts = []
        current_length = 0

        for chunk in prioritized_chunks:
            chunk_length = len(chunk.content.split())  # Rough token estimation

            if current_length + chunk_length > max_tokens:
                break

            context_parts.append(f"[Source: {chunk.source_document}]\n{chunk.content}\n")
            current_length += chunk_length

        return "\n---\n".join(context_parts)

    def add_to_memory(self, query: str, response: str, metadata: Dict[str, Any] = None):
        """Add interaction to conversation memory"""
        memory_item = {
            'query': query,
            'response': response,
            'timestamp': metadata.get('timestamp') if metadata else None,
            'metadata': metadata or {}
        }
        self.conversation_memory.append(memory_item)

    def get_memory_context(self, max_items: int = 5) -> str:
        """Get recent conversation context"""
        recent_memory = list(self.conversation_memory)[-max_items:]
        context_parts = []

        for item in recent_memory:
            context_parts.append(f"Q: {item['query']}\nA: {item['response']}")

        return "\n\n".join(context_parts)

    def summarize_old_context(self, text: str, target_length: int = 200) -> str:
        """
        Summarize old context to preserve key information

        Args:
            text: Text to summarize
            target_length: Target length in words

        Returns:
            Summarized text
        """
        sentences = nltk.sent_tokenize(text)
        if len(sentences) <= 3:
            return text

        # Simple extractive summarization
        # Calculate sentence scores based on word frequency
        words = nltk.word_tokenize(text.lower())
        word_freq = {}
        for word in words:
            if word.isalnum():
                word_freq[word] = word_freq.get(word, 0) + 1

        sentence_scores = {}
        for sentence in sentences:
            sentence_words = nltk.word_tokenize(sentence.lower())
            score = 0
            word_count = 0
            for word in sentence_words:
                if word in word_freq:
                    score += word_freq[word]
                    word_count += 1
            if word_count > 0:
                sentence_scores[sentence] = score / word_count

        # Select top sentences
        sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        summary_sentences = []
        current_length = 0

        for sentence, score in sorted_sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length <= target_length:
                summary_sentences.append(sentence)
                current_length += sentence_length
            else:
                break

        return " ".join(summary_sentences)

    def extract_key_entities(self, text: str) -> List[str]:
        """Extract key entities from text"""
        # Simple entity extraction based on capitalized words
        # In production, you might want to use spaCy or similar
        words = nltk.word_tokenize(text)
        entities = []

        for word in words:
            if (word[0].isupper() and len(word) > 2 and
                not word.isupper() and word.isalpha()):
                entities.append(word)

        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)

        return unique_entities[:20]  # Limit to top 20 entities
