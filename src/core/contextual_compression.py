#!/usr/bin/env python3
"""
Contextual Compression Engine
Selective passage filtering with salience, novelty, and deduplication before LLM processing

Author: Senior IT Specialist
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
import json
import re
from datetime import datetime
import numpy as np
from collections import defaultdict
import hashlib

import structlog
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

logger = structlog.get_logger(__name__)

@dataclass
class PassageScore:
    """Scoring components for passage filtering"""
    salience: float
    novelty: float
    redundancy: float
    final_score: float
    reason: str

@dataclass
class CompressedContext:
    """Result of contextual compression"""
    passages: List[Dict[str, Any]]
    original_token_count: int
    compressed_token_count: int
    compression_ratio: float
    filtering_stats: Dict[str, int]
    budget_used: float

class SalienceCalculator:
    """Calculate passage relevance/salience to query"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_model = None
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )

    async def initialize(self):
        """Initialize salience calculator"""
        model_name = self.config.get("embedding_model", "all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer(model_name)
        logger.info("Salience calculator initialized")

    def calculate_salience_scores(self, query: str, passages: List[str]) -> List[float]:
        """Calculate salience scores using multiple methods"""

        # Method 1: Semantic similarity via embeddings
        semantic_scores = self._calculate_semantic_salience(query, passages)

        # Method 2: TF-IDF based relevance
        tfidf_scores = self._calculate_tfidf_salience(query, passages)

        # Method 3: Keyword overlap
        keyword_scores = self._calculate_keyword_salience(query, passages)

        # Combine scores with weights
        weights = self.config.get("salience_weights", {
            "semantic": 0.5,
            "tfidf": 0.3,
            "keyword": 0.2
        })

        final_scores = []
        for i in range(len(passages)):
            combined_score = (
                weights["semantic"] * semantic_scores[i] +
                weights["tfidf"] * tfidf_scores[i] +
                weights["keyword"] * keyword_scores[i]
            )
            final_scores.append(combined_score)

        return final_scores

    def _calculate_semantic_salience(self, query: str, passages: List[str]) -> List[float]:
        """Calculate semantic similarity scores"""

        # Generate embeddings
        query_embedding = self.embedding_model.encode([query])
        passage_embeddings = self.embedding_model.encode(passages)

        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, passage_embeddings)[0]

        return similarities.tolist()

    def _calculate_tfidf_salience(self, query: str, passages: List[str]) -> List[float]:
        """Calculate TF-IDF based relevance scores"""

        # Combine query and passages for TF-IDF fitting
        all_texts = [query] + passages

        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)

            # Calculate similarities
            query_vector = tfidf_matrix[0]
            passage_vectors = tfidf_matrix[1:]

            similarities = cosine_similarity(query_vector, passage_vectors)[0]
            return similarities.tolist()

        except Exception as e:
            logger.warning(f"TF-IDF calculation failed: {e}")
            return [0.5] * len(passages)  # Fallback to neutral scores

    def _calculate_keyword_salience(self, query: str, passages: List[str]) -> List[float]:
        """Calculate keyword overlap scores"""

        # Extract keywords from query
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))

        scores = []
        for passage in passages:
            passage_words = set(re.findall(r'\b\w{3,}\b', passage.lower()))

            # Calculate Jaccard similarity
            intersection = len(query_words & passage_words)
            union = len(query_words | passage_words)

            jaccard_score = intersection / union if union > 0 else 0
            scores.append(jaccard_score)

        return scores

class NoveltyDetector:
    """Detect novel information in passages"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.novelty_threshold = config.get("novelty_threshold", 0.7)
        self.embedding_model = None

    async def initialize(self):
        """Initialize novelty detector"""
        model_name = self.config.get("embedding_model", "all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer(model_name)
        logger.info("Novelty detector initialized")

    def calculate_novelty_scores(self, passages: List[str]) -> List[float]:
        """Calculate novelty scores for each passage"""

        if len(passages) <= 1:
            return [1.0] * len(passages)

        # Generate embeddings
        embeddings = self.embedding_model.encode(passages)

        novelty_scores = []

        for i, current_embedding in enumerate(embeddings):
            # Calculate similarity to all previous passages
            if i == 0:
                novelty_scores.append(1.0)  # First passage is always novel
                continue

            previous_embeddings = embeddings[:i]
            similarities = cosine_similarity([current_embedding], previous_embeddings)[0]

            # Novelty is inverse of maximum similarity to previous passages
            max_similarity = np.max(similarities)
            novelty_score = 1.0 - max_similarity

            novelty_scores.append(novelty_score)

        return novelty_scores

class RedundancyFilter:
    """Remove redundant passages using content hashing and similarity"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.similarity_threshold = config.get("redundancy_threshold", 0.85)
        self.hash_threshold = config.get("hash_threshold", 0.9)

    def detect_redundancy(self, passages: List[str]) -> List[bool]:
        """Detect redundant passages (True = redundant, should be filtered)"""

        redundant = [False] * len(passages)

        # Method 1: Exact and near-exact duplicates via hashing
        content_hashes = []
        for passage in passages:
            # Create normalized hash
            normalized = re.sub(r'\s+', ' ', passage.lower().strip())
            content_hash = hashlib.md5(normalized.encode()).hexdigest()
            content_hashes.append(content_hash)

        # Mark exact duplicates
        seen_hashes = set()
        for i, content_hash in enumerate(content_hashes):
            if content_hash in seen_hashes:
                redundant[i] = True
            else:
                seen_hashes.add(content_hash)

        # Method 2: High similarity detection
        # Calculate pairwise similarities for non-redundant passages
        non_redundant_indices = [i for i, is_redundant in enumerate(redundant) if not is_redundant]

        if len(non_redundant_indices) > 1:
            # Use simple text similarity for efficiency
            for i in range(len(non_redundant_indices)):
                if redundant[non_redundant_indices[i]]:
                    continue

                for j in range(i + 1, len(non_redundant_indices)):
                    idx1, idx2 = non_redundant_indices[i], non_redundant_indices[j]

                    if redundant[idx2]:
                        continue

                    similarity = self._calculate_text_similarity(passages[idx1], passages[idx2])

                    if similarity > self.similarity_threshold:
                        # Keep the longer passage (more information)
                        if len(passages[idx1]) >= len(passages[idx2]):
                            redundant[idx2] = True
                        else:
                            redundant[idx1] = True
                            break

        return redundant

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity using character n-grams"""

        def get_ngrams(text: str, n: int = 3) -> Set[str]:
            text = re.sub(r'\s+', '', text.lower())
            return set(text[i:i+n] for i in range(len(text) - n + 1))

        ngrams1 = get_ngrams(text1)
        ngrams2 = get_ngrams(text2)

        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)

        return intersection / union if union > 0 else 0.0

class ContextualCompressionEngine:
    """Main compression engine coordinating all filtering strategies"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.compression_config = config.get("compression", {})

        # Compression settings
        self.enabled = self.compression_config.get("enabled", False)
        self.budget_tokens = self.compression_config.get("budget_tokens", 2000)
        self.strategy = self.compression_config.get("strategy", "salience")

        # Strategy weights
        self.strategy_weights = {
            "salience": 0.6,
            "novelty": 0.3,
            "redundancy": 0.1
        }

        # Parse strategy string
        if "+" in self.strategy:
            strategies = self.strategy.split("+")
            if "salience" in strategies and "novelty" in strategies:
                self.strategy_weights = {"salience": 0.5, "novelty": 0.4, "redundancy": 0.1}

        # Components
        self.salience_calculator = SalienceCalculator(config)
        self.novelty_detector = NoveltyDetector(config)
        self.redundancy_filter = RedundancyFilter(config)

        # Token counting (simple approximation)
        self.avg_tokens_per_char = 0.25  # Rough approximation

        self.logger = structlog.get_logger(__name__)

    async def initialize(self):
        """Initialize compression engine"""
        if not self.enabled:
            self.logger.info("Contextual compression disabled")
            return

        await self.salience_calculator.initialize()
        await self.novelty_detector.initialize()

        self.logger.info(f"Contextual compression initialized: {self.strategy} strategy, {self.budget_tokens} token budget")

    async def compress_context(self, query: str, passages: List[Dict[str, Any]]) -> CompressedContext:
        """Main compression method"""

        if not self.enabled or not passages:
            return self._create_uncompressed_result(passages)

        self.logger.info(f"Compressing {len(passages)} passages with {self.strategy} strategy")

        # Extract passage texts
        passage_texts = [p.get("content", "") for p in passages]

        # Calculate original token count
        original_tokens = self._estimate_token_count(passage_texts)

        # Phase 1: Remove redundant passages
        redundant_flags = self.redundancy_filter.detect_redundancy(passage_texts)
        filtered_passages = []
        filtered_texts = []

        for i, (passage, is_redundant) in enumerate(zip(passages, redundant_flags)):
            if not is_redundant:
                filtered_passages.append(passage)
                filtered_texts.append(passage_texts[i])

        redundancy_removed = len(passages) - len(filtered_passages)

        # Phase 2: Calculate scores for remaining passages
        passage_scores = []

        if filtered_texts:
            # Calculate component scores
            salience_scores = self.salience_calculator.calculate_salience_scores(query, filtered_texts)
            novelty_scores = self.novelty_detector.calculate_novelty_scores(filtered_texts)

            # Combine scores
            for i, (salience, novelty) in enumerate(zip(salience_scores, novelty_scores)):
                final_score = (
                    self.strategy_weights["salience"] * salience +
                    self.strategy_weights["novelty"] * novelty
                )

                score = PassageScore(
                    salience=salience,
                    novelty=novelty,
                    redundancy=1.0,  # Already filtered
                    final_score=final_score,
                    reason=f"salience:{salience:.2f} novelty:{novelty:.2f}"
                )
                passage_scores.append(score)

        # Phase 3: Select passages within token budget
        selected_passages = self._select_passages_by_budget(
            filtered_passages, filtered_texts, passage_scores
        )

        # Calculate final metrics
        compressed_tokens = self._estimate_token_count([p.get("content", "") for p in selected_passages])
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        budget_used = compressed_tokens / self.budget_tokens if self.budget_tokens > 0 else 1.0

        filtering_stats = {
            "original_count": len(passages),
            "redundancy_removed": redundancy_removed,
            "salience_filtered": len(filtered_passages) - len(selected_passages),
            "final_count": len(selected_passages)
        }

        self.logger.info(f"Compression completed: {len(passages)} → {len(selected_passages)} passages, "
                        f"ratio: {compression_ratio:.2f}, budget used: {budget_used:.2f}")

        return CompressedContext(
            passages=selected_passages,
            original_token_count=original_tokens,
            compressed_token_count=compressed_tokens,
            compression_ratio=compression_ratio,
            filtering_stats=filtering_stats,
            budget_used=budget_used
        )

    def _select_passages_by_budget(self, passages: List[Dict[str, Any]],
                                 texts: List[str], scores: List[PassageScore]) -> List[Dict[str, Any]]:
        """Select passages within token budget, prioritizing by score"""

        # Create passage-score pairs and sort by score
        passage_score_pairs = list(zip(passages, texts, scores))
        passage_score_pairs.sort(key=lambda x: x[2].final_score, reverse=True)

        selected = []
        current_tokens = 0

        for passage, text, score in passage_score_pairs:
            passage_tokens = self._estimate_token_count([text])

            if current_tokens + passage_tokens <= self.budget_tokens:
                selected.append(passage)
                current_tokens += passage_tokens
            else:
                # Check if we can fit a truncated version
                remaining_budget = self.budget_tokens - current_tokens
                if remaining_budget > 100:  # Minimum meaningful size
                    # Truncate passage to fit budget
                    chars_budget = int(remaining_budget / self.avg_tokens_per_char)
                    truncated_text = text[:chars_budget] + "..."

                    # Create truncated passage
                    truncated_passage = passage.copy()
                    truncated_passage["content"] = truncated_text
                    truncated_passage["truncated"] = True

                    selected.append(truncated_passage)
                    break

        return selected

    def _estimate_token_count(self, texts: List[str]) -> int:
        """Estimate token count for list of texts"""
        total_chars = sum(len(text) for text in texts)
        return int(total_chars * self.avg_tokens_per_char)

    def _create_uncompressed_result(self, passages: List[Dict[str, Any]]) -> CompressedContext:
        """Create result when compression is disabled"""
        passage_texts = [p.get("content", "") for p in passages]
        token_count = self._estimate_token_count(passage_texts)

        return CompressedContext(
            passages=passages,
            original_token_count=token_count,
            compressed_token_count=token_count,
            compression_ratio=1.0,
            filtering_stats={"original_count": len(passages), "final_count": len(passages)},
            budget_used=token_count / self.budget_tokens if self.budget_tokens > 0 else 1.0
        )

def create_contextual_compression_engine(config: Dict[str, Any]) -> ContextualCompressionEngine:
    """Factory function for contextual compression engine"""
    return ContextualCompressionEngine(config)
