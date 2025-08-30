#!/usr/bin/env python3
"""Advanced Re-ranking a Contextual Compression
Pairwise re-ranking + discourse-aware chunking + adaptive compression

Author: Senior Python/MLOps Agent
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
import logging
import re
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ReRankingStrategy(Enum):
    """Strategie re-rankingu"""

    CROSS_ENCODER = "cross_encoder"
    LLM_RATER = "llm_rater"
    HYBRID = "hybrid"


@dataclass
class ReRankingConfig:
    """Konfigurace re-ranking systému"""

    enabled: bool = True
    strategy: ReRankingStrategy = ReRankingStrategy.HYBRID
    top_k_rerank: int = 50  # Počet top dokumentů pro re-ranking

    # Cross-encoder parameters
    cross_encoder_model: str = "ms-marco-MiniLM-L-6-v2"
    cross_encoder_batch_size: int = 16

    # LLM rater parameters
    llm_model: str = "llama3.1:8b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 100

    # Hybrid parameters
    cross_encoder_weight: float = 0.6
    llm_weight: float = 0.4

    # Margin of victory
    margin_threshold: float = 0.1
    calibration_enabled: bool = True

    # Performance thresholds
    max_rerank_time_seconds: float = 30.0
    enable_rationale_logging: bool = True


@dataclass
class CompressionConfig:
    """Konfigurace contextual compression"""

    enabled: bool = True
    target_compression_ratio: float = 0.3  # Cílová kompresní ratio
    max_context_tokens: int = 8000

    # Source priorities
    primary_source_weight: float = 2.0  # Váha primární literatury
    aggregator_penalty: float = 0.5  # Penalizace agregátorů

    # Adaptive chunking
    adaptive_chunking: bool = True
    min_chunk_size: int = 100
    max_chunk_size: int = 500
    overlap_size: int = 50

    # Discourse awareness
    discourse_markers: bool = True
    preserve_citations: bool = True
    preserve_headings: bool = True

    # Salience detection
    salience_threshold: float = 0.7
    use_entity_density: bool = True
    use_keyword_density: bool = True


class CrossEncoderReRanker:
    """Cross-encoder based re-ranking"""

    def __init__(self, config: ReRankingConfig):
        self.config = config
        self.model = None  # Bude inicializován při prvním použití
        self.rerank_stats = {
            "queries_processed": 0,
            "avg_rerank_time": 0.0,
            "avg_score_change": 0.0,
        }

    async def rerank_documents(
        self, query: str, documents: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Re-rank dokumenty pomocí cross-encoder"""
        if not self.config.enabled or len(documents) <= 1:
            return documents, {"reranking_used": False}

        start_time = time.time()

        # Limit na top-k dokumentů
        docs_to_rerank = documents[: self.config.top_k_rerank]
        remaining_docs = documents[self.config.top_k_rerank :]

        try:
            # Initialize model if needed
            await self._ensure_model_loaded()

            # Připrav query-document páry
            pairs = []
            for doc in docs_to_rerank:
                content = doc.get("content", doc.get("text", ""))
                pairs.append((query, content))

            # Batch re-ranking
            rerank_scores = await self._compute_cross_encoder_scores(pairs)

            # Aplikuj nové skóre
            reranked_docs = []
            for i, doc in enumerate(docs_to_rerank):
                new_doc = doc.copy()
                original_score = doc.get("score", 0.0)
                rerank_score = rerank_scores[i] if i < len(rerank_scores) else 0.0

                # Kombinuj original a rerank skóre
                combined_score = 0.7 * rerank_score + 0.3 * original_score

                new_doc.update(
                    {
                        "score": combined_score,
                        "rerank_metadata": {
                            "original_score": original_score,
                            "cross_encoder_score": rerank_score,
                            "score_change": rerank_score - original_score,
                        },
                    }
                )
                reranked_docs.append(new_doc)

            # Seřaď podle nového skóre
            reranked_docs.sort(key=lambda x: x["score"], reverse=True)

            # Kombinuj s remaining docs
            final_docs = reranked_docs + remaining_docs

            elapsed_time = time.time() - start_time

            # Metadata
            rerank_metadata = {
                "reranking_used": True,
                "strategy": "cross_encoder",
                "documents_reranked": len(docs_to_rerank),
                "rerank_time_seconds": elapsed_time,
                "avg_score_change": (
                    np.mean([doc["rerank_metadata"]["score_change"] for doc in reranked_docs])
                    if reranked_docs
                    else 0.0
                ),
            }

            # Update stats
            self._update_stats(
                len(docs_to_rerank), elapsed_time, rerank_metadata["avg_score_change"]
            )

            logger.info(f"Cross-encoder reranking completed in {elapsed_time:.2f}s")

            return final_docs, rerank_metadata

        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            return documents, {"reranking_used": False, "error": str(e)}

    async def _ensure_model_loaded(self):
        """Zajistí, že model je načten"""
        if self.model is None:
            # Mock model pro testování
            logger.info(f"Loading cross-encoder model: {self.config.cross_encoder_model}")
            await asyncio.sleep(0.1)  # Simulate model loading
            self.model = "mock_cross_encoder"

    async def _compute_cross_encoder_scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Vypočítá cross-encoder skóre pro query-document páry"""
        scores = []

        # Process in batches
        batch_size = self.config.cross_encoder_batch_size
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]

            # Mock cross-encoder scoring
            batch_scores = []
            for query, doc in batch:
                # Simple mock scoring based on query-doc overlap
                query_words = set(query.lower().split())
                doc_words = set(doc.lower().split())
                overlap = len(query_words.intersection(doc_words))
                total_words = len(query_words.union(doc_words))

                if total_words > 0:
                    score = overlap / total_words
                else:
                    score = 0.0

                # Add some noise for realism
                score += np.random.normal(0, 0.1)
                score = max(0.0, min(1.0, score))
                batch_scores.append(score)

            scores.extend(batch_scores)

            # Simulate processing time
            await asyncio.sleep(0.01)

        return scores

    def _update_stats(self, docs_processed: int, elapsed_time: float, avg_score_change: float):
        """Aktualizuje statistiky re-rankingu"""
        self.rerank_stats["queries_processed"] += 1

        # Exponential moving average
        alpha = 0.1

        if self.rerank_stats["avg_rerank_time"] == 0:
            self.rerank_stats["avg_rerank_time"] = elapsed_time
        else:
            self.rerank_stats["avg_rerank_time"] = (
                alpha * elapsed_time + (1 - alpha) * self.rerank_stats["avg_rerank_time"]
            )

        if self.rerank_stats["avg_score_change"] == 0:
            self.rerank_stats["avg_score_change"] = avg_score_change
        else:
            self.rerank_stats["avg_score_change"] = (
                alpha * avg_score_change + (1 - alpha) * self.rerank_stats["avg_score_change"]
            )


class LLMRater:
    """LLM-based relevance rating"""

    def __init__(self, config: ReRankingConfig, llm_client=None):
        self.config = config
        self.llm_client = llm_client
        self.rating_stats = {
            "ratings_generated": 0,
            "avg_rating_time": 0.0,
            "calibration_history": [],
        }

    async def rate_documents(
        self, query: str, documents: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Rate dokumenty pomocí LLM"""
        if not self.config.enabled or len(documents) <= 1:
            return documents, {"llm_rating_used": False}

        start_time = time.time()

        # Limit na top-k dokumentů
        docs_to_rate = documents[: self.config.top_k_rerank]
        remaining_docs = documents[self.config.top_k_rerank :]

        try:
            # Rate documents
            ratings_with_rationale = await self._generate_ratings(query, docs_to_rate)

            # Aplikuj ratings
            rated_docs = []
            for i, doc in enumerate(docs_to_rate):
                new_doc = doc.copy()
                original_score = doc.get("score", 0.0)

                if i < len(ratings_with_rationale):
                    rating_info = ratings_with_rationale[i]
                    llm_score = rating_info["score"]
                    rationale = rating_info.get("rationale", "")
                else:
                    llm_score = 0.5  # Neutral rating
                    rationale = "Rating failed"

                # Kombinuj scores
                combined_score = 0.6 * llm_score + 0.4 * original_score

                new_doc.update(
                    {
                        "score": combined_score,
                        "llm_rating_metadata": {
                            "original_score": original_score,
                            "llm_score": llm_score,
                            "rationale": rationale if self.config.enable_rationale_logging else "",
                            "score_change": llm_score - original_score,
                        },
                    }
                )
                rated_docs.append(new_doc)

            # Seřaď podle combined score
            rated_docs.sort(key=lambda x: x["score"], reverse=True)

            # Kombinuj s remaining docs
            final_docs = rated_docs + remaining_docs

            elapsed_time = time.time() - start_time

            # Metadata
            rating_metadata = {
                "llm_rating_used": True,
                "strategy": "llm_rater",
                "documents_rated": len(docs_to_rate),
                "rating_time_seconds": elapsed_time,
                "avg_score_change": (
                    np.mean([doc["llm_rating_metadata"]["score_change"] for doc in rated_docs])
                    if rated_docs
                    else 0.0
                ),
            }

            # Update stats
            self._update_rating_stats(len(docs_to_rate), elapsed_time)

            logger.info(f"LLM rating completed in {elapsed_time:.2f}s")

            return final_docs, rating_metadata

        except Exception as e:
            logger.error(f"LLM rating failed: {e}")
            return documents, {"llm_rating_used": False, "error": str(e)}

    async def _generate_ratings(
        self, query: str, documents: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Generuje LLM ratings pro dokumenty"""
        ratings = []

        for doc in documents:
            content = doc.get("content", doc.get("text", ""))[:1000]  # Limit content

            # Vytvoř rating prompt
            prompt = self._create_rating_prompt(query, content)

            try:
                if self.llm_client:
                    response = await self.llm_client.generate(
                        prompt=prompt,
                        max_tokens=self.config.llm_max_tokens,
                        temperature=self.config.llm_temperature,
                        model=self.config.llm_model,
                    )
                    rating_info = self._parse_llm_response(response)
                else:
                    # Mock LLM rating
                    rating_info = self._generate_mock_rating(query, content)

                ratings.append(rating_info)

            except Exception as e:
                logger.warning(f"LLM rating failed for document: {e}")
                ratings.append({"score": 0.5, "rationale": f"Rating failed: {e}"})

        return ratings

    def _create_rating_prompt(self, query: str, content: str) -> str:
        """Vytvoří prompt pro LLM rating"""
        prompt = f"""Rate the relevance of this document excerpt to the given query on a scale of 0.0 to 1.0.

Query: "{query}"

Document excerpt:
{content}

Provide your rating and brief rationale in this format:
Rating: [0.0-1.0]
Rationale: [Brief explanation of why this document is/isn't relevant]

Focus on:
- Direct relevance to the query topic
- Quality and reliability of information
- Specificity and detail level
- Credibility indicators"""

        return prompt

    def _parse_llm_response(self, response: str) -> dict[str, Any]:
        """Parsuje LLM odpověď"""
        try:
            lines = response.strip().split("\n")
            rating = 0.5
            rationale = "No rationale provided"

            for line in lines:
                if line.startswith("Rating:"):
                    rating_str = line.replace("Rating:", "").strip()
                    try:
                        rating = float(rating_str)
                        rating = max(0.0, min(1.0, rating))  # Clamp to valid range
                    except ValueError:
                        pass
                elif line.startswith("Rationale:"):
                    rationale = line.replace("Rationale:", "").strip()

            return {"score": rating, "rationale": rationale}

        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return {"score": 0.5, "rationale": f"Parse error: {e}"}

    def _generate_mock_rating(self, query: str, content: str) -> dict[str, Any]:
        """Generuje mock rating pro testování"""
        # Simple relevance scoring based on query terms
        query_terms = query.lower().split()
        content_lower = content.lower()

        matches = sum(1 for term in query_terms if term in content_lower)
        base_score = matches / len(query_terms) if query_terms else 0.0

        # Add some variation
        score = base_score + np.random.normal(0, 0.1)
        score = max(0.0, min(1.0, score))

        rationale = f"Found {matches}/{len(query_terms)} query terms in document. "
        if score > 0.7:
            rationale += "Highly relevant content."
        elif score > 0.4:
            rationale += "Moderately relevant content."
        else:
            rationale += "Limited relevance to query."

        return {"score": score, "rationale": rationale}

    def _update_rating_stats(self, docs_rated: int, elapsed_time: float):
        """Aktualizuje rating statistiky"""
        self.rating_stats["ratings_generated"] += docs_rated

        # Exponential moving average
        alpha = 0.1

        if self.rating_stats["avg_rating_time"] == 0:
            self.rating_stats["avg_rating_time"] = elapsed_time
        else:
            self.rating_stats["avg_rating_time"] = (
                alpha * elapsed_time + (1 - alpha) * self.rating_stats["avg_rating_time"]
            )


class DiscourseAwareChunker:
    """Discourse-aware adaptive chunking"""

    def __init__(self, config: CompressionConfig):
        self.config = config
        self.chunk_stats = {"texts_chunked": 0, "total_chunks_created": 0, "avg_chunk_size": 0.0}

    def chunk_text(self, text: str, metadata: dict[str, Any] = None) -> list[dict[str, Any]]:
        """Chunková text s discourse awareness"""
        if not self.config.adaptive_chunking or len(text) < self.config.min_chunk_size:
            return [{"text": text, "chunk_index": 0, "chunk_type": "full"}]

        # Detect discourse structure
        discourse_boundaries = self._detect_discourse_boundaries(text)

        # Create chunks respecting discourse boundaries
        chunks = self._create_discourse_aware_chunks(text, discourse_boundaries)

        # Add metadata
        enriched_chunks = []
        for i, chunk in enumerate(chunks):
            enriched_chunk = {
                "text": chunk["text"],
                "chunk_index": i,
                "chunk_type": chunk["type"],
                "chunk_size": len(chunk["text"]),
                "discourse_markers": chunk.get("markers", []),
                "salience_score": self._calculate_salience(chunk["text"]),
            }

            if metadata:
                enriched_chunk["source_metadata"] = metadata

            enriched_chunks.append(enriched_chunk)

        # Update stats
        self._update_chunk_stats(len(enriched_chunks))

        return enriched_chunks

    def _detect_discourse_boundaries(self, text: str) -> list[dict[str, Any]]:
        """Detekuje discourse boundaries v textu"""
        boundaries = []

        # Heading patterns
        heading_patterns = [
            r"^\s*#{1,6}\s+(.+)$",  # Markdown headings
            r"^\s*([A-Z][^.!?]*):?\s*$",  # All caps headings
            r"^\s*\d+\.\s+(.+)$",  # Numbered sections
        ]

        lines = text.split("\n")
        for i, line in enumerate(lines):
            line_stripped = line.strip()

            # Check for headings
            for pattern in heading_patterns:
                if re.match(pattern, line_stripped, re.MULTILINE):
                    boundaries.append(
                        {
                            "position": i,
                            "type": "heading",
                            "text": line_stripped,
                            "level": self._determine_heading_level(line_stripped),
                        }
                    )
                    break

            # Check for other discourse markers
            if self._is_paragraph_break(line_stripped):
                boundaries.append({"position": i, "type": "paragraph_break", "text": line_stripped})
            elif self._contains_citation(line_stripped):
                boundaries.append({"position": i, "type": "citation", "text": line_stripped})

        return boundaries

    def _create_discourse_aware_chunks(
        self, text: str, boundaries: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Vytvoří chunks respektující discourse strukturu"""
        lines = text.split("\n")
        chunks = []
        current_chunk = []
        current_markers = []

        boundary_positions = set(b["position"] for b in boundaries)

        for i, line in enumerate(lines):
            current_chunk.append(line)

            # Check if this line has discourse markers
            line_boundaries = [b for b in boundaries if b["position"] == i]
            if line_boundaries:
                current_markers.extend(line_boundaries)

            # Decide whether to end current chunk
            should_break = False

            # Break at discourse boundaries
            if i in boundary_positions:
                boundary = next(b for b in boundaries if b["position"] == i)
                if boundary["type"] in ["heading", "paragraph_break"]:
                    should_break = True

            # Break if chunk is getting too long
            chunk_text = "\n".join(current_chunk)
            if len(chunk_text) >= self.config.max_chunk_size:
                should_break = True

            # Create chunk if breaking or at end
            if should_break or i == len(lines) - 1:
                if current_chunk:
                    chunk_text = "\n".join(current_chunk).strip()
                    if chunk_text:
                        chunk_type = self._determine_chunk_type(current_markers)
                        chunks.append(
                            {
                                "text": chunk_text,
                                "type": chunk_type,
                                "markers": current_markers.copy(),
                            }
                        )

                # Start new chunk with overlap if needed
                if should_break and i < len(lines) - 1:
                    overlap_lines = current_chunk[-self._calculate_overlap_lines() :]
                    current_chunk = overlap_lines
                else:
                    current_chunk = []
                current_markers = []

        return chunks

    def _determine_heading_level(self, text: str) -> int:
        """Určí level nadpisu"""
        if text.startswith("#"):
            return text.count("#")
        if text.isupper():
            return 1
        if re.match(r"^\d+\.", text):
            return 2
        return 3

    def _is_paragraph_break(self, line: str) -> bool:
        """Detekuje paragraph break"""
        return len(line.strip()) == 0 or line.strip() in ["---", "***", "___"]

    def _contains_citation(self, line: str) -> bool:
        """Detekuje citace"""
        citation_patterns = [
            r"\[[\d,\s-]+\]",  # [1,2,3] style citations
            r"\(\w+\s+et\s+al\.,?\s+\d{4}\)",  # (Author et al., 2023)
            r"doi:",  # DOI references
            r"http[s]?://",  # URLs
        ]

        return any(re.search(pattern, line, re.IGNORECASE) for pattern in citation_patterns)

    def _determine_chunk_type(self, markers: list[dict[str, Any]]) -> str:
        """Určí typ chunku na základě discourse markers"""
        if not markers:
            return "paragraph"

        marker_types = [m["type"] for m in markers]

        if "heading" in marker_types:
            return "section"
        if "citation" in marker_types:
            return "reference"
        return "paragraph"

    def _calculate_overlap_lines(self) -> int:
        """Vypočítá počet řádků pro overlap"""
        return max(1, self.config.overlap_size // 50)  # Assuming ~50 chars per line

    def _calculate_salience(self, text: str) -> float:
        """Vypočítá salience score pro chunk"""
        if not self.config.use_entity_density and not self.config.use_keyword_density:
            return 0.5

        score = 0.0
        components = 0

        if self.config.use_entity_density:
            entity_score = self._calculate_entity_density(text)
            score += entity_score
            components += 1

        if self.config.use_keyword_density:
            keyword_score = self._calculate_keyword_density(text)
            score += keyword_score
            components += 1

        return score / components if components > 0 else 0.5

    def _calculate_entity_density(self, text: str) -> float:
        """Vypočítá hustotu entit v textu"""
        # Simple entity detection based on capitalization patterns
        words = text.split()
        if not words:
            return 0.0

        entities = 0
        for word in words:
            # Count capitalized words (potential entities)
            if word[0].isupper() and len(word) > 2:
                entities += 1

        return min(entities / len(words), 1.0)

    def _calculate_keyword_density(self, text: str) -> float:
        """Vypočítá hustotu klíčových slov"""
        # Simple keyword density based on common academic terms
        academic_keywords = [
            "research",
            "study",
            "analysis",
            "method",
            "result",
            "conclusion",
            "finding",
            "evidence",
            "data",
            "experiment",
            "theory",
            "model",
            "significant",
            "correlation",
            "hypothesis",
            "validation",
        ]

        text_lower = text.lower()
        words = text_lower.split()
        if not words:
            return 0.0

        keyword_count = sum(1 for word in words if word in academic_keywords)
        return min(keyword_count / len(words), 1.0)

    def _update_chunk_stats(self, num_chunks: int):
        """Aktualizuje chunk statistiky"""
        self.chunk_stats["texts_chunked"] += 1
        self.chunk_stats["total_chunks_created"] += num_chunks

        # Calculate moving average of chunk count
        alpha = 0.1
        new_avg = num_chunks

        if self.chunk_stats["avg_chunk_size"] == 0:
            self.chunk_stats["avg_chunk_size"] = new_avg
        else:
            self.chunk_stats["avg_chunk_size"] = (
                alpha * new_avg + (1 - alpha) * self.chunk_stats["avg_chunk_size"]
            )


class ContextualCompressor:
    """Contextual compression s source-aware budgeting"""

    def __init__(self, config: CompressionConfig):
        self.config = config
        self.chunker = DiscourseAwareChunker(config)
        self.compression_stats = {
            "compressions_performed": 0,
            "avg_compression_ratio": 0.0,
            "context_usage_efficiency": 0.0,
        }

    async def compress_documents(
        self, documents: list[dict[str, Any]], query: str = ""
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Kompresuje dokumenty s source-aware budgetingem"""
        if not self.config.enabled or not documents:
            return documents, {"compression_enabled": False}

        start_time = time.time()

        # Kategorizuj sources
        categorized_docs = self._categorize_sources(documents)

        # Alokuj token budget
        budget_allocation = self._allocate_token_budget(categorized_docs)

        # Kompresuj každou kategorii
        compressed_docs = []
        compression_metadata = {
            "compression_enabled": True,
            "original_documents": len(documents),
            "target_tokens": self.config.max_context_tokens,
            "source_categories": {},
            "budget_allocation": budget_allocation,
        }

        for source_type, docs in categorized_docs.items():
            if not docs:
                continue

            allocated_tokens = budget_allocation.get(source_type, 0)
            if allocated_tokens <= 0:
                continue

            # Kompresuj dokumenty této kategorie
            category_compressed = await self._compress_document_category(
                docs, allocated_tokens, query, source_type
            )

            compressed_docs.extend(category_compressed)

            # Track metadata
            compression_metadata["source_categories"][source_type] = {
                "original_docs": len(docs),
                "compressed_docs": len(category_compressed),
                "allocated_tokens": allocated_tokens,
                "estimated_tokens": sum(
                    len(doc.get("content", "").split()) for doc in category_compressed
                ),
            }

        # Finální seřazení podle skóre
        compressed_docs.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Vypočítaj metriky
        elapsed_time = time.time() - start_time
        original_tokens = sum(len(doc.get("content", "").split()) for doc in documents)
        compressed_tokens = sum(len(doc.get("content", "").split()) for doc in compressed_docs)

        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 0
        context_usage = compressed_tokens / self.config.max_context_tokens

        compression_metadata.update(
            {
                "compression_time_seconds": elapsed_time,
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "compression_ratio": compression_ratio,
                "context_usage_efficiency": context_usage,
                "target_compression_ratio": self.config.target_compression_ratio,
            }
        )

        # Update stats
        self._update_compression_stats(compression_ratio, context_usage)

        logger.info(
            f"Contextual compression: {len(documents)} -> {len(compressed_docs)} docs, "
            f"ratio: {compression_ratio:.2f}"
        )

        return compressed_docs, compression_metadata

    def _categorize_sources(
        self, documents: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Kategorizuje sources podle typu"""
        categories = {
            "primary_literature": [],
            "secondary_sources": [],
            "aggregators": [],
            "unknown": [],
        }

        for doc in documents:
            source_url = doc.get("url", "")
            source_domain = doc.get("source_domain", "")

            category = self._classify_source(source_url, source_domain)
            categories[category].append(doc)

        return categories

    def _classify_source(self, url: str, domain: str) -> str:
        """Klasifikuje source do kategorie"""
        # Primary literature patterns
        primary_domains = [
            "arxiv.org",
            "nature.com",
            "science.org",
            "cell.com",
            "pnas.org",
            "pubmed.ncbi.nlm.nih.gov",
            "acm.org",
            "ieee.org",
        ]

        # Aggregator patterns
        aggregator_domains = [
            "wikipedia.org",
            "reddit.com",
            "twitter.com",
            "facebook.com",
            "medium.com",
            "blog",
            "news",
        ]

        source_text = (url + " " + domain).lower()

        for domain in primary_domains:
            if domain in source_text:
                return "primary_literature"

        for domain in aggregator_domains:
            if domain in source_text:
                return "aggregators"

        # Check for secondary indicators
        if any(indicator in source_text for indicator in ["review", "survey", "proceedings"]):
            return "secondary_sources"

        return "unknown"

    def _allocate_token_budget(
        self, categorized_docs: dict[str, list[dict[str, Any]]]
    ) -> dict[str, int]:
        """Alokuje token budget podle source priorities"""
        total_budget = self.config.max_context_tokens

        # Váhy pro jednotlivé kategorie
        weights = {
            "primary_literature": self.config.primary_source_weight,
            "secondary_sources": 1.0,
            "aggregators": self.config.aggregator_penalty,
            "unknown": 0.8,
        }

        # Vypočítaj proporcionální alokaci
        total_docs = sum(len(docs) for docs in categorized_docs.values())
        if total_docs == 0:
            return {}

        allocation = {}
        total_weighted_docs = 0

        for category, docs in categorized_docs.items():
            if docs:
                weighted_count = len(docs) * weights.get(category, 1.0)
                total_weighted_docs += weighted_count

        for category, docs in categorized_docs.items():
            if docs:
                weighted_count = len(docs) * weights.get(category, 1.0)
                proportion = weighted_count / total_weighted_docs if total_weighted_docs > 0 else 0
                allocation[category] = int(total_budget * proportion)

        return allocation

    async def _compress_document_category(
        self, documents: list[dict[str, Any]], token_budget: int, query: str, source_type: str
    ) -> list[dict[str, Any]]:
        """Kompresuje dokumenty v kategorii"""
        if token_budget <= 0 or not documents:
            return []

        # Chunková dokumenty
        all_chunks = []
        for doc in documents:
            content = doc.get("content", "")
            chunks = self.chunker.chunk_text(content, doc)

            # Obohatí chunks o document metadata
            for chunk in chunks:
                chunk.update(
                    {
                        "doc_id": doc.get("id"),
                        "doc_score": doc.get("score", 0),
                        "source_type": source_type,
                        "url": doc.get("url", ""),
                        "relevance_score": self._calculate_relevance_score(chunk["text"], query),
                    }
                )

            all_chunks.extend(chunks)

        # Seřaď chunks podle salience a relevance
        all_chunks.sort(
            key=lambda x: (x["salience_score"] + x["relevance_score"]) / 2, reverse=True
        )

        # Vybírej chunks do token budgetu
        selected_chunks = []
        current_tokens = 0

        for chunk in all_chunks:
            chunk_tokens = len(chunk["text"].split())

            if current_tokens + chunk_tokens <= token_budget:
                selected_chunks.append(chunk)
                current_tokens += chunk_tokens
            else:
                # Try to fit truncated version
                remaining_tokens = token_budget - current_tokens
                if remaining_tokens > 50:  # Minimum useful chunk size
                    truncated_text = self._truncate_to_tokens(chunk["text"], remaining_tokens)
                    if truncated_text:
                        truncated_chunk = chunk.copy()
                        truncated_chunk["text"] = truncated_text
                        truncated_chunk["truncated"] = True
                        selected_chunks.append(truncated_chunk)
                break

        # Kombinuj chunks zpět do dokumentů
        compressed_docs = self._recombine_chunks(selected_chunks)

        return compressed_docs

    def _calculate_relevance_score(self, text: str, query: str) -> float:
        """Vypočítá relevance score chunku k query"""
        if not query:
            return 0.5

        query_terms = set(query.lower().split())
        text_terms = set(text.lower().split())

        if not query_terms:
            return 0.5

        intersection = query_terms.intersection(text_terms)
        jaccard = len(intersection) / len(query_terms.union(text_terms))

        return jaccard

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Zkrátí text na maximální počet tokenů"""
        words = text.split()
        if len(words) <= max_tokens:
            return text

        # Try to break at sentence boundary
        truncated_words = words[:max_tokens]
        truncated_text = " ".join(truncated_words)

        # Look for last complete sentence
        sentences = truncated_text.split(".")
        if len(sentences) > 1:
            return ".".join(sentences[:-1]) + "."

        return truncated_text

    def _recombine_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rekombinuje chunks zpět do dokumentů"""
        # Group chunks by document
        doc_chunks = {}
        for chunk in chunks:
            doc_id = chunk.get("doc_id", "unknown")
            if doc_id not in doc_chunks:
                doc_chunks[doc_id] = []
            doc_chunks[doc_id].append(chunk)

        # Rebuild documents
        compressed_docs = []
        for doc_id, doc_chunk_list in doc_chunks.items():
            if not doc_chunk_list:
                continue

            # Sort chunks by index
            doc_chunk_list.sort(key=lambda x: x.get("chunk_index", 0))

            # Combine chunk texts
            combined_text = "\n\n".join(chunk["text"] for chunk in doc_chunk_list)

            # Use metadata from first chunk
            first_chunk = doc_chunk_list[0]

            compressed_doc = {
                "id": doc_id,
                "content": combined_text,
                "score": first_chunk.get("doc_score", 0),
                "url": first_chunk.get("url", ""),
                "source_type": first_chunk.get("source_type", "unknown"),
                "compression_metadata": {
                    "chunks_used": len(doc_chunk_list),
                    "total_chunks": len(doc_chunk_list),  # Simplified
                    "avg_salience": np.mean([c["salience_score"] for c in doc_chunk_list]),
                    "truncated": any(c.get("truncated", False) for c in doc_chunk_list),
                },
            }

            compressed_docs.append(compressed_doc)

        return compressed_docs

    def _update_compression_stats(self, compression_ratio: float, context_usage: float):
        """Aktualizuje compression statistiky"""
        self.compression_stats["compressions_performed"] += 1

        # Exponential moving average
        alpha = 0.1

        if self.compression_stats["avg_compression_ratio"] == 0:
            self.compression_stats["avg_compression_ratio"] = compression_ratio
        else:
            self.compression_stats["avg_compression_ratio"] = (
                alpha * compression_ratio
                + (1 - alpha) * self.compression_stats["avg_compression_ratio"]
            )

        if self.compression_stats["context_usage_efficiency"] == 0:
            self.compression_stats["context_usage_efficiency"] = context_usage
        else:
            self.compression_stats["context_usage_efficiency"] = (
                alpha * context_usage
                + (1 - alpha) * self.compression_stats["context_usage_efficiency"]
            )


# Hlavní integration class
class IntegratedReRankingSystem:
    """Integrovaný re-ranking a compression systém"""

    def __init__(
        self,
        rerank_config: ReRankingConfig,
        compression_config: CompressionConfig,
        embedding_model=None,
        llm_client=None,
    ):
        self.rerank_config = rerank_config
        self.compression_config = compression_config

        self.cross_encoder = CrossEncoderReRanker(rerank_config)
        self.llm_rater = LLMRater(rerank_config, llm_client)
        self.compressor = ContextualCompressor(compression_config)

    async def process_documents(
        self, query: str, documents: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Kompletní processing: re-ranking + compression"""
        start_time = time.time()

        processing_metadata = {
            "query": query,
            "input_documents": len(documents),
            "reranking": {},
            "compression": {},
        }

        # 1. Re-ranking
        reranked_docs = documents
        if self.rerank_config.enabled and len(documents) > 1:
            if self.rerank_config.strategy == ReRankingStrategy.CROSS_ENCODER:
                reranked_docs, rerank_meta = await self.cross_encoder.rerank_documents(
                    query, documents
                )
            elif self.rerank_config.strategy == ReRankingStrategy.LLM_RATER:
                reranked_docs, rerank_meta = await self.llm_rater.rate_documents(query, documents)
            elif self.rerank_config.strategy == ReRankingStrategy.HYBRID:
                # Combine both approaches
                cross_docs, cross_meta = await self.cross_encoder.rerank_documents(query, documents)
                llm_docs, llm_meta = await self.llm_rater.rate_documents(query, cross_docs)
                reranked_docs = llm_docs
                rerank_meta = {
                    "strategy": "hybrid",
                    "cross_encoder": cross_meta,
                    "llm_rater": llm_meta,
                }

            processing_metadata["reranking"] = rerank_meta

        # 2. Contextual compression
        compressed_docs, compression_meta = await self.compressor.compress_documents(
            reranked_docs, query
        )
        processing_metadata["compression"] = compression_meta

        # 3. Final metadata
        elapsed_time = time.time() - start_time
        processing_metadata.update(
            {
                "total_processing_time": elapsed_time,
                "output_documents": len(compressed_docs),
                "overall_compression_ratio": (
                    len(compressed_docs) / len(documents) if documents else 0
                ),
            }
        )

        logger.info(
            f"Re-ranking + compression completed in {elapsed_time:.2f}s: "
            f"{len(documents)} -> {len(compressed_docs)} documents"
        )

        return compressed_docs, processing_metadata

    def get_stats(self) -> dict[str, Any]:
        """Získá statistiky celého systému"""
        return {
            "cross_encoder_stats": self.cross_encoder.rerank_stats,
            "llm_rater_stats": self.llm_rater.rating_stats,
            "compression_stats": self.compressor.compression_stats,
            "chunker_stats": self.compressor.chunker.chunk_stats,
        }


# Factory funkce
def create_reranking_system(
    config: dict[str, Any], embedding_model=None, llm_client=None
) -> IntegratedReRankingSystem:
    """Factory funkce pro integrated re-ranking systém"""
    rerank_config_dict = config.get("reranking", {})
    compression_config_dict = config.get("compression", {})

    rerank_config = ReRankingConfig(**rerank_config_dict)
    compression_config = CompressionConfig(**compression_config_dict)

    return IntegratedReRankingSystem(rerank_config, compression_config, embedding_model, llm_client)


# Export hlavních tříd
__all__ = [
    "CompressionConfig",
    "IntegratedReRankingSystem",
    "ReRankingConfig",
    "create_reranking_system",
]
