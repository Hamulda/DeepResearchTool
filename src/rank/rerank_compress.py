#!/usr/bin/env python3
"""Cross-encoder re-ranking a adaptivní contextual compression
FÁZE 2: Pokročilé re-ranking s margin-of-victory a discourse-aware chunking

Author: Senior Python/MLOps Agent
"""

import asyncio
from dataclasses import dataclass
import logging
import math
import re
import time
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RerankingResult:
    """Výsledek re-ranking procesu"""

    reranked_documents: list[dict[str, Any]]
    reranking_scores: dict[str, float]
    margin_of_victory: dict[str, float]
    rationale: dict[str, str]
    calibration_metadata: dict[str, Any]


@dataclass
class CompressionResult:
    """Výsledek contextual compression"""

    compressed_documents: list[dict[str, Any]]
    original_token_count: int
    compressed_token_count: int
    compression_ratio: float
    efficiency_score: float
    source_budget_allocation: dict[str, int]


class CrossEncoderReranker:
    """Cross-encoder re-ranking s LLM-as-rater fallback"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.rerank_config = config.get("rank", {}).get("reranking", {})

        # Re-ranking parameters
        self.enabled = self.rerank_config.get("enabled", True)
        self.top_k_rerank = self.rerank_config.get("top_k_rerank", 20)
        self.cross_encoder_model = self.rerank_config.get("cross_encoder_model", "local_llm")
        self.score_aggregation = self.rerank_config.get("score_aggregation", "margin_of_victory")

        # Calibration parameters
        self.calibration_enabled = self.rerank_config.get("calibration_enabled", True)
        self.calibration_curve = self.rerank_config.get("calibration_curve", "platt_scaling")

        # LLM-as-rater settings
        self.llm_rater_config = self.rerank_config.get("llm_rater", {})
        self.rating_criteria = self.llm_rater_config.get(
            "criteria", ["relevance", "authority", "recency", "evidence_quality"]
        )

        # Performance tracking
        self.rerank_stats = {
            "total_reranks": 0,
            "avg_rerank_time": 0.0,
            "score_distributions": [],
            "margin_distributions": [],
        }

    async def rerank_documents(
        self, query: str, documents: list[dict[str, Any]]
    ) -> RerankingResult:
        """Re-ranking dokumentů s cross-encoder nebo LLM-as-rater

        Args:
            query: Původní dotaz
            documents: Dokumenty k re-rankingu

        Returns:
            RerankingResult s re-ranked dokumenty

        """
        start_time = time.time()

        if not self.enabled or len(documents) <= 1:
            return self._create_passthrough_result(documents)

        # Limit to top-k for re-ranking efficiency
        docs_to_rerank = documents[: self.top_k_rerank]
        remaining_docs = documents[self.top_k_rerank :]

        logger.info(f"Re-ranking {len(docs_to_rerank)} documents with {self.cross_encoder_model}")

        # Generate pairwise scores or direct scores
        if self.cross_encoder_model == "local_llm":
            scores, rationale = await self._llm_based_reranking(query, docs_to_rerank)
        else:
            scores, rationale = await self._cross_encoder_reranking(query, docs_to_rerank)

        # Calculate margin of victory
        margin_of_victory = self._calculate_margin_of_victory(scores)

        # Apply calibration if enabled
        if self.calibration_enabled:
            calibrated_scores, calibration_metadata = await self._apply_score_calibration(scores)
            scores = calibrated_scores
        else:
            calibration_metadata = {"calibration_applied": False}

        # Sort by new scores
        scored_docs = [
            (doc, scores.get(doc.get("id", str(i)), 0.0)) for i, doc in enumerate(docs_to_rerank)
        ]
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Reconstruct document list
        reranked_docs = [doc for doc, score in scored_docs] + remaining_docs

        # Update document scores and add metadata
        for i, doc in enumerate(reranked_docs):
            doc_id = doc.get("id", str(i))
            doc["rerank_score"] = scores.get(doc_id, doc.get("score", 0.0))
            doc["rerank_position"] = i + 1
            doc["margin_of_victory"] = margin_of_victory.get(doc_id, 0.0)

            if doc_id in rationale:
                doc["rerank_rationale"] = rationale[doc_id]

        rerank_time = time.time() - start_time
        self._update_rerank_stats(rerank_time, scores, margin_of_victory)

        logger.info(f"Re-ranking completed in {rerank_time:.3f}s")

        return RerankingResult(
            reranked_documents=reranked_docs,
            reranking_scores=scores,
            margin_of_victory=margin_of_victory,
            rationale=rationale,
            calibration_metadata=calibration_metadata,
        )

    async def _llm_based_reranking(
        self, query: str, documents: list[dict[str, Any]]
    ) -> tuple[dict[str, float], dict[str, str]]:
        """LLM-as-rater re-ranking s detailním rationale"""
        scores = {}
        rationale = {}

        # Create ranking prompt
        doc_summaries = []
        for i, doc in enumerate(documents):
            content_preview = doc.get("content", "")[:300] + "..."
            title = doc.get("title", f"Document {i+1}")
            doc_summaries.append(f"[{i+1}] {title}\n{content_preview}")

        prompt = f"""Rank these documents by relevance to the query: "{query}"

Consider these criteria:
- Relevance: How well does the document answer the query?
- Authority: Is the source credible and authoritative?
- Recency: Is the information current and up-to-date?
- Evidence Quality: Does it provide good supporting evidence?

Documents:
{chr(10).join(doc_summaries)}

For each document, provide:
1. Score (0.0-1.0)
2. Brief rationale explaining the score

Format: [Doc#] Score: X.X | Rationale: explanation"""

        # Mock LLM response - real implementation would call Ollama
        await asyncio.sleep(0.5)  # Simulate LLM inference time

        # Generate mock scores and rationale
        for i, doc in enumerate(documents):
            doc_id = doc.get("id", str(i))

            # Mock scoring with some variance
            base_score = doc.get("score", 0.5)
            noise = (hash(doc_id) % 100) / 1000.0  # Small deterministic noise
            final_score = min(max(base_score + noise, 0.0), 1.0)

            scores[doc_id] = final_score
            rationale[doc_id] = (
                f"Score {final_score:.3f} based on relevance and source quality analysis"
            )

        return scores, rationale

    async def _cross_encoder_reranking(
        self, query: str, documents: list[dict[str, Any]]
    ) -> tuple[dict[str, float], dict[str, str]]:
        """Cross-encoder model re-ranking"""
        scores = {}
        rationale = {}

        # Mock cross-encoder scoring
        await asyncio.sleep(0.2)  # Simulate model inference

        for i, doc in enumerate(documents):
            doc_id = doc.get("id", str(i))
            content = doc.get("content", "")

            # Mock cross-encoder score based on query-document interaction
            query_terms = set(query.lower().split())
            doc_terms = set(content.lower().split())
            overlap = len(query_terms.intersection(doc_terms))

            # Simple relevance scoring
            relevance_score = min(overlap / max(len(query_terms), 1), 1.0)

            # Add position bias (earlier documents get slight boost)
            position_bias = 1.0 - (i * 0.02)

            final_score = relevance_score * position_bias
            scores[doc_id] = final_score
            rationale[doc_id] = f"Cross-encoder relevance: {relevance_score:.3f}, position: {i+1}"

        return scores, rationale

    def _calculate_margin_of_victory(self, scores: dict[str, float]) -> dict[str, float]:
        """Výpočet margin of victory pro každý dokument"""
        sorted_scores = sorted(scores.values(), reverse=True)
        margin_of_victory = {}

        for doc_id, score in scores.items():
            # Find position in sorted list
            position = next(i for i, s in enumerate(sorted_scores) if s == score)

            if position == 0:
                # Winner - margin vs second place
                margin = score - (sorted_scores[1] if len(sorted_scores) > 1 else 0.0)
            else:
                # Margin vs higher-ranked document
                margin = score - sorted_scores[position - 1]

            margin_of_victory[doc_id] = margin

        return margin_of_victory

    async def _apply_score_calibration(
        self, scores: dict[str, float]
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Aplikuje score calibration pro lepší confidence estimates"""
        # Mock Platt scaling calibration
        calibrated_scores = {}

        for doc_id, raw_score in scores.items():
            # Simple sigmoid calibration
            calibrated_score = 1.0 / (1.0 + math.exp(-5.0 * (raw_score - 0.5)))
            calibrated_scores[doc_id] = calibrated_score

        calibration_metadata = {
            "calibration_applied": True,
            "calibration_method": self.calibration_curve,
            "score_shift_mean": sum(calibrated_scores.values()) / len(calibrated_scores)
            - sum(scores.values()) / len(scores),
        }

        return calibrated_scores, calibration_metadata

    def _create_passthrough_result(self, documents: list[dict[str, Any]]) -> RerankingResult:
        """Vytvoří passthrough result bez re-rankingu"""
        scores = {doc.get("id", str(i)): doc.get("score", 0.0) for i, doc in enumerate(documents)}

        return RerankingResult(
            reranked_documents=documents,
            reranking_scores=scores,
            margin_of_victory={},
            rationale={},
            calibration_metadata={"reranking_skipped": True},
        )

    def _update_rerank_stats(
        self, rerank_time: float, scores: dict[str, float], margins: dict[str, float]
    ):
        """Aktualizace re-ranking statistik"""
        self.rerank_stats["total_reranks"] += 1

        # Moving average of rerank time
        total = self.rerank_stats["total_reranks"]
        self.rerank_stats["avg_rerank_time"] = (
            self.rerank_stats["avg_rerank_time"] * (total - 1) + rerank_time
        ) / total

        # Track score and margin distributions
        self.rerank_stats["score_distributions"].extend(scores.values())
        self.rerank_stats["margin_distributions"].extend(margins.values())

        # Keep only recent stats
        for key in ["score_distributions", "margin_distributions"]:
            if len(self.rerank_stats[key]) > 1000:
                self.rerank_stats[key] = self.rerank_stats[key][-1000:]


class AdaptiveContextualCompressor:
    """Contextual compression s discourse-aware chunking a token budgeting"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.compress_config = config.get("compress", {})

        # Token budgeting
        self.max_context_tokens = self.compress_config.get("max_context_tokens", 8000)
        self.salience_threshold = self.compress_config.get("salience_threshold", 0.6)
        self.source_priority_weights = self.compress_config.get(
            "source_priority_weights",
            {
                "primary_literature": 1.4,
                "review_article": 1.2,
                "government": 1.1,
                "news_tier1": 1.0,
                "news_tier2": 0.8,
                "blog": 0.6,
                "aggregator": 0.4,
            },
        )

        # Discourse-aware settings
        self.discourse_markers = self.compress_config.get(
            "discourse_markers",
            {
                "headings": ["introduction", "methods", "results", "conclusion", "abstract"],
                "lists": ["•", "1.", "2.", "3.", "-", "*"],
                "citations": ["(", ")", "[", "]", "et al.", "doi:"],
                "speech_acts": [
                    "however",
                    "therefore",
                    "furthermore",
                    "in contrast",
                    "importantly",
                ],
            },
        )

        # Adaptive chunking parameters
        self.entity_density_threshold = self.compress_config.get("entity_density_threshold", 0.15)
        self.claim_density_threshold = self.compress_config.get("claim_density_threshold", 0.1)

        # Statistics
        self.compression_stats = {
            "total_compressions": 0,
            "avg_compression_ratio": 0.0,
            "avg_efficiency_score": 0.0,
            "source_budget_usage": {},
        }

    async def compress_context(
        self, documents: list[dict[str, Any]], query: str
    ) -> CompressionResult:
        """Adaptivní contextual compression s token budgeting

        Args:
            documents: Dokumenty k kompresi
            query: Původní dotaz pro salience calculation

        Returns:
            CompressionResult s compressed content

        """
        start_time = time.time()

        # Calculate initial token count
        original_tokens = sum(self._estimate_tokens(doc.get("content", "")) for doc in documents)

        if original_tokens <= self.max_context_tokens:
            # No compression needed
            return self._create_passthrough_compression(documents, original_tokens)

        logger.info(f"Compressing context: {original_tokens} → {self.max_context_tokens} tokens")

        # Step 1: Source-aware budget allocation
        source_budgets = self._allocate_source_budgets(documents)

        # Step 2: Discourse-aware chunking
        chunked_docs = await self._perform_discourse_chunking(documents)

        # Step 3: Salience filtering
        salient_chunks = await self._filter_by_salience(chunked_docs, query)

        # Step 4: Token budget enforcement
        compressed_docs = await self._enforce_token_budget(salient_chunks, source_budgets)

        # Calculate final metrics
        compressed_tokens = sum(
            self._estimate_tokens(doc.get("content", "")) for doc in compressed_docs
        )
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        efficiency_score = self._calculate_efficiency_score(documents, compressed_docs, query)

        compression_time = time.time() - start_time
        self._update_compression_stats(compression_ratio, efficiency_score, source_budgets)

        logger.info(
            f"Compression completed: {compression_ratio:.3f} ratio, {efficiency_score:.3f} efficiency in {compression_time:.3f}s"
        )

        return CompressionResult(
            compressed_documents=compressed_docs,
            original_token_count=original_tokens,
            compressed_token_count=compressed_tokens,
            compression_ratio=compression_ratio,
            efficiency_score=efficiency_score,
            source_budget_allocation=source_budgets,
        )

    def _allocate_source_budgets(self, documents: list[dict[str, Any]]) -> dict[str, int]:
        """Alokuje token budget podle source priority"""
        # Group documents by source type
        source_groups = {}
        for doc in documents:
            source_type = doc.get("metadata", {}).get("source_type", "unknown")
            if source_type not in source_groups:
                source_groups[source_type] = []
            source_groups[source_type].append(doc)

        # Calculate weighted budget allocation
        total_weight = 0
        source_weights = {}

        for source_type, docs in source_groups.items():
            weight = self.source_priority_weights.get(source_type, 0.5)
            doc_count = len(docs)
            source_weights[source_type] = weight * doc_count
            total_weight += source_weights[source_type]

        # Allocate budget proportionally
        source_budgets = {}
        for source_type, weight in source_weights.items():
            if total_weight > 0:
                budget = int((weight / total_weight) * self.max_context_tokens)
                source_budgets[source_type] = budget
            else:
                source_budgets[source_type] = self.max_context_tokens // len(source_groups)

        return source_budgets

    async def _perform_discourse_chunking(
        self, documents: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Discourse-aware chunking s adaptivní velikostí"""
        chunked_docs = []

        for doc in documents:
            content = doc.get("content", "")

            # Detect discourse structure
            chunks = self._adaptive_chunk_content(content)

            # Create document chunks
            for i, chunk in enumerate(chunks):
                chunk_doc = doc.copy()
                chunk_doc["content"] = chunk["content"]
                chunk_doc["chunk_id"] = f"{doc.get('id', 'doc')}_{i}"
                chunk_doc["chunk_metadata"] = {
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "discourse_type": chunk["discourse_type"],
                    "entity_density": chunk["entity_density"],
                    "claim_density": chunk["claim_density"],
                }
                chunked_docs.append(chunk_doc)

        return chunked_docs

    def _adaptive_chunk_content(self, content: str) -> list[dict[str, Any]]:
        """Adaptivní chunking based on discourse markers a content density"""
        paragraphs = content.split("\n\n")
        chunks = []
        current_chunk = []

        for para in paragraphs:
            if not para.strip():
                continue

            # Detect discourse type
            discourse_type = self._identify_discourse_type(para)

            # Calculate content densities
            entity_density = self._calculate_entity_density(para)
            claim_density = self._calculate_claim_density(para)

            # Decide chunking strategy
            should_break = (
                discourse_type in ["heading", "conclusion"]
                or entity_density > self.entity_density_threshold
                or claim_density > self.claim_density_threshold
                or len(" ".join(current_chunk + [para]).split()) > 200  # Max chunk size
            )

            if should_break and current_chunk:
                # Finalize current chunk
                chunk_content = "\n\n".join(current_chunk)
                chunks.append(
                    {
                        "content": chunk_content,
                        "discourse_type": "mixed",
                        "entity_density": self._calculate_entity_density(chunk_content),
                        "claim_density": self._calculate_claim_density(chunk_content),
                    }
                )
                current_chunk = []

            current_chunk.append(para)

        # Add final chunk
        if current_chunk:
            chunk_content = "\n\n".join(current_chunk)
            chunks.append(
                {
                    "content": chunk_content,
                    "discourse_type": "mixed",
                    "entity_density": self._calculate_entity_density(chunk_content),
                    "claim_density": self._calculate_claim_density(chunk_content),
                }
            )

        return chunks

    def _identify_discourse_type(self, text: str) -> str:
        """Identifikuje typ discourse v textu"""
        text_lower = text.lower()

        # Check for headings
        for heading in self.discourse_markers["headings"]:
            if heading in text_lower:
                return "heading"

        # Check for lists
        for list_marker in self.discourse_markers["lists"]:
            if text.strip().startswith(list_marker):
                return "list"

        # Check for citations
        citation_count = sum(1 for marker in self.discourse_markers["citations"] if marker in text)
        if citation_count >= 2:
            return "citation_heavy"

        # Check for speech acts
        for speech_act in self.discourse_markers["speech_acts"]:
            if speech_act in text_lower:
                return "argumentative"

        return "narrative"

    def _calculate_entity_density(self, text: str) -> float:
        """Výpočet hustoty entit v textu"""
        words = text.split()
        if not words:
            return 0.0

        # Simple entity detection - kapitalized words, numbers, dates
        entity_patterns = [
            r"\b[A-Z][a-z]+\b",  # Proper nouns
            r"\b\d{4}\b",  # Years
            r"\b\d+\.\d+\b",  # Numbers with decimals
            r"\b[A-Z]{2,}\b",  # Acronyms
        ]

        entity_count = 0
        for pattern in entity_patterns:
            entity_count += len(re.findall(pattern, text))

        return entity_count / len(words)

    def _calculate_claim_density(self, text: str) -> float:
        """Výpočet hustoty tvrzení v textu"""
        sentences = re.split(r"[.!?]+", text)
        if not sentences:
            return 0.0

        claim_indicators = [
            "research shows",
            "studies indicate",
            "evidence suggests",
            "findings reveal",
            "data demonstrates",
            "results show",
            "according to",
            "reported that",
            "concluded that",
        ]

        claim_count = sum(
            1
            for sentence in sentences
            for indicator in claim_indicators
            if indicator in sentence.lower()
        )

        return claim_count / len(sentences)

    async def _filter_by_salience(
        self, chunks: list[dict[str, Any]], query: str
    ) -> list[dict[str, Any]]:
        """Filtruje chunks podle salience vůči query"""
        query_terms = set(query.lower().split())
        salient_chunks = []

        for chunk in chunks:
            content = chunk.get("content", "")

            # Calculate salience score
            salience_score = self._calculate_salience_score(content, query_terms)

            # Add discourse and density bonuses
            chunk_meta = chunk.get("chunk_metadata", {})
            discourse_bonus = {
                "heading": 0.2,
                "citation_heavy": 0.15,
                "argumentative": 0.1,
                "list": 0.05,
                "narrative": 0.0,
                "mixed": 0.02,
            }.get(chunk_meta.get("discourse_type", "mixed"), 0.0)

            density_bonus = (
                chunk_meta.get("entity_density", 0.0) * 0.1
                + chunk_meta.get("claim_density", 0.0) * 0.15
            )

            final_salience = salience_score + discourse_bonus + density_bonus

            chunk["salience_score"] = final_salience

            if final_salience >= self.salience_threshold:
                salient_chunks.append(chunk)

        # Sort by salience
        salient_chunks.sort(key=lambda x: x["salience_score"], reverse=True)

        return salient_chunks

    def _calculate_salience_score(self, content: str, query_terms: set) -> float:
        """Výpočet salience score pro content"""
        content_terms = set(content.lower().split())

        if not content_terms:
            return 0.0

        # Term overlap
        overlap = len(query_terms.intersection(content_terms))
        term_score = overlap / len(query_terms) if query_terms else 0.0

        # Term frequency in content
        tf_score = sum(content.lower().count(term) for term in query_terms) / len(content.split())

        # Combined salience
        return term_score * 0.6 + tf_score * 0.4

    async def _enforce_token_budget(
        self, chunks: list[dict[str, Any]], source_budgets: dict[str, int]
    ) -> list[dict[str, Any]]:
        """Enforces token budget per source type"""
        # Group chunks by source type
        source_chunks = {}
        for chunk in chunks:
            source_type = chunk.get("metadata", {}).get("source_type", "unknown")
            if source_type not in source_chunks:
                source_chunks[source_type] = []
            source_chunks[source_type].append(chunk)

        compressed_chunks = []

        for source_type, type_chunks in source_chunks.items():
            budget = source_budgets.get(source_type, 1000)
            current_tokens = 0

            # Sort by salience and add until budget exhausted
            type_chunks.sort(key=lambda x: x.get("salience_score", 0.0), reverse=True)

            for chunk in type_chunks:
                chunk_tokens = self._estimate_tokens(chunk.get("content", ""))

                if current_tokens + chunk_tokens <= budget:
                    compressed_chunks.append(chunk)
                    current_tokens += chunk_tokens
                else:
                    # Try to fit partial content
                    remaining_budget = budget - current_tokens
                    if remaining_budget > 50:  # Minimum viable chunk
                        truncated_content = self._truncate_content(
                            chunk.get("content", ""), remaining_budget
                        )
                        if truncated_content:
                            truncated_chunk = chunk.copy()
                            truncated_chunk["content"] = truncated_content
                            truncated_chunk["truncated"] = True
                            compressed_chunks.append(truncated_chunk)
                    break

        return compressed_chunks

    def _estimate_tokens(self, text: str) -> int:
        """Odhad počtu tokenů v textu"""
        # Simple approximation: ~0.75 tokens per word
        return int(len(text.split()) * 0.75)

    def _truncate_content(self, content: str, max_tokens: int) -> str:
        """Truncate content to fit token budget"""
        words = content.split()
        max_words = int(max_tokens / 0.75)  # Convert tokens to words

        if len(words) <= max_words:
            return content

        # Try to truncate at sentence boundary
        truncated_words = words[:max_words]
        truncated_text = " ".join(truncated_words)

        # Find last complete sentence
        sentences = re.split(r"[.!?]+", truncated_text)
        if len(sentences) > 1:
            return ".".join(sentences[:-1]) + "."

        return truncated_text + "..."

    def _calculate_efficiency_score(
        self, original_docs: list[dict[str, Any]], compressed_docs: list[dict[str, Any]], query: str
    ) -> float:
        """Výpočet context usage efficiency"""
        # Calculate information retention
        original_unique_terms = set()
        compressed_unique_terms = set()

        for doc in original_docs:
            original_unique_terms.update(doc.get("content", "").lower().split())

        for doc in compressed_docs:
            compressed_unique_terms.update(doc.get("content", "").lower().split())

        term_retention = (
            len(compressed_unique_terms) / len(original_unique_terms)
            if original_unique_terms
            else 0.0
        )

        # Calculate query term coverage
        query_terms = set(query.lower().split())
        covered_query_terms = len(query_terms.intersection(compressed_unique_terms))
        query_coverage = covered_query_terms / len(query_terms) if query_terms else 0.0

        # Combined efficiency score
        return term_retention * 0.4 + query_coverage * 0.6

    def _create_passthrough_compression(
        self, documents: list[dict[str, Any]], token_count: int
    ) -> CompressionResult:
        """Creates passthrough result when no compression needed"""
        return CompressionResult(
            compressed_documents=documents,
            original_token_count=token_count,
            compressed_token_count=token_count,
            compression_ratio=1.0,
            efficiency_score=1.0,
            source_budget_allocation={},
        )

    def _update_compression_stats(
        self, compression_ratio: float, efficiency_score: float, source_budgets: dict[str, int]
    ):
        """Update compression statistics"""
        self.compression_stats["total_compressions"] += 1
        total = self.compression_stats["total_compressions"]

        # Moving averages
        self.compression_stats["avg_compression_ratio"] = (
            self.compression_stats["avg_compression_ratio"] * (total - 1) + compression_ratio
        ) / total
        self.compression_stats["avg_efficiency_score"] = (
            self.compression_stats["avg_efficiency_score"] * (total - 1) + efficiency_score
        ) / total

        # Track source budget usage
        for source_type, budget in source_budgets.items():
            if source_type not in self.compression_stats["source_budget_usage"]:
                self.compression_stats["source_budget_usage"][source_type] = []
            self.compression_stats["source_budget_usage"][source_type].append(budget)

    def get_stats(self) -> dict[str, Any]:
        """Returns compression statistics"""
        return self.compression_stats.copy()
