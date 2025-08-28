#!/usr/bin/env python3
"""
Gated Reranking Engine
Bi-encoder → Cross-encoder pipeline for efficient reranking

Author: Senior Python/MLOps Agent
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class RerankingStage(Enum):
    """Reranking stages in the pipeline"""
    BI_ENCODER = "bi_encoder"
    CROSS_ENCODER = "cross_encoder"
    UNCERTAINTY_GATE = "uncertainty_gate"
    FINAL_RANKING = "final_ranking"


@dataclass
class RerankingConfig:
    """Configuration for gated reranking"""
    # Gate configuration
    top_n_for_cross_encoder: int = 50  # How many docs to send to cross-encoder
    uncertainty_threshold: float = 0.3  # Uncertainty threshold for gating
    min_cross_encoder_candidates: int = 10  # Minimum docs for cross-encoder

    # Bi-encoder configuration
    bi_encoder_enabled: bool = True
    bi_encoder_batch_size: int = 32
    bi_encoder_max_length: int = 256

    # Cross-encoder configuration
    cross_encoder_enabled: bool = True
    cross_encoder_batch_size: int = 8
    cross_encoder_max_length: int = 512

    # Performance thresholds
    bi_encoder_timeout: float = 5.0  # seconds
    cross_encoder_timeout: float = 30.0  # seconds

    # Profile-specific overrides
    profile_configs: Dict[str, Dict[str, Any]] = None

    def __post_init__(self):
        if self.profile_configs is None:
            self.profile_configs = {
                "quick": {
                    "top_n_for_cross_encoder": 20,
                    "cross_encoder_timeout": 10.0,
                    "uncertainty_threshold": 0.4
                },
                "thorough": {
                    "top_n_for_cross_encoder": 100,
                    "cross_encoder_timeout": 60.0,
                    "uncertainty_threshold": 0.2
                }
            }


@dataclass
class RerankingResult:
    """Result of reranking operation"""
    documents: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    stage_times: Dict[str, float]
    total_time: float

    @property
    def reranked_count(self) -> int:
        return len(self.documents)


class BiEncoderReranker:
    """Lightweight bi-encoder for initial reranking"""

    def __init__(self, config: RerankingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize bi-encoder model (placeholder for actual model)"""
        # In production, this would load a sentence transformer model
        # For now, use simple similarity scoring
        logger.info("Bi-encoder reranker initialized (mock implementation)")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Rerank documents using bi-encoder"""

        start_time = time.time()

        if not documents:
            return documents, {"stage": "bi_encoder", "processing_time": 0.0}

        top_k = top_k or len(documents)

        # Mock bi-encoder scoring (in production, use actual embeddings)
        scored_docs = []
        for doc in documents:
            content = doc.get("content", "")

            # Simple similarity score (placeholder)
            similarity_score = self._calculate_mock_similarity(query, content)

            # Add uncertainty estimate
            uncertainty = self._estimate_uncertainty(similarity_score)

            doc_copy = doc.copy()
            doc_copy["bi_encoder_score"] = float(similarity_score)
            doc_copy["bi_encoder_uncertainty"] = float(uncertainty)
            doc_copy["reranking_stage"] = RerankingStage.BI_ENCODER.value

            scored_docs.append(doc_copy)

        # Sort by bi-encoder score
        scored_docs.sort(key=lambda x: x["bi_encoder_score"], reverse=True)

        # Take top-k
        top_docs = scored_docs[:top_k]

        processing_time = time.time() - start_time

        metadata = {
            "stage": "bi_encoder",
            "input_count": len(documents),
            "output_count": len(top_docs),
            "processing_time": processing_time,
            "avg_score": np.mean([d["bi_encoder_score"] for d in top_docs]),
            "score_std": np.std([d["bi_encoder_score"] for d in top_docs])
        }

        logger.info(f"Bi-encoder reranking: {len(documents)} → {len(top_docs)} docs ({processing_time:.2f}s)")

        return top_docs, metadata

    def _calculate_mock_similarity(self, query: str, content: str) -> float:
        """Mock similarity calculation (replace with actual embeddings)"""

        query_words = set(query.lower().split())
        content_words = set(content.lower().split()[:100])  # First 100 words

        if not query_words:
            return 0.0

        # Jaccard similarity + length penalty
        intersection = len(query_words.intersection(content_words))
        union = len(query_words.union(content_words))

        jaccard = intersection / union if union > 0 else 0.0

        # Add some noise to simulate embedding similarity
        noise = np.random.normal(0, 0.1)
        similarity = np.clip(jaccard + noise, 0.0, 1.0)

        return similarity

    def _estimate_uncertainty(self, score: float) -> float:
        """Estimate prediction uncertainty"""
        # High uncertainty for scores near decision boundary (0.5)
        return 1.0 - 2 * abs(score - 0.5)


class CrossEncoderReranker:
    """Heavy cross-encoder for precise reranking"""

    def __init__(self, config: RerankingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize cross-encoder model (placeholder)"""
        # In production, this would load a cross-encoder model
        logger.info("Cross-encoder reranker initialized (mock implementation)")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Rerank documents using cross-encoder"""

        start_time = time.time()

        if not documents:
            return documents, {"stage": "cross_encoder", "processing_time": 0.0}

        # Process in batches
        scored_docs = []
        batch_size = self.config.cross_encoder_batch_size

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_scores = self._score_batch(query, batch)

            for doc, score in zip(batch, batch_scores):
                doc_copy = doc.copy()
                doc_copy["cross_encoder_score"] = float(score)
                doc_copy["reranking_stage"] = RerankingStage.CROSS_ENCODER.value

                # Combine with bi-encoder score if available
                bi_score = doc.get("bi_encoder_score", 0.5)
                combined_score = 0.3 * bi_score + 0.7 * score  # Weighted combination
                doc_copy["combined_rerank_score"] = float(combined_score)

                scored_docs.append(doc_copy)

        # Sort by combined score
        scored_docs.sort(key=lambda x: x["combined_rerank_score"], reverse=True)

        processing_time = time.time() - start_time

        metadata = {
            "stage": "cross_encoder",
            "input_count": len(documents),
            "output_count": len(scored_docs),
            "processing_time": processing_time,
            "batches_processed": (len(documents) + batch_size - 1) // batch_size,
            "avg_score": np.mean([d["cross_encoder_score"] for d in scored_docs]),
            "score_std": np.std([d["cross_encoder_score"] for d in scored_docs])
        }

        logger.info(f"Cross-encoder reranking: {len(documents)} docs ({processing_time:.2f}s)")

        return scored_docs, metadata

    def _score_batch(self, query: str, documents: List[Dict[str, Any]]) -> List[float]:
        """Score a batch of documents (mock implementation)"""

        scores = []
        for doc in documents:
            content = doc.get("content", "")

            # Mock cross-encoder scoring (more sophisticated than bi-encoder)
            query_words = query.lower().split()
            content_words = content.lower().split()

            # Term frequency matching
            tf_score = 0.0
            for word in query_words:
                tf_score += content_words.count(word) / max(len(content_words), 1)

            # Position bonus for early matches
            position_bonus = 0.0
            for i, word in enumerate(content_words[:50]):  # First 50 words
                if word in query_words:
                    position_bonus += (50 - i) / 50 * 0.1

            # Length normalization
            length_penalty = min(len(content) / 1000, 1.0) * 0.1

            # Combine scores
            final_score = tf_score + position_bonus + length_penalty

            # Add noise and normalize
            noise = np.random.normal(0, 0.05)
            score = np.clip(final_score + noise, 0.0, 1.0)

            scores.append(score)

        return scores


class GatedReranker:
    """Main gated reranking engine"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reranking_config = self._load_reranking_config(config)

        # Initialize rerankers
        self.bi_encoder = BiEncoderReranker(self.reranking_config)
        self.cross_encoder = CrossEncoderReranker(self.reranking_config)

        # Statistics
        self.stats = {
            "total_queries": 0,
            "bi_encoder_only": 0,
            "cross_encoder_used": 0,
            "avg_bi_encoder_time": 0.0,
            "avg_cross_encoder_time": 0.0
        }

        logger.info("Gated reranker initialized")

    def _load_reranking_config(self, config: Dict[str, Any]) -> RerankingConfig:
        """Load reranking configuration"""
        rerank_cfg = config.get("reranking", {})
        profile = config.get("profile", "quick")

        base_config = RerankingConfig(
            top_n_for_cross_encoder=rerank_cfg.get("top_n_for_cross_encoder", 50),
            uncertainty_threshold=rerank_cfg.get("uncertainty_threshold", 0.3),
            bi_encoder_enabled=rerank_cfg.get("bi_encoder_enabled", True),
            cross_encoder_enabled=rerank_cfg.get("cross_encoder_enabled", True)
        )

        # Apply profile-specific overrides
        if profile in base_config.profile_configs:
            profile_overrides = base_config.profile_configs[profile]
            for key, value in profile_overrides.items():
                if hasattr(base_config, key):
                    setattr(base_config, key, value)

        return base_config

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> RerankingResult:
        """Main reranking pipeline with gating logic"""

        start_time = time.time()
        stage_times = {}
        all_metadata = {}

        if not documents:
            return RerankingResult(
                documents=[],
                metadata={"message": "No documents to rerank"},
                stage_times={},
                total_time=0.0
            )

        self.stats["total_queries"] += 1
        current_docs = documents.copy()

        # Stage 1: Bi-encoder reranking (if enabled)
        if self.reranking_config.bi_encoder_enabled:
            logger.info(f"Stage 1: Bi-encoder reranking {len(current_docs)} documents")

            bi_start = time.time()
            current_docs, bi_metadata = self.bi_encoder.rerank(
                query, current_docs, top_k=None
            )
            stage_times["bi_encoder"] = time.time() - bi_start
            all_metadata["bi_encoder"] = bi_metadata

        # Stage 2: Uncertainty-based gating
        gate_decision = self._should_use_cross_encoder(current_docs, context)

        if gate_decision["use_cross_encoder"] and self.reranking_config.cross_encoder_enabled:
            logger.info(f"Stage 2: Cross-encoder reranking top {gate_decision['candidates_count']} documents")

            # Select candidates for cross-encoder
            candidates = current_docs[:gate_decision["candidates_count"]]
            remaining = current_docs[gate_decision["candidates_count"]:]

            # Cross-encoder reranking
            cross_start = time.time()
            reranked_candidates, cross_metadata = self.cross_encoder.rerank(query, candidates)
            stage_times["cross_encoder"] = time.time() - cross_start
            all_metadata["cross_encoder"] = cross_metadata

            # Combine reranked candidates with remaining documents
            current_docs = reranked_candidates + remaining

            self.stats["cross_encoder_used"] += 1
        else:
            logger.info("Stage 2: Skipping cross-encoder (gating decision)")
            self.stats["bi_encoder_only"] += 1

        # Final ranking
        final_docs = self._finalize_ranking(current_docs)

        total_time = time.time() - start_time

        # Update statistics
        if "bi_encoder" in stage_times:
            self.stats["avg_bi_encoder_time"] = (
                self.stats["avg_bi_encoder_time"] * (self.stats["total_queries"] - 1) +
                stage_times["bi_encoder"]
            ) / self.stats["total_queries"]

        if "cross_encoder" in stage_times:
            self.stats["avg_cross_encoder_time"] = (
                self.stats["avg_cross_encoder_time"] * (self.stats["cross_encoder_used"] - 1) +
                stage_times["cross_encoder"]
            ) / self.stats["cross_encoder_used"]

        # Compile metadata
        result_metadata = {
            "total_input_docs": len(documents),
            "total_output_docs": len(final_docs),
            "gate_decision": gate_decision,
            "stages_used": list(stage_times.keys()),
            "stage_metadata": all_metadata,
            "config": {
                "top_n_for_cross_encoder": self.reranking_config.top_n_for_cross_encoder,
                "uncertainty_threshold": self.reranking_config.uncertainty_threshold
            }
        }

        logger.info(f"Gated reranking completed: {len(documents)} → {len(final_docs)} docs ({total_time:.2f}s)")

        return RerankingResult(
            documents=final_docs,
            metadata=result_metadata,
            stage_times=stage_times,
            total_time=total_time
        )

    def _should_use_cross_encoder(
        self,
        documents: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Decide whether to use cross-encoder based on uncertainty and other factors"""

        if not documents:
            return {"use_cross_encoder": False, "reason": "no_documents", "candidates_count": 0}

        # Check if we have bi-encoder scores and uncertainties
        has_bi_scores = all("bi_encoder_score" in doc for doc in documents)
        has_uncertainties = all("bi_encoder_uncertainty" in doc for doc in documents)

        if not has_bi_scores:
            # No bi-encoder scores, use cross-encoder for top-N
            candidates_count = min(
                self.reranking_config.top_n_for_cross_encoder,
                len(documents)
            )
            return {
                "use_cross_encoder": True,
                "reason": "no_bi_encoder_scores",
                "candidates_count": candidates_count
            }

        # Calculate uncertainty metrics
        if has_uncertainties:
            uncertainties = [doc["bi_encoder_uncertainty"] for doc in documents[:self.reranking_config.top_n_for_cross_encoder]]
            avg_uncertainty = np.mean(uncertainties)
            max_uncertainty = max(uncertainties)

            if avg_uncertainty > self.reranking_config.uncertainty_threshold:
                candidates_count = min(
                    self.reranking_config.top_n_for_cross_encoder,
                    len(documents)
                )
                return {
                    "use_cross_encoder": True,
                    "reason": "high_uncertainty",
                    "avg_uncertainty": avg_uncertainty,
                    "max_uncertainty": max_uncertainty,
                    "candidates_count": candidates_count
                }

        # Check score distribution
        scores = [doc["bi_encoder_score"] for doc in documents[:self.reranking_config.top_n_for_cross_encoder]]
        score_std = np.std(scores)

        if score_std < 0.1:  # Very similar scores, need better discrimination
            candidates_count = min(
                self.reranking_config.top_n_for_cross_encoder,
                len(documents)
            )
            return {
                "use_cross_encoder": True,
                "reason": "low_score_variance",
                "score_std": score_std,
                "candidates_count": candidates_count
            }

        # Context-based decisions
        if context:
            query_type = context.get("query_type")
            if query_type in ["verification", "navigational"]:
                # High-precision queries benefit from cross-encoder
                candidates_count = min(
                    self.reranking_config.top_n_for_cross_encoder,
                    len(documents)
                )
                return {
                    "use_cross_encoder": True,
                    "reason": "high_precision_query",
                    "query_type": query_type,
                    "candidates_count": candidates_count
                }

        # Default: skip cross-encoder
        return {
            "use_cross_encoder": False,
            "reason": "sufficient_bi_encoder_confidence",
            "candidates_count": 0
        }

    def _finalize_ranking(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Finalize document ranking and clean up metadata"""

        # Determine final score field
        final_docs = []
        for doc in documents:
            doc_copy = doc.copy()

            # Choose best available score
            if "combined_rerank_score" in doc:
                doc_copy["final_rerank_score"] = doc["combined_rerank_score"]
                doc_copy["reranking_method"] = "cross_encoder"
            elif "bi_encoder_score" in doc:
                doc_copy["final_rerank_score"] = doc["bi_encoder_score"]
                doc_copy["reranking_method"] = "bi_encoder"
            else:
                doc_copy["final_rerank_score"] = doc.get("score", 0.5)
                doc_copy["reranking_method"] = "original"

            final_docs.append(doc_copy)

        # Final sort
        final_docs.sort(key=lambda x: x["final_rerank_score"], reverse=True)

        # Add rank positions
        for i, doc in enumerate(final_docs):
            doc["rerank_position"] = i + 1

        return final_docs

    def get_reranking_stats(self) -> Dict[str, Any]:
        """Get reranking statistics"""
        if self.stats["total_queries"] == 0:
            return {"message": "No queries processed yet"}

        cross_encoder_usage_rate = self.stats["cross_encoder_used"] / self.stats["total_queries"]

        return {
            "total_queries": self.stats["total_queries"],
            "bi_encoder_only": self.stats["bi_encoder_only"],
            "cross_encoder_used": self.stats["cross_encoder_used"],
            "cross_encoder_usage_rate": cross_encoder_usage_rate,
            "avg_bi_encoder_time": self.stats["avg_bi_encoder_time"],
            "avg_cross_encoder_time": self.stats["avg_cross_encoder_time"],
            "config": {
                "top_n_for_cross_encoder": self.reranking_config.top_n_for_cross_encoder,
                "uncertainty_threshold": self.reranking_config.uncertainty_threshold
            }
        }


def create_gated_reranker(config: Dict[str, Any]) -> GatedReranker:
    """Factory function for gated reranker"""
    return GatedReranker(config)


# Usage example
if __name__ == "__main__":
    config = {
        "reranking": {
            "top_n_for_cross_encoder": 30,
            "uncertainty_threshold": 0.3,
            "bi_encoder_enabled": True,
            "cross_encoder_enabled": True
        },
        "profile": "thorough"
    }

    reranker = GatedReranker(config)

    # Test documents
    test_docs = [
        {"content": "COVID-19 vaccine effectiveness in clinical trials", "score": 0.8},
        {"content": "Vaccine development and approval process", "score": 0.7},
        {"content": "Side effects of mRNA vaccines", "score": 0.6},
        {"content": "Public health vaccination campaigns", "score": 0.5}
    ]

    result = reranker.rerank("COVID vaccine effectiveness", test_docs)

    print(f"Reranked {result.reranked_count} documents in {result.total_time:.2f}s")
    print(f"Stages: {list(result.stage_times.keys())}")

    for i, doc in enumerate(result.documents[:3]):
        score = doc.get("final_rerank_score", 0)
        method = doc.get("reranking_method", "unknown")
        print(f"{i+1}. Score: {score:.3f} ({method}) - {doc['content'][:50]}...")

    stats = reranker.get_reranking_stats()
    print(f"\nReranking stats: {stats}")
