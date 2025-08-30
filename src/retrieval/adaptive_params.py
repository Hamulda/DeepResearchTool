#!/usr/bin/env python3
"""
Adaptive Qdrant Parameters
Per-query SearchParams optimization based on query type and profile

Author: Senior Python/MLOps Agent
"""

from typing import Dict, Any, Optional, Tuple
from enum import Enum
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Query classification for parameter optimization"""
    EXPLORATORY = "exploratory"  # Broad discovery, needs high recall
    NAVIGATIONAL = "navigational"  # Specific fact finding, needs precision
    VERIFICATION = "verification"  # Cross-checking claims, needs exhaustive search
    TEMPORAL = "temporal"  # Time-sensitive queries, needs recency boost


@dataclass
class SearchParams:
    """Qdrant search parameters with adaptive optimization"""
    hnsw_ef: Optional[int] = None
    exact: bool = False
    limit: int = 100
    score_threshold: Optional[float] = None
    with_payload: bool = True
    with_vectors: bool = False

    def to_qdrant_params(self) -> Dict[str, Any]:
        """Convert to Qdrant API parameters"""
        params = {
            "limit": self.limit,
            "with_payload": self.with_payload,
            "with_vectors": self.with_vectors
        }

        if self.score_threshold is not None:
            params["score_threshold"] = self.score_threshold

        if self.hnsw_ef is not None or self.exact:
            params["search_params"] = {}
            if self.hnsw_ef is not None:
                params["search_params"]["hnsw_ef"] = self.hnsw_ef
            if self.exact:
                params["search_params"]["exact"] = self.exact

        return params


class AdaptiveQdrantParams:
    """Adaptive parameter selection for Qdrant searches"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.retrieval_config = config.get("retrieval", {})
        self.qdrant_config = config.get("qdrant", {})

        # Default parameters by profile
        self.profile_defaults = {
            "quick": {
                "hnsw_ef": 64,
                "limit": 50,
                "exact_fallback_threshold": 0.3
            },
            "thorough": {
                "hnsw_ef": 256,
                "limit": 200,
                "exact_fallback_threshold": 0.5
            }
        }

        # Query type specific parameters
        self.query_type_params = {
            QueryType.EXPLORATORY: {
                "hnsw_ef_multiplier": 1.5,
                "limit_multiplier": 1.2,
                "score_threshold": 0.3
            },
            QueryType.NAVIGATIONAL: {
                "hnsw_ef_multiplier": 1.0,
                "limit_multiplier": 0.8,
                "score_threshold": 0.5
            },
            QueryType.VERIFICATION: {
                "hnsw_ef_multiplier": 2.0,
                "limit_multiplier": 1.5,
                "score_threshold": 0.2,
                "prefer_exact": True
            },
            QueryType.TEMPORAL: {
                "hnsw_ef_multiplier": 1.2,
                "limit_multiplier": 1.0,
                "score_threshold": 0.4,
                "recency_boost": True
            }
        }

        logger.info("Initialized adaptive Qdrant parameters")

    def classify_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryType:
        """Classify query type for parameter optimization"""
        query_lower = query.lower()

        # Temporal indicators
        temporal_keywords = ["latest", "recent", "current", "2024", "2023", "today", "now"]
        if any(keyword in query_lower for keyword in temporal_keywords):
            return QueryType.TEMPORAL

        # Verification indicators
        verification_keywords = ["verify", "check", "confirm", "validate", "true", "false", "accuracy"]
        if any(keyword in query_lower for keyword in verification_keywords):
            return QueryType.VERIFICATION

        # Navigational indicators (specific facts)
        navigational_keywords = ["what is", "who is", "when did", "where is", "how many"]
        if any(keyword in query_lower for keyword in navigational_keywords):
            return QueryType.NAVIGATIONAL

        # Default to exploratory for broad research
        return QueryType.EXPLORATORY

    def get_adaptive_params(
        self,
        query: str,
        profile: str = "quick",
        context: Optional[Dict[str, Any]] = None,
        iteration: int = 1
    ) -> SearchParams:
        """Get adaptive search parameters based on query and context"""

        # Get base parameters for profile
        base_params = self.profile_defaults.get(profile, self.profile_defaults["quick"])

        # Classify query type
        query_type = self.classify_query(query, context)
        type_params = self.query_type_params.get(query_type, {})

        # Calculate adaptive parameters
        hnsw_ef = int(base_params["hnsw_ef"] * type_params.get("hnsw_ef_multiplier", 1.0))
        limit = int(base_params["limit"] * type_params.get("limit_multiplier", 1.0))

        # Iteration-based adaptation (increase search space if needed)
        if iteration > 1:
            hnsw_ef = int(hnsw_ef * min(1.5, 1.0 + 0.2 * (iteration - 1)))
            limit = int(limit * min(1.3, 1.0 + 0.1 * (iteration - 1)))

        # Check for exact search preference
        exact = type_params.get("prefer_exact", False)
        if context and context.get("low_recall_detected"):
            exact = True

        # Score threshold
        score_threshold = type_params.get("score_threshold")

        params = SearchParams(
            hnsw_ef=hnsw_ef if not exact else None,
            exact=exact,
            limit=limit,
            score_threshold=score_threshold
        )

        logger.info(
            f"Adaptive params for {query_type.value} query (iteration {iteration}): "
            f"hnsw_ef={params.hnsw_ef}, exact={params.exact}, limit={params.limit}"
        )

        return params

    def should_fallback_to_exact(
        self,
        results: list,
        params: SearchParams,
        query_type: QueryType
    ) -> bool:
        """Determine if fallback to exact search is needed"""

        if params.exact:
            return False  # Already using exact search

        # Check result quality
        if not results:
            return True

        # Check score distribution
        scores = [r.get("score", 0) for r in results]
        if not scores:
            return True

        avg_score = sum(scores) / len(scores)
        max_score = max(scores)

        # Fallback criteria by query type
        fallback_thresholds = {
            QueryType.EXPLORATORY: 0.3,
            QueryType.NAVIGATIONAL: 0.5,
            QueryType.VERIFICATION: 0.6,
            QueryType.TEMPORAL: 0.4
        }

        threshold = fallback_thresholds.get(query_type, 0.4)

        return max_score < threshold or avg_score < threshold * 0.7

    def get_temporal_boost_params(self, recency_weight: float = 0.3) -> Dict[str, Any]:
        """Get parameters for temporal boosting"""
        return {
            "recency_weight": recency_weight,
            "time_decay_days": 365,  # 1 year decay
            "boost_recent_threshold": 30  # Boost docs newer than 30 days
        }


def create_adaptive_qdrant_params(config: Dict[str, Any]) -> AdaptiveQdrantParams:
    """Factory function for adaptive Qdrant parameters"""
    return AdaptiveQdrantParams(config)


# Usage example
if __name__ == "__main__":
    # Example configuration
    config = {
        "retrieval": {"hierarchical": {"enabled": True}},
        "qdrant": {"collection_name": "research_docs"}
    }

    adapter = AdaptiveQdrantParams(config)

    # Test different query types
    test_queries = [
        "COVID-19 vaccine effectiveness comparison",  # exploratory
        "What is the mortality rate of COVID-19?",   # navigational
        "Verify claims about vaccine side effects",   # verification
        "Latest COVID-19 research findings 2024"     # temporal
    ]

    for query in test_queries:
        params = adapter.get_adaptive_params(query, profile="thorough")
        print(f"Query: {query}")
        print(f"Params: {params}")
        print()
