#!/usr/bin/env python3
"""LTR (Learning to Rank) Fusion Engine
LightGBM-based fusion with RRF fallback for robust ranking

Author: Senior Python/MLOps Agent
"""

from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import pickle
from typing import Any

import numpy as np

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available, falling back to RRF only")

from .rrf import RRFusion

logger = logging.getLogger(__name__)


@dataclass
class RankingFeatures:
    """Feature set for LTR ranking"""

    bm25_score: float = 0.0
    vector_distance: float = 0.0
    vector_score: float = 0.0
    domain_authority: float = 0.0
    recency_score: float = 0.0
    source_diversity: float = 0.0
    reference_depth: int = 0
    content_length: int = 0
    query_overlap: float = 0.0
    semantic_relevance: float = 0.0
    citation_count: int = 0
    source_reliability: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input"""
        return np.array(
            [
                self.bm25_score,
                self.vector_distance,
                self.vector_score,
                self.domain_authority,
                self.recency_score,
                self.source_diversity,
                self.reference_depth,
                self.content_length,
                self.query_overlap,
                self.semantic_relevance,
                self.citation_count,
                self.source_reliability,
            ]
        )

    @classmethod
    def feature_names(cls) -> list[str]:
        """Get feature names for model training"""
        return [
            "bm25_score",
            "vector_distance",
            "vector_score",
            "domain_authority",
            "recency_score",
            "source_diversity",
            "reference_depth",
            "content_length",
            "query_overlap",
            "semantic_relevance",
            "citation_count",
            "source_reliability",
        ]


class LTRFusion:
    """Learning to Rank fusion engine with LightGBM"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.fusion_config = config.get("fusion", {})
        self.ltr_config = self.fusion_config.get("ltr", {})

        # Model configuration
        self.model_path = Path(config.get("model_cache_dir", "models")) / "ltr_fusion_model.pkl"
        self.model_path.parent.mkdir(exist_ok=True)

        # LightGBM model
        self.model: lgb.Booster | None = None
        self.model_trained = False

        # RRF fallback
        self.rrf = RRFusion()
        self.use_ltr = LIGHTGBM_AVAILABLE and self.ltr_config.get("enabled", True)

        # A/B testing configuration
        self.ab_test_ratio = self.ltr_config.get("ab_test_ratio", 0.0)  # 0 = no A/B

        # Feature extractors
        self.domain_authority_cache = {}

        logger.info(f"LTR Fusion initialized (LTR enabled: {self.use_ltr})")

    def extract_features(
        self,
        document: dict[str, Any],
        query: str,
        rankings: list[dict[str, Any]],
        rank_position: int,
    ) -> RankingFeatures:
        """Extract ranking features from document and context"""
        features = RankingFeatures()

        # BM25 score (from ranking)
        features.bm25_score = document.get("bm25_score", 0.0)

        # Vector similarity
        features.vector_distance = document.get("vector_distance", 1.0)
        features.vector_score = max(0, 1.0 - features.vector_distance)

        # Content features
        content = document.get("content", "")
        features.content_length = len(content)
        features.query_overlap = self._calculate_query_overlap(query, content)

        # Source features
        source_url = document.get("source_url", "")
        features.domain_authority = self._get_domain_authority(source_url)
        features.source_reliability = self._get_source_reliability(source_url)

        # Temporal features
        features.recency_score = self._calculate_recency_score(document)

        # Diversity features
        features.source_diversity = self._calculate_source_diversity(document, rankings)

        # Reference features
        metadata = document.get("metadata", {})
        features.reference_depth = metadata.get("reference_count", 0)
        features.citation_count = metadata.get("citation_count", 0)

        # Semantic relevance (if available)
        features.semantic_relevance = document.get("semantic_score", 0.0)

        return features

    def _calculate_query_overlap(self, query: str, content: str) -> float:
        """Calculate query-content overlap ratio"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split()[:200])  # First 200 words

        if not query_words:
            return 0.0

        overlap = len(query_words.intersection(content_words))
        return overlap / len(query_words)

    def _get_domain_authority(self, url: str) -> float:
        """Get domain authority score (cached)"""
        if not url:
            return 0.0

        domain = self._extract_domain(url)

        if domain in self.domain_authority_cache:
            return self.domain_authority_cache[domain]

        # Simple domain authority heuristics
        authority_scores = {
            "arxiv.org": 0.9,
            "pubmed.ncbi.nlm.nih.gov": 0.95,
            "nature.com": 0.9,
            "science.org": 0.9,
            "who.int": 0.85,
            "cdc.gov": 0.85,
            "wikipedia.org": 0.7,
            "scholar.google.com": 0.8,
        }

        # TLD-based scoring
        tld_scores = {".edu": 0.8, ".gov": 0.85, ".org": 0.6, ".com": 0.5}

        score = authority_scores.get(domain, 0.5)

        # Boost by TLD if not in specific list
        if domain not in authority_scores:
            for tld, tld_score in tld_scores.items():
                if domain.endswith(tld):
                    score = max(score, tld_score)
                    break

        self.domain_authority_cache[domain] = score
        return score

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse

            return urlparse(url).netloc.lower()
        except:
            return ""

    def _get_source_reliability(self, url: str) -> float:
        """Get source reliability score"""
        domain = self._extract_domain(url)

        # High reliability sources
        high_reliability = {
            "pubmed.ncbi.nlm.nih.gov",
            "arxiv.org",
            "nature.com",
            "science.org",
            "who.int",
            "cdc.gov",
            "fda.gov",
        }

        if domain in high_reliability:
            return 0.9
        if domain.endswith((".edu", ".gov")):
            return 0.8
        if domain.endswith(".org"):
            return 0.6
        return 0.5

    def _calculate_recency_score(self, document: dict[str, Any]) -> float:
        """Calculate recency score based on publication date"""
        pub_date_str = document.get("metadata", {}).get("publication_date")

        if not pub_date_str:
            return 0.5  # Neutral if no date

        try:
            # Try multiple date formats
            for fmt in ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d"]:
                try:
                    pub_date = datetime.strptime(pub_date_str, fmt)
                    break
                except ValueError:
                    continue
            else:
                return 0.5  # Could not parse date

            days_old = (datetime.now() - pub_date).days

            # Exponential decay: score = exp(-days/365)
            # Recent (0-30 days): ~1.0-0.92
            # Medium (30-365 days): 0.92-0.37
            # Old (1+ years): <0.37
            return np.exp(-days_old / 365)

        except Exception:
            return 0.5

    def _calculate_source_diversity(
        self, document: dict[str, Any], rankings: list[dict[str, Any]]
    ) -> float:
        """Calculate source diversity bonus"""
        current_domain = self._extract_domain(document.get("source_url", ""))

        if not current_domain:
            return 0.5

        # Count domains in top results
        domains_seen = set()
        for rank_doc in rankings[:20]:  # Check top 20
            domain = self._extract_domain(rank_doc.get("source_url", ""))
            if domain:
                domains_seen.add(domain)

        # Diversity bonus: less common domains get higher scores
        domain_frequency = sum(
            1
            for rank_doc in rankings[:20]
            if self._extract_domain(rank_doc.get("source_url", "")) == current_domain
        )

        # Inverse frequency bonus
        diversity_score = 1.0 / (1.0 + domain_frequency - 1)
        return min(diversity_score, 1.0)

    def fuse_rankings(
        self, rankings: list[list[dict[str, Any]]], query: str, use_ab_test: bool = False
    ) -> list[dict[str, Any]]:
        """Fuse multiple rankings using LTR or RRF fallback"""
        if not rankings or not any(rankings):
            return []

        # A/B testing logic
        if use_ab_test and np.random.random() < self.ab_test_ratio:
            logger.info("A/B test: Using RRF for this query")
            return self.rrf.fuse_rankings(rankings)

        # Use LTR if available and trained
        if self.use_ltr and self.model_trained and self.model is not None:
            try:
                return self._ltr_fuse(rankings, query)
            except Exception as e:
                logger.error(f"LTR fusion failed: {e}, falling back to RRF")
                return self.rrf.fuse_rankings(rankings)
        else:
            # Fallback to RRF
            logger.info("Using RRF fusion (LTR not available/trained)")
            return self.rrf.fuse_rankings(rankings)

    def _ltr_fuse(self, rankings: list[list[dict[str, Any]]], query: str) -> list[dict[str, Any]]:
        """Perform LTR-based fusion"""
        # Merge all rankings
        all_docs = {}
        for ranking in rankings:
            for doc in ranking:
                doc_id = doc.get("id", str(hash(doc.get("content", ""))))
                if doc_id not in all_docs:
                    all_docs[doc_id] = doc

        # Extract features and predict
        scored_docs = []
        for doc_id, doc in all_docs.items():
            features = self.extract_features(doc, query, list(all_docs.values()), 0)
            feature_array = features.to_array().reshape(1, -1)

            # Predict relevance score
            relevance_score = self.model.predict(feature_array)[0]

            doc_copy = doc.copy()
            doc_copy["ltr_score"] = float(relevance_score)
            doc_copy["combined_score"] = float(relevance_score)

            scored_docs.append(doc_copy)

        # Sort by LTR score
        scored_docs.sort(key=lambda x: x["ltr_score"], reverse=True)

        logger.info(f"LTR fusion completed: {len(scored_docs)} documents ranked")
        return scored_docs

    def train_model(self, training_data: list[dict[str, Any]]) -> bool:
        """Train the LTR model on provided data"""
        if not LIGHTGBM_AVAILABLE:
            logger.error("Cannot train LTR model: LightGBM not available")
            return False

        try:
            # Prepare training data
            X, y, groups = self._prepare_training_data(training_data)

            if len(X) == 0:
                logger.error("No training data available")
                return False

            # LightGBM parameters for ranking
            params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": [5, 10, 20],
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
            }

            # Create dataset
            train_data = lgb.Dataset(X, label=y, group=groups)

            # Train model
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data],
                callbacks=[lgb.early_stopping(stopping_rounds=10)],
            )

            self.model_trained = True

            # Save model
            with open(self.model_path, "wb") as f:
                pickle.dump(self.model, f)

            logger.info(f"LTR model trained successfully and saved to {self.model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to train LTR model: {e}")
            return False

    def _prepare_training_data(
        self, training_data: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data for LightGBM ranking"""
        X, y, groups = [], [], []

        for query_data in training_data:
            query = query_data["query"]
            documents = query_data["documents"]
            relevance_scores = query_data["relevance_scores"]

            group_size = 0
            for doc, score in zip(documents, relevance_scores, strict=False):
                features = self.extract_features(doc, query, documents, group_size)
                X.append(features.to_array())
                y.append(score)
                group_size += 1

            groups.append(group_size)

        return np.array(X), np.array(y), np.array(groups)

    def load_model(self) -> bool:
        """Load trained LTR model from disk"""
        if not LIGHTGBM_AVAILABLE:
            return False

        try:
            if self.model_path.exists():
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                self.model_trained = True
                logger.info(f"LTR model loaded from {self.model_path}")
                return True
            logger.info("No saved LTR model found")
            return False
        except Exception as e:
            logger.error(f"Failed to load LTR model: {e}")
            return False

    def get_fusion_stats(self) -> dict[str, Any]:
        """Get fusion statistics and model info"""
        stats = {
            "ltr_available": LIGHTGBM_AVAILABLE,
            "ltr_enabled": self.use_ltr,
            "model_trained": self.model_trained,
            "ab_test_ratio": self.ab_test_ratio,
            "fallback_method": "RRF",
        }

        if self.model_trained and self.model is not None:
            stats["model_features"] = RankingFeatures.feature_names()
            stats["model_importance"] = self.model.feature_importance().tolist()

        return stats


def create_ltr_fusion(config: dict[str, Any]) -> LTRFusion:
    """Factory function for LTR fusion engine"""
    fusion = LTRFusion(config)
    fusion.load_model()  # Try to load existing model
    return fusion


# Usage example
if __name__ == "__main__":
    config = {
        "fusion": {"ltr": {"enabled": True, "ab_test_ratio": 0.1}},
        "model_cache_dir": "models",
    }

    fusion = LTRFusion(config)

    # Example rankings
    ranking1 = [
        {"id": "doc1", "content": "COVID vaccine effectiveness", "bm25_score": 0.8},
        {"id": "doc2", "content": "Vaccine side effects study", "bm25_score": 0.6},
    ]

    ranking2 = [
        {"id": "doc2", "content": "Vaccine side effects study", "vector_score": 0.9},
        {"id": "doc3", "content": "mRNA vaccine research", "vector_score": 0.7},
    ]

    result = fusion.fuse_rankings([ranking1, ranking2], "COVID vaccine effectiveness")
    print(f"Fused ranking: {len(result)} documents")

    stats = fusion.get_fusion_stats()
    print(f"Fusion stats: {stats}")
