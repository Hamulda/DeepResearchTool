#!/usr/bin/env python3
"""Reciprocal Rank Fusion (RRF) s authority/recency priory a MMR diversifikací
FÁZE 2: Advanced re-ranking a diversifikace

Author: Senior Python/MLOps Agent
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RRFResult:
    """Výsledek RRF fúze"""

    fused_results: list[dict[str, Any]]
    fusion_metadata: dict[str, Any]
    rrf_scores: dict[str, float]
    source_contributions: dict[str, float]


@dataclass
class MMRResult:
    """Výsledek MMR diversifikace"""

    diversified_results: list[dict[str, Any]]
    diversity_score: float
    removed_duplicates: int
    diversity_metadata: dict[str, Any]


class RRFWithPriors:
    """RRF s configurable priory pro authority/recency/source type"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.rrf_config = config.get("retrieval", {}).get("rrf", {})

        # RRF parameters
        self.k = self.rrf_config.get("k", 60)
        self.weights = self.rrf_config.get("weights", {"bm25": 0.3, "dense": 0.4, "hyde": 0.3})

        # Authority priors (per-source credibility)
        self.authority_priors = self.rrf_config.get(
            "authority_priors",
            {
                "academic": 1.2,
                "government": 1.1,
                "news_tier1": 1.0,
                "news_tier2": 0.9,
                "blog": 0.7,
                "social": 0.5,
                "unknown": 0.8,
            },
        )

        # Recency priors (time-based decay)
        self.recency_config = self.rrf_config.get(
            "recency",
            {
                "enabled": True,
                "half_life_days": 365,  # Score halves after 1 year
                "min_multiplier": 0.5,  # Minimum recency multiplier
            },
        )

        # Source type priors
        self.source_type_priors = self.rrf_config.get(
            "source_type_priors",
            {
                "primary_literature": 1.3,
                "review_article": 1.1,
                "news_article": 0.9,
                "blog_post": 0.7,
                "social_media": 0.5,
            },
        )

    async def fuse_results(self, ranked_lists: dict[str, list[dict[str, Any]]]) -> RRFResult:
        """Fúze multiple ranked lists pomocí RRF s priory

        Args:
            ranked_lists: Dict[source_name, List[documents]]

        Returns:
            RRFResult s fused výsledky

        """
        start_time = asyncio.get_event_loop().time()

        # Collect all unique documents
        all_docs = {}
        doc_rankings = {}

        for source_name, doc_list in ranked_lists.items():
            weight = self.weights.get(source_name, 1.0)

            for rank, doc in enumerate(doc_list):
                doc_id = doc.get("id", f"doc_{rank}_{source_name}")

                if doc_id not in all_docs:
                    all_docs[doc_id] = doc
                    doc_rankings[doc_id] = {}

                # Calculate RRF score for this source
                rrf_score = weight / (self.k + rank + 1)
                doc_rankings[doc_id][source_name] = {
                    "rank": rank,
                    "rrf_score": rrf_score,
                    "weight": weight,
                }

        # Apply priors and calculate final scores
        final_scores = {}
        prior_breakdowns = {}

        for doc_id, doc in all_docs.items():
            # Base RRF score (sum across sources)
            base_rrf = sum(
                ranking_info["rrf_score"] for ranking_info in doc_rankings[doc_id].values()
            )

            # Apply priors
            authority_prior = self._get_authority_prior(doc)
            recency_prior = self._get_recency_prior(doc)
            source_type_prior = self._get_source_type_prior(doc)

            # Combined prior
            combined_prior = authority_prior * recency_prior * source_type_prior

            final_scores[doc_id] = base_rrf * combined_prior
            prior_breakdowns[doc_id] = {
                "base_rrf": base_rrf,
                "authority_prior": authority_prior,
                "recency_prior": recency_prior,
                "source_type_prior": source_type_prior,
                "combined_prior": combined_prior,
                "final_score": final_scores[doc_id],
            }

        # Sort by final score
        sorted_doc_ids = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)
        fused_results = [all_docs[doc_id] for doc_id in sorted_doc_ids]

        # Add scores to documents
        for i, doc in enumerate(fused_results):
            doc_id = doc.get("id", f"doc_{i}")
            doc["rrf_final_score"] = final_scores.get(doc_id, 0.0)
            doc["rrf_rank"] = i + 1
            doc["prior_breakdown"] = prior_breakdowns.get(doc_id, {})

        # Calculate source contributions
        source_contributions = {}
        for source_name in ranked_lists:
            contribution = sum(
                doc_rankings.get(doc_id, {}).get(source_name, {}).get("rrf_score", 0.0)
                for doc_id in sorted_doc_ids[:20]  # Top 20 docs
            )
            source_contributions[source_name] = contribution

        fusion_metadata = {
            "fusion_time": asyncio.get_event_loop().time() - start_time,
            "total_unique_docs": len(all_docs),
            "source_count": len(ranked_lists),
            "rrf_k": self.k,
            "weights_used": self.weights,
            "priors_applied": {
                "authority": True,
                "recency": self.recency_config["enabled"],
                "source_type": True,
            },
        }

        logger.info(
            f"RRF fusion completed: {len(fused_results)} docs in {fusion_metadata['fusion_time']:.3f}s"
        )

        return RRFResult(
            fused_results=fused_results,
            fusion_metadata=fusion_metadata,
            rrf_scores={doc_id: score for doc_id, score in final_scores.items()},
            source_contributions=source_contributions,
        )

    def _get_authority_prior(self, doc: dict[str, Any]) -> float:
        """Výpočet authority prior based on source"""
        source_domain = doc.get("metadata", {}).get("domain", "unknown")
        source_type = doc.get("metadata", {}).get("authority_type", "unknown")

        return self.authority_priors.get(source_type, self.authority_priors["unknown"])

    def _get_recency_prior(self, doc: dict[str, Any]) -> float:
        """Výpočet recency prior based on document age"""
        if not self.recency_config["enabled"]:
            return 1.0

        pub_date_str = doc.get("metadata", {}).get("published_date")
        if not pub_date_str:
            return 0.8  # Default for unknown dates

        try:
            # Parse publication date
            pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
            if pub_date.tzinfo:
                pub_date = pub_date.replace(tzinfo=None)

            # Calculate age in days
            age_days = (datetime.now() - pub_date).days

            # Exponential decay with half-life
            half_life = self.recency_config["half_life_days"]
            decay_factor = math.exp(-0.693 * age_days / half_life)  # ln(2) ≈ 0.693

            # Apply minimum multiplier
            min_mult = self.recency_config["min_multiplier"]
            return max(decay_factor, min_mult)

        except (ValueError, TypeError) as e:
            logger.debug(f"Could not parse date {pub_date_str}: {e}")
            return 0.8

    def _get_source_type_prior(self, doc: dict[str, Any]) -> float:
        """Výpočet source type prior"""
        source_type = doc.get("metadata", {}).get("source_type", "unknown")
        return self.source_type_priors.get(source_type, 0.8)


class MMRDiversifier:
    """Maximum Marginal Relevance diversifikace"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.mmr_config = config.get("retrieval", {}).get("mmr", {})

        # MMR parameters
        self.lambda_param = self.mmr_config.get("lambda", 0.7)  # Relevance vs diversity tradeoff
        self.diversity_k = self.mmr_config.get("diversity_k", 20)  # How many to diversify
        self.similarity_threshold = self.mmr_config.get("similarity_threshold", 0.8)

        # Diversity features to consider
        self.diversity_features = self.mmr_config.get(
            "diversity_features",
            ["domain", "source_type", "author", "publication_year", "topic_keywords"],
        )

    async def diversify_results(self, ranked_results: list[dict[str, Any]]) -> MMRResult:
        """Aplikuje MMR diversifikaci na ranked results

        Args:
            ranked_results: List dokumentů seřazených podle relevance

        Returns:
            MMRResult s diversifikovanými výsledky

        """
        start_time = asyncio.get_event_loop().time()

        if len(ranked_results) <= self.diversity_k:
            # Not enough results to diversify
            return MMRResult(
                diversified_results=ranked_results,
                diversity_score=1.0,
                removed_duplicates=0,
                diversity_metadata={
                    "diversification_applied": False,
                    "reason": "insufficient_results",
                    "total_time": asyncio.get_event_loop().time() - start_time,
                },
            )

        # Step 1: Remove near-duplicates
        deduplicated_results, removed_count = await self._remove_near_duplicates(ranked_results)

        # Step 2: Apply MMR selection
        selected_results = []
        remaining_results = deduplicated_results.copy()

        # Always include the top result
        if remaining_results:
            selected_results.append(remaining_results.pop(0))

        # Select remaining results using MMR
        while remaining_results and len(selected_results) < self.diversity_k:
            best_doc = None
            best_mmr_score = -1
            best_idx = -1

            for idx, candidate in enumerate(remaining_results):
                # Relevance score (normalized)
                relevance_score = candidate.get("rrf_final_score", candidate.get("score", 0.0))

                # Diversity score (average dissimilarity to selected docs)
                diversity_score = self._calculate_diversity_score(candidate, selected_results)

                # MMR score
                mmr_score = (
                    self.lambda_param * relevance_score + (1 - self.lambda_param) * diversity_score
                )

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_doc = candidate
                    best_idx = idx

            if best_doc:
                selected_results.append(best_doc)
                remaining_results.pop(best_idx)
                best_doc["mmr_score"] = best_mmr_score

        # Add remaining results after diversified selection
        selected_results.extend(remaining_results)

        # Calculate diversity metrics
        diversity_score = self._calculate_overall_diversity(selected_results[: self.diversity_k])

        processing_time = asyncio.get_event_loop().time() - start_time

        diversity_metadata = {
            "diversification_applied": True,
            "lambda_param": self.lambda_param,
            "diversity_k": self.diversity_k,
            "deduplication_removed": removed_count,
            "diversity_features_used": self.diversity_features,
            "total_time": processing_time,
            "final_diversity_score": diversity_score,
        }

        logger.info(
            f"MMR diversification completed: {len(selected_results)} docs, diversity={diversity_score:.3f}, removed {removed_count} duplicates"
        )

        return MMRResult(
            diversified_results=selected_results,
            diversity_score=diversity_score,
            removed_duplicates=removed_count,
            diversity_metadata=diversity_metadata,
        )

    async def _remove_near_duplicates(
        self, results: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], int]:
        """Odstraní near-duplicate dokumenty"""
        deduplicated = []
        removed_count = 0
        seen_signatures = set()

        for doc in results:
            # Create content signature for deduplication
            signature = self._create_content_signature(doc)

            # Check for near-duplicates
            is_duplicate = False
            for seen_sig in seen_signatures:
                if (
                    self._calculate_signature_similarity(signature, seen_sig)
                    > self.similarity_threshold
                ):
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduplicated.append(doc)
                seen_signatures.add(signature)
            else:
                removed_count += 1

        return deduplicated, removed_count

    def _create_content_signature(self, doc: dict[str, Any]) -> str:
        """Vytvoří content signature pro deduplication"""
        content = doc.get("content", "")
        title = doc.get("title", "")

        # Simple signature based on first and last parts
        content_words = content.split()
        if len(content_words) > 10:
            signature_parts = content_words[:5] + content_words[-5:]
        else:
            signature_parts = content_words

        signature = " ".join([title] + signature_parts).lower()
        return signature

    def _calculate_signature_similarity(self, sig1: str, sig2: str) -> float:
        """Výpočet similarity mezi signatures"""
        words1 = set(sig1.split())
        words2 = set(sig2.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _calculate_diversity_score(
        self, candidate: dict[str, Any], selected_docs: list[dict[str, Any]]
    ) -> float:
        """Výpočet diversity score pro kandidáta vůči vybraným dokumentům"""
        if not selected_docs:
            return 1.0

        dissimilarities = []

        for selected_doc in selected_docs:
            dissimilarity = 0.0
            feature_count = 0

            for feature in self.diversity_features:
                candidate_value = candidate.get("metadata", {}).get(feature)
                selected_value = selected_doc.get("metadata", {}).get(feature)

                if candidate_value and selected_value:
                    if candidate_value != selected_value:
                        dissimilarity += 1.0
                    feature_count += 1

            if feature_count > 0:
                dissimilarities.append(dissimilarity / feature_count)

        # Average dissimilarity
        return sum(dissimilarities) / len(dissimilarities) if dissimilarities else 0.5

    def _calculate_overall_diversity(self, docs: list[dict[str, Any]]) -> float:
        """Výpočet overall diversity score pro sadu dokumentů"""
        if len(docs) <= 1:
            return 1.0

        total_pairs = 0
        diverse_pairs = 0

        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                total_pairs += 1

                # Check diversity across features
                diverse_features = 0
                total_features = 0

                for feature in self.diversity_features:
                    val1 = docs[i].get("metadata", {}).get(feature)
                    val2 = docs[j].get("metadata", {}).get(feature)

                    if val1 and val2:
                        total_features += 1
                        if val1 != val2:
                            diverse_features += 1

                if total_features > 0 and diverse_features / total_features >= 0.5:
                    diverse_pairs += 1

        return diverse_pairs / total_pairs if total_pairs > 0 else 0.0
