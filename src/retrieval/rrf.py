#!/usr/bin/env python3
"""
Enhanced Reciprocal Rank Fusion (RRF) Implementation
RRF s per-source authority/recency priory a pokročilou deduplikací

Author: Senior Python/MLOps Agent
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import defaultdict
import math

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RankedResult:
    """Single ranked search result with enhanced metadata"""

    id: str
    content: str
    score: float
    rank: int
    source_connector: str
    metadata: Dict[str, Any]
    canonical_url: str = ""
    title: str = ""
    content_hash: str = ""
    # Enhanced fields for FÁZE 1
    authority_score: float = 0.0
    recency_score: float = 0.0
    publication_date: Optional[datetime] = None
    source_authority: float = 0.0


@dataclass
class FusionResult:
    """Enhanced result of RRF fusion"""

    results: List[RankedResult]
    fusion_scores: List[float]
    deduplication_stats: Dict[str, int]
    k_parameter: int
    original_lists_count: int
    # Enhanced metrics for FÁZE 1
    authority_distribution: Dict[str, float]
    recency_distribution: Dict[str, float]
    source_coverage: Dict[str, int]
    fusion_quality_score: float


@dataclass
class SourcePriors:
    """Per-source authority and recency priors"""

    source_name: str
    authority_weight: float
    recency_weight: float
    base_authority: float
    recency_decay_days: int = 365


class DocumentDeduplicator:
    """Cross-connector document deduplication"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dedup_config = config.get("deduplication", {})

        # Deduplication settings
        self.enabled = self.dedup_config.get("enabled", True)
        self.url_similarity_threshold = self.dedup_config.get("url_threshold", 0.8)
        self.content_similarity_threshold = self.dedup_config.get("content_threshold", 0.85)
        self.title_similarity_threshold = self.dedup_config.get("title_threshold", 0.9)

        # Duplicate handling strategy
        self.duplicate_strategy = self.dedup_config.get(
            "strategy", "merge_scores"
        )  # merge_scores, keep_best, penalize
        self.duplicate_penalty = self.dedup_config.get("penalty_factor", 0.8)

        self.logger = structlog.get_logger(__name__)

    def deduplicate_results(
        self, ranked_lists: List[List[RankedResult]]
    ) -> Tuple[List[List[RankedResult]], Dict[str, int]]:
        """Deduplicate across multiple ranked lists"""

        if not self.enabled:
            return ranked_lists, {"duplicates_removed": 0, "duplicates_merged": 0}

        # Create global hash map for all results
        global_hash_map = {}
        duplicate_groups = defaultdict(list)

        # First pass: identify duplicates across all lists
        for list_idx, result_list in enumerate(ranked_lists):
            for result_idx, result in enumerate(result_list):
                # Generate content hash if not present
                if not result.content_hash:
                    result.content_hash = self._generate_content_hash(result)

                # Check for duplicates
                duplicate_key = self._find_duplicate_key(result, global_hash_map)

                if duplicate_key:
                    duplicate_groups[duplicate_key].append((list_idx, result_idx, result))
                else:
                    # New unique result
                    global_hash_map[result.content_hash] = result
                    duplicate_groups[result.content_hash] = [(list_idx, result_idx, result)]

        # Second pass: handle duplicates according to strategy
        deduplicated_lists = [[] for _ in ranked_lists]
        stats = {"duplicates_removed": 0, "duplicates_merged": 0, "unique_results": 0}

        for group_key, group_results in duplicate_groups.items():
            if len(group_results) == 1:
                # Unique result
                list_idx, result_idx, result = group_results[0]
                deduplicated_lists[list_idx].append(result)
                stats["unique_results"] += 1
            else:
                # Handle duplicates
                resolved_result = self._resolve_duplicates(group_results)

                if resolved_result:
                    # Add to the list with the highest-scoring instance
                    best_list_idx = max(group_results, key=lambda x: x[2].score)[0]
                    deduplicated_lists[best_list_idx].append(resolved_result)

                    if self.duplicate_strategy == "merge_scores":
                        stats["duplicates_merged"] += len(group_results) - 1
                    else:
                        stats["duplicates_removed"] += len(group_results) - 1

        self.logger.info(f"Deduplication completed: {stats}")
        return deduplicated_lists, stats

    def _generate_content_hash(self, result: RankedResult) -> str:
        """Generate content hash for deduplication"""

        # Normalize URL
        normalized_url = self._normalize_url(result.canonical_url)

        # Normalize title
        normalized_title = self._normalize_text(result.title)

        # Normalize content snippet
        normalized_content = self._normalize_text(result.content[:500])  # First 500 chars

        # Create composite hash
        composite_string = f"{normalized_url}|{normalized_title}|{normalized_content}"
        return hashlib.md5(composite_string.encode()).hexdigest()

    def _normalize_url(self, url: str) -> str:
        """Normalize URL for comparison"""
        import re

        # Remove protocol, www, trailing slash, query parameters
        normalized = re.sub(r"^https?://", "", url.lower())
        normalized = re.sub(r"^www\.", "", normalized)
        normalized = re.sub(r"/+$", "", normalized)
        normalized = re.sub(r"\?.*$", "", normalized)
        normalized = re.sub(r"#.*$", "", normalized)

        return normalized

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        import re

        # Convert to lowercase, remove extra whitespace, punctuation
        normalized = re.sub(r"[^\w\s]", " ", text.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()

        return normalized

    def _find_duplicate_key(
        self, result: RankedResult, hash_map: Dict[str, RankedResult]
    ) -> Optional[str]:
        """Find if result is duplicate of existing result"""

        # Check exact hash match first
        if result.content_hash in hash_map:
            return result.content_hash

        # Check similarity-based duplicates
        for existing_hash, existing_result in hash_map.items():
            if self._are_duplicates(result, existing_result):
                return existing_hash

        return None

    def _are_duplicates(self, result1: RankedResult, result2: RankedResult) -> bool:
        """Check if two results are duplicates based on similarity thresholds"""

        # URL similarity
        url_similarity = self._calculate_url_similarity(
            result1.canonical_url, result2.canonical_url
        )
        if url_similarity > self.url_similarity_threshold:
            return True

        # Title similarity
        title_similarity = self._calculate_text_similarity(result1.title, result2.title)
        if title_similarity > self.title_similarity_threshold:
            return True

        # Content similarity
        content_similarity = self._calculate_text_similarity(result1.content, result2.content)
        if content_similarity > self.content_similarity_threshold:
            return True

        return False

    def _calculate_url_similarity(self, url1: str, url2: str) -> float:
        """Calculate URL similarity"""
        norm1 = self._normalize_url(url1)
        norm2 = self._normalize_url(url2)

        if norm1 == norm2:
            return 1.0

        # Calculate character-level similarity
        return self._calculate_text_similarity(norm1, norm2)

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using character n-grams"""

        def get_ngrams(text: str, n: int = 3) -> set:
            normalized = self._normalize_text(text)
            return set(normalized[i : i + n] for i in range(len(normalized) - n + 1))

        ngrams1 = get_ngrams(text1)
        ngrams2 = get_ngrams(text2)

        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)

        return intersection / union if union > 0 else 0.0

    def _resolve_duplicates(
        self, duplicate_group: List[Tuple[int, int, RankedResult]]
    ) -> Optional[RankedResult]:
        """Resolve duplicate group according to strategy"""

        if self.duplicate_strategy == "keep_best":
            # Keep the highest scoring result
            best_result = max(duplicate_group, key=lambda x: x[2].score)[2]
            return best_result

        elif self.duplicate_strategy == "merge_scores":
            # Merge scores from all duplicates
            base_result = duplicate_group[0][2]

            # Calculate merged score
            scores = [result.score for _, _, result in duplicate_group]
            merged_score = np.mean(scores) * 1.1  # Small boost for multiple confirmations

            # Create merged result
            merged_result = RankedResult(
                id=base_result.id,
                content=base_result.content,
                score=merged_score,
                rank=base_result.rank,
                source_connector=f"merged_{len(duplicate_group)}",
                metadata={
                    **base_result.metadata,
                    "merged_from": [result.source_connector for _, _, result in duplicate_group],
                    "original_scores": scores,
                },
                canonical_url=base_result.canonical_url,
                title=base_result.title,
                content_hash=base_result.content_hash,
            )

            return merged_result

        elif self.duplicate_strategy == "penalize":
            # Keep best but penalize for being duplicate
            best_result = max(duplicate_group, key=lambda x: x[2].score)[2]
            best_result.score *= self.duplicate_penalty

            return best_result

        return None


class RRFEngine:
    """Reciprocal Rank Fusion engine with parameter optimization"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rrf_config = config.get("rrf", {})

        # RRF parameters
        self.default_k = self.rrf_config.get("k", 40)
        self.profile_k_values = self.rrf_config.get("profile_k", {"quick": 40, "thorough": 60})

        # Components
        self.deduplicator = DocumentDeduplicator(config)

        # Performance tracking
        self.fusion_history = []

        self.logger = structlog.get_logger(__name__)

    def fuse_rankings(
        self,
        ranked_lists: List[List[RankedResult]],
        k: Optional[int] = None,
        profile: str = "quick",
    ) -> FusionResult:
        """Fuse multiple ranked lists using RRF"""

        if k is None:
            k = self.profile_k_values.get(profile, self.default_k)

        self.logger.info(f"Fusing {len(ranked_lists)} ranked lists with k={k}")

        # Step 1: Deduplicate across lists
        deduplicated_lists, dedup_stats = self.deduplicator.deduplicate_results(ranked_lists)

        # Step 2: Apply RRF formula
        fused_results = self._apply_rrf_formula(deduplicated_lists, k)

        # Step 3: Sort by fused score
        fused_results.sort(key=lambda x: x.score, reverse=True)

        # Update ranks
        for new_rank, result in enumerate(fused_results, 1):
            result.rank = new_rank

        # Enhanced metrics calculation
        authority_distribution, recency_distribution, source_coverage = (
            self._calculate_enhanced_metrics(fused_results)
        )

        fusion_quality_score = self._calculate_fusion_quality(fused_results, dedup_stats)

        fusion_result = FusionResult(
            results=fused_results,
            fusion_scores=[r.score for r in fused_results],
            deduplication_stats=dedup_stats,
            k_parameter=k,
            original_lists_count=len(ranked_lists),
            authority_distribution=authority_distribution,
            recency_distribution=recency_distribution,
            source_coverage=source_coverage,
            fusion_quality_score=fusion_quality_score,
        )

        # Track for analysis
        self.fusion_history.append(
            {
                "k": k,
                "profile": profile,
                "lists_count": len(ranked_lists),
                "final_count": len(fused_results),
                "dedup_stats": dedup_stats,
            }
        )

        return fusion_result

    def _apply_rrf_formula(
        self, ranked_lists: List[List[RankedResult]], k: int
    ) -> List[RankedResult]:
        """Apply RRF formula to combine rankings"""

        # Collect all unique results
        all_results = {}
        result_scores = defaultdict(float)

        for list_results in ranked_lists:
            for rank, result in enumerate(list_results, 1):
                # RRF formula: score = 1.0 / (k + rank)
                rrf_score = 1.0 / (k + rank)

                if result.id not in all_results:
                    all_results[result.id] = result

                result_scores[result.id] += rrf_score

        # Create fused results
        fused_results = []
        for result_id, result in all_results.items():
            fused_score = result_scores[result_id]

            # Create new result with fused score
            fused_result = RankedResult(
                id=result.id,
                content=result.content,
                score=fused_score,
                rank=0,  # Will be updated after sorting
                source_connector=result.source_connector,
                metadata={
                    **result.metadata,
                    "rrf_score": fused_score,
                    "original_score": result.score,
                },
                canonical_url=result.canonical_url,
                title=result.title,
                content_hash=result.content_hash,
            )

            fused_results.append(fused_result)

        return fused_results

    def optimize_k_parameter(
        self, test_queries: List[str], ground_truth: Dict[str, List[str]], retrieval_engine
    ) -> Dict[str, Any]:
        """Optimize k parameter using test queries and ground truth"""

        k_values = self.rrf_config.get("k_sweep_range", [20, 40, 60, 80])
        optimization_results = {}

        self.logger.info(f"Optimizing k parameter with values: {k_values}")

        for k in k_values:
            metrics = []

            for query in test_queries:
                # Get rankings from multiple sources
                rankings = []

                # This would call actual retrieval methods
                # For now, placeholder implementation
                vector_results = []  # await retrieval_engine.vector_search(query)
                sparse_results = []  # await retrieval_engine.sparse_search(query)

                if vector_results:
                    rankings.append(vector_results)
                if sparse_results:
                    rankings.append(sparse_results)

                if rankings:
                    # Fuse with current k
                    fusion_result = self.fuse_rankings(rankings, k=k)

                    # Calculate metrics if ground truth available
                    if query in ground_truth:
                        gt_docs = ground_truth[query]
                        query_metrics = self._calculate_metrics(fusion_result.results, gt_docs)
                        metrics.append(query_metrics)

            # Aggregate metrics for this k value
            if metrics:
                avg_metrics = {
                    "ndcg@10": np.mean([m.get("ndcg@10", 0) for m in metrics]),
                    "recall@10": np.mean([m.get("recall@10", 0) for m in metrics]),
                    "mrr": np.mean([m.get("mrr", 0) for m in metrics]),
                }
                optimization_results[k] = avg_metrics

        # Find best k value
        best_k = max(
            optimization_results.keys(), key=lambda k: optimization_results[k].get("ndcg@10", 0)
        )

        return {
            "best_k": best_k,
            "all_results": optimization_results,
            "recommendation": {
                "quick": min(k_values),  # Fast retrieval
                "thorough": best_k,  # Best quality
            },
        }

    def _calculate_metrics(
        self, results: List[RankedResult], ground_truth: List[str]
    ) -> Dict[str, float]:
        """Calculate retrieval metrics"""

        # Convert to IDs for comparison
        result_ids = [r.id for r in results[:10]]  # Top 10
        gt_set = set(ground_truth)

        # Recall@10
        relevant_retrieved = len([rid for rid in result_ids if rid in gt_set])
        recall_at_10 = relevant_retrieved / len(gt_set) if gt_set else 0

        # nDCG@10 (simplified)
        dcg = 0
        for i, result_id in enumerate(result_ids):
            if result_id in gt_set:
                dcg += 1 / np.log2(i + 2)  # +2 because log2(1) = 0

        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(gt_set), 10)))
        ndcg_at_10 = dcg / idcg if idcg > 0 else 0

        # MRR
        mrr = 0
        for i, result_id in enumerate(result_ids):
            if result_id in gt_set:
                mrr = 1 / (i + 1)
                break

        return {"recall@10": recall_at_10, "ndcg@10": ndcg_at_10, "mrr": mrr}

    def _calculate_enhanced_metrics(
        self, results: List[RankedResult]
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
        """Calculate enhanced metrics for authority and recency distribution"""

        authority_distribution = defaultdict(float)
        recency_distribution = defaultdict(float)
        source_coverage = defaultdict(int)

        now = datetime.now()

        for result in results:
            # Authority score contribution
            authority_distribution[result.source_connector] += result.authority_score

            # Recency score contribution (decayed)
            if result.publication_date:
                recency_age = (now - result.publication_date).days
                decay_factor = max(
                    0, 1 - recency_age / result.metadata.get("recency_decay_days", 365)
                )
                recency_distribution[result.source_connector] += result.recency_score * decay_factor

            # Source coverage
            source_coverage[result.source_connector] += 1

        # Normalize distributions
        total_authority = sum(authority_distribution.values())
        total_recency = sum(recency_distribution.values())

        if total_authority > 0:
            for source in authority_distribution:
                authority_distribution[source] /= total_authority

        if total_recency > 0:
            for source in recency_distribution:
                recency_distribution[source] /= total_recency

        return dict(authority_distribution), dict(recency_distribution), dict(source_coverage)

    def _calculate_fusion_quality(
        self, results: List[RankedResult], dedup_stats: Dict[str, int]
    ) -> float:
        """Calculate quality score of the fusion"""

        # Base quality from deduplication stats
        quality_score = dedup_stats.get("unique_results", 0) / max(
            1, dedup_stats.get("duplicates_removed", 1)
        )

        # Authority and recency factors
        authority_factor = np.mean([r.authority_score for r in results]) if results else 0
        recency_factor = np.mean([r.recency_score for r in results]) if results else 0

        # Combine factors into quality score
        quality_score += (authority_factor + recency_factor) * 0.5

        return quality_score

    def get_optimization_report(self) -> Dict[str, Any]:
        """Get RRF optimization analysis report"""

        if not self.fusion_history:
            return {"message": "No fusion history available"}

        # Analyze fusion performance
        k_performance = defaultdict(list)

        for entry in self.fusion_history:
            k = entry["k"]
            final_count = entry["final_count"]
            dedup_ratio = entry["dedup_stats"].get("duplicates_removed", 0) / max(1, final_count)

            k_performance[k].append({"final_count": final_count, "dedup_ratio": dedup_ratio})

        # Generate recommendations
        recommendations = {}
        for k, performances in k_performance.items():
            avg_final_count = np.mean([p["final_count"] for p in performances])
            avg_dedup_ratio = np.mean([p["dedup_ratio"] for p in performances])

            recommendations[k] = {
                "avg_results": avg_final_count,
                "avg_dedup_ratio": avg_dedup_ratio,
                "efficiency_score": avg_final_count * (1 - avg_dedup_ratio),
            }

        return {
            "k_performance": dict(k_performance),
            "recommendations": recommendations,
            "total_fusions": len(self.fusion_history),
        }


def create_rrf_engine(config: Dict[str, Any]) -> RRFEngine:
    """Factory function for RRF engine"""
    return RRFEngine(config)
