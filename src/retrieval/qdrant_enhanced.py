#!/usr/bin/env python3
"""Qdrant integration s per-collection ef_search a advanced deduplication
FÁZE 2: Optimalizované vektorové vyhledávání

Author: Senior Python/MLOps Agent
"""

import asyncio
from dataclasses import dataclass
import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class QdrantConfig:
    """Konfigurace pro Qdrant collection"""

    collection_name: str
    ef_search: int
    vector_size: int
    distance_metric: str
    hnsw_config: dict[str, Any]
    quantization: dict[str, Any] | None = None


@dataclass
class DeduplicationResult:
    """Výsledek deduplication procesu"""

    deduplicated_docs: list[dict[str, Any]]
    duplicate_groups: list[list[str]]
    similarity_matrix: dict[str, dict[str, float]]
    stats: dict[str, Any]


class EnhancedDeduplicator:
    """Advanced deduplication s minhash a cosine similarity"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.dedup_config = config.get("retrieval", {}).get("deduplication", {})

        # Deduplication parameters
        self.similarity_threshold = self.dedup_config.get("similarity_threshold", 0.85)
        self.minhash_num_perm = self.dedup_config.get("minhash_num_perm", 128)
        self.content_similarity_weight = self.dedup_config.get("content_similarity_weight", 0.7)
        self.metadata_similarity_weight = self.dedup_config.get("metadata_similarity_weight", 0.3)

        # Content processing
        self.min_content_length = self.dedup_config.get("min_content_length", 100)
        self.shingle_size = self.dedup_config.get("shingle_size", 3)

        # Audit logging
        self.audit_enabled = self.dedup_config.get("audit_enabled", True)
        self.merge_mappings = {}  # Track what was merged

    async def deduplicate_documents(self, documents: list[dict[str, Any]]) -> DeduplicationResult:
        """Provede advanced deduplication s audit logging

        Args:
            documents: List dokumentů k deduplikaci

        Returns:
            DeduplicationResult s deduplikovanými dokumenty

        """
        start_time = time.time()

        if len(documents) <= 1:
            return DeduplicationResult(
                deduplicated_docs=documents,
                duplicate_groups=[],
                similarity_matrix={},
                stats={
                    "total_input": len(documents),
                    "duplicates_removed": 0,
                    "processing_time": 0.0,
                },
            )

        logger.info(f"Starting deduplication for {len(documents)} documents")

        # Step 1: Generate minhash signatures
        doc_signatures = await self._generate_minhash_signatures(documents)

        # Step 2: Find similarity clusters
        similarity_matrix = await self._calculate_similarity_matrix(doc_signatures, documents)

        # Step 3: Group duplicates
        duplicate_groups = await self._find_duplicate_groups(similarity_matrix)

        # Step 4: Select representatives from each group
        deduplicated_docs = await self._select_representatives(documents, duplicate_groups)

        processing_time = time.time() - start_time

        stats = {
            "total_input": len(documents),
            "duplicates_removed": len(documents) - len(deduplicated_docs),
            "duplicate_groups": len(duplicate_groups),
            "processing_time": processing_time,
            "similarity_threshold": self.similarity_threshold,
            "minhash_perms": self.minhash_num_perm,
        }

        # Audit logging
        if self.audit_enabled:
            await self._log_deduplication_audit(
                documents, deduplicated_docs, duplicate_groups, stats
            )

        logger.info(
            f"Deduplication completed: {len(documents)} → {len(deduplicated_docs)} docs ({stats['duplicates_removed']} removed)"
        )

        return DeduplicationResult(
            deduplicated_docs=deduplicated_docs,
            duplicate_groups=duplicate_groups,
            similarity_matrix=similarity_matrix,
            stats=stats,
        )

    async def _generate_minhash_signatures(
        self, documents: list[dict[str, Any]]
    ) -> dict[str, list[int]]:
        """Generuje MinHash signatures pro dokumenty"""
        signatures = {}

        for doc in documents:
            doc_id = doc.get("id", str(hash(str(doc))))
            content = doc.get("content", "")

            if len(content) < self.min_content_length:
                # Use title + metadata for short content
                content = " ".join([doc.get("title", ""), str(doc.get("metadata", {}))])

            # Generate shingles
            shingles = self._generate_shingles(content)

            # Calculate MinHash signature
            signature = self._calculate_minhash(shingles)
            signatures[doc_id] = signature

        return signatures

    def _generate_shingles(self, text: str) -> set[str]:
        """Generuje k-shingles z textu"""
        text = text.lower().strip()
        words = text.split()

        shingles = set()
        for i in range(len(words) - self.shingle_size + 1):
            shingle = " ".join(words[i : i + self.shingle_size])
            shingles.add(shingle)

        return shingles

    def _calculate_minhash(self, shingles: set[str]) -> list[int]:
        """Výpočet MinHash signature"""
        # Simple MinHash implementation
        signature = []

        for i in range(self.minhash_num_perm):
            min_hash = float("inf")

            for shingle in shingles:
                # Create hash with permutation
                hash_input = f"{shingle}_{i}"
                hash_val = hash(hash_input) % (2**32)
                min_hash = min(min_hash, hash_val)

            signature.append(int(min_hash) if min_hash != float("inf") else 0)

        return signature

    async def _calculate_similarity_matrix(
        self, signatures: dict[str, list[int]], documents: list[dict[str, Any]]
    ) -> dict[str, dict[str, float]]:
        """Výpočet similarity matrix"""
        doc_ids = list(signatures.keys())
        similarity_matrix = {}

        for i, doc_id1 in enumerate(doc_ids):
            similarity_matrix[doc_id1] = {}

            for j, doc_id2 in enumerate(doc_ids):
                if i <= j:
                    if i == j:
                        similarity_matrix[doc_id1][doc_id2] = 1.0
                    else:
                        # Jaccard similarity from MinHash
                        minhash_sim = self._jaccard_similarity_from_minhash(
                            signatures[doc_id1], signatures[doc_id2]
                        )

                        # Metadata similarity
                        doc1 = next(
                            d for d in documents if d.get("id", str(hash(str(d)))) == doc_id1
                        )
                        doc2 = next(
                            d for d in documents if d.get("id", str(hash(str(d)))) == doc_id2
                        )

                        metadata_sim = self._calculate_metadata_similarity(doc1, doc2)

                        # Combined similarity
                        combined_sim = (
                            self.content_similarity_weight * minhash_sim
                            + self.metadata_similarity_weight * metadata_sim
                        )

                        similarity_matrix[doc_id1][doc_id2] = combined_sim
                        similarity_matrix[doc_id2] = (
                            similarity_matrix[doc_id2] if doc_id2 in similarity_matrix else {}
                        )
                        similarity_matrix[doc_id2][doc_id1] = combined_sim

        return similarity_matrix

    def _jaccard_similarity_from_minhash(self, sig1: list[int], sig2: list[int]) -> float:
        """Výpočet Jaccard similarity z MinHash signatures"""
        if len(sig1) != len(sig2):
            return 0.0

        matches = sum(1 for a, b in zip(sig1, sig2, strict=False) if a == b)
        return matches / len(sig1)

    def _calculate_metadata_similarity(self, doc1: dict[str, Any], doc2: dict[str, Any]) -> float:
        """Výpočet metadata similarity"""
        metadata1 = doc1.get("metadata", {})
        metadata2 = doc2.get("metadata", {})

        similarities = []

        # Check key metadata fields
        metadata_fields = ["domain", "author", "publication_year", "source_type", "language"]

        for field in metadata_fields:
            val1 = metadata1.get(field)
            val2 = metadata2.get(field)

            if val1 and val2:
                if val1 == val2:
                    similarities.append(1.0)
                else:
                    similarities.append(0.0)

        # Title similarity
        title1 = doc1.get("title", "").lower()
        title2 = doc2.get("title", "").lower()

        if title1 and title2:
            title_sim = self._string_similarity(title1, title2)
            similarities.append(title_sim)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _string_similarity(self, str1: str, str2: str) -> float:
        """Simple string similarity"""
        words1 = set(str1.split())
        words2 = set(str2.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union

    async def _find_duplicate_groups(
        self, similarity_matrix: dict[str, dict[str, float]]
    ) -> list[list[str]]:
        """Najde skupiny duplicitních dokumentů"""
        doc_ids = list(similarity_matrix.keys())
        visited = set()
        duplicate_groups = []

        for doc_id in doc_ids:
            if doc_id in visited:
                continue

            # Find all similar documents
            group = [doc_id]
            visited.add(doc_id)

            for other_id in doc_ids:
                if other_id != doc_id and other_id not in visited:
                    similarity = similarity_matrix.get(doc_id, {}).get(other_id, 0.0)

                    if similarity >= self.similarity_threshold:
                        group.append(other_id)
                        visited.add(other_id)

            # Only add groups with duplicates
            if len(group) > 1:
                duplicate_groups.append(group)

        return duplicate_groups

    async def _select_representatives(
        self, documents: list[dict[str, Any]], duplicate_groups: list[list[str]]
    ) -> list[dict[str, Any]]:
        """Vybere reprezentanty z duplicate groups"""
        doc_lookup = {}
        for doc in documents:
            doc_id = doc.get("id", str(hash(str(doc))))
            doc_lookup[doc_id] = doc

        # Track which documents to exclude
        excluded_ids = set()

        for group in duplicate_groups:
            # Select best representative from group
            best_doc = None
            best_score = -1

            for doc_id in group:
                doc = doc_lookup.get(doc_id)
                if not doc:
                    continue

                # Score based on content length, metadata completeness, source quality
                score = self._calculate_document_quality_score(doc)

                if score > best_score:
                    best_score = score
                    best_doc = doc

            # Exclude all others from this group
            for doc_id in group:
                if doc_lookup.get(doc_id) != best_doc:
                    excluded_ids.add(doc_id)

                    # Log merge mapping for audit
                    if best_doc:
                        self.merge_mappings[doc_id] = best_doc.get("id", str(hash(str(best_doc))))

        # Return documents not excluded
        result = []
        for doc in documents:
            doc_id = doc.get("id", str(hash(str(doc))))
            if doc_id not in excluded_ids:
                result.append(doc)

        return result

    def _calculate_document_quality_score(self, doc: dict[str, Any]) -> float:
        """Výpočet quality score pro document selection"""
        score = 0.0

        # Content length (longer is generally better)
        content_length = len(doc.get("content", ""))
        score += min(content_length / 1000, 1.0) * 0.3

        # Metadata completeness
        metadata = doc.get("metadata", {})
        important_fields = ["title", "author", "published_date", "source_type"]
        completeness = sum(1 for field in important_fields if metadata.get(field)) / len(
            important_fields
        )
        score += completeness * 0.3

        # Source authority
        authority_score = {
            "academic": 1.0,
            "government": 0.9,
            "news_tier1": 0.8,
            "news_tier2": 0.6,
            "blog": 0.4,
            "unknown": 0.5,
        }.get(metadata.get("authority_type", "unknown"), 0.5)
        score += authority_score * 0.2

        # Existing relevance score
        relevance = doc.get("score", doc.get("rrf_final_score", 0.0))
        score += min(relevance, 1.0) * 0.2

        return score

    async def _log_deduplication_audit(
        self,
        original_docs: list[dict[str, Any]],
        deduplicated_docs: list[dict[str, Any]],
        duplicate_groups: list[list[str]],
        stats: dict[str, Any],
    ):
        """Audit logging pro deduplication"""
        audit_data = {
            "timestamp": time.time(),
            "operation": "deduplication",
            "input_count": len(original_docs),
            "output_count": len(deduplicated_docs),
            "removed_count": stats["duplicates_removed"],
            "duplicate_groups": duplicate_groups,
            "merge_mappings": self.merge_mappings,
            "config": {
                "similarity_threshold": self.similarity_threshold,
                "minhash_num_perm": self.minhash_num_perm,
                "content_weight": self.content_similarity_weight,
                "metadata_weight": self.metadata_similarity_weight,
            },
            "stats": stats,
        }

        # In real implementation, would save to audit log file
        logger.debug(f"Deduplication audit: {json.dumps(audit_data, indent=2)}")


class QdrantVectorStore:
    """Qdrant integration s per-collection ef_search optimization"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.qdrant_config = config.get("retrieval", {}).get("qdrant", {})

        # Connection settings
        self.host = self.qdrant_config.get("host", "localhost")
        self.port = self.qdrant_config.get("port", 6333)
        self.api_key = self.qdrant_config.get("api_key")

        # Per-collection configurations
        self.collection_configs = self._load_collection_configs()

        # Performance tracking
        self.performance_stats = {
            "queries": 0,
            "total_latency": 0.0,
            "p50_latency": [],
            "p95_latency": [],
            "recall_at_k": {},
        }

        # Mock client - real implementation would use qdrant-client
        self.client = None

    def _load_collection_configs(self) -> dict[str, QdrantConfig]:
        """Načte per-collection konfigurace"""
        configs = {}

        collections_config = self.qdrant_config.get("collections", {})

        for collection_name, collection_settings in collections_config.items():
            configs[collection_name] = QdrantConfig(
                collection_name=collection_name,
                ef_search=collection_settings.get("ef_search", 128),
                vector_size=collection_settings.get("vector_size", 384),
                distance_metric=collection_settings.get("distance", "Cosine"),
                hnsw_config=collection_settings.get("hnsw", {"m": 16, "ef_construct": 100}),
                quantization=collection_settings.get("quantization"),
            )

        # Default config if none specified
        if not configs:
            configs["default"] = QdrantConfig(
                collection_name="default",
                ef_search=128,
                vector_size=384,
                distance_metric="Cosine",
                hnsw_config={"m": 16, "ef_construct": 100},
            )

        return configs

    async def initialize(self):
        """Inicializace Qdrant connection"""
        logger.info("Initializing Qdrant vector store...")

        # Mock initialization - real implementation would connect to Qdrant
        await asyncio.sleep(0.1)
        self.client = "mock_qdrant_client"

        logger.info(f"✅ Qdrant initialized with {len(self.collection_configs)} collections")

    async def search_vectors(
        self,
        query_vector: list[float],
        collection_name: str = "default",
        top_k: int = 50,
        filter_conditions: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Vyhledá vectors s optimalizovaným ef_search

        Args:
            query_vector: Query embedding vector
            collection_name: Target collection
            top_k: Number of results to return
            filter_conditions: Optional metadata filters

        Returns:
            Tuple[results, search_metadata]

        """
        start_time = time.time()

        # Get collection config
        collection_config = self.collection_configs.get(
            collection_name, self.collection_configs["default"]
        )

        # Optimize ef_search based on top_k
        optimal_ef_search = max(collection_config.ef_search, top_k * 2)

        # Mock search - real implementation would query Qdrant
        await asyncio.sleep(0.1)  # Simulate search time

        # Generate mock results
        results = []
        for i in range(min(top_k, 20)):  # Mock up to 20 results
            score = 0.9 - (i * 0.03)  # Decreasing relevance

            results.append(
                {
                    "id": f"{collection_name}_doc_{i}",
                    "content": f"Mock document {i} from collection {collection_name}",
                    "score": score,
                    "metadata": {
                        "collection": collection_name,
                        "source_type": "vector_search",
                        "ef_search_used": optimal_ef_search,
                    },
                }
            )

        search_time = time.time() - start_time

        # Update performance stats
        self._update_performance_stats(search_time, len(results), top_k)

        search_metadata = {
            "collection": collection_name,
            "ef_search_used": optimal_ef_search,
            "search_time": search_time,
            "results_count": len(results),
            "top_k_requested": top_k,
            "filter_applied": filter_conditions is not None,
            "vector_dim": len(query_vector),
        }

        logger.debug(
            f"Vector search completed: {len(results)} results in {search_time:.3f}s (ef_search={optimal_ef_search})"
        )

        return results, search_metadata

    def _update_performance_stats(self, latency: float, results_count: int, top_k: int):
        """Aktualizace performance statistik"""
        self.performance_stats["queries"] += 1
        self.performance_stats["total_latency"] += latency

        # Track latencies for percentile calculation
        self.performance_stats["p50_latency"].append(latency)
        self.performance_stats["p95_latency"].append(latency)

        # Keep only last 1000 measurements
        for key in ["p50_latency", "p95_latency"]:
            if len(self.performance_stats[key]) > 1000:
                self.performance_stats[key] = self.performance_stats[key][-1000:]

        # Track recall@k if we have ground truth
        if top_k not in self.performance_stats["recall_at_k"]:
            self.performance_stats["recall_at_k"][top_k] = []

    def get_performance_stats(self) -> dict[str, Any]:
        """Vrací performance statistiky"""
        total_queries = self.performance_stats["queries"]

        if total_queries == 0:
            return {"no_queries": True}

        # Calculate percentiles
        latencies = sorted(self.performance_stats["p50_latency"])
        p50_idx = int(len(latencies) * 0.5)
        p95_idx = int(len(latencies) * 0.95)

        stats = {
            "total_queries": total_queries,
            "avg_latency": self.performance_stats["total_latency"] / total_queries,
            "p50_latency": latencies[p50_idx] if latencies else 0.0,
            "p95_latency": latencies[p95_idx] if latencies else 0.0,
            "collections_configured": len(self.collection_configs),
            "collection_configs": {
                name: {
                    "ef_search": config.ef_search,
                    "vector_size": config.vector_size,
                    "distance_metric": config.distance_metric,
                }
                for name, config in self.collection_configs.items()
            },
        }

        return stats

    async def close(self):
        """Zavře Qdrant connection"""
        if self.client:
            # Real implementation would close connection
            pass
        logger.info("Qdrant connection closed")
