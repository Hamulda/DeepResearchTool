#!/usr/bin/env python3
"""Near-Duplicate Deduplication a Per-Collection Qdrant Optimalizace
MinHash/Cosine deduplication + collection-specific ef_search tuning

Author: Senior Python/MLOps Agent
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationConfig:
    """Konfigurace deduplikace"""

    enabled: bool = True
    similarity_threshold: float = 0.85  # Práh pro near-duplicates
    use_minhash: bool = True  # MinHash pro rychlou deduplikaci
    use_cosine: bool = True  # Cosine similarity pro přesnou deduplikaci
    content_field: str = "content"  # Pole pro porovnání obsahu

    # MinHash parameters
    minhash_permutations: int = 128
    shingle_size: int = 3  # N-gram size pro shingling

    # Cosine similarity parameters
    min_content_length: int = 50  # Minimální délka pro cosine similarity
    batch_size: int = 100  # Batch size pro embedding computation

    # Merge strategy
    merge_strategy: str = "best_score"  # "best_score", "first", "aggregate"
    keep_merge_info: bool = True  # Zachovej info o sloučených dokumentech


@dataclass
class QdrantCollectionConfig:
    """Konfigurace pro Qdrant kolekci"""

    name: str
    ef_search: int = 128  # Search parameter
    ef_construct: int = 200  # Construction parameter
    m_parameter: int = 16  # Number of bi-directional links

    # Performance monitoring
    target_latency_p95: float = 100.0  # ms
    target_recall_at_10: float = 0.9

    # Auto-tuning
    auto_tune: bool = True
    tune_interval_queries: int = 100
    tune_step_size: int = 16


class MinHashDeduplicator:
    """MinHash-based deduplikace pro rychlé near-duplicate detection"""

    def __init__(self, config: DeduplicationConfig):
        self.config = config
        self.permutations = self._generate_permutations()
        self.dedup_stats = {"total_processed": 0, "duplicates_found": 0, "avg_processing_time": 0.0}

    def _generate_permutations(self) -> list[tuple[int, int]]:
        """Generuje permutace pro MinHash"""
        # Velké prvočíslo pro hash functions
        large_prime = 2**31 - 1
        permutations = []

        np.random.seed(42)  # Reproducible permutations
        for i in range(self.config.minhash_permutations):
            a = np.random.randint(1, large_prime)
            b = np.random.randint(0, large_prime)
            permutations.append((a, b))

        return permutations

    def compute_minhash_signature(self, content: str) -> np.ndarray:
        """Vypočítá MinHash signaturu pro obsah"""
        # Vytvoř shingles (n-gramy)
        shingles = self._create_shingles(content)

        if not shingles:
            return np.full(self.config.minhash_permutations, np.inf)

        # Hash každý shingle
        hashed_shingles = [self._hash_shingle(shingle) for shingle in shingles]

        # Vypočítaj MinHash signaturu
        signature = []
        large_prime = 2**31 - 1

        for a, b in self.permutations:
            min_hash = min((a * h + b) % large_prime for h in hashed_shingles)
            signature.append(min_hash)

        return np.array(signature, dtype=np.int64)

    def _create_shingles(self, content: str) -> set[str]:
        """Vytvoří shingles z obsahu"""
        # Normalizuj text
        content = content.lower().strip()
        words = content.split()

        if len(words) < self.config.shingle_size:
            return {content}  # Celý obsah jako jeden shingle

        shingles = set()
        for i in range(len(words) - self.config.shingle_size + 1):
            shingle = " ".join(words[i : i + self.config.shingle_size])
            shingles.add(shingle)

        return shingles

    def _hash_shingle(self, shingle: str) -> int:
        """Hash jednotlivého shingle"""
        return int(hashlib.md5(shingle.encode()).hexdigest(), 16)

    def jaccard_similarity(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Vypočítá Jaccard similaritu mezi MinHash signaturami"""
        if len(sig1) != len(sig2):
            return 0.0

        matches = np.sum(sig1 == sig2)
        return matches / len(sig1)

    def find_near_duplicates(self, documents: list[dict[str, Any]]) -> list[list[int]]:
        """Najde near-duplicate skupiny pomocí MinHash"""
        start_time = time.time()

        if not self.config.enabled or not self.config.use_minhash:
            return []

        # Vypočítaj MinHash signatury
        signatures = []
        for i, doc in enumerate(documents):
            content = doc.get(self.config.content_field, "")
            signature = self.compute_minhash_signature(content)
            signatures.append((i, signature))

        # Najdi duplicate skupiny
        duplicate_groups = []
        processed = set()

        for i, (idx1, sig1) in enumerate(signatures):
            if idx1 in processed:
                continue

            group = [idx1]
            processed.add(idx1)

            for j, (idx2, sig2) in enumerate(signatures[i + 1 :], i + 1):
                if idx2 in processed:
                    continue

                similarity = self.jaccard_similarity(sig1, sig2)
                if similarity >= self.config.similarity_threshold:
                    group.append(idx2)
                    processed.add(idx2)

            if len(group) > 1:
                duplicate_groups.append(group)

        # Aktualizuj statistiky
        elapsed_time = time.time() - start_time
        self._update_stats(len(documents), len(duplicate_groups), elapsed_time)

        logger.info(f"MinHash deduplication: {len(duplicate_groups)} duplicate groups found")

        return duplicate_groups

    def _update_stats(self, total_docs: int, duplicates_found: int, elapsed_time: float):
        """Aktualizuje deduplikační statistiky"""
        self.dedup_stats["total_processed"] += total_docs
        self.dedup_stats["duplicates_found"] += duplicates_found

        # Exponential moving average
        alpha = 0.1
        if self.dedup_stats["avg_processing_time"] == 0:
            self.dedup_stats["avg_processing_time"] = elapsed_time
        else:
            self.dedup_stats["avg_processing_time"] = (
                alpha * elapsed_time + (1 - alpha) * self.dedup_stats["avg_processing_time"]
            )


class CosineDeduplicator:
    """Cosine similarity-based deduplikace pro přesné porovnání"""

    def __init__(self, config: DeduplicationConfig, embedding_model=None):
        self.config = config
        self.embedding_model = embedding_model
        self.dedup_stats = {
            "embeddings_computed": 0,
            "comparisons_made": 0,
            "avg_embedding_time": 0.0,
        }

    async def find_near_duplicates_cosine(
        self,
        documents: list[dict[str, Any]],
        candidate_pairs: list[tuple[int, int]] | None = None,
    ) -> list[list[int]]:
        """Najde near-duplicates pomocí cosine similarity
        candidate_pairs: Páry pro kontrolu (z MinHash pre-filtering)
        """
        if not self.config.enabled or not self.config.use_cosine:
            return []

        start_time = time.time()

        # Filtruj dokumenty podle minimální délky
        valid_docs = []
        doc_mapping = {}  # index in valid_docs -> original index

        for i, doc in enumerate(documents):
            content = doc.get(self.config.content_field, "")
            if len(content) >= self.config.min_content_length:
                doc_mapping[len(valid_docs)] = i
                valid_docs.append(doc)

        if len(valid_docs) < 2:
            return []

        # Vypočítaj embeddings
        embeddings = await self._compute_embeddings(valid_docs)

        if not embeddings:
            return []

        # Najdi duplicate skupiny
        if candidate_pairs:
            # Použij pre-filtered páry z MinHash
            duplicate_groups = self._find_duplicates_from_pairs(
                embeddings, candidate_pairs, doc_mapping
            )
        else:
            # Full pairwise comparison
            duplicate_groups = self._find_duplicates_full_comparison(embeddings, doc_mapping)

        elapsed_time = time.time() - start_time
        logger.info(
            f"Cosine deduplication completed in {elapsed_time:.2f}s: {len(duplicate_groups)} groups"
        )

        return duplicate_groups

    async def _compute_embeddings(self, documents: list[dict[str, Any]]) -> list[np.ndarray]:
        """Vypočítá embeddings pro dokumenty"""
        embeddings = []
        start_time = time.time()

        if not self.embedding_model:
            # Mock embeddings pro testování
            for doc in documents:
                content = doc.get(self.config.content_field, "")
                # Simple hash-based mock embedding
                hash_val = hash(content)
                embedding = np.random.RandomState(hash_val % (2**32)).normal(0, 1, 384)
                embeddings.append(embedding)
            return embeddings

        # Batch processing pro efektivitu
        batch_size = self.config.batch_size
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_contents = [doc.get(self.config.content_field, "") for doc in batch]

            try:
                batch_embeddings = await self.embedding_model.encode_batch(batch_contents)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.warning(f"Embedding batch {i//batch_size} failed: {e}")
                # Fallback na mock embeddings
                for content in batch_contents:
                    hash_val = hash(content)
                    embedding = np.random.RandomState(hash_val % (2**32)).normal(0, 1, 384)
                    embeddings.append(embedding)

        # Aktualizuj statistiky
        elapsed_time = time.time() - start_time
        self.dedup_stats["embeddings_computed"] += len(documents)

        alpha = 0.1
        if self.dedup_stats["avg_embedding_time"] == 0:
            self.dedup_stats["avg_embedding_time"] = elapsed_time
        else:
            self.dedup_stats["avg_embedding_time"] = (
                alpha * elapsed_time + (1 - alpha) * self.dedup_stats["avg_embedding_time"]
            )

        return embeddings

    def _find_duplicates_from_pairs(
        self,
        embeddings: list[np.ndarray],
        candidate_pairs: list[tuple[int, int]],
        doc_mapping: dict[int, int],
    ) -> list[list[int]]:
        """Najde duplicates z candidate pairs"""
        duplicate_groups = []
        processed = set()

        # Vytvoř mapping z original indices na valid doc indices
        reverse_mapping = {orig_idx: valid_idx for valid_idx, orig_idx in doc_mapping.items()}

        for orig_idx1, orig_idx2 in candidate_pairs:
            if orig_idx1 in processed or orig_idx2 in processed:
                continue

            # Převeď na valid doc indices
            valid_idx1 = reverse_mapping.get(orig_idx1)
            valid_idx2 = reverse_mapping.get(orig_idx2)

            if valid_idx1 is None or valid_idx2 is None:
                continue

            # Vypočítaj cosine similarity
            similarity = self._cosine_similarity(embeddings[valid_idx1], embeddings[valid_idx2])
            self.dedup_stats["comparisons_made"] += 1

            if similarity >= self.config.similarity_threshold:
                group = [orig_idx1, orig_idx2]
                duplicate_groups.append(group)
                processed.update(group)

        return duplicate_groups

    def _find_duplicates_full_comparison(
        self, embeddings: list[np.ndarray], doc_mapping: dict[int, int]
    ) -> list[list[int]]:
        """Najde duplicates pomocí full pairwise comparison"""
        duplicate_groups = []
        processed = set()

        for i in range(len(embeddings)):
            orig_idx_i = doc_mapping[i]
            if orig_idx_i in processed:
                continue

            group = [orig_idx_i]
            processed.add(orig_idx_i)

            for j in range(i + 1, len(embeddings)):
                orig_idx_j = doc_mapping[j]
                if orig_idx_j in processed:
                    continue

                similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                self.dedup_stats["comparisons_made"] += 1

                if similarity >= self.config.similarity_threshold:
                    group.append(orig_idx_j)
                    processed.add(orig_idx_j)

            if len(group) > 1:
                duplicate_groups.append(group)

        return duplicate_groups

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Vypočítá cosine similaritu"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)
        except Exception:
            return 0.0


class QdrantCollectionOptimizer:
    """Optimalizace per-collection ef_search parametrů pro Qdrant"""

    def __init__(self, config: dict[str, QdrantCollectionConfig]):
        self.collection_configs = config
        self.performance_history = defaultdict(list)
        self.optimization_stats = {
            "collections_tuned": 0,
            "total_queries_monitored": 0,
            "avg_improvement_percent": 0.0,
        }

    async def optimize_collection(
        self, collection_name: str, qdrant_client, test_queries: list[str] = None
    ) -> dict[str, Any]:
        """Optimalizuje ef_search pro konkrétní kolekci"""
        if collection_name not in self.collection_configs:
            logger.warning(f"Collection {collection_name} not configured for optimization")
            return {}

        config = self.collection_configs[collection_name]

        if not config.auto_tune:
            logger.info(f"Auto-tuning disabled for collection {collection_name}")
            return {"optimized": False, "reason": "auto_tune_disabled"}

        start_time = time.time()

        # Získej současný performance baseline
        baseline_metrics = await self._measure_performance(
            collection_name, qdrant_client, config.ef_search, test_queries
        )

        best_ef_search = config.ef_search
        best_metrics = baseline_metrics

        # Test různé ef_search hodnoty
        test_values = self._generate_test_values(config.ef_search)

        for ef_value in test_values:
            try:
                # Dočasně nastav ef_search
                await self._set_ef_search(qdrant_client, collection_name, ef_value)

                # Změř performance
                metrics = await self._measure_performance(
                    collection_name, qdrant_client, ef_value, test_queries
                )

                # Evaluuj improvement
                if self._is_better_performance(metrics, best_metrics, config):
                    best_ef_search = ef_value
                    best_metrics = metrics
                    logger.info(
                        f"Collection {collection_name}: ef_search {ef_value} shows improvement"
                    )

            except Exception as e:
                logger.warning(f"Failed to test ef_search {ef_value} for {collection_name}: {e}")

        # Nastav optimální hodnotu
        await self._set_ef_search(qdrant_client, collection_name, best_ef_search)
        config.ef_search = best_ef_search

        # Aktualizuj statistiky
        elapsed_time = time.time() - start_time
        optimization_result = {
            "collection": collection_name,
            "optimized": True,
            "old_ef_search": baseline_metrics.get("ef_search", config.ef_search),
            "new_ef_search": best_ef_search,
            "baseline_metrics": baseline_metrics,
            "optimized_metrics": best_metrics,
            "optimization_time_seconds": elapsed_time,
            "improvement_percent": self._calculate_improvement(baseline_metrics, best_metrics),
        }

        self.optimization_stats["collections_tuned"] += 1
        self._update_optimization_stats(optimization_result)

        logger.info(f"Collection {collection_name} optimized: ef_search {best_ef_search}")

        return optimization_result

    async def _measure_performance(
        self, collection_name: str, qdrant_client, ef_search: int, test_queries: list[str] = None
    ) -> dict[str, float]:
        """Změří performance pro danou ef_search hodnotu"""
        if not test_queries:
            # Generuj test queries
            test_queries = [
                "machine learning algorithms",
                "neural network architecture",
                "data processing pipeline",
                "optimization techniques",
                "research methodology",
            ]

        latencies = []
        recall_scores = []

        for query in test_queries[:5]:  # Limit na 5 testů
            try:
                start_time = time.time()

                # Mock search pro testování
                # V reálné implementaci by to byl skutečný Qdrant search
                await asyncio.sleep(0.01 + (ef_search / 10000))  # Simulace latency

                latency = (time.time() - start_time) * 1000  # ms
                latencies.append(latency)

                # Mock recall score
                recall = 0.85 + (ef_search / 1000) * 0.1  # Vyšší ef_search = lepší recall
                recall_scores.append(min(recall, 1.0))

            except Exception as e:
                logger.warning(f"Performance test failed for query '{query}': {e}")

        if not latencies:
            return {"error": "no_valid_measurements"}

        # Vypočítaj metriky
        p95_latency = np.percentile(latencies, 95) if latencies else float("inf")
        avg_recall = np.mean(recall_scores) if recall_scores else 0.0

        return {
            "ef_search": ef_search,
            "p95_latency_ms": p95_latency,
            "avg_recall": avg_recall,
            "measurement_count": len(latencies),
        }

    def _generate_test_values(self, current_ef: int) -> list[int]:
        """Generuje test hodnoty pro ef_search"""
        step = self.collection_configs.get("step_size", 16)

        test_values = []

        # Test nižší hodnoty
        for multiplier in [0.5, 0.75]:
            value = max(16, int(current_ef * multiplier))
            test_values.append(value)

        # Test vyšší hodnoty
        for multiplier in [1.25, 1.5, 2.0]:
            value = min(512, int(current_ef * multiplier))
            test_values.append(value)

        # Remove duplicates a seřaď
        test_values = sorted(list(set(test_values)))

        return test_values

    def _is_better_performance(
        self,
        new_metrics: dict[str, float],
        best_metrics: dict[str, float],
        config: QdrantCollectionConfig,
    ) -> bool:
        """Určí, zda nové metriky jsou lepší"""
        if "error" in new_metrics or "error" in best_metrics:
            return False

        new_latency = new_metrics.get("p95_latency_ms", float("inf"))
        best_latency = best_metrics.get("p95_latency_ms", float("inf"))

        new_recall = new_metrics.get("avg_recall", 0.0)
        best_recall = best_metrics.get("avg_recall", 0.0)

        # Kontrola constraints
        latency_ok = new_latency <= config.target_latency_p95
        recall_ok = new_recall >= config.target_recall_at_10

        if not (latency_ok and recall_ok):
            return False

        # Preference: nejdříve recall, pak latency
        if new_recall > best_recall + 0.01:  # Significant recall improvement
            return True
        if abs(new_recall - best_recall) <= 0.01:  # Similar recall
            return new_latency < best_latency

        return False

    async def _set_ef_search(self, qdrant_client, collection_name: str, ef_search: int):
        """Nastaví ef_search pro kolekci"""
        # Mock implementace - v reálném systému by to byl Qdrant API call
        logger.debug(f"Setting ef_search={ef_search} for collection {collection_name}")
        await asyncio.sleep(0.01)  # Simulate API call

    def _calculate_improvement(
        self, baseline: dict[str, float], optimized: dict[str, float]
    ) -> float:
        """Vypočítá improvement percentage"""
        if "error" in baseline or "error" in optimized:
            return 0.0

        baseline_recall = baseline.get("avg_recall", 0.0)
        optimized_recall = optimized.get("avg_recall", 0.0)

        if baseline_recall == 0:
            return 0.0

        improvement = ((optimized_recall - baseline_recall) / baseline_recall) * 100
        return improvement

    def _update_optimization_stats(self, result: dict[str, Any]):
        """Aktualizuje optimization statistiky"""
        improvement = result.get("improvement_percent", 0.0)

        # Exponential moving average
        alpha = 0.1
        if self.optimization_stats["avg_improvement_percent"] == 0:
            self.optimization_stats["avg_improvement_percent"] = improvement
        else:
            self.optimization_stats["avg_improvement_percent"] = (
                alpha * improvement
                + (1 - alpha) * self.optimization_stats["avg_improvement_percent"]
            )


class IntegratedDeduplication:
    """Integrovaný deduplikační systém kombinující MinHash a Cosine"""

    def __init__(self, config: DeduplicationConfig, embedding_model=None):
        self.config = config
        self.minhash_dedup = MinHashDeduplicator(config)
        self.cosine_dedup = CosineDeduplicator(config, embedding_model)

        self.merge_stats = {"total_merges": 0, "documents_merged": 0, "avg_group_size": 0.0}

    async def deduplicate_documents(
        self, documents: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Kompletní deduplikace s MinHash pre-filtering a Cosine validation
        """
        start_time = time.time()

        if not self.config.enabled or len(documents) < 2:
            return documents, {"deduplication_enabled": False}

        logger.info(f"Starting deduplication of {len(documents)} documents")

        # 1. MinHash pre-filtering
        minhash_groups = []
        if self.config.use_minhash:
            minhash_groups = self.minhash_dedup.find_near_duplicates(documents)
            logger.info(f"MinHash found {len(minhash_groups)} duplicate groups")

        # 2. Cosine similarity validation
        final_groups = []
        if self.config.use_cosine and minhash_groups:
            # Vytvoř candidate pairs z MinHash groups
            candidate_pairs = []
            for group in minhash_groups:
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        candidate_pairs.append((group[i], group[j]))

            cosine_groups = await self.cosine_dedup.find_near_duplicates_cosine(
                documents, candidate_pairs
            )
            final_groups = cosine_groups
            logger.info(f"Cosine validation resulted in {len(final_groups)} final groups")
        else:
            final_groups = minhash_groups

        # 3. Merge duplicate groups
        deduplicated_docs, merge_mapping = self._merge_duplicate_groups(documents, final_groups)

        # 4. Prepare metadata
        elapsed_time = time.time() - start_time
        dedup_metadata = {
            "deduplication_enabled": True,
            "original_count": len(documents),
            "deduplicated_count": len(deduplicated_docs),
            "duplicate_groups_found": len(final_groups),
            "documents_merged": len(documents) - len(deduplicated_docs),
            "merge_mapping": merge_mapping if self.config.keep_merge_info else {},
            "processing_time_seconds": elapsed_time,
            "minhash_stats": self.minhash_dedup.dedup_stats,
            "cosine_stats": self.cosine_dedup.dedup_stats,
        }

        # 5. Update stats
        self._update_merge_stats(final_groups)

        logger.info(
            f"Deduplication completed: {len(documents)} -> {len(deduplicated_docs)} documents"
        )

        return deduplicated_docs, dedup_metadata

    def _merge_duplicate_groups(
        self, documents: list[dict[str, Any]], duplicate_groups: list[list[int]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Merge duplicate dokumenty podle konfigurace"""
        if not duplicate_groups:
            return documents, {}

        merged_docs = []
        merge_mapping = {}
        processed_indices = set()

        # Process duplicate groups
        for group in duplicate_groups:
            if not group:
                continue

            # Merge documents in group
            merged_doc = self._merge_documents([documents[i] for i in group])
            merged_docs.append(merged_doc)

            # Track merge mapping
            if self.config.keep_merge_info:
                merge_mapping[len(merged_docs) - 1] = group

            processed_indices.update(group)

        # Add non-duplicate documents
        for i, doc in enumerate(documents):
            if i not in processed_indices:
                merged_docs.append(doc)

        return merged_docs, merge_mapping

    def _merge_documents(self, docs: list[dict[str, Any]]) -> dict[str, Any]:
        """Merge skupiny duplicate dokumentů"""
        if len(docs) == 1:
            return docs[0]

        if self.config.merge_strategy == "best_score":
            # Vezmi dokument s nejvyšším skóre
            best_doc = max(docs, key=lambda d: d.get("score", 0))
            merged = best_doc.copy()
        elif self.config.merge_strategy == "first":
            # Vezmi první dokument
            merged = docs[0].copy()
        else:  # aggregate
            # Agreguj informace z všech dokumentů
            merged = docs[0].copy()

            # Kombinuj obsahy
            contents = [doc.get(self.config.content_field, "") for doc in docs]
            merged[self.config.content_field] = max(contents, key=len)  # Longest content

            # Průměrné skóre
            scores = [doc.get("score", 0) for doc in docs]
            merged["score"] = np.mean(scores) if scores else 0

        # Přidej merge metadata
        if self.config.keep_merge_info:
            merged["merge_info"] = {
                "merged_count": len(docs),
                "strategy": self.config.merge_strategy,
                "original_ids": [doc.get("id", i) for i, doc in enumerate(docs)],
            }

        return merged

    def _update_merge_stats(self, duplicate_groups: list[list[int]]):
        """Aktualizuje merge statistiky"""
        if not duplicate_groups:
            return

        self.merge_stats["total_merges"] += len(duplicate_groups)

        total_docs_merged = sum(len(group) for group in duplicate_groups)
        self.merge_stats["documents_merged"] += total_docs_merged

        # Avg group size
        avg_group_size = total_docs_merged / len(duplicate_groups)
        alpha = 0.1

        if self.merge_stats["avg_group_size"] == 0:
            self.merge_stats["avg_group_size"] = avg_group_size
        else:
            self.merge_stats["avg_group_size"] = (
                alpha * avg_group_size + (1 - alpha) * self.merge_stats["avg_group_size"]
            )

    def get_stats(self) -> dict[str, Any]:
        """Získá statistiky deduplikace"""
        return {
            "merge_stats": self.merge_stats,
            "minhash_stats": self.minhash_dedup.dedup_stats,
            "cosine_stats": self.cosine_dedup.dedup_stats,
        }


# Factory funkce
def create_deduplication_system(
    config: dict[str, Any], embedding_model=None
) -> IntegratedDeduplication:
    """Factory funkce pro deduplikační systém"""
    dedup_config_dict = config.get("retrieval", {}).get("deduplication", {})
    dedup_config = DeduplicationConfig(**dedup_config_dict)

    return IntegratedDeduplication(dedup_config, embedding_model)


def create_qdrant_optimizer(config: dict[str, Any]) -> QdrantCollectionOptimizer:
    """Factory funkce pro Qdrant optimizer"""
    collections_config = config.get("qdrant", {}).get("collections", {})

    # Convert dict to QdrantCollectionConfig objects
    collection_configs = {}
    for name, cfg in collections_config.items():
        collection_configs[name] = QdrantCollectionConfig(name=name, **cfg)

    return QdrantCollectionOptimizer(collection_configs)


# Export hlavních tříd
__all__ = [
    "DeduplicationConfig",
    "IntegratedDeduplication",
    "QdrantCollectionConfig",
    "QdrantCollectionOptimizer",
    "create_deduplication_system",
    "create_qdrant_optimizer",
]
