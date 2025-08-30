#!/usr/bin/env python3
"""FÁZE 2 Metriky a Reporting Systém
Recall@10, nDCG@10, citation-precision, context_usage_efficiency + export

Author: Senior Python/MLOps Agent
"""

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Phase2Metrics:
    """Metriky pro FÁZI 2"""

    # Core retrieval metrics
    recall_at_10: float = 0.0
    ndcg_at_10: float = 0.0
    citation_precision: float = 0.0
    context_usage_efficiency: float = 0.0

    # Per-channel contributions
    bm25_contribution: float = 0.0
    dense_contribution: float = 0.0
    hyde_contribution: float = 0.0

    # RRF and fusion metrics
    rrf_improvement: float = 0.0
    fusion_effectiveness: float = 0.0
    authority_bonus_impact: float = 0.0
    recency_bonus_impact: float = 0.0

    # MMR diversification
    mmr_diversity_gain: float = 0.0
    similarity_reduction: float = 0.0

    # Deduplication effectiveness
    dedup_precision: float = 0.0
    dedup_recall: float = 0.0
    merge_accuracy: float = 0.0

    # Re-ranking performance
    rerank_improvement: float = 0.0
    cross_encoder_accuracy: float = 0.0
    llm_rating_correlation: float = 0.0

    # Compression metrics
    compression_ratio: float = 0.0
    salience_precision: float = 0.0
    discourse_preservation: float = 0.0

    # Performance metrics
    total_latency_ms: float = 0.0
    hyde_latency_ms: float = 0.0
    rrf_latency_ms: float = 0.0
    dedup_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    compression_latency_ms: float = 0.0

    # Error rates
    hyde_fallback_rate: float = 0.0
    embedding_error_rate: float = 0.0
    llm_error_rate: float = 0.0


class MetricsCalculator:
    """Calculator pro FÁZI 2 metriky"""

    def __init__(self):
        self.calculation_stats = {
            "metrics_calculated": 0,
            "ground_truth_available": 0,
            "avg_calculation_time": 0.0,
        }

    def calculate_recall_at_k(
        self, retrieved_docs: list[dict[str, Any]], relevant_doc_ids: list[str], k: int = 10
    ) -> float:
        """Vypočítá Recall@K"""
        if not relevant_doc_ids:
            return 0.0

        top_k_docs = retrieved_docs[:k]
        retrieved_ids = {doc.get("id", str(i)) for i, doc in enumerate(top_k_docs)}
        relevant_set = set(relevant_doc_ids)

        intersection = retrieved_ids.intersection(relevant_set)
        recall = len(intersection) / len(relevant_set)

        return recall

    def calculate_ndcg_at_k(
        self, retrieved_docs: list[dict[str, Any]], relevance_scores: dict[str, float], k: int = 10
    ) -> float:
        """Vypočítá nDCG@K"""
        if not relevance_scores:
            return 0.0

        # DCG for retrieved docs
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs[:k]):
            doc_id = doc.get("id", str(i))
            relevance = relevance_scores.get(doc_id, 0.0)

            if i == 0:
                dcg += relevance
            else:
                dcg += relevance / math.log2(i + 1)

        # IDCG (Ideal DCG)
        ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = 0.0
        for i, score in enumerate(ideal_scores):
            if i == 0:
                idcg += score
            else:
                idcg += score / math.log2(i + 1)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def calculate_citation_precision(
        self,
        claims: list[dict[str, Any]],
        ground_truth_citations: dict[str, list[str]] | None = None,
    ) -> float:
        """Vypočítá citation precision"""
        if not claims:
            return 0.0

        total_citations = 0
        correct_citations = 0

        for claim in claims:
            citations = claim.get("citations", [])
            claim_text = claim.get("text", "")

            for citation in citations:
                total_citations += 1

                # If ground truth available, use it
                if ground_truth_citations and claim_text in ground_truth_citations:
                    expected_sources = ground_truth_citations[claim_text]
                    citation_source = citation.get("source_id", "")
                    if any(source in citation_source for source in expected_sources):
                        correct_citations += 1
                else:
                    # Heuristic: citation is correct if it contains relevant keywords
                    citation_text = citation.get("passage", "")
                    if self._citation_seems_relevant(claim_text, citation_text):
                        correct_citations += 1

        if total_citations == 0:
            return 0.0

        return correct_citations / total_citations

    def calculate_context_usage_efficiency(
        self, used_tokens: int, available_tokens: int, quality_score: float = 1.0
    ) -> float:
        """Vypočítá context usage efficiency"""
        if available_tokens <= 0:
            return 0.0

        usage_ratio = used_tokens / available_tokens

        # Efficiency = quality per token used
        efficiency = quality_score / max(usage_ratio, 0.01)  # Avoid division by zero

        # Normalize to 0-1 range
        return min(efficiency, 1.0)

    def calculate_per_channel_contributions(
        self, final_docs: list[dict[str, Any]], channel_results: dict[str, list[dict[str, Any]]]
    ) -> dict[str, float]:
        """Vypočítá příspěvky jednotlivých kanálů"""
        contributions = {}

        # Create mapping of doc_id to source channels
        doc_sources = {}
        for channel, docs in channel_results.items():
            for doc in docs:
                doc_id = doc.get("id", "")
                if doc_id not in doc_sources:
                    doc_sources[doc_id] = []
                doc_sources[doc_id].append(channel)

        # Count contributions in final results
        channel_counts = {}
        total_score = 0.0

        for doc in final_docs:
            doc_id = doc.get("id", "")
            doc_score = doc.get("score", 0.0)
            total_score += doc_score

            sources = doc_sources.get(doc_id, ["unknown"])
            score_per_source = doc_score / len(sources)

            for source in sources:
                if source not in channel_counts:
                    channel_counts[source] = 0.0
                channel_counts[source] += score_per_source

        # Normalize to percentages
        if total_score > 0:
            for channel, score in channel_counts.items():
                contributions[channel] = score / total_score

        return contributions

    def calculate_rrf_improvement(
        self,
        baseline_results: list[dict[str, Any]],
        rrf_results: list[dict[str, Any]],
        ground_truth: dict[str, Any] | None = None,
    ) -> float:
        """Vypočítá zlepšení díky RRF"""
        if not baseline_results or not rrf_results:
            return 0.0

        if ground_truth:
            # Use ground truth for comparison
            baseline_ndcg = self.calculate_ndcg_at_k(
                baseline_results, ground_truth.get("relevance_scores", {}), 10
            )
            rrf_ndcg = self.calculate_ndcg_at_k(
                rrf_results, ground_truth.get("relevance_scores", {}), 10
            )

            if baseline_ndcg > 0:
                return (rrf_ndcg - baseline_ndcg) / baseline_ndcg
        else:
            # Use score-based heuristic
            baseline_avg_score = np.mean([doc.get("score", 0) for doc in baseline_results[:10]])
            rrf_avg_score = np.mean([doc.get("score", 0) for doc in rrf_results[:10]])

            if baseline_avg_score > 0:
                return (rrf_avg_score - baseline_avg_score) / baseline_avg_score

        return 0.0

    def calculate_mmr_diversity_gain(
        self, before_mmr: list[dict[str, Any]], after_mmr: list[dict[str, Any]]
    ) -> tuple[float, float]:
        """Vypočítá diversity gain z MMR"""
        if len(before_mmr) < 2 or len(after_mmr) < 2:
            return 0.0, 0.0

        # Calculate average pairwise similarity before and after MMR
        def avg_pairwise_similarity(docs):
            similarities = []
            for i in range(min(10, len(docs))):
                for j in range(i + 1, min(10, len(docs))):
                    sim = self._calculate_doc_similarity(docs[i], docs[j])
                    similarities.append(sim)
            return np.mean(similarities) if similarities else 0.0

        before_similarity = avg_pairwise_similarity(before_mmr)
        after_similarity = avg_pairwise_similarity(after_mmr)

        diversity_gain = max(0, before_similarity - after_similarity)
        similarity_reduction = (before_similarity - after_similarity) / max(before_similarity, 0.01)

        return diversity_gain, similarity_reduction

    def calculate_deduplication_metrics(
        self,
        original_docs: list[dict[str, Any]],
        deduplicated_docs: list[dict[str, Any]],
        merge_mapping: dict[str, Any],
    ) -> tuple[float, float, float]:
        """Vypočítá deduplication metriky"""
        if not merge_mapping:
            return 1.0, 1.0, 1.0  # No deduplication performed

        # Calculate precision: ratio of correct merges
        correct_merges = 0
        total_merges = len(merge_mapping)

        for merged_doc_idx, original_indices in merge_mapping.items():
            if len(original_indices) > 1:
                # Check if merged documents are actually similar
                docs_to_check = [
                    original_docs[i] for i in original_indices if i < len(original_docs)
                ]
                if self._are_docs_similar_group(docs_to_check):
                    correct_merges += 1

        precision = correct_merges / max(total_merges, 1)

        # Calculate recall: how many true duplicates were found
        true_duplicate_pairs = self._find_true_duplicate_pairs(original_docs)
        found_pairs = set()

        for original_indices in merge_mapping.values():
            for i in range(len(original_indices)):
                for j in range(i + 1, len(original_indices)):
                    found_pairs.add(tuple(sorted([original_indices[i], original_indices[j]])))

        if true_duplicate_pairs:
            recall = len(found_pairs.intersection(true_duplicate_pairs)) / len(true_duplicate_pairs)
        else:
            recall = 1.0

        # Merge accuracy: how well content was preserved
        merge_accuracy = self._calculate_merge_accuracy(
            original_docs, deduplicated_docs, merge_mapping
        )

        return precision, recall, merge_accuracy

    def _citation_seems_relevant(self, claim_text: str, citation_text: str) -> bool:
        """Heuristic pro relevantnost citace"""
        claim_words = set(claim_text.lower().split())
        citation_words = set(citation_text.lower().split())

        if not claim_words:
            return False

        overlap = len(claim_words.intersection(citation_words))
        return overlap / len(claim_words) > 0.2  # At least 20% word overlap

    def _calculate_doc_similarity(self, doc1: dict[str, Any], doc2: dict[str, Any]) -> float:
        """Vypočítá similaritu mezi dokumenty"""
        content1 = doc1.get("content", "").lower().split()
        content2 = doc2.get("content", "").lower().split()

        if not content1 or not content2:
            return 0.0

        set1 = set(content1)
        set2 = set(content2)

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0

    def _are_docs_similar_group(self, docs: list[dict[str, Any]], threshold: float = 0.8) -> bool:
        """Zkontroluje, zda jsou dokumenty v skupině podobné"""
        if len(docs) < 2:
            return True

        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                similarity = self._calculate_doc_similarity(docs[i], docs[j])
                if similarity < threshold:
                    return False

        return True

    def _find_true_duplicate_pairs(
        self, docs: list[dict[str, Any]], threshold: float = 0.85
    ) -> set:
        """Najde skutečné duplicate páry"""
        duplicate_pairs = set()

        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                similarity = self._calculate_doc_similarity(docs[i], docs[j])
                if similarity >= threshold:
                    duplicate_pairs.add(tuple(sorted([i, j])))

        return duplicate_pairs

    def _calculate_merge_accuracy(
        self,
        original_docs: list[dict[str, Any]],
        merged_docs: list[dict[str, Any]],
        merge_mapping: dict[str, Any],
    ) -> float:
        """Vypočítá přesnost merge operací"""
        if not merge_mapping:
            return 1.0

        accuracies = []

        for merged_idx, original_indices in merge_mapping.items():
            if merged_idx >= len(merged_docs):
                continue

            merged_content = merged_docs[merged_idx].get("content", "")
            original_contents = [
                original_docs[i].get("content", "")
                for i in original_indices
                if i < len(original_docs)
            ]

            if not original_contents:
                continue

            # Check if merged content preserves key information
            best_original = max(original_contents, key=len)
            similarity = self._text_similarity(merged_content, best_original)
            accuracies.append(similarity)

        return np.mean(accuracies) if accuracies else 1.0

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Vypočítá textovou similaritu"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union


class Phase2MetricsCollector:
    """Collector pro všechny FÁZI 2 metriky"""

    def __init__(self):
        self.calculator = MetricsCalculator()
        self.collection_history = []

    async def collect_comprehensive_metrics(
        self,
        query: str,
        pipeline_results: dict[str, Any],
        ground_truth: dict[str, Any] | None = None,
    ) -> Phase2Metrics:
        """Sbírá komprehensivní metriky pro FÁZI 2"""
        start_time = time.time()

        metrics = Phase2Metrics()

        # Extract results from pipeline
        final_docs = pipeline_results.get("final_documents", [])
        retrieval_metadata = pipeline_results.get("retrieval_metadata", {})
        rrf_metadata = pipeline_results.get("rrf_metadata", {})
        dedup_metadata = pipeline_results.get("dedup_metadata", {})
        rerank_metadata = pipeline_results.get("rerank_metadata", {})
        compression_metadata = pipeline_results.get("compression_metadata", {})

        # Core retrieval metrics
        if ground_truth:
            relevant_docs = ground_truth.get("relevant_documents", [])
            relevance_scores = ground_truth.get("relevance_scores", {})

            metrics.recall_at_10 = self.calculator.calculate_recall_at_k(
                final_docs, relevant_docs, 10
            )
            metrics.ndcg_at_10 = self.calculator.calculate_ndcg_at_k(
                final_docs, relevance_scores, 10
            )
        else:
            # Fallback metrics based on available data
            metrics.recall_at_10 = min(len(final_docs) / 10.0, 1.0)  # Simple heuristic
            metrics.ndcg_at_10 = np.mean([doc.get("score", 0) for doc in final_docs[:10]])

        # Citation precision
        claims = pipeline_results.get("claims", [])
        ground_truth_citations = ground_truth.get("citations", {}) if ground_truth else None
        metrics.citation_precision = self.calculator.calculate_citation_precision(
            claims, ground_truth_citations
        )

        # Context usage efficiency
        used_tokens = compression_metadata.get("compressed_tokens", 0)
        available_tokens = compression_metadata.get("target_tokens", 8000)
        quality_score = metrics.ndcg_at_10
        metrics.context_usage_efficiency = self.calculator.calculate_context_usage_efficiency(
            used_tokens, available_tokens, quality_score
        )

        # Per-channel contributions
        channel_results = retrieval_metadata.get("channel_results", {})
        if channel_results:
            contributions = self.calculator.calculate_per_channel_contributions(
                final_docs, channel_results
            )
            metrics.bm25_contribution = contributions.get("bm25", 0.0)
            metrics.dense_contribution = contributions.get("dense", 0.0)
            metrics.hyde_contribution = contributions.get("hyde", 0.0)

        # RRF metrics
        metrics.rrf_improvement = rrf_metadata.get("improvement_percent", 0.0) / 100.0
        metrics.fusion_effectiveness = rrf_metadata.get("fusion_effectiveness", 0.0)
        metrics.authority_bonus_impact = rrf_metadata.get("avg_authority_bonus", 0.0)
        metrics.recency_bonus_impact = rrf_metadata.get("avg_recency_bonus", 0.0)

        # MMR diversity metrics
        mmr_metadata = rrf_metadata.get("mmr_metadata", {})
        metrics.mmr_diversity_gain = mmr_metadata.get("diversity_gain", 0.0)
        metrics.similarity_reduction = mmr_metadata.get("similarity_reduction", 0.0)

        # Deduplication metrics
        if dedup_metadata.get("deduplication_enabled", False):
            original_docs = pipeline_results.get("original_documents", [])
            merge_mapping = dedup_metadata.get("merge_mapping", {})

            dedup_precision, dedup_recall, merge_accuracy = (
                self.calculator.calculate_deduplication_metrics(
                    original_docs, final_docs, merge_mapping
                )
            )
            metrics.dedup_precision = dedup_precision
            metrics.dedup_recall = dedup_recall
            metrics.merge_accuracy = merge_accuracy

        # Re-ranking metrics
        if rerank_metadata.get("reranking_used", False):
            metrics.rerank_improvement = rerank_metadata.get("avg_score_change", 0.0)
            metrics.cross_encoder_accuracy = rerank_metadata.get("cross_encoder_accuracy", 0.0)
            metrics.llm_rating_correlation = rerank_metadata.get("llm_correlation", 0.0)

        # Compression metrics
        metrics.compression_ratio = compression_metadata.get("compression_ratio", 1.0)
        metrics.salience_precision = compression_metadata.get("salience_precision", 0.0)
        metrics.discourse_preservation = compression_metadata.get("discourse_preservation", 0.0)

        # Performance metrics (latencies)
        metrics.total_latency_ms = pipeline_results.get("total_time_seconds", 0.0) * 1000
        metrics.hyde_latency_ms = retrieval_metadata.get("hyde_time_seconds", 0.0) * 1000
        metrics.rrf_latency_ms = rrf_metadata.get("fusion_time_seconds", 0.0) * 1000
        metrics.dedup_latency_ms = dedup_metadata.get("processing_time_seconds", 0.0) * 1000
        metrics.rerank_latency_ms = rerank_metadata.get("rerank_time_seconds", 0.0) * 1000
        metrics.compression_latency_ms = (
            compression_metadata.get("compression_time_seconds", 0.0) * 1000
        )

        # Error rates
        hyde_stats = retrieval_metadata.get("hyde_stats", {})
        metrics.hyde_fallback_rate = hyde_stats.get("fallback_rate", 0.0)
        metrics.embedding_error_rate = retrieval_metadata.get("embedding_error_rate", 0.0)
        metrics.llm_error_rate = rerank_metadata.get("llm_error_rate", 0.0)

        # Store in history
        collection_time = time.time() - start_time
        self.collection_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "metrics": asdict(metrics),
                "collection_time_seconds": collection_time,
            }
        )

        logger.info(f"Phase 2 metrics collected in {collection_time:.2f}s")

        return metrics

    def get_metrics_summary(self, recent_n: int = 10) -> dict[str, Any]:
        """Získá shrnutí metrik z posledních N běhů"""
        if not self.collection_history:
            return {"error": "No metrics collected yet"}

        recent_metrics = self.collection_history[-recent_n:]

        # Aggregate metrics
        aggregated = {}
        metric_names = list(asdict(Phase2Metrics()).keys())

        for metric_name in metric_names:
            values = [entry["metrics"].get(metric_name, 0.0) for entry in recent_metrics]
            values = [v for v in values if v is not None and not math.isnan(v)]

            if values:
                aggregated[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "count": len(values),
                }
            else:
                aggregated[metric_name] = {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "count": 0,
                }

        return {
            "summary_period": f"Last {len(recent_metrics)} runs",
            "total_runs": len(self.collection_history),
            "aggregated_metrics": aggregated,
            "avg_collection_time": np.mean(
                [entry["collection_time_seconds"] for entry in recent_metrics]
            ),
        }


class Phase2Reporter:
    """Reporter pro FÁZI 2 výsledky a export"""

    def __init__(self, output_dir: str = "artifacts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def export_metrics_json(self, metrics: Phase2Metrics, metadata: dict[str, Any] = None) -> str:
        """Export metrik do JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"phase2_metrics_{timestamp}.json"
        filepath = self.output_dir / filename

        export_data = {
            "timestamp": datetime.now().isoformat(),
            "phase": "FÁZE 2",
            "metrics": asdict(metrics),
            "metadata": metadata or {},
            "export_info": {"format_version": "1.0", "tool": "DeepResearchTool Phase 2 Reporter"},
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Phase 2 metrics exported to {filepath}")
        return str(filepath)

    def export_metrics_markdown(
        self,
        metrics: Phase2Metrics,
        metadata: dict[str, Any] = None,
        include_recommendations: bool = True,
    ) -> str:
        """Export metrik do Markdown reportu"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"phase2_report_{timestamp}.md"
        filepath = self.output_dir / filename

        # Generate markdown content
        markdown_content = self._generate_markdown_report(
            metrics, metadata, include_recommendations
        )

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        logger.info(f"Phase 2 markdown report exported to {filepath}")
        return str(filepath)

    def _generate_markdown_report(
        self, metrics: Phase2Metrics, metadata: dict[str, Any], include_recommendations: bool
    ) -> str:
        """Generuje Markdown report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = f"""# DeepResearchTool - FÁZE 2 Metriky Report

**Generováno:** {timestamp}  
**Fáze:** FÁZE 2 - Retrieval a Re-ranking  
**Verze:** 2.0

## 📊 Core Retrieval Metriky

| Metrika | Hodnota | Cíl | Status |
|---------|---------|-----|--------|
| **Recall@10** | {metrics.recall_at_10:.3f} | ≥0.70 | {"✅" if metrics.recall_at_10 >= 0.7 else "❌"} |
| **nDCG@10** | {metrics.ndcg_at_10:.3f} | ≥0.60 | {"✅" if metrics.ndcg_at_10 >= 0.6 else "❌"} |
| **Citation Precision** | {metrics.citation_precision:.3f} | ≥0.80 | {"✅" if metrics.citation_precision >= 0.8 else "❌"} |
| **Context Usage Efficiency** | {metrics.context_usage_efficiency:.3f} | ≥0.70 | {"✅" if metrics.context_usage_efficiency >= 0.7 else "❌"} |

## 🔄 Per-Channel Contributions

```
BM25 Channel:      {metrics.bm25_contribution:.1%}
Dense Channel:     {metrics.dense_contribution:.1%}
HyDE Channel:      {metrics.hyde_contribution:.1%}
```

## 🚀 RRF a Fusion Effectiveness

| Komponenta | Zlepšení | Impact |
|------------|----------|--------|
| **RRF Improvement** | {metrics.rrf_improvement:.1%} | {"Vysoký" if metrics.rrf_improvement > 0.1 else "Střední" if metrics.rrf_improvement > 0.05 else "Nízký"} |
| **Fusion Effectiveness** | {metrics.fusion_effectiveness:.3f} | {"Efektivní" if metrics.fusion_effectiveness > 0.8 else "Střední"} |
| **Authority Bonus** | {metrics.authority_bonus_impact:.3f} | - |
| **Recency Bonus** | {metrics.recency_bonus_impact:.3f} | - |

## 🎯 MMR Diversifikace

- **Diversity Gain:** {metrics.mmr_diversity_gain:.3f}
- **Similarity Reduction:** {metrics.similarity_reduction:.1%}

## 🔄 Deduplication Performance

| Metrika | Hodnota | Kvalita |
|---------|---------|---------|
| **Precision** | {metrics.dedup_precision:.3f} | {"Vysoká" if metrics.dedup_precision > 0.9 else "Střední" if metrics.dedup_precision > 0.7 else "Nízká"} |
| **Recall** | {metrics.dedup_recall:.3f} | {"Vysoký" if metrics.dedup_recall > 0.9 else "Střední" if metrics.dedup_recall > 0.7 else "Nízký"} |
| **Merge Accuracy** | {metrics.merge_accuracy:.3f} | {"Vysoká" if metrics.merge_accuracy > 0.9 else "Střední"} |

## 🏆 Re-ranking Performance

- **Re-ranking Improvement:** {metrics.rerank_improvement:.1%}
- **Cross-encoder Accuracy:** {metrics.cross_encoder_accuracy:.3f}
- **LLM Rating Correlation:** {metrics.llm_rating_correlation:.3f}

## 🗜️ Contextual Compression

- **Compression Ratio:** {metrics.compression_ratio:.1%}
- **Salience Precision:** {metrics.salience_precision:.3f}
- **Discourse Preservation:** {metrics.discourse_preservation:.3f}

## ⚡ Performance Metriky (Latency)

| Komponenta | Latency (ms) | % z Total |
|------------|--------------|-----------|
| **Total Pipeline** | {metrics.total_latency_ms:.1f} | 100.0% |
| **HyDE Generation** | {metrics.hyde_latency_ms:.1f} | {(metrics.hyde_latency_ms/metrics.total_latency_ms*100):.1f}% |
| **RRF Fusion** | {metrics.rrf_latency_ms:.1f} | {(metrics.rrf_latency_ms/metrics.total_latency_ms*100):.1f}% |
| **Deduplication** | {metrics.dedup_latency_ms:.1f} | {(metrics.dedup_latency_ms/metrics.total_latency_ms*100):.1f}% |
| **Re-ranking** | {metrics.rerank_latency_ms:.1f} | {(metrics.rerank_latency_ms/metrics.total_latency_ms*100):.1f}% |
| **Compression** | {metrics.compression_latency_ms:.1f} | {(metrics.compression_latency_ms/metrics.total_latency_ms*100):.1f}% |

## ⚠️ Error Rates

- **HyDE Fallback Rate:** {metrics.hyde_fallback_rate:.1%}
- **Embedding Error Rate:** {metrics.embedding_error_rate:.1%}
- **LLM Error Rate:** {metrics.llm_error_rate:.1%}

"""

        if include_recommendations:
            report += self._generate_recommendations(metrics)

        if metadata:
            report += f"""
## 📋 Metadata

```json
{json.dumps(metadata, indent=2)}
```
"""

        return report

    def _generate_recommendations(self, metrics: Phase2Metrics) -> str:
        """Generuje doporučení na základě metrik"""
        recommendations = []

        # Recall@10 recommendations
        if metrics.recall_at_10 < 0.7:
            recommendations.append(
                "📈 **Recall@10 nízký:** Zvyš ef_search_param nebo rozšiř HyDE strategii"
            )

        # nDCG@10 recommendations
        if metrics.ndcg_at_10 < 0.6:
            recommendations.append(
                "🎯 **nDCG@10 nízký:** Vylaď RRF váhy nebo vylepši re-ranking model"
            )

        # Citation precision recommendations
        if metrics.citation_precision < 0.8:
            recommendations.append(
                "📚 **Citation precision nízká:** Zpřísni synthesis validation nebo vylepši evidence extraction"
            )

        # Context usage recommendations
        if metrics.context_usage_efficiency < 0.7:
            recommendations.append(
                "🗜️ **Context usage neefektivní:** Optimalizuj compression ratio nebo salience detection"
            )

        # Performance recommendations
        if metrics.total_latency_ms > 30000:  # 30 seconds
            recommendations.append(
                "⚡ **Latency vysoká:** Optimalizuj batch processing nebo parallel execution"
            )

        # HyDE recommendations
        if metrics.hyde_fallback_rate > 0.2:
            recommendations.append(
                "🔄 **HyDE fallback častý:** Zkontroluj LLM connection nebo sniž temperature"
            )

        # Diversity recommendations
        if metrics.similarity_reduction < 0.1:
            recommendations.append(
                "🎨 **MMR diversity nízká:** Zvyš diversity_lambda nebo sniž similarity_threshold"
            )

        if not recommendations:
            recommendations.append("✅ **Všechny metriky v pořádku:** Systém funguje optimálně")

        rec_section = """
## 💡 Doporučení pro Optimalizaci

"""
        for i, rec in enumerate(recommendations, 1):
            rec_section += f"{i}. {rec}\n"

        return rec_section

    def export_performance_comparison(
        self, baseline_metrics: Phase2Metrics, current_metrics: Phase2Metrics
    ) -> str:
        """Export srovnání performance"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"phase2_comparison_{timestamp}.md"
        filepath = self.output_dir / filename

        # Calculate improvements
        improvements = {}
        for field_name, field_type in Phase2Metrics.__annotations__.items():
            baseline_value = getattr(baseline_metrics, field_name)
            current_value = getattr(current_metrics, field_name)

            if baseline_value != 0:
                improvement = ((current_value - baseline_value) / baseline_value) * 100
            else:
                improvement = 0.0

            improvements[field_name] = improvement

        # Generate comparison report
        comparison_content = f"""# FÁZE 2 Performance Comparison

**Generováno:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📊 Key Metrics Comparison

| Metrika | Baseline | Current | Změna | Trend |
|---------|----------|---------|--------|-------|
| **Recall@10** | {baseline_metrics.recall_at_10:.3f} | {current_metrics.recall_at_10:.3f} | {improvements['recall_at_10']:+.1f}% | {"📈" if improvements['recall_at_10'] > 0 else "📉" if improvements['recall_at_10'] < 0 else "➡️"} |
| **nDCG@10** | {baseline_metrics.ndcg_at_10:.3f} | {current_metrics.ndcg_at_10:.3f} | {improvements['ndcg_at_10']:+.1f}% | {"📈" if improvements['ndcg_at_10'] > 0 else "📉" if improvements['ndcg_at_10'] < 0 else "➡️"} |
| **Citation Precision** | {baseline_metrics.citation_precision:.3f} | {current_metrics.citation_precision:.3f} | {improvements['citation_precision']:+.1f}% | {"📈" if improvements['citation_precision'] > 0 else "📉" if improvements['citation_precision'] < 0 else "➡️"} |
| **Context Efficiency** | {baseline_metrics.context_usage_efficiency:.3f} | {current_metrics.context_usage_efficiency:.3f} | {improvements['context_usage_efficiency']:+.1f}% | {"📈" if improvements['context_usage_efficiency'] > 0 else "📉" if improvements['context_usage_efficiency'] < 0 else "➡️"} |

## ⚡ Performance Impact

- **Total Latency:** {baseline_metrics.total_latency_ms:.0f}ms → {current_metrics.total_latency_ms:.0f}ms ({improvements['total_latency_ms']:+.1f}%)
- **RRF Improvement:** {baseline_metrics.rrf_improvement:.1%} → {current_metrics.rrf_improvement:.1%} ({improvements['rrf_improvement']:+.1f}%)
- **Compression Ratio:** {baseline_metrics.compression_ratio:.1%} → {current_metrics.compression_ratio:.1%} ({improvements['compression_ratio']:+.1f}%)

## 🎯 Overall Assessment

"""

        # Overall assessment
        positive_changes = sum(1 for imp in improvements.values() if imp > 5)
        negative_changes = sum(1 for imp in improvements.values() if imp < -5)

        if positive_changes > negative_changes:
            comparison_content += "✅ **Overall Improvement:** Systém vykazuje pozitivní trendy\n"
        elif negative_changes > positive_changes:
            comparison_content += "⚠️ **Performance Degradation:** Některé metriky se zhoršily\n"
        else:
            comparison_content += "➡️ **Stable Performance:** Výkon zůstává konzistentní\n"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(comparison_content)

        logger.info(f"Phase 2 comparison report exported to {filepath}")
        return str(filepath)


# Factory funkce
def create_phase2_metrics_system() -> tuple[Phase2MetricsCollector, Phase2Reporter]:
    """Factory funkce pro FÁZI 2 metrics systém"""
    collector = Phase2MetricsCollector()
    reporter = Phase2Reporter()

    return collector, reporter


# Export hlavních tříd
__all__ = [
    "Phase2Metrics",
    "Phase2MetricsCollector",
    "Phase2Reporter",
    "create_phase2_metrics_system",
]
