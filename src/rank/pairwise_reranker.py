#!/usr/bin/env python3
"""Pairwise Re-ranking Engine
Cross-encoder a LLM-based pairwise ranking s kalibrační křivkou a margin-of-victory

Author: Senior Python/MLOps Agent
"""

from dataclasses import dataclass
import logging
import time
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


@dataclass
class PairwiseComparison:
    """Výsledek pairwise porovnání"""

    doc_a_id: str
    doc_b_id: str
    preference_score: float  # -1.0 (B preferred) to +1.0 (A preferred)
    confidence: float
    margin_of_victory: float
    rationale: str
    comparison_time: float


@dataclass
class RerankingResult:
    """Výsledek re-rankingu"""

    document_id: str
    original_rank: int
    reranked_position: int
    reranking_score: float
    calibrated_score: float
    confidence: float
    ranking_rationale: str
    pairwise_wins: int
    pairwise_losses: int


class PairwiseReranker:
    """Pairwise re-ranking engine s kalibrací"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.reranking_config = config.get("workflow", {}).get("reranking", {})

        # Re-ranking model configuration
        self.model_type = self.reranking_config.get("model_type", "cross_encoder")
        self.model_name = self.reranking_config.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")

        # LLM-as-rater configuration
        self.llm_rater_enabled = self.reranking_config.get("llm_rater_enabled", True)
        self.llm_model = self.reranking_config.get("llm_model", "llama3.2:8b-q4_K_M")

        # Calibration settings
        self.calibration_enabled = self.reranking_config.get("calibration_enabled", True)
        self.calibration_method = self.reranking_config.get("calibration_method", "isotonic")

        # Pairwise comparison settings
        self.max_comparisons = self.reranking_config.get("max_comparisons", 50)
        self.min_margin_threshold = self.reranking_config.get("min_margin_threshold", 0.1)

        # Models (will be initialized)
        self.cross_encoder = None
        self.llm_client = None
        self.calibrator = None

        # Performance tracking
        self.comparison_history = []
        self.calibration_data = []

    async def initialize(self, llm_client=None):
        """Inicializace re-ranking modelů"""
        logger.info("Initializing Pairwise Re-ranker...")
        start_time = time.time()

        try:
            # Initialize cross-encoder if needed
            if self.model_type == "cross_encoder":
                await self._initialize_cross_encoder()

            # Initialize LLM client
            if llm_client and self.llm_rater_enabled:
                self.llm_client = llm_client
                logger.info("✅ LLM-as-rater initialized")

            # Initialize calibrator
            if self.calibration_enabled:
                self._initialize_calibrator()
                logger.info("✅ Score calibrator initialized")

            init_time = time.time() - start_time
            logger.info(f"Pairwise Re-ranker initialized in {init_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to initialize Pairwise Re-ranker: {e}")
            raise

    async def _initialize_cross_encoder(self):
        """Inicializace cross-encoder modelu"""
        try:
            # V reálné implementaci by se zde načetl skutečný cross-encoder
            # Pro demonstraci použiju mock
            logger.info(f"Loading cross-encoder: {self.model_name}")
            self.cross_encoder = MockCrossEncoder(self.model_name)
            logger.info("✅ Cross-encoder initialized")
        except Exception as e:
            logger.error(f"Failed to initialize cross-encoder: {e}")
            self.cross_encoder = None

    def _initialize_calibrator(self):
        """Inicializace kalibračního modelu"""
        if self.calibration_method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
        elif self.calibration_method == "platt":
            # Placeholder pro Platt scaling
            self.calibrator = MockPlattScaling()
        else:
            logger.warning(f"Unknown calibration method: {self.calibration_method}")
            self.calibrator = None

    async def rerank_documents(
        self, query: str, documents: list[dict[str, Any]], top_k: int = 20
    ) -> list[RerankingResult]:
        """Pairwise re-ranking dokumentů

        Args:
            query: Původní dotaz
            documents: Seznam dokumentů k re-rankingu
            top_k: Počet top dokumentů k vrácení

        Returns:
            List RerankingResult s re-ranked pořadím

        """
        logger.info(f"Starting pairwise re-ranking for {len(documents)} documents")
        start_time = time.time()

        if len(documents) <= 1:
            # Single or no document - no re-ranking needed
            return [
                RerankingResult(
                    document_id=doc.get("id", "single_doc"),
                    original_rank=0,
                    reranked_position=0,
                    reranking_score=doc.get("score", 0.5),
                    calibrated_score=doc.get("score", 0.5),
                    confidence=1.0,
                    ranking_rationale="Single document - no comparison needed",
                    pairwise_wins=0,
                    pairwise_losses=0,
                )
                for doc in documents
            ]

        # STEP 1: Perform pairwise comparisons
        logger.debug("Performing pairwise comparisons...")
        comparisons = await self._perform_pairwise_comparisons(query, documents)

        # STEP 2: Aggregate comparison results into rankings
        logger.debug("Aggregating pairwise results...")
        ranking_scores = self._aggregate_pairwise_scores(documents, comparisons)

        # STEP 3: Apply calibration if enabled
        if self.calibration_enabled and self.calibrator:
            logger.debug("Applying score calibration...")
            ranking_scores = self._apply_calibration(ranking_scores)

        # STEP 4: Create final ranking
        results = []

        # Sort by calibrated score
        sorted_docs = sorted(
            enumerate(documents),
            key=lambda x: ranking_scores[x[0]]["calibrated_score"],
            reverse=True,
        )

        for new_rank, (original_rank, doc) in enumerate(sorted_docs[:top_k]):
            score_data = ranking_scores[original_rank]

            result = RerankingResult(
                document_id=doc.get("id", f"doc_{original_rank}"),
                original_rank=original_rank,
                reranked_position=new_rank,
                reranking_score=score_data["raw_score"],
                calibrated_score=score_data["calibrated_score"],
                confidence=score_data["confidence"],
                ranking_rationale=score_data["rationale"],
                pairwise_wins=score_data["wins"],
                pairwise_losses=score_data["losses"],
            )

            results.append(result)

        processing_time = time.time() - start_time
        logger.info(
            f"Pairwise re-ranking completed in {processing_time:.2f}s, "
            f"processed {len(comparisons)} comparisons"
        )

        return results

    async def _perform_pairwise_comparisons(
        self, query: str, documents: list[dict[str, Any]]
    ) -> list[PairwiseComparison]:
        """Provedení pairwise porovnání"""
        comparisons = []
        total_possible = len(documents) * (len(documents) - 1) // 2

        # Limit comparisons for performance
        max_comparisons = min(self.max_comparisons, total_possible)

        # Select most important pairs (top documents vs others)
        comparison_pairs = self._select_comparison_pairs(documents, max_comparisons)

        for i, (doc_a_idx, doc_b_idx) in enumerate(comparison_pairs):
            doc_a = documents[doc_a_idx]
            doc_b = documents[doc_b_idx]

            # Perform comparison
            comparison = await self._compare_document_pair(query, doc_a, doc_b)
            comparisons.append(comparison)

            if (i + 1) % 10 == 0:
                logger.debug(f"Completed {i + 1}/{len(comparison_pairs)} comparisons")

        return comparisons

    def _select_comparison_pairs(
        self, documents: list[dict[str, Any]], max_comparisons: int
    ) -> list[tuple[int, int]]:
        """Výběr nejdůležitějších páru pro porovnání"""
        n_docs = len(documents)
        pairs = []

        # Strategy: Compare top documents with each other and with lower-ranked docs

        # 1. Top vs Top comparisons (most important)
        top_n = min(10, n_docs)
        for i in range(top_n):
            for j in range(i + 1, top_n):
                pairs.append((i, j))

        # 2. Top vs Middle comparisons
        if len(pairs) < max_comparisons:
            middle_start = min(10, n_docs)
            middle_end = min(20, n_docs)

            for i in range(min(5, top_n)):  # Top 5
                for j in range(middle_start, middle_end):
                    if len(pairs) >= max_comparisons:
                        break
                    pairs.append((i, j))

        # 3. Random additional pairs if needed
        if len(pairs) < max_comparisons:
            import random

            remaining = max_comparisons - len(pairs)

            all_possible = [(i, j) for i in range(n_docs) for j in range(i + 1, n_docs)]
            existing_pairs = set(pairs)

            available_pairs = [p for p in all_possible if p not in existing_pairs]
            random.shuffle(available_pairs)

            pairs.extend(available_pairs[:remaining])

        return pairs[:max_comparisons]

    async def _compare_document_pair(
        self, query: str, doc_a: dict[str, Any], doc_b: dict[str, Any]
    ) -> PairwiseComparison:
        """Porovnání jednoho páru dokumentů"""
        start_time = time.time()

        # Try cross-encoder first
        if self.cross_encoder:
            try:
                ce_result = await self._cross_encoder_compare(query, doc_a, doc_b)
                comparison_time = time.time() - start_time

                return PairwiseComparison(
                    doc_a_id=doc_a.get("id", "doc_a"),
                    doc_b_id=doc_b.get("id", "doc_b"),
                    preference_score=ce_result["preference"],
                    confidence=ce_result["confidence"],
                    margin_of_victory=abs(ce_result["preference"]),
                    rationale=f"Cross-encoder comparison: {ce_result['rationale']}",
                    comparison_time=comparison_time,
                )

            except Exception as e:
                logger.warning(f"Cross-encoder comparison failed: {e}")

        # Fallback to LLM-as-rater
        if self.llm_client and self.llm_rater_enabled:
            try:
                llm_result = await self._llm_rater_compare(query, doc_a, doc_b)
                comparison_time = time.time() - start_time

                return PairwiseComparison(
                    doc_a_id=doc_a.get("id", "doc_a"),
                    doc_b_id=doc_b.get("id", "doc_b"),
                    preference_score=llm_result["preference"],
                    confidence=llm_result["confidence"],
                    margin_of_victory=abs(llm_result["preference"]),
                    rationale=f"LLM rater: {llm_result['rationale']}",
                    comparison_time=comparison_time,
                )

            except Exception as e:
                logger.warning(f"LLM rater comparison failed: {e}")

        # Ultimate fallback - score-based comparison
        score_a = doc_a.get("score", 0.5)
        score_b = doc_b.get("score", 0.5)

        preference = np.tanh((score_a - score_b) * 5)  # Scale and normalize
        margin = abs(score_a - score_b)

        comparison_time = time.time() - start_time

        return PairwiseComparison(
            doc_a_id=doc_a.get("id", "doc_a"),
            doc_b_id=doc_b.get("id", "doc_b"),
            preference_score=preference,
            confidence=min(0.8, margin * 2),  # Lower confidence for fallback
            margin_of_victory=margin,
            rationale=f"Score-based fallback: {score_a:.3f} vs {score_b:.3f}",
            comparison_time=comparison_time,
        )

    async def _cross_encoder_compare(
        self, query: str, doc_a: dict[str, Any], doc_b: dict[str, Any]
    ) -> dict[str, Any]:
        """Cross-encoder porovnání"""
        text_a = f"{doc_a.get('title', '')} {doc_a.get('content', '')}"[:500]
        text_b = f"{doc_b.get('title', '')} {doc_b.get('content', '')}"[:500]

        # Mock cross-encoder call
        score_a = await self.cross_encoder.predict(query, text_a)
        score_b = await self.cross_encoder.predict(query, text_b)

        preference = np.tanh((score_a - score_b) * 3)  # Normalize to [-1, 1]
        confidence = min(0.95, abs(score_a - score_b) * 2)

        return {
            "preference": preference,
            "confidence": confidence,
            "rationale": f"CE scores: {score_a:.3f} vs {score_b:.3f}",
        }

    async def _llm_rater_compare(
        self, query: str, doc_a: dict[str, Any], doc_b: dict[str, Any]
    ) -> dict[str, Any]:
        """LLM-as-rater porovnání"""
        prompt = f"""Compare these two documents for relevance to the query: "{query}"

Document A: {doc_a.get('title', '')}
{doc_a.get('content', '')[:300]}

Document B: {doc_b.get('title', '')}
{doc_b.get('content', '')[:300]}

Which document is more relevant? Respond with:
- "A" if Document A is more relevant
- "B" if Document B is more relevant  
- "EQUAL" if they are equally relevant

Provide a confidence score (0.0-1.0) and brief rationale.

Format: PREFERENCE|CONFIDENCE|RATIONALE"""

        try:
            response = await self.llm_client.generate(
                prompt=prompt, max_tokens=150, temperature=0.1
            )

            response_text = response.get("text", "").strip()
            parts = response_text.split("|")

            if len(parts) >= 3:
                preference_text = parts[0].strip()
                confidence = float(parts[1].strip())
                rationale = parts[2].strip()

                # Map preference to score
                if preference_text == "A":
                    preference_score = 0.7
                elif preference_text == "B":
                    preference_score = -0.7
                else:  # EQUAL
                    preference_score = 0.0

                return {
                    "preference": preference_score,
                    "confidence": min(0.95, confidence),
                    "rationale": rationale,
                }

        except Exception as e:
            logger.warning(f"LLM rater parsing failed: {e}")

        # Fallback
        return {
            "preference": 0.0,
            "confidence": 0.3,
            "rationale": "LLM rater failed - defaulting to equal",
        }

    def _aggregate_pairwise_scores(
        self, documents: list[dict[str, Any]], comparisons: list[PairwiseComparison]
    ) -> list[dict[str, Any]]:
        """Agregace pairwise porovnání do ranking scores"""
        n_docs = len(documents)
        scores = []

        for i, doc in enumerate(documents):
            doc_id = doc.get("id", f"doc_{i}")

            # Collect all comparisons involving this document
            wins = 0
            losses = 0
            total_margin = 0.0
            confidences = []
            rationales = []

            for comp in comparisons:
                if comp.doc_a_id == doc_id:
                    if comp.preference_score > self.min_margin_threshold:
                        wins += 1
                        total_margin += comp.margin_of_victory
                    elif comp.preference_score < -self.min_margin_threshold:
                        losses += 1

                    confidences.append(comp.confidence)
                    rationales.append(f"vs {comp.doc_b_id}: {comp.preference_score:.2f}")

                elif comp.doc_b_id == doc_id:
                    if comp.preference_score < -self.min_margin_threshold:
                        wins += 1
                        total_margin += comp.margin_of_victory
                    elif comp.preference_score > self.min_margin_threshold:
                        losses += 1

                    confidences.append(comp.confidence)
                    rationales.append(f"vs {comp.doc_a_id}: {-comp.preference_score:.2f}")

            # Calculate aggregated score
            total_comparisons = wins + losses

            if total_comparisons > 0:
                win_rate = wins / total_comparisons
                avg_margin = total_margin / max(1, wins)
                avg_confidence = np.mean(confidences) if confidences else 0.5

                # Combined score: win rate + margin bonus
                raw_score = win_rate + (avg_margin * 0.2)
            else:
                # No comparisons - use original score
                raw_score = doc.get("score", 0.5)
                avg_confidence = 0.5
                rationales = ["No pairwise comparisons"]

            scores.append(
                {
                    "raw_score": raw_score,
                    "calibrated_score": raw_score,  # Will be updated if calibration enabled
                    "confidence": avg_confidence,
                    "wins": wins,
                    "losses": losses,
                    "rationale": "; ".join(rationales[:3]),  # Top 3 rationales
                }
            )

        return scores

    def _apply_calibration(self, ranking_scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Aplikace kalibrace na ranking scores"""
        if not self.calibrator:
            return ranking_scores

        try:
            # Extract raw scores
            raw_scores = [score["raw_score"] for score in ranking_scores]

            # Apply calibration (mock implementation)
            if hasattr(self.calibrator, "predict"):
                calibrated = self.calibrator.predict(np.array(raw_scores).reshape(-1, 1))
            else:
                # Simple isotonic-style calibration
                calibrated = np.clip(raw_scores, 0, 1)

            # Update scores
            for i, score_dict in enumerate(ranking_scores):
                score_dict["calibrated_score"] = float(calibrated[i])
                score_dict["confidence"] *= 1.1  # Slight confidence boost for calibrated scores
                score_dict["confidence"] = min(0.95, score_dict["confidence"])

            logger.debug("Score calibration applied successfully")

        except Exception as e:
            logger.warning(f"Score calibration failed: {e}")

        return ranking_scores

    def get_reranking_analysis(self) -> dict[str, Any]:
        """Analýza re-ranking performance"""
        if not self.comparison_history:
            return {"message": "No re-ranking history available"}

        # Aggregate comparison metrics
        total_comparisons = sum(len(hist["comparisons"]) for hist in self.comparison_history)
        avg_comparison_time = np.mean(
            [
                comp.comparison_time
                for hist in self.comparison_history
                for comp in hist["comparisons"]
            ]
        )

        # Margin analysis
        margins = [
            comp.margin_of_victory
            for hist in self.comparison_history
            for comp in hist["comparisons"]
        ]

        # Confidence analysis
        confidences = [
            comp.confidence for hist in self.comparison_history for comp in hist["comparisons"]
        ]

        return {
            "total_reranking_sessions": len(self.comparison_history),
            "total_pairwise_comparisons": total_comparisons,
            "avg_comparison_time": avg_comparison_time,
            "margin_of_victory_stats": {
                "mean": np.mean(margins) if margins else 0,
                "std": np.std(margins) if margins else 0,
                "min": min(margins) if margins else 0,
                "max": max(margins) if margins else 0,
            },
            "confidence_stats": {
                "mean": np.mean(confidences) if confidences else 0,
                "std": np.std(confidences) if confidences else 0,
            },
            "calibration_enabled": self.calibration_enabled,
            "model_type": self.model_type,
        }


# Mock classes pro demonstraci
class MockCrossEncoder:
    def __init__(self, model_name: str):
        self.model_name = model_name

    async def predict(self, query: str, text: str) -> float:
        # Mock prediction based on text overlap
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        overlap = len(query_words.intersection(text_words))
        return 0.3 + (overlap / max(len(query_words), 1)) * 0.6


class MockPlattScaling:
    def predict(self, scores):
        # Simple sigmoid calibration
        return 1 / (1 + np.exp(-np.array(scores)))


# Factory function
def create_pairwise_reranker(config: dict[str, Any]) -> PairwiseReranker:
    """Factory function pro Pairwise Re-ranker"""
    return PairwiseReranker(config)
