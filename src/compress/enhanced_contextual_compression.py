#!/usr/bin/env python3
"""
Enhanced Contextual Compression System
Pokročilá komprese kontextu s salience scoring a source-aware budget management

Author: Senior Python/MLOps Agent
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import re
from collections import Counter
import math

logger = logging.getLogger(__name__)


@dataclass
class SalienceScores:
    """Salience scores pro různé komponenty"""
    semantic_score: float
    tfidf_score: float
    keyword_score: float
    novelty_score: float
    redundancy_penalty: float
    combined_score: float


@dataclass
class CompressionUnit:
    """Jednotka pro kompresi (sentence/chunk)"""
    id: str
    text: str
    position: int
    source_type: str
    source_priority: float
    salience_scores: SalienceScores
    token_count: int
    entities: List[str]
    claims_indicators: int
    selected_for_compression: bool
    selection_rationale: str


@dataclass
class CompressionResult:
    """Výsledek komprese"""
    original_units: List[CompressionUnit]
    selected_units: List[CompressionUnit]
    compression_ratio: float
    token_budget_used: int
    token_budget_total: int
    source_distribution: Dict[str, int]
    quality_metrics: Dict[str, float]
    compression_strategy: str


class EnhancedContextualCompressor:
    """Enhanced contextual compression engine"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.compression_config = config.get("compression", {})

        # Budget settings
        self.base_budget = self.compression_config.get("budget_tokens", 2000)
        self.strategy = self.compression_config.get("strategy", "salience")

        # Salience weights
        salience_weights = self.compression_config.get("salience_weights", {})
        self.semantic_weight = salience_weights.get("semantic", 0.5)
        self.tfidf_weight = salience_weights.get("tfidf", 0.3)
        self.keyword_weight = salience_weights.get("keyword", 0.2)

        # Strategy parameters
        self.novelty_threshold = self.compression_config.get("novelty_threshold", 0.7)
        self.redundancy_threshold = self.compression_config.get("redundancy_threshold", 0.85)

        # Source-aware priorities
        self.source_priorities = self._load_source_priorities()

        # Token estimation
        self.tokens_per_word = 1.3  # Average tokens per word for English

        # NLP components
        self.nlp = None
        self.stopwords = set()

        # Cache for efficiency
        self.tfidf_cache = {}
        self.embedding_cache = {}

    def _load_source_priorities(self) -> Dict[str, float]:
        """Načtení source priorities pro budget allocation"""

        source_config = self.compression_config.get("source_priorities", {})

        # Default priorities (higher = more important)
        defaults = {
            "academic": 1.0,      # Primary literature highest priority
            "government": 0.9,    # Official sources high priority
            "wikipedia": 0.7,     # Reliable aggregator medium priority
            "news": 0.6,          # Current events medium priority
            "social_media": 0.3,  # Social aggregators lowest priority
            "unknown": 0.5        # Default medium priority
        }

        # Merge with config overrides
        return {**defaults, **source_config}

    async def initialize(self):
        """Inicializace kompresoru"""

        logger.info("Initializing Enhanced Contextual Compressor...")

        try:
            # Load NLP resources
            await self._initialize_nlp()

            # Load stopwords
            self._load_stopwords()

            logger.info("✅ Enhanced Contextual Compressor initialized")

        except Exception as e:
            logger.error(f"Failed to initialize compressor: {e}")
            raise

    async def _initialize_nlp(self):
        """Inicializace NLP modelů"""
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("spaCy not available, using simplified processing")
            self.nlp = None

    def _load_stopwords(self):
        """Načtení stopwords"""
        # Basic English stopwords
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }

    async def compress_context(self,
                              texts: List[Dict[str, Any]],
                              query: str,
                              token_budget: Optional[int] = None) -> CompressionResult:
        """
        Hlavní compression funkce

        Args:
            texts: Seznam textů s metadata (source_type, content, etc.)
            query: Původní dotaz pro relevance scoring
            token_budget: Token budget (override default)

        Returns:
            CompressionResult s compressed content
        """

        logger.info(f"Starting contextual compression for {len(texts)} texts")

        if not self.nlp:
            await self.initialize()

        # Set budget
        budget = token_budget or self.base_budget

        # STEP 1: Create compression units
        compression_units = await self._create_compression_units(texts, query)

        # STEP 2: Calculate salience scores
        compression_units = await self._calculate_salience_scores(compression_units, query)

        # STEP 3: Apply compression strategy
        selected_units = await self._apply_compression_strategy(compression_units, budget)

        # STEP 4: Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(compression_units, selected_units, query)

        # STEP 5: Create result
        result = self._create_compression_result(
            compression_units, selected_units, budget, quality_metrics
        )

        logger.info(f"Compression completed: {result.compression_ratio:.1%} compression, "
                   f"{result.token_budget_used}/{result.token_budget_total} tokens used")

        return result

    async def _create_compression_units(self, texts: List[Dict[str, Any]],
                                      query: str) -> List[CompressionUnit]:
        """Vytvoření compression units ze vstupních textů"""

        units = []
        unit_counter = 0

        for text_data in texts:
            content = text_data.get("content", "")
            source_type = text_data.get("source_type", "unknown")
            source_priority = self.source_priorities.get(source_type, 0.5)

            # Split into sentences
            sentences = self._split_into_sentences(content)

            for i, sentence in enumerate(sentences):
                if len(sentence.strip()) < 10:  # Skip very short sentences
                    continue

                # Estimate token count
                token_count = self._estimate_tokens(sentence)

                # Extract entities and claims
                entities = await self._extract_entities_simple(sentence)
                claims_indicators = self._count_claim_indicators(sentence)

                unit = CompressionUnit(
                    id=f"unit_{unit_counter}",
                    text=sentence.strip(),
                    position=unit_counter,
                    source_type=source_type,
                    source_priority=source_priority,
                    salience_scores=SalienceScores(0, 0, 0, 0, 0, 0),  # Will be calculated
                    token_count=token_count,
                    entities=entities,
                    claims_indicators=claims_indicators,
                    selected_for_compression=False,
                    selection_rationale=""
                )

                units.append(unit)
                unit_counter += 1

        return units

    def _split_into_sentences(self, text: str) -> List[str]:
        """Rozdělení textu na věty"""

        if self.nlp:
            try:
                doc = self.nlp(text[:5000])  # Limit for performance
                return [sent.text for sent in doc.sents]
            except:
                pass

        # Fallback sentence splitting
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _estimate_tokens(self, text: str) -> int:
        """Odhad počtu tokenů"""
        words = len(text.split())
        return int(words * self.tokens_per_word)

    async def _extract_entities_simple(self, text: str) -> List[str]:
        """Jednoduchá extrakce entit"""

        if self.nlp:
            try:
                doc = self.nlp(text)
                return [ent.text for ent in doc.ents]
            except:
                pass

        # Fallback pattern-based extraction
        entities = []

        # Proper nouns
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend(proper_nouns)

        # Numbers and dates
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
        entities.extend(numbers)

        return entities[:10]  # Limit for performance

    def _count_claim_indicators(self, text: str) -> int:
        """Počítání indikátorů tvrzení"""

        claim_patterns = [
            r'\b(shows?|indicates?|suggests?|demonstrates?|proves?)\b',
            r'\b(evidence|research|study|analysis)\b',
            r'\b(claim|argue|propose|assert|conclude)\b'
        ]

        count = 0
        for pattern in claim_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            count += len(matches)

        return count

    async def _calculate_salience_scores(self, units: List[CompressionUnit],
                                       query: str) -> List[CompressionUnit]:
        """Výpočet salience scores pro všechny units"""

        # Prepare query terms
        query_terms = self._extract_keywords(query)

        # Calculate TF-IDF for all units
        all_texts = [unit.text for unit in units]
        tfidf_scores = self._calculate_tfidf_scores(all_texts, query)

        # Calculate scores for each unit
        for i, unit in enumerate(units):
            # Semantic score (query-text overlap)
            semantic_score = self._calculate_semantic_score(unit.text, query_terms)

            # TF-IDF score
            tfidf_score = tfidf_scores[i]

            # Keyword score (important entities/claims)
            keyword_score = self._calculate_keyword_score(unit)

            # Novelty score (compared to previous units)
            novelty_score = self._calculate_novelty_score(unit, units[:i])

            # Redundancy penalty
            redundancy_penalty = self._calculate_redundancy_penalty(unit, units)

            # Combined score
            combined_score = (
                semantic_score * self.semantic_weight +
                tfidf_score * self.tfidf_weight +
                keyword_score * self.keyword_weight
            )

            # Apply novelty and redundancy adjustments based on strategy
            if "novelty" in self.strategy:
                combined_score *= novelty_score

            if "redundancy" in self.strategy:
                combined_score *= (1 - redundancy_penalty)

            # Apply source priority
            combined_score *= unit.source_priority

            unit.salience_scores = SalienceScores(
                semantic_score=semantic_score,
                tfidf_score=tfidf_score,
                keyword_score=keyword_score,
                novelty_score=novelty_score,
                redundancy_penalty=redundancy_penalty,
                combined_score=combined_score
            )

        return units

    def _extract_keywords(self, text: str) -> List[str]:
        """Extrakce klíčových slov"""

        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if w not in self.stopwords and len(w) > 2]
        return keywords

    def _calculate_tfidf_scores(self, texts: List[str], query: str) -> List[float]:
        """Výpočet TF-IDF scores"""

        # Simple TF-IDF implementation
        query_terms = set(self._extract_keywords(query))

        scores = []
        for text in texts:
            text_terms = self._extract_keywords(text)
            text_term_counts = Counter(text_terms)

            score = 0.0
            for term in query_terms:
                if term in text_term_counts:
                    tf = text_term_counts[term] / len(text_terms) if text_terms else 0

                    # Simple IDF (documents containing term)
                    docs_with_term = sum(1 for t in texts if term in t.lower())
                    idf = math.log(len(texts) / max(docs_with_term, 1))

                    score += tf * idf

            scores.append(score)

        # Normalize scores
        max_score = max(scores) if scores else 1
        return [s / max_score for s in scores]

    def _calculate_semantic_score(self, text: str, query_terms: List[str]) -> float:
        """Výpočet semantic similarity score"""

        text_terms = set(self._extract_keywords(text))
        query_term_set = set(query_terms)

        if not query_term_set:
            return 0.0

        # Jaccard similarity
        intersection = len(text_terms.intersection(query_term_set))
        union = len(text_terms.union(query_term_set))

        return intersection / union if union > 0 else 0.0

    def _calculate_keyword_score(self, unit: CompressionUnit) -> float:
        """Výpočet keyword importance score"""

        score = 0.0

        # Entity score (more entities = more important)
        entity_score = min(len(unit.entities) / 5.0, 1.0)  # Normalize to max 5 entities
        score += entity_score * 0.5

        # Claims indicator score
        claims_score = min(unit.claims_indicators / 3.0, 1.0)  # Normalize to max 3 indicators
        score += claims_score * 0.5

        return score

    def _calculate_novelty_score(self, unit: CompressionUnit,
                                previous_units: List[CompressionUnit]) -> float:
        """Výpočet novelty score (how different from previous content)"""

        if not previous_units:
            return 1.0

        current_terms = set(self._extract_keywords(unit.text))

        max_similarity = 0.0
        for prev_unit in previous_units[-10:]:  # Check last 10 units for efficiency
            prev_terms = set(self._extract_keywords(prev_unit.text))

            if current_terms and prev_terms:
                intersection = len(current_terms.intersection(prev_terms))
                union = len(current_terms.union(prev_terms))
                similarity = intersection / union if union > 0 else 0.0
                max_similarity = max(max_similarity, similarity)

        # Higher novelty score for more novel content
        novelty = 1.0 - max_similarity
        return max(novelty, 0.1)  # Minimum novelty threshold

    def _calculate_redundancy_penalty(self, unit: CompressionUnit,
                                     all_units: List[CompressionUnit]) -> float:
        """Výpočet redundancy penalty"""

        current_terms = set(self._extract_keywords(unit.text))

        redundancy_count = 0
        for other_unit in all_units:
            if other_unit.id == unit.id:
                continue

            other_terms = set(self._extract_keywords(other_unit.text))

            if current_terms and other_terms:
                intersection = len(current_terms.intersection(other_terms))
                similarity = intersection / len(current_terms) if current_terms else 0.0

                if similarity > self.redundancy_threshold:
                    redundancy_count += 1

        # Penalty increases with redundancy count
        penalty = min(redundancy_count * 0.2, 0.8)  # Max 80% penalty
        return penalty

    async def _apply_compression_strategy(self, units: List[CompressionUnit],
                                        budget: int) -> List[CompressionUnit]:
        """Aplikace compression strategy pro výběr units"""

        logger.info(f"Applying compression strategy '{self.strategy}' with budget {budget} tokens")

        # Sort by combined salience score (highest first)
        sorted_units = sorted(units, key=lambda u: u.salience_scores.combined_score, reverse=True)

        selected_units = []
        used_tokens = 0

        # Apply source-aware budget allocation
        source_budgets = self._allocate_source_budgets(sorted_units, budget)

        # Track source usage
        source_usage = {source: 0 for source in source_budgets.keys()}

        for unit in sorted_units:
            source_type = unit.source_type
            source_budget = source_budgets.get(source_type, 0)

            # Check if we can add this unit
            if (used_tokens + unit.token_count <= budget and
                source_usage[source_type] + unit.token_count <= source_budget):

                unit.selected_for_compression = True
                unit.selection_rationale = (
                    f"Score: {unit.salience_scores.combined_score:.3f}, "
                    f"Source: {source_type} ({unit.source_priority:.1f}), "
                    f"Tokens: {unit.token_count}"
                )

                selected_units.append(unit)
                used_tokens += unit.token_count
                source_usage[source_type] += unit.token_count

            else:
                unit.selected_for_compression = False
                if used_tokens + unit.token_count > budget:
                    unit.selection_rationale = f"Exceeded total budget ({used_tokens + unit.token_count} > {budget})"
                else:
                    unit.selection_rationale = f"Exceeded source budget for {source_type}"

        logger.info(f"Selected {len(selected_units)}/{len(units)} units, "
                   f"using {used_tokens}/{budget} tokens")

        # Sort selected units back to original order for coherence
        selected_units.sort(key=lambda u: u.position)

        return selected_units

    def _allocate_source_budgets(self, units: List[CompressionUnit],
                               total_budget: int) -> Dict[str, int]:
        """Alokace budget per source type based on priorities"""

        # Count tokens per source type
        source_tokens = {}
        source_priorities = {}

        for unit in units:
            source_type = unit.source_type
            if source_type not in source_tokens:
                source_tokens[source_type] = 0
                source_priorities[source_type] = unit.source_priority
            source_tokens[source_type] += unit.token_count

        # Calculate budget allocation based on priorities
        total_priority_weight = sum(source_priorities.values())
        source_budgets = {}

        for source_type, priority in source_priorities.items():
            # Base allocation by priority
            base_allocation = int((priority / total_priority_weight) * total_budget)

            # Ensure we don't allocate more than available content
            available_tokens = source_tokens[source_type]
            allocated_budget = min(base_allocation, available_tokens)

            source_budgets[source_type] = allocated_budget

        # Redistribute unused budget
        total_allocated = sum(source_budgets.values())
        if total_allocated < total_budget:
            remaining = total_budget - total_allocated

            # Distribute remaining budget proportionally to high-priority sources
            high_priority_sources = [s for s, p in source_priorities.items() if p >= 0.7]
            if high_priority_sources:
                per_source_bonus = remaining // len(high_priority_sources)
                for source in high_priority_sources:
                    source_budgets[source] += per_source_bonus

        return source_budgets

    def _calculate_quality_metrics(self, all_units: List[CompressionUnit],
                                 selected_units: List[CompressionUnit],
                                 query: str) -> Dict[str, float]:
        """Výpočet quality metrics pro compression"""

        metrics = {}

        # Basic compression metrics
        metrics["compression_ratio"] = len(selected_units) / len(all_units) if all_units else 0

        total_tokens = sum(u.token_count for u in all_units)
        selected_tokens = sum(u.token_count for u in selected_units)
        metrics["token_compression_ratio"] = selected_tokens / total_tokens if total_tokens else 0

        # Salience preservation
        total_salience = sum(u.salience_scores.combined_score for u in all_units)
        preserved_salience = sum(u.salience_scores.combined_score for u in selected_units)
        metrics["salience_preservation"] = preserved_salience / total_salience if total_salience else 0

        # Entity coverage
        all_entities = set()
        for unit in all_units:
            all_entities.update(unit.entities)

        selected_entities = set()
        for unit in selected_units:
            selected_entities.update(unit.entities)

        metrics["entity_coverage"] = len(selected_entities) / len(all_entities) if all_entities else 0

        # Claims coverage
        total_claims = sum(u.claims_indicators for u in all_units)
        selected_claims = sum(u.claims_indicators for u in selected_units)
        metrics["claims_coverage"] = selected_claims / total_claims if total_claims else 0

        # Query relevance preservation
        query_terms = set(self._extract_keywords(query))

        all_query_matches = 0
        selected_query_matches = 0

        for unit in all_units:
            unit_terms = set(self._extract_keywords(unit.text))
            matches = len(unit_terms.intersection(query_terms))
            all_query_matches += matches

            if unit in selected_units:
                selected_query_matches += matches

        metrics["query_relevance_preservation"] = (
            selected_query_matches / all_query_matches if all_query_matches else 0
        )

        # Context usage efficiency
        metrics["context_usage_efficiency"] = metrics["salience_preservation"] / metrics["token_compression_ratio"] if metrics["token_compression_ratio"] > 0 else 0

        return metrics

    def _create_compression_result(self, all_units: List[CompressionUnit],
                                 selected_units: List[CompressionUnit],
                                 budget: int,
                                 quality_metrics: Dict[str, float]) -> CompressionResult:
        """Vytvoření compression result"""

        # Calculate source distribution
        source_distribution = {}
        for unit in selected_units:
            source_type = unit.source_type
            source_distribution[source_type] = source_distribution.get(source_type, 0) + 1

        # Calculate token usage
        tokens_used = sum(u.token_count for u in selected_units)

        result = CompressionResult(
            original_units=all_units,
            selected_units=selected_units,
            compression_ratio=quality_metrics["compression_ratio"],
            token_budget_used=tokens_used,
            token_budget_total=budget,
            source_distribution=source_distribution,
            quality_metrics=quality_metrics,
            compression_strategy=self.strategy
        )

        return result

    def get_compressed_text(self, result: CompressionResult) -> str:
        """Získání komprimovaného textu jako string"""

        selected_texts = [unit.text for unit in result.selected_units]
        return "\n".join(selected_texts)

    def get_compression_report(self, result: CompressionResult) -> Dict[str, Any]:
        """Vytvoření compression report pro audit"""

        report = {
            "compression_summary": {
                "strategy": result.compression_strategy,
                "original_units": len(result.original_units),
                "selected_units": len(result.selected_units),
                "compression_ratio": f"{result.compression_ratio:.1%}",
                "token_usage": f"{result.token_budget_used}/{result.token_budget_total}",
                "token_efficiency": f"{result.token_budget_used/result.token_budget_total:.1%}"
            },
            "quality_metrics": result.quality_metrics,
            "source_distribution": result.source_distribution,
            "selection_details": []
        }

        # Add selection details for audit trail
        for unit in result.original_units:
            detail = {
                "unit_id": unit.id,
                "selected": unit.selected_for_compression,
                "salience_score": unit.salience_scores.combined_score,
                "source_type": unit.source_type,
                "source_priority": unit.source_priority,
                "token_count": unit.token_count,
                "rationale": unit.selection_rationale
            }
            report["selection_details"].append(detail)

        return report
