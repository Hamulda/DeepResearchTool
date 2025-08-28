#!/usr/bin/env python3
"""
Adaptive Chunking System
Dynamicky přizpůsobuje velikost chunků podle hustoty entit a tvrzení

Author: Senior Python/MLOps Agent
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import re
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class EntityDensityMetrics:
    """Metriky hustoty entit"""
    named_entities_count: int
    unique_entities_count: int
    entity_density: float  # entities per 100 words
    entity_diversity: float  # unique/total ratio
    key_entities: List[str]


@dataclass
class ClaimDensityMetrics:
    """Metriky hustoty tvrzení"""
    claim_indicators_count: int
    evidence_markers_count: int
    argumentation_density: float
    confidence_markers_count: int
    uncertainty_markers_count: int


@dataclass
class AdaptiveChunk:
    """Adaptivní chunk s metrikami"""
    id: str
    text: str
    start_position: int
    end_position: int
    target_size: int
    actual_size: int
    entity_metrics: EntityDensityMetrics
    claim_metrics: ClaimDensityMetrics
    adaptive_score: float
    split_rationale: str
    parent_chunk_id: Optional[str] = None
    children_chunk_ids: List[str] = None


class AdaptiveChunker:
    """Adaptivní chunking engine"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.adaptive_config = config.get("compression", {}).get("adaptive_chunking", {})

        # Base chunking parameters
        self.base_chunk_size = self.adaptive_config.get("base_chunk_size", 512)
        self.min_chunk_size = self.adaptive_config.get("min_chunk_size", 100)
        self.max_chunk_size = self.adaptive_config.get("max_chunk_size", 1024)

        # Density thresholds
        self.high_entity_threshold = self.adaptive_config.get("high_entity_threshold", 0.15)  # 15 entities per 100 words
        self.low_entity_threshold = self.adaptive_config.get("low_entity_threshold", 0.05)   # 5 entities per 100 words

        self.high_claim_threshold = self.adaptive_config.get("high_claim_threshold", 0.20)   # 20 claim markers per 100 words
        self.low_claim_threshold = self.adaptive_config.get("low_claim_threshold", 0.05)     # 5 claim markers per 100 words

        # Adaptation factors
        self.entity_adaptation_factor = self.adaptive_config.get("entity_adaptation_factor", 0.3)
        self.claim_adaptation_factor = self.adaptive_config.get("claim_adaptation_factor", 0.4)
        self.diversity_bonus = self.adaptive_config.get("diversity_bonus", 0.2)

        # NLP model for entity extraction
        self.nlp = None

        # Compile patterns
        self._compile_analysis_patterns()

    async def initialize(self):
        """Inicializace NLP modelů"""

        logger.info("Initializing Adaptive Chunker...")

        try:
            # Try to load spaCy model
            import spacy
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("✅ spaCy model loaded for entity extraction")
            except OSError:
                logger.warning("spaCy model not available, using pattern-based entity detection")
                self.nlp = None

            logger.info("✅ Adaptive Chunker initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Adaptive Chunker: {e}")
            raise

    def _compile_analysis_patterns(self):
        """Kompilace patterns pro analýzu"""

        # Claim indicators
        self.claim_patterns = [
            re.compile(r'\b(we\s+claim|we\s+argue|we\s+propose|we\s+assert|we\s+contend)\b', re.IGNORECASE),
            re.compile(r'\b(it\s+is\s+evident|clearly|obviously|undoubtedly|certainly)\b', re.IGNORECASE),
            re.compile(r'\b(demonstrates?|proves?|shows?|indicates?|suggests?)\s+that\b', re.IGNORECASE),
            re.compile(r'\b(our\s+results|our\s+findings|our\s+analysis)\s+(show|indicate|suggest|demonstrate)\b', re.IGNORECASE),
        ]

        # Evidence markers
        self.evidence_patterns = [
            re.compile(r'\b(evidence\s+shows|data\s+indicates?|studies\s+show|research\s+demonstrates?)\b', re.IGNORECASE),
            re.compile(r'\b(according\s+to|based\s+on|as\s+shown\s+by|as\s+reported\s+by)\b', re.IGNORECASE),
            re.compile(r'\b(empirical\s+evidence|experimental\s+results|statistical\s+analysis)\b', re.IGNORECASE),
            re.compile(r'\[(\d+)\]|\(([A-Za-z]+,?\s*\d{4})\)', re.IGNORECASE),  # Citations
        ]

        # Confidence markers
        self.confidence_patterns = [
            re.compile(r'\b(definitely|certainly|undoubtedly|without\s+doubt|conclusively)\b', re.IGNORECASE),
            re.compile(r'\b(strong\s+evidence|robust\s+findings|significant\s+results)\b', re.IGNORECASE),
        ]

        # Uncertainty markers
        self.uncertainty_patterns = [
            re.compile(r'\b(possibly|probably|likely|potentially|presumably|apparently)\b', re.IGNORECASE),
            re.compile(r'\b(may\s+be|might\s+be|could\s+be|seems\s+to|appears\s+to)\b', re.IGNORECASE),
            re.compile(r'\b(tentative|preliminary|inconclusive|uncertain)\b', re.IGNORECASE),
        ]

        # Named entity patterns (fallback if spaCy not available)
        self.entity_patterns = [
            re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'),  # Proper nouns
            re.compile(r'\b\d{4}\b'),  # Years
            re.compile(r'\b[A-Z]{2,}\b'),  # Acronyms
            re.compile(r'\b\d+\.?\d*\s*%\b'),  # Percentages
            re.compile(r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b'),  # Money
        ]

    async def adaptive_chunk_text(self, text: str, document_id: str = "doc") -> List[AdaptiveChunk]:
        """
        Hlavní adaptivní chunking funkce

        Args:
            text: Text k rozchunkování
            document_id: ID dokumentu

        Returns:
            List AdaptiveChunk s adaptivně upravenými velikostmi
        """

        logger.info(f"Starting adaptive chunking for document {document_id}")

        if not self.nlp:
            await self.initialize()

        # STEP 1: Analýza celého textu pro kontext
        global_metrics = await self._analyze_global_context(text)

        # STEP 2: Vytvoření předběžných chunků
        preliminary_chunks = self._create_preliminary_chunks(text, document_id)

        # STEP 3: Analýza hustoty pro každý chunk
        analyzed_chunks = await self._analyze_chunk_densities(preliminary_chunks)

        # STEP 4: Adaptivní úprava velikostí
        adaptive_chunks = self._apply_adaptive_sizing(analyzed_chunks, global_metrics)

        # STEP 5: Optimalizace boundaries
        optimized_chunks = self._optimize_chunk_boundaries(adaptive_chunks, text)

        logger.info(f"Created {len(optimized_chunks)} adaptive chunks")

        return optimized_chunks

    async def _analyze_global_context(self, text: str) -> Dict[str, Any]:
        """Analýza globálního kontextu dokumentu"""

        word_count = len(text.split())

        # Entity analysis
        entities = await self._extract_entities(text)
        entity_density = len(entities) / (word_count / 100) if word_count > 0 else 0

        # Claim analysis
        claim_indicators = self._count_pattern_matches(text, self.claim_patterns)
        evidence_markers = self._count_pattern_matches(text, self.evidence_patterns)
        claim_density = (claim_indicators + evidence_markers) / (word_count / 100) if word_count > 0 else 0

        return {
            "total_words": word_count,
            "global_entity_density": entity_density,
            "global_claim_density": claim_density,
            "total_entities": len(entities),
            "unique_entities": len(set(entities)),
            "document_type": self._classify_document_type(text)
        }

    def _classify_document_type(self, text: str) -> str:
        """Klasifikace typu dokumentu"""

        # Simple heuristics
        academic_indicators = len(re.findall(r'\b(abstract|introduction|methodology|results|conclusion|references)\b', text, re.IGNORECASE))
        news_indicators = len(re.findall(r'\b(reported|according to sources|breaking news|journalist)\b', text, re.IGNORECASE))
        technical_indicators = len(re.findall(r'\b(algorithm|implementation|system|architecture|framework)\b', text, re.IGNORECASE))

        if academic_indicators >= 3:
            return "academic"
        elif news_indicators >= 2:
            return "news"
        elif technical_indicators >= 3:
            return "technical"
        else:
            return "general"

    def _create_preliminary_chunks(self, text: str, document_id: str) -> List[Dict[str, Any]]:
        """Vytvoření předběžných chunků na base_chunk_size"""

        chunks = []
        chunk_id_counter = 0

        # Split on sentence boundaries for better coherence
        sentences = re.split(r'[.!?]+\s+', text)

        current_chunk = ""
        current_start = 0

        for sentence in sentences:
            # Check if adding this sentence would exceed base size
            if len(current_chunk + sentence) > self.base_chunk_size and current_chunk:
                # Create chunk
                chunks.append({
                    "id": f"{document_id}_adaptive_{chunk_id_counter}",
                    "text": current_chunk.strip(),
                    "start_position": current_start,
                    "end_position": current_start + len(current_chunk)
                })

                chunk_id_counter += 1
                current_start += len(current_chunk)
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                "id": f"{document_id}_adaptive_{chunk_id_counter}",
                "text": current_chunk.strip(),
                "start_position": current_start,
                "end_position": current_start + len(current_chunk)
            })

        return chunks

    async def _analyze_chunk_densities(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analýza hustoty entit a tvrzení pro každý chunk"""

        analyzed_chunks = []

        for chunk in chunks:
            text = chunk["text"]
            word_count = len(text.split())

            # Entity analysis
            entities = await self._extract_entities(text)
            unique_entities = list(set(entities))

            entity_metrics = EntityDensityMetrics(
                named_entities_count=len(entities),
                unique_entities_count=len(unique_entities),
                entity_density=len(entities) / (word_count / 100) if word_count > 0 else 0,
                entity_diversity=len(unique_entities) / max(len(entities), 1),
                key_entities=unique_entities[:10]  # Top 10 unique entities
            )

            # Claim analysis
            claim_indicators = self._count_pattern_matches(text, self.claim_patterns)
            evidence_markers = self._count_pattern_matches(text, self.evidence_patterns)
            confidence_markers = self._count_pattern_matches(text, self.confidence_patterns)
            uncertainty_markers = self._count_pattern_matches(text, self.uncertainty_patterns)

            claim_metrics = ClaimDensityMetrics(
                claim_indicators_count=claim_indicators,
                evidence_markers_count=evidence_markers,
                argumentation_density=(claim_indicators + evidence_markers) / (word_count / 100) if word_count > 0 else 0,
                confidence_markers_count=confidence_markers,
                uncertainty_markers_count=uncertainty_markers
            )

            # Calculate adaptive score
            adaptive_score = self._calculate_adaptive_score(entity_metrics, claim_metrics)

            analyzed_chunk = {
                **chunk,
                "entity_metrics": entity_metrics,
                "claim_metrics": claim_metrics,
                "adaptive_score": adaptive_score,
                "word_count": word_count
            }

            analyzed_chunks.append(analyzed_chunk)

        return analyzed_chunks

    async def _extract_entities(self, text: str) -> List[str]:
        """Extrakce entit z textu"""

        entities = []

        if self.nlp:
            try:
                # Use spaCy for better entity extraction
                doc = self.nlp(text[:2000])  # Limit for performance
                entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']]
            except Exception as e:
                logger.warning(f"spaCy entity extraction failed: {e}")
                entities = self._extract_entities_fallback(text)
        else:
            entities = self._extract_entities_fallback(text)

        return entities

    def _extract_entities_fallback(self, text: str) -> List[str]:
        """Fallback entity extraction using patterns"""

        entities = []

        for pattern in self.entity_patterns:
            matches = pattern.findall(text)
            entities.extend(matches)

        # Filter out common words and short matches
        filtered_entities = []
        common_words = {'The', 'This', 'That', 'And', 'Or', 'But', 'For', 'With'}

        for entity in entities:
            if len(entity) > 2 and entity not in common_words:
                filtered_entities.append(entity)

        return filtered_entities

    def _count_pattern_matches(self, text: str, patterns: List[re.Pattern]) -> int:
        """Počítání matches pro seznam patterns"""

        total_matches = 0
        for pattern in patterns:
            matches = pattern.findall(text)
            total_matches += len(matches)

        return total_matches

    def _calculate_adaptive_score(self, entity_metrics: EntityDensityMetrics,
                                claim_metrics: ClaimDensityMetrics) -> float:
        """Výpočet adaptive score pro určení potřeby úpravy velikosti"""

        # Entity score component
        entity_score = 0.0
        if entity_metrics.entity_density > self.high_entity_threshold:
            entity_score = 1.0  # High density - needs smaller chunks
        elif entity_metrics.entity_density < self.low_entity_threshold:
            entity_score = -1.0  # Low density - can use larger chunks
        else:
            # Linear interpolation
            range_size = self.high_entity_threshold - self.low_entity_threshold
            normalized = (entity_metrics.entity_density - self.low_entity_threshold) / range_size
            entity_score = (normalized * 2) - 1  # Map to [-1, 1]

        # Claim score component
        claim_score = 0.0
        if claim_metrics.argumentation_density > self.high_claim_threshold:
            claim_score = 1.0  # High argumentation density - needs smaller chunks
        elif claim_metrics.argumentation_density < self.low_claim_threshold:
            claim_score = -1.0  # Low argumentation density - can use larger chunks
        else:
            # Linear interpolation
            range_size = self.high_claim_threshold - self.low_claim_threshold
            normalized = (claim_metrics.argumentation_density - self.low_claim_threshold) / range_size
            claim_score = (normalized * 2) - 1

        # Diversity bonus
        diversity_bonus = entity_metrics.entity_diversity * self.diversity_bonus

        # Combined adaptive score
        adaptive_score = (
            entity_score * self.entity_adaptation_factor +
            claim_score * self.claim_adaptation_factor +
            diversity_bonus
        )

        return np.clip(adaptive_score, -1.0, 1.0)

    def _apply_adaptive_sizing(self, analyzed_chunks: List[Dict[str, Any]],
                             global_metrics: Dict[str, Any]) -> List[AdaptiveChunk]:
        """Aplikace adaptivního sizingu"""

        adaptive_chunks = []

        for chunk_data in analyzed_chunks:
            adaptive_score = chunk_data["adaptive_score"]
            current_size = len(chunk_data["text"])

            # Calculate target size based on adaptive score
            if adaptive_score > 0:
                # High density - make smaller
                size_factor = 1.0 - (adaptive_score * 0.5)  # Reduce by up to 50%
                target_size = int(self.base_chunk_size * size_factor)
                target_size = max(target_size, self.min_chunk_size)
                split_rationale = f"High density (score: {adaptive_score:.2f}) - reduced size"
            elif adaptive_score < 0:
                # Low density - can make larger
                size_factor = 1.0 + (abs(adaptive_score) * 0.3)  # Increase by up to 30%
                target_size = int(self.base_chunk_size * size_factor)
                target_size = min(target_size, self.max_chunk_size)
                split_rationale = f"Low density (score: {adaptive_score:.2f}) - increased size"
            else:
                # Normal density - keep base size
                target_size = self.base_chunk_size
                split_rationale = f"Normal density (score: {adaptive_score:.2f}) - base size"

            adaptive_chunk = AdaptiveChunk(
                id=chunk_data["id"],
                text=chunk_data["text"],
                start_position=chunk_data["start_position"],
                end_position=chunk_data["end_position"],
                target_size=target_size,
                actual_size=current_size,
                entity_metrics=chunk_data["entity_metrics"],
                claim_metrics=chunk_data["claim_metrics"],
                adaptive_score=adaptive_score,
                split_rationale=split_rationale,
                children_chunk_ids=[]
            )

            adaptive_chunks.append(adaptive_chunk)

        return adaptive_chunks

    def _optimize_chunk_boundaries(self, chunks: List[AdaptiveChunk],
                                 original_text: str) -> List[AdaptiveChunk]:
        """Optimalizace chunk boundaries pro lepší koherenci"""

        optimized_chunks = []

        for i, chunk in enumerate(chunks):
            current_size = chunk.actual_size
            target_size = chunk.target_size

            # Check if chunk needs resizing
            if abs(current_size - target_size) > (target_size * 0.2):  # 20% tolerance
                if current_size > target_size:
                    # Split large chunk
                    split_chunks = self._split_chunk_optimally(chunk, original_text)
                    optimized_chunks.extend(split_chunks)
                else:
                    # Try to merge with next chunk if beneficial
                    if i + 1 < len(chunks):
                        next_chunk = chunks[i + 1]
                        merged_chunk = self._try_merge_chunks(chunk, next_chunk)
                        if merged_chunk:
                            optimized_chunks.append(merged_chunk)
                            chunks[i + 1] = None  # Mark for skipping
                        else:
                            optimized_chunks.append(chunk)
                    else:
                        optimized_chunks.append(chunk)
            else:
                optimized_chunks.append(chunk)

        # Filter out None entries from merging
        return [chunk for chunk in optimized_chunks if chunk is not None]

    def _split_chunk_optimally(self, chunk: AdaptiveChunk,
                             original_text: str) -> List[AdaptiveChunk]:
        """Optimální rozdělení chunku"""

        text = chunk.text
        target_size = chunk.target_size

        # Find good split points (sentence boundaries)
        sentences = re.split(r'([.!?]+\s+)', text)

        sub_chunks = []
        current_text = ""
        current_start = chunk.start_position
        sub_chunk_counter = 0

        for i in range(0, len(sentences), 2):  # Step by 2 to include punctuation
            sentence = sentences[i]
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""

            sentence_with_punct = sentence + punctuation

            if len(current_text + sentence_with_punct) > target_size and current_text:
                # Create sub-chunk
                sub_chunk = AdaptiveChunk(
                    id=f"{chunk.id}_split_{sub_chunk_counter}",
                    text=current_text.strip(),
                    start_position=current_start,
                    end_position=current_start + len(current_text),
                    target_size=target_size,
                    actual_size=len(current_text),
                    entity_metrics=chunk.entity_metrics,  # Inherit from parent
                    claim_metrics=chunk.claim_metrics,
                    adaptive_score=chunk.adaptive_score,
                    split_rationale=f"Split from {chunk.id} - {chunk.split_rationale}",
                    parent_chunk_id=chunk.id,
                    children_chunk_ids=[]
                )

                sub_chunks.append(sub_chunk)
                sub_chunk_counter += 1

                current_start += len(current_text)
                current_text = sentence_with_punct
            else:
                current_text += sentence_with_punct

        # Add final sub-chunk
        if current_text.strip():
            sub_chunk = AdaptiveChunk(
                id=f"{chunk.id}_split_{sub_chunk_counter}",
                text=current_text.strip(),
                start_position=current_start,
                end_position=chunk.end_position,
                target_size=target_size,
                actual_size=len(current_text),
                entity_metrics=chunk.entity_metrics,
                claim_metrics=chunk.claim_metrics,
                adaptive_score=chunk.adaptive_score,
                split_rationale=f"Split from {chunk.id} - {chunk.split_rationale}",
                parent_chunk_id=chunk.id,
                children_chunk_ids=[]
            )

            sub_chunks.append(sub_chunk)

        # Update parent's children references
        chunk.children_chunk_ids = [sc.id for sc in sub_chunks]

        return sub_chunks

    def _try_merge_chunks(self, chunk1: AdaptiveChunk,
                         chunk2: AdaptiveChunk) -> Optional[AdaptiveChunk]:
        """Pokus o sloučení dvou chunků"""

        combined_size = chunk1.actual_size + chunk2.actual_size
        max_target = max(chunk1.target_size, chunk2.target_size)

        # Only merge if combined size doesn't exceed max target by more than 20%
        if combined_size <= max_target * 1.2:
            combined_text = chunk1.text + " " + chunk2.text

            # Average the metrics
            avg_adaptive_score = (chunk1.adaptive_score + chunk2.adaptive_score) / 2

            merged_chunk = AdaptiveChunk(
                id=f"{chunk1.id}_merged_{chunk2.id.split('_')[-1]}",
                text=combined_text,
                start_position=chunk1.start_position,
                end_position=chunk2.end_position,
                target_size=max_target,
                actual_size=combined_size,
                entity_metrics=chunk1.entity_metrics,  # Use first chunk's metrics
                claim_metrics=chunk1.claim_metrics,
                adaptive_score=avg_adaptive_score,
                split_rationale=f"Merged chunks: {chunk1.split_rationale} + {chunk2.split_rationale}",
                children_chunk_ids=[]
            )

            return merged_chunk

        return None

    def get_adaptation_analysis(self, chunks: List[AdaptiveChunk]) -> Dict[str, Any]:
        """Analýza adaptace chunkingu"""

        if not chunks:
            return {"message": "No chunks to analyze"}

        # Size analysis
        target_sizes = [chunk.target_size for chunk in chunks]
        actual_sizes = [chunk.actual_size for chunk in chunks]
        adaptive_scores = [chunk.adaptive_score for chunk in chunks]

        # Entity density analysis
        entity_densities = [chunk.entity_metrics.entity_density for chunk in chunks]
        claim_densities = [chunk.claim_metrics.argumentation_density for chunk in chunks]

        # Adaptation categories
        high_density_chunks = [c for c in chunks if c.adaptive_score > 0.3]
        low_density_chunks = [c for c in chunks if c.adaptive_score < -0.3]
        normal_chunks = [c for c in chunks if -0.3 <= c.adaptive_score <= 0.3]

        return {
            "total_chunks": len(chunks),
            "size_adaptation": {
                "avg_target_size": np.mean(target_sizes),
                "avg_actual_size": np.mean(actual_sizes),
                "size_variance": np.var(actual_sizes),
                "adaptation_effectiveness": np.corrcoef(target_sizes, actual_sizes)[0, 1] if len(target_sizes) > 1 else 0
            },
            "density_analysis": {
                "avg_entity_density": np.mean(entity_densities),
                "avg_claim_density": np.mean(claim_densities),
                "entity_density_range": [min(entity_densities), max(entity_densities)],
                "claim_density_range": [min(claim_densities), max(claim_densities)]
            },
            "adaptation_categories": {
                "high_density_chunks": len(high_density_chunks),
                "low_density_chunks": len(low_density_chunks),
                "normal_chunks": len(normal_chunks),
                "adaptation_rate": (len(high_density_chunks) + len(low_density_chunks)) / len(chunks)
            },
            "adaptive_score_stats": {
                "mean": np.mean(adaptive_scores),
                "std": np.std(adaptive_scores),
                "min": min(adaptive_scores),
                "max": max(adaptive_scores)
            }
        }


# Factory function
def create_adaptive_chunker(config: Dict[str, Any]) -> AdaptiveChunker:
    """Factory function pro Adaptive Chunker"""
    return AdaptiveChunker(config)
