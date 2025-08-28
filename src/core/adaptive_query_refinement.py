#!/usr/bin/env python3
"""
Adaptive Query Refinement Loop
Iteratively extracts uncovered entities and generates sub-queries with confidence-based fan-out

Author: Senior IT Specialist
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
import json
import re
from datetime import datetime
import numpy as np

import structlog
from transformers import AutoTokenizer, AutoModelForTokenClassification
import spacy

logger = structlog.get_logger(__name__)

@dataclass
class QueryEntity:
    """Extracted entity from query"""
    text: str
    label: str  # PERSON, ORG, GPE, PRODUCT, etc.
    confidence: float
    start: int
    end: int

@dataclass
class SubQuery:
    """Generated sub-query for refinement"""
    text: str
    entities: List[QueryEntity]
    confidence: float
    parent_query: str
    iteration: int
    coverage_gain_potential: float

@dataclass
class CoverageAnalysis:
    """Analysis of coverage gaps in current results"""
    uncovered_entities: List[QueryEntity]
    missing_aspects: List[str]
    confidence_gaps: List[Dict[str, Any]]
    coverage_score: float
    potential_gain: float

class EntityExtractor:
    """NER-based entity extraction for query refinement"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("ner_model", "dbmdz/bert-large-cased-finetuned-conll03-english")

        # Initialize models
        self.tokenizer = None
        self.model = None
        self.nlp = None

    async def initialize(self):
        """Initialize NER models"""
        logger.info("Initializing entity extractor")

        # Load transformer NER model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)

        # Load spaCy for additional entity types
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, using basic extraction")
            self.nlp = None

    def extract_entities(self, text: str) -> List[QueryEntity]:
        """Extract entities from text using both transformer and spaCy"""
        entities = []

        # Extract using transformer model
        transformer_entities = self._extract_transformer_entities(text)
        entities.extend(transformer_entities)

        # Extract using spaCy (if available)
        if self.nlp:
            spacy_entities = self._extract_spacy_entities(text)
            entities.extend(spacy_entities)

        # Deduplicate and merge
        entities = self._deduplicate_entities(entities)

        return entities

    def _extract_transformer_entities(self, text: str) -> List[QueryEntity]:
        """Extract entities using transformer NER model"""

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Decode predictions
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_token_class = torch.argmax(predictions, dim=-1)

        # Convert to entities
        entities = []
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        current_entity = None
        current_tokens = []

        for i, (token, pred_id) in enumerate(zip(tokens, predicted_token_class[0])):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue

            label = self.model.config.id2label[pred_id.item()]

            if label.startswith("B-"):  # Beginning of entity
                if current_entity:
                    # Save previous entity
                    entities.append(self._create_entity_from_tokens(current_tokens, current_entity, text))

                current_entity = label[2:]  # Remove B- prefix
                current_tokens = [(token, i)]

            elif label.startswith("I-") and current_entity == label[2:]:  # Inside entity
                current_tokens.append((token, i))

            else:  # Outside entity
                if current_entity:
                    entities.append(self._create_entity_from_tokens(current_tokens, current_entity, text))
                    current_entity = None
                    current_tokens = []

        # Handle last entity
        if current_entity:
            entities.append(self._create_entity_from_tokens(current_tokens, current_entity, text))

        return entities

    def _extract_spacy_entities(self, text: str) -> List[QueryEntity]:
        """Extract entities using spaCy"""
        doc = self.nlp(text)

        entities = []
        for ent in doc.ents:
            entity = QueryEntity(
                text=ent.text,
                label=ent.label_,
                confidence=0.8,  # spaCy doesn't provide confidence scores
                start=ent.start_char,
                end=ent.end_char
            )
            entities.append(entity)

        return entities

    def _create_entity_from_tokens(self, tokens: List[Tuple[str, int]],
                                 entity_type: str, original_text: str) -> QueryEntity:
        """Create entity from transformer tokens"""

        # Reconstruct text from tokens
        entity_text = self.tokenizer.convert_tokens_to_string([t[0] for t in tokens])

        # Find position in original text
        start = original_text.find(entity_text)
        end = start + len(entity_text) if start != -1 else 0

        return QueryEntity(
            text=entity_text,
            label=entity_type,
            confidence=0.9,  # High confidence for transformer predictions
            start=max(0, start),
            end=end
        )

    def _deduplicate_entities(self, entities: List[QueryEntity]) -> List[QueryEntity]:
        """Remove duplicate entities with overlap resolution"""

        # Sort by start position
        entities.sort(key=lambda x: x.start)

        deduplicated = []
        for entity in entities:
            # Check for overlap with existing entities
            overlaps = False
            for existing in deduplicated:
                if (entity.start < existing.end and entity.end > existing.start):
                    # Overlap detected - keep the one with higher confidence
                    if entity.confidence > existing.confidence:
                        deduplicated.remove(existing)
                        deduplicated.append(entity)
                    overlaps = True
                    break

            if not overlaps:
                deduplicated.append(entity)

        return deduplicated

class QueryRefinementEngine:
    """Adaptive query refinement with coverage analysis"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.refinement_config = config.get("query_refinement", {})

        # Refinement settings
        self.max_iterations = self.refinement_config.get("max_iterations", 3)
        self.confidence_threshold = self.refinement_config.get("confidence_threshold", 0.7)
        self.coverage_plateau_threshold = self.refinement_config.get("plateau_threshold", 0.05)
        self.max_subqueries_per_iteration = self.refinement_config.get("max_subqueries", 3)

        # Components
        self.entity_extractor = EntityExtractor(config)

        # State tracking
        self.iteration_history = []
        self.coverage_history = []

        self.logger = structlog.get_logger(__name__)

    async def initialize(self):
        """Initialize refinement engine"""
        await self.entity_extractor.initialize()
        self.logger.info("Query refinement engine initialized")

    async def refine_query_iteratively(self, initial_query: str,
                                     initial_results: List[Dict[str, Any]],
                                     retrieval_engine) -> Dict[str, Any]:
        """Iteratively refine query until coverage plateau reached"""

        self.logger.info(f"Starting iterative refinement: {initial_query[:50]}...")

        # Initialize state
        current_query = initial_query
        all_results = initial_results.copy()
        current_iteration = 0

        refinement_log = {
            "initial_query": initial_query,
            "iterations": [],
            "final_coverage": 0.0,
            "total_subqueries": 0,
            "convergence_reason": ""
        }

        while current_iteration < self.max_iterations:
            self.logger.info(f"Refinement iteration {current_iteration + 1}")

            # Analyze coverage gaps
            coverage = await self._analyze_coverage(current_query, all_results)
            self.coverage_history.append(coverage.coverage_score)

            # Check for plateau
            if self._check_plateau():
                refinement_log["convergence_reason"] = "coverage_plateau"
                break

            # Generate sub-queries for uncovered aspects
            subqueries = await self._generate_subqueries(
                current_query,
                coverage.uncovered_entities,
                coverage.missing_aspects,
                current_iteration
            )

            if not subqueries:
                refinement_log["convergence_reason"] = "no_subqueries_generated"
                break

            # Execute sub-queries
            iteration_results = []
            for subquery in subqueries:
                if subquery.confidence >= self.confidence_threshold:
                    subquery_results = await retrieval_engine.hierarchical_search(
                        subquery.text,
                        top_k=5
                    )
                    iteration_results.extend(subquery_results)

            # Merge with existing results
            all_results = self._merge_results(all_results, iteration_results)

            # Log iteration
            iteration_log = {
                "iteration": current_iteration + 1,
                "coverage_before": coverage.coverage_score,
                "subqueries": [sq.text for sq in subqueries],
                "new_results_count": len(iteration_results),
                "total_results": len(all_results)
            }
            refinement_log["iterations"].append(iteration_log)

            current_iteration += 1

        # Final coverage analysis
        final_coverage = await self._analyze_coverage(current_query, all_results)
        refinement_log["final_coverage"] = final_coverage.coverage_score
        refinement_log["total_subqueries"] = sum(len(it["subqueries"]) for it in refinement_log["iterations"])

        if current_iteration >= self.max_iterations:
            refinement_log["convergence_reason"] = "max_iterations_reached"

        self.logger.info(f"Refinement completed: {refinement_log['convergence_reason']}")

        return {
            "refined_results": all_results,
            "refinement_log": refinement_log,
            "coverage_improvement": final_coverage.coverage_score - (self.coverage_history[0] if self.coverage_history else 0)
        }

    async def _analyze_coverage(self, query: str, results: List[Dict[str, Any]]) -> CoverageAnalysis:
        """Analyze coverage gaps in current results"""

        # Extract entities from original query
        query_entities = self.entity_extractor.extract_entities(query)

        # Extract entities from result content
        result_entities = set()
        for result in results:
            content = result.get("content", "")
            entities = self.entity_extractor.extract_entities(content)
            result_entities.update(e.text.lower() for e in entities)

        # Find uncovered entities
        uncovered_entities = []
        for entity in query_entities:
            if entity.text.lower() not in result_entities:
                uncovered_entities.append(entity)

        # Identify missing aspects (simplified heuristic)
        missing_aspects = self._identify_missing_aspects(query, results)

        # Calculate coverage score
        total_entities = len(query_entities)
        covered_entities = total_entities - len(uncovered_entities)
        coverage_score = covered_entities / total_entities if total_entities > 0 else 1.0

        # Calculate potential gain
        potential_gain = len(uncovered_entities) * 0.1 + len(missing_aspects) * 0.05

        return CoverageAnalysis(
            uncovered_entities=uncovered_entities,
            missing_aspects=missing_aspects,
            confidence_gaps=[],  # Could be enhanced with confidence analysis
            coverage_score=coverage_score,
            potential_gain=potential_gain
        )

    def _identify_missing_aspects(self, query: str, results: List[Dict[str, Any]]) -> List[str]:
        """Identify missing aspects using keyword analysis"""

        # Common research aspects
        aspect_keywords = {
            "methodology": ["method", "approach", "technique", "procedure"],
            "results": ["result", "finding", "outcome", "conclusion"],
            "limitations": ["limitation", "constraint", "drawback", "issue"],
            "future_work": ["future", "next", "recommend", "suggest"],
            "comparison": ["compare", "versus", "different", "alternative"]
        }

        # Check which aspects are missing from results
        result_text = " ".join(r.get("content", "") for r in results).lower()
        missing_aspects = []

        for aspect, keywords in aspect_keywords.items():
            if not any(keyword in result_text for keyword in keywords):
                missing_aspects.append(aspect)

        return missing_aspects

    async def _generate_subqueries(self, original_query: str,
                                 uncovered_entities: List[QueryEntity],
                                 missing_aspects: List[str],
                                 iteration: int) -> List[SubQuery]:
        """Generate sub-queries for uncovered entities and aspects"""

        subqueries = []

        # Generate entity-based sub-queries
        for entity in uncovered_entities[:2]:  # Limit to top 2 entities
            subquery_text = f"{original_query} {entity.text}"

            subquery = SubQuery(
                text=subquery_text,
                entities=[entity],
                confidence=entity.confidence,
                parent_query=original_query,
                iteration=iteration,
                coverage_gain_potential=0.2
            )
            subqueries.append(subquery)

        # Generate aspect-based sub-queries
        aspect_templates = {
            "methodology": f"methodology approach {original_query}",
            "results": f"results findings {original_query}",
            "limitations": f"limitations issues {original_query}",
            "comparison": f"comparison alternatives {original_query}"
        }

        for aspect in missing_aspects[:2]:  # Limit to top 2 aspects
            if aspect in aspect_templates:
                subquery_text = aspect_templates[aspect]

                subquery = SubQuery(
                    text=subquery_text,
                    entities=[],
                    confidence=0.7,  # Medium confidence for aspect queries
                    parent_query=original_query,
                    iteration=iteration,
                    coverage_gain_potential=0.15
                )
                subqueries.append(subquery)

        # Sort by coverage gain potential and confidence
        subqueries.sort(key=lambda sq: (sq.coverage_gain_potential, sq.confidence), reverse=True)

        return subqueries[:self.max_subqueries_per_iteration]

    def _check_plateau(self) -> bool:
        """Check if coverage improvement has plateaued"""

        if len(self.coverage_history) < 2:
            return False

        # Calculate improvement in last iteration
        last_improvement = self.coverage_history[-1] - self.coverage_history[-2]

        return last_improvement < self.coverage_plateau_threshold

    def _merge_results(self, existing_results: List[Dict[str, Any]],
                      new_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge new results with existing ones, avoiding duplicates"""

        # Create set of existing URLs for deduplication
        existing_urls = {r.get("url", "") for r in existing_results}

        # Add new unique results
        merged = existing_results.copy()
        for result in new_results:
            url = result.get("url", "")
            if url not in existing_urls:
                merged.append(result)
                existing_urls.add(url)

        # Sort by score
        merged.sort(key=lambda r: r.get("score", 0), reverse=True)

        return merged

def create_query_refinement_engine(config: Dict[str, Any]) -> QueryRefinementEngine:
    """Factory function for query refinement engine"""
    return QueryRefinementEngine(config)
