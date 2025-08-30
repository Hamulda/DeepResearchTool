#!/usr/bin/env python3
"""Graph-Augmented RAG Module (Optional Feature)
Knowledge graph extraction and subgraph querying for enhanced consistency

Author: Senior IT Specialist
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import logging
import re
from typing import Any

import networkx as nx


@dataclass
class Entity:
    """Knowledge graph entity"""

    entity_id: str
    name: str
    entity_type: str  # "person", "organization", "concept", "location", "date"
    confidence: float = 0.0
    aliases: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    source_evidence: list[str] = field(default_factory=list)


@dataclass
class Relation:
    """Knowledge graph relation"""

    relation_id: str
    subject_id: str
    predicate: str
    object_id: str
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    temporal_info: str | None = None


@dataclass
class KnowledgeGraph:
    """Knowledge graph structure"""

    entities: dict[str, Entity] = field(default_factory=dict)
    relations: dict[str, Relation] = field(default_factory=dict)
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    metadata: dict[str, Any] = field(default_factory=dict)


class EntityExtractor:
    """Extracts entities from text using rule-based and pattern matching"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Entity patterns
        self.patterns = {
            "person": [
                r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b",  # John Smith
                r"\b(Dr\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+)\b",  # Dr. John Smith
                r"\b(Prof\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+)\b"  # Prof. John Smith
            ],
            "organization": [
                r"\b([A-Z][a-z]+\s+(?:University|Institute|Corporation|Company|Inc\.|Ltd\.|LLC))\b",
                r"\b(NASA|FDA|WHO|UN|EU|IBM|Google|Microsoft|Apple)\b",
                r"\b([A-Z]{2,5})\b"  # Acronyms
            ],
            "location": [
                r"\b([A-Z][a-z]+,\s+[A-Z][A-Z])\b",  # City, State
                r"\b(United States|European Union|United Kingdom)\b",
                r"\b([A-Z][a-z]+\s+(?:University|Hospital|Laboratory))\b"
            ],
            "date": [
                r"\b(\d{4})\b",  # Years
                r"\b(\d{1,2}/\d{1,2}/\d{4})\b",  # MM/DD/YYYY
                r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b"
            ],
            "concept": [
                r"\b(artificial intelligence|machine learning|quantum computing|climate change|COVID-19)\b",
                r"\b([a-z]+ology|[a-z]+ics|[a-z]+ism)\b"  # Academic fields
            ]
        }

    def extract_entities(self, text: str, evidence_id: str) -> list[Entity]:
        """Extract entities from text"""
        entities = []
        entity_count = defaultdict(int)

        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)

                for match in matches:
                    entity_text = match.group(1) if match.groups() else match.group(0)
                    entity_text = entity_text.strip()

                    if len(entity_text) < 2:
                        continue

                    # Generate entity ID
                    entity_id = f"{entity_type}_{hash(entity_text.lower()) % 100000}"

                    # Calculate confidence based on pattern specificity
                    confidence = self._calculate_entity_confidence(entity_text, entity_type)

                    entity = Entity(
                        entity_id=entity_id,
                        name=entity_text,
                        entity_type=entity_type,
                        confidence=confidence,
                        source_evidence=[evidence_id]
                    )

                    entities.append(entity)
                    entity_count[entity_type] += 1

        self.logger.debug(f"Extracted {sum(entity_count.values())} entities: {dict(entity_count)}")
        return entities

    def _calculate_entity_confidence(self, entity_text: str, entity_type: str) -> float:
        """Calculate confidence score for entity"""
        base_confidence = 0.5

        # Length-based confidence
        if len(entity_text) > 10:
            base_confidence += 0.2

        # Type-specific rules
        if entity_type == "person":
            if any(title in entity_text for title in ["Dr.", "Prof.", "Mr.", "Ms."]):
                base_confidence += 0.2
        elif entity_type == "organization":
            if any(suffix in entity_text for suffix in ["Inc.", "Ltd.", "LLC", "Corp."]):
                base_confidence += 0.3
        elif entity_type == "date":
            if re.match(r"\d{4}", entity_text):  # Year format
                base_confidence += 0.3

        return min(1.0, base_confidence)


class RelationExtractor:
    """Extracts relations between entities"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Relation patterns
        self.relation_patterns = [
            {
                "pattern": r"(.+?)\s+(?:works at|employed by|affiliated with)\s+(.+)",
                "predicate": "affiliated_with",
                "subject_type": "person",
                "object_type": "organization"
            },
            {
                "pattern": r"(.+?)\s+(?:founded|established|created)\s+(.+)",
                "predicate": "founded",
                "subject_type": "person",
                "object_type": "organization"
            },
            {
                "pattern": r"(.+?)\s+(?:located in|based in|from)\s+(.+)",
                "predicate": "located_in",
                "subject_type": "organization",
                "object_type": "location"
            },
            {
                "pattern": r"(.+?)\s+(?:published|released|announced)\s+(.+?)\s+in\s+(.+)",
                "predicate": "published_in",
                "subject_type": "person",
                "object_type": "date"
            }
        ]

    def extract_relations(self, text: str, entities: list[Entity], evidence_id: str) -> list[Relation]:
        """Extract relations from text using entities"""
        relations = []

        # Create entity lookup
        entity_lookup = {entity.name.lower(): entity for entity in entities}

        # Pattern-based relation extraction
        for relation_def in self.relation_patterns:
            pattern = relation_def["pattern"]
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    subject_text = groups[0].strip()
                    object_text = groups[1].strip()

                    # Find matching entities
                    subject_entity = entity_lookup.get(subject_text.lower())
                    object_entity = entity_lookup.get(object_text.lower())

                    if subject_entity and object_entity:
                        relation_id = f"rel_{hash(f'{subject_entity.entity_id}_{relation_def['predicate']}_{object_entity.entity_id}') % 100000}"

                        relation = Relation(
                            relation_id=relation_id,
                            subject_id=subject_entity.entity_id,
                            predicate=relation_def["predicate"],
                            object_id=object_entity.entity_id,
                            confidence=0.7,  # Pattern-based confidence
                            evidence=[evidence_id]
                        )

                        relations.append(relation)

        # Co-occurrence based relations
        co_occurrence_relations = self._extract_cooccurrence_relations(entities, text, evidence_id)
        relations.extend(co_occurrence_relations)

        self.logger.debug(f"Extracted {len(relations)} relations")
        return relations

    def _extract_cooccurrence_relations(self, entities: list[Entity], text: str, evidence_id: str) -> list[Relation]:
        """Extract relations based on entity co-occurrence"""
        relations = []

        # Simple co-occurrence within sentence
        sentences = re.split(r'[.!?]+', text)

        for sentence in sentences:
            sentence_entities = []

            for entity in entities:
                if entity.name.lower() in sentence.lower():
                    sentence_entities.append(entity)

            # Create co-occurrence relations
            for i, entity1 in enumerate(sentence_entities):
                for entity2 in sentence_entities[i+1:]:
                    if entity1.entity_type != entity2.entity_type:
                        relation_id = f"cooc_{hash(f'{entity1.entity_id}_mentions_{entity2.entity_id}') % 100000}"

                        relation = Relation(
                            relation_id=relation_id,
                            subject_id=entity1.entity_id,
                            predicate="co_mentioned_with",
                            object_id=entity2.entity_id,
                            confidence=0.4,  # Lower confidence for co-occurrence
                            evidence=[evidence_id]
                        )

                        relations.append(relation)

        return relations


class KnowledgeGraphBuilder:
    """Builds and manages knowledge graphs"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.entity_extractor = EntityExtractor()
        self.relation_extractor = RelationExtractor()

        # Configuration
        self.min_entity_confidence = config.get("min_entity_confidence", 0.5)
        self.min_relation_confidence = config.get("min_relation_confidence", 0.4)
        self.max_entities_per_doc = config.get("max_entities_per_doc", 50)

    def build_graph_from_documents(self, documents: list[dict[str, Any]]) -> KnowledgeGraph:
        """Build knowledge graph from document collection"""
        self.logger.info(f"Building knowledge graph from {len(documents)} documents")

        kg = KnowledgeGraph()
        all_entities = []
        all_relations = []

        for doc in documents:
            doc_id = doc.get("doc_id", f"doc_{hash(doc.get('content', ''))}")
            content = doc.get("content", doc.get("snippet", ""))

            if not content:
                continue

            # Extract entities
            entities = self.entity_extractor.extract_entities(content, doc_id)

            # Filter by confidence
            filtered_entities = [e for e in entities if e.confidence >= self.min_entity_confidence]
            filtered_entities = filtered_entities[:self.max_entities_per_doc]

            # Extract relations
            relations = self.relation_extractor.extract_relations(content, filtered_entities, doc_id)
            filtered_relations = [r for r in relations if r.confidence >= self.min_relation_confidence]

            all_entities.extend(filtered_entities)
            all_relations.extend(filtered_relations)

        # Merge duplicate entities
        kg.entities = self._merge_duplicate_entities(all_entities)

        # Add relations
        for relation in all_relations:
            if relation.subject_id in kg.entities and relation.object_id in kg.entities:
                kg.relations[relation.relation_id] = relation

        # Build NetworkX graph
        kg.graph = self._build_networkx_graph(kg.entities, kg.relations)

        # Add metadata
        kg.metadata = {
            "build_timestamp": datetime.now().isoformat(),
            "total_entities": len(kg.entities),
            "total_relations": len(kg.relations),
            "graph_density": nx.density(kg.graph) if kg.graph.number_of_nodes() > 1 else 0.0
        }

        self.logger.info(f"Built knowledge graph: {len(kg.entities)} entities, {len(kg.relations)} relations")
        return kg

    def _merge_duplicate_entities(self, entities: list[Entity]) -> dict[str, Entity]:
        """Merge duplicate entities by name similarity"""
        merged_entities = {}
        entity_groups = defaultdict(list)

        # Group entities by normalized name
        for entity in entities:
            normalized_name = entity.name.lower().strip()
            entity_groups[normalized_name].append(entity)

        # Merge groups
        for normalized_name, group in entity_groups.items():
            if len(group) == 1:
                merged_entities[group[0].entity_id] = group[0]
            else:
                # Merge multiple entities
                primary_entity = max(group, key=lambda e: e.confidence)

                # Combine evidence
                all_evidence = []
                for entity in group:
                    all_evidence.extend(entity.source_evidence)

                primary_entity.source_evidence = list(set(all_evidence))
                primary_entity.confidence = min(1.0, primary_entity.confidence + 0.1 * (len(group) - 1))

                merged_entities[primary_entity.entity_id] = primary_entity

        return merged_entities

    def _build_networkx_graph(self, entities: dict[str, Entity], relations: dict[str, Relation]) -> nx.DiGraph:
        """Build NetworkX graph from entities and relations"""
        graph = nx.DiGraph()

        # Add nodes
        for entity_id, entity in entities.items():
            graph.add_node(entity_id,
                          name=entity.name,
                          entity_type=entity.entity_type,
                          confidence=entity.confidence)

        # Add edges
        for relation_id, relation in relations.items():
            if relation.subject_id in graph and relation.object_id in graph:
                graph.add_edge(relation.subject_id, relation.object_id,
                              predicate=relation.predicate,
                              confidence=relation.confidence,
                              relation_id=relation_id)

        return graph


class GraphAugmentedRAG:
    """Graph-Augmented RAG system with subgraph querying"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Feature flag
        self.enabled = config.get("research", {}).get("graph", {}).get("enabled", False)

        if not self.enabled:
            self.logger.info("Graph-Augmented RAG disabled by feature flag")
            return

        self.kg_builder = KnowledgeGraphBuilder(config.get("graph", {}))
        self.knowledge_graph = None

        # Configuration
        self.subgraph_radius = config.get("graph", {}).get("subgraph_radius", 2)
        self.min_subgraph_nodes = config.get("graph", {}).get("min_subgraph_nodes", 3)

    async def enhance_retrieval_with_graph(self, query: str, retrieved_docs: list[dict[str, Any]]) -> dict[str, Any]:
        """Enhance retrieval using knowledge graph"""
        if not self.enabled:
            return {
                "enhanced_docs": retrieved_docs,
                "graph_info": {"enabled": False}
            }

        self.logger.info("Enhancing retrieval with knowledge graph")

        # Build knowledge graph from retrieved documents
        if not self.knowledge_graph:
            self.knowledge_graph = self.kg_builder.build_graph_from_documents(retrieved_docs)

        # Extract query entities
        query_entities = self.kg_builder.entity_extractor.extract_entities(query, "query")

        if not query_entities:
            self.logger.info("No entities found in query for graph enhancement")
            return {
                "enhanced_docs": retrieved_docs,
                "graph_info": {"entities_found": 0}
            }

        # Find relevant subgraph
        subgraph_nodes = self._find_relevant_subgraph(query_entities)

        # Enhance documents with graph context
        enhanced_docs = self._enhance_documents_with_graph_context(retrieved_docs, subgraph_nodes)

        # Calculate consistency score
        consistency_score = self._calculate_graph_consistency(enhanced_docs, subgraph_nodes)

        return {
            "enhanced_docs": enhanced_docs,
            "graph_info": {
                "enabled": True,
                "query_entities": len(query_entities),
                "subgraph_nodes": len(subgraph_nodes),
                "consistency_score": consistency_score,
                "total_graph_entities": len(self.knowledge_graph.entities),
                "total_graph_relations": len(self.knowledge_graph.relations)
            }
        }

    def _find_relevant_subgraph(self, query_entities: list[Entity]) -> set[str]:
        """Find relevant subgraph based on query entities"""
        if not self.knowledge_graph or not self.knowledge_graph.graph:
            return set()

        subgraph_nodes = set()

        # Find query entities in knowledge graph
        query_entity_ids = []
        for query_entity in query_entities:
            for kg_entity_id, kg_entity in self.knowledge_graph.entities.items():
                if query_entity.name.lower() in kg_entity.name.lower():
                    query_entity_ids.append(kg_entity_id)

        if not query_entity_ids:
            return set()

        # Expand subgraph around query entities
        for entity_id in query_entity_ids:
            if entity_id in self.knowledge_graph.graph:
                # Add entity and neighbors within radius
                neighbors = nx.single_source_shortest_path_length(
                    self.knowledge_graph.graph,
                    entity_id,
                    cutoff=self.subgraph_radius
                )
                subgraph_nodes.update(neighbors.keys())

        self.logger.debug(f"Found subgraph with {len(subgraph_nodes)} nodes")
        return subgraph_nodes

    def _enhance_documents_with_graph_context(self, docs: list[dict[str, Any]],
                                            subgraph_nodes: set[str]) -> list[dict[str, Any]]:
        """Enhance documents with graph context information"""
        enhanced_docs = []

        for doc in docs:
            enhanced_doc = doc.copy()

            # Find entities in document that are in subgraph
            doc_content = doc.get("content", doc.get("snippet", ""))
            doc_entities = []

            for node_id in subgraph_nodes:
                if node_id in self.knowledge_graph.entities:
                    entity = self.knowledge_graph.entities[node_id]
                    if entity.name.lower() in doc_content.lower():
                        doc_entities.append({
                            "entity_id": entity.entity_id,
                            "name": entity.name,
                            "type": entity.entity_type,
                            "confidence": entity.confidence
                        })

            # Add graph context
            enhanced_doc["graph_context"] = {
                "entities_in_subgraph": doc_entities,
                "graph_relevance_score": len(doc_entities) / max(len(subgraph_nodes), 1)
            }

            enhanced_docs.append(enhanced_doc)

        return enhanced_docs

    def _calculate_graph_consistency(self, enhanced_docs: list[dict[str, Any]],
                                   subgraph_nodes: set[str]) -> float:
        """Calculate consistency score based on graph structure"""
        if not subgraph_nodes or not enhanced_docs:
            return 0.0

        # Count entity-heavy documents
        entity_heavy_docs = 0
        total_graph_entities = 0

        for doc in enhanced_docs:
            graph_context = doc.get("graph_context", {})
            doc_entities = graph_context.get("entities_in_subgraph", [])

            if len(doc_entities) >= 2:  # Documents with multiple graph entities
                entity_heavy_docs += 1

            total_graph_entities += len(doc_entities)

        # Consistency score based on graph connectivity
        if len(enhanced_docs) == 0:
            return 0.0

        consistency = (entity_heavy_docs / len(enhanced_docs)) * 0.7 + \
                     (total_graph_entities / (len(enhanced_docs) * 5)) * 0.3  # Normalize by expected entities

        return min(1.0, consistency)


def create_graph_augmented_rag(config: dict[str, Any]) -> GraphAugmentedRAG:
    """Factory function for Graph-Augmented RAG"""
    return GraphAugmentedRAG(config)
