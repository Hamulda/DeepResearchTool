#!/usr/bin/env python3
"""Correlation Engine - Korelační motor a graf vztahů
NER pomocí spaCy a budování grafů vztahů pomocí networkx

Author: GitHub Copilot
Created: August 28, 2025 - Phase 3 Implementation
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import logging
import re
from typing import Any

try:
    import spacy
    from spacy import displacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Reprezentace pojmenované entity"""

    text: str
    label: str
    start: int
    end: int
    confidence: float
    source_doc: str
    context: str
    normalized_form: str = ""
    entity_id: str = field(init=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Generování unikátního ID pro entitu
        self.entity_id = hashlib.md5(
            f"{self.normalized_form or self.text}:{self.label}".encode()
        ).hexdigest()[:12]


@dataclass
class Relationship:
    """Reprezentace vztahu mezi entitami"""

    entity1: Entity
    entity2: Entity
    relationship_type: str
    strength: float
    co_occurrence_count: int
    contexts: list[str]
    source_docs: set[str]
    proximity_score: float
    relationship_id: str = field(init=False)

    def __post_init__(self):
        # Generování ID vztahu
        self.relationship_id = hashlib.md5(
            f"{self.entity1.entity_id}:{self.entity2.entity_id}:{self.relationship_type}".encode()
        ).hexdigest()[:12]


@dataclass
class EntityCluster:
    """Shluk souvisejících entit"""

    cluster_id: str
    entities: list[Entity]
    cluster_type: str
    centrality_scores: dict[str, float]
    internal_connections: int
    external_connections: int
    cluster_strength: float


@dataclass
class NetworkAnalysisResult:
    """Výsledek analýzy síťových vztahů"""

    total_entities: int
    total_relationships: int
    network_density: float
    connected_components: int
    largest_component_size: int
    key_entities: list[Entity]
    important_relationships: list[Relationship]
    entity_clusters: list[EntityCluster]
    network_metrics: dict[str, float]
    anomalous_patterns: list[dict[str, Any]]


class CorrelationEngine:
    """Pokročilý korelační motor pro zpravodajskou analýzu"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.correlation_config = config.get("correlation", {})

        # NLP modely
        self.spacy_model = None
        self.nlp_model_name = self.correlation_config.get("spacy_model", "en_core_web_sm")

        # Graf vztahů
        self.entity_graph = nx.Graph()
        self.relationship_cache: dict[str, list[Relationship]] = {}

        # Entity storage a indexy
        self.entities: dict[str, Entity] = {}
        self.entity_index: dict[str, set[str]] = defaultdict(set)
        self.document_entities: dict[str, list[Entity]] = defaultdict(list)

        # Konfigurace detekce vztahů
        self.proximity_threshold = self.correlation_config.get("proximity_threshold", 100)
        self.min_co_occurrence = self.correlation_config.get("min_co_occurrence", 2)
        self.entity_similarity_threshold = self.correlation_config.get(
            "entity_similarity_threshold", 0.8
        )

        # Statistiky
        self.analysis_stats = defaultdict(int)

        self._initialize_nlp()

    def _initialize_nlp(self):
        """Inicializace NLP komponent"""
        if not SPACY_AVAILABLE:
            logger.warning("spaCy not available - NER analysis will be limited")
            return

        try:
            self.spacy_model = spacy.load(self.nlp_model_name)
            logger.info(f"Loaded spaCy model: {self.nlp_model_name}")
        except OSError:
            logger.warning(f"spaCy model {self.nlp_model_name} not found, trying 'en_core_web_sm'")
            try:
                self.spacy_model = spacy.load("en_core_web_sm")
                logger.info("Loaded fallback spaCy model: en_core_web_sm")
            except OSError:
                logger.error(
                    "No spaCy model available - install with: python -m spacy download en_core_web_sm"
                )
                self.spacy_model = None

    async def extract_entities(self, text: str, source_doc: str = "") -> list[Entity]:
        """Extrakce pojmenovaných entit z textu"""
        if not self.spacy_model:
            logger.warning("No spaCy model available for entity extraction")
            return []

        entities = []

        try:
            # spaCy NER
            doc = self.spacy_model(text)

            for ent in doc.ents:
                # Context kolem entity
                context_start = max(0, ent.start_char - 50)
                context_end = min(len(text), ent.end_char + 50)
                context = text[context_start:context_end]

                # Normalizace entity
                normalized = await self._normalize_entity(ent.text, ent.label_)

                entity = Entity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=float(ent._.get("confidence", 0.8)),  # Default confidence
                    source_doc=source_doc,
                    context=context,
                    normalized_form=normalized,
                    metadata={
                        "extraction_method": "spacy_ner",
                        "sentence": ent.sent.text if ent.sent else "",
                        "pos_tags": [token.pos_ for token in ent],
                        "dep_tags": [token.dep_ for token in ent],
                    },
                )

                entities.append(entity)

                # Indexování
                self.entities[entity.entity_id] = entity
                self.entity_index[entity.label].add(entity.entity_id)
                self.document_entities[source_doc].append(entity)

            # Doplňkové entity pomocí regulárních výrazů
            regex_entities = await self._extract_regex_entities(text, source_doc)
            entities.extend(regex_entities)

            # Statistiky
            self.analysis_stats[f"entities_extracted_{source_doc}"] = len(entities)
            self.analysis_stats["total_entities_extracted"] += len(entities)

            logger.info(f"Extracted {len(entities)} entities from {source_doc}")

        except Exception as e:
            logger.error(f"Error extracting entities from {source_doc}: {e}")

        return entities

    async def _normalize_entity(self, text: str, label: str) -> str:
        """Normalizace entity pro lepší srovnávání"""
        normalized = text.strip().lower()

        # Specifické normalizace podle typu entity
        if label == "PERSON":
            # Normalizace jmen osob
            normalized = re.sub(r"\b(mr|mrs|ms|dr|prof)\b\.?", "", normalized)
            normalized = re.sub(r"\s+", " ", normalized).strip()

        elif label == "ORG":
            # Normalizace organizací
            normalized = re.sub(r"\b(inc|ltd|llc|corp|co)\b\.?", "", normalized)
            normalized = re.sub(r"\s+", " ", normalized).strip()

        elif label in ["GPE", "LOC"]:
            # Normalizace míst
            normalized = re.sub(r"\b(city|town|village|state|country)\b", "", normalized)
            normalized = re.sub(r"\s+", " ", normalized).strip()

        elif label == "MONEY":
            # Normalizace peněžních částek
            normalized = re.sub(r"[,$]", "", normalized)
            normalized = re.sub(r"\s+", " ", normalized).strip()

        return normalized

    async def _extract_regex_entities(self, text: str, source_doc: str) -> list[Entity]:
        """Extrakce entit pomocí regulárních výrazů pro specifické artefakty"""
        regex_entities = []

        # Vzory pro specifické entity
        patterns = {
            "CRYPTO_ADDRESS": r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b|0x[a-fA-F0-9]{40}\b",
            "ONION_ADDRESS": r"\b[a-z2-7]{16,56}\.onion\b",
            "IP_ADDRESS": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
            "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "PHONE": r"\b\+?[1-9]\d{1,14}\b|\b\d{3}-\d{3}-\d{4}\b",
            "HASH": r"\b[a-fA-F0-9]{32,128}\b",
        }

        for label, pattern in patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                context_start = max(0, match.start() - 50)
                context_end = min(len(text), match.end() + 50)
                context = text[context_start:context_end]

                entity = Entity(
                    text=match.group(),
                    label=label,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9,  # Vysoká confidence pro regex
                    source_doc=source_doc,
                    context=context,
                    normalized_form=match.group().lower(),
                    metadata={"extraction_method": "regex", "pattern": pattern},
                )

                regex_entities.append(entity)
                self.entities[entity.entity_id] = entity
                self.entity_index[entity.label].add(entity.entity_id)
                self.document_entities[source_doc].append(entity)

        return regex_entities

    async def build_relationship_graph(self, documents: dict[str, str]) -> nx.Graph:
        """Budování grafu vztahů mezi entitami"""
        # Reset grafu
        self.entity_graph.clear()

        # Extrakce entit ze všech dokumentů
        all_entities = []
        for doc_id, text in documents.items():
            entities = await self.extract_entities(text, doc_id)
            all_entities.extend(entities)

        # Přidání uzlů do grafu
        for entity in all_entities:
            self.entity_graph.add_node(
                entity.entity_id,
                text=entity.text,
                label=entity.label,
                confidence=entity.confidence,
                source_docs=entity.source_doc,
                normalized=entity.normalized_form,
            )

        # Detekce vztahů
        relationships = await self._detect_relationships(all_entities)

        # Přidání hran do grafu
        for relationship in relationships:
            self.entity_graph.add_edge(
                relationship.entity1.entity_id,
                relationship.entity2.entity_id,
                relationship_type=relationship.relationship_type,
                strength=relationship.strength,
                co_occurrence=relationship.co_occurrence_count,
                contexts=relationship.contexts[:3],  # Omezeně pro velikost
                proximity=relationship.proximity_score,
            )

        logger.info(
            f"Built relationship graph with {self.entity_graph.number_of_nodes()} nodes and {self.entity_graph.number_of_edges()} edges"
        )

        return self.entity_graph

    async def _detect_relationships(self, entities: list[Entity]) -> list[Relationship]:
        """Detekce vztahů mezi entitami"""
        relationships = []

        # Grupování entit podle dokumentů
        doc_entities = defaultdict(list)
        for entity in entities:
            doc_entities[entity.source_doc].append(entity)

        # Detekce vztahů v rámci dokumentů
        for doc_id, doc_entities_list in doc_entities.items():
            doc_relationships = await self._detect_document_relationships(doc_entities_list)
            relationships.extend(doc_relationships)

        # Detekce vztahů mezi dokumenty
        cross_doc_relationships = await self._detect_cross_document_relationships(entities)
        relationships.extend(cross_doc_relationships)

        return relationships

    async def _detect_document_relationships(self, entities: list[Entity]) -> list[Relationship]:
        """Detekce vztahů v rámci jednoho dokumentu"""
        relationships = []

        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i + 1 :], i + 1):
                if entity1.entity_id == entity2.entity_id:
                    continue

                # Výpočet proximity
                distance = abs(entity1.start - entity2.start)
                proximity_score = max(0, 1 - (distance / self.proximity_threshold))

                if proximity_score > 0.1:  # Threshold pro vztah
                    relationship_type = await self._determine_relationship_type(entity1, entity2)

                    relationship = Relationship(
                        entity1=entity1,
                        entity2=entity2,
                        relationship_type=relationship_type,
                        strength=proximity_score,
                        co_occurrence_count=1,
                        contexts=[entity1.context, entity2.context],
                        source_docs={entity1.source_doc},
                        proximity_score=proximity_score,
                    )

                    relationships.append(relationship)

        return relationships

    async def _detect_cross_document_relationships(
        self, entities: list[Entity]
    ) -> list[Relationship]:
        """Detekce vztahů mezi dokumenty"""
        relationships = []

        # Index entit podle normalizované formy
        normalized_index = defaultdict(list)
        for entity in entities:
            normalized_index[entity.normalized_form].append(entity)

        # Hledání stejných entit v různých dokumentech
        for normalized_form, entity_list in normalized_index.items():
            if len(entity_list) > 1:
                # Skupiny entit ze stejných dokumentů
                doc_groups = defaultdict(list)
                for entity in entity_list:
                    doc_groups[entity.source_doc].append(entity)

                # Vytváření vztahů mezi dokumenty
                doc_ids = list(doc_groups.keys())
                for i, doc1 in enumerate(doc_ids):
                    for doc2 in doc_ids[i + 1 :]:
                        for entity1 in doc_groups[doc1]:
                            for entity2 in doc_groups[doc2]:
                                similarity = await self._calculate_entity_similarity(
                                    entity1, entity2
                                )

                                if similarity > self.entity_similarity_threshold:
                                    relationship = Relationship(
                                        entity1=entity1,
                                        entity2=entity2,
                                        relationship_type="cross_document_mention",
                                        strength=similarity,
                                        co_occurrence_count=1,
                                        contexts=[entity1.context, entity2.context],
                                        source_docs={entity1.source_doc, entity2.source_doc},
                                        proximity_score=0.0,  # Cross-document, no proximity
                                    )

                                    relationships.append(relationship)

        return relationships

    async def _determine_relationship_type(self, entity1: Entity, entity2: Entity) -> str:
        """Určení typu vztahu mezi entitami"""
        label1, label2 = entity1.label, entity2.label

        # Pravidla pro typy vztahů
        if label1 == "PERSON" and label2 == "ORG":
            return "person_organization"
        if label1 == "ORG" and label2 == "PERSON":
            return "organization_person"
        if label1 == "PERSON" and label2 == "PERSON":
            return "person_person"
        if label1 == "ORG" and label2 == "ORG":
            return "organization_organization"
        if label1 in ["GPE", "LOC"] and label2 == "PERSON":
            return "location_person"
        if label1 == "PERSON" and label2 in ["GPE", "LOC"]:
            return "person_location"
        if label1 in ["GPE", "LOC"] and label2 == "ORG":
            return "location_organization"
        if label1 == "ORG" and label2 in ["GPE", "LOC"]:
            return "organization_location"
        if "CRYPTO" in label1 or "CRYPTO" in label2:
            return "crypto_related"
        if "ONION" in label1 or "ONION" in label2:
            return "darknet_related"
        if label1 == "EMAIL" or label2 == "EMAIL":
            return "communication_related"
        return "co_occurrence"

    async def _calculate_entity_similarity(self, entity1: Entity, entity2: Entity) -> float:
        """Výpočet podobnosti mezi entitami"""
        # Základní textová podobnost
        text_similarity = self._calculate_text_similarity(
            entity1.normalized_form, entity2.normalized_form
        )

        # Bonus za stejný typ entity
        type_bonus = 0.2 if entity1.label == entity2.label else 0.0

        # Kontextová podobnost
        context_similarity = self._calculate_text_similarity(entity1.context, entity2.context) * 0.3

        total_similarity = text_similarity + type_bonus + context_similarity
        return min(1.0, total_similarity)

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Výpočet textové podobnosti"""
        if not text1 or not text2:
            return 0.0

        # Jednoduchá Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    async def analyze_network(self) -> NetworkAnalysisResult:
        """Analýza síťových vlastností grafu vztahů"""
        if not NETWORKX_AVAILABLE:
            logger.error("NetworkX not available for network analysis")
            return NetworkAnalysisResult(
                total_entities=0,
                total_relationships=0,
                network_density=0.0,
                connected_components=0,
                largest_component_size=0,
                key_entities=[],
                important_relationships=[],
                entity_clusters=[],
                network_metrics={},
                anomalous_patterns=[],
            )

        graph = self.entity_graph

        # Základní metriky
        total_entities = graph.number_of_nodes()
        total_relationships = graph.number_of_edges()
        network_density = nx.density(graph)

        # Komponenty souvislosti
        components = list(nx.connected_components(graph))
        connected_components = len(components)
        largest_component_size = len(max(components, key=len)) if components else 0

        # Centrality metriky
        centrality_metrics = {}
        if total_entities > 0:
            try:
                centrality_metrics["degree"] = nx.degree_centrality(graph)
                centrality_metrics["betweenness"] = nx.betweenness_centrality(graph)
                centrality_metrics["closeness"] = nx.closeness_centrality(graph)
                centrality_metrics["pagerank"] = nx.pagerank(graph)
            except Exception as e:
                logger.warning(f"Error calculating centrality metrics: {e}")

        # Klíčové entity
        key_entities = await self._identify_key_entities(centrality_metrics)

        # Důležité vztahy
        important_relationships = await self._identify_important_relationships()

        # Clustering
        entity_clusters = await self._detect_entity_clusters()

        # Síťové metriky
        network_metrics = {
            "density": network_density,
            "clustering_coefficient": nx.average_clustering(graph) if total_entities > 0 else 0.0,
            "diameter": self._safe_diameter(graph),
            "average_path_length": self._safe_average_path_length(graph),
            "assortativity": self._safe_assortativity(graph),
        }

        # Detekce anomálních vzorů
        anomalous_patterns = await self._detect_anomalous_patterns()

        return NetworkAnalysisResult(
            total_entities=total_entities,
            total_relationships=total_relationships,
            network_density=network_density,
            connected_components=connected_components,
            largest_component_size=largest_component_size,
            key_entities=key_entities,
            important_relationships=important_relationships,
            entity_clusters=entity_clusters,
            network_metrics=network_metrics,
            anomalous_patterns=anomalous_patterns,
        )

    async def _identify_key_entities(
        self, centrality_metrics: dict[str, dict[str, float]]
    ) -> list[Entity]:
        """Identifikace klíčových entit podle centrality"""
        key_entities = []

        if not centrality_metrics:
            return key_entities

        # Kombinace různých centrality měr
        combined_scores = defaultdict(float)
        weights = {"degree": 0.3, "betweenness": 0.3, "closeness": 0.2, "pagerank": 0.2}

        for metric, weight in weights.items():
            if metric in centrality_metrics:
                for entity_id, score in centrality_metrics[metric].items():
                    combined_scores[entity_id] += weight * score

        # Top entity podle kombinovaného skóre
        sorted_entities = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        for entity_id, score in sorted_entities[:10]:  # Top 10
            if entity_id in self.entities:
                entity = self.entities[entity_id]
                entity.metadata["centrality_score"] = score
                key_entities.append(entity)

        return key_entities

    async def _identify_important_relationships(self) -> list[Relationship]:
        """Identifikace důležitých vztahů"""
        important_relationships = []

        # Analýza hran podle síly a typu
        for edge in self.entity_graph.edges(data=True):
            entity1_id, entity2_id, edge_data = edge

            if entity1_id in self.entities and entity2_id in self.entities:
                entity1 = self.entities[entity1_id]
                entity2 = self.entities[entity2_id]

                relationship = Relationship(
                    entity1=entity1,
                    entity2=entity2,
                    relationship_type=edge_data.get("relationship_type", "unknown"),
                    strength=edge_data.get("strength", 0.0),
                    co_occurrence_count=edge_data.get("co_occurrence", 1),
                    contexts=edge_data.get("contexts", []),
                    source_docs=set([entity1.source_doc, entity2.source_doc]),
                    proximity_score=edge_data.get("proximity", 0.0),
                )

                # Skórování vztahu
                importance_score = (
                    relationship.strength * 0.4
                    + min(1.0, relationship.co_occurrence_count / 5.0) * 0.3
                    + len(relationship.source_docs) / 10.0 * 0.3
                )

                if importance_score > 0.5:
                    relationship.metadata = {"importance_score": importance_score}
                    important_relationships.append(relationship)

        # Seřazení podle důležitosti
        important_relationships.sort(
            key=lambda r: r.metadata.get("importance_score", 0), reverse=True
        )

        return important_relationships[:20]  # Top 20

    async def _detect_entity_clusters(self) -> list[EntityCluster]:
        """Detekce shluků entit"""
        clusters = []

        if not NETWORKX_AVAILABLE or self.entity_graph.number_of_nodes() == 0:
            return clusters

        try:
            # Community detection
            communities = nx.community.louvain_communities(self.entity_graph)

            for i, community in enumerate(communities):
                if len(community) >= 3:  # Minimální velikost shluku
                    cluster_entities = [
                        self.entities[entity_id]
                        for entity_id in community
                        if entity_id in self.entities
                    ]

                    # Analýza shluku
                    subgraph = self.entity_graph.subgraph(community)
                    internal_connections = subgraph.number_of_edges()

                    # Externí připojení
                    external_connections = 0
                    for node in community:
                        for neighbor in self.entity_graph.neighbors(node):
                            if neighbor not in community:
                                external_connections += 1

                    # Centrality v rámci shluku
                    centrality_scores = {}
                    if len(community) > 2:
                        try:
                            centrality_scores = nx.degree_centrality(subgraph)
                        except:
                            pass

                    # Cluster strength
                    cluster_strength = internal_connections / (
                        internal_connections + external_connections + 1
                    )

                    # Určení typu shluku
                    cluster_type = self._determine_cluster_type(cluster_entities)

                    cluster = EntityCluster(
                        cluster_id=f"cluster_{i}",
                        entities=cluster_entities,
                        cluster_type=cluster_type,
                        centrality_scores=centrality_scores,
                        internal_connections=internal_connections,
                        external_connections=external_connections,
                        cluster_strength=cluster_strength,
                    )

                    clusters.append(cluster)

        except Exception as e:
            logger.warning(f"Error in community detection: {e}")

        return clusters

    def _determine_cluster_type(self, entities: list[Entity]) -> str:
        """Určení typu shluku na základě entit"""
        label_counts = Counter(entity.label for entity in entities)
        most_common_label = label_counts.most_common(1)[0][0]

        if most_common_label == "PERSON":
            return "person_network"
        if most_common_label == "ORG":
            return "organization_network"
        if most_common_label in ["GPE", "LOC"]:
            return "location_network"
        if "CRYPTO" in most_common_label:
            return "cryptocurrency_network"
        if "ONION" in most_common_label:
            return "darknet_network"
        return "mixed_network"

    async def _detect_anomalous_patterns(self) -> list[dict[str, Any]]:
        """Detekce anomálních vzorů v síti"""
        anomalies = []

        graph = self.entity_graph

        # Anomálie v centrality
        if graph.number_of_nodes() > 0:
            degree_centrality = nx.degree_centrality(graph)
            avg_centrality = statistics.mean(degree_centrality.values()) if degree_centrality else 0
            std_centrality = (
                statistics.stdev(degree_centrality.values()) if len(degree_centrality) > 1 else 0
            )

            # Uzly s extrémně vysokou centralitou
            for node_id, centrality in degree_centrality.items():
                if centrality > avg_centrality + 2 * std_centrality and centrality > 0.1:
                    if node_id in self.entities:
                        entity = self.entities[node_id]
                        anomalies.append(
                            {
                                "type": "high_centrality_node",
                                "entity": entity.text,
                                "entity_type": entity.label,
                                "centrality_score": centrality,
                                "description": f"Entity with unusually high centrality: {centrality:.3f}",
                            }
                        )

        # Anomálie v připojení
        degree_sequence = [d for n, d in graph.degree()]
        if degree_sequence:
            avg_degree = statistics.mean(degree_sequence)

            for node, degree in graph.degree():
                if degree > avg_degree * 3 and degree > 5:  # Velmi vysoký stupeň
                    if node in self.entities:
                        entity = self.entities[node]
                        anomalies.append(
                            {
                                "type": "high_degree_node",
                                "entity": entity.text,
                                "entity_type": entity.label,
                                "degree": degree,
                                "description": f"Entity with unusually many connections: {degree}",
                            }
                        )

        # Izolované komponenty
        components = list(nx.connected_components(graph))
        for i, component in enumerate(components):
            if 3 <= len(component) <= 10:  # Malé izolované skupiny
                component_entities = [
                    self.entities[node_id].text for node_id in component if node_id in self.entities
                ]
                anomalies.append(
                    {
                        "type": "isolated_component",
                        "entities": component_entities,
                        "size": len(component),
                        "description": f"Isolated group of {len(component)} entities",
                    }
                )

        return anomalies

    def _safe_diameter(self, graph: nx.Graph) -> float:
        """Bezpečný výpočet průměru grafu"""
        try:
            if graph.number_of_nodes() == 0:
                return 0.0
            if nx.is_connected(graph):
                return nx.diameter(graph)
            # Pro nesouvislý graf vezmi největší komponentu
            largest_component = max(nx.connected_components(graph), key=len)
            subgraph = graph.subgraph(largest_component)
            return nx.diameter(subgraph) if len(largest_component) > 1 else 0.0
        except:
            return 0.0

    def _safe_average_path_length(self, graph: nx.Graph) -> float:
        """Bezpečný výpočet průměrné délky cesty"""
        try:
            if graph.number_of_nodes() == 0:
                return 0.0
            if nx.is_connected(graph):
                return nx.average_shortest_path_length(graph)
            # Pro nesouvislý graf vezmi největší komponentu
            largest_component = max(nx.connected_components(graph), key=len)
            subgraph = graph.subgraph(largest_component)
            return (
                nx.average_shortest_path_length(subgraph) if len(largest_component) > 1 else 0.0
            )
        except:
            return 0.0

    def _safe_assortativity(self, graph: nx.Graph) -> float:
        """Bezpečný výpočet assortativity"""
        try:
            if graph.number_of_edges() == 0:
                return 0.0
            return nx.degree_assortativity_coefficient(graph)
        except:
            return 0.0

    def get_correlation_statistics(self) -> dict[str, Any]:
        """Získání statistik korelační analýzy"""
        return {
            "total_entities": len(self.entities),
            "entity_types": dict(Counter(entity.label for entity in self.entities.values())),
            "documents_processed": len(self.document_entities),
            "graph_nodes": self.entity_graph.number_of_nodes(),
            "graph_edges": self.entity_graph.number_of_edges(),
            "analysis_stats": dict(self.analysis_stats),
            "spacy_available": SPACY_AVAILABLE,
            "networkx_available": NETWORKX_AVAILABLE,
            "cache_size": len(self.relationship_cache),
        }

    async def export_graph(self, format: str = "gexf", filepath: str = "entity_graph") -> str:
        """Export grafu do různých formátů"""
        if not NETWORKX_AVAILABLE:
            raise ValueError("NetworkX not available for graph export")

        output_path = f"{filepath}.{format}"

        try:
            if format.lower() == "gexf":
                nx.write_gexf(self.entity_graph, output_path)
            elif format.lower() == "graphml":
                nx.write_graphml(self.entity_graph, output_path)
            elif format.lower() == "json":
                graph_data = nx.node_link_data(self.entity_graph)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(graph_data, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Graph exported to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error exporting graph: {e}")
            raise

    async def generate_correlation_report(
        self, analysis_result: NetworkAnalysisResult
    ) -> dict[str, Any]:
        """Generování komplexní zprávy o korelační analýze"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "network_overview": {
                "total_entities": analysis_result.total_entities,
                "total_relationships": analysis_result.total_relationships,
                "network_density": analysis_result.network_density,
                "connected_components": analysis_result.connected_components,
                "largest_component_size": analysis_result.largest_component_size,
            },
            "key_findings": {
                "most_central_entities": [
                    {
                        "text": entity.text,
                        "type": entity.label,
                        "centrality_score": entity.metadata.get("centrality_score", 0),
                        "source_docs": entity.source_doc,
                    }
                    for entity in analysis_result.key_entities[:5]
                ],
                "strongest_relationships": [
                    {
                        "entity1": rel.entity1.text,
                        "entity2": rel.entity2.text,
                        "relationship_type": rel.relationship_type,
                        "strength": rel.strength,
                        "co_occurrence": rel.co_occurrence_count,
                    }
                    for rel in analysis_result.important_relationships[:5]
                ],
                "entity_clusters": [
                    {
                        "cluster_id": cluster.cluster_id,
                        "cluster_type": cluster.cluster_type,
                        "size": len(cluster.entities),
                        "strength": cluster.cluster_strength,
                        "key_entities": [entity.text for entity in cluster.entities[:3]],
                    }
                    for cluster in analysis_result.entity_clusters[:5]
                ],
            },
            "anomalous_patterns": analysis_result.anomalous_patterns,
            "network_metrics": analysis_result.network_metrics,
            "statistics": self.get_correlation_statistics(),
        }

        return report

    async def analyze_text(self, text: str) -> dict[str, Any]:
        """Veřejná metoda pro analýzu textu a extrakci entit
        Používá se z AgenticLoop a dalších komponent

        Args:
            text: Text k analýze

        Returns:
            Dict s nalezenými entitami a vztahy

        """
        try:
            # Generování unikátního ID dokumentu
            doc_id = f"doc_{hash(text) % 10000}"

            # Extrakce entit
            entities = await self.extract_entities(text, doc_id)

            # Detekce vztahů mezi entitami
            relationships = await self.detect_relationships(entities)

            # Přidání do grafu znalostí
            for entity in entities:
                self._add_entity_to_graph(entity)

            for relationship in relationships:
                self._add_relationship_to_graph(relationship)

            # Příprava výsledku
            result = {
                "entities": [
                    {
                        "id": entity.entity_id,
                        "text": entity.text,
                        "label": entity.label,
                        "confidence": entity.confidence,
                        "normalized_form": entity.normalized_form,
                    }
                    for entity in entities
                ],
                "relationships": [
                    {
                        "source": rel.source_entity,
                        "target": rel.target_entity,
                        "relation_type": rel.relation_type,
                        "confidence": rel.confidence,
                    }
                    for rel in relationships
                ],
                "document_id": doc_id,
                "entities_count": len(entities),
                "relationships_count": len(relationships),
            }

            return result

        except Exception as e:
            logger.error(f"Chyba při analýze textu: {e}")
            return {
                "entities": [],
                "relationships": [],
                "document_id": f"error_{hash(text) % 1000}",
                "entities_count": 0,
                "relationships_count": 0,
                "error": str(e),
            }

    async def find_related_entities(self, entity_id: str) -> list[dict[str, Any]]:
        """Najde entity související s danou entitou
        Používá se z AgenticLoop pro korelační analýzu

        Args:
            entity_id: ID entity pro kterou hledáme relacionované entity

        Returns:
            List slovníků s relacionovanými entitami

        """
        try:
            if not hasattr(self, "knowledge_graph") or entity_id not in self.knowledge_graph:
                return []

            related = []

            # Najdi všechny sousední uzly v grafu
            if hasattr(self.knowledge_graph, "neighbors"):
                neighbors = list(self.knowledge_graph.neighbors(entity_id))

                for neighbor_id in neighbors:
                    # Získej data o entitě
                    entity_data = self.entities.get(neighbor_id)
                    if entity_data:
                        # Získej informace o vztahu
                        edge_data = self.knowledge_graph.get_edge_data(entity_id, neighbor_id, {})

                        related.append(
                            {
                                "entity_id": neighbor_id,
                                "text": entity_data.text,
                                "label": entity_data.label,
                                "confidence": entity_data.confidence,
                                "relation_type": edge_data.get("relation_type", "related"),
                                "relation_confidence": edge_data.get("confidence", 0.5),
                                "source_docs": edge_data.get("source_docs", []),
                            }
                        )

            # Seřazení podle confidence
            related.sort(key=lambda x: x["relation_confidence"], reverse=True)

            return related[:10]  # Omezení na top 10

        except Exception as e:
            logger.error(f"Chyba při hledání relacionovaných entit pro {entity_id}: {e}")
            return []
