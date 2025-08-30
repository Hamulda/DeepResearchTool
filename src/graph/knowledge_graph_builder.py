"""
Knowledge Graph Builder pro DeepResearchTool
Implementuje extrakci entit a vztahů pomocí LLM a jejich ukládání do Neo4j
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import re
from dataclasses import dataclass
from enum import Enum

import networkx as nx
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, ClientError

from ..core.error_handling import (
    database_circuit_breaker,
    with_circuit_breaker,
    safe_llm_call,
    ErrorAggregator,
    timeout_after
)
from ..core.config import get_settings

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Typy entit pro knowledge graph"""
    PERSON = "Person"
    ORGANIZATION = "Organization"
    LOCATION = "Location"
    EVENT = "Event"
    CONCEPT = "Concept"
    DOCUMENT = "Document"
    DATE = "Date"
    TECHNOLOGY = "Technology"
    PRODUCT = "Product"


class RelationType(Enum):
    """Typy vztahů mezi entitami"""
    WORKS_FOR = "WORKS_FOR"
    LOCATED_IN = "LOCATED_IN"
    PARTICIPATED_IN = "PARTICIPATED_IN"
    FOUNDED = "FOUNDED"
    RELATED_TO = "RELATED_TO"
    MENTIONED_IN = "MENTIONED_IN"
    HAPPENED_ON = "HAPPENED_ON"
    PART_OF = "PART_OF"
    CREATED = "CREATED"
    COLLABORATED_WITH = "COLLABORATED_WITH"


@dataclass
class Entity:
    """Reprezentace entity v knowledge graph"""
    name: str
    entity_type: EntityType
    properties: Dict[str, Any]
    confidence: float
    source_document: str
    mentions: List[Dict[str, Any]]


@dataclass
class Relationship:
    """Reprezentace vztahu mezi entitami"""
    source_entity: str
    target_entity: str
    relationship_type: RelationType
    properties: Dict[str, Any]
    confidence: float
    source_document: str
    context: str


class LLMEntityExtractor:
    """LLM-based entity a relationship extractor"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.error_aggregator = ErrorAggregator()
        
        # Prompts pro extrakci
        self.entity_extraction_prompt = """
Analyze the following text and extract entities with their types. Focus on:
- People (names, roles, positions)
- Organizations (companies, institutions, governments)
- Locations (cities, countries, regions)
- Events (meetings, conferences, incidents)
- Concepts (technologies, methodologies, theories)
- Dates (specific dates, time periods)

Text: {text}

Return a JSON list of entities in this format:
[
    {
        "name": "entity name",
        "type": "Person|Organization|Location|Event|Concept|Date",
        "properties": {
            "description": "brief description",
            "context": "context from text"
        },
        "confidence": 0.0-1.0
    }
]

Only return valid JSON, no additional text.
"""

        self.relationship_extraction_prompt = """
Analyze the following text and extract relationships between entities.

Entities found: {entities}

Text: {text}

Extract relationships in this JSON format:
[
    {
        "source": "source entity name",
        "target": "target entity name", 
        "relationship": "WORKS_FOR|LOCATED_IN|FOUNDED|RELATED_TO|MENTIONED_IN|HAPPENED_ON|PART_OF|CREATED|COLLABORATED_WITH",
        "context": "context explaining the relationship",
        "confidence": 0.0-1.0
    }
]

Only return valid JSON, no additional text.
"""
    
    @timeout_after(60)
    async def extract_entities(self, text: str, document_id: str) -> List[Entity]:
        """Extrahuje entity z textu pomocí LLM"""
        try:
            # Rozdělení dlouhého textu na části
            chunks = self._split_text(text, max_length=2000)
            all_entities = []
            
            for chunk in chunks:
                try:
                    # Připrava promptu
                    prompt = self.entity_extraction_prompt.format(text=chunk)
                    
                    # Volání LLM
                    response = await safe_llm_call(self._call_llm, prompt)
                    
                    if response:
                        entities_data = json.loads(response)
                        
                        for entity_data in entities_data:
                            # Validace a vytvoření Entity objektu
                            entity = self._create_entity(entity_data, document_id, chunk)
                            if entity:
                                all_entities.append(entity)
                    
                    self.error_aggregator.add_success()
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse LLM response for entities: {e}")
                    self.error_aggregator.add_error(e, "entity extraction JSON parsing")
                except Exception as e:
                    logger.error(f"Entity extraction failed for chunk: {e}")
                    self.error_aggregator.add_error(e, "entity extraction")
                
                # Rate limiting
                await asyncio.sleep(0.5)
            
            # Deduplikace entit
            unique_entities = self._deduplicate_entities(all_entities)
            
            logger.info(f"Extracted {len(unique_entities)} unique entities from {len(chunks)} chunks")
            return unique_entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            self.error_aggregator.add_error(e, "entity extraction")
            return []
    
    @timeout_after(60)
    async def extract_relationships(self, text: str, entities: List[Entity], document_id: str) -> List[Relationship]:
        """Extrahuje vztahy mezi entitami pomocí LLM"""
        try:
            if len(entities) < 2:
                return []
            
            # Připrava seznamu entit pro prompt
            entity_names = [entity.name for entity in entities]
            entities_str = ", ".join(entity_names)
            
            chunks = self._split_text(text, max_length=1500)
            all_relationships = []
            
            for chunk in chunks:
                try:
                    # Filtruj entity, které jsou zmíněny v tomto chunku
                    chunk_entities = [e for e in entities if e.name.lower() in chunk.lower()]
                    
                    if len(chunk_entities) < 2:
                        continue
                    
                    chunk_entity_names = [e.name for e in chunk_entities]
                    chunk_entities_str = ", ".join(chunk_entity_names)
                    
                    # Připrava promptu
                    prompt = self.relationship_extraction_prompt.format(
                        entities=chunk_entities_str,
                        text=chunk
                    )
                    
                    # Volání LLM
                    response = await safe_llm_call(self._call_llm, prompt)
                    
                    if response:
                        relationships_data = json.loads(response)
                        
                        for rel_data in relationships_data:
                            # Validace a vytvoření Relationship objektu
                            relationship = self._create_relationship(rel_data, document_id, chunk)
                            if relationship:
                                all_relationships.append(relationship)
                    
                    self.error_aggregator.add_success()
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse LLM response for relationships: {e}")
                    self.error_aggregator.add_error(e, "relationship extraction JSON parsing")
                except Exception as e:
                    logger.error(f"Relationship extraction failed for chunk: {e}")
                    self.error_aggregator.add_error(e, "relationship extraction")
                
                # Rate limiting
                await asyncio.sleep(0.5)
            
            logger.info(f"Extracted {len(all_relationships)} relationships")
            return all_relationships
            
        except Exception as e:
            logger.error(f"Relationship extraction failed: {e}")
            self.error_aggregator.add_error(e, "relationship extraction")
            return []
    
    async def _call_llm(self, prompt: str) -> str:
        """Placeholder pro LLM volání - implementace závisí na konkrétním LLM provideru"""
        # V reálné implementaci by zde bylo volání OpenAI API, Anthropic, nebo lokálního modelu
        try:
            # Mock implementace pro demonstraci
            if "entities" in prompt.lower():
                return '''[
                    {
                        "name": "Example Person",
                        "type": "Person",
                        "properties": {"description": "Example description", "context": "mentioned in text"},
                        "confidence": 0.8
                    }
                ]'''
            else:
                return '''[
                    {
                        "source": "Example Person",
                        "target": "Example Organization",
                        "relationship": "WORKS_FOR",
                        "context": "works for the organization",
                        "confidence": 0.7
                    }
                ]'''
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _split_text(self, text: str, max_length: int = 2000) -> List[str]:
        """Rozdělí text na části pro LLM zpracování"""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        sentences = re.split(r'[.!?]+', text)
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _create_entity(self, entity_data: Dict, document_id: str, text_chunk: str) -> Optional[Entity]:
        """Vytvoří Entity objekt z LLM odpovědi"""
        try:
            name = entity_data.get('name', '').strip()
            entity_type_str = entity_data.get('type', '').strip()
            
            if not name or not entity_type_str:
                return None
            
            # Mapování typu entity
            try:
                entity_type = EntityType(entity_type_str)
            except ValueError:
                # Fallback na Concept pro neznámé typy
                entity_type = EntityType.CONCEPT
            
            properties = entity_data.get('properties', {})
            confidence = float(entity_data.get('confidence', 0.5))
            
            # Najdi mentions v textu
            mentions = self._find_mentions(name, text_chunk)
            
            return Entity(
                name=name,
                entity_type=entity_type,
                properties=properties,
                confidence=confidence,
                source_document=document_id,
                mentions=mentions
            )
            
        except Exception as e:
            logger.warning(f"Failed to create entity from data {entity_data}: {e}")
            return None
    
    def _create_relationship(self, rel_data: Dict, document_id: str, context: str) -> Optional[Relationship]:
        """Vytvoří Relationship objekt z LLM odpovědi"""
        try:
            source = rel_data.get('source', '').strip()
            target = rel_data.get('target', '').strip()
            rel_type_str = rel_data.get('relationship', '').strip()
            
            if not source or not target or not rel_type_str:
                return None
            
            # Mapování typu vztahu
            try:
                rel_type = RelationType(rel_type_str)
            except ValueError:
                # Fallback na RELATED_TO pro neznámé typy
                rel_type = RelationType.RELATED_TO
            
            properties = rel_data.get('properties', {})
            confidence = float(rel_data.get('confidence', 0.5))
            rel_context = rel_data.get('context', context)
            
            return Relationship(
                source_entity=source,
                target_entity=target,
                relationship_type=rel_type,
                properties=properties,
                confidence=confidence,
                source_document=document_id,
                context=rel_context
            )
            
        except Exception as e:
            logger.warning(f"Failed to create relationship from data {rel_data}: {e}")
            return None
    
    def _find_mentions(self, entity_name: str, text: str) -> List[Dict[str, Any]]:
        """Najde všechny zmínky entity v textu"""
        mentions = []
        text_lower = text.lower()
        entity_lower = entity_name.lower()
        
        start = 0
        while True:
            pos = text_lower.find(entity_lower, start)
            if pos == -1:
                break
            
            mentions.append({
                'start': pos,
                'end': pos + len(entity_name),
                'context': text[max(0, pos-50):pos+len(entity_name)+50]
            })
            start = pos + 1
        
        return mentions
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Odstraní duplicitní entity"""
        unique_entities = {}
        
        for entity in entities:
            # Normalizace jména pro deduplikaci
            normalized_name = entity.name.lower().strip()
            
            if normalized_name not in unique_entities:
                unique_entities[normalized_name] = entity
            else:
                # Merge s existující entitou (vyšší confidence)
                existing = unique_entities[normalized_name]
                if entity.confidence > existing.confidence:
                    # Update s lepší entitou
                    existing.confidence = entity.confidence
                    existing.properties.update(entity.properties)
                    existing.mentions.extend(entity.mentions)
        
        return list(unique_entities.values())


class Neo4jKnowledgeGraph:
    """Neo4j knowledge graph implementation"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user 
        self.password = password
        self.driver: Optional[Driver] = None
        self.error_aggregator = ErrorAggregator()
    
    async def connect(self) -> bool:
        """Připojení k Neo4j databázi"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            
            # Test připojení
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                
            if test_value == 1:
                logger.info("Successfully connected to Neo4j")
                await self._create_indexes()
                return True
            else:
                raise Exception("Test query failed")
                
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False
    
    async def _create_indexes(self):
        """Vytvoří indexy pro lepší výkon"""
        indexes = [
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX document_id_index IF NOT EXISTS FOR (d:Document) ON (d.id)",
        ]
        
        with self.driver.session() as session:
            for index_query in indexes:
                try:
                    session.run(index_query)
                except ClientError as e:
                    # Index může už existovat
                    logger.debug(f"Index creation skipped: {e}")
    
    @with_circuit_breaker(database_circuit_breaker)
    async def add_entities(self, entities: List[Entity]) -> int:
        """Přidá entity do knowledge graph"""
        if not self.driver or not entities:
            return 0
        
        added_count = 0
        
        with self.driver.session() as session:
            for entity in entities:
                try:
                    query = """
                    MERGE (e:Entity {name: $name, type: $type})
                    SET e.confidence = $confidence,
                        e.source_document = $source_document,
                        e.created_at = $created_at,
                        e.properties = $properties,
                        e.mentions = $mentions
                    RETURN e
                    """
                    
                    result = session.run(query, {
                        'name': entity.name,
                        'type': entity.entity_type.value,
                        'confidence': entity.confidence,
                        'source_document': entity.source_document,
                        'created_at': datetime.now().isoformat(),
                        'properties': entity.properties,
                        'mentions': [dict(m) for m in entity.mentions]
                    })
                    
                    if result.single():
                        added_count += 1
                        self.error_aggregator.add_success()
                    
                except Exception as e:
                    logger.error(f"Failed to add entity {entity.name}: {e}")
                    self.error_aggregator.add_error(e, f"adding entity {entity.name}")
        
        logger.info(f"Added {added_count}/{len(entities)} entities to knowledge graph")
        return added_count
    
    @with_circuit_breaker(database_circuit_breaker)
    async def add_relationships(self, relationships: List[Relationship]) -> int:
        """Přidá vztahy do knowledge graph"""
        if not self.driver or not relationships:
            return 0
        
        added_count = 0
        
        with self.driver.session() as session:
            for rel in relationships:
                try:
                    query = """
                    MATCH (source:Entity {name: $source_name})
                    MATCH (target:Entity {name: $target_name})
                    MERGE (source)-[r:RELATIONSHIP {type: $rel_type}]->(target)
                    SET r.confidence = $confidence,
                        r.source_document = $source_document,
                        r.context = $context,
                        r.created_at = $created_at,
                        r.properties = $properties
                    RETURN r
                    """
                    
                    result = session.run(query, {
                        'source_name': rel.source_entity,
                        'target_name': rel.target_entity,
                        'rel_type': rel.relationship_type.value,
                        'confidence': rel.confidence,
                        'source_document': rel.source_document,
                        'context': rel.context,
                        'created_at': datetime.now().isoformat(),
                        'properties': rel.properties
                    })
                    
                    if result.single():
                        added_count += 1
                        self.error_aggregator.add_success()
                    
                except Exception as e:
                    logger.error(f"Failed to add relationship {rel.source_entity}->{rel.target_entity}: {e}")
                    self.error_aggregator.add_error(e, f"adding relationship")
        
        logger.info(f"Added {added_count}/{len(relationships)} relationships to knowledge graph")
        return added_count
    
    async def find_entities(self, entity_type: Optional[EntityType] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Najde entity podle typu"""
        if not self.driver:
            return []
        
        if entity_type:
            query = "MATCH (e:Entity {type: $type}) RETURN e LIMIT $limit"
            params = {'type': entity_type.value, 'limit': limit}
        else:
            query = "MATCH (e:Entity) RETURN e LIMIT $limit"
            params = {'limit': limit}
        
        with self.driver.session() as session:
            result = session.run(query, params)
            return [record['e'] for record in result]
    
    async def find_related_entities(self, entity_name: str, max_depth: int = 2) -> Dict[str, Any]:
        """Najde entity související s danou entitou"""
        if not self.driver:
            return {}
        
        query = """
        MATCH path = (start:Entity {name: $entity_name})-[*1..$max_depth]-(related:Entity)
        RETURN start, related, relationships(path) as rels, length(path) as depth
        ORDER BY depth, related.confidence DESC
        LIMIT 50
        """
        
        with self.driver.session() as session:
            result = session.run(query, {
                'entity_name': entity_name,
                'max_depth': max_depth
            })
            
            related_entities = []
            relationships = []
            
            for record in result:
                related_entities.append(dict(record['related']))
                for rel in record['rels']:
                    relationships.append(dict(rel))
            
            return {
                'source_entity': entity_name,
                'related_entities': related_entities,
                'relationships': relationships,
                'total_found': len(related_entities)
            }
    
    async def get_graph_stats(self) -> Dict[str, Any]:
        """Získá statistiky knowledge graph"""
        if not self.driver:
            return {}
        
        queries = {
            'total_entities': "MATCH (e:Entity) RETURN count(e) as count",
            'total_relationships': "MATCH ()-[r:RELATIONSHIP]->() RETURN count(r) as count",
            'entity_types': "MATCH (e:Entity) RETURN e.type as type, count(e) as count ORDER BY count DESC",
            'relationship_types': "MATCH ()-[r:RELATIONSHIP]->() RETURN r.type as type, count(r) as count ORDER BY count DESC"
        }
        
        stats = {}
        
        with self.driver.session() as session:
            for stat_name, query in queries.items():
                try:
                    result = session.run(query)
                    if stat_name in ['total_entities', 'total_relationships']:
                        stats[stat_name] = result.single()['count']
                    else:
                        stats[stat_name] = [dict(record) for record in result]
                except Exception as e:
                    logger.error(f"Failed to get stat {stat_name}: {e}")
                    stats[stat_name] = 0 if 'total' in stat_name else []
        
        return stats
    
    async def close(self):
        """Uzavře připojení k databázi"""
        if self.driver:
            self.driver.close()
            self.driver = None
            logger.info("Neo4j connection closed")


class KnowledgeGraphBuilder:
    """Hlavní třída pro building knowledge graph"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.extractor = LLMEntityExtractor()
        self.graph = Neo4jKnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password)
        self.error_aggregator = ErrorAggregator()
    
    async def initialize(self) -> bool:
        """Inicializace knowledge graph builderu"""
        return await self.graph.connect()
    
    @timeout_after(600)  # 10 minute timeout pro celý proces
    async def process_document(self, text: str, document_id: str) -> Dict[str, Any]:
        """Zpracuje dokument a přidá ho do knowledge graph"""
        try:
            logger.info(f"Processing document {document_id} for knowledge graph")
            
            # Extrakce entit
            entities = await self.extractor.extract_entities(text, document_id)
            
            if not entities:
                logger.warning(f"No entities extracted from document {document_id}")
                return {
                    'document_id': document_id,
                    'entities_found': 0,
                    'relationships_found': 0,
                    'success': False,
                    'error': 'No entities extracted'
                }
            
            # Extrakce vztahů
            relationships = await self.extractor.extract_relationships(text, entities, document_id)
            
            # Přidání do knowledge graph
            entities_added = await self.graph.add_entities(entities)
            relationships_added = await self.graph.add_relationships(relationships)
            
            result = {
                'document_id': document_id,
                'entities_found': len(entities),
                'entities_added': entities_added,
                'relationships_found': len(relationships),
                'relationships_added': relationships_added,
                'success': True,
                'processing_time': datetime.now().isoformat()
            }
            
            logger.info(f"Document {document_id} processed: {entities_added} entities, {relationships_added} relationships")
            self.error_aggregator.add_success()
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process document {document_id}: {e}")
            self.error_aggregator.add_error(e, f"processing document {document_id}")
            
            return {
                'document_id': document_id,
                'entities_found': 0,
                'relationships_found': 0,
                'success': False,
                'error': str(e)
            }
    
    async def process_documents_batch(self, documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Zpracuje více dokumentů v dávce"""
        results = []
        
        # Semafór pro omezení concurrent processing
        semaphore = asyncio.Semaphore(2)  # Max 2 documents at once
        
        async def process_single(doc):
            async with semaphore:
                return await self.process_document(doc['text'], doc['id'])
        
        tasks = [process_single(doc) for doc in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append({
                    'document_id': documents[i].get('id', f'doc_{i}'),
                    'success': False,
                    'error': str(result)
                })
            else:
                final_results.append(result)
        
        return final_results
    
    async def get_stats(self) -> Dict[str, Any]:
        """Získá kompletní statistiky"""
        graph_stats = await self.graph.get_graph_stats()
        extractor_stats = self.extractor.error_aggregator.get_summary()
        builder_stats = self.error_aggregator.get_summary()
        
        return {
            'knowledge_graph': graph_stats,
            'extraction_stats': extractor_stats,
            'builder_stats': builder_stats,
            'total_success_rate': builder_stats['success_rate']
        }
    
    async def close(self):
        """Uzavře všechna připojení"""
        await self.graph.close()


# Utility functions
async def build_knowledge_graph_from_documents(documents: List[Dict[str, str]], 
                                             neo4j_config: Dict[str, str]) -> Dict[str, Any]:
    """Utility pro quick knowledge graph building"""
    builder = KnowledgeGraphBuilder(
        neo4j_uri=neo4j_config['uri'],
        neo4j_user=neo4j_config['user'],
        neo4j_password=neo4j_config['password']
    )
    
    try:
        await builder.initialize()
        results = await builder.process_documents_batch(documents)
        stats = await builder.get_stats()
        
        return {
            'processing_results': results,
            'final_stats': stats,
            'success': True
        }
    finally:
        await builder.close()