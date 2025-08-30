"""Knowledge Graph Manager - Fáze 1: Jádro Znalostního Grafu
Správa grafové databáze Neo4j pro ukládání entit a vztahů
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import json
import logging
import os
from typing import Any

from neo4j import GraphDatabase

# Setup logging
logger = logging.getLogger(__name__)


class KnowledgeGraphManager:
    """Správce znalostního grafu pro persistentní ukládání entit a vztahů"""

    def __init__(self):
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "research2024")

        self.driver = None
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Inicializace připojení
        self._initialize_connection()

    def _initialize_connection(self):
        """Inicializuj připojení k Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
            )

            # Ověř připojení
            with self.driver.session() as session:
                result = session.run("RETURN 'Connection successful' as message")
                message = result.single()["message"]
                logger.info(f"✅ Neo4j připojení úspěšné: {message}")

            # Vytvoř indexy a constraints
            self._create_schema()

        except Exception as e:
            logger.error(f"❌ Chyba při připojování k Neo4j: {e}")
            raise

    def _create_schema(self):
        """Vytvoř indexy a constraints pro optimální výkon"""
        constraints_and_indexes = [
            # Constraints pro jedinečnost
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT source_url_unique IF NOT EXISTS FOR (s:Source) REQUIRE s.url IS UNIQUE",
            # Indexy pro rychlé vyhledávání
            "CREATE INDEX entity_text_index IF NOT EXISTS FOR (e:Entity) ON (e.text)",
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX source_timestamp_index IF NOT EXISTS FOR (s:Source) ON (s.timestamp)",
            "CREATE INDEX relation_type_index IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.type)",
        ]

        with self.driver.session() as session:
            for query in constraints_and_indexes:
                try:
                    session.run(query)
                    logger.debug(f"✅ Schema query executed: {query[:50]}...")
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        logger.warning(f"⚠️ Schema query failed: {e}")

    async def add_entities_and_relations(
        self,
        entities: dict[str, list[dict[str, Any]]],
        relations: list[dict[str, Any]],
        source_url: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Přidej entity a vztahy do znalostního grafu

        Args:
            entities: Slovník entit podle kategorií
            relations: Seznam vztahů ve formátu [{"subject": ..., "predicate": ..., "object": ...}]
            source_url: URL zdroje
            metadata: Metadata o zdroji

        Returns:
            Statistiky o přidaných uzlech a hranách

        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._add_entities_and_relations_sync,
            entities,
            relations,
            source_url,
            metadata,
        )

    def _add_entities_and_relations_sync(
        self,
        entities: dict[str, list[dict[str, Any]]],
        relations: list[dict[str, Any]],
        source_url: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Synchronní verze pro přidání entit a vztahů"""
        stats = {"entities_added": 0, "relations_added": 0, "source_added": False, "errors": []}

        try:
            with self.driver.session() as session:
                # 1. Vytvoř nebo aktualizuj zdrojový uzel
                source_query = """
                MERGE (s:Source {url: $url})
                SET s.title = $title,
                    s.timestamp = $timestamp,
                    s.last_updated = datetime(),
                    s.metadata = $metadata
                RETURN s
                """

                session.run(
                    source_query,
                    {
                        "url": source_url,
                        "title": metadata.get("title", ""),
                        "timestamp": metadata.get(
                            "extracted_at", datetime.now(UTC).isoformat()
                        ),
                        "metadata": json.dumps(metadata),
                    },
                )
                stats["source_added"] = True

                # 2. Přidej entity
                for entity_type, entity_list in entities.items():
                    for entity in entity_list:
                        try:
                            entity_id = self._generate_entity_id(entity["text"], entity_type)

                            entity_query = """
                            MERGE (e:Entity {id: $entity_id})
                            SET e.text = $text,
                                e.type = $type,
                                e.label = $label,
                                e.confidence = $confidence,
                                e.last_seen = datetime()
                            WITH e
                            MATCH (s:Source {url: $source_url})
                            MERGE (e)-[:FOUND_IN]->(s)
                            RETURN e
                            """

                            session.run(
                                entity_query,
                                {
                                    "entity_id": entity_id,
                                    "text": entity["text"],
                                    "type": entity_type,
                                    "label": entity.get("label", entity_type.upper()),
                                    "confidence": entity.get("confidence", 1.0),
                                    "source_url": source_url,
                                },
                            )
                            stats["entities_added"] += 1

                        except Exception as e:
                            stats["errors"].append(f"Entity error: {e}")
                            logger.error(f"❌ Chyba při přidávání entity {entity}: {e}")

                # 3. Přidej vztahy
                for relation in relations:
                    try:
                        self._add_relation(session, relation, source_url)
                        stats["relations_added"] += 1
                    except Exception as e:
                        stats["errors"].append(f"Relation error: {e}")
                        logger.error(f"❌ Chyba při přidávání vztahu {relation}: {e}")

        except Exception as e:
            stats["errors"].append(f"Transaction error: {e}")
            logger.error(f"❌ Chyba při transakci do Neo4j: {e}")

        return stats

    def _add_relation(self, session, relation: dict[str, Any], source_url: str):
        """Přidej vztah mezi entitami"""
        subject_id = self._generate_entity_id(relation["subject"], "unknown")
        object_id = self._generate_entity_id(relation["object"], "unknown")
        predicate = relation["predicate"]

        relation_query = """
        MERGE (subj:Entity {id: $subject_id})
        ON CREATE SET subj.text = $subject_text, subj.type = 'unknown'
        MERGE (obj:Entity {id: $object_id})
        ON CREATE SET obj.text = $object_text, obj.type = 'unknown'
        MERGE (subj)-[r:RELATES_TO {type: $predicate}]->(obj)
        SET r.confidence = $confidence,
            r.source_url = $source_url,
            r.created_at = datetime()
        WITH subj, obj
        MATCH (s:Source {url: $source_url})
        MERGE (subj)-[:FOUND_IN]->(s)
        MERGE (obj)-[:FOUND_IN]->(s)
        RETURN subj, obj, r
        """

        session.run(
            relation_query,
            {
                "subject_id": subject_id,
                "subject_text": relation["subject"],
                "object_id": object_id,
                "object_text": relation["object"],
                "predicate": predicate,
                "confidence": relation.get("confidence", 0.8),
                "source_url": source_url,
            },
        )

    def _generate_entity_id(self, text: str, entity_type: str) -> str:
        """Generuj jedinečné ID pro entitu"""
        import hashlib

        combined = f"{text.lower().strip()}|{entity_type}"
        return hashlib.md5(combined.encode()).hexdigest()

    async def query_entities(
        self, entity_text: str = None, entity_type: str = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Dotaz na entity v grafu"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self._query_entities_sync, entity_text, entity_type, limit
        )

    def _query_entities_sync(
        self, entity_text: str = None, entity_type: str = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Synchronní dotaz na entity"""
        where_clauses = []
        params = {"limit": limit}

        if entity_text:
            where_clauses.append("e.text CONTAINS $text")
            params["text"] = entity_text

        if entity_type:
            where_clauses.append("e.type = $type")
            params["type"] = entity_type

        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        query = f"""
        MATCH (e:Entity)
        {where_clause}
        RETURN e.id as id, e.text as text, e.type as type, 
               e.confidence as confidence, e.last_seen as last_seen
        LIMIT $limit
        """

        try:
            with self.driver.session() as session:
                result = session.run(query, params)
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"❌ Chyba při dotazu na entity: {e}")
            return []

    async def query_relations(
        self, subject: str = None, predicate: str = None, object_entity: str = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Dotaz na vztahy v grafu"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self._query_relations_sync, subject, predicate, object_entity, limit
        )

    def _query_relations_sync(
        self, subject: str = None, predicate: str = None, object_entity: str = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Synchronní dotaz na vztahy"""
        where_clauses = []
        params = {"limit": limit}

        if subject:
            where_clauses.append("subj.text CONTAINS $subject")
            params["subject"] = subject

        if predicate:
            where_clauses.append("r.type = $predicate")
            params["predicate"] = predicate

        if object_entity:
            where_clauses.append("obj.text CONTAINS $object")
            params["object"] = object_entity

        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        query = f"""
        MATCH (subj:Entity)-[r:RELATES_TO]->(obj:Entity)
        {where_clause}
        RETURN subj.text as subject, r.type as predicate, obj.text as object,
               r.confidence as confidence, r.source_url as source_url
        LIMIT $limit
        """

        try:
            with self.driver.session() as session:
                result = session.run(query, params)
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"❌ Chyba při dotazu na vztahy: {e}")
            return []

    async def get_entity_neighbors(
        self, entity_text: str, max_depth: int = 2, limit: int = 20
    ) -> dict[str, Any]:
        """Získej sousedy entity do určité hloubky"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self._get_entity_neighbors_sync, entity_text, max_depth, limit
        )

    def _get_entity_neighbors_sync(
        self, entity_text: str, max_depth: int = 2, limit: int = 20
    ) -> dict[str, Any]:
        """Synchronní získání sousedů entity"""
        query = """
        MATCH (center:Entity)
        WHERE center.text CONTAINS $entity_text
        CALL {
            WITH center
            MATCH (center)-[r:RELATES_TO*1..2]-(neighbor:Entity)
            RETURN neighbor, r
            LIMIT $limit
        }
        RETURN center.text as center_entity,
               collect({
                   entity: neighbor.text,
                   type: neighbor.type,
                   relations: [rel in r | rel.type]
               }) as neighbors
        """

        try:
            with self.driver.session() as session:
                result = session.run(query, {"entity_text": entity_text, "limit": limit})

                records = list(result)
                if records:
                    return records[0].data()
                return {"center_entity": entity_text, "neighbors": []}

        except Exception as e:
            logger.error(f"❌ Chyba při získávání sousedů: {e}")
            return {"center_entity": entity_text, "neighbors": []}

    async def get_graph_statistics(self) -> dict[str, Any]:
        """Získej statistiky o grafu"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._get_graph_statistics_sync)

    def _get_graph_statistics_sync(self) -> dict[str, Any]:
        """Synchronní získání statistik"""
        queries = {
            "total_entities": "MATCH (e:Entity) RETURN count(e) as count",
            "total_relations": "MATCH ()-[r:RELATES_TO]-() RETURN count(r) as count",
            "total_sources": "MATCH (s:Source) RETURN count(s) as count",
            "entity_types": """
                MATCH (e:Entity) 
                RETURN e.type as type, count(e) as count 
                ORDER BY count DESC
            """,
            "relation_types": """
                MATCH ()-[r:RELATES_TO]-() 
                RETURN r.type as type, count(r) as count 
                ORDER BY count DESC
            """,
        }

        stats = {}

        try:
            with self.driver.session() as session:
                for stat_name, query in queries.items():
                    result = session.run(query)
                    if stat_name in ["entity_types", "relation_types"]:
                        stats[stat_name] = [record.data() for record in result]
                    else:
                        record = result.single()
                        stats[stat_name] = record["count"] if record else 0

        except Exception as e:
            logger.error(f"❌ Chyba při získávání statistik: {e}")
            stats = {"error": str(e)}

        return stats

    def close(self):
        """Uzavři připojení"""
        if self.driver:
            self.driver.close()
            logger.info("✅ Neo4j připojení uzavřeno")
