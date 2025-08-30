"""OSINT Automator for Intelligence Correlation
Stubs for public records, breach sources, and entity correlation
Produces entity graphs with provenance and confidence scoring
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class OSINTSourceType(Enum):
    """Types of OSINT sources"""

    PUBLIC_RECORDS = "public_records"
    BREACH_DATABASE = "breach_database"
    SOCIAL_MEDIA = "social_media"
    COURT_RECORDS = "court_records"
    BUSINESS_REGISTRY = "business_registry"
    NEWS_ARTICLES = "news_articles"
    ACADEMIC_PAPERS = "academic_papers"
    GOVERNMENT_DATA = "government_data"


@dataclass
class EntityType(Enum):
    """Entity types for correlation"""

    PERSON = "person"
    ORGANIZATION = "organization"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    DOMAIN = "domain"
    IP_ADDRESS = "ip_address"
    DOCUMENT = "document"


@dataclass
class OSINTEntity:
    """OSINT entity with metadata"""

    entity_id: str
    entity_type: EntityType
    value: str
    confidence: float
    sources: list[str]
    first_seen: datetime
    last_seen: datetime
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.entity_id:
            self.entity_id = self._generate_entity_id()

    def _generate_entity_id(self) -> str:
        """Generate unique entity ID"""
        content = f"{self.entity_type.value}:{self.value}"
        return hashlib.md5(content.encode()).hexdigest()[:16]


@dataclass
class OSINTRelationship:
    """Relationship between OSINT entities"""

    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    confidence: float
    source: str
    discovered_at: datetime
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.relationship_id:
            content = f"{self.source_entity_id}:{self.target_entity_id}:{self.relationship_type}"
            self.relationship_id = hashlib.md5(content.encode()).hexdigest()[:16]


@dataclass
class OSINTResult:
    """Result from OSINT investigation"""

    investigation_id: str
    query: str
    entities: list[OSINTEntity]
    relationships: list[OSINTRelationship]
    confidence_score: float
    sources_used: list[str]
    investigation_time: datetime
    processing_time_seconds: float


@dataclass
class OSINTConfig:
    """OSINT automator configuration"""

    # Source selection
    enabled_sources: list[OSINTSourceType] = field(
        default_factory=lambda: [
            OSINTSourceType.PUBLIC_RECORDS,
            OSINTSourceType.NEWS_ARTICLES,
            OSINTSourceType.ACADEMIC_PAPERS,
        ]
    )

    # API keys and credentials (all optional/stubbed)
    api_keys: dict[str, str] = field(default_factory=dict)

    # Search parameters
    max_results_per_source: int = 100
    confidence_threshold: float = 0.5
    max_investigation_time_minutes: int = 30

    # Entity correlation
    enable_entity_linking: bool = True
    similarity_threshold: float = 0.8
    max_correlation_depth: int = 3

    # Privacy and legal
    respect_rate_limits: bool = True
    log_all_queries: bool = True
    anonymize_pii: bool = True

    # Output settings
    save_entity_graph: bool = True
    export_formats: list[str] = field(default_factory=lambda: ["json", "csv"])


class PublicRecordsConnector:
    """Connector for public records databases (stubbed)"""

    def __init__(self, config: OSINTConfig):
        self.config = config

    async def search_person(self, name: str, location: str | None = None) -> list[OSINTEntity]:
        """Search for person in public records"""
        logger.info(f"STUB: Searching public records for person: {name}")

        # Mock results for demonstration
        entities = []

        if name:
            entity = OSINTEntity(
                entity_id="",
                entity_type=EntityType.PERSON,
                value=name,
                confidence=0.7,
                sources=["public_records_stub"],
                first_seen=datetime.now() - timedelta(days=30),
                last_seen=datetime.now(),
                attributes={
                    "source": "public_records",
                    "location": location or "Unknown",
                    "record_type": "business_registration",
                },
            )
            entities.append(entity)

        return entities

    async def search_business(self, business_name: str) -> list[OSINTEntity]:
        """Search for business in registry"""
        logger.info(f"STUB: Searching business registry for: {business_name}")

        entity = OSINTEntity(
            entity_id="",
            entity_type=EntityType.ORGANIZATION,
            value=business_name,
            confidence=0.8,
            sources=["business_registry_stub"],
            first_seen=datetime.now() - timedelta(days=365),
            last_seen=datetime.now(),
            attributes={
                "registration_status": "active",
                "industry": "technology",
                "employees": "10-50",
            },
        )

        return [entity]


class BreachDatabaseConnector:
    """Connector for breach databases (stubbed for legal compliance)"""

    def __init__(self, config: OSINTConfig):
        self.config = config

    async def check_email_breaches(self, email: str) -> list[OSINTEntity]:
        """Check if email appears in known data breaches"""
        logger.info(
            f"STUB: Checking breach databases for email domain: {email.split('@')[1] if '@' in email else 'invalid'}"
        )

        # Only return anonymized/aggregated information
        if "@" in email:
            domain = email.split("@")[1]
            entity = OSINTEntity(
                entity_id="",
                entity_type=EntityType.DOMAIN,
                value=domain,
                confidence=0.6,
                sources=["breach_database_stub"],
                first_seen=datetime.now() - timedelta(days=180),
                last_seen=datetime.now(),
                attributes={"breach_count": 2, "severity": "medium", "anonymized": True},
            )
            return [entity]

        return []


class NewsSourceConnector:
    """Connector for news and media sources"""

    def __init__(self, config: OSINTConfig):
        self.config = config

    async def search_news_mentions(self, query: str) -> list[OSINTEntity]:
        """Search for mentions in news articles"""
        logger.info(f"STUB: Searching news sources for: {query}")

        entities = []

        # Mock news entity
        entity = OSINTEntity(
            entity_id="",
            entity_type=EntityType.DOCUMENT,
            value=f"News article mentioning '{query}'",
            confidence=0.6,
            sources=["news_api_stub"],
            first_seen=datetime.now() - timedelta(days=7),
            last_seen=datetime.now(),
            attributes={
                "publication": "Example News",
                "headline": f"Research developments related to {query}",
                "sentiment": "neutral",
                "url": f"https://example-news.com/article-{hash(query) % 1000}",
            },
        )
        entities.append(entity)

        return entities


class AcademicSourceConnector:
    """Connector for academic papers and research"""

    def __init__(self, config: OSINTConfig):
        self.config = config

    async def search_academic_papers(self, query: str) -> list[OSINTEntity]:
        """Search academic databases for research papers"""
        logger.info(f"STUB: Searching academic sources for: {query}")

        entity = OSINTEntity(
            entity_id="",
            entity_type=EntityType.DOCUMENT,
            value=f"Academic paper: {query}",
            confidence=0.9,
            sources=["academic_db_stub"],
            first_seen=datetime.now() - timedelta(days=365),
            last_seen=datetime.now(),
            attributes={
                "journal": "Journal of Research",
                "authors": ["Dr. Example Author"],
                "citation_count": 42,
                "doi": f"10.1000/example.{hash(query) % 10000}",
                "abstract": f"This paper discusses research related to {query}...",
            },
        )

        return [entity]


class EntityCorrelationEngine:
    """Engine for correlating entities and finding relationships"""

    def __init__(self, config: OSINTConfig):
        self.config = config

    async def correlate_entities(self, entities: list[OSINTEntity]) -> list[OSINTRelationship]:
        """Find relationships between entities"""
        relationships = []

        # Simple correlation logic
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i + 1 :]:
                relationship = await self._find_relationship(entity1, entity2)
                if relationship:
                    relationships.append(relationship)

        return relationships

    async def _find_relationship(
        self, entity1: OSINTEntity, entity2: OSINTEntity
    ) -> OSINTRelationship | None:
        """Find relationship between two entities"""
        # Name similarity for person-organization relationships
        if (
            entity1.entity_type == EntityType.PERSON
            and entity2.entity_type == EntityType.ORGANIZATION
        ):
            if self._calculate_similarity(entity1.value, entity2.value) > 0.3:
                return OSINTRelationship(
                    relationship_id="",
                    source_entity_id=entity1.entity_id,
                    target_entity_id=entity2.entity_id,
                    relationship_type="associated_with",
                    confidence=0.6,
                    source="correlation_engine",
                    discovered_at=datetime.now(),
                    evidence={"similarity_score": 0.6, "method": "name_correlation"},
                )

        # Domain-email relationships
        if entity1.entity_type == EntityType.EMAIL and entity2.entity_type == EntityType.DOMAIN:
            if entity2.value in entity1.value:
                return OSINTRelationship(
                    relationship_id="",
                    source_entity_id=entity1.entity_id,
                    target_entity_id=entity2.entity_id,
                    relationship_type="belongs_to",
                    confidence=0.9,
                    source="domain_analysis",
                    discovered_at=datetime.now(),
                    evidence={"match_type": "exact_domain"},
                )

        return None

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0


class OSINTAutomator:
    """Main OSINT automation engine"""

    def __init__(self, config: OSINTConfig):
        self.config = config

        # Initialize connectors
        self.public_records = PublicRecordsConnector(config)
        self.breach_db = BreachDatabaseConnector(config)
        self.news_source = NewsSourceConnector(config)
        self.academic_source = AcademicSourceConnector(config)

        # Correlation engine
        self.correlation_engine = EntityCorrelationEngine(config)

        # Investigation tracking
        self.investigations: dict[str, OSINTResult] = {}

    async def investigate(
        self, query: str, entity_types: list[EntityType] | None = None
    ) -> OSINTResult:
        """Conduct OSINT investigation"""
        investigation_id = hashlib.md5(
            f"{query}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        start_time = datetime.now()

        logger.info(f"🔍 Starting OSINT investigation: {investigation_id}")

        all_entities = []
        sources_used = []

        try:
            # Search across enabled sources
            if OSINTSourceType.PUBLIC_RECORDS in self.config.enabled_sources:
                entities = await self.public_records.search_person(query)
                all_entities.extend(entities)
                if entities:
                    sources_used.append("public_records")

            if OSINTSourceType.BREACH_DATABASE in self.config.enabled_sources:
                if "@" in query:  # Email-like query
                    entities = await self.breach_db.check_email_breaches(query)
                    all_entities.extend(entities)
                    if entities:
                        sources_used.append("breach_database")

            if OSINTSourceType.NEWS_ARTICLES in self.config.enabled_sources:
                entities = await self.news_source.search_news_mentions(query)
                all_entities.extend(entities)
                if entities:
                    sources_used.append("news_articles")

            if OSINTSourceType.ACADEMIC_PAPERS in self.config.enabled_sources:
                entities = await self.academic_source.search_academic_papers(query)
                all_entities.extend(entities)
                if entities:
                    sources_used.append("academic_papers")

            # Filter by entity types if specified
            if entity_types:
                all_entities = [e for e in all_entities if e.entity_type in entity_types]

            # Filter by confidence threshold
            all_entities = [
                e for e in all_entities if e.confidence >= self.config.confidence_threshold
            ]

            # Correlate entities to find relationships
            relationships = []
            if self.config.enable_entity_linking and len(all_entities) > 1:
                relationships = await self.correlation_engine.correlate_entities(all_entities)

            # Calculate overall confidence
            confidence_score = self._calculate_investigation_confidence(all_entities, relationships)

            # Create result
            processing_time = (datetime.now() - start_time).total_seconds()

            result = OSINTResult(
                investigation_id=investigation_id,
                query=query,
                entities=all_entities,
                relationships=relationships,
                confidence_score=confidence_score,
                sources_used=sources_used,
                investigation_time=start_time,
                processing_time_seconds=processing_time,
            )

            # Store investigation
            self.investigations[investigation_id] = result

            logger.info(
                f"✅ Investigation completed: {len(all_entities)} entities, {len(relationships)} relationships"
            )
            return result

        except Exception as e:
            logger.error(f"Investigation failed: {e}")
            raise

    def _calculate_investigation_confidence(
        self, entities: list[OSINTEntity], relationships: list[OSINTRelationship]
    ) -> float:
        """Calculate overall confidence for investigation"""
        if not entities:
            return 0.0

        # Average entity confidence
        entity_confidence = sum(e.confidence for e in entities) / len(entities)

        # Relationship confidence bonus
        relationship_bonus = 0.0
        if relationships:
            relationship_confidence = sum(r.confidence for r in relationships) / len(relationships)
            relationship_bonus = relationship_confidence * 0.2  # 20% bonus for relationships

        # Source diversity bonus
        all_sources = set()
        for entity in entities:
            all_sources.update(entity.sources)

        source_bonus = min(len(all_sources) * 0.1, 0.3)  # Up to 30% bonus for multiple sources

        final_confidence = min(entity_confidence + relationship_bonus + source_bonus, 1.0)
        return final_confidence

    async def export_investigation(
        self, investigation_id: str, output_path: Path
    ) -> dict[str, Path]:
        """Export investigation results"""
        if investigation_id not in self.investigations:
            raise ValueError(f"Investigation {investigation_id} not found")

        result = self.investigations[investigation_id]
        exported_files = {}

        # JSON export
        if "json" in self.config.export_formats:
            json_file = output_path / f"investigation_{investigation_id}.json"
            with open(json_file, "w") as f:
                json.dump(self._serialize_result(result), f, indent=2)
            exported_files["json"] = json_file

        # CSV export for entities
        if "csv" in self.config.export_formats:
            import polars as pl

            # Entities CSV
            entities_data = []
            for entity in result.entities:
                entities_data.append(
                    {
                        "entity_id": entity.entity_id,
                        "entity_type": entity.entity_type.value,
                        "value": entity.value,
                        "confidence": entity.confidence,
                        "sources": ",".join(entity.sources),
                        "first_seen": entity.first_seen.isoformat(),
                        "last_seen": entity.last_seen.isoformat(),
                    }
                )

            if entities_data:
                entities_df = pl.DataFrame(entities_data)
                entities_file = output_path / f"investigation_{investigation_id}_entities.csv"
                entities_df.write_csv(entities_file)
                exported_files["entities_csv"] = entities_file

            # Relationships CSV
            relationships_data = []
            for rel in result.relationships:
                relationships_data.append(
                    {
                        "relationship_id": rel.relationship_id,
                        "source_entity": rel.source_entity_id,
                        "target_entity": rel.target_entity_id,
                        "relationship_type": rel.relationship_type,
                        "confidence": rel.confidence,
                        "source": rel.source,
                        "discovered_at": rel.discovered_at.isoformat(),
                    }
                )

            if relationships_data:
                relationships_df = pl.DataFrame(relationships_data)
                relationships_file = (
                    output_path / f"investigation_{investigation_id}_relationships.csv"
                )
                relationships_df.write_csv(relationships_file)
                exported_files["relationships_csv"] = relationships_file

        return exported_files

    def _serialize_result(self, result: OSINTResult) -> dict[str, Any]:
        """Serialize OSINTResult for JSON export"""
        return {
            "investigation_id": result.investigation_id,
            "query": result.query,
            "confidence_score": result.confidence_score,
            "sources_used": result.sources_used,
            "investigation_time": result.investigation_time.isoformat(),
            "processing_time_seconds": result.processing_time_seconds,
            "entities": [
                {
                    "entity_id": e.entity_id,
                    "entity_type": e.entity_type.value,
                    "value": e.value,
                    "confidence": e.confidence,
                    "sources": e.sources,
                    "first_seen": e.first_seen.isoformat(),
                    "last_seen": e.last_seen.isoformat(),
                    "attributes": e.attributes,
                }
                for e in result.entities
            ],
            "relationships": [
                {
                    "relationship_id": r.relationship_id,
                    "source_entity_id": r.source_entity_id,
                    "target_entity_id": r.target_entity_id,
                    "relationship_type": r.relationship_type,
                    "confidence": r.confidence,
                    "source": r.source,
                    "discovered_at": r.discovered_at.isoformat(),
                    "evidence": r.evidence,
                }
                for r in result.relationships
            ],
        }

    def get_investigation_stats(self) -> dict[str, Any]:
        """Get statistics about investigations"""
        if not self.investigations:
            return {"total_investigations": 0}

        results = list(self.investigations.values())

        return {
            "total_investigations": len(results),
            "average_entities_per_investigation": sum(len(r.entities) for r in results)
            / len(results),
            "average_confidence": sum(r.confidence_score for r in results) / len(results),
            "sources_usage": self._count_source_usage(results),
            "entity_types_found": self._count_entity_types(results),
        }

    def _count_source_usage(self, results: list[OSINTResult]) -> dict[str, int]:
        """Count usage of different sources"""
        source_counts = {}
        for result in results:
            for source in result.sources_used:
                source_counts[source] = source_counts.get(source, 0) + 1
        return source_counts

    def _count_entity_types(self, results: list[OSINTResult]) -> dict[str, int]:
        """Count different entity types found"""
        type_counts = {}
        for result in results:
            for entity in result.entities:
                entity_type = entity.entity_type.value
                type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        return type_counts


# Utility functions
async def quick_osint_search(query: str, max_sources: int = 3) -> OSINTResult:
    """Quick OSINT search with default configuration"""
    config = OSINTConfig(
        max_results_per_source=50, confidence_threshold=0.3, max_investigation_time_minutes=5
    )

    automator = OSINTAutomator(config)
    return await automator.investigate(query)


def create_safe_osint_config() -> OSINTConfig:
    """Create safe OSINT configuration for ethical research"""
    return OSINTConfig(
        enabled_sources=[
            OSINTSourceType.PUBLIC_RECORDS,
            OSINTSourceType.NEWS_ARTICLES,
            OSINTSourceType.ACADEMIC_PAPERS,
        ],
        respect_rate_limits=True,
        log_all_queries=True,
        anonymize_pii=True,
        confidence_threshold=0.6,
        max_investigation_time_minutes=15,
    )


__all__ = [
    "EntityType",
    "OSINTAutomator",
    "OSINTConfig",
    "OSINTEntity",
    "OSINTRelationship",
    "OSINTResult",
    "OSINTSourceType",
    "create_safe_osint_config",
    "quick_osint_search",
]
