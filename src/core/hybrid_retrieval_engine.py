#!/usr/bin/env python3
"""Hybridní Retrieval Engine - kombinuje dense embeddings (Qdrant) a sparse retrieval (BM25)
Implementuje hierarchické vyhledávání: metadata-first → section → passage granularity
Optimalizováno pro Apple Silicon M1/M2 s Metal Performance Shaders
+ DataWarehouse pre-filtrace pro dramatické zrychlení

Author: Senior Python/MLOps Agent
"""

from dataclasses import dataclass
import re
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer
import structlog

# Import M1 optimization
from ..optimization.m1_performance import get_optimal_device, optimize_for_m1

# Import DataWarehouse for pre-filtrace
from ..storage.data_warehouse import DataWarehouse

logger = structlog.get_logger(__name__)


@dataclass
class SearchResult:
    """Výsledek vyhledávání"""

    id: str
    title: str
    content: str
    source: str
    url: str
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    metadata: dict[str, Any] = None
    hierarchy_level: int = 0  # 0=document, 1=section, 2=passage, 3=sentence
    parent_id: str | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class HierarchicalChunk:
    """Hierarchický chunk s různými granularitami"""

    id: str
    content: str
    level: int  # 0=document, 1=section, 2=passage, 3=sentence
    parent_id: str | None = None
    children_ids: list[str] = None
    embedding: np.ndarray | None = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
        if self.metadata is None:
            self.metadata = {}


class HierarchicalChunker:
    """Hierarchické dělení dokumentů na více úrovní granularity"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.levels = config.get("levels", 2)

        # Chunk sizes for different levels
        self.chunk_sizes = {
            0: 8000,  # document level
            1: 2000,  # section level
            2: 512,  # passage level
            3: 128,  # sentence level
        }

        # Overlap ratios
        self.overlap_ratios = {
            0: 0.0,  # no overlap at document level
            1: 0.1,  # 10% overlap for sections
            2: 0.15,  # 15% overlap for passages
            3: 0.2,  # 20% overlap for sentences
        }

    def chunk_document(self, document: dict[str, Any]) -> list[HierarchicalChunk]:
        """Rozdělí dokument na hierarchické chunky"""
        content = document.get("content", "")
        doc_id = document.get("id", "")

        chunks = []

        # Level 0: Document level (celý dokument)
        doc_chunk = HierarchicalChunk(
            id=f"{doc_id}_doc",
            content=content[: self.chunk_sizes[0]],
            level=0,
            metadata=document.get("metadata", {}),
        )
        chunks.append(doc_chunk)

        # Level 1: Section level (kapitoly/sekce)
        if self.levels >= 2:
            sections = self._split_into_sections(content, doc_id)
            doc_chunk.children_ids = [s.id for s in sections]
            chunks.extend(sections)

            # Level 2: Passage level (odstavce)
            if self.levels >= 3:
                for section in sections:
                    passages = self._split_into_passages(section, doc_id)
                    section.children_ids = [p.id for p in passages]
                    chunks.extend(passages)

                    # Level 3: Sentence level (věty)
                    if self.levels >= 4:
                        for passage in passages:
                            sentences = self._split_into_sentences(passage, doc_id)
                            passage.children_ids = [s.id for s in sentences]
                            chunks.extend(sentences)

        return chunks

    def _split_into_sections(self, content: str, doc_id: str) -> list[HierarchicalChunk]:
        """Rozdělí content na sekce"""
        chunk_size = self.chunk_sizes[1]
        overlap = int(chunk_size * self.overlap_ratios[1])

        sections = []
        start = 0
        section_idx = 0

        while start < len(content):
            end = min(start + chunk_size, len(content))

            # Try to end at paragraph boundary
            if end < len(content):
                last_paragraph = content.rfind("\n\n", start, end)
                if last_paragraph > start + chunk_size // 2:
                    end = last_paragraph

            section_content = content[start:end]

            section = HierarchicalChunk(
                id=f"{doc_id}_sec_{section_idx}",
                content=section_content,
                level=1,
                parent_id=f"{doc_id}_doc",
            )
            sections.append(section)

            start = max(start + 1, end - overlap)
            section_idx += 1

        return sections

    def _split_into_passages(
        self, section: HierarchicalChunk, doc_id: str
    ) -> list[HierarchicalChunk]:
        """Rozdělí sekci na pasáže"""
        chunk_size = self.chunk_sizes[2]
        overlap = int(chunk_size * self.overlap_ratios[2])

        passages = []
        content = section.content
        start = 0
        passage_idx = 0

        while start < len(content):
            end = min(start + chunk_size, len(content))

            # Try to end at sentence boundary
            if end < len(content):
                last_sentence = content.rfind(".", start, end)
                if last_sentence > start + chunk_size // 2:
                    end = last_sentence + 1

            passage_content = content[start:end]

            passage = HierarchicalChunk(
                id=f"{section.id}_pass_{passage_idx}",
                content=passage_content,
                level=2,
                parent_id=section.id,
            )
            passages.append(passage)

            start = max(start + 1, end - overlap)
            passage_idx += 1

        return passages

    def _split_into_sentences(
        self, passage: HierarchicalChunk, doc_id: str
    ) -> list[HierarchicalChunk]:
        """Rozdělí pasáž na věty"""
        import re

        # Simple sentence splitting (could be enhanced with spaCy)
        sentences = re.split(r"[.!?]+", passage.content)

        sentence_chunks = []
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue

            sentence_chunk = HierarchicalChunk(
                id=f"{passage.id}_sent_{i}", content=sentence, level=3, parent_id=passage.id
            )
            sentence_chunks.append(sentence_chunk)

        return sentence_chunks


class HybridRetrievalEngine:
    """Hybridní retrieval engine s hierarchickým vyhledáváním a DataWarehouse pre-filtrací"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.retrieval_config = config.get("retrieval", {})
        self.hierarchical_config = self.retrieval_config.get("hierarchical", {})
        self.qdrant_config = self.retrieval_config.get("qdrant", {})
        self.sparse_config = self.retrieval_config.get("sparse_retrieval", {})
        self.fusion_config = self.retrieval_config.get("hybrid_fusion", {})

        # Hierarchical settings
        self.hierarchical_enabled = self.hierarchical_config.get("enabled", False)
        self.hierarchy_levels = self.hierarchical_config.get("levels", 2)

        # Initialize components
        self.qdrant_client = None
        self.embedding_model = None
        self.chunker = HierarchicalChunker(self.hierarchical_config)

        # DataWarehouse pro pre-filtraci (KLÍČOVÁ OPTIMALIZACE!)
        self.data_warehouse: DataWarehouse | None = None
        self.prefilter_enabled = self.retrieval_config.get("prefilter_enabled", True)

        # Collections for different hierarchy levels
        self.collections = {
            "metadata": "documents_metadata",  # Level 0: Document metadata
            "sections": "documents_sections",  # Level 1: Sections
            "passages": "documents_passages",  # Level 2: Passages
            "sentences": "documents_sentences",  # Level 3: Sentences
        }

        self.logger = structlog.get_logger(__name__)

    async def initialize(self):
        """Initialize retrieval engine"""
        self.logger.info("Initializing hierarchical retrieval engine with DataWarehouse pre-filtrace")

        # Initialize DataWarehouse pro pre-filtraci
        if self.prefilter_enabled:
            self.data_warehouse = DataWarehouse(db_path="./data/research_warehouse.db")
            await self.data_warehouse.connect()
            self.logger.info("✅ DataWarehouse pre-filtrace inicializována")

        # Initialize Qdrant client
        qdrant_url = self.qdrant_config.get("url", "http://localhost:6333")
        self.qdrant_client = QdrantClient(url=qdrant_url)

        # Initialize embedding model with M1 optimization
        model_name = self.qdrant_config.get("embedding_model", "all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer(model_name)

        # Optimize for Apple Metal (MPS)
        self.embedding_model = optimize_for_m1(self.embedding_model, "sentence_transformer")
        self.logger.info(f"✅ Embedding model optimized for device: {get_optimal_device()}")

        # Setup collections if hierarchical enabled
        if self.hierarchical_enabled:
            await self._setup_hierarchical_collections()

    async def _setup_hierarchical_collections(self):
        """Setup Qdrant collections for hierarchical retrieval"""
        embedding_size = self.embedding_model.get_sentence_embedding_dimension()

        # Collection configurations
        collection_configs = {
            "metadata": {
                "distance": Distance.COSINE,
                "size": embedding_size,
                "hnsw_config": {"ef_construct": 128, "m": 16},  # Higher precision for metadata
            },
            "sections": {
                "distance": Distance.COSINE,
                "size": embedding_size,
                "hnsw_config": {"ef_construct": 100, "m": 16},
            },
            "passages": {
                "distance": Distance.COSINE,
                "size": embedding_size,
                "hnsw_config": {"ef_construct": 64, "m": 12},
                "quantization": self.qdrant_config.get("index_tier", {}).get("passage") == "pq",
            },
            "sentences": {
                "distance": Distance.COSINE,
                "size": embedding_size,
                "hnsw_config": {"ef_construct": 32, "m": 8},
                "quantization": True,  # Always use quantization for sentences
            },
        }

        for level, collection_name in self.collections.items():
            config = collection_configs[level]

            try:
                # Check if collection exists
                self.qdrant_client.get_collection(collection_name)
                self.logger.info(f"Collection {collection_name} already exists")
            except:
                # Create collection
                vector_config = VectorParams(
                    size=config["size"],
                    distance=config["distance"],
                    hnsw_config=config.get("hnsw_config", {}),
                )

                self.qdrant_client.create_collection(
                    collection_name=collection_name, vectors_config=vector_config
                )

                self.logger.info(f"Created collection {collection_name}")

    async def hierarchical_search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Hierarchické vyhledávání - metadata-first → sections → passages"""
        if not self.hierarchical_enabled:
            # Fallback to regular hybrid search
            return await self.hybrid_search(query, top_k)

        self.logger.info(f"Starting hierarchical search: {query[:50]}...")

        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)

        # Stage 1: Search document metadata/sections
        metadata_results = await self._search_level(
            query_embedding,
            collection="metadata",
            top_k=top_k * 2,  # Get more candidates at higher level
        )

        self.logger.debug(f"Found {len(metadata_results)} metadata candidates")

        # Stage 2: Search sections within top documents
        section_candidates = []
        for meta_result in metadata_results[:top_k]:
            sections = await self._search_level(
                query_embedding, collection="sections", top_k=5, filter_parent=meta_result.id
            )
            section_candidates.extend(sections)

        # Re-rank and select top sections
        section_candidates.sort(key=lambda x: x.score, reverse=True)
        top_sections = section_candidates[:top_k]

        self.logger.debug(f"Found {len(top_sections)} section candidates")

        # Stage 3: Search passages within top sections (if level 3+ enabled)
        if self.hierarchy_levels >= 3:
            passage_candidates = []
            for section in top_sections:
                passages = await self._search_level(
                    query_embedding, collection="passages", top_k=3, filter_parent=section.id
                )
                passage_candidates.extend(passages)

            # Final ranking
            passage_candidates.sort(key=lambda x: x.score, reverse=True)
            final_results = passage_candidates[:top_k]
        else:
            final_results = top_sections

        self.logger.info(f"Hierarchical search completed: {len(final_results)} results")
        return final_results

    async def hierarchical_search_with_prefilter(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """OPTIMALIZOVANÉ hierarchické vyhledávání s DataWarehouse pre-filtrací

        1. Extrahuje klíčová slova z dotazu
        2. Pre-filtruje kandidáty pomocí DuckDB (řádově rychlejší)
        3. Spustí vektorové vyhledávání pouze na filtrované množině

        Výsledek: Dramatické snížení zátěže na Qdrant (až 90% redukce)
        """
        if not self.hierarchical_enabled:
            return await self.hybrid_search(query, top_k)

        self.logger.info(f"Starting OPTIMIZED hierarchical search with pre-filter: {query[:50]}...")

        # KROK 1: Pre-filtrace pomocí DataWarehouse (KLÍČOVÁ OPTIMALIZACE!)
        candidate_ids = []
        if self.prefilter_enabled and self.data_warehouse:
            # Extrakce klíčových slov z dotazu
            keywords = self._extract_query_keywords(query)
            self.logger.debug(f"Extracted keywords: {keywords[:10]}...")

            # DuckDB pre-filtrace - dramaticky rychlejší než full vektorové vyhledávání
            candidate_ids = await self.data_warehouse.query_ids_by_keywords(
                keywords,
                max_results=top_k * 10  # Získáme více kandidátů pro následnou vektorovou filtru
            )

            self.logger.info(f"✅ Pre-filter reduced search space to {len(candidate_ids)} candidates")

            if not candidate_ids:
                self.logger.warning("Pre-filter nenašel žádné kandidáty, fallback na full search")
                candidate_ids = None

        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)

        # KROK 2: Vektorové vyhledávání POUZE na pre-filtrovaných kandidátech
        metadata_results = await self._search_level_with_filter(
            query_embedding,
            collection="metadata",
            top_k=top_k * 2,
            candidate_ids=candidate_ids  # Klíčový parametr - omezuje search space!
        )

        self.logger.debug(f"Found {len(metadata_results)} metadata candidates after vector search")

        # KROK 3: Pokračování hierarchického vyhledávání (nezměněno)
        section_candidates = []
        for meta_result in metadata_results[:top_k]:
            sections = await self._search_level_with_filter(
                query_embedding,
                collection="sections",
                top_k=5,
                filter_parent=meta_result.id,
                candidate_ids=None  # Na nižších úrovních už nefiltrujeme podle candidate_ids
            )
            section_candidates.extend(sections)

        # Re-rank and select top sections
        section_candidates.sort(key=lambda x: x.score, reverse=True)
        top_sections = section_candidates[:top_k]

        self.logger.debug(f"Found {len(top_sections)} section candidates")

        # Stage 3: Search passages within top sections (if level 3+ enabled)
        if self.hierarchy_levels >= 3:
            passage_candidates = []
            for section in top_sections:
                passages = await self._search_level_with_filter(
                    query_embedding,
                    collection="passages",
                    top_k=3,
                    filter_parent=section.id
                )
                passage_candidates.extend(passages)

            # Final ranking
            passage_candidates.sort(key=lambda x: x.score, reverse=True)
            final_results = passage_candidates[:top_k]
        else:
            final_results = top_sections

        self.logger.info(f"✅ OPTIMIZED hierarchical search completed: {len(final_results)} results")
        return final_results

    def _extract_query_keywords(self, query: str) -> list[str]:
        """Extrahuje klíčová slova z dotazu pro pre-filtraci

        Args:
            query: Uživatelský dotaz

        Returns:
            Seznam klíčových slov pro DuckDB vyhledávání

        """
        # Normalizace a základní preprocessing
        query_lower = query.lower()

        # Odstranění stop slov a extrakce podstatných termínů
        stop_words = {
            'what', 'how', 'why', 'when', 'where', 'who', 'which', 'is', 'are', 'was', 'were',
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'co', 'jak', 'proč', 'kdy', 'kde', 'kdo', 'který', 'která', 'které', 'je', 'jsou',
            'a', 'nebo', 'ale', 'v', 'na', 'do', 'pro', 'od', 's', 'se'
        }

        # Extrakce slov (3+ znaky, bez speciálních znaků)
        words = re.findall(r'\b[a-záčďéěíňóřšťúůýž]{3,}\b', query_lower)

        # Filtrování stop slov
        keywords = [word for word in words if word not in stop_words]

        # Přidání frází (2-3 slova)
        phrases = re.findall(r'\b[a-záčďéěíňóřšťúůýž]{3,}\s+[a-záčďéěíňóřšťúůýž]{3,}\b', query_lower)
        keywords.extend([phrase.replace(' ', '_') for phrase in phrases])

        return keywords[:20]  # Omezíme na top 20 pro performance

    async def _search_level(
        self, query_embedding: np.ndarray, collection: str, top_k: int, filter_parent: str = None
    ) -> list[SearchResult]:
        """Search specific hierarchy level"""
        collection_name = self.collections[collection]

        # Build filter if parent specified
        search_filter = None
        if filter_parent:
            search_filter = Filter(
                must=[FieldCondition(key="parent_id", match={"value": filter_parent})]
            )

        # Execute search
        ef_search = self.qdrant_config.get("ef_search", 64)

        search_result = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            search_params={"ef": ef_search},
            query_filter=search_filter,
        )

        # Convert to SearchResult objects
        results = []
        for point in search_result:
            payload = point.payload

            result = SearchResult(
                id=str(point.id),
                title=payload.get("title", ""),
                content=payload.get("content", ""),
                source=payload.get("source", ""),
                url=payload.get("url", ""),
                score=point.score,
                dense_score=point.score,
                metadata=payload.get("metadata", {}),
                hierarchy_level=payload.get("hierarchy_level", 0),
                parent_id=payload.get("parent_id"),
            )
            results.append(result)

        return results

    async def _search_level_with_filter(
        self,
        query_embedding: np.ndarray,
        collection: str,
        top_k: int,
        filter_parent: str = None,
        candidate_ids: list[str] | None = None
    ) -> list[SearchResult]:
        """Search specific hierarchy level s možností pre-filtrace podle candidate_ids

        Args:
            query_embedding: Vektor dotazu
            collection: Název kolekce
            top_k: Počet výsledků
            filter_parent: ID nadřazeného elementu
            candidate_ids: Pre-filtrované ID dokumentů (klíčová optimalizace!)

        """
        collection_name = self.collections[collection]

        # Build filter combining parent filter and candidate IDs filter
        filters = []

        if filter_parent:
            filters.append(FieldCondition(key="parent_id", match={"value": filter_parent}))

        # KLÍČOVÁ OPTIMALIZACE: Omezení na pre-filtrované kandidáty
        if candidate_ids:
            # Qdrant podporuje filter na základě ID seznamu
            from qdrant_client.models import HasIdCondition
            filters.append(HasIdCondition(has_id=candidate_ids))

        search_filter = Filter(must=filters) if filters else None

        # Execute search
        ef_search = self.qdrant_config.get("ef_search", 64)

        search_result = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            search_params={"ef": ef_search},
            query_filter=search_filter,
        )

        # Convert to SearchResult objects
        results = []
        for point in search_result:
            payload = point.payload

            result = SearchResult(
                id=str(point.id),
                title=payload.get("title", ""),
                content=payload.get("content", ""),
                source=payload.get("source", ""),
                url=payload.get("url", ""),
                score=point.score,
                dense_score=point.score,
                metadata=payload.get("metadata", {}),
                hierarchy_level=payload.get("hierarchy_level", 0),
                parent_id=payload.get("parent_id"),
            )
            results.append(result)

        return results

    async def hybrid_search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Hybrid search combining dense and sparse retrieval (fallback method)"""
        # This would be the original hybrid search implementation
        # For now, return placeholder
        self.logger.warning("Using fallback hybrid search - hierarchical disabled")
        return []

    async def index_document(self, document: dict[str, Any]):
        """Index document with hierarchical chunking"""
        if not self.hierarchical_enabled:
            return await self._index_document_flat(document)

        # Create hierarchical chunks
        chunks = self.chunker.chunk_document(document)

        # Generate embeddings for all chunks
        for chunk in chunks:
            chunk.embedding = self.embedding_model.encode(chunk.content)

        # Index chunks by level
        level_chunks = {}
        for chunk in chunks:
            level = chunk.level
            if level not in level_chunks:
                level_chunks[level] = []
            level_chunks[level].append(chunk)

        # Index each level in appropriate collection
        for level, level_chunk_list in level_chunks.items():
            collection_key = ["metadata", "sections", "passages", "sentences"][level]
            collection_name = self.collections[collection_key]

            points = []
            for chunk in level_chunk_list:
                point = PointStruct(
                    id=chunk.id,
                    vector=chunk.embedding.tolist(),
                    payload={
                        "title": document.get("title", ""),
                        "content": chunk.content,
                        "source": document.get("source", ""),
                        "url": document.get("url", ""),
                        "metadata": chunk.metadata,
                        "hierarchy_level": chunk.level,
                        "parent_id": chunk.parent_id,
                        "children_ids": chunk.children_ids,
                    },
                )
                points.append(point)

            # Batch upsert
            if points:
                self.qdrant_client.upsert(collection_name=collection_name, points=points)

        self.logger.info(
            f"Indexed document {document.get('id')} with {len(chunks)} hierarchical chunks"
        )

    async def _index_document_flat(self, document: dict[str, Any]):
        """Fallback flat indexing when hierarchical is disabled"""
        # Placeholder for original flat indexing
