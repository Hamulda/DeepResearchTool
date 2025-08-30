"""RAG Pipeline pro zpracování dokumentů s chunking strategiemi
Implementace kompletního procesu ingesci a retrievalu

Author: Senior Python/MLOps Agent
"""

from dataclasses import dataclass
import logging
from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangChainDocument

from .memory import Document, get_memory_store

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Konfigurace pro RAG pipeline"""

    chunk_size: int = 1000
    chunk_overlap: int = 150
    separators: list[str] = None

    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", " ", ""]


class DocumentProcessor:
    """Třída pro zpracování a rozdělování dokumentů"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_text(self, text: str, metadata: dict[str, Any] = None) -> list[Document]:
        """Rozdělení textu na chunky pomocí RecursiveCharacterTextSplitter

        Args:
            text: Text k rozdělení
            metadata: Metadata pro všechny chunky

        Returns:
            Seznam Document objektů reprezentujících chunky

        """
        if metadata is None:
            metadata = {}

        # Vytvoření LangChain dokumentu
        langchain_doc = LangChainDocument(page_content=text, metadata=metadata)

        # Rozdělení na chunky
        chunks = self.text_splitter.split_documents([langchain_doc])

        # Konverze na naše Document objekty
        documents = []
        for i, chunk in enumerate(chunks):
            # Přidání chunk specifických metadat
            chunk_metadata = chunk.metadata.copy()
            chunk_metadata.update(
                {
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk.page_content),
                    "original_length": len(text),
                }
            )

            doc = Document(content=chunk.page_content, metadata=chunk_metadata)
            documents.append(doc)

        return documents

    def chunk_documents(self, documents: list[dict[str, Any]]) -> list[Document]:
        """Zpracování seznamu dokumentů a jejich rozdělení na chunky

        Args:
            documents: Seznam dokumentů s 'content' a volitelně 'metadata'

        Returns:
            Seznam všech chunků ze všech dokumentů

        """
        all_chunks = []

        for doc_index, doc in enumerate(documents):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            # Přidání dokumentu specifických metadat
            metadata["source_document_index"] = doc_index

            # Rozdělení dokumentu na chunky
            chunks = self.chunk_text(content, metadata)
            all_chunks.extend(chunks)

        return all_chunks


class RAGPipeline:
    """Kompletní RAG pipeline pro ingesci a retrieval"""

    def __init__(self, config: dict[str, Any]):
        self.config = config

        # Inicializace komponent
        rag_config = RAGConfig(
            chunk_size=config.get("chunking", {}).get("chunk_size", 1000),
            chunk_overlap=config.get("chunking", {}).get("chunk_overlap", 150),
            separators=config.get("chunking", {}).get("separators"),
        )

        self.document_processor = DocumentProcessor(rag_config)
        self.memory_store = get_memory_store(config)
        self._initialized = False

    async def initialize(self) -> None:
        """Inicializace RAG pipeline"""
        if not self._initialized:
            await self.memory_store.initialize()
            self._initialized = True
            logger.info("RAG Pipeline inicializována")

    async def ingest_documents(self, documents: list[dict[str, Any]]) -> list[str]:
        """Ingesci dokumentů do knowledge base

        Proces:
        1. Načtení dokumentů
        2. Rozdělení na chunky pomocí RecursiveCharacterTextSplitter
        3. Vytvoření embeddingů
        4. Uložení do ChromaDB

        Args:
            documents: Seznam dokumentů s 'content' a volitelně 'metadata'

        Returns:
            Seznam ID uložených chunků

        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"Zahajování ingesci {len(documents)} dokumentů")

        # Rozdělení dokumentů na chunky
        chunks = self.document_processor.chunk_documents(documents)
        logger.info(f"Vytvořeno {len(chunks)} chunků z {len(documents)} dokumentů")

        # Uložení chunků do memory store (embeddingy se vytvoří automaticky)
        chunk_ids = await self.memory_store.add_documents(chunks)

        logger.info(f"Úspěšně uloženo {len(chunk_ids)} chunků do knowledge base")
        return chunk_ids

    async def ingest_text(self, text: str, metadata: dict[str, Any] = None) -> list[str]:
        """Ingesci jednotlivého textu

        Args:
            text: Text k uložení
            metadata: Metadata pro text

        Returns:
            Seznam ID uložených chunků

        """
        documents = [{"content": text, "metadata": metadata or {}}]
        return await self.ingest_documents(documents)

    async def search(
        self, query: str, k: int = 5, filter_metadata: dict[str, Any] = None
    ) -> list[Document]:
        """Vyhledání relevantních dokumentů

        Proces:
        1. Vytvoření embeddingu z dotazu
        2. Vyhledání podobných chunků v ChromaDB
        3. Vrácení nejrelevantnějších chunků

        Args:
            query: Vyhledávací dotaz
            k: Počet výsledků k vrácení
            filter_metadata: Filtry pro metadata

        Returns:
            Seznam nejrelevantnějších dokumentů

        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"Vyhledávání pro dotaz: '{query}' (k={k})")

        # Vyhledání pomocí memory store
        results = await self.memory_store.search(query=query, k=k, where=filter_metadata)

        logger.info(f"Nalezeno {len(results)} relevantních dokumentů")
        return results

    async def get_context_for_query(self, query: str, max_tokens: int = 4000, k: int = 10) -> str:
        """Získání kontextu pro LLM na základě dotazu

        Args:
            query: Dotaz pro vyhledání kontextu
            max_tokens: Maximální počet tokenů v kontextu
            k: Počet dokumentů k vyhledání

        Returns:
            Formátovaný kontext pro LLM

        """
        # Vyhledání relevantních dokumentů
        documents = await self.search(query, k=k)

        if not documents:
            return "Žádné relevantní dokumenty nebyly nalezeny."

        # Sestavení kontextu s omezením tokenů
        context_parts = []
        current_length = 0

        for i, doc in enumerate(documents):
            content = doc.content
            # Hrubý odhad tokenů (1 token ≈ 4 znaky)
            estimated_tokens = len(content) // 4

            if current_length + estimated_tokens > max_tokens:
                break

            # Formátování s metadaty
            source = doc.metadata.get("source", "unknown")
            chunk_info = ""
            if "chunk_index" in doc.metadata:
                chunk_info = f" (chunk {doc.metadata['chunk_index'] + 1}/{doc.metadata.get('total_chunks', '?')})"

            context_part = f"--- Dokument {i+1} [{source}{chunk_info}] ---\n{content}\n"
            context_parts.append(context_part)
            current_length += estimated_tokens

        context = "\n".join(context_parts)

        logger.info(
            f"Vytvořen kontext z {len(context_parts)} dokumentů, odhadovaných {current_length} tokenů"
        )
        return context

    async def clear_knowledge_base(self) -> None:
        """Vymazání celé knowledge base"""
        if not self._initialized:
            await self.initialize()

        await self.memory_store.clear()
        logger.info("Knowledge base byla vymazána")


class HybridRAGPipeline(RAGPipeline):
    """Rozšířená RAG pipeline s hybridním vyhledáváním"""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.keyword_weight = config.get("hybrid", {}).get("keyword_weight", 0.3)
        self.semantic_weight = config.get("hybrid", {}).get("semantic_weight", 0.7)

    async def hybrid_search(
        self, query: str, k: int = 5, filter_metadata: dict[str, Any] = None
    ) -> list[Document]:
        """Hybridní vyhledávání kombinující sémantické a keyword vyhledávání

        Args:
            query: Vyhledávací dotaz
            k: Počet výsledků
            filter_metadata: Filtry pro metadata

        Returns:
            Seznam dokumentů seřazených podle kombinovaného skóre

        """
        # Sémantické vyhledávání
        semantic_results = await self.search(query, k=k * 2, filter_metadata=filter_metadata)

        # Keyword vyhledávání (jednoduchá implementace)
        keyword_results = await self._keyword_search(
            query, k=k * 2, filter_metadata=filter_metadata
        )

        # Kombinace výsledků pomocí RRF (Reciprocal Rank Fusion)
        combined_results = self._combine_results_rrf(semantic_results, keyword_results, k)

        return combined_results[:k]

    async def _keyword_search(
        self, query: str, k: int, filter_metadata: dict[str, Any] = None
    ) -> list[Document]:
        """Jednoduchá keyword vyhledávání implementace"""
        # Pro skutečnou implementaci by se použil např. BM25 nebo Elasticsearch
        # Zde používáme zjednodušenou verzi

        query_words = set(query.lower().split())
        all_docs = await self.memory_store.search("", k=k * 5)  # Získání více dokumentů

        scored_docs = []
        for doc in all_docs:
            content_words = set(doc.content.lower().split())
            overlap = len(query_words.intersection(content_words))

            if overlap > 0:
                score = overlap / len(query_words)
                doc.metadata["keyword_score"] = score
                scored_docs.append(doc)

        # Seřazení podle keyword skóre
        scored_docs.sort(key=lambda x: x.metadata.get("keyword_score", 0), reverse=True)
        return scored_docs[:k]

    def _combine_results_rrf(
        self, semantic_results: list[Document], keyword_results: list[Document], k: int
    ) -> list[Document]:
        """Kombinace výsledků pomocí Reciprocal Rank Fusion

        Args:
            semantic_results: Výsledky sémantického vyhledávání
            keyword_results: Výsledky keyword vyhledávání
            k: Konstanta pro RRF (obvykle 60)

        Returns:
            Kombinované a seřazené výsledky

        """
        rrf_k = 60  # Standardní konstanta pro RRF

        # Vytvoření mapování ID -> dokument
        doc_map = {}
        scores = {}

        # Přidání sémantických výsledků
        for rank, doc in enumerate(semantic_results):
            doc_id = doc.id or doc.content[:50]  # Fallback ID
            doc_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0) + self.semantic_weight / (rank + rrf_k)

        # Přidání keyword výsledků
        for rank, doc in enumerate(keyword_results):
            doc_id = doc.id or doc.content[:50]
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0) + self.keyword_weight / (rank + rrf_k)

        # Seřazení podle kombinovaného skóre
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Přidání RRF skóre do metadat
        result_docs = []
        for doc_id in sorted_ids[:k]:
            doc = doc_map[doc_id]
            doc.metadata["rrf_score"] = scores[doc_id]
            result_docs.append(doc)

        return result_docs
