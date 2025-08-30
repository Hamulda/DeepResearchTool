"""Contextual Compression Engine
Implementuje salience + novelty + redundancy filtering pro M1 s kvantizovaným LLM

Author: Senior Python/MLOps Agent
"""

import asyncio
from dataclasses import dataclass
import logging
from typing import Any

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

from ..optimization.m1_performance import cleanup_memory, get_optimal_device, optimize_for_m1

logger = logging.getLogger(__name__)


@dataclass
class CompressionConfig:
    """Konfigurace pro contextual compression"""

    budget_tokens: int = 2000
    salience_weight: float = 0.4
    novelty_weight: float = 0.3
    redundancy_weight: float = 0.3
    min_relevance_threshold: float = 0.5
    use_llm_rater: bool = True
    source_priority_weights: dict[str, float] = None


@dataclass
class DocumentChunk:
    """Chunk dokumentu s metadaty"""

    id: str
    content: str
    source: str
    url: str
    tokens: int
    salience_score: float = 0.0
    novelty_score: float = 0.0
    redundancy_penalty: float = 0.0
    final_score: float = 0.0
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ContextualCompressor:
    """Contextual Compression Engine pro M1 8GB
    Implementuje salience + novelty + redundancy bez cross-encoderu
    """

    def __init__(self, config: CompressionConfig):
        self.config = config
        self.embedding_model = None
        self.tfidf_vectorizer = None
        self.nlp = None
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Inicializace kompresního enginu"""
        self.logger.info("🚀 Inicializuji Contextual Compressor pro M1...")

        # Lightweight embedding model pro M1
        model_name = "all-MiniLM-L6-v2"  # Pouze 80MB
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_model = optimize_for_m1(self.embedding_model, "sentence_transformer")

        # TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,  # Omezeno pro M1
            stop_words='english',
            ngram_range=(1, 2)
        )

        # Lightweight spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.warning("spaCy model not found, using simple tokenization")
            self.nlp = None

        self.logger.info("✅ Contextual Compressor inicializován")

    async def compress_context(
        self,
        documents: list[dict[str, Any]],
        query: str,
        budget_tokens: int | None = None
    ) -> list[DocumentChunk]:
        """Hlavní funkce pro kompresi kontextu

        Args:
            documents: Seznam dokumentů k kompresi
            query: Původní dotaz pro relevance scoring
            budget_tokens: Token budget (optional override)

        Returns:
            Komprimovaný seznam document chunků

        """
        if not documents:
            return []

        budget = budget_tokens or self.config.budget_tokens
        self.logger.info(f"🗜️ Začínám kompresi {len(documents)} dokumentů, budget: {budget} tokenů")

        # 1. Převod na DocumentChunk objekty
        chunks = self._prepare_chunks(documents)

        # 2. Výpočet salience scores
        await self._calculate_salience_scores(chunks, query)

        # 3. Výpočet novelty scores
        await self._calculate_novelty_scores(chunks)

        # 4. Výpočet redundancy penalties
        await self._calculate_redundancy_penalties(chunks)

        # 5. Finální scoring a selekce
        selected_chunks = self._select_chunks_by_budget(chunks, budget)

        # 6. Memory cleanup
        cleanup_memory()

        self.logger.info(f"✅ Komprese dokončena: {len(selected_chunks)}/{len(chunks)} chunků vybráno")
        return selected_chunks

    def _prepare_chunks(self, documents: list[dict[str, Any]]) -> list[DocumentChunk]:
        """Připraví DocumentChunk objekty"""
        chunks = []

        for i, doc in enumerate(documents):
            # Odhad počtu tokenů (1 token ≈ 4 znaky)
            content = doc.get('content', '')
            estimated_tokens = len(content) // 4

            chunk = DocumentChunk(
                id=f"chunk_{i}",
                content=content,
                source=doc.get('source', 'unknown'),
                url=doc.get('url', ''),
                tokens=estimated_tokens,
                metadata=doc.get('metadata', {})
            )
            chunks.append(chunk)

        return chunks

    async def _calculate_salience_scores(self, chunks: list[DocumentChunk], query: str):
        """Výpočet salience scores (relevance k dotazu)"""
        self.logger.debug("📊 Počítám salience scores...")

        # Semantic similarity pomocí embeddings
        query_embedding = self.embedding_model.encode([query])
        content_embeddings = self.embedding_model.encode([chunk.content for chunk in chunks])

        semantic_similarities = cosine_similarity(query_embedding, content_embeddings)[0]

        # TF-IDF similarity
        all_texts = [query] + [chunk.content for chunk in chunks]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
        tfidf_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

        # Entity/keyword matching
        query_keywords = self._extract_keywords(query)

        for i, chunk in enumerate(chunks):
            # Kombinace semantic + TF-IDF + keywords
            semantic_score = semantic_similarities[i]
            tfidf_score = tfidf_similarities[i]
            keyword_score = self._calculate_keyword_overlap(chunk.content, query_keywords)

            # Source priority weighting
            source_weight = self._get_source_priority(chunk.source)

            # Finální salience score
            salience = (
                0.4 * semantic_score +
                0.3 * tfidf_score +
                0.3 * keyword_score
            ) * source_weight

            chunk.salience_score = salience

    async def _calculate_novelty_scores(self, chunks: list[DocumentChunk]):
        """Výpočet novelty scores (odlišnost od ostatních dokumentů)"""
        self.logger.debug("🆕 Počítám novelty scores...")

        if len(chunks) <= 1:
            for chunk in chunks:
                chunk.novelty_score = 1.0
            return

        # Embeddings pro všechny chunky
        embeddings = self.embedding_model.encode([chunk.content for chunk in chunks])

        # Výpočet průměrné podobnosti s ostatními chunky
        similarity_matrix = cosine_similarity(embeddings)

        for i, chunk in enumerate(chunks):
            # Průměrná podobnost s ostatními (kromě sebe)
            similarities = similarity_matrix[i]
            avg_similarity = (similarities.sum() - similarities[i]) / (len(similarities) - 1)

            # Novelty = 1 - průměrná podobnost
            chunk.novelty_score = 1.0 - avg_similarity

    async def _calculate_redundancy_penalties(self, chunks: list[DocumentChunk]):
        """Výpočet redundancy penalties"""
        self.logger.debug("🔄 Počítám redundancy penalties...")

        # Seřazení podle salience score (sestupně)
        sorted_chunks = sorted(chunks, key=lambda x: x.salience_score, reverse=True)

        selected_contents = []

        for chunk in sorted_chunks:
            if not selected_contents:
                chunk.redundancy_penalty = 0.0
                selected_contents.append(chunk.content)
                continue

            # Výpočet podobnosti s již vybranými chunky
            chunk_embedding = self.embedding_model.encode([chunk.content])
            selected_embeddings = self.embedding_model.encode(selected_contents)

            similarities = cosine_similarity(chunk_embedding, selected_embeddings)[0]
            max_similarity = similarities.max()

            # Redundancy penalty na základě max podobnosti
            if max_similarity > 0.85:
                chunk.redundancy_penalty = 0.8  # Vysoká penalizace
            elif max_similarity > 0.70:
                chunk.redundancy_penalty = 0.5  # Střední penalizace
            elif max_similarity > 0.55:
                chunk.redundancy_penalty = 0.2  # Nízká penalizace
            else:
                chunk.redundancy_penalty = 0.0  # Žádná penalizace
                selected_contents.append(chunk.content)

    def _select_chunks_by_budget(self, chunks: list[DocumentChunk], budget: int) -> list[DocumentChunk]:
        """Selekce chunků podle token budget"""
        self.logger.debug(f"💰 Vybírám chunky pro budget {budget} tokenů...")

        # Výpočet finálních scores
        for chunk in chunks:
            chunk.final_score = (
                self.config.salience_weight * chunk.salience_score +
                self.config.novelty_weight * chunk.novelty_score -
                self.config.redundancy_weight * chunk.redundancy_penalty
            )

        # Seřazení podle finálního score
        sorted_chunks = sorted(chunks, key=lambda x: x.final_score, reverse=True)

        # Greedy selection podle budget
        selected = []
        current_tokens = 0

        for chunk in sorted_chunks:
            if current_tokens + chunk.tokens <= budget:
                selected.append(chunk)
                current_tokens += chunk.tokens

            if current_tokens >= budget * 0.95:  # 95% budget využití
                break

        return selected

    def _extract_keywords(self, text: str) -> list[str]:
        """Extrakce klíčových slov z textu"""
        if self.nlp:
            doc = self.nlp(text)
            keywords = [
                token.lemma_.lower()
                for token in doc
                if not token.is_stop and not token.is_punct and len(token.text) > 2
            ]
        else:
            # Fallback - simple word extraction
            import re
            words = re.findall(r'\b\w{3,}\b', text.lower())
            keywords = list(set(words))

        return keywords

    def _calculate_keyword_overlap(self, content: str, query_keywords: list[str]) -> float:
        """Výpočet keyword overlap"""
        if not query_keywords:
            return 0.0

        content_keywords = self._extract_keywords(content)
        if not content_keywords:
            return 0.0

        overlap = len(set(query_keywords) & set(content_keywords))
        return overlap / len(query_keywords)

    def _get_source_priority(self, source: str) -> float:
        """Získá prioritu zdroje"""
        if not self.config.source_priority_weights:
            return 1.0

        # Normalizace source name
        source_normalized = source.lower()

        for source_type, weight in self.config.source_priority_weights.items():
            if source_type in source_normalized:
                return weight

        return self.config.source_priority_weights.get('unknown', 0.5)

    async def get_compression_stats(self, chunks: list[DocumentChunk]) -> dict[str, Any]:
        """Získá statistiky komprese"""
        if not chunks:
            return {}

        total_tokens = sum(chunk.tokens for chunk in chunks)
        avg_salience = sum(chunk.salience_score for chunk in chunks) / len(chunks)
        avg_novelty = sum(chunk.novelty_score for chunk in chunks) / len(chunks)
        avg_redundancy = sum(chunk.redundancy_penalty for chunk in chunks) / len(chunks)

        return {
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
            "avg_salience_score": round(avg_salience, 3),
            "avg_novelty_score": round(avg_novelty, 3),
            "avg_redundancy_penalty": round(avg_redundancy, 3),
            "compression_ratio": round(total_tokens / self.config.budget_tokens, 2),
            "device": get_optimal_device()
        }


# Utilita pro rychlé použití
async def compress_documents(
    documents: list[dict[str, Any]],
    query: str,
    budget_tokens: int = 2000
) -> tuple[list[DocumentChunk], dict[str, Any]]:
    """Convenience funkce pro rychlou kompresi dokumentů

    Returns:
        Tuple[compressed_chunks, compression_stats]

    """
    config = CompressionConfig(budget_tokens=budget_tokens)
    compressor = ContextualCompressor(config)

    await compressor.initialize()

    compressed = await compressor.compress_context(documents, query, budget_tokens)
    stats = await compressor.get_compression_stats(compressed)

    return compressed, stats


# Příklad použití
async def example_compression():
    """Příklad použití contextual compression"""
    # Testovací dokumenty
    documents = [
        {
            "content": "Artificial intelligence is transforming healthcare through machine learning algorithms.",
            "source": "academic",
            "url": "https://example.com/ai-healthcare"
        },
        {
            "content": "Machine learning models can predict patient outcomes with high accuracy.",
            "source": "medical",
            "url": "https://example.com/ml-prediction"
        },
        {
            "content": "Social media platforms use AI for content recommendation systems.",
            "source": "news",
            "url": "https://example.com/social-ai"
        }
    ]

    query = "How is AI being used in healthcare?"

    # Komprese
    compressed, stats = await compress_documents(documents, query, budget_tokens=500)

    print("Komprese dokončena:")
    print(f"Původní dokumenty: {len(documents)}")
    print(f"Komprimované chunks: {len(compressed)}")
    print(f"Statistiky: {stats}")

    for i, chunk in enumerate(compressed):
        print(f"\nChunk {i+1}:")
        print(f"  Score: {chunk.final_score:.3f}")
        print(f"  Content: {chunk.content[:100]}...")


if __name__ == "__main__":
    asyncio.run(example_compression())
