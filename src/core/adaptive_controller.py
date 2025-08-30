#!/usr/bin/env python3
"""Adaptivní Controller pro Inteligentní Řízení Hloubky Výzkumu
Autonomně ukončuje výzkum když už nenachází nové, relevantní informace

Author: Senior Python/MLOps Agent
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
import logging
import re
from typing import Any

import numpy as np

# Import pro similarity comparison
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class ResearchStage(Enum):
    """Fáze výzkumného procesu"""

    INITIAL_SEARCH = "initial_search"
    ITERATIVE_REFINEMENT = "iterative_refinement"
    DEEP_ANALYSIS = "deep_analysis"
    CONVERGENCE_CHECK = "convergence_check"
    EARLY_TERMINATION = "early_termination"
    COMPLETION = "completion"


@dataclass
class InformationGainMetrics:
    """Metriky informačního přínosu"""

    new_concepts_count: int
    similarity_to_previous: float
    content_novelty_score: float
    citation_overlap_ratio: float
    semantic_diversity_score: float

    # Derived metrics
    information_gain_score: float = 0.0
    plateau_indicator: float = 0.0

    def __post_init__(self):
        self.information_gain_score = self._calculate_information_gain()
        self.plateau_indicator = self._calculate_plateau_indicator()

    def _calculate_information_gain(self) -> float:
        """Výpočet celkového informačního přínosu"""
        # Vážená kombinace metrik
        gain = (
            (self.new_concepts_count / 10.0) * 0.3 +          # Nové koncepty
            (1.0 - self.similarity_to_previous) * 0.4 +       # Nepodobnost k předchozímu
            self.content_novelty_score * 0.2 +                # Novost obsahu
            (1.0 - self.citation_overlap_ratio) * 0.1         # Nové citace
        )

        return max(0.0, min(1.0, gain))

    def _calculate_plateau_indicator(self) -> float:
        """Indikátor zda výzkum dosáhl plateau"""
        # Vysoká podobnost + nízká novost = plateau
        plateau = (
            self.similarity_to_previous * 0.5 +
            (1.0 - self.content_novelty_score) * 0.3 +
            self.citation_overlap_ratio * 0.2
        )

        return max(0.0, min(1.0, plateau))


@dataclass
class ResearchIteration:
    """Reprezentace jedné iterace výzkumu"""

    iteration_number: int
    synthesis: str
    retrieved_docs: list[dict[str, Any]]
    processing_time_ms: float

    # Computed metrics
    concepts_extracted: list[str] = None
    embedding: np.ndarray | None = None
    word_count: int = 0
    unique_sources: int = 0

    def __post_init__(self):
        if self.concepts_extracted is None:
            self.concepts_extracted = []
        self.word_count = len(self.synthesis.split())
        self.unique_sources = len(set(doc.get('source', '') for doc in self.retrieved_docs))


class AdaptiveController:
    """Adaptivní Controller pro Inteligentní Řízení Hloubky Výzkumu
    
    KLÍČOVÉ FUNKCE:
    - Hodnocení informačního přínosu mezi iteracemi
    - Detekce plateau (kdy už výzkum nepřináší nové informace)
    - Autonomní ukončení výzkumu při dosažení sufficiency
    - Prevence nekonečných smyček a plýtvání zdroji
    """

    def __init__(self, config: dict[str, Any]):
        """Inicializace Adaptivního Controlleru
        
        Args:
            config: Konfigurační slovník s adaptive_control sekcí

        """
        self.config = config.get("adaptive_control", {})

        # Thresholdy pro rozhodování
        self.max_iterations = self.config.get("max_iterations", 5)
        self.min_iterations = self.config.get("min_iterations", 2)
        self.information_gain_threshold = self.config.get("information_gain_threshold", 0.15)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.9)
        self.enable_early_stopping = self.config.get("enable_early_stopping", True)

        # Tracking
        self.research_history: list[ResearchIteration] = []
        self.current_stage = ResearchStage.INITIAL_SEARCH

        # Embedding model pro similarity comparison
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Statistics
        self.termination_stats = {
            "early_terminations": 0,
            "natural_completions": 0,
            "plateau_detections": 0,
            "avg_iterations": 0.0
        }

        logger.info("✅ Adaptivní Controller inicializován:")
        logger.info(f"  Max iterací: {self.max_iterations}")
        logger.info(f"  Information gain threshold: {self.information_gain_threshold}")
        logger.info(f"  Similarity threshold: {self.similarity_threshold}")

    async def assess_information_gain(self,
                                    previous_synthesis: str,
                                    current_synthesis: str,
                                    current_docs: list[dict[str, Any]],
                                    llm_client=None) -> float:
        """HLAVNÍ METODA: Hodnocení informačního přínosu mezi syntézami
        
        Args:
            previous_synthesis: Předchozí syntéza
            current_synthesis: Aktuální syntéza
            current_docs: Aktuálně načtené dokumenty
            llm_client: LLM klient pro pokročilé hodnocení
            
        Returns:
            Skóre informačního přínosu (0.0-1.0)

        """
        if not previous_synthesis or not current_synthesis:
            return 1.0  # První iterace má vždy maximální přínos

        logger.debug("🔍 Hodnotím informační přínos iterace")

        # 1. Semantic similarity analysis
        similarity = await self._calculate_semantic_similarity(previous_synthesis, current_synthesis)

        # 2. Content novelty analysis
        novelty_score = await self._analyze_content_novelty(previous_synthesis, current_synthesis)

        # 3. Concept extraction and comparison
        new_concepts = await self._extract_new_concepts(previous_synthesis, current_synthesis, llm_client)

        # 4. Citation overlap analysis
        citation_overlap = self._analyze_citation_overlap()

        # 5. Semantic diversity of sources
        diversity_score = self._calculate_source_diversity(current_docs)

        # Vytvoření metriky
        metrics = InformationGainMetrics(
            new_concepts_count=len(new_concepts),
            similarity_to_previous=similarity,
            content_novelty_score=novelty_score,
            citation_overlap_ratio=citation_overlap,
            semantic_diversity_score=diversity_score
        )

        logger.debug("📊 Information gain metrics:")
        logger.debug(f"  Semantic similarity: {similarity:.3f}")
        logger.debug(f"  Content novelty: {novelty_score:.3f}")
        logger.debug(f"  New concepts: {len(new_concepts)}")
        logger.debug(f"  Overall gain score: {metrics.information_gain_score:.3f}")

        return metrics.information_gain_score

    async def _calculate_semantic_similarity(self, previous: str, current: str) -> float:
        """Výpočet sémantické podobnosti mezi syntézami"""
        try:
            # Embeddings pro oba texty
            prev_embedding = self.embedding_model.encode([previous])
            curr_embedding = self.embedding_model.encode([current])

            # Cosine similarity
            similarity_matrix = cosine_similarity(prev_embedding, curr_embedding)
            similarity = float(similarity_matrix[0][0])

            return max(0.0, min(1.0, similarity))

        except Exception as e:
            logger.warning(f"Chyba při výpočtu semantic similarity: {e}")
            return 0.5  # Neutral default

    async def _analyze_content_novelty(self, previous: str, current: str) -> float:
        """Analýza novosti obsahu pomocí n-gram overlap"""
        try:
            # Tokenizace a vytvoření n-gramů
            prev_tokens = self._tokenize_text(previous)
            curr_tokens = self._tokenize_text(current)

            # Bigrams pro lepší analýzu
            prev_bigrams = set(zip(prev_tokens[:-1], prev_tokens[1:], strict=False))
            curr_bigrams = set(zip(curr_tokens[:-1], curr_tokens[1:], strict=False))

            # Výpočet překryvu
            if not prev_bigrams or not curr_bigrams:
                return 1.0

            overlap = len(prev_bigrams.intersection(curr_bigrams))
            union = len(prev_bigrams.union(curr_bigrams))

            # Jaccard index
            jaccard = overlap / max(1, union)

            # Novelty je inverzní k overlap
            novelty = 1.0 - jaccard

            return max(0.0, min(1.0, novelty))

        except Exception as e:
            logger.warning(f"Chyba při analýze content novelty: {e}")
            return 0.5

    async def _extract_new_concepts(self,
                                   previous: str,
                                   current: str,
                                   llm_client=None) -> list[str]:
        """Extrakce nových konceptů v aktuální syntéze"""
        new_concepts = []

        try:
            # Jednoduchá keyword-based analýza
            prev_keywords = self._extract_keywords(previous)
            curr_keywords = self._extract_keywords(current)

            # Nové klíčové pojmy
            new_keywords = curr_keywords - prev_keywords
            new_concepts.extend(list(new_keywords)[:10])  # Top 10

            # Pokročilá LLM analýza (pokud dostupná)
            if llm_client:
                llm_concepts = await self._llm_concept_extraction(previous, current, llm_client)
                new_concepts.extend(llm_concepts)

            # Deduplikace
            new_concepts = list(set(new_concepts))

        except Exception as e:
            logger.warning(f"Chyba při extrakci konceptů: {e}")

        return new_concepts[:15]  # Omezení na top 15

    async def _llm_concept_extraction(self,
                                     previous: str,
                                     current: str,
                                     llm_client) -> list[str]:
        """LLM-based extrakce nových konceptů"""
        try:
            prompt = f"""Compare these two research syntheses and identify NEW concepts, ideas, or information that appears in the second text but not in the first.

First synthesis:
{previous[:1000]}...

Second synthesis:
{current[:1000]}...

List only the genuinely NEW concepts introduced in the second text:
"""

            response = await llm_client.generate_async(
                prompt,
                max_tokens=200,
                temperature=0.1
            )

            # Parse response pro koncepty
            concepts = []
            lines = response.strip().split('\n')

            for line in lines:
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                    concept = line.lstrip('-•* ').strip()
                    if len(concept) > 5:  # Filtr příliš krátkých
                        concepts.append(concept)

            return concepts[:10]  # Top 10

        except Exception as e:
            logger.warning(f"LLM concept extraction selhala: {e}")
            return []

    def _extract_keywords(self, text: str) -> set:
        """Extrakce klíčových slov z textu"""
        # Normalizace a tokenizace
        text = text.lower()
        words = re.findall(r'\b[a-z]{4,}\b', text)  # Slova 4+ písmen

        # Filtrování stop words
        stop_words = {
            'that', 'this', 'with', 'from', 'they', 'have', 'been', 'were', 'said', 'what', 'when', 'where', 'while', 'would', 'could', 'should', 'about', 'after', 'before', 'between', 'through', 'during', 'above', 'below', 'under', 'over'
        }

        keywords = set(word for word in words if word not in stop_words)

        return keywords

    def _tokenize_text(self, text: str) -> list[str]:
        """Tokenizace textu pro n-gram analýzu"""
        # Normalizace
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)

        # Tokenizace
        tokens = text.split()

        # Filtrování krátkých tokens
        tokens = [token for token in tokens if len(token) > 2]

        return tokens

    def _analyze_citation_overlap(self) -> float:
        """Analýza překryvu citací mezi iteracemi"""
        if len(self.research_history) < 2:
            return 0.0

        try:
            # Porovnání posledních dvou iterací
            prev_iter = self.research_history[-2]
            curr_iter = self.research_history[-1]

            # Extrakce zdrojů
            prev_sources = set(doc.get('source', '') for doc in prev_iter.retrieved_docs)
            curr_sources = set(doc.get('source', '') for doc in curr_iter.retrieved_docs)

            if not prev_sources or not curr_sources:
                return 0.0

            # Výpočet překryvu
            overlap = len(prev_sources.intersection(curr_sources))
            union = len(prev_sources.union(curr_sources))

            overlap_ratio = overlap / max(1, union)

            return max(0.0, min(1.0, overlap_ratio))

        except Exception as e:
            logger.warning(f"Chyba při analýze citation overlap: {e}")
            return 0.0

    def _calculate_source_diversity(self, docs: list[dict[str, Any]]) -> float:
        """Výpočet diverzity zdrojů"""
        if not docs:
            return 0.0

        try:
            # Různé typy zdrojů
            source_types = set()
            domains = set()

            for doc in docs:
                source = doc.get('source', '').lower()

                # Kategorizace typu zdroje
                if 'arxiv' in source or 'pubmed' in source:
                    source_types.add('academic')
                elif '.gov' in source:
                    source_types.add('government')
                elif 'wikipedia' in source:
                    source_types.add('wikipedia')
                elif any(news in source for news in ['reuters', 'bbc', 'cnn', 'nytimes']):
                    source_types.add('news')
                else:
                    source_types.add('other')

                # Extrakce domény
                domain_match = re.search(r'https?://([^/]+)', source)
                if domain_match:
                    domains.add(domain_match.group(1))

            # Diversity score na základě počtu různých typů a domén
            type_diversity = len(source_types) / 5.0  # Max 5 typů
            domain_diversity = min(len(domains) / 10.0, 1.0)  # Max cap na 10 domén

            diversity = (type_diversity + domain_diversity) / 2.0

            return max(0.0, min(1.0, diversity))

        except Exception as e:
            logger.warning(f"Chyba při výpočtu source diversity: {e}")
            return 0.5

    def should_continue_research(self,
                               current_iteration: int,
                               last_information_gain: float,
                               synthesis_history: list[str]) -> tuple[bool, str]:
        """KLÍČOVÉ ROZHODOVÁNÍ: Má se pokračovat ve výzkumu?
        
        Args:
            current_iteration: Číslo aktuální iterace
            last_information_gain: Poslední skóre informačního přínosu
            synthesis_history: Historie syntéz
            
        Returns:
            Tuple (pokračovat?, důvod rozhodnutí)

        """
        # 1. Minimální počet iterací
        if current_iteration < self.min_iterations:
            return True, f"Minimální počet iterací ({self.min_iterations}) nedosažen"

        # 2. Maximální počet iterací
        if current_iteration >= self.max_iterations:
            self.termination_stats["natural_completions"] += 1
            return False, f"Dosažen maximální počet iterací ({self.max_iterations})"

        # 3. Early stopping kontroly (pokud povoleny)
        if self.enable_early_stopping:

            # Informační přínos pod prahem
            if last_information_gain < self.information_gain_threshold:
                self.termination_stats["early_terminations"] += 1
                self.termination_stats["plateau_detections"] += 1
                return False, f"Informační přínos ({last_information_gain:.3f}) pod prahem ({self.information_gain_threshold})"

            # Vysoká podobnost posledních syntéz
            if len(synthesis_history) >= 2:
                recent_similarity = await self._calculate_semantic_similarity(
                    synthesis_history[-2], synthesis_history[-1]
                )

                if recent_similarity > self.similarity_threshold:
                    self.termination_stats["early_terminations"] += 1
                    self.termination_stats["plateau_detections"] += 1
                    return False, f"Vysoká podobnost syntéz ({recent_similarity:.3f}) nad prahem ({self.similarity_threshold})"

        # 4. Pokračuj ve výzkumu
        return True, "Výzkum může pokračovat - dostatečný informační přínos"

    def add_research_iteration(self,
                             iteration_number: int,
                             synthesis: str,
                             retrieved_docs: list[dict[str, Any]],
                             processing_time_ms: float):
        """Přidání iterace do historie výzkumu"""
        iteration = ResearchIteration(
            iteration_number=iteration_number,
            synthesis=synthesis,
            retrieved_docs=retrieved_docs,
            processing_time_ms=processing_time_ms
        )

        # Compute embedding pro future similarity comparisons
        try:
            iteration.embedding = self.embedding_model.encode([synthesis])[0]
        except Exception as e:
            logger.warning(f"Chyba při výpočtu embedding: {e}")

        # Extract concepts
        iteration.concepts_extracted = list(self._extract_keywords(synthesis))[:20]

        self.research_history.append(iteration)

        # Update statistics
        total_sessions = (self.termination_stats["early_terminations"] +
                         self.termination_stats["natural_completions"])

        if total_sessions > 0:
            total_iterations = sum(len(session) for session in [self.research_history])  # Simplified
            self.termination_stats["avg_iterations"] = total_iterations / total_sessions

        logger.debug(f"📝 Přidána iterace {iteration_number} (syntéza: {len(synthesis)} znaků)")

    def get_research_summary(self) -> dict[str, Any]:
        """Získání shrnutí výzkumného procesu"""
        if not self.research_history:
            return {"message": "Žádná výzkumná data"}

        # Analýza historie
        total_docs = sum(len(iter.retrieved_docs) for iter in self.research_history)
        total_time = sum(iter.processing_time_ms for iter in self.research_history)
        unique_sources = set()

        for iteration in self.research_history:
            for doc in iteration.retrieved_docs:
                unique_sources.add(doc.get('source', ''))

        # Progression analysis
        word_counts = [iter.word_count for iter in self.research_history]
        concept_counts = [len(iter.concepts_extracted) for iter in self.research_history]

        return {
            "research_progression": {
                "total_iterations": len(self.research_history),
                "total_documents_processed": total_docs,
                "total_processing_time_ms": total_time,
                "unique_sources_found": len(unique_sources)
            },
            "content_evolution": {
                "synthesis_length_progression": word_counts,
                "concept_extraction_progression": concept_counts,
                "final_synthesis_length": word_counts[-1] if word_counts else 0
            },
            "termination_statistics": self.termination_stats,
            "efficiency_metrics": {
                "avg_time_per_iteration_ms": total_time / max(1, len(self.research_history)),
                "avg_docs_per_iteration": total_docs / max(1, len(self.research_history)),
                "research_convergence_rate": self._calculate_convergence_rate()
            },
            "configuration": {
                "max_iterations": self.max_iterations,
                "information_gain_threshold": self.information_gain_threshold,
                "similarity_threshold": self.similarity_threshold,
                "early_stopping_enabled": self.enable_early_stopping
            }
        }

    def _calculate_convergence_rate(self) -> float:
        """Výpočet rychlosti konvergence výzkumu"""
        if len(self.research_history) < 2:
            return 0.0

        try:
            # Počítej jak rychle se snižuje variabilita mezi iteracemi
            similarities = []

            for i in range(1, len(self.research_history)):
                if (self.research_history[i-1].embedding is not None and
                    self.research_history[i].embedding is not None):

                    sim = cosine_similarity(
                        [self.research_history[i-1].embedding],
                        [self.research_history[i].embedding]
                    )[0][0]
                    similarities.append(sim)

            if not similarities:
                return 0.0

            # Trend similarity - rostoucí znamená konvergenci
            if len(similarities) >= 2:
                trend = (similarities[-1] - similarities[0]) / max(1, len(similarities) - 1)
                return max(0.0, min(1.0, trend))

            return similarities[0]

        except Exception as e:
            logger.warning(f"Chyba při výpočtu convergence rate: {e}")
            return 0.0

    def reset_research_session(self):
        """Reset pro novou research session"""
        self.research_history.clear()
        self.current_stage = ResearchStage.INITIAL_SEARCH

        logger.info("🔄 Research session resetována")


# Factory funkce pro snadné použití
def create_adaptive_controller(config: dict[str, Any]) -> AdaptiveController:
    """Factory funkce pro vytvoření adaptivního controlleru
    
    Args:
        config: Konfigurační slovník
        
    Returns:
        Nakonfigurovaný AdaptiveController

    """
    controller = AdaptiveController(config)

    logger.info("✅ Adaptivní Controller vytvořen:")
    logger.info(f"  Information gain threshold: {controller.information_gain_threshold}")
    logger.info(f"  Max iterations: {controller.max_iterations}")
    logger.info(f"  Early stopping: {controller.enable_early_stopping}")

    return controller


if __name__ == "__main__":
    # Test základní funkcionality
    import asyncio

    async def test_adaptive_controller():
        """Test adaptivního controlleru"""
        config = {
            "adaptive_control": {
                "max_iterations": 3,
                "information_gain_threshold": 0.2,
                "similarity_threshold": 0.8,
                "enable_early_stopping": True
            }
        }

        controller = AdaptiveController(config)

        # Simulace research iterací
        test_syntheses = [
            "Artificial intelligence is a broad field of computer science.",
            "Artificial intelligence encompasses machine learning, deep learning, and neural networks. It represents a major advancement in computational capabilities.",
            "AI technologies including machine learning and deep learning continue to evolve. Neural networks remain central to these developments with minimal new breakthroughs."
        ]

        print("🧪 Test Adaptivního Controlleru:")
        print()

        for i, synthesis in enumerate(test_syntheses):
            # Přidání iterace
            controller.add_research_iteration(
                iteration_number=i+1,
                synthesis=synthesis,
                retrieved_docs=[{"source": f"source_{i}.com", "content": f"content {i}"}],
                processing_time_ms=1000.0
            )

            # Hodnocení informačního přínosu
            if i > 0:
                gain = await controller.assess_information_gain(
                    test_syntheses[i-1],
                    synthesis,
                    [{"source": f"source_{i}.com"}]
                )

                print(f"Iterace {i+1}:")
                print(f"  Information gain: {gain:.3f}")

                # Rozhodnutí o pokračování
                should_continue, reason = controller.should_continue_research(
                    current_iteration=i+1,
                    last_information_gain=gain,
                    synthesis_history=test_syntheses[:i+1]
                )

                print(f"  Pokračovat: {should_continue}")
                print(f"  Důvod: {reason}")
                print()

                if not should_continue:
                    break

        # Finální shrnutí
        summary = controller.get_research_summary()
        print("📊 Research Summary:")
        print(f"  Iterace: {summary['research_progression']['total_iterations']}")
        print(f"  Early terminations: {summary['termination_statistics']['early_terminations']}")
        print(f"  Convergence rate: {summary['efficiency_metrics']['research_convergence_rate']:.3f}")

    # Spuštění testu
    asyncio.run(test_adaptive_controller())
