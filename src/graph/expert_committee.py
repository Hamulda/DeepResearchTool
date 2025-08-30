"""
Multi-Agent Expert Committee Architecture
Implementace výboru expertů pro komplexní výzkumné úlohy
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod

from langgraph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from src.observability.langfuse_integration import trace_research_operation

logger = logging.getLogger(__name__)


class ExpertType(Enum):
    """Typy expertních agentů"""

    ACADEMIC = "academic_expert"
    WEB_ANALYST = "web_analyst"
    TECHNICAL = "technical_expert"
    BUSINESS = "business_analyst"
    COORDINATOR = "coordinator"


@dataclass
class ExpertResponse:
    """Odpověď od expertního agenta"""

    expert_type: ExpertType
    confidence: float
    findings: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class CommitteeState:
    """Stav multi-agentního výboru"""

    original_query: str
    query_analysis: Dict[str, Any]
    expert_responses: List[ExpertResponse]
    synthesis_result: Optional[str] = None
    confidence_scores: Dict[str, float] = None
    final_answer: Optional[str] = None
    iteration: int = 0
    max_iterations: int = 3


class ExpertAgent(ABC):
    """Abstraktní třída pro expertního agenta"""

    def __init__(self, expert_type: ExpertType, llm_client, tools: List[Any]):
        self.expert_type = expert_type
        self.llm_client = llm_client
        self.tools = tools
        self.specialization = self._get_specialization()

    @abstractmethod
    def _get_specialization(self) -> str:
        """Vrátí popis specializace agenta"""
        pass

    @abstractmethod
    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analýza dotazu z pohledu experta"""
        pass

    @trace_research_operation("expert_research")
    async def conduct_research(self, query: str, context: Dict[str, Any] = None) -> ExpertResponse:
        """Provede výzkum v oblasti své specializace"""
        try:
            # Analýza relevantnosti dotazu
            analysis = await self.analyze_query(query)

            if analysis["relevance_score"] < 0.3:
                return ExpertResponse(
                    expert_type=self.expert_type,
                    confidence=0.1,
                    findings=f"Dotaz není relevantní pro {self.specialization}",
                    sources=[],
                    metadata=analysis,
                )

            # Provedení specializovaného výzkumu
            findings = await self._conduct_specialized_research(query, analysis)

            return ExpertResponse(
                expert_type=self.expert_type,
                confidence=analysis["confidence"],
                findings=findings["content"],
                sources=findings["sources"],
                metadata=findings["metadata"],
            )

        except Exception as e:
            logger.error(f"Error in {self.expert_type} research: {e}")
            return ExpertResponse(
                expert_type=self.expert_type,
                confidence=0.0,
                findings=f"Chyba při výzkumu: {str(e)}",
                sources=[],
                metadata={"error": str(e)},
            )

    @abstractmethod
    async def _conduct_specialized_research(
        self, query: str, analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Specializovaný výzkum podle typu experta"""
        pass


class AcademicExpert(ExpertAgent):
    """Expert pro akademický výzkum"""

    def _get_specialization(self) -> str:
        return "akademický výzkum, peer-reviewed publikace, vědecké studie"

    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analýza z akademického pohledu"""
        prompt = f"""
        Analyzuj následující dotaz z pohledu akademického výzkumu:
        
        Dotaz: {query}
        
        Vyhodnoť:
        1. Relevantnost pro akademický výzkum (0.0-1.0)
        2. Potřeba peer-reviewed zdrojů
        3. Vědecké obory, které se dotýká
        4. Metodologické aspekty
        
        Odpověz ve formátu JSON s klíči: relevance_score, confidence, scientific_fields, methodology_needed
        """

        response = await self.llm_client.generate(prompt, temperature=0.1)

        try:
            import json

            return json.loads(response)
        except:
            return {
                "relevance_score": 0.7,
                "confidence": 0.6,
                "scientific_fields": ["general"],
                "methodology_needed": True,
            }

    async def _conduct_specialized_research(
        self, query: str, analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Akademický výzkum pomocí specializovaných nástrojů"""

        # Použití Semantic Scholar a akademických databází
        from src.core.enhanced_tools import semantic_scholar_search

        sources = []
        content_parts = []

        # Vyhledání v akademických databázích
        for field in analysis.get("scientific_fields", ["general"]):
            search_query = f"{query} {field}"
            academic_results = await semantic_scholar_search(search_query, limit=5)

            sources.extend(academic_results.get("sources", []))

            if academic_results.get("content"):
                content_parts.append(f"Akademické výsledky pro {field}:")
                content_parts.append(academic_results["content"])

        # Syntéza akademických zjištění
        synthesis_prompt = f"""
        Jako akademický expert syntetizuj následující výzkumné výsledky:
        
        Původní dotaz: {query}
        
        Akademické zdroje:
        {chr(10).join(content_parts)}
        
        Poskytni:
        1. Shrnutí současného stavu výzkumu
        2. Klíčové studie a jejich závěry
        3. Metodologické aspekty
        4. Oblasti pro další výzkum
        """

        synthesis = await self.llm_client.generate(synthesis_prompt, temperature=0.2)

        return {
            "content": synthesis,
            "sources": sources,
            "metadata": {
                "search_fields": analysis.get("scientific_fields"),
                "peer_reviewed_count": len([s for s in sources if s.get("peer_reviewed", False)]),
            },
        }


class WebAnalyst(ExpertAgent):
    """Expert pro analýzu webových zdrojů"""

    def _get_specialization(self) -> str:
        return "analýza webových zdrojů, aktuální trendy, online databáze"

    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analýza z pohledu webové analýzy"""
        prompt = f"""
        Analyzuj dotaz z pohledu webové analýzy a aktuálních trendů:
        
        Dotaz: {query}
        
        Vyhodnoť:
        1. Potřeba aktuálních informací (0.0-1.0)
        2. Relevantnost pro webové zdroje
        3. Časovou citlivost informací
        4. Typy webových zdrojů potřebných
        
        JSON odpověď s klíči: relevance_score, confidence, time_sensitivity, source_types
        """

        response = await self.llm_client.generate(prompt, temperature=0.1)

        try:
            import json

            return json.loads(response)
        except:
            return {
                "relevance_score": 0.8,
                "confidence": 0.7,
                "time_sensitivity": "high",
                "source_types": ["news", "blogs", "official_sites"],
            }

    async def _conduct_specialized_research(
        self, query: str, analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Webová analýza pomocí scraperů a API"""

        sources = []
        content_parts = []

        # Použití web scraperů a API
        from src.scrapers.firecrawl_scraper import FirecrawlScraper

        scraper = FirecrawlScraper()

        # Vyhledání relevantních webových zdrojů
        search_terms = [query]
        if "recent" in query.lower() or "2024" in query:
            search_terms.append(f"{query} 2024")

        for term in search_terms:
            try:
                web_results = await scraper.search_and_scrape(term, max_results=3)
                sources.extend(web_results.get("sources", []))

                if web_results.get("content"):
                    content_parts.append(f"Webové výsledky pro '{term}':")
                    content_parts.append(web_results["content"])
            except Exception as e:
                logger.warning(f"Web search failed for {term}: {e}")

        # Analýza trendů a aktualit
        synthesis_prompt = f"""
        Jako web analytik analyzuj aktuální trendy a informace:
        
        Dotaz: {query}
        
        Webové zdroje:
        {chr(10).join(content_parts)}
        
        Poskytni:
        1. Aktuální stav a trendy
        2. Klíčové události a vývoj
        3. Relevantní statistiky a data
        4. Budoucí outlook
        """

        synthesis = await self.llm_client.generate(synthesis_prompt, temperature=0.3)

        return {
            "content": synthesis,
            "sources": sources,
            "metadata": {
                "search_terms": search_terms,
                "time_sensitivity": analysis.get("time_sensitivity"),
                "source_freshness": max(
                    [s.get("publish_date", "2020-01-01") for s in sources] + ["2020-01-01"]
                ),
            },
        }


class TechnicalExpert(ExpertAgent):
    """Expert pro technické analýzy"""

    def _get_specialization(self) -> str:
        return "technické specifikace, architektura, implementační detaily"

    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analýza z technického pohledu"""
        prompt = f"""
        Analyzuj dotaz z technického pohledu:
        
        Dotaz: {query}
        
        Vyhodnoť:
        1. Technickou složitost (0.0-1.0)
        2. Potřeba technických specifikací
        3. Relevantní technologie
        4. Implementační aspekty
        
        JSON s klíči: relevance_score, confidence, complexity, technologies, implementation_focus
        """

        response = await self.llm_client.generate(prompt, temperature=0.1)

        try:
            import json

            return json.loads(response)
        except:
            return {
                "relevance_score": 0.6,
                "confidence": 0.8,
                "complexity": 0.7,
                "technologies": ["general"],
                "implementation_focus": True,
            }

    async def _conduct_specialized_research(
        self, query: str, analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Technický výzkum"""

        sources = []
        content_parts = []

        # Technické databáze a dokumentace
        tech_terms = analysis.get("technologies", ["general"])

        for tech in tech_terms:
            # Zde by byly volání na technické API, dokumentace, atd.
            content_parts.append(f"Technické informace pro {tech}:")
            content_parts.append(f"Placeholder pro technické detaily {tech}")

        synthesis_prompt = f"""
        Jako technický expert analyzuj technické aspekty:
        
        Dotaz: {query}
        
        Technické informace:
        {chr(10).join(content_parts)}
        
        Poskytni:
        1. Technické specifikace
        2. Architekturní aspekty
        3. Implementační doporučení
        4. Technická omezení a výzvy
        """

        synthesis = await self.llm_client.generate(synthesis_prompt, temperature=0.2)

        return {
            "content": synthesis,
            "sources": sources,
            "metadata": {
                "technologies": tech_terms,
                "complexity_score": analysis.get("complexity"),
            },
        }


class CoordinatorAgent:
    """Koordinátor pro řízení výboru expertů"""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    async def analyze_query_requirements(self, query: str) -> Dict[str, Any]:
        """Analýza požadavků na experty"""
        prompt = f"""
        Analyzuj tento výzkumný dotaz a urči, kteří experti jsou potřeba:
        
        Dotaz: {query}
        
        Dostupní experti:
        - academic_expert: akademický výzkum, peer-reviewed studie
        - web_analyst: aktuální trendy, webové zdroje
        - technical_expert: technické specifikace, implementace
        - business_analyst: business analýza, trh, finance
        
        Vyhodnoť pro každého experta:
        1. Relevantnost (0.0-1.0)
        2. Prioritu (high/medium/low)
        
        JSON odpověď s experty jako klíči a jejich hodnocením.
        """

        response = await self.llm_client.generate(prompt, temperature=0.1)

        try:
            import json

            return json.loads(response)
        except:
            # Fallback - zapojit všechny experty
            return {
                "academic_expert": {"relevance": 0.7, "priority": "medium"},
                "web_analyst": {"relevance": 0.8, "priority": "high"},
                "technical_expert": {"relevance": 0.6, "priority": "low"},
            }

    async def synthesize_expert_responses(self, query: str, responses: List[ExpertResponse]) -> str:
        """Syntéza odpovědí od expertů"""

        if not responses:
            return "Žádné expertní odpovědi k dispozici."

        # Příprava expertních zjištění
        expert_findings = []
        for response in responses:
            if response.confidence > 0.3:  # Pouze relevantní odpovědi
                expert_findings.append(
                    f"**{response.expert_type.value}** (confidence: {response.confidence:.2f}):\n"
                    f"{response.findings}\n"
                )

        synthesis_prompt = f"""
        Jako koordinátor výboru expertů syntetizuj následující expertní analýzy:
        
        Původní dotaz: {query}
        
        Expertní zjištění:
        {chr(10).join(expert_findings)}
        
        Vytvoř komprehensivní odpověď, která:
        1. Integruje poznatky od všech expertů
        2. Vyhodnotí konzistenci a konflikty
        3. Poskytne vyvážený závěr
        4. Identifikuje oblasti nejistoty
        
        Struktura odpovědi:
        - Shrnutí hlavních zjištění
        - Integrovaná analýza
        - Závěry a doporučení
        - Omezení a nejistoty
        """

        synthesis = await self.llm_client.generate(synthesis_prompt, temperature=0.3)
        return synthesis


class ExpertCommitteeGraph:
    """LangGraph implementace výboru expertů"""

    def __init__(self, llm_client, tools_registry: Dict[str, Any]):
        self.llm_client = llm_client
        self.tools_registry = tools_registry

        # Inicializace expertů
        self.experts = {
            ExpertType.ACADEMIC: AcademicExpert(
                ExpertType.ACADEMIC, llm_client, tools_registry.get("academic", [])
            ),
            ExpertType.WEB_ANALYST: WebAnalyst(
                ExpertType.WEB_ANALYST, llm_client, tools_registry.get("web", [])
            ),
            ExpertType.TECHNICAL: TechnicalExpert(
                ExpertType.TECHNICAL, llm_client, tools_registry.get("technical", [])
            ),
        }

        self.coordinator = CoordinatorAgent(llm_client)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Sestavení LangGraph grafu"""
        graph = StateGraph(CommitteeState)

        # Uzly
        graph.add_node("analyze_requirements", self._analyze_requirements)
        graph.add_node("conduct_research", self._conduct_parallel_research)
        graph.add_node("synthesize_results", self._synthesize_results)
        graph.add_node("evaluate_completeness", self._evaluate_completeness)

        # Hrany
        graph.add_edge(START, "analyze_requirements")
        graph.add_edge("analyze_requirements", "conduct_research")
        graph.add_edge("conduct_research", "synthesize_results")
        graph.add_edge("synthesize_results", "evaluate_completeness")

        # Podmíněné hrany
        graph.add_conditional_edges(
            "evaluate_completeness",
            self._should_continue,
            {"continue": "conduct_research", "finish": END},
        )

        return graph.compile()

    async def _analyze_requirements(self, state: CommitteeState) -> CommitteeState:
        """Analýza požadavků na experty"""
        analysis = await self.coordinator.analyze_query_requirements(state.original_query)
        state.query_analysis = analysis
        return state

    async def _conduct_parallel_research(self, state: CommitteeState) -> CommitteeState:
        """Paralelní výzkum expertů"""
        tasks = []

        for expert_type, expert in self.experts.items():
            expert_key = expert_type.value
            if expert_key in state.query_analysis:
                relevance = state.query_analysis[expert_key].get("relevance", 0.0)
                if relevance > 0.3:  # Pouze relevantní experti
                    tasks.append(expert.conduct_research(state.original_query))

        if tasks:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            valid_responses = [r for r in responses if isinstance(r, ExpertResponse)]
            state.expert_responses.extend(valid_responses)

        return state

    async def _synthesize_results(self, state: CommitteeState) -> CommitteeState:
        """Syntéza výsledků"""
        synthesis = await self.coordinator.synthesize_expert_responses(
            state.original_query, state.expert_responses
        )
        state.synthesis_result = synthesis
        return state

    async def _evaluate_completeness(self, state: CommitteeState) -> CommitteeState:
        """Hodnocení úplnosti odpovědi"""
        # Výpočet confidence scores
        if state.expert_responses:
            avg_confidence = sum(r.confidence for r in state.expert_responses) / len(
                state.expert_responses
            )
            state.confidence_scores = {"overall": avg_confidence}

            # Pokud je confidence vysoké, ukončíme
            if avg_confidence > 0.7 or state.iteration >= state.max_iterations:
                state.final_answer = state.synthesis_result

        state.iteration += 1
        return state

    def _should_continue(self, state: CommitteeState) -> str:
        """Rozhodnutí o pokračování"""
        if state.final_answer or state.iteration >= state.max_iterations:
            return "finish"

        if state.confidence_scores and state.confidence_scores.get("overall", 0) > 0.7:
            return "finish"

        return "continue"

    async def run(self, query: str) -> Dict[str, Any]:
        """Spuštění výboru expertů"""
        initial_state = CommitteeState(
            original_query=query, query_analysis={}, expert_responses=[], max_iterations=3
        )

        final_state = await self.graph.ainvoke(initial_state)

        return {
            "query": query,
            "answer": final_state.final_answer,
            "expert_responses": final_state.expert_responses,
            "confidence_scores": final_state.confidence_scores,
            "iterations": final_state.iteration,
        }
