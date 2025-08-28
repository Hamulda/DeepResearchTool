"""
Moderní stavová architektura pro Research Agent pomocí LangGraph
Implementace centrálního stavového objektu a stavového automatu

Author: Senior Python/MLOps Agent
"""

from typing import TypedDict, List, Dict, Any, Annotated, Optional
import operator
import time
from langgraph import StateGraph, START, END, interrupt
from langchain_core.messages import BaseMessage

from .tools import get_tools_for_agent
from .rag_pipeline import RAGPipeline
from .memory import get_memory_store


class ResearchAgentState(TypedDict):
    """Centrální stavový objekt pro Research Agent"""

    # Vstupní dotaz
    initial_query: str

    # Plán výzkumu jako seznam kroků
    plan: List[str]

    # Získané dokumenty z retrieval procesu
    retrieved_docs: List[Dict[str, Any]]

    # Skóre validace pro různé metriky
    validation_scores: Dict[str, float]

    # Finální syntéza výsledků
    synthesis: str

    # Zprávy pro konverzaci (použití Annotated pro správné spojování)
    messages: Annotated[List[BaseMessage], operator.add]

    # Dodatečné metadata
    current_step: str
    processing_time: float
    errors: List[str]

    # Nové atributy pro zlepšenou funkcionalitát
    validation_threshold: float
    retry_count: int
    human_approval_required: bool
    human_decision: Optional[str]
    pending_action: Optional[Dict[str, Any]]
    sources_used: List[Dict[str, Any]]


class ResearchAgentGraph:
    """Stavový automat pro Research Agent implementovaný pomocí LangGraph"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.graph = None
        self.rag_pipeline = None
        self.tools = get_tools_for_agent()
        self._build_graph()

    def _build_graph(self):
        """Vytvoření stavového grafu s uzly a hranami"""

        # Vytvoření stavového grafu
        workflow = StateGraph(ResearchAgentState)

        # Přidání uzlů (node functions)
        workflow.add_node("plan", self.plan_step)
        workflow.add_node("retrieve", self.retrieve_step)
        workflow.add_node("validate_sources", self.validate_sources_step)
        workflow.add_node("human_approval", self.human_approval_step)
        workflow.add_node("validate", self.validate_step)
        workflow.add_node("synthesize", self.synthesize_step)

        # Definice hran (edge connections)
        workflow.add_edge(START, "plan")
        workflow.add_edge("plan", "retrieve")
        workflow.add_edge("retrieve", "validate_sources")

        # Podmíněná hrana po validaci zdrojů
        workflow.add_conditional_edges(
            "validate_sources",
            self._route_after_validation,
            {
                "continue_to_synthesis": "synthesize",
                "retry_planning": "plan",
                "need_approval": "human_approval"
            }
        )

        workflow.add_conditional_edges(
            "human_approval",
            self._route_after_approval,
            {
                "approved": "synthesize",
                "rejected": "plan"
            }
        )

        workflow.add_edge("validate", "synthesize")
        workflow.add_edge("synthesize", END)

        # Kompilace grafu
        self.graph = workflow.compile()

    async def _initialize_rag_pipeline(self):
        """Inicializace RAG pipeline pokud ještě není inicializována"""
        if self.rag_pipeline is None:
            self.rag_pipeline = RAGPipeline(self.config)
            await self.rag_pipeline.initialize()

    async def plan_step(self, state: ResearchAgentState) -> ResearchAgentState:
        """
        Plánovací krok: Vygeneruje seznam kroků na základě initial_query

        Args:
            state: Aktuální stav agenta

        Returns:
            Aktualizovaný stav s plánem
        """
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage

        # Inicializace LLM
        llm = ChatOpenAI(
            model=self.config.get("llm", {}).get("model", "gpt-4o-mini"),
            temperature=self.config.get("llm", {}).get("temperature", 0.1)
        )

        # Vytvoření prompt pro plánování
        planning_prompt = f"""
        Jsi expert na výzkum a plánování. Na základě dotazu vytvoř strukturovaný plán výzkumu.
        
        Dotaz: {state['initial_query']}
        
        Vytvoř seznam 3-5 konkrétních kroků pro důkladný výzkum tohoto tématu.
        Každý krok by měl být jasný a měl by specifikovat, jaký typ informací hledat.
        
        Odpověz pouze seznamem kroků, jeden na řádek, začínající číslem.
        """

        messages = [
            SystemMessage(content="Jsi expert na výzkumné plánování."),
            HumanMessage(content=planning_prompt)
        ]

        # Generování plánu
        response = await llm.ainvoke(messages)

        # Parsování plánu z odpovědi
        plan_text = response.content
        plan_steps = []

        for line in plan_text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Odstranění číslování a bullet pointů
                clean_step = line.lstrip('0123456789.-• ').strip()
                if clean_step:
                    plan_steps.append(clean_step)

        # Aktualizace stavu
        state["plan"] = plan_steps
        state["current_step"] = "plan_completed"

        return state

    async def retrieve_step(self, state: ResearchAgentState) -> ResearchAgentState:
        """
        Retrieval krok: Iteruje přes plán a sbírá dokumenty

        Args:
            state: Aktuální stav agenta

        Returns:
            Aktualizovaný stav s načtenými dokumenty
        """
        await self._initialize_rag_pipeline()

        retrieved_docs = []

        for step in state["plan"]:
            try:
                # Vyhledání v existující knowledge base
                existing_docs = await self.rag_pipeline.search(step, k=5)

                if existing_docs:
                    # Konverze Document objektů na slovníky
                    for doc in existing_docs:
                        doc_dict = {
                            "content": doc.content,
                            "source": doc.metadata.get("source", "knowledge_base"),
                            "step": step,
                            "metadata": doc.metadata
                        }
                        retrieved_docs.append(doc_dict)
                else:
                    # Pokud nejsou dostupné lokální dokumenty, použij web scraping
                    from .tools import web_scraping_tool
                    web_results = await web_scraping_tool.ainvoke({
                        "query": step,
                        "scrape_type": "search",
                        "max_pages": 2
                    })

                    if web_results and "content" in web_results and web_results["content"]:
                        doc = {
                            "content": web_results["content"],
                            "source": web_results.get("url", "web_search"),
                            "step": step,
                            "metadata": web_results.get("metadata", {})
                        }
                        retrieved_docs.append(doc)

                        # Uložení do knowledge base pro budoucí použití
                        await self.rag_pipeline.ingest_text(
                            web_results["content"],
                            metadata={**web_results.get("metadata", {}), "step": step}
                        )

            except Exception as e:
                error_msg = f"Chyba při retrievalu pro krok '{step}': {str(e)}"
                state["errors"].append(error_msg)

        # Aktualizace stavu
        state["retrieved_docs"] = retrieved_docs
        state["current_step"] = "retrieve_completed"

        return state

    async def validate_sources_step(self, state: ResearchAgentState) -> ResearchAgentState:
        """
        Validace zdrojů: Ověří kvalitu a důvěryhodnost zdrojů

        Args:
            state: Aktuální stav agenta

        Returns:
            Aktualizovaný stav po validaci zdrojů
        """
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatOpenAI(
            model=self.config.get("llm", {}).get("model", "gpt-4o-mini"),
            temperature=0.1
        )

        source_validation_results = []

        if state["retrieved_docs"]:
            for doc in state["retrieved_docs"]:
                content_preview = doc.get('content', '')[:100] + "..."
                source_prompt = f"""
                Ohodnoť důvěryhodnost a kvalitu tohoto zdroje informace na škále 0-1.
                
                Obsah: {content_preview}
                
                Odpověz pouze číslem mezi 0 a 1.
                """

                try:
                    source_response = await llm.ainvoke([
                        SystemMessage(content="Jsi expert na hodnocení kvality zdrojů informací."),
                        HumanMessage(content=source_prompt)
                    ])
                    source_score = float(source_response.content.strip())
                except ValueError:
                    source_score = 0.5

                source_validation_results.append({
                    "source": doc.get("source", ""),
                    "score": source_score,
                    "metadata": doc.get("metadata", {})
                })

        # Uložení výsledků validace zdrojů do stavu
        state["source_validation_results"] = source_validation_results
        state["current_step"] = "validate_sources_completed"

        return state

    async def human_approval_step(self, state: ResearchAgentState) -> ResearchAgentState:
        """
        Krok lidského schválení: Čeká na schválení nebo zamítnutí od člověka

        Args:
            state: Aktuální stav agenta

        Returns:
            Aktualizovaný stav po lidském schválení
        """
        # Tento krok je interaktivní a vyžaduje zásah člověka
        # Simulace lidského schválení pro účely tohoto příkladu
        import random
        user_decision = random.choice(["approved", "rejected"])

        state["human_decision"] = user_decision

        if user_decision == "approved":
            state["current_step"] = "approved"
        else:
            state["current_step"] = "rejected"

        return state

    async def validate_step(self, state: ResearchAgentState) -> ResearchAgentState:
        """
        Validační krok: Ověří kvalitu získaných dokumentů

        Args:
            state: Aktuální stav agenta

        Returns:
            Aktualizovaný stav s validačními skóre
        """
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatOpenAI(
            model=self.config.get("llm", {}).get("model", "gpt-4o-mini"),
            temperature=0.1
        )

        validation_scores = {}

        # Relevance score - jak relevantní jsou dokumenty k dotazu
        if state["retrieved_docs"]:
            doc_previews = []
            for doc in state["retrieved_docs"][:5]:
                content_preview = doc.get('content', '')[:200] + "..."
                doc_previews.append(f"- {content_preview}")

            relevance_prompt = f"""
            Ohodnoť relevanci následujících dokumentů k dotazu na škále 0-1.
            
            Dotaz: {state['initial_query']}
            
            Dokumenty:
            {chr(10).join(doc_previews)}
            
            Odpověz pouze číslem mezi 0 a 1.
            """

            try:
                relevance_response = await llm.ainvoke([
                    SystemMessage(content="Jsi expert na hodnocení relevance dokumentů."),
                    HumanMessage(content=relevance_prompt)
                ])
                validation_scores["relevance"] = float(relevance_response.content.strip())
            except ValueError:
                validation_scores["relevance"] = 0.5
        else:
            validation_scores["relevance"] = 0.0

        # Coverage score - pokrytí všech aspektů plánu
        if state["plan"]:
            coverage_score = min(1.0, len(state["retrieved_docs"]) / len(state["plan"]))
        else:
            coverage_score = 0.0
        validation_scores["coverage"] = coverage_score

        # Quality score - na základě počtu a kvality zdrojů
        quality_score = min(1.0, len(state["retrieved_docs"]) / 10)  # Normalizace na 10 dokumentů
        validation_scores["quality"] = quality_score

        # Aktualizace stavu
        state["validation_scores"] = validation_scores
        state["current_step"] = "validate_completed"

        return state

    async def synthesize_step(self, state: ResearchAgentState) -> ResearchAgentState:
        """
        Syntéza: Vezme validované dokumenty a vygeneruje finální syntézu

        Args:
            state: Aktuální stav agenta

        Returns:
            Aktualizovaný stav s finální syntézou
        """
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatOpenAI(
            model=self.config.get("llm", {}).get("synthesis_model", "gpt-4o"),
            temperature=self.config.get("llm", {}).get("synthesis_temperature", 0.2)
        )

        # Příprava kontextu z dokumentů
        context_parts = []
        max_docs = self.config.get("synthesis", {}).get("max_docs", 10)

        for i, doc in enumerate(state["retrieved_docs"][:max_docs]):
            content = doc.get('content', '')
            source = doc.get('source', 'unknown')
            step = doc.get('step', 'general')

            context_part = f"Dokument {i+1} [Zdroj: {source}, Krok: {step}]:\n{content}\n"
            context_parts.append(context_part)

        context = "\n---\n".join(context_parts)

        synthesis_prompt = f"""
        Na základě následujících dokumentů vytvoř komplexní syntézu odpovídající na dotaz.
        
        Původní dotaz: {state['initial_query']}
        
        Plán výzkumu:
        {chr(10).join([f"{i+1}. {step}" for i, step in enumerate(state['plan'])])}
        
        Dostupné dokumenty:
        {context}
        
        Validační skóre:
        - Relevance: {state['validation_scores'].get('relevance', 'N/A'):.2f}
        - Pokrytí: {state['validation_scores'].get('coverage', 'N/A'):.2f}
        - Kvalita: {state['validation_scores'].get('quality', 'N/A'):.2f}
        
        Vytvoř strukturovanou syntézu, která:
        1. Přímo odpovídá na původní dotaz
        2. Zahrnuje klíčové poznatky z každého kroku plánu
        3. Cituje konkrétní zdroje
        4. Uvádí případná omezení nebo nejistoty
        5. Poskytuje závěr s doporučeními
        
        Použij formát:
        ## Souhrn
        [Krátký souhrn odpovědi]
        
        ## Klíčové poznatky
        [Detailní analýza podle plánu]
        
        ## Zdroje a citace
        [Seznam použitých zdrojů]
        
        ## Závěr a doporučení
        [Závěrečné shrnutí a doporučení]
        """

        messages = [
            SystemMessage(content="Jsi expert na syntézu výzkumných poznatků a tvorbu strukturovaných reportů."),
            HumanMessage(content=synthesis_prompt)
        ]

        # Generování syntézy
        synthesis_response = await llm.ainvoke(messages)

        # Aktualizace stavu
        state["synthesis"] = synthesis_response.content
        state["current_step"] = "synthesis_completed"

        return state

    async def research(self, query: str) -> Dict[str, Any]:
        """
        Hlavní vstupní bod pro spuštění výzkumu

        Args:
            query: Výzkumný dotaz

        Returns:
            Kompletní výsledek výzkumu
        """
        start_time = time.time()

        # Inicializace stavu
        initial_state = ResearchAgentState(
            initial_query=query,
            plan=[],
            retrieved_docs=[],
            validation_scores={},
            synthesis="",
            messages=[],
            current_step="initialized",
            processing_time=0.0,
            errors=[],
            validation_threshold=0.7,  # Výchozí práh pro validaci
            retry_count=0,  # Počáteční počet pokusů
            human_approval_required=False,  # Požadavek na lidské schválení
            human_decision=None,  # Rozhodnutí od člověka (pokud je potřeba)
            pending_action=None,  # Jakákoliv akce čekající na zpracování
            sources_used=[]  # Seznam použitých zdrojů
        )

        # Spuštění stavového automatu
        final_state = await self.graph.ainvoke(initial_state)

        # Výpočet celkového času
        processing_time = time.time() - start_time
        final_state["processing_time"] = processing_time

        # Návrat strukturovaného výsledku
        return {
            "query": query,
            "plan": final_state["plan"],
            "retrieved_docs": final_state["retrieved_docs"],
            "validation_scores": final_state["validation_scores"],
            "synthesis": final_state["synthesis"],
            "processing_time": processing_time,
            "current_step": final_state["current_step"],
            "errors": final_state["errors"],
            "metadata": {
                "architecture": "langgraph",
                "rag_enabled": True,
                "tools_used": len(self.tools),
                "total_documents": len(final_state["retrieved_docs"])
            }
        }

    def _route_after_validation(self, state: ResearchAgentState) -> str:
        """
        Routing funkce pro rozhodování po validaci zdrojů

        Args:
            state: Aktuální stav agenta

        Returns:
            Název dalšího uzlu
        """
        # Výpočet průměrného skóre zdrojů
        if hasattr(state, 'source_validation_results') and state['source_validation_results']:
            avg_score = sum(result['score'] for result in state['source_validation_results']) / len(state['source_validation_results'])
        else:
            avg_score = 0.0

        # Kontrola počtu pokusů
        max_retries = self.config.get("validation", {}).get("max_retries", 2)

        # Rozhodovací logika
        if avg_score >= state.get('validation_threshold', 0.7):
            return "continue_to_synthesis"
        elif state.get('retry_count', 0) < max_retries:
            # Zvýšení počtu pokusů
            state['retry_count'] = state.get('retry_count', 0) + 1
            return "retry_planning"
        else:
            # Požadavek na lidské schválení pokud byla překročena hranice pokusů
            state['human_approval_required'] = True
            state['pending_action'] = {
                "type": "low_quality_sources",
                "avg_score": avg_score,
                "threshold": state.get('validation_threshold', 0.7)
            }
            return "need_approval"

    def _route_after_approval(self, state: ResearchAgentState) -> str:
        """
        Routing funkce pro rozhodování po lidském schválení

        Args:
            state: Aktuální stav agenta

        Returns:
            Název dalšího uzlu
        """
        decision = state.get('human_decision', 'rejected')

        if decision == "approved":
            return "approved"
        else:
            return "rejected"

