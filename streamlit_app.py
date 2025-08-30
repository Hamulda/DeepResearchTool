"""
Streamlit webové rozhraní pro Research Agent
Optimalizováno pro M1 s cachováním a statickou vizualizací

Author: Senior Python/MLOps Agent
"""

import streamlit as st
import asyncio
import json
import time
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx
import io
import base64

# Import agenta a souvisejících komponent
from src.core.langgraph_agent import ResearchAgentGraph, ResearchAgentState
from src.core.config_langgraph import load_config, validate_config
from src.graph.claim_graph import ClaimGraph
from src.optimization.m1_performance import cleanup_memory

# Konfigurace loggingu
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Konfigurace Streamlit stránky
st.set_page_config(
    page_title="Deep Research Tool - M1 Optimized",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Globální session state pro udržení stavu agenta
if "agent" not in st.session_state:
    st.session_state.agent = None
if "research_results" not in st.session_state:
    st.session_state.research_results = None
if "current_state" not in st.session_state:
    st.session_state.current_state = None
if "approval_pending" not in st.session_state:
    st.session_state.approval_pending = False
if "sources_found" not in st.session_state:
    st.session_state.sources_found = []


@st.cache_data(ttl=300)  # Cache na 5 minut
def initialize_agent_cached(config_json: str) -> Optional[ResearchAgentGraph]:
    """
    Cachovaná inicializace research agenta

    Args:
        config_json: JSON string konfigurace

    Returns:
        Inicializovaný Research Agent nebo None
    """
    try:
        config = json.loads(config_json)
        agent = ResearchAgentGraph(config)
        return agent
    except Exception as e:
        st.error(f"Chyba při inicializaci agenta: {e}")
        return None


@st.cache_data(ttl=600)  # Cache na 10 minut
def generate_claim_graph_visualization(claim_graph_data: Dict[str, Any]) -> str:
    """
    Generuje statickou vizualizaci ClaimGraph pomocí matplotlib

    Args:
        claim_graph_data: Data z ClaimGraph

    Returns:
        Base64 encoded obrázek
    """
    try:
        # Vytvoření NetworkX grafu z dat
        G = nx.DiGraph()

        # Přidání nodes
        nodes = claim_graph_data.get("nodes", [])
        for node in nodes:
            node_id = node["id"]
            node_type = node.get("type", "claim")
            label = node.get("label", node_id[:20])

            G.add_node(node_id, type=node_type, label=label)

        # Přidání edges
        edges = claim_graph_data.get("edges", [])
        for edge in edges:
            source = edge["source"]
            target = edge["target"]
            relation_type = edge.get("relation_type", "unknown")

            G.add_edge(source, target, relation=relation_type)

        # Vytvoření vizualizace
        plt.figure(figsize=(12, 8))
        plt.clf()

        # Layout pro lepší rozložení
        if len(G.nodes()) > 0:
            pos = nx.spring_layout(G, k=2, iterations=50)

            # Rozdělení nodes podle typu
            claim_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'claim']
            evidence_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'evidence']

            # Vykreslení claim nodes
            if claim_nodes:
                nx.draw_networkx_nodes(G, pos, nodelist=claim_nodes,
                                     node_color='lightblue', node_size=1000, alpha=0.7)

            # Vykreslení evidence nodes
            if evidence_nodes:
                nx.draw_networkx_nodes(G, pos, nodelist=evidence_nodes,
                                     node_color='lightgreen', node_size=600, alpha=0.7)

            # Vykreslení edges s barvami podle typu
            edge_colors = []
            for edge in G.edges(data=True):
                relation = edge[2].get('relation', 'unknown')
                if relation == 'support':
                    edge_colors.append('green')
                elif relation == 'contradict':
                    edge_colors.append('red')
                else:
                    edge_colors.append('gray')

            nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.6, arrows=True)

            # Labels
            labels = {n: d.get('label', n[:10]) for n, d in G.nodes(data=True)}
            nx.draw_networkx_labels(G, pos, labels, font_size=8)

        else:
            plt.text(0.5, 0.5, 'Žádný graf k zobrazení',
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)

        plt.title("Claim Graph - Vztahy mezi tvrzeními", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()

        # Uložení do base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)

        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return image_base64

    except Exception as e:
        logger.error(f"Error generating claim graph visualization: {e}")
        return ""


@st.cache_data(ttl=1800)  # Cache na 30 minut
def process_research_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cachované zpracování výsledků výzkumu

    Args:
        results: Raw výsledky z agenta

    Returns:
        Zpracované výsledky
    """
    processed = {
        "summary": {},
        "metrics": {},
        "visualizations": {},
        "processed_at": time.time()
    }

    # Základní metriky
    processed["metrics"] = {
        "total_docs": len(results.get("retrieved_docs", [])),
        "processing_time": results.get("processing_time", 0),
        "plan_steps": len(results.get("plan", [])),
        "validation_scores": results.get("validation_scores", {}),
        "errors_count": len(results.get("errors", []))
    }

    # Claim graph visualization
    if "claim_graph" in results and results["claim_graph"]:
        try:
            # Extrakce dat z ClaimGraph pro vizualizaci
            claim_graph = results["claim_graph"]
            if hasattr(claim_graph, 'visualize_subgraph') and claim_graph.claims:
                # Vezmi první claim jako center
                center_claim = list(claim_graph.claims.keys())[0]
                viz_data = claim_graph.visualize_subgraph(center_claim, depth=2)

                if viz_data.get("nodes"):
                    image_b64 = generate_claim_graph_visualization(viz_data)
                    processed["visualizations"]["claim_graph"] = image_b64
        except Exception as e:
            logger.error(f"Error processing claim graph: {e}")

    return processed


def initialize_agent(config: Dict[str, Any]) -> ResearchAgentGraph:
    """
    Inicializuje research agenta s danou konfigurací

    Args:
        config: Konfigurace pro agenta

    Returns:
        Inicializovaný Research Agent
    """
    try:
        agent = ResearchAgentGraph(config)
        return agent
    except Exception as e:
        st.error(f"Chyba při inicializaci agenta: {e}")
        return None


def render_sidebar() -> Dict[str, Any]:
    """
    Vykreslí postranní panel s nastavením

    Returns:
        Slovník s konfigurací
    """
    st.sidebar.title("⚙️ Nastavení")

    # Model selection
    model_option = st.sidebar.selectbox(
        "Vyberte LLM model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0
    )

    # Research depth
    research_depth = st.sidebar.slider(
        "Hloubka výzkumu", min_value=1, max_value=5, value=3, help="Počet kroků v plánu výzkumu"
    )

    # Validation threshold
    validation_threshold = st.sidebar.slider(
        "Práh validace zdrojů",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Minimální skóre pro přijetí zdrojů",
    )

    # Pokročilé možnosti
    with st.sidebar.expander("🔧 Pokročilé nastavení"):
        max_docs = st.slider("Max dokumentů", 5, 50, 20)
        temperature = st.slider("Teplota modelu", 0.0, 1.0, 0.1, step=0.1)
        enable_enhanced_tools = st.checkbox("Povolit rozšířené nástroje", value=True)

    # Sestavení konfigurace
    config = {
        "llm": {
            "model": model_option,
            "temperature": temperature,
            "synthesis_model": "gpt-4o" if model_option != "gpt-4o" else "gpt-4o",
            "synthesis_temperature": 0.2,
        },
        "memory_store": {
            "type": "chroma",
            "collection_name": "research_collection",
            "persist_directory": "./chroma_db",
        },
        "rag": {"chunking": {"chunk_size": 1000, "chunk_overlap": 100}},
        "synthesis": {"max_docs": max_docs},
        "validation": {"threshold": validation_threshold, "max_retries": 2},
        "research": {"depth": research_depth, "enhanced_tools": enable_enhanced_tools},
    }

    return config


def render_main_interface():
    """Vykreslí hlavní rozhraní aplikace"""

    st.title("🔬 Deep Research Tool - Enhanced")
    st.markdown("*Pokročilý výzkumný agent s human-in-the-loop funkcionalitou*")

    # Vstupní oblast pro dotaz
    query = st.text_area(
        "📝 Zadejte váš výzkumný dotaz:",
        height=100,
        placeholder="Napište detailní výzkumný dotaz... Například: 'Jaké jsou nejnovější trendy v oblasti umělé inteligence v medicíně?'",
    )

    # Tlačítka pro ovládání
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        run_research = st.button("🚀 Spustit výzkum", type="primary", disabled=not query.strip())

    with col2:
        if st.button("🗑️ Vymazat"):
            st.session_state.research_results = None
            st.session_state.current_state = None
            st.session_state.sources_found = []
            st.rerun()

    with col3:
        if st.button("📊 Statistiky"):
            show_statistics()

    return query, run_research


def show_real_time_status(state: Dict[str, Any]):
    """
    Zobrazí aktuální stav agenta v reálném čase

    Args:
        state: Aktuální stav agenta
    """
    with st.container():
        st.subheader("🔄 Aktuální stav")

        current_step = state.get("current_step", "neznámý")

        # Progress bar based on step
        step_mapping = {
            "initialized": 0,
            "plan_completed": 20,
            "retrieve_completed": 40,
            "validate_sources_completed": 60,
            "validate_completed": 80,
            "synthesis_completed": 100,
        }

        progress = step_mapping.get(current_step, 0)
        st.progress(progress / 100)

        # Status info
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Aktuální krok", current_step)

        with col2:
            docs_count = len(state.get("retrieved_docs", []))
            st.metric("Nalezené dokumenty", docs_count)

        with col3:
            errors_count = len(state.get("errors", []))
            st.metric("Chyby", errors_count, delta_color="inverse")

        # Plan display
        if state.get("plan"):
            with st.expander("📋 Plán výzkumu"):
                for i, step in enumerate(state["plan"], 1):
                    st.write(f"{i}. {step}")

        # Validation scores
        if state.get("validation_scores"):
            with st.expander("✅ Validační skóre"):
                scores = state["validation_scores"]
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Relevance", f"{scores.get('relevance', 0):.2f}")
                with col2:
                    st.metric("Pokrytí", f"{scores.get('coverage', 0):.2f}")
                with col3:
                    st.metric("Kvalita", f"{scores.get('quality', 0):.2f}")


def handle_human_approval(pending_action: Dict[str, Any]) -> Optional[str]:
    """
    Zpracuje požadavek na lidské schválení

    Args:
        pending_action: Akce čekající na schválení

    Returns:
        Rozhodnutí uživatele nebo None
    """
    if not st.session_state.approval_pending:
        return None

    with st.container():
        st.warning("⚠️ Požadavek na schválení")

        action_type = pending_action.get("type", "neznámá akce")

        if action_type == "low_quality_sources":
            avg_score = pending_action.get("avg_score", 0)
            threshold = pending_action.get("threshold", 0.7)

            st.write(
                f"""
            **Nalezené zdroje mají nízké skóre kvality:**
            - Průměrné skóre: {avg_score:.2f}
            - Požadovaný práh: {threshold:.2f}
            
            Chcete pokračovat i přes nízkou kvalitu zdrojů?
            """
            )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Schválit", type="primary"):
                st.session_state.approval_pending = False
                return "approved"

        with col2:
            if st.button("❌ Zamítnout"):
                st.session_state.approval_pending = False
                return "rejected"

    return None


def display_results(results: Dict[str, Any]):
    """
    Zobrazí finální výsledky výzkumu

    Args:
        results: Výsledky výzkumu
    """
    st.subheader("📋 Výsledky výzkumu")

    # Metadata
    with st.expander("ℹ️ Metadata"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Doba zpracování", f"{results.get('processing_time', 0):.2f}s")
        with col2:
            st.metric("Celkem dokumentů", results.get("metadata", {}).get("total_documents", 0))
        with col3:
            st.metric("Použité nástroje", results.get("metadata", {}).get("tools_used", 0))

    # Syntéza
    if results.get("synthesis"):
        st.markdown("### 📄 Finální syntéza")
        st.markdown(results["synthesis"])

    # Chyby
    if results.get("errors"):
        with st.expander("⚠️ Chyby během zpracování"):
            for error in results["errors"]:
                st.error(error)


def display_sources_sidebar(sources: list):
    """
    Zobrazí seznam zdrojů v postranním panelu

    Args:
        sources: Seznam nalezených zdrojů
    """
    if sources:
        st.sidebar.markdown("### 📚 Použité zdroje")

        for i, source in enumerate(sources, 1):
            source_url = source.get("source", "")
            source_title = source.get("metadata", {}).get("title", f"Zdroj {i}")

            if source_url.startswith("http"):
                st.sidebar.markdown(f"[{i}. {source_title}]({source_url})")
            else:
                st.sidebar.markdown(f"{i}. {source_title}")


def show_statistics():
    """Zobrazí statistiky použití"""
    with st.container():
        st.subheader("📊 Statistiky")

        # Placeholder pro statistiky
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Celkem dotazů", "N/A")
        with col2:
            st.metric("Průměrný čas", "N/A")
        with col3:
            st.metric("Úspěšnost", "N/A")
        with col4:
            st.metric("Zdrojů celkem", "N/A")


async def run_research_async(agent: ResearchAgentGraph, query: str) -> Dict[str, Any]:
    """
    Asynchronně spustí výzkum

    Args:
        agent: Research agent
        query: Dotaz

    Returns:
        Výsledky výzkumu
    """
    return await agent.research(query)


def main():
    """Hlavní funkce aplikace"""

    # Načtení konfigurace z sidebar
    config = render_sidebar()

    # Inicializace agenta
    if st.session_state.agent is None:
        st.session_state.agent = initialize_agent(config)

    # Hlavní rozhraní
    query, run_research = render_main_interface()

    # Spuštění výzkumu
    if run_research and query.strip() and st.session_state.agent:

        # Status container pro real-time updates
        status_container = st.container()

        with st.spinner("🔍 Probíhá výzkum..."):
            try:
                # Spuštění asynchronního výzkumu
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                results = loop.run_until_complete(run_research_async(st.session_state.agent, query))

                st.session_state.research_results = results
                st.session_state.sources_found = results.get("retrieved_docs", [])

                # Zobrazení úspěchu
                st.success("✅ Výzkum dokončen!")

            except Exception as e:
                st.error(f"❌ Chyba během výzkumu: {e}")
                logger.error(f"Research error: {e}")

            finally:
                loop.close()

    # Zobrazení výsledků
    if st.session_state.research_results:
        # Zpracování výsledků s cachováním
        processed_results = process_research_results(st.session_state.research_results)

        # Zobrazení metrik
        st.subheader("📊 Základní metriky")
        metrics = processed_results.get("metrics", {})
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Celkem dokumentů", metrics.get("total_docs", 0))
        with col2:
            st.metric("Doba zpracování", f"{metrics.get('processing_time', 0):.2f}s")
        with col3:
            st.metric("Počet kroků v plánu", metrics.get("plan_steps", 0))
        with col4:
            st.metric("Počet chyb", metrics.get("errors_count", 0))

        # Zobrazení vizualizací
        if "visualizations" in processed_results:
            visualizations = processed_results["visualizations"]

            if "claim_graph" in visualizations:
                st.subheader("🧩 Vizualizace Claim Graph")
                image_base64 = visualizations["claim_graph"]
                st.image(f"data:image/png;base64,{image_base64}", use_column_width=True)

        # Zobrazení syntézy
        if processed_results.get("synthesis"):
            st.markdown("### 📄 Finální syntéza")
            st.markdown(processed_results["synthesis"])

    # Zobrazení zdrojů v sidebar
    if st.session_state.sources_found:
        display_sources_sidebar(st.session_state.sources_found)

    # Zpracování human approval (pokud je potřeba)
    if st.session_state.approval_pending and st.session_state.current_state:
        pending_action = st.session_state.current_state.get("pending_action")
        if pending_action:
            decision = handle_human_approval(pending_action)
            if decision:
                st.rerun()


if __name__ == "__main__":
    main()
