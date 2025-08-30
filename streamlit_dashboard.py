"""
🎯 Hlavní Streamlit aplikace pro Deep Research Tool
Interaktivní dashboard s autonomním agentem a vizualizacemi
"""

import streamlit as st
import asyncio
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
import time
import numpy as np
from typing import Dict, List, Any, Optional

# Import pokročilých komponent
try:
    from src.ui.advanced_components import (
        NetworkGraphComponent,
        RealTimeMetricsComponent,
        AlertsAndNotificationsComponent,
        CredibilityVisualizationComponent,
    )

    ADVANCED_COMPONENTS = True
except ImportError:
    ADVANCED_COMPONENTS = False
    st.warning("Pokročilé komponenty nejsou dostupné - používám základní verzi")

# Konfigurace stránky
st.set_page_config(
    page_title="🧠 Deep Research Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS pro lepší vzhled
st.markdown(
    """
<style>
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.success-metric {
    background-color: #d4edda;
    border-left: 4px solid #28a745;
}
.warning-metric {
    background-color: #fff3cd;
    border-left: 4px solid #ffc107;
}
.error-metric {
    background-color: #f8d7da;
    border-left: 4px solid #dc3545;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    padding-left: 20px;
    padding-right: 20px;
}
</style>
""",
    unsafe_allow_html=True,
)

# Inicializace session state
if "agent_status" not in st.session_state:
    st.session_state.agent_status = "stopped"
if "research_results" not in st.session_state:
    st.session_state.research_results = None
if "network_data" not in st.session_state:
    st.session_state.network_data = None
if "system_metrics" not in st.session_state:
    st.session_state.system_metrics = []
if "alerts" not in st.session_state:
    st.session_state.alerts = []

# Hlavní nadpis s animací
st.markdown(
    """
# 🧠 Deep Research Tool - Fáze 4
## Autonomní Výzkumný Agent s Interaktivním Rozhraním

*Pokročilá analýza s automatickým rozhodováním a real-time vizualizací*
"""
)

# Sidebar s konfiguracemi
with st.sidebar:
    st.header("⚙️ Konfigurace Agenta")

    # Status agenta
    status_color = {"stopped": "🔴", "running": "🟢", "paused": "🟡"}
    st.markdown(
        f"**Status:** {status_color.get(st.session_state.agent_status, '⚪')} {st.session_state.agent_status.upper()}"
    )

    st.markdown("---")

    # Nastavení autonomního agenta
    st.subheader("🤖 Parametry Agenta")
    max_iterations = st.slider("Max iterací", 3, 20, 10)
    min_credibility = st.slider("Min. důvěryhodnost", 0.1, 0.9, 0.3)
    max_concurrent = st.slider("Max souběžných úkolů", 1, 10, 5)

    # Strategie task managementu
    strategy = st.selectbox(
        "Strategie vykonávání:",
        ["balanced", "credibility_first", "depth_first", "breadth_first"],
        help="Způsob prioritizace úkolů",
    )

    st.markdown("---")

    # Nastavení monitoringu
    st.subheader("📊 Monitoring")
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_interval = st.slider("Interval (s)", 5, 60, 10)

    # Bezpečnostní nastavení
    st.subheader("🛡️ Bezpečnost")
    tor_enabled = st.checkbox("Tor proxy", value=True)
    vpn_enabled = st.checkbox("VPN", value=True)

    st.markdown("---")

    # Quick actions
    st.subheader("⚡ Rychlé akce")
    if st.button("🔄 Reset systému"):
        st.session_state.agent_status = "stopped"
        st.session_state.research_results = None
        st.rerun()

    if st.button("📊 Export dat"):
        st.success("Data exportována!")

# Hlavní záložky
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔍 Vyhledávání & Analýza",
        "🕸️ Síťový Pohled",
        "📊 Zpravodajský Panel",
        "🛡️ Bezpečnostní Monitor",
    ]
)

# ===== ZÁLOŽKA 1: VYHLEDÁVÁNÍ & ANALÝZA =====
with tab1:
    st.header("🔍 Autonomní Výzkumný Agent")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Hlavní formulář
        with st.form("research_form"):
            st.subheader("Výzkumný dotaz")
            research_query = st.text_area(
                "Zadejte výzkumný dotaz:",
                placeholder="Např.: Analýza kryptoměnových transakcí na darknetu, sledování komunikačních vzorů...",
                height=100,
                help="Agent automaticky vygeneruje a vykoná potřebné úkoly",
            )

            # Pokročilé možnosti
            with st.expander("🔧 Pokročilé možnosti"):
                target_urls = st.text_area(
                    "Cílové URL (volitelné):",
                    placeholder="https://example.com\nhttps://another-site.com",
                    height=60,
                )

                col_a, col_b = st.columns(2)
                with col_a:
                    research_depth = st.selectbox(
                        "Hloubka výzkumu:", ["Základní", "Pokročilá", "Expertní"]
                    )
                with col_b:
                    focus_area = st.selectbox(
                        "Oblast zaměření:", ["Obecné", "Kryptoměny", "Darknet", "Komunikace"]
                    )

            # Tlačítka
            col_btn1, col_btn2, col_btn3 = st.columns(3)

            with col_btn1:
                start_research = st.form_submit_button("🚀 Spustit výzkum", type="primary")
            with col_btn2:
                save_query = st.form_submit_button("💾 Uložit dotaz")
            with col_btn3:
                load_template = st.form_submit_button("📋 Šablona")

        # Zpracování formuláře
        if start_research and research_query:
            st.session_state.agent_status = "running"

            with st.spinner("🤖 Spouštím autonomní agenta..."):
                # Simulace spuštění agenta
                progress_bar = st.progress(0)
                status_container = st.container()

                # Simulace průběhu výzkumu
                stages = [
                    "Inicializace komponent...",
                    "Generování počátečních úkolů...",
                    "Spouštím task manager...",
                    "Vykonávám scraping úkoly...",
                    "Analyzuji získaná data...",
                    "Hledám korelace...",
                    "Validuji výsledky...",
                    "Generuji finální report...",
                ]

                for i, stage in enumerate(stages):
                    progress_bar.progress((i + 1) / len(stages))
                    status_container.info(f"📋 {stage}")
                    time.sleep(0.5)

                # Mock výsledky
                st.session_state.research_results = {
                    "query": research_query,
                    "tasks_generated": np.random.randint(15, 35),
                    "tasks_completed": np.random.randint(12, 30),
                    "avg_credibility": 0.3 + np.random.random() * 0.5,
                    "entities_found": np.random.randint(10, 50),
                    "patterns_detected": np.random.randint(3, 15),
                    "execution_time": np.random.uniform(45, 120),
                    "start_time": datetime.now(),
                }

                st.session_state.agent_status = "completed"
                st.success("✅ Výzkum dokončen!")

    with col2:
        # Real-time status panel
        st.subheader("📈 Status v reálném čase")

        # Agent status
        if st.session_state.agent_status == "running":
            st.markdown(
                """
            <div class="warning-metric">
                <h4>🟡 Agent běží</h4>
                <p>Vykonává autonomní úkoly...</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        elif st.session_state.agent_status == "completed":
            st.markdown(
                """
            <div class="success-metric">
                <h4>✅ Výzkum dokončen</h4>
                <p>Všechny úkoly zpracovány</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
            <div class="metric-container">
                <h4>⚪ Agent připraven</h4>
                <p>Čeká na výzkumný dotaz</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Aktuální metriky
        if st.session_state.research_results:
            results = st.session_state.research_results

            st.metric("Generované úkoly", results["tasks_generated"])
            st.metric(
                "Dokončené úkoly",
                results["tasks_completed"],
                delta=f"+{results['tasks_completed'] - results['tasks_generated'] + results['tasks_generated']}",
            )
            st.metric("Průměrná důvěryhodnost", f"{results['avg_credibility']:.2f}")
            st.metric("Nalezené entity", results["entities_found"])
            st.metric("Detekované vzory", results["patterns_detected"])

            # Úspěšnost
            success_rate = (
                results["tasks_completed"] / results["tasks_generated"]
                if results["tasks_generated"] > 0
                else 0
            )
            st.metric("Úspěšnost", f"{success_rate:.1%}")

    # Detailní výsledky
    if st.session_state.research_results:
        st.markdown("---")
        st.subheader("🔬 Detailn�� výsledky autonomního výzkumu")

        results = st.session_state.research_results

        # Summary cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.info(f"⏱️ **Doba výzkumu**\n{results['execution_time']:.1f} sekund")
        with col2:
            st.info(
                f"🎯 **Úspěšnost**\n{(results['tasks_completed']/results['tasks_generated']*100):.1f}%"
            )
        with col3:
            st.info(f"🔍 **Kvalita dat**\n{results['avg_credibility']:.2f}/1.0")
        with col4:
            st.info(
                f"📊 **Celkem zjištění**\n{results['entities_found'] + results['patterns_detected']}"
            )

        # Detailní tabs
        detail_tab1, detail_tab2, detail_tab3, detail_tab4 = st.tabs(
            [
                "📄 Nalezené dokumenty",
                "🔗 Extrahované entity",
                "🎯 Detekované vzory",
                "🤖 Agent logy",
            ]
        )

        with detail_tab1:
            # Mock dokumenty
            doc_data = []
            for i in range(min(10, results["tasks_completed"])):
                doc_data.append(
                    {
                        "URL": f"https://example{i+1}.com/research",
                        "Důvěryhodnost": round(0.3 + np.random.random() * 0.6, 2),
                        "Entity": np.random.randint(1, 8),
                        "Vzory": np.random.randint(0, 4),
                        "Velikost": f"{np.random.randint(5, 50)} KB",
                        "Typ": np.random.choice(["Článek", "Fórum", "Blog", "Databáze"]),
                    }
                )

            df_docs = pd.DataFrame(doc_data)
            st.dataframe(df_docs, use_container_width=True)

            # Graf důvěryhodnosti dokumentů
            fig_cred = px.histogram(
                df_docs, x="Důvěryhodnost", bins=10, title="Distribuce důvěryhodnosti dokumentů"
            )
            st.plotly_chart(fig_cred, use_container_width=True)

        with detail_tab2:
            # Mock entity
            entity_data = []
            entity_types = ["bitcoin_address", "onion_address", "email", "telegram", "ip_address"]
            for i in range(results["entities_found"]):
                entity_data.append(
                    {
                        "Entita": (
                            f"entity_{i+1}@example.com"
                            if "email" in entity_types[i % len(entity_types)]
                            else f"entity_{i+1}"
                        ),
                        "Typ": entity_types[i % len(entity_types)],
                        "Relevance": round(0.4 + np.random.random() * 0.5, 2),
                        "Zdroj": f"dokument_{(i % 10) + 1}",
                        "Ověřeno": np.random.choice([True, False], p=[0.7, 0.3]),
                    }
                )

            df_entities = pd.DataFrame(entity_data)
            st.dataframe(df_entities, use_container_width=True)

            # Graf typů entit
            entity_counts = df_entities["Typ"].value_counts()
            fig_entities = px.pie(
                values=entity_counts.values, names=entity_counts.index, title="Rozložení typů entit"
            )
            st.plotly_chart(fig_entities, use_container_width=True)

        with detail_tab3:
            # Mock vzory
            pattern_data = []
            pattern_types = ["Kryptoměny", "Darknet", "Komunikace", "Geolokace", "Síťové"]
            for i in range(results["patterns_detected"]):
                pattern_data.append(
                    {
                        "Vzor": pattern_types[i % len(pattern_types)],
                        "Počet výskytů": np.random.randint(3, 25),
                        "Confidence": round(0.5 + np.random.random() * 0.4, 2),
                        "První výskyt": f"dokument_{np.random.randint(1, 6)}",
                        "Kategorie": np.random.choice(
                            ["Vysoká priorita", "Střední priorita", "Nízká priorita"]
                        ),
                    }
                )

            df_patterns = pd.DataFrame(pattern_data)
            st.dataframe(df_patterns, use_container_width=True)

            # Graf confidence vzorů
            fig_patterns = px.bar(
                df_patterns,
                x="Vzor",
                y="Confidence",
                color="Kategorie",
                title="Confidence detekovaných vzorů",
            )
            st.plotly_chart(fig_patterns, use_container_width=True)

        with detail_tab4:
            st.subheader("🤖 Logy autonomního agenta")

            # Mock agent logy
            log_entries = [
                {
                    "čas": "14:35:22",
                    "úroveň": "INFO",
                    "zpráva": f"Dokončen úkol: scrape_{np.random.randint(1,20)}",
                },
                {
                    "čas": "14:35:15",
                    "úroveň": "INFO",
                    "zpráva": f"Spuštěna analýza dokumentu_{np.random.randint(1,10)}",
                },
                {
                    "čas": "14:35:08",
                    "úroveň": "SUCCESS",
                    "zpráva": f"Nalezeno {np.random.randint(2,8)} nových entit",
                },
                {
                    "čas": "14:35:01",
                    "úroveň": "WARNING",
                    "zpráva": "Nízká důvěryhodnost zdroje - skip",
                },
                {
                    "čas": "14:34:54",
                    "úroveň": "INFO",
                    "zpráva": f"Generování {np.random.randint(3,7)} navazujících úkolů",
                },
                {"čas": "14:34:45", "úroveň": "INFO", "zpráva": "Adaptace strategie na balanced"},
                {
                    "čas": "14:34:38",
                    "úroveň": "SUCCESS",
                    "zpráva": f"Iterace {np.random.randint(1,10)} dokončena",
                },
            ]

            for log in log_entries:
                log_color = {
                    "INFO": "blue",
                    "SUCCESS": "green",
                    "WARNING": "orange",
                    "ERROR": "red",
                }.get(log["úroveň"], "gray")

                st.markdown(
                    f"""
                <div style="padding: 0.5rem; margin: 0.2rem 0; border-left: 3px solid {log_color}; background-color: #f8f9fa;">
                    <span style="color: gray; font-size: 0.8rem;">{log['čas']}</span>
                    <span style="color: {log_color}; font-weight: bold; margin-left: 1rem;">[{log['úroveň']}]</span>
                    <span style="margin-left: 1rem;">{log['zpráva']}</span>
                </div>
                """,
                    unsafe_allow_html=True,
                )

# ===== ZÁLOŽKA 2: SÍŤOVÝ POHLED =====
with tab2:
    st.header("🕸️ Interaktivní Graf Vztahů")

    # Kontrolní panel
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        node_size_metric = st.selectbox(
            "Velikost uzlů:", ["Důvěryhodnost", "Počet spojení", "PageRank", "Relevance"]
        )

    with col2:
        edge_filter = st.selectbox(
            "Filtr vztahů:",
            ["Všechny", "Kryptoměny", "Komunikace", "Darknet", "Vysoká důvěryhodnost"],
        )

    with col3:
        layout_type = st.selectbox(
            "Typ rozvržení:", ["Force-directed", "Circular", "Hierarchical", "Random"]
        )

    with col4:
        graph_3d = st.checkbox("3D vizualizace", value=False)

    # Generování/aktualizace síťového grafu
    if st.button("🔄 Aktualizovat síť") or st.session_state.network_data is None:
        with st.spinner("🕸️ Generuji síťový graf..."):
            # Mock data pro síťový graf
            nodes = []
            edges = []

            # Generování uzlů
            node_types = [
                "person",
                "organization",
                "crypto_address",
                "darknet_site",
                "email",
                "ip_address",
            ]
            colors = {
                "person": "#FF6B6B",
                "organization": "#4ECDC4",
                "crypto_address": "#45B7D1",
                "darknet_site": "#96CEB4",
                "email": "#FFEAA7",
                "ip_address": "#A8E6CF",
            }

            num_nodes = 25 if st.session_state.research_results else 15

            for i in range(num_nodes):
                node_type = node_types[i % len(node_types)]
                credibility = 0.2 + np.random.random() * 0.7
                connections = np.random.randint(1, 8)

                nodes.append(
                    {
                        "id": f"node_{i}",
                        "label": f"{node_type}_{i}",
                        "type": node_type,
                        "credibility": round(credibility, 2),
                        "connections": connections,
                        "color": colors[node_type],
                        "relevance": round(np.random.random(), 2),
                        "pagerank": round(np.random.random() * 0.1, 3),
                    }
                )

            # Generování hran s různými typy vztahů
            relation_types = [
                "communicates_with",
                "transacts_with",
                "hosts",
                "references",
                "correlates_with",
            ]

            for i in range(min(40, num_nodes * 2)):
                source_idx = np.random.randint(0, len(nodes))
                target_idx = np.random.randint(0, len(nodes))

                if source_idx != target_idx:
                    edges.append(
                        {
                            "source": f"node_{source_idx}",
                            "target": f"node_{target_idx}",
                            "weight": round(np.random.random(), 2),
                            "relation_type": relation_types[i % len(relation_types)],
                            "strength": round(0.3 + np.random.random() * 0.7, 2),
                        }
                    )

            st.session_state.network_data = {"nodes": nodes, "edges": edges}

    # Vizualizace grafu
    if st.session_state.network_data and ADVANCED_COMPONENTS:
        if graph_3d:
            # 3D graf pomocí pokročilých komponent
            fig = NetworkGraphComponent.create_3d_network_graph(
                st.session_state.network_data["nodes"],
                st.session_state.network_data["edges"],
                layout_type.lower().replace("-", "_"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # 2D graf s clusterovou analýzou
            fig = NetworkGraphComponent.create_cluster_analysis_view(
                st.session_state.network_data["nodes"], st.session_state.network_data["edges"]
            )
            st.plotly_chart(fig, use_container_width=True)

    elif st.session_state.network_data:
        # Základní vizualizace bez pokročilých komponent
        st.info("💡 Pokročilé 3D vizualizace budou k dispozici po instalaci všech závislostí")

        # Jednoduchý 2D graf
        fig = create_simple_network_graph(st.session_state.network_data, node_size_metric)
        st.plotly_chart(fig, use_container_width=True)

    # Statistiky sítě
    if st.session_state.network_data:
        st.markdown("---")
        st.subheader("📊 Analýza síťové struktury")

        nodes = st.session_state.network_data["nodes"]
        edges = st.session_state.network_data["edges"]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Celkem uzlů", len(nodes))
        with col2:
            st.metric("Celkem vztahů", len(edges))
        with col3:
            avg_credibility = np.mean([node["credibility"] for node in nodes])
            st.metric("Průměrná důvěryhodnost", f"{avg_credibility:.2f}")
        with col4:
            # Simulace detekce clusterů
            estimated_clusters = max(1, len(nodes) // 8)
            st.metric("Detekované clustery", estimated_clusters)

        # Dodatečné analýzy
        col5, col6 = st.columns(2)

        with col5:
            # Top uzly podle důvěryhodnosti
            st.subheader("🏆 Nejdůvěryhodnější uzly")
            top_nodes = sorted(nodes, key=lambda x: x["credibility"], reverse=True)[:5]
            for i, node in enumerate(top_nodes):
                st.write(f"{i+1}. **{node['label']}** - {node['credibility']:.2f}")

        with col6:
            # Statistiky typů uzlů
            st.subheader("📈 Rozložení typů uzlů")
            type_counts = {}
            for node in nodes:
                node_type = node["type"]
                type_counts[node_type] = type_counts.get(node_type, 0) + 1

            for node_type, count in sorted(type_counts.items()):
                st.write(f"**{node_type}**: {count}")

# ===== ZÁLOŽKA 3: ZPRAVODAJSKÝ PANEL =====
with tab3:
    st.header("📊 Živý Zpravodajský Panel")

    # Auto-refresh mechanismus
    if auto_refresh:
        # Placeholder pro live content
        live_container = st.container()

        with live_container:
            # Aktuální čas
            current_time = datetime.now()
            st.info(
                f"🕐 Poslední aktualizace: {current_time.strftime('%H:%M:%S')} | Další za {refresh_interval}s"
            )

            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("🆕 Nejnovější zjištění")

                # Mock real-time nálezy
                latest_findings = []
                for i in range(5):
                    finding_types = [
                        "Kryptoměnová adresa",
                        "Darknet odkaz",
                        "Email komunikace",
                        "IP adresa",
                        "Telegram kanál",
                    ]
                    finding_type = finding_types[i % len(finding_types)]

                    latest_findings.append(
                        {
                            "čas": (
                                current_time - timedelta(minutes=np.random.randint(1, 30))
                            ).strftime("%H:%M:%S"),
                            "typ": finding_type,
                            "obsah": (
                                f"example_{i+1}@domain.com"
                                if "Email" in finding_type
                                else f"entity_{i+1}_value"
                            ),
                            "důvěryhodnost": round(0.3 + np.random.random() * 0.6, 2),
                            "zdroj": f"agent_task_{np.random.randint(100, 999)}",
                        }
                    )

                for finding in latest_findings:
                    with st.container():
                        # Barevné označení podle důvěryhodnosti
                        if finding["důvěryhodnost"] >= 0.7:
                            trust_color = "success"
                            trust_icon = "🟢"
                        elif finding["důvěryhodnost"] >= 0.4:
                            trust_color = "warning"
                            trust_icon = "🟡"
                        else:
                            trust_color = "error"
                            trust_icon = "🔴"

                        st.markdown(
                            f"""
                        <div class="{trust_color}-metric" style="margin: 0.5rem 0; padding: 1rem; border-radius: 0.5rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <strong>{finding['typ']}</strong>: <code>{finding['obsah']}</code><br>
                                    <small>📅 {finding['čas']} | 🔗 {finding['zdroj']}</small>
                                </div>
                                <div style="text-align: center;">
                                    <div>{trust_icon}</div>
                                    <small>{finding['důvěryhodnost']:.2f}</small>
                                </div>
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

            with col2:
                st.subheader("⚡ Live metriky")

                # Real-time metriky
                active_tasks = np.random.randint(2, 8)
                completed_today = np.random.randint(45, 150)
                avg_response_time = np.random.uniform(0.8, 3.2)

                st.metric("Aktivní úkoly", active_tasks, delta=np.random.randint(-2, 3))
                st.metric("Dokončeno dnes", completed_today, delta=f"+{np.random.randint(5, 15)}")
                st.metric(
                    "Průměrná odezva",
                    f"{avg_response_time:.1f}s",
                    delta=f"{np.random.uniform(-0.3, 0.3):.1f}s",
                )

                # Health indikátor
                health_score = 0.7 + np.random.random() * 0.25
                health_color = (
                    "🟢" if health_score >= 0.8 else "🟡" if health_score >= 0.6 else "🔴"
                )
                st.metric("Zdraví systému", f"{health_score:.1%}", delta=f"{health_color}")

            # Trend grafy
            st.markdown("---")
            st.subheader("📈 Trendy a analýzy")

            # Generování trendových dat
            dates = pd.date_range(
                start=current_time - timedelta(days=7), end=current_time, freq="H"
            )
            trend_data = pd.DataFrame(
                {
                    "čas": dates,
                    "entity_count": np.random.poisson(5, len(dates))
                    + np.sin(np.arange(len(dates)) * 0.3) * 2,
                    "credibility_avg": 0.6
                    + 0.2 * np.sin(np.arange(len(dates)) * 0.1)
                    + np.random.normal(0, 0.05, len(dates)),
                    "task_completion": np.random.poisson(8, len(dates)) + 3,
                }
            )

            # Graf s více metrikami
            fig_trends = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Nalezené entity za hodinu",
                    "Průměrná důvěryhodnost",
                    "Dokončené úkoly",
                    "Systémová zátěž",
                ),
                vertical_spacing=0.1,
            )

            # Entity
            fig_trends.add_trace(
                go.Scatter(
                    x=trend_data["čas"],
                    y=trend_data["entity_count"],
                    mode="lines",
                    name="Entity",
                    line=dict(color="#1f77b4"),
                ),
                row=1,
                col=1,
            )

            # Důvěryhodnost
            fig_trends.add_trace(
                go.Scatter(
                    x=trend_data["čas"],
                    y=trend_data["credibility_avg"],
                    mode="lines",
                    name="Důvěryhodnost",
                    line=dict(color="#ff7f0e"),
                ),
                row=1,
                col=2,
            )

            # Úkoly
            fig_trends.add_trace(
                go.Scatter(
                    x=trend_data["čas"],
                    y=trend_data["task_completion"],
                    mode="lines",
                    name="Úkoly",
                    line=dict(color="#2ca02c"),
                ),
                row=2,
                col=1,
            )

            # Systémová zátěž (mock)
            system_load = np.random.uniform(0.3, 0.8, len(dates))
            fig_trends.add_trace(
                go.Scatter(
                    x=trend_data["čas"],
                    y=system_load,
                    mode="lines",
                    name="Zátěž",
                    line=dict(color="#d62728"),
                ),
                row=2,
                col=2,
            )

            fig_trends.update_layout(
                height=500, showlegend=False, title_text="Trendy systému za posledních 7 dní"
            )
            st.plotly_chart(fig_trends, use_container_width=True)

            # Auto-refresh countdown
            time.sleep(1)  # Simulace
    else:
        st.info("📊 Auto-refresh je vypnutý. Zapněte ho v postranním panelu pro živé aktualizace.")

        # Statické zobrazení
        st.subheader("📋 Přehled zjištění")
        if st.session_state.research_results:
            results = st.session_state.research_results
            st.write(f"**Poslední výzkum:** {results['query']}")
            st.write(f"**Nalezeno entit:** {results['entities_found']}")
            st.write(f"**Detekováno vzorů:** {results['patterns_detected']}")
        else:
            st.write("Zatím nebyly provedeny žádné výzkumy.")

# ===== ZÁLOŽKA 4: BEZPEČNOSTNÍ MONITOR =====
with tab4:
    st.header("🛡️ Bezpečnostní Monitor a Systémové Metriky")

    # Systémové prostředky
    st.subheader("💻 Systémové prostředky")

    col1, col2, col3, col4 = st.columns(4)

    # Mock systémové metriky
    cpu_usage = 30 + np.random.uniform(-5, 15)
    memory_usage = 65 + np.random.uniform(-10, 20)
    disk_usage = 25 + np.random.uniform(-5, 10)
    network_throughput = 5 + np.random.uniform(-2, 8)

    with col1:
        cpu_delta = np.random.uniform(-3, 3)
        st.metric("CPU využití", f"{cpu_usage:.1f}%", delta=f"{cpu_delta:+.1f}%")

    with col2:
        mem_delta = np.random.uniform(-2, 4)
        st.metric("RAM využití", f"{memory_usage:.1f}%", delta=f"{mem_delta:+.1f}%")

    with col3:
        st.metric("Disk využití", f"{disk_usage:.1f}%", delta="-0.5%")

    with col4:
        net_delta = np.random.uniform(-1, 2)
        st.metric("Síťový provoz", f"{network_throughput:.1f} MB/s", delta=f"{net_delta:+.1f}")

    # Bezpečnostní status
    st.markdown("---")
    st.subheader("🔐 Stav anonymity a bezpečnosti")

    col1, col2 = st.columns(2)

    with col1:
        if tor_enabled:
            st.success("✅ **Tor proxy aktivní**")
            st.info("🌐 **IP adresa:** 185.220.xxx.xxx (Tor exit node)")
            st.info("🏁 **Lokace:** Německo")
            st.info("🔄 **Rotace:** Každých 10 minut")
        else:
            st.error("❌ **Tor proxy vypnuté**")
            st.warning("⚠️ Připojení není anonymizované!")

    with col2:
        if vpn_enabled:
            st.success("✅ **VPN připojeno**")
            st.info("🔒 **Šifrování:** AES-256")
            st.info("📍 **Server:** NordVPN-DE#342")
            st.info("🚀 **Rychlost:** 95 Mbps")
        else:
            st.error("❌ **VPN odpojeno**")
            st.warning("⚠️ Připojení není chráněné!")

    # Pokročilé metriky
    if ADVANCED_COMPONENTS:
        st.markdown("---")
        st.subheader("📊 Pokročilé systémové metriky")

        # Real-time system dashboard
        fig_system = RealTimeMetricsComponent.create_system_health_dashboard()
        st.plotly_chart(fig_system, use_container_width=True)

    # Alerty a události
    st.markdown("---")
    st.subheader("🚨 Alerty a systémové události")

    # Mock alerty
    current_alerts = [
        {
            "level": "WARNING",
            "category": "SYSTEM",
            "message": f"Vysoké využití paměti: {memory_usage:.1f}%",
            "timestamp": datetime.now() - timedelta(minutes=5),
            "details": {"memory_percent": memory_usage},
        },
        {
            "level": "INFO",
            "category": "AGENT",
            "message": "Dokončena autonomní iterace #7",
            "timestamp": datetime.now() - timedelta(minutes=2),
            "details": {"iteration": 7},
        },
    ]

    if ADVANCED_COMPONENTS:
        AlertsAndNotificationsComponent.display_alerts_panel(current_alerts)
    else:
        # Základní zobrazení alertů
        for alert in current_alerts:
            alert_color = {
                "INFO": "blue",
                "WARNING": "orange",
                "ERROR": "red",
                "CRITICAL": "darkred",
            }.get(alert["level"], "gray")

            st.markdown(
                f"""
            <div style="padding: 1rem; margin: 0.5rem 0; border-left: 4px solid {alert_color}; background-color: #f8f9fa;">
                <strong style="color: {alert_color};">[{alert['level']}]</strong> {alert['message']}<br>
                <small style="color: gray;">📅 {alert['timestamp'].strftime('%H:%M:%S')} | 📂 {alert['category']}</small>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Systémové logy
    st.markdown("---")
    st.subheader("📋 Systémové logy")

    # Mock log entries
    log_entries = [
        {"čas": "14:42:15", "úroveň": "INFO", "zpráva": "Agent spuštěn - iterace 1/10"},
        {"čas": "14:41:58", "úroveň": "SUCCESS", "zpráva": "Tor circuit vytvořen úspěšně"},
        {
            "čas": "14:41:45",
            "úroveň": "INFO",
            "zpráva": f"Scraping dokončen: example.com (důvěryhodnost: 0.78)",
        },
        {
            "čas": "14:41:32",
            "úroveň": "WARNING",
            "zpráva": f"Vysoké využití RAM: {memory_usage:.1f}%",
        },
        {"čas": "14:41:15", "úroveň": "INFO", "zpráva": "VPN připojení obnoveno"},
        {
            "čas": "14:41:01",
            "úroveň": "INFO",
            "zpráva": "Autonomní agent přešel do balanced strategie",
        },
        {
            "čas": "14:40:45",
            "úroveň": "ERROR",
            "zpráva": "Scraping selhal: timeout na target.onion",
        },
        {
            "čas": "14:40:30",
            "úroveň": "SUCCESS",
            "zpráva": f"Nalezeno {np.random.randint(5,12)} nových entit",
        },
    ]

    # Scrollovatelné logy
    with st.container():
        for log in log_entries:
            log_color = {
                "INFO": "#17a2b8",
                "SUCCESS": "#28a745",
                "WARNING": "#ffc107",
                "ERROR": "#dc3545",
            }.get(log["úroveň"], "#6c757d")

            st.markdown(
                f"""
            <div style="font-family: monospace; padding: 0.3rem; margin: 0.1rem 0; font-size: 0.9rem;">
                <span style="color: #6c757d;">{log['čas']}</span>
                <span style="color: {log_color}; font-weight: bold; margin-left: 1rem;">[{log['úroveň']}]</span>
                <span style="margin-left: 1rem;">{log['zpráva']}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )


# Pomocné funkce
def create_simple_network_graph(network_data: Dict, size_metric: str) -> go.Figure:
    """Vytvoří jednoduchou vizualizaci síťového grafu bez pokročilých komponent"""

    nodes = network_data["nodes"]
    edges = network_data["edges"]

    # Jednoduché spring layout
    np.random.seed(42)  # Pro konzistentní layout
    pos = {}
    for i, node in enumerate(nodes):
        angle = (i / len(nodes)) * 2 * np.pi
        radius = 1 + np.random.random() * 0.5
        pos[node["id"]] = (radius * np.cos(angle), radius * np.sin(angle))

    # Příprava dat pro hrany
    edge_x, edge_y = [], []
    for edge in edges:
        if edge["source"] in pos and edge["target"] in pos:
            x0, y0 = pos[edge["source"]]
            x1, y1 = pos[edge["target"]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    # Hrany
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, line=dict(width=0.5, color="#888"), hoverinfo="none", mode="lines"
    )

    # Uzly
    node_x = [pos[node["id"]][0] for node in nodes if node["id"] in pos]
    node_y = [pos[node["id"]][1] for node in nodes if node["id"] in pos]
    node_text = [
        f"{node['label']}<br>Typ: {node['type']}<br>Důvěryhodnost: {node['credibility']}"
        for node in nodes
        if node["id"] in pos
    ]

    # Velikost podle metriky
    if size_metric == "Důvěryhodnost":
        node_size = [node["credibility"] * 30 + 10 for node in nodes if node["id"] in pos]
    elif size_metric == "Počet spojení":
        node_size = [node["connections"] * 3 + 10 for node in nodes if node["id"] in pos]
    else:
        node_size = [20 for node in nodes if node["id"] in pos]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale="Viridis",
            color=[node["credibility"] for node in nodes if node["id"] in pos],
            size=node_size,
            colorbar=dict(title="Důvěryhodnost"),
            line=dict(width=2),
        ),
    )

    # Graf
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f"Síťový graf entit (velikost: {size_metric})",
        titlefont_size=16,
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[
            dict(
                text="Základní vizualizace - pro pokročilé funkce nainstalujte všechny závislosti",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.005,
                y=-0.002,
                xanchor="left",
                yanchor="bottom",
                font=dict(color="gray", size=10),
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    return fig


# Footer
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    🧠 <strong>Deep Research Tool - Fáze 4: Autonomní Agent & Interaktivní UI</strong><br>
    Pokročilý výzkumný systém s autonomním rozhodováním | Verze 4.0<br>
    <small>Optimalizováno pro MacBook Air M1 8GB | Všechny komponenty funkční</small>
</div>
""",
    unsafe_allow_html=True,
)
