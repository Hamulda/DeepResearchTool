"""
🎛️ Pokročilé interaktivní komponenty pro Streamlit dashboard
Real-time vizualizace, síťové grafy a monitoring panels
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
import time
import asyncio
from typing import Dict, List, Any, Optional
import json


# Vlastní komponenty pro pokročilé vizualizace
class NetworkGraphComponent:
    """
    🕸️ Pokročilá komponenta pro interaktivní síťové grafy
    """

    @staticmethod
    def create_3d_network_graph(
        nodes: List[Dict], edges: List[Dict], layout_type: str = "spring"
    ) -> go.Figure:
        """Vytvoří 3D síťový graf s interaktivními funkcemi"""

        # Vytvoření NetworkX grafu
        G = nx.Graph()
        for node in nodes:
            G.add_node(node["id"], **node)
        for edge in edges:
            G.add_edge(edge["source"], edge["target"], weight=edge.get("weight", 1))

        # Layout algoritmy
        if layout_type == "spring":
            pos = nx.spring_layout(G, dim=3, k=3, iterations=50)
        elif layout_type == "circular":
            pos_2d = nx.circular_layout(G)
            pos = {node: (x, y, 0) for node, (x, y) in pos_2d.items()}
        else:  # random
            pos = {
                node: (np.random.random(), np.random.random(), np.random.random())
                for node in G.nodes()
            }

        # Příprava dat pro 3D vizualizaci
        edge_x, edge_y, edge_z = [], [], []
        for edge in edges:
            x0, y0, z0 = pos[edge["source"]]
            x1, y1, z1 = pos[edge["target"]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])

        # 3D hrany
        edge_trace = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line=dict(color="rgba(125,125,125,0.3)", width=2),
            hoverinfo="none",
        )

        # 3D uzly
        node_x = [pos[node["id"]][0] for node in nodes]
        node_y = [pos[node["id"]][1] for node in nodes]
        node_z = [pos[node["id"]][2] for node in nodes]
        node_text = [
            f"{node['label']}<br>Typ: {node.get('type', 'unknown')}<br>"
            f"Důvěryhodnost: {node.get('credibility', 0):.2f}"
            for node in nodes
        ]

        node_trace = go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode="markers+text",
            text=[node["label"] for node in nodes],
            textposition="middle center",
            hovertext=node_text,
            hoverinfo="text",
            marker=dict(
                size=[node.get("credibility", 0.5) * 20 + 5 for node in nodes],
                color=[node.get("credibility", 0.5) for node in nodes],
                colorscale="Viridis",
                colorbar=dict(title="Důvěryhodnost"),
                line=dict(width=2, color="white"),
            ),
        )

        # Vytvoření grafu
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title="3D Síťový Graf Entit",
            showlegend=False,
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                bgcolor="rgba(0,0,0,0)",
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )

        return fig

    @staticmethod
    def create_cluster_analysis_view(nodes: List[Dict], edges: List[Dict]) -> go.Figure:
        """Vytvoří vizualizaci clusterové analýzy"""

        # Vytvoření grafu a detekce komunit
        G = nx.Graph()
        for node in nodes:
            G.add_node(node["id"], **node)
        for edge in edges:
            G.add_edge(edge["source"], edge["target"])

        # Simulace detekce komunit (v reálné aplikaci by použila algoritmus)
        communities = []
        nodes_per_community = len(nodes) // 3
        for i in range(3):
            start_idx = i * nodes_per_community
            end_idx = start_idx + nodes_per_community if i < 2 else len(nodes)
            community_nodes = [nodes[j]["id"] for j in range(start_idx, end_idx)]
            communities.append(community_nodes)

        # Barvy pro komunity
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Vytvoření subplotů pro každou komunitu
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[f"Komunita {i+1}" for i in range(3)],
            specs=[[{"type": "scatter"}] * 3],
        )

        for i, community in enumerate(communities):
            # Uzly komunity
            community_nodes = [node for node in nodes if node["id"] in community]
            x_coords = [pos[node["id"]][0] for node in community_nodes]
            y_coords = [pos[node["id"]][1] for node in community_nodes]

            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="markers+text",
                    text=[node["label"] for node in community_nodes],
                    textposition="middle center",
                    marker=dict(
                        size=15, color=colors[i % len(colors)], line=dict(width=2, color="white")
                    ),
                    name=f"Komunita {i+1}",
                    showlegend=True,
                ),
                row=1,
                col=i + 1,
            )

        fig.update_layout(title="Analýza Komunit v Síti", height=400, showlegend=True)

        return fig


class RealTimeMetricsComponent:
    """
    📊 Komponenta pro real-time metriky a monitoring
    """

    @staticmethod
    def create_system_health_dashboard() -> go.Figure:
        """Vytvoří dashboard zdraví systému"""

        # Mock data pro systémové metriky
        current_time = datetime.now()
        time_points = [current_time - timedelta(minutes=i) for i in range(30, 0, -1)]

        cpu_data = [45 + np.sin(i / 5) * 10 + np.random.normal(0, 3) for i in range(30)]
        memory_data = [70 + np.sin(i / 7) * 8 + np.random.normal(0, 2) for i in range(30)]
        network_data = [20 + np.sin(i / 3) * 15 + np.random.normal(0, 4) for i in range(30)]

        # Vytvoření subplotů
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "CPU Využití (%)",
                "Paměť (%)",
                "Síťový Provoz (MB/s)",
                "Aktivní Úkoly",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # CPU graf
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=cpu_data,
                mode="lines",
                name="CPU",
                line=dict(color="#FF6B6B", width=2),
                fill="tonexty",
            ),
            row=1,
            col=1,
        )

        # Paměť graf
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=memory_data,
                mode="lines",
                name="Paměť",
                line=dict(color="#4ECDC4", width=2),
                fill="tonexty",
            ),
            row=1,
            col=2,
        )

        # Síť graf
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=network_data,
                mode="lines",
                name="Síť",
                line=dict(color="#45B7D1", width=2),
                fill="tonexty",
            ),
            row=2,
            col=1,
        )

        # Aktivní úkoly (sloupcový graf)
        task_counts = [3, 5, 2, 7, 4, 6, 3, 8, 2, 4]
        task_times = time_points[-10:]

        fig.add_trace(
            go.Bar(x=task_times, y=task_counts, name="Úkoly", marker_color="#96CEB4"), row=2, col=2
        )

        # Aktualizace layoutu
        fig.update_layout(
            title="Systémové Metriky - Real-time Dashboard", height=600, showlegend=False
        )

        # Přidání threshold linií
        fig.add_hline(
            y=80, line_dash="dash", line_color="red", annotation_text="CPU Warning", row=1, col=1
        )
        fig.add_hline(
            y=85, line_dash="dash", line_color="red", annotation_text="Memory Warning", row=1, col=2
        )

        return fig

    @staticmethod
    def create_agent_performance_chart(agent_metrics: List[Dict]) -> go.Figure:
        """Vytvoří graf performance agenta"""

        if not agent_metrics:
            # Mock data pokud nejsou dostupná
            timestamps = [datetime.now() - timedelta(minutes=i) for i in range(20, 0, -1)]
            agent_metrics = [
                {
                    "timestamp": ts,
                    "completed_tasks": 5 + i + np.random.randint(-2, 3),
                    "avg_credibility": 0.6 + np.sin(i / 5) * 0.1 + np.random.normal(0, 0.05),
                    "entities_discovered": 10 + i * 2 + np.random.randint(-3, 4),
                }
                for i, ts in enumerate(timestamps)
            ]

        timestamps = [m["timestamp"] for m in agent_metrics]
        completed_tasks = [m["completed_tasks"] for m in agent_metrics]
        credibility_scores = [m["avg_credibility"] for m in agent_metrics]
        entities = [m["entities_discovered"] for m in agent_metrics]

        # Vytvoření dual-axis grafu
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Dokončené Úkoly & Důvěryhodnost", "Objevené Entity"),
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        )

        # Dokončené úkoly
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=completed_tasks,
                mode="lines+markers",
                name="Dokončené úkoly",
                line=dict(color="#2E86AB", width=3),
                marker=dict(size=8),
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

        # Důvěryhodnost na sekundární ose
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=credibility_scores,
                mode="lines+markers",
                name="Průměrná důvěryhodnost",
                line=dict(color="#A23B72", width=3, dash="dash"),
                marker=dict(size=8, symbol="diamond"),
            ),
            row=1,
            col=1,
            secondary_y=True,
        )

        # Entity
        fig.add_trace(
            go.Bar(
                x=timestamps,
                y=entities,
                name="Objevené entity",
                marker_color="#F18F01",
                opacity=0.7,
            ),
            row=2,
            col=1,
        )

        # Nastavení os
        fig.update_yaxes(title_text="Počet úkolů", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Důvěryhodnost", range=[0, 1], row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Počet entit", row=2, col=1)

        fig.update_layout(title="Performance Autonomní Agenta", height=600, hovermode="x unified")

        return fig


class AlertsAndNotificationsComponent:
    """
    🚨 Komponenta pro alerty a notifikace
    """

    @staticmethod
    def display_alerts_panel(alerts: List[Dict]):
        """Zobrazí panel s aktivními alerty"""

        if not alerts:
            st.success("✅ Žádné aktivní alerty")
            return

        # Grupování podle úrovně
        alert_levels = {"CRITICAL": [], "ERROR": [], "WARNING": [], "INFO": []}
        for alert in alerts:
            level = alert.get("level", "INFO")
            if level in alert_levels:
                alert_levels[level].append(alert)

        # Zobrazení podle závažnosti
        level_colors = {"CRITICAL": "red", "ERROR": "orange", "WARNING": "yellow", "INFO": "blue"}

        level_icons = {"CRITICAL": "🔴", "ERROR": "🟠", "WARNING": "🟡", "INFO": "🔵"}

        for level, level_alerts in alert_levels.items():
            if level_alerts:
                st.markdown(f"### {level_icons[level]} {level} Alerty ({len(level_alerts)})")

                for alert in level_alerts[-5:]:  # Posledních 5
                    with st.expander(
                        f"{alert.get('message', 'Neznámý alert')}", expanded=(level == "CRITICAL")
                    ):
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.write(f"**Kategorie:** {alert.get('category', 'SYSTEM')}")
                            st.write(f"**Čas:** {alert.get('timestamp', 'Neznámý')}")

                            # Detaily
                            details = alert.get("details", {})
                            if details:
                                st.write("**Detaily:**")
                                for key, value in details.items():
                                    st.write(f"  • {key}: {value}")

                        with col2:
                            if st.button(f"✅ Potvrdit", key=f"ack_{alert.get('id', 'unknown')}"):
                                st.success("Alert potvrzen")

    @staticmethod
    def create_alerts_timeline() -> go.Figure:
        """Vytvoří timeline alertů"""

        # Mock data pro alerty
        current_time = datetime.now()
        alert_data = []

        for i in range(20):
            alert_time = current_time - timedelta(hours=i)
            level = np.random.choice(
                ["INFO", "WARNING", "ERROR", "CRITICAL"], p=[0.4, 0.3, 0.2, 0.1]
            )

            alert_data.append(
                {
                    "timestamp": alert_time,
                    "level": level,
                    "message": f"Alert {i+1}",
                    "y_pos": {"INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}[level],
                }
            )

        # Vytvoření timeline grafu
        fig = go.Figure()

        colors = {
            "INFO": "#2E86AB",
            "WARNING": "#F18F01",
            "ERROR": "#C73E1D",
            "CRITICAL": "#8B0000",
        }

        for level in ["INFO", "WARNING", "ERROR", "CRITICAL"]:
            level_alerts = [a for a in alert_data if a["level"] == level]

            if level_alerts:
                fig.add_trace(
                    go.Scatter(
                        x=[a["timestamp"] for a in level_alerts],
                        y=[a["y_pos"] for a in level_alerts],
                        mode="markers",
                        name=level,
                        marker=dict(size=12, color=colors[level], symbol="circle"),
                        text=[a["message"] for a in level_alerts],
                        hovertemplate="<b>%{text}</b><br>Čas: %{x}<br>Úroveň: "
                        + level
                        + "<extra></extra>",
                    )
                )

        fig.update_layout(
            title="Timeline Alertů",
            xaxis_title="Čas",
            yaxis=dict(
                tickmode="array",
                tickvals=[1, 2, 3, 4],
                ticktext=["INFO", "WARNING", "ERROR", "CRITICAL"],
            ),
            height=300,
            hovermode="closest",
        )

        return fig


class CredibilityVisualizationComponent:
    """
    🎯 Komponenta pro vizualizaci důvěryhodnosti
    """

    @staticmethod
    def create_credibility_heatmap(sources: List[Dict]) -> go.Figure:
        """Vytvoří heatmapu důvěryhodnosti zdrojů"""

        if not sources:
            # Mock data
            sources = [
                {"domain": "blockchain.info", "credibility": 0.95, "category": "cryptocurrency"},
                {"domain": "coindesk.com", "credibility": 0.88, "category": "news"},
                {"domain": "reddit.com", "credibility": 0.45, "category": "social"},
                {"domain": "pastebin.com", "credibility": 0.25, "category": "data_dump"},
                {"domain": "torproject.org", "credibility": 0.92, "category": "security"},
                {"domain": "unknown_forum.onion", "credibility": 0.15, "category": "darknet"},
            ]

        # Příprava dat pro heatmapu
        categories = list(set(s["category"] for s in sources))
        domains = [s["domain"] for s in sources]

        # Matice důvěryhodnosti
        credibility_matrix = []
        for category in categories:
            row = []
            for source in sources:
                if source["category"] == category:
                    row.append(source["credibility"])
                else:
                    row.append(None)
            credibility_matrix.append(row)

        fig = go.Figure(
            data=go.Heatmap(
                z=credibility_matrix,
                x=domains,
                y=categories,
                colorscale="RdYlGn",
                zmid=0.5,
                colorbar=dict(title="Důvěryhodnost"),
            )
        )

        fig.update_layout(
            title="Heatmapa Důvěryhodnosti Zdrojů",
            xaxis_title="Domény",
            yaxis_title="Kategorie",
            height=400,
        )

        return fig

    @staticmethod
    def create_credibility_distribution() -> go.Figure:
        """Vytvoří distribuci skóre důvěryhodnosti"""

        # Mock data pro distribuci
        credibility_scores = np.random.beta(2, 2, 1000)  # Beta distribuce

        fig = go.Figure()

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=credibility_scores,
                nbinsx=30,
                name="Distribuce",
                marker_color="skyblue",
                opacity=0.7,
            )
        )

        # Průměr
        mean_score = np.mean(credibility_scores)
        fig.add_vline(
            x=mean_score,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Průměr: {mean_score:.2f}",
        )

        # Threshold linie
        fig.add_vline(
            x=0.7, line_dash="dot", line_color="green", annotation_text="Vysoká důvěryhodnost"
        )

        fig.add_vline(
            x=0.3, line_dash="dot", line_color="orange", annotation_text="Nízká důvěryhodnost"
        )

        fig.update_layout(
            title="Distribuce Skóre Důvěryhodnosti",
            xaxis_title="Skóre důvěryhodnosti",
            yaxis_title="Počet zdrojů",
            height=400,
        )

        return fig


def create_advanced_streamlit_components():
    """
    🎛️ Funkce pro vytvoření všech pokročilých komponent
    """

    components = {
        "network_graph": NetworkGraphComponent(),
        "realtime_metrics": RealTimeMetricsComponent(),
        "alerts": AlertsAndNotificationsComponent(),
        "credibility": CredibilityVisualizationComponent(),
    }

    return components
