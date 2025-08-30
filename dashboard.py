"""
Interaktivní Streamlit Dashboard pro DeepResearchTool
Poskytuje vizualizaci výsledků, knowledge graph a real-time monitoring
"""

import streamlit as st
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile
from pathlib import Path

# Import našich modulů
from src.core.config import get_settings, validate_environment
from src.core.async_batch_processor import AsyncBatchProcessor
from src.scrapers.stealth_scraper import StealthScraper
from src.scrapers.tor_stealth_scraper import TorStealthScraper
from src.analysis.ocr_processor import MultiOCRProcessor
from src.analysis.speech_to_text_processor import MultiSpeechProcessor
from src.graph.knowledge_graph_builder import KnowledgeGraphBuilder

logger = logging.getLogger(__name__)

# Streamlit page config
st.set_page_config(
    page_title="DeepResearchTool Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


class DashboardState:
    """Manages dashboard state and caching"""
    
    def __init__(self):
        if 'dashboard_state' not in st.session_state:
            st.session_state.dashboard_state = {
                'research_results': [],
                'knowledge_graph_stats': {},
                'scraping_stats': {},
                'processing_stats': {},
                'last_update': None
            }
    
    def get_state(self) -> Dict[str, Any]:
        return st.session_state.dashboard_state
    
    def update_state(self, key: str, value: Any):
        st.session_state.dashboard_state[key] = value
        st.session_state.dashboard_state['last_update'] = datetime.now()


class ResearchInterface:
    """Main research interface components"""
    
    def __init__(self, dashboard_state: DashboardState):
        self.state = dashboard_state
    
    def render_research_form(self):
        """Render the main research query form"""
        st.markdown("## 🔍 Research Query")
        
        with st.form("research_form"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                query = st.text_input(
                    "Enter your research topic:",
                    placeholder="e.g., artificial intelligence developments 2024",
                    help="Enter a topic you want to research deeply"
                )
            
            with col2:
                research_mode = st.selectbox(
                    "Research Mode:",
                    ["Standard", "Deep Web", "Academic Only", "News Focus"]
                )
            
            # Advanced options
            with st.expander("Advanced Options"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    max_documents = st.slider("Max Documents", 10, 200, 50)
                    include_tor = st.checkbox("Include Tor Networks", value=False)
                
                with col2:
                    time_range = st.selectbox(
                        "Time Range:",
                        ["All Time", "Last Year", "Last 6 Months", "Last Month"]
                    )
                    include_multimedia = st.checkbox("Process Multimedia", value=True)
                
                with col3:
                    output_format = st.selectbox(
                        "Output Format:",
                        ["Interactive Report", "JSON Export", "PDF Report"]
                    )
                    build_knowledge_graph = st.checkbox("Build Knowledge Graph", value=True)
            
            submitted = st.form_submit_button("🚀 Start Research")
            
            if submitted and query:
                return {
                    'query': query,
                    'research_mode': research_mode,
                    'max_documents': max_documents,
                    'include_tor': include_tor,
                    'time_range': time_range,
                    'include_multimedia': include_multimedia,
                    'output_format': output_format,
                    'build_knowledge_graph': build_knowledge_graph
                }
        
        return None
    
    def render_research_progress(self, research_params: Dict[str, Any]):
        """Render research progress indicators"""
        st.markdown("## 📊 Research Progress")
        
        # Progress bars
        progress_container = st.container()
        
        with progress_container:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Documents Collected", "0", "0")
                progress_1 = st.progress(0)
            
            with col2:
                st.metric("Sources Processed", "0", "0")
                progress_2 = st.progress(0)
            
            with col3:
                st.metric("Entities Extracted", "0", "0")
                progress_3 = st.progress(0)
            
            with col4:
                st.metric("Analysis Complete", "0%", "0%")
                progress_4 = st.progress(0)
        
        # Live log output
        st.markdown("### 📝 Live Progress Log")
        log_container = st.empty()
        
        # Simulate research process (in real implementation, this would be actual async research)
        return self._run_research_simulation(research_params, progress_container, log_container)
    
    async def _run_research_simulation(self, params: Dict[str, Any], progress_container, log_container):
        """Simulate the research process with real components"""
        
        # Initialize components
        stealth_scraper = StealthScraper(headless=True)
        
        try:
            # Phase 1: Initialize
            log_container.text("🔄 Initializing research components...")
            await asyncio.sleep(1)
            
            # Phase 2: Web Scraping
            log_container.text("🌐 Starting web scraping...")
            await stealth_scraper.start()
            
            # Simulate scraping some URLs
            sample_urls = [
                "https://httpbin.org/html",
                "https://httpbin.org/json"
            ]
            
            if len(sample_urls) > 0:
                results = await stealth_scraper.scrape_multiple(sample_urls[:2])
                log_container.text(f"✅ Scraped {len(results)} web pages")
            
            # Phase 3: Knowledge Graph Building
            if params.get('build_knowledge_graph'):
                log_container.text("🧠 Building knowledge graph...")
                await asyncio.sleep(2)
                log_container.text("✅ Knowledge graph updated")
            
            # Update final results
            self.state.update_state('research_results', {
                'query': params['query'],
                'documents_found': len(sample_urls),
                'completed_at': datetime.now().isoformat(),
                'success': True
            })
            
            log_container.text("🎉 Research completed successfully!")
            
        except Exception as e:
            log_container.text(f"❌ Research failed: {str(e)}")
            logger.error(f"Research simulation failed: {e}")
        
        finally:
            await stealth_scraper.close()


class KnowledgeGraphViz:
    """Knowledge graph visualization components"""
    
    def __init__(self, dashboard_state: DashboardState):
        self.state = dashboard_state
    
    def render_knowledge_graph(self):
        """Render interactive knowledge graph visualization"""
        st.markdown("## 🧠 Knowledge Graph")
        
        # Sample data for demonstration
        sample_entities = [
            {"name": "Artificial Intelligence", "type": "Concept", "connections": 15},
            {"name": "OpenAI", "type": "Organization", "connections": 8},
            {"name": "GPT-4", "type": "Technology", "connections": 12},
            {"name": "Machine Learning", "type": "Concept", "connections": 20},
            {"name": "Deep Learning", "type": "Concept", "connections": 18}
        ]
        
        sample_relationships = [
            {"source": "OpenAI", "target": "GPT-4", "type": "CREATED"},
            {"source": "GPT-4", "target": "Artificial Intelligence", "type": "PART_OF"},
            {"source": "Machine Learning", "target": "Artificial Intelligence", "type": "PART_OF"},
            {"source": "Deep Learning", "target": "Machine Learning", "type": "PART_OF"}
        ]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Interactive network graph
            self._render_network_graph(sample_entities, sample_relationships)
        
        with col2:
            # Graph statistics
            self._render_graph_stats(sample_entities, sample_relationships)
    
    def _render_network_graph(self, entities: List[Dict], relationships: List[Dict]):
        """Render interactive network using pyvis"""
        try:
            # Create network
            net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black")
            
            # Add nodes
            for entity in entities:
                color = self._get_entity_color(entity['type'])
                net.add_node(
                    entity['name'], 
                    label=entity['name'],
                    color=color,
                    size=min(50, max(20, entity['connections'] * 2))
                )
            
            # Add edges
            for rel in relationships:
                net.add_edge(
                    rel['source'], 
                    rel['target'], 
                    label=rel['type'],
                    color="#cccccc"
                )
            
            # Configure physics
            net.set_options("""
            var options = {
                "physics": {
                    "enabled": true,
                    "stabilization": {"iterations": 100}
                }
            }
            """)
            
            # Save to temp file and display
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                net.save_graph(tmp.name)
                with open(tmp.name, 'r') as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=500)
            
        except Exception as e:
            st.error(f"Failed to render knowledge graph: {e}")
            # Fallback to simple text display
            st.json({"entities": entities[:3], "relationships": relationships[:3]})
    
    def _get_entity_color(self, entity_type: str) -> str:
        """Get color for entity type"""
        colors = {
            "Person": "#ff9999",
            "Organization": "#66b3ff", 
            "Concept": "#99ff99",
            "Technology": "#ffcc99",
            "Location": "#ff99cc"
        }
        return colors.get(entity_type, "#cccccc")
    
    def _render_graph_stats(self, entities: List[Dict], relationships: List[Dict]):
        """Render knowledge graph statistics"""
        st.markdown("### 📈 Graph Statistics")
        
        # Entity type distribution
        entity_types = {}
        for entity in entities:
            entity_type = entity['type']
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        # Create pie chart
        fig = px.pie(
            values=list(entity_types.values()),
            names=list(entity_types.keys()),
            title="Entity Types Distribution"
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Key metrics
        st.metric("Total Entities", len(entities))
        st.metric("Total Relationships", len(relationships))
        st.metric("Average Connections", 
                 round(sum(e['connections'] for e in entities) / len(entities), 1))


class SystemMonitoring:
    """System monitoring and status components"""
    
    def __init__(self, dashboard_state: DashboardState):
        self.state = dashboard_state
    
    def render_system_status(self):
        """Render system status overview"""
        st.markdown("## 🖥️ System Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Configuration status
            config_status = self._check_configuration()
            status_class = "status-success" if config_status['healthy'] else "status-error"
            st.markdown(f'<div class="metric-card">'
                       f'<h4>Configuration</h4>'
                       f'<p class="{status_class}">{config_status["status"]}</p>'
                       f'</div>', unsafe_allow_html=True)
        
        with col2:
            # Database connections
            db_status = self._check_databases()
            status_class = "status-success" if db_status['healthy'] else "status-warning"
            st.markdown(f'<div class="metric-card">'
                       f'<h4>Databases</h4>'
                       f'<p class="{status_class}">{db_status["status"]}</p>'
                       f'</div>', unsafe_allow_html=True)
        
        with col3:
            # Scraping capabilities
            scraping_status = self._check_scraping()
            status_class = "status-success" if scraping_status['healthy'] else "status-warning"
            st.markdown(f'<div class="metric-card">'
                       f'<h4>Scraping</h4>'
                       f'<p class="{status_class}">{scraping_status["status"]}</p>'
                       f'</div>', unsafe_allow_html=True)
        
        with col4:
            # AI/ML models
            ai_status = self._check_ai_models()
            status_class = "status-success" if ai_status['healthy'] else "status-warning"
            st.markdown(f'<div class="metric-card">'
                       f'<h4>AI Models</h4>'
                       f'<p class="{status_class}">{ai_status["status"]}</p>'
                       f'</div>', unsafe_allow_html=True)
    
    def _check_configuration(self) -> Dict[str, Any]:
        """Check configuration status"""
        try:
            settings = get_settings()
            validation = settings.validate_critical_config()
            
            healthy_count = sum(1 for v in validation.values() if v)
            total_count = len(validation)
            
            if healthy_count == total_count:
                return {"healthy": True, "status": "✅ All Good"}
            elif healthy_count > total_count // 2:
                return {"healthy": False, "status": "⚠️ Partial"}
            else:
                return {"healthy": False, "status": "❌ Issues"}
                
        except Exception:
            return {"healthy": False, "status": "❌ Error"}
    
    def _check_databases(self) -> Dict[str, Any]:
        """Check database connections"""
        # This would check actual database connections
        # For demo, returning mock status
        return {"healthy": True, "status": "✅ Connected"}
    
    def _check_scraping(self) -> Dict[str, Any]:
        """Check scraping capabilities"""
        # This would check Playwright, Tor, etc.
        return {"healthy": True, "status": "✅ Ready"}
    
    def _check_ai_models(self) -> Dict[str, Any]:
        """Check AI model availability"""
        # This would check LLM connections, embeddings, etc.
        return {"healthy": True, "status": "⚠️ Limited"}
    
    def render_performance_metrics(self):
        """Render performance metrics and charts"""
        st.markdown("### 📊 Performance Metrics")
        
        # Generate sample performance data
        dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
        
        # Create sample metrics
        scraping_data = pd.DataFrame({
            'date': dates,
            'documents_scraped': np.random.randint(50, 200, len(dates)),
            'success_rate': np.random.uniform(0.85, 0.98, len(dates)),
            'avg_response_time': np.random.uniform(1.5, 4.0, len(dates))
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Documents scraped over time
            fig1 = px.line(
                scraping_data, 
                x='date', 
                y='documents_scraped',
                title='Documents Scraped Over Time'
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Success rate over time
            fig2 = px.line(
                scraping_data,
                x='date',
                y='success_rate',
                title='Scraping Success Rate'
            )
            fig2.update_yaxis(range=[0.8, 1.0])
            st.plotly_chart(fig2, use_container_width=True)


def main():
    """Main dashboard application"""
    # Initialize dashboard state
    dashboard_state = DashboardState()
    
    # Sidebar navigation
    st.sidebar.title("🔍 DeepResearchTool")
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["Research Interface", "Knowledge Graph", "System Monitoring", "Settings"]
    )
    
    # Main header
    st.markdown('<h1 class="main-header">DeepResearchTool Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Route to appropriate page
    if page == "Research Interface":
        research_interface = ResearchInterface(dashboard_state)
        
        # Render research form
        research_params = research_interface.render_research_form()
        
        # If research submitted, show progress
        if research_params:
            research_interface.render_research_progress(research_params)
    
    elif page == "Knowledge Graph":
        kg_viz = KnowledgeGraphViz(dashboard_state)
        kg_viz.render_knowledge_graph()
    
    elif page == "System Monitoring":
        monitoring = SystemMonitoring(dashboard_state)
        monitoring.render_system_status()
        monitoring.render_performance_metrics()
    
    elif page == "Settings":
        st.markdown("## ⚙️ Settings")
        
        with st.expander("Configuration Validation"):
            if st.button("Validate Configuration"):
                try:
                    validate_environment()
                    st.success("Configuration validation completed!")
                except Exception as e:
                    st.error(f"Configuration validation failed: {e}")
        
        with st.expander("System Information"):
            settings = get_settings()
            st.json({
                "Environment": settings.app.environment,
                "Debug Mode": settings.app.debug,
                "Research Profile": settings.app.research_profile,
                "Max Documents": settings.app.max_documents
            })
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Status**: Online 🟢")
    st.sidebar.markdown(f"**Last Update**: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()