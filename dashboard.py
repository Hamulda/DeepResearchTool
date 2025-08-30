#!/usr/bin/env python3
"""
Základní dashboard pro DeepResearchTool
Poskytuje jednoduché webové rozhraní pro monitoring a správu
"""

import streamlit as st
import asyncio
import json
import time
from pathlib import Path
import sys
from typing import Dict, Any

# Přidání src do path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from main import ModernResearchAgent
    from src.core.config import get_settings
except ImportError as e:
    st.error(f"Chyba importu: {e}")
    st.info("Zkontrolujte, že jsou nainstalovány všechny závislosti")

def main():
    """Hlavní dashboard funkce"""
    st.set_page_config(
        page_title="DeepResearchTool Dashboard",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔍 DeepResearchTool Dashboard")
    st.markdown("---")
    
    # Sidebar pro nastavení
    with st.sidebar:
        st.header("⚙️ Nastavení")
        
        profile = st.selectbox(
            "Profil výzkumu",
            ["quick", "thorough", "academic"],
            index=1,
            help="Vyberte profil podle požadované hloubky analýzy"
        )
        
        audit_mode = st.checkbox(
            "Audit mód",
            help="Ukládání detailních logů a artefaktů"
        )
        
        st.markdown("---")
        st.subheader("📊 Info o profilu")
        
        if profile == "quick":
            st.info("⚡ Rychlé výsledky (~1-2 min)")
        elif profile == "thorough":
            st.info("🎯 Detailní analýza (~3-5 min)")
        else:
            st.info("🎓 Akademický standard (~5-10 min)")
    
    # Hlavní oblast
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Research Query")
        
        query = st.text_area(
            "Zadejte váš výzkumný dotaz:",
            placeholder="Např: What are the latest developments in quantum computing?",
            height=100
        )
        
        if st.button("🚀 Spustit Research", type="primary"):
            if query:
                run_research(query, profile, audit_mode)
            else:
                st.error("Zadejte prosím výzkumný dotaz")
    
    with col2:
        st.subheader("📈 Rychlé statistiky")
        
        # Placeholder pro statistiky
        st.metric("Celkem dotazů", "0")
        st.metric("Úspěšnost", "0%")
        st.metric("Průměrný čas", "0s")
    
    # Sekce pro historii
    st.markdown("---")
    st.subheader("📚 Historie výzkumu")
    
    # Načtení historie ze session state
    if 'research_history' not in st.session_state:
        st.session_state.research_history = []
    
    if st.session_state.research_history:
        for i, item in enumerate(reversed(st.session_state.research_history[-5:])):
            with st.expander(f"🔍 {item['query'][:50]}... ({item['timestamp']})"):
                st.write(f"**Profil:** {item['profile']}")
                st.write(f"**Čas:** {item['processing_time']:.2f}s")
                if 'result' in item:
                    st.write(f"**Claims:** {len(item['result'].get('claims', []))}")
                    st.write(f"**Citace:** {len(item['result'].get('citations', []))}")
    else:
        st.info("Zatím žádná historie výzkumu")

def run_research(query: str, profile: str, audit_mode: bool):
    """Spuštění research dotazu"""
    try:
        with st.spinner(f"Spouštím research (profil: {profile})..."):
            # Placeholder pro asynchronní volání
            # V reálné implementaci by zde bylo asyncio.run()
            
            # Mock výsledek pro demonstraci
            result = create_mock_result(query, profile)
            
            # Uložení do historie
            history_item = {
                'query': query,
                'profile': profile,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'processing_time': result['processing_time'],
                'result': result
            }
            
            if 'research_history' not in st.session_state:
                st.session_state.research_history = []
            
            st.session_state.research_history.append(history_item)
            
            # Zobrazení výsledků
            display_results(result)
            
    except Exception as e:
        st.error(f"Chyba během research: {e}")

def create_mock_result(query: str, profile: str) -> Dict[str, Any]:
    """Vytvoření mock výsledku pro demonstraci"""
    return {
        'query': query,
        'profile': profile,
        'processing_time': 45.2,
        'architecture': 'langgraph',
        'claims': [
            {
                'claim': 'Mock claim based on the query',
                'confidence': 0.85,
                'source': 'mock'
            }
        ],
        'citations': [
            {
                'id': 'cite_1',
                'source': 'Mock Source',
                'title': 'Mock Document Title',
                'url': 'https://example.com',
                'relevance_score': 0.9
            }
        ],
        'synthesis': f'Toto je mock syntéza pro dotaz: {query}. V reálné implementaci by zde byl detailní výzkumný obsah.',
        'validation_scores': {
            'groundedness': 0.87,
            'relevance': 0.92,
            'coherence': 0.89
        }
    }

def display_results(result: Dict[str, Any]):
    """Zobrazení výsledků research"""
    st.success("✅ Research dokončen!")
    
    # Základní metriky
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Čas zpracování", f"{result['processing_time']:.1f}s")
    
    with col2:
        st.metric("Claims", len(result['claims']))
    
    with col3:
        st.metric("Citace", len(result['citations']))
    
    with col4:
        if result.get('validation_scores'):
            avg_score = sum(result['validation_scores'].values()) / len(result['validation_scores'])
            st.metric("Avg. Score", f"{avg_score:.2f}")
    
    # Syntéza
    if result.get('synthesis'):
        st.subheader("📄 Syntéza")
        st.write(result['synthesis'])
    
    # Claims
    if result['claims']:
        st.subheader("🎯 Klíčové poznatky")
        for i, claim in enumerate(result['claims'], 1):
            confidence = claim.get('confidence', 0)
            color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
            st.write(f"{i}. {color} {claim['claim']} (Confidence: {confidence:.2f})")
    
    # Validační skóre
    if result.get('validation_scores'):
        st.subheader("✅ Validační skóre")
        cols = st.columns(len(result['validation_scores']))
        for i, (metric, score) in enumerate(result['validation_scores'].items()):
            with cols[i]:
                st.metric(metric.title(), f"{score:.2f}")

if __name__ == "__main__":
    main()