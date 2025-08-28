"""
Demonstrace moderní LangGraph Research Agent architektury
Ukázka použití stavového automatu s RAG pipeline a nástroji

Author: Senior Python/MLOps Agent
"""

import asyncio
import logging
import os
from pathlib import Path

# Nastavení logování
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_langgraph_research():
    """Demonstrace kompletní LangGraph research architektury"""

    try:
        # Import nových komponent
        from src.core.langgraph_agent import ResearchAgentGraph
        from src.core.config_langgraph import load_config, validate_config

        print("🚀 Demonstrace LangGraph Research Agent")
        print("=" * 50)

        # 1. Načtení a validace konfigurace
        print("\n📋 Krok 1: Načítání konfigurace...")
        config = load_config(profile="thorough")

        validation_errors = validate_config(config)
        if validation_errors:
            print("❌ Chyby v konfiguraci:")
            for error in validation_errors:
                print(f"  - {error}")
            return

        print("✅ Konfigurace validována úspěšně")
        print(f"   - LLM model: {config['llm']['model']}")
        print(f"   - Synthesis model: {config['llm']['synthesis_model']}")
        print(f"   - Embedding model: {config['memory_store']['embedding_model']}")
        print(f"   - Chunk size: {config['rag']['chunking']['chunk_size']}")

        # 2. Inicializace LangGraph agenta
        print("\n🤖 Krok 2: Inicializace LangGraph agenta...")
        agent = ResearchAgentGraph(config)
        print("✅ Agent inicializován s stavovým automatem")
        print(f"   - Uzly grafu: plan → retrieve → validate → synthesize")
        print(f"   - Dostupné nástroje: {len(agent.tools)}")

        # 3. Demonstrace základního výzkumu
        print("\n🔍 Krok 3: Spuštění výzkumu...")
        query = "Jaké jsou nejnovější trendy v oblasti umělé inteligence v roce 2024?"

        print(f"Dotaz: {query}")
        print("Spouštím stavový automat...")

        result = await agent.research(query)

        # 4. Zobrazení výsledků
        print("\n📊 Krok 4: Výsledky výzkumu")
        print("=" * 50)

        print(f"\n📝 Vygenerovaný plán ({len(result['plan'])} kroků):")
        for i, step in enumerate(result['plan'], 1):
            print(f"   {i}. {step}")

        print(f"\n📚 Získané dokumenty: {len(result['retrieved_docs'])}")
        for i, doc in enumerate(result['retrieved_docs'][:3], 1):
            source = doc.get('source', 'unknown')
            step = doc.get('step', 'general')
            content_preview = doc.get('content', '')[:100] + "..."
            print(f"   {i}. [{source}] {step}: {content_preview}")

        print(f"\n✅ Validační skóre:")
        for metric, score in result['validation_scores'].items():
            print(f"   - {metric}: {score:.2f}")

        print(f"\n📄 Syntéza (délka: {len(result['synthesis'])} znaků):")
        # Zobrazí pouze první část syntézy
        synthesis_preview = result['synthesis'][:500] + "..." if len(result['synthesis']) > 500 else result['synthesis']
        print(f"   {synthesis_preview}")

        print(f"\n⏱️  Celkový čas zpracování: {result['processing_time']:.2f} sekund")
        print(f"🏗️  Architektura: {result['metadata']['architecture']}")
        print(f"🔗 RAG enabled: {result['metadata']['rag_enabled']}")

        if result['errors']:
            print(f"\n⚠️  Chyby během zpracování:")
            for error in result['errors']:
                print(f"   - {error}")

        return result

    except ImportError as e:
        print(f"❌ Chyba importu: {e}")
        print("Zkontrolujte, zda jsou nainstalovány všechny závislosti:")
        print("pip install langgraph langchain chromadb sentence-transformers firecrawl-py")
        return None

    except Exception as e:
        logger.error(f"Chyba během demonstrace: {e}")
        print(f"❌ Neočekávaná chyba: {e}")
        return None


async def demo_rag_pipeline():
    """Demonstrace RAG pipeline samostatně"""

    print("\n🧠 Demonstrace RAG Pipeline")
    print("=" * 30)

    try:
        from src.core.rag_pipeline import RAGPipeline
        from src.core.config_langgraph import load_config

        config = load_config()
        rag = RAGPipeline(config)
        await rag.initialize()

        print("✅ RAG Pipeline inicializována")

        # Test ingesci dokumentů
        test_documents = [
            {
                "content": "Umělá inteligence v roce 2024 se zaměřuje na velké jazykové modely jako GPT-4, Claude a jejich aplikace.",
                "metadata": {"source": "test_doc_1", "topic": "AI_trends"}
            },
            {
                "content": "ChromaDB je vektorová databáze optimalizovaná pro ukládání a vyhledávání embeddingů v AI aplikacích.",
                "metadata": {"source": "test_doc_2", "topic": "vector_db"}
            }
        ]

        print(f"\n📥 Ingestuji {len(test_documents)} testovacích dokumentů...")
        chunk_ids = await rag.ingest_documents(test_documents)
        print(f"✅ Uloženo {len(chunk_ids)} chunků")

        # Test vyhledávání
        print(f"\n🔍 Test vyhledávání...")
        query = "Jaké jsou trendy v AI?"
        results = await rag.search(query, k=3)

        print(f"Nalezeno {len(results)} relevantních dokumentů:")
        for i, doc in enumerate(results, 1):
            distance = doc.metadata.get('distance', 'N/A')
            print(f"   {i}. Distance: {distance}, Content: {doc.content[:80]}...")

        return True

    except Exception as e:
        print(f"❌ Chyba RAG Pipeline: {e}")
        return False


async def demo_tools():
    """Demonstrace nástrojů (@tool decorator)"""

    print("\n🛠️  Demonstrace nástrojů")
    print("=" * 25)

    try:
        from src.core.tools import web_scraping_tool, knowledge_search_tool, document_analysis_tool

        # Test document analysis tool
        print("📊 Test nástroje pro analýzu dokumentů...")
        test_text = "Toto je ukázkový text pro analýzu. Obsahuje informace o moderních technologiích."

        analysis_result = await document_analysis_tool.ainvoke({
            "text": test_text,
            "analysis_type": "summary"
        })

        print("✅ Analýza dokončena:")
        print(f"   Typ: {analysis_result['analysis_type']}")
        print(f"   Výsledek: {analysis_result['result'][:100]}...")

        return True

    except Exception as e:
        print(f"❌ Chyba nástrojů: {e}")
        return False


async def comprehensive_demo():
    """Kompletní demonstrace všech komponent"""

    print("🎯 Komplexní demonstrace LangGraph Research Agent")
    print("=" * 60)

    # Kontrola environment variables
    required_env_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        print("⚠️  Chybí environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nNastavte je před spuštěním:")
        print("export OPENAI_API_KEY='your-api-key'")
        return

    # Spuštění všech demo částí
    demos = [
        ("RAG Pipeline", demo_rag_pipeline),
        ("Nástroje", demo_tools),
        ("Kompletní Research Agent", demo_langgraph_research)
    ]

    results = {}
    for name, demo_func in demos:
        try:
            print(f"\n{'='*20} {name} {'='*20}")
            result = await demo_func()
            results[name] = result
            print(f"✅ {name} dokončeno úspěšně")
        except Exception as e:
            print(f"❌ {name} selhalo: {e}")
            results[name] = False

    # Souhrn
    print(f"\n{'='*20} SOUHRN {'='*20}")
    for name, success in results.items():
        status = "✅ ÚSPĚCH" if success else "❌ CHYBA"
        print(f"{name}: {status}")


if __name__ == "__main__":
    # Spuštění demonstrace
    asyncio.run(comprehensive_demo())
