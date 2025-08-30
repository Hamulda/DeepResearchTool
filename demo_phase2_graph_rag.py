"""
Demo Script pro Fázi 2: Graph-Powered RAG
Demonstruje hybridní RAG systém kombinující textové a grafové vyhledávání
"""

import asyncio
import json
import tempfile
import polars as pl
from datetime import datetime, timezone
from pathlib import Path
import sys
import os

# Přidej src do path
sys.path.append("/app/src")
sys.path.append("/app/workers")

from graph_powered_rag import GraphPoweredRAG
from knowledge_graph import KnowledgeGraphManager
from processing_worker import EnhancedProcessingWorker


async def demo_graph_powered_rag():
    """Kompletní demo Fáze 2 - Graph-Powered RAG"""

    print("🚀 === DEMO: Fáze 2 - Graph-Powered RAG ===")
    print()

    # 1. Inicializace Graph-Powered RAG systému
    print("📦 1. Inicializace Graph-Powered RAG systému...")

    try:
        rag_system = GraphPoweredRAG()
        print("✅ Graph-Powered RAG systém inicializován")
    except Exception as e:
        print(f"❌ Chyba při inicializaci RAG: {e}")
        return

    # Ověř dostupnost komponent
    available_tables = rag_system.get_available_tables()
    print(f"✅ Dostupné RAG tabulky: {len(available_tables)}")

    graph_stats = await rag_system.get_graph_statistics()
    if "error" not in graph_stats:
        print(f"✅ Knowledge Graph obsahuje {graph_stats.get('total_entities', 0)} entit")
    else:
        print(f"⚠️ Knowledge Graph: {graph_stats['error']}")

    print()

    # 2. Příprava testovacích dat (pokud nejsou k dispozici)
    print("📝 2. Příprava testovacích dat pro demo...")

    if not available_tables:
        print("📊 Vytváření vzorových dat...")
        await create_sample_data()

        # Aktualizuj dostupné tabulky
        available_tables = rag_system.get_available_tables()
        print(f"✅ Vytvořeno {len(available_tables)} RAG tabulek")

    print()

    # 3. Testování jednotlivých komponent
    print("🧪 3. Testování jednotlivých komponent...")

    # Test Text-to-Cypher
    if rag_system.cypher_converter:
        print("🔄 Test Text-to-Cypher převodu:")

        test_queries = [
            "Najdi všechny osoby",
            "Kdo poslal Bitcoin?",
            "Jaké .onion domény jsou v grafu?",
            "Co je spojeno s CryptoKing?",
        ]

        for query in test_queries:
            try:
                cypher_result = await rag_system.cypher_converter.convert_to_cypher(query)
                if cypher_result["success"]:
                    print(
                        f"   ✅ '{query}' → Cypher úspěšně vygenerován ({cypher_result['method']})"
                    )
                else:
                    print(f"   ⚠️ '{query}' → Fallback použit")
            except Exception as e:
                print(f"   ❌ '{query}' → Chyba: {e}")
    else:
        print("⚠️ Text-to-Cypher converter není dostupný")

    print()

    # 4. Hybridní vyhledávání - různé dotazy
    print("🔍 4. Testování hybridního vyhledávání...")

    test_searches = [
        {
            "query": "bitcoin transaction cryptocurrency",
            "description": "Obecný dotaz o kryptoměnách",
            "vector_weight": 0.7,
            "graph_weight": 0.3,
        },
        {
            "query": "CryptoKing trading activities",
            "description": "Specifický dotaz o uživateli",
            "vector_weight": 0.4,
            "graph_weight": 0.6,
        },
        {
            "query": "dark web marketplace connections",
            "description": "Síťový dotaz o spojeních",
            "vector_weight": 0.3,
            "graph_weight": 0.7,
        },
        {
            "query": "privacy tools and organizations",
            "description": "Tematický dotaz",
            "vector_weight": 0.6,
            "graph_weight": 0.4,
        },
    ]

    for i, search in enumerate(test_searches, 1):
        print(f"🔎 {i}. {search['description']}")
        print(f"   Dotaz: '{search['query']}'")
        print(f"   Váhy: Vector {search['vector_weight']}, Graph {search['graph_weight']}")

        try:
            result = await rag_system.hybrid_search(
                query=search["query"],
                limit=5,
                vector_weight=search["vector_weight"],
                graph_weight=search["graph_weight"],
            )

            if result["success"]:
                sources = result["sources"]
                print(
                    f"   ✅ Nalezeno: {sources['vector_results']} textových + {sources['graph_results']} grafových výsledků"
                )
                print(f"   📊 Grafový kontext: {sources['graph_context_items']} položek")

                # Ukázka nejlepších výsledků
                top_results = result["results"][:3]
                for j, res in enumerate(top_results, 1):
                    source_type = res.get("source_type", "unknown")
                    score = res.get("final_score", 0)
                    print(f"      {j}. {source_type.upper()}: skóre {score:.3f}")

                    if source_type == "vector":
                        url = res.get("url", "N/A")
                        print(f"         URL: {url}")
                    elif source_type == "graph":
                        # Zobraz grafové informace
                        graph_info = []
                        for key, value in res.items():
                            if key not in ["source_type", "final_score", "score_components"]:
                                if isinstance(value, dict) and "text" in value:
                                    graph_info.append(f"{key}: {value['text']}")
                                elif not isinstance(value, dict):
                                    graph_info.append(f"{key}: {value}")
                        print(f"         Graf: {' | '.join(graph_info[:2])}")

            else:
                print(f"   ❌ Vyhledávání selhalo: {result.get('error', 'Neznámá chyba')}")

        except Exception as e:
            print(f"   ❌ Chyba při vyhledávání: {e}")

        print()

    # 5. Ukázka LLM kontextu
    print("📖 5. Ukázka LLM kontextu pro komplexní dotaz...")

    complex_query = "How are CryptoKing and DarkMarket connected through bitcoin transactions?"

    try:
        complex_result = await rag_system.hybrid_search(
            query=complex_query, limit=8, vector_weight=0.4, graph_weight=0.6
        )

        if complex_result["success"]:
            llm_context = complex_result["llm_context"]

            print(f"✅ LLM kontext vygenerován ({len(llm_context)} znaků)")
            print("📝 Ukázka kontextu:")
            print("=" * 60)

            # Zobraz první část kontextu
            context_preview = llm_context[:800]
            print(context_preview)
            if len(llm_context) > 800:
                print("... (zkráceno)")

            print("=" * 60)

            print(f"📊 Kontext obsahuje:")
            context_lines = llm_context.split("\n")
            doc_count = len(
                [
                    line
                    for line in context_lines
                    if line.startswith("1.") or line.startswith("2.") or line.startswith("3.")
                ]
            )
            graph_lines = len(
                [line for line in context_lines if "Graf:" in line or "entita" in line.lower()]
            )

            print(f"   • {doc_count} textových dokumentů")
            print(f"   • {graph_lines} grafových informací")
            print(f"   • Celkem {len(context_lines)} řádků kontextu")

        else:
            print(f"❌ Komplexní dotaz selhal: {complex_result.get('error')}")

    except Exception as e:
        print(f"❌ Chyba při komplexním dotazu: {e}")

    print()

    # 6. Porovnání s čistě vektorovým RAG
    print("⚖️ 6. Porovnání Graph-RAG vs čistý Vector-RAG...")

    comparison_query = "bitcoin cryptocurrency transactions"

    try:
        # Čistý vektorový RAG
        vector_only = await rag_system.hybrid_search(
            query=comparison_query, vector_weight=1.0, graph_weight=0.0, limit=5
        )

        # Hybridní Graph-RAG
        hybrid_rag = await rag_system.hybrid_search(
            query=comparison_query, vector_weight=0.5, graph_weight=0.5, limit=5
        )

        print(f"📊 Porovnání pro dotaz: '{comparison_query}'")
        print()

        if vector_only["success"]:
            print("🔤 Čistý Vector-RAG:")
            vector_sources = vector_only["sources"]
            print(f"   • {vector_sources['vector_results']} textových výsledků")
            print(f"   • {vector_sources['graph_results']} grafových výsledků")
            print(f"   • {vector_sources['graph_context_items']} kontextových položek")

        if hybrid_rag["success"]:
            print("🌐 Hybridní Graph-RAG:")
            hybrid_sources = hybrid_rag["sources"]
            print(f"   • {hybrid_sources['vector_results']} textových výsledků")
            print(f"   • {hybrid_sources['graph_results']} grafových výsledků")
            print(f"   • {hybrid_sources['graph_context_items']} kontextových položek")

        print()
        print("💡 Výhody Graph-RAG:")
        print("   ✅ Strukturované informace o vztazích")
        print("   ✅ Síťový kontext pro hlubší analýzu")
        print("   ✅ Kombinace textového a relačního obsahu")
        print("   ✅ Možnost komplexních síťových dotazů")

    except Exception as e:
        print(f"❌ Chyba při porovnání: {e}")

    print()

    # 7. Statistiky a metriky
    print("📈 7. Finální statistiky a metriky...")

    # Graph statistiky
    try:
        final_stats = await rag_system.get_graph_statistics()
        if "error" not in final_stats:
            print("📊 Knowledge Graph statistiky:")
            print(f"   • Celkem entit: {final_stats.get('total_entities', 0)}")
            print(f"   • Celkem vztahů: {final_stats.get('total_relations', 0)}")
            print(f"   • Celkem zdrojů: {final_stats.get('total_sources', 0)}")

            entity_types = final_stats.get("entity_types", [])
            if entity_types:
                print("   • Top typy entit:")
                for et in entity_types[:5]:
                    print(f"     - {et['type']}: {et['count']}")
    except Exception as e:
        print(f"❌ Chyba při získávání statistik: {e}")

    # RAG tabulky
    final_tables = rag_system.get_available_tables()
    print(f"📚 RAG databáze: {len(final_tables)} tabulek")

    print()

    # 8. Závěr
    print("✅ === DEMO DOKONČENO ===")
    print()
    print("🎉 Fáze 2: Graph-Powered RAG byla úspěšně implementována!")
    print()
    print("📋 Implementované funkce:")
    print("   ✅ Text-to-Cypher konvertor s LLM a fallback")
    print("   ✅ Hybridní vyhledávání (Vector + Graph)")
    print("   ✅ Paralelní zpracování dotazů")
    print("   ✅ Kombinování výsledků s váženým skórováním")
    print("   ✅ Strukturovaný LLM kontext")
    print("   ✅ Síťový kontext ze znalostního grafu")
    print("   ✅ Robustní error handling a fallback")
    print()
    print("🔄 Workflow Graph-RAG:")
    print("   1. Paralelní vektorové + grafové vyhledávání")
    print("   2. Text-to-Cypher převod uživatelského dotazu")
    print("   3. Spuštění Cypher dotazů na Neo4j")
    print("   4. Získání síťového kontextu")
    print("   5. Kombinování a skórování výsledků")
    print("   6. Vytvoření strukturovaného kontextu pro LLM")
    print()
    print("🚀 Systém je připraven pro Fázi 3: Multi-Modality!")


async def create_sample_data():
    """Vytvoř vzorová data pokud nejsou k dispozici"""
    try:
        worker = EnhancedProcessingWorker()

        # Vzorový HTML obsah
        sample_html = """
        <html>
        <head><title>Crypto Trading Forum</title></head>
        <body>
            <h1>Advanced Bitcoin Trading Discussion</h1>
            <div class="post">
                <h3>User: CryptoKing</h3>
                <p>Successfully sent 3.2 BTC to address 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa yesterday.</p>
                <p>DarkMarket on darkmarket7rr726xi.onion has the best rates for anonymous trading.</p>
            </div>
            <div class="post">
                <h3>User: BitcoinExpert</h3>
                <p>@CryptoKing good choice! That marketplace is operated by TorProject veterans.</p>
                <p>Always verify hash 5d41402abc4b2a76b9719d911017c592 before transactions.</p>
            </div>
        </body>
        </html>
        """

        # Vytvoř dočasný soubor
        data = {
            "url": ["http://cryptoforum.onion/advanced-trading"],
            "content": [sample_html],
            "metadata": [
                json.dumps(
                    {
                        "scraped_at": datetime.now(timezone.utc).isoformat(),
                        "status_code": 200,
                        "content_type": "text/html",
                    }
                )
            ],
        }

        df = pl.DataFrame(data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".parquet", delete=False) as f:
            temp_file = f.name

        df.write_parquet(temp_file)

        # Zpracuj s Knowledge Graph
        task_id = f"demo_phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        result = await worker.process_with_knowledge_graph(temp_file, task_id)

        if result["success"]:
            print(f"✅ Vzorová data vytvořena a zpracována")
        else:
            print(f"⚠️ Vzorová data vytvořena, ale zpracování selhalo")

        # Vyčisti
        try:
            os.unlink(temp_file)
        except:
            pass

    except Exception as e:
        print(f"❌ Chyba při vytváření vzorových dat: {e}")


if __name__ == "__main__":
    # Nastav environment proměnné
    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_USER", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "research2024")
    os.environ.setdefault("OLLAMA_HOST", "localhost")
    os.environ.setdefault("LLM_MODEL", "llama2")

    # Spusť demo
    asyncio.run(demo_graph_powered_rag())
