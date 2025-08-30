"""
Demo Script pro Fázi 1: Jádro Znalostního Grafu
Demonstruje kompletní funkcionalnost Knowledge Graph systému
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

from knowledge_graph import KnowledgeGraphManager
from llm_relationship_extractor import LLMRelationshipExtractor
from processing_worker import EnhancedProcessingWorker


async def demo_knowledge_graph_phase1():
    """Kompletní demo Fáze 1 - Knowledge Graph Core"""

    print("🚀 === DEMO: Fáze 1 - Jádro Znalostního Grafu ===")
    print()

    # 1. Inicializace komponent
    print("📦 1. Inicializace komponent...")

    try:
        kg_manager = KnowledgeGraphManager()
        print("✅ Knowledge Graph Manager připojen k Neo4j")
    except Exception as e:
        print(f"❌ Neo4j není dostupný: {e}")
        return

    try:
        relation_extractor = LLMRelationshipExtractor()
        print("✅ LLM Relationship Extractor inicializován")
    except Exception as e:
        print(f"⚠️ LLM není dostupný, použije se fallback: {e}")
        relation_extractor = None

    worker = EnhancedProcessingWorker()
    print("✅ Enhanced Processing Worker připraven")
    print()

    # 2. Příprava testovacích dat
    print("📝 2. Příprava testovacích dat...")

    sample_html = """
    <html>
    <head><title>Dark Web Crypto Discussion</title></head>
    <body>
        <h1>Bitcoin Trading Forum</h1>
        <div class="post">
            <h3>User: CryptoKing</h3>
            <p>Just completed a transaction - sent 2.5 BTC to address 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa</p>
            <p>DarkMarket marketplace on domain darkmarket7rr726xi.onion is reliable for purchases.</p>
        </div>
        <div class="post">
            <h3>User: AliceTrader</h3>
            <p>@CryptoKing thanks for the tip! Planning to meet Bob in Prague for crypto exchange.</p>
            <p>TorProject organization provides excellent privacy tools.</p>
        </div>
        <div class="post">
            <h3>User: BitcoinExpert</h3>
            <p>Remember to verify hash 5d41402abc4b2a76b9719d911017c592 before any transaction.</p>
            <p>Contact me via PGP key A1B2C3D4E5F6789012345678901234567890ABCD</p>
        </div>
    </body>
    </html>
    """

    # Vytvoř dočasný Parquet soubor
    data = {
        "url": ["http://cryptoforum.onion/trading-discussion"],
        "content": [sample_html],
        "metadata": [
            json.dumps(
                {
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                    "status_code": 200,
                    "content_type": "text/html",
                    "forum": "crypto_trading",
                }
            )
        ],
    }

    df = pl.DataFrame(data)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".parquet", delete=False) as f:
        temp_file = f.name

    df.write_parquet(temp_file)
    print(f"✅ Testovací data připravena: {temp_file}")
    print()

    # 3. Zpracování s Knowledge Graph
    print("🧠 3. Zpracování dat s Knowledge Graph...")

    task_id = f"demo_kg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        result = await worker.process_with_knowledge_graph(temp_file, task_id)

        if result["success"]:
            print(f"✅ Zpracování úspěšné!")
            print(f"   📊 Zpracováno záznamů: {result['records_processed']}")
            print(f"   📊 Celkem chunků: {result['total_chunks']}")
            print(f"   📊 Celkem entit: {result['total_entities']}")
            print(f"   📊 Celkem vztahů: {result['total_relations']}")

            kg_stats = result.get("knowledge_graph_stats", {})
            print(f"   📊 KG entit přidáno: {kg_stats.get('total_entities_added', 0)}")
            print(f"   📊 KG vztahů přidáno: {kg_stats.get('total_relations_added', 0)}")
            print(f"   📊 Zdrojů zpracováno: {kg_stats.get('sources_processed', 0)}")
        else:
            print(f"❌ Zpracování selhalo: {result.get('error', 'Neznámá chyba')}")
            return

    except Exception as e:
        print(f"❌ Chyba při zpracování: {e}")
        return

    print()

    # 4. Dotazy na Knowledge Graph
    print("🔍 4. Testování dotazů na Knowledge Graph...")

    # Statistiky grafu
    stats = await kg_manager.get_graph_statistics()
    print(f"📈 Statistiky grafu:")
    print(f"   • Celkem entit: {stats.get('total_entities', 0)}")
    print(f"   • Celkem vztahů: {stats.get('total_relations', 0)}")
    print(f"   • Celkem zdrojů: {stats.get('total_sources', 0)}")

    # Typy entit
    entity_types = stats.get("entity_types", [])
    if entity_types:
        print(f"   • Typy entit:")
        for et in entity_types[:5]:  # Top 5
            print(f"     - {et['type']}: {et['count']}")

    print()

    # Vyhledávání konkrétních entit
    print("🔎 Vyhledávání entit:")

    # Hledej CryptoKing
    cryptoking_entities = await kg_manager.query_entities(entity_text="CryptoKing", limit=10)
    print(f"   • 'CryptoKing' nalezen {len(cryptoking_entities)} krát")
    for entity in cryptoking_entities[:3]:
        print(f"     - {entity['text']} ({entity['type']})")

    # Hledej Bitcoin adresy
    crypto_entities = await kg_manager.query_entities(entity_type="crypto_addresses", limit=5)
    print(f"   • Krypto adresy: {len(crypto_entities)} nalezeno")
    for entity in crypto_entities[:3]:
        print(f"     - {entity['text']}")

    print()

    # Dotazy na vztahy
    print("🔗 Vyhledávání vztahů:")

    # Vztahy CryptoKing
    cryptoking_relations = await kg_manager.query_relations(subject="CryptoKing", limit=10)
    print(f"   • CryptoKing má {len(cryptoking_relations)} vztahů:")
    for rel in cryptoking_relations[:5]:
        print(f"     - {rel['subject']} → {rel['predicate']} → {rel['object']}")

    # Všechny SENT_BTC_TO vztahy
    btc_relations = await kg_manager.query_relations(predicate="SENT_BTC_TO", limit=10)
    print(f"   • Bitcoin transakce: {len(btc_relations)} nalezeno")
    for rel in btc_relations[:3]:
        print(f"     - {rel['subject']} → {rel['object']}")

    print()

    # Analýza sousedů
    print("🕸️ Analýza sítě - sousedé entit:")

    if cryptoking_entities:
        neighbors = await kg_manager.get_entity_neighbors("CryptoKing", max_depth=2, limit=10)
        print(f"   • CryptoKing má {len(neighbors['neighbors'])} sousedů:")
        for neighbor in neighbors["neighbors"][:5]:
            relations_str = " → ".join(neighbor["relations"])
            print(f"     - {neighbor['entity']} ({neighbor['type']}) via {relations_str}")

    print()

    # 5. Ukázka Graph-powered vyhledávání
    print("🔍 5. Testování Graph-powered vyhledávání...")

    # RAG vyhledávání
    rag_results = worker.search_rag_documents(
        query="bitcoin transaction cryptocurrency", limit=3, task_id=task_id
    )

    if rag_results["success"]:
        print(f"✅ RAG našel {len(rag_results['results'])} relevantních dokumentů:")
        for i, result in enumerate(rag_results["results"][:3], 1):
            print(f"   {i}. {result.get('url', 'N/A')}")
            print(f"      Podobnost: {result.get('similarity_score', 0):.3f}")
            chunk_text = result.get("chunk_text", "")[:100]
            print(f"      Text: {chunk_text}...")
    else:
        print(f"⚠️ RAG vyhledávání neúspěšné: {rag_results.get('error', 'Neznámá chyba')}")

    print()

    # 6. Ukázka komplexních dotazů
    print("🎯 6. Komplexní síťové dotazy:")

    # Najdi všechny entity spojené s .onion doménami
    onion_entities = await kg_manager.query_entities(entity_type="onion_addresses", limit=5)
    if onion_entities:
        print(f"   • Nalezeno {len(onion_entities)} .onion domén")
        for onion in onion_entities[:3]:
            # Najdi co je s touto doménou spojeno
            domain_neighbors = await kg_manager.get_entity_neighbors(
                onion["text"], max_depth=1, limit=5
            )
            print(
                f"     - {onion['text']} je spojena s {len(domain_neighbors['neighbors'])} entitami"
            )

    # Najdi nejaktivnější uživatele (nejvíce vztahů)
    all_relations = await kg_manager.query_relations(limit=100)
    if all_relations:
        # Spočti vztahy podle subjektu
        subject_counts = {}
        for rel in all_relations:
            subject = rel["subject"]
            subject_counts[subject] = subject_counts.get(subject, 0) + 1

        top_subjects = sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"   • Nejaktivnější entity podle počtu vztahů:")
        for subject, count in top_subjects[:5]:
            print(f"     - {subject}: {count} vztahů")

    print()

    # 7. Závěr
    print("✅ === DEMO DOKONČENO ===")
    print()
    print("🎉 Fáze 1: Jádro Znalostního Grafu bylo úspěšně implementováno!")
    print()
    print("📋 Implementované funkce:")
    print("   ✅ Neo4j grafová databáze s optimalizovaným schématem")
    print("   ✅ Extrakce entit pomocí spaCy s custom patterns")
    print("   ✅ LLM extrakce vztahů s JSON výstupem")
    print("   ✅ Automatické ukládání do znalostního grafu")
    print("   ✅ Pokročilé dotazy a síťové analýzy")
    print("   ✅ Integrace s RAG systémem")
    print("   ✅ Heuristické fallback mechanismy")
    print()
    print("🔗 Graf nyní obsahuje:")
    print(f"   • {stats.get('total_entities', 0)} entit různých typů")
    print(f"   • {stats.get('total_relations', 0)} vztahů mezi entitami")
    print(f"   • {stats.get('total_sources', 0)} zdrojových dokumentů")
    print()
    print("🚀 Systém je připraven pro Fázi 2: Graph-Powered RAG!")

    # Vyčisti
    try:
        os.unlink(temp_file)
    except:
        pass


if __name__ == "__main__":
    # Nastav environment proměnné
    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_USER", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "research2024")
    os.environ.setdefault("OLLAMA_HOST", "localhost")
    os.environ.setdefault("LLM_MODEL", "llama2")

    # Spusť demo
    asyncio.run(demo_knowledge_graph_phase1())
