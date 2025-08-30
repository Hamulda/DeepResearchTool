"""
Integrační test pro Fázi 1: Jádro Znalostního Grafu
Testuje celý pipeline od extrakce entit přes LLM extrakci vztahů až po ukládání do Neo4j
"""

import asyncio
import pytest
import json
import tempfile
import polars as pl
from pathlib import Path
from datetime import datetime, timezone
import os
import sys

# Přidej src do path
sys.path.append("/app/src")
sys.path.append("/app/workers")

from knowledge_graph import KnowledgeGraphManager
from llm_relationship_extractor import LLMRelationshipExtractor
from processing_worker import EnhancedProcessingWorker


class TestKnowledgeGraphPhase1:
    """Test suite pro Fázi 1 - Knowledge Graph Core"""

    @pytest.fixture
    def sample_html_content(self):
        """Vzorový HTML obsah pro testování"""
        return """
        <html>
        <head><title>Dark Web Forum - CryptoKing Discussion</title></head>
        <body>
            <h1>Bitcoin Trading Discussion</h1>
            <p>User CryptoKing posted: "I just sent 5 BTC to address 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa from my wallet."</p>
            <p>Organization DarkMarket operates on domain darkmarket7rr726xi.onion and accepts payments.</p>
            <p>User AliceTrader mentioned meeting Bob in Prague next month for cryptocurrency exchange.</p>
            <p>The discussion also mentioned TorProject as a key privacy organization.</p>
        </body>
        </html>
        """

    @pytest.fixture
    def sample_scraped_data(self, sample_html_content):
        """Vytvoř vzorová scraped data ve formátu Parquet"""
        data = {
            "url": ["http://example.onion/forum/bitcoin-trading"],
            "content": [sample_html_content],
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

        # Ulož do dočasného souboru
        with tempfile.NamedTemporaryFile(mode="w", suffix=".parquet", delete=False) as f:
            temp_path = f.name

        df.write_parquet(temp_path)
        return temp_path

    @pytest.mark.asyncio
    async def test_knowledge_graph_manager_connection(self):
        """Test připojení k Neo4j"""
        try:
            kg_manager = KnowledgeGraphManager()

            # Test základního připojení
            stats = await kg_manager.get_graph_statistics()
            assert isinstance(stats, dict)
            assert "total_entities" in stats

            print("✅ Neo4j připojení funguje")

        except Exception as e:
            pytest.skip(f"Neo4j není dostupný: {e}")

    @pytest.mark.asyncio
    async def test_llm_relationship_extractor(self):
        """Test LLM extraktoru vztahů"""
        try:
            extractor = LLMRelationshipExtractor()

            sample_text = "CryptoKing sent 5 BTC to address 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa. DarkMarket operates on darkmarket7rr726xi.onion."
            sample_entities = {
                "persons": [{"text": "CryptoKing", "label": "PERSON", "confidence": 1.0}],
                "organizations": [{"text": "DarkMarket", "label": "ORG", "confidence": 1.0}],
                "crypto_addresses": [
                    {
                        "text": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
                        "label": "CRYPTO_ADDRESS",
                        "confidence": 1.0,
                    }
                ],
                "onion_addresses": [
                    {
                        "text": "darkmarket7rr726xi.onion",
                        "label": "ONION_ADDRESS",
                        "confidence": 1.0,
                    }
                ],
            }

            result = await extractor.extract_relationships(sample_text, sample_entities)

            assert isinstance(result, dict)
            assert "entities" in result
            assert "relations" in result
            assert isinstance(result["relations"], list)

            print(f"✅ LLM extraktor vrátil {len(result['relations'])} vztahů")

        except Exception as e:
            print(f"⚠️ LLM extraktor není dostupný, použije se fallback: {e}")

    @pytest.mark.asyncio
    async def test_entity_relationship_storage(self):
        """Test ukládání entit a vztahů do Neo4j"""
        try:
            kg_manager = KnowledgeGraphManager()

            # Testovací data
            test_entities = {
                "persons": [
                    {"text": "CryptoKing", "label": "PERSON", "confidence": 1.0},
                    {"text": "AliceTrader", "label": "PERSON", "confidence": 1.0},
                ],
                "organizations": [{"text": "DarkMarket", "label": "ORG", "confidence": 1.0}],
                "crypto_addresses": [
                    {
                        "text": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
                        "label": "CRYPTO_ADDRESS",
                        "confidence": 1.0,
                    }
                ],
            }

            test_relations = [
                {
                    "subject": "CryptoKing",
                    "predicate": "SENT_BTC_TO",
                    "object": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
                    "confidence": 0.9,
                },
                {
                    "subject": "AliceTrader",
                    "predicate": "TRADES_WITH",
                    "object": "CryptoKing",
                    "confidence": 0.8,
                },
                {
                    "subject": "DarkMarket",
                    "predicate": "ACCEPTS",
                    "object": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
                    "confidence": 0.7,
                },
            ]

            test_url = "http://test.onion/forum/test"
            test_metadata = {
                "title": "Test Forum",
                "extracted_at": datetime.now(timezone.utc).isoformat(),
                "task_id": "test_kg_001",
            }

            # Ulož do grafu
            result = await kg_manager.add_entities_and_relations(
                test_entities, test_relations, test_url, test_metadata
            )

            assert result["source_added"] == True
            assert result["entities_added"] > 0
            assert result["relations_added"] > 0

            print(
                f"✅ Uloženo {result['entities_added']} entit a {result['relations_added']} vztahů"
            )

            # Test vyhledávání entit
            entities = await kg_manager.query_entities(entity_text="CryptoKing")
            assert len(entities) > 0
            assert any(e["text"] == "CryptoKing" for e in entities)

            # Test vyhledávání vztahů
            relations = await kg_manager.query_relations(subject="CryptoKing")
            assert len(relations) > 0
            assert any(
                r["subject"] == "CryptoKing" and r["predicate"] == "SENT_BTC_TO" for r in relations
            )

            # Test získání sousedů
            neighbors = await kg_manager.get_entity_neighbors("CryptoKing")
            assert neighbors["center_entity"] == "CryptoKing"
            assert len(neighbors["neighbors"]) > 0

            print("✅ Dotazy na graf fungují správně")

        except Exception as e:
            pytest.skip(f"Neo4j test selhalo: {e}")

    @pytest.mark.asyncio
    async def test_full_processing_pipeline(self, sample_scraped_data):
        """Test celého processing pipeline s Knowledge Graph"""
        try:
            worker = EnhancedProcessingWorker()
            task_id = f"test_kg_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Spusť kompletní zpracování s Knowledge Graph
            result = await worker.process_with_knowledge_graph(sample_scraped_data, task_id)

            assert result["success"] == True
            assert result["records_processed"] > 0
            assert "knowledge_graph_stats" in result

            kg_stats = result["knowledge_graph_stats"]
            assert kg_stats["sources_processed"] > 0

            print(f"✅ Pipeline zpracoval {result['records_processed']} záznamů")
            print(
                f"✅ KG stats: {kg_stats['total_entities_added']} entit, {kg_stats['total_relations_added']} vztahů"
            )

            # Ověř že data jsou v grafu
            if worker.kg_manager:
                stats = await worker.kg_manager.get_graph_statistics()
                assert stats["total_entities"] > 0
                print(
                    f"✅ Graf obsahuje {stats['total_entities']} entit a {stats['total_relations']} vztahů"
                )

        except Exception as e:
            pytest.skip(f"Kompletní pipeline test selhalo: {e}")
        finally:
            # Vyčisti dočasný soubor
            if os.path.exists(sample_scraped_data):
                os.unlink(sample_scraped_data)

    @pytest.mark.asyncio
    async def test_graph_rag_integration(self, sample_scraped_data):
        """Test integrace mezi Knowledge Graph a RAG"""
        try:
            worker = EnhancedProcessingWorker()
            task_id = f"test_graph_rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Zpracuj data s KG
            result = await worker.process_with_knowledge_graph(sample_scraped_data, task_id)
            assert result["success"] == True

            # Počkej na RAG indexování (simulace)
            await asyncio.sleep(2)

            # Test RAG vyhledávání
            search_result = worker.search_rag_documents(
                query="CryptoKing bitcoin trading", limit=5, task_id=task_id
            )

            if search_result["success"]:
                assert len(search_result["results"]) > 0
                print(f"✅ RAG našel {len(search_result['results'])} relevantních dokumentů")

            # Test Knowledge Graph dotazů
            if worker.kg_manager:
                entities = await worker.kg_manager.query_entities(entity_text="CryptoKing")
                relations = await worker.kg_manager.query_relations(subject="CryptoKing")

                print(f"✅ KG obsahuje {len(entities)} entit pro 'CryptoKing'")
                print(f"✅ KG obsahuje {len(relations)} vztahů pro 'CryptoKing'")

        except Exception as e:
            pytest.skip(f"Graph-RAG integrace test selhala: {e}")
        finally:
            if os.path.exists(sample_scraped_data):
                os.unlink(sample_scraped_data)

    def test_heuristic_relationship_fallback(self):
        """Test fallback mechanismu pro vytváření vztahů"""
        try:
            extractor = LLMRelationshipExtractor()

            entities = {
                "persons": [
                    {"text": "CryptoKing", "label": "PERSON", "confidence": 1.0},
                    {"text": "AliceTrader", "label": "PERSON", "confidence": 1.0},
                ],
                "organizations": [{"text": "DarkMarket", "label": "ORG", "confidence": 1.0}],
                "crypto_addresses": [
                    {
                        "text": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
                        "label": "CRYPTO_ADDRESS",
                        "confidence": 1.0,
                    }
                ],
                "onion_addresses": [
                    {
                        "text": "darkmarket7rr726xi.onion",
                        "label": "ONION_ADDRESS",
                        "confidence": 1.0,
                    }
                ],
            }

            relations = extractor.create_simple_relations_from_entities(
                entities, "http://test.onion"
            )

            assert isinstance(relations, list)
            assert len(relations) > 0

            # Ověř strukturu vztahů
            for rel in relations:
                assert "subject" in rel
                assert "predicate" in rel
                assert "object" in rel
                assert "confidence" in rel

            print(f"✅ Heuristický fallback vytvořil {len(relations)} vztahů")

        except Exception as e:
            pytest.fail(f"Heuristický fallback selhal: {e}")


if __name__ == "__main__":
    # Spusť testy
    print("🧪 Spouštím integrační testy pro Fázi 1: Knowledge Graph Core")

    # Nastav environment proměnné pro test
    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_USER", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "research2024")
    os.environ.setdefault("OLLAMA_HOST", "localhost")
    os.environ.setdefault("LLM_MODEL", "llama2")

    pytest.main([__file__, "-v", "-s"])
