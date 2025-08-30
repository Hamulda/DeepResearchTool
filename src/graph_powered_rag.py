"""Graph-Powered RAG System - Fáze 2: Analýza a RAG s podporou grafu
Hybridní RAG systém kombinující textové vyhledávání s grafovými dotazy
"""

import asyncio
from datetime import datetime
import logging
import sys
from typing import Any

import lancedb
from sentence_transformers import SentenceTransformer

# Přidej src do path
sys.path.append("/app/src")

from knowledge_graph import KnowledgeGraphManager
from text_to_cypher import TextToCypherConverter

# Setup logging
logger = logging.getLogger(__name__)


class GraphPoweredRAG:
    """Hybridní RAG systém s podporou znalostního grafu"""

    def __init__(self, vector_db_path: str = "/app/data/vector_db"):
        self.vector_db_path = vector_db_path

        # Inicializuj komponenty
        self.db = lancedb.connect(str(vector_db_path))
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

        try:
            self.kg_manager = KnowledgeGraphManager()
            logger.info("✅ Knowledge Graph Manager připojen")
        except Exception as e:
            logger.error(f"❌ Knowledge Graph nedostupný: {e}")
            self.kg_manager = None

        try:
            self.cypher_converter = TextToCypherConverter()
            logger.info("✅ Text-to-Cypher converter připraven")
        except Exception as e:
            logger.error(f"❌ Text-to-Cypher nedostupný: {e}")
            self.cypher_converter = None

        logger.info("✅ Graph-Powered RAG inicializován")

    async def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4,
        task_id: str = None,
    ) -> dict[str, Any]:
        """Hybridní vyhledávání kombinující vektorové a grafové výsledky

        Args:
            query: Uživatelský dotaz
            limit: Maximální počet výsledků
            vector_weight: Váha vektorových výsledků (0-1)
            graph_weight: Váha grafových výsledků (0-1)
            task_id: ID úlohy pro filtrování

        Returns:
            Kombinované výsledky s kontextem z obou zdrojů

        """
        try:
            logger.info(
                f"🔍 Hybridní vyhledávání: '{query}' (vector: {vector_weight}, graph: {graph_weight})"
            )

            # Paralelní vyhledávání
            vector_task = self._vector_search(query, limit, task_id)
            graph_task = self._graph_search(query, limit)

            vector_results, graph_results = await asyncio.gather(
                vector_task, graph_task, return_exceptions=True
            )

            # Zpracuj výsledky (i v případě chyb)
            if isinstance(vector_results, Exception):
                logger.error(f"❌ Vektorové vyhledávání selhalo: {vector_results}")
                vector_results = {"success": False, "results": []}

            if isinstance(graph_results, Exception):
                logger.error(f"❌ Grafové vyhledávání selhalo: {graph_results}")
                graph_results = {"success": False, "results": [], "graph_context": []}

            # Kombinuj výsledky
            combined_results = self._combine_results(
                vector_results, graph_results, vector_weight, graph_weight, limit
            )

            # Vytvoř finální kontext pro LLM
            llm_context = self._create_llm_context(
                query, combined_results, graph_results.get("graph_context", [])
            )

            return {
                "success": True,
                "query": query,
                "results": combined_results,
                "llm_context": llm_context,
                "sources": {
                    "vector_results": len(vector_results.get("results", [])),
                    "graph_results": len(graph_results.get("results", [])),
                    "graph_context_items": len(graph_results.get("graph_context", [])),
                },
                "search_metadata": {
                    "vector_weight": vector_weight,
                    "graph_weight": graph_weight,
                    "timestamp": datetime.now().isoformat(),
                },
            }

        except Exception as e:
            logger.error(f"❌ Chyba při hybridním vyhledávání: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "fallback_results": await self._fallback_search(query, limit, task_id),
            }

    async def _vector_search(self, query: str, limit: int, task_id: str = None) -> dict[str, Any]:
        """Vektorové vyhledávání v LanceDB"""
        try:
            # Generuj embedding pro query
            query_embedding = self.sentence_model.encode([query])[0].tolist()

            # Najdi relevantní tabulky
            if task_id:
                table_names = [f"rag_documents_{task_id}"]
            else:
                table_names = [
                    name for name in self.db.table_names() if name.startswith("rag_documents_")
                ]

            all_results = []

            for table_name in table_names:
                try:
                    table = self.db.open_table(table_name)
                    results = table.search(query_embedding).limit(limit).to_list()

                    for result in results:
                        result["similarity_score"] = float(result.get("_distance", 0))
                        result["source_type"] = "vector"
                        result["table_source"] = table_name
                        all_results.append(result)

                except Exception as e:
                    logger.warning(f"⚠️ Chyba v tabulce {table_name}: {e}")
                    continue

            # Seřaď podle similarity score
            all_results.sort(key=lambda x: x["similarity_score"], reverse=True)

            return {
                "success": True,
                "results": all_results[:limit],
                "total_found": len(all_results),
            }

        except Exception as e:
            logger.error(f"❌ Vektorové vyhledávání selhalo: {e}")
            return {"success": False, "results": [], "error": str(e)}

    async def _graph_search(self, query: str, limit: int) -> dict[str, Any]:
        """Grafové vyhledávání pomocí Cypher dotazů"""
        try:
            if not self.kg_manager or not self.cypher_converter:
                return {"success": False, "results": [], "graph_context": []}

            # Převeď dotaz na Cypher
            cypher_result = await self.cypher_converter.convert_to_cypher(query)

            if not cypher_result["success"]:
                logger.warning("⚠️ Text-to-Cypher převod selhal")
                return {"success": False, "results": [], "graph_context": []}

            cypher_query = cypher_result["cypher_query"]
            logger.info(f"📊 Spouštím Cypher: {cypher_query[:100]}...")

            # Spusť Cypher dotaz
            graph_results = await self._execute_cypher_query(cypher_query)

            # Získej dodatečný grafový kontext
            graph_context = await self._get_graph_context(query)

            return {
                "success": True,
                "results": graph_results,
                "graph_context": graph_context,
                "cypher_query": cypher_query,
                "cypher_method": cypher_result["method"],
            }

        except Exception as e:
            logger.error(f"❌ Grafové vyhledávání selhalo: {e}")
            return {"success": False, "results": [], "graph_context": [], "error": str(e)}

    async def _execute_cypher_query(self, cypher_query: str) -> list[dict[str, Any]]:
        """Spusť Cypher dotaz a formátuj výsledky"""
        try:
            # Spusť dotaz přes Knowledge Graph Manager
            # Pro obecné Cypher dotazy použijeme Neo4j driver přímo
            with self.kg_manager.driver.session() as session:
                result = session.run(cypher_query)

                formatted_results = []
                for record in result:
                    # Převeď Neo4j record na dict
                    record_dict = {}
                    for key in record.keys():
                        value = record[key]
                        if hasattr(value, "items"):  # Node object
                            record_dict[key] = dict(value)
                        else:
                            record_dict[key] = value

                    record_dict["source_type"] = "graph"
                    formatted_results.append(record_dict)

                return formatted_results

        except Exception as e:
            logger.error(f"❌ Chyba při spouštění Cypher: {e}")
            return []

    async def _get_graph_context(self, query: str) -> list[dict[str, Any]]:
        """Získej dodatečný kontext ze znalostního grafu"""
        try:
            context_items = []

            # Extrahuj klíčová slova z dotazu
            keywords = self._extract_keywords(query)

            for keyword in keywords[:3]:  # Max 3 klíčová slova
                # Najdi entity obsahující klíčové slovo
                entities = await self.kg_manager.query_entities(entity_text=keyword, limit=5)

                for entity in entities:
                    # Získej sousedy této entity
                    neighbors = await self.kg_manager.get_entity_neighbors(
                        entity["text"], max_depth=1, limit=5
                    )

                    context_items.append(
                        {
                            "type": "entity_network",
                            "center_entity": entity["text"],
                            "entity_type": entity["type"],
                            "neighbors": neighbors["neighbors"],
                            "keyword": keyword,
                        }
                    )

            return context_items[:10]  # Omez kontext

        except Exception as e:
            logger.error(f"❌ Chyba při získávání grafového kontextu: {e}")
            return []

    def _extract_keywords(self, query: str) -> list[str]:
        """Extrahuj klíčová slova z dotazu"""
        import re

        # Základní čištění
        query_clean = re.sub(r"[^\w\s]", " ", query.lower())
        words = query_clean.split()

        # Odstraň stop words
        stop_words = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "what",
            "when",
            "where",
            "who",
            "will",
            "with",
            "this",
            "these",
            "they",
            "we",
            "you",
        }

        keywords = [word for word in words if len(word) > 2 and word not in stop_words]

        return keywords[:5]  # Max 5 klíčových slov

    def _combine_results(
        self,
        vector_results: dict[str, Any],
        graph_results: dict[str, Any],
        vector_weight: float,
        graph_weight: float,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Kombinuj výsledky z vektorového a grafového vyhledávání"""
        try:
            combined = []

            # Přidej vektorové výsledky s váhou
            for result in vector_results.get("results", []):
                result_copy = result.copy()
                result_copy["final_score"] = result.get("similarity_score", 0) * vector_weight
                result_copy["score_components"] = {
                    "vector_score": result.get("similarity_score", 0),
                    "graph_score": 0,
                    "vector_weight": vector_weight,
                    "graph_weight": 0,
                }
                combined.append(result_copy)

            # Přidej grafové výsledky s váhou
            for result in graph_results.get("results", []):
                result_copy = result.copy()
                # Pro grafové výsledky použij konstantní skóre
                graph_score = 0.8  # Vysoké skóre pro přesné grafové shody
                result_copy["final_score"] = graph_score * graph_weight
                result_copy["score_components"] = {
                    "vector_score": 0,
                    "graph_score": graph_score,
                    "vector_weight": 0,
                    "graph_weight": graph_weight,
                }
                combined.append(result_copy)

            # Seřaď podle finálního skóre
            combined.sort(key=lambda x: x.get("final_score", 0), reverse=True)

            return combined[:limit]

        except Exception as e:
            logger.error(f"❌ Chyba při kombinování výsledků: {e}")
            return vector_results.get("results", [])[:limit]

    def _create_llm_context(
        self,
        query: str,
        combined_results: list[dict[str, Any]],
        graph_context: list[dict[str, Any]],
    ) -> str:
        """Vytvoř strukturovaný kontext pro LLM"""
        try:
            context_parts = []

            # Přidej uživatelský dotaz
            context_parts.append(f"UŽIVATELSKÝ DOTAZ: {query}\n")

            # Přidej textové výsledky
            context_parts.append("RELEVANTNÍ TEXTOVÉ DOKUMENTY:")
            text_results = [r for r in combined_results if r.get("source_type") == "vector"]

            for i, result in enumerate(text_results[:5], 1):
                chunk_text = result.get("chunk_text", "")[:300]
                url = result.get("url", "N/A")
                score = result.get("final_score", 0)

                context_parts.append(f"{i}. Zdroj: {url}")
                context_parts.append(f"   Skóre: {score:.3f}")
                context_parts.append(f"   Text: {chunk_text}...\n")

            # Přidej grafové výsledky
            graph_results = [r for r in combined_results if r.get("source_type") == "graph"]
            if graph_results:
                context_parts.append("STRUKTUROVANÉ INFORMACE ZE ZNALOSTNÍHO GRAFU:")

                for i, result in enumerate(graph_results[:5], 1):
                    context_parts.append(f"{i}. {self._format_graph_result(result)}")

            # Přidej síťový kontext
            if graph_context:
                context_parts.append("SÍŤOVÝ KONTEXT (entity a jejich spojení):")

                for i, context_item in enumerate(graph_context[:3], 1):
                    if context_item["type"] == "entity_network":
                        center = context_item["center_entity"]
                        neighbors = context_item["neighbors"]

                        context_parts.append(f"{i}. Entita '{center}' je spojena s:")
                        for neighbor in neighbors[:3]:
                            relations = " → ".join(neighbor.get("relations", []))
                            context_parts.append(
                                f"   - {neighbor['entity']} ({neighbor['type']}) via {relations}"
                            )

            return "\n".join(context_parts)

        except Exception as e:
            logger.error(f"❌ Chyba při vytváření LLM kontextu: {e}")
            return f"DOTAZ: {query}\n\nKONTEXT: Chyba při zpracování kontextu."

    def _format_graph_result(self, result: dict[str, Any]) -> str:
        """Formátuj výsledek z grafu pro LLM"""
        try:
            formatted_parts = []

            for key, value in result.items():
                if key in ["source_type", "final_score", "score_components"]:
                    continue

                if isinstance(value, dict):
                    # Neo4j node object
                    if "text" in value and "type" in value:
                        formatted_parts.append(f"{key}: {value['text']} ({value['type']})")
                    else:
                        formatted_parts.append(f"{key}: {str(value)[:100]}")
                else:
                    formatted_parts.append(f"{key}: {value}")

            return " | ".join(formatted_parts)

        except Exception:
            return str(result)[:200]

    async def _fallback_search(
        self, query: str, limit: int, task_id: str = None
    ) -> list[dict[str, Any]]:
        """Fallback vyhledávání při selhání hybridního přístupu"""
        try:
            # Zkus alespoň vektorové vyhledávání
            vector_results = await self._vector_search(query, limit, task_id)
            return vector_results.get("results", [])

        except Exception as e:
            logger.error(f"❌ I fallback vyhledávání selhalo: {e}")
            return []

    async def get_graph_statistics(self) -> dict[str, Any]:
        """Získej statistiky znalostního grafu"""
        try:
            if not self.kg_manager:
                return {"error": "Knowledge Graph nedostupný"}

            return await self.kg_manager.get_graph_statistics()

        except Exception as e:
            logger.error(f"❌ Chyba při získávání statistik: {e}")
            return {"error": str(e)}

    def get_available_tables(self) -> list[str]:
        """Získej seznam dostupných RAG tabulek"""
        try:
            return [name for name in self.db.table_names() if name.startswith("rag_documents_")]
        except Exception as e:
            logger.error(f"❌ Chyba při získávání tabulek: {e}")
            return []
