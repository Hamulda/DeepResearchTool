"""
Text-to-Cypher Converter - Fáze 2: Graph-Powered RAG
Převádí dotazy v přirozeném jazyce na Cypher dotazy pro Neo4j
"""

import logging
import os
import json
import re
from typing import Dict, Any, List, Optional, Tuple
import aiohttp
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)


class TextToCypherConverter:
    """Převodník z přirozeného jazyka na Cypher dotazy"""

    def __init__(self):
        self.ollama_host = os.getenv("OLLAMA_HOST", "localhost")
        self.ollama_port = os.getenv("OLLAMA_PORT", "11434")
        self.ollama_url = f"http://{self.ollama_host}:{self.ollama_port}"
        self.model_name = os.getenv("LLM_MODEL", "llama2")

        # Schéma znalostního grafu
        self.graph_schema = self._create_graph_schema()

        # Template pro Text-to-Cypher
        self.cypher_prompt = self._create_cypher_prompt()

    def _create_graph_schema(self) -> Dict[str, Any]:
        """Definice schématu znalostního grafu"""
        return {
            "node_types": [
                "Entity",  # Základní entity (osoby, organizace, lokace, atd.)
                "Source",  # Zdrojové dokumenty
            ],
            "entity_types": [
                "PERSON",  # Osoby (CryptoKing, AliceTrader)
                "ORG",  # Organizace (DarkMarket, TorProject)
                "LOCATION",  # Lokace (Prague, London)
                "CRYPTO",  # Krypto adresy
                "ONION",  # .onion domény
                "HASH",  # Hash hodnoty
                "PGP_KEY",  # PGP klíče
                "OTHER",  # Ostatní entity
            ],
            "relationship_types": [
                "RELATES_TO",  # Obecný vztah mezi entitami
                "FOUND_IN",  # Entita nalezena ve zdroji
                "SENT_BTC_TO",  # Bitcoin transakce
                "POSTED_ON",  # Publikováno na
                "MEMBER_OF",  # Člen organizace
                "LOCATED_IN",  # Nachází se v
                "CONNECTED_TO",  # Spojen s
                "CREATED",  # Vytvořil
                "OPERATES",  # Provozuje
                "MENTIONED_BY",  # Zmíněn kým
                "ASSOCIATED_WITH",  # Spojen s
                "OWNS",  # Vlastní
                "TRADES_WITH",  # Obchoduje s
            ],
            "entity_properties": [
                "id",  # Jedinečné ID entity
                "text",  # Textová reprezentace
                "type",  # Typ entity
                "confidence",  # Confidence skóre
                "last_seen",  # Poslední výskyt
            ],
            "source_properties": [
                "url",  # URL zdroje
                "title",  # Název
                "timestamp",  # Časové razítko
                "metadata",  # Metadata jako JSON
            ],
        }

    def _create_cypher_prompt(self) -> str:
        """Vytvoř prompt template pro Text-to-Cypher"""
        schema_info = json.dumps(self.graph_schema, indent=2)

        return f"""Jsi expert na převod dotazů z přirozeného jazyka na Cypher dotazy pro Neo4j grafovou databázi.

SCHÉMA ZNALOSTNÍHO GRAFU:
{schema_info}

PRAVIDLA PRO CYPHER DOTAZY:
1. Používej pouze uzly a vztahy definované ve schématu
2. Vždy omez výsledky pomocí LIMIT (max 50)
3. Pro vyhledávání textu používej CONTAINS (case-insensitive)
4. Při hledání entit podle typu používej presnou shodu
5. Pro síťové dotazy používaj proměnlivou délku cest: -[*1..2]-
6. Vždy vrať relevantní vlastnosti uzlů a vztahů
7. Nepoužívej DELETE, CREATE, SET nebo jiné modifikační příkazy
8. Validuj Cypher syntax

PŘÍKLADY DOTAZŮ:

Dotaz: "Najdi všechny osoby"
Cypher: MATCH (e:Entity {{type: 'PERSON'}}) RETURN e.text, e.confidence LIMIT 50

Dotaz: "Kdo poslal Bitcoin?"
Cypher: MATCH (sender:Entity)-[r:RELATES_TO {{type: 'SENT_BTC_TO'}}]->(recipient:Entity) RETURN sender.text, recipient.text, r.confidence LIMIT 50

Dotaz: "Jaké entity jsou spojené s CryptoKing?"
Cypher: MATCH (center:Entity {{text: 'CryptoKing'}})-[r:RELATES_TO]-(connected:Entity) RETURN center.text, connected.text, connected.type, r.type LIMIT 50

Dotaz: "Najdi .onion domény a co s nimi souvisí"
Cypher: MATCH (onion:Entity {{type: 'ONION'}})-[r:RELATES_TO]-(other:Entity) RETURN onion.text, other.text, other.type, r.type LIMIT 50

UŽIVATELSKÝ DOTAZ: {query}

ODPOVĚĎ (pouze Cypher dotaz, žádný další text):"""

    async def convert_to_cypher(self, natural_language_query: str) -> Dict[str, Any]:
        """
        Převeď dotaz v přirozeném jazyce na Cypher

        Args:
            natural_language_query: Dotaz v přirozeném jazyce

        Returns:
            Slovník s Cypher dotazem a metadata
        """
        try:
            logger.info(f"🔄 Převádím dotaz na Cypher: '{natural_language_query}'")

            # Vytvoř prompt
            prompt = self.cypher_prompt.format(query=natural_language_query)

            # Zavolej LLM
            llm_response = await self._call_ollama(prompt)

            if not llm_response:
                logger.warning("⚠️ LLM nevrátil žádnou odpověď")
                return self._fallback_cypher(natural_language_query)

            # Extrahuj Cypher dotaz z odpovědi
            cypher_query = self._extract_cypher_query(llm_response)

            if not cypher_query:
                logger.warning("⚠️ Nepodařilo se extrahovat Cypher dotaz")
                return self._fallback_cypher(natural_language_query)

            # Validuj Cypher dotaz
            validation_result = self._validate_cypher(cypher_query)

            if not validation_result["valid"]:
                logger.warning(f"⚠️ Nevalidní Cypher: {validation_result['error']}")
                return self._fallback_cypher(natural_language_query)

            logger.info(f"✅ Cypher vygenerován: {cypher_query[:100]}...")

            return {
                "success": True,
                "original_query": natural_language_query,
                "cypher_query": cypher_query,
                "method": "llm",
                "confidence": 0.8,
                "generated_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"❌ Chyba při převodu na Cypher: {e}")
            return self._fallback_cypher(natural_language_query)

    async def _call_ollama(self, prompt: str) -> Optional[str]:
        """Zavolej Ollama API pro Text-to-Cypher"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Nízká teplota pro přesné Cypher dotazy
                        "top_p": 0.8,
                        "num_predict": 500,
                    },
                }

                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "")
                    else:
                        logger.error(f"❌ Ollama API error: {response.status}")
                        return None

        except Exception as e:
            logger.error(f"❌ Chyba při volání Ollama API: {e}")
            return None

    def _extract_cypher_query(self, llm_response: str) -> Optional[str]:
        """Extrahuj Cypher dotaz z LLM odpovědi"""
        try:
            # Odstraň zbytečné formátování
            response = llm_response.strip()

            # Pokus se najít Cypher dotaz mezi backticks
            cypher_match = re.search(
                r"```(?:cypher)?\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE
            )
            if cypher_match:
                return cypher_match.group(1).strip()

            # Pokus se najít MATCH dotaz přímo
            match_pattern = re.search(r"(MATCH\s+.*?)(?:\n|$)", response, re.IGNORECASE | re.DOTALL)
            if match_pattern:
                return match_pattern.group(1).strip()

            # Pokud obsahuje základní Cypher klíčová slova, použij celou odpověď
            cypher_keywords = ["MATCH", "RETURN", "WHERE", "WITH", "LIMIT"]
            if any(keyword in response.upper() for keyword in cypher_keywords):
                return response.strip()

            return None

        except Exception as e:
            logger.error(f"❌ Chyba při extrakci Cypher: {e}")
            return None

    def _validate_cypher(self, cypher_query: str) -> Dict[str, Any]:
        """Základní validace Cypher dotazu"""
        try:
            # Základní syntax kontroly
            cypher_upper = cypher_query.upper()

            # Musí obsahovat MATCH a RETURN
            if "MATCH" not in cypher_upper:
                return {"valid": False, "error": "Chybí MATCH clause"}

            if "RETURN" not in cypher_upper:
                return {"valid": False, "error": "Chybí RETURN clause"}

            # Nesmí obsahovat modifikační příkazy
            forbidden = ["DELETE", "CREATE", "SET", "REMOVE", "MERGE", "DROP"]
            for cmd in forbidden:
                if cmd in cypher_upper:
                    return {"valid": False, "error": f"Zakázaný příkaz: {cmd}"}

            # Kontrola základních syntax chyb
            if cypher_query.count("(") != cypher_query.count(")"):
                return {"valid": False, "error": "Nesymetrické závorky"}

            if cypher_query.count("[") != cypher_query.count("]"):
                return {"valid": False, "error": "Nesymetrické hranaté závorky"}

            # Kontrola LIMIT
            if "LIMIT" not in cypher_upper:
                logger.warning("⚠️ Cypher nemá LIMIT, bude přidán")
                # Přidej LIMIT pokud chybí
                if not cypher_query.strip().endswith(";"):
                    cypher_query += " LIMIT 50"
                else:
                    cypher_query = cypher_query.rstrip(";") + " LIMIT 50;"

            return {"valid": True, "query": cypher_query}

        except Exception as e:
            return {"valid": False, "error": f"Validační chyba: {e}"}

    def _fallback_cypher(self, natural_query: str) -> Dict[str, Any]:
        """Fallback Cypher dotazy pro běžné případy"""
        query_lower = natural_query.lower()

        # Heuristické mapování
        if any(word in query_lower for word in ["person", "people", "user", "osoba", "osoby"]):
            cypher = "MATCH (e:Entity {type: 'PERSON'}) RETURN e.text, e.confidence, e.last_seen LIMIT 20"
        elif any(word in query_lower for word in ["bitcoin", "btc", "crypto", "transaction"]):
            cypher = "MATCH (sender:Entity)-[r:RELATES_TO {type: 'SENT_BTC_TO'}]->(recipient:Entity) RETURN sender.text, recipient.text, r.confidence LIMIT 20"
        elif any(word in query_lower for word in ["onion", "domain", "website"]):
            cypher = "MATCH (e:Entity {type: 'ONION'}) RETURN e.text, e.confidence LIMIT 20"
        elif any(word in query_lower for word in ["organization", "org", "company", "organizace"]):
            cypher = "MATCH (e:Entity {type: 'ORG'}) RETURN e.text, e.confidence LIMIT 20"
        elif any(word in query_lower for word in ["relation", "connection", "vztah", "spojení"]):
            cypher = "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) RETURN a.text, r.type, b.text, r.confidence LIMIT 20"
        else:
            # Univerzální dotaz na všechny entity
            cypher = "MATCH (e:Entity) RETURN e.text, e.type, e.confidence ORDER BY e.last_seen DESC LIMIT 20"

        return {
            "success": True,
            "original_query": natural_query,
            "cypher_query": cypher,
            "method": "heuristic_fallback",
            "confidence": 0.6,
            "generated_at": datetime.now().isoformat(),
        }

    def get_common_queries(self) -> List[Dict[str, str]]:
        """Vrať seznam běžných dotazů s jejich Cypher ekvivalenty"""
        return [
            {
                "description": "Všechny osoby v grafu",
                "natural": "Najdi všechny osoby",
                "cypher": "MATCH (e:Entity {type: 'PERSON'}) RETURN e.text, e.confidence ORDER BY e.last_seen DESC LIMIT 50",
            },
            {
                "description": "Bitcoin transakce",
                "natural": "Kdo poslal Bitcoin?",
                "cypher": "MATCH (sender:Entity)-[r:RELATES_TO {type: 'SENT_BTC_TO'}]->(recipient:Entity) RETURN sender.text, recipient.text, r.confidence LIMIT 50",
            },
            {
                "description": "Síťové spojení konkrétní entity",
                "natural": "Co je spojeno s CryptoKing?",
                "cypher": "MATCH (center:Entity)-[r:RELATES_TO]-(connected:Entity) WHERE center.text CONTAINS 'CryptoKing' RETURN center.text, connected.text, connected.type, r.type LIMIT 50",
            },
            {
                "description": ".onion domény a jejich spojení",
                "natural": "Najdi .onion domény",
                "cypher": "MATCH (onion:Entity {type: 'ONION'})-[r:RELATES_TO]-(other:Entity) RETURN onion.text, other.text, other.type, r.type LIMIT 50",
            },
            {
                "description": "Nejaktivnější entity",
                "natural": "Které entity mají nejvíce spojení?",
                "cypher": "MATCH (e:Entity)-[r:RELATES_TO]-() RETURN e.text, e.type, count(r) as connections ORDER BY connections DESC LIMIT 20",
            },
            {
                "description": "Cesty mezi entitami",
                "natural": "Jak jsou spojeny CryptoKing a DarkMarket?",
                "cypher": "MATCH path = (a:Entity)-[*1..3]-(b:Entity) WHERE a.text CONTAINS 'CryptoKing' AND b.text CONTAINS 'DarkMarket' RETURN path LIMIT 10",
            },
        ]
