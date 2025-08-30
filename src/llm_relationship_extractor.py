"""LLM Relationship Extractor - Fáze 1: Jádro Znalostního Grafu
Využívá LLM k extrakci vztahů mezi entitami ve formátu (subjekt, predikát, objekt)
"""

import asyncio
import json
import logging
import os
from typing import Any

import aiohttp

# Setup logging
logger = logging.getLogger(__name__)


class LLMRelationshipExtractor:
    """Extraktor vztahů pomocí LLM pro znalostní graf"""

    def __init__(self):
        self.ollama_host = os.getenv("OLLAMA_HOST", "localhost")
        self.ollama_port = os.getenv("OLLAMA_PORT", "11434")
        self.ollama_url = f"http://{self.ollama_host}:{self.ollama_port}"
        self.model_name = os.getenv("LLM_MODEL", "llama2")

        # Prompt template pro extrakci vztahů
        self.relationship_prompt = self._create_relationship_prompt()

    def _create_relationship_prompt(self) -> str:
        """Vytvoř sofistikovaný prompt pro extrakci vztahů"""
        return """Analyzuj následující text a extrahuj všechny významné vztahy mezi entitami.

INSTRUKCE:
1. Identifikuj entity (osoby, organizace, místa, krypto adresy, .onion domény, atd.)
2. Najdi vztahy mezi těmito entitami
3. Vrať výsledek POUZE ve formátu JSON bez dalšího textu

FORMÁT VÝSTUPU:
{
  "entities": [
    {"text": "název entity", "type": "PERSON|ORG|LOCATION|CRYPTO|ONION|OTHER"},
    ...
  ],
  "relations": [
    {"subject": "entita A", "predicate": "TYP_VZTAHU", "object": "entita B", "confidence": 0.8},
    ...
  ]
}

TYPY VZTAHŮ (používej tyto nebo podobné):
- OWNS: vlastní
- SENT_TO: poslal komu/kam
- POSTED_ON: publikoval na
- MEMBER_OF: člen organizace
- LOCATED_IN: nachází se v
- CONNECTED_TO: spojen s
- CREATED: vytvořil
- OPERATES: provozuje
- MENTIONED_BY: zmíněn kým
- ASSOCIATED_WITH: spojen s

TEXT K ANALÝZE:
{text}

ODPOVĚĎ (pouze JSON):"""

    async def extract_relationships(
        self, text: str, entities: dict[str, list[dict[str, Any]]]
    ) -> dict[str, Any]:
        """Extrahuj vztahy z textu pomocí LLM

        Args:
            text: Text k analýze
            entities: Již extrahované entity pomocí spaCy

        Returns:
            Slovník s entitami a vztahy ve formátu JSON

        """
        try:
            # Ořízni text na rozumnou délku pro LLM
            max_length = 2000
            if len(text) > max_length:
                text = text[:max_length] + "..."

            # Vytvoř prompt s textem
            prompt = self.relationship_prompt.format(text=text)

            # Zavolej LLM
            llm_response = await self._call_ollama(prompt)

            if not llm_response:
                logger.warning("⚠️ LLM nevrátil žádnou odpověď")
                return self._fallback_extraction(entities)

            # Parsuj JSON odpověď
            try:
                result = json.loads(llm_response)

                # Validuj strukturu
                if "entities" not in result:
                    result["entities"] = []
                if "relations" not in result:
                    result["relations"] = []

                # Kombinuj s původními entitami ze spaCy
                result = self._merge_entities(result, entities)

                # Validuj a vyčisti vztahy
                result["relations"] = self._validate_relations(result["relations"])

                logger.info(
                    f"✅ LLM extrahoval {len(result['entities'])} entit a {len(result['relations'])} vztahů"
                )
                return result

            except json.JSONDecodeError as e:
                logger.error(f"❌ LLM nevrátil validní JSON: {e}")
                logger.debug(f"LLM odpověď: {llm_response[:500]}...")
                return self._fallback_extraction(entities)

        except Exception as e:
            logger.error(f"❌ Chyba při extrakci vztahů: {e}")
            return self._fallback_extraction(entities)

    async def _call_ollama(self, prompt: str) -> str | None:
        """Zavolej Ollama API pro generování odpovědi"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Nízká teplota pro konzistentní výstup
                        "top_p": 0.9,
                        "num_predict": 1000,
                    },
                }

                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "")
                    logger.error(f"❌ Ollama API error: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"❌ Chyba při volání Ollama API: {e}")
            return None

    def _merge_entities(
        self, llm_result: dict[str, Any], spacy_entities: dict[str, list[dict[str, Any]]]
    ) -> dict[str, Any]:
        """Kombinuj entity z LLM a spaCy"""
        merged_entities = []

        # Přidej entity ze spaCy
        for entity_type, entity_list in spacy_entities.items():
            for entity in entity_list:
                merged_entities.append(
                    {
                        "text": entity["text"],
                        "type": self._normalize_entity_type(entity["label"]),
                        "source": "spacy",
                        "confidence": entity.get("confidence", 1.0),
                    }
                )

        # Přidej entity z LLM (pokud už nejsou zahrnuty)
        existing_texts = {e["text"].lower() for e in merged_entities}

        for llm_entity in llm_result.get("entities", []):
            if llm_entity["text"].lower() not in existing_texts:
                merged_entities.append(
                    {
                        "text": llm_entity["text"],
                        "type": llm_entity["type"],
                        "source": "llm",
                        "confidence": 0.8,
                    }
                )
                existing_texts.add(llm_entity["text"].lower())

        llm_result["entities"] = merged_entities
        return llm_result

    def _normalize_entity_type(self, spacy_label: str) -> str:
        """Normalizuj spaCy label na naše typy"""
        mapping = {
            "PERSON": "PERSON",
            "ORG": "ORG",
            "GPE": "LOCATION",
            "LOC": "LOCATION",
            "LOCATION": "LOCATION",
            "DATE": "DATE",
            "TIME": "TIME",
            "CRYPTO_ADDRESS": "CRYPTO",
            "ONION_ADDRESS": "ONION",
            "HASH": "HASH",
            "PGP_KEY": "PGP_KEY",
        }
        return mapping.get(spacy_label, "OTHER")

    def _validate_relations(self, relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Validuj a vyčisti vztahy"""
        valid_relations = []

        for relation in relations:
            # Zkontroluj povinná pole
            if not all(key in relation for key in ["subject", "predicate", "object"]):
                continue

            # Zkontroluj, že subjekt a objekt nejsou prázdné
            if not relation["subject"].strip() or not relation["object"].strip():
                continue

            # Zkontroluj, že subjekt a objekt nejsou stejné
            if relation["subject"].strip().lower() == relation["object"].strip().lower():
                continue

            # Přidej výchozí confidence pokud chybí
            if "confidence" not in relation:
                relation["confidence"] = 0.7

            # Normalizuj predikát
            relation["predicate"] = relation["predicate"].upper().replace(" ", "_")

            valid_relations.append(relation)

        return valid_relations

    def _fallback_extraction(
        self, spacy_entities: dict[str, list[dict[str, Any]]]
    ) -> dict[str, Any]:
        """Fallback když LLM selže - použij pouze spaCy entity bez vztahů"""
        entities = []

        for entity_type, entity_list in spacy_entities.items():
            for entity in entity_list:
                entities.append(
                    {
                        "text": entity["text"],
                        "type": self._normalize_entity_type(entity["label"]),
                        "source": "spacy_fallback",
                        "confidence": entity.get("confidence", 1.0),
                    }
                )

        return {"entities": entities, "relations": []}

    async def batch_extract_relationships(self, text_chunks: list[str]) -> list[dict[str, Any]]:
        """Zpracuj více textových chunků paralelně"""
        tasks = []

        for chunk in text_chunks:
            # Pro batch processing použijeme prázdné entity (LLM je extrahuje sám)
            task = self.extract_relationships(chunk, {})
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filtruj úspěšné výsledky
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Chyba při zpracování chunk {i}: {result}")
            else:
                valid_results.append(result)

        return valid_results

    def create_simple_relations_from_entities(
        self, entities: dict[str, list[dict[str, Any]]], source_url: str
    ) -> list[dict[str, Any]]:
        """Vytvoř jednoduché vztahy mezi entitami když LLM není dostupný
        Například propoj osoby s organizacemi, lokacemi atd.
        """
        relations = []

        persons = entities.get("persons", [])
        organizations = entities.get("organizations", [])
        locations = entities.get("locations", [])
        crypto_addresses = entities.get("crypto_addresses", [])
        onion_addresses = entities.get("onion_addresses", [])

        # Propoj osoby s organizacemi
        for person in persons:
            for org in organizations:
                relations.append(
                    {
                        "subject": person["text"],
                        "predicate": "ASSOCIATED_WITH",
                        "object": org["text"],
                        "confidence": 0.5,
                        "source": "heuristic",
                    }
                )

        # Propoj osoby s lokacemi
        for person in persons:
            for location in locations:
                relations.append(
                    {
                        "subject": person["text"],
                        "predicate": "ASSOCIATED_WITH",
                        "object": location["text"],
                        "confidence": 0.4,
                        "source": "heuristic",
                    }
                )

        # Propoj entity s krypto adresami
        for person in persons:
            for crypto in crypto_addresses:
                relations.append(
                    {
                        "subject": person["text"],
                        "predicate": "OWNS",
                        "object": crypto["text"],
                        "confidence": 0.6,
                        "source": "heuristic",
                    }
                )

        # Propoj všechny entity s .onion doménami
        all_entities = persons + organizations
        for entity in all_entities:
            for onion in onion_addresses:
                relations.append(
                    {
                        "subject": entity["text"],
                        "predicate": "OPERATES",
                        "object": onion["text"],
                        "confidence": 0.5,
                        "source": "heuristic",
                    }
                )

        return relations[:20]  # Omez počet heuristických vztahů
