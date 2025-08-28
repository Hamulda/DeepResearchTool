"""
Rozšířené nástroje pro přístup k autoritativním a specializovaným datovým zdrojům
Implementace nástrojů pro Semantic Scholar, Data.gov a Wayback Machine

Author: Senior Python/MLOps Agent
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Any, Optional
from langchain.tools import tool
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@tool
async def semantic_scholar_search(query: str, limit: int = 10) -> Dict[str, Any]:
    """
    Vyhledá akademické články pomocí Semantic Scholar API

    Args:
        query: Vyhledávací dotaz
        limit: Maximální počet výsledků (výchozí 10)

    Returns:
        Slovník s výsledky vyhledávání
    """
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

    params = {
        "query": query,
        "limit": limit,
        "fields": "paperId,title,abstract,authors,year,citationCount,url,venue,publicationTypes"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    formatted_results = []
                    for paper in data.get("data", []):
                        formatted_paper = {
                            "title": paper.get("title", ""),
                            "abstract": paper.get("abstract", ""),
                            "authors": [author.get("name", "") for author in paper.get("authors", [])],
                            "year": paper.get("year"),
                            "citation_count": paper.get("citationCount", 0),
                            "url": paper.get("url", ""),
                            "venue": paper.get("venue", ""),
                            "publication_types": paper.get("publicationTypes", []),
                            "paper_id": paper.get("paperId", "")
                        }
                        formatted_results.append(formatted_paper)

                    return {
                        "success": True,
                        "papers": formatted_results,
                        "total_found": data.get("total", 0),
                        "query": query
                    }
                else:
                    return {
                        "success": False,
                        "error": f"API chyba: {response.status}",
                        "papers": []
                    }
    except Exception as e:
        logger.error(f"Chyba při vyhledávání v Semantic Scholar: {e}")
        return {
            "success": False,
            "error": str(e),
            "papers": []
        }


@tool
async def data_gov_search(query: str, limit: int = 10, resource_type: str = "dataset") -> Dict[str, Any]:
    """
    Vyhledá vládní datové sady pomocí Data.gov API

    Args:
        query: Vyhledávací dotaz
        limit: Maximální počet výsledků (výchozí 10)
        resource_type: Typ zdroje (dataset, article, app)

    Returns:
        Slovník s výsledky vyhledávání
    """
    base_url = "https://catalog.data.gov/api/3/action/package_search"

    params = {
        "q": query,
        "rows": limit,
        "sort": "score desc"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if data.get("success"):
                        formatted_results = []
                        for dataset in data.get("result", {}).get("results", []):
                            formatted_dataset = {
                                "title": dataset.get("title", ""),
                                "description": dataset.get("notes", ""),
                                "organization": dataset.get("organization", {}).get("title", ""),
                                "tags": [tag.get("name", "") for tag in dataset.get("tags", [])],
                                "url": f"https://catalog.data.gov/dataset/{dataset.get('name', '')}",
                                "last_modified": dataset.get("metadata_modified", ""),
                                "resources": len(dataset.get("resources", [])),
                                "dataset_id": dataset.get("id", "")
                            }
                            formatted_results.append(formatted_dataset)

                        return {
                            "success": True,
                            "datasets": formatted_results,
                            "total_found": data.get("result", {}).get("count", 0),
                            "query": query
                        }
                    else:
                        return {
                            "success": False,
                            "error": "API vrátilo neúspěšný výsledek",
                            "datasets": []
                        }
                else:
                    return {
                        "success": False,
                        "error": f"API chyba: {response.status}",
                        "datasets": []
                    }
    except Exception as e:
        logger.error(f"Chyba při vyhledávání v Data.gov: {e}")
        return {
            "success": False,
            "error": str(e),
            "datasets": []
        }


@tool
async def wayback_machine_search(url: str, timestamp: Optional[str] = None) -> Dict[str, Any]:
    """
    Vyhledá historické verze webových stránek pomocí Wayback Machine API

    Args:
        url: URL stránky k vyhledání
        timestamp: Specifický timestamp (YYYYMMDDHHMMSS) nebo None pro nejnovější

    Returns:
        Slovník s historickými verzemi stránky
    """
    # API pro získání dostupných snapshotů
    availability_url = f"http://archive.org/wayback/available"

    params = {"url": url}
    if timestamp:
        params["timestamp"] = timestamp

    try:
        async with aiohttp.ClientSession() as session:
            # Získání dostupných snapshotů
            async with session.get(availability_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if data.get("archived_snapshots"):
                        closest = data["archived_snapshots"].get("closest", {})

                        if closest.get("available"):
                            # Získání obsahu z archivu
                            archived_url = closest.get("url", "")

                            try:
                                async with session.get(archived_url) as content_response:
                                    if content_response.status == 200:
                                        content = await content_response.text()

                                        return {
                                            "success": True,
                                            "archived_url": archived_url,
                                            "timestamp": closest.get("timestamp", ""),
                                            "original_url": url,
                                            "content": content[:5000],  # Omezení na prvních 5000 znaků
                                            "full_content_available": True
                                        }
                                    else:
                                        return {
                                            "success": False,
                                            "error": f"Nepodařilo se získat obsah: {content_response.status}",
                                            "archived_url": archived_url
                                        }
                            except Exception as content_error:
                                return {
                                    "success": False,
                                    "error": f"Chyba při získávání obsahu: {content_error}",
                                    "archived_url": archived_url
                                }
                        else:
                            return {
                                "success": False,
                                "error": "Stránka není dostupná v archivu",
                                "original_url": url
                            }
                    else:
                        return {
                            "success": False,
                            "error": "Žádné archivované snapshoty nenalezeny",
                            "original_url": url
                        }
                else:
                    return {
                        "success": False,
                        "error": f"API chyba: {response.status}",
                        "original_url": url
                    }
    except Exception as e:
        logger.error(f"Chyba při vyhledávání ve Wayback Machine: {e}")
        return {
            "success": False,
            "error": str(e),
            "original_url": url
        }


@tool
async def cross_reference_sources(query: str) -> Dict[str, Any]:
    """
    Kombinuje vyhledávání napříč všemi dostupnými autoritativními zdroji

    Args:
        query: Vyhledávací dotaz

    Returns:
        Agregované výsledky ze všech zdrojů
    """
    results = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "sources": {}
    }

    # Paralelní vyhledávání ve všech zdrojích
    tasks = [
        semantic_scholar_search(query, limit=5),
        data_gov_search(query, limit=5)
    ]

    try:
        # Spuštění paralelního vyhledávání
        semantic_results, data_gov_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Zpracování výsledků z Semantic Scholar
        if not isinstance(semantic_results, Exception) and semantic_results.get("success"):
            results["sources"]["semantic_scholar"] = {
                "status": "success",
                "count": len(semantic_results.get("papers", [])),
                "data": semantic_results.get("papers", [])
            }
        else:
            results["sources"]["semantic_scholar"] = {
                "status": "error",
                "error": str(semantic_results) if isinstance(semantic_results, Exception) else semantic_results.get("error", "Neznámá chyba")
            }

        # Zpracování výsledků z Data.gov
        if not isinstance(data_gov_results, Exception) and data_gov_results.get("success"):
            results["sources"]["data_gov"] = {
                "status": "success",
                "count": len(data_gov_results.get("datasets", [])),
                "data": data_gov_results.get("datasets", [])
            }
        else:
            results["sources"]["data_gov"] = {
                "status": "error",
                "error": str(data_gov_results) if isinstance(data_gov_results, Exception) else data_gov_results.get("error", "Neznámá chyba")
            }

        # Výpočet celkových statistik
        total_sources = sum(
            source_data.get("count", 0)
            for source_data in results["sources"].values()
            if source_data.get("status") == "success"
        )

        results["summary"] = {
            "total_sources_found": total_sources,
            "successful_apis": len([s for s in results["sources"].values() if s.get("status") == "success"]),
            "failed_apis": len([s for s in results["sources"].values() if s.get("status") == "error"])
        }

        return results

    except Exception as e:
        logger.error(f"Chyba při křížové referenci zdrojů: {e}")
        return {
            "query": query,
            "error": str(e),
            "sources": {},
            "summary": {"total_sources_found": 0, "successful_apis": 0, "failed_apis": 0}
        }


def get_enhanced_tools() -> List:
    """
    Vrátí seznam všech rozšířených nástrojů

    Returns:
        Seznam nástrojů pro použití v agentovi
    """
    return [
        semantic_scholar_search,
        data_gov_search,
        wayback_machine_search,
        cross_reference_sources
    ]
