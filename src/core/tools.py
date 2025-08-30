"""Základní nástroje pro Research Agent implementované pomocí LangChain @tool dekorátoru
Webový scraping, vyhledávání a další utility nástroje

Author: Senior Python/MLOps Agent
"""

import logging
from typing import Any

import aiohttp
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


class WebScrapingInput(BaseModel):
    """Input schema pro web scraping nástroj"""

    query: str = Field(description="URL pro scraping nebo klíčová slova pro vyhledání")
    scrape_type: str = Field(
        default="url", description="Typ scrapingu: 'url' pro přímý scraping, 'search' pro vyhledání"
    )
    max_pages: int = Field(default=1, description="Maximální počet stránek k získání")


class SearchInput(BaseModel):
    """Input schema pro vyhledávací nástroj"""

    query: str = Field(description="Vyhledávací dotaz")
    num_results: int = Field(default=5, description="Počet výsledků k vrácení")
    search_type: str = Field(
        default="web", description="Typ vyhledávání: 'web', 'academic', 'news'"
    )


@tool("web_scraping_tool", args_schema=WebScrapingInput)
async def web_scraping_tool(
    query: str, scrape_type: str = "url", max_pages: int = 1
) -> dict[str, Any]:
    """Nástroj pro webový scraping pomocí Firecrawl API

    Podporuje:
    - Přímý scraping URL
    - Vyhledání a scraping výsledků
    - Strukturované získání obsahu stránky

    Args:
        query: URL pro scraping nebo klíčová slova pro vyhledání
        scrape_type: 'url' pro přímý scraping, 'search' pro vyhledání
        max_pages: Maximální počet stránek

    Returns:
        Strukturovaný obsah stránky s metadaty

    """
    try:
        if scrape_type == "url":
            return await _scrape_url_firecrawl(query)
        if scrape_type == "search":
            return await _search_and_scrape_firecrawl(query, max_pages)
        return {
            "error": f"Nepodporovaný typ scrapingu: {scrape_type}",
            "content": "",
            "metadata": {},
        }
    except Exception as e:
        logger.error(f"Chyba při web scrapingu: {e}")
        return {"error": str(e), "content": "", "metadata": {}}


async def _scrape_url_firecrawl(url: str) -> dict[str, Any]:
    """Scraping jednotlivé URL pomocí Firecrawl API

    Args:
        url: URL k scrapingu

    Returns:
        Strukturovaný obsah stránky

    """
    try:
        from firecrawl import FirecrawlApp

        # Inicializace Firecrawl (API klíč by měl být v environment variables)
        app = FirecrawlApp()

        # Scraping s pokročilými možnostmi
        scrape_result = app.scrape_url(
            url,
            params={
                "formats": ["markdown", "html"],
                "includeTags": ["title", "meta", "h1", "h2", "h3", "p", "article"],
                "excludeTags": ["script", "style", "nav", "footer", "sidebar"],
                "waitFor": 2000,  # Čekání na načtení JS
                "screenshot": False,
                "fullPageScreenshot": False,
            },
        )

        if scrape_result.get("success"):
            content = scrape_result.get("markdown", "") or scrape_result.get("html", "")

            return {
                "content": content,
                "url": url,
                "title": scrape_result.get("metadata", {}).get("title", ""),
                "metadata": {
                    "source": "firecrawl",
                    "url": url,
                    "title": scrape_result.get("metadata", {}).get("title", ""),
                    "description": scrape_result.get("metadata", {}).get("description", ""),
                    "language": scrape_result.get("metadata", {}).get("language", ""),
                    "timestamp": scrape_result.get("metadata", {}).get("timestamp"),
                    "content_length": len(content),
                },
            }
        return {
            "error": f"Firecrawl scraping selhal: {scrape_result.get('error', 'Neznámá chyba')}",
            "content": "",
            "metadata": {"url": url},
        }

    except ImportError:
        # Fallback na základní scraping bez Firecrawl
        logger.warning("Firecrawl není dostupný, používám základní scraping")
        return await _basic_web_scraping(url)
    except Exception as e:
        logger.error(f"Chyba při Firecrawl scrapingu: {e}")
        return await _basic_web_scraping(url)


async def _search_and_scrape_firecrawl(query: str, max_pages: int = 1) -> dict[str, Any]:
    """Vyhledání a scraping výsledků pomocí Firecrawl

    Args:
        query: Vyhledávací dotaz
        max_pages: Počet stránek k získání

    Returns:
        Kombinovaný obsah z nalezených stránek

    """
    try:
        from firecrawl import FirecrawlApp

        app = FirecrawlApp()

        # Použití search funkcionality Firecrawl
        search_result = app.search(
            query,
            params={
                "limit": max_pages,
                "formats": ["markdown"],
                "includeTags": ["title", "meta", "h1", "h2", "h3", "p", "article"],
                "excludeTags": ["script", "style", "nav", "footer"],
            },
        )

        if search_result.get("success") and search_result.get("data"):
            combined_content = []
            all_metadata = []

            for result in search_result["data"][:max_pages]:
                content = result.get("markdown", "") or result.get("content", "")
                if content:
                    combined_content.append(
                        f"--- {result.get('title', 'Bez názvu')} ---\n{content}"
                    )
                    all_metadata.append(
                        {
                            "url": result.get("url", ""),
                            "title": result.get("title", ""),
                            "description": result.get("description", ""),
                        }
                    )

            return {
                "content": "\n\n".join(combined_content),
                "query": query,
                "metadata": {
                    "source": "firecrawl_search",
                    "query": query,
                    "num_results": len(all_metadata),
                    "results": all_metadata,
                },
            }
        return {
            "error": f"Vyhledávání selhalo: {search_result.get('error', 'Žádné výsledky')}",
            "content": "",
            "metadata": {"query": query},
        }

    except ImportError:
        # Fallback na základní vyhledávání
        logger.warning("Firecrawl není dostupný, používám základní vyhledávání")
        return await _basic_search_and_scrape(query, max_pages)
    except Exception as e:
        logger.error(f"Chyba při Firecrawl vyhledávání: {e}")
        return await _basic_search_and_scrape(query, max_pages)


async def _basic_web_scraping(url: str) -> dict[str, Any]:
    """Základní web scraping bez Firecrawl (fallback)

    Args:
        url: URL k scrapingu

    Returns:
        Základní obsah stránky

    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    html_content = await response.text()

                    # Základní čištění HTML pomocí BeautifulSoup
                    from bs4 import BeautifulSoup

                    soup = BeautifulSoup(html_content, "html.parser")

                    # Odstranění nežádoucích elementů
                    for tag in soup(["script", "style", "nav", "footer", "aside"]):
                        tag.decompose()

                    # Získání textu
                    text_content = soup.get_text(separator="\n", strip=True)
                    title = soup.find("title")
                    title_text = title.get_text() if title else ""

                    return {
                        "content": text_content,
                        "url": url,
                        "title": title_text,
                        "metadata": {
                            "source": "basic_scraping",
                            "url": url,
                            "title": title_text,
                            "status_code": response.status,
                            "content_length": len(text_content),
                        },
                    }
                return {
                    "error": f"HTTP chyba: {response.status}",
                    "content": "",
                    "metadata": {"url": url, "status_code": response.status},
                }
    except Exception as e:
        return {
            "error": f"Chyba při základním scrapingu: {e!s}",
            "content": "",
            "metadata": {"url": url},
        }


async def _basic_search_and_scrape(query: str, max_pages: int = 1) -> dict[str, Any]:
    """Základní vyhledávání a scraping (fallback implementace)

    Args:
        query: Vyhledávací dotaz
        max_pages: Počet stránek

    Returns:
        Výsledky vyhledávání

    """
    # Jednoduchá implementace - v produkci by se použil skutečný vyhledávač
    return {
        "content": f"Základní vyhledávání pro '{query}' není implementováno. Použijte přímé URL.",
        "query": query,
        "metadata": {
            "source": "basic_search_fallback",
            "query": query,
            "note": "Firecrawl není dostupný",
        },
    }


@tool("knowledge_search_tool", args_schema=SearchInput)
async def knowledge_search_tool(
    query: str, num_results: int = 5, search_type: str = "web"
) -> dict[str, Any]:
    """Nástroj pro vyhledávání v knowledge base a externích zdrojích

    Args:
        query: Vyhledávací dotaz
        num_results: Počet výsledků
        search_type: Typ vyhledávání (web, academic, news)

    Returns:
        Strukturované výsledky vyhledávání

    """
    try:
        # Import RAG pipeline
        from .rag_pipeline import RAGPipeline

        # Načtení konfigurace (zde by byla skutečná konfigurace)
        config = {
            "memory_store": {
                "type": "chroma",
                "collection_name": "research_documents",
                "persist_directory": "./chroma_db",
            }
        }

        # Inicializace RAG pipeline
        rag = RAGPipeline(config)
        await rag.initialize()

        # Vyhledání v knowledge base
        local_results = await rag.search(query, k=num_results)

        results = {
            "query": query,
            "local_results": [
                {
                    "content": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                    "metadata": doc.metadata,
                    "relevance_score": doc.metadata.get("distance", 0),
                }
                for doc in local_results
            ],
            "num_local_results": len(local_results),
            "search_type": search_type,
        }

        # Pokud nejsou lokální výsledky, zkus web scraping
        if not local_results and search_type == "web":
            web_results = await web_scraping_tool(query, scrape_type="search", max_pages=3)
            results["web_fallback"] = web_results

        return results

    except Exception as e:
        logger.error(f"Chyba při vyhledávání v knowledge base: {e}")
        return {"error": str(e), "query": query, "local_results": [], "num_local_results": 0}


@tool("document_analysis_tool")
async def document_analysis_tool(text: str, analysis_type: str = "summary") -> dict[str, Any]:
    """Nástroj pro analýzu dokumentů a textů

    Args:
        text: Text k analýze
        analysis_type: Typ analýzy (summary, keywords, entities, sentiment)

    Returns:
        Výsledky analýzy

    """
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

        if analysis_type == "summary":
            prompt = f"Vytvoř stručný souhrn následujícího textu:\n\n{text[:2000]}"
        elif analysis_type == "keywords":
            prompt = f"Extrahuj 10 nejdůležitějších klíčových slov z textu:\n\n{text[:2000]}"
        elif analysis_type == "entities":
            prompt = (
                f"Identifikuj osoby, místa, organizace a další entity v textu:\n\n{text[:2000]}"
            )
        elif analysis_type == "sentiment":
            prompt = f"Analyzuj sentiment a tón následujícího textu:\n\n{text[:2000]}"
        else:
            prompt = f"Proveď obecnou analýzu textu:\n\n{text[:2000]}"

        response = await llm.ainvoke(
            [
                SystemMessage(content="Jsi expert na analýzu textů a dokumentů."),
                HumanMessage(content=prompt),
            ]
        )

        return {
            "analysis_type": analysis_type,
            "result": response.content,
            "text_length": len(text),
            "analyzed_portion": min(2000, len(text)),
        }

    except Exception as e:
        logger.error(f"Chyba při analýze dokumentu: {e}")
        return {"error": str(e), "analysis_type": analysis_type, "result": ""}


# Seznam všech dostupných nástrojů pro export
available_tools = [web_scraping_tool, knowledge_search_tool, document_analysis_tool]


from .enhanced_tools import get_enhanced_tools


def get_tools_for_agent() -> list:
    """Vrátí kompletní seznam nástrojů pro agenta včetně rozšířených nástrojů

    Returns:
        Seznam všech dostupných nástrojů

    """
    base_tools = [web_scraping_tool, knowledge_search_tool]

    # Přidání rozšířených nástrojů
    enhanced_tools = get_enhanced_tools()

    return base_tools + enhanced_tools
