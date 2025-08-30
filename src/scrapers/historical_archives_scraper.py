#!/usr/bin/env python3
"""Historical Archives Scraper
Access to Qatar Digital Library, Chinese Text Project, European archives and other historical sources

Author: Advanced IT Specialist
"""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
import logging
import time
from typing import Any
from urllib.parse import urljoin

import aiohttp
from bs4 import BeautifulSoup

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


@dataclass
class HistoricalDocument:
    """Represents a historical document"""

    document_id: str
    title: str
    content: str
    original_language: str
    translated_content: str | None
    time_period: str
    dynasty_era: str | None
    geographic_origin: str
    document_type: str
    historical_significance: str
    preservation_status: str
    digitization_date: datetime
    source_archive: str
    catalog_number: str
    metadata: dict[str, Any]


class HistoricalArchivesScraper(BaseScraper):
    """Scraper for historical archives and ancient documents"""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.name = "historical_archives_scraper"

        # Archive endpoints
        self.qatar_digital_library = "https://www.qdl.qa/en/search/"
        self.chinese_text_project = "https://ctext.org/searchbooks.pl"
        self.adam_matthew = "https://www.amdigital.co.uk/primary-sources/"
        self.arcanum_adatbazis = "https://adtplus.arcanum.hu/en/"
        self.europeana = "https://api.europeana.eu/record/v2/"

        # Rate limiting configuration
        self.rate_limits = config.get("historical_archives", {})
        self.last_request_times = {}

        # Language support
        self.supported_languages = (
            config.get("historical_archives", {})
            .get("chinese_text_project", {})
            .get("language_support", ["zh", "en"])
        )

    async def search(self, query: str, **kwargs) -> AsyncGenerator[HistoricalDocument, None]:
        """Search across multiple historical archives"""
        logger.info(f"Searching historical archives for: {query}")

        archive_types = kwargs.get("archive_types", ["all"])
        time_period = kwargs.get("time_period")
        language = kwargs.get("language", "en")

        # Search Qatar Digital Library (Middle East historical documents)
        if "all" in archive_types or "qatar" in archive_types:
            async for doc in self._search_qatar_digital_library(query, **kwargs):
                yield doc

        # Search Chinese Text Project (Pre-modern Chinese texts)
        if "all" in archive_types or "chinese" in archive_types:
            async for doc in self._search_chinese_text_project(query, **kwargs):
                yield doc

        # Search European archives
        if "all" in archive_types or "european" in archive_types:
            async for doc in self._search_european_archives(query, **kwargs):
                yield doc

        # Search Adam Matthew Digital collections
        if "all" in archive_types or "adam_matthew" in archive_types:
            async for doc in self._search_adam_matthew(query, **kwargs):
                yield doc

    async def _search_qatar_digital_library(
        self, query: str, **kwargs
    ) -> AsyncGenerator[HistoricalDocument, None]:
        """Search Qatar Digital Library for Middle East historical documents"""
        try:
            await self._respect_rate_limit("qatar_digital_library")

            # QDL search parameters
            params = {"text": query, "search_type": "full_text", "sort": "relevance", "rows": 50}

            # Add time period filter if specified
            time_period = kwargs.get("time_period")
            if time_period:
                params["date_range"] = time_period

            async with aiohttp.ClientSession() as session:
                async with session.get(self.qatar_digital_library, params=params) as response:
                    if response.status == 200:
                        content = await response.text()
                        documents = await self._parse_qdl_results(content, session)

                        for doc in documents:
                            yield doc
                    else:
                        logger.error(f"Qatar Digital Library search failed: {response.status}")

        except Exception as e:
            logger.error(f"Error searching Qatar Digital Library: {e}")

    async def _search_chinese_text_project(
        self, query: str, **kwargs
    ) -> AsyncGenerator[HistoricalDocument, None]:
        """Search Chinese Text Project for pre-modern Chinese texts"""
        try:
            await self._respect_rate_limit("chinese_text_project")

            # Chinese Text Project search
            params = {"if": "gb", "searchu": query, "searchmode": "text"}  # Use GB encoding

            # Add collection filter
            collections = kwargs.get("collections", ["pre_modern", "ming_qing"])
            if "ming_qing" in collections:
                params["remap"] = "gb"

            async with aiohttp.ClientSession() as session:
                async with session.get(self.chinese_text_project, params=params) as response:
                    if response.status == 200:
                        content = await response.text()
                        documents = await self._parse_ctext_results(content, session)

                        for doc in documents:
                            yield doc
                    else:
                        logger.error(f"Chinese Text Project search failed: {response.status}")

        except Exception as e:
            logger.error(f"Error searching Chinese Text Project: {e}")

    async def _search_european_archives(
        self, query: str, **kwargs
    ) -> AsyncGenerator[HistoricalDocument, None]:
        """Search European archives including Arcanum and Europeana"""
        try:
            # Search Arcanum (Hungarian archives)
            async for doc in self._search_arcanum(query, **kwargs):
                yield doc

            # Search Europeana
            async for doc in self._search_europeana(query, **kwargs):
                yield doc

        except Exception as e:
            logger.error(f"Error searching European archives: {e}")

    async def _search_arcanum(
        self, query: str, **kwargs
    ) -> AsyncGenerator[HistoricalDocument, None]:
        """Search Arcanum database for Hungarian historical content"""
        try:
            await self._respect_rate_limit("arcanum")

            # Arcanum uses a different search mechanism
            search_url = f"{self.arcanum_adatbazis}search"
            params = {
                "q": query,
                "date_from": kwargs.get("date_range", {}).get("start", "1800"),
                "date_to": kwargs.get("date_range", {}).get("end", "2023"),
                "lang": "en",
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params) as response:
                    if response.status == 200:
                        content = await response.text()
                        documents = await self._parse_arcanum_results(content, session)

                        for doc in documents:
                            yield doc

        except Exception as e:
            logger.error(f"Error searching Arcanum: {e}")

    async def _search_europeana(
        self, query: str, **kwargs
    ) -> AsyncGenerator[HistoricalDocument, None]:
        """Search Europeana digital heritage platform"""
        try:
            await self._respect_rate_limit("europeana")

            # Europeana API search
            api_key = self.config.get("europeana_api_key")  # Would need API key
            if not api_key:
                logger.warning("Europeana API key not configured")
                return

            search_url = "https://api.europeana.eu/record/v2/search.json"
            params = {"wskey": api_key, "query": query, "rows": 50, "profile": "rich"}

            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        documents = await self._parse_europeana_results(data, session)

                        for doc in documents:
                            yield doc

        except Exception as e:
            logger.error(f"Error searching Europeana: {e}")

    async def _search_adam_matthew(
        self, query: str, **kwargs
    ) -> AsyncGenerator[HistoricalDocument, None]:
        """Search Adam Matthew Digital collections"""
        try:
            await self._respect_rate_limit("adam_matthew")

            # Adam Matthew typically requires institutional access
            # This would be a mock implementation for demonstration
            collections = kwargs.get("collections", ["multidisciplinary_primary"])

            for collection in collections:
                # Mock search result
                mock_doc = HistoricalDocument(
                    document_id=f"adam_matthew_{hash(query)}",
                    title=f"Historical document related to {query}",
                    content="Mock content from Adam Matthew Digital collection",
                    original_language="en",
                    translated_content=None,
                    time_period="19th-20th century",
                    dynasty_era=None,
                    geographic_origin="Various",
                    document_type="Primary source",
                    historical_significance="High",
                    preservation_status="Digitized",
                    digitization_date=datetime.now(),
                    source_archive="Adam Matthew Digital",
                    catalog_number=f"AMD_{hash(query)}",
                    metadata={"collection": collection, "access": "institutional"},
                )
                yield mock_doc

        except Exception as e:
            logger.error(f"Error searching Adam Matthew: {e}")

    async def _parse_qdl_results(
        self, html_content: str, session: aiohttp.ClientSession
    ) -> list[HistoricalDocument]:
        """Parse Qatar Digital Library search results"""
        documents = []
        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Extract search results (structure depends on QDL website)
            result_items = soup.find_all("div", class_="search-result-item")

            for item in result_items:
                title_elem = item.find("h3") or item.find("a")
                title = title_elem.get_text(strip=True) if title_elem else "Untitled"

                # Extract metadata
                date_elem = item.find("span", class_="date")
                time_period = date_elem.get_text(strip=True) if date_elem else "Unknown period"

                # Get document content
                link_elem = item.find("a", href=True)
                if link_elem:
                    doc_url = urljoin(self.qatar_digital_library, link_elem["href"])
                    content = await self._fetch_document_content(doc_url, session)
                else:
                    content = item.get_text(strip=True)

                doc = HistoricalDocument(
                    document_id=f"qdl_{hash(title)}",
                    title=title,
                    content=content,
                    original_language="ar",  # Arabic is common in QDL
                    translated_content=None,
                    time_period=time_period,
                    dynasty_era=None,
                    geographic_origin="Middle East",
                    document_type="Historical record",
                    historical_significance="Medium",
                    preservation_status="Digitized",
                    digitization_date=datetime.now(),
                    source_archive="Qatar Digital Library",
                    catalog_number=f"QDL_{hash(title)}",
                    metadata={"source_url": doc_url if link_elem else ""},
                )
                documents.append(doc)

        except Exception as e:
            logger.error(f"Error parsing QDL results: {e}")

        return documents

    async def _parse_ctext_results(
        self, html_content: str, session: aiohttp.ClientSession
    ) -> list[HistoricalDocument]:
        """Parse Chinese Text Project search results"""
        documents = []
        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Extract search results from Chinese Text Project
            result_tables = soup.find_all("table")

            for table in result_tables:
                rows = table.find_all("tr")
                for row in rows:
                    cells = row.find_all("td")
                    if len(cells) >= 2:
                        title_cell = cells[0]
                        content_cell = cells[1]

                        title = title_cell.get_text(strip=True)
                        content = content_cell.get_text(strip=True)

                        # Determine dynasty/era from title or content
                        dynasty_era = self._identify_chinese_dynasty(title, content)

                        doc = HistoricalDocument(
                            document_id=f"ctext_{hash(title)}",
                            title=title,
                            content=content,
                            original_language="zh",
                            translated_content=None,  # Would need translation service
                            time_period=self._estimate_chinese_period(dynasty_era),
                            dynasty_era=dynasty_era,
                            geographic_origin="China",
                            document_type="Classical text",
                            historical_significance="High",
                            preservation_status="Digitized",
                            digitization_date=datetime.now(),
                            source_archive="Chinese Text Project",
                            catalog_number=f"CTEXT_{hash(title)}",
                            metadata={"dynasty": dynasty_era},
                        )
                        documents.append(doc)

        except Exception as e:
            logger.error(f"Error parsing Chinese Text Project results: {e}")

        return documents

    async def _parse_arcanum_results(
        self, html_content: str, session: aiohttp.ClientSession
    ) -> list[HistoricalDocument]:
        """Parse Arcanum database search results"""
        documents = []
        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Extract Arcanum search results
            articles = soup.find_all("article") or soup.find_all("div", class_="result")

            for article in articles:
                title_elem = article.find("h2") or article.find("h3") or article.find("a")
                title = title_elem.get_text(strip=True) if title_elem else "Untitled"

                content_elem = article.find("p") or article.find("div", class_="content")
                content = content_elem.get_text(strip=True) if content_elem else ""

                # Extract date information
                date_elem = article.find("span", class_="date") or article.find("time")
                time_period = date_elem.get_text(strip=True) if date_elem else "19th-20th century"

                doc = HistoricalDocument(
                    document_id=f"arcanum_{hash(title)}",
                    title=title,
                    content=content,
                    original_language="hu",  # Hungarian
                    translated_content=None,
                    time_period=time_period,
                    dynasty_era=None,
                    geographic_origin="Hungary/Central Europe",
                    document_type="Historical record",
                    historical_significance="Medium",
                    preservation_status="Digitized",
                    digitization_date=datetime.now(),
                    source_archive="Arcanum Database",
                    catalog_number=f"ARC_{hash(title)}",
                    metadata={"language": "hungarian", "region": "central_europe"},
                )
                documents.append(doc)

        except Exception as e:
            logger.error(f"Error parsing Arcanum results: {e}")

        return documents

    async def _parse_europeana_results(
        self, data: dict[str, Any], session: aiohttp.ClientSession
    ) -> list[HistoricalDocument]:
        """Parse Europeana API search results"""
        documents = []
        try:
            items = data.get("items", [])

            for item in items:
                title = item.get("title", ["Untitled"])[0] if item.get("title") else "Untitled"
                description = (
                    item.get("dcDescription", [""])[0] if item.get("dcDescription") else ""
                )

                # Extract date and geographic information
                date_info = item.get("year", ["Unknown"])[0] if item.get("year") else "Unknown"
                country = item.get("country", ["Unknown"])[0] if item.get("country") else "Unknown"

                # Get document type
                doc_type = item.get("type", "TEXT") if item.get("type") else "TEXT"

                doc = HistoricalDocument(
                    document_id=f"europeana_{item.get('id', hash(title))}",
                    title=title,
                    content=description,
                    original_language=(
                        item.get("language", ["en"])[0] if item.get("language") else "en"
                    ),
                    translated_content=None,
                    time_period=str(date_info),
                    dynasty_era=None,
                    geographic_origin=country,
                    document_type=doc_type.lower(),
                    historical_significance="Medium",
                    preservation_status="Digitized",
                    digitization_date=datetime.now(),
                    source_archive="Europeana",
                    catalog_number=item.get("id", f"EUR_{hash(title)}"),
                    metadata=item,
                )
                documents.append(doc)

        except Exception as e:
            logger.error(f"Error parsing Europeana results: {e}")

        return documents

    def _identify_chinese_dynasty(self, title: str, content: str) -> str | None:
        """Identify Chinese dynasty from title or content"""
        text = f"{title} {content}".lower()

        dynasties = {
            "tang": ["tang", "唐"],
            "song": ["song", "宋"],
            "ming": ["ming", "明"],
            "qing": ["qing", "ching", "清"],
            "han": ["han", "汉", "漢"],
            "yuan": ["yuan", "元"],
            "zhou": ["zhou", "周"],
        }

        for dynasty, keywords in dynasties.items():
            if any(keyword in text for keyword in keywords):
                return dynasty.capitalize()

        return None

    def _estimate_chinese_period(self, dynasty: str | None) -> str:
        """Estimate time period from Chinese dynasty"""
        if not dynasty:
            return "Ancient/Imperial China"

        dynasty_periods = {
            "Tang": "618-907 CE",
            "Song": "960-1279 CE",
            "Ming": "1368-1644 CE",
            "Qing": "1644-1912 CE",
            "Han": "206 BCE - 220 CE",
            "Yuan": "1271-1368 CE",
            "Zhou": "1046-256 BCE",
        }

        return dynasty_periods.get(dynasty, "Ancient/Imperial China")

    async def _fetch_document_content(self, url: str, session: aiohttp.ClientSession) -> str:
        """Fetch full document content from URL"""
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    # Basic content extraction (would need more sophisticated parsing)
                    soup = BeautifulSoup(content, "html.parser")

                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()

                    # Get text content
                    text = soup.get_text()

                    # Clean up whitespace
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = " ".join(chunk for chunk in chunks if chunk)

                    return text[:5000]  # Limit length
                logger.warning(f"Failed to fetch content from {url}: {response.status}")
                return ""
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {e}")
            return ""

    async def _respect_rate_limit(self, archive_name: str):
        """Implement rate limiting for different archives"""
        current_time = time.time()

        # Get rate limit for this archive
        rate_limit = self.rate_limits.get(archive_name, {}).get(
            "rate_limit", 15
        )  # Default 15 requests per minute
        min_delay = 60.0 / rate_limit

        # Check last request time
        last_request = self.last_request_times.get(archive_name, 0)
        time_since_last = current_time - last_request

        if time_since_last < min_delay:
            sleep_time = min_delay - time_since_last
            await asyncio.sleep(sleep_time)

        self.last_request_times[archive_name] = time.time()

    async def health_check(self) -> bool:
        """Check if historical archives are accessible"""
        try:
            async with aiohttp.ClientSession() as session:
                # Test Qatar Digital Library
                async with session.get(self.qatar_digital_library, timeout=10) as response:
                    if response.status not in [200, 403]:  # 403 might be normal for some archives
                        return False

            return True
        except:
            return False

    def get_supported_archives(self) -> list[str]:
        """Get list of supported historical archives"""
        return [
            "qatar_digital_library",
            "chinese_text_project",
            "adam_matthew_digital",
            "arcanum_database",
            "europeana",
            "archives_portal_europe",
        ]

    def get_archive_capabilities(self) -> dict[str, dict[str, Any]]:
        """Get capabilities and metadata for each archive"""
        return {
            "qatar_digital_library": {
                "regions": ["Middle East", "Gulf States"],
                "languages": ["Arabic", "English", "Persian"],
                "time_periods": ["Medieval", "Modern"],
                "document_types": ["manuscripts", "official_records", "maps"],
            },
            "chinese_text_project": {
                "regions": ["China", "East Asia"],
                "languages": ["Classical Chinese", "Modern Chinese"],
                "time_periods": ["Ancient", "Imperial", "Classical"],
                "document_types": ["philosophical_texts", "historical_records", "literature"],
            },
            "arcanum_database": {
                "regions": ["Hungary", "Central Europe"],
                "languages": ["Hungarian", "German", "Latin"],
                "time_periods": ["1800-present"],
                "document_types": ["newspapers", "periodicals", "books"],
            },
            "europeana": {
                "regions": ["Europe", "European colonies"],
                "languages": ["Multiple European languages"],
                "time_periods": ["Medieval to Modern"],
                "document_types": ["manuscripts", "books", "newspapers", "artworks"],
            },
        }
