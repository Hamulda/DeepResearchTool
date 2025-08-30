#!/usr/bin/env python3
"""
Archive Hunter for Deep Research Tool
Advanced scraper for hidden archives, forgotten databases and buried internet content

Author: Advanced IT Specialist
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import aiohttp
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
import json
from fake_useragent import UserAgent

logger = logging.getLogger(__name__)


class ArchiveHunter:
    """Advanced scraper for hidden and forgotten archives"""

    def __init__(self, rate_limit: float = 0.8):
        """
        Initialize Archive Hunter

        Args:
            rate_limit: Requests per second limit
        """
        self.rate_limit = rate_limit
        self.ua = UserAgent()

        # Hidden and alternative archive sources
        self.archive_sources = {
            "library_genesis": {
                "name": "Library Genesis",
                "urls": ["http://libgen.rs/", "http://libgen.is/", "http://gen.lib.rus.ec/"],
                "search_pattern": "/search.php?req={query}&open=0&res=25&view=simple&phrase=1&column=def",
                "type": "academic_books",
            },
            "sci_hub": {
                "name": "Sci-Hub",
                "urls": ["https://sci-hub.se/", "https://sci-hub.st/", "https://sci-hub.ru/"],
                "search_pattern": "/{query}",
                "type": "research_papers",
            },
            "memory_hole": {
                "name": "Memory Hole Archives",
                "urls": ["https://memoryhole.echelon.pl/", "https://archive.today/"],
                "search_pattern": "/search?q={query}",
                "type": "leaked_documents",
            },
            "cryptome": {
                "name": "Cryptome",
                "urls": ["https://cryptome.org/"],
                "search_pattern": "/search.htm?q={query}",
                "type": "intelligence_documents",
            },
            "wikileaks_archive": {
                "name": "WikiLeaks Archives",
                "urls": ["https://wikileaks.org/", "https://file.wikileaks.org/"],
                "search_pattern": "/search?q={query}",
                "type": "leaked_cables",
            },
            "foia_reading_rooms": {
                "name": "FOIA Reading Rooms",
                "urls": [
                    "https://www.cia.gov/readingroom/",
                    "https://vault.fbi.gov/",
                    "https://www.nsa.gov/news-features/declassified-documents/",
                    "https://www.defense.gov/News/Special-Reports/",
                ],
                "search_pattern": "/search?query={query}",
                "type": "declassified_documents",
            },
            "forgotten_databases": {
                "name": "Forgotten Academic Databases",
                "urls": [
                    "https://digital.library.unt.edu/",
                    "https://babel.hathitrust.org/",
                    "https://archive.org/details/texts",
                    "https://www.jstor.org/open/",
                ],
                "search_pattern": "/search?q={query}",
                "type": "academic_archives",
            },
            "conspiracy_archives": {
                "name": "Conspiracy Research Archives",
                "urls": [
                    "https://www.maryferrell.org/",
                    "https://history-matters.com/",
                    "https://www.aarclibrary.org/",
                    "https://jfkfacts.org/",
                ],
                "search_pattern": "/search?q={query}",
                "type": "conspiracy_research",
            },
            "deep_web_libraries": {
                "name": "Deep Web Academic Libraries",
                "urls": [
                    "https://www.worldcat.org/",
                    "https://scholar.google.com/",
                    "https://www.base-search.net/",
                    "https://core.ac.uk/",
                ],
                "search_pattern": "/search?q={query}",
                "type": "deep_academic",
            },
            "historical_newspapers": {
                "name": "Historical Newspaper Archives",
                "urls": [
                    "https://chroniclingamerica.loc.gov/",
                    "https://www.newspapers.com/",
                    "https://newspaperarchive.com/",
                    "https://news.google.com/newspapers",
                ],
                "search_pattern": "/search?q={query}",
                "type": "historical_news",
            },
            "government_archives": {
                "name": "Government Document Archives",
                "urls": [
                    "https://catalog.archives.gov/",
                    "https://www.govinfo.gov/",
                    "https://www.congress.gov/",
                    "https://fraser.stlouisfed.org/",
                ],
                "search_pattern": "/search?q={query}",
                "type": "government_docs",
            },
            "corporate_filings": {
                "name": "Corporate Filing Archives",
                "urls": [
                    "https://www.sec.gov/edgar/",
                    "https://www.opensecrets.org/",
                    "https://littlesis.org/",
                    "https://offshoreleaks.icij.org/",
                ],
                "search_pattern": "/search?q={query}",
                "type": "corporate_docs",
            },
        }

    async def search_async(
        self, topic: str, time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search across all hidden and alternative archives

        Args:
            topic: Search topic
            time_range: Optional time range for search

        Returns:
            List of documents from hidden archives
        """
        results = []
        search_keywords = self._extract_keywords(topic)

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            # Search each archive source
            for source_id, source_config in self.archive_sources.items():
                try:
                    logger.info(f"Searching {source_config['name']}...")
                    source_results = await self._search_archive_source(
                        session, source_id, source_config, search_keywords, time_range
                    )
                    results.extend(source_results)

                    # Rate limiting
                    await asyncio.sleep(1.0 / self.rate_limit)

                except Exception as e:
                    logger.error(f"Error searching {source_config['name']}: {e}")
                    continue

        # Sort by relevance and remove duplicates
        unique_results = self._deduplicate_results(results)
        unique_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        logger.info(f"Archive Hunter found {len(unique_results)} documents from hidden sources")
        return unique_results[:100]  # Limit results

    def search(
        self, topic: str, time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> List[Dict[str, Any]]:
        """Synchronous search method"""
        return asyncio.run(self.search_async(topic, time_range))

    def _extract_keywords(self, topic: str) -> List[str]:
        """Extract search keywords from topic"""
        # Clean topic and extract meaningful terms
        keywords = []

        # Add full topic
        keywords.append(topic.strip())

        # Extract individual words (3+ chars)
        words = re.findall(r"\b\w{3,}\b", topic.lower())
        stop_words = {
            "the",
            "and",
            "for",
            "are",
            "but",
            "not",
            "you",
            "all",
            "can",
            "had",
            "her",
            "was",
            "one",
            "our",
            "out",
            "day",
            "get",
            "has",
            "him",
            "his",
            "how",
            "its",
            "may",
            "new",
            "now",
            "old",
            "see",
            "two",
            "way",
            "who",
            "boy",
            "did",
            "has",
            "let",
            "put",
            "say",
            "she",
            "too",
            "use",
        }
        meaningful_words = [word for word in words if word not in stop_words]
        keywords.extend(meaningful_words[:10])

        return keywords

    async def _search_archive_source(
        self,
        session: aiohttp.ClientSession,
        source_id: str,
        source_config: Dict[str, Any],
        keywords: List[str],
        time_range: Optional[Tuple[datetime, datetime]],
    ) -> List[Dict[str, Any]]:
        """Search individual archive source"""
        results = []

        for base_url in source_config["urls"]:
            try:
                # Try different search approaches
                for keyword in keywords[:3]:  # Limit to first 3 keywords
                    search_results = await self._perform_archive_search(
                        session, base_url, source_config, keyword, time_range
                    )
                    results.extend(search_results)

                    if len(results) >= 20:  # Limit per source
                        break

                if results:
                    break  # If we found results, don't try other URLs

            except Exception as e:
                logger.debug(f"Failed to search {base_url}: {e}")
                continue

        # Add source metadata
        for result in results:
            result["archive_source"] = source_config["name"]
            result["archive_type"] = source_config["type"]
            result["source"] = "hidden_archive"
            result["source_type"] = "archive"

        return results

    async def _perform_archive_search(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        source_config: Dict[str, Any],
        keyword: str,
        time_range: Optional[Tuple[datetime, datetime]],
    ) -> List[Dict[str, Any]]:
        """Perform actual search on archive"""
        results = []

        try:
            # Construct search URL
            search_pattern = source_config.get("search_pattern", "/search?q={query}")
            search_url = base_url.rstrip("/") + search_pattern.format(
                query=keyword.replace(" ", "+")
            )

            headers = {
                "User-Agent": self.ua.random,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Referer": base_url,
            }

            async with session.get(search_url, headers=headers) as response:
                if response.status == 200:
                    content = await response.text()
                    results = await self._parse_archive_results(
                        content, base_url, source_config, keyword
                    )

        except Exception as e:
            logger.debug(f"Search failed for {base_url}: {e}")

        return results

    async def _parse_archive_results(
        self, html_content: str, base_url: str, source_config: Dict[str, Any], keyword: str
    ) -> List[Dict[str, Any]]:
        """Parse search results from archive HTML"""
        results = []

        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Different parsing strategies based on archive type
            archive_type = source_config["type"]

            if archive_type == "academic_books":
                results = self._parse_library_genesis(soup, base_url, keyword)
            elif archive_type == "research_papers":
                results = self._parse_research_papers(soup, base_url, keyword)
            elif archive_type == "leaked_documents":
                results = self._parse_leaked_documents(soup, base_url, keyword)
            elif archive_type == "declassified_documents":
                results = self._parse_government_docs(soup, base_url, keyword)
            elif archive_type == "conspiracy_research":
                results = self._parse_conspiracy_archives(soup, base_url, keyword)
            else:
                results = self._parse_generic_archive(soup, base_url, keyword)

        except Exception as e:
            logger.debug(f"Failed to parse results from {base_url}: {e}")

        return results

    def _parse_library_genesis(
        self, soup: BeautifulSoup, base_url: str, keyword: str
    ) -> List[Dict[str, Any]]:
        """Parse Library Genesis search results"""
        results = []

        # Look for book entries
        book_rows = soup.find_all("tr", limit=20)

        for row in book_rows:
            try:
                cells = row.find_all("td")
                if len(cells) >= 5:
                    title_cell = cells[2] if len(cells) > 2 else None
                    author_cell = cells[1] if len(cells) > 1 else None

                    if title_cell and author_cell:
                        title = title_cell.get_text(strip=True)
                        author = author_cell.get_text(strip=True)

                        if title and len(title) > 5:
                            # Calculate relevance
                            relevance = self._calculate_relevance(title + " " + author, keyword)

                            if relevance > 0.1:
                                results.append(
                                    {
                                        "title": title,
                                        "content": f"Book by {author}: {title}",
                                        "url": base_url,
                                        "date": datetime.now(),
                                        "relevance_score": relevance,
                                        "metadata": {
                                            "author": author,
                                            "type": "academic_book",
                                            "source_archive": "Library Genesis",
                                        },
                                    }
                                )
            except Exception as e:
                continue

        return results

    def _parse_research_papers(
        self, soup: BeautifulSoup, base_url: str, keyword: str
    ) -> List[Dict[str, Any]]:
        """Parse research paper search results"""
        results = []

        # Look for paper links and titles
        paper_links = soup.find_all(
            ["a", "div"], class_=re.compile(r"title|paper|result"), limit=20
        )

        for link in paper_links:
            try:
                title = link.get_text(strip=True)
                href = link.get("href", "")

                if title and len(title) > 10:
                    relevance = self._calculate_relevance(title, keyword)

                    if relevance > 0.1:
                        full_url = urljoin(base_url, href) if href else base_url

                        results.append(
                            {
                                "title": title,
                                "content": title,
                                "url": full_url,
                                "date": datetime.now(),
                                "relevance_score": relevance,
                                "metadata": {"type": "research_paper", "source_archive": "Sci-Hub"},
                            }
                        )
            except Exception as e:
                continue

        return results

    def _parse_leaked_documents(
        self, soup: BeautifulSoup, base_url: str, keyword: str
    ) -> List[Dict[str, Any]]:
        """Parse leaked document search results"""
        results = []

        # Look for document entries
        doc_elements = soup.find_all(["div", "p", "li"], limit=30)

        for element in doc_elements:
            try:
                text = element.get_text(strip=True)
                links = element.find_all("a")

                if text and len(text) > 20:
                    relevance = self._calculate_relevance(text, keyword)

                    if relevance > 0.2:
                        doc_url = base_url
                        if links:
                            href = links[0].get("href", "")
                            if href:
                                doc_url = urljoin(base_url, href)

                        results.append(
                            {
                                "title": text[:100] + "..." if len(text) > 100 else text,
                                "content": text,
                                "url": doc_url,
                                "date": datetime.now(),
                                "relevance_score": relevance,
                                "metadata": {
                                    "type": "leaked_document",
                                    "source_archive": "Memory Hole",
                                },
                            }
                        )
            except Exception as e:
                continue

        return results

    def _parse_government_docs(
        self, soup: BeautifulSoup, base_url: str, keyword: str
    ) -> List[Dict[str, Any]]:
        """Parse government document search results"""
        results = []

        # Look for document titles and links
        doc_links = soup.find_all("a", href=True, limit=25)

        for link in doc_links:
            try:
                title = link.get_text(strip=True)
                href = link.get("href", "")

                # Filter for document-like content
                if (
                    title
                    and len(title) > 15
                    and any(
                        word in title.lower()
                        for word in ["document", "report", "file", "memo", "cable", "classified"]
                    )
                ):

                    relevance = self._calculate_relevance(title, keyword)

                    if relevance > 0.15:
                        full_url = urljoin(base_url, href)

                        results.append(
                            {
                                "title": title,
                                "content": title,
                                "url": full_url,
                                "date": datetime.now(),
                                "relevance_score": relevance,
                                "metadata": {
                                    "type": "government_document",
                                    "source_archive": "Government Archives",
                                },
                            }
                        )
            except Exception as e:
                continue

        return results

    def _parse_conspiracy_archives(
        self, soup: BeautifulSoup, base_url: str, keyword: str
    ) -> List[Dict[str, Any]]:
        """Parse conspiracy research archive results"""
        results = []

        # Look for research documents and articles
        content_elements = soup.find_all(["article", "div", "section"], limit=20)

        for element in content_elements:
            try:
                title_elem = element.find(["h1", "h2", "h3", "h4", "title"])
                if not title_elem:
                    continue

                title = title_elem.get_text(strip=True)
                content = element.get_text(strip=True)[:500]  # First 500 chars

                if title and len(title) > 10:
                    relevance = self._calculate_relevance(title + " " + content, keyword)

                    if relevance > 0.2:
                        # Try to find associated link
                        doc_url = base_url
                        link = element.find("a", href=True)
                        if link:
                            href = link.get("href", "")
                            if href:
                                doc_url = urljoin(base_url, href)

                        results.append(
                            {
                                "title": title,
                                "content": content,
                                "url": doc_url,
                                "date": datetime.now(),
                                "relevance_score": relevance,
                                "metadata": {
                                    "type": "conspiracy_research",
                                    "source_archive": "Conspiracy Archives",
                                },
                            }
                        )
            except Exception as e:
                continue

        return results

    def _parse_generic_archive(
        self, soup: BeautifulSoup, base_url: str, keyword: str
    ) -> List[Dict[str, Any]]:
        """Generic parser for unknown archive types"""
        results = []

        # Look for any content that might be relevant
        all_links = soup.find_all("a", href=True, limit=30)

        for link in all_links:
            try:
                title = link.get_text(strip=True)
                href = link.get("href", "")

                if title and len(title) > 10:
                    relevance = self._calculate_relevance(title, keyword)

                    if relevance > 0.1:
                        full_url = urljoin(base_url, href)

                        results.append(
                            {
                                "title": title,
                                "content": title,
                                "url": full_url,
                                "date": datetime.now(),
                                "relevance_score": relevance,
                                "metadata": {
                                    "type": "archive_document",
                                    "source_archive": "Generic Archive",
                                },
                            }
                        )
            except Exception as e:
                continue

        return results

    def _calculate_relevance(self, text: str, keyword: str) -> float:
        """Calculate relevance score"""
        if not text or not keyword:
            return 0.0

        text_lower = text.lower()
        keyword_lower = keyword.lower()

        # Exact match gets high score
        if keyword_lower in text_lower:
            return 0.8

        # Check for individual words
        keyword_words = keyword_lower.split()
        text_words = text_lower.split()

        matches = sum(1 for word in keyword_words if word in text_words)
        if matches > 0:
            return min(0.7, matches / len(keyword_words))

        return 0.0

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results"""
        seen_titles = set()
        unique_results = []

        for result in results:
            title = result.get("title", "").lower().strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_results.append(result)

        return unique_results

    async def search_specific_archive(self, archive_name: str, topic: str) -> List[Dict[str, Any]]:
        """Search specific archive by name"""
        if archive_name not in self.archive_sources:
            logger.error(f"Unknown archive: {archive_name}")
            return []

        source_config = self.archive_sources[archive_name]
        keywords = self._extract_keywords(topic)

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            results = await self._search_archive_source(
                session, archive_name, source_config, keywords, None
            )

        return results

    def get_available_archives(self) -> List[Dict[str, str]]:
        """Get list of available archive sources"""
        return [
            {
                "id": source_id,
                "name": config["name"],
                "type": config["type"],
                "description": f"Archive for {config['type'].replace('_', ' ')}",
            }
            for source_id, config in self.archive_sources.items()
        ]
