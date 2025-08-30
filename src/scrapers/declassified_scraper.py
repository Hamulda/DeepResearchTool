#!/usr/bin/env python3
"""Declassified Documents Scraper
Access to CIA CREST Database, National Security Archive, and other declassified sources

Author: Advanced IT Specialist
"""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
import logging
import re
import time
from typing import Any
from urllib.parse import urljoin
import xml.etree.ElementTree as ET

import aiohttp

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


@dataclass
class DeclassifiedDocument:
    """Represents a declassified document"""

    document_id: str
    title: str
    content: str
    classification_level: str
    original_classification: str
    declassification_date: datetime
    creation_date: datetime | None
    agency: str
    document_type: str
    topics: list[str]
    entities: list[str]
    redacted_sections: list[str]
    source_url: str
    confidence_score: float


class DeclassifiedScraper(BaseScraper):
    """Scraper for declassified government documents"""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.name = "declassified_scraper"
        self.cia_api_key = config.get("cia_api_key")
        self.rate_limit = config.get("rate_limit", 6)  # requests per minute
        self.max_results = config.get("max_results", 1000)
        self.classification_levels = config.get(
            "classification_levels", ["DECLASSIFIED", "UNCLASSIFIED"]
        )

        # API endpoints
        self.cia_crest_base = "https://www.cia.gov/readingroom/search/site/"
        self.national_archives_api = "https://catalog.archives.gov/api/v1/"
        self.nsa_base = "https://nsarchive2.gwu.edu/"

        # Request tracking for rate limiting
        self.last_request_time = 0
        self.request_count = 0
        self.minute_start = time.time()

    async def search(self, query: str, **kwargs) -> AsyncGenerator[DeclassifiedDocument, None]:
        """Search across multiple declassified document sources"""
        logger.info(f"Searching declassified documents for: {query}")

        # Search CIA CREST Database
        async for doc in self._search_cia_crest(query, **kwargs):
            yield doc

        # Search National Archives
        async for doc in self._search_national_archives(query, **kwargs):
            yield doc

        # Search National Security Archive
        async for doc in self._search_nsa(query, **kwargs):
            yield doc

    async def _search_cia_crest(
        self, query: str, **kwargs
    ) -> AsyncGenerator[DeclassifiedDocument, None]:
        """Search CIA CREST Database - 10+ million pages of declassified documents"""
        try:
            await self._respect_rate_limit()

            # CIA CREST search parameters
            params = {
                "keys": query,
                "format": "json",
                "sort_by": "created",
                "sort_order": "DESC",
                "items_per_page": 50,
            }

            # Add classification level filters
            if self.classification_levels:
                params["f[0]"] = f"field_classification:({' OR '.join(self.classification_levels)})"

            # Add date range if specified
            date_range = kwargs.get("date_range")
            if date_range:
                params["f[1]"] = f"created:[{date_range['start']} TO {date_range['end']}]"

            async with aiohttp.ClientSession() as session:
                async with session.get(self.cia_crest_base, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        for item in data.get("list", []):
                            doc = await self._parse_cia_document(item, session)
                            if doc:
                                yield doc
                    else:
                        logger.error(f"CIA CREST search failed: {response.status}")

        except Exception as e:
            logger.error(f"Error searching CIA CREST: {e}")

    async def _search_national_archives(
        self, query: str, **kwargs
    ) -> AsyncGenerator[DeclassifiedDocument, None]:
        """Search National Archives - US Serial Set, military records, etc."""
        try:
            await self._respect_rate_limit()

            search_url = urljoin(self.national_archives_api, "search")
            params = {"q": query, "rows": 50, "type": "description", "format": "json"}

            # Add collection filters
            collections = kwargs.get("collections", ["us_serial_set", "fold3_military"])
            if collections:
                params["f.parentNaId"] = collections

            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        for item in (
                            data.get("opaResponse", {}).get("results", {}).get("result", [])
                        ):
                            doc = await self._parse_national_archives_document(item, session)
                            if doc:
                                yield doc
                    else:
                        logger.error(f"National Archives search failed: {response.status}")

        except Exception as e:
            logger.error(f"Error searching National Archives: {e}")

    async def _search_nsa(self, query: str, **kwargs) -> AsyncGenerator[DeclassifiedDocument, None]:
        """Search National Security Archive - 100,000+ declassified records"""
        try:
            await self._respect_rate_limit()

            # NSA uses different search mechanism - web scraping approach
            search_url = f"{self.nsa_base}search/"
            params = {"q": query, "sort": "date", "format": "rss"}  # Use RSS for structured data

            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params) as response:
                    if response.status == 200:
                        content = await response.text()
                        docs = await self._parse_nsa_rss(content, session)
                        for doc in docs:
                            yield doc
                    else:
                        logger.error(f"NSA search failed: {response.status}")

        except Exception as e:
            logger.error(f"Error searching NSA: {e}")

    async def _parse_cia_document(
        self, item: dict, session: aiohttp.ClientSession
    ) -> DeclassifiedDocument | None:
        """Parse CIA CREST document from API response"""
        try:
            # Extract basic metadata
            doc_id = item.get("nid", "")
            title = item.get("title", "Untitled Document")

            # Get full document content
            content_url = item.get("url", "")
            content = await self._fetch_document_content(content_url, session)

            # Parse classification information
            classification = item.get("field_classification", "UNKNOWN")
            original_classification = item.get("field_original_classification", classification)

            # Parse dates
            creation_date = self._parse_date(item.get("created"))
            declassification_date = self._parse_date(item.get("field_declassification_date"))

            # Extract entities and topics using regex patterns
            entities = self._extract_entities(content)
            topics = self._extract_topics(content, title)
            redacted_sections = self._detect_redactions(content)

            return DeclassifiedDocument(
                document_id=doc_id,
                title=title,
                content=content,
                classification_level=classification,
                original_classification=original_classification,
                declassification_date=declassification_date,
                creation_date=creation_date,
                agency="CIA",
                document_type=item.get("type", "Unknown"),
                topics=topics,
                entities=entities,
                redacted_sections=redacted_sections,
                source_url=content_url,
                confidence_score=self._calculate_confidence_score(item, content),
            )

        except Exception as e:
            logger.error(f"Error parsing CIA document: {e}")
            return None

    async def _parse_national_archives_document(
        self, item: dict, session: aiohttp.ClientSession
    ) -> DeclassifiedDocument | None:
        """Parse National Archives document"""
        try:
            description = item.get("description", {})

            doc_id = description.get("naId", "")
            title = description.get("title", "Untitled Document")

            # Get document content if available
            content = ""
            digital_objects = description.get("objects", {}).get("object", [])
            if digital_objects:
                # Try to get text content from digital objects
                for obj in digital_objects[:3]:  # Limit to first 3 objects
                    if obj.get("file", {}).get("@mime") == "text/plain":
                        content_url = obj.get("file", {}).get("@url", "")
                        if content_url:
                            obj_content = await self._fetch_document_content(content_url, session)
                            content += obj_content + "\n\n"

            # Parse metadata
            creation_date = self._parse_date(description.get("productionDate"))

            entities = self._extract_entities(content)
            topics = self._extract_topics(content, title)

            return DeclassifiedDocument(
                document_id=doc_id,
                title=title,
                content=content,
                classification_level="DECLASSIFIED",  # National Archives are typically declassified
                original_classification="UNKNOWN",
                declassification_date=None,
                creation_date=creation_date,
                agency="NARA",
                document_type=description.get("type", "Unknown"),
                topics=topics,
                entities=entities,
                redacted_sections=[],
                source_url=f"https://catalog.archives.gov/id/{doc_id}",
                confidence_score=self._calculate_confidence_score(item, content),
            )

        except Exception as e:
            logger.error(f"Error parsing National Archives document: {e}")
            return None

    async def _parse_nsa_rss(
        self, rss_content: str, session: aiohttp.ClientSession
    ) -> list[DeclassifiedDocument]:
        """Parse National Security Archive RSS feed"""
        documents = []
        try:
            root = ET.fromstring(rss_content)

            for item in root.findall(".//item"):
                title = item.find("title").text if item.find("title") is not None else "Untitled"
                description = (
                    item.find("description").text if item.find("description") is not None else ""
                )
                link = item.find("link").text if item.find("link") is not None else ""
                pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""

                # Fetch full content
                content = await self._fetch_document_content(link, session)

                # Extract information
                entities = self._extract_entities(content)
                topics = self._extract_topics(content, title)

                doc = DeclassifiedDocument(
                    document_id=f"nsa_{hash(link)}",
                    title=title,
                    content=content,
                    classification_level="DECLASSIFIED",
                    original_classification="UNKNOWN",
                    declassification_date=self._parse_date(pub_date),
                    creation_date=None,
                    agency="NSA",
                    document_type="Archive Document",
                    topics=topics,
                    entities=entities,
                    redacted_sections=self._detect_redactions(content),
                    source_url=link,
                    confidence_score=0.8,  # Default confidence for NSA documents
                )
                documents.append(doc)

        except Exception as e:
            logger.error(f"Error parsing NSA RSS: {e}")

        return documents

    def _extract_entities(self, content: str) -> list[str]:
        """Extract named entities from document content"""
        entities = []

        # Common entity patterns for declassified documents
        patterns = {
            "person": r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",
            "organization": r"\b(?:CIA|FBI|NSA|DOD|State Department|Pentagon|White House)\b",
            "location": r"\b[A-Z][a-z]+(?:, [A-Z][a-z]+)*\b",
            "date": r"\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b",
            "operation": r"\bOperation [A-Z][a-z]+\b",
            "classification": r"\b(?:TOP SECRET|SECRET|CONFIDENTIAL|CLASSIFIED)\b",
        }

        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                entities.append(f"{entity_type}:{match}")

        return list(set(entities))  # Remove duplicates

    def _extract_topics(self, content: str, title: str) -> list[str]:
        """Extract topics and themes from document"""
        topics = []
        text = f"{title} {content}".lower()

        # Topic keywords for declassified documents
        topic_keywords = {
            "intelligence": ["intelligence", "spy", "surveillance", "covert"],
            "military": ["military", "army", "navy", "air force", "operation"],
            "diplomatic": ["diplomatic", "embassy", "ambassador", "foreign"],
            "cold_war": ["cold war", "soviet", "ussr", "communist"],
            "terrorism": ["terrorist", "terrorism", "threat", "security"],
            "nuclear": ["nuclear", "atomic", "missile", "weapons"],
            "economic": ["economic", "trade", "financial", "sanctions"],
        }

        for topic, keywords in topic_keywords.items():
            if any(keyword in text for keyword in keywords):
                topics.append(topic)

        return topics

    def _detect_redactions(self, content: str) -> list[str]:
        """Detect redacted sections in document content"""
        redaction_patterns = [
            r"\[REDACTED\]",
            r"\[CLASSIFIED\]",
            r"\[DELETED\]",
            r"█+",  # Black bars
            r"▆+",  # Redaction blocks
            r"\*{3,}",  # Multiple asterisks
            r"_{3,}",  # Multiple underscores
            r"X{3,}",  # Multiple X's
        ]

        redacted_sections = []
        for pattern in redaction_patterns:
            matches = re.findall(pattern, content)
            redacted_sections.extend(matches)

        return redacted_sections

    def _calculate_confidence_score(self, metadata: dict, content: str) -> float:
        """Calculate confidence score for document authenticity"""
        score = 0.5  # Base score

        # Increase score based on metadata completeness
        if metadata.get("nid") or metadata.get("naId"):
            score += 0.1
        if metadata.get("title"):
            score += 0.1
        if metadata.get("created") or metadata.get("productionDate"):
            score += 0.1

        # Increase score based on content characteristics
        if len(content) > 500:
            score += 0.1
        if self._detect_redactions(content):
            score += 0.1  # Redactions indicate authentic classified document

        return min(score, 1.0)

    async def _respect_rate_limit(self):
        """Implement rate limiting to respect API constraints"""
        current_time = time.time()

        # Reset counter every minute
        if current_time - self.minute_start >= 60:
            self.request_count = 0
            self.minute_start = current_time

        # Check if we've exceeded rate limit
        if self.request_count >= self.rate_limit:
            sleep_time = 60 - (current_time - self.minute_start)
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
                self.request_count = 0
                self.minute_start = time.time()

        # Ensure minimum delay between requests
        time_since_last = current_time - self.last_request_time
        min_delay = 60.0 / self.rate_limit  # Seconds per request

        if time_since_last < min_delay:
            sleep_time = min_delay - time_since_last
            await asyncio.sleep(sleep_time)

        self.request_count += 1
        self.last_request_time = time.time()

    async def _fetch_document_content(self, url: str, session: aiohttp.ClientSession) -> str:
        """Fetch full document content from URL"""
        try:
            if not url:
                return ""

            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    return self._clean_content(content)
                logger.warning(f"Failed to fetch content from {url}: {response.status}")
                return ""
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {e}")
            return ""

    def _clean_content(self, content: str) -> str:
        """Clean and normalize document content"""
        # Remove HTML tags if present
        content = re.sub(r"<[^>]+>", "", content)

        # Normalize whitespace
        content = re.sub(r"\s+", " ", content)

        # Remove excessive line breaks
        content = re.sub(r"\n\s*\n", "\n\n", content)

        return content.strip()

    def _parse_date(self, date_str: str) -> datetime | None:
        """Parse various date formats found in declassified documents"""
        if not date_str:
            return None

        date_formats = [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%Y-%m-%d %H:%M:%S",
            "%a, %d %b %Y %H:%M:%S %Z",  # RSS format
        ]

        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        logger.warning(f"Could not parse date: {date_str}")
        return None

    async def health_check(self) -> bool:
        """Check if declassified sources are accessible"""
        try:
            async with aiohttp.ClientSession() as session:
                # Test CIA CREST
                async with session.get(self.cia_crest_base, timeout=10) as response:
                    if response.status != 200:
                        return False

                # Test National Archives
                async with session.get(self.national_archives_api, timeout=10) as response:
                    if response.status != 200:
                        return False

            return True
        except:
            return False
