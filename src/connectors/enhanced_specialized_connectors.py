#!/usr/bin/env python3
"""Enhanced Specialized Source Connectors pro FÁZI 3
Common Crawl, Memento, Ahmia/Tor, OpenAlex→Crossref→Unpaywall→Europe PMC, Legal APIs

Author: Senior Python/MLOps Agent
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import logging
import re
import time
from typing import Any
from urllib.parse import urlparse

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class SourceResult:
    """Standardní struktura výsledku ze source"""

    id: str
    title: str
    content: str
    url: str
    source_type: str
    metadata: dict[str, Any]
    timestamp: datetime | None = None
    confidence: float = 1.0
    retrieval_method: str = "direct"


@dataclass
class TemporalDiff:
    """Temporal diff mezi verzemi dokumentu"""

    url: str
    earlier_timestamp: datetime
    later_timestamp: datetime
    diff_type: str  # "content_change", "structure_change", "metadata_change"
    changes: list[dict[str, Any]]
    impact_assessment: str
    confidence: float


class CommonCrawlConnector:
    """Enhanced Common Crawl connector s stabilními WARC offsety"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.base_url = "https://index.commoncrawl.org"
        self.session = None
        self.stats = {
            "queries_performed": 0,
            "documents_retrieved": 0,
            "warc_offsets_cached": 0,
            "avg_retrieval_time": 0.0,
        }
        self.warc_cache = {}  # Cache pro WARC offsety

    async def initialize(self):
        """Inicializace konektoru"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                "User-Agent": self.config.get("user_agent", "DeepResearchTool/3.0 (Research)")
            },
        )

    async def search(self, query: str, max_results: int = 20) -> list[SourceResult]:
        """Vyhledávání v Common Crawl s WARC offsety"""
        if not self.session:
            await self.initialize()

        start_time = time.time()

        try:
            # Search Common Crawl index
            search_results = await self._search_cc_index(query, max_results)

            # Retrieve documents with stable WARC offsety
            documents = []
            for result in search_results:
                doc = await self._retrieve_with_warc_offset(result)
                if doc:
                    documents.append(doc)

            elapsed_time = time.time() - start_time
            self._update_stats(len(documents), elapsed_time)

            logger.info(f"Common Crawl retrieved {len(documents)} documents in {elapsed_time:.2f}s")

            return documents

        except Exception as e:
            logger.error(f"Common Crawl search failed: {e}")
            return []

    async def _search_cc_index(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """Vyhledávání v CC indexu"""
        # Use latest available index
        index_url = f"{self.base_url}/CC-MAIN-2023-40-index"

        search_params = {
            "url": f"*.{query.replace(' ', '*')}*",
            "output": "json",
            "limit": max_results,
        }

        try:
            async with self.session.get(index_url, params=search_params) as response:
                if response.status == 200:
                    text_data = await response.text()
                    # Parse NDJSON format
                    results = []
                    for line in text_data.strip().split("\n"):
                        if line:
                            try:
                                result = json.loads(line)
                                results.append(result)
                            except json.JSONDecodeError:
                                continue
                    return results
                logger.warning(f"CC index search failed: HTTP {response.status}")
                return []

        except Exception as e:
            logger.error(f"CC index search error: {e}")
            return []

    async def _retrieve_with_warc_offset(self, cc_result: dict[str, Any]) -> SourceResult | None:
        """Retrieve dokument s WARC offset pro idempotenci"""
        try:
            url = cc_result.get("url", "")
            warc_filename = cc_result.get("filename", "")
            warc_offset = cc_result.get("offset", 0)
            warc_length = cc_result.get("length", 0)

            # Create stable ID from WARC coordinates
            stable_id = hashlib.md5(
                f"{warc_filename}:{warc_offset}:{warc_length}".encode()
            ).hexdigest()

            # Check cache first
            if stable_id in self.warc_cache:
                cached_result = self.warc_cache[stable_id]
                logger.debug(f"Using cached WARC result for {stable_id}")
                return cached_result

            # Retrieve from WARC
            warc_url = f"https://commoncrawl.s3.amazonaws.com/{warc_filename}"
            headers = {"Range": f"bytes={warc_offset}-{warc_offset + warc_length - 1}"}

            async with self.session.get(warc_url, headers=headers) as response:
                if response.status == 206:  # Partial Content
                    warc_data = await response.read()

                    # Parse WARC record
                    content = self._parse_warc_record(warc_data)

                    if content:
                        result = SourceResult(
                            id=stable_id,
                            title=self._extract_title(content),
                            content=content,
                            url=url,
                            source_type="common_crawl",
                            metadata={
                                "warc_filename": warc_filename,
                                "warc_offset": warc_offset,
                                "warc_length": warc_length,
                                "crawl_timestamp": cc_result.get("timestamp", ""),
                                "mime_type": cc_result.get("mime", ""),
                                "status": cc_result.get("status", ""),
                            },
                            retrieval_method="warc_offset",
                        )

                        # Cache result
                        self.warc_cache[stable_id] = result
                        self.stats["warc_offsets_cached"] += 1

                        return result

        except Exception as e:
            logger.warning(f"WARC retrieval failed for {cc_result.get('url', 'unknown')}: {e}")

        return None

    def _parse_warc_record(self, warc_data: bytes) -> str | None:
        """Parse WARC record a extrahuj content"""
        try:
            # Simple WARC parsing - look for HTML content after headers
            warc_text = warc_data.decode("utf-8", errors="ignore")

            # Find double newline that separates headers from content
            content_start = warc_text.find("\r\n\r\n")
            if content_start == -1:
                content_start = warc_text.find("\n\n")

            if content_start != -1:
                content = warc_text[content_start + 4 :]

                # Extract text from HTML
                text_content = self._extract_text_from_html(content)
                return text_content

        except Exception as e:
            logger.debug(f"WARC parsing error: {e}")

        return None

    def _extract_text_from_html(self, html_content: str) -> str:
        """Extrahuj text z HTML (zjednodušená verze)"""
        # Remove script and style elements
        html_content = re.sub(
            r"<script[^>]*>.*?</script>", "", html_content, flags=re.DOTALL | re.IGNORECASE
        )
        html_content = re.sub(
            r"<style[^>]*>.*?</style>", "", html_content, flags=re.DOTALL | re.IGNORECASE
        )

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", html_content)

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _extract_title(self, content: str) -> str:
        """Extrahuj title z obsahu"""
        # Look for HTML title tag
        title_match = re.search(r"<title[^>]*>(.*?)</title>", content, re.IGNORECASE | re.DOTALL)
        if title_match:
            title = title_match.group(1).strip()
            title = re.sub(r"\s+", " ", title)
            return title[:200]  # Limit title length

        # Fallback: use first line
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:
                return line[:100]

        return "Untitled Document"

    def _update_stats(self, docs_retrieved: int, elapsed_time: float):
        """Aktualizuje statistiky"""
        self.stats["queries_performed"] += 1
        self.stats["documents_retrieved"] += docs_retrieved

        # Exponential moving average
        alpha = 0.1
        if self.stats["avg_retrieval_time"] == 0:
            self.stats["avg_retrieval_time"] = elapsed_time
        else:
            self.stats["avg_retrieval_time"] = (
                alpha * elapsed_time + (1 - alpha) * self.stats["avg_retrieval_time"]
            )

    async def close(self):
        """Zavření konektoru"""
        if self.session:
            await self.session.close()


class MementoTemporalConnector:
    """Enhanced Memento connector s TimeMap→TimeGate orchestrací a diff analýzou"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.memento_aggregator = "http://timetravel.mementoweb.org"
        self.session = None
        self.stats = {
            "temporal_queries": 0,
            "snapshots_retrieved": 0,
            "diffs_generated": 0,
            "avg_diff_time": 0.0,
        }

    async def initialize(self):
        """Inicializace konektoru"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            headers={
                "User-Agent": self.config.get("user_agent", "DeepResearchTool/3.0 (Research)")
            },
        )

    async def search_temporal(
        self, url: str, date_range: tuple[datetime, datetime], max_snapshots: int = 10
    ) -> list[SourceResult]:
        """Temporální vyhledávání s diff analýzou"""
        if not self.session:
            await self.initialize()

        start_time = time.time()

        try:
            # Get TimeMap for URL
            timemap = await self._get_timemap(url, date_range)

            # Select snapshots for analysis
            selected_snapshots = self._select_snapshots(timemap, max_snapshots)

            # Retrieve snapshot content
            snapshots = []
            for snapshot_info in selected_snapshots:
                snapshot = await self._retrieve_snapshot(snapshot_info)
                if snapshot:
                    snapshots.append(snapshot)

            # Generate temporal diffs
            if len(snapshots) >= 2:
                diffs = await self._generate_temporal_diffs(snapshots)

                # Add diff metadata to snapshots
                for snapshot in snapshots:
                    snapshot.metadata["temporal_diffs"] = [
                        diff for diff in diffs if diff.url == snapshot.url
                    ]

            elapsed_time = time.time() - start_time
            self._update_stats(len(snapshots), elapsed_time)

            logger.info(
                f"Memento retrieved {len(snapshots)} temporal snapshots in {elapsed_time:.2f}s"
            )

            return snapshots

        except Exception as e:
            logger.error(f"Memento temporal search failed: {e}")
            return []

    async def _get_timemap(
        self, url: str, date_range: tuple[datetime, datetime]
    ) -> list[dict[str, Any]]:
        """Získá TimeMap pro URL"""
        start_date, end_date = date_range

        timemap_url = f"{self.memento_aggregator}/timemap/json/{url}"

        try:
            async with self.session.get(timemap_url) as response:
                if response.status == 200:
                    timemap_data = await response.json()

                    # Filter by date range
                    filtered_mementos = []
                    for memento in timemap_data.get("mementos", {}).get("list", []):
                        memento_datetime = self._parse_memento_datetime(memento.get("datetime", ""))

                        if memento_datetime and start_date <= memento_datetime <= end_date:
                            filtered_mementos.append(
                                {
                                    "url": memento.get("uri", ""),
                                    "datetime": memento_datetime,
                                    "original_url": url,
                                }
                            )

                    return filtered_mementos
                logger.warning(f"TimeMap request failed: HTTP {response.status}")
                return []

        except Exception as e:
            logger.error(f"TimeMap retrieval error: {e}")
            return []

    def _parse_memento_datetime(self, datetime_str: str) -> datetime | None:
        """Parse Memento datetime"""
        try:
            # Memento datetime format: "YYYYMMDDHHmmss"
            if len(datetime_str) >= 14:
                return datetime.strptime(datetime_str[:14], "%Y%m%d%H%M%S")
        except ValueError:
            pass
        return None

    def _select_snapshots(
        self, timemap: list[dict[str, Any]], max_snapshots: int
    ) -> list[dict[str, Any]]:
        """Vybere reprezentativní snapshots"""
        if len(timemap) <= max_snapshots:
            return timemap

        # Sort by datetime
        sorted_timemap = sorted(timemap, key=lambda x: x["datetime"])

        # Select evenly distributed snapshots
        selected = []
        if sorted_timemap:
            step = len(sorted_timemap) / max_snapshots
            for i in range(max_snapshots):
                index = int(i * step)
                selected.append(sorted_timemap[index])

        return selected

    async def _retrieve_snapshot(self, snapshot_info: dict[str, Any]) -> SourceResult | None:
        """Retrieve snapshot content"""
        try:
            snapshot_url = snapshot_info["url"]
            snapshot_datetime = snapshot_info["datetime"]
            original_url = snapshot_info["original_url"]

            async with self.session.get(snapshot_url) as response:
                if response.status == 200:
                    content = await response.text()

                    # Extract text content
                    text_content = self._extract_text_from_html(content)

                    result = SourceResult(
                        id=f"memento_{hashlib.md5(snapshot_url.encode()).hexdigest()}",
                        title=self._extract_title(content),
                        content=text_content,
                        url=original_url,
                        source_type="memento",
                        metadata={
                            "memento_url": snapshot_url,
                            "snapshot_datetime": snapshot_datetime.isoformat(),
                            "archive_source": self._extract_archive_source(snapshot_url),
                        },
                        timestamp=snapshot_datetime,
                        retrieval_method="memento_timegate",
                    )

                    return result

        except Exception as e:
            logger.warning(
                f"Snapshot retrieval failed for {snapshot_info.get('url', 'unknown')}: {e}"
            )

        return None

    def _extract_archive_source(self, memento_url: str) -> str:
        """Extrahuj zdroj archivu z Memento URL"""
        parsed = urlparse(memento_url)
        domain = parsed.netloc.lower()

        if "archive.org" in domain:
            return "Internet Archive"
        if "archive.today" in domain:
            return "Archive.today"
        if "webcitation.org" in domain:
            return "WebCitation"
        return f"Unknown Archive ({domain})"

    async def _generate_temporal_diffs(self, snapshots: list[SourceResult]) -> list[TemporalDiff]:
        """Generuje temporal diffs mezi snapshots"""
        diffs = []

        # Sort snapshots by timestamp
        sorted_snapshots = sorted(snapshots, key=lambda x: x.timestamp or datetime.min)

        # Generate diffs between consecutive snapshots
        for i in range(len(sorted_snapshots) - 1):
            earlier = sorted_snapshots[i]
            later = sorted_snapshots[i + 1]

            diff = await self._calculate_diff(earlier, later)
            if diff:
                diffs.append(diff)

        self.stats["diffs_generated"] += len(diffs)

        return diffs

    async def _calculate_diff(
        self, earlier: SourceResult, later: SourceResult
    ) -> TemporalDiff | None:
        """Vypočítá diff mezi dvěma snapshots"""
        try:
            # Simple content-based diff
            earlier_content = earlier.content
            later_content = later.content

            # Calculate changes
            changes = []

            # Content length change
            length_change = len(later_content) - len(earlier_content)
            if abs(length_change) > 50:  # Significant change
                changes.append(
                    {
                        "type": "content_length",
                        "change": length_change,
                        "description": f"Content length changed by {length_change} characters",
                    }
                )

            # Word count change
            earlier_words = len(earlier_content.split())
            later_words = len(later_content.split())
            word_change = later_words - earlier_words

            if abs(word_change) > 10:
                changes.append(
                    {
                        "type": "word_count",
                        "change": word_change,
                        "description": f"Word count changed by {word_change} words",
                    }
                )

            # Simple similarity check
            similarity = self._calculate_similarity(earlier_content, later_content)

            if similarity < 0.9:  # Significant content change
                changes.append(
                    {
                        "type": "content_similarity",
                        "change": 1.0 - similarity,
                        "description": f"Content similarity: {similarity:.2f}",
                    }
                )

            if changes:
                # Assess impact
                impact = self._assess_change_impact(changes)

                diff = TemporalDiff(
                    url=earlier.url,
                    earlier_timestamp=earlier.timestamp,
                    later_timestamp=later.timestamp,
                    diff_type="content_change",
                    changes=changes,
                    impact_assessment=impact,
                    confidence=0.8,
                )

                return diff

        except Exception as e:
            logger.warning(f"Diff calculation failed: {e}")

        return None

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Vypočítá similaritu mezi texty"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _assess_change_impact(self, changes: list[dict[str, Any]]) -> str:
        """Posoudí dopad změn"""
        total_impact = sum(abs(change.get("change", 0)) for change in changes)

        if total_impact > 1000:
            return "major_change"
        if total_impact > 100:
            return "moderate_change"
        if total_impact > 10:
            return "minor_change"
        return "minimal_change"

    def _extract_text_from_html(self, html_content: str) -> str:
        """Extrahuj text z HTML"""
        # Simplified HTML text extraction
        html_content = re.sub(
            r"<script[^>]*>.*?</script>", "", html_content, flags=re.DOTALL | re.IGNORECASE
        )
        html_content = re.sub(
            r"<style[^>]*>.*?</style>", "", html_content, flags=re.DOTALL | re.IGNORECASE
        )
        text = re.sub(r"<[^>]+>", " ", html_content)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_title(self, content: str) -> str:
        """Extrahuj title"""
        title_match = re.search(r"<title[^>]*>(.*?)</title>", content, re.IGNORECASE | re.DOTALL)
        if title_match:
            return title_match.group(1).strip()[:200]
        return "Temporal Snapshot"

    def _update_stats(self, snapshots_count: int, elapsed_time: float):
        """Aktualizuje statistiky"""
        self.stats["temporal_queries"] += 1
        self.stats["snapshots_retrieved"] += snapshots_count

        # Exponential moving average
        alpha = 0.1
        if self.stats["avg_diff_time"] == 0:
            self.stats["avg_diff_time"] = elapsed_time
        else:
            self.stats["avg_diff_time"] = (
                alpha * elapsed_time + (1 - alpha) * self.stats["avg_diff_time"]
            )

    async def close(self):
        """Zavření konektoru"""
        if self.session:
            await self.session.close()


class AhmiaOnionConnector:
    """Enhanced Ahmia/Tor connector s legal-only režimem"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.ahmia_base = "https://ahmia.fi/search"
        self.session = None
        self.legal_whitelist = set(config.get("legal_onion_domains", []))
        self.legal_keywords = set(
            config.get(
                "legal_keywords",
                [
                    "research",
                    "academic",
                    "library",
                    "archive",
                    "documentation",
                    "journalism",
                    "whistleblowing",
                    "freedom",
                    "transparency",
                ],
            )
        )
        self.blocked_keywords = set(
            config.get("blocked_keywords", ["drugs", "weapons", "illegal", "black market", "fraud"])
        )

        self.stats = {
            "searches_performed": 0,
            "legal_results_found": 0,
            "blocked_results": 0,
            "avg_search_time": 0.0,
        }

    async def initialize(self):
        """Inicializace s Tor proxy (optional)"""
        connector_args = {}

        # Configure Tor proxy if available
        tor_proxy = self.config.get("tor_proxy")
        if tor_proxy:
            connector_args["connector"] = aiohttp.ProxyConnector.from_url(tor_proxy)

        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                "User-Agent": self.config.get("user_agent", "DeepResearchTool/3.0 (Legal Research)")
            },
            **connector_args,
        )

    async def search_legal_only(self, query: str, max_results: int = 10) -> list[SourceResult]:
        """Vyhledávání pouze v legal onion services"""
        if not self.session:
            await self.initialize()

        start_time = time.time()

        try:
            # Enhance query with legal keywords
            legal_query = self._enhance_query_for_legal(query)

            # Search Ahmia
            search_results = await self._search_ahmia(
                legal_query, max_results * 3
            )  # Get more, filter later

            # Apply legal filtering
            legal_results = self._filter_legal_results(search_results)

            # Retrieve content with safety checks
            documents = []
            for result in legal_results[:max_results]:
                doc = await self._retrieve_legal_content(result)
                if doc:
                    documents.append(doc)

            elapsed_time = time.time() - start_time
            self._update_stats(
                len(documents), len(search_results) - len(legal_results), elapsed_time
            )

            logger.info(
                f"Ahmia legal search found {len(documents)} documents in {elapsed_time:.2f}s"
            )

            return documents

        except Exception as e:
            logger.error(f"Ahmia legal search failed: {e}")
            return []

    def _enhance_query_for_legal(self, query: str) -> str:
        """Vylepší query pro legal obsah"""
        legal_terms = ["research", "academic", "library", "documentation"]

        # Add legal context to query
        enhanced_query = f"{query} {' OR '.join(legal_terms[:2])}"

        return enhanced_query

    async def _search_ahmia(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """Vyhledávání přes Ahmia"""
        search_params = {"q": query, "count": min(max_results, 50)}  # Ahmia limit

        try:
            async with self.session.get(self.ahmia_base, params=search_params) as response:
                if response.status == 200:
                    # Parse Ahmia results (simplified)
                    content = await response.text()
                    results = self._parse_ahmia_results(content)
                    return results
                logger.warning(f"Ahmia search failed: HTTP {response.status}")
                return []

        except Exception as e:
            logger.error(f"Ahmia search error: {e}")
            return []

    def _parse_ahmia_results(self, html_content: str) -> list[dict[str, Any]]:
        """Parse Ahmia search results"""
        results = []

        # Simplified parsing - look for .onion URLs and titles
        onion_pattern = r"([a-z2-7]{16,56}\.onion)"
        title_pattern = r'<h4[^>]*><a[^>]*href="([^"]*)"[^>]*>([^<]+)</a></h4>'

        onion_matches = re.findall(onion_pattern, html_content, re.IGNORECASE)
        title_matches = re.findall(title_pattern, html_content, re.IGNORECASE)

        for i, (url, title) in enumerate(title_matches[: len(onion_matches)]):
            onion_url = onion_matches[i] if i < len(onion_matches) else url

            result = {
                "url": f"http://{onion_url}" if not url.startswith("http") else url,
                "title": title.strip(),
                "onion_domain": onion_url,
                "source": "ahmia",
            }
            results.append(result)

        return results

    def _filter_legal_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filtruje pouze legal výsledky"""
        legal_results = []

        for result in results:
            title = result.get("title", "").lower()
            url = result.get("url", "").lower()
            onion_domain = result.get("onion_domain", "").lower()

            # Check whitelist first
            if onion_domain in self.legal_whitelist:
                legal_results.append(result)
                continue

            # Check for blocked keywords
            if any(blocked in title for blocked in self.blocked_keywords):
                self.stats["blocked_results"] += 1
                continue

            # Check for legal keywords
            if any(legal in title for legal in self.legal_keywords):
                legal_results.append(result)
                continue

            # Conservative approach - if unsure, exclude
            self.stats["blocked_results"] += 1

        return legal_results

    async def _retrieve_legal_content(self, result: dict[str, Any]) -> SourceResult | None:
        """Retrieve content s legal safety checks"""
        try:
            url = result["url"]

            # Additional safety check
            if not self._is_safe_for_research(url):
                return None

            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()

                    # Content safety check
                    if not self._is_content_legal(content):
                        return None

                    # Extract text
                    text_content = self._extract_text_from_html(content)

                    source_result = SourceResult(
                        id=f"ahmia_{hashlib.md5(url.encode()).hexdigest()}",
                        title=result.get("title", "Onion Service"),
                        content=text_content,
                        url=url,
                        source_type="ahmia_legal",
                        metadata={
                            "onion_domain": result.get("onion_domain", ""),
                            "legal_filtered": True,
                            "ahmia_source": True,
                            "content_safety_checked": True,
                        },
                        retrieval_method="tor_legal",
                    )

                    return source_result

        except Exception as e:
            logger.warning(
                f"Legal content retrieval failed for {result.get('url', 'unknown')}: {e}"
            )

        return None

    def _is_safe_for_research(self, url: str) -> bool:
        """Zkontroluje, zda je URL bezpečný pro research"""
        url_lower = url.lower()

        # Check for obviously unsafe patterns
        unsafe_patterns = ["market", "shop", "buy", "sell", "drug", "weapon"]

        for pattern in unsafe_patterns:
            if pattern in url_lower:
                return False

        return True

    def _is_content_legal(self, content: str) -> bool:
        """Zkontroluje legálnost obsahu"""
        content_lower = content.lower()

        # Check for illegal content indicators
        illegal_indicators = [
            "illegal drugs",
            "weapons sale",
            "stolen",
            "fraud",
            "black market",
            "money laundering",
        ]

        for indicator in illegal_indicators:
            if indicator in content_lower:
                return False

        return True

    def _extract_text_from_html(self, html_content: str) -> str:
        """Extrahuj text z HTML"""
        html_content = re.sub(
            r"<script[^>]*>.*?</script>", "", html_content, flags=re.DOTALL | re.IGNORECASE
        )
        html_content = re.sub(
            r"<style[^>]*>.*?</style>", "", html_content, flags=re.DOTALL | re.IGNORECASE
        )
        text = re.sub(r"<[^>]+>", " ", html_content)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _update_stats(self, legal_found: int, blocked_count: int, elapsed_time: float):
        """Aktualizuje statistiky"""
        self.stats["searches_performed"] += 1
        self.stats["legal_results_found"] += legal_found
        self.stats["blocked_results"] += blocked_count

        # Exponential moving average
        alpha = 0.1
        if self.stats["avg_search_time"] == 0:
            self.stats["avg_search_time"] = elapsed_time
        else:
            self.stats["avg_search_time"] = (
                alpha * elapsed_time + (1 - alpha) * self.stats["avg_search_time"]
            )

    async def close(self):
        """Zavření konektoru"""
        if self.session:
            await self.session.close()


class SpecializedConnectorOrchestrator:
    """Orchestrátor pro všechny specializované konektory"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.connectors = {}
        self.orchestrator_stats = {
            "total_searches": 0,
            "successful_connectors": 0,
            "failed_connectors": 0,
        }

    async def initialize(self):
        """Inicializace všech konektorů"""
        # Initialize Common Crawl
        if self.config.get("common_crawl", {}).get("enabled", False):
            self.connectors["common_crawl"] = CommonCrawlConnector(
                self.config.get("common_crawl", {})
            )
            await self.connectors["common_crawl"].initialize()

        # Initialize Memento
        if self.config.get("memento", {}).get("enabled", False):
            self.connectors["memento"] = MementoTemporalConnector(self.config.get("memento", {}))
            await self.connectors["memento"].initialize()

        # Initialize Ahmia (legal only)
        if self.config.get("ahmia", {}).get("enabled", False):
            self.connectors["ahmia"] = AhmiaOnionConnector(self.config.get("ahmia", {}))
            await self.connectors["ahmia"].initialize()

        logger.info(f"Initialized {len(self.connectors)} specialized connectors")

    async def search_all_sources(
        self, query: str, max_results_per_source: int = 10
    ) -> dict[str, list[SourceResult]]:
        """Vyhledávání napříč všemi zdroji"""
        start_time = time.time()
        results = {}

        search_tasks = []

        # Create search tasks for each connector
        for connector_name, connector in self.connectors.items():
            if connector_name == "common_crawl":
                task = asyncio.create_task(
                    connector.search(query, max_results_per_source), name=f"search_{connector_name}"
                )
            elif connector_name == "memento":
                # For Memento, we need a specific URL and date range
                # This is a simplified example
                continue  # Skip for general query search
            elif connector_name == "ahmia":
                task = asyncio.create_task(
                    connector.search_legal_only(query, max_results_per_source),
                    name=f"search_{connector_name}",
                )
            else:
                continue

            search_tasks.append((connector_name, task))

        # Execute searches concurrently
        for connector_name, task in search_tasks:
            try:
                connector_results = await task
                results[connector_name] = connector_results
                self.orchestrator_stats["successful_connectors"] += 1

                logger.info(f"{connector_name}: {len(connector_results)} results")

            except Exception as e:
                logger.error(f"{connector_name} search failed: {e}")
                results[connector_name] = []
                self.orchestrator_stats["failed_connectors"] += 1

        elapsed_time = time.time() - start_time
        self.orchestrator_stats["total_searches"] += 1

        total_results = sum(len(res) for res in results.values())
        logger.info(
            f"Specialized connectors search completed: {total_results} total results in {elapsed_time:.2f}s"
        )

        return results

    async def search_temporal_snapshots(
        self, url: str, start_date: datetime, end_date: datetime, max_snapshots: int = 10
    ) -> list[SourceResult]:
        """Temporální vyhledávání specifické pro Memento"""
        if "memento" not in self.connectors:
            logger.warning("Memento connector not available for temporal search")
            return []

        try:
            return await self.connectors["memento"].search_temporal(
                url, (start_date, end_date), max_snapshots
            )
        except Exception as e:
            logger.error(f"Temporal search failed: {e}")
            return []

    def get_connector_stats(self) -> dict[str, Any]:
        """Získá statistiky všech konektorů"""
        stats = {"orchestrator_stats": self.orchestrator_stats, "connector_stats": {}}

        for name, connector in self.connectors.items():
            if hasattr(connector, "stats"):
                stats["connector_stats"][name] = connector.stats

        return stats

    async def close_all(self):
        """Zavře všechny konektory"""
        for connector in self.connectors.values():
            if hasattr(connector, "close"):
                await connector.close()


# Factory funkce
def create_specialized_connector_orchestrator(
    config: dict[str, Any],
) -> SpecializedConnectorOrchestrator:
    """Factory funkce pro specialized connector orchestrator"""
    return SpecializedConnectorOrchestrator(config)


# Export hlavních tříd
__all__ = [
    "AhmiaOnionConnector",
    "CommonCrawlConnector",
    "MementoTemporalConnector",
    "SourceResult",
    "SpecializedConnectorOrchestrator",
    "TemporalDiff",
    "create_specialized_connector_orchestrator",
]
