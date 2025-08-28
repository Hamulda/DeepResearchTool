#!/usr/bin/env python3
"""
Enhanced Common Crawl Connector
Stabilní práce s WARC offsety, retries a idempotentní cache

Author: Senior Python/MLOps Agent
"""

import asyncio
import logging
import aiohttp
import gzip
import time
from typing import Dict, Any, List, Optional, Tuple, AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
from urllib.parse import urljoin, urlparse
import backoff

logger = logging.getLogger(__name__)


@dataclass
class WARCRecord:
    """WARC record s metadata"""
    record_id: str
    url: str
    timestamp: str
    content_type: str
    content_length: int
    warc_offset: int
    warc_filename: str
    content: str
    headers: Dict[str, str]
    extraction_metadata: Dict[str, Any]


@dataclass
class CommonCrawlResult:
    """Výsledek Common Crawl query"""
    query: str
    total_results: int
    warc_records: List[WARCRecord]
    processing_time: float
    cache_hits: int
    cache_misses: int
    error_count: int
    quality_metrics: Dict[str, float]


class EnhancedCommonCrawlConnector:
    """Enhanced Common Crawl connector s WARC offset handling a caching"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cc_config = config.get("common_crawl", {})

        # API settings
        self.index_api_base = self.cc_config.get("index_api_base", "https://index.commoncrawl.org")
        self.download_base = self.cc_config.get("download_base", "https://data.commoncrawl.org")

        # Cache settings
        self.cache_enabled = self.cc_config.get("cache_enabled", True)
        self.cache_dir = Path(self.cc_config.get("cache_dir", "./cache/common_crawl"))
        self.cache_ttl_hours = self.cc_config.get("cache_ttl_hours", 24)

        # Request settings
        self.max_results = self.cc_config.get("max_results", 1000)
        self.timeout = self.cc_config.get("timeout", 30)
        self.max_retries = self.cc_config.get("max_retries", 3)
        self.retry_delay = self.cc_config.get("retry_delay", 1.0)

        # Quality filters
        self.min_content_length = self.cc_config.get("min_content_length", 500)
        self.max_content_length = self.cc_config.get("max_content_length", 100000)
        self.allowed_content_types = self.cc_config.get("allowed_content_types", ["text/html", "text/plain"])

        # Processing settings
        self.concurrent_downloads = self.cc_config.get("concurrent_downloads", 5)
        self.enable_content_extraction = self.cc_config.get("enable_content_extraction", True)

        # Initialize cache
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Session management
        self.session = None

    async def initialize(self):
        """Inicializace konektoru"""

        logger.info("Initializing Enhanced Common Crawl Connector...")

        # Create HTTP session with appropriate settings
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        headers = {
            "User-Agent": "DeepResearchTool/1.0 (+https://github.com/research/deep-research-tool)"
        }

        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers=headers,
            connector=aiohttp.TCPConnector(limit=10, limit_per_host=5)
        )

        logger.info("✅ Enhanced Common Crawl Connector initialized")

    async def close(self):
        """Zavření konektoru"""
        if self.session:
            await self.session.close()

    async def search_and_fetch(self,
                              query: str,
                              crawl_id: Optional[str] = None,
                              url_pattern: Optional[str] = None) -> CommonCrawlResult:
        """
        Hlavní search a fetch funkce

        Args:
            query: Search query
            crawl_id: Specific crawl ID (e.g., "CC-MAIN-2023-40")
            url_pattern: URL pattern filter

        Returns:
            CommonCrawlResult s WARC records
        """

        start_time = time.time()
        cache_hits = 0
        cache_misses = 0
        error_count = 0

        logger.info(f"Starting Common Crawl search for query: {query}")

        try:
            # STEP 1: Search index for URLs
            index_results = await self._search_index(query, crawl_id, url_pattern)

            # STEP 2: Fetch WARC records for found URLs
            warc_records = []

            # Process in batches for concurrent downloads
            batch_size = self.concurrent_downloads
            for i in range(0, len(index_results), batch_size):
                batch = index_results[i:i + batch_size]

                # Concurrent fetch of WARC records
                batch_tasks = []
                for index_result in batch:
                    task = self._fetch_warc_record(index_result)
                    batch_tasks.append(task)

                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                for result in batch_results:
                    if isinstance(result, Exception):
                        error_count += 1
                        logger.warning(f"WARC fetch failed: {result}")
                    elif result:
                        if result.get("cache_hit"):
                            cache_hits += 1
                        else:
                            cache_misses += 1

                        warc_record = result.get("warc_record")
                        if warc_record:
                            warc_records.append(warc_record)

            # STEP 3: Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(warc_records, index_results)

            processing_time = time.time() - start_time

            result = CommonCrawlResult(
                query=query,
                total_results=len(index_results),
                warc_records=warc_records,
                processing_time=processing_time,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                error_count=error_count,
                quality_metrics=quality_metrics
            )

            logger.info(f"Common Crawl search completed: {len(warc_records)}/{len(index_results)} records fetched")
            logger.info(f"Cache efficiency: {cache_hits}/{cache_hits + cache_misses} hits")

            return result

        except Exception as e:
            logger.error(f"Common Crawl search failed: {e}")
            raise

    async def _search_index(self,
                           query: str,
                           crawl_id: Optional[str] = None,
                           url_pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        """Vyhledání v Common Crawl indexu"""

        # Determine crawl to search
        if not crawl_id:
            crawl_id = await self._get_latest_crawl_id()

        # Build index search URL
        index_url = f"{self.index_api_base}/{crawl_id}-index"

        # Build search parameters
        params = {
            "url": query if query.startswith("http") else f"*{query}*",
            "output": "json",
            "limit": self.max_results
        }

        if url_pattern:
            params["url"] = url_pattern

        logger.info(f"Searching Common Crawl index: {crawl_id}")

        try:
            async with self.session.get(index_url, params=params) as response:
                if response.status == 200:
                    content = await response.text()

                    # Parse JSONL response
                    results = []
                    for line in content.strip().split('\n'):
                        if line:
                            try:
                                result = json.loads(line)
                                results.append(result)
                            except json.JSONDecodeError:
                                continue

                    logger.info(f"Found {len(results)} index results")
                    return results
                else:
                    logger.error(f"Index search failed: {response.status}")
                    return []

        except Exception as e:
            logger.error(f"Index search error: {e}")
            return []

    async def _get_latest_crawl_id(self) -> str:
        """Získání ID nejnovějšího crawlu"""

        try:
            # Check available crawls
            crawls_url = f"{self.index_api_base}/collinfo.json"

            async with self.session.get(crawls_url) as response:
                if response.status == 200:
                    crawls_data = await response.json()

                    # Find latest crawl
                    crawls = crawls_data.get("crawls", [])
                    if crawls:
                        latest_crawl = max(crawls, key=lambda x: x.get("timegate", ""))
                        return latest_crawl.get("id", "CC-MAIN-2023-40")

        except Exception as e:
            logger.warning(f"Failed to get latest crawl ID: {e}")

        # Fallback to recent crawl ID
        return "CC-MAIN-2023-40"

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def _fetch_warc_record(self, index_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fetch WARC record s retry logic"""

        filename = index_result.get("filename")
        offset = index_result.get("offset")
        length = index_result.get("length")
        url = index_result.get("url")
        timestamp = index_result.get("timestamp")

        if not all([filename, offset is not None, length, url]):
            logger.warning("Invalid index result - missing required fields")
            return None

        # Check cache first
        cache_key = self._generate_cache_key(filename, offset, length)
        cached_record = await self._get_from_cache(cache_key)

        if cached_record:
            return {
                "warc_record": cached_record,
                "cache_hit": True
            }

        # Fetch from Common Crawl
        try:
            warc_record = await self._fetch_warc_content(filename, offset, length, url, timestamp)

            if warc_record:
                # Cache the result
                await self._save_to_cache(cache_key, warc_record)

                return {
                    "warc_record": warc_record,
                    "cache_hit": False
                }

        except Exception as e:
            logger.error(f"Failed to fetch WARC record: {e}")
            raise

        return None

    async def _fetch_warc_content(self,
                                 filename: str,
                                 offset: int,
                                 length: int,
                                 url: str,
                                 timestamp: str) -> Optional[WARCRecord]:
        """Fetch konkrétního WARC record content"""

        # Build download URL
        download_url = f"{self.download_base}/{filename}"

        # Set range header for specific offset
        headers = {
            "Range": f"bytes={offset}-{offset + length - 1}"
        }

        try:
            async with self.session.get(download_url, headers=headers) as response:
                if response.status in [200, 206]:  # OK or Partial Content
                    raw_content = await response.read()

                    # Decompress if gzipped
                    if filename.endswith('.gz'):
                        try:
                            raw_content = gzip.decompress(raw_content)
                        except Exception as e:
                            logger.warning(f"Failed to decompress WARC content: {e}")
                            return None

                    # Parse WARC record
                    warc_record = await self._parse_warc_record(
                        raw_content, filename, offset, url, timestamp
                    )

                    return warc_record

                else:
                    logger.error(f"Failed to fetch WARC content: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"WARC fetch error: {e}")
            return None

    async def _parse_warc_record(self,
                                raw_content: bytes,
                                filename: str,
                                offset: int,
                                url: str,
                                timestamp: str) -> Optional[WARCRecord]:
        """Parse WARC record content"""

        try:
            content_str = raw_content.decode('utf-8', errors='ignore')

            # Simple WARC parsing - split headers and content
            parts = content_str.split('\r\n\r\n', 2)
            if len(parts) < 2:
                parts = content_str.split('\n\n', 2)

            if len(parts) < 2:
                logger.warning("Invalid WARC record format")
                return None

            warc_headers_text = parts[0]

            # Skip HTTP headers if present (parts[1])
            if len(parts) >= 3:
                content = parts[2]
            else:
                content = parts[1]

            # Parse WARC headers
            warc_headers = {}
            for line in warc_headers_text.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    warc_headers[key.strip()] = value.strip()

            # Quality filtering
            if len(content) < self.min_content_length:
                logger.debug(f"Content too short: {len(content)} < {self.min_content_length}")
                return None

            if len(content) > self.max_content_length:
                content = content[:self.max_content_length]
                logger.debug(f"Content truncated to {self.max_content_length} characters")

            # Extract clean text content if enabled
            if self.enable_content_extraction:
                content = self._extract_text_content(content)

            # Generate record ID
            record_id = hashlib.md5(f"{filename}:{offset}:{url}".encode()).hexdigest()

            warc_record = WARCRecord(
                record_id=record_id,
                url=url,
                timestamp=timestamp,
                content_type=warc_headers.get("Content-Type", "text/html"),
                content_length=len(content),
                warc_offset=offset,
                warc_filename=filename,
                content=content,
                headers=warc_headers,
                extraction_metadata={
                    "extraction_timestamp": time.time(),
                    "original_length": len(raw_content),
                    "processed_length": len(content)
                }
            )

            return warc_record

        except Exception as e:
            logger.error(f"WARC record parsing failed: {e}")
            return None

    def _extract_text_content(self, html_content: str) -> str:
        """Extrakce čistého textu z HTML"""

        try:
            # Simple HTML tag removal
            import re

            # Remove script and style elements
            html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
            html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)

            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', html_content)

            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            return text

        except Exception as e:
            logger.warning(f"Text extraction failed: {e}")
            return html_content

    def _generate_cache_key(self, filename: str, offset: int, length: int) -> str:
        """Generování cache key"""
        key_data = f"{filename}:{offset}:{length}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    async def _get_from_cache(self, cache_key: str) -> Optional[WARCRecord]:
        """Získání z cache"""

        if not self.cache_enabled:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            if cache_file.exists():
                # Check cache age
                cache_age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
                if cache_age_hours > self.cache_ttl_hours:
                    cache_file.unlink()  # Remove expired cache
                    return None

                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)

                # Reconstruct WARCRecord
                return WARCRecord(**cache_data)

        except Exception as e:
            logger.warning(f"Cache read error: {e}")

        return None

    async def _save_to_cache(self, cache_key: str, warc_record: WARCRecord):
        """Uložení do cache"""

        if not self.cache_enabled:
            return

        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            # Convert to dict for JSON serialization
            cache_data = {
                "record_id": warc_record.record_id,
                "url": warc_record.url,
                "timestamp": warc_record.timestamp,
                "content_type": warc_record.content_type,
                "content_length": warc_record.content_length,
                "warc_offset": warc_record.warc_offset,
                "warc_filename": warc_record.warc_filename,
                "content": warc_record.content,
                "headers": warc_record.headers,
                "extraction_metadata": warc_record.extraction_metadata
            }

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def _calculate_quality_metrics(self,
                                  warc_records: List[WARCRecord],
                                  index_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Výpočet quality metrics"""

        if not warc_records:
            return {"success_rate": 0.0, "avg_content_length": 0.0, "content_type_diversity": 0.0}

        # Success rate
        success_rate = len(warc_records) / len(index_results) if index_results else 0

        # Average content length
        avg_content_length = sum(record.content_length for record in warc_records) / len(warc_records)

        # Content type diversity
        content_types = set(record.content_type for record in warc_records)
        content_type_diversity = len(content_types) / len(warc_records)

        # Temporal distribution
        timestamps = [record.timestamp for record in warc_records]
        temporal_diversity = len(set(ts[:8] for ts in timestamps)) / len(warc_records)  # Unique dates

        return {
            "success_rate": success_rate,
            "avg_content_length": avg_content_length,
            "content_type_diversity": content_type_diversity,
            "temporal_diversity": temporal_diversity
        }

    async def get_connector_status(self) -> Dict[str, Any]:
        """Získání status konektoru"""

        try:
            # Test connection
            test_url = f"{self.index_api_base}/collinfo.json"
            async with self.session.get(test_url) as response:
                online = response.status == 200
        except:
            online = False

        # Cache statistics
        cache_files = 0
        cache_size_mb = 0
        if self.cache_enabled and self.cache_dir.exists():
            cache_files = len(list(self.cache_dir.glob("*.json")))
            cache_size_mb = sum(f.stat().st_size for f in self.cache_dir.glob("*.json")) / (1024 * 1024)

        return {
            "connector_type": "common_crawl",
            "online": online,
            "cache_enabled": self.cache_enabled,
            "cache_files": cache_files,
            "cache_size_mb": round(cache_size_mb, 2),
            "max_results": self.max_results,
            "concurrent_downloads": self.concurrent_downloads
        }
