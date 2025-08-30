#!/usr/bin/env python3
"""
Common Crawl WARC konektor
Stabilní WARC offsety, retries, idempotentní cache

Author: Senior Python/MLOps Agent
"""

import asyncio
import gzip
import json
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import aiohttp
import aiofiles
from pathlib import Path
import warcio
from warcio.archiveiterator import ArchiveIterator


@dataclass
class WARCRecord:
    """WARC záznam s metadaty"""

    warc_record_id: str
    target_uri: str
    warc_date: str
    content_type: str
    content_length: int
    warc_filename: str
    warc_offset: int
    content: str
    headers: Dict[str, str]


class CommonCrawlConnector:
    """Common Crawl konektor s WARC podporou"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get("base_url", "https://commoncrawl.s3.amazonaws.com")
        self.cache_dir = Path(config.get("cache_dir", "research_cache/commoncrawl"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 2.0)

    async def search_crawl_index(
        self, url_pattern: str, crawl_id: str = "latest"
    ) -> List[Dict[str, Any]]:
        """Vyhledá URL v Common Crawl indexu"""
        print(f"🔍 Searching Common Crawl index for: {url_pattern}")

        if crawl_id == "latest":
            crawl_id = await self._get_latest_crawl_id()

        index_url = f"https://index.commoncrawl.org/CC-MAIN-{crawl_id}-index"

        # CDX API query
        cdx_params = {"url": url_pattern, "output": "json", "limit": "1000"}

        results = []
        retry_count = 0

        while retry_count < self.max_retries:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(index_url, params=cdx_params) as response:
                        if response.status == 200:
                            lines = (await response.text()).strip().split("\n")

                            for line in lines:
                                if line.strip():
                                    try:
                                        record = json.loads(line)
                                        results.append(record)
                                    except json.JSONDecodeError:
                                        continue

                            print(f"✅ Found {len(results)} records in Common Crawl")
                            return results
                        else:
                            raise aiohttp.ClientError(f"HTTP {response.status}")

            except Exception as e:
                retry_count += 1
                if retry_count >= self.max_retries:
                    print(f"❌ Common Crawl search failed after {self.max_retries} retries: {e}")
                    return []

                await asyncio.sleep(self.retry_delay * retry_count)

        return results

    async def fetch_warc_record(
        self, warc_filename: str, warc_offset: int, warc_length: int
    ) -> Optional[WARCRecord]:
        """Stáhne WARC záznam podle offsetu"""
        cache_key = f"{warc_filename}_{warc_offset}_{warc_length}"
        cache_file = self.cache_dir / f"{hashlib.md5(cache_key.encode()).hexdigest()}.json"

        # Kontrola cache
        if cache_file.exists():
            async with aiofiles.open(cache_file, "r") as f:
                cached_data = json.loads(await f.read())
                return WARCRecord(**cached_data)

        warc_url = f"{self.base_url}/{warc_filename}"

        retry_count = 0
        while retry_count < self.max_retries:
            try:
                # Range request pro specific WARC offset
                headers = {
                    "Range": f"bytes={warc_offset}-{warc_offset + warc_length - 1}",
                    "User-Agent": "DeepResearchTool/1.0 (+research@example.com)",
                }

                async with aiohttp.ClientSession() as session:
                    async with session.get(warc_url, headers=headers) as response:
                        if response.status in [200, 206]:  # OK or Partial Content
                            warc_data = await response.read()

                            # Parse WARC record
                            record = self._parse_warc_data(warc_data, warc_filename, warc_offset)

                            if record:
                                # Cache výsledek
                                record_dict = {
                                    "warc_record_id": record.warc_record_id,
                                    "target_uri": record.target_uri,
                                    "warc_date": record.warc_date,
                                    "content_type": record.content_type,
                                    "content_length": record.content_length,
                                    "warc_filename": record.warc_filename,
                                    "warc_offset": record.warc_offset,
                                    "content": record.content,
                                    "headers": record.headers,
                                }

                                async with aiofiles.open(cache_file, "w") as f:
                                    await f.write(json.dumps(record_dict, indent=2))

                                return record
                        else:
                            raise aiohttp.ClientError(f"HTTP {response.status}")

            except Exception as e:
                retry_count += 1
                if retry_count >= self.max_retries:
                    print(f"❌ WARC fetch failed after {self.max_retries} retries: {e}")
                    return None

                await asyncio.sleep(self.retry_delay * retry_count)

        return None

    def _parse_warc_data(
        self, warc_data: bytes, warc_filename: str, warc_offset: int
    ) -> Optional[WARCRecord]:
        """Parsuje WARC data"""
        try:
            # Handle gzip compression
            if warc_data.startswith(b"\x1f\x8b"):
                warc_data = gzip.decompress(warc_data)

            # Parse WARC using warcio
            archive_iterator = ArchiveIterator(warc_data)

            for record in archive_iterator:
                if record.rec_type == "response":
                    # Extract content
                    content = record.content_stream().read()

                    # Decode content
                    try:
                        if isinstance(content, bytes):
                            content = content.decode("utf-8", errors="ignore")
                    except Exception:
                        content = str(content)

                    # Extract headers
                    headers = {}
                    if hasattr(record, "http_headers") and record.http_headers:
                        headers = dict(record.http_headers.headers)

                    return WARCRecord(
                        warc_record_id=record.rec_headers.get("WARC-Record-ID", ""),
                        target_uri=record.rec_headers.get("WARC-Target-URI", ""),
                        warc_date=record.rec_headers.get("WARC-Date", ""),
                        content_type=record.rec_headers.get("Content-Type", ""),
                        content_length=int(record.rec_headers.get("Content-Length", 0)),
                        warc_filename=warc_filename,
                        warc_offset=warc_offset,
                        content=content,
                        headers=headers,
                    )

        except Exception as e:
            print(f"❌ WARC parsing failed: {e}")
            return None

        return None

    async def _get_latest_crawl_id(self) -> str:
        """Získá ID nejnovějšího crawlu"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://commoncrawl.org/the-data/get-started/") as response:
                    if response.status == 200:
                        text = await response.text()
                        # Simple regex to find latest crawl ID
                        import re

                        match = re.search(r"CC-MAIN-(\d{4}-\d{2})", text)
                        if match:
                            return match.group(1)
        except Exception as e:
            print(f"❌ Failed to get latest crawl ID: {e}")

        # Fallback to recent crawl
        return "2024-10"

    async def search_and_fetch(self, url_pattern: str, max_records: int = 10) -> List[WARCRecord]:
        """Vyhledá a stáhne WARC záznamy"""
        # Vyhledej v indexu
        index_results = await self.search_crawl_index(url_pattern)

        if not index_results:
            return []

        # Fetch WARC records
        records = []
        for i, result in enumerate(index_results[:max_records]):
            try:
                warc_filename = result.get("filename", "")
                warc_offset = int(result.get("offset", 0))
                warc_length = int(result.get("length", 0))

                if warc_filename and warc_offset >= 0 and warc_length > 0:
                    record = await self.fetch_warc_record(warc_filename, warc_offset, warc_length)
                    if record:
                        records.append(record)

            except Exception as e:
                print(f"❌ Failed to fetch record {i}: {e}")
                continue

        print(f"✅ Successfully fetched {len(records)} WARC records")
        return records

    def get_cache_stats(self) -> Dict[str, Any]:
        """Vrátí statistiky cache"""
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "cache_dir": str(self.cache_dir),
            "cached_records": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "oldest_cache": min((f.stat().st_mtime for f in cache_files), default=0),
            "newest_cache": max((f.stat().st_mtime for f in cache_files), default=0),
        }

    async def cleanup_cache(self, max_age_days: int = 30) -> int:
        """Vyčistí starou cache"""
        import time

        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        removed_count = 0

        cache_files = list(self.cache_dir.glob("*.json"))
        for cache_file in cache_files:
            if current_time - cache_file.stat().st_mtime > max_age_seconds:
                cache_file.unlink()
                removed_count += 1

        print(f"🧹 Cleaned up {removed_count} old cache files")
        return removed_count
