#!/usr/bin/env python3
"""Archive Miner for Phase 2 Implementation
Systematic mining of internet archives with temporal dimension

Author: Advanced AI Research Assistant
Date: August 2025
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import re
import time
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class ArchiveSnapshot:
    """Represents a snapshot from an archive"""

    url: str
    original_url: str
    timestamp: datetime
    archive_source: str
    status_code: int
    content_type: str
    content_length: int
    digest: str | None = None
    title: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "original_url": self.original_url,
            "timestamp": self.timestamp.isoformat(),
            "archive_source": self.archive_source,
            "status_code": self.status_code,
            "content_type": self.content_type,
            "content_length": self.content_length,
            "digest": self.digest,
            "title": self.title,
        }


class WaybackMachineAPI:
    """Interface for Internet Archive Wayback Machine"""

    def __init__(self):
        self.base_url = "https://web.archive.org"
        self.api_url = "https://web.archive.org/cdx/search/cdx"
        self.rate_limit = 1.0  # Respect Archive.org's servers
        self.last_request = 0

    async def _rate_limit_wait(self):
        """Ensure rate limiting"""
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit:
            await asyncio.sleep(self.rate_limit - elapsed)
        self.last_request = time.time()

    async def search_snapshots(
        self,
        url: str,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
        limit: int = 1000,
    ) -> list[ArchiveSnapshot]:
        """Search for snapshots of a URL"""
        await self._rate_limit_wait()

        params = {
            "url": url,
            "output": "json",
            "fl": "timestamp,original,statuscode,mimetype,length,digest",
            "limit": limit,
            "collapse": "digest",  # Remove duplicates
        }

        if from_date:
            params["from"] = from_date.strftime("%Y%m%d")
        if to_date:
            params["to"] = to_date.strftime("%Y%m%d")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.api_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_cdx_response(data)
                    logger.error(f"Wayback API error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error searching Wayback Machine: {e}")
            return []

    def _parse_cdx_response(self, data: list[list[str]]) -> list[ArchiveSnapshot]:
        """Parse CDX API response"""
        snapshots = []

        if not data or len(data) < 2:
            return snapshots

        # Skip header row
        for row in data[1:]:
            try:
                timestamp_str, original_url, status_code, mime_type, length, digest = row

                # Parse timestamp
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")

                # Construct archive URL
                archive_url = f"{self.base_url}/web/{timestamp_str}/{original_url}"

                snapshot = ArchiveSnapshot(
                    url=archive_url,
                    original_url=original_url,
                    timestamp=timestamp,
                    archive_source="wayback_machine",
                    status_code=int(status_code),
                    content_type=mime_type,
                    content_length=int(length) if length.isdigit() else 0,
                    digest=digest,
                )

                snapshots.append(snapshot)

            except (ValueError, IndexError) as e:
                logger.warning(f"Error parsing CDX row {row}: {e}")
                continue

        return snapshots

    async def get_snapshot_content(self, snapshot: ArchiveSnapshot) -> str | None:
        """Get content from a specific snapshot"""
        await self._rate_limit_wait()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(snapshot.url) as response:
                    if response.status == 200:
                        return await response.text()
                    logger.warning(f"Failed to get snapshot content: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error getting snapshot content: {e}")
            return None


class ArchiveTodayAPI:
    """Interface for Archive.today (archive.ph)"""

    def __init__(self):
        self.base_urls = ["https://archive.today", "https://archive.ph", "https://archive.is"]
        self.rate_limit = 2.0  # More conservative rate limiting
        self.last_request = 0

    async def _rate_limit_wait(self):
        """Ensure rate limiting"""
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit:
            await asyncio.sleep(self.rate_limit - elapsed)
        self.last_request = time.time()

    async def search_snapshots(self, url: str) -> list[ArchiveSnapshot]:
        """Search for snapshots on Archive.today"""
        await self._rate_limit_wait()

        snapshots = []

        for base_url in self.base_urls:
            try:
                search_url = f"{base_url}/timemap/json/{url}"

                async with aiohttp.ClientSession() as session:
                    async with session.get(search_url) as response:
                        if response.status == 200:
                            data = await response.json()
                            site_snapshots = self._parse_timemap_response(data, base_url)
                            snapshots.extend(site_snapshots)
                            break  # Use first working mirror
                        continue

            except Exception as e:
                logger.warning(f"Error searching {base_url}: {e}")
                continue

        return snapshots

    def _parse_timemap_response(
        self, data: list[list[str]], base_url: str
    ) -> list[ArchiveSnapshot]:
        """Parse Archive.today timemap response"""
        snapshots = []

        for entry in data:
            try:
                if len(entry) >= 3:
                    timestamp_str = entry[1]
                    archive_url = entry[2]

                    # Parse timestamp (ISO format)
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

                    # Extract original URL from archive URL
                    original_url = archive_url.split("/")[-1] if "/" in archive_url else archive_url

                    snapshot = ArchiveSnapshot(
                        url=archive_url,
                        original_url=original_url,
                        timestamp=timestamp,
                        archive_source="archive_today",
                        status_code=200,  # Assume success if in timemap
                        content_type="text/html",
                        content_length=0,
                    )

                    snapshots.append(snapshot)

            except Exception as e:
                logger.warning(f"Error parsing timemap entry {entry}: {e}")
                continue

        return snapshots

    async def create_snapshot(self, url: str) -> str | None:
        """Create new snapshot on Archive.today"""
        await self._rate_limit_wait()

        for base_url in self.base_urls:
            try:
                submit_url = f"{base_url}/submit/"

                data = {"url": url, "anyway": "1"}

                async with aiohttp.ClientSession() as session:
                    async with session.post(submit_url, data=data) as response:
                        if response.status in [200, 302]:
                            # Extract archive URL from response
                            location = response.headers.get("Location")
                            if location:
                                return location
                            # Try to parse from response text
                            text = await response.text()
                            # Look for archive URL pattern
                            match = re.search(r"https?://archive\.(today|ph|is)/\w+", text)
                            if match:
                                return match.group(0)

            except Exception as e:
                logger.warning(f"Error creating snapshot on {base_url}: {e}")
                continue

        return None


class ArchiveMiner:
    """Main archive mining system"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.wayback = WaybackMachineAPI()
        self.archive_today = ArchiveTodayAPI()

        # Mining configuration
        self.max_snapshots_per_url = config.get("max_snapshots_per_url", 100)
        self.min_time_between_snapshots = config.get("min_time_between_snapshots", 30)  # days
        self.content_filters = config.get("content_filters", [])

        # Results storage
        self.snapshots: dict[str, list[ArchiveSnapshot]] = {}
        self.content_cache: dict[str, str] = {}

    async def mine_url_history(
        self, url: str, from_date: datetime | None = None, to_date: datetime | None = None
    ) -> list[ArchiveSnapshot]:
        """Mine complete history of a URL from all archives"""
        all_snapshots = []

        logger.info(f"Mining archive history for: {url}")

        # Search Wayback Machine
        try:
            wayback_snapshots = await self.wayback.search_snapshots(
                url, from_date, to_date, self.max_snapshots_per_url
            )
            all_snapshots.extend(wayback_snapshots)
            logger.info(f"Found {len(wayback_snapshots)} Wayback Machine snapshots")
        except Exception as e:
            logger.error(f"Error mining Wayback Machine: {e}")

        # Search Archive.today
        try:
            archive_today_snapshots = await self.archive_today.search_snapshots(url)
            all_snapshots.extend(archive_today_snapshots)
            logger.info(f"Found {len(archive_today_snapshots)} Archive.today snapshots")
        except Exception as e:
            logger.error(f"Error mining Archive.today: {e}")

        # Remove duplicates and filter
        filtered_snapshots = self._filter_snapshots(all_snapshots)

        # Store results
        self.snapshots[url] = filtered_snapshots

        logger.info(f"Total unique snapshots for {url}: {len(filtered_snapshots)}")
        return filtered_snapshots

    def _filter_snapshots(self, snapshots: list[ArchiveSnapshot]) -> list[ArchiveSnapshot]:
        """Filter and deduplicate snapshots"""
        # Sort by timestamp
        snapshots.sort(key=lambda s: s.timestamp)

        filtered = []
        last_timestamp = None
        seen_digests = set()

        for snapshot in snapshots:
            # Skip if too close in time to previous snapshot
            if (
                last_timestamp
                and (snapshot.timestamp - last_timestamp).days < self.min_time_between_snapshots
            ):
                continue

            # Skip if we've seen this content before (by digest)
            if snapshot.digest and snapshot.digest in seen_digests:
                continue

            # Apply content filters
            if self._should_filter_snapshot(snapshot):
                continue

            filtered.append(snapshot)
            last_timestamp = snapshot.timestamp
            if snapshot.digest:
                seen_digests.add(snapshot.digest)

        return filtered

    def _should_filter_snapshot(self, snapshot: ArchiveSnapshot) -> bool:
        """Check if snapshot should be filtered out"""
        # Filter by status code
        if snapshot.status_code not in [200, 301, 302]:
            return True

        # Filter by content type
        if not any(
            ct in snapshot.content_type.lower()
            for ct in ["text/html", "text/plain", "application/json"]
        ):
            return True

        # Filter by content length (too small likely not useful)
        if snapshot.content_length < 1000:
            return True

        # Apply custom filters
        for filter_pattern in self.content_filters:
            if re.search(filter_pattern, snapshot.original_url, re.IGNORECASE):
                return True

        return False

    async def get_temporal_content_evolution(self, url: str) -> dict[str, Any]:
        """Analyze how content evolved over time"""
        if url not in self.snapshots:
            await self.mine_url_history(url)

        snapshots = self.snapshots[url]
        if not snapshots:
            return {}

        evolution = {
            "url": url,
            "first_snapshot": snapshots[0].timestamp.isoformat(),
            "last_snapshot": snapshots[-1].timestamp.isoformat(),
            "total_snapshots": len(snapshots),
            "time_span_days": (snapshots[-1].timestamp - snapshots[0].timestamp).days,
            "snapshots_by_year": {},
            "content_changes": [],
        }

        # Group by year
        for snapshot in snapshots:
            year = snapshot.timestamp.year
            if year not in evolution["snapshots_by_year"]:
                evolution["snapshots_by_year"][year] = 0
            evolution["snapshots_by_year"][year] += 1

        # Analyze content changes (sample a few key snapshots)
        sample_snapshots = self._sample_snapshots_for_analysis(snapshots)

        for i, snapshot in enumerate(sample_snapshots):
            content = await self._get_snapshot_content_cached(snapshot)
            if content:
                change_info = {
                    "timestamp": snapshot.timestamp.isoformat(),
                    "content_length": len(content),
                    "title": self._extract_title_from_content(content),
                    "snapshot_url": snapshot.url,
                }

                # Compare with previous snapshot if available
                if i > 0:
                    prev_content = await self._get_snapshot_content_cached(sample_snapshots[i - 1])
                    if prev_content:
                        change_info["content_diff_ratio"] = self._calculate_content_similarity(
                            prev_content, content
                        )

                evolution["content_changes"].append(change_info)

        return evolution

    def _sample_snapshots_for_analysis(
        self, snapshots: list[ArchiveSnapshot], max_samples: int = 10
    ) -> list[ArchiveSnapshot]:
        """Sample snapshots for content analysis"""
        if len(snapshots) <= max_samples:
            return snapshots

        # Sample evenly across time range
        step = len(snapshots) // max_samples
        return [snapshots[i] for i in range(0, len(snapshots), step)]

    async def _get_snapshot_content_cached(self, snapshot: ArchiveSnapshot) -> str | None:
        """Get snapshot content with caching"""
        cache_key = f"{snapshot.archive_source}:{snapshot.digest or snapshot.url}"

        if cache_key in self.content_cache:
            return self.content_cache[cache_key]

        content = None
        if snapshot.archive_source == "wayback_machine":
            content = await self.wayback.get_snapshot_content(snapshot)
        elif snapshot.archive_source == "archive_today":
            # Archive.today content retrieval would be implemented here
            pass

        if content:
            self.content_cache[cache_key] = content

        return content

    def _extract_title_from_content(self, content: str) -> str | None:
        """Extract title from HTML content"""
        try:
            match = re.search(r"<title[^>]*>([^<]+)</title>", content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        except Exception:
            pass
        return None

    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity ratio between two content strings"""
        try:
            from difflib import SequenceMatcher

            matcher = SequenceMatcher(None, content1, content2)
            return matcher.ratio()
        except Exception:
            return 0.0

    async def mine_domain_history(
        self, domain: str, from_date: datetime | None = None, to_date: datetime | None = None
    ) -> dict[str, list[ArchiveSnapshot]]:
        """Mine history for entire domain"""
        domain_snapshots = {}

        # Common URL patterns to try
        url_patterns = [
            f"http://{domain}",
            f"https://{domain}",
            f"http://www.{domain}",
            f"https://www.{domain}",
        ]

        for url in url_patterns:
            try:
                snapshots = await self.mine_url_history(url, from_date, to_date)
                if snapshots:
                    domain_snapshots[url] = snapshots
            except Exception as e:
                logger.error(f"Error mining {url}: {e}")
                continue

        return domain_snapshots

    async def export_mining_results(self, filepath: str):
        """Export mining results to JSON file"""
        export_data = {
            "mining_timestamp": datetime.now().isoformat(),
            "total_urls_mined": len(self.snapshots),
            "total_snapshots_found": sum(len(snapshots) for snapshots in self.snapshots.values()),
            "results": {},
        }

        for url, snapshots in self.snapshots.items():
            export_data["results"][url] = [snapshot.to_dict() for snapshot in snapshots]

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Mining results exported to {filepath}")

    async def create_new_archive_snapshot(self, url: str) -> str | None:
        """Create new archive snapshot for future reference"""
        try:
            archive_url = await self.archive_today.create_snapshot(url)
            if archive_url:
                logger.info(f"Created new archive snapshot: {archive_url}")
                return archive_url
            logger.warning(f"Failed to create archive snapshot for {url}")
            return None
        except Exception as e:
            logger.error(f"Error creating archive snapshot: {e}")
            return None

    def get_mining_statistics(self) -> dict[str, Any]:
        """Get comprehensive mining statistics"""
        total_snapshots = sum(len(snapshots) for snapshots in self.snapshots.values())

        archive_sources = {}
        earliest_snapshot = None
        latest_snapshot = None

        for snapshots in self.snapshots.values():
            for snapshot in snapshots:
                # Count by source
                source = snapshot.archive_source
                archive_sources[source] = archive_sources.get(source, 0) + 1

                # Track date range
                if earliest_snapshot is None or snapshot.timestamp < earliest_snapshot:
                    earliest_snapshot = snapshot.timestamp
                if latest_snapshot is None or snapshot.timestamp > latest_snapshot:
                    latest_snapshot = snapshot.timestamp

        return {
            "urls_mined": len(self.snapshots),
            "total_snapshots": total_snapshots,
            "snapshots_by_source": archive_sources,
            "earliest_snapshot": earliest_snapshot.isoformat() if earliest_snapshot else None,
            "latest_snapshot": latest_snapshot.isoformat() if latest_snapshot else None,
            "time_span_covered_days": (
                (latest_snapshot - earliest_snapshot).days
                if earliest_snapshot and latest_snapshot
                else 0
            ),
            "cache_size": len(self.content_cache),
        }
