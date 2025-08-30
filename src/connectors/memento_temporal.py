#!/usr/bin/env python3
"""Memento Temporal Connector
TimeMap→TimeGate orchestrace přes milníkové datumy s diff analýzou

Author: Senior Python/MLOps Agent
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
import difflib
import hashlib
import logging
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class MementoSnapshot:
    """Memento snapshot s temporal metadata"""

    url: str
    original_url: str
    datetime: datetime
    memento_datetime: str
    content: str
    content_length: int
    content_hash: str
    archive_name: str
    rel_type: str  # first, last, next, prev, memento
    extraction_metadata: dict[str, Any]


@dataclass
class TemporalDiff:
    """Diff mezi dvěma snapshoty"""

    snapshot_a: MementoSnapshot
    snapshot_b: MementoSnapshot
    diff_type: str  # additions, deletions, modifications
    diff_chunks: list[dict[str, Any]]
    similarity_score: float
    change_summary: dict[str, Any]
    significance_score: float


@dataclass
class MementoTemporalResult:
    """Výsledek temporal analysis"""

    original_url: str
    query_timespan: tuple[datetime, datetime]
    snapshots: list[MementoSnapshot]
    temporal_diffs: list[TemporalDiff]
    milestone_analysis: dict[str, Any]
    change_timeline: list[dict[str, Any]]
    processing_time: float
    quality_metrics: dict[str, float]


class MementoTemporalConnector:
    """Memento connector pro temporal analysis s diff tracking"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.memento_config = config.get("memento", {})

        # Memento API settings
        self.memento_aggregator = self.memento_config.get(
            "aggregator", "http://timetravel.mementoweb.org"
        )
        self.archive_endpoints = self.memento_config.get(
            "archive_endpoints",
            {
                "wayback_machine": "http://web.archive.org/web",
                "archive_today": "http://archive.today",
                "library_congress": "http://webarchive.loc.gov",
            },
        )

        # Temporal settings
        self.milestone_intervals = self.memento_config.get(
            "milestone_intervals", ["1_year", "6_months", "3_months", "1_month", "1_week"]
        )
        self.max_snapshots = self.memento_config.get("max_snapshots", 20)
        self.min_content_change = self.memento_config.get(
            "min_content_change", 0.05
        )  # 5% change threshold

        # Request settings
        self.timeout = self.memento_config.get("timeout", 30)
        self.max_retries = self.memento_config.get("max_retries", 3)
        self.concurrent_requests = self.memento_config.get("concurrent_requests", 3)

        # Diff settings
        self.enable_content_diff = self.memento_config.get("enable_content_diff", True)
        self.diff_context_lines = self.memento_config.get("diff_context_lines", 3)
        self.max_diff_size = self.memento_config.get("max_diff_size", 50000)  # characters

        # Cache settings
        self.cache_enabled = self.memento_config.get("cache_enabled", True)
        self.cache_dir = Path(self.memento_config.get("cache_dir", "./cache/memento"))

        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.session = None

    async def initialize(self):
        """Inicializace konektoru"""
        logger.info("Initializing Memento Temporal Connector...")

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        headers = {
            "User-Agent": "DeepResearchTool/1.0 Memento-Temporal-Analysis",
            "Accept": "application/link-format, text/html, */*",
        }

        self.session = aiohttp.ClientSession(
            timeout=timeout, headers=headers, connector=aiohttp.TCPConnector(limit=10)
        )

        logger.info("✅ Memento Temporal Connector initialized")

    async def close(self):
        """Zavření konektoru"""
        if self.session:
            await self.session.close()

    async def analyze_temporal_evolution(
        self, url: str, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> MementoTemporalResult:
        """Hlavní temporal analysis funkce

        Args:
            url: URL pro temporal analysis
            start_date: Počáteční datum (default: 5 let zpět)
            end_date: Konečné datum (default: dnes)

        Returns:
            MementoTemporalResult s temporal evolution analysis

        """
        start_time = asyncio.get_event_loop().time()

        # Set default timespan
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=5 * 365)  # 5 years back

        logger.info(f"Starting temporal analysis for {url} from {start_date} to {end_date}")

        try:
            # STEP 1: Get TimeMap for URL
            timemap = await self._get_timemap(url, start_date, end_date)

            # STEP 2: Select milestone snapshots
            milestone_dates = self._calculate_milestone_dates(start_date, end_date)
            selected_snapshots = await self._select_milestone_snapshots(timemap, milestone_dates)

            # STEP 3: Fetch snapshot content
            snapshots = await self._fetch_snapshots_content(selected_snapshots)

            # STEP 4: Perform diff analysis
            temporal_diffs = []
            if self.enable_content_diff and len(snapshots) > 1:
                temporal_diffs = await self._calculate_temporal_diffs(snapshots)

            # STEP 5: Analyze milestones and changes
            milestone_analysis = self._analyze_milestones(snapshots, temporal_diffs)
            change_timeline = self._build_change_timeline(snapshots, temporal_diffs)

            # STEP 6: Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(snapshots, temporal_diffs)

            processing_time = asyncio.get_event_loop().time() - start_time

            result = MementoTemporalResult(
                original_url=url,
                query_timespan=(start_date, end_date),
                snapshots=snapshots,
                temporal_diffs=temporal_diffs,
                milestone_analysis=milestone_analysis,
                change_timeline=change_timeline,
                processing_time=processing_time,
                quality_metrics=quality_metrics,
            )

            logger.info(
                f"Temporal analysis completed: {len(snapshots)} snapshots, {len(temporal_diffs)} diffs"
            )

            return result

        except Exception as e:
            logger.error(f"Temporal analysis failed: {e}")
            raise

    async def _get_timemap(
        self, url: str, start_date: datetime, end_date: datetime
    ) -> list[dict[str, Any]]:
        """Získání TimeMap pro URL"""
        # Build TimeMap request URL
        timemap_url = f"{self.memento_aggregator}/timemap/link/{url}"

        try:
            async with self.session.get(timemap_url) as response:
                if response.status == 200:
                    link_format_content = await response.text()
                    return self._parse_timemap_links(link_format_content, start_date, end_date)
                logger.warning(f"TimeMap request failed: {response.status}")
                return []

        except Exception as e:
            logger.error(f"TimeMap fetch error: {e}")
            return []

    def _parse_timemap_links(
        self, link_format_content: str, start_date: datetime, end_date: datetime
    ) -> list[dict[str, Any]]:
        """Parse TimeMap link format response"""
        mementos = []

        for line in link_format_content.strip().split("\n"):
            if 'rel="memento"' in line:
                # Parse link format: <url>; rel="memento"; datetime="..."
                url_match = re.search(r"<([^>]+)>", line)
                datetime_match = re.search(r'datetime="([^"]+)"', line)

                if url_match and datetime_match:
                    memento_url = url_match.group(1)
                    datetime_str = datetime_match.group(1)

                    try:
                        memento_datetime = datetime.strptime(
                            datetime_str, "%a, %d %b %Y %H:%M:%S GMT"
                        )

                        # Filter by date range
                        if start_date <= memento_datetime <= end_date:
                            mementos.append(
                                {
                                    "url": memento_url,
                                    "datetime": memento_datetime,
                                    "datetime_str": datetime_str,
                                }
                            )

                    except ValueError:
                        logger.warning(f"Failed to parse memento datetime: {datetime_str}")

        # Sort by datetime
        mementos.sort(key=lambda x: x["datetime"])

        logger.info(f"Found {len(mementos)} mementos in timespan")
        return mementos

    def _calculate_milestone_dates(
        self, start_date: datetime, end_date: datetime
    ) -> list[datetime]:
        """Výpočet milestone dates pro sampling"""
        milestones = []

        for interval in self.milestone_intervals:
            if interval == "1_year":
                delta = timedelta(days=365)
            elif interval == "6_months":
                delta = timedelta(days=183)
            elif interval == "3_months":
                delta = timedelta(days=91)
            elif interval == "1_month":
                delta = timedelta(days=30)
            elif interval == "1_week":
                delta = timedelta(days=7)
            else:
                continue

            current_date = start_date
            while current_date <= end_date:
                milestones.append(current_date)
                current_date += delta

        # Add start and end dates
        milestones.extend([start_date, end_date])

        # Remove duplicates and sort
        milestones = sorted(list(set(milestones)))

        return milestones[: self.max_snapshots]  # Limit number of milestones

    async def _select_milestone_snapshots(
        self, timemap: list[dict[str, Any]], milestone_dates: list[datetime]
    ) -> list[dict[str, Any]]:
        """Výběr snapshots closest to milestone dates"""
        selected = []

        for milestone_date in milestone_dates:
            # Find closest memento to milestone date
            closest_memento = None
            min_time_diff = float("inf")

            for memento in timemap:
                time_diff = abs((memento["datetime"] - milestone_date).total_seconds())
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_memento = memento

            if closest_memento and closest_memento not in selected:
                selected.append(closest_memento)

        # Also include first and last snapshots
        if timemap:
            if timemap[0] not in selected:
                selected.insert(0, timemap[0])
            if timemap[-1] not in selected:
                selected.append(timemap[-1])

        return selected

    async def _fetch_snapshots_content(
        self, selected_snapshots: list[dict[str, Any]]
    ) -> list[MementoSnapshot]:
        """Fetch content for selected snapshots"""
        snapshots = []

        # Process in batches for concurrent requests
        batch_size = self.concurrent_requests
        for i in range(0, len(selected_snapshots), batch_size):
            batch = selected_snapshots[i : i + batch_size]

            batch_tasks = []
            for snapshot_data in batch:
                task = self._fetch_snapshot_content(snapshot_data)
                batch_tasks.append(task)

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    logger.warning(f"Snapshot fetch failed: {result}")
                elif result:
                    snapshots.append(result)

        # Sort by datetime
        snapshots.sort(key=lambda x: x.datetime)

        return snapshots

    async def _fetch_snapshot_content(
        self, snapshot_data: dict[str, Any]
    ) -> MementoSnapshot | None:
        """Fetch content for single snapshot"""
        memento_url = snapshot_data["url"]

        try:
            async with self.session.get(memento_url) as response:
                if response.status == 200:
                    content = await response.text()

                    # Extract clean text content
                    clean_content = self._extract_text_content(content)

                    # Generate content hash for change detection
                    content_hash = hashlib.md5(clean_content.encode()).hexdigest()

                    # Determine archive name from URL
                    archive_name = self._determine_archive_name(memento_url)

                    snapshot = MementoSnapshot(
                        url=memento_url,
                        original_url=snapshot_data.get("original_url", ""),
                        datetime=snapshot_data["datetime"],
                        memento_datetime=snapshot_data["datetime_str"],
                        content=clean_content,
                        content_length=len(clean_content),
                        content_hash=content_hash,
                        archive_name=archive_name,
                        rel_type="memento",
                        extraction_metadata={
                            "fetch_timestamp": asyncio.get_event_loop().time(),
                            "response_status": response.status,
                            "content_type": response.headers.get("content-type", ""),
                            "original_length": len(content),
                        },
                    )

                    return snapshot

                logger.warning(f"Snapshot fetch failed: {response.status} for {memento_url}")
                return None

        except Exception as e:
            logger.error(f"Snapshot content fetch error: {e}")
            return None

    def _extract_text_content(self, html_content: str) -> str:
        """Extrakce čistého textu z HTML snapshot"""
        try:
            # Remove script and style elements
            html_content = re.sub(
                r"<script[^>]*>.*?</script>", "", html_content, flags=re.DOTALL | re.IGNORECASE
            )
            html_content = re.sub(
                r"<style[^>]*>.*?</style>", "", html_content, flags=re.DOTALL | re.IGNORECASE
            )

            # Remove HTML tags
            text = re.sub(r"<[^>]+>", "", html_content)

            # Clean up whitespace
            text = re.sub(r"\s+", " ", text).strip()

            # Limit size for diff analysis
            if len(text) > self.max_diff_size:
                text = text[: self.max_diff_size]

            return text

        except Exception as e:
            logger.warning(f"Text extraction failed: {e}")
            return (
                html_content[: self.max_diff_size]
                if len(html_content) > self.max_diff_size
                else html_content
            )

    def _determine_archive_name(self, memento_url: str) -> str:
        """Určení názvu archivu z URL"""
        if "web.archive.org" in memento_url:
            return "wayback_machine"
        if "archive.today" in memento_url or "archive.is" in memento_url:
            return "archive_today"
        if "webarchive.loc.gov" in memento_url:
            return "library_congress"
        parsed = urlparse(memento_url)
        return parsed.netloc

    async def _calculate_temporal_diffs(
        self, snapshots: list[MementoSnapshot]
    ) -> list[TemporalDiff]:
        """Výpočet temporal diffs mezi consecutive snapshots"""
        temporal_diffs = []

        for i in range(len(snapshots) - 1):
            snapshot_a = snapshots[i]
            snapshot_b = snapshots[i + 1]

            # Skip if content is identical
            if snapshot_a.content_hash == snapshot_b.content_hash:
                continue

            diff = await self._calculate_content_diff(snapshot_a, snapshot_b)
            if diff:
                temporal_diffs.append(diff)

        return temporal_diffs

    async def _calculate_content_diff(
        self, snapshot_a: MementoSnapshot, snapshot_b: MementoSnapshot
    ) -> TemporalDiff | None:
        """Výpočet diff mezi dvěma snapshots"""
        try:
            # Split content into lines for difflib
            lines_a = snapshot_a.content.split("\n")
            lines_b = snapshot_b.content.split("\n")

            # Calculate similarity ratio
            similarity_score = difflib.SequenceMatcher(None, lines_a, lines_b).ratio()

            # Generate unified diff
            diff_lines = list(
                difflib.unified_diff(
                    lines_a,
                    lines_b,
                    fromfile=f"snapshot_{snapshot_a.datetime.isoformat()}",
                    tofile=f"snapshot_{snapshot_b.datetime.isoformat()}",
                    n=self.diff_context_lines,
                )
            )

            # Analyze changes
            change_summary = self._analyze_diff_changes(diff_lines)

            # Determine diff type
            diff_type = self._classify_diff_type(change_summary)

            # Calculate significance score
            significance_score = self._calculate_change_significance(
                change_summary, similarity_score
            )

            # Skip insignificant changes
            if significance_score < self.min_content_change:
                return None

            # Create diff chunks for detailed analysis
            diff_chunks = self._create_diff_chunks(diff_lines)

            temporal_diff = TemporalDiff(
                snapshot_a=snapshot_a,
                snapshot_b=snapshot_b,
                diff_type=diff_type,
                diff_chunks=diff_chunks,
                similarity_score=similarity_score,
                change_summary=change_summary,
                significance_score=significance_score,
            )

            return temporal_diff

        except Exception as e:
            logger.error(f"Diff calculation failed: {e}")
            return None

    def _analyze_diff_changes(self, diff_lines: list[str]) -> dict[str, Any]:
        """Analýza změn v diff"""
        additions = 0
        deletions = 0
        modifications = 0

        for line in diff_lines:
            if line.startswith("+") and not line.startswith("+++"):
                additions += 1
            elif line.startswith("-") and not line.startswith("---"):
                deletions += 1
            elif line.startswith("!"):
                modifications += 1

        total_changes = additions + deletions + modifications

        return {
            "additions": additions,
            "deletions": deletions,
            "modifications": modifications,
            "total_changes": total_changes,
            "change_ratio": total_changes / len(diff_lines) if diff_lines else 0,
        }

    def _classify_diff_type(self, change_summary: dict[str, Any]) -> str:
        """Klasifikace typu změny"""
        additions = change_summary["additions"]
        deletions = change_summary["deletions"]

        if additions > deletions * 2:
            return "major_additions"
        if deletions > additions * 2:
            return "major_deletions"
        if additions > 0 and deletions > 0:
            return "content_modifications"
        if additions > 0:
            return "minor_additions"
        if deletions > 0:
            return "minor_deletions"
        return "no_significant_change"

    def _calculate_change_significance(
        self, change_summary: dict[str, Any], similarity_score: float
    ) -> float:
        """Výpočet significance score pro změnu"""
        # Base significance from change ratio
        change_ratio = change_summary["change_ratio"]

        # Adjust for similarity (more different = more significant)
        dissimilarity = 1.0 - similarity_score

        # Combine factors
        significance = (change_ratio * 0.6) + (dissimilarity * 0.4)

        return min(significance, 1.0)

    def _create_diff_chunks(self, diff_lines: list[str]) -> list[dict[str, Any]]:
        """Vytvoření diff chunks pro detailed analysis"""
        chunks = []
        current_chunk = []
        chunk_type = None

        for line in diff_lines:
            if line.startswith("@@"):  # Chunk header
                if current_chunk:
                    chunks.append({"type": chunk_type, "lines": current_chunk.copy()})
                    current_chunk = []

                chunk_type = "context"
            elif line.startswith("+"):
                chunk_type = "addition"
            elif line.startswith("-"):
                chunk_type = "deletion"
            elif line.startswith("!"):
                chunk_type = "modification"

            current_chunk.append(line)

        # Add final chunk
        if current_chunk:
            chunks.append({"type": chunk_type, "lines": current_chunk})

        return chunks

    def _analyze_milestones(
        self, snapshots: list[MementoSnapshot], temporal_diffs: list[TemporalDiff]
    ) -> dict[str, Any]:
        """Analýza milestones a významných změn"""
        analysis = {
            "total_snapshots": len(snapshots),
            "total_diffs": len(temporal_diffs),
            "timespan_days": (
                (snapshots[-1].datetime - snapshots[0].datetime).days if len(snapshots) >= 2 else 0
            ),
            "major_changes": [],
            "content_evolution": {},
            "archive_distribution": {},
        }

        # Identify major changes
        for diff in temporal_diffs:
            if diff.significance_score >= 0.3:  # Major change threshold
                analysis["major_changes"].append(
                    {
                        "timestamp": diff.snapshot_b.datetime.isoformat(),
                        "change_type": diff.diff_type,
                        "significance": diff.significance_score,
                        "summary": diff.change_summary,
                    }
                )

        # Archive distribution
        archive_counts = {}
        for snapshot in snapshots:
            archive_counts[snapshot.archive_name] = archive_counts.get(snapshot.archive_name, 0) + 1
        analysis["archive_distribution"] = archive_counts

        return analysis

    def _build_change_timeline(
        self, snapshots: list[MementoSnapshot], temporal_diffs: list[TemporalDiff]
    ) -> list[dict[str, Any]]:
        """Sestavení change timeline"""
        timeline = []

        # Add snapshots to timeline
        for snapshot in snapshots:
            timeline.append(
                {
                    "type": "snapshot",
                    "timestamp": snapshot.datetime.isoformat(),
                    "archive": snapshot.archive_name,
                    "content_length": snapshot.content_length,
                    "content_hash": snapshot.content_hash,
                }
            )

        # Add significant changes to timeline
        for diff in temporal_diffs:
            if diff.significance_score >= 0.1:  # Include significant changes
                timeline.append(
                    {
                        "type": "change",
                        "timestamp": diff.snapshot_b.datetime.isoformat(),
                        "change_type": diff.diff_type,
                        "significance": diff.significance_score,
                        "similarity": diff.similarity_score,
                        "changes": diff.change_summary,
                    }
                )

        # Sort by timestamp
        timeline.sort(key=lambda x: x["timestamp"])

        return timeline

    def _calculate_quality_metrics(
        self, snapshots: list[MementoSnapshot], temporal_diffs: list[TemporalDiff]
    ) -> dict[str, float]:
        """Výpočet quality metrics pro temporal analysis"""
        if not snapshots:
            return {
                "temporal_coverage": 0.0,
                "change_detection_rate": 0.0,
                "archive_diversity": 0.0,
            }

        # Temporal coverage
        if len(snapshots) >= 2:
            timespan_days = (snapshots[-1].datetime - snapshots[0].datetime).days
            expected_snapshots = max(1, timespan_days // 30)  # Monthly expectation
            temporal_coverage = min(len(snapshots) / expected_snapshots, 1.0)
        else:
            temporal_coverage = 0.0

        # Change detection rate
        total_possible_comparisons = len(snapshots) - 1
        detected_changes = len(temporal_diffs)
        change_detection_rate = (
            detected_changes / total_possible_comparisons if total_possible_comparisons > 0 else 0
        )

        # Archive diversity
        unique_archives = len(set(snapshot.archive_name for snapshot in snapshots))
        archive_diversity = unique_archives / len(snapshots)

        # Average change significance
        avg_significance = (
            sum(diff.significance_score for diff in temporal_diffs) / len(temporal_diffs)
            if temporal_diffs
            else 0
        )

        return {
            "temporal_coverage": temporal_coverage,
            "change_detection_rate": change_detection_rate,
            "archive_diversity": archive_diversity,
            "avg_change_significance": avg_significance,
        }

    async def get_connector_status(self) -> dict[str, Any]:
        """Získání status konektoru"""
        try:
            # Test Memento aggregator
            test_url = f"{self.memento_aggregator}/timemap/link/example.com"
            async with self.session.get(test_url) as response:
                aggregator_online = response.status in [200, 404]  # 404 is OK for non-existent URLs
        except:
            aggregator_online = False

        # Test individual archives
        archive_status = {}
        for name, endpoint in self.archive_endpoints.items():
            try:
                async with self.session.get(endpoint) as response:
                    archive_status[name] = response.status < 500
            except:
                archive_status[name] = False

        return {
            "connector_type": "memento_temporal",
            "aggregator_online": aggregator_online,
            "archive_endpoints": archive_status,
            "cache_enabled": self.cache_enabled,
            "max_snapshots": self.max_snapshots,
            "milestone_intervals": self.milestone_intervals,
        }
