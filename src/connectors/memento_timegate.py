#!/usr/bin/env python3
"""
Memento TimeMap/TimeGate konektor
Orchestrace snapshots a diff analýza změn v čase

Author: Senior Python/MLOps Agent
"""

import asyncio
import json
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
import aiohttp
import aiofiles
from pathlib import Path
import difflib
import re
from urllib.parse import urljoin, urlparse


@dataclass
class MementoSnapshot:
    """Memento snapshot s metadaty"""
    url: str
    datetime: datetime
    memento_url: str
    archive_name: str
    content: str
    content_hash: str
    headers: Dict[str, str]
    status_code: int


@dataclass
class ContentDiff:
    """Diff mezi snapshots"""
    snapshot1_url: str
    snapshot2_url: str
    snapshot1_date: datetime
    snapshot2_date: datetime
    added_lines: List[str]
    removed_lines: List[str]
    modified_lines: List[Tuple[str, str]]
    similarity_ratio: float
    significant_changes: List[str]


class MementoConnector:
    """Memento konektor pro web archive access"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache_dir = Path(config.get("cache_dir", "research_cache/memento"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timegate_endpoints = config.get("timegate_endpoints", [
            "https://web.archive.org/web/",
            "https://archive.today/",
            "https://timetravel.mementoweb.org/timegate/"
        ])
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)

    async def get_timemap(self, url: str) -> List[MementoSnapshot]:
        """Získá TimeMap pro URL"""
        print(f"🕐 Getting TimeMap for: {url}")

        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_file = self.cache_dir / f"timemap_{cache_key}.json"

        # Kontrola cache
        if cache_file.exists():
            async with aiofiles.open(cache_file, 'r') as f:
                cached_data = json.loads(await f.read())
                snapshots = []
                for item in cached_data:
                    item['datetime'] = datetime.fromisoformat(item['datetime'])
                    snapshots.append(MementoSnapshot(**item))
                return snapshots

        snapshots = []

        for timegate in self.timegate_endpoints:
            try:
                archive_snapshots = await self._get_timemap_from_endpoint(url, timegate)
                snapshots.extend(archive_snapshots)
            except Exception as e:
                print(f"❌ TimeMap failed for {timegate}: {e}")
                continue

        # Deduplikuj podle datetime
        unique_snapshots = {}
        for snapshot in snapshots:
            key = snapshot.datetime.isoformat()
            if key not in unique_snapshots:
                unique_snapshots[key] = snapshot

        final_snapshots = list(unique_snapshots.values())
        final_snapshots.sort(key=lambda x: x.datetime)

        # Cache výsledky
        cache_data = []
        for snapshot in final_snapshots:
            item = {
                "url": snapshot.url,
                "datetime": snapshot.datetime.isoformat(),
                "memento_url": snapshot.memento_url,
                "archive_name": snapshot.archive_name,
                "content": snapshot.content,
                "content_hash": snapshot.content_hash,
                "headers": snapshot.headers,
                "status_code": snapshot.status_code
            }
            cache_data.append(item)

        async with aiofiles.open(cache_file, 'w') as f:
            await f.write(json.dumps(cache_data, indent=2))

        print(f"✅ Found {len(final_snapshots)} unique snapshots")
        return final_snapshots

    async def _get_timemap_from_endpoint(self,
                                       url: str,
                                       timegate: str) -> List[MementoSnapshot]:
        """Získá TimeMap z konkrétního endpointu"""
        snapshots = []
        archive_name = urlparse(timegate).netloc

        # TimeMap URL podle Memento protokolu
        timemap_url = urljoin(timegate, f"timemap/link/{url}")

        retry_count = 0
        while retry_count < self.max_retries:
            try:
                async with aiohttp.ClientSession() as session:
                    headers = {
                        "User-Agent": "DeepResearchTool/1.0 (+research@example.com)",
                        "Accept": "application/link-format"
                    }

                    async with session.get(timemap_url, headers=headers) as response:
                        if response.status == 200:
                            timemap_content = await response.text()
                            snapshots = self._parse_timemap(timemap_content, archive_name)
                            break
                        else:
                            raise aiohttp.ClientError(f"HTTP {response.status}")

            except Exception as e:
                retry_count += 1
                if retry_count >= self.max_retries:
                    raise e
                await asyncio.sleep(self.retry_delay * retry_count)

        return snapshots

    def _parse_timemap(self, timemap_content: str, archive_name: str) -> List[MementoSnapshot]:
        """Parsuje TimeMap response"""
        snapshots = []

        for line in timemap_content.strip().split('\n'):
            if not line.strip():
                continue

            # Parse Link header format
            # Example: <https://web.archive.org/web/20210101000000/http://example.com>; rel="memento"; datetime="Thu, 01 Jan 2021 00:00:00 GMT"
            match = re.search(r'<([^>]+)>.*?datetime="([^"]+)"', line)
            if match:
                memento_url = match.group(1)
                datetime_str = match.group(2)

                try:
                    # Parse datetime
                    dt = datetime.strptime(datetime_str, "%a, %d %b %Y %H:%M:%S %Z")
                    dt = dt.replace(tzinfo=timezone.utc)

                    snapshot = MementoSnapshot(
                        url="",  # Will be filled when fetching content
                        datetime=dt,
                        memento_url=memento_url,
                        archive_name=archive_name,
                        content="",  # Will be filled when fetching
                        content_hash="",
                        headers={},
                        status_code=0
                    )
                    snapshots.append(snapshot)

                except ValueError as e:
                    print(f"❌ Failed to parse datetime: {datetime_str}, {e}")
                    continue

        return snapshots

    async def fetch_snapshot_content(self, snapshot: MementoSnapshot) -> MementoSnapshot:
        """Stáhne obsah snapshot"""
        cache_key = hashlib.md5(snapshot.memento_url.encode()).hexdigest()
        cache_file = self.cache_dir / f"content_{cache_key}.txt"

        # Kontrola cache
        if cache_file.exists():
            async with aiofiles.open(cache_file, 'r') as f:
                content = await f.read()
                snapshot.content = content
                snapshot.content_hash = hashlib.md5(content.encode()).hexdigest()
                return snapshot

        retry_count = 0
        while retry_count < self.max_retries:
            try:
                async with aiohttp.ClientSession() as session:
                    headers = {
                        "User-Agent": "DeepResearchTool/1.0 (+research@example.com)"
                    }

                    async with session.get(snapshot.memento_url, headers=headers) as response:
                        if response.status == 200:
                            content = await response.text()

                            # Clean HTML content
                            cleaned_content = self._clean_html_content(content)

                            snapshot.content = cleaned_content
                            snapshot.content_hash = hashlib.md5(cleaned_content.encode()).hexdigest()
                            snapshot.headers = dict(response.headers)
                            snapshot.status_code = response.status

                            # Cache obsah
                            async with aiofiles.open(cache_file, 'w') as f:
                                await f.write(cleaned_content)

                            return snapshot
                        else:
                            raise aiohttp.ClientError(f"HTTP {response.status}")

            except Exception as e:
                retry_count += 1
                if retry_count >= self.max_retries:
                    print(f"❌ Content fetch failed: {e}")
                    return snapshot
                await asyncio.sleep(self.retry_delay * retry_count)

        return snapshot

    def _clean_html_content(self, html_content: str) -> str:
        """Vyčistí HTML obsah pro diff analýzu"""
        import re

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', html_content)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove common boilerplate
        boilerplate_patterns = [
            r'Skip to main content',
            r'Navigation menu',
            r'Copyright \d{4}',
            r'All rights reserved',
            r'Privacy Policy',
            r'Terms of Service'
        ]

        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        return text.strip()

    async def analyze_content_evolution(self,
                                      snapshots: List[MementoSnapshot]) -> List[ContentDiff]:
        """Analyzuje vývoj obsahu mezi snapshots"""
        print(f"📊 Analyzing content evolution across {len(snapshots)} snapshots")

        diffs = []

        # Fetch content pro všechny snapshots
        for i, snapshot in enumerate(snapshots):
            if not snapshot.content:
                snapshots[i] = await self.fetch_snapshot_content(snapshot)

        # Porovnej consecutive snapshots
        for i in range(len(snapshots) - 1):
            snapshot1 = snapshots[i]
            snapshot2 = snapshots[i + 1]

            if snapshot1.content and snapshot2.content:
                diff = self._calculate_content_diff(snapshot1, snapshot2)
                diffs.append(diff)

        print(f"✅ Generated {len(diffs)} content diffs")
        return diffs

    def _calculate_content_diff(self,
                              snapshot1: MementoSnapshot,
                              snapshot2: MementoSnapshot) -> ContentDiff:
        """Vypočítá diff mezi dvěma snapshots"""
        lines1 = snapshot1.content.split('\n')
        lines2 = snapshot2.content.split('\n')

        # Unified diff
        diff_lines = list(difflib.unified_diff(
            lines1, lines2,
            fromfile=f"snapshot_{snapshot1.datetime.isoformat()}",
            tofile=f"snapshot_{snapshot2.datetime.isoformat()}",
            lineterm=""
        ))

        # Parse diff
        added_lines = []
        removed_lines = []
        modified_lines = []

        for line in diff_lines:
            if line.startswith('+') and not line.startswith('+++'):
                added_lines.append(line[1:])
            elif line.startswith('-') and not line.startswith('---'):
                removed_lines.append(line[1:])

        # Detect modifications (removed + added with similarity)
        for removed in removed_lines[:]:
            for added in added_lines[:]:
                similarity = difflib.SequenceMatcher(None, removed, added).ratio()
                if similarity > 0.6:  # 60% similarity threshold
                    modified_lines.append((removed, added))
                    removed_lines.remove(removed)
                    added_lines.remove(added)
                    break

        # Overall similarity
        similarity_ratio = difflib.SequenceMatcher(None, snapshot1.content, snapshot2.content).ratio()

        # Detect significant changes
        significant_changes = self._detect_significant_changes(added_lines, removed_lines, modified_lines)

        return ContentDiff(
            snapshot1_url=snapshot1.memento_url,
            snapshot2_url=snapshot2.memento_url,
            snapshot1_date=snapshot1.datetime,
            snapshot2_date=snapshot2.datetime,
            added_lines=added_lines,
            removed_lines=removed_lines,
            modified_lines=modified_lines,
            similarity_ratio=similarity_ratio,
            significant_changes=significant_changes
        )

    def _detect_significant_changes(self,
                                   added_lines: List[str],
                                   removed_lines: List[str],
                                   modified_lines: List[Tuple[str, str]]) -> List[str]:
        """Detekuje významné změny"""
        significant_changes = []

        # Keywords indicating significant changes
        significant_keywords = [
            'error', 'warning', 'alert', 'notice', 'update', 'new', 'removed',
            'deprecated', 'changed', 'modified', 'breaking', 'important',
            'security', 'vulnerability', 'fix', 'bug', 'issue'
        ]

        all_changes = added_lines + removed_lines + [f"{old} -> {new}" for old, new in modified_lines]

        for change in all_changes:
            change_lower = change.lower()
            for keyword in significant_keywords:
                if keyword in change_lower:
                    significant_changes.append(f"Detected '{keyword}' in: {change[:100]}...")
                    break

        return significant_changes

    def save_diff_artifacts(self,
                          diffs: List[ContentDiff],
                          output_dir: str) -> Dict[str, str]:
        """Uloží diff artefakty"""
        artifacts = {}

        # Diff summary
        diff_summary = {
            "timestamp": datetime.now().isoformat(),
            "total_diffs": len(diffs),
            "diffs": []
        }

        for i, diff in enumerate(diffs):
            diff_data = {
                "diff_id": i,
                "snapshot1_url": diff.snapshot1_url,
                "snapshot2_url": diff.snapshot2_url,
                "snapshot1_date": diff.snapshot1_date.isoformat(),
                "snapshot2_date": diff.snapshot2_date.isoformat(),
                "similarity_ratio": diff.similarity_ratio,
                "changes_summary": {
                    "added_lines": len(diff.added_lines),
                    "removed_lines": len(diff.removed_lines),
                    "modified_lines": len(diff.modified_lines)
                },
                "significant_changes": diff.significant_changes
            }
            diff_summary["diffs"].append(diff_data)

        summary_file = f"{output_dir}/memento_diffs.json"
        with open(summary_file, "w") as f:
            json.dump(diff_summary, f, indent=2)

        artifacts["diff_summary"] = summary_file

        return artifacts
