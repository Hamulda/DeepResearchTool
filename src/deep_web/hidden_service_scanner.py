"""Hidden Service Scanner for Deep Web Discovery
Onion v3 pattern extraction and recursive link crawl with legal/safety filtering
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import urljoin, urlparse

import polars as pl

from ..core.memory_optimizer import MemoryOptimizer
from .network_manager import NetworkConfig, NetworkManager

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryConfig:
    """Hidden service discovery configuration"""

    # Discovery scope
    max_depth: int = 2
    max_links_per_page: int = 50
    max_total_links: int = 1000
    discovery_timeout_hours: int = 24

    # Onion validation
    validate_onion_format: bool = True
    onion_v3_only: bool = True
    require_https: bool = False

    # Content filtering
    content_filters: list[str] = None
    blocked_keywords: list[str] = None
    language_filters: list[str] = None

    # Rate limiting
    requests_per_minute: int = 30
    respect_robots: bool = True

    # Safety and legal
    legal_categories_only: bool = True
    blocked_categories: list[str] = None
    allowlist_sources: list[str] = None

    # Output
    save_page_content: bool = False
    save_screenshots: bool = False
    output_format: str = "parquet"  # parquet, json, csv

    def __post_init__(self):
        if self.content_filters is None:
            self.content_filters = ["research", "academic", "news", "library", "forum"]

        if self.blocked_keywords is None:
            self.blocked_keywords = [
                "illegal",
                "drugs",
                "weapons",
                "fraud",
                "hacking",
                "marketplace",
                "vendor",
                "shop",
                "buy",
                "sell",
            ]

        if self.blocked_categories is None:
            self.blocked_categories = [
                "marketplace",
                "illegal",
                "drugs",
                "weapons",
                "fraud",
                "hacking",
                "criminal",
                "adult",
                "gambling",
            ]

        if self.allowlist_sources is None:
            # Known legitimate onion services for research
            self.allowlist_sources = [
                "3g2upl4pq6kufc4m.onion",  # DuckDuckGo
                "facebookwkhpilnemxj7asaniu7vnjjbiltxjqhye3mhbshg7kx5tfyd.onion",  # Facebook
                "nytimes3xbfgragh.onion",  # NY Times
                "archivebyd3rzt3ehjpm5c3bjkyxv3hjleiytnvxcn7x32psn2kxcuid.onion",  # Archive.today
            ]


@dataclass
class DiscoveredService:
    """Discovered hidden service information"""

    onion_address: str
    title: str | None
    description: str | None
    category: str | None
    language: str | None
    discovered_at: datetime
    source_url: str
    depth: int
    status_code: int
    content_preview: str | None
    links_found: int
    safety_score: float
    metadata: dict[str, Any]


class OnionValidator:
    """Validate and categorize onion addresses"""

    # Onion v3 pattern (56 character base32)
    ONION_V3_PATTERN = re.compile(r"[a-z2-7]{56}\.onion")

    # Onion v2 pattern (16 character base32) - deprecated
    ONION_V2_PATTERN = re.compile(r"[a-z2-7]{16}\.onion")

    def __init__(self, config: DiscoveryConfig):
        self.config = config

    def is_valid_onion(self, address: str) -> bool:
        """Validate onion address format"""
        if not address.endswith(".onion"):
            return False

        if self.config.onion_v3_only:
            return bool(self.ONION_V3_PATTERN.match(address))
        return bool(
            self.ONION_V3_PATTERN.match(address) or self.ONION_V2_PATTERN.match(address)
        )

    def extract_onions_from_text(self, text: str) -> set[str]:
        """Extract all onion addresses from text"""
        onions = set()

        # Find all potential onion addresses
        if self.config.onion_v3_only:
            matches = self.ONION_V3_PATTERN.findall(text)
        else:
            matches = self.ONION_V3_PATTERN.findall(text) + self.ONION_V2_PATTERN.findall(text)

        for match in matches:
            if self.is_valid_onion(match):
                onions.add(match)

        return onions

    def categorize_service(self, content: str, title: str = "") -> tuple[str, float]:
        """Categorize service and assign safety score"""
        content_lower = (content + " " + title).lower()

        # Check for blocked content
        for keyword in self.config.blocked_keywords:
            if keyword in content_lower:
                return "blocked", 0.0

        # Categorize based on content
        if any(term in content_lower for term in ["research", "academic", "paper", "study"]):
            return "research", 0.9
        if any(term in content_lower for term in ["news", "journalism", "article"]):
            return "news", 0.8
        if any(term in content_lower for term in ["library", "archive", "documentation"]):
            return "library", 0.9
        if any(term in content_lower for term in ["forum", "discussion", "community"]):
            return "forum", 0.7
        if any(term in content_lower for term in ["privacy", "security", "encryption"]):
            return "privacy", 0.8
        if any(term in content_lower for term in ["blog", "personal", "diary"]):
            return "blog", 0.6
        return "unknown", 0.5


class HiddenServiceScanner:
    """Scanner for discovering and cataloging hidden services"""

    def __init__(
        self,
        config: DiscoveryConfig,
        network_manager: NetworkManager,
        optimizer: MemoryOptimizer,
        output_path: Path | None = None,
    ):
        self.config = config
        self.network_manager = network_manager
        self.optimizer = optimizer
        self.validator = OnionValidator(config)

        # State tracking
        self.discovered_services: dict[str, DiscoveredService] = {}
        self.visited_urls: set[str] = set()
        self.queue: list[tuple[str, int]] = []  # (url, depth)
        self.start_time = datetime.now()

        # Output management
        self.output_path = output_path or Path("hidden_services_discovery")
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Rate limiting
        self._last_request = 0.0
        self._request_interval = 60.0 / config.requests_per_minute

    async def discover_from_seeds(self, seed_urls: list[str]) -> AsyncIterator[DiscoveredService]:
        """Start discovery from seed URLs"""
        logger.info(f"Starting hidden service discovery from {len(seed_urls)} seeds")

        # Add seeds to queue
        for url in seed_urls:
            if self._is_safe_url(url):
                self.queue.append((url, 0))

        discovered_count = 0

        while self.queue and discovered_count < self.config.max_total_links:
            # Check timeout
            if datetime.now() - self.start_time > timedelta(
                hours=self.config.discovery_timeout_hours
            ):
                logger.info("Discovery timeout reached")
                break

            # Get next URL
            url, depth = self.queue.pop(0)

            if url in self.visited_urls or depth > self.config.max_depth:
                continue

            try:
                # Rate limiting
                await self._rate_limit()

                # Scan service
                service = await self._scan_service(url, depth)

                if service and service.safety_score > 0.5:
                    self.discovered_services[service.onion_address] = service
                    discovered_count += 1
                    yield service

                    # Extract links for next level
                    if depth < self.config.max_depth:
                        await self._extract_and_queue_links(url, depth + 1)

                # Memory pressure check
                if self.optimizer.check_memory_pressure()["pressure"]:
                    await self._save_checkpoint()
                    self.optimizer.force_gc()

            except Exception as e:
                logger.warning(f"Failed to scan {url}: {e}")
                continue

        logger.info(f"Discovery completed: {discovered_count} services found")

    async def _scan_service(self, url: str, depth: int) -> DiscoveredService | None:
        """Scan individual hidden service"""
        try:
            # Mark as visited
            self.visited_urls.add(url)

            # Make request through network manager
            async with self.network_manager.get(url) as response:
                if response.status != 200:
                    return None

                content = await response.text()

                # Extract metadata
                title = self._extract_title(content)
                description = self._extract_description(content)
                language = self._detect_language(content)

                # Categorize and score
                category, safety_score = self.validator.categorize_service(content, title or "")

                # Skip if blocked category
                if category in self.config.blocked_categories:
                    logger.info(f"Skipping blocked category '{category}': {url}")
                    return None

                # Extract onion address
                parsed_url = urlparse(url)
                onion_address = parsed_url.netloc

                # Count links
                links_found = len(self._extract_links(content, url))

                # Create service record
                service = DiscoveredService(
                    onion_address=onion_address,
                    title=title,
                    description=description,
                    category=category,
                    language=language,
                    discovered_at=datetime.now(),
                    source_url=url,
                    depth=depth,
                    status_code=response.status,
                    content_preview=content[:500] if self.config.save_page_content else None,
                    links_found=links_found,
                    safety_score=safety_score,
                    metadata={
                        "content_length": len(content),
                        "response_headers": dict(response.headers),
                        "scan_duration": time.time() - self._last_request,
                    },
                )

                logger.info(
                    f"Scanned {onion_address} - Category: {category}, Score: {safety_score:.2f}"
                )
                return service

        except Exception as e:
            logger.warning(f"Error scanning {url}: {e}")
            return None

    def _extract_title(self, content: str) -> str | None:
        """Extract page title"""
        title_match = re.search(r"<title[^>]*>([^<]+)</title>", content, re.IGNORECASE)
        if title_match:
            return title_match.group(1).strip()
        return None

    def _extract_description(self, content: str) -> str | None:
        """Extract page description from meta tags"""
        desc_match = re.search(
            r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']',
            content,
            re.IGNORECASE,
        )
        if desc_match:
            return desc_match.group(1).strip()
        return None

    def _detect_language(self, content: str) -> str | None:
        """Simple language detection"""
        # Look for lang attribute
        lang_match = re.search(r'<html[^>]*lang=["\']([^"\']+)["\']', content, re.IGNORECASE)
        if lang_match:
            return lang_match.group(1)

        # Simple heuristic based on common words
        if re.search(r"\b(the|and|or|but|in|on|at|to|for|of|with|by)\b", content, re.IGNORECASE):
            return "en"
        if re.search(
            r"\b(der|die|das|und|oder|aber|in|auf|zu|für|von|mit)\b", content, re.IGNORECASE
        ):
            return "de"
        if re.search(
            r"\b(le|la|les|et|ou|mais|dans|sur|à|pour|de|avec)\b", content, re.IGNORECASE
        ):
            return "fr"

        return None

    def _extract_links(self, content: str, base_url: str) -> list[str]:
        """Extract links from page content"""
        links = []

        # Find all links
        link_pattern = re.compile(r'<a[^>]*href=["\']([^"\']+)["\']', re.IGNORECASE)
        matches = link_pattern.findall(content)

        for href in matches:
            # Resolve relative URLs
            absolute_url = urljoin(base_url, href)

            # Only include .onion links
            if ".onion" in absolute_url and self.validator.is_valid_onion(
                urlparse(absolute_url).netloc
            ):
                links.append(absolute_url)

                if len(links) >= self.config.max_links_per_page:
                    break

        return links

    async def _extract_and_queue_links(self, url: str, next_depth: int):
        """Extract links and add to discovery queue"""
        try:
            async with self.network_manager.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    links = self._extract_links(content, url)

                    for link in links:
                        if link not in self.visited_urls and self._is_safe_url(link):
                            self.queue.append((link, next_depth))

        except Exception as e:
            logger.warning(f"Failed to extract links from {url}: {e}")

    def _is_safe_url(self, url: str) -> bool:
        """Check if URL is safe to scan"""
        # Check allowlist
        if self.config.allowlist_sources:
            domain = urlparse(url).netloc
            if domain not in self.config.allowlist_sources:
                return False

        # Validate onion format
        parsed = urlparse(url)
        if not self.validator.is_valid_onion(parsed.netloc):
            return False

        return True

    async def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self._last_request

        if time_since_last < self._request_interval:
            await asyncio.sleep(self._request_interval - time_since_last)

        self._last_request = time.time()

    async def _save_checkpoint(self):
        """Save discovery progress to disk"""
        checkpoint_data = {
            "discovered_services": {
                addr: {**service.__dict__, "discovered_at": service.discovered_at.isoformat()}
                for addr, service in self.discovered_services.items()
            },
            "visited_urls": list(self.visited_urls),
            "queue": self.queue,
            "start_time": self.start_time.isoformat(),
            "config": self.config.__dict__,
        }

        checkpoint_file = self.output_path / "discovery_checkpoint.json"
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.info(f"Checkpoint saved: {len(self.discovered_services)} services")

    async def save_results(self) -> Path:
        """Save discovery results to configured format"""
        if not self.discovered_services:
            logger.warning("No services to save")
            return None

        if self.config.output_format == "parquet":
            return await self._save_parquet()
        if self.config.output_format == "json":
            return await self._save_json()
        if self.config.output_format == "csv":
            return await self._save_csv()
        raise ValueError(f"Unsupported output format: {self.config.output_format}")

    async def _save_parquet(self) -> Path:
        """Save results as Parquet"""
        data = []
        for service in self.discovered_services.values():
            data.append(
                {
                    "onion_address": service.onion_address,
                    "title": service.title,
                    "description": service.description,
                    "category": service.category,
                    "language": service.language,
                    "discovered_at": service.discovered_at,
                    "source_url": service.source_url,
                    "depth": service.depth,
                    "status_code": service.status_code,
                    "links_found": service.links_found,
                    "safety_score": service.safety_score,
                    "content_length": service.metadata.get("content_length", 0),
                }
            )

        df = pl.DataFrame(data)
        output_file = (
            self.output_path / f"hidden_services_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        )
        df.write_parquet(output_file)

        logger.info(f"Results saved to {output_file}")
        return output_file

    async def _save_json(self) -> Path:
        """Save results as JSON"""
        data = {
            "discovery_metadata": {
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_services": len(self.discovered_services),
                "config": self.config.__dict__,
            },
            "services": [
                {**service.__dict__, "discovered_at": service.discovered_at.isoformat()}
                for service in self.discovered_services.values()
            ],
        }

        output_file = (
            self.output_path / f"hidden_services_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results saved to {output_file}")
        return output_file

    async def _save_csv(self) -> Path:
        """Save results as CSV"""
        data = []
        for service in self.discovered_services.values():
            data.append(
                {
                    "onion_address": service.onion_address,
                    "title": service.title or "",
                    "description": service.description or "",
                    "category": service.category,
                    "language": service.language or "",
                    "discovered_at": service.discovered_at.isoformat(),
                    "source_url": service.source_url,
                    "depth": service.depth,
                    "status_code": service.status_code,
                    "links_found": service.links_found,
                    "safety_score": service.safety_score,
                }
            )

        df = pl.DataFrame(data)
        output_file = (
            self.output_path / f"hidden_services_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        df.write_csv(output_file)

        logger.info(f"Results saved to {output_file}")
        return output_file

    def get_discovery_stats(self) -> dict[str, Any]:
        """Get discovery statistics"""
        services_by_category = {}
        services_by_language = {}

        for service in self.discovered_services.values():
            # Category stats
            category = service.category or "unknown"
            services_by_category[category] = services_by_category.get(category, 0) + 1

            # Language stats
            language = service.language or "unknown"
            services_by_language[language] = services_by_language.get(language, 0) + 1

        return {
            "total_services": len(self.discovered_services),
            "total_visited": len(self.visited_urls),
            "queue_remaining": len(self.queue),
            "discovery_duration": (datetime.now() - self.start_time).total_seconds(),
            "services_by_category": services_by_category,
            "services_by_language": services_by_language,
            "average_safety_score": sum(s.safety_score for s in self.discovered_services.values())
            / max(1, len(self.discovered_services)),
        }


# Utility functions
async def discover_hidden_services(
    seed_urls: list[str], config: DiscoveryConfig = None, output_path: Path = None
) -> dict[str, Any]:
    """Convenience function for hidden service discovery"""
    if config is None:
        config = DiscoveryConfig()

    # Setup network manager for Tor
    network_config = NetworkConfig(enable_tor=True, enable_clearnet=False, enable_i2p=False)

    optimizer = MemoryOptimizer(max_memory_gb=6.0)

    async with NetworkManager(network_config) as network_manager:
        scanner = HiddenServiceScanner(config, network_manager, optimizer, output_path)

        discovered_services = []
        async for service in scanner.discover_from_seeds(seed_urls):
            discovered_services.append(service)

        # Save results
        output_file = await scanner.save_results()
        stats = scanner.get_discovery_stats()

        return {
            "discovered_services": discovered_services,
            "output_file": str(output_file) if output_file else None,
            "statistics": stats,
        }


def create_safe_discovery_config() -> DiscoveryConfig:
    """Create safe configuration for hidden service discovery"""
    return DiscoveryConfig(
        max_depth=1,  # Shallow discovery
        max_total_links=100,  # Limited scope
        legal_categories_only=True,
        onion_v3_only=True,
        requests_per_minute=10,  # Conservative rate
        respect_robots=True,
        save_page_content=False,  # Privacy
        allowlist_sources=[
            # Only well-known legitimate services
            "3g2upl4pq6kufc4m.onion",  # DuckDuckGo
            "facebookwkhpilnemxj7asaniu7vnjjbiltxjqhye3mhbshg7kx5tfyd.onion",  # Facebook
        ],
    )


__all__ = [
    "DiscoveredService",
    "DiscoveryConfig",
    "HiddenServiceScanner",
    "OnionValidator",
    "create_safe_discovery_config",
    "discover_hidden_services",
]
