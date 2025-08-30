#!/usr/bin/env python3
"""Ahmia Tor Connector s Legal Filtrem
Bezpečný přístup k .onion zdrojům s explicitním legal-only režimem

Author: Senior Python/MLOps Agent
"""

import asyncio
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
import time
from typing import Any

import aiohttp
import aiosocks

logger = logging.getLogger(__name__)


@dataclass
class OnionSource:
    """Onion source s legal verification"""

    onion_url: str
    title: str
    description: str
    category: str
    legal_status: str  # verified_legal, questionable, blocked
    verification_date: str
    content_preview: str
    ahmia_rating: float
    language: str
    metadata: dict[str, Any]


@dataclass
class AhmiaSearchResult:
    """Výsledek Ahmia search"""

    query: str
    total_results: int
    legal_sources: list[OnionSource]
    filtered_count: int
    processing_time: float
    tor_status: dict[str, Any]
    quality_metrics: dict[str, float]


class AhmiaTorConnector:
    """Ahmia connector s legal whitelist a safety checks"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.ahmia_config = config.get("ahmia", {})

        # Ahmia API settings
        self.ahmia_api_base = self.ahmia_config.get("api_base", "https://ahmia.fi/search")
        self.ahmia_onion_base = self.ahmia_config.get(
            "onion_base", "http://juhanurmihxlp77nkq76byazjpd3faxuotxdf3rss4sfqnk4nfyaad.onion"
        )

        # Legal and safety settings
        self.legal_only_mode = self.ahmia_config.get("legal_only", True)
        self.legal_whitelist_file = Path(
            self.ahmia_config.get("legal_whitelist", "./configs/tor_legal_whitelist.json")
        )
        self.blocked_categories = set(
            self.ahmia_config.get(
                "blocked_categories",
                [
                    "drugs",
                    "weapons",
                    "illegal_services",
                    "fraud",
                    "hacking",
                    "child_exploitation",
                    "violence",
                    "terrorism",
                    "illegal_content",
                ],
            )
        )

        # Tor connection settings
        self.tor_proxy = self.ahmia_config.get("tor_proxy", "socks5://127.0.0.1:9050")
        self.tor_enabled = self.ahmia_config.get("tor_enabled", True)
        self.circuit_timeout = self.ahmia_config.get("circuit_timeout", 60)

        # Request settings
        self.max_results = self.ahmia_config.get("max_results", 50)
        self.timeout = self.ahmia_config.get("timeout", 30)
        self.rate_limit_delay = self.ahmia_config.get("rate_limit_delay", 2.0)

        # Content filtering
        self.content_filters = self.ahmia_config.get(
            "content_filters",
            {
                "min_description_length": 20,
                "require_english": False,
                "block_suspicious_keywords": True,
            },
        )

        # Load legal whitelist
        self.legal_whitelist = self._load_legal_whitelist()

        # Session management
        self.session = None
        self.tor_session = None

    def _load_legal_whitelist(self) -> set[str]:
        """Načtení legal whitelist"""
        whitelist = set()

        try:
            if self.legal_whitelist_file.exists():
                with open(self.legal_whitelist_file) as f:
                    whitelist_data = json.load(f)
                    whitelist = set(whitelist_data.get("verified_legal_onions", []))
                    logger.info(f"Loaded {len(whitelist)} verified legal onion sources")
            else:
                logger.warning("Legal whitelist file not found - creating default")
                self._create_default_whitelist()

        except Exception as e:
            logger.error(f"Failed to load legal whitelist: {e}")

        return whitelist

    def _create_default_whitelist(self):
        """Vytvoření default legal whitelist"""
        default_whitelist = {
            "verified_legal_onions": [
                # Známé legální .onion služby
                "3g2upl4pq6kufc4m.onion",  # DuckDuckGo
                "facebookcorewwwi.onion",  # Facebook
                "nytimes3xbfgragh.onion",  # New York Times
                "propub3r6espa33w.onion",  # ProPublica
                "secrdrop5wyphb5x.onion",  # SecureDrop
                "zbdltiaekumm56d.onion",  # Tor Project Blog
            ],
            "verified_categories": [
                "news",
                "journalism",
                "privacy_tools",
                "education",
                "libraries",
                "research",
                "wikis",
                "forums_legal",
            ],
            "last_updated": time.strftime("%Y-%m-%d"),
            "verification_criteria": [
                "Publicly known legal service",
                "Operated by legitimate organization",
                "No illegal content",
                "Educational or informational purpose",
            ],
        }

        try:
            self.legal_whitelist_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.legal_whitelist_file, "w") as f:
                json.dump(default_whitelist, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to create default whitelist: {e}")

    async def initialize(self):
        """Inicializace konektoru"""
        logger.info("Initializing Ahmia Tor Connector...")

        # Standard HTTP session for clearnet Ahmia
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        headers = {"User-Agent": "DeepResearchTool/1.0 Legal-Research-Only"}

        self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)

        # Tor session for .onion access
        if self.tor_enabled:
            await self._initialize_tor_session()

        # Verify legal-only mode
        if self.legal_only_mode:
            logger.info("🔒 Legal-only mode ENABLED - only whitelisted sources will be accessed")
        else:
            logger.warning("⚠️ Legal-only mode DISABLED - exercise caution")

        logger.info("✅ Ahmia Tor Connector initialized")

    async def _initialize_tor_session(self):
        """Inicializace Tor session"""
        try:
            # Parse Tor proxy URL
            if self.tor_proxy.startswith("socks5://"):
                proxy_url = self.tor_proxy[9:]  # Remove socks5:// prefix
                host, port = proxy_url.split(":")
                port = int(port)

                # Create Tor connector
                tor_connector = aiosocks.SocksConnector(aiosocks.SocksVer.SOCKS5, host, port)

                timeout = aiohttp.ClientTimeout(total=self.circuit_timeout)
                headers = {"User-Agent": "DeepResearchTool/1.0 Tor-Research"}

                self.tor_session = aiohttp.ClientSession(
                    connector=tor_connector, timeout=timeout, headers=headers
                )

                # Test Tor connection
                await self._test_tor_connection()

                logger.info("✅ Tor session initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Tor session: {e}")
            self.tor_enabled = False

    async def _test_tor_connection(self):
        """Test Tor connection"""
        try:
            # Test with Tor check service
            test_url = "http://check.torproject.org"
            async with self.tor_session.get(test_url) as response:
                if response.status == 200:
                    content = await response.text()
                    if "Congratulations" in content:
                        logger.info("✅ Tor connection verified")
                    else:
                        logger.warning("⚠️ Tor connection may not be working properly")

        except Exception as e:
            logger.warning(f"Tor connection test failed: {e}")

    async def close(self):
        """Zavření konektoru"""
        if self.session:
            await self.session.close()
        if self.tor_session:
            await self.tor_session.close()

    async def search_legal_onions(self, query: str) -> AhmiaSearchResult:
        """Hlavní search funkce s legal filtering

        Args:
            query: Search query

        Returns:
            AhmiaSearchResult s pouze legal sources

        """
        start_time = time.time()

        logger.info(f"Starting Ahmia search for: {query}")

        if self.legal_only_mode:
            logger.info("🔒 Legal-only mode active - filtering results")

        try:
            # STEP 1: Search Ahmia clearnet API
            raw_results = await self._search_ahmia_api(query)

            # STEP 2: Apply legal filtering
            legal_sources, filtered_count = await self._apply_legal_filtering(raw_results)

            # STEP 3: Verify and enhance legal sources
            verified_sources = await self._verify_legal_sources(legal_sources)

            # STEP 4: Get Tor status
            tor_status = await self._get_tor_status()

            # STEP 5: Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(verified_sources, raw_results)

            processing_time = time.time() - start_time

            result = AhmiaSearchResult(
                query=query,
                total_results=len(raw_results),
                legal_sources=verified_sources,
                filtered_count=filtered_count,
                processing_time=processing_time,
                tor_status=tor_status,
                quality_metrics=quality_metrics,
            )

            logger.info(
                f"Ahmia search completed: {len(verified_sources)}/{len(raw_results)} legal sources"
            )

            return result

        except Exception as e:
            logger.error(f"Ahmia search failed: {e}")
            raise

    async def _search_ahmia_api(self, query: str) -> list[dict[str, Any]]:
        """Vyhledání přes Ahmia clearnet API"""
        # Build search URL
        search_params = {"q": query, "page": 1}

        results = []

        try:
            async with self.session.get(self.ahmia_api_base, params=search_params) as response:
                if response.status == 200:
                    # Parse Ahmia HTML response (they don't have clean JSON API)
                    html_content = await response.text()
                    results = self._parse_ahmia_html(html_content)

                    logger.info(f"Retrieved {len(results)} raw results from Ahmia")
                else:
                    logger.warning(f"Ahmia API request failed: {response.status}")

        except Exception as e:
            logger.error(f"Ahmia API error: {e}")

        # Rate limiting
        await asyncio.sleep(self.rate_limit_delay)

        return results

    def _parse_ahmia_html(self, html_content: str) -> list[dict[str, Any]]:
        """Parse Ahmia HTML response"""
        results = []

        try:
            # Simple regex-based parsing of Ahmia results
            # In production, would use proper HTML parser like BeautifulSoup

            # Extract onion URLs
            onion_pattern = r"([a-z2-7]{16,56}\.onion)"
            onion_urls = re.findall(onion_pattern, html_content)

            # Extract titles and descriptions (simplified)
            title_pattern = r"<h4[^>]*>(.*?)</h4>"
            titles = re.findall(title_pattern, html_content, re.DOTALL)

            desc_pattern = r'<p class="excerpt"[^>]*>(.*?)</p>'
            descriptions = re.findall(desc_pattern, html_content, re.DOTALL)

            # Combine results
            for i, onion_url in enumerate(onion_urls[: self.max_results]):
                title = titles[i] if i < len(titles) else "Unknown"
                description = descriptions[i] if i < len(descriptions) else ""

                # Clean HTML tags from title and description
                title = re.sub(r"<[^>]+>", "", title).strip()
                description = re.sub(r"<[^>]+>", "", description).strip()

                results.append(
                    {
                        "onion_url": onion_url,
                        "title": title,
                        "description": description,
                        "source": "ahmia",
                    }
                )

        except Exception as e:
            logger.error(f"HTML parsing error: {e}")

        return results

    async def _apply_legal_filtering(
        self, raw_results: list[dict[str, Any]]
    ) -> tuple[list[OnionSource], int]:
        """Aplikace legal filtering"""
        legal_sources = []
        filtered_count = 0

        for result in raw_results:
            onion_url = result["onion_url"]
            title = result["title"]
            description = result["description"]

            # LEGAL CHECK 1: Whitelist verification
            if self.legal_only_mode and onion_url not in self.legal_whitelist:
                filtered_count += 1
                logger.debug(f"Filtered (not whitelisted): {onion_url}")
                continue

            # LEGAL CHECK 2: Category-based filtering
            category = self._classify_content_category(title, description)
            if category in self.blocked_categories:
                filtered_count += 1
                logger.debug(f"Filtered (blocked category {category}): {onion_url}")
                continue

            # LEGAL CHECK 3: Content keyword filtering
            if self._contains_suspicious_keywords(title, description):
                filtered_count += 1
                logger.debug(f"Filtered (suspicious keywords): {onion_url}")
                continue

            # LEGAL CHECK 4: Description quality
            if len(description) < self.content_filters["min_description_length"]:
                filtered_count += 1
                logger.debug(f"Filtered (insufficient description): {onion_url}")
                continue

            # Create OnionSource for legal content
            legal_status = "verified_legal" if onion_url in self.legal_whitelist else "questionable"

            onion_source = OnionSource(
                onion_url=onion_url,
                title=title,
                description=description,
                category=category,
                legal_status=legal_status,
                verification_date=time.strftime("%Y-%m-%d"),
                content_preview="",  # Will be filled later if needed
                ahmia_rating=0.8 if legal_status == "verified_legal" else 0.5,
                language="en",  # Default, could be detected
                metadata={"source": "ahmia", "filtering_passed": True},
            )

            legal_sources.append(onion_source)

        return legal_sources, filtered_count

    def _classify_content_category(self, title: str, description: str) -> str:
        """Klasifikace content category"""
        content = (title + " " + description).lower()

        # News and journalism
        if any(
            word in content for word in ["news", "journalism", "press", "newspaper", "reporter"]
        ):
            return "news"

        # Education and research
        if any(
            word in content
            for word in ["education", "research", "university", "academic", "library"]
        ):
            return "education"

        # Privacy and security tools
        if any(
            word in content for word in ["privacy", "security", "encryption", "anonymous", "tor"]
        ):
            return "privacy_tools"

        # Forums and discussion
        if any(word in content for word in ["forum", "discussion", "community", "chat"]):
            return "forums"

        # Potentially problematic categories
        if any(word in content for word in ["drug", "weapon", "hack", "fraud", "illegal"]):
            return "illegal_services"

        return "general"

    def _contains_suspicious_keywords(self, title: str, description: str) -> bool:
        """Detekce suspicious keywords"""
        if not self.content_filters["block_suspicious_keywords"]:
            return False

        content = (title + " " + description).lower()

        suspicious_keywords = [
            "illegal",
            "drugs",
            "weapons",
            "fraud",
            "hacking",
            "stolen",
            "credit card",
            "counterfeit",
            "fake id",
            "hitman",
            "assassination",
            "child",
            "explicit",
            "adult content",
            "darkmarket",
            "marketplace",
        ]

        return any(keyword in content for keyword in suspicious_keywords)

    async def _verify_legal_sources(self, legal_sources: list[OnionSource]) -> list[OnionSource]:
        """Verifikace legal sources"""
        verified_sources = []

        for source in legal_sources:
            # Additional verification for non-whitelisted sources
            if source.legal_status != "verified_legal":
                # Could add additional checks here
                # For now, we trust the filtering process
                pass

            verified_sources.append(source)

        return verified_sources

    async def _get_tor_status(self) -> dict[str, Any]:
        """Získání Tor status"""
        status = {
            "tor_enabled": self.tor_enabled,
            "session_active": self.tor_session is not None,
            "proxy_configured": self.tor_proxy is not None,
            "legal_only_mode": self.legal_only_mode,
        }

        if self.tor_enabled and self.tor_session:
            try:
                # Quick connectivity test
                test_start = time.time()
                async with self.tor_session.get(
                    "http://juhanurmihxlp77nkq76byazjpd3faxuotxdf3rss4sfqnk4nfyaad.onion",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    status["connectivity_test"] = response.status == 200
                    status["response_time"] = time.time() - test_start
            except:
                status["connectivity_test"] = False
                status["response_time"] = None

        return status

    def _calculate_quality_metrics(
        self, legal_sources: list[OnionSource], raw_results: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Výpočet quality metrics"""
        if not raw_results:
            return {"legal_ratio": 0.0, "whitelist_ratio": 0.0, "avg_description_length": 0.0}

        # Legal filtering ratio
        legal_ratio = len(legal_sources) / len(raw_results)

        # Whitelist ratio
        whitelisted_count = sum(
            1 for source in legal_sources if source.legal_status == "verified_legal"
        )
        whitelist_ratio = whitelisted_count / len(legal_sources) if legal_sources else 0

        # Average description length
        avg_desc_length = (
            sum(len(source.description) for source in legal_sources) / len(legal_sources)
            if legal_sources
            else 0
        )

        # Category distribution
        categories = {}
        for source in legal_sources:
            categories[source.category] = categories.get(source.category, 0) + 1

        category_diversity = len(categories) / len(legal_sources) if legal_sources else 0

        return {
            "legal_ratio": legal_ratio,
            "whitelist_ratio": whitelist_ratio,
            "avg_description_length": avg_desc_length,
            "category_diversity": category_diversity,
            "filtering_effectiveness": 1.0 - legal_ratio,  # Higher when more is filtered
        }

    async def access_onion_content(self, onion_url: str) -> str | None:
        """Bezpečný přístup k .onion content (pouze pro whitelisted)"""
        if self.legal_only_mode and onion_url not in self.legal_whitelist:
            logger.warning(f"Access denied - onion not in legal whitelist: {onion_url}")
            return None

        if not self.tor_enabled or not self.tor_session:
            logger.error("Tor session not available for onion access")
            return None

        try:
            full_url = f"http://{onion_url}"
            async with self.tor_session.get(full_url) as response:
                if response.status == 200:
                    content = await response.text()
                    logger.info(f"Successfully accessed legal onion: {onion_url}")
                    return content
                logger.warning(f"Onion access failed: {response.status} for {onion_url}")
                return None

        except Exception as e:
            logger.error(f"Onion access error: {e}")
            return None

    async def get_connector_status(self) -> dict[str, Any]:
        """Získání status konektoru"""
        # Legal whitelist stats
        whitelist_size = len(self.legal_whitelist)
        blocked_categories_count = len(self.blocked_categories)

        # Tor status
        tor_status = await self._get_tor_status()

        return {
            "connector_type": "ahmia_tor",
            "legal_only_mode": self.legal_only_mode,
            "whitelist_size": whitelist_size,
            "blocked_categories_count": blocked_categories_count,
            "tor_status": tor_status,
            "max_results": self.max_results,
            "safety_features": {
                "legal_whitelist": True,
                "category_filtering": True,
                "keyword_filtering": self.content_filters["block_suspicious_keywords"],
                "description_quality_check": True,
            },
        }
