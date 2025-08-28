#!/usr/bin/env python3
"""
Deep Web Crawler with Tor Integration
Safe and ethical crawling of .onion domains and deep web content

Author: Advanced IT Specialist
"""

import asyncio
import aiohttp
import aiohttp_socks
import logging
import re
import hashlib
from typing import List, Dict, Any, Optional, AsyncGenerator, Set
from datetime import datetime
from dataclasses import dataclass
import json
import time
from urllib.parse import urljoin, urlparse
import socket

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)

@dataclass
class OnionSite:
    """Represents an onion site"""
    onion_url: str
    title: str
    content: str
    last_seen: datetime
    status: str  # active, inactive, suspicious
    content_type: str
    safety_score: float
    content_hash: str
    links: List[str]
    metadata: Dict[str, Any]

@dataclass
class DeepWebResult:
    """Deep web search result"""
    url: str
    title: str
    content: str
    source_type: str  # onion, i2p, hidden_service
    discovery_method: str
    safety_validated: bool
    content_categories: List[str]
    risk_indicators: List[str]

class DeepWebCrawler(BaseScraper):
    """Safe deep web crawler with Tor integration"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "deep_web_crawler"

        # Tor configuration
        self.tor_enabled = config.get('tor_integration', {}).get('enabled', False)
        self.socks_port = config.get('tor', {}).get('socks_port', 9050)
        self.control_port = config.get('tor', {}).get('control_port', 9051)
        self.proxy_rotation = config.get('deep_web', {}).get('proxy_rotation', True)
        self.safety_checks = config.get('deep_web', {}).get('safety_checks', True)
        self.max_depth = config.get('deep_web', {}).get('max_depth', 3)

        # Safety configuration
        self.content_filters = config.get('deep_web', {}).get('content_filters', [
            'malware', 'illegal', 'harmful', 'explicit'
        ])
        self.max_onion_domains = config.get('deep_web', {}).get('max_onion_domains', 100)

        # Known safe onion directories
        self.safe_directories = [
            'http://3g2upl4pq6kufc4m.onion',  # DuckDuckGo
            'https://facebookwkhpilnemxj7asaniu7vnjjbiltxjqhye3mhbshg7kx5tfyd.onion',  # Facebook
            'https://www.facebookwkhpilnemxj7asaniu7vnjjbiltxjqhye3mhbshg7kx5tfyd.onion',
        ]

        # Ahmia.fi integration for onion indexing
        self.ahmia_api = "https://ahmia.fi/search/"
        self.ahmia_onions_api = "https://ahmia.fi/onions/"

        # Rate limiting
        self.last_request_time = 0
        self.min_delay = 2.0  # Minimum 2 seconds between requests for safety

        # Visited URLs tracking
        self.visited_urls: Set[str] = set()
        self.session_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'blocked_unsafe': 0,
            'onion_sites_found': 0
        }

    async def search(self, query: str, **kwargs) -> AsyncGenerator[DeepWebResult, None]:
        """Search deep web safely using multiple methods"""
        logger.info(f"Starting deep web search for: {query}")

        if not self.tor_enabled:
            logger.warning("Tor integration disabled - skipping deep web search")
            return

        # Check Tor connectivity
        if not await self._check_tor_connectivity():
            logger.error("Tor not available - cannot perform deep web search")
            return

        # Search through Ahmia.fi (safe onion indexing)
        async for result in self._search_ahmia(query, **kwargs):
            yield result

        # Search known safe directories
        async for result in self._search_safe_directories(query, **kwargs):
            yield result

        # Crawl discovered onion sites (with strict safety checks)
        async for result in self._crawl_discovered_onions(query, **kwargs):
            yield result

    async def _search_ahmia(self, query: str, **kwargs) -> AsyncGenerator[DeepWebResult, None]:
        """Search using Ahmia.fi - ethical onion search engine"""
        try:
            # Create Tor-enabled session
            connector = aiohttp_socks.ProxyConnector.from_url(f'socks5://127.0.0.1:{self.socks_port}')

            async with aiohttp.ClientSession(connector=connector) as session:
                # Search Ahmia.fi
                params = {
                    'q': query,
                    'redirect': 'true'
                }

                await self._respect_rate_limit()
                async with session.get(self.ahmia_api, params=params) as response:
                    if response.status == 200:
                        content = await response.text()
                        results = self._parse_ahmia_results(content)

                        for onion_url in results:
                            if await self._is_safe_to_visit(onion_url):
                                result = await self._fetch_onion_content(onion_url, session, query)
                                if result:
                                    yield result
                    else:
                        logger.warning(f"Ahmia search failed: {response.status}")

        except Exception as e:
            logger.error(f"Error searching Ahmia: {e}")

    async def _search_safe_directories(self, query: str, **kwargs) -> AsyncGenerator[DeepWebResult, None]:
        """Search through known safe onion directories"""
        try:
            connector = aiohttp_socks.ProxyConnector.from_url(f'socks5://127.0.0.1:{self.socks_port}')

            async with aiohttp.ClientSession(connector=connector) as session:
                for directory_url in self.safe_directories:
                    try:
                        await self._respect_rate_limit()
                        result = await self._search_directory(directory_url, query, session)
                        if result:
                            yield result
                    except Exception as e:
                        logger.warning(f"Error searching directory {directory_url}: {e}")

        except Exception as e:
            logger.error(f"Error in safe directories search: {e}")

    async def _crawl_discovered_onions(self, query: str, **kwargs) -> AsyncGenerator[DeepWebResult, None]:
        """Crawl discovered onion sites with safety validation"""
        max_sites = kwargs.get('max_sites', 10)  # Limit for safety
        crawled_count = 0

        try:
            connector = aiohttp_socks.ProxyConnector.from_url(f'socks5://127.0.0.1:{self.socks_port}')

            async with aiohttp.ClientSession(connector=connector) as session:
                # Get list of known onions from Ahmia
                onion_list = await self._get_ahmia_onion_list(session)

                for onion_url in onion_list:
                    if crawled_count >= max_sites:
                        break

                    if await self._is_safe_to_visit(onion_url):
                        result = await self._fetch_onion_content(onion_url, session, query)
                        if result and self._content_matches_query(result.content, query):
                            yield result
                            crawled_count += 1

        except Exception as e:
            logger.error(f"Error crawling discovered onions: {e}")

    async def _fetch_onion_content(self, onion_url: str, session: aiohttp.ClientSession, query: str) -> Optional[DeepWebResult]:
        """Safely fetch content from onion site"""
        try:
            if onion_url in self.visited_urls:
                return None

            self.visited_urls.add(onion_url)

            # Strict timeout for safety
            timeout = aiohttp.ClientTimeout(total=30)

            await self._respect_rate_limit()
            async with session.get(onion_url, timeout=timeout, ssl=False) as response:
                if response.status == 200:
                    content = await response.text()

                    # Safety validation
                    if not await self._validate_content_safety(content):
                        self.session_stats['blocked_unsafe'] += 1
                        logger.warning(f"Blocked unsafe content from {onion_url}")
                        return None

                    # Extract metadata
                    title = self._extract_title(content)
                    content_categories = self._categorize_content(content)
                    risk_indicators = self._detect_risk_indicators(content)

                    self.session_stats['successful_requests'] += 1
                    self.session_stats['onion_sites_found'] += 1

                    return DeepWebResult(
                        url=onion_url,
                        title=title,
                        content=self._clean_content(content),
                        source_type="onion",
                        discovery_method="tor_crawl",
                        safety_validated=True,
                        content_categories=content_categories,
                        risk_indicators=risk_indicators
                    )
                else:
                    logger.warning(f"Failed to fetch {onion_url}: {response.status}")

        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching {onion_url}")
        except Exception as e:
            logger.warning(f"Error fetching {onion_url}: {e}")

        self.session_stats['total_requests'] += 1
        return None

    async def _get_ahmia_onion_list(self, session: aiohttp.ClientSession) -> List[str]:
        """Get list of onion sites from Ahmia.fi"""
        try:
            await self._respect_rate_limit()
            async with session.get(self.ahmia_onions_api) as response:
                if response.status == 200:
                    data = await response.json()
                    # Extract onion URLs from Ahmia's API response
                    onions = []
                    for item in data.get('results', []):
                        onion_url = item.get('address')
                        if onion_url and onion_url.endswith('.onion'):
                            onions.append(f"http://{onion_url}")
                    return onions[:self.max_onion_domains]

        except Exception as e:
            logger.error(f"Error getting Ahmia onion list: {e}")

        return []

    def _parse_ahmia_results(self, html_content: str) -> List[str]:
        """Parse Ahmia search results to extract onion URLs"""
        onion_urls = []

        # Extract onion URLs from search results
        onion_pattern = r'href="([a-z0-9]{16,56}\.onion[^"]*)"'
        matches = re.findall(onion_pattern, html_content, re.IGNORECASE)

        for match in matches:
            if not match.startswith('http'):
                match = f"http://{match}"
            onion_urls.append(match)

        return list(set(onion_urls))  # Remove duplicates

    async def _search_directory(self, directory_url: str, query: str, session: aiohttp.ClientSession) -> Optional[DeepWebResult]:
        """Search within a safe onion directory"""
        try:
            await self._respect_rate_limit()
            async with session.get(directory_url) as response:
                if response.status == 200:
                    content = await response.text()

                    # Simple keyword search in content
                    if self._content_matches_query(content, query):
                        title = self._extract_title(content)

                        return DeepWebResult(
                            url=directory_url,
                            title=title,
                            content=self._clean_content(content),
                            source_type="onion",
                            discovery_method="safe_directory",
                            safety_validated=True,
                            content_categories=self._categorize_content(content),
                            risk_indicators=[]
                        )

        except Exception as e:
            logger.warning(f"Error searching directory {directory_url}: {e}")

        return None

    async def _is_safe_to_visit(self, url: str) -> bool:
        """Determine if URL is safe to visit"""
        if not url or not url.endswith('.onion'):
            return False

        # Check against known unsafe patterns
        unsafe_patterns = [
            r'market',
            r'drugs?',
            r'weapon',
            r'hack',
            r'fraud',
            r'illegal',
            r'cp\d*',
            r'exploit'
        ]

        url_lower = url.lower()
        for pattern in unsafe_patterns:
            if re.search(pattern, url_lower):
                logger.warning(f"Blocked potentially unsafe URL: {url}")
                return False

        return True

    async def _validate_content_safety(self, content: str) -> bool:
        """Validate content safety using multiple checks"""
        if not content:
            return False

        content_lower = content.lower()

        # Check for harmful content indicators
        harmful_indicators = [
            'malware', 'virus', 'trojan', 'exploit',
            'illegal', 'drugs', 'weapons', 'fraud',
            'child', 'explicit', 'adult content',
            'marketplace', 'darknet market'
        ]

        for indicator in harmful_indicators:
            if indicator in content_lower:
                return False

        # Check content length (too short might be suspicious)
        if len(content.strip()) < 100:
            return False

        return True

    def _content_matches_query(self, content: str, query: str) -> bool:
        """Check if content matches search query"""
        if not content or not query:
            return False

        content_lower = content.lower()
        query_terms = query.lower().split()

        # Require at least half of query terms to be present
        matches = sum(1 for term in query_terms if term in content_lower)
        return matches >= len(query_terms) // 2

    def _extract_title(self, content: str) -> str:
        """Extract title from HTML content"""
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', content, re.IGNORECASE)
        if title_match:
            return title_match.group(1).strip()
        return "Untitled"

    def _categorize_content(self, content: str) -> List[str]:
        """Categorize content based on keywords"""
        categories = []
        content_lower = content.lower()

        category_keywords = {
            'news': ['news', 'article', 'report', 'journalist'],
            'forum': ['forum', 'discussion', 'post', 'thread'],
            'blog': ['blog', 'diary', 'personal', 'thoughts'],
            'academic': ['research', 'paper', 'study', 'academic'],
            'technology': ['tech', 'software', 'computer', 'programming'],
            'privacy': ['privacy', 'anonymous', 'secure', 'encryption'],
            'activism': ['activist', 'freedom', 'rights', 'protest']
        }

        for category, keywords in category_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                categories.append(category)

        return categories

    def _detect_risk_indicators(self, content: str) -> List[str]:
        """Detect potential risk indicators in content"""
        risk_indicators = []
        content_lower = content.lower()

        risk_patterns = {
            'phishing': ['phishing', 'fake login', 'steal password'],
            'malware': ['download exe', 'run this file', 'install software'],
            'scam': ['send money', 'wire transfer', 'bitcoin payment'],
            'illegal_services': ['illegal service', 'dark market', 'underground']
        }

        for risk_type, patterns in risk_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                risk_indicators.append(risk_type)

        return risk_indicators

    def _clean_content(self, content: str) -> str:
        """Clean and sanitize content"""
        # Remove HTML tags
        content = re.sub(r'<[^>]+>', '', content)

        # Remove scripts and styles
        content = re.sub(r'<script.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<style.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)

        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\n\s*\n', '\n\n', content)

        # Limit content length for safety
        max_length = 10000
        if len(content) > max_length:
            content = content[:max_length] + "... [truncated for safety]"

        return content.strip()

    async def _check_tor_connectivity(self) -> bool:
        """Check if Tor is running and accessible"""
        try:
            # Test SOCKS proxy
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('127.0.0.1', self.socks_port))
            sock.close()

            if result != 0:
                logger.error(f"Cannot connect to Tor SOCKS proxy on port {self.socks_port}")
                return False

            # Test with a simple Tor request
            connector = aiohttp_socks.ProxyConnector.from_url(f'socks5://127.0.0.1:{self.socks_port}')

            async with aiohttp.ClientSession(connector=connector) as session:
                # Test with Tor check service
                async with session.get('https://check.torproject.org/', timeout=aiohttp.ClientTimeout(total=15)) as response:
                    if response.status == 200:
                        content = await response.text()
                        if 'Congratulations. This browser is configured to use Tor.' in content:
                            logger.info("Tor connectivity verified")
                            return True

            logger.error("Tor connectivity test failed")
            return False

        except Exception as e:
            logger.error(f"Error checking Tor connectivity: {e}")
            return False

    async def _respect_rate_limit(self):
        """Implement strict rate limiting for safety"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_delay:
            sleep_time = self.min_delay - time_since_last
            await asyncio.sleep(sleep_time)

        self.last_request_time = time.time()
        self.session_stats['total_requests'] += 1

    async def health_check(self) -> bool:
        """Check if deep web crawler is ready"""
        if not self.tor_enabled:
            return False

        return await self._check_tor_connectivity()

    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        return {
            **self.session_stats,
            'visited_urls_count': len(self.visited_urls),
            'safety_checks_enabled': self.safety_checks,
            'tor_enabled': self.tor_enabled
        }
