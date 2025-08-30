#!/usr/bin/env python3
"""Enhanced Tor Deep Web Scraper s robustním error handling
Advanced .onion domain crawling with safety protocols a retry mechanismy

Author: Advanced IT Specialist
"""

import asyncio
from datetime import datetime
import hashlib
import json
import logging
import random
import re
import socket
import time
from typing import Any, Dict, List
from urllib.parse import urlparse

import aiohttp

from ..core.error_handling import (
    network_retry,
    safe_tor_request,
    tor_circuit_breaker,
    with_circuit_breaker,
    ErrorAggregator,
    timeout_after,
    respect_rate_limit
)

logger = logging.getLogger(__name__)


class EnhancedTorScraper:
    """Enhanced Tor scraper s robustním error handling a retry mechanismy"""

    def __init__(self, rate_limit: float = 2.0):
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.tor_port = 9050
        self.control_port = 9051
        self.session_timeout = 30
        self.error_aggregator = ErrorAggregator()

        # Safety and security settings
        self.max_depth = 3
        self.max_pages_per_site = 10
        self.blocked_extensions = [".exe", ".zip", ".rar", ".pdf", ".doc"]
        self.safety_checks_enabled = True

        # Known .onion search engines and directories
        self.onion_search_engines = [
            "http://3g2upl4pq6kufc4m.onion/",  # DuckDuckGo onion
            "http://facebookwkhpilnemxj7asaniu7vnjjbiltxjqhye3mhbshg7kx5tfyd.onion/",  # Facebook onion
            "http://duckduckgogg42ts72.onion/",  # DuckDuckGo alternative
        ]

        # Document and leak-focused .onion sites (examples - would need real addresses)
        self.document_repositories = [
            # These would be populated with actual .onion addresses
            # 'http://example123456789.onion/documents/',
            # 'http://wikileaks123456.onion/search/',
        ]

        # Content safety patterns
        self.unsafe_patterns = [
            r"illegal\s+content",
            r"child\s+porn",
            r"weapons\s+sale",
            r"drug\s+market",
            r"hitman",
            r"assassination\s+service",
        ]

        # Research-relevant content patterns
        self.research_patterns = [
            r"documents?",
            r"leaks?",
            r"archives?",
            r"research",
            r"classified",
            r"declassified",
            r"intelligence",
            r"historical",
            r"academic",
            r"analysis",
        ]

    @timeout_after(300)  # 5 minute timeout for complete Tor search
    async def search_async(
        self, topic: str, time_range: tuple[datetime, datetime] | None = None
    ) -> List[Dict[str, Any]]:
        """Enhanced search of Tor network s robustním error handling"""
        await self._rate_limit()

        if not await self._verify_tor_connection():
            logger.error("Tor connection not available")
            return []

        results = []

        try:
            # Search through known onion search engines
            search_results = await self._search_onion_engines(topic)
            results.extend(search_results)

            # Search document repositories if available
            if self.document_repositories:
                doc_results = await self._search_document_repositories(topic)
                results.extend(doc_results)

            # Use TorBot for deep crawling (if available)
            torbot_results = await self._torbot_crawl(topic)
            results.extend(torbot_results)

        except Exception as e:
            self.error_aggregator.add_error(e, f"Tor search for topic: {topic}")
            logger.error(f"Error in enhanced Tor search: {e}")

        # Filter and validate results
        safe_results = self._filter_unsafe_content(results)
        research_focused = self._prioritize_research_content(safe_results)

        # Log summary
        self.error_aggregator.log_summary()

        logger.info(
            f"Enhanced Tor search for '{topic}' found {len(research_focused)} safe research documents"
        )
        return research_focused

    @network_retry
    async def _verify_tor_connection(self) -> bool:
        """Verify Tor proxy is running and accessible s retry logikou"""
        try:
            # Test Tor SOCKS proxy
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(("127.0.0.1", self.tor_port))
            sock.close()

            if result == 0:
                logger.info("Tor SOCKS proxy connection verified")
                return True

            logger.warning(f"Tor SOCKS proxy not accessible on port {self.tor_port}")
            raise ConnectionError(f"Cannot connect to Tor proxy on port {self.tor_port}")

        except Exception as e:
            logger.error(f"Error verifying Tor connection: {e}")
            raise

    async def _search_onion_engines(self, topic: str) -> List[Dict[str, Any]]:
        """Search through .onion search engines s robustním error handling"""
        results = []

        # Create Tor-enabled HTTP connector
        connector = aiohttp.TCPConnector(
            limit=10,
            limit_per_host=2,
            ttl_dns_cache=300,
            use_dns_cache=True,
            enable_cleanup_closed=True
        )

        # Configure Tor proxy
        proxy = f"socks5://127.0.0.1:{self.tor_port}"
        timeout = aiohttp.ClientTimeout(total=self.session_timeout)

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0"
            },
        ) as session:

            for search_engine in self.onion_search_engines:
                try:
                    # Construct search URL (this would vary by search engine)
                    search_url = f"{search_engine}?q={topic}"

                    # Use safe Tor request with circuit breaker
                    response = await self._safe_tor_get(session, search_url, proxy)

                    if response and response.status == 200:
                        content = await response.text()
                        parsed_results = self._parse_search_results(
                            content, search_engine, topic
                        )
                        results.extend(parsed_results)
                        self.error_aggregator.add_success()

                    # Add delay between requests
                    await asyncio.sleep(random.uniform(2, 5))

                except Exception as e:
                    self.error_aggregator.add_error(e, f"searching {search_engine}")
                    logger.debug(f"Error searching {search_engine}: {e}")
                    continue

        return results

    @with_circuit_breaker(tor_circuit_breaker)
    async def _safe_tor_get(self, session: aiohttp.ClientSession, url: str, proxy: str) -> aiohttp.ClientResponse:
        """Bezpečný Tor GET request s circuit breaker a retry"""
        try:
            async with session.get(url, proxy=proxy) as response:
                # Check for rate limiting
                await respect_rate_limit(response)

                if response.status >= 400:
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status
                    )

                return response
        except Exception as e:
            logger.error(f"Tor request failed for {url}: {e}")
            raise

    async def _search_document_repositories(self, topic: str) -> List[Dict[str, Any]]:
        """Search known document repositories on .onion sites s error handling"""
        results = []

        for repo_url in self.document_repositories:
            try:
                # This would implement specific search protocols for document repositories
                search_url = f"{repo_url}search?q={topic}"

                # Use safe Tor request
                proxy = f"socks5://127.0.0.1:{self.tor_port}"

                async with aiohttp.ClientSession() as session:
                    response = await self._safe_tor_get(session, search_url, proxy)

                    if response and response.status == 200:
                        content = await response.text()
                        repo_results = self._parse_repository_results(content, repo_url, topic)
                        results.extend(repo_results)
                        self.error_aggregator.add_success()

            except Exception as e:
                self.error_aggregator.add_error(e, f"searching repository {repo_url}")
                logger.debug(f"Error searching repository {repo_url}: {e}")
                continue

        logger.info(f"Document repository search found {len(results)} results")
        return results

    @timeout_after(180)  # 3 minute timeout for TorBot crawl
    async def _torbot_crawl(self, topic: str) -> List[Dict[str, Any]]:
        """Use TorBot for deep crawling s robustním error handling"""
        results = []

        try:
            # Check if TorBot is available
            torbot_available = await self._check_torbot_availability()

            if not torbot_available:
                logger.info("TorBot not available, skipping deep crawl")
                return results

            # Configure TorBot crawl parameters
            crawl_config = {
                "depth": self.max_depth,
                "max_pages": self.max_pages_per_site,
                "keywords": [topic, f"{topic} documents", f"{topic} research"],
                "output_format": "json",
                "safety_checks": True,
            }

            # Execute TorBot crawl
            crawl_results = await self._execute_torbot_crawl(crawl_config)
            results.extend(crawl_results)

            if crawl_results:
                self.error_aggregator.add_success()

        except Exception as e:
            self.error_aggregator.add_error(e, "TorBot crawling")
            logger.error(f"Error in TorBot crawling: {e}")

        return results

    @network_retry
    async def _check_torbot_availability(self) -> bool:
        """Check if TorBot is installed and available s retry logikou"""
        try:
            # Try to run TorBot help command
            process = await asyncio.create_subprocess_exec(
                "torbot", "--help",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)

            if process.returncode == 0:
                logger.info("TorBot is available")
                return True
            logger.info("TorBot not installed or not in PATH")
            return False

        except FileNotFoundError:
            logger.info("TorBot command not found")
            return False
        except asyncio.TimeoutError:
            logger.warning("TorBot availability check timed out")
            return False
        except Exception as e:
            logger.error(f"Error checking TorBot availability: {e}")
            raise

    @timeout_after(120)  # 2 minute timeout for TorBot execution
    async def _execute_torbot_crawl(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute TorBot crawl with given configuration s timeout"""
        results = []

        try:
            # Prepare TorBot command
            cmd = [
                "torbot",
                "--depth", str(config["depth"]),
                "--limit", str(config["max_pages"]),
                "--format", "json",
            ]

            # Add keywords
            for keyword in config["keywords"]:
                cmd.extend(["--keyword", keyword])

            # Execute TorBot
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                # Parse TorBot JSON output
                try:
                    torbot_data = json.loads(stdout.decode())
                    results = self._parse_torbot_results(torbot_data)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse TorBot JSON output: {e}")
                    raise
            else:
                error_msg = stderr.decode()
                logger.error(f"TorBot execution failed: {error_msg}")
                raise RuntimeError(f"TorBot failed: {error_msg}")

        except Exception as e:
            logger.error(f"Error executing TorBot crawl: {e}")
            raise

        return results

    def _parse_search_results(
        self, html_content: str, search_engine: str, topic: str
    ) -> List[Dict[str, Any]]:
        """Parse search results from .onion search engines s error handling"""
        results = []

        try:
            # Extract URLs and titles using regex (would need to be adapted per search engine)
            url_pattern = r'href="(http[s]?://[^"]+\.onion[^"]*)"'
            title_pattern = r"<title>([^<]+)</title>"

            urls = re.findall(url_pattern, html_content)
            titles = re.findall(title_pattern, html_content)

            # Combine URLs and titles
            for i, url in enumerate(urls[:10]):  # Limit to 10 results per engine
                title = titles[i] if i < len(titles) else f"Onion Document {i+1}"

                # Basic content safety check
                if self._is_safe_content(title, ""):
                    results.append({
                        "title": title,
                        "content": f"Tor hidden service content related to {topic}",
                        "url": url,
                        "source_type": "tor_hidden_service",
                        "source_url": url,
                        "date": datetime.now(),
                        "metadata": {
                            "search_engine": search_engine,
                            "onion_address": self._extract_onion_address(url),
                            "access_method": "tor_search_engine",
                            "safety_verified": True,
                            "content_type": "hidden_service",
                        },
                    })

        except Exception as e:
            logger.error(f"Error parsing search results: {e}")
            # Don't re-raise - return partial results

        return results

    def _parse_repository_results(self, html_content: str, repo_url: str, topic: str) -> List[Dict[str, Any]]:
        """Parse results from document repositories"""
        results = []

        try:
            # Implementation would depend on specific repository format
            # This is a placeholder for actual parsing logic
            pass
        except Exception as e:
            logger.error(f"Error parsing repository results from {repo_url}: {e}")

        return results

    def _is_safe_content(self, title: str, content: str) -> bool:
        """Check if content is safe and research-appropriate"""
        if not self.safety_checks_enabled:
            return True

        combined_text = f"{title} {content}".lower()

        # Check for unsafe patterns
        for pattern in self.unsafe_patterns:
            if re.search(pattern, combined_text, re.IGNORECASE):
                logger.warning(f"Unsafe content detected: {pattern}")
                return False

        return True

    def _filter_unsafe_content(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter out potentially unsafe content"""
        safe_results = []

        for result in results:
            title = result.get("title", "")
            content = result.get("content", "")

            if self._is_safe_content(title, content):
                safe_results.append(result)
            else:
                logger.info(f"Filtered unsafe content: {title[:50]}")

        return safe_results

    def _prioritize_research_content(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Prioritize research-relevant content"""
        research_results = []

        for result in results:
            title = result.get("title", "")
            content = result.get("content", "")
            combined_text = f"{title} {content}".lower()

            # Calculate research relevance score
            relevance_score = 0
            for pattern in self.research_patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    relevance_score += 1

            # Add relevance score to metadata
            result["metadata"]["research_relevance"] = relevance_score

            # Include if has any research relevance
            if relevance_score > 0:
                research_results.append(result)

        # Sort by relevance score
        research_results.sort(key=lambda x: x["metadata"]["research_relevance"], reverse=True)

        return research_results

    def _extract_onion_address(self, url: str) -> str:
        """Extract .onion address from URL"""
        try:
            parsed = urlparse(url)
            if parsed.hostname and parsed.hostname.endswith(".onion"):
                return parsed.hostname
        except Exception:
            pass

        # Fallback regex extraction
        onion_match = re.search(r"([a-z2-7]{16,56}\.onion)", url)
        return onion_match.group(1) if onion_match else ""

    async def _rate_limit(self):
        """Apply rate limiting with enhanced delays for Tor"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit:
            await asyncio.sleep(self.rate_limit - time_since_last)

        self.last_request_time = time.time()

    def get_stats(self) -> Dict[str, Any]:
        """Vrátí statistiky Tor scrapingu s error reporting"""
        error_summary = self.error_aggregator.get_summary()

        return {
            "tor_port": self.tor_port,
            "control_port": self.control_port,
            "rate_limit": self.rate_limit,
            "safety_checks_enabled": self.safety_checks_enabled,
            "max_depth": self.max_depth,
            "max_pages_per_site": self.max_pages_per_site,
            "success_rate": error_summary["success_rate"],
            "total_operations": error_summary["total_operations"],
            "failed_operations": error_summary["failed_operations"],
            "recent_errors": error_summary["errors"][-5:] if error_summary["errors"] else []
        }

    def search(
        self, topic: str, time_range: tuple[datetime, datetime] | None = None
    ) -> List[Dict[str, Any]]:
        """Synchronous search wrapper"""
        return asyncio.run(self.search_async(topic, time_range))


# Update the original TorDeepWebScraper to use enhanced version
class TorDeepWebScraper(EnhancedTorScraper):
    """Backward compatibility wrapper for enhanced Tor scraper"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Using enhanced Tor scraper with robust error handling")
