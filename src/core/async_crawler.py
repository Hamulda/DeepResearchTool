"""Asynchronous Crawling Foundation
Rate-limited, resilient web crawling with streaming Parquet output
Optimized for MacBook Air M1 8GB RAM constraints
"""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Any, Callable
from urllib.parse import urljoin, urlparse

import aiohttp

from .memory_optimizer import MemoryOptimizer, ParquetDatasetManager

logger = logging.getLogger(__name__)


@dataclass
class CrawlResult:
    """Single crawl result with metadata"""

    url: str
    status_code: int
    content: str
    headers: dict[str, str]
    timestamp: datetime
    duration_ms: int
    error: str | None = None
    redirect_chain: list[str] | None = None
    content_type: str | None = None
    content_length: int | None = None


@dataclass
class CrawlConfig:
    """Crawling configuration"""

    max_concurrent: int = 10
    request_delay: float = 1.0
    timeout: int = 30
    max_retries: int = 3
    backoff_factor: float = 2.0
    respect_robots: bool = True
    user_agent: str = "DeepResearchTool/2.0 (+https://github.com/hamada/DeepResearchTool)"
    max_content_size: int = 10 * 1024 * 1024  # 10MB
    enable_streaming: bool = True


class PolitenessPolicy:
    """Implements polite crawling with domain-based rate limiting"""

    def __init__(self, default_delay: float = 1.0):
        self.default_delay = default_delay
        self.domain_delays: dict[str, float] = {}
        self.last_request_time: dict[str, float] = {}
        self.request_counts: dict[str, int] = {}

    async def wait_if_needed(self, url: str) -> None:
        """Wait if needed to respect rate limits"""
        domain = urlparse(url).netloc

        current_time = time.time()
        delay = self.domain_delays.get(domain, self.default_delay)

        if domain in self.last_request_time:
            elapsed = current_time - self.last_request_time[domain]
            if elapsed < delay:
                wait_time = delay - elapsed
                await asyncio.sleep(wait_time)

        self.last_request_time[domain] = time.time()
        self.request_counts[domain] = self.request_counts.get(domain, 0) + 1

    def adjust_delay(self, domain: str, response_code: int) -> None:
        """Adjust delay based on response codes"""
        current_delay = self.domain_delays.get(domain, self.default_delay)

        if response_code == 429:  # Too Many Requests
            self.domain_delays[domain] = min(current_delay * 2, 60.0)
        elif response_code in [503, 502, 504]:  # Server errors
            self.domain_delays[domain] = min(current_delay * 1.5, 30.0)
        elif 200 <= response_code < 300:
            # Gradually reduce delay on success
            self.domain_delays[domain] = max(current_delay * 0.95, self.default_delay)


class AsyncCrawler:
    """Asynchronous web crawler with memory optimization"""

    def __init__(
        self, config: CrawlConfig, optimizer: MemoryOptimizer, output_path: Path | None = None
    ):
        self.config = config
        self.optimizer = optimizer
        self.politeness = PolitenessPolicy(config.request_delay)
        self.session: aiohttp.ClientSession | None = None
        self.semaphore = asyncio.Semaphore(config.max_concurrent)
        self.visited_urls: set[str] = set()

        # Setup output
        if output_path:
            self.dataset_manager = ParquetDatasetManager(output_path, optimizer)
        else:
            self.dataset_manager = None

    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)

        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers={"User-Agent": self.config.user_agent},
            connector=aiohttp.TCPConnector(
                limit=self.config.max_concurrent, limit_per_host=5, enable_cleanup_closed=True
            ),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def crawl_single(self, url: str) -> CrawlResult | None:
        """Crawl single URL with retries and error handling"""
        async with self.semaphore:
            await self.politeness.wait_if_needed(url)

            for attempt in range(self.config.max_retries + 1):
                try:
                    start_time = time.time()

                    async with self.session.get(url) as response:
                        content = await response.text()
                        duration_ms = int((time.time() - start_time) * 1000)

                        # Check content size
                        if len(content.encode("utf-8")) > self.config.max_content_size:
                            content = content[: self.config.max_content_size // 2]
                            logger.warning(f"Truncated large content from {url}")

                        result = CrawlResult(
                            url=url,
                            status_code=response.status,
                            content=content,
                            headers=dict(response.headers),
                            timestamp=datetime.now(),
                            duration_ms=duration_ms,
                            redirect_chain=list(response.history) if response.history else None,
                            content_type=response.headers.get("content-type"),
                            content_length=response.headers.get("content-length"),
                        )

                        # Adjust politeness based on response
                        domain = urlparse(url).netloc
                        self.politeness.adjust_delay(domain, response.status)

                        return result

                except TimeoutError:
                    error_msg = f"Timeout after {self.config.timeout}s"
                    logger.warning(f"Timeout crawling {url} (attempt {attempt + 1})")

                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"Error crawling {url} (attempt {attempt + 1}): {e}")

                # Exponential backoff before retry
                if attempt < self.config.max_retries:
                    wait_time = self.config.backoff_factor**attempt
                    await asyncio.sleep(wait_time)

            # All retries failed
            return CrawlResult(
                url=url,
                status_code=0,
                content="",
                headers={},
                timestamp=datetime.now(),
                duration_ms=0,
                error=error_msg,
            )

    async def crawl_batch(
        self, urls: list[str], stream_to_parquet: bool = True
    ) -> AsyncIterator[CrawlResult]:
        """Crawl batch of URLs with streaming output"""
        tasks = []
        results_buffer = []

        # Create tasks for all URLs
        for url in urls:
            if url not in self.visited_urls:
                task = asyncio.create_task(self.crawl_single(url))
                tasks.append(task)
                self.visited_urls.add(url)

        # Process results as they complete
        for completed_task in asyncio.as_completed(tasks):
            try:
                result = await completed_task
                if result:
                    yield result

                    if stream_to_parquet and self.dataset_manager:
                        results_buffer.append(asdict(result))

                        # Write batch when buffer is full
                        batch_size = self.optimizer.get_optimal_batch_size(record_size_bytes=2048)
                        if len(results_buffer) >= batch_size:
                            await self._write_results_batch(results_buffer)
                            results_buffer = []

                            # Check memory pressure
                            if self.optimizer.check_memory_pressure()["pressure"]:
                                self.optimizer.force_gc()

            except Exception as e:
                logger.error(f"Task failed: {e}")

        # Write remaining results
        if results_buffer and stream_to_parquet and self.dataset_manager:
            await self._write_results_batch(results_buffer)

    async def _write_results_batch(self, results: list[dict[str, Any]]) -> None:
        """Write batch of results to Parquet asynchronously"""

        def write_batch():
            return self.dataset_manager.write_streaming_batch(
                iter(results), partition_cols=["timestamp"], compression="snappy"
            )

        # Run in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, write_batch)

    async def recursive_crawl(
        self,
        start_urls: list[str],
        max_depth: int = 2,
        max_pages: int = 1000,
        link_filter: Callable[[str], bool] | None = None,
    ) -> AsyncIterator[CrawlResult]:
        """Recursive crawling with depth and page limits"""
        crawled_count = 0
        urls_to_crawl = [(url, 0) for url in start_urls]  # (url, depth)
        discovered_urls = set(start_urls)

        while urls_to_crawl and crawled_count < max_pages:
            # Process current level
            current_batch = []
            next_urls = []

            for url, depth in urls_to_crawl:
                if depth <= max_depth:
                    current_batch.append(url)

                    if len(current_batch) >= self.config.max_concurrent:
                        break

            # Crawl current batch
            async for result in self.crawl_batch(current_batch):
                yield result
                crawled_count += 1

                # Extract links for next level if not at max depth
                if result.status_code == 200 and result.url in [
                    (url, depth) for url, depth in urls_to_crawl if depth < max_depth
                ]:
                    links = self._extract_links(result.content, result.url)

                    for link in links:
                        if link_filter is None or link_filter(link):
                            if link not in discovered_urls:
                                next_urls.append((link, depth + 1))
                                discovered_urls.add(link)

            # Update URLs to crawl
            urls_to_crawl = [
                item for item in urls_to_crawl if item[0] not in current_batch
            ] + next_urls

            # Memory pressure check
            if self.optimizer.check_memory_pressure()["critical"]:
                logger.warning("Critical memory pressure, pausing crawl")
                await asyncio.sleep(5)
                self.optimizer.force_gc()

    def _extract_links(self, content: str, base_url: str) -> list[str]:
        """Extract links from HTML content"""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(content, "html.parser")
            links = []

            for link in soup.find_all("a", href=True):
                href = link["href"]
                absolute_url = urljoin(base_url, href)

                # Basic filtering
                if absolute_url.startswith(("http://", "https://")):
                    links.append(absolute_url)

            return links

        except Exception as e:
            logger.warning(f"Failed to extract links from {base_url}: {e}")
            return []

    def get_crawl_stats(self) -> dict[str, Any]:
        """Get crawling statistics"""
        return {
            "visited_urls_count": len(self.visited_urls),
            "domain_delays": dict(self.politeness.domain_delays),
            "request_counts": dict(self.politeness.request_counts),
            "memory_stats": self.optimizer.check_memory_pressure(),
        }


# Utility functions for common crawling patterns
async def crawl_urls_to_parquet(
    urls: list[str],
    output_path: Path,
    config: CrawlConfig | None = None,
    max_memory_gb: float = 6.0,
) -> dict[str, Any]:
    """Convenience function to crawl URLs and save to Parquet"""
    if config is None:
        config = CrawlConfig()

    optimizer = MemoryOptimizer(max_memory_gb)

    async with AsyncCrawler(config, optimizer, output_path) as crawler:
        results_count = 0
        start_time = time.time()

        async for result in crawler.crawl_batch(urls, stream_to_parquet=True):
            results_count += 1

            if results_count % 100 == 0:
                logger.info(f"Crawled {results_count} URLs")

        duration = time.time() - start_time
        stats = crawler.get_crawl_stats()

        return {
            "results_count": results_count,
            "duration_seconds": duration,
            "urls_per_second": results_count / duration if duration > 0 else 0,
            "crawler_stats": stats,
        }


__all__ = [
    "AsyncCrawler",
    "CrawlConfig",
    "CrawlResult",
    "PolitenessPolicy",
    "crawl_urls_to_parquet",
]