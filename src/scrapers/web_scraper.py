"""🕸️ Robustní web scraper s pokročilým error handling
Poskytuje jednotné rozhraní pro scraping s retry mechanismy a circuit breaker
"""

import asyncio
from datetime import datetime
import logging
from typing import Any, Dict, List, Tuple

import aiohttp
from bs4 import BeautifulSoup

from .base_scraper import BaseScraper
from ..core.error_handling import (
    scraping_retry,
    safe_aiohttp_get,
    ErrorAggregator,
    respect_rate_limit,
    timeout_after
)


class WebScraper(BaseScraper):
    """🌐 Robustní web scraper s pokročilým error handling

    Poskytuje základní funkcionalitu pro scraping webových stránek
    s podporou pro retry mechanismy, circuit breaker a rate limiting.
    """

    def __init__(self, delay: float = 1.0, timeout: int = 30, max_retries: int = 3):
        super().__init__()
        self.delay = delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = None
        self.scraped_urls = set()
        self.error_aggregator = ErrorAggregator()

    @timeout_after(120)  # 2 minute timeout for individual page scraping
    async def scrape_page(self, url: str, **kwargs) -> str:
        """Scrape jedné webové stránky s robustním error handling

        Args:
            url: URL stránky k scraping
            **kwargs: Dodatečné parametry

        Returns:
            str: Obsah stránky jako text
        """
        try:
            await self._ensure_session()

            # Rate limiting
            await asyncio.sleep(self.delay)

            # Použití bezpečného aiohttp GET s retry
            response = await safe_aiohttp_get(self.session, url, timeout=self.timeout, **kwargs)

            # Kontrola rate limiting
            await respect_rate_limit(response)

            content = await response.text()
            self.scraped_urls.add(url)

            # Základní čištění obsahu
            cleaned_content = self._clean_content(content)

            self.error_aggregator.add_success()
            logging.info(f"✅ Úspěšně scrapen: {url}")
            return cleaned_content

        except Exception as e:
            self.error_aggregator.add_error(e, f"scraping {url}")
            logging.error(f"❌ Chyba při scrapingu {url}: {e}")
            return ""

    async def scrape_multiple(self, urls: List[str], max_concurrent: int = 3) -> Dict[str, str]:
        """Scrape více URL současně s robustním error handling

        Args:
            urls: Seznam URL k scraping
            max_concurrent: Max počet současných požadavků

        Returns:
            Dict[str, str]: Mapování URL -> obsah
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def scrape_with_semaphore(url: str) -> Tuple[str, str]:
            async with semaphore:
                content = await self.scrape_page(url)
                return url, content

        # Spuštění všech úkolů
        tasks = [scrape_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Zpracování výsledků
        scraped_data = {}
        for result in results:
            if isinstance(result, tuple) and len(result) == 2:
                url, content = result
                if content:  # Pouze úspěšné scraping
                    scraped_data[url] = content
            elif isinstance(result, Exception):
                self.error_aggregator.add_error(result, "batch scraping")

        # Logování shrnutí
        self.error_aggregator.log_summary()

        return scraped_data

    @scraping_retry
    async def _ensure_session(self):
        """Zajistí, že HTTP session existuje s retry logikou"""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=3,
                enable_cleanup_closed=True,
                ttl_dns_cache=300,
                use_dns_cache=True
            )
            timeout = aiohttp.ClientTimeout(total=self.timeout)

            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                },
            )

    def _clean_content(self, content: str) -> str:
        """Robustní čištění HTML obsahu s error handling

        Args:
            content: Surový HTML obsahu

        Returns:
            str: Vyčištěný textový obsah
        """
        try:
            soup = BeautifulSoup(content, "html.parser")

            # Odstranění scriptů, stylů a dalších nežádoucích elementů
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()

            # Extrakce textu
            text = soup.get_text()

            # Pokročilé čištění whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)

            return text

        except Exception as e:
            logging.warning(f"Error cleaning content with BeautifulSoup: {e}")
            # Fallback bez BeautifulSoup
            import re
            clean = re.compile("<.*?>")
            return re.sub(clean, "", content)

    @scraping_retry
    async def get_page_metadata(self, url: str) -> Dict[str, Any]:
        """Získá metadata stránky s robustním error handling

        Args:
            url: URL stránky

        Returns:
            Dict s metadaty stránky
        """
        try:
            await self._ensure_session()

            response = await safe_aiohttp_get(self.session, url, timeout=self.timeout)

            metadata = {
                "url": url,
                "status_code": response.status,
                "content_type": response.headers.get("Content-Type", ""),
                "content_length": response.headers.get("Content-Length", ""),
                "last_modified": response.headers.get("Last-Modified", ""),
                "scraped_at": datetime.now().isoformat(),
                "success": response.status == 200,
            }

            if response.status == 200:
                # Extrakce title a description
                content = await response.text()

                try:
                    soup = BeautifulSoup(content, "html.parser")

                    # Title
                    title_tag = soup.find("title")
                    if title_tag:
                        metadata["title"] = title_tag.get_text().strip()

                    # Description
                    desc_tag = soup.find("meta", attrs={"name": "description"})
                    if desc_tag:
                        metadata["description"] = desc_tag.get("content", "").strip()

                    # Keywords
                    keywords_tag = soup.find("meta", attrs={"name": "keywords"})
                    if keywords_tag:
                        metadata["keywords"] = keywords_tag.get("content", "").strip()

                except Exception as e:
                    logging.warning(f"Error extracting metadata from {url}: {e}")

            return metadata

        except Exception as e:
            self.error_aggregator.add_error(e, f"metadata extraction for {url}")
            return {
                "url": url,
                "error": str(e),
                "scraped_at": datetime.now().isoformat(),
                "success": False,
            }

    async def close(self):
        """Bezpečně uzavře HTTP session"""
        try:
            if self.session and not self.session.closed:
                await self.session.close()
                # Krátké čekání na ukončení spojení
                await asyncio.sleep(0.1)
        except Exception as e:
            logging.warning(f"Error closing session: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Vrátí statistiky scrapingu s error reporting"""
        error_summary = self.error_aggregator.get_summary()

        return {
            "total_scraped": len(self.scraped_urls),
            "scraped_urls": list(self.scraped_urls)[-10:],  # Posledních 10
            "delay": self.delay,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "success_rate": error_summary["success_rate"],
            "total_operations": error_summary["total_operations"],
            "failed_operations": error_summary["failed_operations"],
            "recent_errors": error_summary["errors"][-5:] if error_summary["errors"] else []
        }

    def reset_stats(self):
        """Reset statistik a error aggregator"""
        self.scraped_urls.clear()
        self.error_aggregator = ErrorAggregator()
