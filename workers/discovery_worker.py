"""
Discovery Worker - Autonomní discovery engine (Fáze 4)
Automaticky objevuje nové zdroje dat a přidává je do scraping fronty
"""

import asyncio
import logging
import os
import json
import re
import random
from typing import Dict, Any, List, Set, Optional
from pathlib import Path
import redis
import dramatiq
from dramatiq.brokers.redis import RedisBroker
import aiohttp
import aiohttp_socks
from datetime import datetime, timezone, timedelta
from urllib.parse import urljoin, urlparse
import schedule
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis broker setup
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.from_url(redis_url)
broker = RedisBroker(url=redis_url)
dramatiq.set_broker(broker)


class DiscoveryEngine:
    """Autonomní engine pro objevování nových zdrojů"""

    def __init__(self):
        self.tor_proxy_url = os.getenv("TOR_PROXY_URL", "socks5://localhost:9050")
        self.discovery_interval = int(os.getenv("DISCOVERY_INTERVAL", "3600"))  # 1 hour
        self.data_dir = Path("/app/data")
        self.configs_dir = Path("/app/configs")

        # Zajisti že adresáře existují
        self.data_dir.mkdir(exist_ok=True)
        self.configs_dir.mkdir(exist_ok=True)

        # Načti whitelist a konfiguraci
        self.whitelist = self._load_whitelist()
        self.discovered_urls = set()
        self.discovery_stats = {
            "total_discovered": 0,
            "successful_discoveries": 0,
            "last_discovery": None,
            "discovery_sources": {},
        }

        # Discovery sources
        self.discovery_sources = {
            "ahmia": "https://ahmia.fi/search/?q={query}",
            "torch": "http://xmh57jrknzkhv6y3ls3ubitzfqnkrwxhopf5aygthi7d6rplyvk3noyd.onion/cgi-bin/omega/omega?P={query}",
            "hidden_wiki": "http://zqktlwiuavvvqqt4ybvgvi7tyo4hjl5xgfuvpdf6otjiycgwqbym2qad.onion/wiki/index.php/Main_Page",
            "duckduckgo": "https://duckduckgo.com/html/?q={query}+site:*.onion",
        }

        # Discovery patterns
        self.url_patterns = [
            r"https?://[a-z2-7]{16,56}\.onion[/\w\-._~:/?#[\]@!$&\'()*+,;=]*",
            r"http://[a-z2-7]{16,56}\.onion[/\w\-._~:/?#[\]@!$&\'()*+,;=]*",
            r"[a-z2-7]{16,56}\.onion",
            r"https?://[a-z0-9\-]+\.i2p[/\w\-._~:/?#[\]@!$&\'()*+,;=]*",
        ]

        self.session = None

    def _load_whitelist(self) -> Dict[str, Any]:
        """Načti whitelist a compliance pravidla"""
        try:
            whitelist_path = self.configs_dir / "tor_legal_whitelist.json"
            if whitelist_path.exists():
                with open(whitelist_path, "r") as f:
                    return json.load(f)
            else:
                logger.warning("⚠️ Whitelist soubor nenalezen, používám výchozí")
                return {
                    "legal_domains": ["httpbin.org", "example.com"],
                    "blocked_domains": [],
                    "compliance_notes": {"purpose": "Research and educational purposes only"},
                }
        except Exception as e:
            logger.error(f"❌ Chyba při načítání whitelist: {e}")
            return {"legal_domains": [], "blocked_domains": []}

    async def _get_session(self) -> aiohttp.ClientSession:
        """Získej HTTP session s Tor proxy"""
        if self.session is None or self.session.closed:
            connector = aiohttp_socks.ProxyConnector.from_url(self.tor_proxy_url)
            timeout = aiohttp.ClientTimeout(total=30, connect=10)

            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
            )
        return self.session

    def _is_url_allowed(self, url: str) -> bool:
        """Zkontroluj zda je URL povolena podle whitelist"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Zkontroluj blocked domains
            blocked_domains = self.whitelist.get("blocked_domains", [])
            if any(blocked in domain for blocked in blocked_domains):
                return False

            # Pro .onion adresy buď více permisivní (research účely)
            if domain.endswith(".onion"):
                # Základní bezpečnostní kontroly
                if len(domain.split(".")[0]) < 16:  # Minimální délka onion adresy
                    return False
                return True

            # Pro clearnet zkontroluj whitelist
            legal_domains = self.whitelist.get("legal_domains", [])
            if legal_domains and not any(legal in domain for legal in legal_domains):
                return False

            return True

        except Exception as e:
            logger.error(f"❌ URL validation error: {e}")
            return False

    def _extract_urls_from_text(self, text: str) -> Set[str]:
        """Extrahuj URL z textu pomocí regex patterns"""
        urls = set()

        for pattern in self.url_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Normalizuj URL
                if not match.startswith(("http://", "https://")):
                    if match.endswith(".onion"):
                        match = f"http://{match}"
                    elif match.endswith(".i2p"):
                        match = f"http://{match}"

                if self._is_url_allowed(match):
                    urls.add(match)

        return urls

    async def discover_from_source(self, source_name: str, query: str = "research") -> Set[str]:
        """Objevuj URL z konkrétního zdroje"""
        try:
            if source_name not in self.discovery_sources:
                logger.warning(f"⚠️ Neznámý discovery source: {source_name}")
                return set()

            source_url = self.discovery_sources[source_name].format(query=query)
            logger.info(f"🔍 Discovering from {source_name}: {query}")

            session = await self._get_session()
            async with session.get(source_url) as response:
                if response.status == 200:
                    content = await response.text()
                    discovered_urls = self._extract_urls_from_text(content)

                    logger.info(f"✅ Discovered {len(discovered_urls)} URLs from {source_name}")

                    # Aktualizuj statistiky
                    self.discovery_stats["discovery_sources"][source_name] = self.discovery_stats[
                        "discovery_sources"
                    ].get(source_name, 0) + len(discovered_urls)

                    return discovered_urls
                else:
                    logger.warning(f"⚠️ Discovery source {source_name} returned {response.status}")
                    return set()

        except Exception as e:
            logger.error(f"❌ Discovery error from {source_name}: {e}")
            return set()

    async def discover_from_known_sites(self) -> Set[str]:
        """Objevuj nové URL z již známých stránek"""
        try:
            discovered_urls = set()

            # Načti už známé URL z Redis cache
            known_urls_key = "discovery:known_urls"
            known_urls_data = redis_client.get(known_urls_key)

            if known_urls_data:
                known_urls = json.loads(known_urls_data)
            else:
                # Fallback na základní seed URL
                known_urls = [
                    "http://3g2upl4pq6kufc4m.onion",  # DuckDuckGo onion
                    "https://ahmia.fi",  # Ahmia search
                ]

            # Prohledej známé stránky pro nové odkazy
            session = await self._get_session()

            for url in known_urls[:10]:  # Omez na 10 URL pro performance
                try:
                    logger.info(f"🕷️ Crawling known site: {url}")

                    async with session.get(url) as response:
                        if response.status == 200:
                            content = await response.text()
                            new_urls = self._extract_urls_from_text(content)
                            discovered_urls.update(new_urls)

                            await asyncio.sleep(random.uniform(2, 5))  # Zdvořilé čekání

                except Exception as crawl_error:
                    logger.warning(f"⚠️ Crawling error for {url}: {crawl_error}")
                    continue

            logger.info(f"🕸️ Discovered {len(discovered_urls)} URLs from known sites")
            return discovered_urls

        except Exception as e:
            logger.error(f"❌ Known sites discovery error: {e}")
            return set()

    async def run_discovery_cycle(self) -> Dict[str, Any]:
        """Spusť kompletní discovery cyklus"""
        try:
            cycle_start = datetime.now(timezone.utc)
            logger.info("🚀 Spouštím discovery cyklus")

            all_discovered = set()

            # 1. Discovery z search engines
            search_queries = ["research", "academic", "information", "data", "science"]
            for query in search_queries[:2]:  # Omez počet dotazů
                for source in ["ahmia", "duckduckgo"]:  # Bezpečné zdroje
                    urls = await self.discover_from_source(source, query)
                    all_discovered.update(urls)
                    await asyncio.sleep(random.uniform(5, 10))  # Rate limiting

            # 2. Discovery z známých stránek
            known_site_urls = await self.discover_from_known_sites()
            all_discovered.update(known_site_urls)

            # 3. Filtruj duplicity a už zpracované URL
            new_urls = all_discovered - self.discovered_urls
            self.discovered_urls.update(new_urls)

            # 4. Přidej nové URL do scraping fronty
            queued_count = 0
            for url in new_urls:
                try:
                    # Import acquisition worker task
                    from workers.acquisition_worker import scrape_url_enhanced_task

                    task_id = f"discovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{queued_count}"
                    scrape_url_enhanced_task.send(url, task_id, force_tor=True)
                    queued_count += 1

                    if queued_count >= 50:  # Omez počet nových úloh
                        break

                except Exception as queue_error:
                    logger.error(f"❌ Queue error for {url}: {queue_error}")
                    continue

            # 5. Aktualizuj statistiky
            cycle_duration = (datetime.now(timezone.utc) - cycle_start).total_seconds()

            self.discovery_stats.update(
                {
                    "total_discovered": len(all_discovered),
                    "successful_discoveries": len(new_urls),
                    "last_discovery": cycle_start.isoformat(),
                    "cycle_duration": cycle_duration,
                    "queued_for_scraping": queued_count,
                }
            )

            # 6. Ulož discovered URLs do Redis
            redis_client.setex(
                "discovery:known_urls", 86400, json.dumps(list(self.discovered_urls))  # 24 hours
            )

            # 7. Ulož statistiky
            redis_client.setex("discovery:stats", 3600, json.dumps(self.discovery_stats))  # 1 hour

            logger.info(
                f"✅ Discovery cyklus dokončen: {len(new_urls)} nových URL, {queued_count} přidáno do fronty"
            )

            return {
                "success": True,
                "discovered_total": len(all_discovered),
                "new_urls": len(new_urls),
                "queued": queued_count,
                "duration": cycle_duration,
                "cycle_time": cycle_start.isoformat(),
            }

        except Exception as e:
            logger.error(f"❌ Discovery cycle error: {e}")
            return {
                "success": False,
                "error": str(e),
                "cycle_time": datetime.now(timezone.utc).isoformat(),
            }

        finally:
            # Cleanup session
            if self.session and not self.session.closed:
                await self.session.close()
                self.session = None

    async def cleanup(self):
        """Cleanup resources"""
        if self.session and not self.session.closed:
            await self.session.close()


# Dramatiq tasks
@dramatiq.actor(queue_name="discovery")
def run_discovery_cycle() -> Dict[str, Any]:
    """Dramatiq actor pro discovery cyklus"""
    engine = DiscoveryEngine()

    # Spusť async operaci
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        result = loop.run_until_complete(engine.run_discovery_cycle())
        loop.run_until_complete(engine.cleanup())
        return result
    finally:
        loop.close()


@dramatiq.actor(queue_name="discovery")
def schedule_discovery():
    """Plánovaný discovery task"""
    logger.info("⏰ Scheduled discovery task triggered")
    return run_discovery_cycle.send()


# Scheduler pro automatické discovery
def setup_discovery_scheduler():
    """Nastav automatické discovery"""
    interval_hours = int(os.getenv("DISCOVERY_INTERVAL", "3600")) // 3600

    schedule.every(interval_hours).hours.do(lambda: run_discovery_cycle.send())
    logger.info(f"📅 Discovery naplánováno každých {interval_hours} hodin")

    # Spusť první discovery po 5 minutách
    schedule.every(5).minutes.do(lambda: run_discovery_cycle.send()).tag("initial")


if __name__ == "__main__":
    logger.info("Starting Discovery Worker (Phase 4)...")

    # Nastav scheduler
    setup_discovery_scheduler()

    # Spusť první discovery cycle okamžitě (pro testování)
    run_discovery_cycle.send()

    # Spusť scheduler
    while True:
        schedule.run_pending()
        time.sleep(60)  # Zkontroluj každou minutu
