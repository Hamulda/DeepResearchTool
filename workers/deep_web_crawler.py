"""Deep Web Crawler - Rekurzivní crawler pro .onion stránky
Autonomní crawler s rotací Tor identity a bezpečnostními opatřeními

Author: Senior Python/MLOps Agent
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import logging
import re
import time
from typing import Any, Set, List, Dict, Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from src.deep_web.network_manager import NetworkManager, NetworkConfig, NetworkType
from src.scraping.stealth_engine import StealthEngine, StealthConfig
from src.security.osint_sandbox import OSINTSandbox

logger = logging.getLogger(__name__)


@dataclass
class CrawlerConfig:
    """Konfigurace pro deep web crawler"""
    
    # Základní nastavení
    max_depth: int = 3
    max_pages_per_domain: int = 50
    max_total_pages: int = 200
    crawl_delay: float = 2.0
    random_delay_factor: float = 0.5
    
    # Tor specifické
    tor_identity_rotation_interval: int = 10  # stránek před rotací
    tor_circuit_timeout: int = 60  # sekund
    
    # Bezpečnostní limity
    max_page_size: int = 10 * 1024 * 1024  # 10MB
    allowed_content_types: List[str] = field(default_factory=lambda: [
        'text/html', 'text/plain', 'application/xhtml+xml'
    ])
    
    # Filtrování obsahu
    extract_links: bool = True
    extract_text: bool = True
    extract_metadata: bool = True
    save_screenshots: bool = False
    
    # Výstupní formáty
    output_format: str = "json"  # json, csv, markdown
    save_raw_html: bool = False


@dataclass
class CrawledPage:
    """Reprezentace procrawlované stránky"""
    
    url: str
    title: str
    content: str
    links: List[str]
    metadata: Dict[str, Any]
    crawl_time: datetime
    depth: int
    response_time: float
    content_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Konverze na slovník"""
        return {
            'url': self.url,
            'title': self.title,
            'content': self.content[:1000] + '...' if len(self.content) > 1000 else self.content,
            'links': self.links,
            'metadata': self.metadata,
            'crawl_time': self.crawl_time.isoformat(),
            'depth': self.depth,
            'response_time': self.response_time,
            'content_hash': self.content_hash
        }


class DeepWebCrawler:
    """Rekurzivní crawler pro .onion a .i2p stránky"""
    
    def __init__(self, config: CrawlerConfig = None):
        self.config = config or CrawlerConfig()
        
        # Stavové proměnné
        self.visited_urls: Set[str] = set()
        self.url_queue: List[tuple[str, int]] = []  # (url, depth)
        self.crawled_pages: List[CrawledPage] = []
        self.domain_page_counts: Dict[str, int] = {}
        
        # Síťové komponenty
        self.network_manager: Optional[NetworkManager] = None
        self.stealth_engine: Optional[StealthEngine] = None
        self.sandbox: Optional[OSINTSandbox] = None
        
        # Statistiky
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_pages_crawled': 0,
            'total_links_found': 0,
            'tor_identity_rotations': 0,
            'errors_encountered': 0,
            'domains_crawled': set()
        }
        
        # Tor rotace
        self.pages_since_rotation = 0
    
    async def initialize(self):
        """Inicializace crawleru"""
        logger.info("Inicializace Deep Web Crawleru...")
        
        # Konfigurace sítě pro Tor/I2P
        network_config = NetworkConfig(
            enable_tor=True,
            enable_i2p=True,
            enable_clearnet=False,  # Pouze anonymní sítě
            preferred_networks=[NetworkType.TOR, NetworkType.I2P],
            enable_failover=True,
            default_timeout=30
        )
        
        self.network_manager = NetworkManager(network_config)
        await self.network_manager.initialize()
        
        # Konfigurace stealth engine
        stealth_config = StealthConfig(
            user_agent_rotation=True,
            header_randomization=True,
            timing_randomization=True,
            min_delay=self.config.crawl_delay * (1 - self.config.random_delay_factor),
            max_delay=self.config.crawl_delay * (1 + self.config.random_delay_factor)
        )
        
        self.stealth_engine = StealthEngine(stealth_config)
        
        # Bezpečnostní sandbox
        self.sandbox = OSINTSandbox()
        
        logger.info("Deep Web Crawler inicializován")
    
    async def crawl(self, start_urls: List[str]) -> List[CrawledPage]:
        """Hlavní metoda pro crawling
        
        Args:
            start_urls: Seznam výchozích URL
            
        Returns:
            Seznam procrawlovaných stránek
        """
        self.stats['start_time'] = datetime.now()
        
        try:
            # Přidání výchozích URL do fronty
            for url in start_urls:
                if self._is_valid_url(url):
                    self.url_queue.append((url, 0))
                    logger.info(f"Přidána výchozí URL: {url}")
            
            # Hlavní crawling smyčka
            while (self.url_queue and 
                   len(self.crawled_pages) < self.config.max_total_pages):
                
                url, depth = self.url_queue.pop(0)
                
                # Kontrola limitů
                if depth > self.config.max_depth:
                    continue
                
                if url in self.visited_urls:
                    continue
                
                domain = self._extract_domain(url)
                if self.domain_page_counts.get(domain, 0) >= self.config.max_pages_per_domain:
                    logger.debug(f"Dosažen limit stránek pro doménu {domain}")
                    continue
                
                # Rotace Tor identity
                if (self.pages_since_rotation >= self.config.tor_identity_rotation_interval and
                    self.network_manager.tor_manager):
                    await self._rotate_tor_identity()
                
                # Crawling stránky
                crawled_page = await self._crawl_page(url, depth)
                
                if crawled_page:
                    self.crawled_pages.append(crawled_page)
                    self.visited_urls.add(url)
                    self.domain_page_counts[domain] = self.domain_page_counts.get(domain, 0) + 1
                    self.stats['total_pages_crawled'] += 1
                    self.stats['domains_crawled'].add(domain)
                    self.pages_since_rotation += 1
                    
                    # Přidání nalezených odkazů do fronty
                    if depth < self.config.max_depth:
                        for link in crawled_page.links:
                            if self._is_valid_url(link) and link not in self.visited_urls:
                                self.url_queue.append((link, depth + 1))
                                self.stats['total_links_found'] += 1
                
                # Bezpečnostní pauza
                await asyncio.sleep(
                    self.config.crawl_delay + 
                    (self.config.random_delay_factor * self.config.crawl_delay * (2 * asyncio.get_event_loop().time() % 1 - 1))
                )
        
        except Exception as e:
            logger.error(f"Chyba při crawlingu: {e}")
            self.stats['errors_encountered'] += 1
        
        finally:
            self.stats['end_time'] = datetime.now()
            await self._cleanup()
        
        logger.info(f"Crawling dokončen. Procrawlováno {len(self.crawled_pages)} stránek")
        return self.crawled_pages
    
    async def _crawl_page(self, url: str, depth: int) -> Optional[CrawledPage]:
        """Crawling jednotlivé stránky
        
        Args:
            url: URL stránky
            depth: Hloubka v crawling stromu
            
        Returns:
            CrawledPage objekt nebo None při chybě
        """
        start_time = time.time()
        
        try:
            logger.debug(f"Crawling stránky: {url} (hloubka: {depth})")
            
            # Stažení stránky pomocí stealth engine
            result = await self.stealth_engine.scrape_url(url, use_browser=True)
            
            if not result['success']:
                logger.warning(f"Neúspěšné stažení {url}: {result.get('error', 'Neznámá chyba')}")
                return None
            
            # Kontrola velikosti obsahu
            if len(result['content']) > self.config.max_page_size:
                logger.warning(f"Stránka {url} je příliš velká ({len(result['content'])} bytů)")
                return None
            
            # Parsování obsahu
            soup = BeautifulSoup(result['content'], 'html.parser')
            
            # Extrakce základních dat
            title = self._extract_title(soup)
            text_content = self._extract_text(soup) if self.config.extract_text else ""
            links = self._extract_links(soup, url) if self.config.extract_links else []
            metadata = self._extract_metadata(soup, result) if self.config.extract_metadata else {}
            
            # Vytvoření hash obsahu
            content_hash = hashlib.sha256(result['content'].encode()).hexdigest()[:16]
            
            response_time = time.time() - start_time
            
            crawled_page = CrawledPage(
                url=url,
                title=title,
                content=text_content,
                links=links,
                metadata=metadata,
                crawl_time=datetime.now(),
                depth=depth,
                response_time=response_time,
                content_hash=content_hash
            )
            
            logger.info(f"Úspěšně procrawlována stránka: {url} ({len(links)} odkazů)")
            return crawled_page
            
        except Exception as e:
            logger.error(f"Chyba při crawlingu stránky {url}: {e}")
            self.stats['errors_encountered'] += 1
            return None
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extrakce title ze stránky"""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        # Fallback na h1
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()
        
        return "Bez nadpisu"
    
    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extrakce textového obsahu"""
        # Odstranění scriptů a stylů
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extrakce textu
        text = soup.get_text()
        
        # Vyčištění
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text[:5000]  # Limit na 5000 znaků
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extrakce odkazů ze stránky"""
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href'].strip()
            
            if not href or href.startswith('#') or href.startswith('javascript:'):
                continue
            
            # Absolutní URL
            absolute_url = urljoin(base_url, href)
            
            if self._is_valid_url(absolute_url):
                links.append(absolute_url)
        
        return list(set(links))  # Odstranění duplicit
    
    def _extract_metadata(self, soup: BeautifulSoup, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extrakce metadat ze stránky"""
        metadata = {}
        
        # Meta tagy
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            
            if name and content:
                metadata[name] = content
        
        # Další metadata
        metadata.update({
            'final_url': result.get('final_url', ''),
            'content_length': len(result.get('content', '')),
            'anti_bot_detected': result.get('anti_bot_detected', False),
            'status_code': result.get('status_code', 0)
        })
        
        return metadata
    
    def _is_valid_url(self, url: str) -> bool:
        """Kontrola platnosti URL"""
        try:
            parsed = urlparse(url)
            
            # Kontrola schématu
            if parsed.scheme not in ['http', 'https']:
                return False
            
            # Kontrola domény pro anonymní sítě
            if not (parsed.netloc.endswith('.onion') or 
                   parsed.netloc.endswith('.i2p') or
                   parsed.netloc.endswith('.garlic')):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _extract_domain(self, url: str) -> str:
        """Extrakce domény z URL"""
        try:
            return urlparse(url).netloc
        except Exception:
            return "unknown"
    
    async def _rotate_tor_identity(self):
        """Rotace Tor identity"""
        try:
            if self.network_manager and self.network_manager.tor_manager:
                await self.network_manager.tor_manager.rotate_identity()
                self.stats['tor_identity_rotations'] += 1
                self.pages_since_rotation = 0
                logger.info("Tor identita rotována")
                
                # Čekání na nový okruh
                await asyncio.sleep(5)
                
        except Exception as e:
            logger.error(f"Chyba při rotaci Tor identity: {e}")
    
    async def _cleanup(self):
        """Úklid zdrojů"""
        try:
            if self.stealth_engine:
                await self.stealth_engine.cleanup()
            
            if self.network_manager:
                await self.network_manager.cleanup()
                
            if self.sandbox:
                await self.sandbox.cleanup()
                
        except Exception as e:
            logger.error(f"Chyba při úklidu crawleru: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Získání statistik crawlingu"""
        runtime = None
        if self.stats['start_time'] and self.stats['end_time']:
            runtime = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        return {
            **self.stats,
            'domains_crawled': list(self.stats['domains_crawled']),
            'runtime_seconds': runtime,
            'pages_per_second': (
                self.stats['total_pages_crawled'] / runtime 
                if runtime and runtime > 0 else 0
            ),
            'current_queue_size': len(self.url_queue),
            'visited_urls_count': len(self.visited_urls)
        }
    
    def export_results(self, output_path: str = None) -> str:
        """Export výsledků crawlingu"""
        import json
        from pathlib import Path
        
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"deep_web_crawl_{timestamp}.json"
        
        export_data = {
            'crawl_config': {
                'max_depth': self.config.max_depth,
                'max_pages_per_domain': self.config.max_pages_per_domain,
                'max_total_pages': self.config.max_total_pages
            },
            'statistics': self.get_statistics(),
            'crawled_pages': [page.to_dict() for page in self.crawled_pages]
        }
        
        Path(output_path).write_text(
            json.dumps(export_data, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )
        
        logger.info(f"Výsledky exportovány do: {output_path}")
        return output_path


# Utility funkce
async def crawl_onion_sites(start_urls: List[str], max_depth: int = 2) -> List[CrawledPage]:
    """Jednoduché rozhraní pro crawling .onion stránek"""
    config = CrawlerConfig(
        max_depth=max_depth,
        max_pages_per_domain=20,
        max_total_pages=100,
        tor_identity_rotation_interval=5
    )
    
    crawler = DeepWebCrawler(config)
    
    try:
        await crawler.initialize()
        return await crawler.crawl(start_urls)
    finally:
        await crawler._cleanup()


def create_crawler_config(
    max_depth: int = 3,
    max_pages: int = 200,
    rotation_interval: int = 10
) -> CrawlerConfig:
    """Vytvoření konfigurace crawleru"""
    return CrawlerConfig(
        max_depth=max_depth,
        max_pages_per_domain=50,
        max_total_pages=max_pages,
        tor_identity_rotation_interval=rotation_interval,
        crawl_delay=2.0,
        random_delay_factor=0.5
    )


__all__ = [
    'DeepWebCrawler',
    'CrawlerConfig', 
    'CrawledPage',
    'crawl_onion_sites',
    'create_crawler_config'
]