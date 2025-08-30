"""
Enhanced Acquisition Worker - Fáze 2
Pokročilá akvizice dat s Tor/I2P anonymizací a Playwright scraping
"""

import asyncio
import logging
import os
import random
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import redis
import dramatiq
from dramatiq.brokers.redis import RedisBroker
import aiohttp
import aiohttp_socks
import polars as pl
from datetime import datetime, timezone
import hashlib
import re
import urllib.parse
from bs4 import BeautifulSoup

# Tor/Anonymization imports
import stem
from stem import Signal
from stem.control import Controller

# Playwright imports
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from fake_useragent import UserAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis broker setup
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.from_url(redis_url)
broker = RedisBroker(url=redis_url)
dramatiq.set_broker(broker)


class PersonaManager:
    """Správce browser person pro anti-detection"""

    def __init__(self):
        self.personas = self._generate_personas()
        self.domain_assignments = {}  # domain -> persona mapping

    def _generate_personas(self) -> List[Dict[str, Any]]:
        """Generuj pool realistických browser person"""
        ua = UserAgent()
        personas = []

        # Generuj různé kombinace browser fingerprints
        screen_resolutions = [
            {"width": 1920, "height": 1080},
            {"width": 1366, "height": 768},
            {"width": 1440, "height": 900},
            {"width": 1536, "height": 864},
            {"width": 1280, "height": 720},
        ]

        languages = ["en-US", "en-GB", "en-CA", "fr-FR", "de-DE", "es-ES"]
        timezones = ["America/New_York", "Europe/London", "Europe/Paris", "Europe/Berlin"]

        for i in range(int(os.getenv("PERSONA_POOL_SIZE", "10"))):
            resolution = random.choice(screen_resolutions)

            persona = {
                "id": f"persona_{i:03d}",
                "user_agent": ua.random,
                "viewport": resolution,
                "screen": resolution,
                "language": random.choice(languages),
                "timezone": random.choice(timezones),
                "platform": random.choice(["Win32", "MacIntel", "Linux x86_64"]),
                "webgl_vendor": random.choice(
                    ["Google Inc. (NVIDIA)", "Google Inc. (Intel)", "Google Inc. (AMD)"]
                ),
                "webgl_renderer": random.choice(
                    [
                        "ANGLE (NVIDIA GeForce GTX 1060 Direct3D11 vs_5_0 ps_5_0)",
                        "ANGLE (Intel(R) HD Graphics 620 Direct3D11 vs_5_0 ps_5_0)",
                        "ANGLE (AMD Radeon RX 580 Direct3D11 vs_5_0 ps_5_0)",
                    ]
                ),
            }
            personas.append(persona)

        logger.info(f"📋 Vygenerováno {len(personas)} browser person")
        return personas

    def get_persona_for_domain(self, domain: str) -> Dict[str, Any]:
        """Získej nebo přiřaď personu pro doménu"""
        if domain not in self.domain_assignments:
            # Přiřaď náhodnou personu pro novou doménu
            persona = random.choice(self.personas)
            self.domain_assignments[domain] = persona
            logger.info(f"🎭 Přiřazena persona {persona['id']} pro doménu {domain}")

        return self.domain_assignments[domain]


class TorManager:
    """Správce Tor připojení a circuit management"""

    def __init__(self):
        self.tor_proxy_url = os.getenv("TOR_PROXY_URL", "socks5://localhost:9050")
        self.control_password = os.getenv("TOR_CONTROL_PASSWORD", "deepresearch2025")
        self.controller = None

    async def init_controller(self):
        """Inicializuj Tor controller"""
        try:
            self.controller = Controller.from_port(port=9051)
            self.controller.authenticate(password=self.control_password)
            logger.info("✅ Tor controller připojen")
        except Exception as e:
            logger.warning(f"⚠️ Tor controller nedostupný: {e}")
            self.controller = None

    async def new_identity(self):
        """Požádej o novou Tor identitu"""
        if self.controller:
            try:
                self.controller.signal(Signal.NEWNYM)
                logger.info("🔄 Nová Tor identita požádána")
                await asyncio.sleep(5)  # Počkej na změnu okruhu
                return True
            except Exception as e:
                logger.error(f"❌ Chyba při změně Tor identity: {e}")
        return False

    def get_proxy_config(self) -> Dict[str, str]:
        """Získej proxy konfiguraci"""
        return {"http": self.tor_proxy_url, "https": self.tor_proxy_url}


class EnhancedAcquisitionWorker:
    """Pokročilý worker pro sběr dat s anonymizací a stealth scraping"""

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.data_dir = Path("/app/data")
        self.cache_dir = Path("/app/cache")
        self.persona_manager = PersonaManager()
        self.tor_manager = TorManager()

        # Zajisti že adresáře existují
        self.data_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)

        # Statistiky
        self.scraping_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "tor_circuits_changed": 0,
            "playwright_sessions": 0,
        }

    async def init_playwright(self):
        """Inicializuj Playwright browser"""
        if not self.playwright:
            self.playwright = await async_playwright().start()

            # Spusť browser s anti-detection nastavením
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--disable-web-security",
                    "--disable-features=VizDisplayCompositor",
                ],
            )
            logger.info("🎭 Playwright browser inicializován")

    async def create_stealth_context(
        self, persona: Dict[str, Any], use_proxy: bool = True
    ) -> BrowserContext:
        """Vytvoř stealth browser context s danou personou"""
        proxy_config = None
        if use_proxy:
            proxy_config = {
                "server": self.tor_manager.tor_proxy_url.replace("socks5://", "socks5://")
            }

        context = await self.browser.new_context(
            user_agent=persona["user_agent"],
            viewport=persona["viewport"],
            screen=persona["screen"],
            locale=persona["language"],
            timezone_id=persona["timezone"],
            proxy=proxy_config,
            # Anti-detection headers
            extra_http_headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": f"{persona['language']},en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            },
        )

        # Injektuj anti-detection skripty
        await context.add_init_script(
            """
            // Odstranit webdriver vlastnosti
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
            
            // Upravit permissions API
            Object.defineProperty(navigator, 'permissions', {
                get: () => ({
                    query: () => Promise.resolve({ state: 'granted' }),
                }),
            });
            
            // Upravit plugins
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });
            
            // Upravit languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['"""
            + persona["language"]
            + """'],
            });
        """
        )

        return context

    async def scrape_with_playwright(
        self, url: str, persona: Dict[str, Any], task_id: str
    ) -> Dict[str, Any]:
        """Scraping pomocí Playwright s anti-detection"""
        try:
            await self.init_playwright()

            context = await self.create_stealth_context(persona)
            page = await context.new_page()

            # Simulace lidského chování
            await page.set_extra_http_headers({"User-Agent": persona["user_agent"]})

            # Naviguj na stránku s timeoutem
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)

            # Náhodné čekání (simulace čtení)
            await asyncio.sleep(random.uniform(2, 5))

            # Simulace scrollování
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight/3)")
            await asyncio.sleep(random.uniform(1, 2))

            # Získej obsah
            content = await page.content()
            title = await page.title()

            # Metadata
            metadata = {
                "url": url,
                "title": title,
                "content_length": len(content),
                "persona_id": persona["id"],
                "method": "playwright",
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                "task_id": task_id,
            }

            await context.close()
            self.scraping_stats["playwright_sessions"] += 1

            return {"success": True, "content": content, "metadata": metadata}

        except Exception as e:
            logger.error(f"❌ Playwright scraping error for {url}: {e}")
            return {"success": False, "error": str(e), "url": url}

    async def scrape_with_aiohttp(
        self, url: str, persona: Dict[str, Any], task_id: str
    ) -> Dict[str, Any]:
        """Fallback scraping pomocí aiohttp s Tor proxy"""
        try:
            # Konfiguruj proxy pro aiohttp
            connector = aiohttp_socks.ProxyConnector.from_url(self.tor_manager.tor_proxy_url)

            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            async with aiohttp.ClientSession(
                connector=connector, timeout=timeout, headers={"User-Agent": persona["user_agent"]}
            ) as session:

                async with session.get(url) as response:
                    content = await response.text()

                    metadata = {
                        "url": url,
                        "status_code": response.status,
                        "content_type": response.headers.get("content-type", ""),
                        "content_length": len(content),
                        "persona_id": persona["id"],
                        "method": "aiohttp",
                        "scraped_at": datetime.now(timezone.utc).isoformat(),
                        "task_id": task_id,
                    }

                    return {"success": True, "content": content, "metadata": metadata}

        except Exception as e:
            logger.error(f"❌ aiohttp scraping error for {url}: {e}")
            return {"success": False, "error": str(e), "url": url}

    async def scrape_url_enhanced(
        self, url: str, task_id: str, force_tor: bool = False
    ) -> Dict[str, Any]:
        """Enhanced scraping s automatickou metodou selection"""
        try:
            # Extrahuj doménu pro persona assignment
            from urllib.parse import urlparse

            domain = urlparse(url).netloc

            # Získej personu pro tuto doménu
            persona = self.persona_manager.get_persona_for_domain(domain)

            # Detekce typu stránky
            needs_js = any(
                indicator in url.lower()
                for indicator in [
                    "single-page",
                    "spa",
                    "react",
                    "angular",
                    "vue",
                    "javascript",
                    "dynamic",
                    "ajax",
                ]
            )

            # Rozhodnutí o metodě scraping
            if needs_js or ".onion" in url or force_tor:
                logger.info(f"🎭 Používám Playwright pro {url}")
                result = await self.scrape_with_playwright(url, persona, task_id)
            else:
                logger.info(f"🌐 Používám aiohttp pro {url}")
                result = await self.scrape_with_aiohttp(url, persona, task_id)

            self.scraping_stats["total_requests"] += 1
            if result["success"]:
                self.scraping_stats["successful_requests"] += 1

            # Ulož data v Parquet formátu
            if result["success"]:
                raw_data = pl.DataFrame(
                    {
                        "url": [url],
                        "content": [result["content"]],
                        "metadata": [json.dumps(result["metadata"])],
                        "scraped_at": [datetime.now(timezone.utc)],
                    }
                )

                output_path = self.data_dir / f"raw_{task_id}.parquet"
                raw_data.write_parquet(output_path)

                # Pošli do processing queue
                process_scraped_data.send(str(output_path), task_id)

                result["output_path"] = str(output_path)
                result["stats"] = self.scraping_stats.copy()

            return result

        except Exception as e:
            logger.error(f"❌ Enhanced scraping error for {url}: {e}")
            return {"success": False, "url": url, "error": str(e), "task_id": task_id}

    async def extract_and_download_images(
        self, content: str, base_url: str, task_id: str
    ) -> Dict[str, Any]:
        """
        Extrahuj a stáhni obrázky ze stránky (FÁZE 3: Multi-Modality)

        Args:
            content: HTML obsah stránky
            base_url: Základní URL pro relativní odkazy
            task_id: ID úlohy

        Returns:
            Informace o nalezených a stažených obrázcích
        """
        try:
            logger.info(f"🖼️ Detekuji obrázky na stránce: {base_url}")

            soup = BeautifulSoup(content, "html.parser")

            # Najdi všechny img tagy
            img_tags = soup.find_all("img")

            # Extrakce obrázků z různých zdrojů
            image_urls = set()

            # Standard img src
            for img in img_tags:
                src = img.get("src")
                if src:
                    absolute_url = urllib.parse.urljoin(base_url, src)
                    image_urls.add(absolute_url)

                # Lazy loading obrázky
                data_src = img.get("data-src") or img.get("data-lazy-src")
                if data_src:
                    absolute_url = urllib.parse.urljoin(base_url, data_src)
                    image_urls.add(absolute_url)

            # CSS background images
            style_tags = soup.find_all(["div", "section", "header"], style=True)
            for tag in style_tags:
                style = tag.get("style", "")
                bg_matches = re.findall(r'background-image:\s*url\(["\']?([^"\']+)["\']?\)', style)
                for match in bg_matches:
                    absolute_url = urllib.parse.urljoin(base_url, match)
                    image_urls.add(absolute_url)

            # Inline SVG (malé obrázky)
            svg_tags = soup.find_all("svg")
            svg_count = len(svg_tags)

            # Filtruj platné image URLs
            valid_image_urls = []
            image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg", ".ico"}

            for url in image_urls:
                # Kontrola rozšíření
                parsed = urllib.parse.urlparse(url)
                path_lower = parsed.path.lower()

                if (
                    any(path_lower.endswith(ext) for ext in image_extensions)
                    or "image" in parsed.query.lower()
                    or url.startswith("data:image/")
                ):

                    # Filtrace podle velikosti URL (vyloučení malých ikon)
                    if not any(
                        skip in url.lower() for skip in ["favicon", "icon", "logo", "button"]
                    ):
                        valid_image_urls.append(url)

            logger.info(f"📊 Nalezeno {len(valid_image_urls)} potenciálních obrázků na {base_url}")

            # Stahování obrázků (omezeno na rozumný počet)
            max_images = int(os.getenv("MAX_IMAGES_PER_PAGE", "10"))
            download_urls = valid_image_urls[:max_images]

            downloaded_images = []
            images_dir = self.data_dir / "images" / task_id
            images_dir.mkdir(parents=True, exist_ok=True)

            for i, img_url in enumerate(download_urls):
                try:
                    image_result = await self._download_single_image(
                        img_url, images_dir, f"{task_id}_img_{i:03d}", base_url
                    )
                    if image_result["success"]:
                        downloaded_images.append(image_result)

                except Exception as e:
                    logger.warning(f"⚠️ Chyba při stahování {img_url}: {e}")
                    continue

            return {
                "success": True,
                "total_images_found": len(image_urls),
                "valid_images_found": len(valid_image_urls),
                "images_downloaded": len(downloaded_images),
                "svg_elements": svg_count,
                "downloaded_images": downloaded_images,
                "images_directory": str(images_dir),
            }

        except Exception as e:
            logger.error(f"❌ Chyba při extrakci obrázků: {e}")
            return {
                "success": False,
                "error": str(e),
                "total_images_found": 0,
                "images_downloaded": 0,
            }

    async def _download_single_image(
        self, img_url: str, images_dir: Path, filename_prefix: str, source_url: str
    ) -> Dict[str, Any]:
        """Stáhni jeden obrázek"""
        try:
            # Detekce rozšíření
            parsed = urllib.parse.urlparse(img_url)
            extension = Path(parsed.path).suffix.lower()
            if not extension or extension not in {
                ".jpg",
                ".jpeg",
                ".png",
                ".gif",
                ".webp",
                ".bmp",
                ".svg",
            }:
                extension = ".jpg"  # Default

            filename = f"{filename_prefix}{extension}"
            filepath = images_dir / filename

            # Pokud je to data URL, dekóduj přímo
            if img_url.startswith("data:image/"):
                try:
                    header, data = img_url.split(",", 1)
                    image_data = base64.b64decode(data)

                    with open(filepath, "wb") as f:
                        f.write(image_data)

                    return {
                        "success": True,
                        "url": img_url[:100] + "..." if len(img_url) > 100 else img_url,
                        "filepath": str(filepath),
                        "filename": filename,
                        "file_size": len(image_data),
                        "source_url": source_url,
                        "download_method": "data_url",
                    }
                except Exception as e:
                    logger.error(f"❌ Data URL decode error: {e}")
                    return {"success": False, "error": str(e)}

            # Stahování přes HTTP
            timeout = aiohttp.ClientTimeout(total=20, connect=5)

            # Použij proxy pokud je .onion
            if ".onion" in img_url or ".onion" in source_url:
                connector = aiohttp_socks.ProxyConnector.from_url(self.tor_manager.tor_proxy_url)
            else:
                connector = aiohttp.TCPConnector(limit=10)

            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Referer": source_url,
                    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
                }

                async with session.get(img_url, headers=headers) as response:
                    if response.status == 200:
                        content_type = response.headers.get("content-type", "")

                        # Ověř že je to opravdu obrázek
                        if not content_type.startswith("image/"):
                            logger.warning(f"⚠️ {img_url} není obrázek: {content_type}")
                            return {"success": False, "error": f"Not an image: {content_type}"}

                        # Kontrola velikosti (max 10MB)
                        content_length = response.headers.get("content-length")
                        if content_length and int(content_length) > 10 * 1024 * 1024:
                            return {"success": False, "error": "Image too large"}

                        image_data = await response.read()

                        # Uložení souboru
                        with open(filepath, "wb") as f:
                            f.write(image_data)

                        return {
                            "success": True,
                            "url": img_url,
                            "filepath": str(filepath),
                            "filename": filename,
                            "file_size": len(image_data),
                            "content_type": content_type,
                            "source_url": source_url,
                            "download_method": "http",
                        }
                    else:
                        return {"success": False, "error": f"HTTP {response.status}"}

        except Exception as e:
            logger.error(f"❌ Image download error for {img_url}: {e}")
            return {"success": False, "error": str(e)}

    async def cleanup(self):
        """Cleanup resources"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        if self.session and not self.session.closed:
            await self.session.close()


# Enhanced Dramatiq tasks
@dramatiq.actor(queue_name="acquisition")
def scrape_url_enhanced_task(
    url: str, task_id: str = None, force_tor: bool = False
) -> Dict[str, Any]:
    """Enhanced Dramatiq actor pro pokročilý scraping"""
    if task_id is None:
        task_id = f"enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    worker = EnhancedAcquisitionWorker()

    # Spusť async operaci
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Inicializuj Tor manager
        loop.run_until_complete(worker.tor_manager.init_controller())

        # Proveď scraping
        result = loop.run_until_complete(worker.scrape_url_enhanced(url, task_id, force_tor))

        # Cleanup
        loop.run_until_complete(worker.cleanup())

        return result
    finally:
        loop.close()


# Import z processing worker pro cross-service communication
@dramatiq.actor(queue_name="processing")
def process_scraped_data(file_path: str, task_id: str):
    """Forward declaration - implementováno v processing_worker"""
    pass


if __name__ == "__main__":
    logger.info("Starting Enhanced Acquisition Worker (Phase 2)...")

    # Spusť worker
    from dramatiq.cli import main
    import sys

    sys.argv = ["dramatiq", "workers.acquisition_worker", "--processes", "2", "--threads", "4"]
    main()
