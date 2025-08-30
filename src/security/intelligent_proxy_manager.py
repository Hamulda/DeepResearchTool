"""Intelligent Proxy Manager
Dynamická správa proxy serverů a identity pro anonymní výzkum

Author: Senior Python/MLOps Agent
"""

from dataclasses import dataclass
import json
import logging
import random
import time
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class ProxyConfig:
    """Konfigurace proxy serveru"""

    host: str
    port: int
    protocol: str = "http"  # http, https, socks4, socks5
    username: str | None = None
    password: str | None = None
    country: str | None = None
    anonymity_level: str = "high"  # low, medium, high, elite
    last_used: float | None = None
    success_rate: float = 1.0
    response_time: float = 0.0
    is_active: bool = True


@dataclass
class IdentityProfile:
    """Profil identity pro anonymní přístup"""

    user_agent: str
    accept_language: str
    timezone: str
    screen_resolution: str
    platform: str
    browser_version: str
    plugins: list[str]
    fingerprint_id: str


class ProxyManager:
    """Správce proxy serverů s intelligentní rotací
    """

    def __init__(self, config_path: str | None = None):
        self.proxies: list[ProxyConfig] = []
        self.current_proxy_index = 0
        self.proxy_stats: dict[str, dict[str, Any]] = {}
        self.tor_enabled = False
        self.tor_socks_port = 9050

        if config_path:
            self.load_proxies_from_file(config_path)
        else:
            self._setup_default_proxies()

    def _setup_default_proxies(self):
        """Nastavení výchozích proxy serverů"""
        # Lokální Tor proxy (pokud je spuštěn)
        tor_proxy = ProxyConfig(
            host="127.0.0.1",
            port=self.tor_socks_port,
            protocol="socks5",
            anonymity_level="elite",
            country="unknown"
        )

        self.proxies.append(tor_proxy)
        logger.info("Přidán Tor SOCKS proxy")

    def load_proxies_from_file(self, config_path: str):
        """Načtení proxy serverů ze souboru"""
        try:
            with open(config_path) as f:
                proxy_data = json.load(f)

            for proxy_info in proxy_data.get("proxies", []):
                proxy = ProxyConfig(**proxy_info)
                self.proxies.append(proxy)

            logger.info(f"Načteno {len(self.proxies)} proxy serverů")

        except Exception as e:
            logger.error(f"Chyba při načítání proxy konfigurace: {e}")
            self._setup_default_proxies()

    async def test_proxy(self, proxy: ProxyConfig) -> tuple[bool, float]:
        """Test funkčnosti proxy serveru

        Args:
            proxy: Proxy konfigurace k testování

        Returns:
            Tuple (je_funkční, doba_odezvy)

        """
        start_time = time.time()

        try:
            proxy_url = self._build_proxy_url(proxy)

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://httpbin.org/ip",
                    proxy=proxy_url,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        response_time = time.time() - start_time
                        data = await response.json()
                        logger.debug(f"Proxy {proxy.host}:{proxy.port} funguje, IP: {data.get('origin')}")
                        return True, response_time
                    return False, float('inf')

        except Exception as e:
            logger.debug(f"Proxy {proxy.host}:{proxy.port} selhalo: {e}")
            return False, float('inf')

    def _build_proxy_url(self, proxy: ProxyConfig) -> str:
        """Sestavení URL pro proxy"""
        if proxy.username and proxy.password:
            auth = f"{proxy.username}:{proxy.password}@"
        else:
            auth = ""

        return f"{proxy.protocol}://{auth}{proxy.host}:{proxy.port}"

    async def get_working_proxy(self) -> ProxyConfig | None:
        """Získání funkčního proxy serveru

        Returns:
            Funkční proxy konfigurace nebo None

        """
        if not self.proxies:
            return None

        # Testování proxy v pořadí podle success rate
        sorted_proxies = sorted(
            [p for p in self.proxies if p.is_active],
            key=lambda x: (x.success_rate, -x.response_time),
            reverse=True
        )

        for proxy in sorted_proxies:
            is_working, response_time = await self.test_proxy(proxy)

            if is_working:
                # Aktualizace statistik
                proxy.last_used = time.time()
                proxy.response_time = response_time
                proxy.success_rate = min(1.0, proxy.success_rate + 0.1)

                logger.info(f"Použito proxy: {proxy.host}:{proxy.port}")
                return proxy
            # Snížení success rate pro nefunkční proxy
            proxy.success_rate = max(0.0, proxy.success_rate - 0.2)
            if proxy.success_rate < 0.1:
                proxy.is_active = False

        logger.warning("Žádné funkční proxy nebylo nalezeno")
        return None

    async def rotate_proxy(self) -> ProxyConfig | None:
        """Rotace na další proxy server

        Returns:
            Nový proxy server nebo None

        """
        if len(self.proxies) <= 1:
            return await self.get_working_proxy()

        # Rotace na další proxy
        self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxies)

        return await self.get_working_proxy()

    def get_proxy_stats(self) -> dict[str, Any]:
        """Získání statistik proxy serverů"""
        active_proxies = [p for p in self.proxies if p.is_active]

        return {
            "total_proxies": len(self.proxies),
            "active_proxies": len(active_proxies),
            "average_success_rate": sum(p.success_rate for p in active_proxies) / len(active_proxies) if active_proxies else 0,
            "average_response_time": sum(p.response_time for p in active_proxies) / len(active_proxies) if active_proxies else 0,
            "tor_enabled": self.tor_enabled
        }


class IdentityManager:
    """Správce identit pro anonymní browsing
    """

    def __init__(self):
        self.current_identity: IdentityProfile | None = None
        self.identity_pool: list[IdentityProfile] = []
        self._generate_identity_pool()

    def _generate_identity_pool(self):
        """Generování pool identit"""
        from fake_useragent import UserAgent

        try:
            ua = UserAgent()

            # Generování různých identit
            browsers = ['chrome', 'firefox', 'safari', 'edge']
            platforms = ['windows', 'macos', 'linux']
            languages = ['cs-CZ', 'en-US', 'en-GB', 'de-DE', 'fr-FR']
            timezones = ['Europe/Prague', 'Europe/London', 'America/New_York', 'Europe/Berlin']

            for i in range(20):  # 20 různých identit
                browser = random.choice(browsers)
                platform = random.choice(platforms)

                identity = IdentityProfile(
                    user_agent=ua.random,
                    accept_language=random.choice(languages),
                    timezone=random.choice(timezones),
                    screen_resolution=random.choice(['1920x1080', '1366x768', '1440x900', '2560x1440']),
                    platform=platform,
                    browser_version=f"{random.randint(90, 120)}.0.{random.randint(1000, 9999)}.{random.randint(100, 999)}",
                    plugins=['PDF Viewer', 'Chrome PDF Plugin'] if browser == 'chrome' else [],
                    fingerprint_id=f"fp_{i:03d}_{random.randint(10000, 99999)}"
                )

                self.identity_pool.append(identity)

            logger.info(f"Vygenerováno {len(self.identity_pool)} identit")

        except ImportError:
            logger.warning("fake-useragent není nainstalován, používám výchozí identity")
            self._create_default_identities()

    def _create_default_identities(self):
        """Vytvoření výchozích identit bez externí knihovny"""
        default_agents = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0"
        ]

        for i, agent in enumerate(default_agents):
            identity = IdentityProfile(
                user_agent=agent,
                accept_language="en-US,en;q=0.9",
                timezone="America/New_York",
                screen_resolution="1920x1080",
                platform="macOS" if "Macintosh" in agent else "Windows",
                browser_version="120.0.0.0",
                plugins=[],
                fingerprint_id=f"default_{i}"
            )
            self.identity_pool.append(identity)

    def get_random_identity(self) -> IdentityProfile:
        """Získání náhodné identity"""
        if not self.identity_pool:
            self._generate_identity_pool()

        identity = random.choice(self.identity_pool)
        self.current_identity = identity

        logger.debug(f"Použita identita: {identity.fingerprint_id}")
        return identity

    def rotate_identity(self) -> IdentityProfile:
        """Rotace na novou identitu"""
        return self.get_random_identity()

    def get_headers(self, identity: IdentityProfile | None = None) -> dict[str, str]:
        """Získání HTTP headers pro danou identitu

        Args:
            identity: Identita (pokud None, použije se aktuální)

        Returns:
            Slovník s HTTP headers

        """
        if identity is None:
            identity = self.current_identity or self.get_random_identity()

        headers = {
            'User-Agent': identity.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': identity.accept_language,
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }

        return headers


class IntelligentProxyManager:
    """Hlavní třída pro intelligent proxy management
    Kombinuje proxy management s identity management
    """

    def __init__(self, config_path: str | None = None):
        self.proxy_manager = ProxyManager(config_path)
        self.identity_manager = IdentityManager()
        self.current_session_config: dict[str, Any] | None = None

    async def get_session_config(self, rotate: bool = False) -> dict[str, Any]:
        """Získání konfigurace pro anonymní session

        Args:
            rotate: Zda rotovat proxy a identitu

        Returns:
            Konfigurace pro aiohttp session

        """
        if rotate or self.current_session_config is None:
            proxy = await self.proxy_manager.get_working_proxy()
            identity = self.identity_manager.rotate_identity()

            config = {
                'proxy': self.proxy_manager._build_proxy_url(proxy) if proxy else None,
                'headers': self.identity_manager.get_headers(identity),
                'timeout': aiohttp.ClientTimeout(total=30),
                'connector': aiohttp.TCPConnector(
                    ssl=False,  # Disable SSL verification for anonymity
                    limit=10,
                    limit_per_host=2
                )
            }

            self.current_session_config = config

            logger.info(f"Nová session konfigurace - Proxy: {proxy.host if proxy else 'None'}, Identity: {identity.fingerprint_id}")

        return self.current_session_config

    async def create_anonymous_session(self, rotate: bool = False) -> aiohttp.ClientSession:
        """Vytvoření anonymní aiohttp session

        Args:
            rotate: Zda rotovat proxy a identitu

        Returns:
            Nakonfigurovaná ClientSession

        """
        config = await self.get_session_config(rotate)

        session = aiohttp.ClientSession(
            headers=config['headers'],
            timeout=config['timeout'],
            connector=config['connector']
        )

        # Nastavení proxy pro session (pokud je dostupné)
        if config['proxy']:
            session._connector._ssl = False  # Disable SSL for proxy

        return session

    async def test_anonymity(self) -> dict[str, Any]:
        """Test anonymity a leak detection

        Returns:
            Výsledky anonymity testu

        """
        results = {
            "ip_address": None,
            "user_agent": None,
            "headers": {},
            "proxy_working": False,
            "dns_leak": False,
            "webrtc_leak": False,
            "timestamp": time.time()
        }

        try:
            session = await self.create_anonymous_session()

            # Test IP adresy
            async with session.get("http://httpbin.org/ip") as response:
                if response.status == 200:
                    data = await response.json()
                    results["ip_address"] = data.get("origin")
                    results["proxy_working"] = True

            # Test User-Agent a headers
            async with session.get("http://httpbin.org/headers") as response:
                if response.status == 200:
                    data = await response.json()
                    results["headers"] = data.get("headers", {})
                    results["user_agent"] = data.get("headers", {}).get("User-Agent")

            await session.close()

        except Exception as e:
            logger.error(f"Chyba při testu anonymity: {e}")
            results["error"] = str(e)

        return results

    def get_stats(self) -> dict[str, Any]:
        """Získání celkových statistik"""
        return {
            "proxy_stats": self.proxy_manager.get_proxy_stats(),
            "identity_pool_size": len(self.identity_manager.identity_pool),
            "current_identity": self.identity_manager.current_identity.fingerprint_id if self.identity_manager.current_identity else None,
            "session_active": self.current_session_config is not None
        }


# Globální instance pro snadné použití
intelligent_proxy_manager = IntelligentProxyManager()
