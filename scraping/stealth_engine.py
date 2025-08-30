"""
Stealth Engine
Pokročilé techniky pro obcházení detekce a rotaci identity

Author: Senior Python/MLOps Agent
"""

import asyncio
import logging
import random
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path
import aiohttp
from fake_useragent import UserAgent

logger = logging.getLogger(__name__)


@dataclass
class StealthProfile:
    """Profil pro stealth operace"""
    profile_id: str
    user_agent: str
    viewport_size: Tuple[int, int]
    timezone: str
    language: str
    platform: str
    headers: Dict[str, str] = field(default_factory=dict)
    fingerprint_hash: str = ""
    created_at: float = field(default_factory=time.time)
    usage_count: int = 0
    max_usage: int = 20


@dataclass
class RequestMetrics:
    """Metriky pro sledování requestů"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    blocked_requests: int = 0
    average_response_time: float = 0.0
    last_request_time: float = 0.0


class StealthEngine:
    """
    Pokročilý stealth engine pro anonymní web scraping
    Implementuje rotaci User-Agentů, fingerprint masking a evasion techniky
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.stealth_config = config.get("stealth", {})

        # User agent management
        self.ua_generator = None
        self.active_profiles: Dict[str, StealthProfile] = {}
        self.current_profile: Optional[StealthProfile] = None

        # Rate limiting
        self.request_intervals = self.stealth_config.get("request_intervals", [1, 3, 5])
        self.last_request_time = 0.0

        # Metrics
        self.metrics = RequestMetrics()

        # Detection evasion
        self.blocked_indicators = [
            "captcha", "blocked", "forbidden", "rate limit",
            "too many requests", "access denied", "cloudflare"
        ]

        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Inicializace stealth engine"""
        self.logger.info("🥷 Inicializuji Stealth Engine...")

        # Inicializace User Agent generátoru
        try:
            self.ua_generator = UserAgent()
            self.logger.info("✅ User Agent generator inicializován")
        except Exception as e:
            self.logger.warning(f"UA generator failed, using fallback: {e}")
            self.ua_generator = None

        # Vytvoření počátečních profilů
        await self._create_initial_profiles()

        # Výběr prvního profilu
        if self.active_profiles:
            self.current_profile = list(self.active_profiles.values())[0]

    async def _create_initial_profiles(self, count: int = 5):
        """Vytvoří počáteční sadu stealth profilů"""

        for i in range(count):
            profile = await self._generate_stealth_profile()
            self.active_profiles[profile.profile_id] = profile

        self.logger.info(f"✅ Vytvořeno {count} stealth profilů")

    async def _generate_stealth_profile(self) -> StealthProfile:
        """Generuje nový stealth profil"""
        import uuid

        profile_id = f"profile_{uuid.uuid4().hex[:8]}"

        # Generování User Agent
        user_agent = self._generate_user_agent()

        # Generování viewport size
        common_resolutions = [
            (1920, 1080), (1366, 768), (1440, 900), (1536, 864),
            (1280, 720), (1600, 900), (2560, 1440), (1920, 1200)
        ]
        viewport_size = random.choice(common_resolutions)

        # Generování timezone
        timezones = [
            "America/New_York", "Europe/London", "Europe/Berlin",
            "Asia/Tokyo", "Australia/Sydney", "America/Los_Angeles",
            "Europe/Prague", "Europe/Vienna"
        ]
        timezone = random.choice(timezones)

        # Generování language
        languages = [
            "en-US", "en-GB", "de-DE", "fr-FR", "es-ES",
            "it-IT", "cs-CZ", "pl-PL"
        ]
        language = random.choice(languages)

        # Detekce platform z User Agent
        platform = self._extract_platform_from_ua(user_agent)

        # Generování headers
        headers = self._generate_stealth_headers(user_agent, language)

        # Výpočet fingerprint hash
        fingerprint_data = f"{user_agent}{viewport_size}{timezone}{language}{platform}"
        fingerprint_hash = hashlib.md5(fingerprint_data.encode()).hexdigest()[:12]

        profile = StealthProfile(
            profile_id=profile_id,
            user_agent=user_agent,
            viewport_size=viewport_size,
            timezone=timezone,
            language=language,
            platform=platform,
            headers=headers,
            fingerprint_hash=fingerprint_hash,
            max_usage=random.randint(15, 25)  # Random max usage
        )

        return profile

    def _generate_user_agent(self) -> str:
        """Generuje User Agent string"""

        if self.ua_generator:
            try:
                # Pokus o generování specifického typu
                browser_types = ['chrome', 'firefox', 'safari', 'edge']
                browser_type = random.choice(browser_types)

                if browser_type == 'chrome':
                    return self.ua_generator.chrome
                elif browser_type == 'firefox':
                    return self.ua_generator.firefox
                elif browser_type == 'safari':
                    return self.ua_generator.safari
                else:
                    return self.ua_generator.random

            except Exception:
                # Fallback na random
                try:
                    return self.ua_generator.random
                except Exception:
                    pass

        # Fallback User Agents
        fallback_uas = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0"
        ]

        return random.choice(fallback_uas)

    def _extract_platform_from_ua(self, user_agent: str) -> str:
        """Extrahuje platform z User Agent"""
        ua_lower = user_agent.lower()

        if "macintosh" in ua_lower or "mac os" in ua_lower:
            return "MacIntel"
        elif "windows" in ua_lower:
            return "Win32"
        elif "linux" in ua_lower:
            return "Linux x86_64"
        elif "android" in ua_lower:
            return "Linux armv7l"
        else:
            return "Win32"  # Default fallback

    def _generate_stealth_headers(self, user_agent: str, language: str) -> Dict[str, str]:
        """Generuje stealth HTTP headers"""

        headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": f"{language},en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1"
        }

        # Občasné přidání dalších headers pro realističnost
        if random.random() < 0.3:
            headers["DNT"] = "1"

        if random.random() < 0.4:
            headers["Pragma"] = "no-cache"

        if random.random() < 0.2:
            headers["X-Requested-With"] = "XMLHttpRequest"

        return headers

    async def get_stealth_profile(self, force_new: bool = False) -> StealthProfile:
        """Získá aktivní stealth profil nebo vytvoří nový"""

        # Kontrola, zda potřebujeme nový profil
        if force_new or await self._should_rotate_profile():
            await self._rotate_profile()

        # Vrátí aktuální profil
        if self.current_profile and not self._is_profile_exhausted(self.current_profile):
            self.current_profile.usage_count += 1
            return self.current_profile

        # Najde další dostupný profil
        for profile in self.active_profiles.values():
            if not self._is_profile_exhausted(profile):
                self.current_profile = profile
                profile.usage_count += 1
                return profile

        # Vytvoří nový profil pokud všechny jsou vyčerpané
        new_profile = await self._generate_stealth_profile()
        self.active_profiles[new_profile.profile_id] = new_profile
        self.current_profile = new_profile
        return new_profile

    def _is_profile_exhausted(self, profile: StealthProfile) -> bool:
        """Kontroluje, zda je profil vyčerpán"""
        return profile.usage_count >= profile.max_usage

    async def _should_rotate_profile(self) -> bool:
        """Určuje, zda je čas rotovat profil"""

        if not self.current_profile:
            return True

        # Rotace podle usage count
        if self._is_profile_exhausted(self.current_profile):
            return True

        # Rotace podle času (každých 30 minut)
        if time.time() - self.current_profile.created_at > 1800:
            return True

        # Rotace podle počtu failed requestů
        if self.metrics.failed_requests > 5:
            return True

        return False

    async def _rotate_profile(self):
        """Rotuje aktivní profil"""
        self.logger.debug("🔄 Rotating stealth profile...")

        # Vytvoří nový profil
        new_profile = await self._generate_stealth_profile()
        self.active_profiles[new_profile.profile_id] = new_profile

        # Vyčistí staré profily
        current_time = time.time()
        expired_profiles = [
            pid for pid, profile in self.active_profiles.items()
            if current_time - profile.created_at > 3600  # 1 hodina
        ]

        for pid in expired_profiles:
            del self.active_profiles[pid]

        self.current_profile = new_profile

        # Reset metrics při rotaci
        self.metrics.failed_requests = 0

    async def apply_request_delay(self):
        """Aplikuje inteligentní delay mezi requesty"""

        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        # Výběr random intervalu
        min_interval = min(self.request_intervals)
        max_interval = max(self.request_intervals)

        # Adaptivní interval na základě úspěšnosti
        if self.metrics.failed_requests > 2:
            # Zpomalení při chybách
            interval = random.uniform(max_interval, max_interval * 2)
        else:
            # Normální interval
            interval = random.uniform(min_interval, max_interval)

        # Aplikace delay pokud je potřeba
        if time_since_last < interval:
            delay = interval - time_since_last
            await asyncio.sleep(delay)

        self.last_request_time = time.time()

    async def make_stealth_request(
        self,
        method: str,
        url: str,
        session: aiohttp.ClientSession,
        **kwargs
    ) -> aiohttp.ClientResponse:
        """Provede stealth HTTP request"""

        start_time = time.time()

        try:
            # Aplikace delay
            await self.apply_request_delay()

            # Získání stealth profilu
            profile = await self.get_stealth_profile()

            # Aktualizace headers
            if 'headers' not in kwargs:
                kwargs['headers'] = {}

            kwargs['headers'].update(profile.headers)

            # Provedení requestu
            async with session.request(method, url, **kwargs) as response:

                # Aktualizace metrics
                response_time = time.time() - start_time
                self.metrics.total_requests += 1
                self.metrics.last_request_time = time.time()

                # Kontrola na blokování
                if await self._detect_blocking(response):
                    self.metrics.blocked_requests += 1
                    self.metrics.failed_requests += 1
                    self.logger.warning(f"🚫 Request blocked: {url}")

                    # Force profile rotation
                    await self._rotate_profile()
                    raise aiohttp.ClientError("Request appears to be blocked")
                else:
                    self.metrics.successful_requests += 1

                    # Update average response time
                    if self.metrics.successful_requests > 1:
                        self.metrics.average_response_time = (
                            (self.metrics.average_response_time * (self.metrics.successful_requests - 1) + response_time) /
                            self.metrics.successful_requests
                        )
                    else:
                        self.metrics.average_response_time = response_time

                return response

        except Exception as e:
            self.metrics.failed_requests += 1
            self.logger.debug(f"Stealth request failed: {e}")
            raise

    async def _detect_blocking(self, response: aiohttp.ClientResponse) -> bool:
        """Detekuje, zda byla request zablokována"""

        # Kontrola status kódů
        blocking_status_codes = [403, 429, 503, 521, 522, 523, 524]
        if response.status in blocking_status_codes:
            return True

        # Kontrola response headers
        blocking_headers = [
            "cf-ray",  # Cloudflare
            "x-sucuri-id",  # Sucuri
            "server: cloudflare"
        ]

        for header_name, header_value in response.headers.items():
            header_check = f"{header_name.lower()}: {header_value.lower()}"
            if any(indicator in header_check for indicator in blocking_headers):
                return True

        # Kontrola obsahu response (pouze pro text)
        try:
            if response.content_type and "text" in response.content_type:
                content = await response.text()
                content_lower = content.lower()

                if any(indicator in content_lower for indicator in self.blocked_indicators):
                    return True
        except Exception:
            # Pokud nelze číst obsah, nepovažujeme to za blokování
            pass

        return False

    def get_stealth_stats(self) -> Dict[str, Any]:
        """Vrátí statistiky stealth engine"""

        return {
            "active_profiles": len(self.active_profiles),
            "current_profile_id": self.current_profile.profile_id if self.current_profile else None,
            "current_profile_usage": self.current_profile.usage_count if self.current_profile else 0,
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "blocked_requests": self.metrics.blocked_requests,
            "success_rate": (
                self.metrics.successful_requests / self.metrics.total_requests
                if self.metrics.total_requests > 0 else 0
            ),
            "average_response_time": self.metrics.average_response_time,
            "last_request_time": self.metrics.last_request_time
        }

    async def export_profiles(self, filepath: str):
        """Exportuje profily do souboru"""

        export_data = {
            "profiles": {},
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "blocked_requests": self.metrics.blocked_requests
            },
            "exported_at": time.time()
        }

        for profile_id, profile in self.active_profiles.items():
            export_data["profiles"][profile_id] = {
                "user_agent": profile.user_agent,
                "viewport_size": profile.viewport_size,
                "timezone": profile.timezone,
                "language": profile.language,
                "platform": profile.platform,
                "fingerprint_hash": profile.fingerprint_hash,
                "usage_count": profile.usage_count,
                "max_usage": profile.max_usage,
                "created_at": profile.created_at
            }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        self.logger.info(f"✅ Stealth profiles exported to {filepath}")


# Test funkce
async def test_stealth_engine():
    """Test funkce pro stealth engine"""

    config = {
        "stealth": {
            "request_intervals": [1, 2, 3],
            "max_profile_usage": 20
        }
    }

    engine = StealthEngine(config)
    await engine.initialize()

    # Test profile generation
    profile = await engine.get_stealth_profile()
    print(f"📱 Generated profile: {profile.fingerprint_hash}")
    print(f"🌐 User Agent: {profile.user_agent[:80]}...")

    # Test s real requestem
    async with aiohttp.ClientSession() as session:
        try:
            response = await engine.make_stealth_request(
                "GET",
                "https://httpbin.org/user-agent",
                session
            )
            print(f"✅ Stealth request successful: {response.status}")
        except Exception as e:
            print(f"❌ Stealth request failed: {e}")

    # Statistiky
    stats = engine.get_stealth_stats()
    print(f"📊 Stealth stats: {stats}")


if __name__ == "__main__":
    asyncio.run(test_stealth_engine())
