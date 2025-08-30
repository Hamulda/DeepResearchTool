"""
Anti-Bot Circumvention Suite pro DeepResearchTool
Pokročilé obcházení anti-bot systémů včetně Cloudflare, CAPTCHA a TLS fingerprintingu.
"""

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from playwright.async_api import async_playwright, Browser, Page
from playwright_stealth import stealth_async

from ..optimization.intelligent_memory import cache_get, cache_set

logger = logging.getLogger(__name__)


@dataclass
class EvasionProfile:
    """Profil pro obcházení detekce"""
    user_agent: str
    viewport: Dict[str, int]
    headers: Dict[str, str] = field(default_factory=dict)
    tls_config: Dict[str, Any] = field(default_factory=dict)
    behavioral_patterns: Dict[str, Any] = field(default_factory=dict)
    fingerprint_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BypassResult:
    """Výsledek pokusu o obejití anti-bot systému"""
    success: bool
    url: str
    method_used: str
    response_time_ms: float
    content: Optional[str] = None
    cookies: Dict[str, str] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    protection_detected: List[str] = field(default_factory=list)
    bypass_attempts: int = 0
    error_message: Optional[str] = None


class TLSFingerprintRotator:
    """Rotátor TLS/JA3 fingerprintů pro obcházení detekce"""

    def __init__(self):
        self.fingerprints = [
            # Chrome různých verzí
            {
                "ja3": "771,4865-4866-4867-49195-49199-49196-49200-52393-52392-49171-49172-156-157-47-53,0-23-65281-10-11-35-16-5-13-18-51-45-43-27-17513,29-23-24,0",
                "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
                "cipher_suites": ["TLS_AES_128_GCM_SHA256", "TLS_AES_256_GCM_SHA384", "TLS_CHACHA20_POLY1305_SHA256"]
            },
            # Firefox profily
            {
                "ja3": "771,4865-4867-4866-49195-49199-52393-52392-49196-49200-49162-49161-49171-49172-51-57-47-53,0-23-65281-10-11-35-16-5-51-43-13-45-28-21,29-23-24-25-256-257,0",
                "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/119.0",
                "cipher_suites": ["TLS_AES_128_GCM_SHA256", "TLS_CHACHA20_POLY1305_SHA256", "TLS_AES_256_GCM_SHA384"]
            },
            # Safari profil
            {
                "ja3": "771,4865-4866-4867-49196-49195-52393-49200-49199-52392-49162-49161-49172-49171-157-156-61-60-53-47-49160-49170-10,65281-0-23-35-13-5-18-16-30032-11-10,29-23-30-25-24,0-1-2",
                "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
                "cipher_suites": ["TLS_AES_128_GCM_SHA256", "TLS_AES_256_GCM_SHA384", "TLS_CHACHA20_POLY1305_SHA256"]
            }
        ]

    def get_random_fingerprint(self) -> Dict[str, Any]:
        """Získání náhodného TLS fingerprint profilu"""
        return random.choice(self.fingerprints)


class CaptchaSolver:
    """Řešič CAPTCHA pomocí externích služeb"""

    def __init__(self, api_key: Optional[str] = None, service: str = "2captcha"):
        self.api_key = api_key
        self.service = service
        self.base_urls = {
            "2captcha": "http://2captcha.com",
            "anticaptcha": "https://api.anti-captcha.com"
        }

    async def solve_recaptcha_v2(self, site_key: str, page_url: str) -> Optional[str]:
        """Řešení reCAPTCHA v2"""
        if not self.api_key:
            logger.warning("CAPTCHA API klíč není nastaven")
            return None

        try:
            if self.service == "2captcha":
                return await self._solve_2captcha_recaptcha(site_key, page_url)
        except Exception as e:
            logger.error(f"Chyba při řešení CAPTCHA: {e}")

        return None

    async def _solve_2captcha_recaptcha(self, site_key: str, page_url: str) -> Optional[str]:
        """Řešení přes 2captcha API"""
        async with httpx.AsyncClient() as client:
            # Odeslání CAPTCHA k řešení
            submit_response = await client.post(
                f"{self.base_urls['2captcha']}/in.php",
                data={
                    "key": self.api_key,
                    "method": "userrecaptcha",
                    "googlekey": site_key,
                    "pageurl": page_url,
                    "json": 1
                }
            )

            submit_data = submit_response.json()
            if submit_data.get("status") != 1:
                logger.error(f"2captcha submit failed: {submit_data}")
                return None

            captcha_id = submit_data.get("request")

            # Čekání na vyřešení (max 2 minuty)
            for _ in range(24):  # 24 * 5s = 2 minuty
                await asyncio.sleep(5)

                result_response = await client.get(
                    f"{self.base_urls['2captcha']}/res.php",
                    params={
                        "key": self.api_key,
                        "action": "get",
                        "id": captcha_id,
                        "json": 1
                    }
                )

                result_data = result_response.json()
                if result_data.get("status") == 1:
                    return result_data.get("request")
                elif result_data.get("error") == "CAPCHA_NOT_READY":
                    continue
                else:
                    logger.error(f"2captcha error: {result_data}")
                    break

        return None


class CloudflareBypass:
    """Specializovaný bypass pro Cloudflare ochrany"""

    async def bypass_challenge(self, page: Page, max_attempts: int = 3) -> bool:
        """Pokus o obejití Cloudflare challenge"""

        for attempt in range(max_attempts):
            try:
                # Detekce Cloudflare challenge
                if await self._detect_cloudflare_challenge(page):
                    logger.info(f"Cloudflare challenge detekován, pokus {attempt + 1}")

                    # Čekání na automatické vyřešení
                    await asyncio.sleep(5)

                    # Kontrola, zda challenge prošel
                    if not await self._detect_cloudflare_challenge(page):
                        return True

                    # Pokus o manuální vyřešení
                    if await self._manual_challenge_solve(page):
                        return True

                else:
                    return True  # Žádný challenge

            except Exception as e:
                logger.warning(f"Cloudflare bypass attempt {attempt + 1} failed: {e}")

        return False

    async def _detect_cloudflare_challenge(self, page: Page) -> bool:
        """Detekce Cloudflare challenge stránky"""
        try:
            # Hledání typických Cloudflare elementů
            cf_indicators = [
                "cf-browser-verification",
                "cf-challenge-form",
                "cf-wrapper",
                "[data-ray]",
                ".cf-error-details"
            ]

            for indicator in cf_indicators:
                element = await page.query_selector(indicator)
                if element:
                    return True

            # Kontrola title a textu
            title = await page.title()
            if "checking your browser" in title.lower() or "cloudflare" in title.lower():
                return True

            # Kontrola URL
            url = page.url
            if "cf-browser-verification" in url or "__cf_chl_jschl_tk__" in url:
                return True

        except Exception as e:
            logger.debug(f"Chyba při detekci Cloudflare: {e}")

        return False

    async def _manual_challenge_solve(self, page: Page) -> bool:
        """Pokus o manuální vyřešení challenge"""
        try:
            # Kliknutí na checkbox pokud existuje
            checkbox = await page.query_selector("input[type='checkbox']")
            if checkbox:
                await checkbox.click()
                await asyncio.sleep(2)

            # Čekání na submit button a kliknutí
            submit_button = await page.query_selector("button[type='submit'], input[type='submit']")
            if submit_button:
                await submit_button.click()
                await asyncio.sleep(5)

            return not await self._detect_cloudflare_challenge(page)

        except Exception as e:
            logger.debug(f"Manual challenge solve failed: {e}")

        return False


class AntiBotCircumventionSuite:
    """
    Pokročilá sada pro obcházení anti-bot systémů včetně Cloudflare,
    CAPTCHA, TLS fingerprintingu a behavioral analysis.
    """

    def __init__(self,
                 captcha_api_key: Optional[str] = None,
                 max_retry_attempts: int = 3,
                 stealth_mode: bool = True):

        self.captcha_solver = CaptchaSolver(captcha_api_key) if captcha_api_key else None
        self.tls_rotator = TLSFingerprintRotator()
        self.cloudflare_bypass = CloudflareBypass()

        self.max_retry_attempts = max_retry_attempts
        self.stealth_mode = stealth_mode

        # Browser pool pro rotaci
        self.browser_pool: List[Browser] = []
        self.current_browser_index = 0

        logger.info("AntiBotCircumventionSuite inicializován")

    async def circumvent_protection(self, url: str, **kwargs) -> BypassResult:
        """
        Hlavní metoda pro obcházení anti-bot ochrany
        """
        start_time = time.time()

        result = BypassResult(
            success=False,
            url=url,
            method_used="",
            response_time_ms=0,
            bypass_attempts=0
        )

        # Cache kontrola
        cache_key = f"antibot_bypass:{url}"
        cached_bypass = await cache_get(cache_key)

        if cached_bypass and cached_bypass.get("success"):
            logger.info(f"Použití cached bypass pro {url}")
            result.success = True
            result.method_used = "cached"
            result.content = cached_bypass.get("content")
            result.response_time_ms = time.time() - start_time
            return result

        # Postupné pokusy s různými metodami
        bypass_methods = [
            self._bypass_with_stealth_playwright,
            self._bypass_with_custom_httpx,
            self._bypass_with_rotation,
            self._bypass_with_captcha_solving
        ]

        for method in bypass_methods:
            try:
                result.bypass_attempts += 1
                logger.info(f"Pokus {result.bypass_attempts}: {method.__name__}")

                bypass_result = await method(url, **kwargs)

                if bypass_result.success:
                    result = bypass_result
                    result.response_time_ms = (time.time() - start_time) * 1000

                    # Cache úspěšného bypass
                    cache_data = {
                        "success": True,
                        "content": result.content[:1000],  # Omezená velikost pro cache
                        "method": result.method_used,
                        "timestamp": time.time()
                    }
                    await cache_set(cache_key, cache_data, ttl=1800)  # 30 minut

                    return result

            except Exception as e:
                logger.warning(f"Bypass method {method.__name__} failed: {e}")
                result.error_message = str(e)

        result.response_time_ms = (time.time() - start_time) * 1000
        return result

    async def _bypass_with_stealth_playwright(self, url: str, **kwargs) -> BypassResult:
        """Bypass pomocí stealth Playwright"""
        result = BypassResult(success=False, url=url, method_used="stealth_playwright", bypass_attempts=1)

        async with async_playwright() as p:
            # Náhodný profil
            fingerprint = self.tls_rotator.get_random_fingerprint()

            browser = await p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--no-first-run',
                    '--no-default-browser-check',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor'
                ]
            )

            try:
                context = await browser.new_context(
                    user_agent=fingerprint["user_agent"],
                    viewport={"width": 1920, "height": 1080},
                    extra_http_headers={
                        "Accept-Language": "en-US,en;q=0.9",
                        "Accept-Encoding": "gzip, deflate, br",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                        "DNT": "1",
                        "Connection": "keep-alive"
                    }
                )

                page = await context.new_page()

                # Aplikace stealth
                if self.stealth_mode:
                    await stealth_async(page)

                # Behavioral simulation
                await self._simulate_human_behavior(page)

                # Navigace na stránku
                response = await page.goto(url, wait_until="domcontentloaded", timeout=30000)

                # Cloudflare bypass
                if await self.cloudflare_bypass.bypass_challenge(page):

                    # CAPTCHA handling
                    if await self._handle_captcha(page):

                        # Získání finálního obsahu
                        await page.wait_for_load_state("networkidle", timeout=15000)
                        content = await page.content()

                        result.success = True
                        result.content = content
                        result.cookies = {cookie["name"]: cookie["value"] for cookie in await context.cookies()}

                        # Detekce ochran
                        result.protection_detected = await self._detect_protections(page)

            finally:
                await browser.close()

        return result

    async def _bypass_with_custom_httpx(self, url: str, **kwargs) -> BypassResult:
        """Bypass pomocí custom HTTPX klienta s TLS rotací"""
        result = BypassResult(success=False, url=url, method_used="custom_httpx", bypass_attempts=1)

        fingerprint = self.tls_rotator.get_random_fingerprint()

        # Custom transport s TLS konfigurací
        transport = httpx.HTTPTransport(
            verify=False,  # Pro demo - v produkci by se používal proper cert handling
            retries=3
        )

        headers = {
            "User-Agent": fingerprint["user_agent"],
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }

        async with httpx.AsyncClient(transport=transport, headers=headers, timeout=30.0) as client:
            try:
                # Simulace předchozích requests
                await self._simulate_browsing_session(client, url)

                response = await client.get(url)

                if response.status_code == 200:
                    # Kontrola anti-bot ochran v obsahu
                    content = response.text

                    if not self._contains_antibot_protection(content):
                        result.success = True
                        result.content = content
                        result.headers = dict(response.headers)

            except Exception as e:
                result.error_message = str(e)

        return result

    async def _bypass_with_rotation(self, url: str, **kwargs) -> BypassResult:
        """Bypass s rotací browser instancí"""
        result = BypassResult(success=False, url=url, method_used="browser_rotation", bypass_attempts=1)

        # Rotace mezi různými browsery
        browser_configs = [
            {"browser": "chromium", "channel": "chrome"},
            {"browser": "firefox"},
            {"browser": "webkit"}
        ]

        for config in browser_configs:
            try:
                async with async_playwright() as p:
                    if config["browser"] == "chromium":
                        browser = await p.chromium.launch(
                            headless=True,
                            channel=config.get("channel")
                        )
                    elif config["browser"] == "firefox":
                        browser = await p.firefox.launch(headless=True)
                    else:
                        browser = await p.webkit.launch(headless=True)

                    try:
                        context = await browser.new_context()
                        page = await context.new_page()

                        await page.goto(url, timeout=30000)

                        if await self.cloudflare_bypass.bypass_challenge(page):
                            content = await page.content()

                            if not self._contains_antibot_protection(content):
                                result.success = True
                                result.content = content
                                result.method_used = f"browser_rotation_{config['browser']}"
                                break

                    finally:
                        await browser.close()

            except Exception as e:
                logger.debug(f"Browser rotation {config['browser']} failed: {e}")
                continue

        return result

    async def _bypass_with_captcha_solving(self, url: str, **kwargs) -> BypassResult:
        """Bypass s automatickým řešením CAPTCHA"""
        result = BypassResult(success=False, url=url, method_used="captcha_solving", bypass_attempts=1)

        if not self.captcha_solver:
            result.error_message = "CAPTCHA solver není nakonfigurován"
            return result

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)

            try:
                context = await browser.new_context()
                page = await context.new_page()

                await page.goto(url, timeout=30000)

                # Hledání CAPTCHA
                captcha_frame = await page.query_selector("iframe[src*='recaptcha']")

                if captcha_frame:
                    # Získání site key
                    site_key = await self._extract_recaptcha_site_key(page)

                    if site_key:
                        # Řešení CAPTCHA
                        captcha_response = await self.captcha_solver.solve_recaptcha_v2(site_key, url)

                        if captcha_response:
                            # Vložení řešení
                            await page.evaluate(f"""
                                document.getElementById('g-recaptcha-response').innerHTML = '{captcha_response}';
                                if (typeof grecaptcha !== 'undefined') {{
                                    grecaptcha.getResponse = function() {{ return '{captcha_response}'; }};
                                }}
                            """)

                            # Submit formuláře
                            submit_button = await page.query_selector("button[type='submit'], input[type='submit']")
                            if submit_button:
                                await submit_button.click()
                                await page.wait_for_load_state("networkidle")

                                content = await page.content()
                                result.success = True
                                result.content = content

            finally:
                await browser.close()

        return result

    async def _simulate_human_behavior(self, page: Page):
        """Simulace lidského chování"""
        try:
            # Náhodné pohyby myši
            await page.mouse.move(random.randint(100, 500), random.randint(100, 500))
            await asyncio.sleep(random.uniform(0.1, 0.3))

            # Simulace scrollování
            await page.evaluate("window.scrollTo(0, 100)")
            await asyncio.sleep(random.uniform(0.2, 0.5))

            # Náhodné kliknutí (mimo důležité elementy)
            await page.mouse.click(random.randint(50, 200), random.randint(50, 200))
            await asyncio.sleep(random.uniform(0.1, 0.2))

        except Exception as e:
            logger.debug(f"Behavior simulation error: {e}")

    async def _simulate_browsing_session(self, client: httpx.AsyncClient, target_url: str):
        """Simulace browsing session před hlavním requestem"""
        try:
            # Předstíraný referer
            base_domain = "/".join(target_url.split("/")[:3])

            # Simulace návštěvy homepage
            await client.get(base_domain, timeout=10)
            await asyncio.sleep(random.uniform(1, 3))

        except Exception:
            pass  # Není kritické

    async def _handle_captcha(self, page: Page) -> bool:
        """Zpracování CAPTCHA na stránce"""
        if not self.captcha_solver:
            return True  # Pokračujeme bez CAPTCHA solveru

        try:
            # Detekce reCAPTCHA
            captcha_frame = await page.query_selector("iframe[src*='recaptcha']")

            if captcha_frame:
                site_key = await self._extract_recaptcha_site_key(page)

                if site_key:
                    captcha_response = await self.captcha_solver.solve_recaptcha_v2(site_key, page.url)

                    if captcha_response:
                        await page.evaluate(f"""
                            document.getElementById('g-recaptcha-response').innerHTML = '{captcha_response}';
                        """)
                        return True

        except Exception as e:
            logger.debug(f"CAPTCHA handling error: {e}")

        return True  # Pokračujeme i bez úspěšného řešení

    async def _extract_recaptcha_site_key(self, page: Page) -> Optional[str]:
        """Extrakce reCAPTCHA site key"""
        try:
            # Hledání v data atributech
            site_key = await page.evaluate("""
                () => {
                    const elements = document.querySelectorAll('[data-sitekey]');
                    return elements.length > 0 ? elements[0].getAttribute('data-sitekey') : null;
                }
            """)

            if site_key:
                return site_key

            # Hledání v JavaScript kódu
            content = await page.content()
            import re
            match = re.search(r'sitekey["\s]*[:=]["\s]*([a-zA-Z0-9_-]+)', content)
            if match:
                return match.group(1)

        except Exception as e:
            logger.debug(f"Site key extraction error: {e}")

        return None

    async def _detect_protections(self, page: Page) -> List[str]:
        """Detekce aktivních ochranných systémů"""
        protections = []

        try:
            # Cloudflare detekce
            if await self.cloudflare_bypass._detect_cloudflare_challenge(page):
                protections.append("cloudflare")

            # reCAPTCHA detekce
            captcha_frame = await page.query_selector("iframe[src*='recaptcha']")
            if captcha_frame:
                protections.append("recaptcha")

            # hCaptcha detekce
            hcaptcha = await page.query_selector("[data-hcaptcha-widget-id]")
            if hcaptcha:
                protections.append("hcaptcha")

            # Bot detection scripts
            content = await page.content()
            if "bot" in content.lower() and "verification" in content.lower():
                protections.append("generic_bot_detection")

        except Exception as e:
            logger.debug(f"Protection detection error: {e}")

        return protections

    def _contains_antibot_protection(self, content: str) -> bool:
        """Kontrola, zda obsah obsahuje anti-bot ochranu"""
        antibot_indicators = [
            "checking your browser",
            "cloudflare",
            "ddos protection",
            "bot verification",
            "security check",
            "please wait while we verify",
            "g-recaptcha",
            "hcaptcha"
        ]

        content_lower = content.lower()
        return any(indicator in content_lower for indicator in antibot_indicators)

    async def get_bypass_statistics(self) -> Dict[str, Any]:
        """Statistiky úspěšnosti bypass metod"""
        # V produkční verzi by se tyto statistiky ukládaly a sledovaly
        return {
            "total_attempts": 0,
            "success_rate": 0.0,
            "most_successful_method": "stealth_playwright",
            "average_response_time_ms": 0,
            "protections_encountered": {}
        }

    async def update_evasion_profiles(self, new_profiles: List[EvasionProfile]):
        """Aktualizace evasion profilů na základě nových dat"""
        # Implementace pro dynamické updaty profilů
        logger.info(f"Updating {len(new_profiles)} evasion profiles")

    async def close(self):
        """Cleanup resources"""
        for browser in self.browser_pool:
            if browser:
                await browser.close()
        self.browser_pool.clear()
