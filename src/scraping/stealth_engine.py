"""Stealth Engine
Pokročilé techniky pro anonymní web scraping a detekci anti-bot opatření

Author: Senior Python/MLOps Agent
"""

import asyncio
from dataclasses import dataclass
import logging
import random
import re
import time
from typing import Any

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

logger = logging.getLogger(__name__)


@dataclass
class StealthConfig:
    """Konfigurace pro stealth engine"""

    user_agent_rotation: bool = True
    header_randomization: bool = True
    timing_randomization: bool = True
    min_delay: float = 1.0
    max_delay: float = 3.0
    javascript_enabled: bool = True
    cookies_enabled: bool = True
    webrtc_disabled: bool = True
    geolocation_disabled: bool = True
    notifications_disabled: bool = True


class AntiDetectionEngine:
    """Engine pro detekci a obcházení anti-bot opatření
    """

    def __init__(self, config: StealthConfig = None):
        self.config = config or StealthConfig()
        self.detection_patterns = self._load_detection_patterns()
        self.browser_fingerprints = self._generate_fingerprints()

    def _load_detection_patterns(self) -> dict[str, list[str]]:
        """Načtení vzorů pro detekci anti-bot opatření"""
        return {
            "captcha": [
                r"captcha",
                r"recaptcha",
                r"hcaptcha",
                r"I'm not a robot",
                r"verify you are human",
                r"security check"
            ],
            "rate_limiting": [
                r"too many requests",
                r"rate limit",
                r"please wait",
                r"slow down",
                r"try again later"
            ],
            "bot_detection": [
                r"blocked",
                r"access denied",
                r"suspicious activity",
                r"automated traffic",
                r"bot detected"
            ],
            "cloudflare": [
                r"cloudflare",
                r"checking your browser",
                r"please wait while we verify",
                r"security service"
            ]
        }

    def _generate_fingerprints(self) -> list[dict[str, Any]]:
        """Generování realistických browser fingerprintů"""
        fingerprints = []

        # Běžné kombinace OS/Browser
        combinations = [
            ("Windows NT 10.0; Win64; x64", "Chrome", "120.0.0.0"),
            ("Macintosh; Intel Mac OS X 10_15_7", "Chrome", "120.0.0.0"),
            ("X11; Linux x86_64", "Chrome", "120.0.0.0"),
            ("Macintosh; Intel Mac OS X 10_15_7", "Safari", "17.0"),
            ("Windows NT 10.0; Win64; x64", "Firefox", "121.0"),
        ]

        for os_info, browser, version in combinations:
            fingerprint = {
                "user_agent": self._build_user_agent(os_info, browser, version),
                "viewport": random.choice(["1920x1080", "1366x768", "1440x900", "2560x1440"]),
                "platform": os_info.split(";")[0] if ";" in os_info else os_info,
                "language": random.choice(["en-US", "en-GB", "cs-CZ", "de-DE"]),
                "timezone": random.choice(["America/New_York", "Europe/London", "Europe/Prague"]),
                "webgl_vendor": "Google Inc. (Intel)" if "Intel" in os_info else "Google Inc. (AMD)",
                "webgl_renderer": "ANGLE (Intel, Intel Iris Pro Graphics 6200 Direct3D11 vs_5_0 ps_5_0, D3D11)"
            }
            fingerprints.append(fingerprint)

        return fingerprints

    def _build_user_agent(self, os_info: str, browser: str, version: str) -> str:
        """Sestavení User-Agent stringu"""
        if browser == "Chrome":
            return f"Mozilla/5.0 ({os_info}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} Safari/537.36"
        if browser == "Firefox":
            return f"Mozilla/5.0 ({os_info}; rv:109.0) Gecko/20100101 Firefox/{version}"
        if browser == "Safari":
            return f"Mozilla/5.0 ({os_info}) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/{version} Safari/605.1.15"

        return f"Mozilla/5.0 ({os_info}) AppleWebKit/537.36"

    def detect_anti_bot_measures(self, content: str, url: str) -> dict[str, Any]:
        """Detekce anti-bot opatření na stránce

        Args:
            content: HTML obsah stránky
            url: URL stránky

        Returns:
            Slovník s detekcí anti-bot opatření

        """
        detection_results = {
            "captcha_detected": False,
            "rate_limiting": False,
            "bot_detection": False,
            "cloudflare_challenge": False,
            "javascript_challenge": False,
            "detected_patterns": [],
            "risk_level": "low"
        }

        content_lower = content.lower()

        # Kontrola jednotlivých typů detekce
        for detection_type, patterns in self.detection_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    detection_results[detection_type.replace("_", "_")] = True
                    detection_results["detected_patterns"].append(f"{detection_type}: {pattern}")

        # Specifické kontroly
        if "javascript" in content_lower and "disabled" in content_lower:
            detection_results["javascript_challenge"] = True

        # Výpočet risk level
        detected_count = sum([
            detection_results["captcha_detected"],
            detection_results["rate_limiting"],
            detection_results["bot_detection"],
            detection_results["cloudflare_challenge"]
        ])

        if detected_count >= 2:
            detection_results["risk_level"] = "high"
        elif detected_count == 1:
            detection_results["risk_level"] = "medium"

        return detection_results

    def get_random_fingerprint(self) -> dict[str, Any]:
        """Získání náhodného fingerprinu"""
        return random.choice(self.browser_fingerprints)

    async def simulate_human_behavior(self, page: Page):
        """Simulace lidského chování na stránce

        Args:
            page: Playwright Page objekt

        """
        try:
            # Náhodné čekání
            await asyncio.sleep(random.uniform(0.5, 2.0))

            # Simulace pohybu myši
            await page.mouse.move(
                random.randint(100, 800),
                random.randint(100, 600)
            )

            # Náhodný scroll
            if random.random() < 0.7:  # 70% šance na scroll
                scroll_amount = random.randint(100, 500)
                await page.evaluate(f"window.scrollBy(0, {scroll_amount})")
                await asyncio.sleep(random.uniform(0.5, 1.5))

            # Simulace čtení (delší pauza)
            await asyncio.sleep(random.uniform(1.0, 3.0))

        except Exception as e:
            logger.debug(f"Chyba při simulaci lidského chování: {e}")


class StealthBrowser:
    """Stealth Playwright browser s anti-detection funkcemi
    """

    def __init__(self, config: StealthConfig = None):
        self.config = config or StealthConfig()
        self.anti_detection = AntiDetectionEngine(config)
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None

    async def initialize(self, headless: bool = True):
        """Inicializace stealth browseru"""
        playwright = await async_playwright().start()

        # Konfigurace pro stealth mode
        browser_args = [
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-web-security",
            "--disable-features=VizDisplayCompositor",
        ]

        if self.config.webrtc_disabled:
            browser_args.extend([
                "--disable-webrtc",
                "--disable-webrtc-multiple-routes",
                "--disable-webrtc-hw-decoding",
                "--disable-webrtc-hw-encoding"
            ])

        self.browser = await playwright.chromium.launch(
            headless=headless,
            args=browser_args
        )

        # Vytvoření kontextu s fingerprintem
        fingerprint = self.anti_detection.get_random_fingerprint()

        self.context = await self.browser.new_context(
            user_agent=fingerprint["user_agent"],
            viewport={"width": 1920, "height": 1080},
            locale=fingerprint["language"],
            timezone_id=fingerprint["timezone"],
            geolocation=None if self.config.geolocation_disabled else {"latitude": 50.0755, "longitude": 14.4378},
            permissions=[] if self.config.notifications_disabled else ["notifications"],
            extra_http_headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": f"{fingerprint['language']},en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            }
        )

        # Injektování stealth skriptů
        await self._inject_stealth_scripts()

        logger.info("Stealth browser inicializován")

    async def _inject_stealth_scripts(self):
        """Injektování stealth skriptů pro obcházení detekce"""
        stealth_scripts = [
            # Odstranění webdriver property
            """
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
            """,

            # Modifikace plugins
            """
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });
            """,

            # Modifikace languages
            """
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
            });
            """,

            # Chrome detection obejití
            """
            if (navigator.userAgent.indexOf('Chrome') > -1) {
                Object.defineProperty(window, 'chrome', {
                    value: {
                        runtime: {},
                        loadTimes: function() {},
                        csi: function() {},
                        app: {}
                    }
                });
            }
            """,

            # Permission API mock
            """
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                Promise.resolve({ state: Notification.permission }) :
                originalQuery(parameters)
            );
            """
        ]

        for script in stealth_scripts:
            await self.context.add_init_script(script)

    async def create_page(self) -> Page:
        """Vytvoření nové stealth stránky"""
        if not self.context:
            await self.initialize()

        page = await self.context.new_page()

        # Blokování zbytečných requestů pro rychlost
        await page.route("**/*.{png,jpg,jpeg,gif,svg,ico,woff,woff2}", lambda route: route.abort())

        return page

    async def safe_navigate(self, page: Page, url: str) -> dict[str, Any]:
        """Bezpečná navigace s detekcí anti-bot opatření

        Args:
            page: Playwright Page objekt
            url: Cílová URL

        Returns:
            Výsledek navigace s detekcí

        """
        result = {
            "success": False,
            "final_url": url,
            "content": "",
            "anti_bot_detected": False,
            "detection_results": {},
            "error": None
        }

        try:
            # Náhodné čekání před navigací
            if self.config.timing_randomization:
                await asyncio.sleep(random.uniform(self.config.min_delay, self.config.max_delay))

            # Navigace na stránku
            response = await page.goto(url, wait_until="domcontentloaded", timeout=30000)

            if response:
                result["final_url"] = response.url

                # Simulace lidského chování
                await self.anti_detection.simulate_human_behavior(page)

                # Získání obsahu
                result["content"] = await page.content()

                # Detekce anti-bot opatření
                detection_results = self.anti_detection.detect_anti_bot_measures(
                    result["content"],
                    result["final_url"]
                )

                result["detection_results"] = detection_results
                result["anti_bot_detected"] = detection_results["risk_level"] in ["medium", "high"]

                if not result["anti_bot_detected"]:
                    result["success"] = True
                else:
                    logger.warning(f"Anti-bot opatření detekováno na {url}: {detection_results['detected_patterns']}")

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Chyba při navigaci na {url}: {e}")

        return result

    async def close(self):
        """Uzavření browseru"""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()


class StealthEngine:
    """Hlavní stealth engine pro anonymní web scraping
    """

    def __init__(self, config: StealthConfig = None):
        self.config = config or StealthConfig()
        self.browser_manager = StealthBrowser(config)
        self.session_stats = {
            "requests_made": 0,
            "successful_requests": 0,
            "anti_bot_encounters": 0,
            "start_time": time.time()
        }

    async def scrape_url(self, url: str, use_browser: bool = True) -> dict[str, Any]:
        """Hlavní metoda pro scraping URL s anti-detection

        Args:
            url: URL k scrapingu
            use_browser: Zda použít browser nebo HTTP session

        Returns:
            Výsledek scrapingu

        """
        self.session_stats["requests_made"] += 1

        if use_browser:
            return await self._scrape_with_browser(url)
        return await self._scrape_with_http(url)

    async def _scrape_with_browser(self, url: str) -> dict[str, Any]:
        """Scraping pomocí Playwright browseru"""
        page = None

        try:
            page = await self.browser_manager.create_page()
            result = await self.browser_manager.safe_navigate(page, url)

            if result["success"]:
                self.session_stats["successful_requests"] += 1

            if result["anti_bot_detected"]:
                self.session_stats["anti_bot_encounters"] += 1

            return result

        except Exception as e:
            logger.error(f"Chyba při browser scrapingu {url}: {e}")
            return {
                "success": False,
                "final_url": url,
                "content": "",
                "error": str(e)
            }
        finally:
            if page:
                await page.close()

    async def _scrape_with_http(self, url: str) -> dict[str, Any]:
        """Scraping pomocí HTTP session"""
        from .intelligent_proxy_manager import intelligent_proxy_manager

        try:
            session = await intelligent_proxy_manager.create_anonymous_session(rotate=True)

            # Náhodné čekání
            if self.config.timing_randomization:
                await asyncio.sleep(random.uniform(self.config.min_delay, self.config.max_delay))

            async with session.get(url) as response:
                content = await response.text()

                result = {
                    "success": response.status == 200,
                    "final_url": str(response.url),
                    "content": content,
                    "status_code": response.status
                }

                if result["success"]:
                    self.session_stats["successful_requests"] += 1

                    # Detekce anti-bot opatření i v HTTP módu
                    detection_results = self.browser_manager.anti_detection.detect_anti_bot_measures(
                        content, str(response.url)
                    )

                    result["detection_results"] = detection_results
                    result["anti_bot_detected"] = detection_results["risk_level"] in ["medium", "high"]

                    if result["anti_bot_detected"]:
                        self.session_stats["anti_bot_encounters"] += 1

                return result

            await session.close()

        except Exception as e:
            logger.error(f"Chyba při HTTP scrapingu {url}: {e}")
            return {
                "success": False,
                "final_url": url,
                "content": "",
                "error": str(e)
            }

    def get_session_stats(self) -> dict[str, Any]:
        """Získání statistik session"""
        runtime = time.time() - self.session_stats["start_time"]

        return {
            **self.session_stats,
            "runtime_seconds": runtime,
            "success_rate": (self.session_stats["successful_requests"] /
                           max(1, self.session_stats["requests_made"])),
            "anti_bot_rate": (self.session_stats["anti_bot_encounters"] /
                            max(1, self.session_stats["requests_made"]))
        }

    async def cleanup(self):
        """Úklid zdrojů"""
        await self.browser_manager.close()


# Globální instance
stealth_engine = StealthEngine()
