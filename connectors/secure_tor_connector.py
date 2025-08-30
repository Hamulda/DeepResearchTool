"""
Secure Tor Connector
Bezpečný konektor s leak-testem a WebRTC ochranou pro Playwright

Author: Senior Python/MLOps Agent
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import aiohttp
from playwright.async_api import async_playwright, Browser, BrowserContext, Page

logger = logging.getLogger(__name__)


@dataclass
class LeakTestResult:
    """Výsledek leak testu"""
    test_passed: bool
    real_ip: Optional[str] = None
    proxy_ip: Optional[str] = None
    dns_leak_detected: bool = False
    webrtc_leak_detected: bool = False
    test_timestamp: float = 0.0
    error_message: Optional[str] = None


@dataclass
class TorConnectionConfig:
    """Konfigurace Tor připojení"""
    socks_proxy: str = "socks5://127.0.0.1:9050"
    control_port: int = 9051
    timeout: int = 30
    max_retries: int = 3
    enable_leak_test: bool = True
    block_webrtc: bool = True
    enable_javascript: bool = False
    user_data_dir: Optional[str] = None


class SecureTorConnector:
    """
    Bezpečný Tor konektor s pokročilou ochranou proti únikům
    Implementuje leak detection a WebRTC blokování
    """

    def __init__(self, config: TorConnectionConfig):
        self.config = config
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.playwright = None

        # Leak detection URLs
        self.leak_test_urls = {
            "ip_check": "https://httpbin.org/ip",
            "dns_check": "https://1.1.1.1/cdn-cgi/trace",
            "webrtc_check": "https://browserleaks.com/webrtc"
        }

        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Inicializace Tor konektoru"""
        self.logger.info("🔐 Inicializuji Secure Tor Connector...")

        # Spuštění Playwright
        self.playwright = await async_playwright().start()

        # Konfigurace browser args pro bezpečnost
        browser_args = self._get_secure_browser_args()

        # Spuštění prohlížeče s proxy
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=browser_args,
            proxy={
                "server": self.config.socks_proxy
            }
        )

        # Vytvoření bezpečného kontextu
        await self._create_secure_context()

        # Provedení leak testu
        if self.config.enable_leak_test:
            leak_result = await self.perform_comprehensive_leak_test()
            if not leak_result.test_passed:
                raise ConnectionError(f"Leak test failed: {leak_result.error_message}")

        self.logger.info("✅ Secure Tor Connector inicializován")

    def _get_secure_browser_args(self) -> List[str]:
        """Vrátí bezpečné argumenty pro browser"""

        args = [
            # Základní bezpečnostní argumenty
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",

            # WebRTC blokování
            "--disable-webrtc",
            "--disable-webrtc-multiple-routes",
            "--disable-webrtc-hw-decoding",
            "--disable-webrtc-hw-encoding",
            "--enforce-webrtc-ip-permission-check",

            # DNS a leak prevence
            "--disable-background-networking",
            "--disable-default-apps",
            "--disable-extensions",
            "--disable-sync",
            "--disable-translate",

            # Fingerprinting prevence
            "--disable-plugins",
            "--disable-plugins-discovery",
            "--disable-component-extensions-with-background-pages",

            # Tor-specific
            f"--proxy-server={self.config.socks_proxy}",
            "--host-resolver-rules='MAP * ~NOTFOUND , EXCLUDE 127.0.0.1'",
        ]

        return args

    async def _create_secure_context(self):
        """Vytvoří bezpečný browser kontext"""

        # Konfigurace kontextu
        context_options = {
            "viewport": {"width": 1366, "height": 768},  # Běžná rozlišení
            "user_agent": self._get_secure_user_agent(),
            "java_script_enabled": self.config.enable_javascript,
            "accept_downloads": False,
            "ignore_https_errors": True,
            "permissions": [],  # Žádná povolení
        }

        # Přidání user data dir pokud je specifikován
        if self.config.user_data_dir:
            context_options["storage_state"] = self.config.user_data_dir

        self.context = await self.browser.new_context(**context_options)

        # Blokování WebRTC na úrovni kontextu
        if self.config.block_webrtc:
            await self._block_webrtc_in_context()

        # Blokování trackingů a reklam
        await self._setup_content_blocking()

    def _get_secure_user_agent(self) -> str:
        """Vrátí bezpečný User Agent pro Tor"""
        # Tor Browser bundle User Agent
        return "Mozilla/5.0 (Windows NT 10.0; rv:102.0) Gecko/20100101 Firefox/102.0"

    async def _block_webrtc_in_context(self):
        """Blokuje WebRTC na úrovni kontextu"""

        # Injektování skriptu pro blokování WebRTC
        webrtc_block_script = """
        // Block WebRTC
        Object.defineProperty(navigator, 'getUserMedia', {
            value: undefined,
            writable: false
        });
        
        Object.defineProperty(navigator, 'webkitGetUserMedia', {
            value: undefined,
            writable: false
        });
        
        Object.defineProperty(navigator, 'mozGetUserMedia', {
            value: undefined,
            writable: false
        });
        
        Object.defineProperty(window, 'RTCPeerConnection', {
            value: undefined,
            writable: false
        });
        
        Object.defineProperty(window, 'webkitRTCPeerConnection', {
            value: undefined,
            writable: false
        });
        
        Object.defineProperty(window, 'mozRTCPeerConnection', {
            value: undefined,
            writable: false
        });
        
        // Block MediaDevices API
        Object.defineProperty(navigator, 'mediaDevices', {
            value: undefined,
            writable: false
        });
        """

        await self.context.add_init_script(webrtc_block_script)

    async def _setup_content_blocking(self):
        """Nastavení blokování obsahu"""

        # Blokování známých tracking domén
        blocked_domains = [
            "*.doubleclick.net",
            "*.googlesyndication.com",
            "*.google-analytics.com",
            "*.googletagmanager.com",
            "*.facebook.com",
            "*.twitter.com",
            "*.linkedin.com",
            "*.instagram.com",
            "*.tiktok.com",
            "*.youtube.com"
        ]

        # Route handler pro blokování
        async def block_handler(route):
            url = route.request.url
            domain = route.request.url.split('/')[2] if '/' in route.request.url else ''

            # Kontrola blokovaných domén
            for blocked in blocked_domains:
                if blocked.replace('*.', '') in domain:
                    await route.abort()
                    return

            # Blokování podle typu obsahu
            resource_type = route.request.resource_type
            if resource_type in ['image', 'media', 'font', 'stylesheet']:
                await route.abort()
                return

            await route.continue_()

        await self.context.route("**/*", block_handler)

    async def perform_comprehensive_leak_test(self) -> LeakTestResult:
        """Provede komprehenzivní leak test"""

        self.logger.info("🔍 Spouštím comprehensive leak test...")

        result = LeakTestResult(
            test_passed=False,
            test_timestamp=time.time()
        )

        try:
            page = await self.context.new_page()

            # Test 1: IP leak test
            ip_result = await self._test_ip_leak(page)
            result.real_ip = ip_result.get("real_ip")
            result.proxy_ip = ip_result.get("proxy_ip")

            if result.real_ip and result.proxy_ip:
                if result.real_ip == result.proxy_ip:
                    result.error_message = "IP leak detected - real IP equals proxy IP"
                    return result

            # Test 2: DNS leak test
            dns_leak = await self._test_dns_leak(page)
            result.dns_leak_detected = dns_leak

            if dns_leak:
                result.error_message = "DNS leak detected"
                return result

            # Test 3: WebRTC leak test
            webrtc_leak = await self._test_webrtc_leak(page)
            result.webrtc_leak_detected = webrtc_leak

            if webrtc_leak:
                result.error_message = "WebRTC leak detected"
                return result

            # Všechny testy prošly
            result.test_passed = True

            await page.close()

        except Exception as e:
            result.error_message = f"Leak test failed: {str(e)}"

        return result

    async def _test_ip_leak(self, page: Page) -> Dict[str, Optional[str]]:
        """Test úniku IP adresy"""

        result = {"real_ip": None, "proxy_ip": None}

        try:
            # Test bez Tor (pro získání real IP)
            async with aiohttp.ClientSession() as session:
                async with session.get(self.leak_test_urls["ip_check"], timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        result["real_ip"] = data.get("origin")

            # Test přes Tor
            await page.goto(self.leak_test_urls["ip_check"], timeout=30000)

            # Čekání na načtení a extrakce IP
            await page.wait_for_load_state("networkidle")
            content = await page.content()

            # Parsování JSON odpovědi
            import re
            ip_match = re.search(r'"origin":\s*"([^"]+)"', content)
            if ip_match:
                result["proxy_ip"] = ip_match.group(1)

        except Exception as e:
            self.logger.warning(f"IP leak test failed: {e}")

        return result

    async def _test_dns_leak(self, page: Page) -> bool:
        """Test úniku DNS"""

        try:
            await page.goto(self.leak_test_urls["dns_check"], timeout=30000)
            await page.wait_for_load_state("networkidle")

            content = await page.content()

            # Kontrola na DNS leak indikátory
            dns_leak_indicators = [
                "your real location",
                "dns leak",
                "location mismatch"
            ]

            content_lower = content.lower()
            for indicator in dns_leak_indicators:
                if indicator in content_lower:
                    return True

            return False

        except Exception as e:
            self.logger.warning(f"DNS leak test failed: {e}")
            return False

    async def _test_webrtc_leak(self, page: Page) -> bool:
        """Test úniku WebRTC"""

        try:
            # Injektování testu WebRTC
            webrtc_test_script = """
            new Promise((resolve) => {
                let rtcLeak = false;
                
                // Test RTCPeerConnection
                if (typeof RTCPeerConnection !== 'undefined') {
                    rtcLeak = true;
                }
                
                // Test getUserMedia
                if (navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia) {
                    rtcLeak = true;
                }
                
                // Test MediaDevices
                if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                    rtcLeak = true;
                }
                
                resolve(rtcLeak);
            });
            """

            result = await page.evaluate(webrtc_test_script)
            return bool(result)

        except Exception as e:
            self.logger.warning(f"WebRTC leak test failed: {e}")
            return False

    async def create_secure_page(self) -> Page:
        """Vytvoří zabezpečenou stránku"""

        if not self.context:
            raise RuntimeError("Connector not initialized")

        page = await self.context.new_page()

        # Dodatečné security headers
        await page.set_extra_http_headers({
            "DNT": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1"
        })

        return page

    async def safe_navigate(self, page: Page, url: str) -> bool:
        """Bezpečná navigace s leak kontrolou"""

        try:
            # Pre-navigation leak test
            if self.config.enable_leak_test:
                leak_result = await self.perform_comprehensive_leak_test()
                if not leak_result.test_passed:
                    self.logger.error(f"Pre-navigation leak detected: {leak_result.error_message}")
                    return False

            # Navigace
            response = await page.goto(url, timeout=self.config.timeout * 1000)

            if not response or response.status >= 400:
                self.logger.warning(f"Navigation failed with status: {response.status if response else 'None'}")
                return False

            # Post-navigation leak test (rychlý)
            if self.config.enable_leak_test:
                webrtc_leak = await self._test_webrtc_leak(page)
                if webrtc_leak:
                    self.logger.error("Post-navigation WebRTC leak detected")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Safe navigation failed: {e}")
            return False

    async def extract_content(self, page: Page, selectors: List[str] = None) -> Dict[str, Any]:
        """Extrakce obsahu ze stránky"""

        content = {
            "url": page.url,
            "title": await page.title(),
            "text_content": "",
            "html_content": "",
            "metadata": {},
            "extracted_at": time.time()
        }

        try:
            # Čekání na načtení
            await page.wait_for_load_state("networkidle", timeout=10000)

            # Základní obsah
            content["text_content"] = await page.inner_text("body")
            content["html_content"] = await page.content()

            # Custom selektory
            if selectors:
                extracted_elements = {}
                for selector in selectors:
                    try:
                        elements = await page.query_selector_all(selector)
                        extracted_elements[selector] = []
                        for element in elements:
                            text = await element.inner_text()
                            extracted_elements[selector].append(text)
                    except Exception as e:
                        self.logger.debug(f"Selector {selector} failed: {e}")

                content["extracted_elements"] = extracted_elements

            # Metadata
            content["metadata"] = {
                "viewport": await page.viewport_size(),
                "user_agent": await page.evaluate("navigator.userAgent"),
                "content_length": len(content["html_content"]),
                "load_time": time.time() - content["extracted_at"]
            }

        except Exception as e:
            self.logger.error(f"Content extraction failed: {e}")
            content["error"] = str(e)

        return content

    async def close(self):
        """Zavření konektoru"""

        try:
            if self.context:
                await self.context.close()

            if self.browser:
                await self.browser.close()

            if self.playwright:
                await self.playwright.stop()

            self.logger.info("🔒 Secure Tor Connector uzavřen")

        except Exception as e:
            self.logger.error(f"Error closing connector: {e}")

    def get_connection_stats(self) -> Dict[str, Any]:
        """Vrátí statistiky připojení"""

        return {
            "proxy_config": self.config.socks_proxy,
            "leak_testing_enabled": self.config.enable_leak_test,
            "webrtc_blocked": self.config.block_webrtc,
            "javascript_enabled": self.config.enable_javascript,
            "browser_active": self.browser is not None,
            "context_active": self.context is not None
        }


# Test funkce
async def test_secure_tor_connector():
    """Test funkce pro Secure Tor Connector"""

    config = TorConnectionConfig(
        socks_proxy="socks5://127.0.0.1:9050",
        enable_leak_test=True,
        block_webrtc=True,
        enable_javascript=False
    )

    connector = SecureTorConnector(config)

    try:
        await connector.initialize()

        # Test leak detection
        leak_result = await connector.perform_comprehensive_leak_test()
        print(f"🔍 Leak test result: {leak_result}")

        # Test secure browsing
        page = await connector.create_secure_page()

        success = await connector.safe_navigate(page, "https://check.torproject.org")
        if success:
            content = await connector.extract_content(page)
            print(f"✅ Secure navigation successful")
            print(f"📄 Page title: {content['title']}")
            print(f"📊 Content length: {content['metadata']['content_length']} chars")
        else:
            print("❌ Secure navigation failed")

        await page.close()

    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        await connector.close()


if __name__ == "__main__":
    asyncio.run(test_secure_tor_connector())
