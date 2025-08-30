"""
Pokročilý stealth scraper s Playwright a anti-detection technikami
Implementuje dynamickou rotaci fingerprints a obcházení moderních anti-bot systémů
"""

import asyncio
import random
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
from pathlib import Path

from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from playwright_stealth import stealth_async

from ..core.error_handling import (
    scraping_retry,
    ErrorAggregator,
    timeout_after,
    respect_rate_limit
)
from ..core.config import get_settings

logger = logging.getLogger(__name__)


class BrowserFingerprint:
    """Manages browser fingerprinting and rotation"""
    
    def __init__(self):
        self.user_agents = [
            # Chrome on macOS
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            # Firefox on macOS
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0",
            # Safari on macOS
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
            # Chrome on Windows
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        ]
        
        self.screen_resolutions = [
            {"width": 1920, "height": 1080},
            {"width": 1366, "height": 768},
            {"width": 1440, "height": 900},
            {"width": 1536, "height": 864},
            {"width": 1280, "height": 720},
        ]
        
        self.languages = [
            ["en-US", "en"],
            ["en-GB", "en"],
            ["cs-CZ", "cs", "en"],
            ["de-DE", "de", "en"],
            ["fr-FR", "fr", "en"],
        ]
        
        self.timezones = [
            "America/New_York",
            "Europe/London", 
            "Europe/Prague",
            "Europe/Berlin",
            "America/Los_Angeles",
        ]
    
    def generate_fingerprint(self) -> Dict[str, Any]:
        """Generate a random but consistent browser fingerprint"""
        resolution = random.choice(self.screen_resolutions)
        
        return {
            "user_agent": random.choice(self.user_agents),
            "viewport": resolution,
            "screen": resolution,
            "languages": random.choice(self.languages),
            "timezone": random.choice(self.timezones),
            "webgl_vendor": random.choice([
                "Intel Inc.",
                "NVIDIA Corporation", 
                "AMD",
                "Apple Inc."
            ]),
            "platform": random.choice([
                "MacIntel",
                "Win32",
                "Linux x86_64"
            ]),
            "hardware_concurrency": random.choice([4, 8, 12, 16]),
            "device_memory": random.choice([4, 8, 16, 32]),
        }


class CloudflareBypass:
    """Specialized techniques for bypassing Cloudflare protection"""
    
    @staticmethod
    async def wait_for_cloudflare_challenge(page: Page, timeout: int = 30) -> bool:
        """Wait for Cloudflare challenge to complete"""
        try:
            # Wait for either success or failure
            await page.wait_for_load_state("networkidle", timeout=timeout * 1000)
            
            # Check for Cloudflare indicators
            cf_ray = await page.locator("[data-ray]").count()
            cf_challenge = await page.locator(".cf-browser-verification").count()
            
            if cf_ray > 0 or cf_challenge > 0:
                logger.info("Cloudflare challenge detected, waiting...")
                
                # Wait for challenge completion
                await page.wait_for_function(
                    "() => !document.querySelector('.cf-browser-verification')",
                    timeout=timeout * 1000
                )
                
                # Additional wait for page to stabilize
                await asyncio.sleep(2)
                return True
                
            return False
            
        except Exception as e:
            logger.warning(f"Cloudflare bypass error: {e}")
            return False
    
    @staticmethod
    async def bypass_rate_limiting(page: Page) -> bool:
        """Handle Cloudflare rate limiting"""
        try:
            # Check for rate limit page
            title = await page.title()
            if "rate limited" in title.lower() or "429" in title:
                logger.warning("Rate limiting detected")
                
                # Wait for retry-after header or default delay
                retry_after = await page.evaluate("""
                    () => {
                        const retryHeader = document.querySelector('meta[http-equiv="refresh"]');
                        if (retryHeader) {
                            const content = retryHeader.getAttribute('content');
                            const match = content.match(/\\d+/);
                            return match ? parseInt(match[0]) : 60;
                        }
                        return 60;
                    }
                """)
                
                logger.info(f"Waiting {retry_after} seconds for rate limit...")
                await asyncio.sleep(min(retry_after, 300))  # Max 5 minutes
                return True
                
            return False
            
        except Exception as e:
            logger.warning(f"Rate limiting bypass error: {e}")
            return False


class StealthScraper:
    """Advanced stealth scraper with anti-detection capabilities"""
    
    def __init__(self, 
                 headless: bool = True, 
                 proxy: Optional[str] = None,
                 max_concurrent: int = 3):
        self.headless = headless
        self.proxy = proxy
        self.max_concurrent = max_concurrent
        self.browser = None
        self.contexts = []
        self.fingerprint_generator = BrowserFingerprint()
        self.cloudflare_bypass = CloudflareBypass()
        self.error_aggregator = ErrorAggregator()
        
        # Session management
        self.session_rotation_interval = timedelta(minutes=30)
        self.last_session_rotation = None
        
        # Request tracking for rate limiting
        self.request_history = []
        self.max_requests_per_minute = 30
    
    async def start(self) -> None:
        """Initialize the browser and contexts"""
        playwright = await async_playwright().start()
        
        # Launch browser with stealth settings
        launch_options = {
            "headless": self.headless,
            "args": [
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-features=VizDisplayCompositor",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                "--disable-dev-shm-usage",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-default-apps",
                "--disable-extensions",
            ]
        }
        
        if self.proxy:
            launch_options["proxy"] = {"server": self.proxy}
        
        self.browser = await playwright.chromium.launch(**launch_options)
        
        # Create initial contexts
        await self._create_stealth_contexts()
        
        logger.info(f"Stealth scraper started with {len(self.contexts)} contexts")
    
    async def _create_stealth_contexts(self) -> None:
        """Create multiple browser contexts with different fingerprints"""
        for i in range(self.max_concurrent):
            fingerprint = self.fingerprint_generator.generate_fingerprint()
            
            context = await self.browser.new_context(
                user_agent=fingerprint["user_agent"],
                viewport=fingerprint["viewport"],
                screen=fingerprint["screen"],
                locale=fingerprint["languages"][0],
                timezone_id=fingerprint["timezone"],
                extra_http_headers={
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": ",".join(fingerprint["languages"]),
                    "Accept-Encoding": "gzip, deflate, br",
                    "DNT": "1",
                    "Upgrade-Insecure-Requests": "1",
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Site": "none",
                }
            )
            
            # Apply stealth to all pages in context
            await context.add_init_script("""
                // Override navigator properties
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
                
                // Override permissions
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Compat.permission }) :
                        originalQuery(parameters)
                );
                
                // Override plugins
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5],
                });
                
                // Override languages
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['""" + fingerprint["languages"][0] + """'],
                });
            """)
            
            self.contexts.append(context)
    
    async def _get_available_context(self) -> BrowserContext:
        """Get an available browser context for scraping"""
        # Simple round-robin for now
        context = self.contexts[len(self.request_history) % len(self.contexts)]
        
        # Check if session rotation is needed
        if (self.last_session_rotation is None or 
            datetime.now() - self.last_session_rotation > self.session_rotation_interval):
            await self._rotate_session(context)
        
        return context
    
    async def _rotate_session(self, context: BrowserContext) -> None:
        """Rotate session to avoid tracking"""
        try:
            # Clear cookies and storage
            await context.clear_cookies()
            await context.clear_permissions()
            
            # Close and recreate pages to reset state
            for page in context.pages:
                await page.close()
            
            self.last_session_rotation = datetime.now()
            logger.debug("Session rotated successfully")
            
        except Exception as e:
            logger.warning(f"Session rotation failed: {e}")
    
    async def _apply_rate_limiting(self) -> None:
        """Apply intelligent rate limiting"""
        now = datetime.now()
        
        # Clean old requests
        self.request_history = [
            req_time for req_time in self.request_history 
            if now - req_time < timedelta(minutes=1)
        ]
        
        # Check rate limit
        if len(self.request_history) >= self.max_requests_per_minute:
            wait_time = 60 - (now - self.request_history[0]).total_seconds()
            if wait_time > 0:
                logger.info(f"Rate limiting: waiting {wait_time:.1f} seconds")
                await asyncio.sleep(wait_time)
        
        # Add current request
        self.request_history.append(now)
        
        # Random delay to appear more human
        await asyncio.sleep(random.uniform(1, 3))
    
    @timeout_after(60)  # 1 minute timeout per page
    @scraping_retry
    async def scrape_page(self, url: str, wait_for_selector: Optional[str] = None) -> Dict[str, Any]:
        """Scrape a single page with full stealth protection"""
        try:
            await self._apply_rate_limiting()
            
            context = await self._get_available_context()
            page = await context.new_page()
            
            # Apply stealth to the page
            await stealth_async(page)
            
            # Navigate to page
            response = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            
            if not response:
                raise Exception(f"Failed to navigate to {url}")
            
            # Handle Cloudflare challenges
            if response.status in [403, 503, 429]:
                await self.cloudflare_bypass.wait_for_cloudflare_challenge(page)
                await self.cloudflare_bypass.bypass_rate_limiting(page)
            
            # Wait for specific selector if provided
            if wait_for_selector:
                await page.wait_for_selector(wait_for_selector, timeout=10000)
            else:
                await page.wait_for_load_state("networkidle", timeout=10000)
            
            # Extract page data
            result = await self._extract_page_data(page, url)
            
            await page.close()
            self.error_aggregator.add_success()
            
            return result
            
        except Exception as e:
            self.error_aggregator.add_error(e, f"scraping {url}")
            logger.error(f"Failed to scrape {url}: {e}")
            raise
    
    async def _extract_page_data(self, page: Page, url: str) -> Dict[str, Any]:
        """Extract comprehensive data from the page"""
        try:
            # Basic page info
            title = await page.title()
            content = await page.content()
            
            # Meta information
            meta_data = await page.evaluate("""
                () => {
                    const metas = {};
                    document.querySelectorAll('meta').forEach(meta => {
                        const name = meta.getAttribute('name') || meta.getAttribute('property');
                        const content = meta.getAttribute('content');
                        if (name && content) {
                            metas[name] = content;
                        }
                    });
                    return metas;
                }
            """)
            
            # Links
            links = await page.evaluate("""
                () => Array.from(document.querySelectorAll('a[href]')).map(a => ({
                    text: a.textContent.trim(),
                    href: a.href,
                    title: a.title || ''
                })).filter(link => link.text && link.href)
            """)
            
            # Images
            images = await page.evaluate("""
                () => Array.from(document.querySelectorAll('img[src]')).map(img => ({
                    src: img.src,
                    alt: img.alt || '',
                    title: img.title || ''
                }))
            """)
            
            # Text content (cleaned)
            text_content = await page.evaluate("""
                () => {
                    // Remove script and style elements
                    const scripts = document.querySelectorAll('script, style, nav, footer, header');
                    scripts.forEach(el => el.remove());
                    
                    // Get text content
                    return document.body.innerText || document.body.textContent || '';
                }
            """)
            
            return {
                "url": url,
                "title": title,
                "meta_data": meta_data,
                "text_content": text_content,
                "links": links[:50],  # Limit to first 50 links
                "images": images[:20],  # Limit to first 20 images
                "html_content": content,
                "scraped_at": datetime.now().isoformat(),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Failed to extract data from {url}: {e}")
            return {
                "url": url,
                "error": str(e),
                "scraped_at": datetime.now().isoformat(),
                "success": False
            }
    
    async def scrape_multiple(self, urls: List[str], max_concurrent: int = None) -> List[Dict[str, Any]]:
        """Scrape multiple URLs with concurrency control"""
        if max_concurrent is None:
            max_concurrent = min(self.max_concurrent, len(urls))
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_with_semaphore(url: str) -> Dict[str, Any]:
            async with semaphore:
                return await self.scrape_page(url)
        
        # Execute scraping tasks
        tasks = [scrape_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_results = []
        for result in results:
            if isinstance(result, Exception):
                self.error_aggregator.add_error(result, "batch scraping")
                logger.error(f"Batch scraping error: {result}")
            else:
                successful_results.append(result)
        
        self.error_aggregator.log_summary()
        return successful_results
    
    async def close(self) -> None:
        """Clean up browser resources"""
        try:
            if self.contexts:
                for context in self.contexts:
                    await context.close()
                self.contexts.clear()
            
            if self.browser:
                await self.browser.close()
                self.browser = None
                
            logger.info("Stealth scraper closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing stealth scraper: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scraping statistics"""
        error_summary = self.error_aggregator.get_summary()
        
        return {
            "contexts_active": len(self.contexts),
            "headless_mode": self.headless,
            "proxy_enabled": bool(self.proxy),
            "max_concurrent": self.max_concurrent,
            "requests_last_minute": len(self.request_history),
            "success_rate": error_summary["success_rate"],
            "total_requests": error_summary["total_operations"],
            "failed_requests": error_summary["failed_operations"],
            "last_session_rotation": self.last_session_rotation.isoformat() if self.last_session_rotation else None
        }


# Utility functions for easy usage
async def stealth_scrape_url(url: str, proxy: Optional[str] = None) -> Dict[str, Any]:
    """Quick utility for single URL scraping"""
    scraper = StealthScraper(proxy=proxy)
    try:
        await scraper.start()
        return await scraper.scrape_page(url)
    finally:
        await scraper.close()


async def stealth_scrape_urls(urls: List[str], max_concurrent: int = 3, proxy: Optional[str] = None) -> List[Dict[str, Any]]:
    """Quick utility for multiple URL scraping"""
    scraper = StealthScraper(max_concurrent=max_concurrent, proxy=proxy)
    try:
        await scraper.start()
        return await scraper.scrape_multiple(urls)
    finally:
        await scraper.close()