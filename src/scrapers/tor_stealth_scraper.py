"""
Tor integrace pro přístup k deep web obsahu
Implementuje Playwright s Tor proxy a programové řízení Tor identity
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import time
from datetime import datetime

from playwright.async_api import Browser, BrowserContext, Page
import stem
from stem import Signal
from stem.control import Controller

from .stealth_scraper import StealthScraper, BrowserFingerprint
from ..core.error_handling import (
    network_retry,
    tor_circuit_breaker,
    with_circuit_breaker,
    ErrorAggregator,
    timeout_after
)
from ..core.config import get_settings

logger = logging.getLogger(__name__)


class TorController:
    """Manages Tor connection and identity rotation"""
    
    def __init__(self, 
                 socks_port: int = 9050, 
                 control_port: int = 9051,
                 control_password: str = ""):
        self.socks_port = socks_port
        self.control_port = control_port
        self.control_password = control_password
        self.controller = None
        self.is_connected = False
        
        # Circuit management
        self.circuit_rotation_interval = 300  # 5 minutes
        self.last_circuit_rotation = None
        self.circuit_build_timeout = 60
        
        # Statistics
        self.circuits_created = 0
        self.identity_changes = 0
        
    async def connect(self) -> bool:
        """Connect to Tor control port"""
        try:
            self.controller = Controller.from_port(port=self.control_port)
            self.controller.authenticate(password=self.control_password)
            self.is_connected = True
            
            logger.info(f"Connected to Tor control port {self.control_port}")
            
            # Get initial Tor status
            status = await self.get_tor_status()
            logger.info(f"Tor status: {status}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Tor: {e}")
            self.is_connected = False
            return False
    
    async def get_tor_status(self) -> Dict[str, Any]:
        """Get comprehensive Tor status information"""
        if not self.controller:
            return {"connected": False, "error": "No controller"}
        
        try:
            info = self.controller.get_info_map([
                "status/circuit-established",
                "status/bootstrap/summary",
                "traffic/read",
                "traffic/written",
                "version",
                "address",
            ])
            
            return {
                "connected": True,
                "circuit_established": info.get("status/circuit-established") == "1",
                "bootstrap_summary": info.get("status/bootstrap/summary", ""),
                "traffic_read": int(info.get("traffic/read", 0)),
                "traffic_written": int(info.get("traffic/written", 0)),
                "tor_version": info.get("version", ""),
                "current_ip": info.get("address", "unknown"),
                "circuits_created": self.circuits_created,
                "identity_changes": self.identity_changes,
            }
            
        except Exception as e:
            logger.error(f"Failed to get Tor status: {e}")
            return {"connected": False, "error": str(e)}
    
    @network_retry
    async def new_identity(self) -> bool:
        """Request new Tor identity (new circuit)"""
        if not self.controller:
            raise Exception("Tor controller not connected")
        
        try:
            logger.info("Requesting new Tor identity...")
            self.controller.signal(Signal.NEWNYM)
            
            # Wait for new circuit to be established
            await asyncio.sleep(10)
            
            # Verify circuit establishment
            start_time = time.time()
            while time.time() - start_time < self.circuit_build_timeout:
                status = await self.get_tor_status()
                if status.get("circuit_established"):
                    self.identity_changes += 1
                    self.last_circuit_rotation = datetime.now()
                    logger.info("New Tor identity established")
                    return True
                await asyncio.sleep(2)
            
            raise Exception("New circuit build timeout")
            
        except Exception as e:
            logger.error(f"Failed to get new identity: {e}")
            raise
    
    async def should_rotate_circuit(self) -> bool:
        """Check if circuit should be rotated"""
        if not self.last_circuit_rotation:
            return True
        
        elapsed = (datetime.now() - self.last_circuit_rotation).total_seconds()
        return elapsed > self.circuit_rotation_interval
    
    async def close(self):
        """Close Tor controller connection"""
        if self.controller:
            self.controller.close()
            self.controller = None
            self.is_connected = False
            logger.info("Tor controller closed")


class TorStealthScraper(StealthScraper):
    """Stealth scraper enhanced with Tor integration"""
    
    def __init__(self, 
                 tor_socks_port: int = 9050,
                 tor_control_port: int = 9051,
                 tor_control_password: str = "",
                 headless: bool = True,
                 max_concurrent: int = 2):  # Lower concurrency for Tor
        
        # Initialize Tor proxy
        proxy_url = f"socks5://127.0.0.1:{tor_socks_port}"
        super().__init__(headless=headless, proxy=proxy_url, max_concurrent=max_concurrent)
        
        # Tor-specific settings
        self.tor_controller = TorController(
            socks_port=tor_socks_port,
            control_port=tor_control_port,
            control_password=tor_control_password
        )
        
        # Circuit management
        self.requests_per_circuit = 10
        self.current_circuit_requests = 0
        
        # .onion specific settings
        self.onion_timeout = 60  # Longer timeout for .onion sites
        self.connection_retry_delay = 5
        
        # Safety settings
        self.blocked_keywords = [
            "illegal", "child", "weapons", "drugs", "hitman", 
            "assassination", "torture", "violence"
        ]
        
    async def start(self) -> None:
        """Initialize Tor controller and browser"""
        # Connect to Tor first
        if not await self.tor_controller.connect():
            raise Exception("Failed to connect to Tor")
        
        # Verify Tor is working
        status = await self.tor_controller.get_tor_status()
        if not status.get("circuit_established"):
            logger.info("Waiting for Tor circuit to establish...")
            await self.tor_controller.new_identity()
        
        # Initialize browser with Tor proxy
        await super().start()
        
        logger.info("Tor stealth scraper initialized successfully")
    
    @with_circuit_breaker(tor_circuit_breaker)
    @timeout_after(120)  # 2 minute timeout for .onion sites
    async def scrape_onion_page(self, onion_url: str, wait_for_selector: Optional[str] = None) -> Dict[str, Any]:
        """Scrape .onion page with enhanced error handling"""
        
        if not onion_url.endswith('.onion') and '.onion' not in onion_url:
            raise ValueError("URL must be a .onion address")
        
        # Safety check
        if await self._is_safe_onion_content(onion_url):
            logger.warning(f"Potentially unsafe .onion content detected: {onion_url}")
            return {
                "url": onion_url,
                "error": "Content filtered for safety",
                "success": False,
                "scraped_at": datetime.now().isoformat()
            }
        
        # Circuit rotation if needed
        if (self.current_circuit_requests >= self.requests_per_circuit or 
            await self.tor_controller.should_rotate_circuit()):
            logger.info("Rotating Tor circuit...")
            await self.tor_controller.new_identity()
            self.current_circuit_requests = 0
            
            # Wait for circuit to stabilize
            await asyncio.sleep(5)
        
        try:
            context = await self._get_available_context()
            page = await context.new_page()
            
            # Enhanced timeout settings for .onion
            page.set_default_timeout(self.onion_timeout * 1000)
            
            # Apply stealth
            from playwright_stealth import stealth_async
            await stealth_async(page)
            
            # Navigate with retries
            response = None
            for attempt in range(3):
                try:
                    response = await page.goto(
                        onion_url, 
                        wait_until="domcontentloaded", 
                        timeout=self.onion_timeout * 1000
                    )
                    break
                except Exception as e:
                    if attempt < 2:
                        logger.warning(f"Retrying .onion connection (attempt {attempt + 1}): {e}")
                        await asyncio.sleep(self.connection_retry_delay)
                        continue
                    raise
            
            if not response:
                raise Exception(f"Failed to connect to {onion_url}")
            
            # Check response status
            if response.status >= 400:
                logger.warning(f".onion site returned {response.status}: {onion_url}")
            
            # Wait for content to load
            if wait_for_selector:
                await page.wait_for_selector(wait_for_selector, timeout=30000)
            else:
                try:
                    await page.wait_for_load_state("networkidle", timeout=30000)
                except:
                    # .onion sites might not reach networkidle, wait for basic load
                    await page.wait_for_load_state("domcontentloaded", timeout=30000)
            
            # Extract data with safety filtering
            result = await self._extract_onion_data(page, onion_url)
            
            await page.close()
            self.current_circuit_requests += 1
            self.error_aggregator.add_success()
            
            return result
            
        except Exception as e:
            self.error_aggregator.add_error(e, f"scraping onion {onion_url}")
            logger.error(f"Failed to scrape .onion {onion_url}: {e}")
            
            return {
                "url": onion_url,
                "error": str(e),
                "success": False,
                "scraped_at": datetime.now().isoformat()
            }
    
    async def _extract_onion_data(self, page: Page, url: str) -> Dict[str, Any]:
        """Extract data from .onion page with safety filtering"""
        try:
            # Basic extraction
            title = await page.title()
            
            # Safety check on title
            if await self._contains_unsafe_content(title):
                return {
                    "url": url,
                    "error": "Content filtered for safety",
                    "success": False,
                    "scraped_at": datetime.now().isoformat()
                }
            
            # Extract text content
            text_content = await page.evaluate("""
                () => {
                    // Remove scripts, styles, and potential tracking elements
                    const elementsToRemove = document.querySelectorAll(
                        'script, style, iframe, embed, object, noscript'
                    );
                    elementsToRemove.forEach(el => el.remove());
                    
                    return document.body.innerText || document.body.textContent || '';
                }
            """)
            
            # Safety filter on content
            if await self._contains_unsafe_content(text_content):
                return {
                    "url": url,
                    "error": "Content filtered for safety",
                    "success": False,
                    "scraped_at": datetime.now().isoformat()
                }
            
            # Extract links (filtered)
            links = await page.evaluate("""
                () => Array.from(document.querySelectorAll('a[href]')).map(a => ({
                    text: a.textContent.trim(),
                    href: a.href,
                })).filter(link => link.text && link.href && link.href.includes('.onion'))
            """)
            
            # Tor-specific metadata
            tor_status = await self.tor_controller.get_tor_status()
            
            return {
                "url": url,
                "title": title,
                "text_content": text_content[:10000],  # Limit content size
                "onion_links": links[:20],  # Limit links
                "tor_metadata": {
                    "circuit_ip": tor_status.get("current_ip"),
                    "tor_version": tor_status.get("tor_version"),
                    "circuit_requests": self.current_circuit_requests
                },
                "scraped_at": datetime.now().isoformat(),
                "success": True,
                "content_type": "onion_service"
            }
            
        except Exception as e:
            logger.error(f"Failed to extract .onion data: {e}")
            return {
                "url": url,
                "error": str(e),
                "success": False,
                "scraped_at": datetime.now().isoformat()
            }
    
    async def _is_safe_onion_content(self, url: str) -> bool:
        """Basic safety check for .onion URLs"""
        url_lower = url.lower()
        return any(keyword in url_lower for keyword in self.blocked_keywords)
    
    async def _contains_unsafe_content(self, content: str) -> bool:
        """Check if content contains unsafe material"""
        if not content:
            return False
        
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in self.blocked_keywords)
    
    async def scrape_multiple_onions(self, onion_urls: List[str]) -> List[Dict[str, Any]]:
        """Scrape multiple .onion URLs with circuit rotation"""
        results = []
        
        for i, url in enumerate(onion_urls):
            try:
                # Rotate circuit every few requests
                if i > 0 and i % 5 == 0:
                    logger.info(f"Rotating circuit after {i} requests...")
                    await self.tor_controller.new_identity()
                    await asyncio.sleep(10)  # Wait for circuit to stabilize
                
                result = await self.scrape_onion_page(url)
                results.append(result)
                
                # Delay between requests for .onion sites
                await asyncio.sleep(random.uniform(3, 8))
                
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {e}")
                results.append({
                    "url": url,
                    "error": str(e),
                    "success": False,
                    "scraped_at": datetime.now().isoformat()
                })
        
        return results
    
    async def search_onion_directories(self, search_term: str) -> List[Dict[str, Any]]:
        """Search .onion directories for research content"""
        # Known .onion directory sites (these would need to be updated with real addresses)
        directory_sites = [
            # These are examples - real implementation would need actual .onion addresses
            # "http://directoryexample123.onion/search?q=",
            # "http://linklistexample456.onion/find?term=",
        ]
        
        results = []
        
        for directory in directory_sites:
            try:
                search_url = f"{directory}{search_term}"
                result = await self.scrape_onion_page(search_url)
                
                if result.get("success"):
                    # Parse search results from directory
                    parsed_links = self._parse_directory_results(result.get("text_content", ""))
                    results.extend(parsed_links)
                
                await asyncio.sleep(5)  # Delay between directory searches
                
            except Exception as e:
                logger.error(f"Failed to search directory {directory}: {e}")
        
        return results
    
    def _parse_directory_results(self, content: str) -> List[Dict[str, Any]]:
        """Parse .onion links from directory content"""
        import re
        
        # Extract .onion URLs from content
        onion_pattern = r'([a-z2-7]{16,56}\.onion)'
        onion_matches = re.findall(onion_pattern, content)
        
        results = []
        for onion_address in onion_matches[:10]:  # Limit results
            results.append({
                "onion_address": f"http://{onion_address}",
                "source": "directory_search",
                "found_at": datetime.now().isoformat()
            })
        
        return results
    
    async def close(self) -> None:
        """Close Tor controller and browser"""
        await super().close()
        await self.tor_controller.close()
        logger.info("Tor stealth scraper closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive Tor scraping statistics"""
        base_stats = super().get_stats()
        tor_status = asyncio.create_task(self.tor_controller.get_tor_status())
        
        return {
            **base_stats,
            "tor_enabled": True,
            "current_circuit_requests": self.current_circuit_requests,
            "requests_per_circuit": self.requests_per_circuit,
            "tor_status": tor_status.result() if tor_status.done() else {"status": "checking"},
        }


# Utility functions
async def scrape_onion_url(onion_url: str) -> Dict[str, Any]:
    """Quick utility for single .onion URL scraping"""
    scraper = TorStealthScraper()
    try:
        await scraper.start()
        return await scraper.scrape_onion_page(onion_url)
    finally:
        await scraper.close()


async def search_onion_content(search_term: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Quick utility for searching .onion content"""
    scraper = TorStealthScraper()
    try:
        await scraper.start()
        return await scraper.search_onion_directories(search_term)
    finally:
        await scraper.close()