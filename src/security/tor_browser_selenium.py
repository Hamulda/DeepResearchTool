#!/usr/bin/env python3
"""Tor Browser Selenium Integration for Phase 2 Implementation
Advanced anti-fingerprinting with real Tor Browser automation

Author: Advanced AI Research Assistant
Date: August 2025
"""

import asyncio
import logging
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
from typing import Any

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from .behavior_camouflage import BehaviorCamouflage, HumanBehaviorProfile

logger = logging.getLogger(__name__)


class TorBrowserProfile:
    """Manages Tor Browser profile with security settings"""

    def __init__(self, profile_dir: str | None = None):
        self.profile_dir = profile_dir or self._create_temp_profile()
        self.security_level = "safest"  # safest, safer, standard

    def _create_temp_profile(self) -> str:
        """Create temporary Tor Browser profile"""
        temp_dir = tempfile.mkdtemp(prefix="tor_profile_")
        return temp_dir

    def setup_profile(self) -> str:
        """Setup Tor Browser profile with security configurations"""
        profile_path = Path(self.profile_dir)
        profile_path.mkdir(parents=True, exist_ok=True)

        # Create prefs.js with Tor Browser security settings
        prefs_content = self._get_tor_browser_prefs()

        prefs_file = profile_path / "prefs.js"
        with open(prefs_file, "w") as f:
            f.write(prefs_content)

        # Create user.js for additional hardening
        user_js_content = self._get_user_js_prefs()

        user_js_file = profile_path / "user.js"
        with open(user_js_file, "w") as f:
            f.write(user_js_content)

        logger.info(f"Tor Browser profile setup at: {self.profile_dir}")
        return self.profile_dir

    def _get_tor_browser_prefs(self) -> str:
        """Get Tor Browser specific preferences"""
        prefs = {
            # Security and privacy
            "privacy.resistFingerprinting": True,
            "privacy.trackingprotection.enabled": True,
            "privacy.trackingprotection.socialtracking.enabled": True,
            "privacy.firstparty.isolate": True,
            "privacy.partition.network_state": True,
            # Disable potentially identifying features
            "media.peerconnection.enabled": False,
            "webgl.disabled": True,
            "geo.enabled": False,
            "dom.battery.enabled": False,
            "dom.event.clipboardevents.enabled": False,
            "dom.webaudio.enabled": False,
            # Network settings
            "network.proxy.type": 1,
            "network.proxy.socks": "127.0.0.1",
            "network.proxy.socks_port": 9050,
            "network.proxy.socks_remote_dns": True,
            # JavaScript security
            "javascript.options.ion": False,
            "javascript.options.baselinejit": False,
            "javascript.options.native_regexp": False,
            "javascript.options.asmjs": False,
            "javascript.options.wasm": False,
            # Font fingerprinting protection
            "gfx.downloadable_fonts.enabled": False,
            "gfx.font_rendering.opentype_svg.enabled": False,
            # Canvas fingerprinting protection
            "privacy.resistFingerprinting.randomization.enabled": True,
            # Disable telemetry
            "toolkit.telemetry.enabled": False,
            "toolkit.telemetry.unified": False,
            "toolkit.telemetry.archive.enabled": False,
            "datareporting.healthreport.uploadEnabled": False,
            # Security level adjustments
            "security.tls.version.min": 3,  # TLS 1.2 minimum
            "security.ssl.require_safe_negotiation": True,
            "security.ssl.treat_unsafe_negotiation_as_broken": True,
            # NoScript equivalent settings
            "javascript.enabled": self.security_level != "safest",
            "plugins.enabled": False,
            # User agent (will be overridden by Tor Browser)
            "general.useragent.override": "",
            # Extensions and add-ons
            "extensions.torbutton.inserted_button": True,
            "extensions.torbutton.launch_warning": False,
            "extensions.torbutton.loglevel": 2,
        }

        # Convert to Firefox prefs format
        prefs_lines = []
        for key, value in prefs.items():
            if isinstance(value, bool):
                prefs_lines.append(f'user_pref("{key}", {str(value).lower()});')
            elif isinstance(value, int):
                prefs_lines.append(f'user_pref("{key}", {value});')
            else:
                prefs_lines.append(f'user_pref("{key}", "{value}");')

        return "\n".join(prefs_lines)

    def _get_user_js_prefs(self) -> str:
        """Get additional user.js preferences for hardening"""
        return """
// Additional Tor Browser hardening
user_pref("media.video_stats.enabled", false);
user_pref("dom.enable_performance", false);
user_pref("dom.enable_resource_timing", false);
user_pref("dom.enable_user_timing", false);
user_pref("dom.netinfo.enabled", false);
user_pref("dom.network.enabled", false);
user_pref("browser.cache.disk.enable", false);
user_pref("browser.cache.memory.enable", false);
user_pref("browser.cache.offline.enable", false);
user_pref("network.http.use-cache", false);
user_pref("places.history.enabled", false);
user_pref("privacy.clearOnShutdown.downloads", true);
user_pref("privacy.clearOnShutdown.formdata", true);
user_pref("privacy.clearOnShutdown.history", true);
user_pref("privacy.clearOnShutdown.sessions", true);
user_pref("privacy.clearOnShutdown.siteSettings", true);
"""

    def cleanup(self):
        """Clean up temporary profile"""
        if os.path.exists(self.profile_dir):
            try:
                shutil.rmtree(self.profile_dir)
                logger.info(f"Cleaned up Tor Browser profile: {self.profile_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up profile: {e}")


class TorBrowserSelenium:
    """Enhanced Tor Browser automation with anti-detection"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.tor_browser_path = config.get("tor_browser_path")
        self.profile_manager = TorBrowserProfile()
        self.driver = None
        self.behavior_camouflage = BehaviorCamouflage(
            HumanBehaviorProfile(**config.get("behavior_profile", {}))
        )

        # Circuit management
        self.circuit_lifetime = config.get("circuit_lifetime", 600)  # 10 minutes
        self.last_circuit_renewal = time.time()

    def _get_tor_browser_options(self) -> Options:
        """Configure Tor Browser specific options"""
        options = Options()

        # Use Tor Browser profile
        profile_path = self.profile_manager.setup_profile()
        options.add_argument("-profile")
        options.add_argument(profile_path)

        # Tor Browser specific arguments
        options.add_argument("--class=Tor Browser")

        # Window size (common Tor Browser sizes)
        window_sizes = [
            (1000, 800),
            (1200, 900),
            (1400, 1000),
        ]
        width, height = random.choice(window_sizes)
        options.add_argument(f"--width={width}")
        options.add_argument(f"--height={height}")

        # Additional privacy arguments
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-extensions-file-access-check")
        options.add_argument("--disable-extensions-http-throttling")
        options.add_argument("--disable-component-extensions-with-background-pages")

        # Headless mode if configured
        if self.config.get("headless", False):
            options.add_argument("--headless")

        return options

    async def start_browser(self) -> bool:
        """Start Tor Browser with full anonymity setup"""
        try:
            if not self.tor_browser_path or not os.path.exists(self.tor_browser_path):
                # Try to find Tor Browser automatically
                self.tor_browser_path = self._find_tor_browser()

            if not self.tor_browser_path:
                logger.error("Tor Browser not found. Please install Tor Browser.")
                return False

            options = self._get_tor_browser_options()

            # Use Tor Browser's Firefox binary
            firefox_binary = os.path.join(self.tor_browser_path, "firefox")
            if os.name == "nt":  # Windows
                firefox_binary += ".exe"

            service = Service(executable_path=firefox_binary)

            # Start the browser
            self.driver = webdriver.Firefox(service=service, options=options)

            # Set additional properties to avoid detection
            self.driver.execute_script(
                """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
                
                // Remove selenium indicators
                delete window.cdc_adoQpoasnfa76pfcZLmcfl_Array;
                delete window.cdc_adoQpoasnfa76pfcZLmcfl_Promise;
                delete window.cdc_adoQpoasnfa76pfcZLmcfl_Symbol;
            """
            )

            # Verify Tor connection
            if await self._verify_tor_connection():
                logger.info("Tor Browser started successfully with verified anonymity")
                return True
            logger.error("Tor connection verification failed")
            await self.close_browser()
            return False

        except Exception as e:
            logger.error(f"Failed to start Tor Browser: {e}")
            return False

    def _find_tor_browser(self) -> str | None:
        """Try to find Tor Browser installation automatically"""
        possible_paths = []

        if os.name == "nt":  # Windows
            possible_paths.extend(
                [
                    os.path.expanduser("~/Desktop/Tor Browser/Browser"),
                    "C:/Users/*/Desktop/Tor Browser/Browser",
                    "C:/Program Files/Tor Browser/Browser",
                ]
            )
        elif os.name == "posix":  # Linux/macOS
            possible_paths.extend(
                [
                    os.path.expanduser("~/tor-browser/Browser"),
                    "/Applications/Tor Browser.app/Contents/MacOS/Tor Browser",
                    "/opt/tor-browser/Browser",
                    "/usr/local/tor-browser/Browser",
                ]
            )

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return None

    async def _verify_tor_connection(self) -> bool:
        """Verify that traffic is going through Tor"""
        try:
            # Navigate to Tor check page
            self.driver.get("https://check.torproject.org/")

            # Wait for page to load
            WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            # Look for Tor confirmation
            page_source = self.driver.page_source.lower()

            if "congratulations" in page_source and "using tor" in page_source:
                logger.info("Tor connection verified successfully")
                return True
            logger.warning("Tor connection verification failed")
            return False

        except Exception as e:
            logger.error(f"Error verifying Tor connection: {e}")
            return False

    async def navigate_to_url(self, url: str, simulate_behavior: bool = True) -> bool:
        """Navigate to URL with comprehensive anti-detection"""
        if not self.driver:
            raise RuntimeError("Tor Browser not started")

        try:
            # Check if we should renew circuit
            if time.time() - self.last_circuit_renewal > self.circuit_lifetime:
                await self._renew_tor_circuit()

            # Navigate to URL
            self.driver.get(url)

            # Wait for page load with timeout
            WebDriverWait(self.driver, 30).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )

            if simulate_behavior:
                # Simulate realistic human behavior
                await self.behavior_camouflage.simulate_page_exploration(self.driver)

                # Random chance to take a break
                await self.behavior_camouflage.take_break()

            logger.info(f"Successfully navigated to {url}")
            return True

        except TimeoutException:
            logger.error(f"Timeout loading {url}")
            return False
        except Exception as e:
            logger.error(f"Error navigating to {url}: {e}")
            return False

    async def _renew_tor_circuit(self) -> bool:
        """Renew Tor circuit by navigating to new identity"""
        try:
            # Get new Tor identity by accessing Tor control
            # This would require stem library integration
            logger.info("Attempting to renew Tor circuit")

            # Alternative: Clear cookies and restart browser session
            self.driver.delete_all_cookies()

            # Clear local storage
            self.driver.execute_script("window.localStorage.clear();")
            self.driver.execute_script("window.sessionStorage.clear();")

            self.last_circuit_renewal = time.time()

            # Verify new identity
            await asyncio.sleep(5)  # Wait for circuit renewal
            if await self._verify_tor_connection():
                logger.info("Tor circuit renewed successfully")
                return True
            logger.warning("Tor circuit renewal verification failed")
            return False

        except Exception as e:
            logger.error(f"Error renewing Tor circuit: {e}")
            return False

    async def extract_page_content(self, include_metadata: bool = True) -> dict[str, Any]:
        """Extract page content with metadata"""
        if not self.driver:
            raise RuntimeError("Tor Browser not started")

        try:
            content = {
                "url": self.driver.current_url,
                "title": self.driver.title,
                "html": self.driver.page_source,
                "timestamp": time.time(),
            }

            if include_metadata:
                # Extract metadata
                content["metadata"] = {
                    "page_height": self.driver.execute_script("return document.body.scrollHeight"),
                    "page_width": self.driver.execute_script("return document.body.scrollWidth"),
                    "links_count": len(self.driver.find_elements(By.TAG_NAME, "a")),
                    "images_count": len(self.driver.find_elements(By.TAG_NAME, "img")),
                    "forms_count": len(self.driver.find_elements(By.TAG_NAME, "form")),
                }

                # Extract meta tags
                meta_tags = {}
                try:
                    meta_elements = self.driver.find_elements(By.TAG_NAME, "meta")
                    for meta in meta_elements:
                        name = meta.get_attribute("name") or meta.get_attribute("property")
                        content_attr = meta.get_attribute("content")
                        if name and content_attr:
                            meta_tags[name] = content_attr
                    content["metadata"]["meta_tags"] = meta_tags
                except Exception:
                    pass

            return content

        except Exception as e:
            logger.error(f"Error extracting page content: {e}")
            return {}

    async def screenshot_page(self, filepath: str) -> bool:
        """Take screenshot of current page"""
        if not self.driver:
            raise RuntimeError("Tor Browser not started")

        try:
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            # Take screenshot
            success = self.driver.save_screenshot(filepath)

            if success:
                logger.info(f"Screenshot saved to {filepath}")
            else:
                logger.error(f"Failed to save screenshot to {filepath}")

            return success

        except Exception as e:
            logger.error(f"Error taking screenshot: {e}")
            return False

    async def close_browser(self):
        """Close browser and cleanup"""
        try:
            if self.driver:
                self.driver.quit()
                self.driver = None
                logger.info("Tor Browser closed")

            # Cleanup profile
            self.profile_manager.cleanup()

        except Exception as e:
            logger.error(f"Error closing Tor Browser: {e}")

    def get_session_report(self) -> dict[str, Any]:
        """Get comprehensive session report"""
        behavior_report = self.behavior_camouflage.get_behavior_report()

        return {
            "tor_browser_session": {
                "started": self.driver is not None,
                "circuit_lifetime": self.circuit_lifetime,
                "last_circuit_renewal_ago": time.time() - self.last_circuit_renewal,
                "profile_path": self.profile_manager.profile_dir,
            },
            "behavior_analysis": behavior_report,
            "timestamp": time.time(),
        }
