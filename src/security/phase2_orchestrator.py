#!/usr/bin/env python3
"""Phase 2 Advanced Data Acquisition & Anti-Detection Orchestrator
Main orchestration module integrating all Phase 2 components

Author: Advanced AI Research Assistant
Date: August 2025
"""

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Any

from .archive_miner import ArchiveMiner
from .intelligent_proxy_manager import IntelligentProxyManager, ProxyTarget
from .tor_browser_selenium import TorBrowserSelenium

logger = logging.getLogger(__name__)


@dataclass
class Phase2Config:
    """Configuration for Phase 2 system"""

    # Anonymity providers configuration
    providers: dict[str, dict[str, Any]]

    # Proxy management
    proxy_management: dict[str, Any]

    # Behavior camouflage
    behavior_profile: dict[str, Any]

    # Archive mining
    archive_mining: dict[str, Any]

    # Tor Browser Selenium
    tor_browser: dict[str, Any]

    # Global settings
    max_concurrent_sessions: int = 3
    session_timeout: int = 300
    data_export_path: str = "./data/phase2_results"
    enable_comprehensive_logging: bool = True


class AdvancedDataAcquisitionSystem:
    """Main orchestrator for Phase 2 advanced data acquisition"""

    def __init__(self, config: Phase2Config):
        self.config = config

        # Initialize core components
        self.proxy_manager = IntelligentProxyManager(config.proxy_management)
        self.archive_miner = ArchiveMiner(config.archive_mining)
        self.tor_browser = None

        # Session management
        self.active_sessions: dict[str, dict[str, Any]] = {}
        self.session_results: list[dict[str, Any]] = []

        # Performance tracking
        self.start_time = datetime.now()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

    async def initialize(self) -> bool:
        """Initialize all Phase 2 components"""
        logger.info("Initializing Phase 2 Advanced Data Acquisition System...")

        try:
            # Initialize proxy manager and connect providers
            connection_results = await self.proxy_manager.connect_all_providers()
            logger.info(f"Provider connections: {connection_results}")

            # Test all providers
            test_results = await self.proxy_manager.test_all_providers()
            logger.info(f"Provider tests: {test_results}")

            # Initialize Tor Browser if configured
            if self.config.tor_browser.get("enabled", False):
                self.tor_browser = TorBrowserSelenium(self.config.tor_browser)
                tor_started = await self.tor_browser.start_browser()
                if tor_started:
                    logger.info("Tor Browser initialized successfully")
                else:
                    logger.warning("Tor Browser initialization failed")

            # Setup data export directory
            Path(self.config.data_export_path).mkdir(parents=True, exist_ok=True)

            logger.info("Phase 2 system initialization completed")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Phase 2 system: {e}")
            return False

    async def acquire_data_from_url(
        self,
        url: str,
        acquisition_method: str = "intelligent_proxy",
        target_config: ProxyTarget | None = None,
    ) -> dict[str, Any]:
        """Acquire data from URL using specified method"""
        session_id = f"session_{int(time.time())}_{len(self.active_sessions)}"

        session_info = {
            "session_id": session_id,
            "url": url,
            "method": acquisition_method,
            "start_time": datetime.now(),
            "status": "active",
        }

        self.active_sessions[session_id] = session_info

        try:
            if acquisition_method == "intelligent_proxy":
                result = await self._acquire_via_intelligent_proxy(url, target_config)
            elif acquisition_method == "tor_browser":
                result = await self._acquire_via_tor_browser(url)
            elif acquisition_method == "archive_mining":
                result = await self._acquire_via_archive_mining(url)
            elif acquisition_method == "hybrid":
                result = await self._acquire_via_hybrid_approach(url, target_config)
            else:
                raise ValueError(f"Unknown acquisition method: {acquisition_method}")

            session_info["status"] = "completed"
            session_info["success"] = result.get("success", False)
            session_info["end_time"] = datetime.now()
            session_info["duration"] = (
                session_info["end_time"] - session_info["start_time"]
            ).seconds

            self.session_results.append(session_info)

            if result.get("success"):
                self.successful_requests += 1
            else:
                self.failed_requests += 1

            self.total_requests += 1

            return result

        except Exception as e:
            logger.error(f"Error in data acquisition for {url}: {e}")
            session_info["status"] = "failed"
            session_info["error"] = str(e)
            session_info["end_time"] = datetime.now()

            self.failed_requests += 1
            self.total_requests += 1

            return {"success": False, "error": str(e), "session_id": session_id}

        finally:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]

    async def _acquire_via_intelligent_proxy(
        self, url: str, target_config: ProxyTarget | None = None
    ) -> dict[str, Any]:
        """Acquire data using intelligent proxy management"""
        if target_config:
            self.proxy_manager.add_target_config(target_config)

        start_time = time.time()

        try:
            session, provider_name = await self.proxy_manager.get_session_for_url(url)

            async with session.get(url) as response:
                response_time = time.time() - start_time
                content = await response.text()

                # Record success
                await self.proxy_manager.record_request_result(
                    url, provider_name, True, response_time
                )

                return {
                    "success": True,
                    "content": content,
                    "status_code": response.status,
                    "headers": dict(response.headers),
                    "provider_used": provider_name,
                    "response_time": response_time,
                    "content_length": len(content),
                    "method": "intelligent_proxy",
                }

        except Exception as e:
            response_time = time.time() - start_time
            await self.proxy_manager.record_request_result(
                url,
                provider_name if "provider_name" in locals() else "unknown",
                False,
                response_time,
                e,
            )
            raise

    async def _acquire_via_tor_browser(self, url: str) -> dict[str, Any]:
        """Acquire data using Tor Browser Selenium"""
        if not self.tor_browser:
            raise RuntimeError("Tor Browser not initialized")

        try:
            # Navigate to URL with behavior simulation
            success = await self.tor_browser.navigate_to_url(url, simulate_behavior=True)

            if success:
                # Extract page content
                content_data = await self.tor_browser.extract_page_content(include_metadata=True)

                # Take screenshot if configured
                screenshot_path = None
                if self.config.tor_browser.get("take_screenshots", False):
                    screenshot_path = (
                        Path(self.config.data_export_path)
                        / "screenshots"
                        / f"screenshot_{int(time.time())}.png"
                    )
                    await self.tor_browser.screenshot_page(str(screenshot_path))

                return {
                    "success": True,
                    "content": content_data.get("html", ""),
                    "metadata": content_data.get("metadata", {}),
                    "title": content_data.get("title", ""),
                    "final_url": content_data.get("url", url),
                    "screenshot_path": str(screenshot_path) if screenshot_path else None,
                    "method": "tor_browser",
                    "session_report": self.tor_browser.get_session_report(),
                }
            raise RuntimeError("Failed to navigate to URL")

        except Exception as e:
            logger.error(f"Tor Browser acquisition failed: {e}")
            raise

    async def _acquire_via_archive_mining(self, url: str) -> dict[str, Any]:
        """Acquire data through archive mining"""
        try:
            # Mine URL history
            snapshots = await self.archive_miner.mine_url_history(url)

            if not snapshots:
                return {
                    "success": False,
                    "error": "No archive snapshots found",
                    "method": "archive_mining",
                }

            # Get temporal evolution analysis
            evolution = await self.archive_miner.get_temporal_content_evolution(url)

            # Create new snapshot for future reference
            new_snapshot_url = await self.archive_miner.create_new_archive_snapshot(url)

            return {
                "success": True,
                "snapshots_found": len(snapshots),
                "temporal_evolution": evolution,
                "latest_snapshot": snapshots[-1].to_dict() if snapshots else None,
                "oldest_snapshot": snapshots[0].to_dict() if snapshots else None,
                "new_snapshot_created": new_snapshot_url,
                "method": "archive_mining",
                "mining_stats": self.archive_miner.get_mining_statistics(),
            }

        except Exception as e:
            logger.error(f"Archive mining failed: {e}")
            raise

    async def _acquire_via_hybrid_approach(
        self, url: str, target_config: ProxyTarget | None = None
    ) -> dict[str, Any]:
        """Acquire data using hybrid approach combining multiple methods"""
        results = {}
        errors = []

        # Try intelligent proxy first
        try:
            proxy_result = await self._acquire_via_intelligent_proxy(url, target_config)
            results["proxy"] = proxy_result
        except Exception as e:
            errors.append(f"Proxy method failed: {e}")

        # Try archive mining in parallel
        try:
            archive_result = await self._acquire_via_archive_mining(url)
            results["archive"] = archive_result
        except Exception as e:
            errors.append(f"Archive method failed: {e}")

        # Try Tor Browser if others failed or as backup
        if not results or self.config.tor_browser.get("always_use_hybrid", False):
            try:
                if self.tor_browser:
                    tor_result = await self._acquire_via_tor_browser(url)
                    results["tor_browser"] = tor_result
            except Exception as e:
                errors.append(f"Tor Browser method failed: {e}")

        if not results:
            return {"success": False, "errors": errors, "method": "hybrid"}

        # Combine results
        primary_content = None
        if "proxy" in results and results["proxy"].get("success"):
            primary_content = results["proxy"]["content"]
        elif "tor_browser" in results and results["tor_browser"].get("success"):
            primary_content = results["tor_browser"]["content"]

        return {
            "success": True,
            "primary_content": primary_content,
            "all_results": results,
            "errors": errors,
            "method": "hybrid",
            "methods_succeeded": list(results.keys()),
            "total_methods_attempted": len(results) + len(errors),
        }

    async def batch_acquire_urls(
        self,
        urls: list[str],
        method: str = "intelligent_proxy",
        max_concurrent: int | None = None,
    ) -> list[dict[str, Any]]:
        """Acquire data from multiple URLs concurrently"""
        max_concurrent = max_concurrent or self.config.max_concurrent_sessions

        semaphore = asyncio.Semaphore(max_concurrent)

        async def acquire_with_semaphore(url: str):
            async with semaphore:
                return await self.acquire_data_from_url(url, method)

        logger.info(
            f"Starting batch acquisition of {len(urls)} URLs with {max_concurrent} concurrent sessions"
        )

        tasks = [acquire_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    {"success": False, "url": urls[i], "error": str(result), "method": method}
                )
            else:
                processed_results.append(result)

        logger.info(
            f"Batch acquisition completed. Success rate: {sum(1 for r in processed_results if r.get('success')) / len(processed_results) * 100:.1f}%"
        )

        return processed_results

    async def export_session_data(self, export_format: str = "json") -> str:
        """Export all session data and results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "system_uptime_seconds": (datetime.now() - self.start_time).seconds,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (
                (self.successful_requests / self.total_requests * 100)
                if self.total_requests > 0
                else 0
            ),
            "active_sessions": len(self.active_sessions),
            "completed_sessions": len(self.session_results),
            "session_results": self.session_results,
            "proxy_performance": self.proxy_manager.get_performance_report(),
            "archive_mining_stats": self.archive_miner.get_mining_statistics(),
            "config": asdict(self.config),
        }

        # Add Tor Browser session report if available
        if self.tor_browser:
            export_data["tor_browser_session"] = self.tor_browser.get_session_report()

        if export_format == "json":
            filename = f"phase2_session_export_{timestamp}.json"
            filepath = Path(self.config.data_export_path) / filename

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

        else:
            raise ValueError(f"Unsupported export format: {export_format}")

        logger.info(f"Session data exported to {filepath}")
        return str(filepath)

    async def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "system_info": {
                "uptime_seconds": (datetime.now() - self.start_time).seconds,
                "active_sessions": len(self.active_sessions),
                "total_requests": self.total_requests,
                "success_rate": (
                    (self.successful_requests / self.total_requests * 100)
                    if self.total_requests > 0
                    else 0
                ),
            },
            "proxy_manager": {
                "connected_providers": len(
                    [p for p in self.proxy_manager.providers.values() if p.is_connected]
                ),
                "current_provider": self.proxy_manager.current_provider,
                "provider_use_count": self.proxy_manager.provider_use_count,
            },
            "archive_miner": self.archive_miner.get_mining_statistics(),
            "tor_browser": {
                "initialized": self.tor_browser is not None,
                "session_active": (
                    self.tor_browser.driver is not None if self.tor_browser else False
                ),
            },
            "timestamp": datetime.now().isoformat(),
        }

    async def shutdown(self):
        """Gracefully shutdown all components"""
        logger.info("Shutting down Phase 2 Advanced Data Acquisition System...")

        try:
            # Export final session data
            await self.export_session_data()

            # Save proxy performance data
            await self.proxy_manager.save_performance_data(
                str(Path(self.config.data_export_path) / "proxy_performance.json")
            )

            # Save archive mining results
            await self.archive_miner.export_mining_results(
                str(Path(self.config.data_export_path) / "archive_mining_results.json")
            )

            # Disconnect all providers
            await self.proxy_manager.disconnect_all_providers()

            # Close Tor Browser
            if self.tor_browser:
                await self.tor_browser.close_browser()

            logger.info("Phase 2 system shutdown completed")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Factory function for easy initialization
def create_phase2_system(config_dict: dict[str, Any]) -> AdvancedDataAcquisitionSystem:
    """Create Phase 2 system from configuration dictionary"""
    config = Phase2Config(**config_dict)
    return AdvancedDataAcquisitionSystem(config)


# Example usage configuration
EXAMPLE_PHASE2_CONFIG = {
    "providers": {
        "tor": {
            "socks_port": 9050,
            "control_port": 9051,
            "control_password": "",
            "rotation_interval": 600,
        },
        "clearnet": {
            "user_agents": [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            ]
        },
    },
    "proxy_management": {
        "rotation_strategy": "performance_based",
        "min_rotation_interval": 60,
        "max_consecutive_uses": 10,
        "circuit_breaker_threshold": 5,
        "circuit_breaker_timeout": 300,
    },
    "behavior_profile": {
        "reading_speed_wpm": 200,
        "scroll_speed_variance": 0.3,
        "click_delay_range": [0.1, 0.5],
        "typing_speed_cps": 4,
        "mouse_movement_style": "natural",
    },
    "archive_mining": {
        "max_snapshots_per_url": 100,
        "min_time_between_snapshots": 30,
        "content_filters": [r"\.pdf$", r"\.zip$"],
    },
    "tor_browser": {
        "enabled": False,
        "headless": False,
        "take_screenshots": True,
        "tor_browser_path": None,
    },
    "max_concurrent_sessions": 3,
    "session_timeout": 300,
    "data_export_path": "./data/phase2_results",
}
