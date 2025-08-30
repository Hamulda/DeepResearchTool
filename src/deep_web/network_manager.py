"""Network Manager - Transport-Agnostic HTTP Client
Unified interface for Tor, I2P, and clearnet requests
Automatic failover and load balancing between networks
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
import random
import time
from typing import Any

import aiohttp

from .i2p_manager import I2PConfig, I2PManager
from .tor_manager import TorConfig, TorManager

logger = logging.getLogger(__name__)


class NetworkType(Enum):
    """Available network types"""

    CLEARNET = "clearnet"
    TOR = "tor"
    I2P = "i2p"


@dataclass
class NetworkConfig:
    """Network manager configuration"""

    # Network preferences
    preferred_networks: list[NetworkType] = None
    enable_clearnet: bool = True
    enable_tor: bool = False  # Opt-in
    enable_i2p: bool = False  # Opt-in

    # Failover settings
    enable_failover: bool = True
    failover_timeout: float = 30.0
    max_failover_attempts: int = 3

    # Load balancing
    load_balance_strategy: str = "round_robin"  # round_robin, random, latency
    health_check_interval: int = 300  # 5 minutes

    # Request settings
    default_timeout: int = 30
    max_retries_per_network: int = 2
    user_agent: str = "Mozilla/5.0 (compatible; DeepResearchTool/2.0)"

    # Safety settings
    respect_robots: bool = True
    allowlist_only: bool = True
    blocked_domains: list[str] = None

    # Sub-configs
    tor_config: TorConfig | None = None
    i2p_config: I2PConfig | None = None

    def __post_init__(self):
        if self.preferred_networks is None:
            # Default preference order
            self.preferred_networks = []
            if self.enable_clearnet:
                self.preferred_networks.append(NetworkType.CLEARNET)
            if self.enable_tor:
                self.preferred_networks.append(NetworkType.TOR)
            if self.enable_i2p:
                self.preferred_networks.append(NetworkType.I2P)

        if self.blocked_domains is None:
            self.blocked_domains = []


class NetworkStats:
    """Track network performance statistics"""

    def __init__(self):
        self.stats = {
            NetworkType.CLEARNET: {
                "requests": 0,
                "successes": 0,
                "avg_latency": 0.0,
                "last_success": None,
            },
            NetworkType.TOR: {
                "requests": 0,
                "successes": 0,
                "avg_latency": 0.0,
                "last_success": None,
            },
            NetworkType.I2P: {
                "requests": 0,
                "successes": 0,
                "avg_latency": 0.0,
                "last_success": None,
            },
        }
        self._latency_samples = {net: [] for net in NetworkType}

    def record_request(self, network: NetworkType, success: bool, latency: float = 0.0):
        """Record request outcome and latency"""
        self.stats[network]["requests"] += 1

        if success:
            self.stats[network]["successes"] += 1
            self.stats[network]["last_success"] = datetime.now()

            # Update latency average
            samples = self._latency_samples[network]
            samples.append(latency)

            # Keep only last 100 samples
            if len(samples) > 100:
                samples.pop(0)

            self.stats[network]["avg_latency"] = sum(samples) / len(samples)

    def get_success_rate(self, network: NetworkType) -> float:
        """Get success rate for network"""
        stats = self.stats[network]
        return stats["successes"] / max(1, stats["requests"])

    def get_best_network(self) -> NetworkType | None:
        """Get network with best performance"""
        best_network = None
        best_score = -1

        for network in NetworkType:
            stats = self.stats[network]
            if stats["requests"] == 0:
                continue

            # Score based on success rate and latency
            success_rate = self.get_success_rate(network)
            latency_penalty = min(stats["avg_latency"] / 10.0, 1.0)  # Normalize latency
            score = success_rate - latency_penalty

            if score > best_score:
                best_score = score
                best_network = network

        return best_network

    def get_stats_summary(self) -> dict[str, Any]:
        """Get comprehensive statistics"""
        summary = {}
        for network, stats in self.stats.items():
            summary[network.value] = {
                **stats,
                "success_rate": self.get_success_rate(network),
                "last_success": (
                    stats["last_success"].isoformat() if stats["last_success"] else None
                ),
            }

        summary["best_network"] = self.get_best_network().value if self.get_best_network() else None
        return summary


class NetworkManager:
    """Unified network client with automatic failover"""

    def __init__(self, config: NetworkConfig):
        self.config = config
        self.stats = NetworkStats()

        # Network clients
        self.tor_manager: TorManager | None = None
        self.i2p_manager: I2PManager | None = None
        self.clearnet_session: aiohttp.ClientSession | None = None

        # State tracking
        self._network_health = dict.fromkeys(NetworkType, True)
        self._last_health_check = datetime.now()
        self._network_rotation_index = 0

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()

    async def initialize(self):
        """Initialize all configured network clients"""
        logger.info("Initializing network manager...")

        # Initialize Tor if enabled
        if self.config.enable_tor:
            tor_config = self.config.tor_config or TorConfig(enable_tor=True)
            self.tor_manager = TorManager(tor_config)

            try:
                await self.tor_manager.connect()
                logger.info("Tor network initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Tor: {e}")
                self._network_health[NetworkType.TOR] = False

        # Initialize I2P if enabled
        if self.config.enable_i2p:
            i2p_config = self.config.i2p_config or I2PConfig(enable_i2p=True)
            self.i2p_manager = I2PManager(i2p_config)

            try:
                await self.i2p_manager.connect()
                logger.info("I2P network initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize I2P: {e}")
                self._network_health[NetworkType.I2P] = False

        # Initialize clearnet session if enabled
        if self.config.enable_clearnet:
            timeout = aiohttp.ClientTimeout(total=self.config.default_timeout)
            connector = aiohttp.TCPConnector(
                limit=100, limit_per_host=10, use_dns_cache=True, ttl_dns_cache=300
            )

            self.clearnet_session = aiohttp.ClientSession(
                connector=connector, timeout=timeout, headers={"User-Agent": self.config.user_agent}
            )
            logger.info("Clearnet session initialized")

    async def cleanup(self):
        """Cleanup all network clients"""
        if self.tor_manager:
            await self.tor_manager.disconnect()

        if self.i2p_manager:
            await self.i2p_manager.disconnect()

        if self.clearnet_session:
            await self.clearnet_session.close()

        logger.info("Network manager cleaned up")

    def _determine_network_for_url(self, url: str) -> NetworkType:
        """Determine appropriate network for URL"""
        if ".onion" in url:
            return NetworkType.TOR
        if ".i2p" in url:
            return NetworkType.I2P
        return NetworkType.CLEARNET

    def _get_available_networks(self, preferred_network: NetworkType = None) -> list[NetworkType]:
        """Get list of available networks for request"""
        if preferred_network and self._network_health.get(preferred_network, False):
            networks = [preferred_network]
        else:
            networks = []

        # Add other healthy networks based on preference
        for network in self.config.preferred_networks:
            if network not in networks and self._network_health.get(network, False):
                networks.append(network)

        return networks

    def _select_network(self, available_networks: list[NetworkType]) -> NetworkType | None:
        """Select network based on load balancing strategy"""
        if not available_networks:
            return None

        if self.config.load_balance_strategy == "round_robin":
            network = available_networks[self._network_rotation_index % len(available_networks)]
            self._network_rotation_index += 1
            return network

        if self.config.load_balance_strategy == "random":
            return random.choice(available_networks)

        if self.config.load_balance_strategy == "latency":
            # Select network with best performance
            best_network = self.stats.get_best_network()
            if best_network in available_networks:
                return best_network
            return available_networks[0]

        return available_networks[0]

    async def _make_request_on_network(
        self, network: NetworkType, method: str, url: str, **kwargs
    ) -> aiohttp.ClientResponse:
        """Make request on specific network"""
        start_time = time.time()

        try:
            if network == NetworkType.TOR:
                if not self.tor_manager or not self.tor_manager.is_connected:
                    raise ConnectionError("Tor not available")
                response = await self.tor_manager.make_request(method, url, **kwargs)

            elif network == NetworkType.I2P:
                if not self.i2p_manager or not self.i2p_manager.session.is_connected:
                    raise ConnectionError("I2P not available")
                response = await self.i2p_manager.make_request(method, url, **kwargs)

            elif network == NetworkType.CLEARNET:
                if not self.clearnet_session:
                    raise ConnectionError("Clearnet not available")
                async with self.clearnet_session.request(method, url, **kwargs) as response:
                    return response

            else:
                raise ValueError(f"Unknown network type: {network}")

            # Record successful request
            latency = time.time() - start_time
            self.stats.record_request(network, True, latency)
            logger.debug(f"Request successful on {network.value} in {latency:.2f}s")

            return response

        except Exception as e:
            # Record failed request
            latency = time.time() - start_time
            self.stats.record_request(network, False, latency)
            logger.warning(f"Request failed on {network.value}: {e}")
            raise

    async def get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make GET request with automatic network selection and failover"""
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make POST request with automatic network selection and failover"""
        return await self.request("POST", url, **kwargs)

    async def request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make HTTP request with automatic network selection and failover"""
        # Safety checks
        if self.config.allowlist_only and not self._is_url_allowed(url):
            raise ValueError(f"URL not in allowlist: {url}")

        if self._is_url_blocked(url):
            raise ValueError(f"URL is blocked: {url}")

        # Determine preferred network for URL
        preferred_network = self._determine_network_for_url(url)
        available_networks = self._get_available_networks(preferred_network)

        if not available_networks:
            raise ConnectionError("No networks available for request")

        # Attempt request with failover
        last_exception = None

        for attempt in range(self.config.max_failover_attempts):
            # Select network for this attempt
            network = self._select_network(available_networks)
            if not network:
                break

            try:
                response = await self._make_request_on_network(network, method, url, **kwargs)
                logger.debug(f"Request successful: {method} {url} via {network.value}")
                return response

            except Exception as e:
                last_exception = e
                logger.warning(f"Request failed on {network.value} (attempt {attempt + 1}): {e}")

                # Mark network as unhealthy if configured
                if self.config.enable_failover:
                    self._network_health[network] = False
                    available_networks = [n for n in available_networks if n != network]

                # Wait before retry
                if attempt < self.config.max_failover_attempts - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))

        # All attempts failed
        raise last_exception or ConnectionError("All network attempts failed")

    def _is_url_allowed(self, url: str) -> bool:
        """Check if URL is in allowlist (implement based on requirements)"""
        # For now, allow all URLs when allowlist_only is False
        if not self.config.allowlist_only:
            return True

        # TODO: Implement allowlist checking logic
        # Could check against whitelist of domains, patterns, etc.
        return True

    def _is_url_blocked(self, url: str) -> bool:
        """Check if URL is blocked"""
        for blocked_domain in self.config.blocked_domains:
            if blocked_domain in url:
                return True
        return False

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on all networks"""
        health_results = {}

        # Test clearnet
        if self.config.enable_clearnet and self.clearnet_session:
            try:
                async with self.clearnet_session.get(
                    "http://httpbin.org/ip", timeout=5
                ) as response:
                    health_results[NetworkType.CLEARNET.value] = {
                        "healthy": response.status == 200,
                        "latency": response.headers.get("X-Response-Time", "unknown"),
                    }
                    self._network_health[NetworkType.CLEARNET] = response.status == 200
            except Exception as e:
                health_results[NetworkType.CLEARNET.value] = {"healthy": False, "error": str(e)}
                self._network_health[NetworkType.CLEARNET] = False

        # Test Tor
        if self.config.enable_tor and self.tor_manager:
            try:
                status = await self.tor_manager.get_tor_status()
                health_results[NetworkType.TOR.value] = {
                    "healthy": status.get("status") == "connected",
                    "circuits": status.get("circuit_info", {}).get("active_circuits", 0),
                }
                self._network_health[NetworkType.TOR] = status.get("status") == "connected"
            except Exception as e:
                health_results[NetworkType.TOR.value] = {"healthy": False, "error": str(e)}
                self._network_health[NetworkType.TOR] = False

        # Test I2P
        if self.config.enable_i2p and self.i2p_manager:
            try:
                status = await self.i2p_manager.get_i2p_status()
                health_results[NetworkType.I2P.value] = {
                    "healthy": status.get("network_connected", False),
                    "session": status.get("session", {}),
                }
                self._network_health[NetworkType.I2P] = status.get("network_connected", False)
            except Exception as e:
                health_results[NetworkType.I2P.value] = {"healthy": False, "error": str(e)}
                self._network_health[NetworkType.I2P] = False

        self._last_health_check = datetime.now()
        return health_results

    async def get_network_status(self) -> dict[str, Any]:
        """Get comprehensive network status"""
        # Perform health check if needed
        if datetime.now() - self._last_health_check > timedelta(
            seconds=self.config.health_check_interval
        ):
            await self.health_check()

        return {
            "config": {
                "enabled_networks": [net.value for net in self.config.preferred_networks],
                "failover_enabled": self.config.enable_failover,
                "load_balance_strategy": self.config.load_balance_strategy,
            },
            "health": {net.value: healthy for net, healthy in self._network_health.items()},
            "statistics": self.stats.get_stats_summary(),
            "last_health_check": self._last_health_check.isoformat(),
        }


# Utility functions
async def test_all_networks(config: NetworkConfig = None) -> dict[str, Any]:
    """Test connectivity to all configured networks"""
    if config is None:
        config = NetworkConfig(enable_clearnet=True, enable_tor=True, enable_i2p=True)

    async with NetworkManager(config) as network_manager:
        health_results = await network_manager.health_check()
        status = await network_manager.get_network_status()

        return {"health_check": health_results, "network_status": status}


def create_safe_config() -> NetworkConfig:
    """Create safe default configuration"""
    return NetworkConfig(
        enable_clearnet=True,
        enable_tor=False,  # Opt-in
        enable_i2p=False,  # Opt-in
        respect_robots=True,
        allowlist_only=True,
        enable_failover=True,
        load_balance_strategy="round_robin",
    )


__all__ = [
    "NetworkConfig",
    "NetworkManager",
    "NetworkStats",
    "NetworkType",
    "create_safe_config",
    "test_all_networks",
]
