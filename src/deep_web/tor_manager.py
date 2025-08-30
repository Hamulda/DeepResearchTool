"""
Tor Manager for Programmatic Anonymity Layer
Provides control-plane separation with stem Controller and data-plane via SOCKS5
Optimized for MacBook Air M1 8GB RAM constraints
"""

import asyncio
import aiohttp
import httpx
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from pathlib import Path
import logging
import time
import random
import subprocess
import socket
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

try:
    from stem import Signal
    from stem.control import Controller
    from stem.connection import connect

    STEM_AVAILABLE = True
except ImportError:
    STEM_AVAILABLE = False
    logger.warning("Stem library not available - Tor control features disabled")


@dataclass
class TorConfig:
    """Tor configuration settings"""

    socks_port: int = 9050
    control_port: int = 9051
    control_password: Optional[str] = None
    control_socket_path: Optional[str] = None
    data_directory: Optional[Path] = None

    # Circuit management
    circuit_rotation_policy: str = "requests"  # requests, time, errors
    max_requests_per_circuit: int = 50
    circuit_timeout_minutes: int = 10
    max_circuit_retries: int = 3

    # Health monitoring
    health_check_interval: int = 30
    health_check_url: str = "http://httpbin.org/ip"
    max_consecutive_failures: int = 3

    # Proxy settings
    request_timeout: int = 30
    max_retries: int = 3
    backoff_factor: float = 2.0

    # Safety settings
    enable_tor: bool = False  # Opt-in by default
    strict_exit_policy: bool = True
    blocked_countries: List[str] = None
    allowed_exit_nodes: List[str] = None


class TorCircuitStats:
    """Track circuit usage statistics"""

    def __init__(self):
        self.circuits_created = 0
        self.requests_made = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.last_rotation = datetime.now()
        self.current_circuit_requests = 0
        self.consecutive_failures = 0
        self.exit_nodes_used = set()

    def record_request(self, success: bool, exit_node: str = None):
        """Record request statistics"""
        self.requests_made += 1
        self.current_circuit_requests += 1

        if success:
            self.successful_requests += 1
            self.consecutive_failures = 0
        else:
            self.failed_requests += 1
            self.consecutive_failures += 1

        if exit_node:
            self.exit_nodes_used.add(exit_node)

    def record_circuit_rotation(self):
        """Record circuit rotation"""
        self.circuits_created += 1
        self.current_circuit_requests = 0
        self.last_rotation = datetime.now()

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            "circuits_created": self.circuits_created,
            "total_requests": self.requests_made,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.successful_requests / max(1, self.requests_made),
            "current_circuit_requests": self.current_circuit_requests,
            "consecutive_failures": self.consecutive_failures,
            "unique_exit_nodes": len(self.exit_nodes_used),
            "last_rotation": self.last_rotation.isoformat(),
        }


class TorManager:
    """Tor network manager with control and data plane separation"""

    def __init__(self, config: TorConfig):
        if not STEM_AVAILABLE:
            raise ImportError("Stem library required for Tor control functionality")

        self.config = config
        self.stats = TorCircuitStats()
        self.controller: Optional[Controller] = None
        self.is_connected = False
        self.last_health_check = datetime.now()
        self._session: Optional[aiohttp.ClientSession] = None

        # Circuit rotation tracking
        self._rotation_lock = asyncio.Lock()
        self._health_task: Optional[asyncio.Task] = None

        if not config.enable_tor:
            logger.warning("Tor is disabled by configuration - enable_tor=False")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()

    async def connect(self) -> bool:
        """Connect to Tor control port and setup session"""
        if not self.config.enable_tor:
            logger.info("Tor disabled - skipping connection")
            return False

        try:
            # Connect to Tor control port
            if self.config.control_socket_path:
                self.controller = Controller.from_socket_file(self.config.control_socket_path)
            else:
                self.controller = Controller.from_port(port=self.config.control_port)

            self.controller.authenticate(password=self.config.control_password)

            # Verify Tor is running
            if not self.controller.is_alive():
                raise ConnectionError("Tor process is not running")

            # Setup SOCKS5 session
            connector = aiohttp.TCPConnector(
                use_dns_cache=False, ttl_dns_cache=300, limit=100, limit_per_host=10
            )

            # Configure SOCKS5 proxy
            proxy_url = f"socks5://127.0.0.1:{self.config.socks_port}"

            timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0 (compatible; DeepResearchTool/2.0)"},
            )

            self.is_connected = True

            # Start health monitoring
            self._health_task = asyncio.create_task(self._health_monitor())

            # Test initial connection
            await self._test_connection()

            logger.info("Successfully connected to Tor network")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Tor: {e}")
            await self.disconnect()
            return False

    async def disconnect(self):
        """Disconnect from Tor and cleanup resources"""
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        if self._session:
            await self._session.close()
            self._session = None

        if self.controller:
            self.controller.close()
            self.controller = None

        self.is_connected = False
        logger.info("Disconnected from Tor network")

    async def _test_connection(self) -> bool:
        """Test Tor connection with health check URL"""
        try:
            async with self._session.get(
                self.config.health_check_url, proxy=f"socks5://127.0.0.1:{self.config.socks_port}"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Tor connection verified - IP: {data.get('origin', 'unknown')}")
                    return True
                else:
                    logger.warning(f"Health check failed with status {response.status}")
                    return False

        except Exception as e:
            logger.error(f"Tor connection test failed: {e}")
            return False

    async def rotate_circuit(self, reason: str = "manual") -> bool:
        """Rotate Tor circuit (NEWNYM signal)"""
        if not self.is_connected or not self.controller:
            logger.warning("Cannot rotate circuit - not connected to Tor")
            return False

        async with self._rotation_lock:
            try:
                # Send NEWNYM signal
                self.controller.signal(Signal.NEWNYM)

                # Wait for circuit to be established
                await asyncio.sleep(2)

                # Update statistics
                self.stats.record_circuit_rotation()

                logger.info(f"Circuit rotated - reason: {reason}")
                return True

            except Exception as e:
                logger.error(f"Failed to rotate circuit: {e}")
                return False

    async def should_rotate_circuit(self) -> bool:
        """Determine if circuit should be rotated based on policy"""
        if self.config.circuit_rotation_policy == "requests":
            return self.stats.current_circuit_requests >= self.config.max_requests_per_circuit

        elif self.config.circuit_rotation_policy == "time":
            time_since_rotation = datetime.now() - self.stats.last_rotation
            return time_since_rotation >= timedelta(minutes=self.config.circuit_timeout_minutes)

        elif self.config.circuit_rotation_policy == "errors":
            return self.stats.consecutive_failures >= self.config.max_consecutive_failures

        return False

    async def get_circuit_info(self) -> Dict[str, Any]:
        """Get current circuit information"""
        if not self.controller:
            return {}

        try:
            circuits = self.controller.get_circuits()
            active_circuits = [c for c in circuits if c.status == "BUILT"]

            circuit_info = []
            for circuit in active_circuits[:3]:  # Show top 3 circuits
                path = []
                for relay in circuit.path:
                    relay_info = {
                        "fingerprint": relay[0],
                        "nickname": relay[1] if len(relay) > 1 else "unknown",
                    }
                    path.append(relay_info)

                circuit_info.append(
                    {
                        "id": circuit.id,
                        "status": circuit.status,
                        "path_length": len(circuit.path),
                        "path": path,
                    }
                )

            return {
                "total_circuits": len(circuits),
                "active_circuits": len(active_circuits),
                "circuits": circuit_info,
            }

        except Exception as e:
            logger.error(f"Failed to get circuit info: {e}")
            return {}

    async def make_request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make HTTP request through Tor with automatic circuit management"""
        if not self.is_connected:
            raise ConnectionError("Not connected to Tor network")

        # Check if circuit rotation is needed
        if await self.should_rotate_circuit():
            await self.rotate_circuit("policy_triggered")

        # Make request with retries
        last_exception = None
        for attempt in range(self.config.max_retries + 1):
            try:
                # Add proxy to kwargs
                kwargs.setdefault("proxy", f"socks5://127.0.0.1:{self.config.socks_port}")

                async with self._session.request(method, url, **kwargs) as response:
                    # Record successful request
                    exit_node = response.headers.get("X-Exit-Node")  # If available
                    self.stats.record_request(True, exit_node)

                    return response

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exception = e
                self.stats.record_request(False)

                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")

                # Rotate circuit on error if configured
                if self.config.circuit_rotation_policy == "errors":
                    await self.rotate_circuit("request_failed")

                # Exponential backoff
                if attempt < self.config.max_retries:
                    wait_time = self.config.backoff_factor**attempt
                    await asyncio.sleep(wait_time)

        # All retries failed
        raise last_exception

    async def get_tor_status(self) -> Dict[str, Any]:
        """Get comprehensive Tor status information"""
        if not self.controller:
            return {"status": "disconnected"}

        try:
            info = self.controller.get_info(
                ["version", "config-file", "traffic/read", "traffic/written"]
            )
            circuit_info = await self.get_circuit_info()

            return {
                "status": "connected" if self.is_connected else "disconnected",
                "version": info.get("version"),
                "config_file": info.get("config-file"),
                "traffic_read": info.get("traffic/read"),
                "traffic_written": info.get("traffic/written"),
                "circuit_info": circuit_info,
                "stats": self.stats.get_stats(),
                "last_health_check": self.last_health_check.isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get Tor status: {e}")
            return {"status": "error", "error": str(e)}

    async def _health_monitor(self):
        """Background health monitoring task"""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                if self.is_connected:
                    health_ok = await self._test_connection()
                    self.last_health_check = datetime.now()

                    if not health_ok:
                        logger.warning("Tor health check failed - attempting circuit rotation")
                        await self.rotate_circuit("health_check_failed")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")

    @staticmethod
    def check_tor_installation() -> Dict[str, Any]:
        """Check if Tor is installed and accessible"""
        result = {
            "tor_binary": False,
            "tor_running": False,
            "control_port_accessible": False,
            "socks_port_accessible": False,
            "recommendations": [],
        }

        try:
            # Check for Tor binary
            subprocess.run(["tor", "--version"], capture_output=True, check=True, timeout=5)
            result["tor_binary"] = True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            result["recommendations"].append("Install Tor: brew install tor")

        # Check if control port is accessible
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            if sock.connect_ex(("127.0.0.1", 9051)) == 0:
                result["control_port_accessible"] = True
            sock.close()
        except:
            result["recommendations"].append("Start Tor with ControlPort 9051")

        # Check if SOCKS port is accessible
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            if sock.connect_ex(("127.0.0.1", 9050)) == 0:
                result["socks_port_accessible"] = True
            sock.close()
        except:
            result["recommendations"].append("Ensure Tor SOCKS proxy on port 9050")

        result["tor_running"] = (
            result["control_port_accessible"] and result["socks_port_accessible"]
        )

        return result

    @staticmethod
    def generate_torrc_config(config: TorConfig) -> str:
        """Generate torrc configuration file content"""
        torrc_lines = [
            "# DeepResearchTool Tor Configuration",
            "",
            f"SocksPort {config.socks_port}",
            f"ControlPort {config.control_port}",
            "",
            "# Security settings",
            "CookieAuthentication 1",
            "CookieAuthFileGroupReadable 1",
            "",
            "# Performance settings for M1",
            "NumCPUs 4",
            "DisableDebuggerAttachment 0",
            "",
            "# Circuit settings",
            "CircuitBuildTimeout 30",
            "NewCircuitPeriod 60",
            "MaxCircuitDirtiness 600",
            "",
            "# Exit policy",
            "ExitRelay 0",
        ]

        if config.data_directory:
            torrc_lines.extend(["", f"DataDirectory {config.data_directory}"])

        if config.strict_exit_policy:
            torrc_lines.extend(
                [
                    "",
                    "# Strict exit policy",
                    "ExitNodes {us},{ca},{gb},{de},{fr},{nl},{se},{ch}",
                    "StrictNodes 1",
                ]
            )

        if config.blocked_countries:
            excluded = ",".join([f"{{{country.lower()}}}" for country in config.blocked_countries])
            torrc_lines.extend(["", f"ExcludeNodes {excluded}", f"ExcludeExitNodes {excluded}"])

        return "\n".join(torrc_lines)


# Utility functions for Tor management
async def test_tor_connection(config: TorConfig = None) -> Dict[str, Any]:
    """Test Tor connection and return status"""
    if config is None:
        config = TorConfig(enable_tor=True)

    async with TorManager(config) as tor:
        if not tor.is_connected:
            return {"connected": False, "error": "Failed to connect"}

        status = await tor.get_tor_status()
        return {"connected": True, "status": status}


def setup_tor_environment(data_dir: Path = None) -> Dict[str, Any]:
    """Setup Tor environment with configuration files"""
    if data_dir is None:
        data_dir = Path.home() / ".deepresearchtool" / "tor"

    data_dir.mkdir(parents=True, exist_ok=True)

    config = TorConfig(data_directory=data_dir, enable_tor=True)

    # Generate torrc
    torrc_content = TorManager.generate_torrc_config(config)
    torrc_path = data_dir / "torrc"

    with open(torrc_path, "w") as f:
        f.write(torrc_content)

    return {
        "data_directory": str(data_dir),
        "torrc_path": str(torrc_path),
        "torrc_content": torrc_content,
        "start_command": f"tor -f {torrc_path}",
    }


__all__ = [
    "TorManager",
    "TorConfig",
    "TorCircuitStats",
    "test_tor_connection",
    "setup_tor_environment",
]
