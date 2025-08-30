"""I2P Manager for Alternative Anonymous Network Access
SAM (Simple Anonymous Messaging) interface integration
Optional secondary anonymity layer when Tor is compromised
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
import logging
import socket
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

try:
    import i2plib

    I2P_LIB_AVAILABLE = True
except ImportError:
    I2P_LIB_AVAILABLE = False
    logger.warning("i2plib not available - I2P functionality disabled")


@dataclass
class I2PConfig:
    """I2P configuration settings"""

    sam_address: str = "127.0.0.1"
    sam_port: int = 7656
    session_name: str = "DeepResearchTool"
    session_type: str = "STREAM"  # STREAM, DATAGRAM, RAW

    # Connection settings
    connection_timeout: int = 30
    read_timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 2.0

    # Session management
    session_keepalive: bool = True
    auto_reconnect: bool = True
    max_session_duration: int = 3600  # 1 hour

    # Safety settings
    enable_i2p: bool = False  # Opt-in by default
    strict_tunnels: bool = True
    tunnel_length: int = 3
    tunnel_quantity: int = 2


class I2PSession:
    """I2P SAM session wrapper"""

    def __init__(self, config: I2PConfig):
        self.config = config
        self.session = None
        self.is_connected = False
        self.session_id = None
        self.created_at = None
        self._lock = asyncio.Lock()

    async def connect(self) -> bool:
        """Connect to I2P SAM bridge"""
        if not self.config.enable_i2p:
            logger.info("I2P disabled by configuration")
            return False

        if not I2P_LIB_AVAILABLE:
            logger.error("i2plib not available")
            return False

        async with self._lock:
            try:
                # Create SAM session
                self.session = await i2plib.create_session(
                    sam_address=self.config.sam_address,
                    sam_port=self.config.sam_port,
                    session_name=self.config.session_name,
                    session_type=self.config.session_type,
                )

                self.session_id = self.session.session_id
                self.created_at = datetime.now()
                self.is_connected = True

                logger.info(f"I2P session created: {self.session_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to create I2P session: {e}")
                return False

    async def disconnect(self):
        """Disconnect I2P session"""
        async with self._lock:
            if self.session:
                try:
                    await self.session.close()
                except Exception as e:
                    logger.warning(f"Error closing I2P session: {e}")

                self.session = None
                self.is_connected = False
                self.session_id = None
                logger.info("I2P session closed")

    async def create_stream(self, destination: str) -> asyncio.StreamReader | None:
        """Create I2P stream to destination"""
        if not self.is_connected or not self.session:
            raise ConnectionError("I2P session not connected")

        try:
            stream = await self.session.create_stream(destination)
            return stream
        except Exception as e:
            logger.error(f"Failed to create I2P stream to {destination}: {e}")
            return None

    def get_session_info(self) -> dict[str, Any]:
        """Get session information"""
        return {
            "session_id": self.session_id,
            "is_connected": self.is_connected,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "session_name": self.config.session_name,
            "sam_address": f"{self.config.sam_address}:{self.config.sam_port}",
        }


class I2PManager:
    """I2P network manager with HTTP client integration"""

    def __init__(self, config: I2PConfig):
        self.config = config
        self.session = I2PSession(config)
        self._http_session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()

    async def connect(self) -> bool:
        """Connect to I2P network"""
        if not self.config.enable_i2p:
            logger.info("I2P disabled - skipping connection")
            return False

        # Connect I2P session
        session_ok = await self.session.connect()
        if not session_ok:
            return False

        # Setup HTTP session with I2P transport
        try:
            connector = I2PConnector(self.session)
            timeout = aiohttp.ClientTimeout(total=self.config.connection_timeout)

            self._http_session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0 (compatible; DeepResearchTool-I2P/2.0)"},
            )

            logger.info("I2P HTTP client initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to setup I2P HTTP client: {e}")
            await self.session.disconnect()
            return False

    async def disconnect(self):
        """Disconnect from I2P network"""
        if self._http_session:
            await self._http_session.close()
            self._http_session = None

        await self.session.disconnect()

    async def make_request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make HTTP request through I2P network"""
        if not self.session.is_connected or not self._http_session:
            raise ConnectionError("I2P not connected")

        # Validate I2P destination format
        if not self._is_valid_i2p_destination(url):
            raise ValueError(f"Invalid I2P destination: {url}")

        # Make request with retries
        last_exception = None
        for attempt in range(self.config.max_retries + 1):
            try:
                async with self._http_session.request(method, url, **kwargs) as response:
                    return response

            except Exception as e:
                last_exception = e
                logger.warning(f"I2P request failed (attempt {attempt + 1}): {e}")

                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay)

        raise last_exception

    def _is_valid_i2p_destination(self, url: str) -> bool:
        """Validate I2P destination format"""
        if not url:
            return False

        # Check for .i2p domain
        if ".i2p" in url:
            return True

        # Check for base64 destination
        if len(url) > 500 and url.replace("=", "").replace("+", "").replace("/", "").isalnum():
            return True

        return False

    async def resolve_i2p_address(self, address: str) -> str | None:
        """Resolve I2P address to base64 destination"""
        if not self.session.is_connected:
            return None

        try:
            # This would typically use I2P's naming service
            # For now, return as-is for .i2p addresses
            if address.endswith(".i2p"):
                return address

            return address

        except Exception as e:
            logger.error(f"Failed to resolve I2P address {address}: {e}")
            return None

    async def get_i2p_status(self) -> dict[str, Any]:
        """Get I2P network status"""
        status = {
            "session": self.session.get_session_info(),
            "network_connected": False,
            "tunnel_count": 0,
            "bandwidth_in": 0,
            "bandwidth_out": 0,
        }

        if self.session.is_connected:
            try:
                # Get network status from I2P router
                # This would require additional I2P API calls
                status["network_connected"] = True

            except Exception as e:
                logger.error(f"Failed to get I2P status: {e}")
                status["error"] = str(e)

        return status

    @staticmethod
    def check_i2p_installation() -> dict[str, Any]:
        """Check I2P installation and accessibility"""
        result = {
            "i2p_router_running": False,
            "sam_bridge_accessible": False,
            "i2plib_available": I2P_LIB_AVAILABLE,
            "recommendations": [],
        }

        if not I2P_LIB_AVAILABLE:
            result["recommendations"].append("Install i2plib: pip install i2plib")

        # Check SAM bridge accessibility
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            if sock.connect_ex(("127.0.0.1", 7656)) == 0:
                result["sam_bridge_accessible"] = True
                result["i2p_router_running"] = True
            sock.close()
        except Exception:
            result["recommendations"].extend(
                [
                    "Install I2P router: https://geti2p.net/en/download",
                    "Enable SAM bridge in I2P configuration",
                    "Start I2P router and wait for network integration",
                ]
            )

        return result


class I2PConnector(aiohttp.BaseConnector):
    """Custom aiohttp connector for I2P streams"""

    def __init__(self, i2p_session: I2PSession):
        super().__init__()
        self.i2p_session = i2p_session

    async def _create_connection(self, req, traces, timeout):
        """Create I2P stream connection"""
        if not self.i2p_session.is_connected:
            raise ConnectionError("I2P session not connected")

        # Extract destination from URL
        destination = self._extract_destination(req.url)

        # Create I2P stream
        stream = await self.i2p_session.create_stream(destination)
        if not stream:
            raise ConnectionError(f"Failed to create I2P stream to {destination}")

        # Create transport and protocol
        transport = I2PTransport(stream)
        protocol = aiohttp.client_proto.ResponseHandler()

        return transport, protocol

    def _extract_destination(self, url) -> str:
        """Extract I2P destination from URL"""
        host = url.host
        if host.endswith(".i2p"):
            return host

        # For base64 destinations, return as-is
        return host


class I2PTransport:
    """I2P stream transport for aiohttp"""

    def __init__(self, stream):
        self.stream = stream
        self._closed = False

    def write(self, data):
        """Write data to I2P stream"""
        if not self._closed:
            asyncio.create_task(self.stream.write(data))

    def close(self):
        """Close I2P stream"""
        self._closed = True
        if hasattr(self.stream, "close"):
            asyncio.create_task(self.stream.close())

    def is_closing(self):
        """Check if transport is closing"""
        return self._closed


# Utility functions for I2P management
async def test_i2p_connection(config: I2PConfig = None) -> dict[str, Any]:
    """Test I2P connection and return status"""
    if config is None:
        config = I2PConfig(enable_i2p=True)

    try:
        async with I2PManager(config) as i2p:
            if not i2p.session.is_connected:
                return {"connected": False, "error": "Failed to connect to I2P"}

            status = await i2p.get_i2p_status()
            return {"connected": True, "status": status}

    except Exception as e:
        return {"connected": False, "error": str(e)}


def setup_i2p_environment() -> dict[str, Any]:
    """Setup I2P environment with configuration guidance"""
    installation_check = I2PManager.check_i2p_installation()

    config_guidance = {
        "i2p_router_config": {
            "sam_bridge": "Enable SAM bridge on port 7656",
            "bandwidth": "Configure appropriate bandwidth limits",
            "tunnels": "Set tunnel length to 3 hops minimum",
            "participants": "Allow tunnel participation for network health",
        },
        "security_considerations": [
            "I2P provides different anonymity properties than Tor",
            "Best used as secondary network for redundancy",
            "Requires patience for tunnel building and network integration",
            "Consider traffic analysis risks with timing correlation",
        ],
    }

    return {
        "installation_status": installation_check,
        "configuration_guidance": config_guidance,
        "recommended_config": I2PConfig(
            enable_i2p=True, strict_tunnels=True, tunnel_length=3
        ).__dict__,
    }


__all__ = ["I2PConfig", "I2PManager", "I2PSession", "setup_i2p_environment", "test_i2p_connection"]
