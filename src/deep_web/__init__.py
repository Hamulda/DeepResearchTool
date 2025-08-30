"""Deep Web Module Package
Tor/I2P integration and hidden service discovery for privacy-preserving research
"""

# Import with error handling for missing dependencies
try:
    from .tor_manager import TorConfig, TorManager

    TOR_AVAILABLE = True
except ImportError:
    TOR_AVAILABLE = False
    TorManager = None
    TorConfig = None

try:
    from .i2p_manager import I2PConfig, I2PManager

    I2P_AVAILABLE = True
except ImportError:
    I2P_AVAILABLE = False
    I2PManager = None
    I2PConfig = None

try:
    from .network_manager import NetworkConfig, NetworkManager

    NETWORK_MANAGER_AVAILABLE = True
except ImportError:
    NETWORK_MANAGER_AVAILABLE = False
    NetworkManager = None
    NetworkConfig = None

try:
    from .hidden_service_scanner import DiscoveryConfig, HiddenServiceScanner

    SCANNER_AVAILABLE = True
except ImportError:
    SCANNER_AVAILABLE = False
    HiddenServiceScanner = None
    DiscoveryConfig = None

__all__ = [
    "I2P_AVAILABLE",
    "NETWORK_MANAGER_AVAILABLE",
    "SCANNER_AVAILABLE",
    "TOR_AVAILABLE",
    "DiscoveryConfig",
    "HiddenServiceScanner",
    "I2PConfig",
    "I2PManager",
    "NetworkConfig",
    "NetworkManager",
    "TorConfig",
    "TorManager",
]
