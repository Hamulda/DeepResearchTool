"""
Smoke test for Phase 2: Tor/I2P Anonymity Layer
Tests basic functionality without requiring actual Tor/I2P installation
"""

import asyncio
import tempfile
import shutil
from pathlib import Path
import pytest
import logging
from unittest.mock import Mock, AsyncMock, patch

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Phase 2 components
from src.deep_web.tor_manager import TorManager, TorConfig, setup_tor_environment
from src.deep_web.i2p_manager import I2PManager, I2PConfig, setup_i2p_environment
from src.deep_web.network_manager import NetworkManager, NetworkConfig, NetworkType
from src.deep_web.hidden_service_scanner import (
    HiddenServiceScanner,
    DiscoveryConfig,
    OnionValidator,
    create_safe_discovery_config,
)
from src.core.memory_optimizer import MemoryOptimizer


class TestPhase2SmokeTest:
    """Smoke tests for Phase 2 implementation"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def memory_optimizer(self):
        """Create memory optimizer instance"""
        return MemoryOptimizer(max_memory_gb=2.0)

    def test_tor_config_generation(self, temp_dir):
        """Test Tor configuration generation"""
        config = TorConfig(
            enable_tor=True, data_directory=temp_dir / "tor_data", strict_exit_policy=True
        )

        # Generate torrc content
        torrc_content = TorManager.generate_torrc_config(config)

        assert "SocksPort 9050" in torrc_content
        assert "ControlPort 9051" in torrc_content
        assert "ExitRelay 0" in torrc_content
        assert str(config.data_directory) in torrc_content

        logger.info("Tor configuration generation test passed")

    def test_tor_installation_check(self):
        """Test Tor installation detection"""
        # This will work regardless of Tor installation status
        installation_status = TorManager.check_tor_installation()

        assert "tor_binary" in installation_status
        assert "tor_running" in installation_status
        assert "control_port_accessible" in installation_status
        assert "socks_port_accessible" in installation_status
        assert "recommendations" in installation_status

        logger.info(f"Tor installation check completed: {installation_status}")

    def test_i2p_installation_check(self):
        """Test I2P installation detection"""
        installation_status = I2PManager.check_i2p_installation()

        assert "i2p_router_running" in installation_status
        assert "sam_bridge_accessible" in installation_status
        assert "i2plib_available" in installation_status
        assert "recommendations" in installation_status

        logger.info(f"I2P installation check completed: {installation_status}")

    def test_network_config_creation(self):
        """Test network configuration creation"""
        config = NetworkConfig(
            enable_clearnet=True,
            enable_tor=False,  # Disabled for testing
            enable_i2p=False,  # Disabled for testing
            respect_robots=True,
            allowlist_only=True,
        )

        assert NetworkType.CLEARNET in config.preferred_networks
        assert NetworkType.TOR not in config.preferred_networks
        assert NetworkType.I2P not in config.preferred_networks
        assert config.respect_robots is True

        logger.info("Network configuration test passed")

    @pytest.mark.asyncio
    async def test_network_manager_clearnet_only(self):
        """Test network manager with clearnet only"""
        config = NetworkConfig(enable_clearnet=True, enable_tor=False, enable_i2p=False)

        async with NetworkManager(config) as network_manager:
            # Test health check
            health_results = await network_manager.health_check()
            assert "clearnet" in health_results

            # Test status
            status = await network_manager.get_network_status()
            assert "config" in status
            assert "health" in status
            assert "statistics" in status

            logger.info(f"Network manager test passed: {status['config']['enabled_networks']}")

    def test_onion_validator(self):
        """Test onion address validation"""
        config = DiscoveryConfig(onion_v3_only=True)
        validator = OnionValidator(config)

        # Test valid v3 onion
        valid_v3 = "3g2upl4pq6kufc4m3g2upl4pq6kufc4m3g2upl4pq6kufc4m3g2upl4p.onion"
        assert validator.is_valid_onion(valid_v3)

        # Test invalid onions
        assert not validator.is_valid_onion("invalid.onion")
        assert not validator.is_valid_onion("short.onion")
        assert not validator.is_valid_onion("not-onion.com")

        # Test onion extraction
        test_text = f"Visit {valid_v3} for privacy. Also check invalid.onion"
        extracted = validator.extract_onions_from_text(test_text)
        assert valid_v3 in extracted
        assert "invalid.onion" not in extracted

        # Test categorization
        research_content = "This is academic research about privacy and security studies"
        category, score = validator.categorize_service(research_content)
        assert category == "research"
        assert score > 0.8

        blocked_content = "Buy illegal drugs and weapons here"
        category, score = validator.categorize_service(blocked_content)
        assert category == "blocked"
        assert score == 0.0

        logger.info("Onion validator test passed")

    def test_discovery_config_creation(self):
        """Test discovery configuration"""
        config = create_safe_discovery_config()

        assert config.max_depth == 1
        assert config.max_total_links == 100
        assert config.legal_categories_only is True
        assert config.onion_v3_only is True
        assert len(config.allowlist_sources) > 0
        assert "illegal" in config.blocked_keywords

        logger.info(
            f"Safe discovery config: {config.max_total_links} max links, depth {config.max_depth}"
        )

    def test_tor_environment_setup(self, temp_dir):
        """Test Tor environment setup"""
        setup_result = setup_tor_environment(temp_dir / "tor_env")

        assert "data_directory" in setup_result
        assert "torrc_path" in setup_result
        assert "torrc_content" in setup_result
        assert "start_command" in setup_result

        # Check if torrc file was created
        torrc_path = Path(setup_result["torrc_path"])
        assert torrc_path.exists()

        with open(torrc_path, "r") as f:
            content = f.read()
            assert "SocksPort 9050" in content
            assert "ControlPort 9051" in content

        logger.info(f"Tor environment setup test passed: {setup_result['data_directory']}")

    def test_i2p_environment_setup(self):
        """Test I2P environment setup"""
        setup_result = setup_i2p_environment()

        assert "installation_status" in setup_result
        assert "configuration_guidance" in setup_result
        assert "recommended_config" in setup_result

        guidance = setup_result["configuration_guidance"]
        assert "i2p_router_config" in guidance
        assert "security_considerations" in guidance

        logger.info("I2P environment setup test passed")

    @pytest.mark.asyncio
    async def test_mock_tor_manager(self):
        """Test Tor manager with mocked connection"""
        config = TorConfig(enable_tor=False)  # Disabled to avoid real connection

        # Mock the stem library availability
        with patch("src.deep_web.tor_manager.STEM_AVAILABLE", True):
            tor_manager = TorManager(config)

            # Test configuration
            assert tor_manager.config.socks_port == 9050
            assert tor_manager.config.control_port == 9051
            assert not tor_manager.is_connected

            # Test stats
            stats = tor_manager.stats.get_stats()
            assert stats["circuits_created"] == 0
            assert stats["total_requests"] == 0

            logger.info("Mock Tor manager test passed")

    @pytest.mark.asyncio
    async def test_mock_i2p_manager(self):
        """Test I2P manager with mocked connection"""
        config = I2PConfig(enable_i2p=False)  # Disabled to avoid real connection

        i2p_manager = I2PManager(config)

        # Test configuration
        assert i2p_manager.config.sam_port == 7656
        assert i2p_manager.config.session_name == "DeepResearchTool"

        # Test session info
        session_info = i2p_manager.session.get_session_info()
        assert session_info["is_connected"] is False
        assert session_info["session_name"] == "DeepResearchTool"

        logger.info("Mock I2P manager test passed")

    @pytest.mark.asyncio
    async def test_hidden_service_scanner_mock(self, memory_optimizer, temp_dir):
        """Test hidden service scanner with mocked network"""
        config = DiscoveryConfig(max_depth=1, max_total_links=5, legal_categories_only=True)

        # Mock network manager
        mock_network_manager = Mock()
        mock_response = Mock()
        mock_response.status = 200
        mock_response.text = AsyncMock(
            return_value="""
            <html>
            <title>Test Onion Site</title>
            <meta name="description" content="Research site for testing">
            <body>
                <p>This is a test research site about privacy</p>
                <a href="http://another3g2upl4pq6kufc4m3g2upl4pq6kufc4m3g2upl4pq6kufc4m3g2u.onion">Link</a>
            </body>
            </html>
        """
        )
        mock_network_manager.get = AsyncMock(return_value=mock_response)
        mock_network_manager.__aenter__ = AsyncMock(return_value=mock_response)
        mock_network_manager.__aexit__ = AsyncMock(return_value=None)

        scanner = HiddenServiceScanner(config, mock_network_manager, memory_optimizer, temp_dir)

        # Test URL safety check
        safe_url = "http://3g2upl4pq6kufc4m3g2upl4pq6kufc4m3g2upl4pq6kufc4m3g2upl4p.onion"
        assert scanner._is_safe_url(safe_url)

        # Test stats
        stats = scanner.get_discovery_stats()
        assert stats["total_services"] == 0
        assert stats["total_visited"] == 0

        logger.info("Hidden service scanner mock test passed")


# Offline smoke test function for CI/local testing
def run_smoke_test_offline():
    """Run offline smoke test without network dependencies"""
    import sys

    try:
        # Test Tor configuration
        config = TorConfig(enable_tor=False)
        torrc_content = TorManager.generate_torrc_config(config)
        assert "SocksPort 9050" in torrc_content

        # Test I2P configuration
        i2p_config = I2PConfig(enable_i2p=False)
        assert i2p_config.sam_port == 7656

        # Test network configuration
        network_config = NetworkConfig(enable_clearnet=True, enable_tor=False, enable_i2p=False)
        assert NetworkType.CLEARNET in network_config.preferred_networks

        # Test onion validation
        validator = OnionValidator(DiscoveryConfig())
        valid_onion = "3g2upl4pq6kufc4m3g2upl4pq6kufc4m3g2upl4pq6kufc4m3g2upl4p.onion"
        assert validator.is_valid_onion(valid_onion)

        # Test discovery config
        discovery_config = create_safe_discovery_config()
        assert discovery_config.legal_categories_only is True

        print("✅ Phase 2 offline smoke test PASSED")
        return True

    except Exception as e:
        print(f"❌ Phase 2 offline smoke test FAILED: {e}")
        return False


if __name__ == "__main__":
    # Run offline smoke test when called directly
    success = run_smoke_test_offline()
    sys.exit(0 if success else 1)
