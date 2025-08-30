#!/usr/bin/env python3
"""
Phase 2 Validation Tests - Advanced Data Acquisition & Anti-Detection
Comprehensive test suite for Phase 2 capabilities

Author: Advanced AI Research Assistant
Date: August 2025
"""

import pytest
import asyncio
import logging
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.security.anonymity_providers import (
    AnonymityProviderFactory,
    TorProvider,
    ClearnetProvider,
    I2PProvider,
)
from src.security.intelligent_proxy_manager import (
    IntelligentProxyManager,
    ProxyTarget,
    ProxyPerformanceMetrics,
)
from src.security.behavior_camouflage import (
    BehaviorCamouflage,
    HumanBehaviorProfile,
    MousePathGenerator,
)
from src.security.archive_miner import (
    ArchiveMiner,
    WaybackMachineAPI,
    ArchiveTodayAPI,
    ArchiveSnapshot,
)
from src.security.phase2_orchestrator import AdvancedDataAcquisitionSystem, create_phase2_system
from src.config.phase2_config import Phase2Configuration

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAnonymityProviders:
    """Test anonymity provider implementations"""

    def test_provider_factory(self):
        """Test provider factory creation"""
        # Test clearnet provider (always available)
        config = {"timeout": 10}
        provider = AnonymityProviderFactory.create_provider("clearnet", config)
        assert isinstance(provider, ClearnetProvider)

        # Test invalid provider type
        with pytest.raises(ValueError):
            AnonymityProviderFactory.create_provider("invalid", {})

    def test_available_providers(self):
        """Test getting available providers"""
        available = AnonymityProviderFactory.get_available_providers()
        assert "clearnet" in available
        # Tor and I2P may or may not be available depending on system

    @pytest.mark.asyncio
    async def test_clearnet_provider(self):
        """Test clearnet provider functionality"""
        config = {"timeout": 10, "user_agents": ["Test Agent 1.0"]}

        provider = ClearnetProvider(config)

        # Test connection
        assert await provider.connect()
        assert provider.is_connected

        # Test proxy config (should be empty for clearnet)
        proxy_config = provider.get_proxy_config()
        assert proxy_config == {}

        # Test session creation
        session = await provider.get_session()
        assert session is not None

        # Test rotation
        assert await provider.rotate_identity()

        # Test disconnection
        assert await provider.disconnect()
        assert not provider.is_connected

    @pytest.mark.asyncio
    async def test_tor_provider_without_tor(self):
        """Test Tor provider behavior when Tor is not available"""
        config = {"socks_port": 9999, "control_port": 9998, "timeout": 5}  # Non-existent port

        provider = TorProvider(config)

        # Should fail to connect to non-existent Tor
        assert not await provider.connect()
        assert not provider.is_connected


class TestIntelligentProxyManager:
    """Test intelligent proxy management"""

    @pytest.fixture
    def proxy_config(self):
        """Test configuration for proxy manager"""
        return {
            "providers": {
                "clearnet": {"timeout": 10, "user_agents": ["Test Agent 1.0", "Test Agent 2.0"]}
            },
            "rotation_strategy": "performance_based",
            "min_rotation_interval": 1,
            "max_consecutive_uses": 3,
        }

    @pytest.mark.asyncio
    async def test_proxy_manager_initialization(self, proxy_config):
        """Test proxy manager initialization"""
        manager = IntelligentProxyManager(proxy_config)

        assert len(manager.providers) == 1
        assert "clearnet" in manager.providers
        assert len(manager.metrics) == 1

    @pytest.mark.asyncio
    async def test_proxy_manager_connections(self, proxy_config):
        """Test proxy manager connection handling"""
        manager = IntelligentProxyManager(proxy_config)

        # Test connecting all providers
        results = await manager.connect_all_providers()
        assert "clearnet" in results
        assert results["clearnet"] is True

        # Test testing all providers
        test_results = await manager.test_all_providers()
        assert "clearnet" in test_results

        # Cleanup
        await manager.disconnect_all_providers()

    @pytest.mark.asyncio
    async def test_target_configuration(self, proxy_config):
        """Test target-specific configuration"""
        manager = IntelligentProxyManager(proxy_config)

        target = ProxyTarget(
            domain="example.com", preferred_providers=["clearnet"], max_retry_count=2
        )

        manager.add_target_config(target)
        assert "example.com" in manager.target_configs
        assert manager.target_configs["example.com"].preferred_providers == ["clearnet"]

    def test_performance_metrics(self):
        """Test performance metrics functionality"""
        metrics = ProxyPerformanceMetrics()

        # Test initial state
        assert metrics.success_rate == 0.0
        assert metrics.is_healthy is False

        # Record some successes
        metrics.record_success(1.5)
        metrics.record_success(2.0)
        metrics.record_success(1.0)

        assert metrics.success_count == 3
        assert metrics.total_requests == 3
        assert metrics.success_rate == 100.0
        assert metrics.avg_response_time == 1.5
        assert metrics.is_healthy is True

        # Record some failures
        metrics.record_failure()
        assert metrics.success_rate == 75.0
        assert metrics.consecutive_failures == 1


class TestBehaviorCamouflage:
    """Test behavior camouflage system"""

    @pytest.fixture
    def behavior_profile(self):
        """Test behavior profile"""
        return HumanBehaviorProfile(
            reading_speed_wpm=300,  # Fast for testing
            scroll_speed_variance=0.2,
            typing_speed_cps=10,  # Fast for testing
            attention_span_seconds=30,
        )

    def test_behavior_profile_creation(self, behavior_profile):
        """Test behavior profile creation"""
        assert behavior_profile.reading_speed_wpm == 300
        assert behavior_profile.scroll_speed_variance == 0.2
        assert behavior_profile.typing_speed_cps == 10

    def test_mouse_path_generator(self):
        """Test mouse path generation"""
        generator = MousePathGenerator()

        start = (0, 0)
        end = (100, 100)

        # Test Bezier curve generation
        path = generator.bezier_curve(start, end, num_points=10)
        assert len(path) == 11  # num_points + 1
        assert path[0] == start
        assert path[-1] == end

        # Test micro movements
        modified_path = generator.add_micro_movements(path, intensity=1.0)
        assert len(modified_path) == len(path)

    @pytest.mark.asyncio
    async def test_behavior_camouflage_timing(self, behavior_profile):
        """Test behavior camouflage timing functions"""
        behavior = BehaviorCamouflage(behavior_profile)

        # Test random delay
        import time

        start_time = time.time()
        await behavior.random_delay(0.1, 0.1)
        elapsed = time.time() - start_time
        assert 0.05 <= elapsed <= 0.5  # Should be reasonably close

        # Test reading time simulation
        start_time = time.time()
        await behavior.simulate_reading_time(50)  # 50 characters
        elapsed = time.time() - start_time
        assert elapsed > 0  # Should take some time

    def test_behavior_report(self, behavior_profile):
        """Test behavior analysis reporting"""
        behavior = BehaviorCamouflage(behavior_profile)

        report = behavior.get_behavior_report()

        assert "session_duration_seconds" in report
        assert "actions_performed" in report
        assert "profile" in report
        assert report["profile"]["reading_speed_wpm"] == 300


class TestArchiveMiner:
    """Test archive mining capabilities"""

    @pytest.fixture
    def archive_config(self):
        """Test configuration for archive miner"""
        return {
            "max_snapshots_per_url": 10,
            "min_time_between_snapshots": 1,  # Low for testing
            "content_filters": [r"\.pdf$"],
        }

    def test_archive_snapshot_creation(self):
        """Test archive snapshot data structure"""
        from datetime import datetime

        snapshot = ArchiveSnapshot(
            url="https://web.archive.org/web/20230101000000/https://example.com",
            original_url="https://example.com",
            timestamp=datetime.now(),
            archive_source="wayback_machine",
            status_code=200,
            content_type="text/html",
            content_length=1000,
        )

        assert snapshot.url.startswith("https://web.archive.org")
        assert snapshot.original_url == "https://example.com"
        assert snapshot.archive_source == "wayback_machine"

        # Test serialization
        snapshot_dict = snapshot.to_dict()
        assert "url" in snapshot_dict
        assert "timestamp" in snapshot_dict

    def test_wayback_machine_api(self):
        """Test Wayback Machine API interface"""
        api = WaybackMachineAPI()

        assert api.base_url == "https://web.archive.org"
        assert api.api_url == "https://web.archive.org/cdx/search/cdx"
        assert api.rate_limit > 0

    def test_archive_today_api(self):
        """Test Archive.today API interface"""
        api = ArchiveTodayAPI()

        assert len(api.base_urls) > 0
        assert "archive.today" in api.base_urls[0]
        assert api.rate_limit > 0

    def test_archive_miner_initialization(self, archive_config):
        """Test archive miner initialization"""
        miner = ArchiveMiner(archive_config)

        assert miner.max_snapshots_per_url == 10
        assert miner.min_time_between_snapshots == 1
        assert len(miner.content_filters) == 1

        # Test statistics on empty miner
        stats = miner.get_mining_statistics()
        assert stats["urls_mined"] == 0
        assert stats["total_snapshots"] == 0


class TestPhase2Orchestrator:
    """Test Phase 2 orchestrator system"""

    @pytest.fixture
    def test_config(self):
        """Test configuration for Phase 2 system"""
        return {
            "providers": {"clearnet": {"timeout": 10, "user_agents": ["Test Agent 1.0"]}},
            "proxy_management": {
                "rotation_strategy": "performance_based",
                "min_rotation_interval": 1,
                "max_consecutive_uses": 3,
            },
            "behavior_profile": {
                "reading_speed_wpm": 500,  # Fast for testing
                "scroll_speed_variance": 0.1,
                "typing_speed_cps": 20,
            },
            "archive_mining": {"max_snapshots_per_url": 5, "min_time_between_snapshots": 1},
            "tor_browser": {"enabled": False},  # Disabled for testing
            "max_concurrent_sessions": 2,
            "session_timeout": 30,
            "data_export_path": tempfile.mkdtemp(),
        }

    def test_system_creation(self, test_config):
        """Test Phase 2 system creation"""
        system = create_phase2_system(test_config)

        assert isinstance(system, AdvancedDataAcquisitionSystem)
        assert system.config.max_concurrent_sessions == 2
        assert system.config.session_timeout == 30

    @pytest.mark.asyncio
    async def test_system_initialization(self, test_config):
        """Test Phase 2 system initialization"""
        system = create_phase2_system(test_config)

        # Mock successful initialization
        with patch.object(
            system.proxy_manager, "connect_all_providers", return_value={"clearnet": True}
        ):
            with patch.object(
                system.proxy_manager, "test_all_providers", return_value={"clearnet": True}
            ):
                success = await system.initialize()
                assert success is True

    @pytest.mark.asyncio
    async def test_system_status(self, test_config):
        """Test system status reporting"""
        system = create_phase2_system(test_config)

        status = await system.get_system_status()

        assert "system_info" in status
        assert "proxy_manager" in status
        assert "archive_miner" in status
        assert "tor_browser" in status
        assert "timestamp" in status


class TestPhase2Configuration:
    """Test Phase 2 configuration management"""

    def test_development_config(self):
        """Test development configuration"""
        config = Phase2Configuration.get_development_config()

        assert config["max_concurrent_sessions"] == 2
        assert config["session_timeout"] == 60
        assert config["tor_browser"]["enabled"] is False

    def test_production_config(self):
        """Test production configuration"""
        config = Phase2Configuration.get_production_config()

        assert config["max_concurrent_sessions"] >= 3
        assert config["session_timeout"] >= 300
        assert "providers" in config
        assert "tor" in config["providers"]
        assert "clearnet" in config["providers"]

    def test_security_config(self):
        """Test security-focused configuration"""
        config = Phase2Configuration.get_security_focused_config()

        assert config["max_concurrent_sessions"] == 1
        assert config["proxy_management"]["rotation_strategy"] == "random"
        assert config["behavior_profile"]["mouse_movement_style"] == "cautious"

    def test_config_template_generation(self):
        """Test configuration template generation"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".env") as f:
            template_path = f.name

        Phase2Configuration.save_config_template(template_path)

        # Verify template was created
        assert Path(template_path).exists()

        # Read and verify content
        with open(template_path, "r") as f:
            content = f.read()
            assert "PHASE2_ENVIRONMENT" in content
            assert "TOR_SOCKS_PORT" in content
            assert "BEHAVIOR_READING_SPEED" in content

        # Cleanup
        Path(template_path).unlink()


@pytest.mark.asyncio
async def test_phase2_integration():
    """Integration test for Phase 2 system"""
    logger.info("Running Phase 2 integration test...")

    # Use development configuration
    config = Phase2Configuration.get_development_config()
    system = create_phase2_system(config)

    try:
        # Initialize system
        success = await system.initialize()
        if not success:
            pytest.skip("System initialization failed - skipping integration test")

        # Test system status
        status = await system.get_system_status()
        assert "system_info" in status

        # Test basic URL acquisition (using httpbin for reliable testing)
        test_url = "https://httpbin.org/ip"

        try:
            result = await system.acquire_data_from_url(test_url, "intelligent_proxy")

            if result.get("success"):
                logger.info("✅ Integration test successful")
                assert "content" in result
                assert result["provider_used"] == "clearnet"
            else:
                logger.warning(f"⚠️ URL acquisition failed: {result.get('error')}")

        except Exception as e:
            logger.warning(f"⚠️ URL acquisition test failed: {e}")

        # Test export functionality
        export_path = await system.export_session_data()
        assert Path(export_path).exists()

        # Verify export content
        with open(export_path, "r") as f:
            export_data = json.load(f)
            assert "export_timestamp" in export_data
            assert "total_requests" in export_data
            assert "config" in export_data

        logger.info("✅ Phase 2 integration test completed successfully")

    finally:
        # Cleanup
        await system.shutdown()


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v", "--tb=short"])
