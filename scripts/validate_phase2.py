#!/usr/bin/env python3
"""
Phase 2 Validation Script
Simple validation of Tor/I2P anonymity layer without external dependencies
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def validate_phase2():
    """Validate Phase 2 implementation"""
    print("🚀 Phase 2 Validation: Tor/I2P Anonymity Layer")
    print("=" * 50)

    results = {
        "tor_manager": False,
        "i2p_manager": False,
        "network_manager": False,
        "hidden_service_scanner": False,
        "integration": False,
    }

    # Test 1: Tor Manager
    try:
        from src.deep_web.tor_manager import TorManager, TorConfig

        config = TorConfig(enable_tor=False)
        torrc_content = TorManager.generate_torrc_config(config)

        assert "SocksPort 9050" in torrc_content
        assert "ControlPort 9051" in torrc_content

        installation_check = TorManager.check_tor_installation()
        assert "tor_binary" in installation_check

        results["tor_manager"] = True
        print("✅ TorManager: Configuration generation and installation check")

    except Exception as e:
        print(f"❌ TorManager failed: {e}")

    # Test 2: I2P Manager
    try:
        from src.deep_web.i2p_manager import I2PManager, I2PConfig

        config = I2PConfig(enable_i2p=False)
        installation_check = I2PManager.check_i2p_installation()
        assert "i2p_router_running" in installation_check

        results["i2p_manager"] = True
        print("✅ I2PManager: Configuration and installation check")

    except Exception as e:
        print(f"❌ I2PManager failed: {e}")

    # Test 3: Network Manager
    try:
        from src.deep_web.network_manager import NetworkManager, NetworkConfig, NetworkType

        config = NetworkConfig(enable_clearnet=True, enable_tor=False, enable_i2p=False)

        assert NetworkType.CLEARNET in config.preferred_networks
        assert len(config.preferred_networks) == 1

        results["network_manager"] = True
        print("✅ NetworkManager: Configuration and network selection")

    except Exception as e:
        print(f"❌ NetworkManager failed: {e}")

    # Test 4: Hidden Service Scanner
    try:
        from src.deep_web.hidden_service_scanner import (
            OnionValidator,
            DiscoveryConfig,
            create_safe_discovery_config,
        )

        config = DiscoveryConfig(onion_v3_only=True)
        validator = OnionValidator(config)

        # Test valid onion v3
        valid_onion = "3g2upl4pq6kufc4m3g2upl4pq6kufc4m3g2upl4pq6kufc4m3g2upl4p.onion"
        assert validator.is_valid_onion(valid_onion)

        # Test invalid onion
        assert not validator.is_valid_onion("invalid.onion")

        # Test categorization
        research_content = "Academic research on privacy and security"
        category, score = validator.categorize_service(research_content)
        assert category == "research"
        assert score > 0.8

        # Test safe config
        safe_config = create_safe_discovery_config()
        assert safe_config.legal_categories_only is True
        assert safe_config.max_depth == 1

        results["hidden_service_scanner"] = True
        print("✅ HiddenServiceScanner: Onion validation and content categorization")

    except Exception as e:
        print(f"❌ HiddenServiceScanner failed: {e}")

    # Test 5: Integration
    try:
        from src.deep_web import (
            TOR_AVAILABLE,
            I2P_AVAILABLE,
            NETWORK_MANAGER_AVAILABLE,
            SCANNER_AVAILABLE,
        )

        # At least network manager should be available
        integration_ok = NETWORK_MANAGER_AVAILABLE and SCANNER_AVAILABLE

        if integration_ok:
            results["integration"] = True
            print("✅ Integration: Module imports and availability flags")
        else:
            print("❌ Integration: Some modules not available")

    except Exception as e:
        print(f"❌ Integration failed: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("📊 Validation Summary:")

    passed = sum(results.values())
    total = len(results)

    for component, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {component.replace('_', ' ').title()}")

    success_rate = passed / total
    print(f"\n🎯 Success Rate: {passed}/{total} ({success_rate:.1%})")

    if success_rate >= 0.8:
        print("🎉 Phase 2 validation PASSED!")
        return True
    else:
        print("⚠️ Phase 2 validation needs attention")
        return False


if __name__ == "__main__":
    success = validate_phase2()
    sys.exit(0 if success else 1)
