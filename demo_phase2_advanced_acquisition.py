#!/usr/bin/env python3
"""
Phase 2 Demo - Advanced Data Acquisition & Anti-Detection
Comprehensive demonstration of all Phase 2 capabilities

Author: Advanced AI Research Assistant
Date: August 2025
"""

import asyncio
import logging
import json
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.security.phase2_orchestrator import create_phase2_system, EXAMPLE_PHASE2_CONFIG
from src.security.intelligent_proxy_manager import ProxyTarget

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("phase2_demo.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


async def demo_anonymity_providers():
    """Demonstrate anonymity provider functionality"""
    logger.info("=== Testing Anonymity Providers ===")

    from src.security.anonymity_providers import AnonymityProviderFactory

    # Test clearnet provider (always available)
    try:
        clearnet_config = {
            "timeout": 10,
            "user_agents": ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"],
        }

        clearnet = AnonymityProviderFactory.create_provider("clearnet", clearnet_config)

        if await clearnet.connect():
            logger.info("✅ Clearnet provider connected successfully")

            if await clearnet.test_connection():
                logger.info("✅ Clearnet connection test passed")
            else:
                logger.warning("⚠️ Clearnet connection test failed")

            await clearnet.disconnect()
        else:
            logger.error("❌ Clearnet provider connection failed")

    except Exception as e:
        logger.error(f"❌ Clearnet provider error: {e}")


async def demo_intelligent_proxy_manager():
    """Demonstrate intelligent proxy management"""
    logger.info("=== Testing Intelligent Proxy Manager ===")

    from src.security.intelligent_proxy_manager import IntelligentProxyManager, ProxyTarget

    config = {
        "providers": {
            "clearnet": {
                "timeout": 10,
                "user_agents": [
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                ],
            }
        },
        "rotation_strategy": "performance_based",
        "min_rotation_interval": 5,
        "max_consecutive_uses": 3,
    }

    try:
        proxy_manager = IntelligentProxyManager(config)

        # Connect providers
        connection_results = await proxy_manager.connect_all_providers()
        logger.info(f"Provider connections: {connection_results}")

        # Test all providers
        test_results = await proxy_manager.test_all_providers()
        logger.info(f"Provider tests: {test_results}")

        # Add target configuration
        target = ProxyTarget(
            domain="httpbin.org", preferred_providers=["clearnet"], max_retry_count=2
        )
        proxy_manager.add_target_config(target)

        # Test session acquisition
        test_url = "https://httpbin.org/ip"
        try:
            session, provider = await proxy_manager.get_session_for_url(test_url)
            logger.info(f"✅ Got session for {test_url} using {provider}")

            # Test request
            import time

            start_time = time.time()
            async with session.get(test_url) as response:
                response_time = time.time() - start_time
                if response.status == 200:
                    data = await response.json()
                    logger.info(
                        f"✅ Request successful: {data.get('origin')} (⏱️ {response_time:.2f}s)"
                    )

                    # Record success
                    await proxy_manager.record_request_result(
                        test_url, provider, True, response_time
                    )
                else:
                    logger.warning(f"⚠️ Request failed with status {response.status}")
        except Exception as e:
            logger.error(f"❌ Session test failed: {e}")

        # Get performance report
        report = proxy_manager.get_performance_report()
        logger.info(f"Performance report: {json.dumps(report, indent=2)}")

        await proxy_manager.disconnect_all_providers()

    except Exception as e:
        logger.error(f"❌ Proxy manager error: {e}")


async def demo_behavior_camouflage():
    """Demonstrate behavior camouflage system"""
    logger.info("=== Testing Behavior Camouflage ===")

    from src.security.behavior_camouflage import BehaviorCamouflage, HumanBehaviorProfile

    try:
        # Create behavior profile
        profile = HumanBehaviorProfile(
            reading_speed_wpm=180,
            scroll_speed_variance=0.4,
            typing_speed_cps=3,
            mouse_movement_style="natural",
        )

        behavior = BehaviorCamouflage(profile)

        # Test timing functions
        logger.info("Testing human-like delays...")
        start_time = asyncio.get_event_loop().time()
        await behavior.random_delay(1.0, 0.2)
        actual_delay = asyncio.get_event_loop().time() - start_time
        logger.info(f"✅ Random delay: {actual_delay:.2f}s")

        # Test reading time simulation
        test_text = "This is a test article with some content to read. " * 10
        start_time = asyncio.get_event_loop().time()
        await behavior.simulate_reading_time(len(test_text))
        reading_time = asyncio.get_event_loop().time() - start_time
        logger.info(f"✅ Reading time for {len(test_text)} chars: {reading_time:.2f}s")

        # Test break logic
        should_break = await behavior.should_take_break()
        logger.info(f"✅ Should take break: {should_break}")

        # Get behavior report
        report = behavior.get_behavior_report()
        logger.info(f"Behavior report: {json.dumps(report, indent=2)}")

    except Exception as e:
        logger.error(f"❌ Behavior camouflage error: {e}")


async def demo_archive_miner():
    """Demonstrate archive mining capabilities"""
    logger.info("=== Testing Archive Miner ===")

    from src.security.archive_miner import ArchiveMiner

    config = {"max_snapshots_per_url": 20, "min_time_between_snapshots": 30, "content_filters": []}

    try:
        miner = ArchiveMiner(config)

        # Test URL (use a well-archived site)
        test_url = "https://example.com"

        logger.info(f"Mining archive history for {test_url}...")
        snapshots = await miner.mine_url_history(test_url)

        if snapshots:
            logger.info(f"✅ Found {len(snapshots)} archive snapshots")

            # Show sample snapshots
            for i, snapshot in enumerate(snapshots[:3]):
                logger.info(f"  Snapshot {i+1}: {snapshot.timestamp} ({snapshot.archive_source})")

            # Get temporal evolution
            evolution = await miner.get_temporal_content_evolution(test_url)
            logger.info(f"✅ Temporal evolution analysis completed")
            logger.info(f"  Time span: {evolution.get('time_span_days', 0)} days")
            logger.info(f"  Total snapshots: {evolution.get('total_snapshots', 0)}")

        else:
            logger.warning(f"⚠️ No archive snapshots found for {test_url}")

        # Get mining statistics
        stats = miner.get_mining_statistics()
        logger.info(f"Mining statistics: {json.dumps(stats, indent=2)}")

    except Exception as e:
        logger.error(f"❌ Archive mining error: {e}")


async def demo_phase2_orchestrator():
    """Demonstrate complete Phase 2 orchestrator"""
    logger.info("=== Testing Phase 2 Orchestrator ===")

    try:
        # Create Phase 2 system with example config
        config = EXAMPLE_PHASE2_CONFIG.copy()
        config["tor_browser"]["enabled"] = False  # Disable for demo

        system = create_phase2_system(config)

        # Initialize system
        if await system.initialize():
            logger.info("✅ Phase 2 system initialized successfully")

            # Test single URL acquisition
            test_urls = [
                "https://httpbin.org/ip",
                "https://httpbin.org/user-agent",
                "https://httpbin.org/headers",
            ]

            for url in test_urls:
                logger.info(f"Acquiring data from {url}...")

                # Test intelligent proxy method
                result = await system.acquire_data_from_url(url, "intelligent_proxy")

                if result.get("success"):
                    logger.info(f"✅ Successfully acquired {len(result.get('content', ''))} chars")
                    logger.info(f"  Provider: {result.get('provider_used')}")
                    logger.info(f"  Response time: {result.get('response_time', 0):.2f}s")
                else:
                    logger.warning(f"⚠️ Failed to acquire data: {result.get('error')}")

            # Test batch acquisition
            logger.info("Testing batch acquisition...")
            batch_results = await system.batch_acquire_urls(
                ["https://httpbin.org/ip", "https://httpbin.org/uuid"],
                method="intelligent_proxy",
                max_concurrent=2,
            )

            successful_batch = sum(1 for r in batch_results if r.get("success"))
            logger.info(f"✅ Batch acquisition: {successful_batch}/{len(batch_results)} successful")

            # Test archive mining method
            logger.info("Testing archive mining method...")
            archive_result = await system.acquire_data_from_url(
                "https://example.com", "archive_mining"
            )

            if archive_result.get("success"):
                logger.info(
                    f"✅ Archive mining found {archive_result.get('snapshots_found', 0)} snapshots"
                )
            else:
                logger.warning(f"⚠️ Archive mining failed: {archive_result.get('error')}")

            # Get system status
            status = await system.get_system_status()
            logger.info(f"System status: {json.dumps(status, indent=2)}")

            # Export session data
            export_path = await system.export_session_data()
            logger.info(f"✅ Session data exported to: {export_path}")

            # Shutdown system
            await system.shutdown()
            logger.info("✅ Phase 2 system shutdown completed")

        else:
            logger.error("❌ Phase 2 system initialization failed")

    except Exception as e:
        logger.error(f"❌ Phase 2 orchestrator error: {e}")


async def main():
    """Main demo function"""
    logger.info("🚀 Starting Phase 2 Advanced Data Acquisition Demo")
    logger.info("=" * 60)

    # Create results directory
    results_dir = Path("./data/phase2_demo_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Run individual component demos
        await demo_anonymity_providers()
        await asyncio.sleep(1)

        await demo_intelligent_proxy_manager()
        await asyncio.sleep(1)

        await demo_behavior_camouflage()
        await asyncio.sleep(1)

        await demo_archive_miner()
        await asyncio.sleep(1)

        # Run complete orchestrator demo
        await demo_phase2_orchestrator()

        logger.info("🎉 Phase 2 demo completed successfully!")
        logger.info("=" * 60)

        # Demo summary
        logger.info("📊 DEMO SUMMARY:")
        logger.info("✅ Anonymity Providers - Multiple network anonymity options")
        logger.info("✅ Intelligent Proxy Manager - Dynamic proxy selection and rotation")
        logger.info("✅ Behavior Camouflage - Human-like behavior simulation")
        logger.info("✅ Archive Miner - Historical data extraction from web archives")
        logger.info("✅ Phase 2 Orchestrator - Complete system integration")
        logger.info("")
        logger.info("🔧 CONFIGURATION OPTIONS:")
        logger.info("• Tor integration (requires Tor installation)")
        logger.info("• I2P support (experimental)")
        logger.info("• Tor Browser Selenium automation")
        logger.info("• Configurable behavior profiles")
        logger.info("• Performance-based proxy selection")
        logger.info("• Comprehensive archive mining")

    except Exception as e:
        logger.error(f"❌ Demo failed with error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
