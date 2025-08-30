"""
Demo script for Phase 3: Advanced Scraping & Evasion
Demonstrates Playwright interception, stealth capabilities, and CAPTCHA handling
"""

import asyncio
import tempfile
import shutil
from pathlib import Path
import logging
from datetime import datetime
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import Phase 3 components
from src.scraping.playwright_client import PlaywrightClient, discover_and_extract
from src.scraping.evasion_profiles import (
    EvasionProfileManager,
    BrowserProfileGenerator,
    apply_evasion_profile,
)
from src.scraping.captcha_broker import (
    CaptchaBroker,
    CaptchaConfig,
    CaptchaType,
    create_captcha_config,
)
from src.scraping.stealth_engine import StealthEngine, StealthConfig, create_stealth_page
from src.core.memory_optimizer import MemoryOptimizer


async def demo_phase3_advanced_scraping():
    """Comprehensive demo of Phase 3 capabilities"""
    logger.info("🚀 Starting Phase 3 Demo: Advanced Scraping & Evasion")

    # Create temporary workspace
    temp_dir = Path(tempfile.mkdtemp(prefix="phase3_demo_"))
    logger.info(f"📁 Working directory: {temp_dir}")

    try:
        # 1. Browser Profile Generation Demo
        logger.info("\n1️⃣ Browser Profile Generation Demo")

        profile_generator = BrowserProfileGenerator()

        # Generate random profiles
        logger.info("🎭 Generating browser profiles:")
        for i in range(3):
            profile = profile_generator.generate_random_profile()
            logger.info(f"   Profile {i+1}: {profile.platform} - {profile.user_agent[:50]}...")
            logger.info(
                f"   Viewport: {profile.viewport_width}x{profile.viewport_height}, HW Cores: {profile.hardware_concurrency}"
            )

        # Site-specific profiles
        logger.info("\n🎯 Site-specific profiles:")
        test_domains = ["google.com", "facebook.com", "linkedin.com"]
        for domain in test_domains:
            profile = profile_generator.generate_site_specific_profile(domain)
            logger.info(f"   {domain}: {profile.extra_headers.get('sec-ch-ua', 'Default headers')}")

        # 2. Evasion Profile Management Demo
        logger.info("\n2️⃣ Evasion Profile Management Demo")

        profile_manager = EvasionProfileManager(temp_dir / "evasion_profiles")

        # Create profiles for different sites
        test_sites = ["example.com", "test.org", "demo.net"]
        profiles = []

        for site in test_sites:
            profile = profile_manager.get_profile_for_site(site)
            profiles.append(profile)
            logger.info(f"🆔 Created profile for {site}: {profile.profile_id}")
            logger.info(f"   User Agent: {profile.browser_profile.user_agent[:60]}...")
            logger.info(
                f"   Behavioral settings: {list(profile.behavioral_settings.keys())[:3]}..."
            )

        # Simulate usage and success tracking
        logger.info("\n📊 Simulating profile usage:")
        for i, profile in enumerate(profiles):
            # Simulate some successful and failed requests
            successes = 8 + i
            failures = 2 - i

            for _ in range(successes):
                profile_manager.update_profile_success(profile, True)
            for _ in range(failures):
                profile_manager.update_profile_success(profile, False)

            logger.info(
                f"   {profile.site_domain}: {profile.usage_count} uses, {profile.success_rate:.1%} success rate"
            )

        # Profile statistics
        stats = profile_manager.get_profile_stats()
        logger.info(
            f"📈 Profile Manager Stats: {stats['total_profiles']} profiles, {stats['average_success_rate']:.1%} avg success"
        )

        # 3. CAPTCHA Broker Demo
        logger.info("\n3️⃣ CAPTCHA Broker Demo")

        # Create safe CAPTCHA config (stub mode)
        captcha_config = create_captcha_config(
            provider="stub", enable=False, cost_limit=5.0  # Safe default
        )

        captcha_broker = CaptchaBroker(captcha_config)

        logger.info(
            f"🤖 CAPTCHA Broker initialized: Provider={captcha_config.service_provider}, Enabled={captcha_config.enable_captcha_solving}"
        )

        # Demonstrate different CAPTCHA types (stub mode)
        captcha_scenarios = [
            (CaptchaType.RECAPTCHA_V2, "https://example.com", "test_site_key"),
            (CaptchaType.IMAGE, "https://test.org", None),
            (CaptchaType.HCAPTCHA, "https://demo.net", "hcaptcha_key"),
        ]

        logger.info("🔍 Testing CAPTCHA scenarios (stub mode):")
        for captcha_type, site_url, site_key in captcha_scenarios:
            result = await captcha_broker.solve_captcha(
                captcha_type=captcha_type, site_url=site_url, site_key=site_key
            )

            status = "✅ Solved" if result.success else "❌ Failed"
            logger.info(f"   {captcha_type.value}: {status} - {result.error_message or 'Success'}")

        # Broker usage stats
        usage_stats = captcha_broker.get_usage_stats()
        logger.info(
            f"💰 CAPTCHA Stats: ${usage_stats['total_cost_usd']:.4f} spent, {usage_stats['solved_count']} solved"
        )

        # 4. Stealth Engine Demo
        logger.info("\n4️⃣ Stealth Engine Demo")

        stealth_config = StealthConfig(
            enable_stealth=True,
            human_mouse_movement=True,
            randomize_canvas_fingerprint=True,
            spoof_webgl=True,
        )

        stealth_engine = StealthEngine(stealth_config)
        stats = stealth_engine.get_stealth_stats()

        logger.info(f"🥷 Stealth Engine configured:")
        logger.info(
            f"   Features enabled: {sum(stats['features_enabled'].values())}/{len(stats['features_enabled'])}"
        )
        logger.info(f"   Mouse speed: {stats['timing_config']['mouse_speed']}x")
        logger.info(f"   Typing delays: {stats['timing_config']['typing_delay_range']}ms")

        # Demonstrate stealth capabilities (without actual browser)
        logger.info("🎯 Stealth capabilities available:")
        features = [
            "WebDriver property masking",
            "Canvas fingerprint randomization",
            "Audio context spoofing",
            "WebGL vendor/renderer override",
            "Human-like mouse movement",
            "Randomized typing delays",
            "Behavioral scroll patterns",
        ]

        for feature in features:
            logger.info(f"   ✅ {feature}")

        # 5. Playwright Client Demo (Mock Mode)
        logger.info("\n5️⃣ Playwright Client Demo (Mock Mode)")

        # Note: This demo runs in mock mode to avoid requiring browser installation
        logger.info("🎭 Playwright capabilities overview:")

        demo_capabilities = {
            "Discovery Mode": [
                "XHR/fetch request interception",
                "API endpoint capture and analysis",
                "Static content extraction",
                "Interactive element discovery",
            ],
            "Extraction Mode": [
                "API replay through Tor/I2P",
                "Parameterized request modification",
                "Rate-limited data extraction",
                "Session persistence",
            ],
            "Integration Features": [
                "Evasion profile application",
                "Stealth behavior simulation",
                "CAPTCHA detection and solving",
                "Memory-optimized processing",
            ],
        }

        for category, features in demo_capabilities.items():
            logger.info(f"   📋 {category}:")
            for feature in features:
                logger.info(f"      • {feature}")

        # Mock discovery session
        logger.info("\n🔍 Mock Discovery Session:")
        mock_session = {
            "session_id": "discovery_20241128_demo",
            "page_url": "https://example.com/app",
            "apis_discovered": 5,
            "endpoints": [
                {"method": "GET", "url": "/api/search", "type": "data"},
                {"method": "POST", "url": "/api/login", "type": "auth"},
                {"method": "GET", "url": "/api/content", "type": "content"},
            ],
        }

        logger.info(f"   🆔 Session: {mock_session['session_id']}")
        logger.info(f"   🌐 Target: {mock_session['page_url']}")
        logger.info(f"   📊 APIs found: {mock_session['apis_discovered']}")

        for endpoint in mock_session["endpoints"]:
            logger.info(f"      • {endpoint['method']} {endpoint['url']} ({endpoint['type']})")

        # 6. Anti-Bot Evasion Strategies
        logger.info("\n6️⃣ Anti-Bot Evasion Strategies")

        evasion_strategies = {
            "Fingerprint Masking": {
                "description": "Randomize browser fingerprints",
                "techniques": ["Canvas noise injection", "WebGL spoofing", "Audio context masking"],
                "effectiveness": "High",
            },
            "Behavioral Simulation": {
                "description": "Human-like interaction patterns",
                "techniques": [
                    "Mouse movement curves",
                    "Typing rhythm variation",
                    "Natural scrolling",
                ],
                "effectiveness": "Very High",
            },
            "Request Patterns": {
                "description": "Avoid detection through timing",
                "techniques": ["Rate limiting", "Random delays", "Session rotation"],
                "effectiveness": "Medium",
            },
            "Network Anonymity": {
                "description": "IP and traffic obfuscation",
                "techniques": ["Tor circuits", "I2P tunnels", "Proxy rotation"],
                "effectiveness": "High",
            },
        }

        logger.info("🛡️ Available evasion strategies:")
        for strategy, details in evasion_strategies.items():
            logger.info(f"   📋 {strategy}: {details['description']}")
            logger.info(f"      Effectiveness: {details['effectiveness']}")
            logger.info(f"      Techniques: {', '.join(details['techniques'][:2])}...")

        # 7. Memory and Performance Optimization
        logger.info("\n7️⃣ Memory and Performance Optimization")

        optimizer = MemoryOptimizer(max_memory_gb=6.0)
        memory_stats = optimizer.check_memory_pressure()

        logger.info(f"💾 Memory optimization for M1:")
        logger.info(f"   Available: {memory_stats['available_gb']:.1f}GB")
        logger.info(f"   Usage: {memory_stats['used_percent']:.1f}%")
        logger.info(f"   Pressure: {'⚠️ Yes' if memory_stats['pressure'] else '✅ No'}")

        optimization_tips = [
            "Minimize concurrent browser sessions",
            "Use API replay instead of full DOM parsing",
            "Stream large datasets to Parquet",
            "Rotate browser contexts regularly",
            "Enable lazy loading for content",
            "Batch API requests efficiently",
        ]

        logger.info("⚡ M1 optimization strategies:")
        for tip in optimization_tips:
            logger.info(f"   • {tip}")

        # 8. Security and Legal Considerations
        logger.info("\n8️⃣ Security and Legal Considerations")

        security_measures = [
            "Respect robots.txt by default",
            "Implement rate limiting",
            "Use allowlists for target sites",
            "Log all scraping activities",
            "Disable features by default (opt-in)",
            "Provide CAPTCHA solving interfaces only",
            "Support legal research use cases",
        ]

        logger.info("🛡️ Built-in security measures:")
        for measure in security_measures:
            logger.info(f"   ✅ {measure}")

        legal_guidelines = [
            "Only access publicly available content",
            "Respect website terms of service",
            "Implement appropriate delays",
            "Avoid overloading target servers",
            "Document legitimate research purposes",
        ]

        logger.info("⚖️ Legal compliance guidelines:")
        for guideline in legal_guidelines:
            logger.info(f"   📋 {guideline}")

        # 9. Integration with Previous Phases
        logger.info("\n9️⃣ Integration with Previous Phases")

        integration_points = {
            "Phase 1 (Memory Core)": [
                "Stream scraped data to Parquet",
                "Use DuckDB for scraped data analysis",
                "Apply memory optimization during scraping",
                "Lazy evaluation for large datasets",
            ],
            "Phase 2 (Anonymity)": [
                "Route scraping through Tor/I2P",
                "Apply network failover during scraping",
                "Use hidden service discovery for targets",
                "Maintain anonymity during extraction",
            ],
        }

        for phase, features in integration_points.items():
            logger.info(f"🔗 {phase} integration:")
            for feature in features:
                logger.info(f"   • {feature}")

        # 10. Configuration Examples
        logger.info("\n🔟 Configuration Examples")

        config_examples = {
            "Conservative": {
                "stealth": "Basic fingerprint masking",
                "captcha": "Disabled (stub mode)",
                "rate_limit": "High delays, respectful",
                "features": "Minimal evasion",
            },
            "Balanced": {
                "stealth": "Full stealth suite enabled",
                "captcha": "Manual solving only",
                "rate_limit": "Moderate delays",
                "features": "Standard evasion",
            },
            "Aggressive": {
                "stealth": "Maximum evasion techniques",
                "captcha": "Automated solving enabled",
                "rate_limit": "Optimized for speed",
                "features": "All techniques active",
            },
        }

        logger.info("⚙️ Configuration profiles:")
        for profile_name, settings in config_examples.items():
            logger.info(f"   📋 {profile_name}:")
            for setting, value in settings.items():
                logger.info(f"      {setting.title()}: {value}")

        logger.info("\n🎉 Phase 3 Demo completed successfully!")
        logger.info("✨ Advanced scraping & evasion capabilities are ready for ethical use")

    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise
    finally:
        # Cleanup
        logger.info(f"🧹 Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


async def demo_playwright_workflow():
    """Demonstrate complete Playwright workflow (conceptual)"""
    logger.info("\n🎭 Playwright Workflow Demo (Conceptual)")

    workflow_steps = [
        {
            "step": "1. Browser Initialization",
            "description": "Launch browser with stealth configuration",
            "code": "browser = await playwright.chromium.launch(headless=True)",
        },
        {
            "step": "2. Profile Application",
            "description": "Apply evasion profile to new page",
            "code": "await apply_evasion_profile(page, profile)",
        },
        {
            "step": "3. Stealth Navigation",
            "description": "Navigate with human-like behavior",
            "code": "await stealth_engine.navigate_stealthily(page, url)",
        },
        {
            "step": "4. API Discovery",
            "description": "Capture XHR/fetch requests",
            "code": "discovery_result = await client.discover_page_apis(url)",
        },
        {
            "step": "5. API Replay",
            "description": "Replay APIs through Tor/I2P",
            "code": "extraction_result = await client.extract_with_replay(discovery_result)",
        },
        {
            "step": "6. Data Processing",
            "description": "Stream results to Parquet",
            "code": "await stream_to_parquet(extraction_result)",
        },
    ]

    logger.info("📋 Complete scraping workflow:")
    for workflow_step in workflow_steps:
        logger.info(f"   {workflow_step['step']}: {workflow_step['description']}")
        logger.info(f"      Code: {workflow_step['code']}")


async def demo_captcha_integration():
    """Demonstrate CAPTCHA integration workflow"""
    logger.info("\n🤖 CAPTCHA Integration Demo")

    # Different CAPTCHA service configurations
    captcha_configs = {
        "Development": create_captcha_config("stub", enable=False),
        "Testing": create_captcha_config("2captcha", api_key="test_key", enable=False),
        "Production": create_captcha_config(
            "anticaptcha", api_key="prod_key", enable=False, cost_limit=50.0
        ),
    }

    logger.info("🔧 CAPTCHA service configurations:")
    for env, config in captcha_configs.items():
        logger.info(f"   {env}: Provider={config.service_provider}, Limit=${config.cost_limit_usd}")

    # CAPTCHA solving workflow
    workflow = [
        "1. Detect CAPTCHA on page",
        "2. Extract CAPTCHA parameters (site key, image)",
        "3. Submit to solving service",
        "4. Poll for solution with timeout",
        "5. Apply solution to form",
        "6. Continue with scraping",
    ]

    logger.info("📋 CAPTCHA solving workflow:")
    for step in workflow:
        logger.info(f"   {step}")


if __name__ == "__main__":

    async def main():
        """Run complete Phase 3 demo"""
        logger.info("🎬 DeepResearchTool Phase 3 Demo Starting...")

        await demo_phase3_advanced_scraping()
        await demo_playwright_workflow()
        await demo_captcha_integration()

        logger.info("\n🏁 All Phase 3 demos completed!")
        logger.info("🚀 Ready to proceed with Phase 4: Intelligence Layer & RAG")

    # Run the demo
    asyncio.run(main())
