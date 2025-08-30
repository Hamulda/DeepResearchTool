#!/usr/bin/env python3
"""Phase 2 Configuration - Advanced Data Acquisition & Anti-Detection
Production-ready configuration for Phase 2 capabilities

Author: Advanced AI Research Assistant
Date: August 2025
"""

import os
from pathlib import Path
from typing import Any

# Base configuration directory
CONFIG_DIR = Path(__file__).parent / "configs"
DATA_DIR = Path(__file__).parent / "data"
LOGS_DIR = Path(__file__).parent / "logs"

# Ensure directories exist
CONFIG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


class Phase2Configuration:
    """Configuration manager for Phase 2 system"""

    @staticmethod
    def get_production_config() -> dict[str, Any]:
        """Get production-ready Phase 2 configuration"""
        return {
            "providers": {
                "tor": {
                    "socks_port": int(os.getenv("TOR_SOCKS_PORT", "9050")),
                    "control_port": int(os.getenv("TOR_CONTROL_PORT", "9051")),
                    "control_password": os.getenv("TOR_CONTROL_PASSWORD", ""),
                    "rotation_interval": int(os.getenv("TOR_ROTATION_INTERVAL", "600")),
                    "timeout": int(os.getenv("TOR_TIMEOUT", "30")),
                    "user_agent": "Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0",
                },
                "clearnet": {
                    "timeout": int(os.getenv("CLEARNET_TIMEOUT", "10")),
                    "user_agents": [
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0",
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:91.0) Gecko/20100101 Firefox/91.0",
                    ],
                },
                "i2p": {
                    "enabled": bool(os.getenv("I2P_ENABLED", "false").lower() == "true"),
                    "sam_port": int(os.getenv("I2P_SAM_PORT", "7656")),
                    "http_proxy_port": int(os.getenv("I2P_HTTP_PROXY_PORT", "4444")),
                    "timeout": int(os.getenv("I2P_TIMEOUT", "60")),
                },
            },
            "proxy_management": {
                "rotation_strategy": os.getenv("PROXY_ROTATION_STRATEGY", "performance_based"),
                "min_rotation_interval": int(os.getenv("PROXY_MIN_ROTATION_INTERVAL", "60")),
                "max_consecutive_uses": int(os.getenv("PROXY_MAX_CONSECUTIVE_USES", "10")),
                "circuit_breaker_threshold": int(os.getenv("PROXY_CIRCUIT_BREAKER_THRESHOLD", "5")),
                "circuit_breaker_timeout": int(os.getenv("PROXY_CIRCUIT_BREAKER_TIMEOUT", "300")),
            },
            "behavior_profile": {
                "reading_speed_wpm": int(os.getenv("BEHAVIOR_READING_SPEED", "200")),
                "scroll_speed_variance": float(os.getenv("BEHAVIOR_SCROLL_VARIANCE", "0.3")),
                "click_delay_range": [0.1, 0.5],
                "typing_speed_cps": int(os.getenv("BEHAVIOR_TYPING_SPEED", "4")),
                "mouse_movement_style": os.getenv("BEHAVIOR_MOUSE_STYLE", "natural"),
                "attention_span_seconds": int(os.getenv("BEHAVIOR_ATTENTION_SPAN", "180")),
                "break_frequency": float(os.getenv("BEHAVIOR_BREAK_FREQUENCY", "0.15")),
                "break_duration_range": [5, 30],
            },
            "archive_mining": {
                "max_snapshots_per_url": int(os.getenv("ARCHIVE_MAX_SNAPSHOTS", "100")),
                "min_time_between_snapshots": int(os.getenv("ARCHIVE_MIN_TIME_BETWEEN", "30")),
                "content_filters": [
                    r"\.pdf$",
                    r"\.zip$",
                    r"\.rar$",
                    r"\.exe$",
                    r"\.dmg$",
                    r"\.iso$",
                    r"\.tar\.gz$",
                ],
                "wayback_rate_limit": float(os.getenv("WAYBACK_RATE_LIMIT", "1.0")),
                "archive_today_rate_limit": float(os.getenv("ARCHIVE_TODAY_RATE_LIMIT", "2.0")),
            },
            "tor_browser": {
                "enabled": bool(os.getenv("TOR_BROWSER_ENABLED", "false").lower() == "true"),
                "headless": bool(os.getenv("TOR_BROWSER_HEADLESS", "true").lower() == "true"),
                "take_screenshots": bool(
                    os.getenv("TOR_BROWSER_SCREENSHOTS", "false").lower() == "true"
                ),
                "tor_browser_path": os.getenv("TOR_BROWSER_PATH"),
                "circuit_lifetime": int(os.getenv("TOR_BROWSER_CIRCUIT_LIFETIME", "600")),
                "always_use_hybrid": bool(
                    os.getenv("TOR_BROWSER_HYBRID", "false").lower() == "true"
                ),
                "behavior_profile": {
                    "reading_speed_wpm": 150,
                    "scroll_speed_variance": 0.4,
                    "mouse_movement_style": "cautious",
                },
            },
            "max_concurrent_sessions": int(os.getenv("MAX_CONCURRENT_SESSIONS", "3")),
            "session_timeout": int(os.getenv("SESSION_TIMEOUT", "300")),
            "data_export_path": os.getenv("DATA_EXPORT_PATH", str(DATA_DIR / "phase2_results")),
            "enable_comprehensive_logging": bool(
                os.getenv("COMPREHENSIVE_LOGGING", "true").lower() == "true"
            ),
        }

    @staticmethod
    def get_development_config() -> dict[str, Any]:
        """Get development configuration with safer defaults"""
        config = Phase2Configuration.get_production_config()

        # Override with development-safe settings
        config.update(
            {
                "max_concurrent_sessions": 2,
                "session_timeout": 60,
                "behavior_profile": {
                    **config["behavior_profile"],
                    "reading_speed_wpm": 300,  # Faster for testing
                    "attention_span_seconds": 60,
                    "break_frequency": 0.05,  # Less frequent breaks
                },
                "archive_mining": {
                    **config["archive_mining"],
                    "max_snapshots_per_url": 20,  # Fewer snapshots for testing
                    "wayback_rate_limit": 2.0,  # More conservative for testing
                    "archive_today_rate_limit": 3.0,
                },
                "tor_browser": {
                    **config["tor_browser"],
                    "enabled": False,  # Disabled by default in development
                    "headless": True,
                    "take_screenshots": False,
                },
            }
        )

        return config

    @staticmethod
    def get_security_focused_config() -> dict[str, Any]:
        """Get configuration optimized for maximum security and anonymity"""
        config = Phase2Configuration.get_production_config()

        # Override with security-focused settings
        config.update(
            {
                "providers": {
                    "tor": {
                        **config["providers"]["tor"],
                        "rotation_interval": 300,  # More frequent rotation
                        "timeout": 60,  # Longer timeout for Tor
                    },
                    # Remove clearnet provider for maximum anonymity
                    "clearnet": None,
                },
                "proxy_management": {
                    **config["proxy_management"],
                    "rotation_strategy": "random",  # Less predictable
                    "min_rotation_interval": 30,  # More frequent rotation
                    "max_consecutive_uses": 5,  # Fewer uses per provider
                },
                "behavior_profile": {
                    **config["behavior_profile"],
                    "mouse_movement_style": "cautious",
                    "break_frequency": 0.25,  # More frequent breaks
                    "break_duration_range": [10, 60],  # Longer breaks
                },
                "tor_browser": {
                    **config["tor_browser"],
                    "enabled": True,
                    "circuit_lifetime": 300,  # More frequent circuit renewal
                    "behavior_profile": {
                        "reading_speed_wpm": 120,  # Slower, more careful
                        "scroll_speed_variance": 0.2,
                        "mouse_movement_style": "cautious",
                    },
                },
                "max_concurrent_sessions": 1,  # Single session for maximum stealth
            }
        )

        # Remove clearnet provider
        if config["providers"]["clearnet"] is None:
            del config["providers"]["clearnet"]

        return config

    @staticmethod
    def get_config_by_environment() -> dict[str, Any]:
        """Get configuration based on environment variable"""
        environment = os.getenv("PHASE2_ENVIRONMENT", "development").lower()

        if environment == "production":
            return Phase2Configuration.get_production_config()
        if environment == "security":
            return Phase2Configuration.get_security_focused_config()
        # development
        return Phase2Configuration.get_development_config()

    @staticmethod
    def save_config_template(filepath: str):
        """Save configuration template to file"""
        template = {
            "# Phase 2 Configuration Template": "Copy to .env file and customize",
            "# Environment": "development, production, or security",
            "PHASE2_ENVIRONMENT": "development",
            "# Tor Configuration": "",
            "TOR_SOCKS_PORT": "9050",
            "TOR_CONTROL_PORT": "9051",
            "TOR_CONTROL_PASSWORD": "your_tor_password_here",
            "TOR_ROTATION_INTERVAL": "600",
            "TOR_TIMEOUT": "30",
            "# I2P Configuration (experimental)": "",
            "I2P_ENABLED": "false",
            "I2P_SAM_PORT": "7656",
            "I2P_HTTP_PROXY_PORT": "4444",
            "# Proxy Management": "",
            "PROXY_ROTATION_STRATEGY": "performance_based",
            "PROXY_MIN_ROTATION_INTERVAL": "60",
            "PROXY_MAX_CONSECUTIVE_USES": "10",
            "# Behavior Configuration": "",
            "BEHAVIOR_READING_SPEED": "200",
            "BEHAVIOR_SCROLL_VARIANCE": "0.3",
            "BEHAVIOR_TYPING_SPEED": "4",
            "BEHAVIOR_MOUSE_STYLE": "natural",
            "# Archive Mining": "",
            "ARCHIVE_MAX_SNAPSHOTS": "100",
            "ARCHIVE_MIN_TIME_BETWEEN": "30",
            "WAYBACK_RATE_LIMIT": "1.0",
            "# Tor Browser Selenium": "",
            "TOR_BROWSER_ENABLED": "false",
            "TOR_BROWSER_HEADLESS": "true",
            "TOR_BROWSER_SCREENSHOTS": "false",
            "TOR_BROWSER_PATH": "/path/to/tor-browser",
            "# System Configuration": "",
            "MAX_CONCURRENT_SESSIONS": "3",
            "SESSION_TIMEOUT": "300",
            "DATA_EXPORT_PATH": "./data/phase2_results",
            "COMPREHENSIVE_LOGGING": "true",
        }

        with open(filepath, "w") as f:
            for key, value in template.items():
                if key.startswith("#"):
                    f.write(f"\n{key}\n")
                else:
                    f.write(f"{key}={value}\n")


# Export commonly used configurations
DEVELOPMENT_CONFIG = Phase2Configuration.get_development_config()
PRODUCTION_CONFIG = Phase2Configuration.get_production_config()
SECURITY_CONFIG = Phase2Configuration.get_security_focused_config()

# Current configuration based on environment
CURRENT_CONFIG = Phase2Configuration.get_config_by_environment()

if __name__ == "__main__":
    # Create configuration template
    Phase2Configuration.save_config_template("phase2_config_template.env")
    print("Phase 2 configuration template saved to phase2_config_template.env")

    # Display current configuration
    import json

    print("\nCurrent Phase 2 Configuration:")
    print(json.dumps(CURRENT_CONFIG, indent=2))
