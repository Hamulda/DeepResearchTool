#!/usr/bin/env python3
"""
Configuration utilities for Deep Research Tool v2.0
Load and validate configuration with profile support

Author: Senior IT Specialist
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def load_config(profile: str = "quick") -> Dict[str, Any]:
    """Load configuration with profile override"""

    # Determine config file path
    config_file = os.getenv("DRT_CONFIG", "config_m1_local.yaml")
    config_path = Path(config_file)

    if not config_path.exists():
        # Try relative path from project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / config_file

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    # Load base configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Apply profile-specific settings
    if "profiles" in config and profile in config["profiles"]:
        profile_config = config["profiles"][profile]

        # Deep merge profile settings into base config
        config = deep_merge(config, profile_config)

        # Set profile in config for reference
        config["active_profile"] = profile

    # Apply environment variable overrides
    config = apply_env_overrides(config)

    # Validate configuration
    validate_config(config)

    logger.info(f"Configuration loaded: profile={profile}, file={config_path}")

    return config

def deep_merge(base_dict: Dict[str, Any], override_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries"""

    result = base_dict.copy()

    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result

def apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides"""

    # Map of environment variables to config paths
    env_mappings = {
        "QDRANT_URL": ["qdrant", "url"],
        "OLLAMA_URL": ["ollama", "url"],
        "DRT_PROFILE": ["active_profile"],
        "DRT_LOG_LEVEL": ["logging", "level"],
        "DRT_ENABLE_COMPRESSION": ["compression", "enabled"],
        "DRT_ENABLE_HIERARCHICAL": ["retrieval", "hierarchical", "enabled"],
        "DRT_RRF_K": ["retrieval", "rrf", "k"]
    }

    for env_var, config_path in env_mappings.items():
        env_value = os.getenv(env_var)
        if env_value is not None:
            # Convert string values to appropriate types
            if env_value.lower() in ("true", "false"):
                env_value = env_value.lower() == "true"
            elif env_value.isdigit():
                env_value = int(env_value)
            elif env_value.replace(".", "").isdigit():
                env_value = float(env_value)

            # Set value in config
            current_dict = config
            for key in config_path[:-1]:
                current_dict = current_dict.setdefault(key, {})
            current_dict[config_path[-1]] = env_value

            logger.info(f"Applied environment override: {env_var}={env_value}")

    return config

def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration completeness and correctness"""

    required_sections = [
        "profiles",
        "qdrant",
        "retrieval",
        "compression",
        "claim_graph"
    ]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")

    # Validate profile settings
    active_profile = config.get("active_profile", "quick")
    if active_profile not in config.get("profiles", {}):
        logger.warning(f"Active profile '{active_profile}' not found in profiles")

    # Validate Qdrant URL
    qdrant_url = config.get("qdrant", {}).get("url")
    if not qdrant_url:
        raise ValueError("Qdrant URL not configured")

    # Validate feature flags consistency
    features = config.get("features", {})
    if features.get("hierarchical_retrieval") and not config.get("retrieval", {}).get("hierarchical", {}).get("enabled"):
        logger.warning("Hierarchical retrieval feature flag enabled but not configured")

    # Validate model configurations
    profiles = config.get("profiles", {})
    for profile_name, profile_config in profiles.items():
        llm_config = profile_config.get("llm", {}).get("verification", {})
        if not llm_config.get("primary_model"):
            logger.warning(f"No primary model configured for profile: {profile_name}")

def get_profile_setting(config: Dict[str, Any], setting_path: str, default=None) -> Any:
    """Get setting from active profile with fallback to global config"""

    active_profile = config.get("active_profile", "quick")

    # Try profile-specific setting first
    if "profiles" in config and active_profile in config["profiles"]:
        profile_config = config["profiles"][active_profile]

        # Navigate setting path
        current = profile_config
        for key in setting_path.split("."):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                current = None
                break

        if current is not None:
            return current

    # Fallback to global setting
    current = config
    for key in setting_path.split("."):
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default

    return current if current is not None else default

def save_config_override(config: Dict[str, Any], output_path: str = "config_override.yaml") -> None:
    """Save configuration with current settings for debugging"""

    # Remove sensitive information
    safe_config = config.copy()

    # Remove any API keys or sensitive data
    sensitive_keys = ["api_key", "password", "secret", "token"]
    safe_config = remove_sensitive_keys(safe_config, sensitive_keys)

    with open(output_path, 'w') as f:
        yaml.dump(safe_config, f, indent=2, default_flow_style=False)

    logger.info(f"Configuration saved to: {output_path}")

def remove_sensitive_keys(data: Any, sensitive_keys: list) -> Any:
    """Remove sensitive keys from configuration data"""

    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                result[key] = "***REDACTED***"
            else:
                result[key] = remove_sensitive_keys(value, sensitive_keys)
        return result
    elif isinstance(data, list):
        return [remove_sensitive_keys(item, sensitive_keys) for item in data]
    else:
        return data

def create_default_config() -> Dict[str, Any]:
    """Create default configuration for new installations"""

    default_config = {
        "globals": {
            "seeds": {"python": 1337, "numpy": 1337, "torch": 1337},
            "checkpoints": {"enabled": True, "path": "./.checkpoints"}
        },
        "profiles": {
            "quick": {
                "retrieval": {
                    "hierarchical": {"enabled": True, "levels": 2},
                    "rrf_k": 40,
                    "dedup": True,
                    "compression": {"enabled": True, "budget_tokens": 2000}
                },
                "qdrant": {"ef_search": 64},
                "llm": {
                    "verification": {
                        "primary_model": "qwen2.5:7b-q4_K_M",
                        "confidence_threshold": 0.6
                    }
                }
            }
        },
        "qdrant": {
            "url": "http://localhost:6333",
            "embedding_model": "all-MiniLM-L6-v2"
        },
        "retrieval": {
            "hierarchical": {"enabled": True, "levels": 2},
            "rrf": {"k": 40},
            "deduplication": {"enabled": True}
        },
        "compression": {
            "enabled": True,
            "budget_tokens": 2000,
            "strategy": "salience"
        },
        "claim_graph": {
            "enabled": True,
            "contradiction_detection": {"enabled": True}
        },
        "features": {
            "hierarchical_retrieval": True,
            "contextual_compression": True,
            "adaptive_query_refinement": True,
            "claim_graph": True,
            "contradiction_detection": True
        },
        "logging": {
            "level": "INFO",
            "file": "./logs/deep_research_tool.log"
        }
    }

    return default_config
