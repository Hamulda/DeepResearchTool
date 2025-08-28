"""
Core configuration loader for DeepResearchTool v3.0
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_file = Path(config_path)

    if not config_file.exists():
        # Return default config if file doesn't exist
        return get_default_config()

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Merge with defaults
    default_config = get_default_config()
    return merge_configs(default_config, config)


def get_default_config() -> Dict[str, Any]:
    """Default configuration for automatic research agent"""
    return {
        "retrieval": {
            "max_context_tokens": 8000,
            "top_k": 50,
            "ef_search": 200,
            "hyde": {
                "enabled": True,
                "budget_tokens": 500
            },
            "rrf": {
                "k": 60,
                "weights": {
                    "bm25": 0.3,
                    "dense": 0.4,
                    "hyde": 0.3
                }
            }
        },
        "synthesis": {
            "max_claims": 8,
            "min_citations_per_claim": 2,
            "template_mode": "evidence_driven"
        },
        "gates": {
            "evidence": {
                "min_citations_per_claim": 2,
                "min_source_diversity": 0.7
            },
            "compliance": {
                "max_rate_violations": 0,
                "max_robots_violations": 0
            },
            "metrics": {
                "min_recall": 0.7,
                "min_precision": 0.8,
                "min_groundedness": 0.85
            },
            "quality": {
                "max_token_budget": 8000,
                "min_claim_quality": 0.7
            }
        },
        "connectors": {
            "common_crawl": {"enabled": True},
            "memento": {"enabled": True},
            "ahmia": {"enabled": True, "legal_only": True},
            "legal_apis": {"enabled": True},
            "open_alex": {"enabled": True}
        }
    }


def merge_configs(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge configuration dictionaries"""
    result = default.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result
