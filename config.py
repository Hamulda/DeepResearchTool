#!/usr/bin/env python3
"""
Mock Config Module pro FÁZI 1 testování
"""

import yaml
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load mock configuration for Phase 1 testing"""
    # Mock configuration that satisfies validation gates
    mock_config = {
        "research_profile": "thorough",
        "max_documents": 50,
        "synthesis": {
            "max_claims": 10
        },
        "retrieval": {
            "timeout_seconds": 120
        },
        "validation_gates": {
            "min_citations_per_claim": 2,
            "min_independent_sources": 2,
            "max_hallucination_rate": 0.05,
            "robots_txt_respect": True,
            "rate_limit_respect": True,
            "min_recall_at_10": 0.7,
            "min_ndcg_at_10": 0.6,
            "min_citation_precision": 0.8,
            "max_context_usage": 0.9,
            "pii_redaction_enabled": True,
            "secret_scanning_enabled": True,
            "min_groundedness": 0.8,
            "min_evidence_coverage": 0.7
        },
        "m1_optimization": {
            "enabled": True,
            "ollama": {
                "host": "http://localhost:11434"
            }
        }
    }

    # Try to load real config if exists, otherwise use mock
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            real_config = yaml.safe_load(f)
            # Merge with mock config to ensure all required keys exist
            mock_config.update(real_config)
    except FileNotFoundError:
        pass

    return mock_config
