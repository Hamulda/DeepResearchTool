#!/usr/bin/env python3
"""
Legacy config loader - přesměrováno na nový bezpečný config systém
"""

import warnings
from typing import Dict, Any

# Import nového bezpečného config systému
try:
    from src.core.config import load_config as new_load_config
except ImportError:
    # Fallback pro případy, kdy nový systém není dostupný
    def new_load_config(config_path: str = None) -> Dict[str, Any]:
        return {
            "research_profile": "thorough",
            "max_documents": 50,
            "synthesis": {"max_claims": 10},
            "retrieval": {"timeout_seconds": 120},
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
                "min_evidence_coverage": 0.7,
            },
            "m1_optimization": {"enabled": True, "ollama": {"host": "http://localhost:11434"}},
        }


def load_config(config_path: str) -> Dict[str, Any]:
    """Legacy function - používá nový bezpečný config systém"""
    warnings.warn(
        "This legacy config loader is deprecated. Use src.core.config.get_settings() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    return new_load_config(config_path)
