#!/usr/bin/env python3
"""
Zjednodušený konfigurační modul pro DeepResearchTool
Funkční verze bez problematických Pydantic Settings
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class SimpleConfig:
    """Jednoduchá konfigurační třída bez Pydantic komplikací"""
    
    def __init__(self):
        # Základní nastavení
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "true").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.workers = int(os.getenv("WORKERS", "1"))
        
        # LLM konfigurace
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        
        # M1 optimalizace
        self.m1_optimization_enabled = os.getenv("M1_OPTIMIZATION_ENABLED", "true").lower() == "true"
        
        # Aplikační nastavení
        self.research_profile = "thorough"
        self.max_documents = 50
        self.max_claims = 10
        self.timeout_seconds = 120
        
        # Validační brány
        self.min_citations_per_claim = 2
        self.min_independent_sources = 2
        self.max_hallucination_rate = 0.05
        self.min_recall_at_10 = 0.7
        self.min_ndcg_at_10 = 0.6
        self.min_citation_precision = 0.8
        self.max_context_usage = 0.9
        self.min_groundedness = 0.8
        self.min_evidence_coverage = 0.7
        
        # Databáze (základní)
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.chroma_host = os.getenv("CHROMA_HOST", "http://localhost:8001")
        self.qdrant_host = os.getenv("QDRANT_HOST", "http://localhost:6333")
        
        # Bezpečnost (základní)
        self.robots_txt_respect = True
        self.rate_limit_respect = True
        self.pii_redaction_enabled = True
        self.secret_scanning_enabled = True


class Settings:
    """Centrální settings manager s jednoduchým přístupem"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.app = SimpleConfig()
        
        # Load additional config from YAML if provided
        if config_path and Path(config_path).exists():
            self._load_yaml_config(config_path)
    
    def _load_yaml_config(self, config_path: str) -> None:
        """Načte dodatečnou konfiguraci z YAML souboru"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f)
                
            # Update settings from YAML
            if "research_profile" in yaml_config:
                self.app.research_profile = yaml_config["research_profile"]
            if "max_documents" in yaml_config:
                self.app.max_documents = yaml_config["max_documents"]
                
        except Exception as e:
            print(f"Warning: Could not load YAML config from {config_path}: {e}")
    
    def validate_critical_config(self) -> Dict[str, bool]:
        """Validuje kritické konfigurační hodnoty"""
        validation_results = {
            "has_llm_api_key": bool(self.app.openai_api_key or self.app.anthropic_api_key),
            "has_database_config": bool(self.app.redis_url),
            "has_security_config": True,  # Základní bezpečnost je vždy nastavena
            "environment_set": bool(self.app.environment),
        }
        
        return validation_results
    
    def get_mock_config_for_testing(self) -> Dict[str, Any]:
        """Vrací mock konfiguraci pro testování (zachovává zpětnou kompatibilitu)"""
        return {
            "research_profile": self.app.research_profile,
            "max_documents": self.app.max_documents,
            "synthesis": {"max_claims": self.app.max_claims},
            "retrieval": {"timeout_seconds": self.app.timeout_seconds},
            "validation_gates": {
                "min_citations_per_claim": self.app.min_citations_per_claim,
                "min_independent_sources": self.app.min_independent_sources,
                "max_hallucination_rate": self.app.max_hallucination_rate,
                "robots_txt_respect": self.app.robots_txt_respect,
                "rate_limit_respect": self.app.rate_limit_respect,
                "min_recall_at_10": self.app.min_recall_at_10,
                "min_ndcg_at_10": self.app.min_ndcg_at_10,
                "min_citation_precision": self.app.min_citation_precision,
                "max_context_usage": self.app.max_context_usage,
                "pii_redaction_enabled": self.app.pii_redaction_enabled,
                "secret_scanning_enabled": self.app.secret_scanning_enabled,
                "min_groundedness": self.app.min_groundedness,
                "min_evidence_coverage": self.app.min_evidence_coverage,
            },
            "m1_optimization": {
                "enabled": self.app.m1_optimization_enabled,
                "ollama": {"host": self.app.ollama_host}
            },
        }


# Global settings instance
settings = Settings()


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Legacy function for backward compatibility
    Vrací mock konfiguraci pro existující kód
    """
    if config_path:
        global_settings = Settings(config_path)
    else:
        global_settings = settings
    
    return global_settings.get_mock_config_for_testing()


def get_settings() -> Settings:
    """Vrací globální settings instanci"""
    return settings


def validate_environment() -> None:
    """Validuje kritické proměnné prostředí při startu aplikace"""
    validation = settings.validate_critical_config()
    
    missing_configs = [key for key, valid in validation.items() if not valid]
    
    if missing_configs:
        print("⚠️  Warning: Missing critical configuration:")
        for config in missing_configs:
            print(f"   - {config}")
        print("Check your .env file and environment variables")
    else:
        print("✅ Configuration validation passed")


if __name__ == "__main__":
    # Test konfigurace
    validate_environment()
    print("Settings loaded successfully!")