#!/usr/bin/env python3
"""
Bezpečný konfigurační modul pro DeepResearchTool
Všechny citlivé údaje jsou načítány z proměnných prostředí
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseSettings, Field, validator
from pydantic_settings import BaseSettings as PydanticBaseSettings


class DatabaseConfig(PydanticBaseSettings):
    """Konfigurace databází"""
    
    # PostgreSQL/Vector DB
    database_url: str = Field(default="", env="DATABASE_URL")
    pgvector_host: str = Field(default="localhost", env="PGVECTOR_HOST")
    pgvector_port: int = Field(default=5432, env="PGVECTOR_PORT")
    pgvector_database: str = Field(default="vectors", env="PGVECTOR_DATABASE")
    pgvector_user: str = Field(default="", env="PGVECTOR_USER")
    pgvector_password: str = Field(default="", env="PGVECTOR_PASSWORD")
    
    # Vector stores
    chroma_host: str = Field(default="http://localhost:8001", env="CHROMA_HOST")
    qdrant_host: str = Field(default="http://localhost:6333", env="QDRANT_HOST")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    
    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field(default="", env="NEO4J_USER")
    neo4j_password: str = Field(default="", env="NEO4J_PASSWORD")


class LLMConfig(PydanticBaseSettings):
    """Konfigurace pro LLM služby"""
    
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    ollama_host: str = Field(default="http://localhost:11434", env="OLLAMA_HOST")
    
    # Langfuse observability
    langfuse_secret_key: str = Field(default="", env="LANGFUSE_SECRET_KEY")
    langfuse_public_key: str = Field(default="", env="LANGFUSE_PUBLIC_KEY")
    langfuse_host: str = Field(default="http://localhost:3000", env="LANGFUSE_HOST")
    langfuse_enabled: bool = Field(default=False, env="LANGFUSE_ENABLED")


class ScrapingConfig(PydanticBaseSettings):
    """Konfigurace pro scraping služby"""
    
    # Enterprise scraping
    bright_data_username: str = Field(default="", env="BRIGHT_DATA_USERNAME")
    bright_data_password: str = Field(default="", env="BRIGHT_DATA_PASSWORD")
    apify_api_token: str = Field(default="", env="APIFY_API_TOKEN")
    scraperapi_key: str = Field(default="", env="SCRAPERAPI_KEY")
    
    # Tor configuration
    tor_socks_port: int = Field(default=9050, env="TOR_SOCKS_PORT")
    tor_control_port: int = Field(default=9051, env="TOR_CONTROL_PORT")
    tor_control_password: str = Field(default="", env="TOR_CONTROL_PASSWORD")


class SecurityConfig(PydanticBaseSettings):
    """Bezpečnostní konfigurace"""
    
    jwt_secret_key: str = Field(default="", env="JWT_SECRET_KEY")
    api_rate_limit: int = Field(default=100, env="API_RATE_LIMIT")
    api_rate_window: int = Field(default=60, env="API_RATE_WINDOW")
    
    # Security flags
    pii_redaction_enabled: bool = Field(default=True, env="PII_REDACTION_ENABLED")
    secret_scanning_enabled: bool = Field(default=True, env="SECRET_SCANNING_ENABLED")
    robots_txt_respect: bool = Field(default=True, env="ROBOTS_TXT_RESPECT")
    rate_limit_respect: bool = Field(default=True, env="RATE_LIMIT_RESPECT")


class MonitoringConfig(PydanticBaseSettings):
    """Konfigurace monitoringu"""
    
    prometheus_url: str = Field(default="http://localhost:9090", env="PROMETHEUS_URL")
    grafana_url: str = Field(default="http://localhost:3001", env="GRAFANA_URL")
    alert_webhook_url: str = Field(default="", env="ALERT_WEBHOOK_URL")
    
    # Sentry
    sentry_dsn: str = Field(default="", env="SENTRY_DSN")
    sentry_environment: str = Field(default="development", env="SENTRY_ENVIRONMENT")


class AppConfig(PydanticBaseSettings):
    """Hlavní aplikační konfigurace"""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    workers: int = Field(default=1, env="WORKERS")
    
    # Application settings
    research_profile: str = Field(default="thorough")
    max_documents: int = Field(default=50)
    max_claims: int = Field(default=10)
    timeout_seconds: int = Field(default=120)
    
    # M1 Optimization
    m1_optimization_enabled: bool = Field(default=True, env="M1_OPTIMIZATION_ENABLED")
    
    # Validation gates
    min_citations_per_claim: int = Field(default=2)
    min_independent_sources: int = Field(default=2)
    max_hallucination_rate: float = Field(default=0.05)
    min_recall_at_10: float = Field(default=0.7)
    min_ndcg_at_10: float = Field(default=0.6)
    min_citation_precision: float = Field(default=0.8)
    max_context_usage: float = Field(default=0.9)
    min_groundedness: float = Field(default=0.8)
    min_evidence_coverage: float = Field(default=0.7)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class Settings:
    """Centrální settings manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.app = AppConfig()
        self.database = DatabaseConfig()
        self.llm = LLMConfig()
        self.scraping = ScrapingConfig()
        self.security = SecurityConfig()
        self.monitoring = MonitoringConfig()
        
        # Load additional config from YAML if provided
        if config_path and Path(config_path).exists():
            self._load_yaml_config(config_path)
    
    def _load_yaml_config(self, config_path: str) -> None:
        """Načte dodatečnou konfiguraci z YAML souboru (bez citlivých údajů)"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f)
                
            # Update non-sensitive settings only
            if "research_profile" in yaml_config:
                self.app.research_profile = yaml_config["research_profile"]
            if "max_documents" in yaml_config:
                self.app.max_documents = yaml_config["max_documents"]
            # Add other non-sensitive overrides as needed
                
        except Exception as e:
            # Log warning but don't fail - environment variables are primary
            print(f"Warning: Could not load YAML config from {config_path}: {e}")
    
    def validate_critical_config(self) -> Dict[str, bool]:
        """Validuje kritické konfigurační hodnoty"""
        validation_results = {
            "has_llm_api_key": bool(self.llm.openai_api_key or self.llm.anthropic_api_key),
            "has_database_config": bool(self.database.database_url or self.database.redis_url),
            "has_security_config": bool(self.security.jwt_secret_key),
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
                "robots_txt_respect": self.security.robots_txt_respect,
                "rate_limit_respect": self.security.rate_limit_respect,
                "min_recall_at_10": self.app.min_recall_at_10,
                "min_ndcg_at_10": self.app.min_ndcg_at_10,
                "min_citation_precision": self.app.min_citation_precision,
                "max_context_usage": self.app.max_context_usage,
                "pii_redaction_enabled": self.security.pii_redaction_enabled,
                "secret_scanning_enabled": self.security.secret_scanning_enabled,
                "min_groundedness": self.app.min_groundedness,
                "min_evidence_coverage": self.app.min_evidence_coverage,
            },
            "m1_optimization": {
                "enabled": self.app.m1_optimization_enabled,
                "ollama": {"host": self.llm.ollama_host}
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