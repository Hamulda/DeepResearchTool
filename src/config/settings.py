#!/usr/bin/env python3
"""Centralizovaná správa konfigurace pomocí pydantic-settings
Všechny konfigurační parametry a tajemství v jednom místě

Author: Senior Python/MLOps Agent
"""

from functools import lru_cache
from pathlib import Path

from pydantic import BaseSettings, Field, SecretStr, validator
from pydantic_settings import BaseSettings as PydanticBaseSettings


class DatabaseSettings(BaseSettings):
    """Nastavení databází"""

    # Redis
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_password: SecretStr | None = Field(default=None, description="Redis password")
    redis_db: int = Field(default=0, description="Redis database number")

    # PostgreSQL
    postgres_host: str = Field(default="localhost", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_user: str = Field(default="deepresearch", description="PostgreSQL user")
    postgres_password: SecretStr = Field(description="PostgreSQL password")
    postgres_db: str = Field(default="deepresearch", description="PostgreSQL database")

    # Elasticsearch
    elasticsearch_url: str = Field(default="http://localhost:9200", description="Elasticsearch URL")
    elasticsearch_api_key: SecretStr | None = Field(
        default=None, description="Elasticsearch API key"
    )

    # Qdrant Vector DB
    qdrant_host: str = Field(default="localhost", description="Qdrant host")
    qdrant_port: int = Field(default=6333, description="Qdrant port")
    qdrant_api_key: SecretStr | None = Field(default=None, description="Qdrant API key")

    class Config:
        env_prefix = "DB_"


class AIModelSettings(BaseSettings):
    """Nastavení AI modelů a API"""

    # OpenAI
    openai_api_key: SecretStr = Field(description="OpenAI API key")
    openai_model: str = Field(default="gpt-4", description="OpenAI model name")
    openai_embedding_model: str = Field(
        default="text-embedding-ada-002", description="OpenAI embedding model"
    )
    openai_max_tokens: int = Field(default=4000, description="Max tokens per request")
    openai_temperature: float = Field(default=0.7, description="Model temperature")

    # Anthropic
    anthropic_api_key: SecretStr | None = Field(default=None, description="Anthropic API key")
    anthropic_model: str = Field(
        default="claude-3-sonnet-20240229", description="Anthropic model name"
    )

    # Ollama (local models)
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama base URL")
    ollama_model: str = Field(default="llama2", description="Ollama model name")

    # Model limits
    max_concurrent_requests: int = Field(default=10, description="Max concurrent API requests")
    request_timeout: int = Field(default=120, description="Request timeout in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")

    class Config:
        env_prefix = "AI_"


class SecuritySettings(BaseSettings):
    """Bezpečnostní nastavení"""

    # Encryption
    secret_key: SecretStr = Field(description="Application secret key")
    jwt_secret: SecretStr = Field(description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_hours: int = Field(default=24, description="JWT token expiration in hours")

    # API security
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    rate_limit_per_minute: int = Field(default=100, description="Rate limit per minute")

    # TOR settings
    tor_proxy_host: str = Field(default="127.0.0.1", description="TOR proxy host")
    tor_proxy_port: int = Field(default=9050, description="TOR proxy port")
    tor_control_port: int = Field(default=9051, description="TOR control port")
    tor_control_password: SecretStr | None = Field(
        default=None, description="TOR control password"
    )

    # Security compliance
    enable_data_encryption: bool = Field(default=True, description="Enable data encryption at rest")
    enable_audit_logging: bool = Field(default=True, description="Enable audit logging")
    pii_detection_enabled: bool = Field(default=True, description="Enable PII detection")

    class Config:
        env_prefix = "SECURITY_"


class ScrapingSettings(BaseSettings):
    """Nastavení pro web scraping"""

    # User agents
    default_user_agent: str = Field(
        default="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        description="Default user agent",
    )

    # Request settings
    request_delay: float = Field(default=1.0, description="Delay between requests in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    timeout: int = Field(default=30, description="Request timeout in seconds")

    # Proxy settings
    use_proxies: bool = Field(default=False, description="Enable proxy usage")
    proxy_list: list[str] = Field(default_factory=list, description="List of proxy URLs")

    # Rate limiting
    requests_per_second: float = Field(default=1.0, description="Maximum requests per second")
    concurrent_requests: int = Field(default=5, description="Maximum concurrent requests")

    # Content limits
    max_content_size: int = Field(default=10485760, description="Max content size in bytes (10MB)")
    max_pages_per_domain: int = Field(default=1000, description="Max pages per domain")

    class Config:
        env_prefix = "SCRAPING_"


class ProcessingSettings(BaseSettings):
    """Nastavení pro zpracování dat"""

    # Chunking
    chunk_size: int = Field(default=1000, description="Default chunk size for text processing")
    chunk_overlap: int = Field(default=200, description="Chunk overlap size")
    max_chunk_size: int = Field(default=4000, description="Maximum chunk size")

    # Processing limits
    max_file_size: int = Field(default=104857600, description="Max file size in bytes (100MB)")
    max_documents_per_batch: int = Field(
        default=100, description="Max documents per processing batch"
    )

    # Language processing
    default_language: str = Field(default="en", description="Default language for processing")
    supported_languages: list[str] = Field(
        default_factory=lambda: ["en", "cs", "sk", "de", "fr", "es"],
        description="Supported languages",
    )

    # Embedding settings
    embedding_dimension: int = Field(default=1536, description="Embedding vector dimension")
    similarity_threshold: float = Field(
        default=0.7, description="Similarity threshold for matching"
    )

    class Config:
        env_prefix = "PROCESSING_"


class MonitoringSettings(BaseSettings):
    """Nastavení pro monitoring a observability"""

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json/text)")
    log_file_path: str | None = Field(default=None, description="Log file path")

    # Metrics
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=8000, description="Metrics server port")

    # Tracing
    enable_tracing: bool = Field(default=True, description="Enable distributed tracing")
    jaeger_endpoint: str | None = Field(default=None, description="Jaeger endpoint")

    # Performance monitoring
    enable_profiling: bool = Field(default=False, description="Enable performance profiling")
    profile_sample_rate: float = Field(default=0.1, description="Profiling sample rate")

    class Config:
        env_prefix = "MONITORING_"


class ApplicationSettings(PydanticBaseSettings):
    """Hlavní konfigurace aplikace"""

    # Application info
    app_name: str = Field(default="DeepResearchTool", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: str = Field(
        default="development", description="Environment (development/staging/production)"
    )
    debug: bool = Field(default=False, description="Debug mode")

    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8080, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")

    # Paths
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    cache_dir: Path = Field(default=Path("cache"), description="Cache directory")
    logs_dir: Path = Field(default=Path("logs"), description="Logs directory")

    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    ai_models: AIModelSettings = Field(default_factory=AIModelSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    scraping: ScrapingSettings = Field(default_factory=ScrapingSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)

    @validator("environment")
    def validate_environment(cls, v):
        """Validace prostředí"""
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of: {allowed}")
        return v

    @validator("data_dir", "cache_dir", "logs_dir")
    def create_directories(cls, v):
        """Vytvoření adresářů pokud neexistují"""
        v.mkdir(parents=True, exist_ok=True)
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache
def get_settings() -> ApplicationSettings:
    """Získání singleton instance nastavení"""
    return ApplicationSettings()


# Aliasy pro pohodlnější použití
settings = get_settings()
