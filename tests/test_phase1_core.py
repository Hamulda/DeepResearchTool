#!/usr/bin/env python3
"""
Základní testy pro kritické komponenty Fáze 1
Validuje konfigurační systém a error handling
"""

import pytest
import asyncio
import os
from unittest.mock import Mock, patch, AsyncMock
import tempfile
from pathlib import Path

from src.core.config import Settings, validate_environment, get_settings
from src.core.error_handling import (
    network_retry,
    scraping_retry,
    ErrorAggregator,
    CircuitBreaker,
    safe_requests_get,
    safe_aiohttp_get
)


class TestConfigurationSystem:
    """Testy pro nový konfigurační systém"""
    
    def test_settings_initialization(self):
        """Test základní inicializace settings"""
        settings = Settings()
        
        assert settings.app.environment == "development"
        assert settings.app.debug is True
        assert settings.app.research_profile == "thorough"
        assert settings.app.max_documents == 50
        
    def test_config_validation(self):
        """Test validace kritické konfigurace"""
        settings = Settings()
        validation_results = settings.validate_critical_config()
        
        assert isinstance(validation_results, dict)
        assert "has_llm_api_key" in validation_results
        assert "has_database_config" in validation_results
        assert "has_security_config" in validation_results
        assert "environment_set" in validation_results
        
    def test_environment_variable_loading(self):
        """Test načítání z environment variables"""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_key_123',
            'ENVIRONMENT': 'testing',
            'DEBUG': 'false'
        }):
            settings = Settings()
            
            assert settings.llm.openai_api_key == 'test_key_123'
            assert settings.app.environment == 'testing'
            assert settings.app.debug is False
            
    def test_yaml_config_override(self):
        """Test YAML konfigurace přepíše výchozí hodnoty"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
research_profile: "fast"
max_documents: 100
""")
            config_path = f.name
            
        try:
            settings = Settings(config_path)
            assert settings.app.research_profile == "fast"
            assert settings.app.max_documents == 100
        finally:
            os.unlink(config_path)
            
    def test_mock_config_compatibility(self):
        """Test zpětné kompatibility s mock config"""
        settings = Settings()
        mock_config = settings.get_mock_config_for_testing()
        
        assert "research_profile" in mock_config
        assert "validation_gates" in mock_config
        assert "m1_optimization" in mock_config
        assert mock_config["validation_gates"]["min_citations_per_claim"] == 2


class TestErrorHandling:
    """Testy pro robustní error handling"""
    
    def test_error_aggregator(self):
        """Test error aggregator functionality"""
        aggregator = ErrorAggregator()
        
        # Přidání úspěchů a chyb
        aggregator.add_success()
        aggregator.add_success()
        aggregator.add_error(ValueError("test error"), "test context")
        
        summary = aggregator.get_summary()
        
        assert summary["total_operations"] == 3
        assert summary["successful_operations"] == 2
        assert summary["failed_operations"] == 1
        assert summary["success_rate"] == 2/3
        assert len(summary["errors"]) == 1
        
    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker v zavřeném stavu"""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        
        def success_func():
            return "success"
            
        result = cb.call(success_func)
        assert result == "success"
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0
        
    def test_circuit_breaker_open_state(self):
        """Test circuit breaker se otevře po threshold failures"""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60)
        
        def failing_func():
            raise ValueError("test failure")
            
        # První dvě chyby
        with pytest.raises(ValueError):
            cb.call(failing_func)
        assert cb.state == "CLOSED"
        
        with pytest.raises(ValueError):
            cb.call(failing_func)
        assert cb.state == "OPEN"
        
        # Třetí volání by mělo být blokováno circuit breaker
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            cb.call(failing_func)
            
    @pytest.mark.asyncio
    async def test_retry_decorator_success(self):
        """Test retry dekorátoru při úspěšném volání"""
        call_count = 0
        
        @network_retry
        async def test_func():
            nonlocal call_count
            call_count += 1
            return "success"
            
        result = await test_func()
        assert result == "success"
        assert call_count == 1
        
    @pytest.mark.asyncio
    async def test_retry_decorator_eventual_success(self):
        """Test retry dekorátoru s eventual success"""
        call_count = 0
        
        @network_retry
        async def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("temporary failure")
            return "success"
            
        result = await test_func()
        assert result == "success"
        assert call_count == 2
        
    @pytest.mark.asyncio
    async def test_safe_aiohttp_get_mock(self):
        """Test safe aiohttp GET s mock response"""
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = AsyncMock()
        
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        # Test úspěšného volání
        result = await safe_aiohttp_get(mock_session, "http://example.com")
        assert result == mock_response
        mock_response.raise_for_status.assert_called_once()


class TestScrapingIntegration:
    """Integrační testy pro scraping s error handling"""
    
    @pytest.mark.asyncio
    async def test_web_scraper_error_aggregation(self):
        """Test error aggregation ve web scraperu"""
        from src.scrapers.web_scraper import WebScraper
        
        scraper = WebScraper(delay=0.1, timeout=5)
        
        # Test scraping neexistující URL
        result = await scraper.scrape_page("http://nonexistent-url-12345.com")
        assert result == ""  # Měl by vrátit prázdný string při chybě
        
        stats = scraper.get_stats()
        assert stats["failed_operations"] >= 1
        assert stats["success_rate"] < 1.0
        
        await scraper.close()
        
    @pytest.mark.asyncio 
    async def test_web_scraper_multiple_urls(self):
        """Test batch scrapingu s error handling"""
        from src.scrapers.web_scraper import WebScraper
        
        scraper = WebScraper(delay=0.1, timeout=5)
        
        urls = [
            "http://httpbin.org/html",  # Toto by mohlo fungovat
            "http://nonexistent-12345.com",  # Toto selže
            "http://httpbin.org/status/500"  # Toto vrátí 500
        ]
        
        results = await scraper.scrape_multiple(urls, max_concurrent=2)
        
        # Měl by vrátit pouze úspěšné výsledky
        assert isinstance(results, dict)
        # Neočekáváme nutně žádné úspěšné výsledky v testovacím prostředí
        
        stats = scraper.get_stats()
        assert "success_rate" in stats
        assert "total_operations" in stats
        
        await scraper.close()


class TestSecurityValidation:
    """Testy pro bezpečnostní validaci"""
    
    def test_no_hardcoded_secrets_in_config(self):
        """Validuje, že konfigurace neobsahuje hardcoded secrets"""
        settings = Settings()
        
        # Kontrola, že výchozí hodnoty jsou prázdné nebo bezpečné
        assert settings.llm.openai_api_key == ""
        assert settings.llm.anthropic_api_key == ""
        assert settings.database.pgvector_password == ""
        assert settings.security.jwt_secret_key == ""
        
    def test_environment_validation_warns_missing_config(self, capfd):
        """Test, že validate_environment varuje při chybějící konfiguraci"""
        with patch.dict(os.environ, {}, clear=True):
            validate_environment()
            
        captured = capfd.readouterr()
        assert "Warning: Missing critical configuration" in captured.out
        
    def test_pii_redaction_enabled_by_default(self):
        """Test, že PII redaction je defaultně zapnutá"""
        settings = Settings()
        assert settings.security.pii_redaction_enabled is True
        assert settings.security.secret_scanning_enabled is True


if __name__ == "__main__":
    # Spuštění testů
    pytest.main([__file__, "-v", "--tb=short"])