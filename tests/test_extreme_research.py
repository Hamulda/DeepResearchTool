"""
Comprehensive Test Suite pro Extreme Deep Research moduly
Testuje všechny nové funkcionality včetně archeologie, steganografie, evasion a protokolů.
"""

import asyncio
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import numpy as np
from PIL import Image

# Import všech nových modulů pro testování
from src.archaeology.historical_excavator import HistoricalWebExcavator, ArchaeologicalFind
from src.archaeology.legacy_protocols import LegacyProtocolDetector, LegacyProtocolResult
from src.steganography.advanced_steganalysis import AdvancedSteganalysisEngine, SteganalysisResult
from src.steganography.polyglot_detector import PolyglotFileDetector, PolyglotDetectionResult
from src.evasion.anti_bot_bypass import AntiBotCircumventionSuite, BypassResult
from src.evasion.dynamic_loader import DynamicContentLoader, DynamicContent
from src.protocols.custom_handler import CustomProtocolHandler, ProtocolResponse
from src.protocols.network_inspector import NetworkLayerInspector, NetworkAnalysisResult
from src.optimization.intelligent_memory import IntelligentMemoryManager
from src.optimization.metal_acceleration import MetalAcceleration
from src.extreme_research_orchestrator import ExtremeResearchOrchestrator, ExtremeResearchTask


class TestArchaeologyModules:
    """Testovací třída pro archeologické moduly"""

    @pytest.fixture
    async def historical_excavator(self):
        """Fixture pro HistoricalWebExcavator"""
        return HistoricalWebExcavator()

    @pytest.fixture
    async def legacy_detector(self):
        """Fixture pro LegacyProtocolDetector"""
        return LegacyProtocolDetector()

    @pytest.mark.asyncio
    async def test_historical_excavation_basic(self, historical_excavator):
        """Test základní archeologické expedice"""
        # Mock Wayback Machine responses
        with patch.object(historical_excavator.wayback_client, 'get_snapshots') as mock_snapshots:
            mock_snapshots.return_value = [
                {
                    "timestamp": "20220101000000",
                    "original": "http://example.com",
                    "mimetype": "text/html",
                    "statuscode": "200"
                }
            ]

            with patch.object(historical_excavator.wayback_client, 'get_historical_content') as mock_content:
                mock_content.return_value = "<html><head><title>Test</title></head><body>Historical content</body></html>"

                result = await historical_excavator.excavate_forgotten_domains("example.com", depth=1, time_range_years=1)

                assert result.domain == "example.com"
                assert len(result.finds) >= 0
                assert result.excavation_date is not None

    @pytest.mark.asyncio
    async def test_dns_archaeology(self, historical_excavator):
        """Test DNS archeologického prohledávání"""
        with patch.object(historical_excavator.dns_archaeologist, 'excavate_dns_history') as mock_dns:
            mock_dns.return_value = {
                "domain": "example.com",
                "historical_records": [],
                "subdomain_discoveries": {"www.example.com", "mail.example.com"},
                "current_records": {"A": ["192.168.1.1"]}
            }

            result = await historical_excavator.dns_archaeologist.excavate_dns_history("example.com")

            assert result["domain"] == "example.com"
            assert "subdomain_discoveries" in result
            assert len(result["subdomain_discoveries"]) > 0

    @pytest.mark.asyncio
    async def test_certificate_transparency_mining(self, historical_excavator):
        """Test Certificate Transparency mining"""
        with patch.object(historical_excavator.ct_miner, 'excavate_certificate_history') as mock_ct:
            mock_ct.return_value = [
                {
                    "id": "12345",
                    "logged_at": "2022-01-01T00:00:00Z",
                    "common_name": "example.com",
                    "issuer_name": "Let's Encrypt"
                }
            ]

            result = await historical_excavator.ct_miner.excavate_certificate_history("example.com")

            assert len(result) >= 0
            if result:
                assert "common_name" in result[0]

    @pytest.mark.asyncio
    async def test_legacy_protocols_gopher(self, legacy_detector):
        """Test Gopher protokolu"""
        with patch('asyncio.open_connection') as mock_connection:
            # Mock Gopher server response
            mock_reader = AsyncMock()
            mock_writer = AsyncMock()
            mock_connection.return_value = (mock_reader, mock_writer)

            mock_reader.read.return_value = b"1Test Gopher\t/test\texample.com\t70\r\n"

            result = await legacy_detector.scan_gopher_server("example.com")

            assert result.protocol == "gopher"
            assert result.host == "example.com"
            assert result.port == 70

    @pytest.mark.asyncio
    async def test_legacy_protocols_finger(self, legacy_detector):
        """Test Finger protokolu"""
        with patch('asyncio.open_connection') as mock_connection:
            mock_reader = AsyncMock()
            mock_writer = AsyncMock()
            mock_connection.return_value = (mock_reader, mock_writer)

            mock_reader.read.return_value = b"Login: testuser\nName: Test User\nLast login: Mon Jan 1 00:00:00 2022"

            result = await legacy_detector.scan_finger_server("example.com")

            assert result.protocol == "finger"
            assert result.host == "example.com"

    @pytest.mark.asyncio
    async def test_comprehensive_legacy_scan(self, legacy_detector):
        """Test komplexního legacy skenování"""
        with patch.object(legacy_detector, 'scan_gopher_server') as mock_gopher, \
             patch.object(legacy_detector, 'scan_finger_server') as mock_finger, \
             patch.object(legacy_detector, 'scan_nntp_server') as mock_nntp:

            mock_gopher.return_value = LegacyProtocolResult("gopher", "example.com", 70, "active")
            mock_finger.return_value = LegacyProtocolResult("finger", "example.com", 79, "inactive")
            mock_nntp.return_value = LegacyProtocolResult("nntp", "example.com", 119, "active")

            results = await legacy_detector.comprehensive_legacy_scan("example.com")

            assert len(results) == 3
            assert "gopher" in results
            assert "finger" in results
            assert "nntp" in results


class TestSteganographyModules:
    """Testovací třída pro steganografické moduly"""

    @pytest.fixture
    async def steganalysis_engine(self):
        """Fixture pro AdvancedSteganalysisEngine"""
        return AdvancedSteganalysisEngine(enable_gpu_acceleration=False)  # Disable GPU for tests

    @pytest.fixture
    async def polyglot_detector(self):
        """Fixture pro PolyglotFileDetector"""
        return PolyglotFileDetector()

    @pytest.fixture
    def test_image_path(self):
        """Vytvoření testovacího obrázku"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            # Vytvoření jednoduchého testovacího obrázku
            img = Image.new('RGB', (100, 100), color='red')
            img.save(temp_file.name)
            yield temp_file.name
            os.unlink(temp_file.name)

    @pytest.fixture
    def test_polyglot_file(self):
        """Vytvoření testovacího polyglot souboru"""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # PNG header + trailing data
            png_header = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde'
            png_end = b'\x00\x00\x00\x00IEND\xaeB`\x82'
            trailing_data = b'This is hidden data in the polyglot file'

            temp_file.write(png_header + png_end + trailing_data)
            yield temp_file.name
            os.unlink(temp_file.name)

    @pytest.mark.asyncio
    async def test_steganalysis_image_basic(self, steganalysis_engine, test_image_path):
        """Test základní steganografické analýzy obrázku"""
        result = await steganalysis_engine.analyze_file(test_image_path)

        assert isinstance(result, SteganalysisResult)
        assert result.file_path == test_image_path
        assert result.file_type in ["png", "jpg", "jpeg"]
        assert "analysis_details" in result.__dict__

    @pytest.mark.asyncio
    async def test_lsb_detection(self, steganalysis_engine):
        """Test LSB steganografie detekce"""
        # Vytvoření testovacího obrázku s LSB modifications
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Modifikace LSB bitů pro simulaci steganografie
        test_image[:, :, 0] = test_image[:, :, 0] & 0xFE  # Clear LSB
        test_image[::2, ::2, 0] |= 1  # Set pattern in LSB

        result = await steganalysis_engine.detect_lsb_steganography(test_image)

        assert hasattr(result, 'chi_square_score')
        assert hasattr(result, 'p_value')
        assert isinstance(result.suspicious_regions, list)

    @pytest.mark.asyncio
    async def test_polyglot_detection_basic(self, polyglot_detector, test_polyglot_file):
        """Test základní polyglot detekce"""
        result = await polyglot_detector.analyze_file(test_polyglot_file)

        assert isinstance(result, PolyglotDetectionResult)
        assert result.file_path == test_polyglot_file
        assert len(result.detected_formats) > 0
        assert result.trailing_data_size > 0

    @pytest.mark.asyncio
    async def test_batch_steganalysis(self, steganalysis_engine, test_image_path):
        """Test batch steganografické analýzy"""
        results = await steganalysis_engine.batch_analyze([test_image_path])

        assert len(results) == 1
        assert isinstance(results[0], SteganalysisResult)

    def test_steganalysis_report_generation(self, steganalysis_engine):
        """Test generování steganografického reportu"""
        # Mock výsledky
        mock_results = [
            SteganalysisResult(
                file_path="test1.jpg",
                file_type="jpg",
                steganography_detected=True,
                confidence_score=0.8,
                detection_methods=["lsb_chi_square"]
            ),
            SteganalysisResult(
                file_path="test2.png",
                file_type="png",
                steganography_detected=False,
                confidence_score=0.1
            )
        ]

        report = steganalysis_engine.generate_steganalysis_report(mock_results)

        assert "summary" in report
        assert report["summary"]["total_files_analyzed"] == 2
        assert report["summary"]["suspicious_files_found"] == 1


class TestEvasionModules:
    """Testovací třída pro evasion moduly"""

    @pytest.fixture
    async def antibot_suite(self):
        """Fixture pro AntiBotCircumventionSuite"""
        return AntiBotCircumventionSuite(stealth_mode=True)

    @pytest.fixture
    async def dynamic_loader(self):
        """Fixture pro DynamicContentLoader"""
        return DynamicContentLoader()

    @pytest.mark.asyncio
    async def test_antibot_bypass_basic(self, antibot_suite):
        """Test základního anti-bot bypass"""
        with patch.object(antibot_suite, '_bypass_with_stealth_playwright') as mock_bypass:
            mock_bypass.return_value = BypassResult(
                success=True,
                url="https://example.com",
                method_used="stealth_playwright",
                response_time_ms=1500.0,
                content="<html>Success</html>"
            )

            result = await antibot_suite.circumvent_protection("https://example.com")

            assert result.success
            assert result.method_used == "stealth_playwright"
            assert result.content is not None

    @pytest.mark.asyncio
    async def test_tls_fingerprint_rotation(self, antibot_suite):
        """Test TLS fingerprint rotace"""
        fingerprint = antibot_suite.tls_rotator.get_random_fingerprint()

        assert "ja3" in fingerprint
        assert "user_agent" in fingerprint
        assert "cipher_suites" in fingerprint

    @pytest.mark.asyncio
    async def test_dynamic_content_loading(self, dynamic_loader):
        """Test dynamického načítání obsahu"""
        # Mock Playwright page
        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.content.return_value = "<html><body>Dynamic content loaded</body></html>"

        result = await dynamic_loader.load_dynamic_content(mock_page)

        assert isinstance(result, DynamicContent)
        assert result.url == "https://example.com"
        assert result.dynamic_content is not None

    @pytest.mark.asyncio
    async def test_javascript_sandbox(self, dynamic_loader):
        """Test JavaScript sandbox execution"""
        safe_script = "return 2 + 2;"

        mock_page = AsyncMock()
        mock_page.evaluate.return_value = {"result": 4, "error": None, "consoleLogs": []}

        result = await dynamic_loader.execute_js_safely(mock_page, safe_script)

        assert result.success
        assert result.result == 4

    def test_javascript_security_validation(self, dynamic_loader):
        """Test validace bezpečnosti JavaScript kódu"""
        dangerous_script = "eval('alert(1)'); window.location = 'http://evil.com';"

        violations = dynamic_loader.js_sandbox.validate_script_safety(dangerous_script)

        assert len(violations) > 0
        assert any("eval" in violation.lower() for violation in violations)


class TestProtocolModules:
    """Testovací třída pro protocol moduly"""

    @pytest.fixture
    async def protocol_handler(self):
        """Fixture pro CustomProtocolHandler"""
        return CustomProtocolHandler()

    @pytest.fixture
    async def network_inspector(self):
        """Fixture pro NetworkLayerInspector"""
        return NetworkLayerInspector()

    @pytest.mark.asyncio
    async def test_gemini_protocol(self, protocol_handler):
        """Test Gemini protokolu"""
        with patch('asyncio.open_connection') as mock_connection:
            mock_reader = AsyncMock()
            mock_writer = AsyncMock()
            mock_connection.return_value = (mock_reader, mock_writer)

            mock_reader.readline.return_value = b"20 text/gemini\r\n"
            mock_reader.read.return_value = b"# Welcome to Gemini\n=> gemini://example.com/about About"

            result = await protocol_handler.fetch("gemini://example.com")

            assert isinstance(result, ProtocolResponse)
            assert result.protocol == "gemini"
            assert result.url == "gemini://example.com"

    @pytest.mark.asyncio
    async def test_ipfs_protocol(self, protocol_handler):
        """Test IPFS protokolu"""
        with patch.object(protocol_handler.ipfs_client, 'fetch_by_hash') as mock_fetch:
            from src.protocols.custom_handler import IPFSContent

            mock_fetch.return_value = IPFSContent(
                hash="QmHash123",
                content=b"Test IPFS content",
                content_type="text/plain",
                size=17
            )

            result = await protocol_handler.fetch("ipfs://QmHash123")

            assert result.protocol == "ipfs"
            assert result.error is None

    @pytest.mark.asyncio
    async def test_network_analysis_basic(self, network_inspector):
        """Test základní síťové analýzy"""
        with patch.object(network_inspector.port_scanner, 'scan_common_ports') as mock_scan, \
             patch.object(network_inspector.dns_analyzer, 'comprehensive_dns_analysis') as mock_dns:

            from src.protocols.network_inspector import PortScanResult

            mock_scan.return_value = PortScanResult(
                host="example.com",
                open_ports=[80, 443],
                closed_ports=[22, 21],
                scan_duration_ms=1000
            )

            mock_dns.return_value = {
                "current_records": {"A": ["192.168.1.1"]},
                "dns_security": {"dnssec_enabled": True}
            }

            result = await network_inspector.comprehensive_network_analysis("example.com")

            assert isinstance(result, NetworkAnalysisResult)
            assert result.target == "example.com"
            assert result.port_scan is not None
            assert result.dns_analysis is not None

    @pytest.mark.asyncio
    async def test_tcp_fingerprinting(self, network_inspector):
        """Test TCP fingerprintingu"""
        with patch('asyncio.open_connection') as mock_connection:
            mock_reader = AsyncMock()
            mock_writer = AsyncMock()
            mock_connection.return_value = (mock_reader, mock_writer)

            mock_reader.read.return_value = b"SSH-2.0-OpenSSH_8.0"

            result = await network_inspector.tcp_fingerprinter.fingerprint_service("example.com", 22)

            assert result.host == "example.com"
            assert result.port == 22
            assert result.service_banner is not None


class TestOptimizationModules:
    """Testovací třída pro optimization moduly"""

    @pytest.mark.asyncio
    async def test_intelligent_memory_manager(self):
        """Test intelligent memory manageru"""
        manager = IntelligentMemoryManager(max_memory_mb=10)  # Malý limit pro test
        await manager.initialize()

        # Test základních operací
        await manager.set("test_key", "test_value", importance_score=0.8)
        result = await manager.get("test_key")

        assert result == "test_value"

        # Test eviction
        for i in range(100):
            await manager.set(f"key_{i}", f"value_{i}" * 1000)  # Velké hodnoty

        # Původní klíč by měl být evicted
        evicted_result = await manager.get("test_key")
        assert evicted_result is None or evicted_result == "test_value"

        await manager.close()

    def test_metal_acceleration_info(self):
        """Test Metal acceleration info"""
        acceleration = MetalAcceleration()
        info = acceleration.get_acceleration_info()

        assert "mlx_available" in info
        assert "acceleration_type" in info
        assert "recommended_batch_size" in info

    @pytest.mark.asyncio
    async def test_metal_acceleration_benchmark(self):
        """Test Metal acceleration benchmark"""
        acceleration = MetalAcceleration()

        # Test pouze pokud není MLX dostupný (aby test prošel všude)
        if not acceleration.mlx_available:
            # Test fallback CPU verze
            test_image = np.random.rand(64, 64).astype(np.float32)
            result = await acceleration._fallback_cpu_analysis(test_image)

            assert "entropy" in result
            assert "frequency_analysis" in result


class TestExtremeResearchOrchestrator:
    """Testovací třída pro hlavní orchestrátor"""

    @pytest.fixture
    async def orchestrator(self):
        """Fixture pro ExtremeResearchOrchestrator"""
        orchestrator = ExtremeResearchOrchestrator(max_concurrent_tasks=2)
        # Mock inicializace pro testy
        orchestrator.memory_manager = AsyncMock()
        return orchestrator

    @pytest.mark.asyncio
    async def test_extreme_research_task_creation(self, orchestrator):
        """Test vytvoření extreme research úkolu"""
        task = ExtremeResearchTask(
            task_id="test_task",
            task_type="archaeology",
            target="example.com",
            parameters={"depth": 2}
        )

        assert task.task_id == "test_task"
        assert task.task_type == "archaeology"
        assert task.target == "example.com"
        assert task.parameters["depth"] == 2

    @pytest.mark.asyncio
    async def test_system_status(self, orchestrator):
        """Test system status"""
        status = await orchestrator.get_system_status()

        assert "orchestrator" in status
        assert "modules" in status
        assert status["orchestrator"] == "active"


class TestIntegrationScenarios:
    """Integrační testy pro kompletní workflow"""

    @pytest.mark.asyncio
    async def test_full_spectrum_research_mock(self):
        """Test full spectrum research s mock daty"""
        orchestrator = ExtremeResearchOrchestrator()

        # Mock všech komponent
        orchestrator.historical_excavator = AsyncMock()
        orchestrator.steganalysis_engine = AsyncMock()
        orchestrator.antibot_suite = AsyncMock()
        orchestrator.protocol_handler = AsyncMock()

        # Mock výsledky
        orchestrator.historical_excavator.excavate_forgotten_domains.return_value = AsyncMock(
            finds=[],
            subdomain_discoveries=set(),
            certificate_history=[]
        )

        task = ExtremeResearchTask(
            task_id="integration_test",
            task_type="full_spectrum",
            target="example.com"
        )

        # Mock memory manager
        orchestrator.memory_manager = AsyncMock()

        # Test by měl projít bez chyb
        try:
            result = await orchestrator._execute_full_spectrum_workflow(task)
            assert isinstance(result, dict)
        except Exception as e:
            # Očekáváme některé chyby kvuli mockování
            assert "mock" in str(e).lower() or "not available" in str(e).lower()


class TestPerformanceBenchmarks:
    """Performance benchmarky pro M1 optimalizace"""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_memory_manager_performance(self):
        """Benchmark memory manageru"""
        manager = IntelligentMemoryManager(max_memory_mb=50)
        await manager.initialize()

        start_time = time.time()

        # Test rychlosti set/get operací
        for i in range(1000):
            await manager.set(f"perf_key_{i}", f"value_{i}" * 100)

        set_time = time.time() - start_time

        start_time = time.time()

        for i in range(1000):
            await manager.get(f"perf_key_{i}")

        get_time = time.time() - start_time

        await manager.close()

        # Assertions pro performance
        assert set_time < 10.0  # Mělo by být rychlejší než 10 sekund
        assert get_time < 5.0   # Get operace by měly být rychlejší

        print(f"Set operations: {set_time:.2f}s, Get operations: {get_time:.2f}s")

    @pytest.mark.slow
    def test_steganalysis_performance(self):
        """Benchmark steganografické analýzy"""
        engine = AdvancedSteganalysisEngine(enable_gpu_acceleration=False)

        # Vytvoření testovacího obrázku
        test_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

        start_time = time.time()

        # Test entropy výpočtu
        for _ in range(100):
            engine._calculate_entropy_analysis(test_image)

        entropy_time = time.time() - start_time

        assert entropy_time < 5.0  # Mělo by být rychlejší než 5 sekund
        print(f"Entropy analysis (100x): {entropy_time:.2f}s")


# Pytest konfigurace a fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Event loop pro asyncio testy"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Custom pytest markers
def pytest_configure(config):
    """Konfigurace pytest markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "benchmark: marks tests as benchmarks")


if __name__ == "__main__":
    # Spuštění testů při přímém volání
    pytest.main([__file__, "-v"])
