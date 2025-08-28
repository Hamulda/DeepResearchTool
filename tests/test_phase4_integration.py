#!/usr/bin/env python3
"""
FÁZE 4 Integration Test
Test evaluace, M1 výkonu, bezpečnosti a CI gates

Author: Senior Python/MLOps Agent
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, AsyncMock
import pytest

# Import centralizovaných nastavení a logování
from src.config.settings import get_settings
from src.utils.logging import get_logger, get_performance_logger

# Původní importy
from src.evaluation.comprehensive_evaluator import ComprehensiveEvaluator
from src.optimization.m1_device_manager import M1DeviceManager
from src.security.compliance_manager import SecurityManager


@pytest.mark.integration
class TestPhase4Integration:
    """Test suite pro FÁZE 4 komponenty s využitím pytest"""

    @pytest.fixture(autouse=True)
    def setup(self, test_settings):
        """Setup pro každý test"""
        self.settings = test_settings
        self.logger = get_logger("phase4_test", phase=4)
        self.perf_logger = get_performance_logger("phase4_test")

        self.test_results = {
            "phase": 4,
            "start_time": datetime.now().isoformat(),
            "tests": []
        }

        self.logger.info("Starting Phase 4 integration tests",
                        test_session=self.test_results["start_time"])

    def _get_test_config(self):
        """Test konfigurace pro FÁZI 4"""
        return {
            "evaluation": {
                "regression_thresholds": {
                    "recall_at_10": -0.05,
                    "ndcg_at_10": -0.02,
                    "citation_precision": -0.03,
                    "groundedness_score": -0.05
                }
            },
            "performance": {
                "fp16_enabled": True,
                "metal_enabled": True,
                "adaptive_batching": True,
                "streaming_enabled": True
            },
            "security": {
                "allow_domains": ["example.com", "test.com"],
                "deny_domains": ["malicious.com"],
                "default_rate_limits": {
                    "requests_per_minute": 10,
                    "requests_per_hour": 100
                },
                "user_agent": "DeepResearchTool/1.0 (+test@example.com)"
            },
            "ci_gates": {
                "max_smoke_test_time": 60,  # seconds
                "min_claims_required": 1,
                "min_citations_per_claim": 2,
                "max_regression_tolerance": 0.05
            }
        }

    @pytest.mark.asyncio
    async def test_comprehensive_evaluator(self):
        """Test komprehensivního evaluátoru"""
        self.logger.info("Testing comprehensive evaluator")

        try:
            config = self._get_test_config()
            evaluator = ComprehensiveEvaluator(config["evaluation"])

            # Mock research system
            class MockResearchSystem:
                async def process_query(self, query):
                    return {
                        "retrieved_docs": [f"doc_{i}" for i in range(1, 11)],
                        "claims": [f"Test claim about {query}"],
                        "claim_citations": {"Test claim": ["doc_1", "doc_2"]},
                        "context_tokens_used": 2048,
                        "context_tokens_budget": 4096
                    }

            mock_system = MockResearchSystem()

            # Spusť evaluaci na subset domén
            start_time = time.time()
            metrics = await evaluator.run_comprehensive_evaluation(
                mock_system,
                evaluation_domains=["climate_science"]
            )
            evaluation_time = time.time() - start_time

            # Performance logging
            self.perf_logger.timing(
                operation="comprehensive_evaluation",
                duration_ms=evaluation_time * 1000
            )

            # Kontrola výsledků
            assert metrics.total_queries > 0, "Žádné dotazy nebyly evaluovány"
            assert metrics.mean_ndcg_10 > 0, "NDCG@10 musí být větší než 0"
            assert evaluation_time < 300, "Evaluace nesmí trvat více než 5 minut"

            self.logger.info("Comprehensive evaluator test passed",
                           total_queries=metrics.total_queries,
                           mean_ndcg=metrics.mean_ndcg_10,
                           duration_s=evaluation_time)

            return {
                "test": "comprehensive_evaluator",
                "status": "passed",
                "metrics": {
                    "total_queries": metrics.total_queries,
                    "mean_ndcg_10": metrics.mean_ndcg_10,
                    "evaluation_time": evaluation_time
                }
            }

        except Exception as e:
            self.logger.error("Comprehensive evaluator test failed", error=str(e))
            pytest.fail(f"Comprehensive evaluator test failed: {e}")

    @pytest.mark.asyncio
    async def test_m1_performance_optimization(self):
        """Test M1 výkonnostních optimalizací"""
        self.logger.info("Testing M1 performance optimization")

        try:
            config = self._get_test_config()
            m1_manager = M1DeviceManager(
                fp16_enabled=config["performance"]["fp16_enabled"],
                metal_enabled=config["performance"]["metal_enabled"]
            )

            # Test základní funkcionality
            device_info = await m1_manager.get_device_info()
            assert device_info is not None, "Device info nesmí být None"

            # Test optimalizace
            start_time = time.time()
            optimized_result = await m1_manager.optimize_inference(
                model_name="test_model",
                batch_size=4
            )
            optimization_time = time.time() - start_time

            self.perf_logger.timing(
                operation="m1_optimization",
                duration_ms=optimization_time * 1000
            )

            assert optimized_result is not None, "Optimalizace musí vrátit výsledek"
            assert optimization_time < 10, "Optimalizace nesmí trvat více než 10 sekund"

            self.logger.info("M1 performance optimization test passed",
                           device_info=device_info,
                           optimization_time=optimization_time)

            return {
                "test": "m1_performance_optimization",
                "status": "passed",
                "metrics": {
                    "device_info": device_info,
                    "optimization_time": optimization_time
                }
            }

        except Exception as e:
            self.logger.error("M1 performance optimization test failed", error=str(e))
            pytest.fail(f"M1 performance optimization test failed: {e}")

    @pytest.mark.asyncio
    async def test_security_compliance(self):
        """Test bezpečnostního compliance"""
        self.logger.info("Testing security compliance")

        try:
            config = self._get_test_config()
            security_manager = SecurityManager(config["security"])

            # Test rate limiting
            rate_limit_result = await security_manager.check_rate_limit(
                user_id="test_user",
                endpoint="/api/search"
            )
            assert rate_limit_result["allowed"], "Rate limit musí povolit request"

            # Test domain validation
            allowed_domain = security_manager.validate_domain("example.com")
            denied_domain = security_manager.validate_domain("malicious.com")

            assert allowed_domain, "Povolená doména musí projít validací"
            assert not denied_domain, "Zakázaná doména nesmí projít validací"

            # Test PII detection
            pii_result = await security_manager.detect_pii(
                "Test text without sensitive information"
            )
            assert not pii_result["contains_pii"], "Text nesmí obsahovat PII"

            self.logger.info("Security compliance test passed",
                           rate_limit_ok=rate_limit_result["allowed"],
                           domain_validation_ok=True,
                           pii_detection_ok=True)

            return {
                "test": "security_compliance",
                "status": "passed",
                "metrics": {
                    "rate_limit_check": rate_limit_result,
                    "domain_validation": {"allowed": allowed_domain, "denied": denied_domain},
                    "pii_detection": pii_result
                }
            }

        except Exception as e:
            self.logger.error("Security compliance test failed", error=str(e))
            pytest.fail(f"Security compliance test failed: {e}")

    @pytest.mark.asyncio
    async def test_ci_gates_validation(self):
        """Test CI gates validace"""
        self.logger.info("Testing CI gates validation")

        try:
            config = self._get_test_config()

            # Mock systém pro CI gates
            class MockCISystem:
                async def run_smoke_tests(self):
                    await asyncio.sleep(0.1)  # Simulace testů
                    return {"passed": True, "duration": 0.1}

                async def validate_output_quality(self, output):
                    claims = output.get("claims", [])
                    citations = output.get("claim_citations", {})

                    return {
                        "claims_count": len(claims),
                        "citations_per_claim": len(citations) / max(len(claims), 1),
                        "quality_passed": len(claims) >= 1 and len(citations) >= 2
                    }

            ci_system = MockCISystem()

            # Test smoke tests
            start_time = time.time()
            smoke_result = await ci_system.run_smoke_tests()
            smoke_duration = time.time() - start_time

            assert smoke_result["passed"], "Smoke testy musí projít"
            assert smoke_duration < config["ci_gates"]["max_smoke_test_time"], \
                f"Smoke testy nesmí trvat více než {config['ci_gates']['max_smoke_test_time']} sekund"

            # Test výstupní kvality
            mock_output = {
                "claims": ["Test claim 1", "Test claim 2"],
                "claim_citations": {"Test claim 1": ["doc1", "doc2"], "Test claim 2": ["doc3", "doc4"]}
            }

            quality_result = await ci_system.validate_output_quality(mock_output)

            assert quality_result["quality_passed"], "Kvalita výstupu musí projít validací"
            assert quality_result["claims_count"] >= config["ci_gates"]["min_claims_required"], \
                "Počet claimů musí splňovat minimum"

            self.perf_logger.timing(
                operation="ci_gates_validation",
                duration_ms=smoke_duration * 1000
            )

            self.logger.info("CI gates validation test passed",
                           smoke_duration=smoke_duration,
                           quality_metrics=quality_result)

            return {
                "test": "ci_gates_validation",
                "status": "passed",
                "metrics": {
                    "smoke_test_duration": smoke_duration,
                    "quality_validation": quality_result
                }
            }

        except Exception as e:
            self.logger.error("CI gates validation test failed", error=str(e))
            pytest.fail(f"CI gates validation test failed: {e}")

    @pytest.mark.asyncio
    async def test_end_to_end_phase4_integration(self):
        """End-to-end test celé FÁZE 4"""
        self.logger.info("Running end-to-end Phase 4 integration test")

        start_time = time.time()

        try:
            # Spusť všechny komponenty
            eval_result = await self.test_comprehensive_evaluator()
            m1_result = await self.test_m1_performance_optimization()
            security_result = await self.test_security_compliance()
            ci_result = await self.test_ci_gates_validation()

            total_duration = time.time() - start_time

            # Agregace výsledků
            all_results = [eval_result, m1_result, security_result, ci_result]
            all_passed = all(result["status"] == "passed" for result in all_results)

            self.perf_logger.timing(
                operation="phase4_end_to_end",
                duration_ms=total_duration * 1000
            )

            self.test_results["tests"] = all_results
            self.test_results["end_time"] = datetime.now().isoformat()
            self.test_results["total_duration"] = total_duration
            self.test_results["all_passed"] = all_passed

            # Uložení výsledků
            results_path = Path("artifacts/phase4_test_result.json")
            results_path.parent.mkdir(exist_ok=True)

            with open(results_path, "w") as f:
                json.dump(self.test_results, f, indent=2)

            self.logger.info("Phase 4 end-to-end test completed",
                           all_passed=all_passed,
                           total_duration=total_duration,
                           results_saved=str(results_path))

            assert all_passed, "Všechny FÁZE 4 testy musí projít"

            return self.test_results

        except Exception as e:
            self.logger.error("Phase 4 end-to-end test failed", error=str(e))
            pytest.fail(f"Phase 4 end-to-end test failed: {e}")


# Legacy compatibility - pokud je spuštěno přímo
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
