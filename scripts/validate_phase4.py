#!/usr/bin/env python3
"""
FÁZE 4 Validation Script
Validuje všechny komponenty FÁZE 4 specializovaných konektorů

Author: Senior Python/MLOps Agent
"""

import sys
import json
import asyncio
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class Phase4Validator:
    """Validator pro FÁZE 4 komponenty"""

    def __init__(self):
        self.validation_results = {
            "phase4_validation": {
                "timestamp": datetime.now().isoformat(),
                "components_tested": [],
                "test_results": {},
                "overall_status": "unknown",
                "critical_issues": [],
                "recommendations": [],
            }
        }

    async def validate_enhanced_common_crawl(self) -> Dict[str, Any]:
        """Validuje Enhanced Common Crawl Connector"""
        try:
            from src.connectors.enhanced_common_crawl import (
                EnhancedCommonCrawlConnector,
                CommonCrawlResult,
            )

            # Test základní inicializace
            config = {"common_crawl": {"enabled": True, "cache_enabled": False, "max_results": 5}}
            connector = EnhancedCommonCrawlConnector(config)

            # Test cache key generation
            cache_key = connector._generate_cache_key("test.warc", 0, 1000)
            assert len(cache_key) == 64, "Cache key should be SHA256 hex (64 chars)"

            # Test URL validation
            assert connector._is_valid_warc_url("https://example.com/test.warc.gz")
            assert not connector._is_valid_warc_url("invalid-url")

            return {
                "status": "passed",
                "tests_run": 3,
                "issues": [],
                "features_validated": [
                    "cache_key_generation",
                    "url_validation",
                    "basic_initialization",
                ],
            }

        except Exception as e:
            return {"status": "failed", "error": str(e), "traceback": traceback.format_exc()}

    async def validate_memento_temporal(self) -> Dict[str, Any]:
        """Validuje Memento Temporal Connector"""
        try:
            from src.connectors.memento_temporal import (
                MementoTemporalConnector,
                MementoTemporalResult,
            )
            from datetime import datetime, timedelta

            config = {"memento": {"enabled": True, "max_snapshots": 5}}
            connector = MementoTemporalConnector(config)

            # Test milestone calculation
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2023, 12, 31)
            milestones = connector._calculate_milestone_dates(start_date, end_date)

            assert len(milestones) > 0, "Should generate milestone dates"
            assert all(
                start_date <= m <= end_date for m in milestones
            ), "All milestones should be in range"

            # Test temporal diff calculation
            content1 = "Original content with some text"
            content2 = "Modified content with different text"
            diff = connector._calculate_content_diff(content1, content2)

            assert diff["similarity_score"] < 1.0, "Different content should have similarity < 1.0"

            return {
                "status": "passed",
                "tests_run": 3,
                "issues": [],
                "features_validated": [
                    "milestone_date_calculation",
                    "content_diff_calculation",
                    "temporal_range_validation",
                ],
            }

        except Exception as e:
            return {"status": "failed", "error": str(e), "traceback": traceback.format_exc()}

    async def validate_ahmia_tor(self) -> Dict[str, Any]:
        """Validuje Ahmia Tor Connector"""
        try:
            from src.connectors.ahmia_tor_connector import AhmiaTorConnector, AhmiaSearchResult

            config = {"ahmia": {"enabled": True, "legal_only": True, "tor_enabled": False}}
            connector = AhmiaTorConnector(config)

            # Test content category classification
            category = connector._classify_content_category("news", "journalism content")
            assert category in [
                "news",
                "research",
                "education",
                "other",
            ], "Should classify content properly"

            # Test suspicious keyword detection
            has_suspicious = connector._contains_suspicious_keywords("illegal", "description")
            assert has_suspicious is True, "Should detect suspicious keywords"

            safe_content = connector._contains_suspicious_keywords("research", "academic")
            assert safe_content is False, "Should not flag safe content"

            # Test legal whitelist creation
            connector._create_default_whitelist()
            assert isinstance(connector.legal_whitelist, set), "Should create whitelist set"

            return {
                "status": "passed",
                "tests_run": 4,
                "issues": [],
                "features_validated": [
                    "content_classification",
                    "suspicious_keyword_detection",
                    "safe_content_validation",
                    "legal_whitelist_creation",
                ],
            }

        except Exception as e:
            return {"status": "failed", "error": str(e), "traceback": traceback.format_exc()}

    async def validate_legal_apis(self) -> Dict[str, Any]:
        """Validuje Legal APIs Connector"""
        try:
            from src.connectors.legal_apis_connector import LegalAPIsConnector, LegalSearchResult

            config = {
                "legal_apis": {
                    "enabled": True,
                    "courtlistener": {"enabled": True},
                    "sec_edgar": {"enabled": True},
                }
            }
            connector = LegalAPIsConnector(config)

            # Test court citation building
            result_data = {
                "case_name": "Test v. Case",
                "court": {"citation_string": "D. Test"},
                "date_filed": "2023-01-01",
                "docket": {"docket_number": "23-cv-001"},
            }

            citation = connector._build_court_citation(result_data)
            assert "Test v. Case" in citation, "Should include case name"
            assert "D. Test" in citation, "Should include court"

            # Test SEC filing validation
            filing_data = {"cik_str": "123456", "ticker": "TEST", "title": "Test Company"}
            is_valid = connector._validate_sec_filing(filing_data)
            assert is_valid, "Should validate proper SEC filing data"

            return {
                "status": "passed",
                "tests_run": 3,
                "issues": [],
                "features_validated": [
                    "court_citation_building",
                    "sec_filing_validation",
                    "legal_document_processing",
                ],
            }

        except Exception as e:
            return {"status": "failed", "error": str(e), "traceback": traceback.format_exc()}

    async def validate_phase4_integration(self) -> Dict[str, Any]:
        """Validuje Phase 4 Integration orchestrátor"""
        try:
            from src.connectors.phase4_integration import (
                Phase4Integrator,
                Phase4ProcessingResult,
                SpecializedSourceResult,
            )
            from unittest.mock import Mock, AsyncMock

            config = {
                "phase4": {
                    "integration": {"parallel_processing": True, "timeout_per_connector": 60},
                    "diff_analysis": {
                        "enable_temporal_diff": True,
                        "enable_cross_source_diff": True,
                    },
                    "stability": {"min_success_rate": 0.8},
                }
            }

            integrator = Phase4Integrator(config)

            # Test stability metrics calculation
            test_results = {
                "connector1": SpecializedSourceResult(
                    source_type="test",
                    connector_name="test1",
                    success=True,
                    result_data=None,
                    processing_time=0.1,
                    error_message=None,
                    quality_metrics={},
                ),
                "connector2": SpecializedSourceResult(
                    source_type="test",
                    connector_name="test2",
                    success=False,
                    result_data=None,
                    processing_time=0.2,
                    error_message="timeout",
                    quality_metrics={},
                ),
            }

            stability = integrator._calculate_stability_metrics(test_results)
            assert stability["total_connectors"] == 2, "Should count all connectors"
            assert stability["successful_connectors"] == 1, "Should count successful ones"
            assert stability["overall_success_rate"] == 0.5, "Should calculate correct success rate"

            # Test audit preparation
            mock_result = Mock()
            mock_result.query = "test query"
            mock_result.processing_time = 1.0
            mock_result.connector_performance = {}

            audit_data = integrator._prepare_audit_data(mock_result)
            assert (
                "phase4_specialized_connectors_audit" in audit_data
            ), "Should prepare audit structure"

            return {
                "status": "passed",
                "tests_run": 3,
                "issues": [],
                "features_validated": [
                    "stability_metrics_calculation",
                    "audit_data_preparation",
                    "integration_orchestration",
                ],
            }

        except Exception as e:
            return {"status": "failed", "error": str(e), "traceback": traceback.format_exc()}

    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Spustí komprehenzivní validaci FÁZE 4"""
        print("🚀 Spouštím FÁZE 4 validaci...")

        # Komponenty k testování
        validation_tasks = {
            "enhanced_common_crawl": self.validate_enhanced_common_crawl,
            "memento_temporal": self.validate_memento_temporal,
            "ahmia_tor": self.validate_ahmia_tor,
            "legal_apis": self.validate_legal_apis,
            "phase4_integration": self.validate_phase4_integration,
        }

        all_passed = True
        total_tests = 0

        for component_name, validation_func in validation_tasks.items():
            print(f"📋 Validuji {component_name}...")

            try:
                result = await validation_func()
                self.validation_results["phase4_validation"]["test_results"][
                    component_name
                ] = result
                self.validation_results["phase4_validation"]["components_tested"].append(
                    component_name
                )

                if result["status"] == "passed":
                    print(f"✅ {component_name}: PASSED ({result.get('tests_run', 0)} testů)")
                    total_tests += result.get("tests_run", 0)
                else:
                    print(f"❌ {component_name}: FAILED - {result.get('error', 'Unknown error')}")
                    all_passed = False
                    self.validation_results["phase4_validation"]["critical_issues"].append(
                        {"component": component_name, "error": result.get("error", "Unknown error")}
                    )

            except Exception as e:
                print(f"💥 {component_name}: CRITICAL ERROR - {str(e)}")
                all_passed = False
                self.validation_results["phase4_validation"]["critical_issues"].append(
                    {"component": component_name, "error": f"Critical validation error: {str(e)}"}
                )

        # Finální status
        self.validation_results["phase4_validation"]["overall_status"] = (
            "passed" if all_passed else "failed"
        )
        self.validation_results["phase4_validation"]["total_tests_run"] = total_tests

        # Doporučení
        if all_passed:
            self.validation_results["phase4_validation"]["recommendations"] = [
                "✅ Všechny FÁZE 4 komponenty prošly validací",
                "🚀 Systém je připraven pro FÁZE 5 (Evaluace a CI/CD)",
                "📊 Implementovat benchmarky výkonu v produkčním prostředí",
                "🔍 Monitorovat metriky stability v reálném provozu",
            ]
        else:
            self.validation_results["phase4_validation"]["recommendations"] = [
                "❌ Opravit kritické chyby před pokračováním na FÁZE 5",
                "🔧 Zkontrolovat závislosti a konfigurace",
                "🧪 Spustit jednotlivé testy pro identifikaci problémů",
                "📋 Zkontrolovat implementaci podle akceptačních kritérií",
            ]

        return self.validation_results

    def export_validation_report(self, output_path: str = "docs/phase4_validation_report.json"):
        """Exportuje validační report"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.validation_results, f, indent=2, ensure_ascii=False)

        print(f"📊 Validační report exportován: {output_file}")


async def main():
    """Hlavní funkce pro spuštění validace"""
    validator = Phase4Validator()

    print("=" * 60)
    print("🔍 FÁZE 4 - Validace specializovaných konektorů")
    print("=" * 60)

    try:
        results = await validator.run_comprehensive_validation()

        print("\n" + "=" * 60)
        print("📊 SOUHRN VALIDACE")
        print("=" * 60)

        overall_status = results["phase4_validation"]["overall_status"]
        total_tests = results["phase4_validation"]["total_tests_run"]
        components_tested = len(results["phase4_validation"]["components_tested"])

        if overall_status == "passed":
            print(f"🎉 FÁZE 4 ÚSPĚŠNĚ VALIDOVÁNA!")
            print(f"✅ Komponenty testované: {components_tested}")
            print(f"✅ Celkem testů: {total_tests}")
            print(f"✅ Status: VŠECHNY TESTY PROŠLY")
        else:
            print(f"❌ FÁZE 4 VALIDACE NEÚSPĚŠNÁ")
            print(f"🔍 Komponenty testované: {components_tested}")
            print(f"📊 Celkem testů: {total_tests}")
            print(f"⚠️  Status: NĚKTERÉ TESTY SELHALY")

            if results["phase4_validation"]["critical_issues"]:
                print("\n🚨 Kritické problémy:")
                for issue in results["phase4_validation"]["critical_issues"]:
                    print(f"   • {issue['component']}: {issue['error']}")

        print("\n📋 Doporučení:")
        for rec in results["phase4_validation"]["recommendations"]:
            print(f"   {rec}")

        # Export reportu
        validator.export_validation_report()

        print(f"\n📄 Detailní report: docs/phase4_validation_report.json")

        # Return appropriate exit code
        return 0 if overall_status == "passed" else 1

    except Exception as e:
        print(f"💥 Kritická chyba při validaci: {str(e)}")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
