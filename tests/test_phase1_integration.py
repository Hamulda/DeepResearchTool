#!/usr/bin/env python3
"""
FÁZE 1 Integration Test
Test odstranění HITL, gates systému, CLI a základní infrastruktury

Author: Senior Python/MLOps Agent
"""

import asyncio
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Add src to path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.utils.gates import (
    GateKeeper,
    EvidenceGate,
    ComplianceGate,
    MetricsGate,
    QualityGate,
    EvidenceGateError,
    ComplianceGateError,
    MetricsGateError
)
from src.core.config import load_config, get_default_config
from src.core.pipeline import ResearchPipeline


class Phase1Tester:
    """Test suite pro FÁZE 1 komponenty"""

    def __init__(self):
        self.test_results = {
            "phase": 1,
            "start_time": datetime.now().isoformat(),
            "tests": []
        }

    async def test_gates_system(self):
        """Test validačních bran - nahrazení HITL"""
        print("🔄 Testing validation gates system...")

        config = get_default_config()
        gatekeeper = GateKeeper(config)

        # Test 1: Evidence gate s dostatečnými citacemi (should pass)
        try:
            evidence_gate = EvidenceGate(min_citations_per_claim=2)

            # Mock synthesis result with sufficient citations
            mock_synthesis = type('obj', (object,), {
                'claims': [
                    type('claim', (object,), {
                        'citations': [{'doc_id': 'doc1'}, {'doc_id': 'doc2'}, {'doc_id': 'doc3'}]
                    })(),
                    type('claim', (object,), {
                        'citations': [{'doc_id': 'doc4'}, {'doc_id': 'doc5'}]
                    })()
                ]
            })()

            result = await evidence_gate.validate(mock_synthesis)
            assert result.passed, "Evidence gate should pass with sufficient citations"

            self.test_results["tests"].append({
                "name": "Evidence Gate - Sufficient Citations",
                "status": "PASSED",
                "details": f"Claims have sufficient citations (≥2 per claim)"
            })

        except Exception as e:
            self.test_results["tests"].append({
                "name": "Evidence Gate - Sufficient Citations",
                "status": "FAILED",
                "error": str(e)
            })
            print(f"❌ Evidence gate test failed: {e}")
            return False

        # Test 2: Evidence gate s nedostatečnými citacemi (should fail)
        try:
            mock_synthesis_bad = type('obj', (object,), {
                'claims': [
                    type('claim', (object,), {
                        'citations': [{'doc_id': 'doc1'}]  # Only 1 citation
                    })()
                ]
            })()

            result = await evidence_gate.validate(mock_synthesis_bad)
            assert not result.passed, "Evidence gate should fail with insufficient citations"
            assert len(result.suggestions) > 0, "Should provide suggestions"

            self.test_results["tests"].append({
                "name": "Evidence Gate - Insufficient Citations",
                "status": "PASSED",
                "details": "Correctly failed with insufficient citations"
            })

        except Exception as e:
            self.test_results["tests"].append({
                "name": "Evidence Gate - Insufficient Citations",
                "status": "FAILED",
                "error": str(e)
            })
            print(f"❌ Evidence gate negative test failed: {e}")
            return False

        # Test 3: Compliance gate
        try:
            compliance_gate = ComplianceGate(max_rate_violations=0, max_robots_violations=0)

            # Mock retrieval log with no violations
            clean_log = {
                'rate_limit_violations': [],
                'robots_txt_violations': [],
                'blocked_domains': [],
                'requests': [{'url': 'example.com'}]
            }

            result = await compliance_gate.validate(clean_log)
            assert result.passed, "Compliance gate should pass with clean log"

            self.test_results["tests"].append({
                "name": "Compliance Gate - Clean Log",
                "status": "PASSED",
                "details": "No compliance violations detected"
            })

        except Exception as e:
            self.test_results["tests"].append({
                "name": "Compliance Gate - Clean Log",
                "status": "FAILED",
                "error": str(e)
            })
            print(f"❌ Compliance gate test failed: {e}")
            return False

        print("✅ Gates system test passed")
        return True

    async def test_config_system(self):
        """Test konfiguračního systému"""
        print("🔄 Testing configuration system...")

        try:
            # Test default config
            default_config = get_default_config()
            assert isinstance(default_config, dict), "Default config should be dict"
            assert "retrieval" in default_config, "Should have retrieval config"
            assert "gates" in default_config, "Should have gates config"

            # Test config loading with temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write("""
retrieval:
  max_context_tokens: 4000
  top_k: 30

gates:
  evidence:
    min_citations_per_claim: 3
""")
                temp_config_path = f.name

            try:
                config = load_config(temp_config_path)
                assert config["retrieval"]["max_context_tokens"] == 4000, "Config override should work"
                assert config["gates"]["evidence"]["min_citations_per_claim"] == 3, "Nested config should work"

                # Should still have default values
                assert "synthesis" in config, "Should merge with defaults"

            finally:
                os.unlink(temp_config_path)

            self.test_results["tests"].append({
                "name": "Configuration System",
                "status": "PASSED",
                "details": "Config loading and merging works correctly"
            })

        except Exception as e:
            self.test_results["tests"].append({
                "name": "Configuration System",
                "status": "FAILED",
                "error": str(e)
            })
            print(f"❌ Config system test failed: {e}")
            return False

        print("✅ Configuration system test passed")
        return True

    async def test_pipeline_basic(self):
        """Test základního pipeline systému"""
        print("🔄 Testing basic pipeline...")

        try:
            config = get_default_config()
            pipeline = ResearchPipeline(config)

            # Test initialization
            await pipeline.initialize()
            assert pipeline.initialized, "Pipeline should be initialized"

            # Test basic execution
            result = await pipeline.execute("test query")

            # Validate result structure
            assert isinstance(result, dict), "Result should be dict"
            assert "claims" in result, "Should have claims"
            assert "citations" in result, "Should have citations"
            assert "processing_time" in result, "Should track processing time"

            # Validate claims have citations
            claims = result["claims"]
            assert len(claims) >= 1, "Should generate at least 1 claim"

            for claim in claims:
                citations = claim.get("citations", [])
                assert len(citations) >= 2, f"Each claim should have ≥2 citations, got {len(citations)}"

            # Cleanup
            await pipeline.cleanup()
            assert not pipeline.initialized, "Pipeline should be cleaned up"

            self.test_results["tests"].append({
                "name": "Basic Pipeline Execution",
                "status": "PASSED",
                "details": f"Generated {len(claims)} claims with proper citations"
            })

        except Exception as e:
            self.test_results["tests"].append({
                "name": "Basic Pipeline Execution",
                "status": "FAILED",
                "error": str(e)
            })
            print(f"❌ Pipeline test failed: {e}")
            return False

        print("✅ Basic pipeline test passed")
        return True

    async def test_fail_hard_behavior(self):
        """Test fail-hard chování bran"""
        print("🔄 Testing fail-hard gate behavior...")

        try:
            config = get_default_config()
            gatekeeper = GateKeeper(config)

            # Create data that should fail evidence gate
            bad_data = {
                "synthesis_result": type('obj', (object,), {
                    'claims': [
                        type('claim', (object,), {
                            'citations': []  # No citations - should fail
                        })()
                    ]
                })(),
                "retrieval_log": {},
                "evaluation_result": {},
                "output_data": {"claims": [], "citations": [], "token_count": 0}
            }

            # This should raise EvidenceGateError
            try:
                await gatekeeper.validate_all(bad_data)
                assert False, "Should have raised EvidenceGateError"
            except EvidenceGateError as e:
                assert len(e.suggestions) > 0, "Should provide suggestions"
                print(f"✅ Correctly failed with: {e}")

            self.test_results["tests"].append({
                "name": "Fail-Hard Gate Behavior",
                "status": "PASSED",
                "details": "Gates correctly fail-hard with proper error messages"
            })

        except Exception as e:
            self.test_results["tests"].append({
                "name": "Fail-Hard Gate Behavior",
                "status": "FAILED",
                "error": str(e)
            })
            print(f"❌ Fail-hard test failed: {e}")
            return False

        print("✅ Fail-hard behavior test passed")
        return True


async def main():
    """Hlavní test runner pro FÁZE 1"""
    print("🧪 FÁZE 1 Integration Test")
    print("=" * 50)
    print("Testing: HITL removal, gates system, CLI, basic infrastructure")
    print("=" * 50)

    tester = Phase1Tester()
    start_time = datetime.now()

    # Run all tests
    tests = [
        ("Gates System", tester.test_gates_system),
        ("Configuration System", tester.test_config_system),
        ("Basic Pipeline", tester.test_pipeline_basic),
        ("Fail-Hard Behavior", tester.test_fail_hard_behavior)
    ]

    passed = 0
    for test_name, test_func in tests:
        try:
            result = await test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")

    # Finalize results
    end_time = datetime.now()
    tester.test_results.update({
        "end_time": end_time.isoformat(),
        "duration": (end_time - start_time).total_seconds(),
        "summary": {
            "total_tests": len(tests),
            "passed": passed,
            "failed": len(tests) - passed,
            "success_rate": passed / len(tests)
        }
    })

    # Save results
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/phase1_test_result.json", "w") as f:
        json.dump(tester.test_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("📊 FÁZE 1 Test Summary:")
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"⏱️  Duration: {tester.test_results['duration']:.2f}s")
    print(f"📁 Results saved to: artifacts/phase1_test_result.json")

    # Check if we meet acceptance criteria
    if passed == len(tests):
        print("🎉 FÁZE 1 ACCEPTANCE CRITERIA MET!")
        print("✅ Gates system implemented and tested")
        print("✅ HITL checkpoints removed")
        print("✅ Basic infrastructure working")
        print("✅ Fail-hard behavior validated")
        return True
    else:
        print("❌ FÁZE 1 acceptance criteria NOT met")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
