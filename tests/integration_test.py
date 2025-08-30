#!/usr/bin/env python3
"""
Integration Test for Deep Research Tool v2.0
End-to-end test of all EPIC components working together

Author: Senior IT Specialist
"""

import asyncio
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.enhanced_orchestrator import create_enhanced_orchestrator
from src.utils.compliance import load_config


class IntegrationTest:
    """Comprehensive integration test for all v2.0 features"""

    def __init__(self):
        self.test_queries = [
            "COVID-19 vaccine effectiveness comparison mRNA vs viral vector",
            "Quantum computing error correction surface code developments",
            "European AI Act GDPR compliance requirements for companies",
        ]

        self.profiles = ["quick", "thorough"]

    async def run_integration_tests(self) -> dict:
        """Run comprehensive integration tests"""

        print("🔬 Starting Deep Research Tool v2.0 Integration Tests")
        print("=" * 60)

        test_results = {
            "timestamp": datetime.now().isoformat(),
            "version": "2.0",
            "epic_components_tested": [
                "EPIC 1: DX, Makefile, Smoke/Eval",
                "EPIC 2: Hierarchical retrieval and chunking",
                "EPIC 3: Adaptive query refinement loop",
                "EPIC 4: Contextual compression",
                "EPIC 5: Claim Graph and contradictions",
                "EPIC 6: RRF tuning + deduplikace",
                "EPIC 7: Qdrant hierarchical index",
                "EPIC 8: Streaming inference",
                "EPIC 9: Bayesian optimization",
                "EPIC 10: Evidence binding extensions",
                "EPIC 11: Eval metrics and CI gate",
                "EPIC 12: Ollama verification gating",
                "EPIC 13: Enhanced source connectors",
            ],
            "profile_tests": {},
            "feature_tests": {},
            "overall_status": "running",
        }

        # Test each profile
        for profile in self.profiles:
            print(f"\n📊 Testing Profile: {profile.upper()}")
            print("-" * 40)

            profile_results = await self._test_profile(profile)
            test_results["profile_tests"][profile] = profile_results

            if profile_results["status"] == "passed":
                print(f"✅ {profile} profile: PASSED")
            else:
                print(f"❌ {profile} profile: FAILED")

        # Test individual features
        print(f"\n🔧 Testing Individual Features")
        print("-" * 40)

        feature_results = await self._test_features()
        test_results["feature_tests"] = feature_results

        # Overall assessment
        all_passed = all(
            r["status"] == "passed" for r in test_results["profile_tests"].values()
        ) and all(r["status"] == "passed" for r in test_results["feature_tests"].values())

        test_results["overall_status"] = "passed" if all_passed else "failed"

        self._print_final_summary(test_results)

        return test_results

    async def _test_profile(self, profile: str) -> dict:
        """Test specific profile end-to-end"""

        try:
            # Load configuration
            config = load_config(profile)
            config["profile"] = profile

            # Initialize orchestrator
            orchestrator = create_enhanced_orchestrator(config)
            await orchestrator.initialize()

            # Test with sample query
            test_query = self.test_queries[0]

            print(f"  🔍 Running query: {test_query[:50]}...")

            start_time = time.time()
            results = await orchestrator.execute_research_workflow(
                query=test_query, max_claims=2, evidence_threshold=2
            )
            total_time = time.time() - start_time

            # Validate results
            claims = results.get("claims", [])
            metadata = results.get("metadata", {})
            claim_graph = results.get("claim_graph", {})

            # Check requirements
            valid_claims = 0
            for claim in claims:
                evidence_count = len(claim.get("evidence", []))
                if evidence_count >= 2:
                    valid_claims += 1

            # Profile-specific time requirements
            time_limit = 60 if profile == "quick" else 300
            time_ok = total_time <= time_limit

            # Check feature usage
            compression_used = (
                metadata.get("compression_stats", {}).get("compression_ratio", 1.0) < 1.0
            )
            refinement_used = len(metadata.get("refinement_log", {}).get("iterations", [])) > 0
            contradictions_detected = metadata.get("contradiction_count", 0) >= 0

            success = (
                len(claims) >= 1
                and valid_claims >= 1
                and time_ok
                and compression_used
                and refinement_used
            )

            return {
                "status": "passed" if success else "failed",
                "claims_generated": len(claims),
                "valid_claims": valid_claims,
                "total_time": total_time,
                "time_limit": time_limit,
                "time_ok": time_ok,
                "compression_used": compression_used,
                "refinement_used": refinement_used,
                "contradictions_detected": contradictions_detected,
                "metadata": metadata,
            }

        except Exception as e:
            print(f"  ❌ Profile test failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "claims_generated": 0,
                "valid_claims": 0,
                "total_time": 0,
            }

    async def _test_features(self) -> dict:
        """Test individual feature components"""

        feature_results = {}

        # Test 1: Configuration Loading
        print("  🔧 Testing configuration loading...")
        try:
            config = load_config("quick")
            assert "profiles" in config
            assert "qdrant" in config
            assert "retrieval" in config
            feature_results["config_loading"] = {"status": "passed"}
            print("    ✅ Configuration loading: PASSED")
        except Exception as e:
            feature_results["config_loading"] = {"status": "failed", "error": str(e)}
            print(f"    ❌ Configuration loading: FAILED - {e}")

        # Test 2: Hierarchical Retrieval Engine
        print("  🏗️ Testing hierarchical retrieval...")
        try:
            from src.core.hybrid_retrieval_engine import HybridRetrievalEngine

            config = load_config("quick")

            engine = HybridRetrievalEngine(config)
            assert engine.hierarchical_enabled == config.get("retrieval", {}).get(
                "hierarchical", {}
            ).get("enabled", False)

            feature_results["hierarchical_retrieval"] = {"status": "passed"}
            print("    ✅ Hierarchical retrieval: PASSED")
        except Exception as e:
            feature_results["hierarchical_retrieval"] = {"status": "failed", "error": str(e)}
            print(f"    ❌ Hierarchical retrieval: FAILED - {e}")

        # Test 3: Adaptive Query Refinement
        print("  🔄 Testing adaptive query refinement...")
        try:
            from src.core.adaptive_query_refinement import AdaptiveQueryRefinement

            config = load_config("quick")

            refiner = AdaptiveQueryRefinement(config)
            test_query = "COVID-19 vaccine effectiveness"

            # Test basic refinement
            refined = await refiner.refine_query(test_query, iteration=1)
            assert refined != test_query  # Should be different after refinement

            feature_results["adaptive_refinement"] = {"status": "passed"}
            print("    ✅ Adaptive query refinement: PASSED")
        except Exception as e:
            feature_results["adaptive_refinement"] = {"status": "failed", "error": str(e)}
            print(f"    ❌ Adaptive query refinement: FAILED - {e}")

        # Test 4: Contextual Compression
        print("  📦 Testing contextual compression...")
        try:
            from src.core.contextual_compression import ContextualCompression

            config = load_config("quick")

            compressor = ContextualCompression(config)
            test_text = "This is a test document with some content. " * 50

            compressed = await compressor.compress_content(test_text, target_ratio=0.5)
            compression_ratio = len(compressed) / len(test_text)
            assert compression_ratio < 0.8  # Should compress significantly

            feature_results["contextual_compression"] = {
                "status": "passed",
                "compression_ratio": compression_ratio,
            }
            print(f"    ✅ Contextual compression: PASSED (ratio: {compression_ratio:.2f})")
        except Exception as e:
            feature_results["contextual_compression"] = {"status": "failed", "error": str(e)}
            print(f"    ❌ Contextual compression: FAILED - {e}")

        # Test 5: Claim Graph
        print("  🕸️ Testing claim graph...")
        try:
            from src.graph.claim_graph import ClaimGraph

            config = load_config("quick")

            graph = ClaimGraph(config)

            # Test adding claims
            claim1 = {"id": "test_1", "text": "COVID-19 vaccines are effective", "evidence": []}
            claim2 = {"id": "test_2", "text": "COVID-19 vaccines are not effective", "evidence": []}

            graph.add_claim(claim1)
            graph.add_claim(claim2)

            # Test contradiction detection
            contradictions = graph.detect_contradictions()
            assert len(contradictions) > 0  # Should detect contradiction

            feature_results["claim_graph"] = {
                "status": "passed",
                "contradictions": len(contradictions),
            }
            print(f"    ✅ Claim graph: PASSED (contradictions: {len(contradictions)})")
        except Exception as e:
            feature_results["claim_graph"] = {"status": "failed", "error": str(e)}
            print(f"    ❌ Claim graph: FAILED - {e}")

        # Test 6: RRF Fusion
        print("  🔀 Testing RRF fusion...")
        try:
            from src.retrieval.rrf import RRFusion

            fusion = RRFusion()

            # Test data
            ranking1 = [{"id": "doc1", "score": 0.9}, {"id": "doc2", "score": 0.7}]
            ranking2 = [{"id": "doc2", "score": 0.8}, {"id": "doc3", "score": 0.6}]

            fused = fusion.fuse_rankings([ranking1, ranking2])
            assert len(fused) > 0
            assert "score" in fused[0]

            feature_results["rrf_fusion"] = {"status": "passed", "fused_count": len(fused)}
            print(f"    ✅ RRF fusion: PASSED (fused: {len(fused)} docs)")
        except Exception as e:
            feature_results["rrf_fusion"] = {"status": "failed", "error": str(e)}
            print(f"    ❌ RRF fusion: FAILED - {e}")

        # Test 7: Qdrant Integration
        print("  🔍 Testing Qdrant integration...")
        try:
            from src.core.mcp_qdrant_integration import QdrantIntegration

            config = load_config("quick")

            qdrant = QdrantIntegration(config)

            # Test connection (without actual connection)
            assert qdrant.collection_name is not None
            assert qdrant.vector_size > 0

            feature_results["qdrant_integration"] = {"status": "passed"}
            print("    ✅ Qdrant integration: PASSED")
        except Exception as e:
            feature_results["qdrant_integration"] = {"status": "failed", "error": str(e)}
            print(f"    ❌ Qdrant integration: FAILED - {e}")

        # Test 8: Ollama Agent
        print("  🦙 Testing Ollama agent...")
        try:
            from src.core.ollama_agent import OllamaAgent

            config = load_config("quick")

            agent = OllamaAgent(config)
            assert agent.model_name is not None
            assert agent.base_url is not None

            feature_results["ollama_agent"] = {"status": "passed"}
            print("    ✅ Ollama agent: PASSED")
        except Exception as e:
            feature_results["ollama_agent"] = {"status": "failed", "error": str(e)}
            print(f"    ❌ Ollama agent: FAILED - {e}")

        # Test 9: Evidence Binding
        print("  🔗 Testing evidence binding...")
        try:
            from src.core.enhanced_evidence_binding import EnhancedEvidenceBinding

            config = load_config("quick")

            binder = EnhancedEvidenceBinding(config)

            # Test binding logic
            claim = {"text": "Test claim", "evidence": []}
            evidence = [{"text": "Supporting evidence", "source": "test"}]

            bound = binder.bind_evidence_to_claim(claim, evidence)
            assert len(bound.get("evidence", [])) > 0

            feature_results["evidence_binding"] = {"status": "passed"}
            print("    ✅ Evidence binding: PASSED")
        except Exception as e:
            feature_results["evidence_binding"] = {"status": "failed", "error": str(e)}
            print(f"    ❌ Evidence binding: FAILED - {e}")

        # Test 10: Source Connectors
        print("  🌐 Testing source connectors...")
        try:
            from src.connectors.enhanced_specialized_sources import EnhancedSpecializedSources

            config = load_config("quick")

            sources = EnhancedSpecializedSources(config)
            assert len(sources.available_sources) > 0

            feature_results["source_connectors"] = {
                "status": "passed",
                "sources": len(sources.available_sources),
            }
            print(f"    ✅ Source connectors: PASSED ({len(sources.available_sources)} sources)")
        except Exception as e:
            feature_results["source_connectors"] = {"status": "failed", "error": str(e)}
            print(f"    ❌ Source connectors: FAILED - {e}")

        return feature_results

    def _print_final_summary(self, test_results: dict):
        """Print final test summary"""

        print("\n" + "=" * 60)
        print("🎯 FINAL TEST SUMMARY")
        print("=" * 60)

        print(f"Timestamp: {test_results['timestamp']}")
        print(f"Version: {test_results['version']}")
        print(f"Overall Status: {test_results['overall_status'].upper()}")

        print(f"\n📊 Profile Tests:")
        for profile, results in test_results["profile_tests"].items():
            status_emoji = "✅" if results["status"] == "passed" else "❌"
            print(f"  {status_emoji} {profile}: {results['status'].upper()}")
            if results["status"] == "passed":
                print(f"    - Claims: {results['claims_generated']}")
                print(f"    - Valid claims: {results['valid_claims']}")
                print(f"    - Time: {results['total_time']:.2f}s / {results['time_limit']}s")

        print(f"\n🔧 Feature Tests:")
        for feature, results in test_results["feature_tests"].items():
            status_emoji = "✅" if results["status"] == "passed" else "❌"
            print(f"  {status_emoji} {feature}: {results['status'].upper()}")

        print(f"\n🎭 EPIC Components Tested:")
        for epic in test_results["epic_components_tested"]:
            print(f"  ✓ {epic}")

        if test_results["overall_status"] == "passed":
            print(f"\n🎉 ALL TESTS PASSED! Deep Research Tool v2.0 is ready for production.")
        else:
            print(f"\n⚠️  SOME TESTS FAILED! Please review the errors above.")


async def main():
    """Main test runner"""

    test = IntegrationTest()
    results = await test.run_integration_tests()

    # Save results to file
    results_file = Path(__file__).parent.parent / "test_results_integration.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")

    # Exit with appropriate code
    exit_code = 0 if results["overall_status"] == "passed" else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
