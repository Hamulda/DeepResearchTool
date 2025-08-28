#!/usr/bin/env python3
"""
FÁZE 2 Integration Test
Test HyDE, RRF, MMR, dedup, compression a Qdrant optimizace

Author: Senior Python/MLOps Agent
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.retrieval.hyde import HyDEGenerator, HyDERetrieval
from src.rank.rrf_mmr import RRFWithPriors, MMRDiversifier
from src.retrieval.qdrant_enhanced import EnhancedDeduplicator, QdrantVectorStore
from src.rank.rerank_compress import CrossEncoderReranker, AdaptiveContextualCompressor


class Phase2Tester:
    """Test suite pro FÁZE 2 komponenty"""

    def __init__(self):
        self.test_results = {
            "phase": 2,
            "start_time": datetime.now().isoformat(),
            "tests": []
        }
        self.config = self._get_test_config()

    def _get_test_config(self):
        """Test konfigurace pro FÁZI 2"""
        return {
            "retrieval": {
                "hyde": {
                    "enabled": True,
                    "budget_tokens": 512,
                    "model": "llama3.2:3b-instruct-q4_K_M"
                },
                "qdrant": {
                    "url": ":memory:",
                    "ef_search": {
                        "arxiv": 128,
                        "pubmed": 64,
                        "default": 32
                    }
                }
            },
            "ranking": {
                "rrf": {
                    "k": 60,
                    "priors": {
                        "authority": 0.3,
                        "recency": 0.2
                    }
                },
                "mmr": {
                    "lambda": 0.6,
                    "diversity_k": 10
                }
            },
            "compression": {
                "token_budget": 4096,
                "source_weights": {
                    "primary": 1.0,
                    "secondary": 0.7,
                    "aggregator": 0.3
                }
            }
        }

    async def test_hyde_generation(self):
        """Test HyDE generace hypotetických dokumentů"""
        print("🧪 Testing HyDE generation...")

        try:
            hyde_gen = HyDEGenerator(self.config["retrieval"]["hyde"])

            query = "What are the effects of climate change on Arctic ice?"
            hypothetical_doc = await hyde_gen.generate_hypothetical_document(query)

            assert len(hypothetical_doc) > 100, "HyDE dokument je příliš krátký"
            assert "climate" in hypothetical_doc.lower(), "HyDE neobsahuje klíčová slova"

            test_result = {
                "test": "hyde_generation",
                "status": "passed",
                "metrics": {
                    "doc_length": len(hypothetical_doc),
                    "contains_keywords": "climate" in hypothetical_doc.lower()
                },
                "timestamp": datetime.now().isoformat()
            }

            self.test_results["tests"].append(test_result)
            print("✅ HyDE generation test passed")

        except Exception as e:
            test_result = {
                "test": "hyde_generation",
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.test_results["tests"].append(test_result)
            print(f"❌ HyDE generation test failed: {e}")

    async def test_rrf_with_priors(self):
        """Test RRF s priors (authority, recency)"""
        print("🧪 Testing RRF with priors...")

        try:
            rrf = RRFWithPriors(self.config["ranking"]["rrf"])

            # Mock ranking lists s různými scores
            dense_results = [
                {"doc_id": "doc1", "score": 0.95, "metadata": {"authority": 0.8, "recency": 0.9}},
                {"doc_id": "doc2", "score": 0.85, "metadata": {"authority": 0.6, "recency": 0.7}},
                {"doc_id": "doc3", "score": 0.75, "metadata": {"authority": 0.9, "recency": 0.5}}
            ]

            bm25_results = [
                {"doc_id": "doc2", "score": 0.9, "metadata": {"authority": 0.6, "recency": 0.7}},
                {"doc_id": "doc1", "score": 0.8, "metadata": {"authority": 0.8, "recency": 0.9}},
                {"doc_id": "doc4", "score": 0.7, "metadata": {"authority": 0.5, "recency": 0.8}}
            ]

            fused_results = rrf.fuse_rankings([dense_results, bm25_results])

            assert len(fused_results) > 0, "RRF vrátil prázdné výsledky"
            assert "rrf_score" in fused_results[0], "RRF score chybí"

            test_result = {
                "test": "rrf_with_priors",
                "status": "passed",
                "metrics": {
                    "results_count": len(fused_results),
                    "top_doc": fused_results[0]["doc_id"],
                    "top_score": fused_results[0]["rrf_score"]
                },
                "timestamp": datetime.now().isoformat()
            }

            self.test_results["tests"].append(test_result)
            print("✅ RRF with priors test passed")

        except Exception as e:
            test_result = {
                "test": "rrf_with_priors",
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.test_results["tests"].append(test_result)
            print(f"❌ RRF with priors test failed: {e}")

    async def test_mmr_diversification(self):
        """Test MMR diversifikace"""
        print("🧪 Testing MMR diversification...")

        try:
            mmr = MMRDiversifier(self.config["ranking"]["mmr"])

            # Mock podobné dokumenty (by měly být diversifikovány)
            results = [
                {"doc_id": "doc1", "score": 0.95, "content": "Climate change affects Arctic ice significantly"},
                {"doc_id": "doc2", "score": 0.90, "content": "Arctic ice melting due to climate change"},
                {"doc_id": "doc3", "score": 0.85, "content": "Ocean acidification from CO2 emissions"},
                {"doc_id": "doc4", "score": 0.80, "content": "Renewable energy solutions for climate"}
            ]

            diversified = mmr.diversify_results(results, "climate change Arctic")

            assert len(diversified) > 0, "MMR vrátil prázdné výsledky"
            assert len(diversified) <= self.config["ranking"]["mmr"]["diversity_k"], "MMR překročil diversity_k"

            test_result = {
                "test": "mmr_diversification",
                "status": "passed",
                "metrics": {
                    "original_count": len(results),
                    "diversified_count": len(diversified),
                    "diversity_ratio": len(diversified) / len(results)
                },
                "timestamp": datetime.now().isoformat()
            }

            self.test_results["tests"].append(test_result)
            print("✅ MMR diversification test passed")

        except Exception as e:
            test_result = {
                "test": "mmr_diversification",
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.test_results["tests"].append(test_result)
            print(f"❌ MMR diversification test failed: {e}")

    async def test_enhanced_deduplication(self):
        """Test enhanced deduplikace"""
        print("🧪 Testing enhanced deduplication...")

        try:
            deduplicator = EnhancedDeduplicator()

            # Mock duplikované dokumenty
            documents = [
                {"doc_id": "doc1", "content": "Climate change is a major global issue"},
                {"doc_id": "doc2", "content": "Climate change is a major global issue"},  # exact duplicate
                {"doc_id": "doc3", "content": "Global warming is a significant worldwide problem"},  # near duplicate
                {"doc_id": "doc4", "content": "Renewable energy solutions are important"}  # different
            ]

            deduplicated, merge_map = deduplicator.deduplicate_documents(documents)

            assert len(deduplicated) < len(documents), "Deduplikace neodstranila duplikáty"
            assert len(merge_map) > 0, "Merge mapa je prázdná"

            test_result = {
                "test": "enhanced_deduplication",
                "status": "passed",
                "metrics": {
                    "original_count": len(documents),
                    "deduplicated_count": len(deduplicated),
                    "dedup_ratio": len(deduplicated) / len(documents),
                    "merges": len(merge_map)
                },
                "timestamp": datetime.now().isoformat()
            }

            self.test_results["tests"].append(test_result)
            print("✅ Enhanced deduplication test passed")

        except Exception as e:
            test_result = {
                "test": "enhanced_deduplication",
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.test_results["tests"].append(test_result)
            print(f"❌ Enhanced deduplication test failed: {e}")

    async def test_contextual_compression(self):
        """Test kontextová komprese"""
        print("🧪 Testing contextual compression...")

        try:
            compressor = AdaptiveContextualCompressor(self.config["compression"])

            # Mock dlouhý kontext
            contexts = [
                {
                    "doc_id": "doc1",
                    "content": "This is a very long document about climate change. " * 50,
                    "source_type": "primary",
                    "relevance_score": 0.9
                },
                {
                    "doc_id": "doc2",
                    "content": "Another document with useful information. " * 30,
                    "source_type": "secondary",
                    "relevance_score": 0.7
                }
            ]

            query = "climate change effects"
            compressed = compressor.compress_contexts(contexts, query)

            # Spočítej tokeny před a po
            original_tokens = sum(len(ctx["content"].split()) for ctx in contexts)
            compressed_tokens = sum(len(ctx["content"].split()) for ctx in compressed)

            assert compressed_tokens < original_tokens, "Komprese nesnížila počet tokenů"
            assert compressed_tokens <= self.config["compression"]["token_budget"], "Komprese překročila budget"

            efficiency = compressed_tokens / original_tokens

            test_result = {
                "test": "contextual_compression",
                "status": "passed",
                "metrics": {
                    "original_tokens": original_tokens,
                    "compressed_tokens": compressed_tokens,
                    "compression_ratio": 1 - efficiency,
                    "context_usage_efficiency": efficiency
                },
                "timestamp": datetime.now().isoformat()
            }

            self.test_results["tests"].append(test_result)
            print("✅ Contextual compression test passed")

        except Exception as e:
            test_result = {
                "test": "contextual_compression",
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.test_results["tests"].append(test_result)
            print(f"❌ Contextual compression test failed: {e}")

    async def run_all_tests(self):
        """Spustí všechny FÁZE 2 testy"""
        print("🚀 Starting FÁZE 2 Integration Tests...")

        tests = [
            self.test_hyde_generation(),
            self.test_rrf_with_priors(),
            self.test_mmr_diversification(),
            self.test_enhanced_deduplication(),
            self.test_contextual_compression()
        ]

        await asyncio.gather(*tests)

        # Vypočítej celkové metriky
        passed_tests = sum(1 for test in self.test_results["tests"] if test["status"] == "passed")
        total_tests = len(self.test_results["tests"])

        self.test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "end_time": datetime.now().isoformat()
        }

        # Ulož výsledky
        artifacts_dir = Path(parent_dir) / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        results_file = artifacts_dir / "phase2_test_result.json"
        with open(results_file, "w") as f:
            json.dump(self.test_results, f, indent=2)

        print(f"\n📊 FÁZE 2 Test Results:")
        print(f"   Passed: {passed_tests}/{total_tests}")
        print(f"   Success Rate: {self.test_results['summary']['success_rate']:.2%}")
        print(f"   Results saved to: {results_file}")

        return self.test_results


async def main():
    """Main test runner pro FÁZI 2"""
    tester = Phase2Tester()
    results = await tester.run_all_tests()

    # Fail hard pokud testy neprošly
    if results["summary"]["success_rate"] < 1.0:
        print("❌ FÁZE 2 testy neprošly - fail hard!")
        sys.exit(1)
    else:
        print("✅ FÁZE 2 testy úspěšně dokončeny!")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
