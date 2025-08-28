"""
Pytest suite pro automatizovanou evaluaci Research Agent
Integrace do CI/CD pipeline s regression detection
"""

import pytest
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any

from src.evaluation.evaluation_pipeline import EvaluationPipeline, RAGEvaluator
from src.core.langgraph_agent import ResearchAgentGraph
from src.core.config_langgraph import load_config


class TestResearchAgentEvaluation:
    """Test suite pro evaluaci Research Agent pomocí Golden Dataset"""

    @pytest.fixture(scope="class")
    async def research_agent(self):
        """Inicializace Research Agent pro testování"""
        config = load_config()
        agent = ResearchAgentGraph(config)
        await agent.initialize()
        return agent

    @pytest.fixture(scope="class")
    def evaluation_pipeline(self, research_agent):
        """Inicializace evaluační pipeline"""
        # Mock LLM client pro evaluaci (v produkci by byl skutečný)
        class MockLLMClient:
            async def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 100):
                # Pro testy vrací simulované odpovědi
                if "relevant" in prompt.lower():
                    return "relevant relevant not_relevant"
                elif "score" in prompt.lower():
                    return "0.8"
                else:
                    return "2"

        llm_client = MockLLMClient()
        return EvaluationPipeline(research_agent, llm_client)

    @pytest.mark.asyncio
    async def test_golden_dataset_loads(self, evaluation_pipeline):
        """Test načtení Golden Dataset"""
        assert len(evaluation_pipeline.golden_dataset) > 0
        assert len(evaluation_pipeline.golden_dataset) >= 15  # Minimálně 15 otázek

        # Ověření struktury dat
        first_item = evaluation_pipeline.golden_dataset[0]
        assert hasattr(first_item, 'id')
        assert hasattr(first_item, 'query')
        assert hasattr(first_item, 'expected_answer')
        assert hasattr(first_item, 'relevant_sources')

    @pytest.mark.asyncio
    async def test_single_query_evaluation(self, evaluation_pipeline):
        """Test evaluace jedné otázky"""
        if not evaluation_pipeline.golden_dataset:
            pytest.skip("Golden dataset not available")

        test_item = evaluation_pipeline.golden_dataset[0]
        result = await evaluation_pipeline.evaluate_single_query(test_item)

        # Ověření struktury výsledku
        assert result.query_id == test_item.id
        assert result.query == test_item.query
        assert 0.0 <= result.context_precision <= 1.0
        assert 0.0 <= result.context_recall <= 1.0
        assert 0.0 <= result.faithfulness <= 1.0
        assert 0.0 <= result.answer_relevance <= 1.0
        assert 0.0 <= result.answer_correctness <= 1.0
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_full_evaluation_pipeline(self, evaluation_pipeline):
        """Test kompletní evaluace - pouze pro CI/CD"""
        if not os.getenv("CI"):
            pytest.skip("Full evaluation only in CI/CD")

        results = await evaluation_pipeline.run_full_evaluation()

        assert "results" in results
        assert "metrics" in results
        assert "results_path" in results

        metrics = results["metrics"]

        # Kontrola existence všech požadovaných metrik
        required_metrics = [
            "avg_context_precision",
            "avg_context_recall",
            "avg_faithfulness",
            "avg_answer_relevance",
            "avg_answer_correctness",
            "overall_score",
            "success_rate"
        ]

        for metric in required_metrics:
            assert metric in metrics
            assert 0.0 <= metrics[metric] <= 1.0

    @pytest.mark.asyncio
    async def test_regression_thresholds(self, evaluation_pipeline):
        """Test regression detection thresholds"""
        # Test s dobrými metrikami
        good_metrics = {
            "overall_score": 0.75,
            "avg_faithfulness": 0.80,
            "avg_answer_correctness": 0.70,
            "success_rate": 0.95
        }

        assert evaluation_pipeline.check_regression_thresholds(good_metrics) == True

        # Test s špatnými metrikami
        bad_metrics = {
            "overall_score": 0.60,  # Pod threshold 0.70
            "avg_faithfulness": 0.70,  # Pod threshold 0.75
            "avg_answer_correctness": 0.60,  # Pod threshold 0.65
            "success_rate": 0.85   # Pod threshold 0.90
        }

        assert evaluation_pipeline.check_regression_thresholds(bad_metrics) == False

    @pytest.mark.asyncio
    async def test_rag_evaluator_metrics(self):
        """Test jednotlivých RAG metrik"""
        class MockLLMClient:
            async def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 100):
                if "relevant" in prompt.lower():
                    return "relevant relevant not_relevant"
                elif "faithfulness" in prompt.lower():
                    return "0.8"
                elif "relevance" in prompt.lower():
                    return "0.9"
                elif "correctness" in prompt.lower():
                    return "0.7"
                else:
                    return "2"

        evaluator = RAGEvaluator(MockLLMClient())

        # Test context precision
        contexts = ["relevant context 1", "relevant context 2", "irrelevant context"]
        precision = await evaluator.evaluate_context_precision("test query", contexts)
        assert 0.0 <= precision <= 1.0

        # Test context recall
        expected_sources = ["source1", "source2"]
        recall = await evaluator.evaluate_context_recall("test query", contexts, expected_sources)
        assert 0.0 <= recall <= 1.0

        # Test faithfulness
        faithfulness = await evaluator.evaluate_faithfulness("test answer", contexts)
        assert 0.0 <= faithfulness <= 1.0

        # Test answer relevance
        relevance = await evaluator.evaluate_answer_relevance("test query", "test answer")
        assert 0.0 <= relevance <= 1.0

        # Test answer correctness
        correctness = await evaluator.evaluate_answer_correctness("expected", "generated")
        assert 0.0 <= correctness <= 1.0


@pytest.mark.integration
class TestCIIntegration:
    """Testy specifické pro CI/CD integraci"""

    def test_results_directory_exists(self):
        """Test existence adresáře pro výsledky"""
        results_dir = Path("evaluation")
        assert results_dir.exists() or results_dir.parent.exists()

    def test_environment_variables(self):
        """Test požadovaných environment variables pro CI"""
        if os.getenv("CI"):
            # V CI prostředí musí být nastaveny klíče
            assert os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
            assert os.getenv("LANGFUSE_SECRET_KEY")
            assert os.getenv("LANGFUSE_PUBLIC_KEY")

    @pytest.mark.asyncio
    async def test_ci_evaluation_with_thresholds(self):
        """Hlavní test pro CI/CD - spustí evaluaci a zkontroluje thresholds"""
        if not os.getenv("CI"):
            pytest.skip("CI integration test only in CI/CD environment")

        # Tento test se spustí v CI a může způsobit selhání buildu
        config = load_config()
        agent = ResearchAgentGraph(config)

        class ProductionLLMClient:
            async def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 100):
                # V produkci by zde byl skutečný LLM call
                # Pro demo vracíme simulované hodnoty
                import random
                if "score" in prompt.lower():
                    return str(random.uniform(0.7, 0.9))
                elif "relevant" in prompt.lower():
                    return "relevant " * random.randint(2, 4)
                else:
                    return str(random.randint(1, 3))

        pipeline = EvaluationPipeline(agent, ProductionLLMClient())

        # Spuštění evaluace na první 3 otázky (pro rychlost v CI)
        limited_dataset = pipeline.golden_dataset[:3]
        pipeline.golden_dataset = limited_dataset

        results = await pipeline.run_full_evaluation()
        metrics = results["metrics"]

        # Kritická kontrola - pokud selže, build musí selhat
        regression_check = pipeline.check_regression_thresholds(metrics)

        if not regression_check:
            pytest.fail(
                f"REGRESSION DETECTED! Metrics below thresholds: {metrics}"
            )

        # Uložení výsledků pro reporting
        with open("evaluation/ci_results.json", "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    # Pro lokální spuštění
    pytest.main([__file__, "-v"])
