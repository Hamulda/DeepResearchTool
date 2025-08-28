"""
Demo pro pokročilý Research Agent s human-in-the-loop funkcionalitou
Ukázka všech nových funkcí: validace zdrojů, rozšířené nástroje, Streamlit UI

Author: Senior Python/MLOps Agent
"""

import asyncio
import json
import time
from typing import Dict, Any
import logging
import os
from pathlib import Path

from src.core.langgraph_agent import ResearchAgentGraph, ResearchAgentState
from src.core.config_langgraph import load_config
from src.core.enhanced_tools import (
    semantic_scholar_search,
    data_gov_search,
    wayback_machine_search,
    cross_reference_sources
)
from src.observability.langfuse_integration import get_observability_manager
from src.evaluation.evaluation_pipeline import EvaluationPipeline, RAGEvaluator
from src.graph.expert_committee import ExpertCommitteeGraph, ExpertType

# Konfigurace loggingu
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedResearchDemo:
    """Demo třída pro ukázku pokročilých funkcí Research Agenta"""

    def __init__(self):
        """Inicializace demo"""
        self.config = self._load_demo_config()
        self.agent = None
        self.expert_committee = None
        self.evaluation_pipeline = None
        self.observability = get_observability_manager()

    def _load_demo_config(self) -> Dict[str, Any]:
        """Načte konfiguraci pro demo"""
        return {
            "llm": {
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "synthesis_model": "gpt-4o",
                "synthesis_temperature": 0.2
            },
            "memory_store": {
                "type": "chroma",
                "collection_name": "enhanced_demo_collection",
                "persist_directory": "./demo_chroma_db"
            },
            "tools": {
                "enabled": ["semantic_scholar", "web_scraper", "wayback_machine"],
                "scraper_config": {
                    "max_pages": 5,
                    "timeout": 30
                }
            },
            "evaluation": {
                "golden_dataset_path": "evaluation/golden_dataset.json",
                "metrics_enabled": True
            },
            "observability": {
                "langfuse_enabled": True,
                "trace_level": "detailed"
            }
        }

    async def initialize(self):
        """Inicializace všech komponent"""
        logger.info("🚀 Inicializace Enhanced Research Agent Demo...")

        # Inicializace hlavního agenta
        self.agent = ResearchAgentGraph(self.config)
        await self.agent.initialize()

        # Inicializace expert committee
        tools_registry = {
            "academic": ["semantic_scholar_search", "arxiv_search"],
            "web": ["firecrawl_scraper", "web_search"],
            "technical": ["github_search", "documentation_search"]
        }

        # Mock LLM client pro demo
        class DemoLLMClient:
            async def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 1000):
                # Simulace LLM odpovědi
                if "academic" in prompt.lower():
                    return "Akademické výsledky ukazují významné pokroky v dané oblasti..."
                elif "technical" in prompt.lower():
                    return "Z technického hlediska jsou klíčové následující aspekty..."
                elif "synthesis" in prompt.lower():
                    return "Kombinace expertních poznatků naznačuje..."
                else:
                    return "Relevantní informace z dostupných zdrojů..."

        llm_client = DemoLLMClient()
        self.expert_committee = ExpertCommitteeGraph(llm_client, tools_registry)

        # Inicializace evaluační pipeline
        self.evaluation_pipeline = EvaluationPipeline(self.agent, llm_client)

        logger.info("✅ Inicializace dokončena")

    async def demo_basic_research(self):
        """Demo základního výzkumu"""
        print("\n" + "="*60)
        print("🔍 DEMO: Základní výzkum s observability")
        print("="*60)

        query = "Jaké jsou nejnovější trendy v AI safety research?"

        if self.observability.is_enabled():
            tracer = self.observability.get_tracer()
            with tracer.trace_research_session(query, {"demo": "basic_research"}) as trace:
                result = await self.agent.run(query)

                # Zaznamenání metrik
                tracer.log_metrics({
                    "query_complexity": 0.7,
                    "sources_found": len(result.get("sources", [])),
                    "processing_time": 2.5
                }, trace_id=trace.id if trace else None)
        else:
            result = await self.agent.run(query)

        print(f"Dotaz: {query}")
        print(f"Odpověď: {result.get('answer', 'N/A')[:200]}...")
        print(f"Počet zdrojů: {len(result.get('sources', []))}")

        return result

    async def demo_expert_committee(self):
        """Demo multi-agentní expert committee"""
        print("\n" + "="*60)
        print("🤝 DEMO: Multi-Agent Expert Committee")
        print("="*60)

        complex_query = "Analyzuj dopad kvantového počítání na kryptografii a bezpečnost"

        print(f"Komplexní dotaz: {complex_query}")
        print("Zapojení expertů...")

        result = await self.expert_committee.run(complex_query)

        print(f"\nVýsledek od výboru expertů:")
        print(f"Odpověď: {result.get('answer', 'N/A')[:300]}...")
        print(f"Počet expertních odpovědí: {len(result.get('expert_responses', []))}")
        print(f"Celková confidence: {result.get('confidence_scores', {}).get('overall', 0):.2f}")
        print(f"Iterace: {result.get('iterations', 0)}")

        return result

    async def demo_evaluation_pipeline(self):
        """Demo evaluační pipeline"""
        print("\n" + "="*60)
        print("📊 DEMO: Automatizovaná evaluace")
        print("="*60)

        # Test na prvních 3 otázkách z Golden Dataset
        if not self.evaluation_pipeline.golden_dataset:
            print("❌ Golden Dataset není k dispozici")
            return None

        print(f"Golden Dataset obsahuje {len(self.evaluation_pipeline.golden_dataset)} otázek")
        print("Spouštím evaluaci na vzorku...")

        # Testování na prvních 3 otázkách pro demo
        sample_dataset = self.evaluation_pipeline.golden_dataset[:3]
        self.evaluation_pipeline.golden_dataset = sample_dataset

        results = await self.evaluation_pipeline.run_full_evaluation()
        metrics = results["metrics"]

        print("\n📈 Výsledky evaluace:")
        print(f"Overall Score: {metrics.get('overall_score', 0):.3f}")
        print(f"Context Precision: {metrics.get('avg_context_precision', 0):.3f}")
        print(f"Context Recall: {metrics.get('avg_context_recall', 0):.3f}")
        print(f"Faithfulness: {metrics.get('avg_faithfulness', 0):.3f}")
        print(f"Answer Relevance: {metrics.get('avg_answer_relevance', 0):.3f}")
        print(f"Answer Correctness: {metrics.get('avg_answer_correctness', 0):.3f}")
        print(f"Success Rate: {metrics.get('success_rate', 0):.3f}")

        # Kontrola regresi
        regression_check = self.evaluation_pipeline.check_regression_thresholds(metrics)
        print(f"\n🔍 Regression Check: {'✅ PASSED' if regression_check else '❌ FAILED'}")

        return results

    async def demo_observability_dashboard(self):
        """Demo observability funkcí"""
        print("\n" + "="*60)
        print("📊 DEMO: Observability & Monitoring")
        print("="*60)

        if not self.observability.is_enabled():
            print("❌ Langfuse observability není zapnutá")
            print("Pro aktivaci nastavte LANGFUSE_SECRET_KEY a LANGFUSE_PUBLIC_KEY")
            return

        print("✅ Langfuse observability je aktivní")
        print("🔗 Dashboard: http://localhost:3000")

        # Demo traced operace
        queries = [
            "Co je GPT-4?",
            "Vysvětli kvantovou supremacii",
            "Jaké jsou trendy v renewable energy?"
        ]

        for i, query in enumerate(queries):
            print(f"\n🔄 Trace {i+1}/3: {query}")

            tracer = self.observability.get_tracer()
            with tracer.trace_research_session(
                query,
                {"demo_batch": True, "query_index": i}
            ) as trace:
                start_time = time.time()
                result = await self.agent.run(query)
                duration = time.time() - start_time

                # Detailní metriky
                tracer.log_metrics({
                    "duration_seconds": duration,
                    "query_length": len(query),
                    "answer_length": len(result.get("answer", "")),
                    "sources_count": len(result.get("sources", [])),
                    "demo_run": True
                }, trace_id=trace.id if trace else None)

                print(f"  ✅ Completed in {duration:.2f}s")

        print("\n📊 Všechny traces jsou dostupné v Langfuse dashboard")

    async def demo_production_readiness(self):
        """Demo production readiness checklist"""
        print("\n" + "="*60)
        print("🚀 DEMO: Production Readiness Checklist")
        print("="*60)

        checklist = {
            "🔍 Observability": self.observability.is_enabled(),
            "📊 Evaluation Pipeline": len(self.evaluation_pipeline.golden_dataset) >= 15,
            "🤝 Multi-Agent Support": self.expert_committee is not None,
            "🔧 Configuration Management": self.config is not None,
            "📁 Golden Dataset": Path("evaluation/golden_dataset.json").exists(),
            "🐳 Docker Support": Path("docker-compose.observability.yml").exists(),
            "🔄 CI/CD Pipeline": Path(".github/workflows/ci-cd-pipeline.yml").exists(),
            "📋 Scaling Plan": Path("docs/production_scaling_plan.md").exists()
        }

        total_checks = len(checklist)
        passed_checks = sum(checklist.values())

        print("Production Readiness Status:")
        for item, status in checklist.items():
            print(f"  {item}: {'✅' if status else '❌'}")

        print(f"\n📊 Overall Score: {passed_checks}/{total_checks} ({passed_checks/total_checks*100:.1f}%)")

        if passed_checks >= total_checks * 0.8:
            print("🎉 READY FOR PRODUCTION!")
        else:
            print("⚠️  Needs more work before production deployment")

    async def run_complete_demo(self):
        """Spuštění kompletního demo"""
        print("🎯 ENHANCED RESEARCH AGENT - COMPLETE DEMO")
        print("="*80)

        await self.initialize()

        # Postupné demo všech funkcí
        await self.demo_basic_research()
        await self.demo_expert_committee()
        await self.demo_evaluation_pipeline()
        await self.demo_observability_dashboard()
        await self.demo_production_readiness()

        print("\n" + "="*80)
        print("🎊 DEMO DOKONČENO!")
        print("="*80)
        print("\nDalší kroky:")
        print("1. 🐳 Spusťte Langfuse: docker-compose -f docker-compose.observability.yml up")
        print("2. 🧪 Spusťte testy: pytest tests/test_evaluation_pipeline.py -v")
        print("3. 🚀 Nasaďte do produkce dle docs/production_scaling_plan.md")


async def main():
    """Hlavní funkce"""
    demo = EnhancedResearchDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())
