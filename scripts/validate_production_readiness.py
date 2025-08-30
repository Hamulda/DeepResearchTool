#!/usr/bin/env python3
"""
Validační skript pro kompletní systém Research Agent
Ověřuje připravenost na produkční nasazení
"""

import asyncio
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import subprocess

# Nastavení loggingu
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ProductionValidation:
    """Validační třída pro produkční připravenost"""

    def __init__(self):
        self.validation_results = {}
        self.total_checks = 0
        self.passed_checks = 0

    def check_environment_variables(self) -> bool:
        """Kontrola environment variables"""
        logger.info("🔍 Kontrola environment variables...")

        required_vars = ["OPENAI_API_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY"]

        optional_vars = ["ANTHROPIC_API_KEY", "DATABASE_URL", "REDIS_URL"]

        missing_required = []
        missing_optional = []

        for var in required_vars:
            if not os.getenv(var):
                missing_required.append(var)

        for var in optional_vars:
            if not os.getenv(var):
                missing_optional.append(var)

        if missing_required:
            logger.error(f"❌ Chybí povinné environment variables: {missing_required}")
            return False

        if missing_optional:
            logger.warning(f"⚠️ Chybí volitelné environment variables: {missing_optional}")

        logger.info("✅ Environment variables OK")
        return True

    def check_file_structure(self) -> bool:
        """Kontrola struktury souborů"""
        logger.info("🔍 Kontrola struktury souborů...")

        required_files = [
            "src/observability/langfuse_integration.py",
            "src/evaluation/evaluation_pipeline.py",
            "src/graph/expert_committee.py",
            "evaluation/golden_dataset.json",
            "docker-compose.observability.yml",
            ".github/workflows/ci-cd-pipeline.yml",
            "docs/production_scaling_plan.md",
            "tests/test_evaluation_pipeline.py",
        ]

        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)

        if missing_files:
            logger.error(f"❌ Chybí soubory: {missing_files}")
            return False

        logger.info("✅ Struktura souborů OK")
        return True

    def check_dependencies(self) -> bool:
        """Kontrola závislostí"""
        logger.info("🔍 Kontrola závislostí...")

        try:
            import langfuse
            import pytest
            import docker

            logger.info("✅ Hlavní závislosti OK")
            return True
        except ImportError as e:
            logger.error(f"❌ Chybí závislost: {e}")
            return False

    def check_golden_dataset(self) -> bool:
        """Kontrola Golden Dataset"""
        logger.info("🔍 Kontrola Golden Dataset...")

        dataset_path = Path("evaluation/golden_dataset.json")
        if not dataset_path.exists():
            logger.error("❌ Golden Dataset soubor neexistuje")
            return False

        try:
            with open(dataset_path, "r") as f:
                dataset = json.load(f)

            if len(dataset) < 15:
                logger.error(f"❌ Golden Dataset má pouze {len(dataset)} otázek, minimum je 15")
                return False

            # Kontrola struktury první otázky
            if dataset:
                required_fields = ["id", "query", "expected_answer", "relevant_sources", "category"]
                first_item = dataset[0]
                missing_fields = [field for field in required_fields if field not in first_item]

                if missing_fields:
                    logger.error(f"❌ Golden Dataset má chybějící pole: {missing_fields}")
                    return False

            logger.info(f"✅ Golden Dataset OK ({len(dataset)} otázek)")
            return True

        except json.JSONDecodeError:
            logger.error("❌ Golden Dataset není platný JSON")
            return False

    async def check_observability_integration(self) -> bool:
        """Kontrola Langfuse integrace"""
        logger.info("🔍 Kontrola observability integrace...")

        try:
            from src.observability.langfuse_integration import get_observability_manager

            manager = get_observability_manager()
            if manager.is_enabled():
                logger.info("✅ Langfuse observability je zapnutá")
                return True
            else:
                logger.warning("⚠️ Langfuse observability není zapnutá (možná chybí API klíče)")
                return True  # Ne-fatální pro validaci

        except ImportError as e:
            logger.error(f"❌ Problém s observability integrací: {e}")
            return False

    async def check_evaluation_pipeline(self) -> bool:
        """Kontrola evaluační pipeline"""
        logger.info("🔍 Kontrola evaluační pipeline...")

        try:
            from src.evaluation.evaluation_pipeline import EvaluationPipeline, RAGEvaluator

            # Mock LLM client pro test
            class MockLLMClient:
                async def generate(
                    self, prompt: str, temperature: float = 0.1, max_tokens: int = 100
                ):
                    return "0.8"  # Mock odpověď

            evaluator = RAGEvaluator(MockLLMClient())

            # Test základních metrik
            precision = await evaluator.evaluate_context_precision(
                "test query", ["context1", "context2"]
            )
            if not (0.0 <= precision <= 1.0):
                logger.error("❌ Context precision mimo rozsah")
                return False

            logger.info("✅ Evaluační pipeline OK")
            return True

        except Exception as e:
            logger.error(f"❌ Problém s evaluační pipeline: {e}")
            return False

    async def check_expert_committee(self) -> bool:
        """Kontrola multi-agentní architektury"""
        logger.info("🔍 Kontrola expert committee...")

        try:
            from src.graph.expert_committee import ExpertCommitteeGraph, ExpertType

            # Mock LLM a tools pro test
            class MockLLMClient:
                async def generate(
                    self, prompt: str, temperature: float = 0.1, max_tokens: int = 1000
                ):
                    return '{"relevance_score": 0.8, "confidence": 0.7}'

            tools_registry = {"academic": [], "web": [], "technical": []}
            committee = ExpertCommitteeGraph(MockLLMClient(), tools_registry)

            if committee.experts and committee.coordinator:
                logger.info("✅ Expert committee OK")
                return True
            else:
                logger.error("❌ Expert committee není správně inicializován")
                return False

        except Exception as e:
            logger.error(f"❌ Problém s expert committee: {e}")
            return False

    def check_docker_setup(self) -> bool:
        """Kontrola Docker konfigurace"""
        logger.info("🔍 Kontrola Docker setup...")

        docker_files = ["docker-compose.observability.yml", "Dockerfile", "Dockerfile.production"]

        missing_files = [f for f in docker_files if not Path(f).exists()]

        if missing_files:
            logger.error(f"❌ Chybí Docker soubory: {missing_files}")
            return False

        # Kontrola Docker daemon
        try:
            result = subprocess.run(
                ["docker", "--version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                logger.info("✅ Docker je dostupný")
            else:
                logger.warning("⚠️ Docker daemon neběží")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("⚠️ Docker není nainstalován nebo není dostupný")

        logger.info("✅ Docker konfigurace OK")
        return True

    def check_ci_cd_pipeline(self) -> bool:
        """Kontrola CI/CD pipeline"""
        logger.info("🔍 Kontrola CI/CD pipeline...")

        github_workflow = Path(".github/workflows/ci-cd-pipeline.yml")
        if not github_workflow.exists():
            logger.error("❌ GitHub Actions workflow neexistuje")
            return False

        # Kontrola test souborů
        test_files = ["tests/test_evaluation_pipeline.py", "tests/conftest.py"]

        missing_tests = [f for f in test_files if not Path(f).exists()]
        if missing_tests:
            logger.warning(f"⚠️ Chybí test soubory: {missing_tests}")

        logger.info("✅ CI/CD pipeline OK")
        return True

    async def run_validation(self) -> Dict[str, Any]:
        """Spuštění kompletní validace"""
        logger.info("🚀 Spouštím produkční validaci...")

        checks = [
            ("Environment Variables", self.check_environment_variables),
            ("File Structure", self.check_file_structure),
            ("Dependencies", self.check_dependencies),
            ("Golden Dataset", self.check_golden_dataset),
            ("Observability Integration", self.check_observability_integration),
            ("Evaluation Pipeline", self.check_evaluation_pipeline),
            ("Expert Committee", self.check_expert_committee),
            ("Docker Setup", self.check_docker_setup),
            ("CI/CD Pipeline", self.check_ci_cd_pipeline),
        ]

        results = {}

        for check_name, check_func in checks:
            self.total_checks += 1
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()

                results[check_name] = result
                if result:
                    self.passed_checks += 1

            except Exception as e:
                logger.error(f"❌ Chyba v {check_name}: {e}")
                results[check_name] = False

        self.validation_results = results
        return results

    def generate_report(self) -> str:
        """Generování validačního reportu"""
        success_rate = (self.passed_checks / self.total_checks) * 100

        report = f"""
{'='*80}
🔍 PRODUKČNÍ VALIDAČNÍ REPORT
{'='*80}

Celkový výsledek: {self.passed_checks}/{self.total_checks} ({success_rate:.1f}%)

Detailní výsledky:
"""

        for check_name, result in self.validation_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            report += f"  {check_name:<25}: {status}\n"

        report += f"\n{'='*80}\n"

        if success_rate >= 90:
            report += "🎉 SYSTÉM JE PŘIPRAVEN NA PRODUKCI!\n"
        elif success_rate >= 70:
            report += "⚠️  Systém potřebuje drobné úpravy před produkčním nasazením\n"
        else:
            report += "❌ Systém NENÍ připraven na produkci - vyžaduje významné úpravy\n"

        report += f"{'='*80}\n"

        return report


async def main():
    """Hlavní funkce validace"""
    print("🎯 RESEARCH AGENT - PRODUCTION VALIDATION")
    print("=" * 80)

    validator = ProductionValidation()
    await validator.run_validation()

    report = validator.generate_report()
    print(report)

    # Uložení reportu
    report_path = Path("validation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"📄 Report uložen do {report_path}")

    # Exit code pro CI/CD
    success_rate = (validator.passed_checks / validator.total_checks) * 100
    if success_rate < 70:
        sys.exit(1)  # Selhání pro CI/CD
    else:
        sys.exit(0)  # Úspěch


if __name__ == "__main__":
    asyncio.run(main())
