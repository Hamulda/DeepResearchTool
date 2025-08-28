"""
Validační skript pro Fázi 1 - Základní Architektura a Bezpečnost.
Ověřuje správnou implementaci všech komponent ELT pipeline, RAG systému a bezpečnosti.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
import tempfile
import shutil
from datetime import datetime
import subprocess
import os

import structlog
import docker
import pytest

# Konfigurace logování
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class Phase1Validator:
    """Hlavní validátor pro Fázi 1."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "phase": "1",
            "description": "Základní Architektura a Bezpečnost",
            "tests": {},
            "overall_status": "pending"
        }

    async def run_all_validations(self) -> Dict[str, Any]:
        """Spustí všechny validace pro Fázi 1."""
        logger.info("Starting Phase 1 validation suite")

        validations = [
            ("project_structure", self._validate_project_structure),
            ("dependencies", self._validate_dependencies),
            ("docker_setup", self._validate_docker_setup),
            ("security_config", self._validate_security_config),
            ("elt_pipeline", self._validate_elt_pipeline),
            ("rag_system", self._validate_rag_system),
            ("local_llm", self._validate_local_llm),
            ("autonomous_server", self._validate_autonomous_server),
            ("integration", self._validate_integration)
        ]

        for test_name, test_func in validations:
            try:
                logger.info(f"Running validation: {test_name}")
                result = await test_func()
                self.test_results["tests"][test_name] = result

                if result["status"] == "failed":
                    logger.error(f"Validation failed: {test_name}")
                else:
                    logger.info(f"Validation passed: {test_name}")

            except Exception as e:
                logger.error(f"Validation error: {test_name}", error=str(e))
                self.test_results["tests"][test_name] = {
                    "status": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                }

        # Vyhodnocení celkového stavu
        self._calculate_overall_status()

        return self.test_results

    async def _validate_project_structure(self) -> Dict[str, Any]:
        """Validuje strukturu projektu."""
        required_files = [
            "docker-compose.autonomous.yml",
            "Dockerfile.autonomous",
            ".env.template",
            "requirements.txt",
            "src/core/elt_pipeline.py",
            "src/core/rag_system.py",
            "src/core/local_llm.py",
            "src/core/autonomous_server.py",
            "monitoring/prometheus.yml",
            "configs/tor_legal_whitelist.json"
        ]

        required_dirs = [
            "src/core",
            "monitoring",
            "configs",
            "data",
            "models"
        ]

        missing_files = []
        missing_dirs = []

        # Kontrola souborů
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)

        # Kontrola adresářů
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                missing_dirs.append(dir_path)

        status = "passed" if not missing_files and not missing_dirs else "failed"

        return {
            "status": status,
            "missing_files": missing_files,
            "missing_directories": missing_dirs,
            "message": "Project structure validation",
            "timestamp": datetime.now().isoformat()
        }

    async def _validate_dependencies(self) -> Dict[str, Any]:
        """Validuje závislosti v requirements.txt."""
        requirements_file = self.project_root / "requirements.txt"

        if not requirements_file.exists():
            return {
                "status": "failed",
                "message": "requirements.txt not found",
                "timestamp": datetime.now().isoformat()
            }

        required_packages = [
            "pyarrow",
            "duckdb",
            "polars",
            "milvus-lite",
            "sentence-transformers",
            "llama-cpp-python",
            "structlog",
            "fastapi",
            "uvicorn",
            "prometheus-client",
            "docker",
            "cryptography",
            "ggshield"
        ]

        content = requirements_file.read_text()
        missing_packages = []

        for package in required_packages:
            if package not in content:
                missing_packages.append(package)

        status = "passed" if not missing_packages else "failed"

        return {
            "status": status,
            "missing_packages": missing_packages,
            "message": "Dependencies validation",
            "timestamp": datetime.now().isoformat()
        }

    async def _validate_docker_setup(self) -> Dict[str, Any]:
        """Validuje Docker konfiguraci."""
        try:
            # Kontrola Docker Compose souboru
            compose_file = self.project_root / "docker-compose.autonomous.yml"
            if not compose_file.exists():
                return {
                    "status": "failed",
                    "message": "docker-compose.autonomous.yml not found",
                    "timestamp": datetime.now().isoformat()
                }

            # Kontrola Docker client
            client = docker.from_env()
            client.ping()

            # Validace compose souboru
            result = subprocess.run(
                ["docker-compose", "-f", str(compose_file), "config"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )

            if result.returncode != 0:
                return {
                    "status": "failed",
                    "message": f"Docker Compose validation failed: {result.stderr}",
                    "timestamp": datetime.now().isoformat()
                }

            return {
                "status": "passed",
                "message": "Docker setup validation successful",
                "docker_version": client.version()["Version"],
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "status": "failed",
                "message": f"Docker validation error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def _validate_security_config(self) -> Dict[str, Any]:
        """Validuje bezpečnostní konfiguraci."""
        checks = []

        # Kontrola .gitignore
        gitignore_file = self.project_root / ".gitignore"
        if gitignore_file.exists():
            content = gitignore_file.read_text()
            has_env = ".env" in content
            has_secrets = "*.key" in content
            checks.append(("gitignore_env", has_env))
            checks.append(("gitignore_secrets", has_secrets))
        else:
            checks.append(("gitignore_exists", False))

        # Kontrola .env template
        env_template = self.project_root / ".env.template"
        checks.append(("env_template_exists", env_template.exists()))

        # Kontrola Tor whitelist
        tor_whitelist = self.project_root / "configs" / "tor_legal_whitelist.json"
        if tor_whitelist.exists():
            try:
                whitelist_data = json.loads(tor_whitelist.read_text())
                has_domains = "allowed_domains" in whitelist_data
                has_ethics = "ethical_guidelines" in whitelist_data
                checks.append(("tor_whitelist_valid", has_domains and has_ethics))
            except json.JSONDecodeError:
                checks.append(("tor_whitelist_valid", False))
        else:
            checks.append(("tor_whitelist_exists", False))

        # Vyhodnocení
        failed_checks = [name for name, passed in checks if not passed]
        status = "passed" if not failed_checks else "failed"

        return {
            "status": status,
            "failed_checks": failed_checks,
            "all_checks": dict(checks),
            "message": "Security configuration validation",
            "timestamp": datetime.now().isoformat()
        }

    async def _validate_elt_pipeline(self) -> Dict[str, Any]:
        """Validuje ELT pipeline implementaci."""
        try:
            # Import test
            sys.path.insert(0, str(self.project_root))
            from src.core.elt_pipeline import ELTPipeline, ParquetStreamWriter, DuckDBProcessor

            # Základní test s mock daty
            with tempfile.TemporaryDirectory() as temp_dir:
                pipeline = ELTPipeline(Path(temp_dir), chunk_size=10)

                # Mock data stream
                async def mock_stream():
                    for i in range(5):
                        yield {"id": i, "content": f"test content {i}"}

                # Test extract & load
                await pipeline.extract_and_load(
                    mock_stream(), "test_table", "test_source"
                )

                # Test analyze
                stats = await pipeline.transform_and_analyze("test_table")

                pipeline.cleanup()

                has_stats = "row_count" in stats

            return {
                "status": "passed" if has_stats else "failed",
                "message": "ELT pipeline validation successful",
                "test_stats": stats if has_stats else None,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "status": "failed",
                "message": f"ELT pipeline validation failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def _validate_rag_system(self) -> Dict[str, Any]:
        """Validuje RAG systém implementaci."""
        try:
            # Import test (bez inicializace Milvus - vyžaduje běžící server)
            sys.path.insert(0, str(self.project_root))
            from src.core.rag_system import EmbeddingGenerator, DocumentChunk

            # Test embedding generator
            embedding_gen = EmbeddingGenerator(device="cpu")  # Force CPU pro test

            test_texts = ["Test document 1", "Test document 2"]
            embeddings = embedding_gen.encode_batch(test_texts)

            has_correct_shape = len(embeddings) == 2
            has_correct_dim = embeddings.shape[1] == embedding_gen.embedding_dimension

            return {
                "status": "passed" if has_correct_shape and has_correct_dim else "failed",
                "message": "RAG system validation successful",
                "embedding_dimension": embedding_gen.embedding_dimension,
                "device": embedding_gen.device,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "status": "failed",
                "message": f"RAG system validation failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def _validate_local_llm(self) -> Dict[str, Any]:
        """Validuje lokální LLM implementaci."""
        try:
            # Import test (bez načítání modelu)
            sys.path.insert(0, str(self.project_root))
            from src.core.local_llm import LLMConfig, ModelDownloader

            # Test konfigurace
            config = LLMConfig(
                model_path="/fake/path/model.gguf",
                n_ctx=2048,
                metal=True
            )

            # Test model downloader
            available_models = ModelDownloader.list_available_models()
            has_models = len(available_models) > 0

            return {
                "status": "passed" if has_models else "failed",
                "message": "Local LLM validation successful",
                "available_models": list(available_models.keys()),
                "config_valid": True,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "status": "failed",
                "message": f"Local LLM validation failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def _validate_autonomous_server(self) -> Dict[str, Any]:
        """Validuje autonomní server implementaci."""
        try:
            # Import test
            sys.path.insert(0, str(self.project_root))
            from src.core.autonomous_server import AutonomousServer, create_app

            # Test vytvoření aplikace
            app = create_app()

            # Kontrola routes
            routes = [route.path for route in app.routes]
            required_routes = ["/health", "/metrics", "/query", "/chat", "/stats"]

            missing_routes = [route for route in required_routes if route not in routes]

            return {
                "status": "passed" if not missing_routes else "failed",
                "message": "Autonomous server validation successful",
                "available_routes": routes,
                "missing_routes": missing_routes,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "status": "failed",
                "message": f"Autonomous server validation failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def _validate_integration(self) -> Dict[str, Any]:
        """Validuje integraci všech komponent."""
        try:
            # Simulace integračního testu
            integration_checks = []

            # Kontrola importů
            sys.path.insert(0, str(self.project_root))

            try:
                from src.core.elt_pipeline import ELTPipeline
                integration_checks.append(("elt_import", True))
            except ImportError:
                integration_checks.append(("elt_import", False))

            try:
                from src.core.rag_system import LocalRAGSystem
                integration_checks.append(("rag_import", True))
            except ImportError:
                integration_checks.append(("rag_import", False))

            try:
                from src.core.local_llm import RAGLLMPipeline
                integration_checks.append(("llm_import", True))
            except ImportError:
                integration_checks.append(("llm_import", False))

            try:
                from src.core.autonomous_server import create_app
                integration_checks.append(("server_import", True))
            except ImportError:
                integration_checks.append(("server_import", False))

            failed_integrations = [name for name, passed in integration_checks if not passed]

            return {
                "status": "passed" if not failed_integrations else "failed",
                "message": "Integration validation",
                "failed_integrations": failed_integrations,
                "all_integrations": dict(integration_checks),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "status": "failed",
                "message": f"Integration validation failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    def _calculate_overall_status(self) -> None:
        """Vypočítá celkový stav validace."""
        test_results = self.test_results["tests"]

        if not test_results:
            self.test_results["overall_status"] = "no_tests"
            return

        statuses = [test["status"] for test in test_results.values()]

        if all(status == "passed" for status in statuses):
            self.test_results["overall_status"] = "passed"
        elif any(status == "error" for status in statuses):
            self.test_results["overall_status"] = "error"
        else:
            self.test_results["overall_status"] = "failed"

        # Statistiky
        self.test_results["summary"] = {
            "total_tests": len(statuses),
            "passed": statuses.count("passed"),
            "failed": statuses.count("failed"),
            "errors": statuses.count("error")
        }

    def save_results(self, output_path: Path) -> None:
        """Uloží výsledky validace."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)

        logger.info(f"Validation results saved to: {output_path}")


async def main():
    """Hlavní funkce pro spuštění validace."""
    project_root = Path(__file__).parent.parent

    validator = Phase1Validator(project_root)
    results = await validator.run_all_validations()

    # Uložení výsledků
    output_path = project_root / "artifacts" / "phase1_validation_results.json"
    validator.save_results(output_path)

    # Výpis výsledků
    print("\n" + "="*50)
    print("FÁZE 1 VALIDACE - VÝSLEDKY")
    print("="*50)

    print(f"Celkový stav: {results['overall_status'].upper()}")
    print(f"Testů celkem: {results['summary']['total_tests']}")
    print(f"Prošlo: {results['summary']['passed']}")
    print(f"Selhalo: {results['summary']['failed']}")
    print(f"Chyby: {results['summary']['errors']}")

    print("\nDetail testů:")
    for test_name, test_result in results["tests"].items():
        status_icon = "✅" if test_result["status"] == "passed" else "❌"
        print(f"{status_icon} {test_name}: {test_result['status']}")
        if test_result["status"] != "passed":
            print(f"   Zpráva: {test_result['message']}")

    print(f"\nVýsledky uloženy: {output_path}")

    # Exit code pro CI/CD
    sys.exit(0 if results["overall_status"] == "passed" else 1)


if __name__ == "__main__":
    asyncio.run(main())
