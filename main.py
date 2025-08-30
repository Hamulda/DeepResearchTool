#!/usr/bin/env python3
"""
Aktualizovaný hlavní vstupní bod s podporou LangGraph architektury
Zachovává kompatibilitu se starým API a přidává novou funkcionalität

Author: Senior Python/MLOps Agent
"""

# Pre-flight kontrola architektury - MUSÍ být jako první!
import sys
from pathlib import Path

# Přidání scripts do path pro import
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

try:
    from verify_environment import verify_arm64_architecture, verify_memory_availability
    # Spuštění pre-flight kontroly
    verify_arm64_architecture()
    verify_memory_availability()
except ImportError:
    print("VAROVÁNÍ: Pre-flight kontrola architektury není dostupná")
except SystemExit:
    # Re-raise SystemExit z verify_environment
    raise

import asyncio
import argparse
import json
import logging
import time
from typing import Dict, Any, Optional

# Původní komponenty
from src.utils.gates import GateKeeper, EvidenceGateError, ComplianceGateError, MetricsGateError
from src.core.config import load_config
from src.core.pipeline import ResearchPipeline

# Nové LangGraph komponenty
from src.core.langgraph_agent import ResearchAgentGraph
from src.core.config_langgraph import load_config as load_langgraph_config, validate_config

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModernResearchAgent:
    """
    Moderní Research Agent s LangGraph stavovou architekturou
    Nahrazuje původní AutomaticResearchAgent s pokročilými funkcemi
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        profile: str = "thorough",
        use_langgraph: bool = True,
    ):
        """
        Inicializace moderního research agenta

        Args:
            config_path: Cesta ke konfiguračnímu souboru (legacy)
            profile: Profil výzkumu (quick/thorough/academic)
            use_langgraph: Použít LangGraph architekturu (default: True)
        """
        self.profile = profile
        self.use_langgraph = use_langgraph

        if use_langgraph:
            # Nová LangGraph architektura
            self.config = load_langgraph_config(profile=profile)

            # Validace konfigurace
            validation_errors = validate_config(self.config)
            if validation_errors:
                raise ValueError(f"Chyby v konfiguraci: {validation_errors}")

            self.agent = ResearchAgentGraph(self.config)
            self.pipeline = None
            self.gatekeeper = None

            logger.info(f"Inicializován moderní LangGraph agent (profil: {profile})")
        else:
            # Legacy architektura pro zpětnou kompatibilitu
            self.config = load_config(config_path or "config.yaml")
            self.agent = None
            self.pipeline = None
            self.gatekeeper = GateKeeper(self.config)
            self._configure_profile()

            logger.info(f"Inicializován legacy agent (profil: {profile})")

    def _configure_profile(self):
        """Konfigurace podle profilu (pro legacy režim)"""
        if self.profile == "quick":
            # Quick profile: 4k context, lower precision, faster
            self.config.update(
                {
                    "retrieval": {
                        **self.config.get("retrieval", {}),
                        "max_context_tokens": 4000,
                        "top_k": 20,
                        "ef_search": 100,
                    },
                    "synthesis": {
                        **self.config.get("synthesis", {}),
                        "max_claims": 3,
                        "min_citations_per_claim": 2,
                    },
                    "gates": {
                        **self.config.get("gates", {}),
                        "metrics": {
                            "min_recall": 0.6,
                            "min_precision": 0.7,
                            "min_groundedness": 0.8,
                        },
                    },
                }
            )
        elif self.profile == "thorough":
            # Thorough profile: 8k context, higher precision, slower
            self.config.update(
                {
                    "retrieval": {
                        **self.config.get("retrieval", {}),
                        "max_context_tokens": 8000,
                        "top_k": 50,
                        "ef_search": 200,
                    },
                    "synthesis": {
                        **self.config.get("synthesis", {}),
                        "max_claims": 8,
                        "min_citations_per_claim": 3,
                    },
                    "gates": {
                        **self.config.get("gates", {}),
                        "metrics": {
                            "min_recall": 0.7,
                            "min_precision": 0.8,
                            "min_groundedness": 0.85,
                        },
                    },
                }
            )

    async def research(self, query: str, audit_mode: bool = False) -> Dict[str, Any]:
        """
        Hlavní research funkce s podporou obou architektur

        Args:
            query: Výzkumný dotaz
            audit_mode: Zda ukládat detailní audit artefakty

        Returns:
            Výsledek research s claims a citacemi
        """
        start_time = time.time()

        logger.info(f"Starting research for query: {query}")
        logger.info(f"Architecture: {'LangGraph' if self.use_langgraph else 'Legacy'}")
        logger.info(f"Profile: {self.profile}, Audit mode: {audit_mode}")

        try:
            if self.use_langgraph:
                # Nová LangGraph architektura
                result = await self.agent.research(query)

                # Konverze na kompatibilní formát
                final_result = {
                    "query": query,
                    "claims": self._extract_claims_from_synthesis(result["synthesis"]),
                    "citations": self._extract_citations_from_docs(result["retrieved_docs"]),
                    "synthesis": result["synthesis"],
                    "processing_time": result["processing_time"],
                    "profile": self.profile,
                    "architecture": "langgraph",
                    "validation_scores": result["validation_scores"],
                    "plan": result["plan"],
                    "errors": result["errors"],
                    "metadata": result["metadata"],
                }

                logger.info(f"LangGraph research completed in {result['processing_time']:.2f}s")
                return final_result

            else:
                # Legacy architektura
                self.pipeline = ResearchPipeline(self.config)
                await self.pipeline.initialize()

                result = await self.pipeline.execute(query)

                # Prepare data for gate validation
                gate_data = {
                    "synthesis_result": result.get("synthesis"),
                    "retrieval_log": result.get("retrieval_log", {}),
                    "evaluation_result": result.get("evaluation", {}),
                    "output_data": {
                        "claims": result.get("claims", []),
                        "citations": result.get("citations", []),
                        "token_count": result.get("token_count", 0),
                    },
                }

                # Run all validation gates (fail-hard)
                logger.info("Running validation gates...")
                gate_results = await self.gatekeeper.validate_all(gate_data)

                # If we reach here, all gates passed
                processing_time = time.time() - start_time

                final_result = {
                    "query": query,
                    "claims": result.get("claims", []),
                    "citations": result.get("citations", []),
                    "processing_time": processing_time,
                    "profile": self.profile,
                    "gate_results": [
                        {
                            "name": gr.gate_name,
                            "passed": gr.passed,
                            "score": gr.score,
                            "message": gr.message,
                        }
                        for gr in gate_results
                    ],
                    "metadata": {
                        "token_count": result.get("token_count", 0),
                        "retrieval_stats": result.get("retrieval_log", {}).get("stats", {}),
                        "synthesis_stats": result.get("synthesis", {}).get("stats", {}),
                        "audit_mode": audit_mode,
                    },
                }

                # Save audit artifacts if requested
                if audit_mode:
                    await self._save_audit_artifacts(final_result, result)

                logger.info(f"Research completed successfully in {processing_time:.2f}s")
                logger.info(
                    f"Generated {len(final_result['claims'])} claims with {len(final_result['citations'])} citations"
                )

                return final_result

        except Exception as e:
            logger.error(f"Research failed: {e}")
            return {
                "query": query,
                "error": str(e),
                "claims": [],
                "citations": [],
                "processing_time": time.time() - start_time,
                "profile": self.profile,
                "architecture": "langgraph" if self.use_langgraph else "legacy",
            }

    def _extract_claims_from_synthesis(self, synthesis: str) -> list:
        """Extrakce claims ze syntézy (pro kompatibilitu)"""
        # Jednoduchá implementace - v praxi by byla sofistikovanější
        claims = []
        lines = synthesis.split("\n")

        for line in lines:
            line = line.strip()
            if line and (
                line.startswith("-")
                or line.startswith("•")
                or line.startswith("*")
                or any(line.startswith(f"{i}.") for i in range(1, 20))
            ):
                clean_claim = line.lstrip("-•*0123456789. ").strip()
                if len(clean_claim) > 20:  # Filtr příliš krátkých řádků
                    claims.append(
                        {
                            "claim": clean_claim,
                            "confidence": 0.8,  # Default confidence
                            "source": "synthesis",
                        }
                    )

        return claims[: self.config.get("synthesis", {}).get("max_claims", 8)]

    def _extract_citations_from_docs(self, docs: list) -> list:
        """Extrakce citací z dokumentů (pro kompatibilitu)"""
        citations = []

        for i, doc in enumerate(docs):
            citation = {
                "id": f"cite_{i+1}",
                "source": doc.get("source", "unknown"),
                "url": doc.get("metadata", {}).get("url", ""),
                "title": doc.get("metadata", {}).get("title", f"Document {i+1}"),
                "content_preview": doc.get("content", "")[:200] + "...",
                "relevance_score": doc.get("metadata", {}).get("distance", 0.5),
            }
            citations.append(citation)

        return citations

    async def _save_audit_artifacts(self, final_result: Dict[str, Any], raw_result: Dict[str, Any]):
        """Uložení audit artefaktů"""
        timestamp = int(time.time())
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)

        # Main result
        with open(artifacts_dir / f"research_result_{timestamp}.json", "w") as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)

        # Detailed logs
        with open(artifacts_dir / f"research_audit_{timestamp}.json", "w") as f:
            json.dump(raw_result, f, indent=2, ensure_ascii=False)

        logger.info(f"Audit artifacts saved to artifacts/research_*_{timestamp}.json")


# Zachování zpětné kompatibility
class AutomaticResearchAgent(ModernResearchAgent):
    """Alias pro zpětnou kompatibilitu"""

    def __init__(self, config_path: str, profile: str = "thorough"):
        super().__init__(config_path=config_path, profile=profile, use_langgraph=True)


async def main():
    """Hlavní funkce s podporou obou architektur"""
    parser = argparse.ArgumentParser(description="Deep Research Tool - Moderní AI Research Agent")
    parser.add_argument("query", help="Výzkumný dotaz")
    parser.add_argument(
        "--profile",
        choices=["quick", "thorough", "academic"],
        default="thorough",
        help="Profil výzkumu",
    )
    parser.add_argument(
        "--legacy", action="store_true", help="Použít legacy architekturu místo LangGraph"
    )
    parser.add_argument("--audit", action="store_true", help="Zapnout audit mode")
    parser.add_argument("--output", help="Soubor pro uložení výsledků (JSON)")
    parser.add_argument("--config", help="Cesta ke konfiguračnímu souboru")

    args = parser.parse_args()

    try:
        # Inicializace agenta
        agent = ModernResearchAgent(
            config_path=args.config, profile=args.profile, use_langgraph=not args.legacy
        )

        # Spuštění výzkumu
        result = await agent.research(args.query, audit_mode=args.audit)

        # Výstup výsledků
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Výsledky uloženy do: {args.output}")
        else:
            # Výpis na konzoli
            print("\n" + "=" * 60)
            print(f"VÝSLEDKY VÝZKUMU")
            print("=" * 60)
            print(f"Dotaz: {result['query']}")
            print(f"Architektura: {result.get('architecture', 'legacy')}")
            print(f"Profil: {result['profile']}")
            print(f"Čas zpracování: {result['processing_time']:.2f}s")

            if "plan" in result and result["plan"]:
                print(f"\nPlán výzkumu:")
                for i, step in enumerate(result["plan"], 1):
                    print(f"  {i}. {step}")

            print(f"\nNalezené dokumenty: {len(result.get('retrieved_docs', []))}")
            print(f"Extrahované claims: {len(result['claims'])}")
            print(f"Citace: {len(result['citations'])}")

            if "validation_scores" in result:
                print(f"\nValidační skóre:")
                for metric, score in result["validation_scores"].items():
                    print(f"  {metric}: {score:.2f}")

            if "synthesis" in result and result["synthesis"]:
                print(f"\nSyntéza:")
                print("-" * 40)
                print(result["synthesis"])

            if result.get("errors"):
                print(f"\n⚠️  Chyby:")
                for error in result["errors"]:
                    print(f"  - {error}")

        return 0

    except KeyboardInterrupt:
        print("\nPřerušeno uživatelem")
        return 1
    except Exception as e:
        logger.error(f"Kritická chyba: {e}")
        print(f"❌ Chyba: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
