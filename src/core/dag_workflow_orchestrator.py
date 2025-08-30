#!/usr/bin/env python3
"""DAG Workflow Orchestrator pro Deep Research Tool
Implementuje řízený pipeline: retrieval → re-ranking → synthesis → verification

Author: Senior IT Specialist
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
import time
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class WorkflowPhase(Enum):
    """Fáze workflow"""

    RETRIEVAL = "retrieval"
    RERANKING = "reranking"
    SYNTHESIS = "synthesis"
    VERIFICATION = "verification"


class TaskStatus(Enum):
    """Status úkolu"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowMetrics:
    """Metriky workflow"""

    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    phase_times: dict[str, float] = field(default_factory=dict)
    documents_processed: int = 0
    queries_executed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    def get_total_duration(self) -> float:
        """Celková doba běhu"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()


@dataclass
class ResearchTask:
    """Jednotlivý výzkumný úkol"""

    id: str
    query: str
    phase: WorkflowPhase
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 1
    parent_task_id: str | None = None
    subqueries: list[str] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None


class WorkflowContract:
    """Kontrakt mezi fázemi workflow"""

    @staticmethod
    def retrieval_output_schema() -> dict[str, Any]:
        """Schema výstupu z retrieval fáze"""
        return {
            "documents": [
                {
                    "id": "str",
                    "title": "str",
                    "content": "str",
                    "source": "str",
                    "url": "str",
                    "timestamp": "datetime",
                    "score": "float",
                    "metadata": "dict",
                }
            ],
            "query_expansion": ["str"],
            "total_found": "int",
        }

    @staticmethod
    def reranking_output_schema() -> dict[str, Any]:
        """Schema výstupu z re-ranking fáze"""
        return {
            "ranked_documents": [
                {
                    "document_id": "str",
                    "relevance_score": "float",
                    "authority_score": "float",
                    "novelty_score": "float",
                    "combined_score": "float",
                    "ranking_reason": "str",
                }
            ]
        }


class DAGWorkflowOrchestrator:
    """Hlavní orchestrátor DAG workflow"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.workflow_config = config.get("workflow", {})
        self.tasks: dict[str, ResearchTask] = {}
        self.metrics = WorkflowMetrics()
        self.logger = structlog.get_logger(__name__)

        # Komponenty workflow
        self.retrieval_engine = None
        self.reranking_engine = None
        self.synthesis_engine = None
        self.verification_engine = None

        # Fan-out/fan-in konfigurace
        self.max_parallel_subqueries = (
            self.workflow_config.get("phases", {})
            .get("retrieval", {})
            .get("parallel_subqueries", 4)
        )
        self.human_checkpoints = self.workflow_config.get("human_checkpoints", {})

    async def initialize_engines(self):
        """Inicializace všech engines"""
        self.logger.info("Inicializace workflow engines")

        # Import a inicializace komponent
        from .hybrid_retrieval_engine import HybridRetrievalEngine
        from .reranking_engine import ReRankingEngine
        from .synthesis_engine import SynthesisEngine
        from .verification_engine import VerificationEngine

        self.retrieval_engine = HybridRetrievalEngine(self.config)
        self.reranking_engine = ReRankingEngine(self.config)
        self.synthesis_engine = SynthesisEngine(self.config)
        self.verification_engine = VerificationEngine(self.config)

        await self.retrieval_engine.initialize()
        await self.reranking_engine.initialize()
        await self.synthesis_engine.initialize()
        await self.verification_engine.initialize()

    async def execute_research_workflow(
        self, main_query: str, research_depth: int = 3, max_documents: int = 100
    ) -> dict[str, Any]:
        """Hlavní metoda pro spuštění výzkumného workflow"""
        self.logger.info(
            "Spouštím research workflow",
            query=main_query,
            depth=research_depth,
            max_docs=max_documents,
        )

        try:
            # Fáze 1: Retrieval s query expansion a fan-out
            retrieval_result = await self._execute_retrieval_phase(
                main_query, research_depth, max_documents
            )

            # Human checkpoint po retrieval (volitelně)
            if self.human_checkpoints.get("after_retrieval", False):
                retrieval_result = await self._human_checkpoint("retrieval", retrieval_result)

            # Fáze 2: Re-ranking kandidátů
            reranking_result = await self._execute_reranking_phase(retrieval_result)

            # Fáze 3: Synthesis s evidence binding
            synthesis_result = await self._execute_synthesis_phase(main_query, reranking_result)

            # Human checkpoint po synthesis (volitelně)
            if self.human_checkpoints.get("after_synthesis", False):
                synthesis_result = await self._human_checkpoint("synthesis", synthesis_result)

            # Fáze 4: Verification nezávislým modelem
            verification_result = await self._execute_verification_phase(synthesis_result)

            # Sestavení finálního výsledku
            final_result = await self._compile_final_result(
                main_query,
                retrieval_result,
                reranking_result,
                synthesis_result,
                verification_result,
            )

            self.metrics.end_time = datetime.now()

            # Export research grafu a metrik
            await self._export_research_graph(final_result)
            await self._export_metrics()

            return final_result

        except Exception as e:
            self.logger.error("Chyba ve workflow", error=str(e))
            raise

    async def _execute_retrieval_phase(
        self, main_query: str, depth: int, max_docs: int
    ) -> dict[str, Any]:
        """Fáze 1: Retrieval s query expansion"""
        phase_start = time.time()
        self.logger.info("Spouštím retrieval fázi")

        # Query expansion pro generování subdotazů
        expanded_queries = await self.retrieval_engine.expand_query(
            main_query, max_expansions=self.max_parallel_subqueries
        )

        # Paralelní spuštění subdotazů (fan-out)
        retrieval_tasks = []
        for i, subquery in enumerate(expanded_queries):
            task = ResearchTask(
                id=f"retrieval_{i}",
                query=subquery,
                phase=WorkflowPhase.RETRIEVAL,
                parent_task_id="main",
            )
            self.tasks[task.id] = task
            retrieval_tasks.append(
                self.retrieval_engine.search(
                    subquery, max_results=max_docs // len(expanded_queries)
                )
            )

        # Čekání na dokončení všech subdotazů
        retrieval_results = await asyncio.gather(*retrieval_tasks)

        # Slučování a deduplikace výsledků (fan-in)
        merged_results = await self._merge_retrieval_results(retrieval_results)

        self.metrics.phase_times["retrieval"] = time.time() - phase_start
        self.metrics.documents_processed += len(merged_results.get("documents", []))
        self.metrics.queries_executed += len(expanded_queries)

        return {
            "phase": "retrieval",
            "main_query": main_query,
            "expanded_queries": expanded_queries,
            "documents": merged_results.get("documents", []),
            "total_documents": len(merged_results.get("documents", [])),
            "sources_used": merged_results.get("sources", []),
        }

    async def _execute_reranking_phase(self, retrieval_result: dict[str, Any]) -> dict[str, Any]:
        """Fáze 2: Re-ranking dokumentů"""
        phase_start = time.time()
        self.logger.info("Spouštím re-ranking fázi")

        documents = retrieval_result.get("documents", [])
        main_query = retrieval_result.get("main_query", "")

        # Re-ranking s konfigurovatelným K
        top_k = self.workflow_config.get("phases", {}).get("reranking", {}).get("final_k", 20)

        ranked_results = await self.reranking_engine.rerank_documents(
            query=main_query, documents=documents, top_k=top_k
        )

        self.metrics.phase_times["reranking"] = time.time() - phase_start

        return {
            "phase": "reranking",
            "main_query": main_query,
            "ranked_documents": ranked_results["ranked_documents"],
            "reranking_metrics": ranked_results.get("metrics", {}),
        }

    async def _execute_synthesis_phase(
        self, main_query: str, reranking_result: dict[str, Any]
    ) -> dict[str, Any]:
        """Fáze 3: Synthesis s evidence binding"""
        phase_start = time.time()
        self.logger.info("Spouštím synthesis fázi")

        ranked_documents = reranking_result.get("ranked_documents", [])

        # Synthesis s povinným evidence binding
        synthesis_result = await self.synthesis_engine.synthesize_with_evidence(
            query=main_query,
            documents=ranked_documents,
            min_evidence_per_claim=self.workflow_config.get("phases", {})
            .get("synthesis", {})
            .get("min_evidence_per_claim", 2),
        )

        self.metrics.phase_times["synthesis"] = time.time() - phase_start

        return {
            "phase": "synthesis",
            "main_query": main_query,
            "claims": synthesis_result["claims"],
            "evidence_bindings": synthesis_result["evidence_bindings"],
            "overall_confidence": synthesis_result["confidence"],
            "synthesis_metadata": synthesis_result.get("metadata", {}),
        }

    async def _execute_verification_phase(self, synthesis_result: dict[str, Any]) -> dict[str, Any]:
        """Fáze 4: Nezávislá verifikace"""
        phase_start = time.time()
        self.logger.info("Spouštím verification fázi")

        claims = synthesis_result.get("claims", [])
        evidence_bindings = synthesis_result.get("evidence_bindings", {})

        # Nezávislá verifikace jiným modelem
        verification_result = await self.verification_engine.verify_claims(
            claims=claims,
            evidence_bindings=evidence_bindings,
            threshold=self.workflow_config.get("phases", {})
            .get("verification", {})
            .get("verification_threshold", 0.8),
        )

        self.metrics.phase_times["verification"] = time.time() - phase_start

        return {
            "phase": "verification",
            "verified_claims": verification_result["verified_claims"],
            "flagged_claims": verification_result["flagged_claims"],
            "verification_confidence": verification_result["confidence"],
            "verification_reasoning": verification_result.get("reasoning", []),
        }

    async def _merge_retrieval_results(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Sloučení a deduplikace výsledků z retrieval"""
        all_documents = []
        all_sources = set()

        for result in results:
            documents = result.get("documents", [])
            all_documents.extend(documents)
            all_sources.update(result.get("sources", []))

        # Deduplikace podle URL/ID
        seen_urls = set()
        unique_documents = []

        for doc in all_documents:
            doc_id = doc.get("url", doc.get("id", ""))
            if doc_id not in seen_urls:
                seen_urls.add(doc_id)
                unique_documents.append(doc)

        return {"documents": unique_documents, "sources": list(all_sources)}

    async def _human_checkpoint(self, phase: str, data: dict[str, Any]) -> dict[str, Any]:
        """Human-in-the-loop checkpoint"""
        self.logger.info(f"Human checkpoint pro fázi: {phase}")

        # Export dat pro lidskou kontrolu
        checkpoint_file = f"./data/checkpoints/{phase}_{datetime.now().isoformat()}.json"
        Path(checkpoint_file).parent.mkdir(parents=True, exist_ok=True)

        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        print(f"\n🔍 HUMAN CHECKPOINT - {phase.upper()}")
        print(f"Data exportována do: {checkpoint_file}")
        print("Zkontrolujte data a stiskněte Enter pro pokračování...")
        input()

        return data

    async def _compile_final_result(
        self,
        main_query: str,
        retrieval_result: dict[str, Any],
        reranking_result: dict[str, Any],
        synthesis_result: dict[str, Any],
        verification_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Sestavení finálního výsledku"""
        return {
            "query": main_query,
            "timestamp": datetime.now().isoformat(),
            "workflow_metrics": {
                "total_duration_seconds": self.metrics.get_total_duration(),
                "phase_durations": self.metrics.phase_times,
                "documents_processed": self.metrics.documents_processed,
                "queries_executed": self.metrics.queries_executed,
                "cache_performance": {
                    "hits": self.metrics.cache_hits,
                    "misses": self.metrics.cache_misses,
                    "hit_rate": (
                        self.metrics.cache_hits
                        / (self.metrics.cache_hits + self.metrics.cache_misses)
                        if (self.metrics.cache_hits + self.metrics.cache_misses) > 0
                        else 0
                    ),
                },
            },
            "retrieval_summary": {
                "expanded_queries": retrieval_result.get("expanded_queries", []),
                "total_documents_found": retrieval_result.get("total_documents", 0),
                "sources_consulted": retrieval_result.get("sources_used", []),
            },
            "reranking_summary": {
                "documents_reranked": len(reranking_result.get("ranked_documents", [])),
                "metrics": reranking_result.get("reranking_metrics", {}),
            },
            "verified_claims": verification_result.get("verified_claims", []),
            "flagged_claims": verification_result.get("flagged_claims", []),
            "evidence_bindings": synthesis_result.get("evidence_bindings", {}),
            "overall_confidence": verification_result.get("verification_confidence", 0.0),
            "citation_summary": self._generate_citation_summary(
                synthesis_result, verification_result
            ),
        }

    def _generate_citation_summary(
        self, synthesis_result: dict[str, Any], verification_result: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generování přehledu citací"""
        citations = []
        evidence_bindings = synthesis_result.get("evidence_bindings", {})

        for claim_id, evidence_list in evidence_bindings.items():
            for evidence in evidence_list:
                citation = {
                    "claim_id": claim_id,
                    "source": evidence.get("source", ""),
                    "citation": evidence.get("citation", ""),
                    "passage": evidence.get("passage", ""),
                    "confidence": evidence.get("confidence", 0.0),
                    "timestamp": evidence.get("timestamp", ""),
                    "verification_status": (
                        "verified"
                        if claim_id
                        in [c.get("id") for c in verification_result.get("verified_claims", [])]
                        else "flagged"
                    ),
                }
                citations.append(citation)

        return citations

    async def _export_research_graph(self, final_result: dict[str, Any]):
        """Export research grafu pro audit"""
        if (
            not self.config.get("observability", {})
            .get("tracing", {})
            .get("export_research_graph", False)
        ):
            return

        graph_data = {
            "nodes": [],
            "edges": [],
            "metadata": {
                "query": final_result["query"],
                "timestamp": final_result["timestamp"],
                "total_duration": final_result["workflow_metrics"]["total_duration_seconds"],
            },
        }

        # Přidání uzlů pro každý úkol
        for task_id, task in self.tasks.items():
            graph_data["nodes"].append(
                {
                    "id": task_id,
                    "label": task.query,
                    "phase": task.phase.value,
                    "status": task.status.value,
                    "confidence": task.confidence,
                }
            )

            # Přidání hran parent-child
            if task.parent_task_id:
                graph_data["edges"].append(
                    {"source": task.parent_task_id, "target": task_id, "type": "derivation"}
                )

        # Export ve více formátech
        graph_formats = (
            self.config.get("observability", {}).get("tracing", {}).get("graph_formats", ["json"])
        )

        for format_type in graph_formats:
            if format_type == "json":
                graph_file = (
                    f"./data/graphs/research_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                Path(graph_file).parent.mkdir(parents=True, exist_ok=True)
                with open(graph_file, "w", encoding="utf-8") as f:
                    json.dump(graph_data, f, indent=2)

            elif format_type == "graphml":
                # GraphML export pro Gephi/Cytoscape
                import networkx as nx

                G = nx.DiGraph()

                for node in graph_data["nodes"]:
                    G.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})

                for edge in graph_data["edges"]:
                    G.add_edge(edge["source"], edge["target"], type=edge["type"])

                graphml_file = f"./data/graphs/research_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.graphml"
                nx.write_graphml(G, graphml_file)

    async def _export_metrics(self):
        """Export metrik workflow"""
        metrics_data = {
            "timestamp": datetime.now().isoformat(),
            "total_duration_seconds": self.metrics.get_total_duration(),
            "phase_durations": self.metrics.phase_times,
            "documents_processed": self.metrics.documents_processed,
            "queries_executed": self.metrics.queries_executed,
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "tasks_summary": {
                "total": len(self.tasks),
                "by_status": {
                    status.value: len([t for t in self.tasks.values() if t.status == status])
                    for status in TaskStatus
                },
                "by_phase": {
                    phase.value: len([t for t in self.tasks.values() if t.phase == phase])
                    for phase in WorkflowPhase
                },
            },
        }

        metrics_file = (
            f"./data/metrics/workflow_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        Path(metrics_file).parent.mkdir(parents=True, exist_ok=True)

        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)

        self.logger.info("Workflow metriky exportovány", file=metrics_file)
