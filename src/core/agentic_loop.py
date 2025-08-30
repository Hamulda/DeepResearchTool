"""🤖 Autonomní výzkumný agent s rekurzivní smyčkou
Implementuje inteligentní rozhodování o dalších krocích na základě zjištění
"""

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any
import uuid

from ..scrapers.web_scraper import WebScraper
from ..storage.document_store import DocumentStore
from ..synthesis.correlation_engine import CorrelationEngine
from ..synthesis.credibility_assessor import CredibilityAssessor
from ..synthesis.deep_pattern_detector import DeepPatternDetector
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class TaskType(Enum):
    """Typy úkolů pro autonomní agenta"""

    SCRAPE = "scrape"
    ANALYZE = "analyze"
    CORRELATE = "correlate"
    VALIDATE = "validate"
    EXPAND_SEARCH = "expand_search"
    DEEP_DIVE = "deep_dive"
    CROSS_REFERENCE = "cross_reference"


class TaskPriority(Enum):
    """Priorita úkolů"""

    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class AgentTask:
    """Struktura úkolu pro autonomní agenta"""

    id: str
    task_type: TaskType
    priority: TaskPriority
    parameters: dict[str, Any]
    created_at: datetime
    parent_task_id: str | None = None
    attempts: int = 0
    max_attempts: int = 3
    completed: bool = False
    failed: bool = False
    result: dict[str, Any] | None = None
    credibility_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["task_type"] = self.task_type.value
        data["priority"] = self.priority.value
        data["created_at"] = self.created_at.isoformat()
        return data


class TaskQueue:
    """Prioritní fronta úkolů s inteligentním plánováním"""

    def __init__(self, max_concurrent_tasks: int = 5):
        self.tasks: list[AgentTask] = []
        self.completed_tasks: list[AgentTask] = []
        self.failed_tasks: list[AgentTask] = []
        self.max_concurrent_tasks = max_concurrent_tasks
        self.running_tasks: set[str] = set()

    def add_task(self, task: AgentTask) -> None:
        """Přidá úkol do fronty podle priority"""
        # Kontrola duplicit
        if self._is_duplicate_task(task):
            logger.info(f"Duplicitní úkol ignorován: {task.task_type.value}")
            return

        self.tasks.append(task)
        self.tasks.sort(key=lambda t: (t.priority.value, t.created_at))
        logger.info(f"Přidán úkol: {task.task_type.value} (priorita: {task.priority.value})")

    def get_next_task(self) -> AgentTask | None:
        """Vrátí další úkol k vykonání"""
        if len(self.running_tasks) >= self.max_concurrent_tasks:
            return None

        for task in self.tasks:
            if not task.completed and not task.failed and task.id not in self.running_tasks:
                if task.attempts < task.max_attempts:
                    return task
        return None

    def mark_task_running(self, task_id: str) -> None:
        """Označí úkol jako běžící"""
        self.running_tasks.add(task_id)

    def complete_task(self, task_id: str, result: dict[str, Any]) -> None:
        """Označí úkol jako dokončený"""
        for task in self.tasks:
            if task.id == task_id:
                task.completed = True
                task.result = result
                self.completed_tasks.append(task)
                self.running_tasks.discard(task_id)
                break

    def fail_task(self, task_id: str, error: str) -> None:
        """Označí úkol jako neúspěšný"""
        for task in self.tasks:
            if task.id == task_id:
                task.attempts += 1
                if task.attempts >= task.max_attempts:
                    task.failed = True
                    self.failed_tasks.append(task)
                self.running_tasks.discard(task_id)
                logger.warning(f"Úkol {task_id} selhal: {error}")
                break

    def _is_duplicate_task(self, new_task: AgentTask) -> bool:
        """Kontroluje duplicitní úkoly"""
        for task in self.tasks:
            if (
                task.task_type == new_task.task_type
                and task.parameters == new_task.parameters
                and not task.completed
                and not task.failed
            ):
                return True
        return False

    def get_stats(self) -> dict[str, int]:
        """Vrátí statistiky fronty"""
        return {
            "pending": len([t for t in self.tasks if not t.completed and not t.failed]),
            "running": len(self.running_tasks),
            "completed": len(self.completed_tasks),
            "failed": len(self.failed_tasks),
        }


class AgenticLoop:
    """🤖 Autonomní výzkumný agent s rekurzivní smyčkou

    Analyzuje zjištění z LLM a generuje nové strukturované úkoly
    na základě důvěryhodnosti a relevance nalezených informací.
    """

    def __init__(
        self,
        document_store: DocumentStore,
        credibility_assessor: CredibilityAssessor,
        correlation_engine: CorrelationEngine,
        pattern_detector: DeepPatternDetector,
        web_scraper: WebScraper,
        max_iterations: int = 10,
        min_credibility_threshold: float = 0.3,
    ):

        self.document_store = document_store
        self.credibility_assessor = credibility_assessor
        self.correlation_engine = correlation_engine
        self.pattern_detector = pattern_detector
        self.web_scraper = web_scraper

        self.task_queue = TaskQueue()
        self.max_iterations = max_iterations
        self.min_credibility_threshold = min_credibility_threshold
        self.current_iteration = 0
        self.is_running = False

        # Metriky a monitoring
        self.total_tasks_generated = 0
        self.total_discoveries = 0
        self.credibility_scores: list[float] = []

    async def start_research_cycle(
        self, initial_query: str, target_urls: list[str] = None
    ) -> dict[str, Any]:
        """🚀 Spustí autonomní výzkumný cyklus

        Args:
            initial_query: Počáteční výzkumný dotaz
            target_urls: Volitelné počáteční URL pro scraping

        Returns:
            Dict s výsledky výzkumu a metrikami

        """
        logger.info(f"🚀 Spouštím autonomní výzkumný cyklus pro: {initial_query}")

        self.is_running = True
        self.current_iteration = 0

        # Generování počátečních úkolů
        await self._generate_initial_tasks(initial_query, target_urls)

        try:
            # Hlavní smyčka agenta
            while (
                self.is_running
                and self.current_iteration < self.max_iterations
                and self._should_continue_research()
            ):

                await self._execute_iteration()
                self.current_iteration += 1

                # Krátká pauza mezi iteracemi
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Chyba v autonomní smyčce: {e}")
        finally:
            self.is_running = False

        return await self._generate_final_report()

    async def _generate_initial_tasks(self, query: str, target_urls: list[str] = None) -> None:
        """Generuje počáteční úkoly na základě dotazu"""
        # Analýza úvodního dotazu pro extrakci entit
        analysis_task = AgentTask(
            id=str(uuid.uuid4()),
            task_type=TaskType.ANALYZE,
            priority=TaskPriority.HIGH,
            parameters={"text": query, "source": "initial_query"},
            created_at=datetime.now(),
        )
        self.task_queue.add_task(analysis_task)

        # Scrapování zadaných URL
        if target_urls:
            for url in target_urls:
                scrape_task = AgentTask(
                    id=str(uuid.uuid4()),
                    task_type=TaskType.SCRAPE,
                    priority=TaskPriority.HIGH,
                    parameters={"url": url, "depth": 1},
                    created_at=datetime.now(),
                )
                self.task_queue.add_task(scrape_task)

        # Rozšířené vyhledávání na základě dotazu
        search_task = AgentTask(
            id=str(uuid.uuid4()),
            task_type=TaskType.EXPAND_SEARCH,
            priority=TaskPriority.MEDIUM,
            parameters={"query": query, "max_results": 10},
            created_at=datetime.now(),
        )
        self.task_queue.add_task(search_task)

    async def _execute_iteration(self) -> None:
        """Vykoná jednu iteraci agentní smyčky"""
        logger.info(f"🔄 Iterace {self.current_iteration + 1}/{self.max_iterations}")

        # Paralelní vykonávání úkolů
        running_tasks = []

        # Získání úkolů k vykonání
        while len(running_tasks) < self.task_queue.max_concurrent_tasks:
            task = self.task_queue.get_next_task()
            if not task:
                break

            self.task_queue.mark_task_running(task.id)
            running_tasks.append(self._execute_task(task))

        # Čekání na dokončení úkolů
        if running_tasks:
            await asyncio.gather(*running_tasks, return_exceptions=True)

        # Generování nových úkolů na základě výsledků
        await self._generate_follow_up_tasks()

    async def _execute_task(self, task: AgentTask) -> None:
        """Vykoná konkrétní úkol"""
        try:
            logger.info(f"🎯 Vykonávám úkol: {task.task_type.value}")

            if task.task_type == TaskType.SCRAPE:
                result = await self._execute_scrape_task(task)
            elif task.task_type == TaskType.ANALYZE:
                result = await self._execute_analyze_task(task)
            elif task.task_type == TaskType.CORRELATE:
                result = await self._execute_correlate_task(task)
            elif task.task_type == TaskType.VALIDATE:
                result = await self._execute_validate_task(task)
            elif task.task_type == TaskType.EXPAND_SEARCH:
                result = await self._execute_expand_search_task(task)
            elif task.task_type == TaskType.DEEP_DIVE:
                result = await self._execute_deep_dive_task(task)
            elif task.task_type == TaskType.CROSS_REFERENCE:
                result = await self._execute_cross_reference_task(task)
            else:
                raise ValueError(f"Neznámý typ úkolu: {task.task_type}")

            # Hodnocení důvěryhodnosti výsledku
            if result and "content" in result:
                credibility = await self.credibility_assessor.assess_content(
                    result["content"], result.get("url", "")
                )
                result["credibility_score"] = credibility
                task.credibility_score = credibility
                self.credibility_scores.append(credibility)

            self.task_queue.complete_task(task.id, result)
            logger.info(f"✅ Úkol dokončen: {task.task_type.value}")

        except Exception as e:
            logger.error(f"❌ Chyba při vykonávání úkolu {task.id}: {e}")
            self.task_queue.fail_task(task.id, str(e))

    async def _execute_scrape_task(self, task: AgentTask) -> dict[str, Any]:
        """Vykoná scraping úkol"""
        url = task.parameters["url"]
        depth = task.parameters.get("depth", 1)

        content = await self.web_scraper.scrape_page(url)

        # Uložení do document store
        doc_id = await self.document_store.store_document(
            {
                "url": url,
                "content": content,
                "timestamp": datetime.now(),
                "source": "autonomous_scraping",
            }
        )

        return {"url": url, "content": content, "document_id": doc_id, "depth": depth}

    async def _execute_analyze_task(self, task: AgentTask) -> dict[str, Any]:
        """Vykoná analýzu textu"""
        text = task.parameters["text"]

        # Pattern detection
        patterns = await self.pattern_detector.analyze_text(text)

        # Korelační analýza
        correlations = await self.correlation_engine.analyze_text(text)

        return {
            "text": text,
            "patterns": patterns,
            "correlations": correlations,
            "entities_found": len(correlations.get("entities", [])),
            "patterns_found": len(patterns.get("artifacts", [])),
        }

    async def _execute_correlate_task(self, task: AgentTask) -> dict[str, Any]:
        """Vykoná korelační analýzu"""
        entity_id = task.parameters["entity_id"]

        # Najít související entity
        related = await self.correlation_engine.find_related_entities(entity_id)

        return {
            "entity_id": entity_id,
            "related_entities": related,
            "correlation_strength": len(related),
        }

    async def _execute_validate_task(self, task: AgentTask) -> dict[str, Any]:
        """Vykoná validaci informací"""
        claim = task.parameters["claim"]
        sources = task.parameters.get("sources", [])

        # Křížová validace mezi zdroji
        validation_score = await self._cross_validate_claim(claim, sources)

        return {"claim": claim, "validation_score": validation_score, "sources_count": len(sources)}

    async def _execute_expand_search_task(self, task: AgentTask) -> dict[str, Any]:
        """Vykoná rozšířené vyhledávání"""
        query = task.parameters["query"]
        max_results = task.parameters.get("max_results", 5)

        # Zde by byla integrace s vyhledávači
        # Pro demonstraci vracíme mock data
        results = [f"expanded_result_{i}_for_{query}" for i in range(min(max_results, 3))]

        return {"query": query, "results": results, "results_count": len(results)}

    async def _execute_deep_dive_task(self, task: AgentTask) -> dict[str, Any]:
        """Vykoná hloubkovou analýzu"""
        target = task.parameters["target"]

        # Detailní analýza konkrétní entity nebo tématu
        deep_analysis = {
            "target": target,
            "depth_level": task.parameters.get("depth_level", 2),
            "findings": f"deep_analysis_of_{target}",
        }

        return deep_analysis

    async def _execute_cross_reference_task(self, task: AgentTask) -> dict[str, Any]:
        """Vykoná křížové porovnání"""
        entities = task.parameters["entities"]

        # Porovnání entit napříč zdroji
        cross_ref = {"entities": entities, "common_attributes": [], "discrepancies": []}

        return cross_ref

    async def _generate_follow_up_tasks(self) -> None:
        """Generuje navazující úkoly na základě výsledků"""
        # Analyzuje dokončené úkoly z poslední iterace
        recent_completed = [
            task
            for task in self.task_queue.completed_tasks
            if task.result and task.credibility_score >= self.min_credibility_threshold
        ]

        for task in recent_completed[-5:]:  # Posledních 5 úkolů
            await self._analyze_task_for_follow_ups(task)

    async def _analyze_task_for_follow_ups(self, completed_task: AgentTask) -> None:
        """Analyzuje dokončený úkol a generuje navazující úkoly"""
        if completed_task.task_type == TaskType.SCRAPE:
            # Nalezené vzory → další analýza
            if completed_task.result.get("patterns_found", 0) > 0:
                analyze_task = AgentTask(
                    id=str(uuid.uuid4()),
                    task_type=TaskType.ANALYZE,
                    priority=TaskPriority.MEDIUM,
                    parameters={"text": completed_task.result["content"]},
                    created_at=datetime.now(),
                    parent_task_id=completed_task.id,
                )
                self.task_queue.add_task(analyze_task)

        elif completed_task.task_type == TaskType.ANALYZE:
            # Nalezené entity → korelace
            entities = completed_task.result.get("correlations", {}).get("entities", [])
            for entity in entities[:3]:  # Max 3 entity
                correlate_task = AgentTask(
                    id=str(uuid.uuid4()),
                    task_type=TaskType.CORRELATE,
                    priority=TaskPriority.LOW,
                    parameters={"entity_id": entity.get("id", str(uuid.uuid4()))},
                    created_at=datetime.now(),
                    parent_task_id=completed_task.id,
                )
                self.task_queue.add_task(correlate_task)

    def _should_continue_research(self) -> bool:
        """Rozhoduje, zda pokračovat ve výzkumu"""
        stats = self.task_queue.get_stats()

        # Pokračuj pokud jsou úkoly k vykonání
        if stats["pending"] > 0 or stats["running"] > 0:
            return True

        # Pokračuj pokud byly nedávno generovány kvalitní výsledky
        recent_quality = [
            score
            for score in self.credibility_scores[-10:]
            if score >= self.min_credibility_threshold
        ]

        return len(recent_quality) > 2

    async def _cross_validate_claim(self, claim: str, sources: list[str]) -> float:
        """Křížová validace tvrzení napříč zdroji"""
        if not sources:
            return 0.0

        # Simulace křížové validace
        # V reálné implementaci by porovnávala tvrzení napříč zdroji
        return min(1.0, len(sources) * 0.3)

    async def _generate_final_report(self) -> dict[str, Any]:
        """Generuje finální report z výzkumného cyklu"""
        stats = self.task_queue.get_stats()

        # Nejlépe hodnocené výsledky
        top_findings = sorted(
            [task for task in self.task_queue.completed_tasks if task.result],
            key=lambda t: t.credibility_score,
            reverse=True,
        )[:10]

        avg_credibility = (
            sum(self.credibility_scores) / len(self.credibility_scores)
            if self.credibility_scores
            else 0.0
        )

        report = {
            "research_summary": {
                "iterations_completed": self.current_iteration,
                "total_tasks_executed": stats["completed"],
                "failed_tasks": stats["failed"],
                "average_credibility": avg_credibility,
                "top_findings_count": len(top_findings),
            },
            "key_discoveries": [
                {
                    "task_type": task.task_type.value,
                    "credibility_score": task.credibility_score,
                    "summary": self._summarize_task_result(task.result),
                }
                for task in top_findings
            ],
            "network_insights": await self._generate_network_insights(),
            "recommendations": self._generate_recommendations(top_findings),
            "metadata": {
                "completed_at": datetime.now().isoformat(),
                "total_runtime_minutes": self.current_iteration * 0.5,  # Estimate
            },
        }

        logger.info(
            f"🎉 Výzkumný cyklus dokončen: {stats['completed']} úkolů, průměrná důvěryhodnost: {avg_credibility:.2f}"
        )

        return report

    def _summarize_task_result(self, result: dict[str, Any]) -> str:
        """Vytvoří stručné shrnutí výsledku úkolu"""
        if not result:
            return "Žádný výsledek"

        summary_parts = []

        if "url" in result:
            summary_parts.append(f"URL: {result['url']}")
        if "patterns_found" in result:
            summary_parts.append(f"Vzory: {result['patterns_found']}")
        if "entities_found" in result:
            summary_parts.append(f"Entity: {result['entities_found']}")
        if "credibility_score" in result:
            summary_parts.append(f"Důvěryhodnost: {result['credibility_score']:.2f}")

        return "; ".join(summary_parts) if summary_parts else "Výsledek zpracován"

    async def _generate_network_insights(self) -> dict[str, Any]:
        """Generuje insights ze síťové analýzy"""
        # Zde by byla integrace s CorrelationEngine
        return {
            "total_entities": (
                len(self.correlation_engine.knowledge_graph.nodes)
                if hasattr(self.correlation_engine, "knowledge_graph")
                else 0
            ),
            "total_relationships": 0,
            "key_clusters": [],
            "anomalies_detected": 0,
        }

    def _generate_recommendations(self, top_findings: list[AgentTask]) -> list[str]:
        """Generuje doporučení na základě zjištění"""
        recommendations = []

        high_credibility_count = len(
            [task for task in top_findings if task.credibility_score >= 0.7]
        )

        if high_credibility_count >= 3:
            recommendations.append(
                "Nalezeny vysoce důvěryhodné zdroje - doporučujeme detailní manuální analýzu"
            )

        if len(top_findings) >= 5:
            recommendations.append("Dostatek dat pro statistickou analýzu trendů")

        low_credibility_count = len(
            [
                task
                for task in self.task_queue.completed_tasks
                if task.credibility_score < self.min_credibility_threshold
            ]
        )

        if low_credibility_count > len(top_findings):
            recommendations.append(
                "Vysoký podíl nedůvěryhodných zdrojů - nutná pečlivější filtrace"
            )

        return recommendations

    def stop_research_cycle(self) -> None:
        """Zastaví výzkumný cyklus"""
        logger.info("🛑 Zastavuji výzkumný cyklus")
        self.is_running = False

    def get_current_status(self) -> dict[str, Any]:
        """Vrátí aktuální stav agenta"""
        stats = self.task_queue.get_stats()

        return {
            "is_running": self.is_running,
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "task_queue_stats": stats,
            "average_credibility": (
                sum(self.credibility_scores) / len(self.credibility_scores)
                if self.credibility_scores
                else 0.0
            ),
            "total_discoveries": len(self.credibility_scores),
        }
