"""
🎯 Pokročilý Task Manager pro autonomní agenta
Inteligentní plánování a optimalizace úkolů na základě důvěryhodnosti
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import heapq
import networkx as nx
from collections import defaultdict, deque

from ..synthesis.credibility_assessor import CredibilityAssessor
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class TaskExecutionStrategy(Enum):
    """Strategie vykonávání úkolů"""

    BREADTH_FIRST = "breadth_first"  # Nejprv šířka
    DEPTH_FIRST = "depth_first"  # Nejprv hloubka
    CREDIBILITY_FIRST = "credibility_first"  # Nejprv nejvěrohodnější
    BALANCED = "balanced"  # Vyvážená strategie


@dataclass
class TaskDependency:
    """Závislost mezi úkoly"""

    task_id: str
    depends_on: str
    dependency_type: str  # "requires", "enhances", "blocks"
    strength: float = 1.0


class IntelligentTaskManager:
    """
    🧠 Inteligentní správce úkolů s pokročilým plánováním

    Optimalizuje pořadí úkolů na základě:
    - Důvěryhodnosti zdrojů
    - Závislostí mezi úkoly
    - Očekávané hodnoty výsledků
    - Dostupných zdrojů
    """

    def __init__(self, credibility_assessor: CredibilityAssessor):
        self.credibility_assessor = credibility_assessor
        self.task_graph = nx.DiGraph()
        self.execution_history: List[Dict] = []
        self.resource_usage = defaultdict(float)
        self.strategy = TaskExecutionStrategy.BALANCED

        # Metriky pro optimalizaci
        self.success_rates: Dict[str, float] = defaultdict(lambda: 0.5)
        self.execution_times: Dict[str, float] = defaultdict(lambda: 30.0)
        self.resource_costs: Dict[str, float] = defaultdict(lambda: 1.0)

    async def add_task_with_intelligence(
        self, task: "AgentTask", related_tasks: List["AgentTask"] = None
    ) -> None:
        """Přidá úkol s inteligentní analýzou závislostí"""

        # Analýza potenciální hodnoty úkolu
        task_value = await self._estimate_task_value(task)

        # Detekce závislostí
        dependencies = await self._detect_dependencies(task, related_tasks or [])

        # Přidání do grafu úkolů
        self.task_graph.add_node(
            task.id,
            task=task,
            value=task_value,
            estimated_time=self.execution_times[task.task_type.value],
            estimated_cost=self.resource_costs[task.task_type.value],
        )

        # Přidání závislostí
        for dep in dependencies:
            self.task_graph.add_edge(dep.depends_on, task.id, dependency=dep)

        logger.info(f"Úkol {task.id} přidán s hodnotou {task_value:.2f}")

    async def _estimate_task_value(self, task: "AgentTask") -> float:
        """Odhadne potenciální hodnotu úkolu"""
        base_value = 1.0

        # Bonus za vysokou důvěryhodnost zdroje
        if hasattr(task, "credibility_score") and task.credibility_score > 0:
            base_value *= 1 + task.credibility_score

        # Bonus za typ úkolu
        type_bonuses = {
            "scrape": 1.2,  # Nová data
            "analyze": 1.5,  # Analýza je hodnotná
            "correlate": 1.8,  # Korelace jsou velmi cenné
            "validate": 1.3,  # Validace je důležitá
            "expand_search": 1.1,
            "deep_dive": 2.0,  # Hloubková analýza nejvíce
            "cross_reference": 1.6,
        }
        base_value *= type_bonuses.get(task.task_type.value, 1.0)

        # Penalizace za opakované úkoly
        similar_completed = len(
            [
                h
                for h in self.execution_history
                if h.get("task_type") == task.task_type.value
                and h.get("parameters") == task.parameters
            ]
        )
        if similar_completed > 0:
            base_value *= 0.8**similar_completed

        # Bonus za vysokou prioritu
        priority_bonuses = {1: 2.0, 2: 1.5, 3: 1.0, 4: 0.7}
        base_value *= priority_bonuses.get(task.priority.value, 1.0)

        return min(base_value, 5.0)  # Cap na 5.0

    async def _detect_dependencies(
        self, task: "AgentTask", related_tasks: List["AgentTask"]
    ) -> List[TaskDependency]:
        """Detekuje závislosti mezi úkoly"""
        dependencies = []

        for related_task in related_tasks:
            if related_task.id == task.id:
                continue

            # Silné závislosti
            if self._has_strong_dependency(task, related_task):
                dependencies.append(
                    TaskDependency(
                        task_id=task.id,
                        depends_on=related_task.id,
                        dependency_type="requires",
                        strength=1.0,
                    )
                )

            # Slabé závislosti (enhancement)
            elif self._has_weak_dependency(task, related_task):
                dependencies.append(
                    TaskDependency(
                        task_id=task.id,
                        depends_on=related_task.id,
                        dependency_type="enhances",
                        strength=0.5,
                    )
                )

        return dependencies

    def _has_strong_dependency(self, task: "AgentTask", other_task: "AgentTask") -> bool:
        """Kontroluje silnou závislost mezi úkoly"""

        # ANALYZE úkoly vyžadují SCRAPE úkoly pro stejný zdroj
        if task.task_type.value == "analyze" and other_task.task_type.value == "scrape":

            task_url = task.parameters.get("url", "")
            other_url = other_task.parameters.get("url", "")
            if task_url and task_url == other_url:
                return True

        # CORRELATE úkoly vyžadují ANALYZE úkoly
        if task.task_type.value == "correlate" and other_task.task_type.value == "analyze":
            return True

        # VALIDATE úkoly vyžadují nějaké předchozí findings
        if task.task_type.value == "validate" and other_task.task_type.value in [
            "scrape",
            "analyze",
        ]:
            return True

        return False

    def _has_weak_dependency(self, task: "AgentTask", other_task: "AgentTask") -> bool:
        """Kontroluje slabou závislost (enhancement)"""

        # DEEP_DIVE úkoly jsou vylepšeny předchozími analýzami
        if task.task_type.value == "deep_dive" and other_task.task_type.value in [
            "analyze",
            "correlate",
        ]:
            return True

        # CROSS_REFERENCE úkoly jsou vylepšeny více zdroji
        if task.task_type.value == "cross_reference" and other_task.task_type.value == "scrape":
            return True

        return False

    def get_optimal_execution_order(self, available_tasks: List["AgentTask"]) -> List["AgentTask"]:
        """Vrátí optimální pořadí vykonávání úkolů"""

        if self.strategy == TaskExecutionStrategy.CREDIBILITY_FIRST:
            return self._credibility_first_order(available_tasks)
        elif self.strategy == TaskExecutionStrategy.DEPTH_FIRST:
            return self._depth_first_order(available_tasks)
        elif self.strategy == TaskExecutionStrategy.BREADTH_FIRST:
            return self._breadth_first_order(available_tasks)
        else:  # BALANCED
            return self._balanced_order(available_tasks)

    def _credibility_first_order(self, tasks: List["AgentTask"]) -> List["AgentTask"]:
        """Řadí úkoly podle důvěryhodnosti"""
        return sorted(tasks, key=lambda t: (t.credibility_score, t.priority.value), reverse=True)

    def _depth_first_order(self, tasks: List["AgentTask"]) -> List["AgentTask"]:
        """Řadí úkoly pro hloubkový průchod"""
        # Topologické řazení grafu úkolů
        try:
            task_ids = [t.id for t in tasks]
            subgraph = self.task_graph.subgraph(task_ids)
            topo_order = list(nx.topological_sort(subgraph))

            # Mapování zpět na úkoly
            task_map = {t.id: t for t in tasks}
            return [task_map[tid] for tid in topo_order if tid in task_map]
        except nx.NetworkXError:
            # Fallback na prioritu
            return sorted(tasks, key=lambda t: t.priority.value)

    def _breadth_first_order(self, tasks: List["AgentTask"]) -> List["AgentTask"]:
        """Řadí úkoly pro šířkový průchod"""
        # Grupování podle typu a pak podle priority
        grouped = defaultdict(list)
        for task in tasks:
            grouped[task.task_type.value].append(task)

        result = []
        for task_type, type_tasks in grouped.items():
            type_tasks.sort(key=lambda t: t.priority.value)
            result.extend(type_tasks)

        return result

    def _balanced_order(self, tasks: List["AgentTask"]) -> List["AgentTask"]:
        """Vyvážené řazení kombinující více faktorů"""

        def task_score(task: "AgentTask") -> float:
            # Kombinovaný score
            score = 0.0

            # Hodnota úkolu (30%)
            if task.id in self.task_graph:
                task_value = self.task_graph.nodes[task.id].get("value", 1.0)
                score += task_value * 0.3

            # Priorita (25%)
            priority_score = (5 - task.priority.value) / 4  # Inverted
            score += priority_score * 0.25

            # Důvěryhodnost (25%)
            score += task.credibility_score * 0.25

            # Success rate pro tento typ úkolu (20%)
            success_rate = self.success_rates[task.task_type.value]
            score += success_rate * 0.20

            return score

        return sorted(tasks, key=task_score, reverse=True)

    async def optimize_resource_allocation(self, tasks: List["AgentTask"]) -> Dict[str, Any]:
        """Optimalizuje alokaci zdrojů pro úkoly"""

        total_estimated_time = sum(self.execution_times[task.task_type.value] for task in tasks)

        total_estimated_cost = sum(self.resource_costs[task.task_type.value] for task in tasks)

        # Analýza bottlenecků
        bottlenecks = self._identify_bottlenecks(tasks)

        # Doporučení pro optimalizaci
        recommendations = []

        if total_estimated_time > 300:  # 5 minut
            recommendations.append("Zvážit paralelizaci úkolů")

        if total_estimated_cost > 10:
            recommendations.append("Vysoké náklady - prioritizovat důležité úkoly")

        if bottlenecks:
            recommendations.append(f"Bottlenecky v typech: {', '.join(bottlenecks)}")

        return {
            "estimated_total_time": total_estimated_time,
            "estimated_total_cost": total_estimated_cost,
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "optimal_parallelism": min(len(tasks), 5),
        }

    def _identify_bottlenecks(self, tasks: List["AgentTask"]) -> List[str]:
        """Identifikuje bottlenecky v typech úkolů"""

        type_counts = defaultdict(int)
        for task in tasks:
            type_counts[task.task_type.value] += 1

        # Bottleneck = více než 3 úkoly stejného typu
        bottlenecks = [task_type for task_type, count in type_counts.items() if count > 3]

        return bottlenecks

    def record_execution_result(
        self, task_id: str, success: bool, execution_time: float, result: Dict[str, Any]
    ) -> None:
        """Zaznamenává výsledek vykonání úkolu pro učení"""

        if task_id not in self.task_graph:
            return

        task_data = self.task_graph.nodes[task_id]
        task_type = task_data["task"].task_type.value

        # Aktualizace metrik
        old_success_rate = self.success_rates[task_type]
        self.success_rates[task_type] = old_success_rate * 0.8 + (1.0 if success else 0.0) * 0.2

        old_exec_time = self.execution_times[task_type]
        self.execution_times[task_type] = old_exec_time * 0.8 + execution_time * 0.2

        # Záznam do historie
        self.execution_history.append(
            {
                "task_id": task_id,
                "task_type": task_type,
                "success": success,
                "execution_time": execution_time,
                "timestamp": datetime.now(),
                "result_quality": self._assess_result_quality(result),
            }
        )

        # Omezení historie na posledních 1000 záznamů
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]

    def _assess_result_quality(self, result: Dict[str, Any]) -> float:
        """Posuzuje kvalitu výsledku úkolu"""
        if not result:
            return 0.0

        quality = 0.5  # Base quality

        # Bonus za důvěryhodnost
        if "credibility_score" in result:
            quality += result["credibility_score"] * 0.3

        # Bonus za množství dat
        if "entities_found" in result and result["entities_found"] > 0:
            quality += min(result["entities_found"] * 0.05, 0.2)

        if "patterns_found" in result and result["patterns_found"] > 0:
            quality += min(result["patterns_found"] * 0.1, 0.2)

        return min(quality, 1.0)

    def get_learning_insights(self) -> Dict[str, Any]:
        """Vrátí insights z učení agenta"""

        if not self.execution_history:
            return {"message": "Nedostatek dat pro analýzu"}

        # Analýza úspěšnosti podle typu
        type_stats = defaultdict(lambda: {"success": 0, "total": 0, "avg_time": 0})

        for record in self.execution_history[-100:]:  # Posledních 100
            task_type = record["task_type"]
            type_stats[task_type]["total"] += 1
            if record["success"]:
                type_stats[task_type]["success"] += 1
            type_stats[task_type]["avg_time"] += record["execution_time"]

        # Výpočet průměrů
        for task_type, stats in type_stats.items():
            if stats["total"] > 0:
                stats["success_rate"] = stats["success"] / stats["total"]
                stats["avg_time"] /= stats["total"]

        # Nejlepší a nejhorší typy úkolů
        best_type = (
            max(type_stats.items(), key=lambda x: x[1]["success_rate"])[0] if type_stats else None
        )
        worst_type = (
            min(type_stats.items(), key=lambda x: x[1]["success_rate"])[0] if type_stats else None
        )

        return {
            "total_executions": len(self.execution_history),
            "type_statistics": dict(type_stats),
            "best_performing_type": best_type,
            "worst_performing_type": worst_type,
            "overall_success_rate": sum(r["success"] for r in self.execution_history[-100:])
            / min(100, len(self.execution_history)),
            "current_strategy": self.strategy.value,
        }

    def adapt_strategy(self) -> None:
        """Adaptuje strategii na základě výsledků"""
        insights = self.get_learning_insights()

        if "overall_success_rate" not in insights:
            return

        success_rate = insights["overall_success_rate"]

        # Adaptace strategie podle úspěšnosti
        if success_rate < 0.5:
            self.strategy = TaskExecutionStrategy.CREDIBILITY_FIRST
            logger.info("Přepínám na CREDIBILITY_FIRST strategii - nízká úspěšnost")
        elif success_rate > 0.8:
            self.strategy = TaskExecutionStrategy.BALANCED
            logger.info("Přepínám na BALANCED strategii - vysoká úspěšnost")

        logger.info(f"Aktuální strategie: {self.strategy.value}, úspěšnost: {success_rate:.2f}")


class TaskExecutionMonitor:
    """
    📊 Monitor vykonávání úkolů s real-time metrikami
    """

    def __init__(self):
        self.active_tasks: Dict[str, Dict] = {}
        self.completion_times: List[float] = []
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.resource_usage_history: List[Dict] = []

    def start_task_monitoring(self, task_id: str, task_type: str) -> None:
        """Začne monitorovat úkol"""
        self.active_tasks[task_id] = {
            "task_type": task_type,
            "start_time": datetime.now(),
            "status": "running",
        }

    def complete_task_monitoring(self, task_id: str, success: bool, result_size: int = 0) -> None:
        """Dokončí monitorování úkolu"""
        if task_id not in self.active_tasks:
            return

        task_info = self.active_tasks[task_id]
        execution_time = (datetime.now() - task_info["start_time"]).total_seconds()

        self.completion_times.append(execution_time)

        if not success:
            self.error_counts[task_info["task_type"]] += 1

        # Záznam využití zdrojů
        self.resource_usage_history.append(
            {
                "timestamp": datetime.now(),
                "task_type": task_info["task_type"],
                "execution_time": execution_time,
                "success": success,
                "result_size": result_size,
            }
        )

        # Odstranění z aktivních
        del self.active_tasks[task_id]

        # Omezení historie
        if len(self.resource_usage_history) > 500:
            self.resource_usage_history = self.resource_usage_history[-500:]

    def get_current_metrics(self) -> Dict[str, Any]:
        """Vrátí aktuální metriky"""
        now = datetime.now()

        # Průměrný čas dokončení
        avg_completion_time = (
            sum(self.completion_times[-50:]) / len(self.completion_times[-50:])
            if self.completion_times
            else 0
        )

        # Aktivní úkoly
        running_tasks = len(self.active_tasks)

        # Chybovost podle typu
        recent_errors = {}
        for task_type, error_count in self.error_counts.items():
            recent_errors[task_type] = error_count

        # Throughput (úkoly za minutu)
        recent_completions = [
            r for r in self.resource_usage_history if (now - r["timestamp"]).total_seconds() < 60
        ]
        throughput = len(recent_completions)

        return {
            "active_tasks": running_tasks,
            "avg_completion_time": avg_completion_time,
            "throughput_per_minute": throughput,
            "error_counts_by_type": recent_errors,
            "total_completed_tasks": len(self.completion_times),
            "system_load": min(running_tasks / 10, 1.0),  # Normalized load
        }
