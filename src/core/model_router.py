#!/usr/bin/env python3
"""Model Router pro Kaskádové Modely - KLÍČOVÁ M1 OPTIMALIZACE
Inteligentní delegování úkolů mezi rychlým klasifikačním modelem (CPU) 
a velkým syntézním modelem (GPU) pro maximální efektivitu

Author: Senior Python/MLOps Agent
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
import logging
import time
from typing import Any

# Import centralizované konfigurace
from ..core.config import get_app_settings

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Typy úkolů pro model routing"""

    CLASSIFICATION = "classification"           # Rychlé filtrování/klasifikace
    RELEVANCE_CHECK = "relevance_check"        # Rychlé hodnocení relevance
    SIMPLE_QA = "simple_qa"                    # Jednoduché otázky

    SYNTHESIS = "synthesis"                    # Komplexní syntéza
    COMPLEX_REASONING = "complex_reasoning"    # Složité uvažování
    GENERATION = "generation"                  # Generování textu


class ModelTier(Enum):
    """Úrovně modelů podle složitosti"""

    FAST = "fast"           # Rychlý model pro jednoduché úkoly (CPU)
    STANDARD = "standard"   # Standardní model pro střední úkoly
    ADVANCED = "advanced"   # Pokročilý model pro složité úkoly (GPU)


@dataclass
class ModelConfig:
    """Konfigurace jednotlivého modelu"""

    name: str
    model_path: str
    device: str                    # cpu, mps, cuda
    max_tokens: int
    temperature: float

    # Performance charakteristiky
    avg_latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    memory_usage_mb: float = 0.0

    # Schopnosti
    supports_tasks: list[TaskType] = None
    quality_score: float = 0.5     # 0.0-1.0

    def __post_init__(self):
        if self.supports_tasks is None:
            self.supports_tasks = []


@dataclass
class RoutingDecision:
    """Rozhodnutí o routingu úkolu"""

    selected_model: str
    model_tier: ModelTier
    task_type: TaskType
    confidence: float
    reasoning: str
    expected_latency_ms: float
    fallback_model: str | None = None


class ModelRouter:
    """Inteligentní Router pro Kaskádové Modely
    
    KLÍČOVÁ OPTIMALIZACE:
    - Rychlé úkoly → Malý model (CPU) 
    - Složité úkoly → Velký model (GPU)
    - Dynamické rozhodování na základě úkolu a zátěže
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Inicializace model routeru
        
        Args:
            config: Konfigurační slovník (nebo načte z centralizované konfigurace)

        """
        # Načtení konfigurace
        if config is None:
            app_settings = get_app_settings()
            self.routing_config = app_settings.model_routing.dict()
        else:
            self.routing_config = config.get("model_routing", {})

        # Registrace dostupných modelů
        self.models: dict[str, ModelConfig] = {}
        self.task_model_mapping: dict[TaskType, list[str]] = {}

        # Performance tracking
        self.routing_stats = {
            "total_requests": 0,
            "fast_model_usage": 0,
            "advanced_model_usage": 0,
            "avg_routing_time_ms": 0.0,
            "model_performance": {}
        }

        # Inicializace modelů
        self._initialize_models()
        self._setup_task_mappings()

        logger.info(f"✅ Model Router inicializován s {len(self.models)} modely")

    def _initialize_models(self):
        """Inicializace dostupných modelů podle konfigurace"""
        # RYCHLÝ KLASIFIKAČNÍ MODEL (CPU) - pro filtrování a jednoduché úkoly
        fast_model = ModelConfig(
            name="classification",
            model_path=self.routing_config.get("classification_model", "sentence-transformers/all-MiniLM-L6-v2"),
            device=self.routing_config.get("classification_device", "cpu"),
            max_tokens=512,
            temperature=0.1,
            avg_latency_ms=50.0,     # Velmi rychlý
            tokens_per_second=200.0,
            memory_usage_mb=500.0,   # Nízká paměťová náročnost
            supports_tasks=[
                TaskType.CLASSIFICATION,
                TaskType.RELEVANCE_CHECK,
                TaskType.SIMPLE_QA
            ],
            quality_score=0.7        # Dobrá kvalita pro jednoduché úkoly
        )

        # POKROČILÝ SYNTÉZNÍ MODEL (GPU) - pro komplexní úkoly
        advanced_model = ModelConfig(
            name="synthesis",
            model_path=self.routing_config.get("synthesis_model", "models/llama-2-7b-chat.Q4_K_M.gguf"),
            device=self.routing_config.get("synthesis_device", "mps"),
            max_tokens=4096,
            temperature=0.3,
            avg_latency_ms=2000.0,   # Pomalejší ale přesnější
            tokens_per_second=25.0,
            memory_usage_mb=4096.0,  # Vyšší paměťová náročnost
            supports_tasks=[
                TaskType.SYNTHESIS,
                TaskType.COMPLEX_REASONING,
                TaskType.GENERATION,
                # Může také jednoduché úkoly (fallback)
                TaskType.CLASSIFICATION,
                TaskType.RELEVANCE_CHECK,
                TaskType.SIMPLE_QA
            ],
            quality_score=0.95       # Vysoká kvalita pro všechny úkoly
        )

        # Registrace modelů
        self.models["fast"] = fast_model
        self.models["advanced"] = advanced_model

        logger.info(f"📱 Rychlý model: {fast_model.name} ({fast_model.device})")
        logger.info(f"🧠 Pokročilý model: {advanced_model.name} ({advanced_model.device})")

    def _setup_task_mappings(self):
        """Nastavení mapování úkolů na modely"""
        # Mapování: které modely mohou řešit které úkoly
        for task_type in TaskType:
            self.task_model_mapping[task_type] = []

            for model_name, model_config in self.models.items():
                if task_type in model_config.supports_tasks:
                    self.task_model_mapping[task_type].append(model_name)

        logger.debug(f"🗺️ Task mappings: {dict(self.task_model_mapping)}")

    def get_model(self, task_type: str | TaskType,
                  context: dict[str, Any] | None = None,
                  priority: str = "balanced") -> RoutingDecision:
        """HLAVNÍ METODA: Výběr optimálního modelu pro úkol
        
        Args:
            task_type: Typ úkolu
            context: Kontext úkolu (délka textu, složitost, atd.)
            priority: Priorita (speed, quality, balanced)
            
        Returns:
            Rozhodnutí o routingu s vybraným modelem

        """
        start_time = time.time()

        # Normalizace task_type
        if isinstance(task_type, str):
            try:
                task_type = TaskType(task_type)
            except ValueError:
                logger.warning(f"Neznámý typ úkolu: {task_type}, použiju classification")
                task_type = TaskType.CLASSIFICATION

        # Získání kandidátních modelů
        candidate_models = self.task_model_mapping.get(task_type, [])

        if not candidate_models:
            logger.warning(f"Žádné modely pro úkol {task_type}, použiju pokročilý model")
            decision = RoutingDecision(
                selected_model="advanced",
                model_tier=ModelTier.ADVANCED,
                task_type=task_type,
                confidence=0.5,
                reasoning="Fallback - žádný specializovaný model",
                expected_latency_ms=self.models["advanced"].avg_latency_ms
            )
        else:
            decision = self._make_routing_decision(task_type, candidate_models, context, priority)

        # Performance tracking
        routing_time = (time.time() - start_time) * 1000
        self._update_routing_stats(decision, routing_time)

        logger.debug(f"🎯 Routing decision: {task_type.value} → {decision.selected_model} "
                    f"({decision.expected_latency_ms:.0f}ms)")

        return decision

    def _make_routing_decision(self,
                              task_type: TaskType,
                              candidate_models: list[str],
                              context: dict[str, Any] | None,
                              priority: str) -> RoutingDecision:
        """Inteligentní rozhodování o výběru modelu
        
        KLÍČOVÁ LOGIKA OPTIMALIZACE:
        - Jednoduché úkoly → rychlý model (CPU)
        - Složité úkoly → pokročilý model (GPU)
        - Kontextové rozhodování na základě délky textu, složitosti
        """
        # Analýza kontextu
        text_length = context.get("text_length", 0) if context else 0
        complexity_score = context.get("complexity_score", 0.5) if context else 0.5
        user_priority = context.get("user_priority", priority) if context else priority

        # ROZHODOVACÍ LOGIKA

        # 1. Jednoduché klasifikační úkoly → rychlý model
        if task_type in [TaskType.CLASSIFICATION, TaskType.RELEVANCE_CHECK]:
            if text_length < 1000 and complexity_score < 0.3:
                return self._create_decision("fast", task_type,
                                            "Jednoduchý klasifikační úkol → rychlý model",
                                            confidence=0.9)

        # 2. Priorita rychlosti → rychlý model (pokud podporuje úkol)
        if user_priority == "speed" and "fast" in candidate_models:
            return self._create_decision("fast", task_type,
                                       "Priorita rychlosti → rychlý model",
                                       confidence=0.8)

        # 3. Priorita kvality → pokročilý model
        if user_priority == "quality" and "advanced" in candidate_models:
            return self._create_decision("advanced", task_type,
                                       "Priorita kvality → pokročilý model",
                                       confidence=0.9)

        # 4. Složité úkoly → pokročilý model
        if task_type in [TaskType.SYNTHESIS, TaskType.COMPLEX_REASONING, TaskType.GENERATION]:
            return self._create_decision("advanced", task_type,
                                       "Složitý úkol → pokročilý model",
                                       confidence=0.9)

        # 5. Dlouhý text nebo vysoká složitost → pokročilý model
        if text_length > 2000 or complexity_score > 0.7:
            return self._create_decision("advanced", task_type,
                                       f"Dlouhý text ({text_length}) nebo vysoká složitost → pokročilý model",
                                       confidence=0.8)

        # 6. Balanced priorita - dynamické rozhodování
        if user_priority == "balanced":
            # Skóre pro rychlý vs pokročilý model
            fast_score = self._calculate_model_score("fast", task_type, context, weight_speed=0.7)
            advanced_score = self._calculate_model_score("advanced", task_type, context, weight_speed=0.3)

            if fast_score >= advanced_score and "fast" in candidate_models:
                return self._create_decision("fast", task_type,
                                           f"Balanced: rychlý model optimální (skóre: {fast_score:.2f})",
                                           confidence=0.7)
            return self._create_decision("advanced", task_type,
                                       f"Balanced: pokročilý model optimální (skóre: {advanced_score:.2f})",
                                       confidence=0.7)

        # 7. Fallback - preferuj rychlý model pokud možno
        if "fast" in candidate_models:
            return self._create_decision("fast", task_type,
                                       "Fallback → rychlý model",
                                       confidence=0.6)
        return self._create_decision("advanced", task_type,
                                   "Fallback → pokročilý model",
                                   confidence=0.6)

    def _calculate_model_score(self,
                              model_name: str,
                              task_type: TaskType,
                              context: dict[str, Any] | None,
                              weight_speed: float = 0.5) -> float:
        """Výpočet skóre modelu pro daný úkol
        
        Args:
            model_name: Název modelu
            task_type: Typ úkolu
            context: Kontext úkolu
            weight_speed: Váha rychlosti vs kvality (0.0-1.0)
            
        Returns:
            Skóre modelu (0.0-1.0)

        """
        model_config = self.models.get(model_name)
        if not model_config:
            return 0.0

        # Normalizované metriky
        speed_score = 1.0 - (model_config.avg_latency_ms / 5000.0)  # Normalizace na 5s max
        speed_score = max(0.0, min(1.0, speed_score))

        quality_score = model_config.quality_score

        # Podpora úkolu
        task_support_score = 1.0 if task_type in model_config.supports_tasks else 0.0

        # Kontextové faktory
        context_score = 1.0
        if context:
            text_length = context.get("text_length", 0)
            # Rychlý model je horší pro dlouhé texty
            if model_name == "fast" and text_length > 1500:
                context_score *= 0.5

        # Vážená kombinace
        weight_quality = 1.0 - weight_speed
        final_score = (speed_score * weight_speed +
                      quality_score * weight_quality +
                      task_support_score * 0.3 +
                      context_score * 0.2) / 2.5

        return max(0.0, min(1.0, final_score))

    def _create_decision(self,
                        model_name: str,
                        task_type: TaskType,
                        reasoning: str,
                        confidence: float) -> RoutingDecision:
        """Vytvoření routing decision objektu"""
        model_config = self.models[model_name]

        # Určení tier
        if model_name == "fast":
            tier = ModelTier.FAST
        elif model_name == "advanced":
            tier = ModelTier.ADVANCED
        else:
            tier = ModelTier.STANDARD

        # Fallback model
        fallback = None
        if model_name == "fast":
            fallback = "advanced"  # Pokud rychlý model selže, použij pokročilý

        return RoutingDecision(
            selected_model=model_name,
            model_tier=tier,
            task_type=task_type,
            confidence=confidence,
            reasoning=reasoning,
            expected_latency_ms=model_config.avg_latency_ms,
            fallback_model=fallback
        )

    def _update_routing_stats(self, decision: RoutingDecision, routing_time_ms: float):
        """Aktualizace statistik routingu"""
        self.routing_stats["total_requests"] += 1

        if decision.model_tier == ModelTier.FAST:
            self.routing_stats["fast_model_usage"] += 1
        elif decision.model_tier == ModelTier.ADVANCED:
            self.routing_stats["advanced_model_usage"] += 1

        # Průměrný čas routingu
        total_requests = self.routing_stats["total_requests"]
        current_avg = self.routing_stats["avg_routing_time_ms"]
        self.routing_stats["avg_routing_time_ms"] = (
            (current_avg * (total_requests - 1) + routing_time_ms) / total_requests
        )

    def get_routing_stats(self) -> dict[str, Any]:
        """Získání statistik routingu"""
        total_requests = self.routing_stats["total_requests"]

        if total_requests == 0:
            return {"message": "Žádné požadavky zatím zpracovány"}

        fast_usage_percent = (self.routing_stats["fast_model_usage"] / total_requests) * 100
        advanced_usage_percent = (self.routing_stats["advanced_model_usage"] / total_requests) * 100

        # Odhad úspory výkonu
        avg_fast_latency = self.models["fast"].avg_latency_ms
        avg_advanced_latency = self.models["advanced"].avg_latency_ms

        # Kdyby všechny úkoly běžely na pokročilém modelu
        total_latency_if_all_advanced = total_requests * avg_advanced_latency

        # Skutečná latence s routingem
        actual_total_latency = (
            self.routing_stats["fast_model_usage"] * avg_fast_latency +
            self.routing_stats["advanced_model_usage"] * avg_advanced_latency
        )

        performance_improvement = ((total_latency_if_all_advanced - actual_total_latency) /
                                 max(1, total_latency_if_all_advanced)) * 100

        return {
            "total_requests": total_requests,
            "model_usage": {
                "fast_model_percent": round(fast_usage_percent, 1),
                "advanced_model_percent": round(advanced_usage_percent, 1)
            },
            "performance_metrics": {
                "avg_routing_time_ms": round(self.routing_stats["avg_routing_time_ms"], 2),
                "estimated_performance_improvement_percent": round(performance_improvement, 1),
                "total_latency_saved_ms": round(total_latency_if_all_advanced - actual_total_latency, 0)
            },
            "model_configurations": {
                model_name: {
                    "device": config.device,
                    "avg_latency_ms": config.avg_latency_ms,
                    "quality_score": config.quality_score,
                    "memory_usage_mb": config.memory_usage_mb
                }
                for model_name, config in self.models.items()
            }
        }

    async def test_model_performance(self, model_name: str, sample_tasks: list[dict[str, Any]]) -> dict[str, Any]:
        """Test výkonu modelu na vzorových úkolech
        
        Args:
            model_name: Název modelu k testování
            sample_tasks: Seznam vzorových úkolů
            
        Returns:
            Performance metriky

        """
        if model_name not in self.models:
            return {"error": f"Model {model_name} není dostupný"}

        model_config = self.models[model_name]

        logger.info(f"🧪 Testuji výkon modelu: {model_name}")

        latencies = []
        start_time = time.time()

        for i, task in enumerate(sample_tasks):
            task_start = time.time()

            # Simulace úkolu (v reálné implementaci by se volal skutečný model)
            await asyncio.sleep(model_config.avg_latency_ms / 1000.0)

            task_latency = (time.time() - task_start) * 1000
            latencies.append(task_latency)

            logger.debug(f"Task {i+1}/{len(sample_tasks)}: {task_latency:.1f}ms")

        total_time = (time.time() - start_time) * 1000

        # Aktualizace model config s naměřenými hodnotami
        measured_avg_latency = sum(latencies) / len(latencies)
        self.models[model_name].avg_latency_ms = measured_avg_latency

        return {
            "model_name": model_name,
            "tasks_tested": len(sample_tasks),
            "total_time_ms": round(total_time, 1),
            "average_latency_ms": round(measured_avg_latency, 1),
            "min_latency_ms": round(min(latencies), 1),
            "max_latency_ms": round(max(latencies), 1),
            "throughput_tasks_per_second": round(len(sample_tasks) / (total_time / 1000), 2),
            "updated_config": True
        }


# Globální instance routeru (singleton pattern)
_model_router: ModelRouter | None = None


def get_model_router(config: dict[str, Any] | None = None) -> ModelRouter:
    """Získání globální instance model routeru
    
    Args:
        config: Konfigurační slovník (optional)
        
    Returns:
        ModelRouter instance

    """
    global _model_router

    if _model_router is None:
        _model_router = ModelRouter(config)

    return _model_router


def reset_model_router():
    """Reset globální instance (pro testování)"""
    global _model_router
    _model_router = None


# Convenience funkce pro snadné použití
def route_task(task_type: str,
               context: dict[str, Any] | None = None,
               priority: str = "balanced") -> RoutingDecision:
    """Rychlá funkce pro routing úkolu
    
    Args:
        task_type: Typ úkolu (classification, synthesis, atd.)
        context: Kontext úkolu
        priority: Priorita (speed, quality, balanced)
        
    Returns:
        Routing decision

    """
    router = get_model_router()
    return router.get_model(task_type, context, priority)


if __name__ == "__main__":
    # Testování functionality
    import asyncio

    async def test_model_router():
        """Test základní funkcionality"""
        router = ModelRouter()

        # Test různých typů úkolů
        test_cases = [
            {"task": "classification", "context": {"text_length": 200}, "priority": "speed"},
            {"task": "relevance_check", "context": {"text_length": 500}, "priority": "balanced"},
            {"task": "synthesis", "context": {"text_length": 1500}, "priority": "quality"},
            {"task": "complex_reasoning", "context": {"complexity_score": 0.8}, "priority": "quality"},
        ]

        for case in test_cases:
            decision = router.get_model(
                case["task"],
                case.get("context"),
                case.get("priority", "balanced")
            )

            print(f"Task: {case['task']}")
            print(f"  → Model: {decision.selected_model}")
            print(f"  → Tier: {decision.model_tier.value}")
            print(f"  → Reasoning: {decision.reasoning}")
            print(f"  → Expected latency: {decision.expected_latency_ms:.0f}ms")
            print()

        # Zobrazení statistik
        stats = router.get_routing_stats()
        print("📊 Routing Statistics:")
        print(f"  Fast model usage: {stats['model_usage']['fast_model_percent']:.1f}%")
        print(f"  Advanced model usage: {stats['model_usage']['advanced_model_percent']:.1f}%")
        print(f"  Performance improvement: {stats['performance_metrics']['estimated_performance_improvement_percent']:.1f}%")

    # Spuštění testu
    asyncio.run(test_model_router())
