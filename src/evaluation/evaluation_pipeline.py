"""
Automatizovaná evaluační pipeline pro Research Agent
Implementuje RAG Triad metriky a Golden Dataset testing
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

import pandas as pd
from pydantic import BaseModel

from src.observability.langfuse_integration import get_observability_manager

logger = logging.getLogger(__name__)


@dataclass
class GoldenDatasetItem:
    """Jednotka Golden Datasetu"""
    id: str
    query: str
    expected_answer: str
    relevant_sources: List[str]
    category: str
    difficulty: str  # "easy", "medium", "hard"
    metadata: Dict[str, Any]


@dataclass
class EvaluationResult:
    """Výsledek evaluace jedné otázky"""
    query_id: str
    query: str
    generated_answer: str
    expected_answer: str

    # RAG Triad metriky
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevance: float
    answer_correctness: float

    # Dodatečné metriky
    latency_ms: float
    token_count: int
    retrieved_sources: List[str]
    execution_error: Optional[str] = None


class RAGEvaluator:
    """Implementuje RAG Triad metriky pomocí LLM-as-a-Judge"""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    async def evaluate_context_precision(self,
                                       query: str,
                                       retrieved_contexts: List[str]) -> float:
        """
        Context Precision: Podíl relevantních dokumentů z načtených dokumentů
        """
        if not retrieved_contexts:
            return 0.0

        prompt = f"""
        Evaluuj relevantnost každého kontextu pro danou otázku.
        
        Otázka: {query}
        
        Kontexty:
        {chr(10).join([f"{i+1}. {ctx[:500]}" for i, ctx in enumerate(retrieved_contexts)])}
        
        Pro každý kontext odpověz pouze "relevant" nebo "not_relevant":
        """

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=100
            )

            relevant_count = response.lower().count("relevant") - response.lower().count("not_relevant")
            return max(0, relevant_count) / len(retrieved_contexts)

        except Exception as e:
            logger.error(f"Error in context precision evaluation: {e}")
            return 0.0

    async def evaluate_context_recall(self,
                                    query: str,
                                    retrieved_contexts: List[str],
                                    expected_sources: List[str]) -> float:
        """
        Context Recall: Podíl očekávaných zdrojů, které byly načteny
        """
        if not expected_sources:
            return 1.0  # Pokud nejsou očekávané zdroje, považujeme za úspěch

        prompt = f"""
        Zkontroluj, zda načtené kontexty pokrývají informace z očekávaných zdrojů.
        
        Otázka: {query}
        
        Očekávané zdroje: {', '.join(expected_sources)}
        
        Načtené kontexty:
        {chr(10).join([f"{i+1}. {ctx[:300]}" for i, ctx in enumerate(retrieved_contexts)])}
        
        Kolik z očekávaných zdrojů je pokryto načtenými kontexty? 
        Odpověz pouze číslem od 0 do {len(expected_sources)}:
        """

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=10
            )

            covered_count = int(''.join(filter(str.isdigit, response)) or "0")
            return min(covered_count, len(expected_sources)) / len(expected_sources)

        except Exception as e:
            logger.error(f"Error in context recall evaluation: {e}")
            return 0.0

    async def evaluate_faithfulness(self,
                                  answer: str,
                                  contexts: List[str]) -> float:
        """
        Faithfulness: Míra, do jaké je odpověď věrná poskytnutým zdrojům
        """
        prompt = f"""
        Evaluuj, zda je odpověď věrná poskytnutým kontextům.
        
        Kontexty:
        {chr(10).join([f"{i+1}. {ctx[:400]}" for i, ctx in enumerate(contexts)])}
        
        Odpověď: {answer}
        
        Je odpověď plně podporována kontexty? Skóre od 0.0 do 1.0:
        0.0 = Odpověď zcela protichází kontextům
        0.5 = Odpověď je částečně podporována
        1.0 = Odpověď je plně podporována kontexty
        
        Odpověz pouze číslem:
        """

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=10
            )

            score = float(''.join(c for c in response if c.isdigit() or c == '.') or "0")
            return min(max(score, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Error in faithfulness evaluation: {e}")
            return 0.0

    async def evaluate_answer_relevance(self,
                                      query: str,
                                      answer: str) -> float:
        """
        Answer Relevance: Míra relevantnosti odpovědi k otázce
        """
        prompt = f"""
        Evaluuj relevantnost odpovědi k otázce.
        
        Otázka: {query}
        Odpověď: {answer}
        
        Skóre relevantnosti od 0.0 do 1.0:
        0.0 = Odpověď zcela nesouvisí s otázkou
        0.5 = Odpověď částečně odpovídá na otázku
        1.0 = Odpověď plně a přesně odpovídá na otázku
        
        Odpověz pouze číslem:
        """

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=10
            )

            score = float(''.join(c for c in response if c.isdigit() or c == '.') or "0")
            return min(max(score, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Error in answer relevance evaluation: {e}")
            return 0.0

    async def evaluate_answer_correctness(self,
                                        expected_answer: str,
                                        generated_answer: str) -> float:
        """
        Answer Correctness: Faktická správnost odpovědi
        """
        prompt = f"""
        Porovnej faktickou správnost vygenerované odpovědi s očekávanou odpovědí.
        
        Očekávaná odpověď: {expected_answer}
        Vygenerovaná odpověď: {generated_answer}
        
        Skóre faktické správnosti od 0.0 do 1.0:
        0.0 = Vygenerovaná odpověď je fakticky nesprávná
        0.5 = Vygenerovaná odpověď je částečně správná
        1.0 = Vygenerovaná odpověď je fakticky správná
        
        Odpověz pouze číslem:
        """

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=10
            )

            score = float(''.join(c for c in response if c.isdigit() or c == '.') or "0")
            return min(max(score, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Error in answer correctness evaluation: {e}")
            return 0.0


class EvaluationPipeline:
    """Hlavní třída pro automatizovanou evaluaci"""

    def __init__(self,
                 research_agent,
                 llm_client,
                 golden_dataset_path: str = "evaluation/golden_dataset.json"):
        self.research_agent = research_agent
        self.evaluator = RAGEvaluator(llm_client)
        self.golden_dataset_path = Path(golden_dataset_path)
        self.observability = get_observability_manager()

        self.golden_dataset = self._load_golden_dataset()

    def _load_golden_dataset(self) -> List[GoldenDatasetItem]:
        """Načte Golden Dataset"""
        if not self.golden_dataset_path.exists():
            logger.warning(f"Golden dataset not found at {self.golden_dataset_path}")
            return []

        with open(self.golden_dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return [GoldenDatasetItem(**item) for item in data]

    async def evaluate_single_query(self,
                                  golden_item: GoldenDatasetItem) -> EvaluationResult:
        """Evaluace jedné otázky z Golden Datasetu"""
        start_time = datetime.now()

        try:
            # Spuštění Research Agent
            result = await self.research_agent.run(
                query=golden_item.query,
                session_id=f"eval_{golden_item.id}"
            )

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Extrakce výsledků
            generated_answer = result.get("answer", "")
            retrieved_sources = result.get("sources", [])
            retrieved_contexts = [source.get("content", "") for source in retrieved_sources]

            # Výpočet metrik
            context_precision = await self.evaluator.evaluate_context_precision(
                golden_item.query, retrieved_contexts
            )

            context_recall = await self.evaluator.evaluate_context_recall(
                golden_item.query, retrieved_contexts, golden_item.relevant_sources
            )

            faithfulness = await self.evaluator.evaluate_faithfulness(
                generated_answer, retrieved_contexts
            )

            answer_relevance = await self.evaluator.evaluate_answer_relevance(
                golden_item.query, generated_answer
            )

            answer_correctness = await self.evaluator.evaluate_answer_correctness(
                golden_item.expected_answer, generated_answer
            )

            return EvaluationResult(
                query_id=golden_item.id,
                query=golden_item.query,
                generated_answer=generated_answer,
                expected_answer=golden_item.expected_answer,
                context_precision=context_precision,
                context_recall=context_recall,
                faithfulness=faithfulness,
                answer_relevance=answer_relevance,
                answer_correctness=answer_correctness,
                latency_ms=latency_ms,
                token_count=result.get("token_count", 0),
                retrieved_sources=[s.get("url", "") for s in retrieved_sources]
            )

        except Exception as e:
            logger.error(f"Error evaluating query {golden_item.id}: {e}")
            return EvaluationResult(
                query_id=golden_item.id,
                query=golden_item.query,
                generated_answer="",
                expected_answer=golden_item.expected_answer,
                context_precision=0.0,
                context_recall=0.0,
                faithfulness=0.0,
                answer_relevance=0.0,
                answer_correctness=0.0,
                latency_ms=0.0,
                token_count=0,
                retrieved_sources=[],
                execution_error=str(e)
            )

    async def run_full_evaluation(self) -> Dict[str, Any]:
        """Spuštění kompletní evaluace na celém Golden Datasetu"""
        if not self.golden_dataset:
            raise ValueError("Golden dataset is empty or not loaded")

        logger.info(f"Starting evaluation on {len(self.golden_dataset)} queries")

        results = []
        for item in self.golden_dataset:
            logger.info(f"Evaluating query: {item.id}")
            result = await self.evaluate_single_query(item)
            results.append(result)

        # Výpočet agregovaných metrik
        metrics = self._calculate_aggregate_metrics(results)

        # Uložení výsledků
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"evaluation/results_{timestamp}.json"
        self._save_results(results, metrics, results_path)

        return {
            "results": results,
            "metrics": metrics,
            "results_path": results_path
        }

    def _calculate_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Výpočet agregovaných metrik"""
        if not results:
            return {}

        valid_results = [r for r in results if r.execution_error is None]

        if not valid_results:
            return {"error": "No valid results"}

        metrics = {
            "avg_context_precision": sum(r.context_precision for r in valid_results) / len(valid_results),
            "avg_context_recall": sum(r.context_recall for r in valid_results) / len(valid_results),
            "avg_faithfulness": sum(r.faithfulness for r in valid_results) / len(valid_results),
            "avg_answer_relevance": sum(r.answer_relevance for r in valid_results) / len(valid_results),
            "avg_answer_correctness": sum(r.answer_correctness for r in valid_results) / len(valid_results),
            "avg_latency_ms": sum(r.latency_ms for r in valid_results) / len(valid_results),
            "total_token_count": sum(r.token_count for r in valid_results),
            "success_rate": len(valid_results) / len(results),
            "total_queries": len(results)
        }

        # Overall score jako průměr hlavních metrik
        metrics["overall_score"] = (
            metrics["avg_context_precision"] * 0.15 +
            metrics["avg_context_recall"] * 0.15 +
            metrics["avg_faithfulness"] * 0.25 +
            metrics["avg_answer_relevance"] * 0.25 +
            metrics["avg_answer_correctness"] * 0.20
        )

        return metrics

    def _save_results(self,
                     results: List[EvaluationResult],
                     metrics: Dict[str, float],
                     path: str):
        """Uložení výsledků evaluace"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        output = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "results": [asdict(result) for result in results]
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation results saved to {path}")

    def check_regression_thresholds(self, metrics: Dict[str, float]) -> bool:
        """Kontrola regresních threshold pro CI/CD"""
        thresholds = {
            "overall_score": 0.70,
            "avg_faithfulness": 0.75,
            "avg_answer_correctness": 0.65,
            "success_rate": 0.90
        }

        for metric, threshold in thresholds.items():
            if metrics.get(metric, 0) < threshold:
                logger.error(f"Regression detected: {metric} = {metrics.get(metric):.3f} < {threshold}")
                return False

        return True
