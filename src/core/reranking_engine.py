#!/usr/bin/env python3
"""
Re-ranking Engine pro Deep Research Tool
Implementuje cross-encoder re-ranking s konfigurovatelními kritérii

Author: Senior IT Specialist
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from datetime import datetime
import json

import structlog

logger = structlog.get_logger(__name__)

@dataclass
class RankingCriteria:
    """Kritéria pro re-ranking"""
    relevance_weight: float = 0.5
    authority_weight: float = 0.3
    novelty_weight: float = 0.2
    recency_weight: float = 0.1

@dataclass
class RankedDocument:
    """Přeřazený dokument s detailním skórováním"""
    document_id: str
    original_rank: int
    new_rank: int
    relevance_score: float
    authority_score: float
    novelty_score: float
    recency_score: float
    combined_score: float
    ranking_reason: str
    metadata: Dict[str, Any]

class ReRankingEngine:
    """Re-ranking engine s cross-encoder nebo LLM rater"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reranking_config = config.get("workflow", {}).get("phases", {}).get("reranking", {})
        self.m1_config = config.get("m1_optimization", {})

        # Model konfigurace
        self.model_name = self.reranking_config.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.use_llm_rater = self.reranking_config.get("use_llm_rater", False)

        # Cross-encoder komponenty
        self.tokenizer = None
        self.model = None

        # M1 optimalizace
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        # Ranking kritéria
        self.criteria = RankingCriteria()

        self.logger = structlog.get_logger(__name__)

    async def initialize(self):
        """Inicializace re-ranking enginu"""
        self.logger.info("Inicializace re-ranking enginu")

        if not self.use_llm_rater:
            await self._initialize_cross_encoder()
        else:
            await self._initialize_llm_rater()

        self.logger.info("Re-ranking engine inicializován",
                        model=self.model_name,
                        device=self.device,
                        use_llm=self.use_llm_rater)

    async def _initialize_cross_encoder(self):
        """Inicializace cross-encoder modelu"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

            # M1 optimalizace
            if self.device == "mps":
                self.model = self.model.to(self.device)

            self.model.eval()

            self.logger.info("Cross-encoder načten", model=self.model_name)

        except Exception as e:
            self.logger.error("Chyba při načítání cross-encoder", error=str(e))
            raise

    async def _initialize_llm_rater(self):
        """Inicializace LLM rateru"""
        from ..core.ollama_agent import OllamaResearchAgent

        self.llm_agent = OllamaResearchAgent(self.config)
        self.llm_model = self.m1_config.get("ollama", {}).get("models", {}).get("reranking", "qwen2.5:7b")

        self.logger.info("LLM rater inicializován", model=self.llm_model)

    async def rerank_documents(self,
                             query: str,
                             documents: List[Dict[str, Any]],
                             top_k: int = 20) -> Dict[str, Any]:
        """Hlavní metoda pro re-ranking dokumentů"""

        self.logger.info("Spouštím re-ranking",
                        query=query,
                        documents=len(documents),
                        top_k=top_k)

        if not documents:
            return {"ranked_documents": [], "metrics": {}}

        start_time = datetime.now()

        if self.use_llm_rater:
            ranked_results = await self._llm_rerank(query, documents, top_k)
        else:
            ranked_results = await self._cross_encoder_rerank(query, documents, top_k)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Metriky re-ranking procesu
        metrics = {
            "duration_seconds": duration,
            "input_documents": len(documents),
            "output_documents": len(ranked_results),
            "reranking_method": "llm" if self.use_llm_rater else "cross_encoder",
            "average_score_change": self._calculate_score_change(documents, ranked_results)
        }

        return {
            "ranked_documents": ranked_results,
            "metrics": metrics
        }

    async def _cross_encoder_rerank(self,
                                  query: str,
                                  documents: List[Dict[str, Any]],
                                  top_k: int) -> List[RankedDocument]:
        """Re-ranking pomocí cross-encoder"""

        ranked_docs = []

        # Zpracování v dávkách pro M1 optimalizaci
        batch_size = 8

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_results = await self._process_cross_encoder_batch(query, batch, i)
            ranked_docs.extend(batch_results)

        # Výpočet dalších skóre (autorита, novost, aktualnost)
        for doc in ranked_docs:
            doc.authority_score = await self._calculate_authority_score(doc)
            doc.novelty_score = await self._calculate_novelty_score(doc)
            doc.recency_score = await self._calculate_recency_score(doc)

            # Kombinované skóre
            doc.combined_score = (
                self.criteria.relevance_weight * doc.relevance_score +
                self.criteria.authority_weight * doc.authority_score +
                self.criteria.novelty_weight * doc.novelty_score +
                self.criteria.recency_weight * doc.recency_score
            )

            doc.ranking_reason = self._generate_ranking_reason(doc)

        # Finální řazení podle kombinovaného skóre
        ranked_docs.sort(key=lambda x: x.combined_score, reverse=True)

        # Aktualizace nových pozic
        for new_rank, doc in enumerate(ranked_docs):
            doc.new_rank = new_rank + 1

        return ranked_docs[:top_k]

    async def _process_cross_encoder_batch(self,
                                         query: str,
                                         batch: List[Dict[str, Any]],
                                         start_idx: int) -> List[RankedDocument]:
        """Zpracování dávky dokumentů cross-encoderem"""

        # Příprava vstupů pro cross-encoder
        query_doc_pairs = []
        for doc in batch:
            content = doc.get("content", "")[:512]  # Truncate pro rychlost
            query_doc_pairs.append(f"{query} [SEP] {content}")

        try:
            # Tokenizace
            inputs = self.tokenizer(
                query_doc_pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # M1 optimalizace
            if self.device == "mps":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.softmax(outputs.logits, dim=-1)[:, 1]  # Pozitivní třída

            # Konverze na CPU numpy
            relevance_scores = scores.cpu().numpy()

        except Exception as e:
            self.logger.error("Chyba při cross-encoder inference", error=str(e))
            relevance_scores = np.zeros(len(batch))

        # Vytvoření RankedDocument objektů
        ranked_docs = []
        for i, (doc, relevance_score) in enumerate(zip(batch, relevance_scores)):
            ranked_doc = RankedDocument(
                document_id=doc.get("id", f"doc_{start_idx + i}"),
                original_rank=start_idx + i + 1,
                new_rank=0,  # Bude aktualizováno později
                relevance_score=float(relevance_score),
                authority_score=0.0,
                novelty_score=0.0,
                recency_score=0.0,
                combined_score=0.0,
                ranking_reason="",
                metadata=doc
            )
            ranked_docs.append(ranked_doc)

        return ranked_docs

    async def _llm_rerank(self,
                         query: str,
                         documents: List[Dict[str, Any]],
                         top_k: int) -> List[RankedDocument]:
        """Re-ranking pomocí LLM as rater"""

        # Příprava prompt pro LLM rating
        documents_text = ""
        for i, doc in enumerate(documents[:20]):  # Limit pro LLM context
            title = doc.get("title", "")
            content = doc.get("content", "")[:300]  # Truncate
            source = doc.get("source", "")

            documents_text += f"""
Document {i+1}:
Title: {title}
Source: {source}
Content: {content}
---
"""

        rating_prompt = f"""
Jako expert na hodnocení relevance dokumentů, ohodnoť následující dokumenty podle jejich relevance k dotazu.

Dotaz: {query}

Dokumenty k hodnocení:
{documents_text}

Pro každý dokument poskytni:
1. Skóre relevance (0.0-1.0)
2. Skóre autority zdroje (0.0-1.0) 
3. Skóre novosti informací (0.0-1.0)
4. Stručné odůvodnění

Odpověz ve formátu JSON:
{{
  "ratings": [
    {{
      "document_id": 1,
      "relevance_score": 0.85,
      "authority_score": 0.9,
      "novelty_score": 0.7,
      "reasoning": "Dokument přímo odpovídá na dotaz..."
    }},
    ...
  ]
}}
"""

        try:
            response = await self.llm_agent.generate_response(
                rating_prompt,
                model=self.llm_model,
                max_tokens=1000
            )

            # Parsování JSON odpovědi
            ratings_data = json.loads(response)
            ratings = ratings_data.get("ratings", [])

            # Vytvoření RankedDocument objektů
            ranked_docs = []
            for i, (doc, rating) in enumerate(zip(documents, ratings)):
                if i < len(ratings):
                    relevance = rating.get("relevance_score", 0.5)
                    authority = rating.get("authority_score", 0.5)
                    novelty = rating.get("novelty_score", 0.5)
                    reasoning = rating.get("reasoning", "")
                else:
                    relevance = authority = novelty = 0.5
                    reasoning = "Nedostatečná data"

                # Recency score z metadat
                recency = await self._calculate_recency_score_from_doc(doc)

                combined_score = (
                    self.criteria.relevance_weight * relevance +
                    self.criteria.authority_weight * authority +
                    self.criteria.novelty_weight * novelty +
                    self.criteria.recency_weight * recency
                )

                ranked_doc = RankedDocument(
                    document_id=doc.get("id", f"doc_{i}"),
                    original_rank=i + 1,
                    new_rank=0,
                    relevance_score=relevance,
                    authority_score=authority,
                    novelty_score=novelty,
                    recency_score=recency,
                    combined_score=combined_score,
                    ranking_reason=reasoning,
                    metadata=doc
                )
                ranked_docs.append(ranked_doc)

            # Řazení podle kombinovaného skóre
            ranked_docs.sort(key=lambda x: x.combined_score, reverse=True)

            # Aktualizace nových pozic
            for new_rank, doc in enumerate(ranked_docs):
                doc.new_rank = new_rank + 1

            return ranked_docs[:top_k]

        except Exception as e:
            self.logger.error("Chyba při LLM re-ranking", error=str(e))
            # Fallback na původní pořadí
            return self._create_fallback_ranking(documents, top_k)

    async def _calculate_authority_score(self, doc: RankedDocument) -> float:
        """Výpočet skóre autority zdroje"""

        source = doc.metadata.get("source", "").lower()
        url = doc.metadata.get("url", "").lower()

        # Autoritativní domény
        authority_domains = {
            "arxiv.org": 0.9,
            "pubmed.ncbi.nlm.nih.gov": 0.95,
            "scholar.google.com": 0.8,
            "ieee.org": 0.9,
            "acm.org": 0.9,
            "nature.com": 0.95,
            "science.org": 0.95,
            "gov": 0.85,
            "edu": 0.8,
            "org": 0.7
        }

        authority_score = 0.5  # Default

        for domain, score in authority_domains.items():
            if domain in url or domain in source:
                authority_score = score
                break

        return authority_score

    async def _calculate_novelty_score(self, doc: RankedDocument) -> float:
        """Výpočet skóre novosti informací"""

        # Jednoduchá heuristika založená na zdroji a obsahu
        content = doc.metadata.get("content", "").lower()
        title = doc.metadata.get("title", "").lower()

        novelty_indicators = [
            "new", "novel", "recent", "latest", "breakthrough",
            "innovative", "first", "unprecedented", "cutting-edge"
        ]

        novelty_count = sum(1 for indicator in novelty_indicators
                          if indicator in content or indicator in title)

        # Normalizace na 0-1
        novelty_score = min(novelty_count / 3.0, 1.0)

        return novelty_score

    async def _calculate_recency_score(self, doc: RankedDocument) -> float:
        """Výpočet skóre aktualnosti"""
        return await self._calculate_recency_score_from_doc(doc.metadata)

    async def _calculate_recency_score_from_doc(self, doc: Dict[str, Any]) -> float:
        """Výpočet skóre aktualnosti z dokumentu"""

        try:
            timestamp = doc.get("timestamp", "")
            if not timestamp:
                return 0.5  # Neznámé datum

            from datetime import datetime
            import dateutil.parser

            doc_date = dateutil.parser.parse(timestamp)
            now = datetime.now(doc_date.tzinfo) if doc_date.tzinfo else datetime.now()

            # Skóre klesá s věkem (exponenciálně)
            days_old = (now - doc_date).days
            recency_score = np.exp(-days_old / 365.0)  # Polovina po roce

            return min(max(recency_score, 0.0), 1.0)

        except Exception:
            return 0.5

    def _generate_ranking_reason(self, doc: RankedDocument) -> str:
        """Generování důvodu pro re-ranking"""

        reasons = []

        if doc.relevance_score > 0.8:
            reasons.append("vysoká relevance")
        elif doc.relevance_score < 0.3:
            reasons.append("nízká relevance")

        if doc.authority_score > 0.8:
            reasons.append("autoritativní zdroj")

        if doc.novelty_score > 0.7:
            reasons.append("nové informace")

        if doc.recency_score > 0.8:
            reasons.append("aktuální obsah")

        if not reasons:
            reasons.append("standardní skórování")

        return ", ".join(reasons)

    def _calculate_score_change(self,
                              original_docs: List[Dict[str, Any]],
                              ranked_docs: List[RankedDocument]) -> float:
        """Výpočet průměrné změny skóre"""

        if not ranked_docs:
            return 0.0

        total_change = 0.0
        count = 0

        for ranked_doc in ranked_docs:
            original_rank = ranked_doc.original_rank
            new_rank = ranked_doc.new_rank

            rank_change = abs(original_rank - new_rank)
            total_change += rank_change
            count += 1

        return total_change / count if count > 0 else 0.0

    def _create_fallback_ranking(self,
                               documents: List[Dict[str, Any]],
                               top_k: int) -> List[RankedDocument]:
        """Fallback ranking při chybě"""

        ranked_docs = []

        for i, doc in enumerate(documents[:top_k]):
            ranked_doc = RankedDocument(
                document_id=doc.get("id", f"doc_{i}"),
                original_rank=i + 1,
                new_rank=i + 1,
                relevance_score=0.5,
                authority_score=0.5,
                novelty_score=0.5,
                recency_score=0.5,
                combined_score=0.5,
                ranking_reason="fallback ranking",
                metadata=doc
            )
            ranked_docs.append(ranked_doc)

        return ranked_docs

    async def update_ranking_criteria(self, new_criteria: Dict[str, float]):
        """Aktualizace kritérií pro ranking"""

        self.criteria.relevance_weight = new_criteria.get("relevance_weight", self.criteria.relevance_weight)
        self.criteria.authority_weight = new_criteria.get("authority_weight", self.criteria.authority_weight)
        self.criteria.novelty_weight = new_criteria.get("novelty_weight", self.criteria.novelty_weight)
        self.criteria.recency_weight = new_criteria.get("recency_weight", self.criteria.recency_weight)

        self.logger.info("Ranking kritéria aktualizována", criteria=new_criteria)
