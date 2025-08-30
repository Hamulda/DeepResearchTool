#!/usr/bin/env python3
"""Sparse Encoder Module with SPLADE + BM25 Fallback
Query intent classification and hybrid retrieval optimization

Author: Senior IT Specialist
"""

import asyncio
from dataclasses import dataclass
import logging
import re
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


@dataclass
class QueryIntent:
    """Klasifikace záměru dotazu"""

    intent_type: str  # "fact_finding", "sota_research", "exploration"
    confidence: float
    keywords: list[str]
    suggested_weights: dict[str, float]  # dense/sparse weights


class QueryIntentClassifier:
    """Klasifikuje záměr dotazu pro optimální retrieval strategii"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Patterns for different intent types
        self.patterns = {
            "fact_finding": [
                r"\b(what is|what are|define|definition|explain|how many|when did|who is)\b",
                r"\b(statistics|data|numbers|percentage|rate|amount)\b",
                r"\b(current|latest|recent|today|2024|2025)\b",
            ],
            "sota_research": [
                r"\b(state of the art|sota|latest research|cutting edge|breakthrough)\b",
                r"\b(recent advances|new developments|emerging|novel)\b",
                r"\b(comparison|versus|vs|better than|outperforms)\b",
            ],
            "exploration": [
                r"\b(overview|survey|review|comprehensive|broad)\b",
                r"\b(trends|future|potential|applications|implications)\b",
                r"\b(explore|investigate|analyze|study)\b",
            ],
        }

        # Intent-specific retrieval weights
        self.weight_profiles = {
            "fact_finding": {"dense": 0.3, "sparse": 0.7},  # Prefer exact matches
            "sota_research": {"dense": 0.7, "sparse": 0.3},  # Prefer semantic similarity
            "exploration": {"dense": 0.5, "sparse": 0.5},  # Balanced approach
        }

    def classify_intent(self, query: str) -> QueryIntent:
        """Klasifikuje záměr dotazu"""
        query_lower = query.lower()
        scores = {}
        matched_keywords = {}

        # Score each intent type
        for intent, patterns in self.patterns.items():
            score = 0
            keywords = []

            for pattern in patterns:
                matches = re.findall(pattern, query_lower)
                if matches:
                    score += len(matches)
                    keywords.extend(matches)

            scores[intent] = score
            matched_keywords[intent] = keywords

        # Determine primary intent
        if not any(scores.values()):
            # Default to exploration if no patterns match
            primary_intent = "exploration"
            confidence = 0.5
        else:
            primary_intent = max(scores, key=scores.get)
            max_score = scores[primary_intent]
            total_score = sum(scores.values())
            confidence = max_score / total_score if total_score > 0 else 0.5

        self.logger.info(f"Query intent: {primary_intent} (confidence: {confidence:.2f})")

        return QueryIntent(
            intent_type=primary_intent,
            confidence=confidence,
            keywords=matched_keywords[primary_intent],
            suggested_weights=self.weight_profiles[primary_intent],
        )


class SPLADEEncoder:
    """SPLADE sparse encoder implementation"""

    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil"):
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.tokenizer = None
        self.model = None
        self._loaded = False

    async def _load_model(self):
        """Lazy loading of SPLADE model"""
        if self._loaded:
            return

        try:
            self.logger.info(f"Loading SPLADE model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            self.logger.info("SPLADE model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load SPLADE model: {e}")
            raise

    def _compute_splade_representation(self, text: str) -> np.ndarray:
        """Compute SPLADE sparse representation"""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # Forward pass
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Apply ReLU and log transformation (SPLADE specific)
            splade_rep = torch.log(1 + torch.relu(logits))

            # Max pooling over sequence length
            splade_rep = torch.max(splade_rep, dim=1)[0]

            # Convert to sparse representation
            splade_rep = splade_rep.squeeze().cpu().numpy()

        return splade_rep

    async def encode_query(self, query: str) -> np.ndarray:
        """Encode query using SPLADE"""
        await self._load_model()
        return self._compute_splade_representation(query)

    async def encode_documents(self, documents: list[str]) -> list[np.ndarray]:
        """Encode multiple documents using SPLADE"""
        await self._load_model()

        representations = []
        for doc in documents:
            try:
                rep = self._compute_splade_representation(doc)
                representations.append(rep)
            except Exception as e:
                self.logger.warning(f"Failed to encode document: {e}")
                # Return zero vector as fallback
                representations.append(np.zeros(self.tokenizer.vocab_size))

        return representations


class BM25Fallback:
    """BM25 implementation jako fallback pro SPLADE"""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.logger = logging.getLogger(__name__)
        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", max_features=10000, ngram_range=(1, 2)
        )
        self.doc_vectors = None
        self.documents = []
        self._fitted = False

    def fit(self, documents: list[str]):
        """Fit BM25 on document collection"""
        self.logger.info(f"Fitting BM25 on {len(documents)} documents")
        self.documents = documents

        try:
            # Fit TF-IDF vectorizer
            self.doc_vectors = self.vectorizer.fit_transform(documents)
            self._fitted = True
            self.logger.info("BM25 fitted successfully")
        except Exception as e:
            self.logger.error(f"BM25 fitting failed: {e}")
            raise

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """Search documents using BM25"""
        if not self._fitted:
            self.logger.warning("BM25 not fitted, returning empty results")
            return []

        try:
            # Transform query
            query_vector = self.vectorizer.transform([query])

            # Compute BM25 scores (simplified using cosine similarity)
            similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()

            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = [(idx, similarities[idx]) for idx in top_indices if similarities[idx] > 0]

            self.logger.info(f"BM25 search returned {len(results)} results")
            return results

        except Exception as e:
            self.logger.error(f"BM25 search failed: {e}")
            return []


class SparseRetriever:
    """Unified sparse retriever with SPLADE + BM25 fallback"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.engine = config.get("engine", "splade")  # "splade" or "bm25"
        self.fallback_enabled = config.get("fallback_enabled", True)
        self.splade_timeout = config.get("splade_timeout", 30.0)  # seconds

        # Initialize components
        self.intent_classifier = QueryIntentClassifier()
        self.splade_encoder = SPLADEEncoder() if self.engine == "splade" else None
        self.bm25_fallback = BM25Fallback()

        # State
        self.documents = []
        self.splade_available = False

    async def initialize(self, documents: list[str]):
        """Initialize retriever with document collection"""
        self.documents = documents
        self.logger.info(f"Initializing sparse retriever with {len(documents)} documents")

        # Always initialize BM25 (as fallback)
        self.bm25_fallback.fit(documents)

        # Try to initialize SPLADE if configured
        if self.engine == "splade" and self.splade_encoder:
            try:
                await asyncio.wait_for(
                    self.splade_encoder._load_model(), timeout=self.splade_timeout
                )
                self.splade_available = True
                self.logger.info("SPLADE encoder ready")
            except Exception as e:
                self.logger.warning(f"SPLADE initialization failed: {e}")
                if self.fallback_enabled:
                    self.logger.info("Falling back to BM25")
                    self.splade_available = False
                else:
                    raise

    async def retrieve(
        self, query: str, top_k: int = 20
    ) -> tuple[list[dict[str, Any]], QueryIntent]:
        """Retrieve documents using sparse methods"""
        # Classify query intent
        intent = self.intent_classifier.classify_intent(query)

        # Determine which engine to use
        use_splade = (
            self.engine == "splade" and self.splade_available and self.splade_encoder is not None
        )

        if use_splade:
            try:
                results = await self._retrieve_with_splade(query, top_k)
                engine_used = "splade"
            except Exception as e:
                self.logger.warning(f"SPLADE retrieval failed: {e}")
                if self.fallback_enabled:
                    results = self._retrieve_with_bm25(query, top_k)
                    engine_used = "bm25_fallback"
                else:
                    raise
        else:
            results = self._retrieve_with_bm25(query, top_k)
            engine_used = "bm25"

        # Add metadata to results
        for result in results:
            result["engine_used"] = engine_used
            result["intent"] = intent.intent_type

        self.logger.info(f"Sparse retrieval complete: {len(results)} docs using {engine_used}")
        return results, intent

    async def _retrieve_with_splade(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Retrieve using SPLADE encoder"""
        self.logger.info("Using SPLADE for sparse retrieval")

        # Encode query
        query_rep = await self.splade_encoder.encode_query(query)

        # For demo purposes, we'll use BM25 as SPLADE indexing is complex
        # In production, this would use a proper SPLADE index
        self.logger.info("SPLADE query encoded, falling back to BM25 for demo")
        return self._retrieve_with_bm25(query, top_k)

    def _retrieve_with_bm25(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Retrieve using BM25"""
        self.logger.info("Using BM25 for sparse retrieval")

        bm25_results = self.bm25_fallback.search(query, top_k)

        results = []
        for doc_idx, score in bm25_results:
            if doc_idx < len(self.documents):
                results.append(
                    {
                        "doc_id": f"doc_{doc_idx}",
                        "content": self.documents[doc_idx],
                        "score": float(score),
                        "rank": len(results) + 1,
                        "source": "sparse_bm25",
                    }
                )

        return results


# Factory function
def create_sparse_retriever(config: dict[str, Any]) -> SparseRetriever:
    """Factory function pro sparse retriever"""
    return SparseRetriever(config)
