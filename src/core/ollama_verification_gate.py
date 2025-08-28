#!/usr/bin/env python3
"""
Ollama Verification Gating System
Quality-based model switching and context budget optimization

Author: Senior IT Specialist
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import re

try:
    import ollama
except ImportError:
    print("❌ Ollama library not installed: pip install ollama")


@dataclass
class VerificationResult:
    """Result of verification process"""
    confidence: float
    reasoning: str
    model_used: str
    processing_time: float
    context_tokens: int
    fallback_triggered: bool = False
    quality_score: float = 0.0


@dataclass
class ContextBudget:
    """Context window management"""
    max_tokens: int
    used_tokens: int = 0
    chunk_overlap: int = 100
    adaptive_chunking: bool = True


class ModelPerformanceTracker:
    """Tracks model performance and quality metrics"""

    def __init__(self):
        self.performance_history = {}
        self.quality_thresholds = {
            "qwen2.5:3b-q4_K_M": {"min_confidence": 0.5, "max_latency": 5.0},
            "qwen2.5:7b-q4_K_M": {"min_confidence": 0.7, "max_latency": 10.0},
            "llama3.2:8b-q4_K_M": {"min_confidence": 0.8, "max_latency": 15.0}
        }
        self.logger = logging.getLogger(__name__)

    def record_performance(self, model: str, confidence: float,
                          latency: float, context_tokens: int):
        """Record model performance metrics"""
        if model not in self.performance_history:
            self.performance_history[model] = {
                "calls": 0,
                "total_latency": 0.0,
                "confidence_scores": [],
                "context_usage": [],
                "last_updated": datetime.now().isoformat()
            }

        history = self.performance_history[model]
        history["calls"] += 1
        history["total_latency"] += latency
        history["confidence_scores"].append(confidence)
        history["context_usage"].append(context_tokens)
        history["last_updated"] = datetime.now().isoformat()

        # Keep only recent history (last 100 calls)
        if len(history["confidence_scores"]) > 100:
            history["confidence_scores"] = history["confidence_scores"][-100:]
            history["context_usage"] = history["context_usage"][-100:]

    def get_model_stats(self, model: str) -> Dict[str, Any]:
        """Get performance statistics for model"""
        if model not in self.performance_history:
            return {"avg_confidence": 0.0, "avg_latency": 0.0, "calls": 0}

        history = self.performance_history[model]

        return {
            "calls": history["calls"],
            "avg_confidence": sum(history["confidence_scores"]) / len(history["confidence_scores"]) if history["confidence_scores"] else 0.0,
            "avg_latency": history["total_latency"] / history["calls"] if history["calls"] > 0 else 0.0,
            "avg_context_tokens": sum(history["context_usage"]) / len(history["context_usage"]) if history["context_usage"] else 0,
            "last_updated": history["last_updated"]
        }

    def should_fallback(self, model: str, confidence: float, latency: float) -> bool:
        """Determine if should fallback to better model"""
        if model not in self.quality_thresholds:
            return False

        thresholds = self.quality_thresholds[model]

        # Check confidence threshold
        if confidence < thresholds["min_confidence"]:
            self.logger.info(f"Fallback triggered: confidence {confidence:.3f} < {thresholds['min_confidence']}")
            return True

        # Check latency (optional - might indicate model issues)
        if latency > thresholds["max_latency"] * 2:  # 2x normal latency
            self.logger.warning(f"High latency detected: {latency:.1f}s > {thresholds['max_latency'] * 2:.1f}s")

        return False


class ContextOptimizer:
    """Optimizes context usage and chunking strategies"""

    def __init__(self, max_context: int = 4096):
        self.max_context = max_context
        self.logger = logging.getLogger(__name__)

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token average)"""
        return len(text) // 4

    def optimize_chunks(self, passages: List[str], query: str,
                       budget: ContextBudget) -> List[str]:
        """Optimize passage chunks for context window"""
        query_tokens = self.estimate_tokens(query)
        available_tokens = budget.max_tokens - query_tokens - 200  # Reserve for response

        if not budget.adaptive_chunking:
            # Simple truncation
            optimized_passages = []
            current_tokens = 0

            for passage in passages:
                passage_tokens = self.estimate_tokens(passage)
                if current_tokens + passage_tokens <= available_tokens:
                    optimized_passages.append(passage)
                    current_tokens += passage_tokens
                else:
                    # Truncate last passage to fit
                    remaining_tokens = available_tokens - current_tokens
                    if remaining_tokens > 100:  # Minimum viable chunk
                        chars_limit = remaining_tokens * 4
                        truncated = passage[:chars_limit] + "..."
                        optimized_passages.append(truncated)
                    break

            return optimized_passages

        # Adaptive chunking with overlap
        optimized_passages = []
        current_tokens = 0

        for passage in passages:
            passage_tokens = self.estimate_tokens(passage)

            if passage_tokens <= available_tokens - current_tokens:
                # Passage fits completely
                optimized_passages.append(passage)
                current_tokens += passage_tokens
            else:
                # Need to chunk passage
                remaining_tokens = available_tokens - current_tokens

                if remaining_tokens > budget.chunk_overlap:
                    # Split into chunks with overlap
                    chars_per_token = len(passage) / passage_tokens
                    chunk_size = int((remaining_tokens - budget.chunk_overlap) * chars_per_token)

                    if chunk_size > 200:  # Minimum viable chunk
                        chunk = passage[:chunk_size]

                        # Find good break point (sentence end)
                        for i in range(len(chunk) - 1, max(len(chunk) - 200, 0), -1):
                            if chunk[i] in '.!?':
                                chunk = chunk[:i + 1]
                                break

                        optimized_passages.append(chunk)
                        current_tokens += self.estimate_tokens(chunk)

                break  # No more space

        budget.used_tokens = current_tokens + query_tokens

        self.logger.info(f"Context optimization: {len(passages)} → {len(optimized_passages)} passages, "
                        f"{budget.used_tokens}/{budget.max_tokens} tokens")

        return optimized_passages

    def tune_overlap(self, domain: str, performance_history: Dict[str, Any]) -> int:
        """Auto-tune chunk overlap based on domain and performance"""
        base_overlap = 100

        # Domain-specific adjustments
        domain_multipliers = {
            "legal": 1.5,      # Legal documents need more context
            "medical": 1.3,    # Medical texts are dense
            "technical": 1.2,  # Technical docs have dependencies
            "news": 0.8,       # News articles are more independent
            "general": 1.0
        }

        multiplier = domain_multipliers.get(domain, 1.0)

        # Performance-based adjustment
        if performance_history.get("avg_confidence", 0) < 0.6:
            multiplier *= 1.2  # Increase overlap if confidence is low

        optimized_overlap = int(base_overlap * multiplier)

        self.logger.debug(f"Tuned overlap for {domain}: {optimized_overlap} tokens")
        return optimized_overlap


class OllamaVerificationGate:
    """Main verification gate with model fallback"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Model configuration
        self.primary_model = config.get("primary_model", "qwen2.5:7b-q4_K_M")
        self.fallback_model = config.get("fallback_model", "llama3.2:8b-q4_K_M")
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.top_k_passages = config.get("top_k", 5)

        # Context management
        self.max_context = config.get("max_context", 4096)
        self.enable_adaptive_chunking = config.get("adaptive_chunking", True)

        # Components
        self.performance_tracker = ModelPerformanceTracker()
        self.context_optimizer = ContextOptimizer(self.max_context)

        # Ollama client
        self.client = ollama.Client()

    async def verify_claim_with_evidence(self, claim: str, evidence_passages: List[str],
                                       domain: str = "general") -> VerificationResult:
        """
        Verify claim against evidence with quality gating

        Args:
            claim: Claim to verify
            evidence_passages: Supporting evidence passages
            domain: Domain for context optimization

        Returns:
            VerificationResult with confidence and reasoning
        """
        self.logger.info(f"Verifying claim with {len(evidence_passages)} evidence passages")

        # Optimize context usage
        budget = ContextBudget(
            max_tokens=self.max_context,
            chunk_overlap=self.context_optimizer.tune_overlap(domain, {}),
            adaptive_chunking=self.enable_adaptive_chunking
        )

        # Select and optimize top passages
        top_passages = evidence_passages[:self.top_k_passages]
        optimized_passages = self.context_optimizer.optimize_chunks(
            top_passages, claim, budget
        )

        # First attempt with primary model
        start_time = time.time()

        primary_result = await self._verify_with_model(
            claim, optimized_passages, self.primary_model
        )

        primary_latency = time.time() - start_time

        # Record performance
        self.performance_tracker.record_performance(
            self.primary_model,
            primary_result.confidence,
            primary_latency,
            budget.used_tokens
        )

        # Check if fallback is needed
        should_fallback = self.performance_tracker.should_fallback(
            self.primary_model,
            primary_result.confidence,
            primary_latency
        )

        if should_fallback and self.fallback_model != self.primary_model:
            self.logger.info(f"Quality gate triggered, using fallback model: {self.fallback_model}")

            # Re-optimize for fallback model (might have different context limits)
            fallback_start = time.time()

            fallback_result = await self._verify_with_model(
                claim, optimized_passages, self.fallback_model
            )

            fallback_latency = time.time() - fallback_start

            # Record fallback performance
            self.performance_tracker.record_performance(
                self.fallback_model,
                fallback_result.confidence,
                fallback_latency,
                budget.used_tokens
            )

            # Use fallback result
            fallback_result.fallback_triggered = True
            fallback_result.processing_time = primary_latency + fallback_latency

            self.logger.info(f"Fallback verification complete: confidence {fallback_result.confidence:.3f}")
            return fallback_result

        else:
            primary_result.processing_time = primary_latency
            self.logger.info(f"Primary verification complete: confidence {primary_result.confidence:.3f}")
            return primary_result

    async def _verify_with_model(self, claim: str, evidence_passages: List[str],
                               model: str) -> VerificationResult:
        """Verify claim using specific model"""
        # Construct verification prompt
        evidence_text = "\n\n".join([
            f"Evidence {i+1}: {passage}"
            for i, passage in enumerate(evidence_passages)
        ])

        prompt = f"""You are a fact-checking expert. Analyze the claim against the provided evidence and determine how well it is supported.

CLAIM: {claim}

EVIDENCE:
{evidence_text}

Evaluate the claim based on:
1. Direct support from evidence
2. Logical consistency
3. Potential contradictions
4. Evidence quality and reliability

Respond with:
CONFIDENCE: X.XX (0.00 to 1.00)
REASONING: Brief explanation of your assessment
QUALITY: X.XX (evidence quality score 0.00 to 1.00)

Be concise but thorough in your reasoning."""

        try:
            response = await asyncio.to_thread(
                self.client.generate,
                model=model,
                prompt=prompt,
                options={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 300
                }
            )

            response_text = response.get("response", "")

            # Parse response
            confidence = self._extract_confidence(response_text)
            reasoning = self._extract_reasoning(response_text)
            quality_score = self._extract_quality(response_text)

            # Estimate context tokens
            context_tokens = self.context_optimizer.estimate_tokens(prompt + response_text)

            return VerificationResult(
                confidence=confidence,
                reasoning=reasoning,
                model_used=model,
                processing_time=0.0,  # Set by caller
                context_tokens=context_tokens,
                quality_score=quality_score
            )

        except Exception as e:
            self.logger.error(f"Verification failed with model {model}: {e}")

            # Return low-confidence result on error
            return VerificationResult(
                confidence=0.1,
                reasoning=f"Verification failed: {str(e)}",
                model_used=model,
                processing_time=0.0,
                context_tokens=0,
                quality_score=0.0
            )

    def _extract_confidence(self, response: str) -> float:
        """Extract confidence score from response"""
        confidence_match = re.search(r"CONFIDENCE:\s*(\d+\.?\d*)", response, re.IGNORECASE)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
                return max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
            except ValueError:
                pass

        # Fallback parsing
        confidence_patterns = [
            r"confidence.*?(\d+\.?\d*)(?:%|\s|$)",
            r"(\d+\.?\d*).*?confidence",
            r"score.*?(\d+\.?\d*)"
        ]

        for pattern in confidence_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    if value > 1.0:  # Assume percentage
                        value /= 100.0
                    return max(0.0, min(1.0, value))
                except ValueError:
                    continue

        # Default to medium confidence if can't parse
        return 0.5

    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning from response"""
        reasoning_match = re.search(r"REASONING:\s*(.+?)(?:\n(?:QUALITY|$))", response, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            return reasoning_match.group(1).strip()

        # Fallback: return first paragraph that's not confidence/quality
        lines = response.split('\n')
        for line in lines:
            if (len(line.strip()) > 20 and
                not re.match(r"(CONFIDENCE|QUALITY|SCORE):", line, re.IGNORECASE)):
                return line.strip()

        return "Unable to extract reasoning from response"

    def _extract_quality(self, response: str) -> float:
        """Extract quality score from response"""
        quality_match = re.search(r"QUALITY:\s*(\d+\.?\d*)", response, re.IGNORECASE)
        if quality_match:
            try:
                quality = float(quality_match.group(1))
                return max(0.0, min(1.0, quality))
            except ValueError:
                pass

        # Default quality score
        return 0.7

    async def batch_verify_claims(self, claims_with_evidence: List[Dict[str, Any]],
                                domain: str = "general") -> List[VerificationResult]:
        """Verify multiple claims in batch"""
        self.logger.info(f"Batch verifying {len(claims_with_evidence)} claims")

        results = []
        total_start = time.time()

        for i, claim_data in enumerate(claims_with_evidence, 1):
            self.logger.info(f"Verifying claim {i}/{len(claims_with_evidence)}")

            claim_text = claim_data.get("text", "")
            evidence_passages = [
                ev.get("snippet", ev.get("content", ""))
                for ev in claim_data.get("evidence", [])
            ]

            if not claim_text or not evidence_passages:
                # Skip claims without text or evidence
                results.append(VerificationResult(
                    confidence=0.0,
                    reasoning="No claim text or evidence provided",
                    model_used="none",
                    processing_time=0.0,
                    context_tokens=0
                ))
                continue

            result = await self.verify_claim_with_evidence(
                claim_text, evidence_passages, domain
            )
            results.append(result)

        total_time = time.time() - total_start

        # Log batch statistics
        avg_confidence = sum(r.confidence for r in results) / len(results) if results else 0
        fallback_count = sum(1 for r in results if r.fallback_triggered)

        self.logger.info(f"Batch verification complete:")
        self.logger.info(f"  Average confidence: {avg_confidence:.3f}")
        self.logger.info(f"  Fallback triggered: {fallback_count}/{len(results)} times")
        self.logger.info(f"  Total time: {total_time:.1f}s")

        return results

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all models"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "models": {}
        }

        for model in [self.primary_model, self.fallback_model]:
            stats = self.performance_tracker.get_model_stats(model)
            summary["models"][model] = stats

        return summary


def create_verification_gate(config: Dict[str, Any]) -> OllamaVerificationGate:
    """Factory function for verification gate"""
    return OllamaVerificationGate(config)
