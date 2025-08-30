#!/usr/bin/env python3
"""FÁZE 6: Streaming Engine with Progressive Context Building
Streaming inference s early-exit a adaptive batch sizing pro M1

Author: Senior Python/MLOps Agent
"""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
import time
from typing import Any

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class StreamingChunk:
    """Chunk dat pro streaming"""

    chunk_id: int
    content: str
    tokens: int
    timestamp: float
    novelty_score: float
    context_window_used: int
    is_early_exit: bool = False


@dataclass
class ProgressiveContext:
    """Progressive context building state"""

    accumulated_context: str
    context_length: int
    quality_score: float
    information_density: float
    chunks_processed: int
    early_exits: int


@dataclass
class StreamingMetrics:
    """Streaming performance metriky"""

    total_chunks: int
    total_tokens: int
    streaming_time_s: float
    tokens_per_second: float
    context_efficiency: float
    early_exit_rate: float
    memory_usage_mb: float
    progressive_quality_score: float


class AdaptiveBatchSizer:
    """Adaptive batch sizing pro M1 optimalizaci"""

    def __init__(self, initial_batch_size: int = 8, min_batch: int = 2, max_batch: int = 32):
        self.current_batch_size = initial_batch_size
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.performance_history = []

    def adjust_batch_size(
        self, latency_ms: float, memory_usage_mb: float, memory_limit_mb: float
    ) -> int:
        """Adjustuje batch size na základě performance"""
        self.performance_history.append(
            {
                "batch_size": self.current_batch_size,
                "latency_ms": latency_ms,
                "memory_usage_mb": memory_usage_mb,
                "timestamp": time.time(),
            }
        )

        # Keep only recent history
        if len(self.performance_history) > 10:
            self.performance_history = self.performance_history[-10:]

        # Memory pressure check
        memory_pressure = memory_usage_mb / memory_limit_mb
        if memory_pressure > 0.85:  # High memory pressure
            self.current_batch_size = max(self.min_batch, self.current_batch_size - 2)
        elif memory_pressure < 0.5 and latency_ms < 1000:  # Low pressure, good latency
            self.current_batch_size = min(self.max_batch, self.current_batch_size + 1)

        return self.current_batch_size

    def get_optimal_batch_size(self) -> int:
        """Vrací aktuální optimální batch size"""
        return self.current_batch_size


class EarlyExitController:
    """Controller pro early exit při nízké novosti"""

    def __init__(self, novelty_threshold: float = 0.15, min_chunks: int = 5):
        self.novelty_threshold = novelty_threshold
        self.min_chunks = min_chunks
        self.novelty_history = []

    def should_exit_early(self, chunk: StreamingChunk, total_chunks: int) -> bool:
        """Rozhodne, zda ukončit stream early"""
        if total_chunks < self.min_chunks:
            return False

        self.novelty_history.append(chunk.novelty_score)

        # Keep sliding window of recent novelty scores
        if len(self.novelty_history) > 5:
            self.novelty_history = self.novelty_history[-5:]

        # Check if recent chunks have consistently low novelty
        if len(self.novelty_history) >= 3:
            recent_avg_novelty = np.mean(self.novelty_history[-3:])
            return recent_avg_novelty < self.novelty_threshold

        return False

    def calculate_novelty_score(self, new_content: str, existing_context: str) -> float:
        """Vypočítá novelty score pro nový content"""
        if not existing_context or not new_content:
            return 1.0

        # Simple novelty calculation based on word overlap
        new_words = set(new_content.lower().split())
        existing_words = set(existing_context.lower().split())

        if not new_words:
            return 0.0

        # Calculate ratio of truly new words
        novel_words = new_words - existing_words
        novelty_ratio = len(novel_words) / len(new_words)

        return novelty_ratio


class M1StreamingEngine:
    """M1 optimalizovaný streaming engine"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.batch_sizer = AdaptiveBatchSizer(
            initial_batch_size=config.get("streaming", {}).get("initial_batch_size", 8)
        )
        self.early_exit_controller = EarlyExitController(
            novelty_threshold=config.get("streaming", {}).get("novelty_threshold", 0.15)
        )
        self.ollama_client = None
        self._initialize_streaming()

    def _initialize_streaming(self):
        """Inicializuje streaming components"""
        try:
            import ollama

            self.ollama_client = ollama.Client()
            print("✅ Streaming engine initialized with Ollama")
        except ImportError:
            print("⚠️ Ollama not available - using mock streaming")

    async def stream_with_progressive_context(
        self,
        query: str,
        model: str = "qwen2.5:7b-q4_K_M",
        context_window: int = 8192,
        max_tokens: int = 4096,
        memory_limit_mb: int = 8192,
    ) -> tuple[ProgressiveContext, StreamingMetrics]:
        """Streaming s progressive context building"""
        print(f"🚀 Starting progressive streaming for: {query[:50]}...")

        start_time = time.time()
        progressive_context = ProgressiveContext(
            accumulated_context="",
            context_length=0,
            quality_score=0.0,
            information_density=0.0,
            chunks_processed=0,
            early_exits=0,
        )

        streaming_chunks = []
        total_tokens = 0

        try:
            if self.ollama_client:
                # Real Ollama streaming
                async for chunk in self._ollama_stream_generator(
                    query, model, context_window, max_tokens
                ):
                    streaming_chunks.append(chunk)
                    total_tokens += chunk.tokens

                    # Update progressive context
                    progressive_context = self._update_progressive_context(
                        progressive_context, chunk
                    )

                    # Adaptive batch sizing
                    current_memory = self._estimate_memory_usage()
                    latency = (time.time() - start_time) * 1000
                    self.batch_sizer.adjust_batch_size(latency, current_memory, memory_limit_mb)

                    # Early exit check
                    if self.early_exit_controller.should_exit_early(chunk, len(streaming_chunks)):
                        chunk.is_early_exit = True
                        progressive_context.early_exits += 1
                        print(
                            f"🛑 Early exit triggered at chunk {len(streaming_chunks)} (novelty: {chunk.novelty_score:.3f})"
                        )
                        break

                    # Memory pressure check
                    if current_memory > memory_limit_mb:
                        print(
                            f"⚠️ Memory limit reached: {current_memory:.0f}MB > {memory_limit_mb}MB"
                        )
                        break
            else:
                # Mock streaming for testing
                async for chunk in self._mock_stream_generator(query, max_tokens):
                    streaming_chunks.append(chunk)
                    total_tokens += chunk.tokens

                    progressive_context = self._update_progressive_context(
                        progressive_context, chunk
                    )

                    if self.early_exit_controller.should_exit_early(chunk, len(streaming_chunks)):
                        chunk.is_early_exit = True
                        progressive_context.early_exits += 1
                        break

        except Exception as e:
            print(f"❌ Streaming error: {e!s}")

        total_time = time.time() - start_time

        # Calculate final metrics
        metrics = StreamingMetrics(
            total_chunks=len(streaming_chunks),
            total_tokens=total_tokens,
            streaming_time_s=total_time,
            tokens_per_second=total_tokens / total_time if total_time > 0 else 0,
            context_efficiency=progressive_context.information_density,
            early_exit_rate=(
                progressive_context.early_exits / len(streaming_chunks) if streaming_chunks else 0
            ),
            memory_usage_mb=self._estimate_memory_usage(),
            progressive_quality_score=progressive_context.quality_score,
        )

        print(
            f"✅ Streaming completed: {len(streaming_chunks)} chunks, {total_tokens} tokens, {total_time:.1f}s"
        )

        return progressive_context, metrics

    async def _ollama_stream_generator(
        self, query: str, model: str, context_window: int, max_tokens: int
    ) -> AsyncGenerator[StreamingChunk, None]:
        """Generator pro Ollama streaming"""
        try:
            stream = self.ollama_client.generate(
                model=model,
                prompt=query,
                stream=True,
                options={
                    "num_ctx": context_window,
                    "num_predict": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9,
                },
            )

            chunk_id = 0
            accumulated_context = ""

            for response_chunk in stream:
                if response_chunk.get("done", False):
                    break

                content = response_chunk.get("response", "")
                if not content:
                    continue

                # Calculate novelty
                novelty_score = self.early_exit_controller.calculate_novelty_score(
                    content, accumulated_context
                )

                chunk = StreamingChunk(
                    chunk_id=chunk_id,
                    content=content,
                    tokens=len(content.split()),
                    timestamp=time.time(),
                    novelty_score=novelty_score,
                    context_window_used=len(accumulated_context.split()),
                )

                accumulated_context += " " + content
                chunk_id += 1

                yield chunk

                # Brief pause to allow for async processing
                await asyncio.sleep(0.01)

        except Exception as e:
            print(f"❌ Ollama streaming error: {e!s}")

    async def _mock_stream_generator(
        self, query: str, max_tokens: int
    ) -> AsyncGenerator[StreamingChunk, None]:
        """Mock generator pro testing"""
        # Generate realistic streaming chunks
        sample_responses = [
            "Based on recent research,",
            "quantum computing has made significant advances",
            "in error correction methodologies.",
            "Key breakthroughs include topological qubits",
            "and improved quantum error correction codes.",
            "These developments show promise for",
            "building more stable quantum systems.",
            "However, challenges remain in scaling",
            "to practical quantum advantage.",
        ]

        accumulated_context = ""

        for i, response in enumerate(sample_responses):
            if len(accumulated_context.split()) > max_tokens:
                break

            # Calculate novelty
            novelty_score = self.early_exit_controller.calculate_novelty_score(
                response, accumulated_context
            )

            chunk = StreamingChunk(
                chunk_id=i,
                content=response,
                tokens=len(response.split()),
                timestamp=time.time(),
                novelty_score=novelty_score,
                context_window_used=len(accumulated_context.split()),
            )

            accumulated_context += " " + response

            yield chunk

            # Simulate streaming delay
            await asyncio.sleep(0.1)

    def _update_progressive_context(
        self, context: ProgressiveContext, chunk: StreamingChunk
    ) -> ProgressiveContext:
        """Update progressive context s novým chunkem"""
        # Add content to context
        context.accumulated_context += " " + chunk.content
        context.context_length = len(context.accumulated_context.split())
        context.chunks_processed += 1

        # Update quality metrics
        if context.chunks_processed > 0:
            # Information density = unique concepts per total length
            unique_words = len(set(context.accumulated_context.lower().split()))
            total_words = context.context_length
            context.information_density = unique_words / total_words if total_words > 0 else 0

            # Quality score based on novelty and information density
            context.quality_score = chunk.novelty_score * 0.6 + context.information_density * 0.4

        return context

    def _estimate_memory_usage(self) -> float:
        """Odhad memory usage v MB"""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 1024.0  # Default estimate

    async def benchmark_streaming_performance(
        self, test_queries: list[str], profile_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Benchmark streaming performance"""
        print(f"🏁 Benchmarking streaming performance with {len(test_queries)} queries...")

        results = []
        total_start = time.time()

        for i, query in enumerate(test_queries):
            print(f"📋 Query {i+1}/{len(test_queries)}: {query[:50]}...")

            try:
                context, metrics = await self.stream_with_progressive_context(
                    query=query,
                    model=profile_config.get("model", "qwen2.5:7b-q4_K_M"),
                    context_window=profile_config.get("context_window", 8192),
                    max_tokens=profile_config.get("max_tokens", 4096),
                    memory_limit_mb=profile_config.get("memory_limit_mb", 8192),
                )

                result = {
                    "query": query,
                    "execution_time_s": metrics.streaming_time_s,
                    "total_tokens": metrics.total_tokens,
                    "tokens_per_second": metrics.tokens_per_second,
                    "chunks_processed": metrics.total_chunks,
                    "early_exit_rate": metrics.early_exit_rate,
                    "context_efficiency": metrics.context_efficiency,
                    "memory_usage_mb": metrics.memory_usage_mb,
                    "quality_score": metrics.progressive_quality_score,
                }

                results.append(result)

                print(
                    f"✅ Query {i+1}: {metrics.streaming_time_s:.1f}s, {metrics.tokens_per_second:.1f} tok/s"
                )

            except Exception as e:
                print(f"❌ Query {i+1} failed: {e!s}")
                results.append(
                    {"query": query, "error": str(e), "execution_time_s": 0, "total_tokens": 0}
                )

        total_time = time.time() - total_start

        # Calculate aggregate metrics
        successful_results = [r for r in results if "error" not in r]

        benchmark_report = {
            "streaming_benchmark": {
                "timestamp": datetime.now().isoformat(),
                "total_queries": len(test_queries),
                "successful_queries": len(successful_results),
                "total_benchmark_time_s": total_time,
                "profile_config": profile_config,
                "aggregate_metrics": {},
                "individual_results": results,
            }
        }

        if successful_results:
            benchmark_report["streaming_benchmark"]["aggregate_metrics"] = {
                "avg_execution_time_s": np.mean(
                    [r["execution_time_s"] for r in successful_results]
                ),
                "avg_tokens_per_second": np.mean(
                    [r["tokens_per_second"] for r in successful_results]
                ),
                "avg_early_exit_rate": np.mean([r["early_exit_rate"] for r in successful_results]),
                "avg_context_efficiency": np.mean(
                    [r["context_efficiency"] for r in successful_results]
                ),
                "avg_memory_usage_mb": np.mean([r["memory_usage_mb"] for r in successful_results]),
                "p95_execution_time_s": np.percentile(
                    [r["execution_time_s"] for r in successful_results], 95
                ),
                "success_rate": len(successful_results) / len(test_queries),
            }

        return benchmark_report


# Factory function
def create_m1_streaming_engine(config: dict[str, Any]) -> M1StreamingEngine:
    """Factory function pro vytvoření M1 streaming engine"""
    return M1StreamingEngine(config)


if __name__ == "__main__":
    # Test streaming engine
    config = {
        "streaming": {
            "initial_batch_size": 8,
            "novelty_threshold": 0.15,
            "enable_early_exit": True,
            "progressive_context": True,
        }
    }

    engine = create_m1_streaming_engine(config)
    print("✅ M1 Streaming Engine initialized!")

    # Test queries
    test_queries = [
        "What are quantum computing error correction advances?",
        "Explain machine learning model compression techniques",
        "Recent developments in renewable energy storage",
    ]

    async def test_streaming():
        context, metrics = await engine.stream_with_progressive_context(
            "Test query for streaming engine", max_tokens=1000
        )
        print(
            f"Streaming test completed: {metrics.total_chunks} chunks, {metrics.tokens_per_second:.1f} tok/s"
        )

    # asyncio.run(test_streaming())
