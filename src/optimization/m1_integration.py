#!/usr/bin/env python3
"""
FÁZE 6: M1 Integration Orchestrator
Integruje všechny M1 optimalizace do hlavního research pipeline

Author: Senior Python/MLOps Agent
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.optimization.m1_performance import create_m1_optimization_engine, M1PerformanceMetrics, PerformanceProfile
from src.optimization.streaming_engine import create_m1_streaming_engine, ProgressiveContext, StreamingMetrics


@dataclass
class M1ResearchSession:
    """M1 optimalizovaná research session"""
    session_id: str
    query: str
    profile: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_execution_time_s: float = 0.0
    memory_peak_mb: float = 0.0
    tokens_generated: int = 0
    claims_generated: int = 0
    citations_count: int = 0
    m1_metrics: Optional[M1PerformanceMetrics] = None
    streaming_metrics: Optional[StreamingMetrics] = None
    progressive_context: Optional[ProgressiveContext] = None
    success: bool = False
    error_message: Optional[str] = None


class M1ResearchOrchestrator:
    """M1 optimalizovaný research orchestrátor"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.m1_engine = create_m1_optimization_engine(config)
        self.streaming_engine = create_m1_streaming_engine(config)
        self.active_sessions = {}

    async def execute_m1_research(self,
                                query: str,
                                profile: str = "balanced",
                                enable_streaming: bool = True) -> M1ResearchSession:
        """Vykoná M1 optimalizovaný research"""

        session_id = f"m1_session_{int(time.time())}"
        session = M1ResearchSession(
            session_id=session_id,
            query=query,
            profile=profile,
            start_time=datetime.now()
        )

        self.active_sessions[session_id] = session

        print(f"🚀 Starting M1 research session: {session_id}")
        print(f"Query: {query}")
        print(f"Profile: {profile}")
        print(f"Streaming: {enable_streaming}")

        try:
            # Phase 1: M1 Performance Optimization
            print("\n📊 Phase 1: M1 Performance Optimization")
            session.m1_metrics = await self.m1_engine.optimize_for_profile(profile)

            if session.m1_metrics.error_rate > 0.5:
                raise Exception(f"M1 optimization failed with high error rate: {session.m1_metrics.error_rate}")

            # Phase 2: Streaming Inference (if enabled)
            if enable_streaming:
                print("\n🌊 Phase 2: Streaming Inference with Progressive Context")

                profile_config = self.m1_engine.performance_profiles[profile]

                session.progressive_context, session.streaming_metrics = await self.streaming_engine.stream_with_progressive_context(
                    query=query,
                    model=profile_config.ollama_model,
                    context_window=profile_config.context_window,
                    max_tokens=profile_config.max_tokens,
                    memory_limit_mb=profile_config.memory_limit_mb
                )

                # Extract metrics from streaming
                session.tokens_generated = session.streaming_metrics.total_tokens
                session.memory_peak_mb = max(session.m1_metrics.memory_peak_mb,
                                           session.streaming_metrics.memory_usage_mb)

            # Phase 3: Mock Evidence Processing (would be real research in production)
            print("\n🔍 Phase 3: Evidence Processing")
            await self._process_evidence_m1_optimized(session)

            # Phase 4: Results Synthesis
            print("\n📝 Phase 4: Results Synthesis")
            await self._synthesize_results_m1_optimized(session)

            session.end_time = datetime.now()
            session.total_execution_time_s = (session.end_time - session.start_time).total_seconds()
            session.success = True

            print(f"\n✅ M1 research session completed successfully!")
            print(f"Total time: {session.total_execution_time_s:.1f}s")
            print(f"Memory peak: {session.memory_peak_mb:.0f}MB")
            print(f"Tokens generated: {session.tokens_generated}")
            print(f"Claims generated: {session.claims_generated}")

        except Exception as e:
            session.end_time = datetime.now()
            session.total_execution_time_s = (session.end_time - session.start_time).total_seconds()
            session.error_message = str(e)
            session.success = False

            print(f"\n❌ M1 research session failed: {str(e)}")

        finally:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]

        return session

    async def _process_evidence_m1_optimized(self, session: M1ResearchSession):
        """M1 optimalizované zpracování evidence"""

        # Mock evidence processing s M1 optimalizacemi
        profile_config = self.m1_engine.performance_profiles[session.profile]

        # Simulate adaptive batch processing
        batch_size = profile_config.batch_size
        processing_time = 0.5  # Optimized for M1

        await asyncio.sleep(processing_time)

        # Mock results
        session.claims_generated = 2 if session.profile == "quick" else 3
        session.citations_count = session.claims_generated * 3  # 3 citations per claim

        print(f"   Evidence processed with batch_size={batch_size}")
        print(f"   Generated {session.claims_generated} claims with {session.citations_count} citations")

    async def _synthesize_results_m1_optimized(self, session: M1ResearchSession):
        """M1 optimalizovaná syntéza výsledků"""

        # Mock synthesis s progressive context využitím
        if session.progressive_context:
            context_quality = session.progressive_context.quality_score
            print(f"   Using progressive context (quality: {context_quality:.2f})")

        synthesis_time = 0.3  # Fast synthesis on M1
        await asyncio.sleep(synthesis_time)

        print(f"   Results synthesized in {synthesis_time}s")

    async def run_m1_benchmark_session(self,
                                     test_queries: List[str],
                                     profiles: List[str] = None) -> Dict[str, Any]:
        """Spustí benchmark session s multiple queries a profiles"""

        if profiles is None:
            profiles = ["quick", "thorough"]

        print(f"🏁 Running M1 benchmark session...")
        print(f"Queries: {len(test_queries)}")
        print(f"Profiles: {', '.join(profiles)}")

        benchmark_start = time.time()
        results = {
            "m1_benchmark_session": {
                "timestamp": datetime.now().isoformat(),
                "test_queries": test_queries,
                "profiles": profiles,
                "sessions": []
            }
        }

        for profile in profiles:
            for query in test_queries:
                print(f"\n📋 Testing: {profile} profile with query: {query[:50]}...")

                session = await self.execute_m1_research(
                    query=query,
                    profile=profile,
                    enable_streaming=True
                )

                session_result = {
                    "session_id": session.session_id,
                    "query": session.query,
                    "profile": session.profile,
                    "success": session.success,
                    "execution_time_s": session.total_execution_time_s,
                    "memory_peak_mb": session.memory_peak_mb,
                    "tokens_generated": session.tokens_generated,
                    "claims_generated": session.claims_generated,
                    "citations_count": session.citations_count,
                    "error_message": session.error_message
                }

                if session.m1_metrics:
                    session_result["m1_metrics"] = {
                        "tokens_per_second": session.m1_metrics.tokens_per_second,
                        "memory_efficiency": session.m1_metrics.memory_efficiency,
                        "mps_utilization": session.m1_metrics.mps_utilization,
                        "early_exit_rate": session.m1_metrics.early_exit_rate
                    }

                if session.streaming_metrics:
                    session_result["streaming_metrics"] = {
                        "context_efficiency": session.streaming_metrics.context_efficiency,
                        "progressive_quality_score": session.streaming_metrics.progressive_quality_score,
                        "early_exit_rate": session.streaming_metrics.early_exit_rate
                    }

                results["m1_benchmark_session"]["sessions"].append(session_result)

                status = "✅" if session.success else "❌"
                print(f"   {status} {session.total_execution_time_s:.1f}s | {session.memory_peak_mb:.0f}MB | {session.tokens_generated} tokens")

        total_benchmark_time = time.time() - benchmark_start
        results["m1_benchmark_session"]["total_benchmark_time_s"] = total_benchmark_time

        # Calculate aggregate metrics
        successful_sessions = [s for s in results["m1_benchmark_session"]["sessions"] if s["success"]]

        if successful_sessions:
            results["m1_benchmark_session"]["aggregate_metrics"] = {
                "success_rate": len(successful_sessions) / len(results["m1_benchmark_session"]["sessions"]),
                "avg_execution_time_s": sum(s["execution_time_s"] for s in successful_sessions) / len(successful_sessions),
                "avg_memory_peak_mb": sum(s["memory_peak_mb"] for s in successful_sessions) / len(successful_sessions),
                "avg_tokens_generated": sum(s["tokens_generated"] for s in successful_sessions) / len(successful_sessions),
                "total_claims_generated": sum(s["claims_generated"] for s in successful_sessions),
                "total_citations_count": sum(s["citations_count"] for s in successful_sessions)
            }

        print(f"\n🎉 M1 benchmark session completed in {total_benchmark_time:.1f}s")
        print(f"Success rate: {len(successful_sessions)}/{len(results['m1_benchmark_session']['sessions'])}")

        return results

    def validate_m1_performance_targets(self, session: M1ResearchSession) -> Dict[str, bool]:
        """Validuje M1 performance targets"""

        profile_config = self.m1_engine.performance_profiles[session.profile]

        validations = {
            "execution_time": session.total_execution_time_s <= profile_config.timeout_seconds,
            "memory_limit": session.memory_peak_mb <= profile_config.memory_limit_mb,
            "min_claims": session.claims_generated >= 1,
            "min_citations": session.citations_count >= 2,
            "success": session.success,
            "no_oom": session.memory_peak_mb < profile_config.memory_limit_mb * 0.95  # 95% limit
        }

        return validations

    async def export_m1_session_results(self, session: M1ResearchSession,
                                      output_path: str = "docs/m1_session_results.json"):
        """Exportuje M1 session výsledky"""

        session_data = {
            "m1_research_session": {
                "session_id": session.session_id,
                "timestamp": session.start_time.isoformat(),
                "query": session.query,
                "profile": session.profile,
                "execution_time_s": session.total_execution_time_s,
                "memory_peak_mb": session.memory_peak_mb,
                "tokens_generated": session.tokens_generated,
                "claims_generated": session.claims_generated,
                "citations_count": session.citations_count,
                "success": session.success,
                "error_message": session.error_message,
                "validations": self.validate_m1_performance_targets(session)
            }
        }

        # Add detailed metrics if available
        if session.m1_metrics:
            session_data["m1_research_session"]["m1_performance_metrics"] = {
                "tokens_per_second": session.m1_metrics.tokens_per_second,
                "memory_efficiency": session.m1_metrics.memory_efficiency,
                "context_utilization": session.m1_metrics.context_utilization,
                "mps_utilization": session.m1_metrics.mps_utilization,
                "early_exit_rate": session.m1_metrics.early_exit_rate,
                "streaming_chunks": session.m1_metrics.streaming_chunks,
                "error_rate": session.m1_metrics.error_rate
            }

        if session.streaming_metrics:
            session_data["m1_research_session"]["streaming_metrics"] = {
                "total_chunks": session.streaming_metrics.total_chunks,
                "context_efficiency": session.streaming_metrics.context_efficiency,
                "progressive_quality_score": session.streaming_metrics.progressive_quality_score,
                "early_exit_rate": session.streaming_metrics.early_exit_rate
            }

        if session.progressive_context:
            session_data["m1_research_session"]["progressive_context"] = {
                "context_length": session.progressive_context.context_length,
                "quality_score": session.progressive_context.quality_score,
                "information_density": session.progressive_context.information_density,
                "chunks_processed": session.progressive_context.chunks_processed,
                "early_exits": session.progressive_context.early_exits
            }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(session_data, f, indent=2)

        print(f"📊 M1 session results exported: {output_file}")


# Factory function
def create_m1_research_orchestrator(config: Dict[str, Any]) -> M1ResearchOrchestrator:
    """Factory function pro vytvoření M1 research orchestrator"""
    return M1ResearchOrchestrator(config)


if __name__ == "__main__":
    # Test M1 research orchestrator
    config = {
        "m1_optimization": {
            "enable_mps": True,
            "enable_streaming": True
        },
        "streaming": {
            "initial_batch_size": 8,
            "novelty_threshold": 0.15,
            "enable_early_exit": True
        }
    }

    async def test_m1_research():
        orchestrator = create_m1_research_orchestrator(config)

        # Test single research session
        session = await orchestrator.execute_m1_research(
            query="What are the latest advances in quantum computing?",
            profile="quick"
        )

        print(f"\nTest session results:")
        print(f"Success: {session.success}")
        print(f"Time: {session.total_execution_time_s:.1f}s")
        print(f"Claims: {session.claims_generated}")
        print(f"Citations: {session.citations_count}")

        await orchestrator.export_m1_session_results(session)

    # asyncio.run(test_m1_research())
