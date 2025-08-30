"""
Langfuse observability integration pro Research Agent
Poskytuje end-to-end tracing, metriky a monitoring
"""

import os
import time
import logging
from typing import Dict, Any, Optional, List
from functools import wraps
from contextlib import contextmanager

from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from langfuse.callback import CallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LangfuseConfig(BaseModel):
    """Konfigurace pro Langfuse"""

    secret_key: str
    public_key: str
    host: str = "http://localhost:3000"
    enabled: bool = True
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class ResearchAgentTracer:
    """Hlavní třída pro tracing Research Agent operací"""

    def __init__(self, config: LangfuseConfig):
        self.config = config
        self.langfuse = None
        self.callback_handler = None

        if config.enabled:
            self._initialize_langfuse()

    def _initialize_langfuse(self):
        """Inicializace Langfuse klienta"""
        try:
            self.langfuse = Langfuse(
                secret_key=self.config.secret_key,
                public_key=self.config.public_key,
                host=self.config.host,
            )

            self.callback_handler = CallbackHandler(
                secret_key=self.config.secret_key,
                public_key=self.config.public_key,
                host=self.config.host,
                session_id=self.config.session_id,
                user_id=self.config.user_id,
            )

            logger.info(f"Langfuse initialized with host: {self.config.host}")

        except Exception as e:
            logger.error(f"Failed to initialize Langfuse: {e}")
            self.config.enabled = False

    def get_callback_handler(self) -> Optional[BaseCallbackHandler]:
        """Vrátí callback handler pro LangChain integration"""
        return self.callback_handler if self.config.enabled else None

    @contextmanager
    def trace_research_session(self, query: str, metadata: Dict[str, Any] = None):
        """Context manager pro tracing celé research session"""
        if not self.config.enabled:
            yield None
            return

        trace = self.langfuse.trace(
            name="research_agent_session",
            input={"query": query},
            metadata=metadata or {},
            session_id=self.config.session_id,
            user_id=self.config.user_id,
        )

        try:
            yield trace
        finally:
            trace.update(output={"status": "completed"}, end_time=time.time())

    def trace_node_execution(
        self, node_name: str, input_data: Dict[str, Any], trace_id: Optional[str] = None
    ) -> Optional[Any]:
        """Trace jednotlivého uzlu v grafu"""
        if not self.config.enabled:
            return None

        span = self.langfuse.span(
            trace_id=trace_id,
            name=f"node_{node_name}",
            input=input_data,
            metadata={"node_type": node_name},
        )

        return span

    def trace_tool_usage(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Any,
        span_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ):
        """Trace použití nástroje"""
        if not self.config.enabled:
            return

        self.langfuse.span(
            trace_id=trace_id,
            parent_span_id=span_id,
            name=f"tool_{tool_name}",
            input=tool_input,
            output={"result": str(tool_output)[:1000]},  # Limit délky
            metadata={"tool_type": tool_name, "execution_time": time.time()},
        )

    def log_metrics(self, metrics: Dict[str, float], trace_id: Optional[str] = None):
        """Logování metrik (latence, náklady, etc.)"""
        if not self.config.enabled:
            return

        self.langfuse.score(
            trace_id=trace_id,
            name="performance_metrics",
            value=metrics.get("overall_score", 0.0),
            metadata=metrics,
        )

    def record_token_usage(
        self, prompt_tokens: int, completion_tokens: int, model: str, trace_id: Optional[str] = None
    ):
        """Zaznamenání využití tokenů pro cost tracking"""
        if not self.config.enabled:
            return

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "model": model,
        }

        self.langfuse.span(trace_id=trace_id, name="token_usage", metadata=usage)


def trace_research_operation(operation_name: str):
    """Decorator pro tracing research operací"""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = getattr(args[0], "tracer", None) if args else None

            if tracer and tracer.config.enabled:
                span = tracer.trace_node_execution(
                    operation_name, {"args": str(args)[:500], "kwargs": str(kwargs)[:500]}
                )

                try:
                    result = await func(*args, **kwargs)
                    if span:
                        span.update(output={"result": str(result)[:1000]})
                    return result
                except Exception as e:
                    if span:
                        span.update(output={"error": str(e)}, level="ERROR")
                    raise
            else:
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = getattr(args[0], "tracer", None) if args else None

            if tracer and tracer.config.enabled:
                span = tracer.trace_node_execution(
                    operation_name, {"args": str(args)[:500], "kwargs": str(kwargs)[:500]}
                )

                try:
                    result = func(*args, **kwargs)
                    if span:
                        span.update(output={"result": str(result)[:1000]})
                    return result
                except Exception as e:
                    if span:
                        span.update(output={"error": str(e)}, level="ERROR")
                    raise
            else:
                return func(*args, **kwargs)

        return (
            async_wrapper
            if hasattr(func, "__code__") and func.__code__.co_flags & 0x80
            else sync_wrapper
        )

    return decorator


class ObservabilityManager:
    """Centrální manager pro observability"""

    def __init__(self):
        self.tracer: Optional[ResearchAgentTracer] = None
        self._initialize_from_env()

    def _initialize_from_env(self):
        """Inicializace z environment variables"""
        config = LangfuseConfig(
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
            enabled=os.getenv("LANGFUSE_ENABLED", "true").lower() == "true",
        )

        if config.secret_key and config.public_key:
            self.tracer = ResearchAgentTracer(config)
        else:
            logger.warning("Langfuse credentials not found. Observability disabled.")

    def get_tracer(self) -> Optional[ResearchAgentTracer]:
        """Vrátí aktuální tracer"""
        return self.tracer

    def is_enabled(self) -> bool:
        """Kontrola, zda je observability zapnutá"""
        return self.tracer is not None and self.tracer.config.enabled


# Global instance
observability_manager = ObservabilityManager()


def get_observability_manager() -> ObservabilityManager:
    """Getter pro global observability manager"""
    return observability_manager
