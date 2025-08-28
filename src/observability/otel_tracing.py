#!/usr/bin/env python3
"""
OpenTelemetry Tracing for DAG Pipeline
Comprehensive observability with spans, metrics, and performance tracking

Author: Senior Python/MLOps Agent
"""

from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
import logging
import json
from pathlib import Path
from contextlib import contextmanager
from functools import wraps
import threading
from collections import defaultdict, deque

try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.util.types import Attributes
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logging.warning("OpenTelemetry not available - using mock implementation")

logger = logging.getLogger(__name__)


@dataclass
class SpanMetrics:
    """Metrics collected for a span"""
    duration_ms: float
    tokens_processed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    error_count: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_ms": self.duration_ms,
            "tokens_processed": self.tokens_processed,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "error_count": self.error_count,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent
        }


@dataclass
class DAGNodeMetrics:
    """Aggregated metrics for a DAG node"""
    node_name: str
    total_executions: int = 0
    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0.0
    success_count: int = 0
    error_count: int = 0
    success_rate: float = 0.0
    recent_durations: deque = field(default_factory=lambda: deque(maxlen=100))

    def update(self, duration_ms: float, success: bool):
        self.total_executions += 1
        self.total_duration_ms += duration_ms
        self.avg_duration_ms = self.total_duration_ms / self.total_executions
        self.min_duration_ms = min(self.min_duration_ms, duration_ms)
        self.max_duration_ms = max(self.max_duration_ms, duration_ms)
        self.recent_durations.append(duration_ms)

        if success:
            self.success_count += 1
        else:
            self.error_count += 1

        self.success_rate = self.success_count / self.total_executions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_name": self.node_name,
            "total_executions": self.total_executions,
            "avg_duration_ms": self.avg_duration_ms,
            "min_duration_ms": self.min_duration_ms if self.min_duration_ms != float('inf') else 0.0,
            "max_duration_ms": self.max_duration_ms,
            "success_rate": self.success_rate,
            "recent_avg_duration": sum(self.recent_durations) / len(self.recent_durations) if self.recent_durations else 0.0
        }


class MockTracer:
    """Mock tracer for when OpenTelemetry is not available"""

    def __init__(self, name: str):
        self.name = name

    def start_span(self, name: str, **kwargs):
        return MockSpan(name)


class MockSpan:
    """Mock span for when OpenTelemetry is not available"""

    def __init__(self, name: str):
        self.name = name
        self.start_time = time.time()
        self.attributes = {}

    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value

    def set_status(self, status, description: str = ""):
        self.status = status
        self.description = description

    def end(self):
        self.end_time = time.time()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()


class DAGTracer:
    """Enhanced tracer for DAG pipeline observability"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tracing_config = config.get("observability", {}).get("tracing", {})

        # Service information
        self.service_name = config.get("service_name", "deep-research-tool")
        self.service_version = config.get("version", "2.0.0")

        # Initialize OpenTelemetry
        self.tracer_provider = None
        self.tracer = None
        self.meter = None
        self._initialize_telemetry()

        # Node metrics tracking
        self.node_metrics: Dict[str, DAGNodeMetrics] = defaultdict(lambda: DAGNodeMetrics("unknown"))
        self.metrics_lock = threading.Lock()

        # Current context
        self.current_pipeline_id = None
        self.current_profile = "unknown"

        logger.info(f"DAG tracer initialized (OTEL available: {OTEL_AVAILABLE})")

    def _initialize_telemetry(self):
        """Initialize OpenTelemetry providers and exporters"""

        if not OTEL_AVAILABLE:
            self.tracer = MockTracer(self.service_name)
            return

        # Resource information
        resource = Resource.create({
            "service.name": self.service_name,
            "service.version": self.service_version,
            "deployment.environment": self.config.get("environment", "development")
        })

        # Trace provider
        self.tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(self.tracer_provider)

        # Configure exporters based on config
        self._setup_trace_exporters()
        self._setup_metric_exporters()

        # Get tracer
        self.tracer = trace.get_tracer(self.service_name, self.service_version)

    def _setup_trace_exporters(self):
        """Setup trace exporters (Jaeger, OTLP, etc.)"""

        exporters_config = self.tracing_config.get("exporters", {})

        # Jaeger exporter
        if exporters_config.get("jaeger", {}).get("enabled", False):
            jaeger_config = exporters_config["jaeger"]
            jaeger_exporter = JaegerExporter(
                agent_host_name=jaeger_config.get("host", "localhost"),
                agent_port=jaeger_config.get("port", 6831),
            )

            span_processor = BatchSpanProcessor(jaeger_exporter)
            self.tracer_provider.add_span_processor(span_processor)
            logger.info("Jaeger trace exporter configured")

        # OTLP exporter
        if exporters_config.get("otlp", {}).get("enabled", False):
            otlp_config = exporters_config["otlp"]
            otlp_exporter = OTLPSpanExporter(
                endpoint=otlp_config.get("endpoint", "http://localhost:4317"),
                insecure=otlp_config.get("insecure", True)
            )

            span_processor = BatchSpanProcessor(otlp_exporter)
            self.tracer_provider.add_span_processor(span_processor)
            logger.info("OTLP trace exporter configured")

    def _setup_metric_exporters(self):
        """Setup metric exporters"""

        if not OTEL_AVAILABLE:
            return

        exporters_config = self.tracing_config.get("exporters", {})

        if exporters_config.get("otlp", {}).get("enabled", False):
            otlp_config = exporters_config["otlp"]
            metric_exporter = OTLPMetricExporter(
                endpoint=otlp_config.get("endpoint", "http://localhost:4317"),
                insecure=otlp_config.get("insecure", True)
            )

            metric_reader = PeriodicExportingMetricReader(
                exporter=metric_exporter,
                export_interval_millis=30000  # 30 seconds
            )

            meter_provider = MeterProvider(
                resource=self.tracer_provider.resource,
                metric_readers=[metric_reader]
            )

            metrics.set_meter_provider(meter_provider)
            self.meter = metrics.get_meter(self.service_name, self.service_version)
            logger.info("OTLP metric exporter configured")

    def start_pipeline(self, pipeline_id: str, profile: str, query: str) -> 'PipelineSpan':
        """Start tracing a complete pipeline execution"""

        self.current_pipeline_id = pipeline_id
        self.current_profile = profile

        span = self.tracer.start_span(
            "research_pipeline",
            attributes={
                "pipeline.id": pipeline_id,
                "pipeline.profile": profile,
                "pipeline.query": query[:100],  # Truncate long queries
                "pipeline.start_time": datetime.now(timezone.utc).isoformat()
            }
        )

        return PipelineSpan(self, span, pipeline_id, profile)

    def trace_dag_node(
        self,
        node_name: str,
        operation: str = "execute",
        **attributes
    ) -> 'DAGNodeSpan':
        """Start tracing a DAG node execution"""

        span_name = f"dag_node.{node_name}.{operation}"

        span_attributes = {
            "dag.node_name": node_name,
            "dag.operation": operation,
            "dag.pipeline_id": self.current_pipeline_id,
            "dag.profile": self.current_profile,
            **attributes
        }

        span = self.tracer.start_span(span_name, attributes=span_attributes)

        return DAGNodeSpan(self, span, node_name)

    def record_node_metrics(self, node_name: str, duration_ms: float, success: bool, **metrics):
        """Record metrics for a DAG node"""

        with self.metrics_lock:
            node_metrics = self.node_metrics[node_name]
            if node_metrics.node_name == "unknown":
                node_metrics.node_name = node_name

            node_metrics.update(duration_ms, success)

            # Store additional metrics
            for key, value in metrics.items():
                setattr(node_metrics, key, value)

    def get_node_metrics(self, node_name: Optional[str] = None) -> Union[Dict[str, DAGNodeMetrics], DAGNodeMetrics]:
        """Get metrics for specific node or all nodes"""

        with self.metrics_lock:
            if node_name:
                return self.node_metrics.get(node_name, DAGNodeMetrics(node_name))
            else:
                return dict(self.node_metrics)

    def export_metrics_summary(self) -> Dict[str, Any]:
        """Export comprehensive metrics summary"""

        with self.metrics_lock:
            summary = {
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "service_info": {
                    "name": self.service_name,
                    "version": self.service_version
                },
                "node_metrics": {
                    name: metrics.to_dict()
                    for name, metrics in self.node_metrics.items()
                },
                "pipeline_stats": {
                    "current_pipeline_id": self.current_pipeline_id,
                    "current_profile": self.current_profile,
                    "total_nodes_tracked": len(self.node_metrics)
                }
            }

            # Calculate overall statistics
            if self.node_metrics:
                all_durations = []
                total_executions = 0
                total_errors = 0

                for metrics in self.node_metrics.values():
                    all_durations.extend(metrics.recent_durations)
                    total_executions += metrics.total_executions
                    total_errors += metrics.error_count

                summary["overall_stats"] = {
                    "total_executions": total_executions,
                    "total_errors": total_errors,
                    "overall_error_rate": total_errors / max(total_executions, 1),
                    "avg_execution_time": sum(all_durations) / len(all_durations) if all_durations else 0.0
                }

            return summary


class PipelineSpan:
    """Span for entire pipeline execution"""

    def __init__(self, tracer: DAGTracer, span, pipeline_id: str, profile: str):
        self.tracer = tracer
        self.span = span
        self.pipeline_id = pipeline_id
        self.profile = profile
        self.start_time = time.time()

    def set_attribute(self, key: str, value: Any):
        """Set span attribute"""
        self.span.set_attribute(key, value)

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to span"""
        if OTEL_AVAILABLE and hasattr(self.span, 'add_event'):
            self.span.add_event(name, attributes or {})

    def set_status(self, success: bool, message: str = ""):
        """Set span status"""
        if OTEL_AVAILABLE:
            status = StatusCode.OK if success else StatusCode.ERROR
            self.span.set_status(Status(status, message))

    def end(self, **final_attributes):
        """End pipeline span"""

        duration_ms = (time.time() - self.start_time) * 1000

        # Set final attributes
        final_attrs = {
            "pipeline.duration_ms": duration_ms,
            "pipeline.end_time": datetime.now(timezone.utc).isoformat(),
            **final_attributes
        }

        for key, value in final_attrs.items():
            self.span.set_attribute(key, value)

        self.span.end()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        error_msg = str(exc_val) if exc_val else ""
        self.set_status(success, error_msg)
        self.end()


class DAGNodeSpan:
    """Span for individual DAG node execution"""

    def __init__(self, tracer: DAGTracer, span, node_name: str):
        self.tracer = tracer
        self.span = span
        self.node_name = node_name
        self.start_time = time.time()
        self.metrics = SpanMetrics(0.0)

    def set_attribute(self, key: str, value: Any):
        """Set span attribute"""
        self.span.set_attribute(key, value)

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to span"""
        if OTEL_AVAILABLE and hasattr(self.span, 'add_event'):
            self.span.add_event(name, attributes or {})

    def record_tokens(self, count: int):
        """Record token processing"""
        self.metrics.tokens_processed += count
        self.set_attribute("tokens.processed", self.metrics.tokens_processed)

    def record_cache_hit(self):
        """Record cache hit"""
        self.metrics.cache_hits += 1
        self.set_attribute("cache.hits", self.metrics.cache_hits)

    def record_cache_miss(self):
        """Record cache miss"""
        self.metrics.cache_misses += 1
        self.set_attribute("cache.misses", self.metrics.cache_misses)

    def record_error(self, error: Exception):
        """Record error"""
        self.metrics.error_count += 1
        self.set_attribute("error.count", self.metrics.error_count)
        self.set_attribute("error.type", type(error).__name__)
        self.set_attribute("error.message", str(error))

    def record_memory_usage(self, mb: float):
        """Record memory usage"""
        self.metrics.memory_usage_mb = mb
        self.set_attribute("memory.usage_mb", mb)

    def set_status(self, success: bool, message: str = ""):
        """Set span status"""
        if OTEL_AVAILABLE:
            status = StatusCode.OK if success else StatusCode.ERROR
            self.span.set_status(Status(status, message))

    def end(self, **final_attributes):
        """End node span and record metrics"""

        duration_ms = (time.time() - self.start_time) * 1000
        self.metrics.duration_ms = duration_ms

        # Set final attributes
        final_attrs = {
            "node.duration_ms": duration_ms,
            "node.end_time": datetime.now(timezone.utc).isoformat(),
            **self.metrics.to_dict(),
            **final_attributes
        }

        for key, value in final_attrs.items():
            self.span.set_attribute(key, value)

        # Record node metrics
        success = self.metrics.error_count == 0
        self.tracer.record_node_metrics(
            self.node_name,
            duration_ms,
            success,
            tokens_processed=self.metrics.tokens_processed,
            cache_hit_rate=self.metrics.cache_hits / max(self.metrics.cache_hits + self.metrics.cache_misses, 1)
        )

        self.span.end()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            self.record_error(exc_val)

        success = exc_type is None
        error_msg = str(exc_val) if exc_val else ""
        self.set_status(success, error_msg)
        self.end()


def traced_dag_node(node_name: str, operation: str = "execute"):
    """Decorator for automatic DAG node tracing"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to find tracer in arguments or kwargs
            tracer = None
            for arg in args:
                if hasattr(arg, 'tracer') and isinstance(arg.tracer, DAGTracer):
                    tracer = arg.tracer
                    break

            if not tracer:
                # Fallback to global tracer if available
                tracer = getattr(traced_dag_node, '_global_tracer', None)

            if tracer:
                with tracer.trace_dag_node(node_name, operation) as span:
                    try:
                        result = func(*args, **kwargs)

                        # Record result metadata if available
                        if isinstance(result, dict):
                            if "tokens" in result:
                                span.record_tokens(result["tokens"])
                            if "cache_hit" in result:
                                if result["cache_hit"]:
                                    span.record_cache_hit()
                                else:
                                    span.record_cache_miss()

                        return result
                    except Exception as e:
                        span.record_error(e)
                        raise
            else:
                # No tracer available, run function normally
                return func(*args, **kwargs)

        return wrapper
    return decorator


def set_global_tracer(tracer: DAGTracer):
    """Set global tracer for decorator usage"""
    traced_dag_node._global_tracer = tracer


def create_dag_tracer(config: Dict[str, Any]) -> DAGTracer:
    """Factory function for DAG tracer"""
    return DAGTracer(config)


# Usage example
if __name__ == "__main__":
    config = {
        "service_name": "deep-research-tool",
        "version": "2.0.0",
        "observability": {
            "tracing": {
                "exporters": {
                    "jaeger": {
                        "enabled": False,
                        "host": "localhost",
                        "port": 6831
                    },
                    "otlp": {
                        "enabled": False,
                        "endpoint": "http://localhost:4317",
                        "insecure": True
                    }
                }
            }
        }
    }

    tracer = DAGTracer(config)
    set_global_tracer(tracer)

    # Example pipeline execution
    with tracer.start_pipeline("test_pipeline", "thorough", "COVID vaccine effectiveness") as pipeline:

        # Retrieval node
        with tracer.trace_dag_node("retrieval", "hybrid_search") as span:
            span.record_tokens(500)
            span.record_cache_miss()
            time.sleep(0.1)  # Simulate work

        # Reranking node
        with tracer.trace_dag_node("reranking", "cross_encoder") as span:
            span.record_tokens(1000)
            span.record_cache_hit()
            time.sleep(0.2)  # Simulate work

        # Synthesis node
        with tracer.trace_dag_node("synthesis", "generate_claims") as span:
            span.record_tokens(2000)
            span.record_cache_miss()
            time.sleep(0.3)  # Simulate work

        pipeline.set_attribute("pipeline.claims_generated", 3)
        pipeline.set_attribute("pipeline.total_evidence", 15)

    # Export metrics
    metrics_summary = tracer.export_metrics_summary()
    print(f"Pipeline execution completed")
    print(f"Nodes tracked: {len(metrics_summary['node_metrics'])}")

    for node_name, metrics in metrics_summary['node_metrics'].items():
        print(f"- {node_name}: {metrics['avg_duration_ms']:.1f}ms avg, {metrics['success_rate']:.2f} success rate")
