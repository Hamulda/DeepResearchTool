#!/usr/bin/env python3
"""
Metrics Collector
Comprehensive performance and operational metrics collection

Author: Senior Python/MLOps Agent
"""

from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import time
import threading
import json
import psutil
import logging
from pathlib import Path
from collections import defaultdict, deque
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected"""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricValue:
    """Individual metric value with metadata"""

    value: Union[int, float]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "timestamp": self.timestamp.isoformat(), "tags": self.tags}


@dataclass
class MetricDefinition:
    """Definition of a metric including aggregation rules"""

    name: str
    metric_type: MetricType
    description: str
    unit: str = ""
    retention_hours: int = 24
    aggregation_window_minutes: int = 5

    def __post_init__(self):
        self.values: deque = deque(
            maxlen=int(self.retention_hours * 60 / self.aggregation_window_minutes)
        )
        self.lock = threading.Lock()


class SystemMetricsCollector:
    """Collector for system-level metrics (CPU, memory, disk, etc.)"""

    def __init__(self):
        self.process = psutil.Process()
        self.system_start_time = time.time()

    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics"""

        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_used_mb = memory.used / (1024 * 1024)
            memory_percent = memory.percent

            # Process-specific metrics
            process_memory = self.process.memory_info()
            process_memory_mb = process_memory.rss / (1024 * 1024)
            process_cpu_percent = self.process.cpu_percent()

            # Disk metrics for current directory
            disk_usage = psutil.disk_usage(".")
            disk_used_gb = disk_usage.used / (1024 * 1024 * 1024)
            disk_free_gb = disk_usage.free / (1024 * 1024 * 1024)
            disk_percent = (disk_usage.used / disk_usage.total) * 100

            # Network metrics (if available)
            network_stats = psutil.net_io_counters()
            network_bytes_sent = network_stats.bytes_sent if network_stats else 0
            network_bytes_recv = network_stats.bytes_recv if network_stats else 0

            return {
                "system.cpu.percent": cpu_percent,
                "system.cpu.count": cpu_count,
                "system.memory.used_mb": memory_used_mb,
                "system.memory.percent": memory_percent,
                "process.memory.rss_mb": process_memory_mb,
                "process.cpu.percent": process_cpu_percent,
                "disk.used_gb": disk_used_gb,
                "disk.free_gb": disk_free_gb,
                "disk.percent": disk_percent,
                "network.bytes_sent": network_bytes_sent,
                "network.bytes_recv": network_bytes_recv,
                "uptime_seconds": time.time() - self.system_start_time,
            }

        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
            return {}


class MetricsCollector:
    """Main metrics collection and aggregation system"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_config = config.get("observability", {}).get("metrics", {})

        # Storage
        self.storage_dir = Path(config.get("metrics_storage", "metrics_data"))
        self.storage_dir.mkdir(exist_ok=True)

        # Metric definitions
        self.metrics: Dict[str, MetricDefinition] = {}
        self.metric_values: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))

        # System metrics
        self.system_collector = SystemMetricsCollector()

        # Collection state
        self.collection_enabled = self.metrics_config.get("enabled", True)
        self.collection_interval = self.metrics_config.get("collection_interval_seconds", 30)
        self.collection_thread = None
        self.stop_collection = threading.Event()

        # Performance tracking
        self.performance_metrics = {
            "total_metrics_collected": 0,
            "collection_errors": 0,
            "last_collection_time": None,
            "avg_collection_duration_ms": 0.0,
        }

        # Initialize standard metrics
        self._initialize_standard_metrics()

        # Start background collection
        if self.collection_enabled:
            self.start_collection()

        logger.info(f"Metrics collector initialized (storage: {self.storage_dir})")

    def _initialize_standard_metrics(self):
        """Initialize standard DeepResearchTool metrics"""

        standard_metrics = [
            # Pipeline metrics
            MetricDefinition(
                "pipeline.executions", MetricType.COUNTER, "Total pipeline executions"
            ),
            MetricDefinition(
                "pipeline.duration_ms", MetricType.TIMER, "Pipeline execution duration", "ms"
            ),
            MetricDefinition(
                "pipeline.success_rate", MetricType.GAUGE, "Pipeline success rate", "%"
            ),
            MetricDefinition(
                "pipeline.claims_generated", MetricType.HISTOGRAM, "Claims generated per pipeline"
            ),
            # Retrieval metrics
            MetricDefinition("retrieval.queries", MetricType.COUNTER, "Total retrieval queries"),
            MetricDefinition("retrieval.duration_ms", MetricType.TIMER, "Retrieval duration", "ms"),
            MetricDefinition(
                "retrieval.documents_found", MetricType.HISTOGRAM, "Documents found per query"
            ),
            MetricDefinition("retrieval.cache_hit_rate", MetricType.GAUGE, "Cache hit rate", "%"),
            # Reranking metrics
            MetricDefinition(
                "reranking.operations", MetricType.COUNTER, "Total reranking operations"
            ),
            MetricDefinition("reranking.duration_ms", MetricType.TIMER, "Reranking duration", "ms"),
            MetricDefinition(
                "reranking.cross_encoder_usage", MetricType.GAUGE, "Cross-encoder usage rate", "%"
            ),
            # LLM metrics
            MetricDefinition("llm.requests", MetricType.COUNTER, "Total LLM requests"),
            MetricDefinition("llm.tokens_input", MetricType.COUNTER, "Input tokens processed"),
            MetricDefinition("llm.tokens_output", MetricType.COUNTER, "Output tokens generated"),
            MetricDefinition("llm.duration_ms", MetricType.TIMER, "LLM request duration", "ms"),
            # Verification metrics
            MetricDefinition("verification.claims_checked", MetricType.COUNTER, "Claims verified"),
            MetricDefinition(
                "verification.contradictions_found", MetricType.COUNTER, "Contradictions detected"
            ),
            MetricDefinition(
                "verification.confidence_avg", MetricType.GAUGE, "Average verification confidence"
            ),
            # Cache metrics
            MetricDefinition("cache.hits", MetricType.COUNTER, "Cache hits"),
            MetricDefinition("cache.misses", MetricType.COUNTER, "Cache misses"),
            MetricDefinition("cache.size_mb", MetricType.GAUGE, "Cache size", "MB"),
            # Error metrics
            MetricDefinition("errors.total", MetricType.COUNTER, "Total errors"),
            MetricDefinition("errors.rate", MetricType.GAUGE, "Error rate", "errors/min"),
            # System metrics
            MetricDefinition("system.cpu_percent", MetricType.GAUGE, "CPU usage", "%"),
            MetricDefinition("system.memory_mb", MetricType.GAUGE, "Memory usage", "MB"),
            MetricDefinition("system.disk_percent", MetricType.GAUGE, "Disk usage", "%"),
        ]

        for metric in standard_metrics:
            self.register_metric(metric)

    def register_metric(self, metric_def: MetricDefinition):
        """Register a new metric definition"""
        self.metrics[metric_def.name] = metric_def
        logger.debug(f"Registered metric: {metric_def.name} ({metric_def.metric_type.value})")

    def record_metric(
        self,
        name: str,
        value: Union[int, float],
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None,
    ):
        """Record a metric value"""

        if not self.collection_enabled:
            return

        if name not in self.metrics:
            logger.warning(f"Unknown metric: {name}")
            return

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        if tags is None:
            tags = {}

        metric_value = MetricValue(value, timestamp, tags)

        with self.metrics[name].lock:
            self.metrics[name].values.append(metric_value)

        self.performance_metrics["total_metrics_collected"] += 1

    def increment_counter(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        self.record_metric(name, 1, tags)

    def set_gauge(self, name: str, value: Union[int, float], tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric value"""
        self.record_metric(name, value, tags)

    def record_timer(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None):
        """Record a timer metric"""
        self.record_metric(name, duration_ms, tags)

    def record_histogram(
        self, name: str, value: Union[int, float], tags: Optional[Dict[str, str]] = None
    ):
        """Record a histogram metric"""
        self.record_metric(name, value, tags)

    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations"""
        return MetricTimer(self, name, tags)

    def get_metric_values(
        self, name: str, since: Optional[datetime] = None, until: Optional[datetime] = None
    ) -> List[MetricValue]:
        """Get metric values within time range"""

        if name not in self.metrics:
            return []

        with self.metrics[name].lock:
            values = list(self.metrics[name].values)

        if since or until:
            filtered_values = []
            for value in values:
                if since and value.timestamp < since:
                    continue
                if until and value.timestamp > until:
                    continue
                filtered_values.append(value)
            values = filtered_values

        return values

    def get_metric_aggregates(
        self, name: str, since: Optional[datetime] = None, window_minutes: int = 5
    ) -> Dict[str, float]:
        """Get aggregated metric statistics"""

        if since is None:
            since = datetime.now(timezone.utc) - timedelta(hours=1)

        values = self.get_metric_values(name, since)

        if not values:
            return {}

        numeric_values = [v.value for v in values]

        aggregates = {
            "count": len(numeric_values),
            "sum": sum(numeric_values),
            "avg": statistics.mean(numeric_values),
            "min": min(numeric_values),
            "max": max(numeric_values),
            "latest": numeric_values[-1] if numeric_values else 0,
        }

        if len(numeric_values) > 1:
            aggregates["std"] = statistics.stdev(numeric_values)
            aggregates["median"] = statistics.median(numeric_values)

        return aggregates

    def collect_system_metrics(self):
        """Collect and record system metrics"""

        start_time = time.time()

        try:
            system_metrics = self.system_collector.collect_system_metrics()

            for metric_name, value in system_metrics.items():
                # Map system metrics to our metric names
                if metric_name == "system.cpu.percent":
                    self.set_gauge("system.cpu_percent", value)
                elif metric_name == "process.memory.rss_mb":
                    self.set_gauge("system.memory_mb", value)
                elif metric_name == "disk.percent":
                    self.set_gauge("system.disk_percent", value)

                # Record all system metrics with system prefix
                full_name = (
                    f"system.{metric_name}"
                    if not metric_name.startswith("system.")
                    else metric_name
                )
                self.record_metric(full_name, value, {"source": "system_collector"})

            self.performance_metrics["last_collection_time"] = datetime.now(timezone.utc)

        except Exception as e:
            self.performance_metrics["collection_errors"] += 1
            logger.error(f"System metrics collection failed: {e}")

        # Update collection duration
        duration_ms = (time.time() - start_time) * 1000
        current_avg = self.performance_metrics["avg_collection_duration_ms"]
        total_collections = self.performance_metrics["total_metrics_collected"]

        if total_collections > 0:
            self.performance_metrics["avg_collection_duration_ms"] = (
                current_avg * (total_collections - 1) + duration_ms
            ) / total_collections
        else:
            self.performance_metrics["avg_collection_duration_ms"] = duration_ms

    def start_collection(self):
        """Start background metrics collection"""

        if self.collection_thread and self.collection_thread.is_alive():
            return

        self.stop_collection.clear()
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()

        logger.info(f"Started metrics collection (interval: {self.collection_interval}s)")

    def stop_collection_thread(self):
        """Stop background metrics collection"""

        self.stop_collection.set()
        if self.collection_thread:
            self.collection_thread.join(timeout=5)

        logger.info("Stopped metrics collection")

    def _collection_loop(self):
        """Background collection loop"""

        while not self.stop_collection.is_set():
            self.collect_system_metrics()

            # Wait for next collection interval
            self.stop_collection.wait(self.collection_interval)

    def export_metrics(
        self, format: str = "json", since: Optional[datetime] = None, include_system: bool = True
    ) -> Dict[str, Any]:
        """Export metrics in specified format"""

        if since is None:
            since = datetime.now(timezone.utc) - timedelta(hours=1)

        exported_data = {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "export_range_start": since.isoformat(),
            "format": format,
            "metrics": {},
            "performance": self.performance_metrics.copy(),
        }

        for metric_name, metric_def in self.metrics.items():
            if not include_system and metric_name.startswith("system."):
                continue

            values = self.get_metric_values(metric_name, since)
            aggregates = self.get_metric_aggregates(metric_name, since)

            exported_data["metrics"][metric_name] = {
                "definition": {
                    "type": metric_def.metric_type.value,
                    "description": metric_def.description,
                    "unit": metric_def.unit,
                },
                "aggregates": aggregates,
                "recent_values": [v.to_dict() for v in values[-10:]],  # Last 10 values
            }

        return exported_data

    def save_metrics(self, filepath: Optional[Path] = None):
        """Save metrics to file"""

        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.storage_dir / f"metrics_export_{timestamp}.json"

        exported_data = self.export_metrics()

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(exported_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Metrics saved to: {filepath}")
        return filepath

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data formatted for dashboard display"""

        since = datetime.now(timezone.utc) - timedelta(minutes=30)

        dashboard_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_health": {},
            "pipeline_performance": {},
            "error_rates": {},
            "cache_performance": {},
        }

        # System health
        system_metrics = ["system.cpu_percent", "system.memory_mb", "system.disk_percent"]
        for metric in system_metrics:
            aggregates = self.get_metric_aggregates(metric, since)
            if aggregates:
                dashboard_data["system_health"][metric] = aggregates["latest"]

        # Pipeline performance
        pipeline_metrics = [
            "pipeline.duration_ms",
            "pipeline.success_rate",
            "pipeline.claims_generated",
        ]
        for metric in pipeline_metrics:
            aggregates = self.get_metric_aggregates(metric, since)
            if aggregates:
                dashboard_data["pipeline_performance"][metric] = {
                    "avg": aggregates.get("avg", 0),
                    "latest": aggregates.get("latest", 0),
                    "count": aggregates.get("count", 0),
                }

        # Error rates
        error_aggregates = self.get_metric_aggregates("errors.total", since)
        if error_aggregates:
            dashboard_data["error_rates"]["total_errors"] = error_aggregates["sum"]
            dashboard_data["error_rates"]["error_rate_per_min"] = (
                error_aggregates["sum"] / 30
            )  # 30 min window

        # Cache performance
        cache_hits = self.get_metric_aggregates("cache.hits", since)
        cache_misses = self.get_metric_aggregates("cache.misses", since)

        if cache_hits and cache_misses:
            total_cache_ops = cache_hits["sum"] + cache_misses["sum"]
            cache_hit_rate = cache_hits["sum"] / total_cache_ops if total_cache_ops > 0 else 0
            dashboard_data["cache_performance"]["hit_rate"] = cache_hit_rate
            dashboard_data["cache_performance"]["total_operations"] = total_cache_ops

        return dashboard_data


class MetricTimer:
    """Context manager for timing operations"""

    def __init__(
        self, collector: MetricsCollector, name: str, tags: Optional[Dict[str, str]] = None
    ):
        self.collector = collector
        self.name = name
        self.tags = tags or {}
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            self.collector.record_timer(self.name, duration_ms, self.tags)


def create_metrics_collector(config: Dict[str, Any]) -> MetricsCollector:
    """Factory function for metrics collector"""
    return MetricsCollector(config)


# Usage example
if __name__ == "__main__":
    config = {
        "observability": {"metrics": {"enabled": True, "collection_interval_seconds": 10}},
        "metrics_storage": "test_metrics",
    }

    collector = MetricsCollector(config)

    # Record some test metrics
    collector.increment_counter("pipeline.executions", {"profile": "thorough"})
    collector.set_gauge("pipeline.success_rate", 95.5, {"profile": "thorough"})
    collector.record_histogram("pipeline.claims_generated", 5, {"profile": "thorough"})

    # Use timer
    with collector.timer("pipeline.duration_ms", {"profile": "thorough"}):
        time.sleep(0.1)  # Simulate work

    # Wait for system metrics collection
    time.sleep(2)

    # Get dashboard data
    dashboard = collector.get_dashboard_data()
    print(f"Dashboard data: {len(dashboard)} sections")

    # Export metrics
    exported = collector.export_metrics()
    print(f"Exported {len(exported['metrics'])} metrics")

    # Save to file
    saved_file = collector.save_metrics()
    print(f"Metrics saved to: {saved_file}")

    # Stop collection
    collector.stop_collection_thread()
