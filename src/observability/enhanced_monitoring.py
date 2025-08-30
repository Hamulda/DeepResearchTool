#!/usr/bin/env python3
"""
Enhanced Monitoring and Observability System for Deep Research Tool
Comprehensive metrics, alerting, and operational visibility
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import psutil
import json
from enum import Enum

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SystemMetric:
    """System performance metric"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    labels: Dict[str, str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels or {}
        }


@dataclass
class SecurityEvent:
    """Security monitoring event"""
    event_type: str
    severity: AlertSeverity
    description: str
    source_ip: Optional[str]
    user_agent: Optional[str]
    details: Dict[str, Any]
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class MetricsCollector:
    """Enhanced metrics collection system"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_buffer = deque(maxlen=10000)  # Keep last 10k metrics
        self.alerts_buffer = deque(maxlen=1000)   # Keep last 1k alerts
        self.collection_interval = config.get("metrics_interval", 30)  # seconds
        self.running = False

        # Performance thresholds
        self.thresholds = {
            "cpu_usage": config.get("thresholds", {}).get("cpu_usage", 80.0),
            "memory_usage": config.get("thresholds", {}).get("memory_usage", 85.0),
            "disk_usage": config.get("thresholds", {}).get("disk_usage", 90.0),
            "response_time": config.get("thresholds", {}).get("response_time", 5.0),
            "error_rate": config.get("thresholds", {}).get("error_rate", 5.0)
        }

        # Application-specific metrics
        self.app_metrics = {
            "tasks_processed": 0,
            "tasks_failed": 0,
            "active_workers": 0,
            "queue_size": 0,
            "tor_circuits_active": 0,
            "scraping_success_rate": 0.0,
            "data_processed_mb": 0.0
        }

    async def start_collection(self):
        """Start metrics collection loop"""
        self.running = True
        logger.info("Starting enhanced metrics collection")

        while self.running:
            try:
                await self._collect_system_metrics()
                await self._collect_application_metrics()
                await self._check_thresholds()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(5)

    async def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
        logger.info("Stopped metrics collection")

    async def _collect_system_metrics(self):
        """Collect system performance metrics"""
        now = datetime.now()

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()

        self._add_metric("system_cpu_usage", cpu_percent, "percent", now)
        self._add_metric("system_cpu_count", cpu_count, "count", now)

        # Memory metrics
        memory = psutil.virtual_memory()
        self._add_metric("system_memory_usage", memory.percent, "percent", now)
        self._add_metric("system_memory_available", memory.available / (1024**3), "GB", now)
        self._add_metric("system_memory_total", memory.total / (1024**3), "GB", now)

        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self._add_metric("system_disk_usage", disk_percent, "percent", now)
        self._add_metric("system_disk_free", disk.free / (1024**3), "GB", now)

        # Network metrics
        network = psutil.net_io_counters()
        self._add_metric("system_network_bytes_sent", network.bytes_sent, "bytes", now)
        self._add_metric("system_network_bytes_recv", network.bytes_recv, "bytes", now)

        # Process metrics
        process = psutil.Process()
        self._add_metric("process_memory_usage", process.memory_percent(), "percent", now)
        self._add_metric("process_cpu_usage", process.cpu_percent(), "percent", now)
        self._add_metric("process_threads", process.num_threads(), "count", now)

    async def _collect_application_metrics(self):
        """Collect application-specific metrics"""
        now = datetime.now()

        # Add application metrics to buffer
        for metric_name, value in self.app_metrics.items():
            self._add_metric(f"app_{metric_name}", value, "count", now)

    def _add_metric(self, name: str, value: float, unit: str, timestamp: datetime, labels: Dict[str, str] = None):
        """Add metric to buffer"""
        metric = SystemMetric(name, value, unit, timestamp, labels)
        self.metrics_buffer.append(metric)

    async def _check_thresholds(self):
        """Check metrics against thresholds and generate alerts"""
        latest_metrics = {}

        # Get latest values for each metric
        for metric in reversed(self.metrics_buffer):
            if metric.name not in latest_metrics:
                latest_metrics[metric.name] = metric.value

        # Check CPU threshold
        cpu_usage = latest_metrics.get("system_cpu_usage", 0)
        if cpu_usage > self.thresholds["cpu_usage"]:
            await self._generate_alert(
                "high_cpu_usage",
                AlertSeverity.HIGH,
                f"CPU usage at {cpu_usage:.1f}% (threshold: {self.thresholds['cpu_usage']}%)",
                {"cpu_usage": cpu_usage}
            )

        # Check memory threshold
        memory_usage = latest_metrics.get("system_memory_usage", 0)
        if memory_usage > self.thresholds["memory_usage"]:
            await self._generate_alert(
                "high_memory_usage",
                AlertSeverity.HIGH,
                f"Memory usage at {memory_usage:.1f}% (threshold: {self.thresholds['memory_usage']}%)",
                {"memory_usage": memory_usage}
            )

        # Check disk threshold
        disk_usage = latest_metrics.get("system_disk_usage", 0)
        if disk_usage > self.thresholds["disk_usage"]:
            await self._generate_alert(
                "high_disk_usage",
                AlertSeverity.CRITICAL,
                f"Disk usage at {disk_usage:.1f}% (threshold: {self.thresholds['disk_usage']}%)",
                {"disk_usage": disk_usage}
            )

    async def _generate_alert(self, alert_type: str, severity: AlertSeverity, message: str, details: Dict[str, Any]):
        """Generate and store alert"""
        alert = {
            "type": alert_type,
            "severity": severity.value,
            "message": message,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }

        self.alerts_buffer.append(alert)
        logger.warning(f"ALERT [{severity.value.upper()}]: {message}")

    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get metrics summary for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_buffer if m.timestamp > cutoff_time]

        if not recent_metrics:
            return {"message": "No recent metrics available"}

        # Group metrics by name
        metric_groups = defaultdict(list)
        for metric in recent_metrics:
            metric_groups[metric.name].append(metric.value)

        # Calculate statistics
        summary = {}
        for name, values in metric_groups.items():
            summary[name] = {
                "current": values[-1] if values else 0,
                "average": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values)
            }

        return {
            "time_range_hours": hours,
            "metrics_count": len(recent_metrics),
            "metrics": summary,
            "alerts_count": len([a for a in self.alerts_buffer
                               if datetime.fromisoformat(a["timestamp"]) > cutoff_time])
        }

    def update_app_metric(self, metric_name: str, value: float):
        """Update application metric"""
        if metric_name in self.app_metrics:
            self.app_metrics[metric_name] = value


class SecurityMonitor:
    """Security monitoring and threat detection"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.security_events = deque(maxlen=5000)
        self.threat_patterns = self._load_threat_patterns()
        self.rate_limiters = defaultdict(lambda: deque(maxlen=100))

    def _load_threat_patterns(self) -> Dict[str, List[str]]:
        """Load threat detection patterns"""
        return {
            "sql_injection": [
                r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
                r"(--|/\*|\*/|;)",
                r"(\b(OR|AND)\s+\d+\s*=\s*\d+)"
            ],
            "xss_attempts": [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*="
            ],
            "path_traversal": [
                r"\.\./",
                r"\.\.\\",
                r"/etc/passwd",
                r"/proc/self/environ"
            ],
            "command_injection": [
                r"[;&|`$]",
                r"\b(cat|ls|pwd|whoami|id|uname)\b"
            ]
        }

    async def monitor_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor incoming request for security threats"""
        threat_score = 0.0
        detected_threats = []

        ip_address = request_data.get("ip_address", "unknown")
        user_agent = request_data.get("user_agent", "")
        url_path = request_data.get("path", "")
        query_params = request_data.get("query_params", {})

        # Rate limiting check
        now = time.time()
        self.rate_limiters[ip_address].append(now)

        # Check request rate
        recent_requests = [t for t in self.rate_limiters[ip_address] if now - t < 60]  # Last minute
        if len(recent_requests) > 100:  # More than 100 requests per minute
            threat_score += 50.0
            detected_threats.append("rate_limit_exceeded")

        # Check for threat patterns
        all_input = f"{url_path} {user_agent} {' '.join(str(v) for v in query_params.values())}"

        for threat_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                import re
                if re.search(pattern, all_input, re.IGNORECASE):
                    threat_score += 25.0
                    detected_threats.append(threat_type)
                    break

        # Generate security event if threats detected
        if threat_score > 25.0:
            severity = AlertSeverity.HIGH if threat_score > 50.0 else AlertSeverity.MEDIUM

            await self._log_security_event(
                "threat_detected",
                severity,
                f"Threat detected from {ip_address}",
                ip_address,
                user_agent,
                {
                    "threat_score": threat_score,
                    "detected_threats": detected_threats,
                    "request_path": url_path
                }
            )

        return {
            "threat_score": threat_score,
            "detected_threats": detected_threats,
            "action": "block" if threat_score > 75.0 else "monitor"
        }

    async def _log_security_event(self, event_type: str, severity: AlertSeverity,
                                 description: str, source_ip: str, user_agent: str,
                                 details: Dict[str, Any]):
        """Log security event"""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            description=description,
            source_ip=source_ip,
            user_agent=user_agent,
            details=details
        )

        self.security_events.append(event)
        logger.warning(f"SECURITY EVENT [{severity.value.upper()}]: {description}")

    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security events summary"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_events = [e for e in self.security_events if e.timestamp > cutoff_time]

        # Count by severity
        severity_counts = defaultdict(int)
        event_type_counts = defaultdict(int)

        for event in recent_events:
            severity_counts[event.severity.value] += 1
            event_type_counts[event.event_type] += 1

        return {
            "time_range_hours": hours,
            "total_events": len(recent_events),
            "severity_breakdown": dict(severity_counts),
            "event_type_breakdown": dict(event_type_counts),
            "recent_events": [
                {
                    "type": e.event_type,
                    "severity": e.severity.value,
                    "description": e.description,
                    "timestamp": e.timestamp.isoformat()
                }
                for e in list(recent_events)[-10:]  # Last 10 events
            ]
        }


class OperationalDashboard:
    """Operational dashboard with real-time metrics"""

    def __init__(self, metrics_collector: MetricsCollector, security_monitor: SecurityMonitor):
        self.metrics_collector = metrics_collector
        self.security_monitor = security_monitor

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        metrics_summary = self.metrics_collector.get_metrics_summary(hours=1)
        security_summary = self.security_monitor.get_security_summary(hours=1)

        # Determine overall health
        health_score = 100.0
        health_status = "healthy"
        issues = []

        # Check system metrics
        if "metrics" in metrics_summary:
            cpu_usage = metrics_summary["metrics"].get("system_cpu_usage", {}).get("current", 0)
            memory_usage = metrics_summary["metrics"].get("system_memory_usage", {}).get("current", 0)
            disk_usage = metrics_summary["metrics"].get("system_disk_usage", {}).get("current", 0)

            if cpu_usage > 80:
                health_score -= 20
                issues.append(f"High CPU usage: {cpu_usage:.1f}%")

            if memory_usage > 85:
                health_score -= 25
                issues.append(f"High memory usage: {memory_usage:.1f}%")

            if disk_usage > 90:
                health_score -= 30
                issues.append(f"High disk usage: {disk_usage:.1f}%")

        # Check security events
        critical_events = security_summary.get("severity_breakdown", {}).get("critical", 0)
        high_events = security_summary.get("severity_breakdown", {}).get("high", 0)

        if critical_events > 0:
            health_score -= 40
            issues.append(f"{critical_events} critical security events")

        if high_events > 5:
            health_score -= 20
            issues.append(f"{high_events} high-severity security events")

        # Determine status
        if health_score < 50:
            health_status = "critical"
        elif health_score < 70:
            health_status = "degraded"
        elif health_score < 90:
            health_status = "warning"

        return {
            "health_score": max(0, health_score),
            "status": health_status,
            "issues": issues,
            "timestamp": datetime.now().isoformat(),
            "system_metrics": metrics_summary,
            "security_summary": security_summary
        }

    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get performance dashboard data"""
        return {
            "system_health": self.get_system_health(),
            "worker_status": self._get_worker_status(),
            "queue_status": self._get_queue_status(),
            "tor_status": self._get_tor_status(),
            "recent_tasks": self._get_recent_tasks()
        }

    def _get_worker_status(self) -> Dict[str, Any]:
        """Get worker status information"""
        # This would integrate with actual worker monitoring
        return {
            "active_workers": 5,
            "idle_workers": 2,
            "busy_workers": 3,
            "failed_workers": 0,
            "average_task_time": 12.5
        }

    def _get_queue_status(self) -> Dict[str, Any]:
        """Get queue status information"""
        # This would integrate with Redis monitoring
        return {
            "acquisition_queue": 15,
            "processing_queue": 8,
            "completed_tasks": 1247,
            "failed_tasks": 23
        }

    def _get_tor_status(self) -> Dict[str, Any]:
        """Get Tor network status"""
        return {
            "circuits_active": 3,
            "exit_nodes": ["DE", "NL", "US"],
            "connection_quality": "good",
            "anonymity_level": "high"
        }

    def _get_recent_tasks(self) -> List[Dict[str, Any]]:
        """Get recent task information"""
        return [
            {
                "task_id": "task_001",
                "type": "url_scraping",
                "status": "completed",
                "duration": 8.5,
                "timestamp": datetime.now().isoformat()
            }
        ]


def create_monitoring_system(config: Dict[str, Any]) -> Dict[str, Any]:
    """Factory function to create monitoring components"""
    metrics_collector = MetricsCollector(config)
    security_monitor = SecurityMonitor(config)
    dashboard = OperationalDashboard(metrics_collector, security_monitor)

    return {
        "metrics_collector": metrics_collector,
        "security_monitor": security_monitor,
        "dashboard": dashboard
    }
