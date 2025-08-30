"""📊 Real-time monitoring systém pro autonomní agenta
Pokročilé metriky, alerting a performance tracking
"""

import asyncio
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import time
from typing import Any

import psutil

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SystemMetrics:
    """Systémové metriky"""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_io_bytes: int
    active_connections: int
    tor_status: bool
    vpn_status: bool


@dataclass
class AgentMetrics:
    """Metriky autonomní agenta"""

    timestamp: datetime
    active_tasks: int
    completed_tasks: int
    failed_tasks: int
    avg_credibility: float
    entities_discovered: int
    patterns_detected: int
    current_iteration: int
    queue_size: int


@dataclass
class Alert:
    """Systémový alert"""

    id: str
    timestamp: datetime
    level: str  # INFO, WARNING, ERROR, CRITICAL
    category: str  # SYSTEM, AGENT, SECURITY, PERFORMANCE
    message: str
    details: dict[str, Any]
    acknowledged: bool = False


class MetricsCollector:
    """📊 Kolector metrik s real-time monitoring
    """

    def __init__(self, collection_interval: int = 5):
        self.collection_interval = collection_interval
        self.is_collecting = False
        self.system_metrics: deque = deque(maxlen=1440)  # 12h při 30s intervalech
        self.agent_metrics: deque = deque(maxlen=1440)
        self.alerts: list[Alert] = []

        # Thresholdy pro alerting
        self.alert_thresholds = {
            "cpu_high": 80.0,
            "memory_high": 85.0,
            "disk_high": 90.0,
            "credibility_low": 0.3,
            "failure_rate_high": 0.4,
        }

        # Callbacks pro real-time aktualizace
        self.metric_callbacks: list[Callable] = []

    async def start_collection(self):
        """Spustí sběr metrik"""
        self.is_collecting = True
        logger.info("🚀 Spouštím sběr metrik")

        while self.is_collecting:
            try:
                # Sběr systémových metrik
                system_metrics = await self._collect_system_metrics()
                self.system_metrics.append(system_metrics)

                # Kontrola alertů
                await self._check_system_alerts(system_metrics)

                # Volání callbacks
                for callback in self.metric_callbacks:
                    try:
                        await callback(system_metrics)
                    except Exception as e:
                        logger.warning(f"Callback chyba: {e}")

                await asyncio.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"Chyba při sběru metrik: {e}")
                await asyncio.sleep(self.collection_interval)

    def stop_collection(self):
        """Zastaví sběr metrik"""
        self.is_collecting = False
        logger.info("🛑 Zastavujem sběr metrik")

    async def _collect_system_metrics(self) -> SystemMetrics:
        """Sbírá systémové metriky"""
        # CPU a paměť
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        # Disk
        disk = psutil.disk_usage("/")

        # Síť
        network = psutil.net_io_counters()
        network_bytes = network.bytes_sent + network.bytes_recv

        # Spojení
        connections = len(psutil.net_connections())

        # Status Tor/VPN (mock pro demo)
        tor_status = await self._check_tor_status()
        vpn_status = await self._check_vpn_status()

        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_usage_percent=disk.percent,
            network_io_bytes=network_bytes,
            active_connections=connections,
            tor_status=tor_status,
            vpn_status=vpn_status,
        )

    async def _check_tor_status(self) -> bool:
        """Kontroluje stav Tor připojení"""
        # V reálné implementaci by kontrolovalo Tor SOCKS proxy
        try:
            # Mock kontrola - v praxi by testovalo připojení na 127.0.0.1:9050
            return True
        except:
            return False

    async def _check_vpn_status(self) -> bool:
        """Kontroluje stav VPN připojení"""
        # V reálné implementaci by kontrolovalo VPN interface
        try:
            # Mock kontrola - v praxi by testovalo síťové rozhraní
            return True
        except:
            return False

    def record_agent_metrics(self, metrics: AgentMetrics):
        """Zaznamenává metriky agenta"""
        self.agent_metrics.append(metrics)

        # Kontrola alertů pro agenta
        asyncio.create_task(self._check_agent_alerts(metrics))

    async def _check_system_alerts(self, metrics: SystemMetrics):
        """Kontroluje systémové alerty"""
        # Vysoké využití CPU
        if metrics.cpu_percent > self.alert_thresholds["cpu_high"]:
            await self._create_alert(
                level="WARNING",
                category="SYSTEM",
                message=f"Vysoké využití CPU: {metrics.cpu_percent:.1f}%",
                details={"cpu_percent": metrics.cpu_percent},
            )

        # Vysoké využití paměti
        if metrics.memory_percent > self.alert_thresholds["memory_high"]:
            await self._create_alert(
                level="WARNING",
                category="SYSTEM",
                message=f"Vysoké využití paměti: {metrics.memory_percent:.1f}%",
                details={"memory_percent": metrics.memory_percent},
            )

        # Vysoké využití disku
        if metrics.disk_usage_percent > self.alert_thresholds["disk_high"]:
            await self._create_alert(
                level="ERROR",
                category="SYSTEM",
                message=f"Vysoké využití disku: {metrics.disk_usage_percent:.1f}%",
                details={"disk_percent": metrics.disk_usage_percent},
            )

        # Ztráta Tor připojení
        if not metrics.tor_status:
            await self._create_alert(
                level="CRITICAL",
                category="SECURITY",
                message="Tor připojení není dostupné!",
                details={"tor_status": False},
            )

        # Ztráta VPN připojení
        if not metrics.vpn_status:
            await self._create_alert(
                level="CRITICAL",
                category="SECURITY",
                message="VPN připojení není dostupné!",
                details={"vpn_status": False},
            )

    async def _check_agent_alerts(self, metrics: AgentMetrics):
        """Kontroluje alerty agenta"""
        # Nízká důvěryhodnost
        if metrics.avg_credibility < self.alert_thresholds["credibility_low"]:
            await self._create_alert(
                level="WARNING",
                category="AGENT",
                message=f"Nízká průměrná důvěryhodnost: {metrics.avg_credibility:.2f}",
                details={"avg_credibility": metrics.avg_credibility},
            )

        # Vysoká chybovost
        if metrics.failed_tasks > 0 and metrics.completed_tasks > 0:
            failure_rate = metrics.failed_tasks / (metrics.completed_tasks + metrics.failed_tasks)
            if failure_rate > self.alert_thresholds["failure_rate_high"]:
                await self._create_alert(
                    level="ERROR",
                    category="AGENT",
                    message=f"Vysoká chybovost agenta: {failure_rate:.2f}",
                    details={"failure_rate": failure_rate},
                )

        # Velká fronta úkolů
        if metrics.queue_size > 50:
            await self._create_alert(
                level="INFO",
                category="AGENT",
                message=f"Velká fronta úkolů: {metrics.queue_size}",
                details={"queue_size": metrics.queue_size},
            )

    async def _create_alert(self, level: str, category: str, message: str, details: dict[str, Any]):
        """Vytvoří nový alert"""
        # Kontrola duplicit (stejný alert za posledních 5 minut)
        now = datetime.now()
        recent_alerts = [
            a
            for a in self.alerts
            if (now - a.timestamp).total_seconds() < 300
            and a.message == message
            and not a.acknowledged
        ]

        if recent_alerts:
            return  # Neposílat duplicitní alerty

        alert = Alert(
            id=f"alert_{int(time.time())}_{len(self.alerts)}",
            timestamp=now,
            level=level,
            category=category,
            message=message,
            details=details,
        )

        self.alerts.append(alert)

        # Omezení počtu alertů
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]

        logger.info(f"🚨 {level} alert: {message}")

    def get_recent_metrics(self, minutes: int = 30) -> dict[str, Any]:
        """Vrátí nedávné metriky"""
        cutoff = datetime.now() - timedelta(minutes=minutes)

        recent_system = [m for m in self.system_metrics if m.timestamp >= cutoff]
        recent_agent = [m for m in self.agent_metrics if m.timestamp >= cutoff]

        return {
            "system_metrics": [asdict(m) for m in recent_system],
            "agent_metrics": [asdict(m) for m in recent_agent],
            "timespan_minutes": minutes,
            "system_points": len(recent_system),
            "agent_points": len(recent_agent),
        }

    def get_active_alerts(self, acknowledged: bool = False) -> list[Alert]:
        """Vrátí aktivní alerty"""
        return [alert for alert in self.alerts if alert.acknowledged == acknowledged]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Potvrdí alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                logger.info(f"Alert {alert_id} potvrzen")
                return True
        return False

    def get_system_health_score(self) -> float:
        """Vypočítá celkové skóre zdraví systému"""
        if not self.system_metrics:
            return 0.5

        latest = self.system_metrics[-1]
        score = 1.0

        # Penalizace za vysoké využití zdrojů
        score -= max(0, (latest.cpu_percent - 50) / 100)  # Penalizace nad 50%
        score -= max(0, (latest.memory_percent - 60) / 100)  # Penalizace nad 60%
        score -= max(0, (latest.disk_usage_percent - 80) / 100)  # Penalizace nad 80%

        # Penalizace za vypnuté bezpečnostní služby
        if not latest.tor_status:
            score -= 0.3
        if not latest.vpn_status:
            score -= 0.2

        # Penalizace za aktivní alerty
        critical_alerts = len([a for a in self.get_active_alerts() if a.level == "CRITICAL"])
        error_alerts = len([a for a in self.get_active_alerts() if a.level == "ERROR"])

        score -= critical_alerts * 0.2
        score -= error_alerts * 0.1

        return max(0.0, min(1.0, score))

    def add_metric_callback(self, callback: Callable):
        """Přidá callback pro real-time aktualizace"""
        self.metric_callbacks.append(callback)

    def export_metrics(self, filepath: str, format: str = "json"):
        """Exportuje metriky do souboru"""
        data = {
            "export_timestamp": datetime.now().isoformat(),
            "system_metrics": [asdict(m) for m in self.system_metrics],
            "agent_metrics": [asdict(m) for m in self.agent_metrics],
            "alerts": [asdict(a) for a in self.alerts],
        }

        path = Path(filepath)

        if format == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        else:
            raise ValueError(f"Nepodporovaný formát: {format}")

        logger.info(f"Metriky exportovány do {filepath}")


class PerformanceProfiler:
    """⚡ Performance profiler pro optimalizaci agenta
    """

    def __init__(self):
        self.execution_profiles: dict[str, list[float]] = defaultdict(list)
        self.memory_profiles: dict[str, list[float]] = defaultdict(list)
        self.bottleneck_analysis: dict[str, Any] = {}

    def profile_task_execution(self, task_type: str, execution_time: float, memory_usage: float):
        """Profiluje vykonání úkolu"""
        self.execution_profiles[task_type].append(execution_time)
        self.memory_profiles[task_type].append(memory_usage)

        # Omezení historie
        if len(self.execution_profiles[task_type]) > 100:
            self.execution_profiles[task_type] = self.execution_profiles[task_type][-100:]
            self.memory_profiles[task_type] = self.memory_profiles[task_type][-100:]

    def analyze_performance_patterns(self) -> dict[str, Any]:
        """Analyzuje performance vzory"""
        analysis = {}

        for task_type, times in self.execution_profiles.items():
            if len(times) < 5:
                continue

            analysis[task_type] = {
                "avg_execution_time": sum(times) / len(times),
                "min_execution_time": min(times),
                "max_execution_time": max(times),
                "execution_variance": self._calculate_variance(times),
                "trend": self._calculate_trend(times),
                "sample_count": len(times),
            }

            # Memory analýza
            if task_type in self.memory_profiles:
                memory_times = self.memory_profiles[task_type]
                analysis[task_type]["avg_memory_usage"] = sum(memory_times) / len(memory_times)
                analysis[task_type]["max_memory_usage"] = max(memory_times)

        return analysis

    def _calculate_variance(self, values: list[float]) -> float:
        """Vypočítá variance"""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance

    def _calculate_trend(self, values: list[float]) -> str:
        """Vypočítá trend (improving/stable/degrading)"""
        if len(values) < 10:
            return "insufficient_data"

        # Porovnání první a druhé poloviny
        mid = len(values) // 2
        first_half_avg = sum(values[:mid]) / mid
        second_half_avg = sum(values[mid:]) / (len(values) - mid)

        diff_ratio = (second_half_avg - first_half_avg) / first_half_avg

        if diff_ratio < -0.1:
            return "improving"
        if diff_ratio > 0.1:
            return "degrading"
        return "stable"

    def identify_bottlenecks(self) -> list[dict[str, Any]]:
        """Identifikuje bottlenecky"""
        analysis = self.analyze_performance_patterns()
        bottlenecks = []

        for task_type, stats in analysis.items():
            # Bottleneck kritéria
            if (
                stats["avg_execution_time"] > 60  # Více než minutu
                or stats["execution_variance"] > 100  # Vysoká variance
                or stats["trend"] == "degrading"
            ):  # Zhoršující se trend

                bottlenecks.append(
                    {
                        "task_type": task_type,
                        "severity": self._calculate_bottleneck_severity(stats),
                        "avg_time": stats["avg_execution_time"],
                        "variance": stats["execution_variance"],
                        "trend": stats["trend"],
                        "recommendations": self._generate_optimization_recommendations(
                            task_type, stats
                        ),
                    }
                )

        # Řazení podle závažnosti
        bottlenecks.sort(key=lambda x: x["severity"], reverse=True)
        return bottlenecks

    def _calculate_bottleneck_severity(self, stats: dict[str, Any]) -> float:
        """Vypočítá závažnost bottlenecku"""
        severity = 0.0

        # Čas vykonání
        if stats["avg_execution_time"] > 120:
            severity += 0.4
        elif stats["avg_execution_time"] > 60:
            severity += 0.2

        # Variance
        if stats["execution_variance"] > 500:
            severity += 0.3
        elif stats["execution_variance"] > 100:
            severity += 0.1

        # Trend
        if stats["trend"] == "degrading":
            severity += 0.3
        elif stats["trend"] == "improving":
            severity -= 0.1

        return min(1.0, max(0.0, severity))

    def _generate_optimization_recommendations(
        self, task_type: str, stats: dict[str, Any]
    ) -> list[str]:
        """Generuje doporučení pro optimalizaci"""
        recommendations = []

        if stats["avg_execution_time"] > 60:
            recommendations.append("Zvážit paralelizaci nebo optimalizaci algoritmu")

        if stats["execution_variance"] > 100:
            recommendations.append("Vysoká variance - zkontrolovat externí závislosti")

        if stats["trend"] == "degrading":
            recommendations.append("Zhoršující se výkon - možná memory leak nebo fragmentace")

        if task_type == "scrape" and stats["avg_execution_time"] > 30:
            recommendations.append("Scraping trvá dlouho - zkontrolovat síťové připojení")

        if task_type == "analyze" and stats["avg_execution_time"] > 45:
            recommendations.append("Analýza je pomalá - optimalizovat NLP pipeline")

        return recommendations
