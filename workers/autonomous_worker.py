"""
Autonomous Worker - Phase 4
Kontinuální, proaktivní systém pro autonomní monitorování a analýzu
"""

import asyncio
import logging
import os
import json
import hashlib
import smtplib
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
from datetime import datetime, timezone, timedelta
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
import schedule
import time
import threading
from dataclasses import dataclass, asdict
from urllib.parse import urlparse

# Redis a Dramatiq setup s fallback
try:
    import redis
    import dramatiq
    from dramatiq.brokers.redis import RedisBroker

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = redis.from_url(redis_url)
    broker = RedisBroker(url=redis_url)
    dramatiq.set_broker(broker)
    DRAMATIQ_AVAILABLE = True
    logger.info("✅ Redis/Dramatiq dostupné")
except ImportError as e:
    DRAMATIQ_AVAILABLE = False
    logger.warning(f"⚠️ Redis/Dramatiq není dostupný: {e}")

    # Mock objekty pro lokální testování
    class MockRedisClient:
        def from_url(self, url):
            return self

    class MockBroker:
        def __init__(self, url):
            pass

    class MockDramatiq:
        def set_broker(self, broker):
            pass

        def actor(self, queue_name=None):
            def decorator(func):
                return func

            return decorator

    redis_client = MockRedisClient()
    broker = MockBroker("")
    dramatiq = MockDramatiq()

# Import workers
import sys

# Přidej lokální cestu místo Docker cesty
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from acquisition_worker import scrape_url

    ACQUISITION_WORKER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Acquisition worker není dostupný: {e}")
    ACQUISITION_WORKER_AVAILABLE = False

try:
    from processing_worker import process_scraped_data_enhanced, process_with_knowledge_graph

    PROCESSING_WORKER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Processing worker není dostupný: {e}")
    PROCESSING_WORKER_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MonitoredSource:
    """Reprezentace monitorovaného zdroje"""

    url: str
    check_interval: int  # v minutách
    last_check: Optional[datetime] = None
    last_content_hash: Optional[str] = None
    last_etag: Optional[str] = None
    last_modified: Optional[str] = None
    priority: int = 1  # 1=nízká, 5=vysoká
    enabled: bool = True
    keywords: List[str] = None  # klíčová slova pro upozornění
    task_id: Optional[str] = None


@dataclass
class AlertRule:
    """Pravidlo pro upozornění"""

    rule_id: str
    name: str
    condition_type: str  # "keyword_mention", "entity_connection", "new_source"
    condition_params: Dict[str, Any]
    notification_channels: List[str]  # "email", "file", "telegram"
    enabled: bool = True
    created_at: datetime = None


@dataclass
class Alert:
    """Vygenerované upozornění"""

    alert_id: str
    rule_id: str
    title: str
    message: str
    severity: str  # "low", "medium", "high", "critical"
    source_url: str
    triggered_at: datetime
    sent: bool = False


class ChangeDetector:
    """Detektor změn na webových stránkách"""

    def __init__(self):
        # Použij lokální cestu místo Docker cesty
        base_dir = Path.cwd() / "data"
        self.data_dir = base_dir
        self.cache_dir = base_dir / "cache"
        self.data_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)

    def check_page_changes(self, source: MonitoredSource) -> Dict[str, Any]:
        """
        Zkontroluj změny na stránce s optimalizovaným přístupem
        """
        try:
            logger.info(f"🔍 Kontroluji změny: {source.url}")

            headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}

            # Přidej podmíněné hlavičky pokud jsou dostupné
            if source.last_etag:
                headers["If-None-Match"] = source.last_etag
            if source.last_modified:
                headers["If-Modified-Since"] = source.last_modified

            # Nejprve zkus HEAD request
            try:
                head_response = requests.head(source.url, headers=headers, timeout=30)

                # Pokud 304 Not Modified, žádné změny
                if head_response.status_code == 304:
                    logger.info(f"✅ Žádné změny (304): {source.url}")
                    return {"changed": False, "reason": "not_modified_304", "status_code": 304}

                # Zkontroluj ETag a Last-Modified
                current_etag = head_response.headers.get("ETag")
                current_modified = head_response.headers.get("Last-Modified")

                if current_etag and source.last_etag and current_etag == source.last_etag:
                    logger.info(f"✅ Žádné změny (ETag): {source.url}")
                    return {"changed": False, "reason": "etag_match", "etag": current_etag}

            except requests.RequestException as e:
                logger.warning(f"⚠️ HEAD request selhal pro {source.url}: {e}")
                # Pokračuj s GET requestem

            # Pokud HEAD nedorazil nebo naznačuje změny, stáhni obsah
            response = requests.get(source.url, headers=headers, timeout=30)
            response.raise_for_status()

            # Vypočítej hash obsahu
            content_hash = hashlib.sha256(response.content).hexdigest()

            # Porovnej s předchozím hashem
            if source.last_content_hash and content_hash == source.last_content_hash:
                logger.info(f"✅ Žádné změny (content hash): {source.url}")
                return {
                    "changed": False,
                    "reason": "content_hash_match",
                    "content_hash": content_hash,
                }

            # Změny detekované!
            logger.info(f"🔥 Změny detekované: {source.url}")

            return {
                "changed": True,
                "content": response.text,
                "content_hash": content_hash,
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
                "status_code": response.status_code,
                "size": len(response.content),
            }

        except Exception as e:
            logger.error(f"❌ Chyba při kontrole {source.url}: {e}")
            return {"changed": False, "error": str(e)}

    def update_source_metadata(self, source: MonitoredSource, check_result: Dict[str, Any]):
        """Aktualizuj metadata zdroje po kontrole"""
        source.last_check = datetime.now(timezone.utc)

        if not check_result.get("error"):
            if check_result.get("content_hash"):
                source.last_content_hash = check_result["content_hash"]
            if check_result.get("etag"):
                source.last_etag = check_result["etag"]
            if check_result.get("last_modified"):
                source.last_modified = check_result["last_modified"]


class EventDrivenAnalyzer:
    """Analyzér pro event-driven analýzy"""

    def __init__(self):
        # Použij lokální cestu místo Docker cesty
        self.data_dir = Path.cwd() / "data"
        self.data_dir.mkdir(exist_ok=True)

    async def analyze_new_content(self, source: MonitoredSource, content: str) -> Dict[str, Any]:
        """
        Analyzuj nový obsah a spusť navazující úlohy
        """
        try:
            logger.info(f"🧠 Event-driven analýza: {source.url}")

            # Vytvoř dočasný task ID
            task_id = f"autonomous_{int(datetime.now().timestamp())}"

            # Spusť scraping úlohu pro nový obsah
            scrape_result = await self._trigger_scraping(source.url, task_id, content)

            if not scrape_result.get("success"):
                return {"success": False, "error": f"Scraping failed: {scrape_result.get('error')}"}

            # Spusť zpracování s Knowledge Graph
            processing_result = await self._trigger_processing(
                scrape_result["output_file"], task_id, use_knowledge_graph=True
            )

            # Analyzuj výsledky pro potenciální upozornění
            alerts = await self._analyze_for_alerts(processing_result, source)

            return {
                "success": True,
                "task_id": task_id,
                "scrape_result": scrape_result,
                "processing_result": processing_result,
                "alerts_generated": len(alerts),
                "alerts": alerts,
            }

        except Exception as e:
            logger.error(f"❌ Event-driven analýza selhala: {e}")
            return {"success": False, "error": str(e)}

    async def _trigger_scraping(self, url: str, task_id: str, content: str) -> Dict[str, Any]:
        """Spusť scraping úlohu"""
        try:
            # Přímo zpracuj obsah místo opětovného stahování
            temp_file = self.data_dir / f"temp_content_{task_id}.html"
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(content)

            # Vytvoř parquet se základními daty
            data = [
                {
                    "url": url,
                    "content": content,
                    "metadata": json.dumps(
                        {
                            "scraped_at": datetime.now(timezone.utc).isoformat(),
                            "autonomous": True,
                            "task_id": task_id,
                        }
                    ),
                    "task_id": task_id,
                }
            ]

            df = pl.DataFrame(data)
            output_file = self.data_dir / f"scraped_autonomous_{task_id}.parquet"
            df.write_parquet(output_file)

            return {"success": True, "output_file": str(output_file), "records": 1}

        except Exception as e:
            logger.error(f"❌ Autonomous scraping error: {e}")
            return {"success": False, "error": str(e)}

    async def _trigger_processing(
        self, file_path: str, task_id: str, use_knowledge_graph: bool = True
    ) -> Dict[str, Any]:
        """Spusť zpracování dat"""
        try:
            if use_knowledge_graph:
                # Spusť processing s Knowledge Graph
                result = process_with_knowledge_graph.send(file_path, task_id)
                return {"success": True, "knowledge_graph": True, "dramatiq_result": str(result)}
            else:
                # Spusť standardní enhanced processing
                result = process_scraped_data_enhanced.send(file_path, task_id)
                return {"success": True, "knowledge_graph": False, "dramatiq_result": str(result)}

        except Exception as e:
            logger.error(f"❌ Processing trigger error: {e}")
            return {"success": False, "error": str(e)}

    async def _analyze_for_alerts(
        self, processing_result: Dict[str, Any], source: MonitoredSource
    ) -> List[Alert]:
        """Analyzuj výsledky zpracování pro potenciální upozornění"""
        alerts = []

        try:
            # Zkontroluj klíčová slova pokud jsou definována
            if source.keywords:
                # Toto by mělo číst z výsledků zpracování
                # Pro demo účely vytvoříme základní alert
                alert = Alert(
                    alert_id=f"alert_{int(datetime.now().timestamp())}",
                    rule_id="keyword_watch",
                    title=f"Změny na monitorované stránce: {source.url}",
                    message=f"Detekována změna na {source.url}. Spuštěna automatická analýza.",
                    severity="medium",
                    source_url=source.url,
                    triggered_at=datetime.now(timezone.utc),
                )
                alerts.append(alert)

            return alerts

        except Exception as e:
            logger.error(f"❌ Alert analysis error: {e}")
            return []


class AlertManager:
    """Správce upozornění"""

    def __init__(self):
        # Použij lokální cestu místo Docker cesty
        base_dir = Path.cwd() / "data"
        self.data_dir = base_dir
        self.alerts_file = self.data_dir / "alerts.json"
        self.rules_file = self.data_dir / "alert_rules.json"
        self.data_dir.mkdir(exist_ok=True)

        # Email konfigurace
        self.smtp_server = os.getenv("SMTP_SERVER", "localhost")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.email_user = os.getenv("EMAIL_USER", "")
        self.email_password = os.getenv("EMAIL_PASSWORD", "")
        self.email_from = os.getenv("EMAIL_FROM", "research-bot@localhost")
        self.email_to = os.getenv("EMAIL_TO", "admin@localhost")

        # Telegram konfigurace
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

        self._load_rules()

    def _load_rules(self):
        """Načti pravidla pro upozornění"""
        try:
            if self.rules_file.exists():
                with open(self.rules_file, "r") as f:
                    rules_data = json.load(f)
                    self.rules = [AlertRule(**rule) for rule in rules_data]
            else:
                # Vytvoř základní pravidla
                self.rules = [
                    AlertRule(
                        rule_id="change_detected",
                        name="Změna na monitorované stránce",
                        condition_type="page_change",
                        condition_params={},
                        notification_channels=["file", "email"],
                        created_at=datetime.now(timezone.utc),
                    )
                ]
                self._save_rules()

        except Exception as e:
            logger.error(f"❌ Chyba při načítání pravidel: {e}")
            self.rules = []

    def _save_rules(self):
        """Ulož pravidla"""
        try:
            with open(self.rules_file, "w") as f:
                json.dump([asdict(rule) for rule in self.rules], f, indent=2, default=str)
        except Exception as e:
            logger.error(f"❌ Chyba při ukládání pravidel: {e}")

    async def send_alert(self, alert: Alert) -> bool:
        """Pošli upozornění podle konfigurace"""
        success = False

        try:
            # Ulož alert do souboru
            if await self._send_file_alert(alert):
                success = True

            # Pošli email pokud je nakonfigurován
            if self.email_user and await self._send_email_alert(alert):
                success = True

            # Pošli Telegram zprávu pokud je nakonfigurováno
            if self.telegram_bot_token and await self._send_telegram_alert(alert):
                success = True

            alert.sent = success

            return success

        except Exception as e:
            logger.error(f"❌ Chyba při odesílání alertu: {e}")
            return False

    async def _send_file_alert(self, alert: Alert) -> bool:
        """Ulož alert do souboru"""
        try:
            alerts = []

            # Načti existující alerty
            if self.alerts_file.exists():
                with open(self.alerts_file, "r") as f:
                    alerts = json.load(f)

            # Přidej nový alert
            alerts.append(asdict(alert))

            # Ulož zpět
            with open(self.alerts_file, "w") as f:
                json.dump(alerts, f, indent=2, default=str)

            logger.info(f"📝 Alert uložen do souboru: {alert.title}")
            return True

        except Exception as e:
            logger.error(f"❌ File alert error: {e}")
            return False

    async def _send_email_alert(self, alert: Alert) -> bool:
        """Pošli email alert"""
        try:
            if not self.email_user:
                return False

            msg = MimeMultipart()
            msg["From"] = self.email_from
            msg["To"] = self.email_to
            msg["Subject"] = f"[Research Alert] {alert.title}"

            body = f"""
Upozornění z autonomního výzkumného systému:

Název: {alert.title}
Závažnost: {alert.severity}
Zdroj: {alert.source_url}
Čas: {alert.triggered_at}

Zpráva:
{alert.message}

---
Automaticky generováno výzkumným botem
            """

            msg.attach(MimeText(body, "plain", "utf-8"))

            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_user, self.email_password)
            server.send_message(msg)
            server.quit()

            logger.info(f"📧 Email alert odeslán: {alert.title}")
            return True

        except Exception as e:
            logger.error(f"❌ Email alert error: {e}")
            return False

    async def _send_telegram_alert(self, alert: Alert) -> bool:
        """Pošli Telegram alert"""
        try:
            if not self.telegram_bot_token:
                return False

            message = f"""
🔔 *Research Alert*

📌 {alert.title}
⚠️ Závažnost: {alert.severity}
🌐 Zdroj: {alert.source_url}
🕐 Čas: {alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S')}

📝 {alert.message}
            """

            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            data = {"chat_id": self.telegram_chat_id, "text": message, "parse_mode": "Markdown"}

            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()

            logger.info(f"📱 Telegram alert odeslán: {alert.title}")
            return True

        except Exception as e:
            logger.error(f"❌ Telegram alert error: {e}")
            return False


class AutonomousWorker:
    """Hlavní autonomní worker"""

    def __init__(self):
        # Použij lokální cestu místo Docker cesty
        base_dir = Path.cwd() / "data"
        self.data_dir = base_dir
        self.sources_file = self.data_dir / "monitored_sources.json"
        self.data_dir.mkdir(exist_ok=True)

        self.change_detector = ChangeDetector()
        self.event_analyzer = EventDrivenAnalyzer()
        self.alert_manager = AlertManager()

        self.monitored_sources: List[MonitoredSource] = []
        self.running = False

        self._load_sources()
        self._setup_scheduler()

        logger.info("🤖 Autonomous Worker inicializován")

    def _load_sources(self):
        """Načti monitorované zdroje"""
        try:
            if self.sources_file.exists():
                with open(self.sources_file, "r") as f:
                    sources_data = json.load(f)
                    self.monitored_sources = [MonitoredSource(**source) for source in sources_data]
                logger.info(f"✅ Načteno {len(self.monitored_sources)} monitorovaných zdrojů")
            else:
                # Vytvoř ukázkové zdroje
                self.monitored_sources = [
                    MonitoredSource(
                        url="https://news.ycombinator.com",
                        check_interval=60,  # každou hodinu
                        priority=3,
                        keywords=["security", "privacy", "crypto"],
                    )
                ]
                self._save_sources()

        except Exception as e:
            logger.error(f"❌ Chyba při načítání zdrojů: {e}")
            self.monitored_sources = []

    def _save_sources(self):
        """Ulož monitorované zdroje"""
        try:
            with open(self.sources_file, "w") as f:
                json.dump(
                    [asdict(source) for source in self.monitored_sources], f, indent=2, default=str
                )
        except Exception as e:
            logger.error(f"❌ Chyba při ukládání zdrojů: {e}")

    def _setup_scheduler(self):
        """Nastav plánovač úloh"""
        # Naplánuj kontroly podle priorit
        schedule.every(5).minutes.do(self._check_high_priority_sources)
        schedule.every(30).minutes.do(self._check_medium_priority_sources)
        schedule.every(2).hours.do(self._check_low_priority_sources)

        # Naplánuj údržbové úlohy
        schedule.every().day.at("02:00").do(self._cleanup_old_data)
        schedule.every().week.do(self._generate_summary_report)

        logger.info("⏰ Plánovač úloh nastaven")

    def _check_high_priority_sources(self):
        """Zkontroluj vysoce prioritní zdroje (priority >= 4)"""
        asyncio.create_task(self._check_sources_by_priority(4, 5))

    def _check_medium_priority_sources(self):
        """Zkontroluj středně prioritní zdroje (priority 2-3)"""
        asyncio.create_task(self._check_sources_by_priority(2, 3))

    def _check_low_priority_sources(self):
        """Zkontroluj nízko prioritní zdroje (priority 1)"""
        asyncio.create_task(self._check_sources_by_priority(1, 1))

    async def _check_sources_by_priority(self, min_priority: int, max_priority: int):
        """Zkontroluj zdroje podle priority"""
        try:
            sources_to_check = [
                source
                for source in self.monitored_sources
                if (
                    min_priority <= source.priority <= max_priority
                    and source.enabled
                    and self._should_check_source(source)
                )
            ]

            if not sources_to_check:
                return

            logger.info(
                f"🔍 Kontroluji {len(sources_to_check)} zdrojů (priorita {min_priority}-{max_priority})"
            )

            for source in sources_to_check:
                await self._check_single_source(source)

        except Exception as e:
            logger.error(f"❌ Chyba při kontrole zdrojů: {e}")

    def _should_check_source(self, source: MonitoredSource) -> bool:
        """Zjisti zda je čas zkontrolovat zdroj"""
        if not source.last_check:
            return True

        time_since_check = datetime.now(timezone.utc) - source.last_check
        return time_since_check.total_seconds() >= (source.check_interval * 60)

    async def _check_single_source(self, source: MonitoredSource):
        """Zkontroluj jednotlivý zdroj"""
        try:
            logger.info(f"🔍 Kontroluji zdroj: {source.url}")

            # Detekuj změny
            check_result = self.change_detector.check_page_changes(source)

            # Aktualizuj metadata
            self.change_detector.update_source_metadata(source, check_result)

            # Pokud jsou změny, spusť analýzu
            if check_result.get("changed") and check_result.get("content"):
                logger.info(f"🔥 Spouštím analýzu změn pro: {source.url}")

                analysis_result = await self.event_analyzer.analyze_new_content(
                    source, check_result["content"]
                )

                # Pošli upozornění
                if analysis_result.get("alerts"):
                    for alert in analysis_result["alerts"]:
                        await self.alert_manager.send_alert(alert)

                logger.info(f"✅ Analýza dokončena pro: {source.url}")

            # Ulož aktualizované zdroje
            self._save_sources()

        except Exception as e:
            logger.error(f"❌ Chyba při kontrole zdroje {source.url}: {e}")

    def _cleanup_old_data(self):
        """Vyčisti stará data"""
        try:
            logger.info("🧹 Spouštím cleanup starých dat")

            # Vymaž soubory starší než 30 dní
            cutoff_date = datetime.now() - timedelta(days=30)

            for file_path in self.data_dir.glob("*.parquet"):
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    file_path.unlink()
                    logger.info(f"🗑️ Vymazán starý soubor: {file_path}")

        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

    def _generate_summary_report(self):
        """Generuj týdenní souhrn"""
        try:
            logger.info("📊 Generuji týdenní souhrn")

            # Přečti alerty z posledního týdne
            week_ago = datetime.now(timezone.utc) - timedelta(days=7)

            recent_alerts = []
            if self.alert_manager.alerts_file.exists():
                with open(self.alert_manager.alerts_file, "r") as f:
                    all_alerts = json.load(f)
                    for alert_data in all_alerts:
                        alert_time = datetime.fromisoformat(
                            alert_data["triggered_at"].replace("Z", "+00:00")
                        )
                        if alert_time >= week_ago:
                            recent_alerts.append(alert_data)

            summary = {
                "period": f"{week_ago.date()} - {datetime.now().date()}",
                "monitored_sources": len(self.monitored_sources),
                "active_sources": len([s for s in self.monitored_sources if s.enabled]),
                "alerts_generated": len(recent_alerts),
                "alerts_by_severity": {},
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

            # Spočítej alerty podle závažnosti
            for alert in recent_alerts:
                severity = alert.get("severity", "unknown")
                summary["alerts_by_severity"][severity] = (
                    summary["alerts_by_severity"].get(severity, 0) + 1
                )

            # Ulož souhrn
            summary_file = (
                self.data_dir / f"weekly_summary_{datetime.now().strftime('%Y_%m_%d')}.json"
            )
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)

            logger.info(f"📊 Týdenní souhrn uložen: {summary_file}")

        except Exception as e:
            logger.error(f"❌ Summary generation error: {e}")

    def add_monitored_source(
        self, url: str, check_interval: int = 60, priority: int = 1, keywords: List[str] = None
    ) -> bool:
        """Přidej nový monitorovaný zdroj"""
        try:
            source = MonitoredSource(
                url=url, check_interval=check_interval, priority=priority, keywords=keywords or []
            )

            self.monitored_sources.append(source)
            self._save_sources()

            logger.info(f"✅ Přidán nový monitorovaný zdroj: {url}")
            return True

        except Exception as e:
            logger.error(f"❌ Chyba při přidávání zdroje: {e}")
            return False

    def start(self):
        """Spusť autonomní worker"""
        self.running = True
        logger.info("🚀 Autonomous Worker spuštěn")

        def run_scheduler():
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # Kontroluj každou minutu

        # Spusť plánovač v separátním threadu
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()

        logger.info("⏰ Plánovač spuštěn v background threadu")

    def stop(self):
        """Zastav autonomní worker"""
        self.running = False
        logger.info("🛑 Autonomous Worker zastaven")


# Dramatiq actors pro autonomní operace
@dramatiq.actor(queue_name="autonomous")
def check_monitored_sources() -> Dict[str, Any]:
    """Dramatiq actor pro kontrolu monitorovaných zdrojů"""
    worker = AutonomousWorker()
    # Spusť jednorázovou kontrolu všech zdrojů
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(worker._check_sources_by_priority(1, 5))
    finally:
        loop.close()


@dramatiq.actor(queue_name="autonomous")
def add_monitored_source(
    url: str, check_interval: int = 60, priority: int = 1, keywords: List[str] = None
) -> bool:
    """Dramatiq actor pro přidání monitorovaného zdroje"""
    worker = AutonomousWorker()
    return worker.add_monitored_source(url, check_interval, priority, keywords)


# Global autonomous worker instance
autonomous_worker = AutonomousWorker()

logger.info("🤖 Autonomous Worker připraven pro Phase 4!")
