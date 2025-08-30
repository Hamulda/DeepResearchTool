"""🔧 Logging utilities pro Deep Research Tool
Poskytuje konzistentní logging napříč celým projektem
"""

from datetime import datetime
import logging
from pathlib import Path
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Vytvoří a nakonfiguruje logger pro modul

    Args:
        name: Název loggeru (obvykle __name__)
        level: Úroveň logování

    Returns:
        logging.Logger: Nakonfigurovaný logger

    """
    logger = logging.getLogger(name)

    # Pokud logger už má handlers, nekonfikguruj znovu
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # Prevence duplikování logů
    logger.propagate = False

    return logger


def setup_file_logging(log_file: str | None = None, level: int = logging.INFO) -> None:
    """Nastaví logování do souboru pro celý projekt

    Args:
        log_file: Cesta k log souboru (default: logs/app.log)
        level: Úroveň logování

    """
    if log_file is None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "app.log"

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)

    # Detailed formatter pro soubory
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # Přidání do root loggeru
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(level)


def get_research_logger(component: str) -> logging.Logger:
    """Vytvoří specializovaný logger pro výzkumné komponenty

    Args:
        component: Název komponenty (např. 'agent', 'scraper', 'analyzer')

    Returns:
        logging.Logger: Nakonfigurovaný logger

    """
    logger_name = f"research.{component}"
    return get_logger(logger_name)


def log_task_execution(
    logger: logging.Logger,
    task_id: str,
    task_type: str,
    start_time: datetime,
    end_time: datetime,
    success: bool,
) -> None:
    """Loguje vykonání úkolu s detailními informacemi

    Args:
        logger: Logger instance
        task_id: ID úkolu
        task_type: Typ úkolu
        start_time: Čas začátku
        end_time: Čas konce
        success: Zda úkol uspěl

    """
    duration = (end_time - start_time).total_seconds()
    status = "SUCCESS" if success else "FAILED"

    logger.info(
        f"TASK_EXECUTION | ID: {task_id} | Type: {task_type} | "
        f"Status: {status} | Duration: {duration:.2f}s"
    )


def log_agent_iteration(
    logger: logging.Logger,
    iteration: int,
    total_iterations: int,
    tasks_completed: int,
    avg_credibility: float,
) -> None:
    """Loguje iteraci autonomního agenta

    Args:
        logger: Logger instance
        iteration: Číslo aktuální iterace
        total_iterations: Celkový počet iterací
        tasks_completed: Počet dokončených úkolů
        avg_credibility: Průměrná důvěryhodnost

    """
    logger.info(
        f"AGENT_ITERATION | {iteration}/{total_iterations} | "
        f"Tasks: {tasks_completed} | Credibility: {avg_credibility:.2f}"
    )


def log_discovery(
    logger: logging.Logger, entity_type: str, entity_value: str, credibility: float, source: str
) -> None:
    """Loguje objevení nové entity

    Args:
        logger: Logger instance
        entity_type: Typ entity
        entity_value: Hodnota entity
        credibility: Důvěryhodnost
        source: Zdroj objevení

    """
    logger.info(
        f"DISCOVERY | Type: {entity_type} | Value: {entity_value[:50]}... | "
        f"Credibility: {credibility:.2f} | Source: {source}"
    )


def configure_production_logging() -> None:
    """Konfiguruje logování pro produkční prostředí"""
    # Strukturované logování pro produkci
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/production.log", encoding="utf-8"),
        ],
    )

    # Nastavení úrovní pro různé moduly
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def configure_debug_logging() -> None:
    """Konfiguruje podrobné logování pro debug"""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/debug.log", encoding="utf-8"),
        ],
    )


class LogContext:
    """Context manager pro logování s dodatečným kontextem"""

    def __init__(self, logger: logging.Logger, context: str):
        self.logger = logger
        self.context = context
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"START | {self.context}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        if exc_type is None:
            self.logger.info(f"COMPLETE | {self.context} | Duration: {duration:.2f}s")
        else:
            self.logger.error(
                f"FAILED | {self.context} | Duration: {duration:.2f}s | Error: {exc_val}"
            )

    def log(self, message: str, level: int = logging.INFO):
        """Loguje zprávu s kontextem"""
        self.logger.log(level, f"{self.context} | {message}")


# Convenience funkce pro rychlé použití
def info(message: str, component: str = "main") -> None:
    """Rychlé info logování"""
    logger = get_research_logger(component)
    logger.info(message)


def warning(message: str, component: str = "main") -> None:
    """Rychlé warning logování"""
    logger = get_research_logger(component)
    logger.warning(message)


def error(message: str, component: str = "main") -> None:
    """Rychlé error logování"""
    logger = get_research_logger(component)
    logger.error(message)
