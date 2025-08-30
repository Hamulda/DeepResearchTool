#!/usr/bin/env python3
"""Strukturované logování pomocí structlog
Nahrazuje všechna print() a základní logging volání

Author: Senior Python/MLOps Agent
"""

from datetime import datetime
import logging
from pathlib import Path
import sys

import structlog
from structlog.processors import CallsiteParameterAdder, JSONRenderer, TimeStamper

from ..config.settings import get_settings


def configure_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: str | None = None,
    enable_console: bool = True,
) -> None:
    """Konfigurace strukturovaného logování

    Args:
        log_level: Úroveň logování (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Formát logů (json, text)
        log_file: Cesta k log souboru (optional)
        enable_console: Povolit výstup na konzoli

    """
    settings = get_settings()

    # Nastavení úrovně logování
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Procesory pro structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        TimeStamper(fmt="ISO"),
        CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.PATHNAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
    ]

    # Konfigurace výstupních handlerů
    handlers = []

    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)

        if log_format == "json":
            processors.append(JSONRenderer())
            console_handler.setFormatter(logging.Formatter("%(message)s"))
        else:
            processors.append(structlog.dev.ConsoleRenderer(colors=True))
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )

        handlers.append(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)

        # Soubory vždy v JSON formátu pro lepší parsování
        file_processors = processors[:-1] + [JSONRenderer()]
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(file_handler)

    # Konfigurace standardního loggingu
    logging.basicConfig(level=numeric_level, handlers=handlers, format="%(message)s")

    # Konfigurace structlog
    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


class StructuredLogger:
    """Wrapper pro strukturované logování s kontextovými informacemi
    """

    def __init__(self, name: str, **context):
        self.logger = structlog.get_logger(name)
        self.context = context

        # Přidání základního kontextu
        if context:
            self.logger = self.logger.bind(**context)

    def debug(self, message: str, **kwargs) -> None:
        """Debug level log"""
        self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Info level log"""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Warning level log"""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Error level log"""
        self.logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Critical level log"""
        self.logger.critical(message, **kwargs)

    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback"""
        self.logger.exception(message, **kwargs)

    def bind(self, **kwargs) -> "StructuredLogger":
        """Vytvoří nový logger s dodatečným kontextem"""
        new_logger = StructuredLogger(self.logger.name)
        new_logger.logger = self.logger.bind(**kwargs)
        new_logger.context = {**self.context, **kwargs}
        return new_logger


class AuditLogger(StructuredLogger):
    """Specializovaný logger pro audit trail
    """

    def __init__(self, component: str):
        super().__init__("audit", component=component)

    def user_action(
        self, action: str, user_id: str | None = None, resource: str | None = None, **kwargs
    ) -> None:
        """Logování uživatelských akcí"""
        self.info(
            "User action performed",
            action=action,
            user_id=user_id,
            resource=resource,
            timestamp=datetime.utcnow().isoformat(),
            **kwargs
        )

    def security_event(self, event_type: str, severity: str, description: str, **kwargs) -> None:
        """Logování bezpečnostních událostí"""
        self.warning(
            "Security event detected",
            event_type=event_type,
            severity=severity,
            description=description,
            timestamp=datetime.utcnow().isoformat(),
            **kwargs
        )

    def data_access(
        self,
        operation: str,
        table_name: str | None = None,
        record_count: int | None = None,
        **kwargs
    ) -> None:
        """Logování přístupu k datům"""
        self.info(
            "Data access logged",
            operation=operation,
            table_name=table_name,
            record_count=record_count,
            timestamp=datetime.utcnow().isoformat(),
            **kwargs
        )


class PerformanceLogger(StructuredLogger):
    """Specializovaný logger pro výkonnostní metriky
    """

    def __init__(self, component: str):
        super().__init__("performance", component=component)

    def timing(self, operation: str, duration_ms: float, **kwargs) -> None:
        """Logování časování operací"""
        self.info("Operation timing", operation=operation, duration_ms=duration_ms, **kwargs)

    def resource_usage(
        self,
        cpu_percent: float | None = None,
        memory_mb: float | None = None,
        disk_io_mb: float | None = None,
        **kwargs
    ) -> None:
        """Logování využití zdrojů"""
        self.info(
            "Resource usage",
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            disk_io_mb=disk_io_mb,
            **kwargs
        )

    def throughput(
        self, operation: str, items_per_second: float, total_items: int, **kwargs
    ) -> None:
        """Logování propustnosti"""
        self.info(
            "Throughput measurement",
            operation=operation,
            items_per_second=items_per_second,
            total_items=total_items,
            **kwargs
        )


def get_logger(name: str, **context) -> StructuredLogger:
    """Factory funkce pro získání loggeru

    Args:
        name: Název loggeru
        **context: Dodatečný kontext

    Returns:
        StructuredLogger instance

    """
    return StructuredLogger(name, **context)


def get_audit_logger(component: str) -> AuditLogger:
    """Factory funkce pro audit logger

    Args:
        component: Název komponenty

    Returns:
        AuditLogger instance

    """
    return AuditLogger(component)


def get_performance_logger(component: str) -> PerformanceLogger:
    """Factory funkce pro performance logger

    Args:
        component: Název komponenty

    Returns:
        PerformanceLogger instance

    """
    return PerformanceLogger(component)


# Automatická inicializace při importu modulu
def _initialize_logging():
    """Automatická inicializace logování při importu"""
    try:
        settings = get_settings()
        configure_logging(
            log_level=settings.monitoring.log_level,
            log_format=settings.monitoring.log_format,
            log_file=settings.monitoring.log_file_path,
            enable_console=True,
        )
    except Exception as e:
        # Fallback na základní konfiguraci
        configure_logging()
        logger = get_logger(__name__)
        logger.warning("Failed to load settings for logging, using defaults", error=str(e))


# Inicializace při importu
_initialize_logging()
