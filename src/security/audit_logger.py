#!/usr/bin/env python3
"""
Audit Logger
Comprehensive audit logging for security events and compliance

Author: Senior Python/MLOps Agent
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import json
import logging
import hashlib
import threading
from pathlib import Path
from enum import Enum
from collections import deque
import gzip

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events"""
    ACCESS_ATTEMPT = "access_attempt"
    CONTENT_PROCESSING = "content_processing"
    PII_DETECTION = "pii_detection"
    GDPR_PROCESSING = "gdpr_processing"
    SECURITY_VIOLATION = "security_violation"
    DATA_EXPORT = "data_export"
    CONFIGURATION_CHANGE = "configuration_change"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    SYSTEM_EVENT = "system_event"


class AuditSeverity(Enum):
    """Severity levels for audit events"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Individual audit event"""
    event_id: str
    event_type: AuditEventType
    severity: AuditSeverity
    timestamp: datetime
    source_component: str
    event_data: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "source_component": self.source_component,
            "event_data": self.event_data,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent
        }

    def get_hash(self) -> str:
        """Get integrity hash for the event"""
        event_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(event_str.encode()).hexdigest()


class AuditLogger:
    """Main audit logging system with integrity protection"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.audit_config = config.get("audit_logging", {})

        # Storage configuration
        self.storage_dir = Path(config.get("audit_storage", "audit_logs"))
        self.storage_dir.mkdir(exist_ok=True)

        # Current log file
        self.current_log_file = None
        self.log_rotation_size = self.audit_config.get("rotation_size_mb", 100) * 1024 * 1024
        self.compression_enabled = self.audit_config.get("compression", True)

        # In-memory buffer for high-performance logging
        self.buffer_size = self.audit_config.get("buffer_size", 1000)
        self.event_buffer: deque = deque(maxlen=self.buffer_size)
        self.buffer_lock = threading.Lock()

        # Integrity protection
        self.integrity_enabled = self.audit_config.get("integrity_protection", True)
        self.integrity_chain: List[str] = []

        # Background flush thread
        self.flush_interval = self.audit_config.get("flush_interval_seconds", 30)
        self.stop_flush = threading.Event()
        self.flush_thread = None

        # Statistics
        self.stats = {
            "total_events": 0,
            "events_by_type": {},
            "events_by_severity": {},
            "files_created": 0,
            "integrity_violations": 0
        }

        self._initialize_logging()

        logger.info(f"Audit logger initialized (storage: {self.storage_dir})")

    def _initialize_logging(self):
        """Initialize audit logging system"""

        # Create initial log file
        self._rotate_log_file()

        # Start background flush thread
        if self.flush_interval > 0:
            self.flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
            self.flush_thread.start()

    def log_event(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        source_component: str,
        event_data: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> str:
        """Log an audit event"""

        # Generate unique event ID
        event_id = self._generate_event_id()

        # Create audit event
        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            severity=severity,
            timestamp=datetime.now(timezone.utc),
            source_component=source_component,
            event_data=event_data,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent
        )

        # Add to buffer
        with self.buffer_lock:
            self.event_buffer.append(event)

        # Update statistics
        self.stats["total_events"] += 1
        self.stats["events_by_type"][event_type.value] = self.stats["events_by_type"].get(event_type.value, 0) + 1
        self.stats["events_by_severity"][severity.value] = self.stats["events_by_severity"].get(severity.value, 0) + 1

        # Immediate flush for critical events
        if severity == AuditSeverity.CRITICAL:
            self._flush_buffer()

        logger.debug(f"Logged audit event: {event_id} ({event_type.value})")
        return event_id

    def log_access_attempt(
        self,
        url: str,
        result: str,
        rule_matched: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """Log access attempt"""

        severity = AuditSeverity.WARNING if result == "blocked" else AuditSeverity.INFO

        event_data = {
            "url": url,
            "access_result": result,
            "rule_matched": rule_matched,
            "access_timestamp": datetime.now(timezone.utc).isoformat()
        }

        self.log_event(
            AuditEventType.ACCESS_ATTEMPT,
            severity,
            "osint_sandbox",
            event_data,
            ip_address=ip_address,
            user_agent=user_agent
        )

    def log_pii_detection(
        self,
        pii_count: int,
        pii_types: List[str],
        redacted: bool,
        processing_id: str,
        content_hash: str
    ):
        """Log PII detection event"""

        severity = AuditSeverity.WARNING if pii_count > 0 else AuditSeverity.INFO

        event_data = {
            "pii_count": pii_count,
            "pii_types": pii_types,
            "redacted": redacted,
            "processing_id": processing_id,
            "content_hash": content_hash
        }

        self.log_event(
            AuditEventType.PII_DETECTION,
            severity,
            "pii_redactor",
            event_data
        )

    def log_security_violation(
        self,
        violation_type: str,
        description: str,
        source_ip: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """Log security violation"""

        event_data = {
            "violation_type": violation_type,
            "description": description,
            "detected_timestamp": datetime.now(timezone.utc).isoformat()
        }

        self.log_event(
            AuditEventType.SECURITY_VIOLATION,
            AuditSeverity.CRITICAL,
            "security_monitor",
            event_data,
            user_id=user_id,
            ip_address=source_ip
        )

    def log_data_export(
        self,
        export_type: str,
        data_types: List[str],
        record_count: int,
        destination: str,
        user_id: Optional[str] = None
    ):
        """Log data export event"""

        event_data = {
            "export_type": export_type,
            "data_types": data_types,
            "record_count": record_count,
            "destination": destination,
            "export_timestamp": datetime.now(timezone.utc).isoformat()
        }

        self.log_event(
            AuditEventType.DATA_EXPORT,
            AuditSeverity.WARNING,
            "data_exporter",
            event_data,
            user_id=user_id
        )

    def log_configuration_change(
        self,
        component: str,
        changes: Dict[str, Any],
        user_id: Optional[str] = None
    ):
        """Log configuration change"""

        event_data = {
            "component": component,
            "changes": changes,
            "change_timestamp": datetime.now(timezone.utc).isoformat()
        }

        self.log_event(
            AuditEventType.CONFIGURATION_CHANGE,
            AuditSeverity.WARNING,
            "configuration_manager",
            event_data,
            user_id=user_id
        )

    def _flush_buffer(self):
        """Flush event buffer to disk"""

        events_to_flush = []

        with self.buffer_lock:
            events_to_flush = list(self.event_buffer)
            self.event_buffer.clear()

        if not events_to_flush:
            return

        # Check if log rotation is needed
        if self._should_rotate_log():
            self._rotate_log_file()

        # Write events to file
        try:
            with open(self.current_log_file, 'a', encoding='utf-8') as f:
                for event in events_to_flush:
                    event_dict = event.to_dict()

                    # Add integrity hash if enabled
                    if self.integrity_enabled:
                        event_hash = event.get_hash()
                        event_dict["integrity_hash"] = event_hash
                        event_dict["previous_hash"] = self.integrity_chain[-1] if self.integrity_chain else None
                        self.integrity_chain.append(event_hash)

                    f.write(json.dumps(event_dict) + '\n')

            logger.debug(f"Flushed {len(events_to_flush)} audit events to disk")

        except Exception as e:
            logger.error(f"Failed to flush audit events: {e}")

            # Put events back in buffer
            with self.buffer_lock:
                self.event_buffer.extendleft(reversed(events_to_flush))

    def _should_rotate_log(self) -> bool:
        """Check if log file should be rotated"""

        if not self.current_log_file or not self.current_log_file.exists():
            return True

        return self.current_log_file.stat().st_size >= self.log_rotation_size

    def _rotate_log_file(self):
        """Rotate to a new log file"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_log_file = self.storage_dir / f"audit_log_{timestamp}.jsonl"

        # Compress old log file if enabled
        if self.current_log_file and self.current_log_file.exists() and self.compression_enabled:
            self._compress_log_file(self.current_log_file)

        self.current_log_file = new_log_file
        self.stats["files_created"] += 1

        logger.info(f"Rotated to new audit log file: {new_log_file}")

    def _compress_log_file(self, log_file: Path):
        """Compress a log file"""

        compressed_file = log_file.with_suffix(log_file.suffix + '.gz')

        try:
            with open(log_file, 'rb') as f_in:
                with gzip.open(compressed_file, 'wb') as f_out:
                    f_out.writelines(f_in)

            # Remove original file after successful compression
            log_file.unlink()
            logger.info(f"Compressed audit log: {compressed_file}")

        except Exception as e:
            logger.error(f"Failed to compress audit log {log_file}: {e}")

    def _flush_loop(self):
        """Background thread for periodic buffer flushing"""

        while not self.stop_flush.wait(self.flush_interval):
            self._flush_buffer()

    def _generate_event_id(self) -> str:
        """Generate unique event ID"""

        timestamp = datetime.now(timezone.utc).timestamp()
        random_part = hashlib.md5(f"{timestamp}_{self.stats['total_events']}".encode()).hexdigest()[:8]
        return f"audit_{int(timestamp)}_{random_part}"

    def query_events(
        self,
        event_type: Optional[AuditEventType] = None,
        severity: Optional[AuditSeverity] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Query audit events with filters"""

        events = []

        # Search in current buffer
        with self.buffer_lock:
            for event in self.event_buffer:
                if self._event_matches_filter(event, event_type, severity, start_time, end_time):
                    events.append(event.to_dict())

        # Search in log files if needed
        if len(events) < limit:
            file_events = self._search_log_files(event_type, severity, start_time, end_time, limit - len(events))
            events.extend(file_events)

        return events[:limit]

    def _event_matches_filter(
        self,
        event: AuditEvent,
        event_type: Optional[AuditEventType],
        severity: Optional[AuditSeverity],
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> bool:
        """Check if event matches query filters"""

        if event_type and event.event_type != event_type:
            return False

        if severity and event.severity != severity:
            return False

        if start_time and event.timestamp < start_time:
            return False

        if end_time and event.timestamp > end_time:
            return False

        return True

    def _search_log_files(
        self,
        event_type: Optional[AuditEventType],
        severity: Optional[AuditSeverity],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search audit events in log files"""

        events = []

        # Get all log files (including compressed)
        log_files = list(self.storage_dir.glob("audit_log_*.jsonl*"))
        log_files.sort(reverse=True)  # Most recent first

        for log_file in log_files:
            if len(events) >= limit:
                break

            try:
                # Handle compressed files
                if log_file.suffix == '.gz':
                    file_opener = lambda f: gzip.open(f, 'rt', encoding='utf-8')
                else:
                    file_opener = lambda f: open(f, 'r', encoding='utf-8')

                with file_opener(log_file) as f:
                    for line in f:
                        try:
                            event_dict = json.loads(line.strip())

                            # Create temporary event object for filtering
                            event_timestamp = datetime.fromisoformat(event_dict["timestamp"])
                            temp_event = AuditEvent(
                                event_id=event_dict["event_id"],
                                event_type=AuditEventType(event_dict["event_type"]),
                                severity=AuditSeverity(event_dict["severity"]),
                                timestamp=event_timestamp,
                                source_component=event_dict["source_component"],
                                event_data=event_dict["event_data"]
                            )

                            if self._event_matches_filter(temp_event, event_type, severity, start_time, end_time):
                                events.append(event_dict)

                                if len(events) >= limit:
                                    break

                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue

            except Exception as e:
                logger.error(f"Failed to search log file {log_file}: {e}")

        return events

    def verify_integrity(self) -> Dict[str, Any]:
        """Verify audit log integrity"""

        if not self.integrity_enabled:
            return {"status": "disabled", "message": "Integrity protection not enabled"}

        verification_results = {
            "status": "verified",
            "total_events_checked": 0,
            "integrity_violations": 0,
            "missing_hashes": 0,
            "broken_chains": 0
        }

        # Check integrity chain
        log_files = list(self.storage_dir.glob("audit_log_*.jsonl*"))
        log_files.sort()

        previous_hash = None

        for log_file in log_files:
            try:
                if log_file.suffix == '.gz':
                    file_opener = lambda f: gzip.open(f, 'rt', encoding='utf-8')
                else:
                    file_opener = lambda f: open(f, 'r', encoding='utf-8')

                with file_opener(log_file) as f:
                    for line in f:
                        try:
                            event_dict = json.loads(line.strip())
                            verification_results["total_events_checked"] += 1

                            # Check integrity hash
                            if "integrity_hash" in event_dict:
                                stored_hash = event_dict.pop("integrity_hash")
                                stored_previous = event_dict.pop("previous_hash", None)

                                # Recalculate hash
                                calculated_hash = hashlib.sha256(
                                    json.dumps(event_dict, sort_keys=True).encode()
                                ).hexdigest()

                                if stored_hash != calculated_hash:
                                    verification_results["integrity_violations"] += 1

                                if previous_hash != stored_previous:
                                    verification_results["broken_chains"] += 1

                                previous_hash = stored_hash
                            else:
                                verification_results["missing_hashes"] += 1

                        except (json.JSONDecodeError, KeyError):
                            continue

            except Exception as e:
                logger.error(f"Failed to verify integrity of {log_file}: {e}")

        # Update status
        if (verification_results["integrity_violations"] > 0 or
            verification_results["broken_chains"] > 0):
            verification_results["status"] = "violations_detected"

        return verification_results

    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get audit logging statistics"""

        return {
            "total_events": self.stats["total_events"],
            "events_by_type": self.stats["events_by_type"],
            "events_by_severity": self.stats["events_by_severity"],
            "files_created": self.stats["files_created"],
            "integrity_violations": self.stats["integrity_violations"],
            "buffer_size": len(self.event_buffer),
            "buffer_capacity": self.buffer_size,
            "storage_location": str(self.storage_dir),
            "current_log_file": str(self.current_log_file) if self.current_log_file else None,
            "integrity_protection": self.integrity_enabled,
            "compression_enabled": self.compression_enabled
        }

    def shutdown(self):
        """Gracefully shutdown audit logger"""

        logger.info("Shutting down audit logger...")

        # Stop flush thread
        if self.flush_thread:
            self.stop_flush.set()
            self.flush_thread.join(timeout=5)

        # Final flush
        self._flush_buffer()

        logger.info("Audit logger shutdown complete")


def create_audit_logger(config: Dict[str, Any]) -> AuditLogger:
    """Factory function for audit logger"""
    return AuditLogger(config)


# Usage example
if __name__ == "__main__":
    config = {
        "audit_logging": {
            "buffer_size": 100,
            "flush_interval_seconds": 10,
            "rotation_size_mb": 1,  # Small for testing
            "compression": True,
            "integrity_protection": True
        },
        "audit_storage": "test_audit_logs"
    }

    audit_logger = AuditLogger(config)

    # Log various events
    audit_logger.log_access_attempt(
        "https://example.com/test",
        "allowed",
        ip_address="192.168.1.1"
    )

    audit_logger.log_pii_detection(
        pii_count=2,
        pii_types=["email", "phone"],
        redacted=True,
        processing_id="proc_123",
        content_hash="abc123"
    )

    audit_logger.log_security_violation(
        violation_type="rate_limit_exceeded",
        description="Too many requests from IP",
        source_ip="192.168.1.100"
    )

    # Query events
    events = audit_logger.query_events(
        event_type=AuditEventType.ACCESS_ATTEMPT,
        limit=10
    )

    print(f"Found {len(events)} access attempt events")

    # Get statistics
    stats = audit_logger.get_audit_statistics()
    print(f"Total events logged: {stats['total_events']}")
    print(f"Events by type: {stats['events_by_type']}")

    # Verify integrity
    integrity_result = audit_logger.verify_integrity()
    print(f"Integrity status: {integrity_result['status']}")

    # Shutdown
    audit_logger.shutdown()
