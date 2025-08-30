"""WARC Tracking System
Sledování a archivace webových zdrojů pro forenzní analýzu

Author: Senior Python/MLOps Agent
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
import gzip
import hashlib
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import uuid

logger = logging.getLogger(__name__)


@dataclass
class WARCRecord:
    """Záznam pro WARC archiv"""

    record_id: str
    url: str
    timestamp: datetime
    content_type: str
    content_length: int
    content_hash: str
    http_status: int | None = None
    headers: dict[str, str] = field(default_factory=dict)
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    source_query: str | None = None
    retrieval_method: str = "http"  # http, browser, tor


@dataclass
class WARCHeader:
    """WARC header informace"""

    warc_type: str
    warc_record_id: str
    warc_date: str
    content_length: int
    content_type: str = "application/http; msgtype=response"
    warc_target_uri: str | None = None


class WARCWriter:
    """Writer pro WARC formát s kompresí
    """

    def __init__(self, output_dir: str = "./warc_archives"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.current_file: Path | None = None
        self.current_writer = None
        self.records_written = 0
        self.max_records_per_file = 1000

    def _get_warc_filename(self) -> str:
        """Generování názvu WARC souboru"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"research_archive_{timestamp}.warc.gz"

    def _create_warc_header(self, record: WARCRecord) -> str:
        """Vytvoření WARC header"""
        warc_date = record.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

        header_lines = [
            "WARC/1.0",
            "WARC-Type: response",
            f"WARC-Record-ID: <urn:uuid:{record.record_id}>",
            f"WARC-Date: {warc_date}",
            f"WARC-Target-URI: {record.url}",
            f"Content-Type: {record.content_type}",
            f"Content-Length: {record.content_length}",
            ""  # Prázdný řádek na konci header
        ]

        return "\r\n".join(header_lines) + "\r\n"

    def _create_http_response(self, record: WARCRecord) -> str:
        """Vytvoření HTTP response části"""
        status_line = f"HTTP/1.1 {record.http_status or 200} OK\r\n"

        headers = []
        for key, value in record.headers.items():
            headers.append(f"{key}: {value}")

        headers.append(f"Content-Length: {len(record.content.encode('utf-8'))}")
        headers.append("")  # Prázdný řádek před tělem

        http_response = status_line + "\r\n".join(headers) + "\r\n" + record.content

        return http_response

    async def write_record(self, record: WARCRecord):
        """Zápis WARC záznamu"""
        try:
            # Otevření nového souboru pokud je potřeba
            if (self.current_writer is None or
                self.records_written >= self.max_records_per_file):
                await self._open_new_file()

            # Vytvoření HTTP response
            http_response = self._create_http_response(record)
            http_response_bytes = http_response.encode('utf-8')

            # Aktualizace content length
            record.content_length = len(http_response_bytes)

            # Vytvoření WARC header
            warc_header = self._create_warc_header(record)

            # Zápis do souboru
            self.current_writer.write(warc_header.encode('utf-8'))
            self.current_writer.write(http_response_bytes)
            self.current_writer.write(b"\r\n\r\n")  # Separátor mezi záznamy

            self.records_written += 1

            logger.debug(f"WARC záznam zapsán: {record.url}")

        except Exception as e:
            logger.error(f"Chyba při zápisu WARC záznamu: {e}")

    async def _open_new_file(self):
        """Otevření nového WARC souboru"""
        if self.current_writer:
            self.current_writer.close()

        filename = self._get_warc_filename()
        self.current_file = self.output_dir / filename

        self.current_writer = gzip.open(self.current_file, 'wb')
        self.records_written = 0

        logger.info(f"Nový WARC soubor vytvořen: {filename}")

    def close(self):
        """Uzavření WARC writer"""
        if self.current_writer:
            self.current_writer.close()
            self.current_writer = None


class WARCTracker:
    """Hlavní třída pro sledování a archivaci webových zdrojů
    """

    def __init__(self, output_dir: str = "./warc_archives"):
        self.writer = WARCWriter(output_dir)
        self.tracked_urls: dict[str, WARCRecord] = {}
        self.session_metadata = {
            "session_id": str(uuid.uuid4()),
            "start_time": datetime.now(UTC),
            "total_records": 0,
            "unique_domains": set(),
            "queries_tracked": set()
        }

    def _calculate_content_hash(self, content: str) -> str:
        """Výpočet hash obsahu"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _extract_domain(self, url: str) -> str:
        """Extrakce domény z URL"""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except:
            return "unknown"

    async def track_document(self,
                           url: str,
                           content: str,
                           headers: dict[str, str] = None,
                           source_query: str = None,
                           retrieval_method: str = "http",
                           http_status: int = 200) -> str:
        """Sledování a archivace dokumentu

        Args:
            url: URL dokumentu
            content: Obsah dokumentu
            headers: HTTP headers
            source_query: Původní dotaz
            retrieval_method: Metoda získání
            http_status: HTTP status kód

        Returns:
            ID záznamu

        """
        record_id = str(uuid.uuid4())
        content_hash = self._calculate_content_hash(content)

        # Kontrola duplicity podle hash
        existing_record = None
        for existing_id, existing in self.tracked_urls.items():
            if existing.content_hash == content_hash and existing.url == url:
                existing_record = existing
                break

        if existing_record:
            logger.debug(f"Dokument již archivován: {url}")
            return existing_record.record_id

        # Vytvoření nového záznamu
        record = WARCRecord(
            record_id=record_id,
            url=url,
            timestamp=datetime.now(UTC),
            content_type="text/html; charset=utf-8",
            content_length=len(content.encode('utf-8')),
            content_hash=content_hash,
            http_status=http_status,
            headers=headers or {},
            content=content,
            source_query=source_query,
            retrieval_method=retrieval_method,
            metadata={
                "domain": self._extract_domain(url),
                "content_length_chars": len(content),
                "archive_timestamp": datetime.now(UTC).isoformat()
            }
        )

        # Uložení záznamu
        self.tracked_urls[record_id] = record

        # Zápis do WARC archivu
        await self.writer.write_record(record)

        # Aktualizace session metadat
        self.session_metadata["total_records"] += 1
        self.session_metadata["unique_domains"].add(self._extract_domain(url))
        if source_query:
            self.session_metadata["queries_tracked"].add(source_query)

        logger.info(f"Dokument archivován: {url} (ID: {record_id[:8]})")

        return record_id

    def get_record(self, record_id: str) -> WARCRecord | None:
        """Získání záznamu podle ID"""
        return self.tracked_urls.get(record_id)

    def search_records(self,
                      query: str = None,
                      domain: str = None,
                      source_query: str = None) -> list[WARCRecord]:
        """Vyhledání záznamů podle kritérií

        Args:
            query: Textové vyhledávání v obsahu
            domain: Filtr podle domény
            source_query: Filtr podle původního dotazu

        Returns:
            Seznam odpovídajících záznamů

        """
        results = []

        for record in self.tracked_urls.values():
            match = True

            if domain and record.metadata.get("domain") != domain.lower():
                match = False

            if source_query and record.source_query != source_query:
                match = False

            if query:
                query_lower = query.lower()
                if (query_lower not in record.content.lower() and
                    query_lower not in record.url.lower()):
                    match = False

            if match:
                results.append(record)

        return results

    def get_session_stats(self) -> dict[str, Any]:
        """Získání statistik session"""
        runtime = datetime.now(UTC) - self.session_metadata["start_time"]

        # Statistiky podle domén
        domain_stats = {}
        for record in self.tracked_urls.values():
            domain = record.metadata.get("domain", "unknown")
            if domain not in domain_stats:
                domain_stats[domain] = {"count": 0, "total_size": 0}
            domain_stats[domain]["count"] += 1
            domain_stats[domain]["total_size"] += record.content_length

        return {
            "session_id": self.session_metadata["session_id"],
            "runtime_seconds": runtime.total_seconds(),
            "total_records": self.session_metadata["total_records"],
            "unique_domains": len(self.session_metadata["unique_domains"]),
            "unique_queries": len(self.session_metadata["queries_tracked"]),
            "domain_breakdown": domain_stats,
            "total_content_size": sum(r.content_length for r in self.tracked_urls.values()),
            "average_record_size": (sum(r.content_length for r in self.tracked_urls.values()) /
                                  max(1, len(self.tracked_urls))),
            "retrieval_methods": self._get_method_stats()
        }

    def _get_method_stats(self) -> dict[str, int]:
        """Statistiky podle metod získání"""
        methods = {}
        for record in self.tracked_urls.values():
            method = record.retrieval_method
            methods[method] = methods.get(method, 0) + 1
        return methods

    def export_metadata(self, filepath: str):
        """Export metadat do JSON souboru"""
        metadata = {
            "session": self.session_metadata,
            "records": []
        }

        for record in self.tracked_urls.values():
            record_meta = {
                "record_id": record.record_id,
                "url": record.url,
                "timestamp": record.timestamp.isoformat(),
                "content_hash": record.content_hash,
                "content_length": record.content_length,
                "source_query": record.source_query,
                "retrieval_method": record.retrieval_method,
                "http_status": record.http_status,
                "metadata": record.metadata
            }
            metadata["records"].append(record_meta)

        # Konverze set na list pro JSON serializaci
        metadata["session"]["unique_domains"] = list(metadata["session"]["unique_domains"])
        metadata["session"]["queries_tracked"] = list(metadata["session"]["queries_tracked"])
        metadata["session"]["start_time"] = metadata["session"]["start_time"].isoformat()

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Metadata exportována do {filepath}")

    def get_provenance_chain(self, record_id: str) -> dict[str, Any]:
        """Získání řetězce provenience pro daný záznam

        Args:
            record_id: ID záznamu

        Returns:
            Řetězec provenience

        """
        record = self.get_record(record_id)
        if not record:
            return {}

        return {
            "record_id": record.record_id,
            "original_url": record.url,
            "retrieval_timestamp": record.timestamp.isoformat(),
            "retrieval_method": record.retrieval_method,
            "source_query": record.source_query,
            "content_integrity": {
                "hash_algorithm": "sha256",
                "content_hash": record.content_hash,
                "content_length": record.content_length
            },
            "technical_metadata": {
                "http_status": record.http_status,
                "headers": record.headers,
                "domain": record.metadata.get("domain"),
                "archive_timestamp": record.metadata.get("archive_timestamp")
            }
        }

    async def close(self):
        """Uzavření trackeru a uložení dat"""
        self.writer.close()

        # Export finálních metadat
        metadata_file = self.writer.output_dir / f"session_{self.session_metadata['session_id'][:8]}_metadata.json"
        self.export_metadata(str(metadata_file))

        logger.info(f"WARC Tracker uzavřen. Celkem {self.session_metadata['total_records']} záznamů.")


# Globální instance
warc_tracker = WARCTracker()
