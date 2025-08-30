"""Paměťově efektivní ELT (Extract, Load, Transform) pipeline pro Fázi 1.
Implementuje streaming processing s minimální využitím RAM.
"""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Optional imports with fallbacks
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False
    duckdb = None

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    pl = None

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    pa = None
    pq = None

try:
    import structlog
    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False
    structlog = None

# Fallback logging
if HAS_STRUCTLOG:
    # Konfigurace strukturovaného logování
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logger = structlog.get_logger(__name__)
else:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class DataChunk:
    """Reprezentuje jeden chunk dat pro streaming processing."""

    data: dict[str, Any]
    timestamp: datetime
    source: str
    chunk_id: str


class ParquetStreamWriter:
    """Paměťově efektivní writer pro Apache Parquet soubory.
    Zapisuje data v malých chuncích přímo na disk.
    """

    def __init__(self, output_dir: Path, compression: str = "snappy"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.writers: dict[str, pq.ParquetWriter] = {}
        self.schemas: dict[str, pa.Schema] = {}

    async def write_chunk(self, table_name: str, chunk: DataChunk) -> None:
        """Zapíše jeden chunk dat do Parquet souboru."""
        try:
            # Konverze dat na Arrow Table
            df = pl.DataFrame([chunk.data])
            arrow_table = df.to_arrow()

            # Inicializace writeru pokud neexistuje
            if table_name not in self.writers:
                self._init_writer(table_name, arrow_table.schema)

            # Zápis chunku
            self.writers[table_name].write_table(arrow_table)

            logger.info(
                "Chunk written to Parquet",
                table=table_name,
                chunk_id=chunk.chunk_id,
                source=chunk.source,
            )

        except Exception as e:
            logger.error(
                "Failed to write chunk", table=table_name, chunk_id=chunk.chunk_id, error=str(e)
            )
            raise

    def _init_writer(self, table_name: str, schema: pa.Schema) -> None:
        """Inicializuje Parquet writer pro danou tabulku."""
        file_path = self.output_dir / f"{table_name}.parquet"

        self.writers[table_name] = pq.ParquetWriter(
            file_path,
            schema,
            compression=self.compression,
            use_dictionary=True,
            row_group_size=10000,  # Optimalizace pro čtení
        )

        self.schemas[table_name] = schema
        logger.info("Initialized Parquet writer", table=table_name, path=str(file_path))

    def close_all(self) -> None:
        """Uzavře všechny otevřené writery."""
        for table_name, writer in self.writers.items():
            writer.close()
            logger.info("Closed Parquet writer", table=table_name)

        self.writers.clear()
        self.schemas.clear()


class DuckDBProcessor:
    """DuckDB procesor pro analýzu Parquet souborů bez načítání do paměti.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

        if not HAS_DUCKDB:
            logger.warning("DuckDB not available, processor disabled")
            self.conn = None
            return

        self.conn = duckdb.connect(":memory:")

        # Optimalizace pro Apple Silicon
        self.conn.execute("SET threads=8")
        self.conn.execute("SET memory_limit='4GB'")

    async def analyze_table(self, table_name: str) -> dict[str, Any]:
        """Provede základní analýzu Parquet tabulky."""
        if not self.conn:
            logger.error("DuckDB not available for table analysis")
            return {"error": "DuckDB not available"}

        parquet_path = self.data_dir / f"{table_name}.parquet"

        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        try:
            # Základní statistiky bez načítání do paměti
            stats_query = f"""
            SELECT 
                COUNT(*) as row_count,
                COUNT(DISTINCT source) as unique_sources,
                MIN(timestamp) as earliest_timestamp,
                MAX(timestamp) as latest_timestamp
            FROM read_parquet('{parquet_path}')
            """

            result = self.conn.execute(stats_query).fetchone()

            return {
                "table_name": table_name,
                "row_count": result[0],
                "unique_sources": result[1],
                "earliest_timestamp": result[2],
                "latest_timestamp": result[3],
                "file_size_mb": parquet_path.stat().st_size / (1024 * 1024),
            }

        except Exception as e:
            logger.error("Failed to analyze table", table=table_name, error=str(e))
            raise

    async def query_data(self, query: str) -> list[dict[str, Any]]:
        """Spustí SQL dotaz nad Parquet soubory."""
        try:
            # Nahrazení placeholder s cestami k Parquet souborům
            for parquet_file in self.data_dir.glob("*.parquet"):
                table_name = parquet_file.stem
                query = query.replace(f"{table_name}", f"read_parquet('{parquet_file}')")

            result = self.conn.execute(query).fetchall()
            columns = [desc[0] for desc in self.conn.description]

            return [dict(zip(columns, row, strict=False)) for row in result]

        except Exception as e:
            logger.error("Failed to execute query", query=query, error=str(e))
            raise

    def close(self) -> None:
        """Uzavře DuckDB spojení."""
        self.conn.close()


class ELTPipeline:
    """Hlavní ELT pipeline třída pro paměťově efektivní zpracování dat.
    """

    def __init__(self, data_dir: Path, chunk_size: int = 1000):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size

        self.parquet_writer = ParquetStreamWriter(data_dir)
        self.duckdb_processor = DuckDBProcessor(data_dir)

        logger.info("ELT Pipeline initialized", data_dir=str(data_dir), chunk_size=chunk_size)

    async def extract_and_load(
        self, data_stream: AsyncGenerator[dict[str, Any], None], table_name: str, source: str
    ) -> None:
        """Extrahuje data ze streamu a okamžitě je ukládá do Parquet.
        """
        chunk_buffer = []
        chunk_counter = 0

        async for item in data_stream:
            chunk_buffer.append(item)

            # Flush buffer když dosáhne chunk_size
            if len(chunk_buffer) >= self.chunk_size:
                await self._flush_chunk_buffer(chunk_buffer, table_name, source, chunk_counter)
                chunk_buffer = []
                chunk_counter += 1

        # Flush posledního chunku
        if chunk_buffer:
            await self._flush_chunk_buffer(chunk_buffer, table_name, source, chunk_counter)

    async def _flush_chunk_buffer(
        self, buffer: list[dict[str, Any]], table_name: str, source: str, chunk_counter: int
    ) -> None:
        """Vyprázdní buffer do Parquet souboru."""
        for i, item in enumerate(buffer):
            chunk = DataChunk(
                data=item,
                timestamp=datetime.now(),
                source=source,
                chunk_id=f"{table_name}_{chunk_counter}_{i}",
            )

            await self.parquet_writer.write_chunk(table_name, chunk)

    async def transform_and_analyze(self, table_name: str) -> dict[str, Any]:
        """Provede transformaci a analýzu dat."""
        return await self.duckdb_processor.analyze_table(table_name)

    async def execute_query(self, query: str) -> list[dict[str, Any]]:
        """Spustí SQL dotaz nad uloženými daty."""
        return await self.duckdb_processor.query_data(query)

    def cleanup(self) -> None:
        """Vyčistí zdroje."""
        self.parquet_writer.close_all()
        self.duckdb_processor.close()
        logger.info("ELT Pipeline cleanup completed")


# Příklad použití ELT pipeline
async def example_usage():
    """Příklad použití ELT pipeline."""

    # Simulace datového streamu
    async def mock_data_stream():
        for i in range(10000):
            yield {
                "id": i,
                "title": f"Document {i}",
                "content": f"This is content for document {i}",
                "url": f"https://example.com/doc/{i}",
                "scraped_at": datetime.now().isoformat(),
            }

    # Inicializace pipeline
    pipeline = ELTPipeline(Path("./data/parquet"))

    try:
        # Extract & Load
        await pipeline.extract_and_load(mock_data_stream(), "scraped_documents", "example_scraper")

        # Transform & Analyze
        stats = await pipeline.transform_and_analyze("scraped_documents")
        print(f"Table statistics: {stats}")

        # Query data
        recent_docs = await pipeline.execute_query(
            """
            SELECT title, url, scraped_at 
            FROM scraped_documents 
            ORDER BY scraped_at DESC 
            LIMIT 10
        """
        )
        print(f"Recent documents: {recent_docs}")

    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    asyncio.run(example_usage())
