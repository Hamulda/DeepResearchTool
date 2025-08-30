"""Memory-Efficient Data Core Module
Optimized for MacBook Air M1 8GB RAM constraints
"""

from collections.abc import Iterator
from datetime import datetime
import gc
import logging
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
    # Type alias for when Polars is available
    LazyFrameType = pl.LazyFrame
    DataFrameType = pl.DataFrame
except ImportError:
    HAS_POLARS = False
    pl = None
    # Fallback type aliases
    LazyFrameType = Any
    DataFrameType = Any

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    pa = None
    pq = None

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """Memory optimization utilities for constrained environments"""

    def __init__(self, max_memory_gb: float = 6.0):
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = int(max_memory_gb * 1024**3)

    def check_memory_pressure(self) -> dict[str, Any]:
        """Monitor current memory usage"""
        if not HAS_PSUTIL:
            logger.warning("psutil not available, using fallback memory monitoring")
            return {
                "total_gb": 8.0,  # Fallback for M1 MacBook Air
                "available_gb": 4.0,
                "used_percent": 50.0,
                "pressure": False,
                "critical": False,
            }

        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / 1024**3,
            "available_gb": memory.available / 1024**3,
            "used_percent": memory.percent,
            "pressure": memory.percent > 80,
            "critical": memory.percent > 90,
        }

    def force_gc(self) -> dict[str, int]:
        """Force garbage collection and return stats"""
        if HAS_PSUTIL:
            before = psutil.Process().memory_info().rss
        else:
            before = 0

        collected = gc.collect()

        if HAS_PSUTIL:
            after = psutil.Process().memory_info().rss
            freed_bytes = before - after
        else:
            freed_bytes = 0

        return {
            "collected_objects": collected,
            "freed_bytes": freed_bytes,
            "freed_mb": freed_bytes / 1024**2 if freed_bytes > 0 else 0,
        }

    def get_optimal_batch_size(self, record_size_bytes: int = 1024) -> int:
        """Calculate optimal batch size based on available memory"""
        memory_stats = self.check_memory_pressure()
        available_bytes = memory_stats["available_gb"] * 1024**3

        # Use 25% of available memory for batching
        batch_memory = available_bytes * 0.25
        batch_size = int(batch_memory / record_size_bytes)

        # Reasonable bounds
        return max(100, min(batch_size, 10000))


class LazyDataPipeline:
    """Lazy evaluation pipeline for large datasets using Polars"""

    def __init__(self, optimizer: MemoryOptimizer):
        self.optimizer = optimizer
        if not HAS_POLARS:
            logger.warning("Polars not available, LazyDataPipeline will use fallback mode")

    def create_lazy_frame(self, source: str | Path | dict):
        """Create lazy frame from various sources"""
        if not HAS_POLARS:
            logger.error("Polars not available for lazy frame operations")
            return None

        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix == ".parquet":
                return pl.scan_parquet(source)
            if path.suffix == ".csv":
                return pl.scan_csv(source)
            if path.suffix == ".json":
                return pl.scan_ndjson(source)
        elif isinstance(source, dict):
            # From in-memory data
            return pl.DataFrame(source).lazy()

        raise ValueError(f"Unsupported source type: {type(source)}")

    def streaming_transform(
        self, lazy_frame: LazyFrameType, batch_size: int | None = None
    ) -> Iterator[DataFrameType]:
        """Stream lazy frame in memory-efficient batches"""
        if not HAS_POLARS or lazy_frame is None:
            logger.warning("Polars not available or invalid lazy frame")
            return iter([])

        if batch_size is None:
            batch_size = self.optimizer.get_optimal_batch_size()

        # Use Polars streaming mode for large datasets
        try:
            for batch in lazy_frame.collect():
                yield batch
        except Exception as e:
            logger.error(f"Error in streaming transform: {e}")
            return iter([])

    def apply_transformations(
        self, lazy_frame: LazyFrameType, transformations: list[dict[str, Any]]
    ) -> LazyFrameType:
        """Apply list of transformations to lazy frame"""
        if not HAS_POLARS or lazy_frame is None:
            logger.warning("Polars not available or invalid lazy frame")
            return None

        result = lazy_frame

        try:
            for transform in transformations:
                op = transform["operation"]
                params = transform.get("params", {})

                if op == "filter":
                    result = result.filter(pl.col(params["column"]) == params["value"])
                elif op == "select":
                    result = result.select(params["columns"])
                elif op == "group_by":
                    result = result.group_by(params["columns"]).agg(params["aggregations"])
                elif op == "sort":
                    result = result.sort(params["column"], descending=params.get("desc", False))
                elif op == "with_columns":
                    result = result.with_columns(params["expressions"])

            return result
        except Exception as e:
            logger.error(f"Error applying transformations: {e}")
            return lazy_frame


class ParquetDatasetManager:
    """Manage partitioned Parquet datasets for efficient storage and querying"""

    def __init__(self, base_path: str | Path, optimizer: MemoryOptimizer):
        self.base_path = Path(base_path)
        self.optimizer = optimizer
        self.base_path.mkdir(parents=True, exist_ok=True)

    def write_streaming_batch(
        self,
        data_iterator: Iterator[dict[str, Any]],
        partition_cols: list[str] = None,
        compression: str = "snappy",
    ) -> list[Path]:
        """Write streaming data to partitioned Parquet files"""
        written_files = []
        batch_size = self.optimizer.get_optimal_batch_size()

        batch_data = []
        batch_count = 0

        for record in data_iterator:
            batch_data.append(record)

            if len(batch_data) >= batch_size:
                file_path = self._write_batch(batch_data, batch_count, partition_cols, compression)
                written_files.append(file_path)

                batch_data = []
                batch_count += 1

                # Check memory pressure
                if self.optimizer.check_memory_pressure()["pressure"]:
                    self.optimizer.force_gc()

        # Write remaining data
        if batch_data:
            file_path = self._write_batch(batch_data, batch_count, partition_cols, compression)
            written_files.append(file_path)

        return written_files

    def _write_batch(
        self,
        batch_data: list[dict[str, Any]],
        batch_number: int,
        partition_cols: list[str] = None,
        compression: str = "snappy",
    ) -> Path:
        """Write single batch to Parquet"""
        df = pl.DataFrame(batch_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"batch_{batch_number:06d}_{timestamp}.parquet"

        if partition_cols:
            # Create partitioned structure
            for col in partition_cols:
                if col in df.columns:
                    unique_values = df[col].unique().to_list()
                    for value in unique_values:
                        partition_dir = self.base_path / f"{col}={value}"
                        partition_dir.mkdir(parents=True, exist_ok=True)

                        filtered_df = df.filter(pl.col(col) == value)
                        file_path = partition_dir / file_name
                        filtered_df.write_parquet(file_path, compression=compression)
        else:
            file_path = self.base_path / file_name
            df.write_parquet(file_path, compression=compression)

        return file_path

    def get_dataset_schema(self) -> pa.Schema | None:
        """Get schema from existing Parquet files"""
        parquet_files = list(self.base_path.glob("**/*.parquet"))
        if parquet_files:
            return pq.read_schema(parquet_files[0])
        return None

    def read_partitioned_lazy(
        self, filters: list | None = None, columns: list[str] | None = None
    ) -> LazyFrameType:
        """Read partitioned dataset as lazy frame with predicate pushdown"""
        if not HAS_POLARS:
            logger.warning("Polars not available for lazy reading")
            return None

        pattern = str(self.base_path / "**/*.parquet")

        try:
            lazy_df = pl.scan_parquet(pattern)

            if columns:
                lazy_df = lazy_df.select(columns)

            # Apply filters for predicate pushdown
            if filters:
                for filter_expr in filters:
                    lazy_df = lazy_df.filter(filter_expr)

            return lazy_df
        except Exception as e:
            logger.error(f"Error reading partitioned data: {e}")
            return None


class DuckDBQueryEngine:
    """DuckDB integration for in-situ SQL over Parquet files"""

    def __init__(self, optimizer: MemoryOptimizer):
        self.optimizer = optimizer
        if not HAS_DUCKDB:
            logger.warning("DuckDB not available, query engine disabled")
            self.conn = None
            return

        self.conn = duckdb.connect(":memory:")
        self._configure_memory_limits()

    def _configure_memory_limits(self):
        """Configure DuckDB memory limits for M1 constraints"""
        if not self.conn:
            return

        memory_stats = self.optimizer.check_memory_pressure()
        available_gb = memory_stats["available_gb"]

        # Use 50% of available memory for DuckDB
        memory_limit = f"{available_gb * 0.5:.1f}GB"

        self.conn.execute(f"SET memory_limit='{memory_limit}'")
        self.conn.execute("SET threads=8")  # M1 has 8 cores

    def register_parquet_dataset(
        self, name: str, dataset_path: str | Path, recursive: bool = True
    ):
        """Register Parquet dataset for SQL queries"""
        path_pattern = str(
            Path(dataset_path) / "**/*.parquet" if recursive else Path(dataset_path) / "*.parquet"
        )

        query = f"""
        CREATE OR REPLACE VIEW {name} AS 
        SELECT * FROM read_parquet('{path_pattern}')
        """
        self.conn.execute(query)

    def execute_query(self, query: str, return_polars: bool = True):
        """Execute SQL query and return results"""
        if not self.conn:
            logger.error("DuckDB not available")
            return None

        result = self.conn.execute(query)

        if return_polars and HAS_POLARS:
            # Convert to Polars for consistency
            arrow_table = result.fetch_arrow_table()
            return pl.from_arrow(arrow_table)

        return result

    def analyze_dataset(self, table_name: str) -> dict[str, Any]:
        """Generate dataset statistics"""
        queries = {
            "row_count": f"SELECT COUNT(*) as count FROM {table_name}",
            "column_info": f"DESCRIBE {table_name}",
            "memory_usage": f"SELECT SUM(strlen(column_name)) as estimated_size FROM information_schema.columns WHERE table_name = '{table_name}'",
        }

        results = {}
        for stat_name, query in queries.items():
            try:
                results[stat_name] = self.execute_query(query)
            except Exception as e:
                logger.warning(f"Failed to compute {stat_name}: {e}")
                results[stat_name] = None

        return results

    def close(self):
        """Close DuckDB connection"""
        self.conn.close()


# Export main classes
__all__ = ["DuckDBQueryEngine", "LazyDataPipeline", "MemoryOptimizer", "ParquetDatasetManager"]
