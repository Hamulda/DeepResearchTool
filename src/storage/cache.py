#!/usr/bin/env python3
"""
Enhanced Research Cache with Semantic Similarity
Supports vector-based query matching and intelligent cache retrieval
"""

import json
import hashlib
import pickle
import time
import sqlite3
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import threading

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Enhanced cache entry with semantic information"""
    key: str
    data: Any
    timestamp: datetime
    expiry_time: datetime
    access_count: int = 0
    last_access: datetime = None
    query_text: str = ""
    query_vector: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    similarity_threshold: float = 0.8

@dataclass
class CacheStats:
    """Cache statistics"""
    total_entries: int
    hit_rate: float
    semantic_hits: int
    exact_hits: int
    total_requests: int
    cache_size_mb: float
    avg_retrieval_time_ms: float

class SemanticResearchCache:
    """Enhanced cache with semantic similarity matching"""

    def __init__(self, max_size_gb: float = 10, retention_days: int = 30,
                 similarity_threshold: float = 0.8, max_vector_cache: int = 1000):
        self.max_size = max_size_gb * 1024 * 1024 * 1024  # Convert to bytes
        self.retention_days = retention_days
        self.similarity_threshold = similarity_threshold
        self.max_vector_cache = max_vector_cache

        # Cache storage
        self.cache_dir = Path("research_cache")
        self.cache_dir.mkdir(exist_ok=True)

        # Database for metadata and vectors
        self.db_path = self.cache_dir / "semantic_cache.db"
        self.init_database()

        # In-memory components
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.query_vectors = {}  # Cache for query vectors
        self.vector_cache_fitted = False

        # Statistics
        self.stats = {
            'total_requests': 0,
            'exact_hits': 0,
            'semantic_hits': 0,
            'misses': 0,
            'retrieval_times': []
        }

        # Thread safety
        self.lock = threading.RLock()

        logger.info(f"Initialized semantic cache with {similarity_threshold} similarity threshold")

    def init_database(self):
        """Initialize SQLite database for cache metadata"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    query_text TEXT,
                    query_vector BLOB,
                    timestamp TEXT,
                    expiry_time TEXT,
                    access_count INTEGER DEFAULT 0,
                    last_access TEXT,
                    file_path TEXT,
                    metadata TEXT,
                    similarity_threshold REAL
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expiry_time ON cache_entries(expiry_time)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON cache_entries(timestamp)
            """)

    def _create_cache_key(self, query: str, **kwargs) -> str:
        """Create cache key from query and parameters"""
        # Normalize query
        normalized_query = query.lower().strip()

        # Include relevant parameters in key
        key_data = {
            'query': normalized_query,
            **{k: v for k, v in kwargs.items() if v is not None}
        }

        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def _vectorize_query(self, query: str) -> Optional[np.ndarray]:
        """Convert query to vector representation"""
        try:
            if not self.vector_cache_fitted:
                # If vectorizer not fitted, return None for first few queries
                return None

            query_vector = self.vectorizer.transform([query.lower().strip()])
            return query_vector.toarray()[0]
        except Exception as e:
            logger.warning(f"Error vectorizing query: {e}")
            return None

    def _fit_vectorizer_if_needed(self, queries: List[str]):
        """Fit vectorizer on existing queries if not already fitted"""
        if self.vector_cache_fitted or len(queries) < 5:
            return

        try:
            # Fit vectorizer on existing queries
            self.vectorizer.fit(queries)
            self.vector_cache_fitted = True

            # Update existing entries with vectors
            self._update_existing_vectors(queries)

            logger.info(f"Fitted vectorizer on {len(queries)} queries")
        except Exception as e:
            logger.error(f"Error fitting vectorizer: {e}")

    def _update_existing_vectors(self, queries: List[str]):
        """Update existing cache entries with vectors"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for query in queries:
                    if query.strip():
                        vector = self._vectorize_query(query)
                        if vector is not None:
                            vector_blob = pickle.dumps(vector)
                            conn.execute(
                                "UPDATE cache_entries SET query_vector = ? WHERE query_text = ?",
                                (vector_blob, query)
                            )
        except Exception as e:
            logger.error(f"Error updating existing vectors: {e}")

    def _find_similar_entries(self, query: str, query_vector: Optional[np.ndarray] = None) -> List[Tuple[str, float]]:
        """Find semantically similar cache entries"""
        if query_vector is None or not self.vector_cache_fitted:
            return []

        similar_entries = []

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT key, query_text, query_vector, similarity_threshold 
                    FROM cache_entries 
                    WHERE query_vector IS NOT NULL AND expiry_time > ?
                """, (datetime.now().isoformat(),))

                for row in cursor.fetchall():
                    cache_key, cache_query, vector_blob, threshold = row

                    try:
                        cached_vector = pickle.loads(vector_blob)
                        similarity = cosine_similarity([query_vector], [cached_vector])[0][0]

                        if similarity >= max(threshold, self.similarity_threshold):
                            similar_entries.append((cache_key, similarity))
                    except Exception as e:
                        logger.warning(f"Error calculating similarity for {cache_key}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error finding similar entries: {e}")

        # Sort by similarity descending
        similar_entries.sort(key=lambda x: x[1], reverse=True)
        return similar_entries[:5]  # Return top 5 similar entries

    def get(self, query: str, **kwargs) -> Optional[Any]:
        """Get data from cache with semantic similarity matching"""
        start_time = time.time()

        with self.lock:
            self.stats['total_requests'] += 1

            # Create exact cache key
            exact_key = self._create_cache_key(query, **kwargs)

            # Try exact match first
            exact_result = self._get_exact(exact_key)
            if exact_result is not None:
                self.stats['exact_hits'] += 1
                retrieval_time = (time.time() - start_time) * 1000
                self.stats['retrieval_times'].append(retrieval_time)
                logger.debug(f"Exact cache hit for query: {query[:50]}...")
                return exact_result

            # Try semantic similarity matching
            query_vector = self._vectorize_query(query)
            if query_vector is not None:
                similar_entries = self._find_similar_entries(query, query_vector)

                for similar_key, similarity in similar_entries:
                    similar_result = self._get_exact(similar_key)
                    if similar_result is not None:
                        self.stats['semantic_hits'] += 1
                        retrieval_time = (time.time() - start_time) * 1000
                        self.stats['retrieval_times'].append(retrieval_time)
                        logger.info(f"Semantic cache hit (similarity: {similarity:.3f}) for query: {query[:50]}...")
                        return similar_result

            # No match found
            self.stats['misses'] += 1
            retrieval_time = (time.time() - start_time) * 1000
            self.stats['retrieval_times'].append(retrieval_time)
            return None

    def _get_exact(self, cache_key: str) -> Optional[Any]:
        """Get exact match from cache"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT file_path, expiry_time, access_count 
                    FROM cache_entries 
                    WHERE key = ?
                """, (cache_key,))

                row = cursor.fetchone()
                if not row:
                    return None

                file_path, expiry_time_str, access_count = row
                expiry_time = datetime.fromisoformat(expiry_time_str)

                # Check if expired
                if expiry_time < datetime.now():
                    self._delete_entry(cache_key)
                    return None

                # Load data from file
                cache_file = Path(file_path)
                if not cache_file.exists():
                    self._delete_entry(cache_key)
                    return None

                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)

                # Update access statistics
                conn.execute("""
                    UPDATE cache_entries 
                    SET access_count = ?, last_access = ?
                    WHERE key = ?
                """, (access_count + 1, datetime.now().isoformat(), cache_key))

                return data

        except Exception as e:
            logger.error(f"Error getting cache entry {cache_key}: {e}")
            return None

    def set(self, query: str, data: Any, expiry_hours: int = 24, **kwargs) -> bool:
        """Set data in cache with semantic indexing"""
        with self.lock:
            try:
                cache_key = self._create_cache_key(query, **kwargs)

                # Create cache file
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)

                # Create vector representation
                query_vector = self._vectorize_query(query)
                vector_blob = pickle.dumps(query_vector) if query_vector is not None else None

                # Store metadata in database
                expiry_time = datetime.now() + timedelta(hours=expiry_hours)

                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO cache_entries 
                        (key, query_text, query_vector, timestamp, expiry_time, 
                         access_count, last_access, file_path, metadata, similarity_threshold)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        cache_key,
                        query.strip(),
                        vector_blob,
                        datetime.now().isoformat(),
                        expiry_time.isoformat(),
                        0,
                        datetime.now().isoformat(),
                        str(cache_file),
                        json.dumps(kwargs),
                        self.similarity_threshold
                    ))

                # Update vectorizer if needed
                if not self.vector_cache_fitted:
                    existing_queries = self._get_all_queries()
                    if len(existing_queries) >= 5:
                        self._fit_vectorizer_if_needed(existing_queries)

                logger.debug(f"Cached query: {query[:50]}...")
                return True

            except Exception as e:
                logger.error(f"Error setting cache entry: {e}")
                return False

    def _get_all_queries(self) -> List[str]:
        """Get all cached query texts"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT query_text FROM cache_entries WHERE query_text != ''")
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting all queries: {e}")
            return []

    def _delete_entry(self, cache_key: str):
        """Delete cache entry and associated file"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get file path
                cursor = conn.execute("SELECT file_path FROM cache_entries WHERE key = ?", (cache_key,))
                row = cursor.fetchone()

                if row:
                    file_path = Path(row[0])
                    if file_path.exists():
                        file_path.unlink()

                # Delete from database
                conn.execute("DELETE FROM cache_entries WHERE key = ?", (cache_key,))

        except Exception as e:
            logger.error(f"Error deleting cache entry {cache_key}: {e}")

    def cleanup_expired(self):
        """Remove expired cache entries"""
        with self.lock:
            try:
                current_time = datetime.now()

                with sqlite3.connect(self.db_path) as conn:
                    # Get expired entries
                    cursor = conn.execute("""
                        SELECT key, file_path FROM cache_entries 
                        WHERE expiry_time < ?
                    """, (current_time.isoformat(),))

                    expired_entries = cursor.fetchall()

                    # Delete files and database entries
                    for cache_key, file_path in expired_entries:
                        try:
                            Path(file_path).unlink(missing_ok=True)
                        except Exception as e:
                            logger.warning(f"Error deleting cache file {file_path}: {e}")

                    # Delete from database
                    conn.execute("DELETE FROM cache_entries WHERE expiry_time < ?", (current_time.isoformat(),))

                    if expired_entries:
                        logger.info(f"Cleaned up {len(expired_entries)} expired cache entries")

            except Exception as e:
                logger.error(f"Error during cache cleanup: {e}")

    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
                    total_entries = cursor.fetchone()[0]

                # Calculate cache size
                cache_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl") if f.is_file())
                cache_size_mb = cache_size / (1024 * 1024)

                # Calculate statistics
                total_requests = self.stats['total_requests']
                hit_rate = 0.0
                if total_requests > 0:
                    total_hits = self.stats['exact_hits'] + self.stats['semantic_hits']
                    hit_rate = total_hits / total_requests

                avg_retrieval_time = 0.0
                if self.stats['retrieval_times']:
                    avg_retrieval_time = sum(self.stats['retrieval_times']) / len(self.stats['retrieval_times'])

                return CacheStats(
                    total_entries=total_entries,
                    hit_rate=hit_rate,
                    semantic_hits=self.stats['semantic_hits'],
                    exact_hits=self.stats['exact_hits'],
                    total_requests=total_requests,
                    cache_size_mb=cache_size_mb,
                    avg_retrieval_time_ms=avg_retrieval_time
                )

            except Exception as e:
                logger.error(f"Error getting cache stats: {e}")
                return CacheStats(0, 0.0, 0, 0, 0, 0.0, 0.0)

    def clear_cache(self):
        """Clear all cache entries"""
        with self.lock:
            try:
                # Delete all cache files
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()

                # Clear database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM cache_entries")

                # Reset statistics
                self.stats = {
                    'total_requests': 0,
                    'exact_hits': 0,
                    'semantic_hits': 0,
                    'misses': 0,
                    'retrieval_times': []
                }

                # Reset vectorizer
                self.vector_cache_fitted = False
                self.query_vectors.clear()

                logger.info("Cache cleared successfully")

            except Exception as e:
                logger.error(f"Error clearing cache: {e}")

# Backward compatibility
ResearchCache = SemanticResearchCache
