"""
Intelligent Memory Manager pro DeepResearchTool
Implementuje pokročilou správu paměti s LRU cache, Redis integrací a smart eviction.
"""

import asyncio
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from weakref import WeakValueDictionary

import redis.asyncio as redis
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry s metadaty pro smart eviction"""
    key: str
    value: Any
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    creation_time: float = field(default_factory=time.time)
    size_bytes: int = 0
    importance_score: float = 1.0
    tags: Set[str] = field(default_factory=set)


class MemoryMetrics(BaseModel):
    """Metriky využití paměti"""
    total_entries: int = 0
    total_size_bytes: int = 0
    hit_ratio: float = 0.0
    eviction_count: int = 0
    access_patterns: Dict[str, int] = Field(default_factory=dict)


class IntelligentMemoryManager:
    """
    Pokročilý memory manager s LRU cache, Redis integrací a smart eviction
    pro optimální výkon na M1 architektuře.
    """

    def __init__(
        self,
        max_memory_mb: int = 2048,
        redis_url: str = "redis://localhost:6379",
        enable_redis: bool = True,
        eviction_threshold: float = 0.8,
        compression_enabled: bool = True
    ):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.eviction_threshold = eviction_threshold
        self.compression_enabled = compression_enabled

        # Lokální cache s OrderedDict pro LRU
        self._local_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._current_size = 0
        self._access_lock = asyncio.Lock()

        # Redis pro meziprocesovou komunikaci
        self._redis_client: Optional[redis.Redis] = None
        self._redis_enabled = enable_redis
        self._redis_url = redis_url

        # Metriky a monitoring
        self._metrics = MemoryMetrics()
        self._hit_count = 0
        self._miss_count = 0

        # Weak references pro automatické čištění
        self._object_registry: WeakValueDictionary = WeakValueDictionary()

        logger.info(f"IntelligentMemoryManager inicializován s {max_memory_mb}MB limitem")

    async def initialize(self) -> None:
        """Inicializace Redis připojení"""
        if self._redis_enabled:
            try:
                self._redis_client = redis.from_url(self._redis_url)
                await self._redis_client.ping()
                logger.info("Redis připojení úspěšně navázano")
            except Exception as e:
                logger.warning(f"Redis nedostupný, pokračuje bez distribované cache: {e}")
                self._redis_enabled = False

    async def get(self, key: str, default: Any = None) -> Any:
        """
        Získání hodnoty z cache s intelligent access tracking
        """
        async with self._access_lock:
            # Kontrola lokální cache
            if key in self._local_cache:
                entry = self._local_cache[key]
                entry.access_count += 1
                entry.last_access = time.time()

                # Přesun na konec pro LRU
                self._local_cache.move_to_end(key)
                self._hit_count += 1

                logger.debug(f"Cache hit pro klíč: {key}")
                return entry.value

            # Kontrola Redis cache
            if self._redis_enabled and self._redis_client:
                try:
                    redis_value = await self._redis_client.get(key)
                    if redis_value:
                        value = json.loads(redis_value)
                        # Přidání do lokální cache
                        await self._add_to_local_cache(key, value, importance_score=0.8)
                        self._hit_count += 1
                        logger.debug(f"Redis cache hit pro klíč: {key}")
                        return value
                except Exception as e:
                    logger.warning(f"Chyba při čtení z Redis: {e}")

            self._miss_count += 1
            logger.debug(f"Cache miss pro klíč: {key}")
            return default

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        importance_score: float = 1.0,
        tags: Optional[Set[str]] = None
    ) -> None:
        """
        Uložení hodnoty do cache s intelligent metadata
        """
        async with self._access_lock:
            # Přidání do lokální cache
            await self._add_to_local_cache(key, value, importance_score, tags or set())

            # Uložení do Redis
            if self._redis_enabled and self._redis_client:
                try:
                    serialized_value = json.dumps(value, default=str)
                    if ttl:
                        await self._redis_client.setex(key, ttl, serialized_value)
                    else:
                        await self._redis_client.set(key, serialized_value)
                    logger.debug(f"Hodnota uložena do Redis: {key}")
                except Exception as e:
                    logger.warning(f"Chyba při ukládání do Redis: {e}")

    async def _add_to_local_cache(
        self,
        key: str,
        value: Any,
        importance_score: float = 1.0,
        tags: Optional[Set[str]] = None
    ) -> None:
        """Přidání hodnoty do lokální cache s intelligent eviction"""

        # Výpočet velikosti objektu
        size_bytes = self._estimate_size(value)

        # Kontrola, zda je potřeba eviction
        if self._current_size + size_bytes > self.max_memory_bytes * self.eviction_threshold:
            await self._smart_eviction(size_bytes)

        # Vytvoření cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            size_bytes=size_bytes,
            importance_score=importance_score,
            tags=tags or set()
        )

        # Odebrání starého záznamu pokud existuje
        if key in self._local_cache:
            old_entry = self._local_cache[key]
            self._current_size -= old_entry.size_bytes

        # Přidání nového záznamu
        self._local_cache[key] = entry
        self._current_size += size_bytes

        # Přesun na konec pro LRU
        self._local_cache.move_to_end(key)

        logger.debug(f"Přidán do cache: {key}, velikost: {size_bytes}B")

    async def _smart_eviction(self, required_bytes: int) -> None:
        """
        Intelligent eviction založený na LRU, frekvenci přístupu a importance score
        """
        freed_bytes = 0
        evicted_keys = []

        # Seřazení podle eviction score (nejnižší první)
        sorted_entries = sorted(
            self._local_cache.items(),
            key=lambda x: self._calculate_eviction_score(x[1])
        )

        for key, entry in sorted_entries:
            if freed_bytes >= required_bytes:
                break

            freed_bytes += entry.size_bytes
            evicted_keys.append(key)

        # Odebrání vybraných položek
        for key in evicted_keys:
            del self._local_cache[key]
            self._metrics.eviction_count += 1

        self._current_size -= freed_bytes

        logger.info(f"Smart eviction: odebráno {len(evicted_keys)} položek, "
                   f"uvolněno {freed_bytes}B")

    def _calculate_eviction_score(self, entry: CacheEntry) -> float:
        """
        Výpočet eviction score kombinující LRU, frekvenci a důležitost
        """
        current_time = time.time()

        # Komponenty skóre
        age_factor = current_time - entry.last_access  # Čím starší, tím vyšší
        frequency_factor = 1.0 / max(entry.access_count, 1)  # Čím méně přístupů, tím vyšší
        size_factor = entry.size_bytes / 1024  # Velikost v KB
        importance_factor = 1.0 / max(entry.importance_score, 0.1)  # Inverzní důležitost

        # Vážený součet (nižší = kandidát na eviction)
        score = (
            age_factor * 0.4 +
            frequency_factor * 0.3 +
            size_factor * 0.2 +
            importance_factor * 0.1
        )

        return score

    def _estimate_size(self, obj: Any) -> int:
        """Odhad velikosti objektu v bytech"""
        try:
            import sys
            if hasattr(obj, '__sizeof__'):
                return sys.getsizeof(obj)
            else:
                # Fallback pro komplexní objekty
                return len(str(obj).encode('utf-8'))
        except Exception:
            return 1024  # Fallback odhad

    async def invalidate(self, key: str) -> bool:
        """Invalidace konkrétního klíče"""
        async with self._access_lock:
            removed = False

            # Odebrání z lokální cache
            if key in self._local_cache:
                entry = self._local_cache[key]
                self._current_size -= entry.size_bytes
                del self._local_cache[key]
                removed = True

            # Odebrání z Redis
            if self._redis_enabled and self._redis_client:
                try:
                    await self._redis_client.delete(key)
                except Exception as e:
                    logger.warning(f"Chyba při mazání z Redis: {e}")

            return removed

    async def invalidate_by_tags(self, tags: Set[str]) -> int:
        """Invalidace podle tagů"""
        async with self._access_lock:
            keys_to_remove = []

            for key, entry in self._local_cache.items():
                if entry.tags.intersection(tags):
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                await self.invalidate(key)

            return len(keys_to_remove)

    async def clear(self) -> None:
        """Vymazání celé cache"""
        async with self._access_lock:
            self._local_cache.clear()
            self._current_size = 0

            if self._redis_enabled and self._redis_client:
                try:
                    await self._redis_client.flushdb()
                except Exception as e:
                    logger.warning(f"Chyba při mazání Redis: {e}")

    def get_metrics(self) -> MemoryMetrics:
        """Získání aktuálních metrik"""
        total_requests = self._hit_count + self._miss_count
        hit_ratio = self._hit_count / max(total_requests, 1)

        self._metrics.total_entries = len(self._local_cache)
        self._metrics.total_size_bytes = self._current_size
        self._metrics.hit_ratio = hit_ratio

        return self._metrics

    async def optimize_memory(self) -> Dict[str, Any]:
        """
        Proaktivní optimalizace paměti
        """
        async with self._access_lock:
            initial_size = self._current_size
            initial_count = len(self._local_cache)

            # Identifikace neaktivních položek
            current_time = time.time()
            inactive_keys = []

            for key, entry in self._local_cache.items():
                if current_time - entry.last_access > 3600:  # 1 hodina neaktivity
                    inactive_keys.append(key)

            # Odebrání neaktivních položek
            for key in inactive_keys:
                await self.invalidate(key)

            # Komprese cache pokud je povolena
            if self.compression_enabled and len(self._local_cache) > 1000:
                await self._compress_cache()

            optimization_result = {
                "removed_inactive_entries": len(inactive_keys),
                "memory_freed_bytes": initial_size - self._current_size,
                "entries_before": initial_count,
                "entries_after": len(self._local_cache),
                "compression_applied": self.compression_enabled
            }

            logger.info(f"Memory optimization dokončena: {optimization_result}")
            return optimization_result

    async def _compress_cache(self) -> None:
        """Komprese cache dat pro úsporu paměti"""
        # Implementace komprese pro velké objekty
        import gzip
        import pickle

        compressed_count = 0

        for key, entry in list(self._local_cache.items()):
            if entry.size_bytes > 10240:  # Komprese objektů > 10KB
                try:
                    original_data = pickle.dumps(entry.value)
                    compressed_data = gzip.compress(original_data)

                    if len(compressed_data) < len(original_data) * 0.8:  # Úspora alespoň 20%
                        entry.value = ('compressed', compressed_data)
                        old_size = entry.size_bytes
                        entry.size_bytes = len(compressed_data)
                        self._current_size = self._current_size - old_size + entry.size_bytes
                        compressed_count += 1

                except Exception as e:
                    logger.warning(f"Komprese selhala pro {key}: {e}")

        logger.info(f"Komprimováno {compressed_count} položek")

    async def close(self) -> None:
        """Zavření připojení a cleanup"""
        if self._redis_client:
            await self._redis_client.aclose()

        await self.clear()
        logger.info("IntelligentMemoryManager uzavřen")


# Singleton instance pro globální použití
_memory_manager: Optional[IntelligentMemoryManager] = None


async def get_memory_manager() -> IntelligentMemoryManager:
    """Factory function pro získání singleton instance"""
    global _memory_manager

    if _memory_manager is None:
        _memory_manager = IntelligentMemoryManager()
        await _memory_manager.initialize()

    return _memory_manager


# Convenience funkce pro snadné použití
async def cache_get(key: str, default: Any = None) -> Any:
    """Convenience funkce pro získání z cache"""
    manager = await get_memory_manager()
    return await manager.get(key, default)


async def cache_set(
    key: str,
    value: Any,
    ttl: Optional[int] = None,
    importance_score: float = 1.0,
    tags: Optional[Set[str]] = None
) -> None:
    """Convenience funkce pro uložení do cache"""
    manager = await get_memory_manager()
    await manager.set(key, value, ttl, importance_score, tags)


async def cache_invalidate(key: str) -> bool:
    """Convenience funkce pro invalidaci"""
    manager = await get_memory_manager()
    return await manager.invalidate(key)
