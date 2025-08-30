"""
Asynchronní batch processor pro výkonné zpracování dat
Implementuje dávkové operace s optimalizací pro M1 MacBook
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass
from datetime import datetime
import time
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import threading

from ..core.error_handling import ErrorAggregator, scraping_retry, timeout_after

T = TypeVar('T')
R = TypeVar('R')

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Konfigurace pro batch operace"""
    batch_size: int = 100
    max_concurrent_batches: int = 3
    retry_failed_items: bool = True
    max_retries: int = 3
    delay_between_batches: float = 0.1
    timeout_per_batch: int = 300  # 5 minut


class AsyncBatchProcessor(Generic[T, R]):
    """Asynchronní batch processor pro výkonné zpracování dat"""
    
    def __init__(self, config: BatchConfig = None):
        self.config = config or BatchConfig()
        self.error_aggregator = ErrorAggregator()
        self._stats = {
            'total_items': 0,
            'processed_items': 0,
            'failed_items': 0,
            'batches_processed': 0,
            'start_time': None,
            'end_time': None
        }
    
    async def process_items(
        self, 
        items: List[T], 
        processor_func: Callable[[List[T]], R],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[R]:
        """
        Zpracuje seznam položek v dávkách s asynchronním přístupem
        
        Args:
            items: Seznam položek k zpracování
            processor_func: Funkce pro zpracování dávky položek
            progress_callback: Volitelný callback pro sledování pokroku
            
        Returns:
            Seznam výsledků zpracování
        """
        self._stats['start_time'] = time.time()
        self._stats['total_items'] = len(items)
        
        logger.info(f"Začínám batch zpracování {len(items)} položek v dávkách po {self.config.batch_size}")
        
        # Rozdělení na dávky
        batches = self._create_batches(items)
        results = []
        
        # Semafór pro omezení současných dávek
        semaphore = asyncio.Semaphore(self.config.max_concurrent_batches)
        
        async def process_batch_with_semaphore(batch_idx: int, batch: List[T]) -> Optional[R]:
            async with semaphore:
                return await self._process_single_batch(batch_idx, batch, processor_func)
        
        # Spuštění všech dávek současně
        tasks = [
            process_batch_with_semaphore(i, batch) 
            for i, batch in enumerate(batches)
        ]
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Zpracování výsledků
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                self.error_aggregator.add_error(result, f"batch {i}")
                logger.error(f"Batch {i} failed: {result}")
            else:
                if result is not None:
                    results.append(result)
                self.error_aggregator.add_success()
            
            # Progress callback
            if progress_callback:
                progress_callback(i + 1, len(batches))
        
        self._stats['end_time'] = time.time()
        self._stats['processed_items'] = len(results)
        self._stats['failed_items'] = self._stats['total_items'] - self._stats['processed_items']
        self._stats['batches_processed'] = len(batches)
        
        self._log_statistics()
        
        return results
    
    @timeout_after(300)  # 5 minute timeout per batch
    async def _process_single_batch(
        self, 
        batch_idx: int, 
        batch: List[T], 
        processor_func: Callable[[List[T]], R]
    ) -> Optional[R]:
        """Zpracuje jednu dávku s retry logikou"""
        
        @scraping_retry
        async def process_with_retry():
            # Spuštění v thread pool pro CPU-intensive operace
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=2) as executor:
                return await loop.run_in_executor(executor, processor_func, batch)
        
        try:
            logger.debug(f"Zpracovávám batch {batch_idx} s {len(batch)} položkami")
            result = await process_with_retry()
            
            # Delay mezi batches pro rate limiting
            if self.config.delay_between_batches > 0:
                await asyncio.sleep(self.config.delay_between_batches)
            
            return result
            
        except Exception as e:
            logger.error(f"Batch {batch_idx} failed after retries: {e}")
            return None
    
    def _create_batches(self, items: List[T]) -> List[List[T]]:
        """Rozdělí položky na dávky"""
        batches = []
        for i in range(0, len(items), self.config.batch_size):
            batch = items[i:i + self.config.batch_size]
            batches.append(batch)
        return batches
    
    def _log_statistics(self):
        """Zaloguje statistiky zpracování"""
        duration = self._stats['end_time'] - self._stats['start_time']
        success_rate = self._stats['processed_items'] / self._stats['total_items'] if self._stats['total_items'] > 0 else 0
        items_per_second = self._stats['processed_items'] / duration if duration > 0 else 0
        
        logger.info(f"Batch zpracování dokončeno:")
        logger.info(f"  - Celkem položek: {self._stats['total_items']}")
        logger.info(f"  - Zpracováno: {self._stats['processed_items']}")
        logger.info(f"  - Selhalo: {self._stats['failed_items']}")
        logger.info(f"  - Úspěšnost: {success_rate:.1%}")
        logger.info(f"  - Doba trvání: {duration:.1f}s")
        logger.info(f"  - Rychlost: {items_per_second:.1f} položek/s")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Vrátí detailní statistiky"""
        duration = None
        if self._stats['start_time'] and self._stats['end_time']:
            duration = self._stats['end_time'] - self._stats['start_time']
        
        return {
            **self._stats,
            'duration': duration,
            'success_rate': self._stats['processed_items'] / self._stats['total_items'] if self._stats['total_items'] > 0 else 0,
            'items_per_second': self._stats['processed_items'] / duration if duration and duration > 0 else 0,
            'error_summary': self.error_aggregator.get_summary()
        }


class AsyncScrapingBatch:
    """Specializovaný batch processor pro web scraping"""
    
    def __init__(self, max_concurrent: int = 10, delay: float = 0.5):
        self.max_concurrent = max_concurrent
        self.delay = delay
        self.session = None
        self.error_aggregator = ErrorAggregator()
    
    async def scrape_urls_batch(self, urls: List[str]) -> Dict[str, str]:
        """
        Současný scraping více URL s rate limiting
        
        Args:
            urls: Seznam URL k scraping
            
        Returns:
            Dictionary mapující URL na scraped content
        """
        await self._ensure_session()
        
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results = {}
        
        async def scrape_single_url(url: str) -> tuple[str, str]:
            async with semaphore:
                try:
                    await asyncio.sleep(self.delay)  # Rate limiting
                    
                    async with self.session.get(url, timeout=30) as response:
                        if response.status == 200:
                            content = await response.text()
                            self.error_aggregator.add_success()
                            return url, content
                        else:
                            self.error_aggregator.add_error(
                                Exception(f"HTTP {response.status}"), 
                                f"scraping {url}"
                            )
                            return url, ""
                            
                except Exception as e:
                    self.error_aggregator.add_error(e, f"scraping {url}")
                    return url, ""
        
        # Spuštění všech URL současně
        tasks = [scrape_single_url(url) for url in urls]
        url_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Zpracování výsledků
        for result in url_results:
            if isinstance(result, tuple) and len(result) == 2:
                url, content = result
                if content:  # Pouze úspěšné výsledky
                    results[url] = content
        
        logger.info(f"Batch scraping dokončen: {len(results)}/{len(urls)} úspěšných")
        return results
    
    async def _ensure_session(self):
        """Zajistí HTTP session"""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(
                limit=self.max_concurrent + 5,
                limit_per_host=5,
                enable_cleanup_closed=True
            )
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=30),
                headers={
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                }
            )
    
    async def close(self):
        """Uzavře HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Vrátí statistiky scraping batch"""
        return self.error_aggregator.get_summary()


class DatabaseBatchProcessor:
    """Batch processor optimalizovaný pro databázové operace"""
    
    def __init__(self, batch_size: int = 200):
        self.batch_size = batch_size
        self.error_aggregator = ErrorAggregator()
    
    async def batch_insert_vector_data(
        self, 
        vector_store, 
        documents: List[Dict[str, Any]]
    ) -> int:
        """
        Dávkové vkládání do vector store
        
        Args:
            vector_store: Instance vector store
            documents: Seznam dokumentů k vložení
            
        Returns:
            Počet úspěšně vložených dokumentů
        """
        if not documents:
            return 0
        
        total_inserted = 0
        batches = [
            documents[i:i + self.batch_size] 
            for i in range(0, len(documents), self.batch_size)
        ]
        
        logger.info(f"Vkládám {len(documents)} dokumentů v {len(batches)} dávkách")
        
        for i, batch in enumerate(batches):
            try:
                # Příprava dat pro batch insert
                texts = [doc.get('text', '') for doc in batch]
                metadatas = [doc.get('metadata', {}) for doc in batch]
                ids = [doc.get('id', f'doc_{i}_{j}') for j, doc in enumerate(batch)]
                
                # Batch insert do vector store
                if hasattr(vector_store, 'add_texts'):
                    vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
                elif hasattr(vector_store, 'upsert'):
                    # Pro Qdrant nebo jiné
                    vector_store.upsert(
                        points=[
                            {'id': id_, 'vector': None, 'payload': {'text': text, **metadata}}
                            for id_, text, metadata in zip(ids, texts, metadatas)
                        ]
                    )
                
                total_inserted += len(batch)
                self.error_aggregator.add_success()
                
                logger.debug(f"Batch {i+1}/{len(batches)}: vloženo {len(batch)} dokumentů")
                
                # Krátká pauza mezi batches
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.error_aggregator.add_error(e, f"vector batch {i}")
                logger.error(f"Failed to insert batch {i}: {e}")
        
        logger.info(f"Batch insert dokončen: {total_inserted}/{len(documents)} dokumentů")
        return total_inserted


# Utility funkce pro rychlé použití
async def batch_process_urls(urls: List[str], max_concurrent: int = 10) -> Dict[str, str]:
    """Rychlá utility pro batch scraping URL"""
    processor = AsyncScrapingBatch(max_concurrent=max_concurrent)
    try:
        return await processor.scrape_urls_batch(urls)
    finally:
        await processor.close()


async def batch_process_items(
    items: List[T], 
    processor_func: Callable[[List[T]], R], 
    batch_size: int = 100
) -> List[R]:
    """Rychlá utility pro obecné batch zpracování"""
    config = BatchConfig(batch_size=batch_size)
    processor = AsyncBatchProcessor[T, R](config)
    return await processor.process_items(items, processor_func)