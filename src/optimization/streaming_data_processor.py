"""
Streamované zpracování dat pro M1 optimalizaci
Zpracování velkých datových souborů s konstantní paměťovou stopou
Použití generátorů a ijson pro streaming JSON parsing
"""

import gc
import logging
import json
from pathlib import Path
from typing import Generator, Dict, Any, Iterator, Optional, List, Union
from dataclasses import dataclass
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

logger = logging.getLogger(__name__)

try:
    import ijson
    IJSON_AVAILABLE = True
except ImportError:
    logger.warning("ijson not available - falling back to standard json")
    IJSON_AVAILABLE = False


@dataclass
class StreamingConfig:
    """Konfigurace pro streamované zpracování"""
    chunk_size: int = 1000  # Velikost chunků pro zpracování
    memory_limit_mb: int = 512  # Limit paměti pro jeden worker
    max_workers: int = 2  # Počet paralelních workerů (konzervativní pro M1)
    gc_frequency: int = 50  # GC po N chuncích
    progress_frequency: int = 100  # Progress log po N záznamech


class M1StreamingDataProcessor:
    """
    M1-optimalizovaný procesor pro streamované zpracování dat

    Klíčové vlastnosti:
    - Konstantní paměťová stopa nezávisle na velikosti souboru
    - Generátory pro lazy evaluation
    - Automatický garbage collection
    - Memory pressure monitoring
    - Batch processing s M1-optimalizovanými velikostmi
    """

    def __init__(self, config: Optional[StreamingConfig] = None):
        self.config = config or StreamingConfig()
        self._processed_count = 0
        self._start_time = None
        self._lock = threading.Lock()

        logger.info(f"🚀 M1 Streaming procesor inicializován")
        logger.info(f"📊 Chunk size: {self.config.chunk_size}")
        logger.info(f"💾 Memory limit: {self.config.memory_limit_mb}MB")

    def stream_json_file(self, file_path: Union[str, Path]) -> Generator[Dict[str, Any], None, None]:
        """
        Streamování JSON souboru po jednotlivých objektech

        Args:
            file_path: Cesta k JSON souboru

        Yields:
            Jednotlivé JSON objekty
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Soubor neexistuje: {file_path}")

        logger.info(f"📁 Streamování JSON souboru: {file_path}")
        file_size_mb = file_path.stat().st_size / 1024 / 1024
        logger.info(f"📏 Velikost souboru: {file_size_mb:.2f}MB")

        self._start_time = time.time()

        if IJSON_AVAILABLE:
            yield from self._stream_with_ijson(file_path)
        else:
            yield from self._stream_with_standard_json(file_path)

    def _stream_with_ijson(self, file_path: Path) -> Generator[Dict[str, Any], None, None]:
        """Streamování pomocí ijson (efektivnější pro velké soubory)"""
        try:
            with open(file_path, 'rb') as file:
                # Předpokládáme array JSON objektů nebo jednotlivé objekty na řádcích
                parser = ijson.parse(file)
                current_object = {}
                object_depth = 0

                for prefix, event, value in parser:
                    if event == 'start_map':
                        if object_depth == 0:
                            current_object = {}
                        object_depth += 1
                    elif event == 'end_map':
                        object_depth -= 1
                        if object_depth == 0 and current_object:
                            yield current_object
                            self._processed_count += 1
                            self._maybe_log_progress()
                            self._maybe_trigger_gc()
                            current_object = {}
                    elif event in ('string', 'number', 'boolean', 'null') and object_depth == 1:
                        # Přidání hodnoty do objektu
                        key = prefix.split('.')[-1] if '.' in prefix else prefix
                        current_object[key] = value

        except Exception as e:
            logger.error(f"❌ Chyba při ijson streamování: {e}")
            # Fallback na standard JSON
            yield from self._stream_with_standard_json(file_path)

    def _stream_with_standard_json(self, file_path: Path) -> Generator[Dict[str, Any], None, None]:
        """Fallback streamování pomocí standard JSON"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                # Pokus o načtení jako JSON array
                try:
                    data = json.load(file)
                    if isinstance(data, list):
                        for item in data:
                            yield item
                            self._processed_count += 1
                            self._maybe_log_progress()
                            self._maybe_trigger_gc()
                    else:
                        # Jednotlivý objekt
                        yield data
                        self._processed_count += 1

                except json.JSONDecodeError:
                    # Možná JSONL formát (jeden JSON objekt na řádek)
                    file.seek(0)
                    for line_num, line in enumerate(file, 1):
                        line = line.strip()
                        if line:
                            try:
                                obj = json.loads(line)
                                yield obj
                                self._processed_count += 1
                                self._maybe_log_progress()
                                self._maybe_trigger_gc()
                            except json.JSONDecodeError as e:
                                logger.warning(f"⚠️ Nevalidní JSON na řádku {line_num}: {e}")

        except Exception as e:
            logger.error(f"❌ Chyba při standardním JSON streamování: {e}")
            raise

    def process_data_chunks(self,
                           data_stream: Generator[Dict[str, Any], None, None],
                           processor_func: callable,
                           chunk_size: Optional[int] = None) -> Generator[List[Any], None, None]:
        """
        Zpracování dat po chuncích s danou funkcí

        Args:
            data_stream: Generator dat
            processor_func: Funkce pro zpracování chunků
            chunk_size: Velikost chunků (None = použij config)

        Yields:
            Zpracované chunky
        """
        chunk_size = chunk_size or self.config.chunk_size
        chunk = []

        logger.info(f"⚙️ Začínám chunk processing (velikost: {chunk_size})")

        try:
            for item in data_stream:
                chunk.append(item)

                if len(chunk) >= chunk_size:
                    # Zpracování chunky
                    processed_chunk = processor_func(chunk)
                    yield processed_chunk

                    # Memory cleanup
                    chunk.clear()
                    self._maybe_trigger_gc()

            # Zpracování posledního chunky
            if chunk:
                processed_chunk = processor_func(chunk)
                yield processed_chunk

        except Exception as e:
            logger.error(f"❌ Chyba při chunk processingu: {e}")
            raise
        finally:
            self._log_final_stats()

    def parallel_process_chunks(self,
                               data_stream: Generator[Dict[str, Any], None, None],
                               processor_func: callable,
                               max_workers: Optional[int] = None) -> Generator[Any, None, None]:
        """
        Paralelní zpracování chunků s ThreadPoolExecutor

        Args:
            data_stream: Generator dat
            processor_func: Funkce pro zpracování
            max_workers: Počet workerů (None = použij config)

        Yields:
            Zpracované výsledky
        """
        max_workers = max_workers or self.config.max_workers
        chunk_size = self.config.chunk_size

        logger.info(f"🔄 Paralelní zpracování s {max_workers} workery")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            chunk = []

            try:
                for item in data_stream:
                    chunk.append(item)

                    if len(chunk) >= chunk_size:
                        # Odeslání chunky na zpracování
                        future = executor.submit(processor_func, chunk.copy())
                        futures.append(future)
                        chunk.clear()

                        # Kontrola dokončených úkolů
                        self._check_completed_futures(futures)

                        # Memory pressure check
                        if len(futures) > max_workers * 2:
                            # Čekání na dokončení některých úkolů
                            completed_futures = []
                            for future in as_completed(futures[:max_workers]):
                                try:
                                    result = future.result()
                                    yield result
                                    completed_futures.append(future)
                                except Exception as e:
                                    logger.error(f"❌ Chyba při zpracování chunky: {e}")

                            # Odstranění dokončených
                            for f in completed_futures:
                                futures.remove(f)

                # Zpracování posledního chunky
                if chunk:
                    future = executor.submit(processor_func, chunk)
                    futures.append(future)

                # Čekání na dokončení všech úkolů
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        yield result
                    except Exception as e:
                        logger.error(f"❌ Chyba při finálním zpracování: {e}")

            except Exception as e:
                logger.error(f"❌ Chyba při paralelním zpracování: {e}")
                raise
            finally:
                self._log_final_stats()

    def _check_completed_futures(self, futures: List):
        """Kontrola a cleanup dokončených future objektů"""
        completed = [f for f in futures if f.done()]
        for f in completed:
            futures.remove(f)

    def stream_csv_file(self, file_path: Union[str, Path],
                       delimiter: str = ',',
                       encoding: str = 'utf-8') -> Generator[Dict[str, str], None, None]:
        """
        Streamování CSV souboru po řádcích

        Args:
            file_path: Cesta k CSV souboru
            delimiter: Oddělovač
            encoding: Kódování souboru

        Yields:
            Řádky jako slovníky
        """
        import csv

        file_path = Path(file_path)
        logger.info(f"📊 Streamování CSV souboru: {file_path}")

        self._start_time = time.time()

        try:
            with open(file_path, 'r', encoding=encoding) as file:
                reader = csv.DictReader(file, delimiter=delimiter)

                for row in reader:
                    yield row
                    self._processed_count += 1
                    self._maybe_log_progress()
                    self._maybe_trigger_gc()

        except Exception as e:
            logger.error(f"❌ Chyba při CSV streamování: {e}")
            raise

    def memory_efficient_file_processor(self,
                                      file_paths: List[Union[str, Path]],
                                      processor_func: callable) -> Generator[Any, None, None]:
        """
        Memory-efficient zpracování více souborů

        Args:
            file_paths: Seznam cest k souborům
            processor_func: Funkce pro zpracování každého souboru

        Yields:
            Výsledky zpracování
        """
        logger.info(f"📁 Zpracovávám {len(file_paths)} souborů")

        for i, file_path in enumerate(file_paths, 1):
            logger.info(f"📄 Zpracovávám soubor {i}/{len(file_paths)}: {Path(file_path).name}")

            try:
                # Zpracování jednoho souboru
                if Path(file_path).suffix.lower() == '.json':
                    data_stream = self.stream_json_file(file_path)
                elif Path(file_path).suffix.lower() == '.csv':
                    data_stream = self.stream_csv_file(file_path)
                else:
                    logger.warning(f"⚠️ Nepodporovaný formát souboru: {file_path}")
                    continue

                result = processor_func(data_stream)
                yield result

                # Force GC po každém souboru
                gc.collect()
                self._log_memory_usage()

            except Exception as e:
                logger.error(f"❌ Chyba při zpracování souboru {file_path}: {e}")
                continue

    def _maybe_log_progress(self):
        """Logování pokroku při zpracování"""
        if self._processed_count % self.config.progress_frequency == 0:
            elapsed = time.time() - self._start_time if self._start_time else 0
            rate = self._processed_count / elapsed if elapsed > 0 else 0
            logger.info(f"📈 Zpracováno {self._processed_count} záznamů ({rate:.1f}/s)")

    def _maybe_trigger_gc(self):
        """Spuštění GC při potřebě"""
        if self._processed_count % self.config.gc_frequency == 0:
            collected = gc.collect()
            if collected > 0:
                logger.debug(f"🧹 GC: {collected} objektů uvolněno")

    def _log_memory_usage(self):
        """Logování aktuálního využití paměti"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_percent = (memory_mb / self.config.memory_limit_mb) * 100

            if memory_percent > 80:
                logger.warning(f"⚠️ Vysoké využití paměti: {memory_mb:.1f}MB ({memory_percent:.1f}%)")
            else:
                logger.debug(f"💾 Paměť: {memory_mb:.1f}MB ({memory_percent:.1f}%)")

        except Exception as e:
            logger.debug(f"Chyba při kontrole paměti: {e}")

    def _log_final_stats(self):
        """Finální statistiky zpracování"""
        if self._start_time:
            elapsed = time.time() - self._start_time
            rate = self._processed_count / elapsed if elapsed > 0 else 0

            logger.info(f"✅ Zpracování dokončeno:")
            logger.info(f"📊 Celkem záznamů: {self._processed_count}")
            logger.info(f"⏱️ Čas: {elapsed:.2f}s")
            logger.info(f"🚀 Rychlost: {rate:.1f} záznamů/s")

            self._log_memory_usage()


# Utility funkce pro jednoduché použití
def stream_large_json(file_path: Union[str, Path],
                     chunk_size: int = 1000) -> Generator[List[Dict[str, Any]], None, None]:
    """
    Jednoduchá funkce pro streamování velkého JSON souboru

    Args:
        file_path: Cesta k JSON souboru
        chunk_size: Velikost chunků

    Yields:
        Chunky JSON objektů
    """
    config = StreamingConfig(chunk_size=chunk_size)
    processor = M1StreamingDataProcessor(config)

    data_stream = processor.stream_json_file(file_path)

    def identity_processor(chunk):
        return chunk

    yield from processor.process_data_chunks(data_stream, identity_processor)


if __name__ == "__main__":
    # Test streamovaného zpracování
    print("🧪 Testování M1 streamovaného procesoru...")

    # Vytvoření test dat
    test_file = Path("test_large_data.json")
    test_data = [{"id": i, "text": f"Document {i}", "value": i * 0.1} for i in range(10000)]

    with open(test_file, 'w') as f:
        json.dump(test_data, f)

    print(f"📁 Vytvořen test soubor: {test_file} ({test_file.stat().st_size / 1024:.1f}KB)")

    # Test streamování
    processor = M1StreamingDataProcessor()

    def test_processor(chunk):
        # Simulace zpracování
        return len(chunk)

    total_processed = 0
    for chunk_result in processor.process_data_chunks(
        processor.stream_json_file(test_file),
        test_processor,
        chunk_size=500
    ):
        total_processed += chunk_result
        print(f"Chunk zpracován: {chunk_result} záznamů")

    print(f"✅ Celkem zpracováno: {total_processed} záznamů")

    # Cleanup
    test_file.unlink()
    print("🧹 Test soubor smazán")
