"""
Hierarchická sumarizace pro M1 RAG optimalizaci
Rekurzivní komprese velkých dokumentů do zhuštěných souhrných textů
Dramatická redukce velikosti kontextu pro vektorovou databázi
"""

import gc
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Generator
from dataclasses import dataclass
from pathlib import Path
import time
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalConfig:
    """Konfigurace pro hierarchickou sumarizaci"""
    max_chunk_size: int = 2000  # Maximální velikost jednoho chunky v tokenech
    overlap_size: int = 200  # Překryv mezi chunky
    compression_ratio: float = 0.3  # Cílový poměr komprese (30% originální velikosti)
    max_recursion_depth: int = 3  # Maximální hloubka rekurze
    min_chunk_size: int = 500  # Minimální velikost chunky pro další dělení
    summary_template: str = "Shrň následující text do {target_length} slov, zachovej klíčové informace a faktický obsah:"


class M1HierarchicalSummarizer:
    """
    M1-optimalizovaný hierarchický sumarizátor

    Algoritmus:
    1. Rozdělí dokument na sémanticky související části
    2. Vytvoří souhrn každé části pomocí lokálního LLM
    3. Rekurzivně spojí a zhustí dílčí souhrny
    4. Výsledek: dramaticky zmenšený text s zachovaným obsahem

    M1 optimalizace:
    - Malé batch sizes pro stabilní inference
    - Streaming processing pro velké dokumenty
    - Memory pressure monitoring
    - Cachování pro opakované úseky
    """

    def __init__(self,
                 llm_client=None,
                 config: Optional[HierarchicalConfig] = None,
                 cache_dir: Optional[str] = None):
        self.llm_client = llm_client
        self.config = config or HierarchicalConfig()
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./summary_cache")

        # Thread safety a performance tracking
        self._lock = threading.Lock()
        self._cache = {}
        self._stats = {
            'documents_processed': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'cache_hits': 0,
            'compression_achieved': 0.0
        }

        # Inicializace cache adresáře
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"🧠 Hierarchický sumarizátor inicializován")
        logger.info(f"⚙️ Max chunk size: {self.config.max_chunk_size} tokenů")
        logger.info(f"🗜️ Cílová komprese: {self.config.compression_ratio * 100}%")

    def summarize_document(self,
                          document: str,
                          target_compression: Optional[float] = None) -> Dict[str, Any]:
        """
        Hlavní metoda pro hierarchickou sumarizaci dokumentu

        Args:
            document: Vstupní text dokumentu
            target_compression: Cílový poměr komprese (None = použij config)

        Returns:
            Slovník s výsledky sumarizace
        """
        target_compression = target_compression or self.config.compression_ratio

        logger.info(f"📄 Začínám hierarchickou sumarizaci")
        logger.info(f"📏 Vstupní délka: {len(document)} znaků")
        logger.info(f"🎯 Cílová komprese: {target_compression * 100}%")

        start_time = time.time()

        try:
            # Fáze 1: Sémantické chunking
            chunks = self._semantic_chunking(document)
            logger.info(f"🔪 Dokument rozdělen na {len(chunks)} chunků")

            # Fáze 2: Hierarchická komprese
            summary_result = self._hierarchical_compression(chunks, target_compression)

            # Statistiky
            processing_time = time.time() - start_time
            input_length = len(document)
            output_length = len(summary_result['final_summary'])
            actual_compression = output_length / input_length

            result = {
                'original_text': document,
                'final_summary': summary_result['final_summary'],
                'compression_tree': summary_result['compression_tree'],
                'statistics': {
                    'input_length': input_length,
                    'output_length': output_length,
                    'compression_ratio': actual_compression,
                    'target_compression': target_compression,
                    'compression_achieved': actual_compression <= target_compression,
                    'processing_time_seconds': processing_time,
                    'chunks_created': len(chunks),
                    'recursion_levels': summary_result['recursion_levels']
                }
            }

            # Update global stats
            with self._lock:
                self._stats['documents_processed'] += 1
                self._stats['total_input_tokens'] += input_length // 4  # Approx tokens
                self._stats['total_output_tokens'] += output_length // 4
                self._stats['compression_achieved'] += actual_compression

            logger.info(f"✅ Sumarizace dokončena za {processing_time:.2f}s")
            logger.info(f"🗜️ Komprese: {input_length} → {output_length} znaků ({actual_compression:.1%})")

            return result

        except Exception as e:
            logger.error(f"❌ Chyba při sumarizaci dokumentu: {e}")
            raise

    def _semantic_chunking(self, document: str) -> List[Dict[str, Any]]:
        """
        Sémantické rozdělení dokumentu na chunky

        Args:
            document: Vstupní text

        Returns:
            Seznam chunků s metadaty
        """
        # Jednoduché rozdělení po odstavcích s překryvem
        paragraphs = [p.strip() for p in document.split('\n\n') if p.strip()]

        chunks = []
        current_chunk = ""
        current_size = 0
        chunk_id = 0

        for paragraph in paragraphs:
            paragraph_size = len(paragraph)

            # Pokud by přidání odstavce překročilo limit
            if current_size + paragraph_size > self.config.max_chunk_size and current_chunk:
                # Uložit současný chunk
                chunks.append({
                    'id': chunk_id,
                    'text': current_chunk.strip(),
                    'size': current_size,
                    'start_paragraph': chunk_id * 10,  # Aproximace
                    'end_paragraph': chunk_id * 10 + current_chunk.count('\n\n')
                })

                # Začít nový chunk s překryvem
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + "\n\n" + paragraph
                current_size = len(current_chunk)
                chunk_id += 1
            else:
                # Přidat odstavec k současnému chunku
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_size += paragraph_size

        # Přidat poslední chunk
        if current_chunk:
            chunks.append({
                'id': chunk_id,
                'text': current_chunk.strip(),
                'size': current_size,
                'start_paragraph': chunk_id * 10,
                'end_paragraph': chunk_id * 10 + current_chunk.count('\n\n')
            })

        return chunks

    def _get_overlap_text(self, text: str) -> str:
        """Získání posledních N znaků pro překryv"""
        if len(text) <= self.config.overlap_size:
            return text
        return text[-self.config.overlap_size:]

    def _hierarchical_compression(self,
                                 chunks: List[Dict[str, Any]],
                                 target_compression: float) -> Dict[str, Any]:
        """
        Hierarchická komprese chunků

        Args:
            chunks: Seznam chunků k sumarizaci
            target_compression: Cílový poměr komprese

        Returns:
            Výsledky hierarchické komprese
        """
        compression_tree = []
        current_level = 0
        current_chunks = chunks.copy()

        logger.info(f"🌳 Začínám hierarchickou kompresi s {len(current_chunks)} chunky")

        while len(current_chunks) > 1 and current_level < self.config.max_recursion_depth:
            logger.info(f"📊 Úroveň {current_level}: {len(current_chunks)} chunků")

            # Sumarizace všech chunků na této úrovni
            level_summaries = []

            for chunk in current_chunks:
                summary = self._summarize_chunk(
                    chunk['text'],
                    target_length=int(len(chunk['text']) * target_compression)
                )

                level_summaries.append({
                    'id': f"level_{current_level}_chunk_{chunk['id']}",
                    'text': summary,
                    'size': len(summary),
                    'original_chunk_id': chunk['id'],
                    'compression_ratio': len(summary) / len(chunk['text'])
                })

            # Uložit úroveň do stromu
            compression_tree.append({
                'level': current_level,
                'input_chunks': len(current_chunks),
                'output_summaries': len(level_summaries),
                'total_input_size': sum(c['size'] for c in current_chunks),
                'total_output_size': sum(s['size'] for s in level_summaries)
            })

            # Příprava pro další úroveň
            current_chunks = level_summaries
            current_level += 1

            # Memory cleanup
            gc.collect()

        # Finální sumarizace
        if len(current_chunks) == 1:
            final_summary = current_chunks[0]['text']
        else:
            # Spojení zbývajících chunků a finální sumarizace
            combined_text = "\n\n".join(chunk['text'] for chunk in current_chunks)
            final_summary = self._summarize_chunk(
                combined_text,
                target_length=int(len(combined_text) * target_compression)
            )

        logger.info(f"✅ Hierarchická komprese dokončena po {current_level} úrovních")

        return {
            'final_summary': final_summary,
            'compression_tree': compression_tree,
            'recursion_levels': current_level
        }

    def _summarize_chunk(self, text: str, target_length: int) -> str:
        """
        Sumarizace jednotlivého chunky

        Args:
            text: Text k sumarizaci
            target_length: Cílová délka v znacích

        Returns:
            Sumarizovaný text
        """
        # Cache kontrola
        cache_key = self._get_cache_key(text, target_length)
        cached_result = self._get_from_cache(cache_key)

        if cached_result:
            self._stats['cache_hits'] += 1
            return cached_result

        try:
            # Příprava promptu
            target_words = max(50, target_length // 6)  # Aproximace znaků na slova
            prompt = self.config.summary_template.format(target_length=target_words)
            full_prompt = f"{prompt}\n\nText:\n{text}\n\nSouhrn:"

            # LLM sumarizace
            if self.llm_client:
                summary = self._call_llm(full_prompt)
            else:
                # Fallback: jednoduchá extrakce prvních vět
                summary = self._simple_summarize(text, target_length)

            # Cache uložení
            self._save_to_cache(cache_key, summary)

            return summary

        except Exception as e:
            logger.warning(f"⚠️ Chyba při sumarizaci, použiji fallback: {e}")
            return self._simple_summarize(text, target_length)

    def _call_llm(self, prompt: str) -> str:
        """Volání LLM pro sumarizaci"""
        try:
            # Předpokládáme, že llm_client má metodu generate
            if hasattr(self.llm_client, 'generate'):
                response = self.llm_client.generate(
                    prompt=prompt,
                    max_tokens=min(512, len(prompt) // 4),  # Conservative pro M1
                    temperature=0.3,  # Nízká teplota pro konzistentní souhrny
                )
                return response.strip()
            else:
                logger.warning("LLM client nemá metodu 'generate'")
                return self._simple_summarize(prompt.split("Text:\n")[1].split("\n\nSouhrn:")[0], 500)

        except Exception as e:
            logger.warning(f"LLM volání selhalo: {e}")
            raise

    def _simple_summarize(self, text: str, target_length: int) -> str:
        """Jednoduchá fallback sumarizace"""
        sentences = text.split('. ')

        if len(text) <= target_length:
            return text

        # Vezmi první N vět do cílové délky
        summary = ""
        for sentence in sentences:
            if len(summary) + len(sentence) <= target_length:
                summary += sentence + ". "
            else:
                break

        return summary.strip()

    def _get_cache_key(self, text: str, target_length: int) -> str:
        """Vytvoření cache klíče"""
        content = f"{text}:{target_length}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """Načtení z cache"""
        if cache_key in self._cache:
            return self._cache[cache_key]

        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    summary = data.get('summary')
                    if summary:
                        self._cache[cache_key] = summary
                        return summary
            except Exception as e:
                logger.debug(f"Cache read error: {e}")

        return None

    def _save_to_cache(self, cache_key: str, summary: str):
        """Uložení do cache"""
        self._cache[cache_key] = summary

        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'summary': summary,
                    'timestamp': time.time()
                }, f)
        except Exception as e:
            logger.debug(f"Cache write error: {e}")

    def batch_summarize_documents(self,
                                 documents: List[str],
                                 target_compression: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Batch sumarizace více dokumentů s M1 optimalizacemi

        Args:
            documents: Seznam dokumentů
            target_compression: Cílový poměr komprese

        Returns:
            Seznam výsledků sumarizace
        """
        target_compression = target_compression or self.config.compression_ratio
        results = []

        logger.info(f"📚 Batch sumarizace {len(documents)} dokumentů")

        start_time = time.time()

        for i, document in enumerate(documents, 1):
            logger.info(f"📄 Zpracovávám dokument {i}/{len(documents)}")

            try:
                result = self.summarize_document(document, target_compression)
                results.append(result)

                # M1 Memory management
                if i % 5 == 0:  # GC každých 5 dokumentů
                    gc.collect()
                    logger.debug(f"🧹 Memory cleanup po {i} dokumentech")

            except Exception as e:
                logger.error(f"❌ Chyba při zpracování dokumentu {i}: {e}")
                results.append({
                    'error': str(e),
                    'original_text': document[:100] + "...",
                    'final_summary': None
                })

        total_time = time.time() - start_time
        logger.info(f"✅ Batch sumarizace dokončena za {total_time:.2f}s")
        logger.info(f"📊 Průměrný čas na dokument: {total_time/len(documents):.2f}s")

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Získání statistik sumarizátoru"""
        with self._lock:
            avg_compression = (self._stats['compression_achieved'] /
                             max(1, self._stats['documents_processed']))

            return {
                **self._stats,
                'average_compression_ratio': avg_compression,
                'cache_size': len(self._cache),
                'cache_hit_rate': (self._stats['cache_hits'] /
                                 max(1, self._stats['documents_processed']))
            }


# Utility funkce pro jednoduché použití
def create_m1_summarizer(llm_client=None,
                        max_chunk_size: int = 2000,
                        compression_ratio: float = 0.3) -> M1HierarchicalSummarizer:
    """Factory funkce pro vytvoření M1-optimalizovaného sumarizátoru"""
    config = HierarchicalConfig(
        max_chunk_size=max_chunk_size,
        compression_ratio=compression_ratio
    )

    return M1HierarchicalSummarizer(
        llm_client=llm_client,
        config=config
    )


if __name__ == "__main__":
    # Test hierarchické sumarizace
    print("🧪 Testování M1 hierarchické sumarizace...")

    # Test dokument
    test_document = """
    Umělá inteligence (AI) je oblast počítačové vědy, která se zaměřuje na vytváření 
    systémů schopných vykonávat úkoly, které typicky vyžadují lidskou inteligenci.
    
    Historie AI sahá do 50. let 20. století, kdy Alan Turing navrhl test pro určení,
    zda stroj může vykazovat inteligentní chování nerozeznatelné od člověka.
    
    Moderní AI zahrnuje mnoho různých přístupů, včetně strojového učení, hlubokého učení,
    zpracování přirozeného jazyka, počítačového vidění a robotiky.
    
    Strojové učení je podoblast AI, která umožňuje počítačům učit se a zlepšovat
    ze zkušeností bez explicitního programování pro každou specifickou úlohu.
    
    Hluboké učení je specifický typ strojového učení inspirovaný strukturou
    a funkcí lidského mozku, využívající neuronové sítě s mnoha vrstvami.
    
    Aplikace AI jsou rozšířené v mnoha oblastech, včetně zdravotnictví, financí,
    dopravy, vzdělávání a zábavy. AI systémy pomáhají s diagnózou nemocí,
    automatizací obchodování, řízením autonomních vozidel a personalizací obsahu.
    """ * 5  # Zvětšení pro test

    # Vytvoření sumarizátoru (bez LLM klienta - použije fallback)
    summarizer = create_m1_summarizer(
        max_chunk_size=1000,
        compression_ratio=0.4
    )

    # Test sumarizace
    result = summarizer.summarize_document(test_document)

    print(f"📊 Výsledky:")
    print(f"Vstupní délka: {result['statistics']['input_length']} znaků")
    print(f"Výstupní délka: {result['statistics']['output_length']} znaků")
    print(f"Komprese: {result['statistics']['compression_ratio']:.1%}")
    print(f"Čas zpracování: {result['statistics']['processing_time_seconds']:.2f}s")
    print(f"Počet chunků: {result['statistics']['chunks_created']}")
    print(f"Úrovní rekurze: {result['statistics']['recursion_levels']}")

    print(f"\n📝 Souhrn:")
    print(result['final_summary'][:200] + "...")

    print("✅ Test dokončen!")
