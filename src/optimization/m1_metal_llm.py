"""
M1 Metal-optimalizovaný LLM klient
Plné využití Metal Performance Shaders pro dramatické zrychlení inference
Optimalizované pro MacBook Air M1 (8GB RAM) s kvantizovanými modely
"""

import gc
import logging
import time
import threading
from typing import Dict, Any, Optional, List, Union, Generator
from dataclasses import dataclass
from pathlib import Path
import psutil
import json

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = torch.backends.mps.is_available()
    logger.info(f"🔥 Metal Performance Shaders: {'✅ Dostupné' if TORCH_AVAILABLE else '❌ Nedostupné'}")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch není dostupný")

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama není dostupný")


@dataclass
class M1ModelConfig:
    """Konfigurace pro M1-optimalizované modely"""
    model_name: str = "llama3:8b-instruct-q4_K_M"  # Kvantizovaný model
    max_tokens: int = 2048  # Konzervativní pro 8GB RAM
    temperature: float = 0.7
    top_p: float = 0.9
    repeat_penalty: float = 1.1

    # M1-specifické optimalizace
    use_metal: bool = True
    num_gpu_layers: int = -1  # Všechny vrstvy na GPU
    num_threads: int = 4  # Využití P-cores
    context_window: int = 4096  # Rozumná velikost kontextu

    # Memory management
    low_vram: bool = True  # Optimalizace pro unified memory
    batch_size: int = 1  # Konzervativní batch size
    offload_kqv: bool = True  # Offload key-query-value na GPU


class M1MetalLLMClient:
    """
    M1 Metal-optimalizovaný LLM klient

    Klíčové optimalizace:
    - Metal Performance Shaders pro GPU acceleration
    - Kvantizované modely (Q4_K_M) pro paměťovou efektivitu
    - Smart batch processing s M1 memory constraints
    - Automatické fallback mechanismy
    - Token/s monitoring a optimalizace
    """

    def __init__(self,
                 config: Optional[M1ModelConfig] = None,
                 model_cache_dir: Optional[str] = None):
        self.config = config or M1ModelConfig()
        self.model_cache_dir = Path(model_cache_dir) if model_cache_dir else Path("./models")

        # Performance tracking
        self._stats = {
            'total_tokens_generated': 0,
            'total_inference_time': 0.0,
            'requests_count': 0,
            'metal_acceleration_used': False,
            'average_tokens_per_second': 0.0
        }

        # Thread safety
        self._lock = threading.Lock()

        # Model state
        self._model_loaded = False
        self._last_memory_check = 0

        # Inicializace
        self._setup_metal_environment()
        self._validate_model_availability()

        logger.info(f"🚀 M1 Metal LLM klient inicializován")
        logger.info(f"🤖 Model: {self.config.model_name}")
        logger.info(f"🔥 Metal: {'✅' if self.config.use_metal and TORCH_AVAILABLE else '❌'}")

    def _setup_metal_environment(self):
        """Nastavení Metal environment pro M1"""
        if not TORCH_AVAILABLE:
            logger.warning("⚠️ Metal Performance Shaders nejsou dostupné")
            self.config.use_metal = False
            return

        try:
            # Nastavení PyTorch pro MPS
            if torch.backends.mps.is_available():
                torch.backends.mps.empty_cache()
                self._stats['metal_acceleration_used'] = True
                logger.info("✅ Metal Performance Shaders aktivovány")
            else:
                logger.warning("⚠️ MPS není dostupný, použiji CPU")
                self.config.use_metal = False

        except Exception as e:
            logger.error(f"❌ Chyba při nastavení Metal: {e}")
            self.config.use_metal = False

    def _validate_model_availability(self):
        """Kontrola dostupnosti modelu"""
        if not OLLAMA_AVAILABLE:
            logger.error("❌ Ollama není dostupný. Nainstalujte: brew install ollama")
            return

        try:
            # Kontrola běžícího Ollama serveru
            models = ollama.list()
            available_models = [model['name'] for model in models['models']]

            if self.config.model_name in available_models:
                logger.info(f"✅ Model {self.config.model_name} je dostupný")
            else:
                logger.warning(f"⚠️ Model {self.config.model_name} není dostupný")
                logger.info(f"📋 Dostupné modely: {available_models}")

                # Pokus o pull modelu
                self._pull_model_if_needed()

        except Exception as e:
            logger.error(f"❌ Chyba při kontrole modelu: {e}")

    def _pull_model_if_needed(self):
        """Stažení modelu pokud není dostupný"""
        try:
            logger.info(f"📥 Stahuji model {self.config.model_name}...")
            ollama.pull(self.config.model_name)
            logger.info(f"✅ Model {self.config.model_name} úspěšně stažen")
        except Exception as e:
            logger.error(f"❌ Chyba při stahování modelu: {e}")

    def generate(self,
                prompt: str,
                max_tokens: Optional[int] = None,
                temperature: Optional[float] = None,
                stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """
        Generování textu s M1 Metal optimalizacemi

        Args:
            prompt: Vstupní prompt
            max_tokens: Maximální počet tokenů (None = použij config)
            temperature: Teplota (None = použij config)
            stream: Streamování výstupu

        Returns:
            Vygenerovaný text nebo generator pro streaming
        """
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature

        logger.debug(f"🧠 Generuji odpověď (max_tokens: {max_tokens}, temp: {temperature})")

        # Memory check před inference
        self._check_memory_pressure()

        start_time = time.time()

        try:
            # Příprava Ollama parametrů s M1 optimalizacemi
            options = self._prepare_ollama_options(max_tokens, temperature)

            if stream:
                return self._generate_stream(prompt, options)
            else:
                return self._generate_complete(prompt, options, start_time)

        except Exception as e:
            logger.error(f"❌ Chyba při generování: {e}")
            raise

    def _prepare_ollama_options(self, max_tokens: int, temperature: float) -> Dict[str, Any]:
        """Příprava Ollama options s M1 optimalizacemi"""
        options = {
            'num_predict': max_tokens,
            'temperature': temperature,
            'top_p': self.config.top_p,
            'repeat_penalty': self.config.repeat_penalty,

            # M1-specifické optimalizace
            'num_gpu': self.config.num_gpu_layers if self.config.use_metal else 0,
            'num_thread': self.config.num_threads,
            'num_ctx': self.config.context_window,

            # Memory optimalizace
            'low_vram': self.config.low_vram,
            'f16_kv': True,  # Half precision pro key-value cache
            'use_mlock': False,  # Nenechat model v paměti permanent
        }

        return options

    def _generate_complete(self, prompt: str, options: Dict[str, Any], start_time: float) -> str:
        """Kompletní generování (non-streaming)"""
        try:
            response = ollama.generate(
                model=self.config.model_name,
                prompt=prompt,
                options=options,
                stream=False
            )

            generated_text = response['response']

            # Performance tracking
            inference_time = time.time() - start_time
            tokens_generated = len(generated_text.split())  # Aproximace
            tokens_per_second = tokens_generated / inference_time if inference_time > 0 else 0

            # Update stats
            with self._lock:
                self._stats['total_tokens_generated'] += tokens_generated
                self._stats['total_inference_time'] += inference_time
                self._stats['requests_count'] += 1
                self._stats['average_tokens_per_second'] = (
                    self._stats['total_tokens_generated'] / self._stats['total_inference_time']
                    if self._stats['total_inference_time'] > 0 else 0
                )

            logger.debug(f"✅ Generování dokončeno: {tokens_generated} tokenů za {inference_time:.2f}s ({tokens_per_second:.1f} tok/s)")

            # Memory cleanup po inference
            self._cleanup_after_inference()

            return generated_text

        except Exception as e:
            logger.error(f"❌ Chyba při kompletním generování: {e}")
            raise

    def _generate_stream(self, prompt: str, options: Dict[str, Any]) -> Generator[str, None, None]:
        """Streamované generování"""
        try:
            response_stream = ollama.generate(
                model=self.config.model_name,
                prompt=prompt,
                options=options,
                stream=True
            )

            for chunk in response_stream:
                if 'response' in chunk:
                    yield chunk['response']

        except Exception as e:
            logger.error(f"❌ Chyba při streamovaném generování: {e}")
            raise

    def batch_generate(self,
                      prompts: List[str],
                      max_tokens: Optional[int] = None,
                      temperature: Optional[float] = None) -> List[str]:
        """
        Batch generování s M1 memory management

        Args:
            prompts: Seznam promptů
            max_tokens: Maximální počet tokenů
            temperature: Teplota

        Returns:
            Seznam vygenerovaných odpovědí
        """
        logger.info(f"🔄 Batch generování {len(prompts)} promptů")

        results = []
        batch_start_time = time.time()

        for i, prompt in enumerate(prompts, 1):
            logger.debug(f"📝 Zpracovávám prompt {i}/{len(prompts)}")

            try:
                result = self.generate(prompt, max_tokens, temperature)
                results.append(result)

                # M1 Memory management
                if i % 3 == 0:  # GC každé 3 requesty
                    self._cleanup_after_inference()

                # Progress log
                if i % 5 == 0 or i == len(prompts):
                    elapsed = time.time() - batch_start_time
                    avg_time = elapsed / i
                    eta = avg_time * (len(prompts) - i)
                    logger.info(f"📊 Pokrok: {i}/{len(prompts)} ({avg_time:.1f}s/prompt, ETA: {eta:.0f}s)")

            except Exception as e:
                logger.error(f"❌ Chyba při zpracování promptu {i}: {e}")
                results.append(f"ERROR: {str(e)}")

        total_time = time.time() - batch_start_time
        logger.info(f"✅ Batch generování dokončeno za {total_time:.2f}s")

        return results

    def _check_memory_pressure(self):
        """Kontrola memory pressure před inference"""
        current_time = time.time()

        # Kontrola pouze každých 30 sekund
        if current_time - self._last_memory_check < 30:
            return

        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            # Varování při vysokém využití paměti
            if memory_mb > 6000:  # 6GB+ na 8GB systému
                logger.warning(f"⚠️ Vysoké využití paměti: {memory_mb:.0f}MB")
                self._cleanup_after_inference()

            # Metal memory cache cleanup
            if self.config.use_metal and TORCH_AVAILABLE:
                torch.backends.mps.empty_cache()

            self._last_memory_check = current_time

        except Exception as e:
            logger.debug(f"Memory check error: {e}")

    def _cleanup_after_inference(self):
        """Cleanup po inference pro M1 optimalizaci"""
        try:
            # Python garbage collection
            collected = gc.collect()

            # Metal cache cleanup
            if self.config.use_metal and TORCH_AVAILABLE:
                torch.backends.mps.empty_cache()

            if collected > 0:
                logger.debug(f"🧹 Cleanup: {collected} objektů uvolněno")

        except Exception as e:
            logger.debug(f"Cleanup error: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Informace o aktuálním modelu"""
        try:
            if not OLLAMA_AVAILABLE:
                return {'error': 'Ollama není dostupný'}

            # Ollama show command pro model info
            model_info = ollama.show(self.config.model_name)

            return {
                'model_name': self.config.model_name,
                'model_info': model_info,
                'metal_enabled': self.config.use_metal and TORCH_AVAILABLE,
                'configuration': {
                    'max_tokens': self.config.max_tokens,
                    'context_window': self.config.context_window,
                    'num_gpu_layers': self.config.num_gpu_layers,
                    'temperature': self.config.temperature
                }
            }

        except Exception as e:
            logger.error(f"❌ Chyba při získávání model info: {e}")
            return {'error': str(e)}

    def get_performance_stats(self) -> Dict[str, Any]:
        """Statistiky výkonu"""
        with self._lock:
            # Memory usage
            try:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
            except:
                memory_mb = 0

            return {
                **self._stats,
                'current_memory_mb': round(memory_mb, 1),
                'metal_acceleration': self.config.use_metal and TORCH_AVAILABLE,
                'model_loaded': self._model_loaded,
                'requests_per_minute': (self._stats['requests_count'] * 60 /
                                      max(1, self._stats['total_inference_time']))
            }

    def optimize_for_m1(self):
        """Spuštění M1-specifických optimalizací"""
        logger.info("⚡ Spouštím M1 LLM optimalizace...")

        # Memory cleanup
        self._cleanup_after_inference()

        # Model cache optimalizace
        if OLLAMA_AVAILABLE:
            try:
                # Preload model do cache
                ollama.generate(
                    model=self.config.model_name,
                    prompt="Test",
                    options={'num_predict': 1}
                )
                self._model_loaded = True
                logger.info("✅ Model preloaded do cache")
            except Exception as e:
                logger.warning(f"⚠️ Model preload selhal: {e}")

        # Performance stats
        stats = self.get_performance_stats()
        logger.info(f"📊 LLM Performance:")
        logger.info(f"   Tokens/s: {stats['average_tokens_per_second']:.1f}")
        logger.info(f"   Memory: {stats['current_memory_mb']}MB")
        logger.info(f"   Metal: {'✅' if stats['metal_acceleration'] else '❌'}")

        return stats


# Utility funkce pro snadné použití
def create_m1_llm_client(model_name: str = "llama3:8b-instruct-q4_K_M",
                        max_tokens: int = 2048,
                        use_metal: bool = True) -> M1MetalLLMClient:
    """Factory funkce pro vytvoření M1-optimalizovaného LLM klienta"""
    config = M1ModelConfig(
        model_name=model_name,
        max_tokens=max_tokens,
        use_metal=use_metal
    )

    return M1MetalLLMClient(config=config)


if __name__ == "__main__":
    # Test M1 Metal LLM klienta
    print("🧪 Testování M1 Metal LLM klienta...")

    # Vytvoření klienta
    client = create_m1_llm_client()

    # Test single generation
    test_prompt = "Vysvětli stručně, co je umělá inteligence."

    print(f"🤖 Test prompt: {test_prompt}")

    try:
        start_time = time.time()
        response = client.generate(test_prompt, max_tokens=100)
        elapsed = time.time() - start_time

        print(f"✅ Odpověď ({elapsed:.2f}s):")
        print(response[:200] + "..." if len(response) > 200 else response)

        # Performance stats
        stats = client.get_performance_stats()
        print(f"\n📊 Performance:")
        print(f"Tokens/s: {stats['average_tokens_per_second']:.1f}")
        print(f"Memory: {stats['current_memory_mb']}MB")
        print(f"Metal: {stats['metal_acceleration']}")

    except Exception as e:
        print(f"❌ Test selhal: {e}")

    print("✅ Test dokončen!")
