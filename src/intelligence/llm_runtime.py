"""LLM Runtime for On-Device Intelligence
Small quantized LLMs with Metal/MPS acceleration for M1 optimization
Load/unload models to manage 8GB RAM constraints
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
import gc
import logging
from typing import Any

import psutil

logger = logging.getLogger(__name__)

try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama not available - using mock LLM runtime")

try:
    import torch

    TORCH_AVAILABLE = True
    MPS_AVAILABLE = torch.backends.mps.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    MPS_AVAILABLE = False


@dataclass
class ModelProfile:
    """LLM model profile configuration"""

    name: str
    model_id: str
    quantization: str  # q4_K_M, q8_0, q5_K_M
    context_length: int
    parameter_count: str  # 3B, 7B, 13B
    memory_usage_gb: float
    use_case: str  # synthesis, analysis, qa, classification
    performance_score: float = 0.0

    def get_ollama_model_name(self) -> str:
        """Get Ollama-compatible model name"""
        return f"{self.model_id}:{self.quantization}"


class ModelRegistry:
    """Registry of available M1-optimized models"""

    @classmethod
    def get_available_models(cls) -> list[ModelProfile]:
        """Get list of M1-optimized model profiles"""
        return [
            # Qwen2.5 series - excellent for research tasks
            ModelProfile(
                name="Qwen2.5 3B Q4",
                model_id="qwen2.5",
                quantization="3b-instruct-q4_K_M",
                context_length=4096,
                parameter_count="3B",
                memory_usage_gb=2.5,
                use_case="quick_synthesis",
                performance_score=0.85,
            ),
            ModelProfile(
                name="Qwen2.5 7B Q4",
                model_id="qwen2.5",
                quantization="7b-instruct-q4_K_M",
                context_length=8192,
                parameter_count="7B",
                memory_usage_gb=4.5,
                use_case="thorough_analysis",
                performance_score=0.92,
            ),
            # Phi-3 series - Microsoft's efficient models
            ModelProfile(
                name="Phi-3 Mini",
                model_id="phi3",
                quantization="mini-q4_K_M",
                context_length=4096,
                parameter_count="3.8B",
                memory_usage_gb=2.8,
                use_case="classification",
                performance_score=0.80,
            ),
            # Llama 3.2 series - Meta's latest small models
            ModelProfile(
                name="Llama 3.2 3B",
                model_id="llama3.2",
                quantization="3b-instruct-q4_K_M",
                context_length=8192,
                parameter_count="3B",
                memory_usage_gb=2.3,
                use_case="qa",
                performance_score=0.87,
            ),
            # Gemma 2 series - Google's efficient models
            ModelProfile(
                name="Gemma 2 2B",
                model_id="gemma2",
                quantization="2b-instruct-q4_K_M",
                context_length=4096,
                parameter_count="2B",
                memory_usage_gb=1.8,
                use_case="fast_qa",
                performance_score=0.78,
            ),
        ]

    @classmethod
    def get_model_for_use_case(
        cls, use_case: str, max_memory_gb: float = 4.0
    ) -> ModelProfile | None:
        """Get best model for specific use case within memory constraints"""
        models = cls.get_available_models()

        # Filter by memory constraint
        suitable_models = [m for m in models if m.memory_usage_gb <= max_memory_gb]

        if not suitable_models:
            return None

        # Filter by use case
        use_case_models = [m for m in suitable_models if use_case in m.use_case]

        if use_case_models:
            # Return best performing model for use case
            return max(use_case_models, key=lambda m: m.performance_score)
        # Return best general model within memory limit
        return max(suitable_models, key=lambda m: m.performance_score)


@dataclass
class LLMConfig:
    """LLM runtime configuration"""

    # Model selection
    model_profile: ModelProfile | None = None
    auto_select_model: bool = True
    preferred_use_case: str = "synthesis"
    max_memory_gb: float = 4.0

    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048
    context_length: int = 4096

    # Performance settings
    enable_mps: bool = True
    num_threads: int = 8  # M1 has 8 performance cores
    batch_size: int = 1
    enable_streaming: bool = True

    # Memory management
    auto_unload: bool = True
    idle_timeout_minutes: int = 10
    force_gc_after_generation: bool = True


class LLMRuntime:
    """On-device LLM runtime with M1 optimization"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.current_model: ModelProfile | None = None
        self.client = None
        self.last_used = datetime.now()
        self.generation_count = 0
        self.total_tokens_generated = 0

        # Memory monitoring
        self.peak_memory_gb = 0.0
        self.current_memory_gb = 0.0

    async def initialize(self) -> bool:
        """Initialize LLM runtime"""
        logger.info("🧠 Initializing LLM Runtime...")

        if not OLLAMA_AVAILABLE:
            logger.warning("Ollama not available - using mock runtime")
            return False

        try:
            # Initialize Ollama client
            self.client = ollama.AsyncClient()

            # Auto-select model if needed
            if self.config.auto_select_model and not self.config.model_profile:
                self.config.model_profile = ModelRegistry.get_model_for_use_case(
                    self.config.preferred_use_case, self.config.max_memory_gb
                )

            if not self.config.model_profile:
                logger.error("No suitable model found for configuration")
                return False

            # Check if model is available
            await self._ensure_model_available()

            logger.info(f"✅ LLM Runtime initialized with {self.config.model_profile.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize LLM runtime: {e}")
            return False

    async def _ensure_model_available(self) -> bool:
        """Ensure model is downloaded and available"""
        try:
            model_name = self.config.model_profile.get_ollama_model_name()

            # Check if model exists
            models = await self.client.list()
            model_names = [model["name"] for model in models["models"]]

            if model_name not in model_names:
                logger.info(f"📥 Downloading model: {model_name}")
                await self.client.pull(model_name)
                logger.info(f"✅ Model downloaded: {model_name}")

            return True

        except Exception as e:
            logger.error(f"Error ensuring model availability: {e}")
            return False

    async def load_model(self, model_profile: ModelProfile | None = None) -> bool:
        """Load specific model into memory"""
        if model_profile:
            self.config.model_profile = model_profile

        if not self.config.model_profile:
            logger.error("No model profile specified")
            return False

        try:
            # Check memory availability
            available_memory = self._get_available_memory_gb()
            required_memory = self.config.model_profile.memory_usage_gb

            if available_memory < required_memory:
                logger.warning(
                    f"Insufficient memory: {available_memory:.1f}GB available, {required_memory:.1f}GB required"
                )

                # Try to free memory
                if self.current_model:
                    await self.unload_model()
                    gc.collect()
                    available_memory = self._get_available_memory_gb()

                if available_memory < required_memory:
                    return False

            # Load model (Ollama handles this automatically on first use)
            self.current_model = self.config.model_profile
            self.current_memory_gb = required_memory
            self.peak_memory_gb = max(self.peak_memory_gb, self.current_memory_gb)

            logger.info(
                f"📂 Loaded model: {self.current_model.name} ({self.current_memory_gb:.1f}GB)"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    async def unload_model(self) -> bool:
        """Unload current model from memory"""
        if not self.current_model:
            return True

        try:
            # Ollama doesn't have explicit unload, but we can clear our reference
            model_name = self.current_model.name
            self.current_model = None
            self.current_memory_gb = 0.0

            # Force garbage collection
            gc.collect()

            logger.info(f"📤 Unloaded model: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to unload model: {e}")
            return False

    async def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stream: bool = None,
    ) -> str | AsyncIterator[str]:
        """Generate text with the loaded model"""
        if not self.current_model:
            if not await self.load_model():
                raise RuntimeError("No model loaded and failed to load default model")

        # Update last used time
        self.last_used = datetime.now()

        # Use config defaults if not specified
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        stream = stream if stream is not None else self.config.enable_streaming

        try:
            model_name = self.current_model.get_ollama_model_name()

            generation_params = {
                "model": model_name,
                "prompt": prompt,
                "options": {
                    "temperature": temperature,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                    "num_predict": max_tokens,
                    "num_ctx": self.config.context_length,
                    "num_thread": self.config.num_threads,
                },
                "stream": stream,
            }

            if stream:
                return self._generate_streaming(generation_params)
            response = await self.client.generate(**generation_params)

            # Update statistics
            self.generation_count += 1
            self.total_tokens_generated += len(response["response"].split())

            # Force garbage collection if configured
            if self.config.force_gc_after_generation:
                gc.collect()

            return response["response"]

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    async def _generate_streaming(self, params: dict[str, Any]) -> AsyncIterator[str]:
        """Generate streaming response"""
        try:
            async for chunk in await self.client.generate(**params):
                if "response" in chunk:
                    yield chunk["response"]

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise
        finally:
            # Update statistics
            self.generation_count += 1

            if self.config.force_gc_after_generation:
                gc.collect()

    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        stream: bool = None,
    ) -> str | AsyncIterator[str]:
        """Chat-based generation with conversation history"""
        if not self.current_model:
            if not await self.load_model():
                raise RuntimeError("No model loaded and failed to load default model")

        self.last_used = datetime.now()

        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        stream = stream if stream is not None else self.config.enable_streaming

        try:
            model_name = self.current_model.get_ollama_model_name()

            chat_params = {
                "model": model_name,
                "messages": messages,
                "options": {
                    "temperature": temperature,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                    "num_predict": max_tokens,
                    "num_ctx": self.config.context_length,
                    "num_thread": self.config.num_threads,
                },
                "stream": stream,
            }

            if stream:
                return self._chat_streaming(chat_params)
            response = await self.client.chat(**chat_params)

            self.generation_count += 1
            self.total_tokens_generated += len(response["message"]["content"].split())

            if self.config.force_gc_after_generation:
                gc.collect()

            return response["message"]["content"]

        except Exception as e:
            logger.error(f"Chat generation failed: {e}")
            raise

    async def _chat_streaming(self, params: dict[str, Any]) -> AsyncIterator[str]:
        """Generate streaming chat response"""
        try:
            async for chunk in await self.client.chat(**params):
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]

        except Exception as e:
            logger.error(f"Streaming chat failed: {e}")
            raise
        finally:
            self.generation_count += 1

            if self.config.force_gc_after_generation:
                gc.collect()

    def _get_available_memory_gb(self) -> float:
        """Get available system memory in GB"""
        memory = psutil.virtual_memory()
        return memory.available / (1024**3)

    async def auto_unload_if_idle(self) -> bool:
        """Automatically unload model if idle for too long"""
        if not self.config.auto_unload or not self.current_model:
            return False

        idle_time = datetime.now() - self.last_used
        idle_minutes = idle_time.total_seconds() / 60

        if idle_minutes > self.config.idle_timeout_minutes:
            logger.info(f"🔄 Auto-unloading model after {idle_minutes:.1f} minutes idle")
            return await self.unload_model()

        return False

    def get_runtime_stats(self) -> dict[str, Any]:
        """Get runtime statistics"""
        return {
            "current_model": self.current_model.name if self.current_model else None,
            "model_memory_gb": self.current_memory_gb,
            "peak_memory_gb": self.peak_memory_gb,
            "generation_count": self.generation_count,
            "total_tokens_generated": self.total_tokens_generated,
            "last_used": self.last_used.isoformat(),
            "available_memory_gb": self._get_available_memory_gb(),
            "mps_available": MPS_AVAILABLE,
            "ollama_available": OLLAMA_AVAILABLE,
        }

    async def cleanup(self):
        """Cleanup runtime resources"""
        if self.current_model:
            await self.unload_model()

        if self.client:
            # Ollama client doesn't need explicit cleanup
            self.client = None


# Utility functions
async def get_available_models() -> list[dict[str, Any]]:
    """Get list of available models from Ollama"""
    if not OLLAMA_AVAILABLE:
        return []

    try:
        client = ollama.AsyncClient()
        models = await client.list()
        return models["models"]
    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        return []


async def quick_generate(
    prompt: str, use_case: str = "synthesis", max_memory_gb: float = 3.0
) -> str:
    """Quick generation with auto model selection"""
    config = LLMConfig(
        auto_select_model=True,
        preferred_use_case=use_case,
        max_memory_gb=max_memory_gb,
        auto_unload=True,
    )

    runtime = LLMRuntime(config)

    try:
        if await runtime.initialize():
            return await runtime.generate(prompt)
        return "Error: Failed to initialize LLM runtime"
    finally:
        await runtime.cleanup()


__all__ = [
    "LLMConfig",
    "LLMRuntime",
    "ModelProfile",
    "ModelRegistry",
    "get_available_models",
    "quick_generate",
]
