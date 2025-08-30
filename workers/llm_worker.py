"""
LLM Worker - Lokální inference pro RAG systém (Fáze 4)
Poskytuje API pro generování textu pomocí lokálních modelů
"""

import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.7
    context: Optional[List[str]] = None


class GenerationResponse(BaseModel):
    text: str
    model: str
    tokens_used: int
    processing_time: float


class RecursiveRAGRequest(BaseModel):
    query: str
    max_iterations: int = 3
    context_limit: int = 5


class LLMInferenceEngine:
    """Lokální LLM inference engine s podporou různých backend"""

    def __init__(self):
        self.model_name = os.getenv("MODEL_NAME", "mistral-7b-instruct")
        self.models_dir = Path("/app/models")
        self.models_dir.mkdir(exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.backend = None

        # Inicializuj model
        self._initialize_model()

    def _initialize_model(self):
        """Inicializuj lokální model s fallback strategií"""
        try:
            # Pokus 1: Apple MLX (optimalizováno pro M1/M2)
            if self._try_initialize_mlx():
                logger.info("✅ Používám Apple MLX backend")
                return

            # Pokus 2: Transformers s quantization
            if self._try_initialize_transformers():
                logger.info("✅ Používám HuggingFace Transformers backend")
                return

            # Pokus 3: llama.cpp
            if self._try_initialize_llamacpp():
                logger.info("✅ Používám llama.cpp backend")
                return

            # Fallback: Mock model pro testování
            self._initialize_mock_model()
            logger.warning("⚠️ Používám mock model - pouze pro testování!")

        except Exception as e:
            logger.error(f"❌ Chyba při inicializaci LLM: {e}")
            self._initialize_mock_model()

    def _try_initialize_mlx(self) -> bool:
        """Pokus o inicializaci Apple MLX"""
        try:
            import mlx.core as mx
            from mlx_lm import load, generate

            # Načti model pomocí MLX
            self.model, self.tokenizer = load(f"mlx-community/{self.model_name}")
            self.backend = "mlx"
            return True

        except ImportError:
            logger.info("MLX není dostupný")
            return False
        except Exception as e:
            logger.warning(f"MLX inicializace selhala: {e}")
            return False

    def _try_initialize_transformers(self) -> bool:
        """Pokus o inicializaci HuggingFace Transformers"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            model_path = f"microsoft/{self.model_name}"

            # Načti tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, cache_dir=str(self.models_dir)
            )

            # Načti model s quantization pro úsporu paměti
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                cache_dir=str(self.models_dir),
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True,  # 8-bit quantization
            )

            self.backend = "transformers"
            return True

        except Exception as e:
            logger.warning(f"Transformers inicializace selhala: {e}")
            return False

    def _try_initialize_llamacpp(self) -> bool:
        """Pokus o inicializaci llama.cpp"""
        try:
            from llama_cpp import Llama

            # Hledej GGUF model soubory
            gguf_files = list(self.models_dir.glob("*.gguf"))
            if not gguf_files:
                logger.info("Žádné GGUF soubory nenalezeny")
                return False

            model_path = gguf_files[0]

            self.model = Llama(model_path=str(model_path), n_ctx=2048, n_threads=4, verbose=False)

            self.backend = "llamacpp"
            return True

        except Exception as e:
            logger.warning(f"llama.cpp inicializace selhala: {e}")
            return False

    def _initialize_mock_model(self):
        """Mock model pro testování"""
        self.model = "mock"
        self.tokenizer = "mock"
        self.backend = "mock"

    def generate(
        self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generuj text pomocí načteného modelu"""
        import time

        start_time = time.time()

        try:
            if self.backend == "mlx":
                return self._generate_mlx(prompt, max_tokens, temperature, start_time)
            elif self.backend == "transformers":
                return self._generate_transformers(prompt, max_tokens, temperature, start_time)
            elif self.backend == "llamacpp":
                return self._generate_llamacpp(prompt, max_tokens, temperature, start_time)
            else:
                return self._generate_mock(prompt, max_tokens, temperature, start_time)

        except Exception as e:
            logger.error(f"❌ Generation error: {e}")
            return {
                "text": f"Error during generation: {str(e)}",
                "model": self.backend,
                "tokens_used": 0,
                "processing_time": time.time() - start_time,
            }

    def _generate_mlx(
        self, prompt: str, max_tokens: int, temperature: float, start_time: float
    ) -> Dict[str, Any]:
        """Generování pomocí Apple MLX"""
        from mlx_lm import generate

        response = generate(
            self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens, temp=temperature
        )

        return {
            "text": response,
            "model": f"mlx-{self.model_name}",
            "tokens_used": len(response.split()),
            "processing_time": time.time() - start_time,
        }

    def _generate_transformers(
        self, prompt: str, max_tokens: int, temperature: float, start_time: float
    ) -> Dict[str, Any]:
        """Generování pomocí HuggingFace Transformers"""
        import torch

        inputs = self.tokenizer.encode(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0][len(inputs[0]) :], skip_special_tokens=True)

        return {
            "text": response,
            "model": f"transformers-{self.model_name}",
            "tokens_used": len(outputs[0]) - len(inputs[0]),
            "processing_time": time.time() - start_time,
        }

    def _generate_llamacpp(
        self, prompt: str, max_tokens: int, temperature: float, start_time: float
    ) -> Dict[str, Any]:
        """Generování pomocí llama.cpp"""
        response = self.model(
            prompt, max_tokens=max_tokens, temperature=temperature, stop=["</s>", "\n\n"]
        )

        generated_text = response["choices"][0]["text"]

        return {
            "text": generated_text,
            "model": f"llamacpp-{self.model_name}",
            "tokens_used": response["usage"]["completion_tokens"],
            "processing_time": time.time() - start_time,
        }

    def _generate_mock(
        self, prompt: str, max_tokens: int, temperature: float, start_time: float
    ) -> Dict[str, Any]:
        """Mock generování pro testování"""
        import time

        time.sleep(0.5)  # Simulace zpracování

        mock_responses = [
            "Based on the provided context, here are the key findings from the research data...",
            "The analysis reveals several important patterns in the collected information...",
            "According to the indexed documents, the main insights include...",
            "The evidence suggests that the primary conclusions are...",
        ]

        import random

        response = random.choice(mock_responses)

        return {
            "text": response,
            "model": "mock-model",
            "tokens_used": len(response.split()),
            "processing_time": time.time() - start_time,
        }


# FastAPI app pro LLM API
app = FastAPI(
    title="DeepResearchTool LLM API",
    description="Lokální LLM inference pro RAG systém",
    version="1.0.0",
)

# Globální instance
llm_engine = None


@app.on_event("startup")
async def startup_event():
    """Inicializuj LLM engine při startu"""
    global llm_engine
    logger.info("🚀 Inicializuji LLM inference engine...")
    llm_engine = LLMInferenceEngine()
    logger.info("✅ LLM API je připraveno")


@app.get("/")
async def root():
    """Health check"""
    return {
        "service": "DeepResearchTool LLM API",
        "model": llm_engine.model_name if llm_engine else "not_loaded",
        "backend": llm_engine.backend if llm_engine else "not_loaded",
        "status": "ready" if llm_engine else "loading",
    }


@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Generuj text pomocí LLM"""
    if not llm_engine:
        raise HTTPException(status_code=503, detail="LLM engine not ready")

    try:
        # Přidej kontext do promptu pokud je poskytnut
        full_prompt = request.prompt
        if request.context:
            context_text = "\n".join(request.context)
            full_prompt = f"Context:\n{context_text}\n\nQuestion: {request.prompt}\n\nAnswer:"

        result = llm_engine.generate(full_prompt, request.max_tokens, request.temperature)

        return GenerationResponse(**result)

    except Exception as e:
        logger.error(f"❌ Generation request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag-query")
async def rag_query(request: RecursiveRAGRequest):
    """Rekurzivní RAG dotaz s automatickým zpřesňováním"""
    try:
        # Simulace rekurzivního RAG workflow
        # V reálné implementaci by se volaly processing worker funkce

        iterations = []
        current_query = request.query

        for i in range(request.max_iterations):
            logger.info(f"🔄 RAG iterace {i+1}: {current_query}")

            # 1. Vyhledej relevantní dokumenty (simulace)
            context_docs = [
                "Document 1: Relevant information about the query...",
                "Document 2: Additional context and details...",
                "Document 3: Supporting evidence and data...",
            ]

            # 2. Generuj odpověď s kontextem
            generation_result = llm_engine.generate(
                f"Context:\n{chr(10).join(context_docs)}\n\nQuestion: {current_query}\n\nProvide a comprehensive answer and suggest 2-3 follow-up questions:",
                max_tokens=512,
                temperature=0.7,
            )

            iteration_result = {
                "iteration": i + 1,
                "query": current_query,
                "context_docs": len(context_docs),
                "response": generation_result["text"],
                "processing_time": generation_result["processing_time"],
            }

            iterations.append(iteration_result)

            # 3. Extrahuj follow-up otázky pro další iteraci (zjednodušeno)
            if i < request.max_iterations - 1:
                # V reálné implementaci by se použil LLM k extrakci follow-up otázek
                follow_up_queries = [
                    f"What are the implications of {current_query}?",
                    f"How does {current_query} relate to current trends?",
                    f"What additional evidence supports {current_query}?",
                ]
                current_query = follow_up_queries[0] if follow_up_queries else current_query

        return {
            "original_query": request.query,
            "iterations": iterations,
            "total_processing_time": sum(it["processing_time"] for it in iterations),
            "final_synthesis": "Based on the recursive analysis, the comprehensive answer incorporates multiple perspectives and follow-up insights.",
        }

    except Exception as e:
        logger.error(f"❌ RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """Seznam dostupných modelů"""
    return {
        "current_model": llm_engine.model_name if llm_engine else None,
        "backend": llm_engine.backend if llm_engine else None,
        "available_models": ["mistral-7b-instruct", "llama-2-7b-chat", "phi-2"],
    }


if __name__ == "__main__":
    port = int(os.getenv("API_PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
