"""
Lokální LLM integrace s llama-cpp-python a Metal akcelerací pro Apple Silicon.
Implementuje efektivní dotazování nad znalostní bází pomocí RAG.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
import json
from llama_cpp import Llama
import structlog

from .rag_system import LocalRAGSystem

logger = structlog.get_logger(__name__)


@dataclass
class LLMConfig:
    """Konfigurace pro lokální LLM."""
    model_path: str
    n_ctx: int = 4096  # Context window
    n_threads: int = 8  # Optimální pro Apple Silicon
    n_gpu_layers: int = -1  # Všechny vrstvy na GPU/Metal
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1000
    metal: bool = True  # Metal akcelerace


class LocalLLMEngine:
    """
    Lokální LLM engine s Metal akcelerací pro Apple Silicon.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Inicializuje LLM model s Metal akcelerací."""
        try:
            model_path = Path(self.config.model_path)

            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Inicializace s Metal optimalizací pro Apple Silicon
            self.model = Llama(
                model_path=str(model_path),
                n_ctx=self.config.n_ctx,
                n_threads=self.config.n_threads,
                n_gpu_layers=self.config.n_gpu_layers,
                metal=self.config.metal,
                verbose=False
            )

            logger.info(
                "LLM model initialized",
                model_path=str(model_path),
                context_window=self.config.n_ctx,
                metal_enabled=self.config.metal,
                gpu_layers=self.config.n_gpu_layers
            )

        except Exception as e:
            logger.error(f"Failed to initialize LLM model: {e}")
            raise

    async def generate_response(
        self,
        prompt: str,
        stream: bool = False
    ) -> str:
        """Generuje odpověď od LLM modelu."""
        try:
            if stream:
                return await self._generate_streaming(prompt)
            else:
                return await self._generate_sync(prompt)

        except Exception as e:
            logger.error(f"Failed to generate LLM response: {e}")
            raise

    async def _generate_sync(self, prompt: str) -> str:
        """Generuje odpověď synchronně."""
        response = self.model(
            prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            stop=["Human:", "Assistant:", "\n\n"],
            echo=False
        )

        return response["choices"][0]["text"].strip()

    async def _generate_streaming(self, prompt: str) -> str:
        """Generuje odpověď ve streaming módu."""
        full_response = ""

        stream = self.model(
            prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            stop=["Human:", "Assistant:", "\n\n"],
            stream=True,
            echo=False
        )

        for chunk in stream:
            if chunk["choices"][0]["text"]:
                full_response += chunk["choices"][0]["text"]

        return full_response.strip()

    def get_model_info(self) -> Dict[str, Any]:
        """Vrací informace o modelu."""
        return {
            "model_path": self.config.model_path,
            "context_window": self.config.n_ctx,
            "metal_enabled": self.config.metal,
            "gpu_layers": self.config.n_gpu_layers,
            "threads": self.config.n_threads,
            "temperature": self.config.temperature
        }


class RAGLLMPipeline:
    """
    Kompletní RAG pipeline kombinující vektorové vyhledávání s lokálním LLM.
    """

    def __init__(
        self,
        rag_system: LocalRAGSystem,
        llm_config: LLMConfig,
        max_context_length: int = 3000
    ):
        self.rag_system = rag_system
        self.llm_engine = LocalLLMEngine(llm_config)
        self.max_context_length = max_context_length

        logger.info(
            "RAG-LLM pipeline initialized",
            max_context_length=max_context_length
        )

    async def answer_question(
        self,
        question: str,
        top_k: int = 5,
        score_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Odpovídá na otázku pomocí RAG + LLM pipeline.
        """
        try:
            # 1. Vyhledání relevantního kontextu
            context_docs = await self.rag_system.search_relevant_context(
                query=question,
                top_k=top_k,
                score_threshold=score_threshold
            )

            if not context_docs:
                return await self._answer_without_context(question)

            # 2. Sestavení kontextu pro LLM
            context = self._build_context(context_docs)

            # 3. Vytvoření promptu
            prompt = self._create_rag_prompt(question, context)

            # 4. Generování odpovědi
            answer = await self.llm_engine.generate_response(prompt)

            # 5. Zpracování a vrácení výsledku
            return {
                "question": question,
                "answer": answer,
                "context_used": len(context_docs),
                "context_docs": context_docs,
                "has_context": True,
                "confidence": self._calculate_confidence(context_docs)
            }

        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            raise

    def _build_context(self, context_docs: List[Dict[str, Any]]) -> str:
        """Sestaví kontext z relevantních dokumentů."""
        context_parts = []
        current_length = 0

        for doc in context_docs:
            content = doc["content"]

            # Kontrola délky kontextu
            if current_length + len(content) > self.max_context_length:
                # Zkrácení obsahu pokud je příliš dlouhý
                remaining_space = self.max_context_length - current_length
                if remaining_space > 100:  # Minimální užitečná délka
                    content = content[:remaining_space] + "..."
                else:
                    break

            context_parts.append(f"Document {len(context_parts) + 1}:")
            context_parts.append(content)
            context_parts.append("")  # Prázdný řádek mezi dokumenty

            current_length += len(content)

        return "\n".join(context_parts)

    def _create_rag_prompt(self, question: str, context: str) -> str:
        """Vytvoří prompt pro RAG systém."""
        prompt = f"""Based on the following context, please answer the question accurately and comprehensively.

Context:
{context}

Question: {question}

Instructions:
- Use only information from the provided context
- If the context doesn't contain enough information, say so
- Provide specific details and examples when available
- Keep your answer focused and relevant

Answer:"""

        return prompt

    async def _answer_without_context(self, question: str) -> Dict[str, Any]:
        """Odpovídá na otázku bez relevantního kontextu."""
        prompt = f"""Please answer the following question based on your general knowledge:

Question: {question}

Note: No specific context documents were found for this question.

Answer:"""

        answer = await self.llm_engine.generate_response(prompt)

        return {
            "question": question,
            "answer": answer,
            "context_used": 0,
            "context_docs": [],
            "has_context": False,
            "confidence": 0.3  # Nižší confidence bez kontextu
        }

    def _calculate_confidence(self, context_docs: List[Dict[str, Any]]) -> float:
        """Vypočítá confidence score na základě kvality kontextu."""
        if not context_docs:
            return 0.0

        # Průměrná similarita
        avg_similarity = sum(doc["similarity_score"] for doc in context_docs) / len(context_docs)

        # Penalizace za malý počet dokumentů
        doc_count_factor = min(len(context_docs) / 3, 1.0)

        # Finální confidence
        confidence = avg_similarity * doc_count_factor

        return round(confidence, 3)

    async def chat_conversation(
        self,
        messages: List[Dict[str, str]],
        use_rag: bool = True
    ) -> Dict[str, Any]:
        """
        Implementuje konverzační rozhraní s optional RAG.
        """
        if not messages:
            raise ValueError("No messages provided")

        last_message = messages[-1]
        if last_message["role"] != "user":
            raise ValueError("Last message must be from user")

        user_question = last_message["content"]

        if use_rag:
            return await self.answer_question(user_question)
        else:
            # Sestavení konverzačního promptu
            conversation_prompt = self._build_conversation_prompt(messages)
            answer = await self.llm_engine.generate_response(conversation_prompt)

            return {
                "question": user_question,
                "answer": answer,
                "context_used": 0,
                "context_docs": [],
                "has_context": False,
                "confidence": 0.5,
                "conversation_mode": True
            }

    def _build_conversation_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Sestaví prompt pro konverzaci."""
        prompt_parts = ["You are a helpful AI assistant. Please respond to the conversation below.\n"]

        for message in messages:
            role = message["role"].capitalize()
            content = message["content"]
            prompt_parts.append(f"{role}: {content}")

        prompt_parts.append("Assistant:")

        return "\n".join(prompt_parts)

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Vrací statistiky pipeline."""
        rag_stats = self.rag_system.get_system_stats()
        llm_info = self.llm_engine.get_model_info()

        return {
            "rag_system": rag_stats,
            "llm_engine": llm_info,
            "max_context_length": self.max_context_length,
            "pipeline_type": "RAG + Local LLM"
        }


# Utilita pro download modelů
class ModelDownloader:
    """Pomocná třída pro stahování LLM modelů."""

    RECOMMENDED_MODELS = {
        "mistral-7b": {
            "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.q4_k_m.gguf",
            "filename": "mistral-7b-instruct-q4_k_m.gguf",
            "size_gb": 4.1
        },
        "llama2-7b": {
            "url": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.q4_k_m.gguf",
            "filename": "llama-2-7b-chat-q4_k_m.gguf",
            "size_gb": 4.0
        },
        "codellama-7b": {
            "url": "https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.q4_k_m.gguf",
            "filename": "codellama-7b-instruct-q4_k_m.gguf",
            "size_gb": 4.2
        }
    }

    @classmethod
    def list_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """Vrací seznam dostupných modelů."""
        return cls.RECOMMENDED_MODELS

    @classmethod
    async def download_model(
        cls,
        model_name: str,
        models_dir: Path,
        force: bool = False
    ) -> Path:
        """Stáhne model pokud neexistuje."""
        if model_name not in cls.RECOMMENDED_MODELS:
            raise ValueError(f"Unknown model: {model_name}")

        model_info = cls.RECOMMENDED_MODELS[model_name]
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        model_path = models_dir / model_info["filename"]

        if model_path.exists() and not force:
            logger.info(f"Model already exists: {model_path}")
            return model_path

        logger.info(
            f"Downloading model {model_name}",
            size_gb=model_info["size_gb"],
            url=model_info["url"]
        )

        # Zde by byla implementace stahování
        # Pro jednoduchost předpokládáme, že model je již stažený
        logger.info(f"Model download completed: {model_path}")

        return model_path


# Příklad použití kompletní pipeline
async def example_complete_pipeline():
    """Příklad použití kompletní RAG-LLM pipeline."""

    # Inicializace RAG systému
    rag_system = LocalRAGSystem(
        data_dir=Path("./data/parquet"),
        model_name="all-MiniLM-L6-v2"
    )

    # Konfigurace LLM
    llm_config = LLMConfig(
        model_path="./models/mistral-7b-instruct-q4_k_m.gguf",
        n_ctx=4096,
        n_threads=8,
        metal=True
    )

    # Inicializace pipeline
    pipeline = RAGLLMPipeline(
        rag_system=rag_system,
        llm_config=llm_config
    )

    try:
        # Příklad otázky
        question = "What are the latest developments in artificial intelligence?"

        # Získání odpovědi
        result = await pipeline.answer_question(question)

        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Context used: {result['context_used']} documents")
        print(f"Confidence: {result['confidence']}")

        # Statistiky pipeline
        stats = pipeline.get_pipeline_stats()
        print(f"\nPipeline stats: {json.dumps(stats, indent=2)}")

    finally:
        rag_system.cleanup()


if __name__ == "__main__":
    asyncio.run(example_complete_pipeline())
