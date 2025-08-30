"""Autonomní server pro Fázi 1 - spojuje ELT pipeline, RAG systém a lokální LLM.
Poskytuje REST API pro kompletní research workflow.
"""

from contextlib import asynccontextmanager
from datetime import datetime
import os
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response
import structlog
import uvicorn

from .elt_pipeline import ELTPipeline
from .local_llm import LLMConfig, ModelDownloader, RAGLLMPipeline
from .rag_system import LocalRAGSystem

# Konfigurace strukturovaného logování
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Prometheus metriky
REQUEST_COUNT = Counter("autonomous_requests_total", "Total requests", ["method", "endpoint"])
REQUEST_DURATION = Histogram("autonomous_request_duration_seconds", "Request duration")
DOCUMENTS_INDEXED = Counter("autonomous_documents_indexed_total", "Documents indexed")
QUERIES_PROCESSED = Counter("autonomous_queries_processed_total", "Queries processed")


# Pydantic modely pro API
class DocumentData(BaseModel):
    id: str
    title: str
    content: str
    url: str
    source: str = "api"
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)
    score_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    use_rag: bool = True


class ConversationMessage(BaseModel):
    role: str = Field(..., regex="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[ConversationMessage]
    use_rag: bool = True


class IndexingRequest(BaseModel):
    table_name: str
    content_column: str = "content"
    chunk_size: int = Field(default=500, ge=100, le=2000)


class SystemStats(BaseModel):
    elt_pipeline: dict[str, Any]
    rag_system: dict[str, Any]
    llm_engine: dict[str, Any]
    uptime_seconds: float
    total_documents: int
    total_queries: int


class AutonomousServer:
    """Hlavní třída autonomního serveru."""

    def __init__(self):
        self.start_time = datetime.now()
        self.data_dir = Path(os.getenv("DATA_DIR", "./data"))
        self.models_dir = Path(os.getenv("MODELS_DIR", "./models"))

        # Inicializace komponent
        self.elt_pipeline = None
        self.rag_system = None
        self.rag_llm_pipeline = None

        self.app = FastAPI(
            title="Autonomous Research Platform",
            description="Fáze 1: Základní architektura s ELT, RAG a lokálním LLM",
            version="1.0.0",
            lifespan=self.lifespan,
        )

        self._setup_middleware()
        self._setup_routes()

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """Lifecycle management pro FastAPI aplikaci."""
        # Startup
        await self._initialize_components()
        logger.info("Autonomous server started successfully")

        yield

        # Shutdown
        await self._cleanup_components()
        logger.info("Autonomous server shut down")

    def _setup_middleware(self):
        """Nastaví middleware pro CORS a další funkce."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        """Nastaví API routes."""

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            }

        @self.app.get("/metrics")
        async def metrics():
            """Prometheus metriky endpoint."""
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

        @self.app.post("/documents/ingest")
        async def ingest_documents(
            documents: list[DocumentData], background_tasks: BackgroundTasks
        ):
            """Ingestuje dokumenty do ELT pipeline."""
            REQUEST_COUNT.labels(method="POST", endpoint="/documents/ingest").inc()

            try:
                # Konverze do formátu pro ELT pipeline
                async def document_stream():
                    for doc in documents:
                        yield {
                            "id": doc.id,
                            "title": doc.title,
                            "content": doc.content,
                            "url": doc.url,
                            "source": doc.source,
                            "metadata": doc.metadata,
                            "ingested_at": datetime.now().isoformat(),
                        }

                # Spuštění ingestu na pozadí
                background_tasks.add_task(
                    self._ingest_documents_background, document_stream(), "api_documents"
                )

                DOCUMENTS_INDEXED.inc(len(documents))

                return {
                    "status": "accepted",
                    "documents_count": len(documents),
                    "message": "Documents are being processed in background",
                }

            except Exception as e:
                logger.error(f"Failed to ingest documents: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/query")
        async def query_knowledge_base(request: QueryRequest):
            """Dotazuje se na znalostní bázi pomocí RAG + LLM."""
            REQUEST_COUNT.labels(method="POST", endpoint="/query").inc()

            with REQUEST_DURATION.time():
                try:
                    if not self.rag_llm_pipeline:
                        raise HTTPException(
                            status_code=503, detail="RAG-LLM pipeline not initialized"
                        )

                    result = await self.rag_llm_pipeline.answer_question(
                        question=request.question,
                        top_k=request.top_k,
                        score_threshold=request.score_threshold,
                    )

                    QUERIES_PROCESSED.inc()

                    return result

                except Exception as e:
                    logger.error(f"Failed to process query: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/chat")
        async def chat_conversation(request: ChatRequest):
            """Konverzační rozhraní s RAG podporou."""
            REQUEST_COUNT.labels(method="POST", endpoint="/chat").inc()

            try:
                if not self.rag_llm_pipeline:
                    raise HTTPException(status_code=503, detail="RAG-LLM pipeline not initialized")

                # Konverze Pydantic modelů na slovníky
                messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

                result = await self.rag_llm_pipeline.chat_conversation(
                    messages=messages, use_rag=request.use_rag
                )

                QUERIES_PROCESSED.inc()

                return result

            except Exception as e:
                logger.error(f"Failed to process chat: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/index")
        async def index_documents(request: IndexingRequest, background_tasks: BackgroundTasks):
            """Indexuje dokumenty z Parquet souboru do vektorové databáze."""
            REQUEST_COUNT.labels(method="POST", endpoint="/index").inc()

            try:
                if not self.rag_system:
                    raise HTTPException(status_code=503, detail="RAG system not initialized")

                # Spuštění indexování na pozadí
                background_tasks.add_task(
                    self._index_documents_background,
                    request.table_name,
                    request.content_column,
                    request.chunk_size,
                )

                return {
                    "status": "accepted",
                    "table_name": request.table_name,
                    "message": "Indexing started in background",
                }

            except Exception as e:
                logger.error(f"Failed to start indexing: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/stats", response_model=SystemStats)
        async def get_system_stats():
            """Vrací statistiky systému."""
            REQUEST_COUNT.labels(method="GET", endpoint="/stats").inc()

            try:
                uptime = (datetime.now() - self.start_time).total_seconds()

                # Získání statistik z komponent
                elt_stats = (
                    {"status": "active"} if self.elt_pipeline else {"status": "not_initialized"}
                )
                rag_stats = (
                    self.rag_system.get_system_stats()
                    if self.rag_system
                    else {"status": "not_initialized"}
                )
                pipeline_stats = (
                    self.rag_llm_pipeline.get_pipeline_stats()
                    if self.rag_llm_pipeline
                    else {"status": "not_initialized"}
                )

                return SystemStats(
                    elt_pipeline=elt_stats,
                    rag_system=rag_stats,
                    llm_engine=pipeline_stats.get("llm_engine", {}),
                    uptime_seconds=uptime,
                    total_documents=int(DOCUMENTS_INDEXED._value.sum()),
                    total_queries=int(QUERIES_PROCESSED._value.sum()),
                )

            except Exception as e:
                logger.error(f"Failed to get system stats: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/models")
        async def list_available_models():
            """Vrací seznam dostupných LLM modelů."""
            return {
                "available_models": ModelDownloader.list_available_models(),
                "models_directory": str(self.models_dir),
                "currently_loaded": (
                    self.rag_llm_pipeline.llm_engine.get_model_info()
                    if self.rag_llm_pipeline
                    else None
                ),
            }

    async def _initialize_components(self):
        """Inicializuje všechny komponenty systému."""
        try:
            # Vytvoření adresářů
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.models_dir.mkdir(parents=True, exist_ok=True)

            # ELT Pipeline
            self.elt_pipeline = ELTPipeline(
                data_dir=self.data_dir / "parquet", chunk_size=int(os.getenv("CHUNK_SIZE", "1000"))
            )

            # RAG System
            self.rag_system = LocalRAGSystem(
                data_dir=self.data_dir / "parquet",
                model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            )

            # LLM konfigurace
            model_path = os.getenv("LLM_MODEL_PATH")
            if model_path and Path(model_path).exists():
                llm_config = LLMConfig(
                    model_path=model_path,
                    n_ctx=int(os.getenv("LLM_CONTEXT", "4096")),
                    n_threads=int(os.getenv("LLM_THREADS", "8")),
                    temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
                    metal=os.getenv("LLM_METAL", "true").lower() == "true",
                )

                # RAG-LLM Pipeline
                self.rag_llm_pipeline = RAGLLMPipeline(
                    rag_system=self.rag_system, llm_config=llm_config
                )

                logger.info("All components initialized successfully")
            else:
                logger.warning("LLM model not found, RAG-LLM pipeline not initialized")
                logger.info("Use /models endpoint to see available models")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    async def _cleanup_components(self):
        """Vyčistí komponenty při shutdown."""
        try:
            if self.elt_pipeline:
                self.elt_pipeline.cleanup()

            if self.rag_system:
                self.rag_system.cleanup()

            logger.info("Components cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def _ingest_documents_background(self, document_stream, table_name: str):
        """Background task pro ingest dokumentů."""
        try:
            await self.elt_pipeline.extract_and_load(
                data_stream=document_stream, table_name=table_name, source="api"
            )

            logger.info(f"Document ingestion completed for table: {table_name}")

        except Exception as e:
            logger.error(f"Background ingestion failed: {e}")

    async def _index_documents_background(
        self, table_name: str, content_column: str, chunk_size: int
    ):
        """Background task pro indexování dokumentů."""
        try:
            await self.rag_system.index_documents_from_parquet(
                table_name=table_name, content_column=content_column, chunk_size=chunk_size
            )

            logger.info(f"Document indexing completed for table: {table_name}")

        except Exception as e:
            logger.error(f"Background indexing failed: {e}")


def create_app() -> FastAPI:
    """Factory funkce pro vytvoření FastAPI aplikace."""
    server = AutonomousServer()
    return server.app


def run_server():
    """Spustí autonomní server."""
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8000"))
    workers = int(os.getenv("SERVER_WORKERS", "1"))

    logger.info("Starting autonomous server", host=host, port=port, workers=workers)

    uvicorn.run(
        "src.core.autonomous_server:create_app",
        factory=True,
        host=host,
        port=port,
        workers=workers,
        reload=False,
        access_log=True,
    )


if __name__ == "__main__":
    run_server()
