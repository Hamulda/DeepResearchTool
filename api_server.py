#!/usr/bin/env python3
"""
REST API Server pro Deep Research Tool
Implementuje FastAPI endpoint pro batch processing a vzdálený přístup

Author: Senior IT Specialist
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import yaml

import structlog

logger = structlog.get_logger(__name__)

# Request/Response modely
class ResearchRequest(BaseModel):
    query: str = Field(..., description="Výzkumný dotaz")
    research_depth: int = Field(3, ge=1, le=5, description="Hloubka výzkumu (1-5)")
    max_documents: int = Field(50, ge=10, le=200, description="Maximum dokumentů")
    profile: str = Field("thorough", description="Profil výzkumu (quick/thorough)")
    include_evidence: bool = Field(True, description="Zahrnout evidence binding")
    human_checkpoints: bool = Field(False, description="Povolit human checkpoints")

class BatchResearchRequest(BaseModel):
    queries: List[ResearchRequest] = Field(..., description="Seznam výzkumných dotazů")
    parallel_limit: int = Field(2, ge=1, le=5, description="Limit paralelních úloh")
    callback_url: Optional[str] = Field(None, description="URL pro callback po dokončení")

class ResearchResponse(BaseModel):
    request_id: str
    query: str
    status: str  # "completed", "failed", "processing"
    verified_claims: List[Dict[str, Any]] = []
    flagged_claims: List[Dict[str, Any]] = []
    evidence_bindings: Dict[str, List[Dict[str, Any]]] = {}
    workflow_metrics: Dict[str, Any] = {}
    citation_summary: List[Dict[str, Any]] = []
    overall_confidence: float = 0.0
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

class BatchResponse(BaseModel):
    batch_id: str
    total_queries: int
    completed: int
    failed: int
    status: str  # "processing", "completed", "failed"
    results: List[ResearchResponse] = []
    created_at: datetime
    estimated_completion: Optional[datetime] = None

class EvaluationRequest(BaseModel):
    test_cases: Optional[List[str]] = Field(None, description="Specifické test cases (nebo všechny)")
    metrics: Optional[List[str]] = Field(None, description="Specifické metriky k měření")
    benchmark_runs: int = Field(3, ge=1, le=10, description="Počet běhů pro každý test case")

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    components: Dict[str, str]

# Global state
app_state = {
    "orchestrator": None,
    "config": None,
    "active_requests": {},
    "batch_requests": {},
    "start_time": datetime.now()
}

# FastAPI app
app = FastAPI(
    title="Deep Research Tool API",
    description="Advanced Research Agent with Evidence-Based Synthesis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def get_orchestrator():
    """Dependency pro získání orchestrátoru"""
    if app_state["orchestrator"] is None:
        raise HTTPException(status_code=503, detail="Orchestrator není inicializován")
    return app_state["orchestrator"]

async def get_config():
    """Dependency pro získání konfigurace"""
    if app_state["config"] is None:
        raise HTTPException(status_code=503, detail="Konfigurace není načtena")
    return app_state["config"]

@app.on_event("startup")
async def startup_event():
    """Inicializace při startu serveru"""
    logger.info("Spouštím Deep Research Tool API server")

    try:
        # Načtení konfigurace
        config_path = "config_m1_local.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            app_state["config"] = yaml.safe_load(f)

        # Inicializace orchestrátoru
        from src.core.dag_workflow_orchestrator import DAGWorkflowOrchestrator
        app_state["orchestrator"] = DAGWorkflowOrchestrator(app_state["config"])
        await app_state["orchestrator"].initialize_engines()

        logger.info("API server úspěšně inicializován")

    except Exception as e:
        logger.error("Chyba při inicializaci API serveru", error=str(e))
        raise

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""

    uptime = (datetime.now() - app_state["start_time"]).total_seconds()

    # Kontrola stavu komponent
    components = {
        "orchestrator": "healthy" if app_state["orchestrator"] else "unhealthy",
        "config": "loaded" if app_state["config"] else "not_loaded"
    }

    # Kontrola dalších komponent
    try:
        if app_state["orchestrator"]:
            # Test Qdrant připojení
            if hasattr(app_state["orchestrator"], 'retrieval_engine'):
                stats = await app_state["orchestrator"].retrieval_engine.get_collection_stats()
                components["qdrant"] = "healthy" if stats.get("qdrant", {}).get("points_count", 0) >= 0 else "unhealthy"
            else:
                components["qdrant"] = "not_initialized"
    except Exception as e:
        components["qdrant"] = f"error: {str(e)}"

    overall_status = "healthy" if all(status in ["healthy", "loaded"] for status in components.values()) else "degraded"

    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        uptime_seconds=uptime,
        components=components
    )

@app.post("/research", response_model=ResearchResponse)
async def create_research_request(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
    orchestrator = Depends(get_orchestrator)
):
    """
    Vytvoření nového výzkumného požadavku

    Spustí asynchronní výzkumný workflow a vrátí request_id pro sledování pokroku.
    """

    request_id = str(uuid.uuid4())

    # Uložení do active requests
    app_state["active_requests"][request_id] = ResearchResponse(
        request_id=request_id,
        query=request.query,
        status="processing",
        created_at=datetime.now()
    )

    # Spuštění background task
    background_tasks.add_task(
        _process_research_request,
        request_id,
        request,
        orchestrator
    )

    logger.info("Nový výzkumný požadavek vytvořen", request_id=request_id, query=request.query)

    return app_state["active_requests"][request_id]

async def _process_research_request(
    request_id: str,
    request: ResearchRequest,
    orchestrator
):
    """Background task pro zpracování výzkumného požadavku"""

    try:
        logger.info("Zpracovávám výzkumný požadavek", request_id=request_id)

        # Spuštění workflow
        result = await orchestrator.execute_research_workflow(
            main_query=request.query,
            research_depth=request.research_depth,
            max_documents=request.max_documents
        )

        # Aktualizace výsledku
        app_state["active_requests"][request_id] = ResearchResponse(
            request_id=request_id,
            query=request.query,
            status="completed",
            verified_claims=result.get("verified_claims", []),
            flagged_claims=result.get("flagged_claims", []),
            evidence_bindings=result.get("evidence_bindings", {}),
            workflow_metrics=result.get("workflow_metrics", {}),
            citation_summary=result.get("citation_summary", []),
            overall_confidence=result.get("overall_confidence", 0.0),
            created_at=app_state["active_requests"][request_id].created_at,
            completed_at=datetime.now()
        )

        logger.info("Výzkumný požadavek dokončen", request_id=request_id)

    except Exception as e:
        logger.error("Chyba při zpracování výzkumného požadavku",
                    request_id=request_id,
                    error=str(e))

        # Aktualizace s chybou
        app_state["active_requests"][request_id] = ResearchResponse(
            request_id=request_id,
            query=request.query,
            status="failed",
            error_message=str(e),
            created_at=app_state["active_requests"][request_id].created_at,
            completed_at=datetime.now()
        )

@app.get("/research/{request_id}", response_model=ResearchResponse)
async def get_research_result(request_id: str):
    """Získání výsledku výzkumného požadavku"""

    if request_id not in app_state["active_requests"]:
        raise HTTPException(status_code=404, detail="Request ID nenalezen")

    return app_state["active_requests"][request_id]

@app.post("/batch", response_model=BatchResponse)
async def create_batch_request(
    request: BatchResearchRequest,
    background_tasks: BackgroundTasks,
    orchestrator = Depends(get_orchestrator)
):
    """
    Vytvoření batch požadavku pro více výzkumných dotazů

    Zpracuje všechny dotazy paralelně s nastaveným limitem.
    """

    batch_id = str(uuid.uuid4())

    # Vytvoření batch response
    batch_response = BatchResponse(
        batch_id=batch_id,
        total_queries=len(request.queries),
        completed=0,
        failed=0,
        status="processing",
        created_at=datetime.now()
    )

    app_state["batch_requests"][batch_id] = batch_response

    # Spuštění background task
    background_tasks.add_task(
        _process_batch_request,
        batch_id,
        request,
        orchestrator
    )

    logger.info("Nový batch požadavek vytvořen",
               batch_id=batch_id,
               queries=len(request.queries))

    return batch_response

async def _process_batch_request(
    batch_id: str,
    request: BatchResearchRequest,
    orchestrator
):
    """Background task pro zpracování batch požadavku"""

    try:
        logger.info("Zpracovávám batch požadavek", batch_id=batch_id)

        # Semafora pro omezení paralelismu
        semaphore = asyncio.Semaphore(request.parallel_limit)

        async def process_single_query(query_request: ResearchRequest):
            async with semaphore:
                try:
                    result = await orchestrator.execute_research_workflow(
                        main_query=query_request.query,
                        research_depth=query_request.research_depth,
                        max_documents=query_request.max_documents
                    )

                    return ResearchResponse(
                        request_id=str(uuid.uuid4()),
                        query=query_request.query,
                        status="completed",
                        verified_claims=result.get("verified_claims", []),
                        flagged_claims=result.get("flagged_claims", []),
                        evidence_bindings=result.get("evidence_bindings", {}),
                        workflow_metrics=result.get("workflow_metrics", {}),
                        citation_summary=result.get("citation_summary", []),
                        overall_confidence=result.get("overall_confidence", 0.0),
                        created_at=datetime.now(),
                        completed_at=datetime.now()
                    )

                except Exception as e:
                    return ResearchResponse(
                        request_id=str(uuid.uuid4()),
                        query=query_request.query,
                        status="failed",
                        error_message=str(e),
                        created_at=datetime.now(),
                        completed_at=datetime.now()
                    )

        # Spuštění všech úloh
        tasks = [process_single_query(query) for query in request.queries]
        results = await asyncio.gather(*tasks)

        # Aktualizace batch response
        completed = len([r for r in results if r.status == "completed"])
        failed = len([r for r in results if r.status == "failed"])

        app_state["batch_requests"][batch_id] = BatchResponse(
            batch_id=batch_id,
            total_queries=len(request.queries),
            completed=completed,
            failed=failed,
            status="completed",
            results=results,
            created_at=app_state["batch_requests"][batch_id].created_at
        )

        logger.info("Batch požadavek dokončen",
                   batch_id=batch_id,
                   completed=completed,
                   failed=failed)

        # Callback pokud je nastaven
        if request.callback_url:
            await _send_callback(request.callback_url, app_state["batch_requests"][batch_id])

    except Exception as e:
        logger.error("Chyba při zpracování batch požadavku",
                    batch_id=batch_id,
                    error=str(e))

        app_state["batch_requests"][batch_id].status = "failed"

async def _send_callback(callback_url: str, batch_result: BatchResponse):
    """Odeslání callback s výsledkem batch požadavku"""

    import aiohttp

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                callback_url,
                json=batch_result.dict(),
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    logger.info("Callback úspěšně odeslán", url=callback_url)
                else:
                    logger.warning("Callback selhal", url=callback_url, status=response.status)

    except Exception as e:
        logger.error("Chyba při odesílání callback", url=callback_url, error=str(e))

@app.get("/batch/{batch_id}", response_model=BatchResponse)
async def get_batch_result(batch_id: str):
    """Získání výsledku batch požadavku"""

    if batch_id not in app_state["batch_requests"]:
        raise HTTPException(status_code=404, detail="Batch ID nenalezen")

    return app_state["batch_requests"][batch_id]

@app.post("/evaluate")
async def run_evaluation(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    config = Depends(get_config)
):
    """
    Spuštění evaluace na regression test set

    Měří recall@k, evidence coverage, answer faithfulness a další metriky.
    """

    eval_id = str(uuid.uuid4())

    # Spuštění background task
    background_tasks.add_task(
        _process_evaluation_request,
        eval_id,
        request,
        config
    )

    return {
        "evaluation_id": eval_id,
        "status": "started",
        "message": "Evaluace spuštěna na pozadí. Zkontrolujte /evaluate/{eval_id} pro výsledky."
    }

async def _process_evaluation_request(
    eval_id: str,
    request: EvaluationRequest,
    config: dict
):
    """Background task pro evaluaci"""

    try:
        from src.evaluation.evaluation_system import EvaluationSystem

        # Aktualizace konfigurace podle požadavku
        if request.metrics:
            config["evaluation"]["metrics"] = request.metrics
        if request.benchmark_runs:
            config["evaluation"]["benchmark_runs"] = request.benchmark_runs

        eval_system = EvaluationSystem(config)
        await eval_system.initialize()

        result = await eval_system.run_full_evaluation()

        # Uložení výsledků
        results_file = Path(f"./evaluation_results/eval_{eval_id}.json")
        results_file.parent.mkdir(exist_ok=True)

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)

        logger.info("Evaluace dokončena", eval_id=eval_id, file=str(results_file))

    except Exception as e:
        logger.error("Chyba při evaluaci", eval_id=eval_id, error=str(e))

@app.get("/evaluate/{eval_id}")
async def get_evaluation_result(eval_id: str):
    """Získání výsledku evaluace"""

    results_file = Path(f"./evaluation_results/eval_{eval_id}.json")

    if not results_file.exists():
        raise HTTPException(status_code=404, detail="Evaluation ID nenalezen nebo ještě není dokončen")

    with open(results_file, 'r', encoding='utf-8') as f:
        result = json.load(f)

    return result

@app.get("/status")
async def get_system_status():
    """Získání stavu systému a aktivních požadavků"""

    active_count = len([req for req in app_state["active_requests"].values() if req.status == "processing"])
    batch_count = len([batch for batch in app_state["batch_requests"].values() if batch.status == "processing"])

    return {
        "active_research_requests": active_count,
        "active_batch_requests": batch_count,
        "total_completed_requests": len([req for req in app_state["active_requests"].values() if req.status == "completed"]),
        "total_failed_requests": len([req for req in app_state["active_requests"].values() if req.status == "failed"]),
        "uptime_seconds": (datetime.now() - app_state["start_time"]).total_seconds(),
        "memory_usage": {
            "active_requests_count": len(app_state["active_requests"]),
            "batch_requests_count": len(app_state["batch_requests"])
        }
    }

@app.delete("/cleanup")
async def cleanup_completed_requests():
    """Vyčištění dokončených požadavků z paměti"""

    # Odstranění dokončených/chybných požadavků starších než 1 hodina
    cutoff_time = datetime.now().timestamp() - 3600  # 1 hodina

    removed_requests = 0
    removed_batches = 0

    # Cleanup individual requests
    to_remove = []
    for request_id, request in app_state["active_requests"].items():
        if (request.status in ["completed", "failed"] and
            request.completed_at and
            request.completed_at.timestamp() < cutoff_time):
            to_remove.append(request_id)

    for request_id in to_remove:
        del app_state["active_requests"][request_id]
        removed_requests += 1

    # Cleanup batch requests
    to_remove_batches = []
    for batch_id, batch in app_state["batch_requests"].items():
        if (batch.status in ["completed", "failed"] and
            batch.created_at.timestamp() < cutoff_time):
            to_remove_batches.append(batch_id)

    for batch_id in to_remove_batches:
        del app_state["batch_requests"][batch_id]
        removed_batches += 1

    return {
        "removed_requests": removed_requests,
        "removed_batches": removed_batches,
        "remaining_requests": len(app_state["active_requests"]),
        "remaining_batches": len(app_state["batch_requests"])
    }

if __name__ == "__main__":
    # Spuštění API serveru
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
