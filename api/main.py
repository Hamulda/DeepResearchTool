"""
API Gateway - Hlavní rozhraní pro DeepResearchTool mikroslužby
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import redis
import dramatiq
from dramatiq.brokers.redis import RedisBroker
import os
import logging
from datetime import datetime
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis broker setup
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.from_url(redis_url)
broker = RedisBroker(url=redis_url)
dramatiq.set_broker(broker)

app = FastAPI(
    title="DeepResearchTool API",
    description="Mikroslužbová architektura pro pokročilý web research",
    version="1.0.0",
)


# Pydantic models
class ScrapeRequest(BaseModel):
    url: str
    task_id: Optional[str] = None


class ScrapeResponse(BaseModel):
    task_id: str
    status: str
    message: str


class TaskStatus(BaseModel):
    task_id: str
    status: str
    created_at: str
    updated_at: str
    details: Dict[str, Any]


# Import workers tasks
from workers.acquisition_worker import scrape_url_task


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "DeepResearchTool API Gateway",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Test Redis connection
        redis_client.ping()
        redis_status = "healthy"
    except Exception as e:
        redis_status = f"unhealthy: {str(e)}"

    return {"api": "healthy", "redis": redis_status, "timestamp": datetime.now().isoformat()}


@app.post("/scrape", response_model=ScrapeResponse)
async def scrape_url(request: ScrapeRequest):
    """Spustí scraping úlohu"""
    try:
        # Generuj task_id pokud není poskytnut
        task_id = request.task_id or f"scrape_{uuid.uuid4().hex[:8]}"

        # Odešli úlohu do acquisition queue
        message = scrape_url_task.send(request.url, task_id)

        # Ulož task status do Redis
        task_status = {
            "task_id": task_id,
            "status": "queued",
            "url": request.url,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "message_id": message.message_id,
        }

        redis_client.setex(f"task:{task_id}", 3600, str(task_status))  # 1 hour expiry

        return ScrapeResponse(
            task_id=task_id, status="queued", message=f"Scraping task queued for URL: {request.url}"
        )

    except Exception as e:
        logger.error(f"Error creating scrape task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/task/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Získej status úlohy"""
    try:
        # Načti status z Redis
        task_data = redis_client.get(f"task:{task_id}")

        if not task_data:
            raise HTTPException(status_code=404, detail="Task not found")

        # Pro nyní vracíme základní status - později přidáme detailnější sledování
        return TaskStatus(
            task_id=task_id,
            status="processing",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            details={"message": "Task is being processed"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
async def list_tasks(limit: int = 10):
    """Seznam posledních úloh"""
    try:
        # Načti klíče úloh z Redis
        task_keys = redis_client.keys("task:*")

        tasks = []
        for key in task_keys[:limit]:
            task_id = key.decode().split(":")[-1]
            tasks.append(
                {"task_id": task_id, "status": "unknown"}  # Později přidáme detailnější sledování
            )

        return {"tasks": tasks, "total": len(task_keys), "showing": len(tasks)}

    except Exception as e:
        logger.error(f"Error listing tasks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Statistiky systému"""
    try:
        # Redis statistiky
        redis_info = redis_client.info()

        return {
            "redis": {
                "connected_clients": redis_info.get("connected_clients", 0),
                "used_memory_human": redis_info.get("used_memory_human", "unknown"),
                "total_commands_processed": redis_info.get("total_commands_processed", 0),
            },
            "queues": {
                "acquisition": redis_client.llen("dramatiq:default.DQ"),
                "processing": redis_client.llen("dramatiq:processing.DQ"),
            },
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
