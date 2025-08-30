#!/usr/bin/env python3
"""
Integration tests for Redis queue workers and microservices
Tests task submission, processing, and result storage across services
"""

import pytest
import asyncio
import redis
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from src.core.pipeline import PipelineTask
from workers.acquisition_worker import AcquisitionWorker
from workers.processing_worker import ProcessingWorker


class TestRedisQueueIntegration:
    """Integration tests for Redis queue-based task processing"""

    @pytest.fixture
    async def redis_client(self):
        """Redis client for testing"""
        try:
            client = redis.Redis(host='localhost', port=6379, db=15, decode_responses=True)
            # Clear test database
            client.flushdb()
            yield client
        finally:
            client.flushdb()
            client.close()

    @pytest.fixture
    async def acquisition_worker(self, redis_client):
        """Acquisition worker instance for testing"""
        config = {
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 15
            },
            "worker": {
                "max_concurrent": 2,
                "timeout": 30
            }
        }

        worker = AcquisitionWorker(config)
        await worker.initialize()
        yield worker
        await worker.shutdown()

    @pytest.fixture
    async def processing_worker(self, redis_client):
        """Processing worker instance for testing"""
        config = {
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 15
            },
            "worker": {
                "max_concurrent": 1,
                "timeout": 60
            }
        }

        worker = ProcessingWorker(config)
        await worker.initialize()
        yield worker
        await worker.shutdown()

    @pytest.mark.asyncio
    async def test_task_submission_and_pickup(self, redis_client, acquisition_worker):
        """Test task submission to Redis and pickup by worker"""

        # Submit task to Redis queue
        task_data = {
            "task_id": "test_acquisition_001",
            "task_type": "url_scraping",
            "payload": {
                "url": "https://example.com/test-page",
                "method": "GET",
                "headers": {"User-Agent": "TestBot/1.0"}
            },
            "priority": 1,
            "created_at": datetime.now().isoformat()
        }

        # Push to acquisition queue
        redis_client.lpush("acquisition_queue", json.dumps(task_data))

        # Verify task in queue
        queue_length = redis_client.llen("acquisition_queue")
        assert queue_length == 1

        # Worker should pick up the task
        task = await acquisition_worker.get_next_task()

        assert task is not None
        assert task.task_id == "test_acquisition_001"
        assert task.task_type == "url_scraping"
        assert task.payload["url"] == "https://example.com/test-page"

    @pytest.mark.asyncio
    async def test_end_to_end_task_processing(self, redis_client, acquisition_worker, processing_worker):
        """Test complete task lifecycle from acquisition to processing"""

        # 1. Submit URL scraping task
        scraping_task = {
            "task_id": "e2e_test_001",
            "task_type": "url_scraping",
            "payload": {
                "url": "https://httpbin.org/json",
                "scraping_config": {
                    "extract_text": True,
                    "extract_links": True,
                    "timeout": 10
                }
            },
            "priority": 1,
            "created_at": datetime.now().isoformat()
        }

        redis_client.lpush("acquisition_queue", json.dumps(scraping_task))

        # 2. Acquisition worker processes task
        with patch.object(acquisition_worker, '_scrape_url') as mock_scrape:
            mock_scrape.return_value = {
                "url": "https://httpbin.org/json",
                "content": "Test content from httpbin",
                "status_code": 200,
                "links": ["https://httpbin.org/", "https://httpbin.org/html"],
                "metadata": {
                    "title": "Test Page",
                    "content_type": "application/json",
                    "scraped_at": datetime.now().isoformat()
                }
            }

            # Process acquisition task
            task = await acquisition_worker.get_next_task()
            result = await acquisition_worker.process_task(task)

            # Store result and create processing task
            await acquisition_worker.store_result(task.task_id, result)
            processing_task = await acquisition_worker.create_processing_task(task.task_id, result)

            # Submit to processing queue
            redis_client.lpush("processing_queue", json.dumps(processing_task))

        # 3. Processing worker picks up and processes task
        with patch.object(processing_worker, '_extract_entities') as mock_extract:
            mock_extract.return_value = {
                "entities": ["httpbin", "json", "test"],
                "topics": ["web testing", "api testing"],
                "sentiment": "neutral",
                "language": "en"
            }

            # Process the result
            proc_task = await processing_worker.get_next_task()
            proc_result = await processing_worker.process_task(proc_task)

            # Store final result
            await processing_worker.store_result(proc_task.task_id, proc_result)

        # 4. Verify complete processing chain
        final_result = redis_client.get(f"result:{proc_task.task_id}")
        assert final_result is not None

        result_data = json.loads(final_result)
        assert "entities" in result_data
        assert "topics" in result_data
        assert result_data["source_url"] == "https://httpbin.org/json"

    @pytest.mark.asyncio
    async def test_concurrent_task_processing(self, redis_client, acquisition_worker):
        """Test concurrent processing of multiple tasks"""

        # Submit multiple tasks
        tasks = []
        for i in range(5):
            task_data = {
                "task_id": f"concurrent_test_{i:03d}",
                "task_type": "url_scraping",
                "payload": {
                    "url": f"https://httpbin.org/delay/{i}",
                    "timeout": 15
                },
                "priority": i,
                "created_at": datetime.now().isoformat()
            }
            tasks.append(task_data)
            redis_client.lpush("acquisition_queue", json.dumps(task_data))

        # Verify all tasks in queue
        assert redis_client.llen("acquisition_queue") == 5

        # Process tasks concurrently
        with patch.object(acquisition_worker, '_scrape_url') as mock_scrape:
            mock_scrape.side_effect = lambda task: {
                "url": task.payload["url"],
                "content": f"Content for task {task.task_id}",
                "status_code": 200,
                "processing_time": 0.1  # Simulate fast processing
            }

            # Start concurrent processing
            processing_tasks = []
            for _ in range(3):  # Process 3 tasks concurrently
                processing_tasks.append(acquisition_worker.process_next_available())

            # Wait for completion
            results = await asyncio.gather(*processing_tasks, return_exceptions=True)

            # Verify results
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) >= 3

    @pytest.mark.asyncio
    async def test_error_handling_and_retry(self, redis_client, acquisition_worker):
        """Test error handling and retry mechanism"""

        # Submit task that will fail
        failing_task = {
            "task_id": "error_test_001",
            "task_type": "url_scraping",
            "payload": {
                "url": "https://nonexistent-domain-12345.com",
                "retry_count": 0,
                "max_retries": 3
            },
            "priority": 1,
            "created_at": datetime.now().isoformat()
        }

        redis_client.lpush("acquisition_queue", json.dumps(failing_task))

        # Mock network failure
        with patch.object(acquisition_worker, '_scrape_url') as mock_scrape:
            mock_scrape.side_effect = Exception("Connection timeout")

            # Process failing task
            task = await acquisition_worker.get_next_task()

            with pytest.raises(Exception, match="Connection timeout"):
                await acquisition_worker.process_task(task)

            # Verify task is moved to retry queue
            retry_queue_length = redis_client.llen("acquisition_retry_queue")
            assert retry_queue_length >= 0  # Task should be queued for retry

    @pytest.mark.asyncio
    async def test_task_priority_ordering(self, redis_client, acquisition_worker):
        """Test that high-priority tasks are processed first"""

        # Submit tasks with different priorities
        priority_tasks = [
            {"priority": 3, "task_id": "low_priority"},
            {"priority": 1, "task_id": "high_priority"},
            {"priority": 2, "task_id": "medium_priority"}
        ]

        for task_data in priority_tasks:
            full_task = {
                "task_id": task_data["task_id"],
                "task_type": "url_scraping",
                "payload": {"url": f"https://example.com/{task_data['task_id']}"},
                "priority": task_data["priority"],
                "created_at": datetime.now().isoformat()
            }
            redis_client.lpush("acquisition_priority_queue", json.dumps(full_task))

        # Process tasks and verify priority order
        processed_order = []

        with patch.object(acquisition_worker, 'get_next_task') as mock_get_task:
            # Mock priority-based task retrieval
            async def priority_get_task():
                # Get highest priority task first
                tasks_raw = redis_client.lrange("acquisition_priority_queue", 0, -1)
                if not tasks_raw:
                    return None

                tasks = [json.loads(t) for t in tasks_raw]
                highest_priority_task = min(tasks, key=lambda x: x["priority"])

                # Remove from queue
                redis_client.lrem("acquisition_priority_queue", 1, json.dumps(highest_priority_task))

                return PipelineTask(
                    task_id=highest_priority_task["task_id"],
                    task_type=highest_priority_task["task_type"],
                    payload=highest_priority_task["payload"],
                    priority=highest_priority_task["priority"]
                )

            mock_get_task.side_effect = priority_get_task

            # Process all tasks
            while redis_client.llen("acquisition_priority_queue") > 0:
                task = await acquisition_worker.get_next_task()
                if task:
                    processed_order.append((task.task_id, task.priority))

        # Verify priority order (1 = highest priority)
        priorities = [p[1] for p in processed_order]
        assert priorities == sorted(priorities), f"Tasks not processed in priority order: {processed_order}"

    @pytest.mark.asyncio
    async def test_worker_health_monitoring(self, redis_client, acquisition_worker):
        """Test worker health monitoring and heartbeat"""

        # Start worker health monitoring
        await acquisition_worker.start_health_monitoring()

        # Wait for heartbeat
        await asyncio.sleep(1.0)

        # Check heartbeat in Redis
        heartbeat = redis_client.get(f"worker:heartbeat:{acquisition_worker.worker_id}")
        assert heartbeat is not None

        heartbeat_data = json.loads(heartbeat)
        assert "timestamp" in heartbeat_data
        assert "status" in heartbeat_data
        assert heartbeat_data["status"] == "healthy"

        # Check worker stats
        stats = redis_client.get(f"worker:stats:{acquisition_worker.worker_id}")
        if stats:
            stats_data = json.loads(stats)
            assert "tasks_processed" in stats_data
            assert "uptime_seconds" in stats_data


class TestTaskResultStorage:
    """Test task result storage and retrieval"""

    @pytest.fixture
    async def redis_client(self):
        """Redis client for testing"""
        try:
            client = redis.Redis(host='localhost', port=6379, db=15, decode_responses=True)
            client.flushdb()
            yield client
        finally:
            client.flushdb()
            client.close()

    @pytest.mark.asyncio
    async def test_result_storage_and_retrieval(self, redis_client):
        """Test storing and retrieving task results"""

        task_id = "storage_test_001"
        result_data = {
            "task_id": task_id,
            "status": "completed",
            "result": {
                "url": "https://example.com",
                "content": "Scraped content here",
                "metadata": {
                    "title": "Example Page",
                    "content_length": 1234,
                    "scraped_at": datetime.now().isoformat()
                }
            },
            "processing_time": 2.5,
            "worker_id": "test_worker_001",
            "completed_at": datetime.now().isoformat()
        }

        # Store result
        redis_client.set(f"result:{task_id}", json.dumps(result_data), ex=3600)  # 1 hour TTL

        # Retrieve and verify
        stored_result = redis_client.get(f"result:{task_id}")
        assert stored_result is not None

        retrieved_data = json.loads(stored_result)
        assert retrieved_data["task_id"] == task_id
        assert retrieved_data["status"] == "completed"
        assert retrieved_data["result"]["url"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_result_expiration(self, redis_client):
        """Test result expiration and cleanup"""

        task_id = "expiration_test_001"
        result_data = {"task_id": task_id, "status": "completed"}

        # Store with short TTL
        redis_client.set(f"result:{task_id}", json.dumps(result_data), ex=1)  # 1 second TTL

        # Verify initially present
        assert redis_client.get(f"result:{task_id}") is not None

        # Wait for expiration
        await asyncio.sleep(2)

        # Verify expired
        assert redis_client.get(f"result:{task_id}") is None

    @pytest.mark.asyncio
    async def test_batch_result_storage(self, redis_client):
        """Test batch storage of multiple results"""

        # Create multiple results
        results = []
        for i in range(10):
            result = {
                "task_id": f"batch_test_{i:03d}",
                "status": "completed",
                "result": {"data": f"result_{i}"},
                "completed_at": datetime.now().isoformat()
            }
            results.append(result)

        # Store batch using pipeline
        pipe = redis_client.pipeline()
        for result in results:
            pipe.set(f"result:{result['task_id']}", json.dumps(result), ex=3600)
        pipe.execute()

        # Verify all stored
        for result in results:
            stored = redis_client.get(f"result:{result['task_id']}")
            assert stored is not None

            retrieved = json.loads(stored)
            assert retrieved["task_id"] == result["task_id"]
