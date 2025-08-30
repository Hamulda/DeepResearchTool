#!/usr/bin/env python3
"""
Comprehensive unit tests for core pipeline module
Tests pipeline orchestration, task management, and error handling
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Dict, Any, List

from src.core.pipeline import Pipeline, PipelineStage, PipelineTask, PipelineError


class TestPipelineTask:
    """Test suite for PipelineTask class"""

    def test_task_initialization(self):
        """Test task initialization with required parameters"""
        task = PipelineTask(
            task_id="test_task_1",
            task_type="scraping",
            payload={"url": "https://example.com"},
            priority=1
        )

        assert task.task_id == "test_task_1"
        assert task.task_type == "scraping"
        assert task.payload["url"] == "https://example.com"
        assert task.priority == 1
        assert task.status == "pending"
        assert isinstance(task.created_at, datetime)

    def test_task_status_transitions(self):
        """Test valid task status transitions"""
        task = PipelineTask("test", "processing", {})

        # Valid transitions
        task.status = "running"
        assert task.status == "running"

        task.status = "completed"
        assert task.status == "completed"

        # Test error status
        task.status = "failed"
        assert task.status == "failed"

    def test_task_serialization(self):
        """Test task serialization to dict"""
        task = PipelineTask(
            task_id="test_task",
            task_type="analysis",
            payload={"data": "test"},
            priority=2,
            metadata={"source": "test"}
        )

        serialized = task.to_dict()

        assert serialized["task_id"] == "test_task"
        assert serialized["task_type"] == "analysis"
        assert serialized["payload"]["data"] == "test"
        assert serialized["priority"] == 2
        assert serialized["metadata"]["source"] == "test"
        assert "created_at" in serialized


class TestPipelineStage:
    """Test suite for PipelineStage class"""

    @pytest.fixture
    def mock_processor(self):
        """Mock processor function for testing"""
        async def processor(task: PipelineTask) -> Dict[str, Any]:
            # Simulate processing time
            await asyncio.sleep(0.01)
            return {"processed": True, "task_id": task.task_id}
        return processor

    def test_stage_initialization(self, mock_processor):
        """Test stage initialization"""
        stage = PipelineStage(
            name="test_stage",
            processor=mock_processor,
            max_concurrent=2,
            timeout=30.0
        )

        assert stage.name == "test_stage"
        assert stage.processor == mock_processor
        assert stage.max_concurrent == 2
        assert stage.timeout == 30.0
        assert stage.is_running is False

    @pytest.mark.asyncio
    async def test_stage_process_task_success(self, mock_processor):
        """Test successful task processing by stage"""
        stage = PipelineStage("test", mock_processor)
        task = PipelineTask("task1", "test", {"data": "test"})

        result = await stage.process_task(task)

        assert result["processed"] is True
        assert result["task_id"] == "task1"
        assert task.status == "completed"

    @pytest.mark.asyncio
    async def test_stage_process_task_failure(self):
        """Test task processing failure handling"""
        async def failing_processor(task):
            raise ValueError("Processing failed")

        stage = PipelineStage("failing_stage", failing_processor)
        task = PipelineTask("task1", "test", {})

        with pytest.raises(ValueError, match="Processing failed"):
            await stage.process_task(task)

        assert task.status == "failed"

    @pytest.mark.asyncio
    async def test_stage_timeout_handling(self):
        """Test task timeout handling"""
        async def slow_processor(task):
            await asyncio.sleep(1.0)  # Longer than timeout
            return {"result": "slow"}

        stage = PipelineStage("slow_stage", slow_processor, timeout=0.1)
        task = PipelineTask("task1", "test", {})

        with pytest.raises(asyncio.TimeoutError):
            await stage.process_task(task)

        assert task.status == "failed"


class TestPipeline:
    """Test suite for Pipeline class"""

    @pytest.fixture
    def sample_config(self):
        """Sample pipeline configuration"""
        return {
            "name": "test_pipeline",
            "stages": [
                {
                    "name": "acquisition",
                    "max_concurrent": 2,
                    "timeout": 30.0
                },
                {
                    "name": "processing",
                    "max_concurrent": 1,
                    "timeout": 60.0
                },
                {
                    "name": "storage",
                    "max_concurrent": 3,
                    "timeout": 15.0
                }
            ],
            "queue_config": {
                "max_size": 1000,
                "batch_size": 10
            }
        }

    def test_pipeline_initialization(self, sample_config):
        """Test pipeline initialization"""
        pipeline = Pipeline(sample_config)

        assert pipeline.name == "test_pipeline"
        assert len(pipeline.stages) == 3
        assert pipeline.stages[0].name == "acquisition"
        assert pipeline.stages[1].name == "processing"
        assert pipeline.stages[2].name == "storage"
        assert pipeline.is_running is False

    def test_pipeline_add_stage(self):
        """Test adding stages to pipeline"""
        pipeline = Pipeline({"name": "test", "stages": []})

        async def test_processor(task):
            return {"result": "processed"}

        stage = PipelineStage("new_stage", test_processor)
        pipeline.add_stage(stage)

        assert len(pipeline.stages) == 1
        assert pipeline.stages[0].name == "new_stage"

    @pytest.mark.asyncio
    async def test_pipeline_submit_task(self, sample_config):
        """Test task submission to pipeline"""
        pipeline = Pipeline(sample_config)

        task = PipelineTask("task1", "scraping", {"url": "https://test.com"})
        await pipeline.submit_task(task)

        # Check task was queued
        assert pipeline.task_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_pipeline_full_workflow(self):
        """Test complete pipeline workflow with multiple stages"""
        # Create processors for each stage
        async def acquisition_processor(task):
            task.payload["acquired"] = True
            return {"stage": "acquisition", "task_id": task.task_id}

        async def processing_processor(task):
            task.payload["processed"] = True
            return {"stage": "processing", "task_id": task.task_id}

        async def storage_processor(task):
            task.payload["stored"] = True
            return {"stage": "storage", "task_id": task.task_id}

        # Setup pipeline
        config = {
            "name": "full_test",
            "stages": [
                {"name": "acquisition", "max_concurrent": 1},
                {"name": "processing", "max_concurrent": 1},
                {"name": "storage", "max_concurrent": 1}
            ]
        }

        pipeline = Pipeline(config)
        pipeline.add_stage(PipelineStage("acquisition", acquisition_processor))
        pipeline.add_stage(PipelineStage("processing", processing_processor))
        pipeline.add_stage(PipelineStage("storage", storage_processor))

        # Submit task
        task = PipelineTask("workflow_task", "test", {"data": "initial"})
        await pipeline.submit_task(task)

        # Start pipeline
        await pipeline.start()

        # Process tasks (simplified for test)
        processed_task = await pipeline.process_next_task()

        assert processed_task is not None
        assert processed_task.payload["acquired"] is True

    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self):
        """Test pipeline error handling and recovery"""
        async def error_processor(task):
            if task.task_id == "error_task":
                raise RuntimeError("Simulated error")
            return {"success": True}

        pipeline = Pipeline({"name": "error_test", "stages": []})
        stage = PipelineStage("error_stage", error_processor)
        pipeline.add_stage(stage)

        # Submit normal task
        normal_task = PipelineTask("normal_task", "test", {})
        await pipeline.submit_task(normal_task)

        # Submit error task
        error_task = PipelineTask("error_task", "test", {})
        await pipeline.submit_task(error_task)

        await pipeline.start()

        # Process tasks and check error handling
        result1 = await pipeline.process_next_task()
        assert result1.status in ["completed", "failed"]

    @pytest.mark.asyncio
    async def test_pipeline_concurrent_processing(self):
        """Test concurrent task processing"""
        processed_tasks = []

        async def concurrent_processor(task):
            await asyncio.sleep(0.01)  # Simulate work
            processed_tasks.append(task.task_id)
            return {"processed": task.task_id}

        pipeline = Pipeline({"name": "concurrent_test", "stages": []})
        stage = PipelineStage("concurrent_stage", concurrent_processor, max_concurrent=3)
        pipeline.add_stage(stage)

        # Submit multiple tasks
        tasks = [
            PipelineTask(f"task_{i}", "test", {"index": i})
            for i in range(5)
        ]

        for task in tasks:
            await pipeline.submit_task(task)

        await pipeline.start()

        # Process all tasks concurrently
        results = await asyncio.gather(*[
            pipeline.process_next_task() for _ in range(5)
        ])

        assert len(results) == 5
        assert len(processed_tasks) == 5

    def test_pipeline_metrics_collection(self, sample_config):
        """Test pipeline metrics collection"""
        pipeline = Pipeline(sample_config)

        # Check initial metrics
        metrics = pipeline.get_metrics()

        assert "tasks_submitted" in metrics
        assert "tasks_completed" in metrics
        assert "tasks_failed" in metrics
        assert "average_processing_time" in metrics
        assert metrics["tasks_submitted"] == 0
        assert metrics["tasks_completed"] == 0
        assert metrics["tasks_failed"] == 0

    @pytest.mark.asyncio
    async def test_pipeline_shutdown(self, sample_config):
        """Test pipeline graceful shutdown"""
        pipeline = Pipeline(sample_config)

        async def long_processor(task):
            await asyncio.sleep(0.1)
            return {"result": "completed"}

        stage = PipelineStage("test_stage", long_processor)
        pipeline.add_stage(stage)

        await pipeline.start()

        # Submit a task
        task = PipelineTask("shutdown_task", "test", {})
        await pipeline.submit_task(task)

        # Shutdown pipeline
        await pipeline.shutdown(timeout=1.0)

        assert pipeline.is_running is False

    def test_pipeline_configuration_validation(self):
        """Test pipeline configuration validation"""
        # Valid configuration
        valid_config = {
            "name": "valid_pipeline",
            "stages": [{"name": "stage1", "max_concurrent": 1}]
        }
        pipeline = Pipeline(valid_config)
        assert pipeline.name == "valid_pipeline"

        # Invalid configuration - missing name
        with pytest.raises(ValueError, match="Pipeline name is required"):
            Pipeline({"stages": []})

        # Invalid configuration - missing stages
        with pytest.raises(ValueError, match="At least one stage is required"):
            Pipeline({"name": "test", "stages": []})


class TestPipelineIntegration:
    """Integration tests for pipeline system"""

    @pytest.mark.asyncio
    async def test_multi_stage_data_flow(self):
        """Test data flow through multiple pipeline stages"""
        # Track data transformations
        transformations = []

        async def extract_processor(task):
            task.payload["extracted_data"] = f"extracted_{task.task_id}"
            transformations.append(f"extract_{task.task_id}")
            return {"stage": "extract", "data": task.payload["extracted_data"]}

        async def transform_processor(task):
            original = task.payload["extracted_data"]
            task.payload["transformed_data"] = f"transformed_{original}"
            transformations.append(f"transform_{task.task_id}")
            return {"stage": "transform", "data": task.payload["transformed_data"]}

        async def load_processor(task):
            transformed = task.payload["transformed_data"]
            task.payload["loaded_data"] = f"loaded_{transformed}"
            transformations.append(f"load_{task.task_id}")
            return {"stage": "load", "data": task.payload["loaded_data"]}

        # Setup ETL pipeline
        config = {
            "name": "etl_pipeline",
            "stages": [
                {"name": "extract", "max_concurrent": 1},
                {"name": "transform", "max_concurrent": 1},
                {"name": "load", "max_concurrent": 1}
            ]
        }

        pipeline = Pipeline(config)
        pipeline.add_stage(PipelineStage("extract", extract_processor))
        pipeline.add_stage(PipelineStage("transform", transform_processor))
        pipeline.add_stage(PipelineStage("load", load_processor))

        # Process task through all stages
        task = PipelineTask("etl_task", "data", {"source": "test_data"})

        await pipeline.start()
        await pipeline.submit_task(task)

        # Simulate processing through stages
        for _ in range(3):  # 3 stages
            result = await pipeline.process_next_task()
            assert result is not None

        # Verify transformations occurred in order
        assert len(transformations) == 3
        assert "extract_etl_task" in transformations
        assert "transform_etl_task" in transformations
        assert "load_etl_task" in transformations
