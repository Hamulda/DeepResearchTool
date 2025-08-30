#!/usr/bin/env python3
"""
Enhanced DAG Workflow Orchestrator with Reliability Features
Retriable nodes, exponential backoff, checkpointing and deterministic execution

Author: Senior IT Specialist
"""

import asyncio
import random
import time
import json
import logging
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import pickle


class NodeStatus(Enum):
    """DAG node execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    SKIPPED = "skipped"


@dataclass
class RetryConfig:
    """Retry configuration for DAG nodes"""
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on_exceptions: List[type] = field(default_factory=lambda: [Exception])


@dataclass
class RateLimitConfig:
    """Rate limiting configuration per connector"""
    requests_per_second: float = 1.0
    burst_size: int = 5
    cooldown_period: float = 60.0  # seconds after rate limit hit


@dataclass
class NodeExecution:
    """Execution record for a DAG node"""
    node_id: str
    status: NodeStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: float = 0.0
    attempt_count: int = 0
    last_error: Optional[str] = None
    result: Any = None
    checkpoint_hash: Optional[str] = None


@dataclass
class DAGCheckpoint:
    """DAG execution checkpoint"""
    checkpoint_id: str
    timestamp: datetime
    config_version: str
    completed_nodes: Dict[str, Any]
    node_executions: Dict[str, NodeExecution]
    global_state: Dict[str, Any]
    deterministic_seed: int


class DeterministicSeedManager:
    """Manages deterministic seeds for reproducible execution"""

    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self.node_seeds = {}
        self.logger = logging.getLogger(__name__)

    def get_node_seed(self, node_id: str, config_hash: str) -> int:
        """Get deterministic seed for node execution"""
        seed_key = f"{node_id}:{config_hash}"

        if seed_key not in self.node_seeds:
            # Generate deterministic seed from node_id and config
            hasher = hashlib.sha256()
            hasher.update(f"{self.base_seed}:{seed_key}".encode())
            self.node_seeds[seed_key] = int(hasher.hexdigest()[:8], 16)

        return self.node_seeds[seed_key]

    def set_seeds(self, node_id: str, config_hash: str):
        """Set all random seeds for deterministic execution"""
        seed = self.get_node_seed(node_id, config_hash)

        random.seed(seed)
        # Set numpy seed if available
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass

        # Set torch seed if available
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        except ImportError:
            pass

        self.logger.debug(f"Set deterministic seed {seed} for node {node_id}")


class RateLimiter:
    """Token bucket rate limiter"""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.tokens = config.burst_size
        self.last_update = time.time()
        self.last_rate_limit = 0
        self.logger = logging.getLogger(__name__)

    async def acquire(self, connector_name: str = "default") -> bool:
        """Acquire permission to make request"""
        now = time.time()

        # Check if we're in cooldown period
        if now - self.last_rate_limit < self.config.cooldown_period:
            self.logger.warning(f"Rate limiter in cooldown for {connector_name}")
            await asyncio.sleep(1.0)
            return False

        # Add tokens based on time elapsed
        time_passed = now - self.last_update
        tokens_to_add = time_passed * self.config.requests_per_second
        self.tokens = min(self.config.burst_size, self.tokens + tokens_to_add)
        self.last_update = now

        # Check if we have tokens
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        else:
            # Rate limit hit
            self.last_rate_limit = now
            wait_time = (1.0 - self.tokens) / self.config.requests_per_second
            self.logger.info(f"Rate limit hit for {connector_name}, waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
            return False


class RetriableNode:
    """DAG node with retry capabilities"""

    def __init__(self, node_id: str, func: Callable, retry_config: RetryConfig = None):
        self.node_id = node_id
        self.func = func
        self.retry_config = retry_config or RetryConfig()
        self.execution = NodeExecution(node_id=node_id, status=NodeStatus.PENDING)
        self.logger = logging.getLogger(__name__)

    async def execute(self, *args, **kwargs) -> Any:
        """Execute node with retry logic"""
        self.execution.status = NodeStatus.RUNNING
        self.execution.start_time = datetime.now(timezone.utc)

        for attempt in range(self.retry_config.max_retries + 1):
            self.execution.attempt_count = attempt + 1

            try:
                if attempt > 0:
                    self.execution.status = NodeStatus.RETRYING
                    delay = self._calculate_delay(attempt)
                    self.logger.info(f"Retrying node {self.node_id}, attempt {attempt + 1}, delay {delay:.1f}s")
                    await asyncio.sleep(delay)

                # Execute the function
                if asyncio.iscoroutinefunction(self.func):
                    result = await self.func(*args, **kwargs)
                else:
                    result = self.func(*args, **kwargs)

                # Success
                self.execution.status = NodeStatus.COMPLETED
                self.execution.result = result
                self.execution.end_time = datetime.now(timezone.utc)
                self.execution.duration = (self.execution.end_time - self.execution.start_time).total_seconds()

                self.logger.info(f"Node {self.node_id} completed successfully on attempt {attempt + 1}")
                return result

            except Exception as e:
                self.execution.last_error = str(e)

                # Check if this exception type should trigger retry
                should_retry = any(isinstance(e, exc_type) for exc_type in self.retry_config.retry_on_exceptions)

                if attempt < self.retry_config.max_retries and should_retry:
                    self.logger.warning(f"Node {self.node_id} failed on attempt {attempt + 1}: {e}")
                    continue
                else:
                    # Final failure
                    self.execution.status = NodeStatus.FAILED
                    self.execution.end_time = datetime.now(timezone.utc)
                    self.execution.duration = (self.execution.end_time - self.execution.start_time).total_seconds()

                    self.logger.error(f"Node {self.node_id} failed permanently after {attempt + 1} attempts: {e}")
                    raise

        # Should never reach here
        raise RuntimeError(f"Unexpected end of retry loop for node {self.node_id}")

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff and jitter"""
        delay = min(
            self.retry_config.base_delay * (self.retry_config.exponential_base ** (attempt - 1)),
            self.retry_config.max_delay
        )

        if self.retry_config.jitter:
            # Add random jitter (±25%)
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0.1, delay)  # Minimum 0.1 second delay


class CheckpointManager:
    """Manages DAG execution checkpoints"""

    def __init__(self, checkpoint_dir: str = "research_cache/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def generate_config_version(self, config: Dict[str, Any]) -> str:
        """Generate version hash for configuration"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def save_checkpoint(self, checkpoint: DAGCheckpoint) -> str:
        """Save checkpoint to disk"""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{checkpoint.checkpoint_id}.pkl"

        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)

            self.logger.info(f"Checkpoint saved: {checkpoint_file}")
            return str(checkpoint_file)

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            raise

    def load_checkpoint(self, checkpoint_id: str) -> Optional[DAGCheckpoint]:
        """Load checkpoint from disk"""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{checkpoint_id}.pkl"

        if not checkpoint_file.exists():
            return None

        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)

            self.logger.info(f"Checkpoint loaded: {checkpoint_file}")
            return checkpoint

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return None

    def find_latest_checkpoint(self, config_version: str) -> Optional[DAGCheckpoint]:
        """Find latest checkpoint for given config version"""
        checkpoints = []

        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.pkl"):
            try:
                with open(checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)

                if checkpoint.config_version == config_version:
                    checkpoints.append(checkpoint)

            except Exception as e:
                self.logger.warning(f"Could not load checkpoint {checkpoint_file}: {e}")
                continue

        if checkpoints:
            # Return most recent
            latest = max(checkpoints, key=lambda c: c.timestamp)
            self.logger.info(f"Found latest checkpoint: {latest.checkpoint_id}")
            return latest

        return None

    def cleanup_old_checkpoints(self, keep_count: int = 5):
        """Clean up old checkpoint files"""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pkl"))

        if len(checkpoint_files) <= keep_count:
            return

        # Sort by modification time
        checkpoint_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        # Remove old files
        for old_file in checkpoint_files[keep_count:]:
            try:
                old_file.unlink()
                self.logger.info(f"Removed old checkpoint: {old_file}")
            except Exception as e:
                self.logger.warning(f"Could not remove checkpoint {old_file}: {e}")


class ReliableDAGOrchestrator:
    """Enhanced DAG orchestrator with reliability features"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Core components
        self.seed_manager = DeterministicSeedManager(
            base_seed=config.get("deterministic_seed", 42)
        )
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.get("checkpoint_dir", "research_cache/checkpoints")
        )

        # Rate limiters per connector
        self.rate_limiters = {}
        self._setup_rate_limiters()

        # DAG state
        self.nodes = {}
        self.dependencies = {}
        self.node_executions = {}
        self.global_state = {}
        self.config_version = self.checkpoint_manager.generate_config_version(config)

        # Execution tracking
        self.execution_id = self._generate_execution_id()
        self.start_time = None
        self.current_checkpoint = None

    def _setup_rate_limiters(self):
        """Setup rate limiters for connectors"""
        rate_limits = self.config.get("rate_limits", {})

        for connector, limit_config in rate_limits.items():
            self.rate_limiters[connector] = RateLimiter(RateLimitConfig(
                requests_per_second=limit_config.get("requests_per_second", 1.0),
                burst_size=limit_config.get("burst_size", 5),
                cooldown_period=limit_config.get("cooldown_period", 60.0)
            ))

        # Default rate limiter
        if "default" not in self.rate_limiters:
            self.rate_limiters["default"] = RateLimiter(RateLimitConfig())

    def _generate_execution_id(self) -> str:
        """Generate unique execution ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = random.randint(1000, 9999)
        return f"dag_exec_{timestamp}_{random_suffix}"

    def add_node(self, node_id: str, func: Callable, dependencies: List[str] = None,
                retry_config: RetryConfig = None):
        """Add node to DAG"""
        dependencies = dependencies or []

        node = RetriableNode(node_id, func, retry_config)
        self.nodes[node_id] = node
        self.dependencies[node_id] = dependencies

        self.logger.debug(f"Added node {node_id} with dependencies: {dependencies}")

    async def execute_dag(self, initial_data: Dict[str, Any] = None,
                         resume_from_checkpoint: bool = True) -> Dict[str, Any]:
        """Execute complete DAG with checkpointing"""
        self.start_time = datetime.now(timezone.utc)
        self.global_state = initial_data or {}

        self.logger.info(f"Starting DAG execution: {self.execution_id}")
        self.logger.info(f"Config version: {self.config_version}")

        # Try to resume from checkpoint
        if resume_from_checkpoint:
            checkpoint = self.checkpoint_manager.find_latest_checkpoint(self.config_version)
            if checkpoint:
                await self._resume_from_checkpoint(checkpoint)

        try:
            # Execute nodes in topological order
            execution_order = self._topological_sort()

            for node_id in execution_order:
                if node_id in self.node_executions and self.node_executions[node_id].status == NodeStatus.COMPLETED:
                    self.logger.info(f"Skipping completed node: {node_id}")
                    continue

                await self._execute_node(node_id)

                # Create checkpoint after each successful node
                await self._create_checkpoint()

            # Final results
            results = {
                "execution_id": self.execution_id,
                "status": "completed",
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now(timezone.utc).isoformat(),
                "node_results": {node_id: execution.result for node_id, execution in self.node_executions.items() if execution.status == NodeStatus.COMPLETED},
                "global_state": self.global_state
            }

            self.logger.info(f"DAG execution completed successfully: {self.execution_id}")
            return results

        except Exception as e:
            self.logger.error(f"DAG execution failed: {e}")

            # Create failure checkpoint
            await self._create_checkpoint()

            raise

    async def _execute_node(self, node_id: str):
        """Execute single node with rate limiting and deterministic seeds"""
        node = self.nodes[node_id]

        self.logger.info(f"Executing node: {node_id}")

        # Set deterministic seeds
        self.seed_manager.set_seeds(node_id, self.config_version)

        # Check rate limiting (if node has connector info)
        connector = self._get_node_connector(node_id)
        if connector and connector in self.rate_limiters:
            rate_limiter = self.rate_limiters[connector]
            while not await rate_limiter.acquire(connector):
                pass  # Wait for rate limit

        # Prepare node inputs from dependencies
        node_inputs = await self._prepare_node_inputs(node_id)

        # Execute node
        try:
            result = await node.execute(**node_inputs)

            # Store execution record
            self.node_executions[node_id] = node.execution

            # Update global state
            self.global_state[f"node_{node_id}_result"] = result

            self.logger.info(f"Node {node_id} completed in {node.execution.duration:.1f}s")

        except Exception as e:
            self.node_executions[node_id] = node.execution
            self.logger.error(f"Node {node_id} failed: {e}")
            raise

    def _get_node_connector(self, node_id: str) -> Optional[str]:
        """Get connector name for node (for rate limiting)"""
        # This would be configured based on node type
        node_config = self.config.get("nodes", {}).get(node_id, {})
        return node_config.get("connector")

    async def _prepare_node_inputs(self, node_id: str) -> Dict[str, Any]:
        """Prepare inputs for node execution"""
        inputs = {"global_state": self.global_state}

        # Add dependency results
        for dep_id in self.dependencies.get(node_id, []):
            if dep_id in self.node_executions:
                execution = self.node_executions[dep_id]
                if execution.status == NodeStatus.COMPLETED:
                    inputs[f"dep_{dep_id}"] = execution.result

        return inputs

    def _topological_sort(self) -> List[str]:
        """Topological sort of DAG nodes"""
        in_degree = {node_id: 0 for node_id in self.nodes}

        # Calculate in-degrees
        for node_id, deps in self.dependencies.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[node_id] += 1

        # Kahn's algorithm
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node_id = queue.pop(0)
            result.append(node_id)

            # Update in-degrees of dependents
            for dependent, deps in self.dependencies.items():
                if node_id in deps:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        if len(result) != len(self.nodes):
            raise ValueError("DAG contains cycles")

        return result

    async def _create_checkpoint(self):
        """Create execution checkpoint"""
        checkpoint = DAGCheckpoint(
            checkpoint_id=f"{self.execution_id}_{len(self.node_executions)}",
            timestamp=datetime.now(timezone.utc),
            config_version=self.config_version,
            completed_nodes={node_id: execution.result for node_id, execution in self.node_executions.items() if execution.status == NodeStatus.COMPLETED},
            node_executions=self.node_executions.copy(),
            global_state=self.global_state.copy(),
            deterministic_seed=self.seed_manager.base_seed
        )

        self.checkpoint_manager.save_checkpoint(checkpoint)
        self.current_checkpoint = checkpoint

    async def _resume_from_checkpoint(self, checkpoint: DAGCheckpoint):
        """Resume execution from checkpoint"""
        self.logger.info(f"Resuming from checkpoint: {checkpoint.checkpoint_id}")

        self.node_executions = checkpoint.node_executions
        self.global_state = checkpoint.global_state
        self.seed_manager.base_seed = checkpoint.deterministic_seed

        # Log resume status
        completed_count = sum(1 for ex in self.node_executions.values() if ex.status == NodeStatus.COMPLETED)
        self.logger.info(f"Resumed with {completed_count} completed nodes")


def create_reliable_dag_orchestrator(config: Dict[str, Any]) -> ReliableDAGOrchestrator:
    """Factory function for reliable DAG orchestrator"""
    return ReliableDAGOrchestrator(config)
