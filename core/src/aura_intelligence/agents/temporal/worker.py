"""
Temporal Worker Implementation for AURA Intelligence

Workers execute workflows and activities with proper lifecycle management,
observability, and error handling.
"""

from typing import List, Optional, Type, Any, Dict
from dataclasses import dataclass, field
import asyncio
import signal
import sys

from temporalio.client import Client
from temporalio.worker import Worker
from temporalio import workflow, activity
import structlog
from opentelemetry import trace, metrics

from .workflows import (
    AgentWorkflow,
    MultiAgentOrchestrationWorkflow,
    ResearchAnalysisPipeline,
    ConsensusWorkflow
)
from .activities import (
    AgentActivity,
    StateManagementActivity,
    KafkaProducerActivity,
    ObservabilityActivity
)

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

# Metrics
worker_status = meter.create_gauge(
    name="temporal.worker.status",
    description="Worker status (1=running, 0=stopped)"
)

worker_tasks_completed = meter.create_counter(
    name="temporal.worker.tasks.completed",
    description="Number of tasks completed by worker"
)


@dataclass
class WorkerConfig:
    """Configuration for Temporal worker."""
    task_queue: str
    namespace: str = "default"
    temporal_host: str = "localhost:7233"
    max_concurrent_workflow_tasks: int = 100
    max_concurrent_activity_tasks: int = 100
    max_cached_workflows: int = 100
    graceful_shutdown_timeout: int = 30
    identity: Optional[str] = None
    build_id: Optional[str] = None
    use_worker_versioning: bool = True
    log_in_replay: bool = False
    
    # Feature flags
    enable_session_worker: bool = False
    sticky_queue_schedule_to_start_timeout: int = 10
    
    # Workflows to register
    workflows: List[Type] = field(default_factory=lambda: [
        AgentWorkflow,
        MultiAgentOrchestrationWorkflow,
        ResearchAnalysisPipeline,
        ConsensusWorkflow
    ])
    
    # Activities to register
    activities: List[Any] = field(default_factory=lambda: [
        AgentActivity.process,
        AgentActivity.compute_consensus,
        StateManagementActivity.create_initial_state,
        StateManagementActivity.persist_state,
        StateManagementActivity.load_state,
        KafkaProducerActivity.publish_event,
        ObservabilityActivity.start_workflow_span,
        ObservabilityActivity.record_workflow_metrics
    ])


class TemporalWorker:
    """
    Temporal worker with lifecycle management.
    
    Features:
    - Graceful shutdown
    - Health checks
    - Metrics collection
    - Error recovery
    """
    
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.client: Optional[Client] = None
        self.worker: Optional[Worker] = None
        self._running = False
        self._shutdown_event = asyncio.Event()
        
    async def start(self) -> None:
        """Start the worker."""
        logger.info(
            "Starting Temporal worker",
            task_queue=self.config.task_queue,
            namespace=self.config.namespace
        )
        
        try:
            # Create client
            self.client = await Client.connect(
                self.config.temporal_host,
                namespace=self.config.namespace
            )
            
            # Create worker
            self.worker = Worker(
                self.client,
                task_queue=self.config.task_queue,
                workflows=self.config.workflows,
                activities=self.config.activities,
                max_concurrent_workflow_tasks=self.config.max_concurrent_workflow_tasks,
                max_concurrent_activity_tasks=self.config.max_concurrent_activity_tasks,
                max_cached_workflows=self.config.max_cached_workflows,
                identity=self.config.identity,
                build_id=self.config.build_id,
                use_worker_versioning=self.config.use_worker_versioning,
                graceful_shutdown_timeout=self.config.graceful_shutdown_timeout,
                log_in_replay=self.config.log_in_replay
            )
            
            # Set up signal handlers
            self._setup_signal_handlers()
            
            # Update metrics
            worker_status.set(1, {"task_queue": self.config.task_queue})
            
            self._running = True
            logger.info("Temporal worker started successfully")
            
            # Run worker
            await self.worker.run()
            
        except Exception as e:
            logger.error(f"Failed to start worker: {e}")
            raise
        finally:
            await self.shutdown()
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the worker."""
        if not self._running:
            return
            
        logger.info("Shutting down Temporal worker")
        self._running = False
        
        # Update metrics
        worker_status.set(0, {"task_queue": self.config.task_queue})
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Close client
        if self.client:
            await self.client.close()
            
        logger.info("Temporal worker shutdown complete")
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, initiating shutdown")
            asyncio.create_task(self.shutdown())
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check worker health."""
        health = {
            "status": "healthy" if self._running else "stopped",
            "task_queue": self.config.task_queue,
            "namespace": self.config.namespace
        }
        
        if self.worker and self._running:
            # Add worker metrics
            health.update({
                "workflows_registered": len(self.config.workflows),
                "activities_registered": len(self.config.activities),
                "max_concurrent_workflows": self.config.max_concurrent_workflow_tasks,
                "max_concurrent_activities": self.config.max_concurrent_activity_tasks
            })
        
        return health


async def create_worker(
    task_queue: str,
    namespace: str = "default",
    temporal_host: str = "localhost:7233",
    **kwargs
) -> TemporalWorker:
    """
    Create and configure a Temporal worker.
    
    Args:
        task_queue: The task queue name
        namespace: Temporal namespace
        temporal_host: Temporal server address
        **kwargs: Additional configuration options
        
    Returns:
        Configured TemporalWorker instance
    """
    config = WorkerConfig(
        task_queue=task_queue,
        namespace=namespace,
        temporal_host=temporal_host,
        **kwargs
    )
    
    return TemporalWorker(config)


class WorkerPool:
    """
    Manages multiple Temporal workers for different task queues.
    
    Useful for:
    - Separating workflow types
    - Resource isolation
    - Scaling different workloads independently
    """
    
    def __init__(self):
        self.workers: Dict[str, TemporalWorker] = {}
        self._running = False
        
    async def add_worker(
        self,
        name: str,
        task_queue: str,
        **config_kwargs
    ) -> None:
        """Add a worker to the pool."""
        if name in self.workers:
            raise ValueError(f"Worker {name} already exists")
            
        worker = await create_worker(task_queue, **config_kwargs)
        self.workers[name] = worker
        
        if self._running:
            # Start the worker if pool is already running
            asyncio.create_task(worker.start())
    
    async def start_all(self) -> None:
        """Start all workers in the pool."""
        self._running = True
        
        tasks = []
        for name, worker in self.workers.items():
            logger.info(f"Starting worker: {name}")
            task = asyncio.create_task(worker.start())
            tasks.append(task)
        
        # Wait for all workers to start
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def shutdown_all(self) -> None:
        """Shutdown all workers in the pool."""
        self._running = False
        
        tasks = []
        for name, worker in self.workers.items():
            logger.info(f"Shutting down worker: {name}")
            task = asyncio.create_task(worker.shutdown())
            tasks.append(task)
        
        # Wait for all workers to shutdown
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def health_check_all(self) -> Dict[str, Any]:
        """Get health status of all workers."""
        health = {}
        
        for name, worker in self.workers.items():
            health[name] = await worker.health_check()
        
        return {
            "pool_status": "running" if self._running else "stopped",
            "worker_count": len(self.workers),
            "workers": health
        }


# Example worker configurations for different workloads
def create_default_worker_pool() -> WorkerPool:
    """Create a default worker pool with standard configuration."""
    pool = WorkerPool()
    
    # Main agent workflow worker
    asyncio.create_task(pool.add_worker(
        "main",
        "agent-workflows",
        max_concurrent_workflow_tasks=50,
        max_concurrent_activity_tasks=100
    ))
    
    # High-priority consensus worker
    asyncio.create_task(pool.add_worker(
        "consensus",
        "consensus-workflows",
        max_concurrent_workflow_tasks=20,
        max_concurrent_activity_tasks=50,
        workflows=[ConsensusWorkflow]  # Only consensus workflows
    ))
    
    # Research pipeline worker (resource intensive)
    asyncio.create_task(pool.add_worker(
        "research",
        "research-pipelines",
        max_concurrent_workflow_tasks=10,
        max_concurrent_activity_tasks=30,
        workflows=[ResearchAnalysisPipeline]
    ))
    
    return pool


if __name__ == "__main__":
    # Example: Run a single worker
    async def main():
        worker = await create_worker(
            task_queue="agent-workflows",
            identity="worker-1"
        )
        
        try:
            await worker.start()
        except KeyboardInterrupt:
            logger.info("Worker interrupted")
        finally:
            await worker.shutdown()
    
    asyncio.run(main())