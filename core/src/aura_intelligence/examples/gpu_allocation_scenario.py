"""
Multi-Agent GPU Allocation Scenario

Complete example demonstrating:
- Multiple agents competing for GPU resources
- Full resilience stack (Circuit Breaker, Bulkhead, Retry, Timeout)
- Temporal workflow orchestration
- Consensus for resource allocation
- Kafka event streaming
- OpenTelemetry observability

This serves as both a test and a reference implementation.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
import structlog
from dataclasses import dataclass

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from ..resilience import (
    ResilienceManager,
    ResilienceConfig,
    ResilienceContext,
    ResilienceLevel,
    AdaptiveCircuitBreaker,
    CircuitBreakerConfig,
    DynamicBulkhead,
    BulkheadConfig,
    ResourceRequest,
    ResourceType,
    PriorityLevel,
    ContextAwareRetry,
    RetryConfig,
    RetryStrategy,
    AdaptiveTimeout,
    TimeoutConfig,
    resilience_metrics,
    metrics_collector
)

from ..agents.temporal import TemporalClient, execute_workflow
from ..agents.v2 import ObserverAgent, ExecutorAgent, CoordinatorAgent
from ..consensus import ConsensusManager, ConsensusRequest, DecisionType
from ..events import EventProducer, AgentEvent, EventType

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)


@dataclass
class GPUAllocationRequest:
    """Request for GPU allocation."""
    agent_id: str
    agent_type: str
    gpu_count: int
    duration: timedelta
    priority: PriorityLevel
    workload_type: str  # "training", "inference", "research"
    estimated_cost: float
    metadata: Dict[str, Any]


class GPUResourceManager:
    """
    Manages GPU resources with full resilience stack.
    
    Demonstrates real-world integration of all components.
    """
    
    def __init__(self):
        # Initialize resilience components
        self._init_resilience()
        
        # Initialize AURA components
        self._init_aura_components()
        
        # GPU inventory
        self.total_gpus = 8
        self.allocated_gpus: Dict[str, int] = {}
        
    def _init_resilience(self):
        """Initialize resilience patterns."""
        # Circuit breaker for GPU operations
        self.circuit_breaker = AdaptiveCircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=0.5,
                adaptive_enabled=True,
                use_ml_prediction=True,
                window_size=50
            ),
            name="gpu_allocation"
        )
        
        # Bulkhead for resource isolation
        self.bulkhead = DynamicBulkhead(
            BulkheadConfig(
                min_capacity=4,
                max_capacity=16,
                initial_capacity=8,
                gpu_partitions={
                    "inference": 0.5,    # 4 GPUs
                    "training": 0.375,   # 3 GPUs
                    "emergency": 0.125   # 1 GPU
                },
                cost_aware=True,
                max_cost_per_minute=100.0,
                use_consensus=True
            ),
            name="gpu_bulkhead"
        )
        
        # Retry for transient failures
        self.retry = ContextAwareRetry(
            RetryConfig(
                max_attempts=3,
                strategy=RetryStrategy.EXPONENTIAL,
                budget_enabled=True,
                hedged_requests=True,
                hedge_delay=0.5
            )
        )
        
        # Timeout for operations
        self.timeout = AdaptiveTimeout(
            TimeoutConfig(
                default_timeout_ms=5000,
                strategy=TimeoutConfig.TimeoutStrategy.ADAPTIVE,
                deadline_propagation=True
            )
        )
        
        # Resilience manager
        self.resilience_manager = ResilienceManager(
            ResilienceConfig(
                enable_circuit_breaker=True,
                enable_bulkhead=True,
                enable_retry=True,
                enable_timeout=True,
                enable_model_fallback=True
            )
        )
        
    def _init_aura_components(self):
        """Initialize AURA components."""
        # Temporal client
        self.temporal_client = TemporalClient(
            temporal_host="localhost:7233",
            namespace="aura-gpu"
        )
        
        # Consensus manager
        self.consensus_manager = ConsensusManager(
            node_id="gpu-manager",
            peers=["gpu-node-1", "gpu-node-2", "gpu-node-3"],
            kafka_servers="localhost:9092"
        )
        
        # Event producer
        self.event_producer = EventProducer(
            bootstrap_servers="localhost:9092",
            client_id="gpu-manager"
        )
    
    async def start(self):
        """Start all components."""
        # Start resilience components
        await self.bulkhead.start()
        
        # Start AURA components
        await self.temporal_client.connect()
        await self.consensus_manager.start()
        await self.event_producer.start()
        
        # Start metrics collection
        metrics_collector.register_component("gpu_circuit_breaker", self.circuit_breaker)
        metrics_collector.register_component("gpu_bulkhead", self.bulkhead)
        await metrics_collector.start()
        
        logger.info("GPU Resource Manager started")
    
    async def stop(self):
        """Stop all components."""
        await self.bulkhead.stop()
        await self.consensus_manager.stop()
        await self.event_producer.stop()
        await metrics_collector.stop()
    
    async def allocate_gpu(self, request: GPUAllocationRequest) -> Dict[str, Any]:
        """
        Allocate GPU with full resilience stack.
        
        Flow:
        1. Create resilience context
        2. Check bulkhead capacity
        3. Get consensus for allocation
        4. Execute allocation with circuit breaker
        5. Publish events
        6. Start Temporal workflow for management
        """
        with tracer.start_as_current_span("gpu.allocate") as span:
            span.set_attributes({
                "agent_id": request.agent_id,
                "gpu_count": request.gpu_count,
                "priority": request.priority.name,
                "workload_type": request.workload_type
            })
            
            try:
                # Create resilience context
                context = ResilienceContext(
                    operation_name="gpu_allocation",
                    criticality=self._get_criticality(request.priority),
                    timeout=timedelta(seconds=30),
                    metadata={
                        "agent_id": request.agent_id,
                        "cost_estimate": request.estimated_cost
                    }
                )
                
                # Execute with full resilience
                result = await self.resilience_manager.execute_with_resilience(
                    self._allocate_with_consensus,
                    context,
                    request
                )
                
                span.set_status(Status(StatusCode.OK))
                return result
                
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                
                # Publish failure event
                await self._publish_allocation_event(
                    request,
                    success=False,
                    error=str(e)
                )
                
                raise
    
    async def _allocate_with_consensus(self, request: GPUAllocationRequest) -> Dict[str, Any]:
        """Allocate GPU with consensus and bulkhead protection."""
        # Create bulkhead request
        bulkhead_request = ResourceRequest(
            id=f"gpu-{request.agent_id}-{datetime.now(timezone.utc).timestamp()}",
            operation_name=f"gpu_allocation_{request.workload_type}",
            priority=request.priority,
            resources={
                ResourceType.GPU: float(request.gpu_count),
                ResourceType.MEMORY: 16.0 * request.gpu_count  # 16GB per GPU
            },
            estimated_duration=request.duration,
            cost_estimate=request.estimated_cost,
            metadata=request.metadata
        )
        
        # Execute with bulkhead
        return await self.bulkhead.execute(
            self._consensus_allocation,
            bulkhead_request,
            request
        )
    
    async def _consensus_allocation(self, request: GPUAllocationRequest) -> Dict[str, Any]:
        """Get consensus for GPU allocation."""
        # Only use consensus for critical allocations
        if request.priority in [PriorityLevel.CRITICAL, PriorityLevel.HIGH]:
            consensus_request = ConsensusRequest(
                request_id=f"gpu-consensus-{request.agent_id}",
                decision_type=DecisionType.RESOURCE_ALLOCATION,
                proposal={
                    "agent_id": request.agent_id,
                    "gpu_count": request.gpu_count,
                    "duration_minutes": request.duration.total_seconds() / 60,
                    "cost": request.estimated_cost
                },
                timeout=timedelta(seconds=5),
                priority=request.priority.value
            )
            
            # Get consensus with circuit breaker
            consensus_result = await self.circuit_breaker.execute(
                self.consensus_manager.propose,
                consensus_request
            )
            
            if consensus_result.status.value != "accepted":
                raise Exception(f"Consensus rejected: {consensus_result.reason}")
        
        # Execute actual allocation
        return await self._execute_allocation(request)
    
    async def _execute_allocation(self, request: GPUAllocationRequest) -> Dict[str, Any]:
        """Execute the actual GPU allocation."""
        # Check availability
        available = self.total_gpus - sum(self.allocated_gpus.values())
        
        if available < request.gpu_count:
            raise Exception(f"Insufficient GPUs: {available} < {request.gpu_count}")
        
        # Allocate
        self.allocated_gpus[request.agent_id] = request.gpu_count
        
        # Start Temporal workflow for lifecycle management
        workflow_id = f"gpu-lifecycle-{request.agent_id}-{datetime.now(timezone.utc).timestamp()}"
        
        workflow_handle = await self.temporal_client.start_workflow(
            "GPULifecycleWorkflow",
            {
                "agent_id": request.agent_id,
                "gpu_count": request.gpu_count,
                "duration": request.duration.total_seconds(),
                "workload_type": request.workload_type
            },
            id=workflow_id,
            task_queue="gpu-management"
        )
        
        # Publish success event
        await self._publish_allocation_event(request, success=True)
        
        return {
            "success": True,
            "agent_id": request.agent_id,
            "gpu_count": request.gpu_count,
            "workflow_id": workflow_id,
            "allocated_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": (datetime.now(timezone.utc) + request.duration).isoformat()
        }
    
    async def _publish_allocation_event(
        self,
        request: GPUAllocationRequest,
        success: bool,
        error: Optional[str] = None
    ):
        """Publish GPU allocation event."""
        event = AgentEvent(
            agent_id=request.agent_id,
            event_type=EventType.RESOURCE_ALLOCATED if success else EventType.RESOURCE_FAILED,
            data={
                "resource_type": "gpu",
                "count": request.gpu_count,
                "workload_type": request.workload_type,
                "success": success,
                "error": error
            },
            severity="info" if success else "error"
        )
        
        await self.event_producer.send_event(
            event.to_kafka_record()
        )
    
    def _get_criticality(self, priority: PriorityLevel) -> ResilienceLevel:
        """Map priority to resilience criticality."""
        mapping = {
            PriorityLevel.CRITICAL: ResilienceLevel.CRITICAL,
            PriorityLevel.HIGH: ResilienceLevel.CRITICAL,
            PriorityLevel.NORMAL: ResilienceLevel.STANDARD,
            PriorityLevel.LOW: ResilienceLevel.BEST_EFFORT,
            PriorityLevel.SCAVENGER: ResilienceLevel.BEST_EFFORT
        }
        return mapping.get(priority, ResilienceLevel.STANDARD)
    
    async def get_allocation_status(self) -> Dict[str, Any]:
        """Get current allocation status."""
        return {
            "total_gpus": self.total_gpus,
            "allocated_gpus": sum(self.allocated_gpus.values()),
            "available_gpus": self.total_gpus - sum(self.allocated_gpus.values()),
            "allocations": self.allocated_gpus.copy(),
            "bulkhead_metrics": self.bulkhead.get_metrics(),
            "circuit_breaker_metrics": self.circuit_breaker.get_metrics(),
            "resilience_score": resilience_metrics.calculate_resilience_score()
        }


async def simulate_multi_agent_scenario():
    """
    Simulate multiple agents competing for GPU resources.
    
    This demonstrates:
    - Priority-based allocation
    - Resource contention handling
    - Failure scenarios
    - Recovery patterns
    """
    # Initialize GPU manager
    gpu_manager = GPUResourceManager()
    await gpu_manager.start()
    
    # Define agent requests
    agent_requests = [
        # Critical research agent - should succeed
        GPUAllocationRequest(
            agent_id="research_agent_1",
            agent_type="research",
            gpu_count=2,
            duration=timedelta(hours=4),
            priority=PriorityLevel.CRITICAL,
            workload_type="training",
            estimated_cost=50.0,
            metadata={"model": "llama-70b", "experiment": "exp-123"}
        ),
        
        # High priority inference agents
        GPUAllocationRequest(
            agent_id="inference_agent_1",
            agent_type="inference",
            gpu_count=1,
            duration=timedelta(hours=1),
            priority=PriorityLevel.HIGH,
            workload_type="inference",
            estimated_cost=10.0,
            metadata={"endpoint": "api/v1/chat"}
        ),
        
        GPUAllocationRequest(
            agent_id="inference_agent_2",
            agent_type="inference",
            gpu_count=1,
            duration=timedelta(hours=1),
            priority=PriorityLevel.HIGH,
            workload_type="inference",
            estimated_cost=10.0,
            metadata={"endpoint": "api/v1/embeddings"}
        ),
        
        # Normal priority training
        GPUAllocationRequest(
            agent_id="training_agent_1",
            agent_type="training",
            gpu_count=3,
            duration=timedelta(hours=8),
            priority=PriorityLevel.NORMAL,
            workload_type="training",
            estimated_cost=80.0,
            metadata={"dataset": "custom-dataset-v2"}
        ),
        
        # Low priority batch job
        GPUAllocationRequest(
            agent_id="batch_agent_1",
            agent_type="batch",
            gpu_count=2,
            duration=timedelta(hours=2),
            priority=PriorityLevel.LOW,
            workload_type="inference",
            estimated_cost=20.0,
            metadata={"job_type": "batch_embeddings"}
        ),
        
        # Scavenger agent - should use leftover resources
        GPUAllocationRequest(
            agent_id="scavenger_agent_1",
            agent_type="scavenger",
            gpu_count=1,
            duration=timedelta(minutes=30),
            priority=PriorityLevel.SCAVENGER,
            workload_type="inference",
            estimated_cost=2.0,
            metadata={"opportunistic": True}
        )
    ]
    
    # Submit all requests concurrently
    logger.info("Starting multi-agent GPU allocation scenario")
    
    tasks = []
    for request in agent_requests:
        task = asyncio.create_task(
            allocate_with_logging(gpu_manager, request)
        )
        tasks.append((request.agent_id, task))
        
        # Small delay to simulate realistic submission
        await asyncio.sleep(0.1)
    
    # Wait for all allocations
    results = []
    for agent_id, task in tasks:
        try:
            result = await task
            results.append((agent_id, "success", result))
            logger.info(f"Allocation succeeded for {agent_id}", result=result)
        except Exception as e:
            results.append((agent_id, "failed", str(e)))
            logger.error(f"Allocation failed for {agent_id}", error=str(e))
    
    # Print summary
    print("\n" + "="*80)
    print("MULTI-AGENT GPU ALLOCATION RESULTS")
    print("="*80)
    
    success_count = sum(1 for _, status, _ in results if status == "success")
    print(f"\nTotal Requests: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(results) - success_count}")
    
    print("\nDetailed Results:")
    for agent_id, status, result in results:
        print(f"\n{agent_id}:")
        print(f"  Status: {status}")
        if status == "success":
            print(f"  GPUs: {result['gpu_count']}")
            print(f"  Workflow: {result['workflow_id']}")
        else:
            print(f"  Error: {result}")
    
    # Get final status
    status = await gpu_manager.get_allocation_status()
    print(f"\nFinal GPU Status:")
    print(f"  Total: {status['total_gpus']}")
    print(f"  Allocated: {status['allocated_gpus']}")
    print(f"  Available: {status['available_gpus']}")
    print(f"  Resilience Score: {status['resilience_score']:.1f}/100")
    
    # Simulate some failures to test resilience
    print("\n" + "="*80)
    print("SIMULATING FAILURE SCENARIOS")
    print("="*80)
    
    # Simulate network partition
    print("\n1. Simulating network partition...")
    gpu_manager.circuit_breaker.state = gpu_manager.circuit_breaker.state.__class__.OPEN
    
    try:
        await gpu_manager.allocate_gpu(
            GPUAllocationRequest(
                agent_id="test_agent_network",
                agent_type="test",
                gpu_count=1,
                duration=timedelta(minutes=10),
                priority=PriorityLevel.NORMAL,
                workload_type="inference",
                estimated_cost=5.0,
                metadata={}
            )
        )
    except Exception as e:
        print(f"   Expected failure: {e}")
    
    # Wait for circuit breaker recovery
    await asyncio.sleep(2)
    gpu_manager.circuit_breaker.state = gpu_manager.circuit_breaker.state.__class__.CLOSED
    
    # Simulate resource exhaustion
    print("\n2. Simulating resource exhaustion...")
    try:
        await gpu_manager.allocate_gpu(
            GPUAllocationRequest(
                agent_id="test_agent_exhaust",
                agent_type="test",
                gpu_count=10,  # More than available
                duration=timedelta(minutes=10),
                priority=PriorityLevel.HIGH,
                workload_type="training",
                estimated_cost=100.0,
                metadata={}
            )
        )
    except Exception as e:
        print(f"   Expected failure: {e}")
    
    # Cleanup
    await gpu_manager.stop()
    
    print("\n" + "="*80)
    print("SCENARIO COMPLETED")
    print("="*80)


async def allocate_with_logging(
    gpu_manager: GPUResourceManager,
    request: GPUAllocationRequest
) -> Dict[str, Any]:
    """Helper to allocate with detailed logging."""
    logger.info(
        f"Requesting GPU allocation",
        agent_id=request.agent_id,
        gpu_count=request.gpu_count,
        priority=request.priority.name
    )
    
    start_time = datetime.now(timezone.utc)
    result = await gpu_manager.allocate_gpu(request)
    duration = (datetime.now(timezone.utc) - start_time).total_seconds()
    
    logger.info(
        f"GPU allocation completed",
        agent_id=request.agent_id,
        duration_seconds=duration,
        success=True
    )
    
    return result


if __name__ == "__main__":
    # Run the scenario
    asyncio.run(simulate_multi_agent_scenario())