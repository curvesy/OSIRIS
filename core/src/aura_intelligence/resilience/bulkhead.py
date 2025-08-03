"""
Dynamic Bulkhead implementation for AURA Intelligence.

Features:
- Auto-scaling resource pools based on load
- Priority-based queuing and scheduling
- GPU resource management
- Agent isolation
- Integration with consensus for resource allocation
- Cost-aware admission control
"""

from typing import Dict, Any, Optional, List, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, IntEnum
import asyncio
from asyncio import Queue, Semaphore
import numpy as np
from collections import defaultdict, deque
import structlog

from opentelemetry import trace, metrics

from ..consensus import SimpleConsensus, Decision
from ..events import EventProducer

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

T = TypeVar('T')

# Metrics
bulkhead_active = meter.create_gauge(
    name="aura.resilience.bulkhead.active",
    description="Number of active executions"
)

bulkhead_queued = meter.create_gauge(
    name="aura.resilience.bulkhead.queued", 
    description="Number of queued requests"
)

bulkhead_rejected = meter.create_counter(
    name="aura.resilience.bulkhead.rejected",
    description="Number of rejected requests"
)

bulkhead_capacity = meter.create_gauge(
    name="aura.resilience.bulkhead.capacity",
    description="Current bulkhead capacity"
)

queue_time = meter.create_histogram(
    name="aura.resilience.bulkhead.queue_time",
    description="Time spent in queue",
    unit="ms"
)


class PriorityLevel(IntEnum):
    """Priority levels for bulkhead queuing."""
    CRITICAL = 0  # Highest priority
    HIGH = 1
    NORMAL = 2
    LOW = 3
    SCAVENGER = 4  # Lowest priority


class ResourceType(Enum):
    """Types of resources managed by bulkhead."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    API_QUOTA = "api_quota"
    AGENT_SLOT = "agent_slot"


@dataclass
class BulkheadConfig:
    """Configuration for dynamic bulkhead."""
    # Capacity settings
    min_capacity: int = 10
    max_capacity: int = 100
    initial_capacity: int = 20
    
    # Scaling settings
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    scale_factor: float = 1.5
    scale_cooldown: timedelta = timedelta(seconds=30)
    
    # Queue settings
    queue_size: int = 1000
    priority_enabled: bool = True
    fair_queuing: bool = True
    
    # GPU settings
    gpu_partitions: Dict[str, float] = field(default_factory=lambda: {
        "inference": 0.6,
        "training": 0.3,
        "emergency": 0.1
    })
    
    # Cost settings
    cost_aware: bool = True
    max_cost_per_minute: float = 100.0
    
    # Integration
    use_consensus: bool = True
    kafka_servers: str = "localhost:9092"


@dataclass
class ResourceRequest:
    """Request for bulkhead resources."""
    id: str
    operation_name: str
    priority: PriorityLevel = PriorityLevel.NORMAL
    resources: Dict[ResourceType, float] = field(default_factory=dict)
    estimated_duration: Optional[timedelta] = None
    cost_estimate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_gpu_request(self) -> bool:
        return ResourceType.GPU in self.resources


class ResourcePool:
    """Manages a pool of resources with dynamic scaling."""
    
    def __init__(self, config: BulkheadConfig, resource_type: ResourceType):
        self.config = config
        self.resource_type = resource_type
        self.capacity = config.initial_capacity
        self.available = config.initial_capacity
        self.lock = asyncio.Lock()
        
        # Scaling state
        self.last_scale_time = datetime.utcnow()
        self.scaling_history = deque(maxlen=100)
        
        # Utilization tracking
        self.utilization_history = deque(maxlen=60)  # 1 minute window
        
    async def acquire(self, amount: float = 1.0) -> bool:
        """Try to acquire resources."""
        async with self.lock:
            if self.available >= amount:
                self.available -= amount
                self._update_metrics()
                return True
            return False
    
    async def release(self, amount: float = 1.0):
        """Release resources back to pool."""
        async with self.lock:
            self.available = min(self.available + amount, self.capacity)
            self._update_metrics()
    
    async def scale(self, factor: float):
        """Scale the pool capacity."""
        async with self.lock:
            old_capacity = self.capacity
            new_capacity = int(self.capacity * factor)
            
            # Clamp to bounds
            new_capacity = max(self.config.min_capacity, 
                             min(self.config.max_capacity, new_capacity))
            
            # Update capacity
            capacity_delta = new_capacity - old_capacity
            self.capacity = new_capacity
            self.available += capacity_delta
            
            # Record scaling event
            self.scaling_history.append({
                "timestamp": datetime.utcnow(),
                "old_capacity": old_capacity,
                "new_capacity": new_capacity,
                "factor": factor
            })
            
            self.last_scale_time = datetime.utcnow()
            self._update_metrics()
            
            logger.info(
                f"Scaled {self.resource_type.value} pool",
                old_capacity=old_capacity,
                new_capacity=new_capacity
            )
    
    def get_utilization(self) -> float:
        """Get current utilization percentage."""
        if self.capacity == 0:
            return 0.0
        return (self.capacity - self.available) / self.capacity
    
    def should_scale_up(self) -> bool:
        """Check if should scale up."""
        if not self._can_scale():
            return False
        
        # Check recent utilization
        recent_utils = list(self.utilization_history)[-10:]
        if not recent_utils:
            return False
        
        avg_utilization = np.mean(recent_utils)
        return avg_utilization > self.config.scale_up_threshold
    
    def should_scale_down(self) -> bool:
        """Check if should scale down."""
        if not self._can_scale():
            return False
        
        recent_utils = list(self.utilization_history)[-10:]
        if not recent_utils:
            return False
        
        avg_utilization = np.mean(recent_utils)
        return avg_utilization < self.config.scale_down_threshold
    
    def _can_scale(self) -> bool:
        """Check if scaling cooldown has passed."""
        time_since_scale = datetime.utcnow() - self.last_scale_time
        return time_since_scale > self.config.scale_cooldown
    
    def _update_metrics(self):
        """Update resource pool metrics."""
        utilization = self.get_utilization()
        self.utilization_history.append(utilization)
        
        bulkhead_capacity.set(
            self.capacity,
            {"resource": self.resource_type.value}
        )


class PriorityQueue(Generic[T]):
    """Priority queue with fair queuing support."""
    
    def __init__(self, maxsize: int, fair_queuing: bool = True):
        self.maxsize = maxsize
        self.fair_queuing = fair_queuing
        self.queues: Dict[PriorityLevel, Queue] = {
            level: Queue(maxsize=maxsize // len(PriorityLevel))
            for level in PriorityLevel
        }
        self.total_size = 0
        self.last_served: Dict[PriorityLevel, datetime] = {}
        
    async def put(self, item: T, priority: PriorityLevel):
        """Add item to queue with priority."""
        if self.total_size >= self.maxsize:
            raise asyncio.QueueFull("Priority queue is full")
        
        await self.queues[priority].put(item)
        self.total_size += 1
        
    async def get(self) -> T:
        """Get next item based on priority and fairness."""
        # Try priorities in order, with fairness
        for priority in PriorityLevel:
            queue = self.queues[priority]
            
            if not queue.empty():
                # Check fairness
                if self.fair_queuing and self._should_skip_for_fairness(priority):
                    continue
                
                item = await queue.get()
                self.total_size -= 1
                self.last_served[priority] = datetime.utcnow()
                return item
        
        # If all queues empty, wait on highest priority
        item = await self.queues[PriorityLevel.CRITICAL].get()
        self.total_size -= 1
        return item
    
    def _should_skip_for_fairness(self, priority: PriorityLevel) -> bool:
        """Check if should skip priority level for fairness."""
        if priority == PriorityLevel.CRITICAL:
            return False  # Never skip critical
        
        # Simple fairness: ensure each level gets served at least once per second
        last_time = self.last_served.get(priority)
        if not last_time:
            return False
        
        time_since = datetime.utcnow() - last_time
        return time_since < timedelta(seconds=1)
    
    def qsize(self) -> int:
        """Get total queue size."""
        return self.total_size


class GPUPartition:
    """Manages GPU partitions for different workloads."""
    
    def __init__(self, total_gpus: int, partitions: Dict[str, float]):
        self.total_gpus = total_gpus
        self.partitions = partitions
        self.allocated: Dict[str, float] = defaultdict(float)
        self.lock = asyncio.Lock()
        
    async def allocate(self, partition: str, amount: float) -> bool:
        """Allocate GPU from partition."""
        async with self.lock:
            max_allowed = self.total_gpus * self.partitions.get(partition, 0)
            current = self.allocated[partition]
            
            if current + amount <= max_allowed:
                self.allocated[partition] += amount
                return True
            
            # Try borrowing from emergency partition
            if partition != "emergency":
                emergency_available = (
                    self.total_gpus * self.partitions["emergency"] - 
                    self.allocated["emergency"]
                )
                if emergency_available >= amount:
                    self.allocated["emergency"] += amount
                    logger.info(f"Borrowed {amount} GPUs from emergency for {partition}")
                    return True
            
            return False
    
    async def release(self, partition: str, amount: float):
        """Release GPU back to partition."""
        async with self.lock:
            self.allocated[partition] = max(0, self.allocated[partition] - amount)


class DynamicBulkhead:
    """
    Dynamic bulkhead with auto-scaling and priority queuing.
    
    Features:
    - Automatic capacity adjustment
    - Priority-based scheduling
    - GPU resource management
    - Cost-aware admission
    - Consensus integration
    """
    
    def __init__(self, config: BulkheadConfig, name: str = "default"):
        self.config = config
        self.name = name
        
        # Resource pools
        self.pools: Dict[ResourceType, ResourcePool] = {
            ResourceType.AGENT_SLOT: ResourcePool(config, ResourceType.AGENT_SLOT),
            ResourceType.CPU: ResourcePool(config, ResourceType.CPU),
            ResourceType.MEMORY: ResourcePool(config, ResourceType.MEMORY),
        }
        
        # GPU management
        self.gpu_partition = GPUPartition(
            total_gpus=config.max_capacity,  # Simplified
            partitions=config.gpu_partitions
        )
        
        # Queue
        self.queue = PriorityQueue(
            maxsize=config.queue_size,
            fair_queuing=config.fair_queuing
        )
        
        # Cost tracking
        self.cost_tracker = CostTracker(config.max_cost_per_minute)
        
        # Integration
        if config.use_consensus:
            self.consensus = SimpleConsensus(
                node_id=f"bulkhead-{name}",
                peers=[],  # Would be configured
                kafka_servers=config.kafka_servers
            )
        
        # Create proper ProducerConfig for EventProducer
        from ..events.producers import ProducerConfig
        producer_config = ProducerConfig(
            bootstrap_servers=config.kafka_servers,
            client_id=f"bulkhead-{name}"
        )
        self.event_producer = EventProducer(producer_config)
        
        # Background tasks
        self._scaling_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start bulkhead background tasks."""
        await self.event_producer.start()
        
        self._scaling_task = asyncio.create_task(self._auto_scaling_loop())
        self._metrics_task = asyncio.create_task(self._metrics_loop())
        
        logger.info(f"Started dynamic bulkhead {self.name}")
    
    async def stop(self):
        """Stop bulkhead."""
        if self._scaling_task:
            self._scaling_task.cancel()
        if self._metrics_task:
            self._metrics_task.cancel()
        
        await self.event_producer.stop()
    
    async def execute(
        self,
        operation: Callable[..., T],
        request: Optional[ResourceRequest] = None,
        *args,
        **kwargs
    ) -> T:
        """Execute operation with bulkhead protection."""
        if not request:
            # Create default request
            request = ResourceRequest(
                id=f"req-{datetime.utcnow().timestamp()}",
                operation_name=operation.__name__,
                resources={ResourceType.AGENT_SLOT: 1.0}
            )
        
        # Admission control
        if not await self._admit_request(request):
            bulkhead_rejected.add(1, {"bulkhead": self.name, "reason": "admission"})
            raise BulkheadRejectedException(f"Request {request.id} rejected by admission control")
        
        # Queue if needed
        acquired = await self._acquire_resources(request)
        if not acquired:
            await self._queue_request(request)
            acquired = await self._wait_for_resources(request)
        
        if not acquired:
            bulkhead_rejected.add(1, {"bulkhead": self.name, "reason": "timeout"})
            raise BulkheadRejectedException(f"Request {request.id} timed out waiting for resources")
        
        # Execute with resources
        start_time = datetime.utcnow()
        try:
            # Update metrics
            bulkhead_active.set(1, {"bulkhead": self.name})
            
            # Execute operation
            result = await operation(*args, **kwargs)
            
            # Track cost
            if self.config.cost_aware:
                duration = (datetime.utcnow() - start_time).total_seconds()
                await self.cost_tracker.record_usage(request, duration)
            
            return result
            
        finally:
            # Release resources
            await self._release_resources(request)
            bulkhead_active.set(0, {"bulkhead": self.name})
            
            # Publish event
            await self._publish_execution_event(request, start_time)
    
    async def _admit_request(self, request: ResourceRequest) -> bool:
        """Check if request should be admitted."""
        # Cost-based admission
        if self.config.cost_aware:
            if not await self.cost_tracker.can_admit(request):
                logger.warning(f"Request {request.id} rejected due to cost limit")
                return False
        
        # Consensus-based admission for critical resources
        if self.config.use_consensus and request.is_gpu_request:
            decision = Decision(
                id=request.id,
                type="gpu_allocation",
                data={
                    "request_id": request.id,
                    "gpu_count": request.resources.get(ResourceType.GPU, 0),
                    "priority": request.priority.value
                },
                requester=request.operation_name
            )
            
            result = await self.consensus.decide(decision)
            if result.get("status") != "accepted":
                logger.warning(f"Request {request.id} rejected by consensus")
                return False
        
        return True
    
    async def _acquire_resources(self, request: ResourceRequest) -> bool:
        """Try to acquire all requested resources."""
        acquired = []
        
        try:
            # Try to acquire each resource type
            for resource_type, amount in request.resources.items():
                if resource_type == ResourceType.GPU:
                    # Special handling for GPU
                    partition = self._get_gpu_partition(request)
                    if await self.gpu_partition.allocate(partition, amount):
                        acquired.append((ResourceType.GPU, amount, partition))
                    else:
                        return False
                else:
                    # Regular resource pool
                    pool = self.pools.get(resource_type)
                    if pool and await pool.acquire(amount):
                        acquired.append((resource_type, amount, None))
                    else:
                        return False
            
            return True
            
        except Exception:
            # Rollback on failure
            for resource_type, amount, partition in acquired:
                if resource_type == ResourceType.GPU:
                    await self.gpu_partition.release(partition, amount)
                else:
                    pool = self.pools.get(resource_type)
                    if pool:
                        await pool.release(amount)
            return False
    
    async def _release_resources(self, request: ResourceRequest):
        """Release all resources for request."""
        for resource_type, amount in request.resources.items():
            if resource_type == ResourceType.GPU:
                partition = self._get_gpu_partition(request)
                await self.gpu_partition.release(partition, amount)
            else:
                pool = self.pools.get(resource_type)
                if pool:
                    await pool.release(amount)
    
    async def _queue_request(self, request: ResourceRequest):
        """Queue request for later execution."""
        queue_entry = QueueEntry(
            request=request,
            enqueue_time=datetime.utcnow(),
            future=asyncio.Future()
        )
        
        await self.queue.put(queue_entry, request.priority)
        bulkhead_queued.add(1, {"bulkhead": self.name})
    
    async def _wait_for_resources(self, request: ResourceRequest) -> bool:
        """Wait for resources to become available."""
        # This would be implemented with proper queue processing
        # For now, simplified timeout
        timeout = request.estimated_duration or timedelta(seconds=30)
        
        try:
            await asyncio.wait_for(
                self._acquire_resources(request),
                timeout=timeout.total_seconds()
            )
            return True
        except asyncio.TimeoutError:
            return False
    
    def _get_gpu_partition(self, request: ResourceRequest) -> str:
        """Determine GPU partition for request."""
        if request.priority == PriorityLevel.CRITICAL:
            return "emergency"
        elif "training" in request.operation_name.lower():
            return "training"
        else:
            return "inference"
    
    async def _auto_scaling_loop(self):
        """Background task for auto-scaling."""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                for resource_type, pool in self.pools.items():
                    if pool.should_scale_up():
                        await pool.scale(self.config.scale_factor)
                    elif pool.should_scale_down():
                        await pool.scale(1 / self.config.scale_factor)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")
    
    async def _metrics_loop(self):
        """Background task for metrics collection."""
        while True:
            try:
                await asyncio.sleep(1)  # Update every second
                
                # Update queue metrics
                bulkhead_queued.set(self.queue.qsize(), {"bulkhead": self.name})
                
                # Update pool metrics
                for resource_type, pool in self.pools.items():
                    pool._update_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics error: {e}")
    
    async def _publish_execution_event(self, request: ResourceRequest, start_time: datetime):
        """Publish execution event to Kafka."""
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Import here to avoid circular imports
        from aura_intelligence.events.schemas import SystemEvent, EventType
        
        event = SystemEvent(
            event_type=EventType.SYSTEM_METRIC,
            component="bulkhead",
            instance_id=self.name,
            data={
                "bulkhead": self.name,
                "request_id": request.id,
                "operation": request.operation_name,
                "priority": request.priority.value,
                "resources": {k.value: v for k, v in request.resources.items()},
                "duration_seconds": duration
            }
        )
        
        await self.event_producer.send_event("bulkhead.executions", event)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current bulkhead metrics."""
        return {
            "name": self.name,
            "queue_size": self.queue.qsize(),
            "pools": {
                resource_type.value: {
                    "capacity": pool.capacity,
                    "available": pool.available,
                    "utilization": pool.get_utilization()
                }
                for resource_type, pool in self.pools.items()
            },
            "gpu_allocated": dict(self.gpu_partition.allocated),
            "cost_per_minute": self.cost_tracker.get_current_rate()
        }


@dataclass
class QueueEntry:
    """Entry in the priority queue."""
    request: ResourceRequest
    enqueue_time: datetime
    future: asyncio.Future


class CostTracker:
    """Tracks cost of bulkhead usage."""
    
    def __init__(self, max_cost_per_minute: float):
        self.max_cost_per_minute = max_cost_per_minute
        self.usage_history = deque(maxlen=60)  # 1 minute window
        self.lock = asyncio.Lock()
    
    async def can_admit(self, request: ResourceRequest) -> bool:
        """Check if request can be admitted based on cost."""
        current_rate = self.get_current_rate()
        projected_rate = current_rate + request.cost_estimate
        
        return projected_rate <= self.max_cost_per_minute
    
    async def record_usage(self, request: ResourceRequest, duration: float):
        """Record actual usage cost."""
        async with self.lock:
            cost = request.cost_estimate * (duration / 60.0)  # Per minute
            self.usage_history.append({
                "timestamp": datetime.utcnow(),
                "cost": cost
            })
    
    def get_current_rate(self) -> float:
        """Get current cost rate per minute."""
        now = datetime.utcnow()
        recent_costs = [
            entry["cost"]
            for entry in self.usage_history
            if now - entry["timestamp"] < timedelta(minutes=1)
        ]
        
        return sum(recent_costs)


class BulkheadRejectedException(Exception):
    """Raised when bulkhead rejects a request."""
    pass