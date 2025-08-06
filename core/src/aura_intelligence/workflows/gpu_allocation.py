"""
GPU Allocation Workflow for AURA Intelligence.

Implements GPU resource allocation using:
- LNN council agents for decision making
- Context-aware allocation strategies
- Fair scheduling algorithms
- Cost optimization
- Automatic deallocation
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import uuid

from temporalio import workflow, activity
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError
import structlog

from ..agents.council.lnn_council import (
    LNNCouncilAgent, CouncilTask, CouncilVote, VoteType
)
from ..consensus.types import DecisionType, ConsensusState
from ..events.schemas import EventType, AgentEvent

logger = structlog.get_logger()


class GPUType(str, Enum):
    """Types of GPUs available."""
    T4 = "t4"
    V100 = "v100"
    A100 = "a100"
    H100 = "h100"


class AllocationPriority(str, Enum):
    """Allocation priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class GPUAllocationRequest:
    """GPU allocation request."""
    request_id: str
    requester_id: str
    gpu_type: GPUType
    gpu_count: int
    duration_hours: int
    priority: AllocationPriority
    
    # Workload information
    workload_type: str  # training, inference, research
    estimated_memory_gb: float
    estimated_compute_tflops: float
    
    # Constraints
    preferred_nodes: Optional[List[str]] = None
    anti_affinity: Optional[List[str]] = None  # Don't co-locate with these
    deadline: Optional[datetime] = None
    
    # Cost constraints
    max_cost_per_hour: Optional[float] = None
    budget_remaining: Optional[float] = None
    
    # Context
    project_id: Optional[str] = None
    previous_allocations: Optional[List[str]] = None
    metadata: Dict[str, Any] = None


@dataclass
class GPUAllocationResult:
    """Result of GPU allocation."""
    allocation_id: str
    request_id: str
    status: str  # allocated, rejected, queued
    
    # Allocation details
    allocated_gpus: List[Dict[str, Any]]  # node_id, gpu_id, gpu_type
    start_time: datetime
    end_time: datetime
    
    # Cost information
    estimated_cost: float
    cost_per_hour: float
    
    # Decision information
    council_votes: List[CouncilVote]
    consensus_achieved: bool
    reasoning: str
    
    # Monitoring
    allocation_token: str  # For tracking usage
    metrics_endpoint: str


@dataclass
class GPUResourceState:
    """Current state of GPU resources."""
    total_gpus: Dict[GPUType, int]
    available_gpus: Dict[GPUType, int]
    allocated_gpus: Dict[GPUType, List[Dict[str, Any]]]
    
    # Utilization metrics
    utilization_by_type: Dict[GPUType, float]
    average_wait_time: Dict[GPUType, float]
    
    # Cost metrics
    current_cost_per_hour: Dict[GPUType, float]
    spot_pricing_available: bool


class GPUAllocationActivities:
    """Activities for GPU allocation workflow."""
    
    @activity.defn
    async def check_gpu_availability(
        self,
        gpu_type: GPUType,
        gpu_count: int
    ) -> Dict[str, Any]:
        """Check if requested GPUs are available."""
        # In production, query actual GPU cluster state
        # For now, mock implementation
        available = {
            GPUType.T4: 20,
            GPUType.V100: 10,
            GPUType.A100: 8,
            GPUType.H100: 4
        }
        
        current_available = available.get(gpu_type, 0)
        
        return {
            "sufficient": current_available >= gpu_count,
            "available": current_available,
            "requested": gpu_count,
            "utilization": 1.0 - (current_available / (current_available + 10))
        }
    
    @activity.defn
    async def calculate_allocation_cost(
        self,
        gpu_type: GPUType,
        gpu_count: int,
        duration_hours: int
    ) -> Dict[str, Any]:
        """Calculate cost for GPU allocation."""
        # Pricing per GPU hour
        pricing = {
            GPUType.T4: 0.526,
            GPUType.V100: 2.48,
            GPUType.A100: 5.12,
            GPUType.H100: 8.00
        }
        
        base_price = pricing.get(gpu_type, 1.0)
        
        # Apply spot pricing discount if available
        spot_discount = 0.7  # 30% discount
        
        # Apply volume discount
        volume_discount = 1.0
        if gpu_count >= 4:
            volume_discount = 0.9
        elif gpu_count >= 8:
            volume_discount = 0.85
            
        cost_per_hour = base_price * gpu_count * volume_discount
        total_cost = cost_per_hour * duration_hours
        
        return {
            "cost_per_hour": cost_per_hour,
            "total_cost": total_cost,
            "base_price": base_price,
            "discounts_applied": {
                "spot": spot_discount if duration_hours < 6 else 1.0,
                "volume": volume_discount
            }
        }
    
    @activity.defn
    async def create_council_task(
        self,
        request: GPUAllocationRequest,
        availability: Dict[str, Any],
        cost_info: Dict[str, Any]
    ) -> CouncilTask:
        """Create council task for allocation decision."""
        return CouncilTask(
            task_id=f"gpu_alloc_{request.request_id}",
            task_type="gpu_allocation",
            payload={
                "action": "allocate_gpu",
                "request": {
                    "gpu_type": request.gpu_type,
                    "gpu_count": request.gpu_count,
                    "duration_hours": request.duration_hours,
                    "workload_type": request.workload_type
                },
                "availability": availability,
                "cost": cost_info,
                "constraints": {
                    "max_cost_per_hour": request.max_cost_per_hour,
                    "budget_remaining": request.budget_remaining,
                    "deadline": request.deadline.isoformat() if request.deadline else None
                }
            },
            context={
                "requester_id": request.requester_id,
                "project_id": request.project_id,
                "previous_allocations": request.previous_allocations or []
            },
            priority=self._map_priority(request.priority),
            deadline=request.deadline
        )
    
    @activity.defn
    async def gather_council_votes(
        self,
        council_task: CouncilTask,
        agent_ids: List[str]
    ) -> List[CouncilVote]:
        """Gather votes from council agents."""
        # In production, this would actually invoke the agents
        # For now, simulate the voting
        votes = []
        
        for agent_id in agent_ids:
            # Create mock vote based on task analysis
            confidence = 0.85 if council_task.payload["availability"]["sufficient"] else 0.3
            vote_type = VoteType.APPROVE if confidence > 0.6 else VoteType.REJECT
            
            vote = CouncilVote(
                agent_id=agent_id,
                vote=vote_type,
                confidence=confidence,
                reasoning=f"Based on availability ({council_task.payload['availability']['available']}) and cost (${council_task.payload['cost']['cost_per_hour']}/hr)",
                supporting_evidence=[
                    {"type": "availability", "data": council_task.payload["availability"]},
                    {"type": "cost", "data": council_task.payload["cost"]}
                ],
                timestamp=datetime.now(timezone.utc)
            )
            votes.append(vote)
            
        return votes
    
    @activity.defn
    async def allocate_gpus(
        self,
        request: GPUAllocationRequest
    ) -> Dict[str, Any]:
        """Actually allocate the GPUs."""
        # In production, interact with Kubernetes GPU operator
        # or cloud provider APIs
        allocation_id = str(uuid.uuid4())
        
        # Mock allocation
        allocated_gpus = []
        for i in range(request.gpu_count):
            allocated_gpus.append({
                "node_id": f"node-{i % 4}",
                "gpu_id": f"gpu-{uuid.uuid4().hex[:8]}",
                "gpu_type": request.gpu_type,
                "pci_address": f"0000:0{i}:00.0"
            })
            
        return {
            "allocation_id": allocation_id,
            "allocated_gpus": allocated_gpus,
            "allocation_token": f"token-{allocation_id}",
            "metrics_endpoint": f"http://metrics.aura.ai/gpu/{allocation_id}"
        }
    
    @activity.defn
    async def deallocate_gpus(
        self,
        allocation_id: str,
        allocated_gpus: List[Dict[str, Any]]
    ):
        """Deallocate GPUs after duration expires."""
        logger.info("Deallocating GPUs", 
                   allocation_id=allocation_id,
                   gpu_count=len(allocated_gpus))
        
        # In production, release GPUs back to pool
        # Clean up any running processes
        # Archive usage metrics
        
    @activity.defn
    async def emit_allocation_event(
        self,
        event_type: EventType,
        allocation_result: GPUAllocationResult
    ):
        """Emit allocation event to Kafka."""
        event = AgentEvent(
            event_type=event_type,
            source_id="gpu_allocation_workflow",
            source_type="workflow",
            agent_id="gpu_allocator",
            agent_type="resource_manager",
            data={
                "allocation_id": allocation_result.allocation_id,
                "request_id": allocation_result.request_id,
                "status": allocation_result.status,
                "gpu_count": len(allocation_result.allocated_gpus),
                "cost_per_hour": allocation_result.cost_per_hour
            }
        )
        
        # In production, publish to Kafka
        logger.info("Emitted allocation event", event_type=event_type)
    
    def _map_priority(self, priority: AllocationPriority) -> int:
        """Map allocation priority to numeric value."""
        mapping = {
            AllocationPriority.LOW: 3,
            AllocationPriority.NORMAL: 5,
            AllocationPriority.HIGH: 7,
            AllocationPriority.CRITICAL: 9
        }
        return mapping.get(priority, 5)


@workflow.defn
class GPUAllocationWorkflow:
    """
    GPU allocation workflow with LNN council decision making.
    
    Features:
    - Context-aware allocation using historical data
    - Multi-agent council voting
    - Cost optimization
    - Fair scheduling
    - Automatic deallocation
    """
    
    @workflow.run
    async def run(self, request: GPUAllocationRequest) -> GPUAllocationResult:
        """Execute GPU allocation workflow."""
        workflow_id = workflow.info().workflow_id
        logger.info("Starting GPU allocation workflow", 
                   request_id=request.request_id,
                   gpu_type=request.gpu_type,
                   gpu_count=request.gpu_count)
        
        # Step 1: Check availability
        availability = await workflow.execute_activity(
            GPUAllocationActivities.check_gpu_availability,
            request.gpu_type,
            request.gpu_count,
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=RetryPolicy(maximum_attempts=3)
        )
        
        # Step 2: Calculate cost
        cost_info = await workflow.execute_activity(
            GPUAllocationActivities.calculate_allocation_cost,
            request.gpu_type,
            request.gpu_count,
            request.duration_hours,
            start_to_close_timeout=timedelta(seconds=5)
        )
        
        # Check budget constraints
        if request.max_cost_per_hour and cost_info["cost_per_hour"] > request.max_cost_per_hour:
            return GPUAllocationResult(
                allocation_id="",
                request_id=request.request_id,
                status="rejected",
                allocated_gpus=[],
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                estimated_cost=0,
                cost_per_hour=0,
                council_votes=[],
                consensus_achieved=False,
                reasoning=f"Cost ${cost_info['cost_per_hour']}/hr exceeds maximum ${request.max_cost_per_hour}/hr",
                allocation_token="",
                metrics_endpoint=""
            )
        
        # Step 3: Create council task
        council_task = await workflow.execute_activity(
            GPUAllocationActivities.create_council_task,
            request,
            availability,
            cost_info,
            start_to_close_timeout=timedelta(seconds=5)
        )
        
        # Step 4: Gather council votes
        # In production, these would be actual LNN council agents
        council_agent_ids = [
            "lnn_cost_optimizer",
            "lnn_fairness_enforcer",
            "lnn_performance_predictor"
        ]
        
        votes = await workflow.execute_activity(
            GPUAllocationActivities.gather_council_votes,
            council_task,
            council_agent_ids,
            start_to_close_timeout=timedelta(seconds=30)
        )
        
        # Step 5: Determine consensus
        approve_count = sum(1 for v in votes if v.vote == VoteType.APPROVE)
        consensus_achieved = approve_count >= len(votes) // 2 + 1
        
        if not consensus_achieved:
            # Emit rejection event
            result = GPUAllocationResult(
                allocation_id="",
                request_id=request.request_id,
                status="rejected",
                allocated_gpus=[],
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                estimated_cost=0,
                cost_per_hour=0,
                council_votes=votes,
                consensus_achieved=False,
                reasoning="Council consensus not achieved for allocation",
                allocation_token="",
                metrics_endpoint=""
            )
            
            await workflow.execute_activity(
                GPUAllocationActivities.emit_allocation_event,
                EventType.AGENT_DECISION_MADE,
                result,
                start_to_close_timeout=timedelta(seconds=5)
            )
            
            return result
        
        # Step 6: Allocate GPUs
        allocation_info = await workflow.execute_activity(
            GPUAllocationActivities.allocate_gpus,
            request,
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=3)
        )
        
        # Step 7: Create result
        start_time = datetime.now(timezone.utc)
        end_time = start_time + timedelta(hours=request.duration_hours)
        
        result = GPUAllocationResult(
            allocation_id=allocation_info["allocation_id"],
            request_id=request.request_id,
            status="allocated",
            allocated_gpus=allocation_info["allocated_gpus"],
            start_time=start_time,
            end_time=end_time,
            estimated_cost=cost_info["total_cost"],
            cost_per_hour=cost_info["cost_per_hour"],
            council_votes=votes,
            consensus_achieved=True,
            reasoning=f"Allocated {request.gpu_count} {request.gpu_type} GPUs for {request.duration_hours} hours",
            allocation_token=allocation_info["allocation_token"],
            metrics_endpoint=allocation_info["metrics_endpoint"]
        )
        
        # Step 8: Emit allocation event
        await workflow.execute_activity(
            GPUAllocationActivities.emit_allocation_event,
            EventType.AGENT_DECISION_MADE,
            result,
            start_to_close_timeout=timedelta(seconds=5)
        )
        
        # Step 9: Schedule deallocation
        await workflow.sleep(timedelta(hours=request.duration_hours))
        
        await workflow.execute_activity(
            GPUAllocationActivities.deallocate_gpus,
            allocation_info["allocation_id"],
            allocation_info["allocated_gpus"],
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=5)
        )
        
        logger.info("GPU allocation workflow completed",
                   allocation_id=result.allocation_id,
                   duration_hours=request.duration_hours)
        
        return result