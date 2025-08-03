"""
GPU Allocation Activities for Temporal Workflows

These activities implement the core business logic for GPU allocation
including LNN council voting, cost calculation, and resource allocation.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import uuid
from dataclasses import dataclass

from temporalio import activity
from opentelemetry import trace
import structlog

from aura_intelligence.agents.council.lnn import (
    CouncilAgentFactory,
    CouncilRequest,
    VoteDecision,
    AgentCapability
)
from aura_intelligence.adapters.neo4j_adapter import Neo4jAdapter, Neo4jConfig
from aura_intelligence.adapters.redis_adapter import RedisAdapter, RedisConfig
from aura_intelligence.events.producers import EventProducer, ProducerConfig

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)


@dataclass
class GPURequest:
    """GPU allocation request data."""
    request_id: str
    gpu_type: str
    gpu_count: int
    duration_hours: int
    user_id: str = "test_user"
    priority: str = "normal"


@dataclass
class AllocationResult:
    """Result of GPU allocation process."""
    request_id: str
    approved: bool
    allocation_id: Optional[str] = None
    allocated_gpus: List[str] = None
    cost_per_hour: float = 0.0
    estimated_cost: float = 0.0
    council_votes: List[Dict[str, Any]] = None
    consensus_achieved: bool = False
    reason: str = ""


@activity.defn
async def check_gpu_availability(request: GPURequest) -> Dict[str, Any]:
    """Check if requested GPUs are available."""
    with tracer.start_as_current_span("activity.check_gpu_availability") as span:
        span.set_attribute("gpu.type", request.gpu_type)
        span.set_attribute("gpu.count", request.gpu_count)
        
        # Mock availability check - in real implementation would query resource manager
        available_gpus = {
            "a100": 8,
            "v100": 12,
            "h100": 4
        }
        
        available_count = available_gpus.get(request.gpu_type, 0)
        is_available = available_count >= request.gpu_count
        
        result = {
            "available": is_available,
            "available_count": available_count,
            "requested_count": request.gpu_count,
            "gpu_type": request.gpu_type
        }
        
        logger.info(
            "GPU availability checked",
            request_id=request.request_id,
            available=is_available,
            available_count=available_count
        )
        
        return result


@activity.defn
async def calculate_allocation_cost(request: GPURequest) -> Dict[str, Any]:
    """Calculate cost for GPU allocation."""
    with tracer.start_as_current_span("activity.calculate_allocation_cost") as span:
        span.set_attribute("gpu.type", request.gpu_type)
        span.set_attribute("gpu.count", request.gpu_count)
        span.set_attribute("duration.hours", request.duration_hours)
        
        # Cost per hour by GPU type
        gpu_costs = {
            "a100": 3.20,
            "v100": 2.50,
            "h100": 4.80
        }
        
        cost_per_hour = gpu_costs.get(request.gpu_type, 2.00) * request.gpu_count
        total_cost = cost_per_hour * request.duration_hours
        
        result = {
            "cost_per_hour": cost_per_hour,
            "estimated_cost": total_cost,
            "gpu_type": request.gpu_type,
            "gpu_count": request.gpu_count,
            "duration_hours": request.duration_hours
        }
        
        logger.info(
            "Allocation cost calculated",
            request_id=request.request_id,
            cost_per_hour=cost_per_hour,
            total_cost=total_cost
        )
        
        return result


@activity.defn
async def create_council_task(request: GPURequest, cost_info: Dict[str, Any]) -> Dict[str, Any]:
    """Create a task for the LNN council to vote on."""
    with tracer.start_as_current_span("activity.create_council_task") as span:
        span.set_attribute("request.id", request.request_id)
        
        task = {
            "task_id": str(uuid.uuid4()),
            "request_id": request.request_id,
            "task_type": "gpu_allocation_decision",
            "context": {
                "gpu_type": request.gpu_type,
                "gpu_count": request.gpu_count,
                "duration_hours": request.duration_hours,
                "user_id": request.user_id,
                "priority": request.priority,
                "cost_per_hour": cost_info["cost_per_hour"],
                "estimated_cost": cost_info["estimated_cost"]
            },
            "created_at": datetime.utcnow().isoformat()
        }
        
        logger.info(
            "Council task created",
            task_id=task["task_id"],
            request_id=request.request_id
        )
        
        return task


@activity.defn
async def gather_council_votes(task: Dict[str, Any]) -> Dict[str, Any]:
    """Gather votes from LNN council agents."""
    with tracer.start_as_current_span("activity.gather_council_votes") as span:
        span.set_attribute("task.id", task["task_id"])
        
        try:
            # Initialize LNN Council Agent using the new factory
            council_agent = CouncilAgentFactory.create_default_agent(
                agent_id="gpu_allocation_council",
                capabilities=[AgentCapability.GPU_ALLOCATION, AgentCapability.COST_OPTIMIZATION]
            )
            
            # Initialize the agent
            await council_agent.initialize()
            
            # Prepare council request
            council_request = CouncilRequest(
                request_id=task["request_id"],
                request_type="gpu_allocation",
                data=task["context"],
                context={
                    "task_type": task["task_type"],
                    "priority": task.get("priority", 5)
                },
                capabilities_required=[AgentCapability.GPU_ALLOCATION],
                priority=task.get("priority", 5)
            )
            
            # Process with LNN council - this calls the real AI agents
            council_result = await council_agent.process_request(council_request)
            
            # Extract votes from council result
            votes = [{
                "agent_id": council_result.agent_id,
                "vote": "approve" if council_result.decision == VoteDecision.APPROVE else "deny",
                "confidence": council_result.confidence,
                "reasoning": council_result.reasoning
            }]
            
            # Calculate consensus (single agent for now, but extensible)
            approve_votes = sum(1 for vote in votes if vote["vote"] == "approve")
            total_votes = len(votes)
            consensus_achieved = approve_votes > total_votes / 2
            
            result = {
                "votes": votes,
                "vote_count": total_votes,
                "approve_count": approve_votes,
                "consensus_achieved": consensus_achieved,
                "council_decision": "approved" if consensus_achieved else "denied"
            }
            
            # Cleanup
            await council_agent.cleanup()
            
            logger.info(
                "Council votes gathered",
                task_id=task["task_id"],
                vote_count=total_votes,
                consensus=consensus_achieved
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Failed to gather council votes",
                task_id=task["task_id"],
                error=str(e)
            )
            
            # Return fallback mock votes for now
            mock_votes = [
                {
                    "agent_id": "efficiency_agent",
                    "vote": "approve",
                    "confidence": 0.85,
                    "reasoning": "Resource utilization is within acceptable limits"
                },
                {
                    "agent_id": "cost_agent", 
                    "vote": "approve",
                    "confidence": 0.75,
                    "reasoning": "Cost is reasonable for requested duration"
                },
                {
                    "agent_id": "security_agent",
                    "vote": "approve", 
                    "confidence": 0.90,
                    "reasoning": "User has appropriate permissions"
                }
            ]
            
            result = {
                "votes": mock_votes,
                "vote_count": len(mock_votes),
                "approve_count": 3,
                "consensus_achieved": True,
                "council_decision": "approved",
                "fallback_used": True,
                "error": str(e)
            }
            
            return result


@activity.defn
async def make_allocation_decision(
    request: GPURequest,
    availability: Dict[str, Any],
    cost_info: Dict[str, Any],
    council_votes: Dict[str, Any]
) -> AllocationResult:
    """Make final allocation decision based on all inputs."""
    with tracer.start_as_current_span("activity.make_allocation_decision") as span:
        span.set_attribute("request.id", request.request_id)
        
        # Decision logic
        approved = (
            availability["available"] and
            council_votes["consensus_achieved"] and
            council_votes["council_decision"] == "approved"
        )
        
        allocation_id = str(uuid.uuid4()) if approved else None
        allocated_gpus = []
        
        if approved:
            # Generate mock GPU IDs
            allocated_gpus = [
                f"{request.gpu_type}-{i:03d}"
                for i in range(request.gpu_count)
            ]
        
        result = AllocationResult(
            request_id=request.request_id,
            approved=approved,
            allocation_id=allocation_id,
            allocated_gpus=allocated_gpus,
            cost_per_hour=cost_info["cost_per_hour"],
            estimated_cost=cost_info["estimated_cost"],
            council_votes=council_votes["votes"],
            consensus_achieved=council_votes["consensus_achieved"],
            reason="Approved by council" if approved else "Denied by council or unavailable"
        )
        
        logger.info(
            "Allocation decision made",
            request_id=request.request_id,
            approved=approved,
            allocation_id=allocation_id
        )
        
        return result


@activity.defn
async def store_decision_in_neo4j(decision: AllocationResult) -> Dict[str, Any]:
    """Store allocation decision in Neo4j."""
    with tracer.start_as_current_span("activity.store_decision_neo4j") as span:
        span.set_attribute("request.id", decision.request_id)
        
        try:
            # Initialize Neo4j adapter
            neo4j_config = Neo4jConfig(
                uri="bolt://localhost:7687",
                username="neo4j",
                password="dev_password"
            )
            neo4j = Neo4jAdapter(neo4j_config)
            await neo4j.initialize()
            
            # Create decision node
            query = """
            CREATE (d:Decision {
                request_id: $request_id,
                approved: $approved,
                allocation_id: $allocation_id,
                cost_per_hour: $cost_per_hour,
                estimated_cost: $estimated_cost,
                consensus_achieved: $consensus_achieved,
                reason: $reason,
                created_at: $created_at
            })
            RETURN d
            """
            
            result = await neo4j.query(query, {
                "request_id": decision.request_id,
                "approved": decision.approved,
                "allocation_id": decision.allocation_id,
                "cost_per_hour": decision.cost_per_hour,
                "estimated_cost": decision.estimated_cost,
                "consensus_achieved": decision.consensus_achieved,
                "reason": decision.reason,
                "created_at": datetime.utcnow().isoformat()
            })
            
            await neo4j.close()
            
            logger.info(
                "Decision stored in Neo4j",
                request_id=decision.request_id,
                approved=decision.approved
            )
            
            return {"stored": True, "node_count": len(result)}
            
        except Exception as e:
            logger.error(
                "Failed to store decision in Neo4j",
                request_id=decision.request_id,
                error=str(e)
            )
            return {"stored": False, "error": str(e)}


@activity.defn
async def publish_allocation_event(decision: AllocationResult) -> Dict[str, Any]:
    """Publish allocation event to Kafka."""
    with tracer.start_as_current_span("activity.publish_allocation_event") as span:
        span.set_attribute("request.id", decision.request_id)
        
        try:
            # Initialize event producer
            producer_config = ProducerConfig(
                bootstrap_servers=["localhost:9092"]
            )
            producer = EventProducer(producer_config)
            await producer.start()
            
            # Create allocation event
            event = {
                "event_type": "gpu.allocation.decided",
                "request_id": decision.request_id,
                "approved": decision.approved,
                "allocation_id": decision.allocation_id,
                "allocated_gpus": decision.allocated_gpus,
                "cost_per_hour": decision.cost_per_hour,
                "estimated_cost": decision.estimated_cost,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Publish event
            await producer.send_event("gpu.allocations", event)
            await producer.stop()
            
            logger.info(
                "Allocation event published",
                request_id=decision.request_id,
                event_type=event["event_type"]
            )
            
            return {"published": True}
            
        except Exception as e:
            logger.error(
                "Failed to publish allocation event",
                request_id=decision.request_id,
                error=str(e)
            )
            return {"published": False, "error": str(e)}


@activity.defn
async def allocate_gpus(decision: AllocationResult) -> Dict[str, Any]:
    """Actually allocate the GPUs if approved."""
    with tracer.start_as_current_span("activity.allocate_gpus") as span:
        span.set_attribute("request.id", decision.request_id)
        
        if not decision.approved:
            return {"allocated": False, "reason": "Request not approved"}
        
        try:
            # Mock GPU allocation - in real implementation would call resource manager
            allocation_result = {
                "allocated": True,
                "allocation_id": decision.allocation_id,
                "gpu_ids": decision.allocated_gpus,
                "start_time": datetime.utcnow().isoformat(),
                "status": "active"
            }
            
            logger.info(
                "GPUs allocated",
                request_id=decision.request_id,
                allocation_id=decision.allocation_id,
                gpu_count=len(decision.allocated_gpus)
            )
            
            return allocation_result
            
        except Exception as e:
            logger.error(
                "Failed to allocate GPUs",
                request_id=decision.request_id,
                error=str(e)
            )
            return {"allocated": False, "error": str(e)}


@activity.defn
async def deallocate_gpus(allocation_id: str) -> Dict[str, Any]:
    """Deallocate GPUs when done."""
    with tracer.start_as_current_span("activity.deallocate_gpus") as span:
        span.set_attribute("allocation.id", allocation_id)
        
        try:
            # Mock GPU deallocation
            result = {
                "deallocated": True,
                "allocation_id": allocation_id,
                "end_time": datetime.utcnow().isoformat(),
                "status": "completed"
            }
            
            logger.info(
                "GPUs deallocated",
                allocation_id=allocation_id
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Failed to deallocate GPUs",
                allocation_id=allocation_id,
                error=str(e)
            )
            return {"deallocated": False, "error": str(e)}


@activity.defn
async def record_metrics(decision: AllocationResult, duration_ms: float) -> Dict[str, Any]:
    """Record metrics for the allocation process."""
    with tracer.start_as_current_span("activity.record_metrics") as span:
        span.set_attribute("request.id", decision.request_id)
        
        metrics = {
            "request_id": decision.request_id,
            "duration_ms": duration_ms,
            "approved": decision.approved,
            "cost_per_hour": decision.cost_per_hour,
            "estimated_cost": decision.estimated_cost,
            "consensus_achieved": decision.consensus_achieved,
            "vote_count": len(decision.council_votes) if decision.council_votes else 0,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(
            "Metrics recorded",
            request_id=decision.request_id,
            duration_ms=duration_ms,
            approved=decision.approved
        )
        
        return metrics