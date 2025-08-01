"""
Temporal Workflows for Consensus Integration.

Bridges consensus protocols with durable workflow orchestration,
enabling reliable distributed decision-making with automatic retries
and state persistence.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import timedelta
import structlog

from temporalio import workflow
from temporalio.common import RetryPolicy

from .types import (
    ConsensusRequest,
    ConsensusResult,
    DecisionType,
    ConsensusState
)

logger = structlog.get_logger()


@dataclass
class ConsensusWorkflowInput:
    """Input for consensus workflow."""
    request: ConsensusRequest
    consensus_type: Optional[str] = None  # "raft", "bft", or None for auto
    validators: Optional[List[str]] = None
    timeout_seconds: int = 30


@dataclass
class ConsensusWorkflowResult:
    """Result from consensus workflow."""
    consensus_result: ConsensusResult
    execution_time_ms: float
    consensus_type_used: str
    retry_count: int = 0


@workflow.defn
class ConsensusWorkflow:
    """
    Main consensus workflow that orchestrates distributed decision-making.
    
    Features:
    - Automatic consensus type selection based on decision criticality
    - Retry handling with exponential backoff
    - State persistence across failures
    - Integration with event mesh for notifications
    """
    
    @workflow.run
    async def run(self, input: ConsensusWorkflowInput) -> ConsensusWorkflowResult:
        """Execute consensus workflow."""
        start_time = workflow.now()
        retry_count = 0
        
        # Log workflow start
        workflow.logger.info(
            "Starting consensus workflow",
            request_id=input.request.request_id,
            decision_type=input.request.decision_type.value
        )
        
        # Determine consensus type if not specified
        consensus_type = input.consensus_type
        if not consensus_type:
            consensus_type = await workflow.execute_activity(
                "determine_consensus_type",
                input.request.decision_type,
                start_to_close_timeout=timedelta(seconds=5)
            )
        
        # Execute consensus with retries
        retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=1),
            maximum_interval=timedelta(seconds=10),
            maximum_attempts=3,
            backoff_coefficient=2.0
        )
        
        try:
            # Pre-consensus validation
            validation_result = await workflow.execute_activity(
                "validate_consensus_request",
                input.request,
                retry_policy=retry_policy,
                start_to_close_timeout=timedelta(seconds=10)
            )
            
            if not validation_result["valid"]:
                return ConsensusWorkflowResult(
                    consensus_result=ConsensusResult(
                        request_id=input.request.request_id,
                        status=ConsensusState.REJECTED,
                        reason=validation_result["reason"]
                    ),
                    execution_time_ms=self._calculate_duration_ms(start_time),
                    consensus_type_used="none",
                    retry_count=0
                )
            
            # Execute consensus
            consensus_result = await workflow.execute_activity(
                f"execute_{consensus_type}_consensus",
                input.request,
                input.validators,
                retry_policy=retry_policy,
                start_to_close_timeout=timedelta(seconds=input.timeout_seconds),
                heartbeat_timeout=timedelta(seconds=5)
            )
            
            # Post-consensus actions
            await workflow.execute_activity(
                "publish_consensus_result",
                consensus_result,
                retry_policy=retry_policy,
                start_to_close_timeout=timedelta(seconds=5)
            )
            
            # Record metrics
            await workflow.execute_activity(
                "record_consensus_metrics",
                {
                    "request_id": input.request.request_id,
                    "consensus_type": consensus_type,
                    "duration_ms": self._calculate_duration_ms(start_time),
                    "status": consensus_result.status.value,
                    "retry_count": retry_count
                },
                start_to_close_timeout=timedelta(seconds=5)
            )
            
            return ConsensusWorkflowResult(
                consensus_result=consensus_result,
                execution_time_ms=self._calculate_duration_ms(start_time),
                consensus_type_used=consensus_type,
                retry_count=retry_count
            )
            
        except Exception as e:
            workflow.logger.error(
                "Consensus workflow failed",
                request_id=input.request.request_id,
                error=str(e)
            )
            
            # Return failure result
            return ConsensusWorkflowResult(
                consensus_result=ConsensusResult(
                    request_id=input.request.request_id,
                    status=ConsensusState.FAILED,
                    reason=str(e)
                ),
                execution_time_ms=self._calculate_duration_ms(start_time),
                consensus_type_used=consensus_type or "none",
                retry_count=retry_count
            )
    
    def _calculate_duration_ms(self, start_time) -> float:
        """Calculate duration in milliseconds."""
        duration = workflow.now() - start_time
        return duration.total_seconds() * 1000


@workflow.defn
class ConsensusVotingWorkflow:
    """
    Workflow for collecting and tallying votes in consensus protocols.
    
    Used by both Raft and BFT for reliable vote collection.
    """
    
    @workflow.run
    async def run(
        self,
        proposal_id: str,
        voters: List[str],
        vote_timeout_seconds: int = 10,
        required_votes: int = None
    ) -> Dict[str, Any]:
        """Collect votes from validators."""
        
        if required_votes is None:
            required_votes = len(voters) // 2 + 1
        
        # Start vote collection
        vote_futures = []
        for voter in voters:
            future = workflow.execute_activity(
                "request_vote",
                proposal_id,
                voter,
                start_to_close_timeout=timedelta(seconds=vote_timeout_seconds),
                retry_policy=RetryPolicy(maximum_attempts=2)
            )
            vote_futures.append(future)
        
        # Collect votes with early termination
        votes = []
        approve_count = 0
        reject_count = 0
        
        for future in vote_futures:
            try:
                vote = await future
                votes.append(vote)
                
                if vote["vote_type"] == "APPROVE":
                    approve_count += 1
                else:
                    reject_count += 1
                
                # Early termination if we have enough votes
                if approve_count >= required_votes:
                    workflow.logger.info(
                        "Consensus reached early",
                        proposal_id=proposal_id,
                        approve_count=approve_count
                    )
                    break
                    
                if reject_count > len(voters) - required_votes:
                    workflow.logger.info(
                        "Consensus rejected early",
                        proposal_id=proposal_id,
                        reject_count=reject_count
                    )
                    break
                    
            except Exception as e:
                workflow.logger.warning(
                    "Vote collection failed",
                    proposal_id=proposal_id,
                    error=str(e)
                )
        
        # Return voting results
        return {
            "proposal_id": proposal_id,
            "total_voters": len(voters),
            "votes_collected": len(votes),
            "approve_count": approve_count,
            "reject_count": reject_count,
            "consensus_reached": approve_count >= required_votes,
            "votes": votes
        }


@workflow.defn
class BFTConsensusWorkflow:
    """
    Byzantine Fault Tolerant consensus workflow.
    
    Implements three-phase commit with Byzantine fault detection.
    """
    
    @workflow.run
    async def run(
        self,
        request: ConsensusRequest,
        validators: List[str],
        byzantine_threshold: float = 0.33
    ) -> ConsensusResult:
        """Execute BFT consensus workflow."""
        
        workflow.logger.info(
            "Starting BFT consensus",
            request_id=request.request_id,
            validator_count=len(validators)
        )
        
        # Calculate thresholds
        total_validators = len(validators)
        required_votes = int(total_validators * (2/3)) + 1
        
        # Phase 1: Prepare
        prepare_result = await workflow.execute_child_workflow(
            ConsensusVotingWorkflow.run,
            f"{request.request_id}-prepare",
            validators,
            vote_timeout_seconds=5,
            required_votes=required_votes,
            id=f"bft-prepare-{request.request_id}"
        )
        
        if not prepare_result["consensus_reached"]:
            return ConsensusResult(
                request_id=request.request_id,
                status=ConsensusState.REJECTED,
                reason="Prepare phase failed"
            )
        
        # Phase 2: Pre-commit
        precommit_result = await workflow.execute_child_workflow(
            ConsensusVotingWorkflow.run,
            f"{request.request_id}-precommit",
            validators,
            vote_timeout_seconds=5,
            required_votes=required_votes,
            id=f"bft-precommit-{request.request_id}"
        )
        
        if not precommit_result["consensus_reached"]:
            return ConsensusResult(
                request_id=request.request_id,
                status=ConsensusState.REJECTED,
                reason="Pre-commit phase failed"
            )
        
        # Phase 3: Commit
        commit_result = await workflow.execute_child_workflow(
            ConsensusVotingWorkflow.run,
            f"{request.request_id}-commit",
            validators,
            vote_timeout_seconds=5,
            required_votes=required_votes,
            id=f"bft-commit-{request.request_id}"
        )
        
        if not commit_result["consensus_reached"]:
            return ConsensusResult(
                request_id=request.request_id,
                status=ConsensusState.REJECTED,
                reason="Commit phase failed"
            )
        
        # Byzantine detection
        byzantine_nodes = await workflow.execute_activity(
            "detect_byzantine_behavior",
            {
                "prepare_votes": prepare_result["votes"],
                "precommit_votes": precommit_result["votes"],
                "commit_votes": commit_result["votes"]
            },
            start_to_close_timeout=timedelta(seconds=10)
        )
        
        if byzantine_nodes:
            workflow.logger.warning(
                "Byzantine nodes detected",
                request_id=request.request_id,
                byzantine_nodes=byzantine_nodes
            )
        
        # Consensus achieved!
        return ConsensusResult(
            request_id=request.request_id,
            status=ConsensusState.ACCEPTED,
            decision=request.proposal,
            consensus_type="bft",
            metadata={
                "phases": {
                    "prepare": prepare_result,
                    "precommit": precommit_result,
                    "commit": commit_result
                },
                "byzantine_nodes": byzantine_nodes
            }
        )


@workflow.defn
class ResourceAllocationWorkflow:
    """
    Specialized workflow for resource allocation using consensus.
    
    Example of selective consensus usage for critical resources.
    """
    
    @workflow.run
    async def run(
        self,
        agent_id: str,
        resource_type: str,
        quantity: int,
        duration_hours: int
    ) -> Dict[str, Any]:
        """Allocate resources with consensus."""
        
        # Check resource availability
        available = await workflow.execute_activity(
            "check_resource_availability",
            resource_type,
            quantity,
            start_to_close_timeout=timedelta(seconds=5)
        )
        
        if not available["sufficient"]:
            return {
                "status": "rejected",
                "reason": "Insufficient resources",
                "available": available["current"],
                "requested": quantity
            }
        
        # Create consensus request
        consensus_request = ConsensusRequest(
            request_id=f"alloc-{agent_id}-{workflow.now().timestamp()}",
            decision_type=DecisionType.RESOURCE_ALLOCATION,
            proposal={
                "action": "allocate",
                "agent_id": agent_id,
                "resource_type": resource_type,
                "quantity": quantity,
                "duration_hours": duration_hours
            },
            timeout=timedelta(seconds=10),
            requester=agent_id
        )
        
        # Execute consensus (will use Raft for operational decisions)
        consensus_result = await workflow.execute_child_workflow(
            ConsensusWorkflow.run,
            ConsensusWorkflowInput(
                request=consensus_request,
                consensus_type="raft"  # Fast consensus for resources
            ),
            id=f"resource-consensus-{consensus_request.request_id}"
        )
        
        if consensus_result.consensus_result.status != ConsensusState.ACCEPTED:
            return {
                "status": "rejected",
                "reason": consensus_result.consensus_result.reason
            }
        
        # Allocate resources
        allocation = await workflow.execute_activity(
            "allocate_resources",
            {
                "agent_id": agent_id,
                "resource_type": resource_type,
                "quantity": quantity,
                "duration_hours": duration_hours
            },
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=RetryPolicy(maximum_attempts=3)
        )
        
        # Schedule deallocation
        await workflow.sleep(timedelta(hours=duration_hours))
        
        await workflow.execute_activity(
            "deallocate_resources",
            allocation["allocation_id"],
            start_to_close_timeout=timedelta(seconds=10)
        )
        
        return {
            "status": "completed",
            "allocation_id": allocation["allocation_id"],
            "consensus_proof": consensus_result.consensus_result.consensus_proof
        }