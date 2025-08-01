"""
Hierarchical Consensus Manager for AURA Intelligence.

Routes decisions to appropriate consensus protocols based on criticality.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
import structlog

from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode

from .types import (
    DecisionType,
    ConsensusRequest,
    ConsensusResult,
    ConsensusState,
    ConsensusConfig,
    Vote
)
from .raft import RaftConsensus, RaftConfig
from .byzantine import ByzantineConsensus, BFTConfig
from .multi_raft import MultiRaftConsensus, MultiRaftConfig
from .validation import NeuroSymbolicValidator, ValidatorConfig
from ..events import EventProducer, ProducerConfig, ConsensusDecisionEvent
from ..agents.temporal import TemporalClient

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

# Metrics
consensus_requests = meter.create_counter(
    name="consensus.requests",
    description="Number of consensus requests",
    unit="1"
)

consensus_decisions = meter.create_counter(
    name="consensus.decisions",
    description="Number of consensus decisions",
    unit="1"
)

consensus_latency = meter.create_histogram(
    name="consensus.latency",
    description="Consensus decision latency",
    unit="ms"
)

consensus_failures = meter.create_counter(
    name="consensus.failures",
    description="Number of consensus failures",
    unit="1"
)


class ConsensusManager:
    """
    Hierarchical consensus manager that routes decisions
    to appropriate consensus protocols.
    """
    
    def __init__(self, config: ConsensusConfig):
        self.config = config
        
        # Initialize consensus protocols
        self.raft_consensus = self._init_raft(config)
        self.bft_consensus = self._init_bft(config)
        self.multi_raft = self._init_multi_raft(config)
        
        # Initialize validator
        self.validator = self._init_validator(config)
        
        # Integration clients
        self.temporal_client = TemporalClient(
            namespace=config.temporal_namespace
        )
        self.event_producer = EventProducer(
            ProducerConfig(
                bootstrap_servers=config.kafka_bootstrap_servers
            )
        )
        
        # State tracking
        self.active_requests: Dict[str, ConsensusRequest] = {}
        self._started = False
    
    def _init_raft(self, config: ConsensusConfig) -> RaftConsensus:
        """Initialize Raft consensus."""
        raft_config = RaftConfig(
            node_id="consensus-manager",
            peers=["node-1", "node-2", "node-3"],  # TODO: Dynamic discovery
            election_timeout_ms=150,
            heartbeat_interval_ms=50
        )
        return RaftConsensus(raft_config)
    
    def _init_bft(self, config: ConsensusConfig) -> ByzantineConsensus:
        """Initialize Byzantine consensus."""
        bft_config = BFTConfig(
            node_id="consensus-manager",
            validators=["validator-1", "validator-2", "validator-3", "validator-4"],
            view_timeout_ms=5000,
            batch_size=100
        )
        return ByzantineConsensus(bft_config)
    
    def _init_multi_raft(self, config: ConsensusConfig) -> MultiRaftConsensus:
        """Initialize Multi-Raft consensus."""
        multi_raft_config = MultiRaftConfig(
            groups=[
                {"name": "agents", "nodes": ["agent-1", "agent-2", "agent-3"]},
                {"name": "workflows", "nodes": ["workflow-1", "workflow-2", "workflow-3"]},
                {"name": "resources", "nodes": ["resource-1", "resource-2", "resource-3"]}
            ]
        )
        return MultiRaftConsensus(multi_raft_config)
    
    def _init_validator(self, config: ConsensusConfig) -> NeuroSymbolicValidator:
        """Initialize neuro-symbolic validator."""
        validator_config = ValidatorConfig(
            model_path="models/consensus_validator.pt",
            confidence_threshold=config.neural_confidence_threshold,
            rules_path="config/consensus_rules.yaml"
        )
        return NeuroSymbolicValidator(validator_config)
    
    async def start(self):
        """Start the consensus manager."""
        if self._started:
            return
        
        logger.info("Starting consensus manager")
        
        # Start components
        await self.event_producer.start()
        await self.temporal_client.connect()
        
        # Start consensus protocols
        await self.raft_consensus.start()
        if self.config.use_bft_for_strategic:
            await self.bft_consensus.start()
        if self.config.use_multi_raft_for_tactical:
            await self.multi_raft.start()
        
        self._started = True
        logger.info("Consensus manager started")
    
    async def stop(self):
        """Stop the consensus manager."""
        if not self._started:
            return
        
        logger.info("Stopping consensus manager")
        
        # Stop protocols
        await self.raft_consensus.stop()
        await self.bft_consensus.stop()
        await self.multi_raft.stop()
        
        # Stop clients
        await self.event_producer.stop()
        
        self._started = False
        logger.info("Consensus manager stopped")
    
    async def propose(self, request: ConsensusRequest) -> ConsensusResult:
        """
        Route consensus request to appropriate protocol.
        
        Args:
            request: Consensus request with proposal and metadata
            
        Returns:
            ConsensusResult with decision and explanation
        """
        if not self._started:
            await self.start()
        
        # Start span for tracing
        with tracer.start_as_current_span(
            "consensus.propose",
            attributes={
                "request_id": request.request_id,
                "decision_type": request.decision_type.value,
                "proposer_id": request.proposer_id
            }
        ) as span:
            start_time = datetime.utcnow()
            
            try:
                # Track active request
                self.active_requests[request.request_id] = request
                
                # Record metric
                consensus_requests.add(
                    1,
                    {
                        "decision_type": request.decision_type.value,
                        "priority": str(request.priority)
                    }
                )
                
                # Pre-validation
                validation = await self._pre_validate(request)
                if not validation.is_valid:
                    result = ConsensusResult(
                        request_id=request.request_id,
                        status=ConsensusState.REJECTED,
                        reason=validation.reason,
                        started_at=start_time,
                        completed_at=datetime.utcnow()
                    )
                    span.set_status(Status(StatusCode.ERROR, validation.reason))
                    return result
                
                # Route to appropriate consensus protocol
                result = await self._route_request(request)
                
                # Post-validation and explanation
                if result.is_successful() and self.config.require_explanation:
                    explanation = await self.validator.explain_decision(request, result)
                    result.explanation = explanation
                
                # Publish decision event
                await self._publish_decision(request, result)
                
                # Record metrics
                duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                consensus_latency.record(
                    duration_ms,
                    {
                        "decision_type": request.decision_type.value,
                        "status": result.status.value
                    }
                )
                
                consensus_decisions.add(
                    1,
                    {
                        "decision_type": request.decision_type.value,
                        "status": result.status.value
                    }
                )
                
                span.set_status(Status(StatusCode.OK))
                span.set_attribute("consensus.status", result.status.value)
                
                return result
                
            except Exception as e:
                logger.error(
                    "Consensus proposal failed",
                    request_id=request.request_id,
                    error=str(e)
                )
                
                consensus_failures.add(
                    1,
                    {
                        "decision_type": request.decision_type.value,
                        "error": type(e).__name__
                    }
                )
                
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                
                return ConsensusResult(
                    request_id=request.request_id,
                    status=ConsensusState.FAILED,
                    reason=str(e),
                    started_at=start_time,
                    completed_at=datetime.utcnow()
                )
                
            finally:
                # Clean up
                self.active_requests.pop(request.request_id, None)
    
    async def _pre_validate(self, request: ConsensusRequest):
        """Pre-validate consensus request."""
        return await self.validator.pre_validate(request)
    
    async def _route_request(self, request: ConsensusRequest) -> ConsensusResult:
        """Route request to appropriate consensus protocol."""
        # Set default quorum sizes if not specified
        if request.quorum_size is None:
            if request.decision_type == DecisionType.OPERATIONAL:
                request.quorum_size = self.config.operational_quorum
            elif request.decision_type == DecisionType.TACTICAL:
                request.quorum_size = self.config.tactical_quorum
            elif request.decision_type == DecisionType.STRATEGIC:
                request.quorum_size = self.config.strategic_quorum
            else:  # EMERGENCY
                request.quorum_size = self.config.operational_quorum
        
        # Set timeout based on decision type
        if request.decision_type == DecisionType.OPERATIONAL:
            request.timeout = self.config.operational_timeout
        elif request.decision_type == DecisionType.TACTICAL:
            request.timeout = self.config.tactical_timeout
        elif request.decision_type == DecisionType.STRATEGIC:
            request.timeout = self.config.strategic_timeout
        
        # Route to protocol
        if request.decision_type == DecisionType.OPERATIONAL:
            return await self.raft_consensus.propose(request)
            
        elif request.decision_type == DecisionType.TACTICAL:
            if self.config.use_multi_raft_for_tactical:
                return await self.multi_raft.propose(request)
            else:
                return await self.raft_consensus.propose(request)
                
        elif request.decision_type == DecisionType.STRATEGIC:
            if self.config.use_bft_for_strategic:
                return await self.bft_consensus.propose(request)
            else:
                # Use Raft with higher quorum
                request.quorum_size = self.config.strategic_quorum
                return await self.raft_consensus.propose(request)
                
        elif request.decision_type == DecisionType.EMERGENCY:
            # Fast path for emergency decisions
            return await self._emergency_consensus(request)
            
        else:
            raise ValueError(f"Unknown decision type: {request.decision_type}")
    
    async def _emergency_consensus(self, request: ConsensusRequest) -> ConsensusResult:
        """
        Fast consensus for emergency decisions.
        Uses first responder with validation.
        """
        # Get first available validator
        validators = request.validators or ["emergency-1", "emergency-2", "emergency-3"]
        
        # Collect votes quickly with short timeout
        votes = []
        vote_tasks = []
        
        for validator in validators[:3]:  # Only need 3 for speed
            task = asyncio.create_task(
                self._get_emergency_vote(validator, request)
            )
            vote_tasks.append(task)
        
        # Wait for first valid vote
        done, pending = await asyncio.wait(
            vote_tasks,
            timeout=0.1,  # 100ms timeout
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
        
        # Process completed votes
        for task in done:
            try:
                vote = await task
                if vote and vote.vote_type == VoteType.APPROVE:
                    votes.append(vote)
                    
                    # One valid vote is enough for emergency
                    return ConsensusResult(
                        request_id=request.request_id,
                        status=ConsensusState.ACCEPTED,
                        decision=request.proposal,
                        votes=votes,
                        consensus_type="emergency",
                        reason="Emergency consensus achieved"
                    )
            except Exception as e:
                logger.warning(f"Emergency vote failed: {e}")
        
        # No approval received
        return ConsensusResult(
            request_id=request.request_id,
            status=ConsensusState.REJECTED,
            votes=votes,
            consensus_type="emergency",
            reason="No emergency approval received"
        )
    
    async def _get_emergency_vote(
        self,
        validator_id: str,
        request: ConsensusRequest
    ) -> Optional[Vote]:
        """Get emergency vote from validator."""
        # In production, this would call the actual validator
        # For now, simulate with simple logic
        await asyncio.sleep(0.05)  # Simulate network delay
        
        # Emergency approval logic
        if "safety" in str(request.proposal).lower():
            return Vote(
                voter_id=validator_id,
                vote_type=VoteType.APPROVE,
                reason="Safety-critical decision approved"
            )
        
        return Vote(
            voter_id=validator_id,
            vote_type=VoteType.REJECT,
            reason="Not a valid emergency"
        )
    
    async def _publish_decision(
        self,
        request: ConsensusRequest,
        result: ConsensusResult
    ):
        """Publish consensus decision to event mesh."""
        try:
            event = ConsensusDecisionEvent(
                proposal_id=request.request_id,
                decision=result.decision,
                status=result.status.value,
                consensus_type=result.consensus_type,
                votes=[v.to_dict() for v in result.votes],
                explanation=(
                    result.explanation.to_natural_language()
                    if result.explanation else None
                ),
                duration_ms=result.duration_ms,
                participation_rate=result.participation_rate
            )
            
            await self.event_producer.send_event(
                "consensus.decisions",
                event
            )
            
        except Exception as e:
            logger.error(f"Failed to publish decision: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get consensus manager status."""
        return {
            "started": self._started,
            "active_requests": len(self.active_requests),
            "protocols": {
                "raft": await self.raft_consensus.get_status(),
                "bft": await self.bft_consensus.get_status() if self.config.use_bft_for_strategic else None,
                "multi_raft": await self.multi_raft.get_status() if self.config.use_multi_raft_for_tactical else None
            }
        }


class HierarchicalConsensus:
    """
    Wrapper for hierarchical consensus with multiple managers.
    Useful for large-scale deployments.
    """
    
    def __init__(self, levels: List[ConsensusConfig]):
        self.managers = [ConsensusManager(config) for config in levels]
        self.levels = len(levels)
    
    async def propose(
        self,
        request: ConsensusRequest,
        level: int = 0
    ) -> ConsensusResult:
        """Propose at specific hierarchy level."""
        if level >= self.levels:
            raise ValueError(f"Invalid level {level}, max is {self.levels - 1}")
        
        return await self.managers[level].propose(request)
    
    async def escalate(
        self,
        request: ConsensusRequest,
        from_level: int
    ) -> ConsensusResult:
        """Escalate decision to higher level."""
        next_level = from_level + 1
        
        if next_level >= self.levels:
            raise ValueError("Cannot escalate beyond top level")
        
        # Modify request for escalation
        request.decision_type = DecisionType.STRATEGIC
        request.context["escalated_from"] = from_level
        request.priority += 10  # Increase priority
        
        return await self.propose(request, next_level)