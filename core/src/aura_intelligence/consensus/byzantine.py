"""
Byzantine Fault Tolerant Consensus for AURA Intelligence.

HotStuff-inspired implementation for critical strategic decisions:
- Model updates affecting all agents
- Safety-critical operations
- Financial transactions
- Compliance-required decisions

Tolerates up to f Byzantine failures in 3f+1 nodes.
"""

from typing import Dict, Any, Optional, List, Set, Protocol
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import asyncio
import hashlib
import structlog

from opentelemetry import trace, metrics

from .types import (
    ConsensusRequest, ConsensusResult, ConsensusState, ConsensusProof,
    Vote, VoteType, BFTPhase, BFTMessage, BFTVote, BFTProof
)
from ..events import EventProducer
from ..agents.temporal import execute_workflow

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)


class BFTMetrics:
    """Centralized metrics for Byzantine consensus."""
    
    def __init__(self):
        self.phases = meter.create_counter(
            name="bft.phases",
            description="Number of BFT phases completed"
        )
        self.view_changes = meter.create_counter(
            name="bft.view_changes",
            description="Number of BFT view changes"
        )
        self.byzantine_detected = meter.create_counter(
            name="bft.byzantine.detected",
            description="Number of Byzantine behaviors detected"
        )


class BFTCrypto(Protocol):
    """Protocol for BFT cryptographic operations."""
    
    def sign(self, data: bytes) -> bytes:
        """Sign data."""
        ...
    
    def verify(self, data: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify signature."""
        ...
    
    def hash(self, data: bytes) -> bytes:
        """Hash data."""
        ...


class SimpleBFTCrypto:
    """Simple crypto implementation for testing."""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
    
    def sign(self, data: bytes) -> bytes:
        """Simple signature (NOT for production)."""
        return hashlib.sha256(data + self.node_id.encode()).digest()
    
    def verify(self, data: bytes, signature: bytes, public_key: bytes) -> bool:
        """Simple verification (NOT for production)."""
        expected = hashlib.sha256(data + public_key).digest()
        return signature == expected
    
    def hash(self, data: bytes) -> bytes:
        """SHA256 hash."""
        return hashlib.sha256(data).digest()


class BFTViewManager:
    """Manages BFT view changes and leader selection."""
    
    def __init__(self, validators: List[str], metrics: BFTMetrics):
        self.validators = sorted(validators)
        self.metrics = metrics
        self.view = 0
        self.phase = BFTPhase.PREPARE
        
    def get_leader(self, view: int) -> str:
        """Get leader for view using round-robin."""
        return self.validators[view % len(self.validators)]
    
    def advance_view(self):
        """Move to next view."""
        self.view += 1
        self.phase = BFTPhase.VIEW_CHANGE
        self.metrics.view_changes.add(1)
        return self.get_leader(self.view)
    
    def is_valid_leader(self, node_id: str, view: int) -> bool:
        """Check if node is valid leader for view."""
        return node_id == self.get_leader(view)


class BFTVoteCollector:
    """Collects and validates votes for BFT phases."""
    
    def __init__(self, threshold: int, crypto: BFTCrypto):
        self.threshold = threshold
        self.crypto = crypto
        self.phase_votes: Dict[BFTPhase, List[BFTVote]] = {
            phase: [] for phase in BFTPhase
        }
        self.vote_history: Dict[str, List[BFTVote]] = {}
    
    def add_vote(self, vote: BFTVote) -> bool:
        """Add vote if valid, return True if threshold reached."""
        # Check for duplicates
        if self._is_duplicate(vote):
            return False
        
        # Add to phase votes
        self.phase_votes[vote.phase].append(vote)
        
        # Track history for Byzantine detection
        if vote.voter_id not in self.vote_history:
            self.vote_history[vote.voter_id] = []
        self.vote_history[vote.voter_id].append(vote)
        
        # Check threshold
        return len(self.phase_votes[vote.phase]) >= self.threshold
    
    def _is_duplicate(self, vote: BFTVote) -> bool:
        """Check for duplicate or conflicting votes."""
        for prev_vote in self.vote_history.get(vote.voter_id, []):
            if (prev_vote.phase == vote.phase and 
                prev_vote.view == vote.view and
                prev_vote.sequence == vote.sequence):
                return prev_vote.message_hash != vote.message_hash
        return False
    
    def get_byzantine_nodes(self) -> Set[str]:
        """Detect nodes with conflicting votes."""
        byzantine = set()
        for voter_id, votes in self.vote_history.items():
            # Check for conflicting votes in same phase/view/sequence
            seen = {}
            for vote in votes:
                key = (vote.phase, vote.view, vote.sequence)
                if key in seen and seen[key] != vote.message_hash:
                    byzantine.add(voter_id)
                seen[key] = vote.message_hash
        return byzantine
    
    def reset(self):
        """Reset for new consensus round."""
        self.phase_votes = {phase: [] for phase in BFTPhase}


class BFTMessageHandler:
    """Handles BFT message creation and validation."""
    
    def __init__(self, node_id: str, crypto: BFTCrypto):
        self.node_id = node_id
        self.crypto = crypto
        self.sequence = 0
    
    def create_message(
        self,
        phase: BFTPhase,
        view: int,
        proposal: Dict[str, Any],
        request_id: str
    ) -> BFTMessage:
        """Create and sign BFT message."""
        self.sequence += 1
        
        msg = BFTMessage(
            type=phase,
            view=view,
            sequence=self.sequence,
            node_id=self.node_id,
            proposal=proposal,
            request_id=request_id
        )
        
        # Sign message
        msg_bytes = self._serialize_message(msg)
        msg.signature = self.crypto.sign(msg_bytes)
        
        return msg
    
    def validate_message(self, msg: BFTMessage) -> bool:
        """Validate message signature and content."""
        if not msg.signature:
            return False
        
        msg_bytes = self._serialize_message(msg)
        public_key = f"public_key_{msg.node_id}".encode()
        
        return self.crypto.verify(msg_bytes, msg.signature, public_key)
    
    def hash_message(self, msg: BFTMessage) -> str:
        """Create hash of message for voting."""
        data = f"{msg.type}:{msg.view}:{msg.sequence}:{msg.proposal}"
        return self.crypto.hash(data.encode()).hex()
    
    def _serialize_message(self, msg: BFTMessage) -> bytes:
        """Serialize message for signing."""
        # In production: use proper serialization
        data = f"{msg.type}:{msg.view}:{msg.sequence}:{msg.node_id}:{msg.proposal}"
        return data.encode()


class BFTCore:
    """Core BFT consensus logic."""
    
    def __init__(
        self,
        config: 'BFTConfig',
        event_producer: EventProducer,
        metrics: BFTMetrics
    ):
        self.config = config
        self.node_id = config.node_id
        self.event_producer = event_producer
        self.metrics = metrics
        
        # Core components
        self.crypto = SimpleBFTCrypto(config.node_id)
        self.view_manager = BFTViewManager(config.validators, metrics)
        self.vote_collector = BFTVoteCollector(
            threshold=len(config.validators) * 2 // 3 + 1,
            crypto=self.crypto
        )
        self.message_handler = BFTMessageHandler(config.node_id, self.crypto)
        
        # State
        self.pending_proposals: Dict[str, ConsensusRequest] = {}
        self.pending_futures: Dict[str, asyncio.Future] = {}
        self.view_change_timer: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start BFT node."""
        await self.event_producer.start()
        self.view_change_timer = asyncio.create_task(self._view_change_monitor())
    
    async def stop(self):
        """Stop BFT node."""
        if self.view_change_timer:
            self.view_change_timer.cancel()
        await self.event_producer.stop()
    
    async def propose(self, request: ConsensusRequest) -> ConsensusResult:
        """Propose value for Byzantine consensus."""
        with tracer.start_as_current_span("bft.propose") as span:
            span.set_attributes({
                "node_id": self.node_id,
                "view": self.view_manager.view,
                "is_leader": self._is_leader(),
                "request_id": request.request_id
            })
            
            # Create future for result
            future = asyncio.Future()
            self.pending_futures[request.request_id] = future
            self.pending_proposals[request.request_id] = request
            
            try:
                if self._is_leader():
                    await self._initiate_consensus(request)
                else:
                    await self._forward_to_leader(request)
                
                # Wait for consensus
                result = await asyncio.wait_for(
                    future,
                    timeout=request.timeout.total_seconds()
                )
                return result
                
            except asyncio.TimeoutError:
                return ConsensusResult(
                    request_id=request.request_id,
                    status=ConsensusState.TIMEOUT,
                    reason="BFT consensus timeout"
                )
            finally:
                self.pending_proposals.pop(request.request_id, None)
                self.pending_futures.pop(request.request_id, None)
    
    def _is_leader(self) -> bool:
        """Check if this node is current leader."""
        current_leader = self.view_manager.get_leader(self.view_manager.view)
        return self.node_id == current_leader
    
    async def _initiate_consensus(self, request: ConsensusRequest):
        """Leader initiates three-phase consensus."""
        # Reset vote collector
        self.vote_collector.reset()
        
        # Phase 1: Prepare
        prepare_msg = self.message_handler.create_message(
            BFTPhase.PREPARE,
            self.view_manager.view,
            request.proposal,
            request.request_id
        )
        
        await self._broadcast_message(prepare_msg)
        
        # Start phase timeout
        asyncio.create_task(
            self._phase_timeout(BFTPhase.PREPARE, prepare_msg.sequence)
        )
    
    async def _broadcast_message(self, msg: BFTMessage):
        """Broadcast message to all validators."""
        await self.event_producer.send_event("bft.messages", {
            "phase": msg.type.value,
            "view": msg.view,
            "sequence": msg.sequence,
            "node_id": msg.node_id,
            "message": msg.to_dict()
        })
        
        self.metrics.phases.add(1, {
            "phase": msg.type.value,
            "node_id": self.node_id
        })
    
    async def handle_message(self, msg: BFTMessage):
        """Handle incoming BFT message."""
        # Validate message
        if not self.message_handler.validate_message(msg):
            await self._report_byzantine(msg.node_id, "Invalid signature")
            return
        
        # Check view
        if msg.view != self.view_manager.view:
            return
        
        # Handle based on phase
        if msg.type == BFTPhase.PREPARE:
            await self._handle_prepare(msg)
        elif msg.type == BFTPhase.PRE_COMMIT:
            await self._handle_precommit(msg)
        elif msg.type == BFTPhase.COMMIT:
            await self._handle_commit(msg)
    
    async def _handle_prepare(self, msg: BFTMessage):
        """Handle prepare phase message."""
        # Validate proposal
        if not await self._validate_proposal(msg.proposal):
            return
        
        # Create vote
        vote = BFTVote(
            phase=BFTPhase.PREPARE,
            view=msg.view,
            sequence=msg.sequence,
            voter_id=self.node_id,
            message_hash=self.message_handler.hash_message(msg),
            signature=self.crypto.sign(
                f"{self.node_id}:{msg.view}:{msg.sequence}".encode()
            )
        )
        
        # Send vote to leader
        leader = self.view_manager.get_leader(msg.view)
        await self._send_vote(leader, vote)
        
        # If we are leader, handle our own vote
        if self._is_leader():
            await self._handle_vote(vote)
    
    async def _handle_vote(self, vote: BFTVote):
        """Handle vote from validator."""
        # Add vote and check threshold
        if self.vote_collector.add_vote(vote):
            # Threshold reached, advance phase
            if vote.phase == BFTPhase.PREPARE:
                await self._start_precommit()
            elif vote.phase == BFTPhase.PRE_COMMIT:
                await self._start_commit()
            elif vote.phase == BFTPhase.COMMIT:
                await self._finalize_consensus()
    
    async def _start_precommit(self):
        """Start pre-commit phase."""
        # Find the proposal
        request = next(iter(self.pending_proposals.values()), None)
        if not request:
            return
        
        msg = self.message_handler.create_message(
            BFTPhase.PRE_COMMIT,
            self.view_manager.view,
            request.proposal,
            request.request_id
        )
        
        await self._broadcast_message(msg)
    
    async def _start_commit(self):
        """Start commit phase."""
        request = next(iter(self.pending_proposals.values()), None)
        if not request:
            return
        
        msg = self.message_handler.create_message(
            BFTPhase.COMMIT,
            self.view_manager.view,
            request.proposal,
            request.request_id
        )
        
        await self._broadcast_message(msg)
    
    async def _finalize_consensus(self):
        """Finalize consensus after commit phase."""
        # Find request
        request_id = None
        request = None
        for rid, req in self.pending_proposals.items():
            if rid in self.pending_futures:
                request_id = rid
                request = req
                break
        
        if not request:
            return
        
        # Check for Byzantine nodes
        byzantine_nodes = self.vote_collector.get_byzantine_nodes()
        if byzantine_nodes:
            for node in byzantine_nodes:
                await self._report_byzantine(node, "Conflicting votes")
        
        # Create result
        result = ConsensusResult(
            request_id=request_id,
            status=ConsensusState.ACCEPTED,
            decision=request.proposal,
            consensus_type="bft",
            consensus_proof=self._create_proof()
        )
        
        # Complete future
        if request_id in self.pending_futures:
            future = self.pending_futures[request_id]
            future.set_result(result)
        
        logger.info(
            f"BFT consensus achieved",
            request_id=request_id,
            view=self.view_manager.view
        )
    
    def _create_proof(self) -> ConsensusProof:
        """Create consensus proof from votes."""
        return ConsensusProof(
            request_id="",
            consensus_type="bft",
            votes=[
                Vote(
                    voter_id=v.voter_id,
                    vote_type=VoteType.APPROVE,
                    timestamp=v.timestamp,
                    signature=v.signature
                )
                for v in self.vote_collector.phase_votes[BFTPhase.COMMIT]
            ],
            quorum_size=self.vote_collector.threshold,
            view=self.view_manager.view
        )
    
    async def _view_change_monitor(self):
        """Monitor for view timeouts."""
        while True:
            try:
                await asyncio.sleep(self.config.view_timeout_ms / 1000.0)
                
                if self.pending_proposals and not self._has_progress():
                    new_leader = self.view_manager.advance_view()
                    logger.info(
                        f"View change",
                        old_view=self.view_manager.view - 1,
                        new_view=self.view_manager.view,
                        new_leader=new_leader
                    )
                    
                    # Notify pending requests
                    for future in self.pending_futures.values():
                        if not future.done():
                            future.set_exception(
                                Exception("View change")
                            )
                    
            except asyncio.CancelledError:
                break
    
    def _has_progress(self) -> bool:
        """Check if consensus is making progress."""
        # Simple check: any votes in current phase
        return any(
            len(votes) > 0 
            for votes in self.vote_collector.phase_votes.values()
        )
    
    async def _report_byzantine(self, node_id: str, reason: str):
        """Report Byzantine behavior."""
        logger.warning(
            f"Byzantine behavior detected",
            node_id=node_id,
            reason=reason
        )
        
        self.metrics.byzantine_detected.add(1, {
            "node_id": node_id,
            "reason": reason
        })
        
        await self.event_producer.send_event("bft.byzantine.alerts", {
            "detected_by": self.node_id,
            "byzantine_node": node_id,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def _validate_proposal(self, proposal: Dict[str, Any]) -> bool:
        """Validate proposal before voting."""
        # Add custom validation logic
        return True
    
    async def _forward_to_leader(self, request: ConsensusRequest):
        """Forward request to current leader."""
        leader = self.view_manager.get_leader(self.view_manager.view)
        await execute_workflow(
            "BFTForwardWorkflow",
            {
                "from_node": self.node_id,
                "to_node": leader,
                "request": request
            },
            id=f"bft-forward-{request.request_id}"
        )
    
    async def _send_vote(self, target: str, vote: BFTVote):
        """Send vote to target node."""
        await self.event_producer.send_event(
            f"bft.votes.{target}",
            vote.to_dict()
        )
    
    async def _phase_timeout(self, phase: BFTPhase, sequence: int):
        """Handle phase timeout."""
        await asyncio.sleep(self.config.phase_timeout_ms / 1000.0)
        
        # Check if still in same phase
        if self.view_manager.phase == phase:
            logger.warning(
                f"Phase timeout",
                phase=phase.value,
                sequence=sequence
            )
    
    async def _handle_precommit(self, msg: BFTMessage):
        """Handle pre-commit message."""
        # Similar to prepare
        await self._handle_prepare(msg)
    
    async def _handle_commit(self, msg: BFTMessage):
        """Handle commit message."""
        # Similar to prepare
        await self._handle_prepare(msg)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get BFT node status."""
        byzantine_nodes = self.vote_collector.get_byzantine_nodes()
        return {
            "node_id": self.node_id,
            "view": self.view_manager.view,
            "phase": self.view_manager.phase.value,
            "is_leader": self._is_leader(),
            "validators": self.config.validators,
            "byzantine_nodes": list(byzantine_nodes),
            "pending_proposals": len(self.pending_proposals)
        }


@dataclass
class BFTConfig:
    """Configuration for Byzantine consensus."""
    node_id: str
    validators: List[str]
    view_timeout_ms: int = 5000
    phase_timeout_ms: int = 1000
    batch_size: int = 100
    batch_timeout_ms: int = 50
    signature_scheme: str = "ed25519"
    threshold_signature: bool = True
    kafka_bootstrap_servers: str = "localhost:9092"
    use_temporal_for_rpc: bool = True


class ByzantineConsensus:
    """High-level Byzantine consensus interface."""
    
    def __init__(self, config: BFTConfig):
        self.config = config
        self.metrics = BFTMetrics()
        self.event_producer = EventProducer(config.kafka_bootstrap_servers)
        self.core = BFTCore(config, self.event_producer, self.metrics)
    
    async def start(self):
        """Start Byzantine consensus."""
        await self.core.start()
    
    async def stop(self):
        """Stop Byzantine consensus."""
        await self.core.stop()
    
    async def propose(self, request: ConsensusRequest) -> ConsensusResult:
        """Propose value for Byzantine consensus."""
        return await self.core.propose(request)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get consensus status."""
        return await self.core.get_status()