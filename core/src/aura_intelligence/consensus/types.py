"""
Core types and data structures for consensus protocols.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, Field, validator
import uuid

from ..agents.base import AgentState


class DecisionType(str, Enum):
    """Types of decisions requiring consensus."""
    OPERATIONAL = "operational"    # Fast, low-stakes (e.g., task assignment)
    TACTICAL = "tactical"          # Medium speed (e.g., resource allocation)
    STRATEGIC = "strategic"        # Slow, high-stakes (e.g., model updates)
    EMERGENCY = "emergency"        # Fast, critical (e.g., safety shutdown)


class VoteType(str, Enum):
    """Types of votes in consensus."""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


class RaftState(str, Enum):
    """Raft node states."""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


class BFTPhase(str, Enum):
    """Byzantine consensus phases."""
    PREPARE = "prepare"
    PRE_COMMIT = "pre_commit"
    COMMIT = "commit"
    VIEW_CHANGE = "view_change"


class ConsensusState(str, Enum):
    """Overall consensus state."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class Vote:
    """Individual vote in consensus."""
    voter_id: str
    vote_type: VoteType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    signature: Optional[str] = None
    reason: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "voter_id": self.voter_id,
            "vote_type": self.vote_type.value,
            "timestamp": self.timestamp.isoformat(),
            "signature": self.signature,
            "reason": self.reason,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


@dataclass
class ConsensusRequest:
    """Request for consensus decision."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    decision_type: DecisionType = DecisionType.OPERATIONAL
    proposal: Dict[str, Any] = field(default_factory=dict)
    proposer_id: str = ""
    deadline: Optional[datetime] = None
    priority: int = 0  # Higher is more urgent
    
    # Consensus parameters
    quorum_size: Optional[int] = None
    validators: Optional[List[str]] = None
    timeout: timedelta = timedelta(seconds=30)
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    parent_request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    # Validation
    requires_explanation: bool = True
    requires_unanimous: bool = False
    allow_abstention: bool = True
    
    def __post_init__(self):
        """Set defaults after initialization."""
        if self.deadline is None:
            self.deadline = datetime.now(timezone.utc) + self.timeout
        if self.correlation_id is None:
            self.correlation_id = self.request_id


@dataclass
class ConsensusProof:
    """Proof of consensus achievement."""
    request_id: str
    consensus_type: str  # "raft", "bft", "multi_raft"
    votes: List[Vote]
    quorum_size: int
    achieved_at: datetime = field(default_factory=datetime.utcnow)
    
    # Protocol-specific proof
    term: Optional[int] = None  # For Raft
    view: Optional[int] = None  # For BFT
    phase_proofs: Optional[Dict[str, List[Vote]]] = None  # For BFT phases
    
    # Cryptographic proof
    merkle_root: Optional[str] = None
    aggregate_signature: Optional[str] = None
    
    def is_valid(self) -> bool:
        """Check if proof is valid."""
        # Count valid votes
        valid_votes = [v for v in self.votes if v.vote_type == VoteType.APPROVE]
        return len(valid_votes) >= self.quorum_size


@dataclass
class CausalNode:
    """Node in causal reasoning path."""
    node_id: str
    description: str
    node_type: str  # "cause", "effect", "mediator", "confounder"
    probability: float
    evidence: List[str] = field(default_factory=list)


@dataclass
class DecisionExplanation:
    """Explanation for consensus decision."""
    decision_id: str
    explanation: str
    confidence: float
    
    # Reasoning components
    causal_path: List[CausalNode] = field(default_factory=list)
    supporting_evidence: List[str] = field(default_factory=list)
    counter_evidence: List[str] = field(default_factory=list)
    
    # Dissent analysis
    dissenting_opinions: List[Vote] = field(default_factory=list)
    dissent_summary: Optional[str] = None
    
    # Formal proof (if available)
    symbolic_proof: Optional[str] = None
    proof_steps: List[str] = field(default_factory=list)
    
    # Recommendations
    alternative_decisions: List[Dict[str, Any]] = field(default_factory=list)
    risk_assessment: Optional[Dict[str, float]] = None
    
    def to_natural_language(self) -> str:
        """Convert explanation to natural language."""
        parts = [self.explanation]
        
        if self.causal_path:
            parts.append("\nCausal reasoning:")
            for node in self.causal_path:
                parts.append(f"  - {node.description} ({node.node_type}, p={node.probability:.2f})")
        
        if self.dissenting_opinions:
            parts.append(f"\nDissenting opinions: {len(self.dissenting_opinions)}")
            if self.dissent_summary:
                parts.append(f"  Summary: {self.dissent_summary}")
        
        if self.risk_assessment:
            parts.append("\nRisk assessment:")
            for risk, prob in self.risk_assessment.items():
                parts.append(f"  - {risk}: {prob:.2%}")
        
        return "\n".join(parts)


@dataclass
class ConsensusResult:
    """Result of consensus process."""
    request_id: str
    status: ConsensusState
    decision: Optional[Dict[str, Any]] = None
    
    # Consensus details
    consensus_proof: Optional[ConsensusProof] = None
    votes: List[Vote] = field(default_factory=list)
    participation_rate: float = 0.0
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    
    # Explanation
    explanation: Optional[DecisionExplanation] = None
    reason: Optional[str] = None
    
    # Metadata
    consensus_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived fields."""
        if self.started_at and self.completed_at and not self.duration_ms:
            self.duration_ms = (self.completed_at - self.started_at).total_seconds() * 1000
        
        if self.votes and self.consensus_proof:
            total_validators = len(self.consensus_proof.votes)
            if total_validators > 0:
                self.participation_rate = len(self.votes) / total_validators
    
    def is_successful(self) -> bool:
        """Check if consensus was successful."""
        return self.status in [ConsensusState.ACCEPTED, ConsensusState.REJECTED]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "decision": self.decision,
            "votes": [v.to_dict() for v in self.votes],
            "participation_rate": self.participation_rate,
            "duration_ms": self.duration_ms,
            "reason": self.reason,
            "consensus_type": self.consensus_type,
            "metadata": self.metadata
        }


# Raft-specific types

@dataclass
class LogEntry:
    """Entry in Raft log."""
    term: int
    index: int
    command: Dict[str, Any]
    request_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    committed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "term": self.term,
            "index": self.index,
            "command": self.command,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "committed": self.committed
        }


@dataclass
class RaftVoteRequest:
    """Request for vote in Raft election."""
    term: int
    candidate_id: str
    last_log_index: int
    last_log_term: int
    
    
@dataclass
class RaftVoteResponse:
    """Response to Raft vote request."""
    term: int
    vote_granted: bool
    voter_id: str
    reason: Optional[str] = None


@dataclass
class AppendEntriesRequest:
    """Raft append entries RPC."""
    term: int
    leader_id: str
    prev_log_index: int
    prev_log_term: int
    entries: List[LogEntry]
    leader_commit: int


@dataclass
class AppendEntriesResponse:
    """Response to append entries."""
    term: int
    success: bool
    follower_id: str
    match_index: Optional[int] = None
    reason: Optional[str] = None


# Byzantine-specific types

@dataclass
class BFTMessage:
    """Message in Byzantine consensus."""
    type: BFTPhase
    view: int
    sequence: int
    node_id: str
    proposal: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    signature: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "view": self.view,
            "sequence": self.sequence,
            "node_id": self.node_id,
            "proposal": self.proposal,
            "request_id": self.request_id,
            "signature": self.signature,
            "timestamp": self.timestamp.isoformat()
        }
    
    def sign(self, private_key: Any) -> None:
        """Sign the message."""
        # In production, use proper cryptographic signing
        import hashlib
        message_bytes = str(self.to_dict()).encode()
        self.signature = hashlib.sha256(message_bytes).hexdigest()


@dataclass
class BFTVote:
    """Vote in Byzantine consensus."""
    phase: BFTPhase
    view: int
    sequence: int
    voter_id: str
    message_hash: str
    signature: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def verify(self, public_key: Any) -> bool:
        """Verify vote signature."""
        # In production, use proper cryptographic verification
        return True  # Placeholder


@dataclass
class BFTProof:
    """Proof of Byzantine consensus."""
    view: int
    sequence: int
    phase: BFTPhase
    votes: List[BFTVote]
    threshold: int
    message: BFTMessage
    
    def is_valid(self) -> bool:
        """Check if proof has enough valid votes."""
        valid_votes = [v for v in self.votes if v.verify(None)]
        return len(valid_votes) >= self.threshold


# Validation types

class ValidationResult(BaseModel):
    """Result of consensus validation."""
    is_valid: bool
    confidence: float = Field(ge=0.0, le=1.0)
    
    # Validation components
    neural_score: float = Field(ge=0.0, le=1.0)
    symbolic_valid: bool = True
    causal_valid: bool = True
    
    # Reasons
    reason: str = ""
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    
    # Evidence
    supporting_rules: List[str] = Field(default_factory=list)
    violated_rules: List[str] = Field(default_factory=list)
    
    @validator('confidence')
    def calculate_confidence(cls, v, values):
        """Calculate overall confidence."""
        if 'neural_score' in values:
            # Weighted average of validation components
            weights = {
                'neural': 0.4,
                'symbolic': 0.4,
                'causal': 0.2
            }
            
            score = values['neural_score'] * weights['neural']
            if values.get('symbolic_valid', True):
                score += weights['symbolic']
            if values.get('causal_valid', True):
                score += weights['causal']
                
            return min(score, 1.0)
        return v


# Configuration types

@dataclass
class ConsensusConfig:
    """Configuration for consensus manager."""
    # Decision routing
    operational_timeout: timedelta = timedelta(milliseconds=100)
    tactical_timeout: timedelta = timedelta(milliseconds=500)
    strategic_timeout: timedelta = timedelta(seconds=2)
    
    # Quorum sizes
    operational_quorum: int = 3
    tactical_quorum: int = 5
    strategic_quorum: int = 7
    
    # Protocol selection
    use_bft_for_strategic: bool = True
    use_multi_raft_for_tactical: bool = True
    
    # Validation
    require_explanation: bool = True
    neural_confidence_threshold: float = 0.8
    
    # Integration
    temporal_namespace: str = "default"
    kafka_bootstrap_servers: str = "localhost:9092"
    redis_url: str = "redis://localhost:6379"