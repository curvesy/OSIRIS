"""
AURA Intelligence Consensus Protocols

Advanced consensus mechanisms for distributed decision-making:
- Raft consensus for operational decisions
- Byzantine fault tolerance for critical decisions
- Multi-Raft for scalability
- Neuro-symbolic validation for explainability

Key Features:
- Hierarchical consensus based on decision criticality
- Integration with Temporal workflows and Kafka event mesh
- Explainable decisions with causal reasoning
- Self-healing and Byzantine fault detection
"""

from .types import (
    DecisionType,
    ConsensusRequest,
    ConsensusResult,
    ConsensusState,
    Vote,
    VoteType,
    RaftState,
    BFTPhase,
    ConsensusProof,
    DecisionExplanation
)

from .manager import (
    ConsensusManager,
    ConsensusConfig,
    HierarchicalConsensus
)

from .raft import (
    RaftConsensus,
    RaftConfig
    # RaftNode, RaftElectionWorkflow, RaftReplicationWorkflow - not implemented yet
)
# LogEntry is imported from types above

from .byzantine import (
    ByzantineConsensus,
    BFTConfig
    # BFTMessage, HotStuffConsensus, BFTProof - not implemented yet
)

# from .multi_raft import (
#     MultiRaftConsensus,
#     MultiRaftConfig,
#     RaftGroup,
#     CrossGroupCoordinator
# ) - module not implemented yet

# from .validation import (
#     NeuroSymbolicValidator,
#     ValidatorConfig,
#     ValidationResult,
#     CausalInferenceEngine,
#     SymbolicReasoner
# ) - module not implemented yet

from .workflows import (
    ConsensusWorkflow,
    ConsensusWorkflowInput,
    ConsensusVotingWorkflow,
    BFTConsensusWorkflow
)

# from .events import (
#     ConsensusProposalEvent,
#     ConsensusVoteEvent,
#     ConsensusDecisionEvent,
#     ConsensusStreamProcessor
# ) - module not implemented yet

# Temporary stubs for missing classes
class SimpleConsensus:
    """Stub for SimpleConsensus - to be implemented."""
    def __init__(self, node_id=None, peers=None, kafka_servers=None, **kwargs):
        self.node_id = node_id
        self.peers = peers or []
        self.kafka_servers = kafka_servers
        # Accept any other parameters for compatibility
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    async def decide(self, request):
        return {"decision": "approved", "confidence": 0.8}

class Decision:
    """Stub for Decision - to be implemented."""
    def __init__(self, decision_id, result, confidence=1.0):
        self.decision_id = decision_id
        self.result = result
        self.confidence = confidence

__version__ = "1.0.0"

__all__ = [
    # Types
    "DecisionType",
    "ConsensusRequest",
    "ConsensusResult",
    "ConsensusState",
    "Vote",
    "VoteType",
    "RaftState",
    "BFTPhase",
    "ConsensusProof",
    "DecisionExplanation",
    
    # Manager
    "ConsensusManager",
    "ConsensusConfig",
    "HierarchicalConsensus",
    
    # Raft
    "RaftConsensus",
    "RaftConfig",
    "RaftNode",
    "LogEntry",
    "RaftElectionWorkflow",
    "RaftReplicationWorkflow",
    
    # Byzantine
    "ByzantineConsensus",
    "BFTConfig",
    "BFTMessage",
    "HotStuffConsensus",
    "BFTProof",
    
    # Multi-Raft
    "MultiRaftConsensus",
    "MultiRaftConfig",
    "RaftGroup",
    "CrossGroupCoordinator",
    
    # Validation
    "NeuroSymbolicValidator",
    "ValidatorConfig",
    "ValidationResult",
    "CausalInferenceEngine",
    "SymbolicReasoner",
    
    # Workflows
    "ConsensusWorkflow",
    "ConsensusWorkflowInput",
    "ConsensusVotingWorkflow",
    "BFTConsensusWorkflow",
    
    # Events
    "ConsensusProposalEvent",
    "ConsensusVoteEvent",
    "ConsensusDecisionEvent",
    "ConsensusStreamProcessor"
]