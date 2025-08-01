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
    RaftConfig,
    RaftNode,
    LogEntry,
    RaftElectionWorkflow,
    RaftReplicationWorkflow
)

from .byzantine import (
    ByzantineConsensus,
    BFTConfig,
    BFTMessage,
    HotStuffConsensus,
    BFTProof
)

from .multi_raft import (
    MultiRaftConsensus,
    MultiRaftConfig,
    RaftGroup,
    CrossGroupCoordinator
)

from .validation import (
    NeuroSymbolicValidator,
    ValidatorConfig,
    ValidationResult,
    CausalInferenceEngine,
    SymbolicReasoner
)

from .workflows import (
    ConsensusWorkflow,
    ConsensusWorkflowInput,
    ConsensusVotingWorkflow,
    BFTConsensusWorkflow
)

from .events import (
    ConsensusProposalEvent,
    ConsensusVoteEvent,
    ConsensusDecisionEvent,
    ConsensusStreamProcessor
)

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