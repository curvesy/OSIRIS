"""
LNN Council Agent - Modular Architecture

A truly modular, production-grade implementation following 2025 best practices.
Separates concerns into distinct, composable modules with clean interfaces.
"""

from .contracts import (
    CouncilRequest,
    CouncilResponse,
    VoteDecision,
    VoteConfidence,
    DecisionEvidence,
    AgentCapability,
    ContextScope
)

from .interfaces import (
    ICouncilAgent,
    INeuralEngine,
    IContextProvider,
    IDecisionMaker,
    IEvidenceCollector,
    IReasoningEngine
)

from .agent import LNNCouncilAgent
from .factory import CouncilAgentFactory
from .registry import AgentRegistry, get_global_registry

__all__ = [
    # Contracts
    "CouncilRequest",
    "CouncilResponse",
    "VoteDecision",
    "VoteConfidence",
    "DecisionEvidence",
    "AgentCapability",
    "ContextScope",
    
    # Interfaces
    "ICouncilAgent",
    "INeuralEngine",
    "IContextProvider",
    "IDecisionMaker",
    "IEvidenceCollector",
    "IReasoningEngine",
    
    # Implementation
    "LNNCouncilAgent",
    "CouncilAgentFactory",
    "AgentRegistry",
    "get_global_registry"
]