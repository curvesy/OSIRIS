"""
🚀 Distributed Orchestration Module

2025 Ray Serve and CrewAI Flows integration for distributed AI agent orchestration.
Provides enterprise-scale distributed inference, auto-scaling, and hierarchical
agent coordination with TDA intelligence integration.

Key Components:
- Ray Serve agent ensemble deployments
- CrewAI Flows hierarchical orchestration
- Distributed coordination layer
- TDA-aware load balancing and routing

TDA Integration:
- TDA context for intelligent agent selection
- Load balancing based on TDA patterns
- Distributed performance correlation
- Cross-service TDA event propagation
"""

from .ray_orchestrator import (
    RayServeOrchestrator,
    AgentEnsembleDeployment,
    TDALoadBalancer,
    DistributedAgentConfig
)

from .crewai_orchestrator import (
    CrewAIFlowOrchestrator,
    AuraDistributedFlow,
    HierarchicalFlowConfig,
    FlowExecutionResult
)

from .distributed_coordinator import (
    DistributedCoordinator,
    DistributedExecutionPlan,
    CrossServiceCheckpoint,
    DistributedRecoveryStrategy,
    DistributedExecutionMode
)

# Feature flags
RAY_SERVE_AVAILABLE = True
CREWAI_FLOWS_AVAILABLE = True
DISTRIBUTED_ORCHESTRATION_ENABLED = True

try:
    import ray
    from ray import serve
except ImportError:
    RAY_SERVE_AVAILABLE = False
    DISTRIBUTED_ORCHESTRATION_ENABLED = False

try:
    from crewai_flows import Flow
except ImportError:
    CREWAI_FLOWS_AVAILABLE = False

__all__ = [
    # Ray Serve orchestration
    "RayServeOrchestrator",
    "AgentEnsembleDeployment",
    "TDALoadBalancer",
    "DistributedAgentConfig",
    
    # CrewAI Flows orchestration
    "CrewAIFlowOrchestrator",
    "AuraDistributedFlow",
    "HierarchicalFlowConfig",
    "FlowExecutionResult",
    
    # Distributed coordination
    "DistributedCoordinator",
    "DistributedExecutionPlan",
    "CrossServiceCheckpoint",
    "DistributedRecoveryStrategy",
    "DistributedExecutionMode",
    
    # Feature flags
    "RAY_SERVE_AVAILABLE",
    "CREWAI_FLOWS_AVAILABLE",
    "DISTRIBUTED_ORCHESTRATION_ENABLED"
]