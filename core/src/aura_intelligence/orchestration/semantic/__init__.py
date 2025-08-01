"""
🧠 Semantic Orchestration Module

2025 research-driven semantic orchestration with TDA integration.
Implements LangGraph StateGraph patterns, semantic routing, and 
intelligent agent coordination based on TDA context.

Key Components:
- LangGraph Semantic Orchestrator with StateGraph patterns
- Semantic Pattern Matcher with TDA correlation
- TDA Context Integration Layer
- Semantic Routing Engine with capability-based selection

Integration Points:
- TDA Event Mesh (Kafka) for agent communication
- TDA Feature Flags for progressive rollout
- TDA Tracing system for observability
- TDA Pattern Analysis for routing decisions
"""

from typing import Dict, Any, List, Optional

# Core semantic orchestration interfaces
from .base_interfaces import (
    AgentState, SemanticOrchestrator, TDAContext, TDAIntegration,
    SemanticAnalysis, OrchestrationStrategy, UrgencyLevel
)

from .langgraph_orchestrator import (
    LangGraphSemanticOrchestrator,
    SemanticWorkflowConfig,
    LANGGRAPH_AVAILABLE,
    POSTGRES_CHECKPOINTING_AVAILABLE
)

from .semantic_patterns import (
    SemanticPatternMatcher
)

from .tda_integration import (
    TDAContextIntegration,
    MockTDAIntegration
)

from .semantic_router import (
    SemanticRouter,
    AgentProfile,
    AgentCapability,
    RoutingDecision
)

# Feature flag integration with TDA system
try:
    from aura_common.feature_flags.manager import FeatureFlagManager
    feature_flags = FeatureFlagManager()
    SEMANTIC_ORCHESTRATION_ENABLED = feature_flags.is_enabled("semantic_orchestration")
except ImportError:
    SEMANTIC_ORCHESTRATION_ENABLED = False

# TDA integration check
try:
    from aura_intelligence.tda.streaming import get_current_patterns
    TDA_INTEGRATION_AVAILABLE = True
except ImportError:
    TDA_INTEGRATION_AVAILABLE = False

__all__ = [
    # Base interfaces
    "AgentState", "SemanticOrchestrator", "TDAContext", "TDAIntegration",
    "SemanticAnalysis", "OrchestrationStrategy", "UrgencyLevel",
    
    # Core orchestration
    "LangGraphSemanticOrchestrator",
    "SemanticWorkflowConfig",
    
    # Feature flags
    "LANGGRAPH_AVAILABLE",
    "POSTGRES_CHECKPOINTING_AVAILABLE",
    
    # Pattern matching
    "SemanticPatternMatcher",
    
    # TDA integration
    "TDAContextIntegration",
    "MockTDAIntegration",
    
    # Routing
    "SemanticRouter", 
    "AgentProfile",
    "AgentCapability",
    "RoutingDecision",
    
    # Feature flags
    "SEMANTIC_ORCHESTRATION_ENABLED",
    "TDA_INTEGRATION_AVAILABLE"
]

# Module version for compatibility tracking
__version__ = "1.0.0"