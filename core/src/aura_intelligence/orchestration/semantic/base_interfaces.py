"""
🏗️ Base Interfaces for Semantic Orchestration

Defines core interfaces and data structures for semantic orchestration
with TDA integration. All implementations must adhere to these contracts
for consistency and testability.

Design Principles:
- <150 lines per module
- Single responsibility
- TDA integration by design
- Comprehensive type hints
- Abstract base classes for testability
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# TDA integration imports
try:
    from aura_intelligence.observability.tracing import get_tracer
    tracer = get_tracer(__name__)
except ImportError:
    tracer = None

class OrchestrationStrategy(Enum):
    """Orchestration strategies based on semantic analysis"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel" 
    HIERARCHICAL = "hierarchical"
    CONSENSUS = "consensus"
    EVENT_DRIVEN = "event_driven"

class UrgencyLevel(Enum):
    """Urgency levels for orchestration prioritization"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class TDAContext:
    """TDA context for semantic orchestration"""
    correlation_id: str
    pattern_confidence: float
    anomaly_severity: float
    current_patterns: Dict[str, Any]
    temporal_window: str
    metadata: Dict[str, Any]

@dataclass
class SemanticAnalysis:
    """Results of semantic analysis"""
    complexity_score: float
    urgency_level: UrgencyLevel
    coordination_pattern: OrchestrationStrategy
    suggested_agents: List[str]
    confidence: float
    tda_correlation: Optional[TDAContext]

class AgentState(TypedDict):
    """LangGraph-compatible agent state with TDA integration"""
    messages: Annotated[List[Dict[str, Any]], "Conversation messages"]
    context: Dict[str, Any]
    agent_outputs: Dict[str, Any]
    workflow_metadata: Dict[str, Any]
    execution_trace: List[Dict[str, Any]]
    tda_context: Optional[TDAContext]

class SemanticOrchestrator(ABC):
    """Base interface for semantic orchestrators"""
    
    @abstractmethod
    async def analyze_semantically(
        self, 
        input_data: Dict[str, Any],
        tda_context: Optional[TDAContext] = None
    ) -> SemanticAnalysis:
        """Perform semantic analysis with TDA correlation"""
        pass
    
    @abstractmethod
    async def route_to_agents(
        self, 
        analysis: SemanticAnalysis,
        available_agents: List[str]
    ) -> List[str]:
        """Route to optimal agents based on semantic analysis"""
        pass
    
    @abstractmethod
    async def execute_orchestration(
        self,
        strategy: OrchestrationStrategy,
        agents: List[str],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute orchestration with specified strategy"""
        pass

class TDAIntegration(ABC):
    """Base interface for TDA integration"""
    
    @abstractmethod
    async def get_context(self, correlation_id: str) -> Optional[TDAContext]:
        """Get TDA context for correlation ID"""
        pass
    
    @abstractmethod
    async def send_orchestration_result(
        self, 
        result: Dict[str, Any],
        correlation_id: str
    ) -> bool:
        """Send orchestration result to TDA for pattern analysis"""
        pass
    
    @abstractmethod
    async def get_current_patterns(self, window: str = "1h") -> Dict[str, Any]:
        """Get current TDA patterns for context"""
        pass