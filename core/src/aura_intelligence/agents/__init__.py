"""
🤖 AURA Intelligence Agent SDK - Phase 2D: The Collective

Advanced multi-agent system built on the proven Hot→Cold→Wise Intelligence Flywheel.
Combines cutting-edge agent orchestration with operational reliability.

Key Components:
- UnifiedMemory: LlamaIndex fusion over existing memory tiers
- ACP Protocol: Formal agent-to-agent communication
- Agent Base Classes: Observer, Analyst, Executor, Coordinator
- OpenTelemetry Integration: Full observability
- LangGraph Orchestration: State-of-the-art workflows

Production-grade multi-agent implementation using LangGraph
with comprehensive observability and resilience patterns.
"""

# Phase 2 new implementations
from .base import AgentBase, AgentConfig, AgentState
from .observability import AgentInstrumentor, AgentMetrics

# Original schemas (to be implemented/migrated)
from .schemas.acp import ACPEnvelope, ACPEndpoint, MessageType, Priority
from .schemas.state import AgentState as LegacyAgentState, DossierEntry, TaskStatus
from .schemas.log import AgentActionEvent, ActionResult

# Memory & Communication (to be implemented/migrated)
from .memory.unified import UnifiedMemory, MemoryTier, QueryResult
from .communication.protocol import ACPProtocol, MessageBus
from .communication.transport import RedisStreamsTransport

# Base Classes (to be migrated to new base)
from .base_classes.agent import BaseAgent, AgentRole, AgentCapability
from .base_classes.instrumentation import instrument_agent, AgentMetrics as LegacyAgentMetrics

# TODO: Fix missing orchestration module
# from .orchestration.workflow import WorkflowEngine, WorkflowState
# from .orchestration.langgraph import LangGraphOrchestrator

# TODO: Fix missing core agent implementations
# from .core.observer import ObserverAgent
# from .core.analyst import AnalystAgent  
# from .core.executor import ExecutorAgent
# from .core.coordinator import CoordinatorAgent

# TODO: Fix missing advanced agents
# from .advanced.router import RouterAgent
# from .advanced.consensus import ConsensusAgent
# from .advanced.supervisor import SupervisorAgent

__version__ = "2.0.0"
__author__ = "AURA Intelligence Team"

# Export main classes for easy import
__all__ = [
    # Phase 2 New Base
    "AgentBase",
    "AgentConfig", 
    "AgentState",
    "AgentInstrumentor",
    "AgentMetrics",
    
    # Schemas (Legacy - to be migrated)
    "ACPEnvelope", "ACPEndpoint", "MessageType", "Priority",
    "LegacyAgentState", "DossierEntry", "TaskStatus", 
    "AgentActionEvent", "ActionResult",
    
    # Memory & Communication (to be implemented)
    "UnifiedMemory", "MemoryTier", "QueryResult",
    "ACPProtocol", "MessageBus", "RedisStreamsTransport",
    
    # Base Classes (Legacy - to be migrated)
    "BaseAgent", "AgentRole", "AgentCapability",
    "instrument_agent", "LegacyAgentMetrics",
    
    # Orchestration (to be integrated)
    "WorkflowEngine", "WorkflowState", "LangGraphOrchestrator",
    
    # Core Agents (to be reimplemented)
    "ObserverAgent", "AnalystAgent", "ExecutorAgent", "CoordinatorAgent",
    
    # Advanced Agents (future)
    "RouterAgent", "ConsensusAgent", "SupervisorAgent"
]
