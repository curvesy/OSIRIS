"""
⏱️ Durable Orchestration Module

2025 Temporal.io integration for fault-tolerant, durable workflows with
saga patterns and distributed transaction handling. Provides enterprise-grade
reliability for multi-agent orchestration.

Key Components:
- Temporal.io workflow definitions
- Saga pattern compensation logic
- TDA-aware checkpointing
- Automatic retry and recovery

TDA Integration:
- TDA context for workflow planning
- Anomaly-aware compensation
- Pattern-based workflow adaptation
- Performance correlation tracking
"""

from .temporal_orchestrator import (
    TemporalDurableOrchestrator,
    DurableWorkflowConfig,
    WorkflowExecutionResult,
    CompensationAction
)

from .saga_patterns import (
    SagaOrchestrator,
    SagaStep,
    CompensationHandler
)

from .checkpoint_manager import (
    CheckpointManager,
    WorkflowCheckpoint,
    RecoveryStrategy
)

from .hybrid_checkpointer import (
    HybridCheckpointManager,
    HybridCheckpointConfig,
    HybridCheckpointResult,
    CheckpointLevel,
    RecoveryMode
)

# Feature flags
TEMPORAL_AVAILABLE = True
DURABLE_ORCHESTRATION_ENABLED = True

try:
    import temporalio
except ImportError:
    TEMPORAL_AVAILABLE = False
    DURABLE_ORCHESTRATION_ENABLED = False

__all__ = [
    # Core durable orchestration
    "TemporalDurableOrchestrator",
    "DurableWorkflowConfig", 
    "WorkflowExecutionResult",
    "CompensationAction",
    
    # Saga patterns
    "SagaOrchestrator",
    "SagaStep",
    "CompensationHandler",
    
    # Checkpoint management
    "CheckpointManager",
    "WorkflowCheckpoint",
    "RecoveryStrategy",
    
    # Hybrid checkpointing
    "HybridCheckpointManager",
    "HybridCheckpointConfig", 
    "HybridCheckpointResult",
    "CheckpointLevel",
    "RecoveryMode",
    
    # Feature flags
    "TEMPORAL_AVAILABLE",
    "DURABLE_ORCHESTRATION_ENABLED"
]