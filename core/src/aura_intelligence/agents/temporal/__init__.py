"""
AURA Intelligence Temporal Integration

Durable workflow orchestration with Temporal for reliable, 
stateful agent execution at scale.

Key Features:
- Durable execution with automatic retries
- Stateful workflows with versioning
- Event sourcing and replay
- Distributed saga patterns
- Long-running agent orchestration
"""

from .workflows import (
    AgentWorkflow,
    MultiAgentOrchestrationWorkflow,
    ResearchAnalysisPipeline,
    ConsensusWorkflow
)

from .activities import (
    AgentActivity,
    KafkaProducerActivity,
    StateManagementActivity,
    ObservabilityActivity
)

from .worker import (
    TemporalWorker,
    WorkerConfig,
    create_worker
)

from .client import (
    TemporalClient,
    WorkflowHandle,
    execute_workflow
)

__all__ = [
    # Workflows
    "AgentWorkflow",
    "MultiAgentOrchestrationWorkflow",
    "ResearchAnalysisPipeline",
    "ConsensusWorkflow",
    
    # Activities
    "AgentActivity",
    "KafkaProducerActivity",
    "StateManagementActivity",
    "ObservabilityActivity",
    
    # Worker
    "TemporalWorker",
    "WorkerConfig",
    "create_worker",
    
    # Client
    "TemporalClient",
    "WorkflowHandle",
    "execute_workflow"
]