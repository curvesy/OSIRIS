"""
🎼 Workflow State Management
Clean state definitions for LangGraph workflows.
"""

from typing import Dict, Any, List, Annotated, Sequence, Optional
from typing_extensions import TypedDict
from datetime import datetime, timezone

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from aura_intelligence.config import AURASettings


class CollectiveState(TypedDict):
    """
    Advanced state for collective intelligence workflows.
    
    Uses LangGraph TypedDict patterns for type safety and
    state management across workflow nodes.
    """
    # Core messaging
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Workflow identification
    workflow_id: str
    thread_id: str
    
    # Evidence and decisions
    evidence_log: List[Dict[str, Any]]
    supervisor_decisions: List[Dict[str, Any]]
    
    # Context and configuration
    memory_context: Dict[str, Any]
    active_config: Dict[str, Any]
    
    # Workflow progress
    current_step: str
    risk_assessment: Optional[Dict[str, Any]]
    execution_results: List[Dict[str, Any]]
    
    # Error handling
    error_log: List[Dict[str, Any]]
    error_recovery_attempts: int
    last_error: Optional[Dict[str, Any]]
    
    # System health
    system_health: Dict[str, Any]
    validation_results: Optional[Dict[str, Any]]
    
    # Shadow mode (A/B testing)
    shadow_mode_enabled: bool
    shadow_predictions: List[Dict[str, Any]]


class WorkflowMetadata(BaseModel):
    """Metadata for workflow execution."""
    
    workflow_id: str = Field(..., description="Unique workflow identifier")
    thread_id: str = Field(..., description="Thread/conversation ID")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Correlation for distributed tracing
    correlation_id: Optional[str] = Field(None, description="Request correlation ID")
    trace_id: Optional[str] = Field(None, description="OpenTelemetry trace ID")
    
    # Feature flags
    features: Dict[str, bool] = Field(
        default_factory=dict,
        description="Active feature flags for this workflow"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class NodeResult(BaseModel):
    """Standard result from workflow nodes."""
    
    success: bool = Field(..., description="Whether node execution succeeded")
    node_name: str = Field(..., description="Name of the executing node")
    
    # Results
    output: Optional[Dict[str, Any]] = Field(None, description="Node output data")
    messages: List[BaseMessage] = Field(default_factory=list)
    
    # Metrics
    duration_ms: float = Field(0.0, description="Execution time in milliseconds")
    
    # Error info
    error: Optional[str] = Field(None, description="Error message if failed")
    error_type: Optional[str] = Field(None, description="Error classification")
    
    # Next steps
    next_node: Optional[str] = Field(None, description="Suggested next node")
    routing_metadata: Dict[str, Any] = Field(default_factory=dict)


def create_initial_state(
    workflow_id: str,
    thread_id: str,
    initial_message: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> CollectiveState:
    """
    Create initial workflow state with defaults.
    
    Args:
        workflow_id: Unique workflow identifier
        thread_id: Thread/conversation ID
        initial_message: Optional initial human message
        config: Optional configuration overrides
        
    Returns:
        Initialized CollectiveState
    """
    # Get system config
    system_config = AURASettings()
    
    # Build initial state
    state: CollectiveState = {
        "messages": [],
        "workflow_id": workflow_id,
        "thread_id": thread_id,
        "evidence_log": [],
        "supervisor_decisions": [],
        "memory_context": {},
        "active_config": config or {},
        "current_step": "initialization",
        "risk_assessment": None,
        "execution_results": [],
        "error_log": [],
        "error_recovery_attempts": 0,
        "last_error": None,
        "system_health": {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        "validation_results": None,
        "shadow_mode_enabled": system_config.logging.shadow_mode_enabled,
        "shadow_predictions": []
    }
    
    # Add initial message if provided
    if initial_message:
        from langchain_core.messages import HumanMessage
        state["messages"] = [HumanMessage(content=initial_message)]
    
    return state


def update_state_safely(
    current_state: CollectiveState,
    updates: Dict[str, Any]
) -> CollectiveState:
    """
    Safely update workflow state (immutability pattern).
    
    Args:
        current_state: Current state
        updates: Updates to apply
        
    Returns:
        New state with updates applied
    """
    # Create new state dict
    new_state = dict(current_state)
    
    # Apply updates
    for key, value in updates.items():
        if key in new_state:
            # Special handling for lists (append, don't replace)
            if isinstance(new_state[key], list) and isinstance(value, list):
                new_state[key] = new_state[key] + value
            else:
                new_state[key] = value
    
    # Update timestamp
    if "system_health" in new_state:
        new_state["system_health"]["last_updated"] = datetime.now(timezone.utc).isoformat()
    
    return CollectiveState(**new_state)  # type: ignore