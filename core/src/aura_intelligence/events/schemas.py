"""
Event Schemas for AURA Intelligence Event Mesh

Defines all event types with:
- Avro schema definitions
- Pydantic models for validation
- Schema evolution support
- Backward/forward compatibility
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from dataclasses import dataclass, field
import json
import uuid

from pydantic import BaseModel, Field, validator
import structlog

logger = structlog.get_logger()


class EventType(str, Enum):
    """Event type enumeration."""
    # Agent events
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"
    AGENT_STATE_CHANGED = "agent.state.changed"
    AGENT_DECISION_MADE = "agent.decision.made"
    
    # Workflow events
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"
    WORKFLOW_STATE_CHANGED = "workflow.state.changed"
    WORKFLOW_STEP_COMPLETED = "workflow.step.completed"
    
    # System events
    SYSTEM_HEALTH_CHECK = "system.health.check"
    SYSTEM_ALERT = "system.alert"
    SYSTEM_METRIC = "system.metric"
    SYSTEM_CONFIG_CHANGED = "system.config.changed"
    
    # Coordination events
    CONSENSUS_REQUESTED = "consensus.requested"
    CONSENSUS_ACHIEVED = "consensus.achieved"
    CONSENSUS_FAILED = "consensus.failed"
    
    # Stream processing events
    STREAM_CHECKPOINT = "stream.checkpoint"
    STREAM_REBALANCE = "stream.rebalance"
    STREAM_ERROR = "stream.error"


class EventPriority(str, Enum):
    """Event priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class EventSchema(BaseModel):
    """Base event schema with common fields."""
    
    # Event metadata
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    event_version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Source information
    source_id: str
    source_type: str
    source_version: Optional[str] = None
    
    # Correlation and tracing
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    
    # Event properties
    priority: EventPriority = EventPriority.NORMAL
    partition_key: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    
    # Payload
    data: Dict[str, Any]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @validator('partition_key', always=True)
    def set_partition_key(cls, v, values):
        """Set partition key if not provided."""
        if v is None:
            # Use source_id as default partition key for ordering
            return values.get('source_id', str(uuid.uuid4()))
        return v
    
    def to_avro_schema(self) -> Dict[str, Any]:
        """Convert to Avro schema definition."""
        return {
            "type": "record",
            "name": self.__class__.__name__,
            "namespace": "com.aura.intelligence.events",
            "fields": [
                {"name": "event_id", "type": "string"},
                {"name": "event_type", "type": "string"},
                {"name": "event_version", "type": "string"},
                {"name": "timestamp", "type": "string"},
                {"name": "source_id", "type": "string"},
                {"name": "source_type", "type": "string"},
                {"name": "source_version", "type": ["null", "string"], "default": None},
                {"name": "correlation_id", "type": ["null", "string"], "default": None},
                {"name": "causation_id", "type": ["null", "string"], "default": None},
                {"name": "trace_id", "type": ["null", "string"], "default": None},
                {"name": "span_id", "type": ["null", "string"], "default": None},
                {"name": "priority", "type": "string"},
                {"name": "partition_key", "type": "string"},
                {"name": "headers", "type": {"type": "map", "values": "string"}},
                {"name": "data", "type": "string"}  # JSON encoded
            ]
        }
    
    def to_kafka_record(self) -> Dict[str, Any]:
        """Convert to Kafka record format."""
        # Use model_dump with JSON serialization mode to handle datetime objects
        value_dict = self.model_dump(mode='json')
        
        return {
            "key": self.partition_key,
            "value": value_dict,
            "headers": [
                ("event_type", self.event_type.value.encode()),
                ("event_version", self.event_version.encode()),
                ("source_id", self.source_id.encode()),
                ("priority", self.priority.value.encode())
            ] + [
                (k, v.encode()) for k, v in self.headers.items()
            ]
        }


class AgentEvent(EventSchema):
    """Agent-specific event schema."""
    
    # Agent information
    agent_id: str
    agent_type: str
    agent_version: str
    
    # Agent state
    state_before: Optional[Dict[str, Any]] = None
    state_after: Optional[Dict[str, Any]] = None
    
    # Performance metrics
    duration_ms: Optional[float] = None
    tokens_used: Optional[Dict[str, int]] = None
    
    def __init__(self, **data):
        # Set source information from agent data
        if 'source_id' not in data:
            data['source_id'] = data.get('agent_id', '')
        if 'source_type' not in data:
            data['source_type'] = f"agent.{data.get('agent_type', 'unknown')}"
        
        super().__init__(**data)
    
    @classmethod
    def create_started_event(
        cls,
        agent_id: str,
        agent_type: str,
        agent_version: str,
        input_data: Dict[str, Any],
        **kwargs
    ) -> "AgentEvent":
        """Create agent started event."""
        return cls(
            event_type=EventType.AGENT_STARTED,
            agent_id=agent_id,
            agent_type=agent_type,
            agent_version=agent_version,
            data={
                "status": "started",
                "input": input_data
            },
            **kwargs
        )
    
    @classmethod
    def create_completed_event(
        cls,
        agent_id: str,
        agent_type: str,
        agent_version: str,
        output_data: Dict[str, Any],
        duration_ms: float,
        tokens_used: Optional[Dict[str, int]] = None,
        **kwargs
    ) -> "AgentEvent":
        """Create agent completed event."""
        return cls(
            event_type=EventType.AGENT_COMPLETED,
            agent_id=agent_id,
            agent_type=agent_type,
            agent_version=agent_version,
            duration_ms=duration_ms,
            tokens_used=tokens_used,
            data={
                "status": "completed",
                "output": output_data
            },
            **kwargs
        )
    
    @classmethod
    def create_decision_event(
        cls,
        agent_id: str,
        agent_type: str,
        agent_version: str,
        decision: str,
        reason: str,
        confidence: float,
        alternatives: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> "AgentEvent":
        """Create agent decision event."""
        return cls(
            event_type=EventType.AGENT_DECISION_MADE,
            agent_id=agent_id,
            agent_type=agent_type,
            agent_version=agent_version,
            data={
                "decision": decision,
                "reason": reason,
                "confidence": confidence,
                "alternatives": alternatives or []
            },
            **kwargs
        )


class WorkflowEvent(EventSchema):
    """Workflow-specific event schema."""
    
    # Workflow information
    workflow_id: str
    workflow_type: str
    workflow_version: str
    
    # Execution details
    run_id: str
    parent_workflow_id: Optional[str] = None
    
    # State information
    current_step: Optional[str] = None
    next_step: Optional[str] = None
    steps_completed: Optional[List[str]] = None
    
    def __init__(self, **data):
        # Set source information from workflow data
        if 'source_id' not in data:
            data['source_id'] = data.get('workflow_id', '')
        if 'source_type' not in data:
            data['source_type'] = f"workflow.{data.get('workflow_type', 'unknown')}"
        
        super().__init__(**data)
    
    @classmethod
    def create_started_event(
        cls,
        workflow_id: str,
        workflow_type: str,
        workflow_version: str,
        run_id: str,
        input_data: Dict[str, Any],
        **kwargs
    ) -> "WorkflowEvent":
        """Create workflow started event."""
        return cls(
            event_type=EventType.WORKFLOW_STARTED,
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            workflow_version=workflow_version,
            run_id=run_id,
            data={
                "status": "started",
                "input": input_data
            },
            **kwargs
        )
    
    @classmethod
    def create_step_completed_event(
        cls,
        workflow_id: str,
        workflow_type: str,
        workflow_version: str,
        run_id: str,
        step_name: str,
        step_output: Dict[str, Any],
        duration_ms: float,
        **kwargs
    ) -> "WorkflowEvent":
        """Create workflow step completed event."""
        return cls(
            event_type=EventType.WORKFLOW_STEP_COMPLETED,
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            workflow_version=workflow_version,
            run_id=run_id,
            current_step=step_name,
            data={
                "step": step_name,
                "output": step_output,
                "duration_ms": duration_ms
            },
            **kwargs
        )


class SystemEvent(EventSchema):
    """System-level event schema."""
    
    # System information
    component: str
    instance_id: str
    environment: str = "production"
    
    # Event severity
    severity: str = "info"  # debug, info, warning, error, critical
    
    def __init__(self, **data):
        # Set source information from system data
        if 'source_id' not in data:
            data['source_id'] = data.get('instance_id', '')
        if 'source_type' not in data:
            data['source_type'] = f"system.{data.get('component', 'unknown')}"
        
        super().__init__(**data)
    
    @classmethod
    def create_health_check_event(
        cls,
        component: str,
        instance_id: str,
        status: str,
        checks: Dict[str, Any],
        **kwargs
    ) -> "SystemEvent":
        """Create system health check event."""
        return cls(
            event_type=EventType.SYSTEM_HEALTH_CHECK,
            component=component,
            instance_id=instance_id,
            severity="info",
            data={
                "status": status,
                "checks": checks,
                "timestamp": datetime.utcnow().isoformat()
            },
            **kwargs
        )
    
    @classmethod
    def create_alert_event(
        cls,
        component: str,
        instance_id: str,
        alert_type: str,
        message: str,
        severity: str = "warning",
        details: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> "SystemEvent":
        """Create system alert event."""
        return cls(
            event_type=EventType.SYSTEM_ALERT,
            component=component,
            instance_id=instance_id,
            severity=severity,
            priority=EventPriority.HIGH if severity in ["error", "critical"] else EventPriority.NORMAL,
            data={
                "alert_type": alert_type,
                "message": message,
                "details": details or {},
                "timestamp": datetime.utcnow().isoformat()
            },
            **kwargs
        )


# Schema registry for managing event schemas
SCHEMA_REGISTRY = {
    "EventSchema": EventSchema,
    "AgentEvent": AgentEvent,
    "WorkflowEvent": WorkflowEvent,
    "SystemEvent": SystemEvent
}


def get_event_schema(event_type: str) -> type[EventSchema]:
    """Get event schema class by event type."""
    # Map event types to schema classes
    if event_type.startswith("agent."):
        return AgentEvent
    elif event_type.startswith("workflow."):
        return WorkflowEvent
    elif event_type.startswith("system."):
        return SystemEvent
    else:
        return EventSchema


def validate_event(event_data: Dict[str, Any]) -> EventSchema:
    """Validate and parse event data."""
    event_type = event_data.get("event_type", "")
    schema_class = get_event_schema(event_type)
    
    try:
        return schema_class(**event_data)
    except Exception as e:
        logger.error(f"Event validation failed: {e}", event_data=event_data)
        raise