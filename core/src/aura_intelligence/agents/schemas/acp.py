"""
🔐 Agent Communication Protocol (ACP) - Formal Multi-Agent Messaging

Enterprise-grade agent-to-agent communication with:
- Cryptographic signatures for integrity
- Correlation IDs for end-to-end tracing  
- Versioned payloads for schema evolution
- Priority-based routing and processing
- OpenTelemetry integration for observability

Based on the advanced schemas from phas02d.md and kakakagan.md research.
"""

import uuid
import hashlib
import hmac
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator


class MessageType(str, Enum):
    """Types of messages in the ACP protocol."""
    REQUEST = "request"
    RESPONSE = "response" 
    NOTIFICATION = "notification"
    BROADCAST = "broadcast"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


class Priority(str, Enum):
    """Message priority levels for routing and processing."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class ACPEndpoint(BaseModel):
    """Identifies the sender or recipient of a message."""
    agent_id: str = Field(..., description="The unique identifier of the agent")
    role: str = Field(..., description="The functional role of the agent (e.g., 'observer', 'analyst')")
    instance_id: Optional[str] = Field(None, description="Specific instance ID for scaled agents")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities for routing")
    
    @validator('agent_id')
    def validate_agent_id(cls, v):
        if not v or len(v) < 3:
            raise ValueError("agent_id must be at least 3 characters")
        return v
    
    def __str__(self) -> str:
        return f"{self.role}:{self.agent_id}"


class ACPEnvelope(BaseModel):
    """
    The standard envelope for all agent-to-agent communication.
    
    Provides security, traceability, and reliability for multi-agent systems.
    """
    
    # Message Identity
    message_id: str = Field(
        default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this specific message"
    )
    correlation_id: str = Field(
        ...,
        description="Identifier linking all messages within a single workflow or task"
    )
    
    # Routing Information
    sender: ACPEndpoint = Field(..., description="The agent sending the message")
    recipient: ACPEndpoint = Field(..., description="The intended recipient agent or service")
    message_type: MessageType = Field(..., description="Type of message for routing")
    priority: Priority = Field(default=Priority.NORMAL, description="Message priority level")
    
    # Temporal Information
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Timestamp of message creation in ISO 8601 format"
    )
    expires_at: Optional[str] = Field(None, description="Message expiration time")
    
    # Security & Integrity
    signature: str = Field(
        ...,
        description="HMAC-SHA256 signature of the payload for integrity verification"
    )
    
    # Content
    payload: Dict[str, Any] = Field(
        ...,
        description="The actual content of the message, structured as a dictionary"
    )
    payload_version: str = Field("1.0", description="The version of the payload schema")
    
    # Metadata
    trace_id: Optional[str] = Field(None, description="OpenTelemetry trace ID for observability")
    span_id: Optional[str] = Field(None, description="OpenTelemetry span ID for observability")
    retry_count: int = Field(default=0, description="Number of delivery attempts")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    
    @validator('correlation_id')
    def validate_correlation_id(cls, v):
        if not v or len(v) < 8:
            raise ValueError("correlation_id must be at least 8 characters")
        return v
    
    def sign_payload(self, secret_key: str) -> str:
        """
        Create HMAC-SHA256 signature of the payload.
        
        Args:
            secret_key: Secret key for signing
            
        Returns:
            Hex-encoded signature
        """
        payload_str = str(sorted(self.payload.items()))
        signature = hmac.new(
            secret_key.encode('utf-8'),
            payload_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def verify_signature(self, secret_key: str) -> bool:
        """
        Verify the message signature.
        
        Args:
            secret_key: Secret key for verification
            
        Returns:
            True if signature is valid
        """
        expected_signature = self.sign_payload(secret_key)
        return hmac.compare_digest(self.signature, expected_signature)
    
    def is_expired(self) -> bool:
        """Check if the message has expired."""
        if not self.expires_at:
            return False
        
        expiry = datetime.fromisoformat(self.expires_at.replace('Z', '+00:00'))
        return datetime.now(timezone.utc) > expiry
    
    def should_retry(self) -> bool:
        """Check if the message should be retried."""
        return self.retry_count < self.max_retries
    
    def increment_retry(self) -> 'ACPEnvelope':
        """Create a new envelope with incremented retry count."""
        return self.copy(update={
            'retry_count': self.retry_count + 1,
            'timestamp_utc': datetime.now(timezone.utc).isoformat()
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ACPEnvelope':
        """Create envelope from dictionary."""
        return cls(**data)
    
    def __str__(self) -> str:
        return f"ACP[{self.message_type.value}] {self.sender} -> {self.recipient} ({self.correlation_id[:8]})"


class ACPResponse(BaseModel):
    """Standard response format for ACP requests."""
    
    success: bool = Field(..., description="Whether the request was successful")
    result: Optional[Dict[str, Any]] = Field(None, description="Response data if successful")
    error: Optional[str] = Field(None, description="Error message if failed")
    error_code: Optional[str] = Field(None, description="Machine-readable error code")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    
    def to_payload(self) -> Dict[str, Any]:
        """Convert to payload format for ACP envelope."""
        return self.dict(exclude_none=True)


class ACPError(Exception):
    """Exception for ACP protocol errors."""
    
    def __init__(self, message: str, error_code: str = "ACP_ERROR", details: Optional[Dict] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
    
    def to_response(self) -> ACPResponse:
        """Convert to ACP response format."""
        return ACPResponse(
            success=False,
            error=str(self),
            error_code=self.error_code,
            result=self.details if self.details else None
        )


# Common payload schemas for different message types
class HeartbeatPayload(BaseModel):
    """Payload for heartbeat messages."""
    status: str = Field(..., description="Agent status (healthy, degraded, unhealthy)")
    load: float = Field(..., description="Current load percentage (0.0-1.0)")
    memory_usage: float = Field(..., description="Memory usage percentage (0.0-1.0)")
    active_tasks: int = Field(..., description="Number of active tasks")
    last_activity: str = Field(..., description="Timestamp of last activity")


class TaskAssignmentPayload(BaseModel):
    """Payload for task assignment messages."""
    task_id: str = Field(..., description="Unique task identifier")
    task_type: str = Field(..., description="Type of task to perform")
    parameters: Dict[str, Any] = Field(..., description="Task parameters")
    deadline: Optional[str] = Field(None, description="Task deadline")
    dependencies: List[str] = Field(default_factory=list, description="Task dependencies")


class ObservationPayload(BaseModel):
    """Payload for observation messages from Observer agents."""
    observation_id: str = Field(..., description="Unique observation identifier")
    observation_type: str = Field(..., description="Type of observation")
    severity: str = Field(..., description="Severity level (low, medium, high, critical)")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    data: Dict[str, Any] = Field(..., description="Observation data")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


class AnalysisPayload(BaseModel):
    """Payload for analysis results from Analyst agents."""
    analysis_id: str = Field(..., description="Unique analysis identifier")
    root_cause: Optional[str] = Field(None, description="Identified root cause")
    confidence: float = Field(..., description="Analysis confidence (0.0-1.0)")
    recommendations: List[Dict[str, Any]] = Field(default_factory=list, description="Recommended actions")
    evidence: List[Dict[str, Any]] = Field(default_factory=list, description="Supporting evidence")
    risk_assessment: Dict[str, Any] = Field(default_factory=dict, description="Risk analysis")


class ExecutionPayload(BaseModel):
    """Payload for execution results from Executor agents."""
    execution_id: str = Field(..., description="Unique execution identifier")
    action_taken: str = Field(..., description="Description of action taken")
    status: str = Field(..., description="Execution status (success, failed, partial)")
    result: Dict[str, Any] = Field(..., description="Execution result data")
    side_effects: List[str] = Field(default_factory=list, description="Observed side effects")
    rollback_procedure: Optional[str] = Field(None, description="Rollback procedure if needed")
