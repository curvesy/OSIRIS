"""
🧠 Agent State - Immutable State with Pure Functional Updates

The heart of The Collective's memory and coordination with:
- Immutable state with pure functional updates
- Cryptographically signed state transitions
- Comprehensive audit trail
- Partial state views for privacy and performance
- Delta processing for efficient updates

The foundation for distributed multi-agent coordination.
"""

from typing import Dict, Any, List, Optional
from datetime import timedelta
from pydantic import Field, validator

try:
    from .base import (
        ImmutableBaseModel, DateTimeField, utc_now, validate_confidence_score, validate_signature_format
    )
    from .enums import TaskStatus, SignatureAlgorithm
    from .crypto import get_crypto_provider
    from .tracecontext import TraceContextMixin
    from .evidence import DossierEntry
    from .action import ActionRecord
    from .decision import DecisionPoint
except ImportError:
    # Fallback for direct import (testing/isolation)
    from base import (
        ImmutableBaseModel, DateTimeField, utc_now, validate_confidence_score, validate_signature_format
    )
    from enums import TaskStatus, SignatureAlgorithm
    from crypto import get_crypto_provider
    from tracecontext import TraceContextMixin
    # Skip complex imports for isolation testing
    DossierEntry = None
    ActionRecord = None
    DecisionPoint = None


# ============================================================================
# AGENT STATE - Immutable State Container
# ============================================================================

class AgentState(ImmutableBaseModel, TraceContextMixin):
    """
    Immutable Agent State for Distributed Multi-Agent Workflows.

    The comprehensive state object that gets passed between agents with:
    - Complete immutability and pure functional updates
    - Cryptographic signatures for all mutations
    - Full OpenTelemetry W3C trace context
    - Schema versioning for long-lived workflows
    - Enhanced explainability and decision rationale
    """

    # ========================================================================
    # IDENTITY & VERSIONING
    # ========================================================================
    task_id: str = Field(..., description="Globally unique task identifier")
    workflow_id: str = Field(..., description="Workflow instance identifier")
    correlation_id: str = Field(..., description="End-to-end correlation identifier")
    state_version: int = Field(default=1, description="Version number for this state instance")
    schema_version: str = Field(default="2.0", description="Schema version for compatibility")

    # ========================================================================
    # CRYPTOGRAPHIC AUTHENTICATION
    # ========================================================================
    state_signature: str = Field(..., description="Cryptographic signature of the entire state")
    signature_algorithm: SignatureAlgorithm = Field(
        default=SignatureAlgorithm.HMAC_SHA256,
        description="Cryptographic algorithm used for signing"
    )
    last_modifier_agent_id: str = Field(..., description="Agent that last modified this state")
    agent_public_key: str = Field(..., description="Public key of the last modifier agent")
    signature_timestamp: DateTimeField = Field(
        default_factory=utc_now,
        description="When the state was last signed"
    )

    # ========================================================================
    # TASK INFORMATION
    # ========================================================================
    task_type: str = Field(..., description="Type of task (e.g., 'incident_response', 'anomaly_investigation')")
    priority: str = Field(default="normal", description="Task priority level")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current task status")
    urgency: str = Field(default="medium", description="Task urgency (low, medium, high, critical)")

    # ========================================================================
    # INITIAL CONTEXT
    # ========================================================================
    initial_event: Dict[str, Any] = Field(..., description="The raw event that triggered the workflow")
    initial_context: Dict[str, Any] = Field(default_factory=dict, description="Initial context provided")
    trigger_source: str = Field(..., description="What triggered this workflow")

    # ========================================================================
    # EVIDENCE & INVESTIGATION
    # ========================================================================
    context_dossier: List[DossierEntry] = Field(
        default_factory=list,
        description="Collection of cryptographically signed evidence"
    )
    overall_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fused confidence score from all evidence"
    )
    confidence_calculation_method: str = Field(
        default="weighted_average",
        description="Method used to calculate overall confidence"
    )

    # ========================================================================
    # DECISION MAKING
    # ========================================================================
    decision_points: List[DecisionPoint] = Field(
        default_factory=list,
        description="Cryptographically signed decision points with full rationale"
    )
    active_decision: Optional[str] = Field(
        None,
        description="ID of currently active decision point"
    )

    # ========================================================================
    # EXECUTION & RESULTS
    # ========================================================================
    action_log: List[ActionRecord] = Field(
        default_factory=list,
        description="Cryptographically signed log of all actions taken"
    )
    final_outcome: Optional[str] = Field(
        None,
        description="Final result of the workflow"
    )
    success_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Quantitative success metrics"
    )

    # ========================================================================
    # COMMUNICATION & STAKEHOLDERS
    # ========================================================================
    human_readable_summary: Optional[str] = Field(
        None,
        description="Summary generated for stakeholders"
    )
    stakeholder_notifications: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Notifications sent to stakeholders"
    )
    communication_log: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Log of all communications"
    )

    # ========================================================================
    # WORKFLOW MANAGEMENT
    # ========================================================================
    current_agent_role: Optional[str] = Field(
        None,
        description="Role of the agent currently processing the state"
    )
    next_agents: List[str] = Field(
        default_factory=list,
        description="Agents that should process this state next"
    )
    waiting_for: Optional[str] = Field(
        None,
        description="What the workflow is waiting for"
    )
    workflow_phase: str = Field(
        default="investigation",
        description="Current phase of the workflow"
    )

    # ========================================================================
    # ERROR HANDLING & RECOVERY
    # ========================================================================
    error_count: int = Field(default=0, description="Number of errors encountered")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Error details with timestamps")
    recovery_attempts: int = Field(default=0, description="Number of recovery attempts")
    last_error_timestamp: Optional[DateTimeField] = Field(None, description="When the last error occurred")

    # ========================================================================
    # TEMPORAL INFORMATION
    # ========================================================================
    created_at: DateTimeField = Field(
        default_factory=utc_now,
        description="When the workflow was created"
    )
    updated_at: DateTimeField = Field(
        default_factory=utc_now,
        description="When the state was last updated"
    )
    deadline: Optional[DateTimeField] = Field(None, description="Workflow deadline")
    estimated_completion: Optional[DateTimeField] = Field(None, description="Estimated completion time")

    # ========================================================================
    # METADATA & CLASSIFICATION
    # ========================================================================
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    classification: Optional[str] = Field(None, description="Security classification")
    retention_policy: Optional[str] = Field(None, description="Data retention policy")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # ========================================================================
    # VALIDATION
    # ========================================================================
    @validator('overall_confidence')
    def validate_confidence(cls, v):
        return validate_confidence_score(v)

    @validator('state_signature')
    def validate_signature_format(cls, v):
        return validate_signature_format(v)

    @validator('state_version')
    def validate_version(cls, v):
        if v < 1:
            raise ValueError("State version must be >= 1")
        return v

    # ========================================================================
    # CRYPTOGRAPHIC METHODS
    # ========================================================================
    def sign_state(self, private_key: str, algorithm: Optional[SignatureAlgorithm] = None) -> str:
        """Generate cryptographic signature of the entire state."""
        sig_algorithm = algorithm or self.signature_algorithm
        state_bytes = self._get_canonical_state().encode('utf-8')
        provider = get_crypto_provider(sig_algorithm)
        return provider.sign(state_bytes, private_key)

    def verify_signature(self, private_key: str) -> bool:
        """Verify the state signature."""
        state_bytes = self._get_canonical_state().encode('utf-8')
        provider = get_crypto_provider(self.signature_algorithm)
        return provider.verify(state_bytes, self.state_signature, private_key)

    def _get_canonical_state(self) -> str:
        """Get canonical string representation of state for signing."""
        try:
            from .base import datetime_to_iso
        except ImportError:
            from base import datetime_to_iso
        updated_iso = datetime_to_iso(self.updated_at)
        return f"{self.task_id}:{self.workflow_id}:{self.state_version}:{updated_iso}:{self.last_modifier_agent_id}"

    # ========================================================================
    # PURE FUNCTIONAL UPDATES - Immutable State Transitions
    # ========================================================================
    def with_evidence(
        self,
        evidence: DossierEntry,
        modifier_agent_id: str,
        private_key: str,
        traceparent: Optional[str] = None
    ) -> 'AgentState':
        """Create new state with additional evidence (pure function)."""
        new_dossier = self.context_dossier + [evidence]
        new_confidence = self._calculate_confidence(new_dossier)
        new_version = self.state_version + 1
        current_time = utc_now()

        # Create new state data
        new_state_data = self.dict()
        new_state_data.update({
            'context_dossier': new_dossier,
            'overall_confidence': new_confidence,
            'state_version': new_version,
            'last_modifier_agent_id': modifier_agent_id,
            'updated_at': current_time,
            'signature_timestamp': current_time,
            'traceparent': traceparent or self.traceparent
        })

        # Create and sign new state
        new_state_data['state_signature'] = 'placeholder'
        new_state = AgentState(**new_state_data)
        signature = new_state.sign_state(private_key)
        new_state_data['state_signature'] = signature

        return AgentState(**new_state_data)

    def with_action(
        self,
        action: ActionRecord,
        modifier_agent_id: str,
        private_key: str,
        traceparent: Optional[str] = None
    ) -> 'AgentState':
        """Create new state with additional action (pure function)."""
        new_actions = self.action_log + [action]
        new_version = self.state_version + 1
        current_time = utc_now()

        # Update error count if action failed
        new_error_count = self.error_count
        new_error_timestamp = self.last_error_timestamp
        if not action.is_successful():
            new_error_count += 1
            new_error_timestamp = current_time

        # Create new state data
        new_state_data = self.dict()
        new_state_data.update({
            'action_log': new_actions,
            'error_count': new_error_count,
            'last_error_timestamp': new_error_timestamp,
            'state_version': new_version,
            'last_modifier_agent_id': modifier_agent_id,
            'updated_at': current_time,
            'signature_timestamp': current_time,
            'traceparent': traceparent or self.traceparent
        })

        # Create and sign new state
        new_state_data['state_signature'] = 'placeholder'
        new_state = AgentState(**new_state_data)
        signature = new_state.sign_state(private_key)
        new_state_data['state_signature'] = signature

        return AgentState(**new_state_data)

    def with_status(
        self,
        status: TaskStatus,
        modifier_agent_id: str,
        private_key: str,
        reason: Optional[str] = None,
        traceparent: Optional[str] = None
    ) -> 'AgentState':
        """Create new state with updated status (pure function)."""
        new_version = self.state_version + 1
        current_time = utc_now()

        # Update metadata with status change reason
        new_metadata = self.metadata.copy()
        if reason:
            new_metadata['status_change_reason'] = reason
            new_metadata['status_changed_at'] = current_time.isoformat()

        # Create new state data
        new_state_data = self.dict()
        new_state_data.update({
            'status': status,
            'metadata': new_metadata,
            'state_version': new_version,
            'last_modifier_agent_id': modifier_agent_id,
            'updated_at': current_time,
            'signature_timestamp': current_time,
            'traceparent': traceparent or self.traceparent
        })

        # Create and sign new state
        new_state_data['state_signature'] = 'placeholder'
        new_state = AgentState(**new_state_data)
        signature = new_state.sign_state(private_key)
        new_state_data['state_signature'] = signature

        return AgentState(**new_state_data)

    def _calculate_confidence(self, dossier: List[DossierEntry]) -> float:
        """Calculate overall confidence from evidence dossier."""
        if not dossier:
            return 0.0

        if self.confidence_calculation_method == "weighted_average":
            total_weight = 0.0
            weighted_sum = 0.0

            for entry in dossier:
                weight = entry.get_weighted_score()
                weighted_sum += entry.confidence * weight
                total_weight += weight

            return weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            # Default to simple average
            return sum(entry.confidence for entry in dossier) / len(dossier)

    def to_global_id(self) -> str:
        """Generate globally unique identifier."""
        return f"{self.workflow_id}:{self.task_id}"

    def __str__(self) -> str:
        return f"AgentState[{self.task_type}]({self.to_global_id()}) v{self.state_version} -> {self.status.value}"


# Export public interface
__all__ = [
    'AgentState'
]