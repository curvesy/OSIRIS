"""
⚡ Action Models - Structured Action Intent & Execution

Comprehensive action recording and management with:
- Structured action intent and rationale
- Cryptographically signed action records
- Risk assessment and business justification
- Execution tracking and rollback support
- Global references and audit trails

The foundation for accountable agent actions.
"""

from typing import Dict, Any, List, Optional
from pydantic import Field, validator, model_validator

try:
    from .base import (
        GloballyIdentifiable, MetadataSupport, TemporalSupport, QualityMetrics,
        DateTimeField, utc_now, validate_confidence_score, validate_signature_format
    )
    from .enums import ActionType, ActionCategory, ActionResult, RiskLevel, SignatureAlgorithm, get_action_category
    from .crypto import get_crypto_provider
    from .tracecontext import TraceContextMixin
    from .evidence import EvidenceReference
except ImportError:
    # Fallback for direct import (testing/isolation)
    from base import (
        GloballyIdentifiable, MetadataSupport, TemporalSupport, QualityMetrics,
        DateTimeField, utc_now, validate_confidence_score, validate_signature_format
    )
    from enums import ActionType, ActionCategory, ActionResult, RiskLevel, SignatureAlgorithm, get_action_category
    from crypto import get_crypto_provider
    from tracecontext import TraceContextMixin
    # Skip complex imports for isolation testing
    EvidenceReference = None


# ============================================================================
# STRUCTURED ACTION INTENT - Comprehensive Rationale
# ============================================================================

class ActionIntent(QualityMetrics):
    """
    Structured action intent for analytics and explainability.
    
    Captures the complete rationale, risk assessment, and business
    context for every action taken by agents.
    """
    
    # ========================================================================
    # PRIMARY INTENT
    # ========================================================================
    primary_goal: str = Field(..., description="Primary goal of the action")
    expected_outcome: str = Field(..., description="Expected outcome")
    success_criteria: List[str] = Field(..., description="Criteria for success")
    
    # ========================================================================
    # RISK ASSESSMENT
    # ========================================================================
    risk_level: RiskLevel = Field(..., description="Risk level assessment")
    potential_failures: List[str] = Field(default_factory=list, description="Potential failure modes")
    mitigation_strategies: List[str] = Field(default_factory=list, description="Risk mitigation strategies")
    impact_assessment: str = Field(..., description="Assessment of potential impact")
    
    # ========================================================================
    # DEPENDENCIES & CONSTRAINTS
    # ========================================================================
    prerequisites: List[str] = Field(default_factory=list, description="Action prerequisites")
    dependencies: List[str] = Field(default_factory=list, description="System dependencies")
    constraints: List[str] = Field(default_factory=list, description="Operational constraints")
    
    # ========================================================================
    # IMPACT ANALYSIS
    # ========================================================================
    affected_systems: List[str] = Field(default_factory=list, description="Systems that will be affected")
    estimated_duration: Optional[str] = Field(None, description="Estimated duration")
    resource_requirements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Required resources (CPU, memory, network, etc.)"
    )
    rollback_plan: Optional[str] = Field(None, description="Rollback plan if needed")
    
    # ========================================================================
    # BUSINESS CONTEXT
    # ========================================================================
    business_justification: str = Field(..., description="Business justification")
    stakeholder_impact: Optional[str] = Field(None, description="Impact on stakeholders")
    compliance_notes: Optional[str] = Field(None, description="Compliance considerations")
    cost_benefit_analysis: Optional[str] = Field(None, description="Cost-benefit analysis")
    
    # ========================================================================
    # APPROVAL & AUTHORIZATION
    # ========================================================================
    requires_approval: bool = Field(default=False, description="Whether action requires approval")
    approval_level: Optional[str] = Field(None, description="Required approval level")
    emergency_override: bool = Field(default=False, description="Emergency override flag")
    
    def get_risk_score(self) -> float:
        """Calculate numeric risk score."""
        base_score = self.risk_level.get_numeric_value() / 5.0  # Normalize to 0-1
        
        # Adjust based on factors
        failure_factor = min(len(self.potential_failures) * 0.1, 0.3)
        mitigation_factor = min(len(self.mitigation_strategies) * 0.05, 0.2)
        
        return min(1.0, base_score + failure_factor - mitigation_factor)


# ============================================================================
# ACTION REFERENCE - Global Lineage Tracking
# ============================================================================

class ActionReference(GloballyIdentifiable):
    """Reference to action with full lineage for audit trails."""
    
    action_id: str = Field(..., description="Unique action identifier")
    agent_id: str = Field(..., description="Agent that executed the action")
    action_timestamp: DateTimeField = Field(..., description="When the action was executed")
    action_type: str = Field(..., description="Type of action")
    
    def __str__(self) -> str:
        return f"Action[{self.action_type}]({self.to_global_id()})"


# ============================================================================
# ACTION RECORD - Main Action Container
# ============================================================================

class ActionRecord(
    GloballyIdentifiable,
    MetadataSupport,
    TemporalSupport,
    TraceContextMixin
):
    """
    Cryptographically signed action record with full audit trail.
    
    The main container for all actions taken by agents with:
    - Structured action intent
    - Cryptographic signatures
    - Execution tracking
    - Risk assessment
    - Global lineage tracking
    """
    
    # ========================================================================
    # IDENTITY
    # ========================================================================
    action_id: str = Field(..., description="Globally unique action identifier")
    
    # ========================================================================
    # CRYPTOGRAPHIC AUTHENTICATION
    # ========================================================================
    executing_agent_id: str = Field(..., description="Agent that executed this action")
    agent_public_key: str = Field(..., description="Public key of the executing agent")
    signature_algorithm: SignatureAlgorithm = Field(
        default=SignatureAlgorithm.HMAC_SHA256,
        description="Cryptographic algorithm used for signing"
    )
    action_signature: str = Field(..., description="Cryptographic signature of the action")
    signature_timestamp: DateTimeField = Field(
        default_factory=utc_now,
        description="When the action was signed"
    )
    
    # ========================================================================
    # STRUCTURED ACTION DETAILS
    # ========================================================================
    action_type: ActionType = Field(..., description="Structured type of action taken")
    action_category: ActionCategory = Field(..., description="High-level action category")
    action_name: str = Field(..., description="Specific name of the action")
    description: str = Field(..., description="Human-readable description of the action")
    structured_intent: ActionIntent = Field(..., description="Structured action intent and rationale")
    
    # ========================================================================
    # EXECUTION INFORMATION
    # ========================================================================
    execution_timestamp: DateTimeField = Field(
        default_factory=utc_now,
        description="When the action was executed"
    )
    duration_ms: Optional[float] = Field(None, description="How long the action took")
    timeout_ms: Optional[float] = Field(None, description="Action timeout if applicable")
    
    # ========================================================================
    # RESULTS & OUTCOMES
    # ========================================================================
    result: ActionResult = Field(..., description="Structured result of the action")
    result_data: Optional[Dict[str, Any]] = Field(None, description="Detailed result data")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Error details if failed")
    
    # ========================================================================
    # IMPACT ASSESSMENT
    # ========================================================================
    side_effects: List[str] = Field(default_factory=list, description="Observed side effects")
    affected_systems: List[str] = Field(default_factory=list, description="Systems affected by action")
    rollback_available: bool = Field(default=False, description="Whether rollback is possible")
    rollback_procedure: Optional[str] = Field(None, description="How to rollback if needed")
    
    # ========================================================================
    # EVIDENCE & JUSTIFICATION
    # ========================================================================
    supporting_evidence: List[EvidenceReference] = Field(
        default_factory=list,
        description="Evidence that supported this action"
    )
    decision_rationale: str = Field(..., description="Why this action was chosen")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the action")
    
    # ========================================================================
    # APPROVAL & AUTHORIZATION
    # ========================================================================
    requires_approval: bool = Field(default=False, description="Whether action required approval")
    approved_by: Optional[str] = Field(None, description="Who approved the action")
    approval_timestamp: Optional[DateTimeField] = Field(None, description="When approval was granted")
    authorization_level: Optional[str] = Field(None, description="Required authorization level")
    
    # ========================================================================
    # VALIDATION
    # ========================================================================
    @validator('confidence')
    def validate_confidence(cls, v):
        return validate_confidence_score(v)
    
    @validator('action_signature')
    def validate_signature_format(cls, v):
        return validate_signature_format(v)
    
    @model_validator(mode='before')
    @classmethod
    def validate_action_category(cls, values):
        """Ensure action category matches action type."""
        if isinstance(values, dict):
            action_type = values.get('action_type')
            action_category = values.get('action_category')
            
            if action_type and not action_category:
                values['action_category'] = get_action_category(action_type)
            elif action_type and action_category:
                expected_category = get_action_category(action_type)
                if action_category != expected_category:
                    raise ValueError(f"Action category {action_category} doesn't match type {action_type}")
        
        return values
    
    # ========================================================================
    # CRYPTOGRAPHIC METHODS
    # ========================================================================
    def sign_action(self, private_key: str, algorithm: Optional[SignatureAlgorithm] = None) -> str:
        """Generate cryptographic signature of the action."""
        sig_algorithm = algorithm or self.signature_algorithm
        action_bytes = self._get_canonical_action().encode('utf-8')
        provider = get_crypto_provider(sig_algorithm)
        return provider.sign(action_bytes, private_key)
    
    def verify_signature(self, private_key: str) -> bool:
        """Verify the action signature."""
        action_bytes = self._get_canonical_action().encode('utf-8')
        provider = get_crypto_provider(self.signature_algorithm)
        return provider.verify(action_bytes, self.action_signature, private_key)
    
    def _get_canonical_action(self) -> str:
        """Get canonical string representation of action for signing."""
        try:
            from .base import datetime_to_iso
        except ImportError:
            from base import datetime_to_iso
        timestamp_iso = datetime_to_iso(self.execution_timestamp)
        return f"{self.action_type.value}:{self.action_name}:{timestamp_iso}:{self.executing_agent_id}"
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    def get_reference(self) -> ActionReference:
        """Get a reference to this action record."""
        return ActionReference(
            workflow_id=self.workflow_id,
            task_id=self.task_id,
            correlation_id=self.correlation_id,
            action_id=self.action_id,
            agent_id=self.executing_agent_id,
            action_timestamp=self.execution_timestamp,
            action_type=self.action_type.value,
            schema_version=self.schema_version
        )
    
    def is_successful(self) -> bool:
        """Check if the action was successful."""
        return self.result.is_successful()
    
    def requires_retry(self) -> bool:
        """Check if the action requires retry."""
        return self.result.requires_retry()
    
    def requires_rollback(self) -> bool:
        """Check if the action should be rolled back."""
        return (
            self.result == ActionResult.FAILURE and
            self.rollback_available and
            self.rollback_procedure is not None
        )
    
    def get_risk_score(self) -> float:
        """Get the risk score from structured intent."""
        return self.structured_intent.get_risk_score()
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of action execution."""
        return {
            'action_id': self.action_id,
            'action_type': self.action_type.value,
            'action_name': self.action_name,
            'result': self.result.value,
            'duration_ms': self.duration_ms,
            'confidence': self.confidence,
            'risk_score': self.get_risk_score(),
            'successful': self.is_successful(),
            'requires_rollback': self.requires_rollback(),
            'affected_systems': len(self.affected_systems),
            'side_effects': len(self.side_effects)
        }
    
    def __str__(self) -> str:
        return f"Action[{self.action_type.value}]({self.to_global_id()}) -> {self.result.value}"


# Export public interface
__all__ = [
    # Intent and rationale
    'ActionIntent',
    
    # References
    'ActionReference',
    
    # Main models
    'ActionRecord'
]
