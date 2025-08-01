"""
🧠 Decision Models - Enhanced Explainability & Option Scoring

Comprehensive decision making and management with:
- Multi-criteria decision analysis
- Scored options with evidence relationships
- Enhanced explainability and rationale
- Cryptographically signed decisions
- Global references and audit trails

The foundation for transparent agent decision making.
"""

from typing import Dict, Any, List, Optional
from pydantic import Field, validator

try:
    from .base import (
        GloballyIdentifiable, MetadataSupport, TemporalSupport, QualityMetrics,
        DateTimeField, utc_now, validate_confidence_score, validate_signature_format
    )
    from .enums import DecisionType, DecisionMethod, RiskLevel, SignatureAlgorithm
    from .crypto import get_crypto_provider
    from .tracecontext import TraceContextMixin
    from .evidence import EvidenceReference
except ImportError:
    # Fallback for direct import (testing/isolation)
    from base import (
        GloballyIdentifiable, MetadataSupport, TemporalSupport, QualityMetrics,
        DateTimeField, utc_now, validate_confidence_score, validate_signature_format
    )
    from enums import DecisionType, DecisionMethod, RiskLevel, SignatureAlgorithm
    from crypto import get_crypto_provider
    from tracecontext import TraceContextMixin
    # Skip complex imports for isolation testing
    EvidenceReference = None


# ============================================================================
# DECISION CRITERION - Weighted Criteria for Multi-Criteria Analysis
# ============================================================================

class DecisionCriterion(QualityMetrics):
    """A criterion used for decision making with weights and measurement."""
    
    criterion_id: str = Field(..., description="Unique identifier for this criterion")
    name: str = Field(..., description="Name of the criterion")
    description: str = Field(..., description="Description of what this criterion measures")
    weight: float = Field(..., ge=0.0, le=1.0, description="Weight of this criterion in decision")
    measurement_method: str = Field(..., description="How this criterion is measured")
    target_value: Optional[float] = Field(None, description="Target value for this criterion")
    acceptable_range: Optional[Dict[str, float]] = Field(None, description="Acceptable range")
    
    @validator('weight')
    def validate_weight(cls, v):
        return validate_confidence_score(v)


# ============================================================================
# DECISION OPTION - Scored Options with Evidence Relationships
# ============================================================================

class DecisionOption(QualityMetrics):
    """A single decision option with scoring and comprehensive rationale."""
    
    option_id: str = Field(..., description="Unique identifier for this option")
    name: str = Field(..., description="Human-readable name of the option")
    description: str = Field(..., description="Detailed description of the option")
    
    # ========================================================================
    # SCORING & ASSESSMENT
    # ========================================================================
    score: float = Field(..., ge=0.0, le=1.0, description="Overall score for this option")
    criteria_scores: Dict[str, float] = Field(..., description="Scores for each decision criterion")
    model_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Model confidence in scoring")
    
    # ========================================================================
    # IMPACT ASSESSMENT
    # ========================================================================
    expected_outcome: str = Field(..., description="Expected outcome if this option is chosen")
    risk_level: RiskLevel = Field(..., description="Risk level assessment")
    effort_required: str = Field(..., description="Effort required (low, medium, high)")
    time_to_implement: Optional[str] = Field(None, description="Estimated time to implement")
    cost_estimate: Optional[float] = Field(None, description="Estimated cost")
    
    # ========================================================================
    # DEPENDENCIES & CONSTRAINTS
    # ========================================================================
    prerequisites: List[str] = Field(default_factory=list, description="Prerequisites for this option")
    constraints: List[str] = Field(default_factory=list, description="Constraints that apply")
    side_effects: List[str] = Field(default_factory=list, description="Potential side effects")
    
    # ========================================================================
    # ENHANCED EVIDENCE RELATIONSHIPS
    # ========================================================================
    supporting_evidence: List[EvidenceReference] = Field(
        default_factory=list,
        description="Evidence that supports this option"
    )
    contradicting_evidence: List[EvidenceReference] = Field(
        default_factory=list,
        description="Evidence that contradicts this option"
    )
    evidence_strength: Dict[str, float] = Field(
        default_factory=dict,
        description="Strength of evidence support (evidence_id -> strength)"
    )
    
    # ========================================================================
    # REJECTION ANALYSIS
    # ========================================================================
    rejection_rationale: Optional[str] = Field(None, description="Why this option was rejected")
    rejection_evidence: List[EvidenceReference] = Field(
        default_factory=list,
        description="Evidence that led to rejection"
    )
    
    # ========================================================================
    # METADATA
    # ========================================================================
    proposed_by: Optional[str] = Field(None, description="Agent that proposed this option")
    proposal_timestamp: Optional[DateTimeField] = Field(None, description="When this option was proposed")
    
    @validator('score', 'model_confidence')
    def validate_scores(cls, v):
        if v is not None:
            return validate_confidence_score(v)
        return v
    
    def get_evidence_support_score(self) -> float:
        """Calculate overall evidence support score."""
        if not self.evidence_strength:
            return 0.5  # Neutral if no evidence
        
        total_strength = sum(self.evidence_strength.values())
        evidence_count = len(self.evidence_strength)
        
        return min(1.0, total_strength / evidence_count) if evidence_count > 0 else 0.5
    
    def get_risk_score(self) -> float:
        """Get numeric risk score."""
        return self.risk_level.get_numeric_value() / 5.0  # Normalize to 0-1


# ============================================================================
# DECISION POINT - Main Decision Container
# ============================================================================

class DecisionPoint(
    GloballyIdentifiable,
    MetadataSupport,
    TemporalSupport,
    TraceContextMixin
):
    """
    Cryptographically signed decision point with enhanced explainability.
    
    Captures decision-making process with full rationale, option scoring,
    and audit trail for enterprise compliance and explainability.
    """
    
    # ========================================================================
    # IDENTITY
    # ========================================================================
    decision_id: str = Field(..., description="Globally unique decision identifier")
    
    # ========================================================================
    # CRYPTOGRAPHIC AUTHENTICATION
    # ========================================================================
    deciding_agent_id: str = Field(..., description="Agent that made this decision")
    agent_public_key: str = Field(..., description="Public key of the deciding agent")
    signature_algorithm: SignatureAlgorithm = Field(
        default=SignatureAlgorithm.HMAC_SHA256,
        description="Cryptographic algorithm used for signing"
    )
    decision_signature: str = Field(..., description="Cryptographic signature of the decision")
    signature_timestamp: DateTimeField = Field(
        default_factory=utc_now,
        description="When the decision was signed"
    )
    
    # ========================================================================
    # DECISION CONTEXT
    # ========================================================================
    decision_name: str = Field(..., description="Name of the decision being made")
    description: str = Field(..., description="What decision needs to be made")
    decision_type: DecisionType = Field(..., description="Type of decision")
    urgency: str = Field(..., description="Urgency level (low, medium, high, critical)")
    impact_scope: str = Field(..., description="Scope of impact (local, system, global)")
    
    # ========================================================================
    # ENHANCED DECISION FRAMEWORK
    # ========================================================================
    criteria: List[DecisionCriterion] = Field(..., description="Decision criteria with weights")
    options: List[DecisionOption] = Field(..., description="Available options with scoring")
    
    # Decision Algorithm
    decision_method: DecisionMethod = Field(..., description="Method used for decision making")
    algorithm_version: str = Field(default="1.0", description="Version of decision algorithm")
    
    # ========================================================================
    # DECISION RESULT
    # ========================================================================
    chosen_option_id: Optional[str] = Field(None, description="ID of the chosen option")
    decision_rationale: Optional[str] = Field(None, description="Detailed rationale for the decision")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence in the decision")
    
    # Alternative Analysis
    runner_up_option_id: Optional[str] = Field(None, description="Second-best option")
    option_comparison: Optional[Dict[str, Any]] = Field(None, description="Comparison between top options")
    
    # ========================================================================
    # SUPPORTING INFORMATION
    # ========================================================================
    supporting_evidence: List[EvidenceReference] = Field(
        default_factory=list,
        description="Evidence that informed this decision"
    )
    consulted_agents: List[str] = Field(
        default_factory=list,
        description="Other agents consulted for this decision"
    )
    
    # ========================================================================
    # TEMPORAL INFORMATION
    # ========================================================================
    decision_timestamp: DateTimeField = Field(
        default_factory=utc_now,
        description="When the decision was made"
    )
    deadline: Optional[DateTimeField] = Field(None, description="Decision deadline if applicable")
    time_to_decide_ms: Optional[float] = Field(None, description="Time taken to make decision")
    
    # ========================================================================
    # METADATA & AUDIT
    # ========================================================================
    review_required: bool = Field(default=False, description="Whether decision requires review")
    reviewed_by: Optional[str] = Field(None, description="Who reviewed the decision")
    review_timestamp: Optional[DateTimeField] = Field(None, description="When decision was reviewed")
    
    # ========================================================================
    # VALIDATION
    # ========================================================================
    @validator('confidence')
    def validate_confidence(cls, v):
        if v is not None:
            return validate_confidence_score(v)
        return v
    
    @validator('decision_signature')
    def validate_signature_format(cls, v):
        return validate_signature_format(v)
    
    # ========================================================================
    # CRYPTOGRAPHIC METHODS
    # ========================================================================
    def sign_decision(self, private_key: str, algorithm: Optional[SignatureAlgorithm] = None) -> str:
        """Generate cryptographic signature of the decision."""
        sig_algorithm = algorithm or self.signature_algorithm
        decision_bytes = self._get_canonical_decision().encode('utf-8')
        provider = get_crypto_provider(sig_algorithm)
        return provider.sign(decision_bytes, private_key)
    
    def verify_signature(self, private_key: str) -> bool:
        """Verify the decision signature."""
        decision_bytes = self._get_canonical_decision().encode('utf-8')
        provider = get_crypto_provider(self.signature_algorithm)
        return provider.verify(decision_bytes, self.decision_signature, private_key)
    
    def _get_canonical_decision(self) -> str:
        """Get canonical string representation of decision for signing."""
        try:
            from .base import datetime_to_iso
        except ImportError:
            from base import datetime_to_iso
        chosen_option = self.chosen_option_id or "none"
        timestamp_iso = datetime_to_iso(self.decision_timestamp)
        return f"{self.decision_name}:{chosen_option}:{timestamp_iso}:{self.deciding_agent_id}"
    
    # ========================================================================
    # DECISION ANALYSIS METHODS
    # ========================================================================
    def get_chosen_option(self) -> Optional[DecisionOption]:
        """Get the chosen decision option."""
        if not self.chosen_option_id:
            return None
        
        for option in self.options:
            if option.option_id == self.chosen_option_id:
                return option
        
        return None
    
    def get_runner_up_option(self) -> Optional[DecisionOption]:
        """Get the runner-up decision option."""
        if not self.runner_up_option_id:
            return None
        
        for option in self.options:
            if option.option_id == self.runner_up_option_id:
                return option
        
        return None
    
    def calculate_option_scores(self) -> Dict[str, float]:
        """Calculate weighted scores for all options."""
        scores = {}
        
        for option in self.options:
            total_score = 0.0
            total_weight = 0.0
            
            for criterion in self.criteria:
                if criterion.criterion_id in option.criteria_scores:
                    score = option.criteria_scores[criterion.criterion_id]
                    total_score += score * criterion.weight
                    total_weight += criterion.weight
            
            # Normalize by total weight
            if total_weight > 0:
                scores[option.option_id] = total_score / total_weight
            else:
                scores[option.option_id] = 0.0
        
        return scores
    
    def get_decision_explanation(self) -> Dict[str, Any]:
        """Generate detailed explanation of the decision."""
        chosen_option = self.get_chosen_option()
        runner_up = self.get_runner_up_option()
        scores = self.calculate_option_scores()
        
        explanation = {
            "decision_summary": {
                "decision_name": self.decision_name,
                "chosen_option": chosen_option.name if chosen_option else None,
                "confidence": self.confidence,
                "rationale": self.decision_rationale,
                "method": self.decision_method.value
            },
            "option_analysis": {
                "total_options": len(self.options),
                "scores": scores,
                "chosen_score": scores.get(self.chosen_option_id, 0.0) if self.chosen_option_id else 0.0,
                "runner_up_score": scores.get(self.runner_up_option_id, 0.0) if self.runner_up_option_id else 0.0
            },
            "criteria_breakdown": [
                {
                    "name": criterion.name,
                    "weight": criterion.weight,
                    "chosen_score": chosen_option.criteria_scores.get(criterion.criterion_id, 0.0) if chosen_option else 0.0,
                    "runner_up_score": runner_up.criteria_scores.get(criterion.criterion_id, 0.0) if runner_up else 0.0
                }
                for criterion in self.criteria
            ],
            "evidence_analysis": {
                "supporting_evidence_count": len(self.supporting_evidence),
                "chosen_option_evidence": len(chosen_option.supporting_evidence) if chosen_option else 0,
                "chosen_option_contradictions": len(chosen_option.contradicting_evidence) if chosen_option else 0
            },
            "consultation": {
                "agents_consulted": len(self.consulted_agents),
                "decision_time_ms": self.time_to_decide_ms
            }
        }
        
        return explanation
    
    def __str__(self) -> str:
        chosen = self.get_chosen_option()
        chosen_name = chosen.name if chosen else "none"
        return f"Decision[{self.decision_name}]({self.to_global_id()}) -> {chosen_name}"


# Export public interface
__all__ = [
    # Decision framework
    'DecisionCriterion', 'DecisionOption',
    
    # Main models
    'DecisionPoint'
]
