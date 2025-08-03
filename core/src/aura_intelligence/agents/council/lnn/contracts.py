"""
Council Agent Contracts

Defines all data contracts used by the LNN Council Agent system.
Following Domain-Driven Design principles with immutable value objects.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Dict, List, Optional, Any, FrozenSet
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, Field, validator


class VoteDecision(str, Enum):
    """Possible voting decisions."""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"
    DELEGATE = "delegate"
    ESCALATE = "escalate"


class VoteConfidence(float):
    """Vote confidence score between 0.0 and 1.0."""
    
    def __new__(cls, value: float):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {value}")
        return float.__new__(cls, value)
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """Pydantic core schema for VoteConfidence."""
        from pydantic_core import core_schema
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.float_schema(ge=0.0, le=1.0)
        )


class ContextScope(str, Enum):
    """Scope of context retrieval."""
    LOCAL = "local"
    HISTORICAL = "historical"
    GLOBAL = "global"
    FEDERATED = "federated"


class AgentCapability(str, Enum):
    """Agent capabilities for registry."""
    GPU_ALLOCATION = "gpu_allocation"
    RESOURCE_MANAGEMENT = "resource_management"
    COST_OPTIMIZATION = "cost_optimization"
    RISK_ASSESSMENT = "risk_assessment"
    COMPLIANCE_CHECK = "compliance_check"


@dataclass(frozen=True)
class ResourceRequest:
    """Immutable resource request details."""
    resource_type: str
    quantity: int
    duration_hours: float
    priority: int = 5
    constraints: FrozenSet[str] = field(default_factory=frozenset)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        if self.duration_hours <= 0:
            raise ValueError("Duration must be positive")
        if not 0 <= self.priority <= 10:
            raise ValueError("Priority must be between 0 and 10")


@dataclass(frozen=True)
class DecisionEvidence:
    """Immutable evidence supporting a decision."""
    evidence_type: str
    source: str
    confidence: VoteConfidence
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.evidence_type,
            "source": self.source,
            "confidence": float(self.confidence),
            "data": self.data,
            "timestamp": self.timestamp.isoformat()
        }


class CouncilRequest(BaseModel):
    """Validated council request using Pydantic."""
    request_id: UUID = Field(default_factory=uuid4)
    request_type: str
    payload: Dict[str, Any]
    context: Dict[str, Any] = Field(default_factory=dict)
    requester_id: str
    priority: int = Field(default=5, ge=0, le=10)
    deadline: Optional[datetime] = None
    capabilities_required: List[AgentCapability] = Field(default_factory=list)
    context_scope: ContextScope = ContextScope.LOCAL
    
    class Config:
        use_enum_values = True
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }
    
    @validator('deadline')
    def deadline_must_be_future(cls, v):
        if v and v < datetime.now(timezone.utc):
            raise ValueError('Deadline must be in the future')
        return v


class CouncilResponse(BaseModel):
    """Validated council response."""
    request_id: UUID
    agent_id: str
    decision: VoteDecision
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    evidence: List[DecisionEvidence] = Field(default_factory=list)
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
            DecisionEvidence: lambda v: v.to_dict()
        }
    
    @validator('confidence')
    def validate_confidence(cls, v):
        return VoteConfidence(v)


@dataclass(frozen=True)
class NeuralFeatures:
    """Immutable neural network features."""
    raw_features: np.ndarray
    normalized_features: np.ndarray
    feature_names: List[str]
    extraction_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if len(self.raw_features) != len(self.normalized_features):
            raise ValueError("Raw and normalized features must have same length")
        if len(self.feature_names) != len(self.raw_features):
            raise ValueError("Feature names must match feature length")


@dataclass(frozen=True)
class ContextSnapshot:
    """Immutable context snapshot."""
    snapshot_id: UUID = field(default_factory=uuid4)
    historical_patterns: List[Dict[str, Any]] = field(default_factory=list)
    recent_decisions: List[Dict[str, Any]] = field(default_factory=list)
    entity_relationships: Dict[str, List[str]] = field(default_factory=dict)
    temporal_context: Dict[str, Any] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def merge_with(self, other: 'ContextSnapshot') -> 'ContextSnapshot':
        """Create new snapshot by merging with another."""
        return ContextSnapshot(
            historical_patterns=self.historical_patterns + other.historical_patterns,
            recent_decisions=self.recent_decisions + other.recent_decisions,
            entity_relationships={**self.entity_relationships, **other.entity_relationships},
            temporal_context={**self.temporal_context, **other.temporal_context},
            confidence_scores={**self.confidence_scores, **other.confidence_scores}
        )


@dataclass(frozen=True)
class AgentMetrics:
    """Immutable agent performance metrics."""
    total_decisions: int = 0
    approval_rate: float = 0.0
    average_confidence: float = 0.0
    average_processing_time_ms: float = 0.0
    error_rate: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def update(self, **kwargs) -> 'AgentMetrics':
        """Create new metrics with updated values."""
        return AgentMetrics(
            total_decisions=kwargs.get('total_decisions', self.total_decisions),
            approval_rate=kwargs.get('approval_rate', self.approval_rate),
            average_confidence=kwargs.get('average_confidence', self.average_confidence),
            average_processing_time_ms=kwargs.get('average_processing_time_ms', self.average_processing_time_ms),
            error_rate=kwargs.get('error_rate', self.error_rate),
            last_updated=datetime.now(timezone.utc)
        )