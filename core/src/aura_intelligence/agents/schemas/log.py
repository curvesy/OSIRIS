"""
📝 Agent Action Logging Schema - Memory Flywheel Integration

Comprehensive logging of agent actions back into the Hot→Cold→Wise memory system.
Enables the Intelligence Flywheel to learn from agent decisions and outcomes.

Key Features:
- Structured action logging with context
- Integration with existing memory tiers
- Performance metrics and outcomes
- Causal relationship tracking
- Learning feedback loops

Based on the advanced schemas from phas02d.md and kakakagan.md research.
"""

import uuid
import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, validator


class ActionType(str, Enum):
    """Types of actions that agents can perform."""
    OBSERVATION = "observation"
    ANALYSIS = "analysis"
    INVESTIGATION = "investigation"
    DECISION = "decision"
    EXECUTION = "execution"
    COMMUNICATION = "communication"
    ESCALATION = "escalation"
    RECOVERY = "recovery"
    LEARNING = "learning"


class ActionResult(str, Enum):
    """Possible outcomes of agent actions."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    RETRY_NEEDED = "retry_needed"


class ImpactLevel(str, Enum):
    """Impact level of the action."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ContextSource(str, Enum):
    """Sources of context used in decision making."""
    HOT_MEMORY = "hot_memory"
    COLD_STORAGE = "cold_storage"
    SEMANTIC_MEMORY = "semantic_memory"
    CAUSAL_GRAPH = "causal_graph"
    EXTERNAL_API = "external_api"
    HUMAN_INPUT = "human_input"
    AGENT_REASONING = "agent_reasoning"


class AgentActionEvent(BaseModel):
    """
    The schema for an event logged by an agent back into the memory flywheel.
    
    This creates a feedback loop where agent actions become part of the
    intelligence system's memory, enabling continuous learning and improvement.
    """
    
    # Event Identity
    event_id: str = Field(
        default_factory=lambda: f"action_{uuid.uuid4().hex[:12]}",
        description="A unique ID for this action event"
    )
    
    # Temporal Information
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Timestamp of when the action was taken"
    )
    duration_ms: Optional[float] = Field(None, description="How long the action took to complete")
    
    # Agent Information
    agent_id: str = Field(..., description="The ID of the agent that performed the action")
    agent_role: str = Field(..., description="The role of the agent (observer, analyst, etc.)")
    agent_instance: Optional[str] = Field(None, description="Specific agent instance ID")
    agent_version: str = Field(default="1.0", description="Version of the agent")
    
    # Task Context
    task_id: str = Field(..., description="The ID of the task the agent was working on")
    workflow_id: str = Field(..., description="The workflow instance ID")
    correlation_id: str = Field(..., description="The correlation ID of the workflow")
    
    # Action Details
    action_type: ActionType = Field(..., description="Type of action performed")
    action_name: str = Field(..., description="Specific name of the action")
    action_description: str = Field(..., description="Human-readable description of the action")
    
    # Action Content
    action_taken: Dict[str, Any] = Field(
        ..., 
        description="A structured description of the action performed"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters used for the action"
    )
    
    # Results & Outcomes
    action_result: ActionResult = Field(..., description="The outcome of the action")
    result_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed result data"
    )
    success_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Quantitative success metrics"
    )
    
    # Impact Assessment
    impact_level: ImpactLevel = Field(default=ImpactLevel.LOW, description="Assessed impact level")
    side_effects: List[str] = Field(
        default_factory=list,
        description="Observed side effects of the action"
    )
    unintended_consequences: List[str] = Field(
        default_factory=list,
        description="Unintended consequences observed"
    )
    
    # Context & Decision Making
    context_used: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Context information used to make the decision"
    )
    context_sources: List[ContextSource] = Field(
        default_factory=list,
        description="Sources of context information"
    )
    context_used_hashes: List[str] = Field(
        default_factory=list,
        description="A list of hashes or IDs of the memory entries used to make this decision"
    )
    
    # Decision Quality
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in the action (0.0 to 1.0)"
    )
    certainty: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Certainty about the decision context (0.0 to 1.0)"
    )
    risk_assessment: Dict[str, Any] = Field(
        default_factory=dict,
        description="Risk assessment performed before action"
    )
    
    # Learning & Feedback
    feedback_received: Optional[Dict[str, Any]] = Field(
        None,
        description="Feedback received about the action"
    )
    lessons_learned: List[str] = Field(
        default_factory=list,
        description="Lessons learned from this action"
    )
    improvement_suggestions: List[str] = Field(
        default_factory=list,
        description="Suggestions for improvement"
    )
    
    # Error Information (if applicable)
    error_details: Optional[Dict[str, Any]] = Field(
        None,
        description="Error details if the action failed"
    )
    recovery_actions: List[str] = Field(
        default_factory=list,
        description="Recovery actions taken after failure"
    )
    
    # Relationships
    triggered_by: Optional[str] = Field(None, description="Event ID that triggered this action")
    triggers: List[str] = Field(
        default_factory=list,
        description="Event IDs of actions triggered by this action"
    )
    related_actions: List[str] = Field(
        default_factory=list,
        description="Related action event IDs"
    )
    
    # Memory Integration
    memory_tier_written: Optional[str] = Field(
        None,
        description="Which memory tier this event was written to"
    )
    signature_hash: Optional[str] = Field(
        None,
        description="Hash of the topological signature if generated"
    )
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('confidence', 'certainty')
    def validate_scores(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Score must be between 0.0 and 1.0")
        return v
    
    def generate_signature_hash(self) -> str:
        """Generate a hash of the action for deduplication and reference."""
        content = f"{self.agent_id}:{self.action_type.value}:{self.action_name}:{self.timestamp_utc}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def add_context_reference(self, source: ContextSource, reference_id: str, data: Dict[str, Any]) -> None:
        """Add a reference to context used in the decision."""
        self.context_sources.append(source)
        self.context_used_hashes.append(reference_id)
        self.context_used.append({
            'source': source.value,
            'reference_id': reference_id,
            'data': data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    
    def add_feedback(self, feedback_type: str, feedback_data: Dict[str, Any]) -> None:
        """Add feedback about the action."""
        if self.feedback_received is None:
            self.feedback_received = {}
        
        self.feedback_received[feedback_type] = {
            'data': feedback_data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def calculate_success_score(self) -> float:
        """Calculate an overall success score based on result and metrics."""
        base_score = {
            ActionResult.SUCCESS: 1.0,
            ActionResult.PARTIAL_SUCCESS: 0.7,
            ActionResult.FAILURE: 0.0,
            ActionResult.TIMEOUT: 0.2,
            ActionResult.CANCELLED: 0.0,
            ActionResult.SKIPPED: 0.0,
            ActionResult.RETRY_NEEDED: 0.3
        }.get(self.action_result, 0.5)
        
        # Adjust based on confidence and certainty
        confidence_factor = (self.confidence + self.certainty) / 2
        
        # Adjust based on impact (higher impact actions need higher success)
        impact_factor = {
            ImpactLevel.NONE: 1.0,
            ImpactLevel.LOW: 1.0,
            ImpactLevel.MEDIUM: 0.9,
            ImpactLevel.HIGH: 0.8,
            ImpactLevel.CRITICAL: 0.7
        }.get(self.impact_level, 1.0)
        
        return base_score * confidence_factor * impact_factor
    
    def to_memory_signature(self) -> Dict[str, Any]:
        """Convert to a format suitable for storage in the memory system."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp_utc,
            'agent_id': self.agent_id,
            'agent_role': self.agent_role,
            'action_type': self.action_type.value,
            'action_result': self.action_result.value,
            'confidence': self.confidence,
            'impact_level': self.impact_level.value,
            'success_score': self.calculate_success_score(),
            'context_sources': [source.value for source in self.context_sources],
            'signature_hash': self.signature_hash or self.generate_signature_hash(),
            'metadata': {
                'task_id': self.task_id,
                'correlation_id': self.correlation_id,
                'duration_ms': self.duration_ms,
                'tags': self.tags
            }
        }
    
    def __str__(self) -> str:
        return f"ActionEvent[{self.action_type.value}] {self.agent_role}:{self.agent_id} -> {self.action_result.value}"


class ActionBatch(BaseModel):
    """A batch of related actions for efficient logging."""
    
    batch_id: str = Field(
        default_factory=lambda: f"batch_{uuid.uuid4().hex[:8]}",
        description="Unique batch identifier"
    )
    
    actions: List[AgentActionEvent] = Field(..., description="Actions in this batch")
    
    batch_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="When the batch was created"
    )
    
    correlation_id: str = Field(..., description="Correlation ID for the batch")
    
    def add_action(self, action: AgentActionEvent) -> None:
        """Add an action to the batch."""
        self.actions.append(action)
    
    def get_success_rate(self) -> float:
        """Calculate the success rate for actions in this batch."""
        if not self.actions:
            return 0.0
        
        successful = sum(1 for action in self.actions if action.action_result == ActionResult.SUCCESS)
        return successful / len(self.actions)
    
    def get_average_confidence(self) -> float:
        """Calculate average confidence across actions."""
        if not self.actions:
            return 0.0
        
        return sum(action.confidence for action in self.actions) / len(self.actions)


class LearningFeedback(BaseModel):
    """Feedback for improving agent performance."""
    
    feedback_id: str = Field(
        default_factory=lambda: f"feedback_{uuid.uuid4().hex[:8]}",
        description="Unique feedback identifier"
    )
    
    action_event_id: str = Field(..., description="ID of the action this feedback relates to")
    
    feedback_type: str = Field(..., description="Type of feedback (human, system, outcome)")
    feedback_source: str = Field(..., description="Source of the feedback")
    
    rating: Optional[float] = Field(None, ge=0.0, le=1.0, description="Numerical rating (0.0 to 1.0)")
    comments: Optional[str] = Field(None, description="Textual feedback")
    
    improvement_areas: List[str] = Field(
        default_factory=list,
        description="Areas identified for improvement"
    )
    
    positive_aspects: List[str] = Field(
        default_factory=list,
        description="Positive aspects to reinforce"
    )
    
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="When the feedback was provided"
    )
