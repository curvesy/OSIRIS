"""
🔄 Observability Context Managers - Latest 2025 Patterns
Professional context management for neural observability system.
"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

try:
    from .config import ObservabilityConfig
except ImportError:
    # Fallback for direct import
    from config import ObservabilityConfig


@dataclass
class ObservabilityContext:
    """
    Context object for observability operations.
    
    Provides correlation context across all observability systems with:
    - Workflow identification and metadata
    - Timing information
    - Status tracking
    - Error context
    - Configuration access
    """
    
    # Core identification
    workflow_id: str
    workflow_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    config: Optional[ObservabilityConfig] = None
    
    # Timing information
    start_time: float = field(default_factory=time.time)
    duration: Optional[float] = None
    
    # Status tracking
    status: str = "running"
    error: Optional[str] = None
    
    # Correlation context
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    
    # Component-specific contexts
    opentelemetry_span: Optional[Any] = None
    langsmith_run: Optional[Any] = None
    prometheus_labels: Dict[str, str] = field(default_factory=dict)
    
    # Additional context
    tags: list = field(default_factory=list)
    custom_attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization setup."""
        
        # Set default prometheus labels
        if not self.prometheus_labels:
            self.prometheus_labels = {
                "workflow_type": self.workflow_type,
                "organism_id": self.config.organism_id if self.config else "unknown"
            }
        
        # Add default tags
        if self.config:
            self.tags.extend([
                f"generation:{self.config.organism_generation}",
                f"environment:{self.config.deployment_environment}",
                "neural-observability"
            ])
    
    def add_attribute(self, key: str, value: Any) -> None:
        """Add custom attribute to context."""
        self.custom_attributes[key] = value
    
    def add_tag(self, tag: str) -> None:
        """Add tag to context."""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def set_error(self, error: str) -> None:
        """Set error information."""
        self.error = error
        self.status = "failed"
        self.add_tag("error")
    
    def set_success(self) -> None:
        """Mark context as successful."""
        self.status = "success"
        self.add_tag("success")
    
    def calculate_duration(self) -> float:
        """Calculate and set duration."""
        if self.duration is None:
            self.duration = time.time() - self.start_time
        return self.duration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        
        return {
            "workflow_id": self.workflow_id,
            "workflow_type": self.workflow_type,
            "metadata": self.metadata,
            "start_time": self.start_time,
            "duration": self.duration,
            "status": self.status,
            "error": self.error,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "tags": self.tags,
            "custom_attributes": self.custom_attributes,
            "prometheus_labels": self.prometheus_labels,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def get_correlation_id(self) -> str:
        """Get correlation ID for log correlation."""
        return f"{self.workflow_id}:{self.trace_id or 'no-trace'}"
    
    def is_successful(self) -> bool:
        """Check if context represents successful operation."""
        return self.status == "success"
    
    def is_failed(self) -> bool:
        """Check if context represents failed operation."""
        return self.status == "failed"
    
    def is_running(self) -> bool:
        """Check if context represents running operation."""
        return self.status == "running"


@dataclass
class AgentContext:
    """
    Context object for agent operations.
    
    Specialized context for individual agent/tool calls with:
    - Agent identification
    - Tool information
    - Input/output tracking
    - Performance metrics
    """
    
    # Core identification
    agent_name: str
    tool_name: str
    workflow_context: Optional[ObservabilityContext] = None
    
    # Input/output
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Optional[Dict[str, Any]] = None
    
    # Timing
    start_time: float = field(default_factory=time.time)
    duration: Optional[float] = None
    
    # Status
    status: str = "running"
    error: Optional[str] = None
    
    # Performance metrics
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None
    
    # Correlation
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    
    def set_outputs(self, outputs: Dict[str, Any]) -> None:
        """Set operation outputs."""
        self.outputs = outputs
    
    def set_performance_metrics(self, tokens_used: int = None, cost_usd: float = None) -> None:
        """Set performance metrics."""
        if tokens_used is not None:
            self.tokens_used = tokens_used
        if cost_usd is not None:
            self.cost_usd = cost_usd
    
    def calculate_duration(self) -> float:
        """Calculate and set duration."""
        if self.duration is None:
            self.duration = time.time() - self.start_time
        return self.duration
    
    def set_error(self, error: str) -> None:
        """Set error information."""
        self.error = error
        self.status = "failed"
    
    def set_success(self) -> None:
        """Mark operation as successful."""
        self.status = "success"
    
    def get_workflow_id(self) -> str:
        """Get workflow ID from parent context."""
        if self.workflow_context:
            return self.workflow_context.workflow_id
        return "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        
        return {
            "agent_name": self.agent_name,
            "tool_name": self.tool_name,
            "workflow_id": self.get_workflow_id(),
            "inputs": self.inputs,
            "outputs": self.outputs,
            "start_time": self.start_time,
            "duration": self.duration,
            "status": self.status,
            "error": self.error,
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@dataclass
class LLMUsageContext:
    """
    Context object for LLM usage tracking.
    
    Specialized context for LLM operations with:
    - Model information
    - Token usage
    - Cost tracking
    - Performance metrics
    """
    
    # Model information
    model_name: str
    provider: str = "unknown"
    
    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    # Performance
    latency_seconds: float = 0.0
    throughput_tokens_per_second: Optional[float] = None
    
    # Cost
    cost_usd: Optional[float] = None
    cost_per_token: Optional[float] = None
    
    # Context
    workflow_id: Optional[str] = None
    agent_name: Optional[str] = None
    tool_name: Optional[str] = None
    
    # Timing
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def __post_init__(self):
        """Post-initialization calculations."""
        
        # Calculate total tokens
        self.total_tokens = self.input_tokens + self.output_tokens
        
        # Calculate throughput
        if self.latency_seconds > 0 and self.total_tokens > 0:
            self.throughput_tokens_per_second = self.total_tokens / self.latency_seconds
        
        # Calculate cost per token
        if self.cost_usd is not None and self.total_tokens > 0:
            self.cost_per_token = self.cost_usd / self.total_tokens
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        
        return {
            "model_name": self.model_name,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_seconds": self.latency_seconds,
            "throughput_tokens_per_second": self.throughput_tokens_per_second,
            "cost_usd": self.cost_usd,
            "cost_per_token": self.cost_per_token,
            "workflow_id": self.workflow_id,
            "agent_name": self.agent_name,
            "tool_name": self.tool_name,
            "timestamp": self.timestamp
        }


def create_workflow_context(
    workflow_id: str,
    workflow_type: str,
    config: ObservabilityConfig,
    **metadata
) -> ObservabilityContext:
    """
    Create workflow observability context.
    
    Args:
        workflow_id: Unique workflow identifier
        workflow_type: Type of workflow
        config: Observability configuration
        **metadata: Additional metadata
        
    Returns:
        ObservabilityContext: Configured context
    """
    
    return ObservabilityContext(
        workflow_id=workflow_id,
        workflow_type=workflow_type,
        config=config,
        metadata=metadata
    )


def create_agent_context(
    agent_name: str,
    tool_name: str,
    workflow_context: Optional[ObservabilityContext] = None,
    **inputs
) -> AgentContext:
    """
    Create agent observability context.
    
    Args:
        agent_name: Name of the agent
        tool_name: Name of the tool
        workflow_context: Parent workflow context
        **inputs: Tool inputs
        
    Returns:
        AgentContext: Configured context
    """
    
    return AgentContext(
        agent_name=agent_name,
        tool_name=tool_name,
        workflow_context=workflow_context,
        inputs=inputs
    )


def create_llm_usage_context(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    latency_seconds: float,
    cost_usd: Optional[float] = None,
    **context_info
) -> LLMUsageContext:
    """
    Create LLM usage context.
    
    Args:
        model_name: Name of the LLM model
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        latency_seconds: Response latency
        cost_usd: Cost in USD (optional)
        **context_info: Additional context information
        
    Returns:
        LLMUsageContext: Configured context
    """
    
    return LLMUsageContext(
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_seconds=latency_seconds,
        cost_usd=cost_usd,
        **context_info
    )
