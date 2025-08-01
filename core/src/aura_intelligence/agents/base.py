"""
Base Agent Implementation with Built-in Observability

Provides the foundation for all AURA agents with automatic
instrumentation, error handling, and state management.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import uuid

from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import structlog

from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode
from opentelemetry.metrics import CallbackOptions, Observation

from aura_common.atomic.base import AtomicComponent
from aura_common.atomic.base.exceptions import ComponentError

# Type variables for generic agent implementation
TInput = TypeVar('TInput')
TOutput = TypeVar('TOutput')
TState = TypeVar('TState', bound='AgentState')

# Get tracer and meter
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

# Create metrics
agent_invocation_counter = meter.create_counter(
    name="agent.invocations",
    description="Number of agent invocations",
    unit="1"
)

agent_duration_histogram = meter.create_histogram(
    name="agent.duration",
    description="Agent execution duration",
    unit="ms"
)

agent_error_counter = meter.create_counter(
    name="agent.errors",
    description="Number of agent errors",
    unit="1"
)


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    
    name: str
    model: str = "gpt-4"
    temperature: float = 0.7
    max_retries: int = 3
    timeout_seconds: int = 30
    enable_memory: bool = True
    enable_tools: bool = True
    
    def validate(self) -> None:
        """Validate agent configuration."""
        if not self.name:
            raise ValueError("Agent name is required")
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        if self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")


class AgentState(BaseModel):
    """Base state for agent workflows."""
    
    # Core state fields
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    
    # Workflow control
    current_step: str = "start"
    next_step: Optional[str] = None
    completed: bool = False
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Error tracking
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    
    def add_message(self, role: str, content: str, **kwargs) -> None:
        """Add a message to the conversation."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        })
        self.updated_at = datetime.utcnow()
    
    def add_error(self, error: Exception, step: str) -> None:
        """Record an error."""
        self.errors.append({
            "step": step,
            "error": str(error),
            "type": type(error).__name__,
            "timestamp": datetime.utcnow().isoformat()
        })
        self.updated_at = datetime.utcnow()
    
    def get_last_message(self) -> Optional[Dict[str, Any]]:
        """Get the most recent message."""
        return self.messages[-1] if self.messages else None


class AgentBase(AtomicComponent[TInput, TOutput, AgentConfig], ABC, Generic[TInput, TOutput, TState]):
    """
    Base class for all AURA agents with built-in observability.
    
    Features:
    - Automatic OpenTelemetry instrumentation
    - State management with LangGraph
    - Error handling and retries
    - Structured logging
    - Performance metrics
    """
    
    def __init__(self, config: AgentConfig):
        """Initialize the agent with configuration."""
        super().__init__(config.name, config)
        self.logger = structlog.get_logger().bind(agent=config.name)
        self.state_graph: Optional[StateGraph] = None
        
        # Initialize metrics
        self._invocation_count = 0
        self._error_count = 0
        self._total_duration_ms = 0
    
    def _validate_config(self) -> None:
        """Validate agent configuration."""
        self.config.validate()
    
    @abstractmethod
    def build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow for this agent.
        
        Returns:
            StateGraph defining the agent's workflow
        """
        pass
    
    @abstractmethod
    async def _execute_step(self, state: TState, step_name: str) -> TState:
        """
        Execute a specific step in the agent workflow.
        
        Args:
            state: Current agent state
            step_name: Name of the step to execute
            
        Returns:
            Updated state after step execution
        """
        pass
    
    async def _process(self, input_data: TInput) -> TOutput:
        """
        Process input through the agent workflow.
        
        Args:
            input_data: Input to process
            
        Returns:
            Agent output
        """
        # Create initial state
        state = self._create_initial_state(input_data)
        
        # Build graph if not already built
        if self.state_graph is None:
            self.state_graph = self.build_graph()
            self.compiled_graph = self.state_graph.compile()
        
        # Execute workflow with tracing
        with tracer.start_as_current_span(
            f"agent.{self.name}.workflow",
            attributes={
                "agent.name": self.name,
                "agent.model": self.config.model,
                "agent.id": state.agent_id
            }
        ) as span:
            try:
                # Run the graph
                final_state = await self._run_graph(state)
                
                # Extract output
                output = self._extract_output(final_state)
                
                # Update metrics
                self._invocation_count += 1
                agent_invocation_counter.add(
                    1,
                    {"agent.name": self.name, "status": "success"}
                )
                
                return output
                
            except Exception as e:
                self._error_count += 1
                agent_error_counter.add(
                    1,
                    {"agent.name": self.name, "error.type": type(e).__name__}
                )
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    
    async def _run_graph(self, state: TState) -> TState:
        """Run the compiled graph with the given state."""
        # For now, manually execute steps (LangGraph integration pending)
        current_state = state
        
        while not current_state.completed:
            step_name = current_state.current_step
            
            with tracer.start_as_current_span(
                f"agent.{self.name}.step.{step_name}",
                attributes={
                    "agent.name": self.name,
                    "step.name": step_name
                }
            ) as span:
                try:
                    # Execute the step
                    current_state = await self._execute_step(current_state, step_name)
                    
                    # Check for completion
                    if current_state.next_step is None or current_state.next_step == END:
                        current_state.completed = True
                    else:
                        current_state.current_step = current_state.next_step
                        
                except Exception as e:
                    current_state.add_error(e, step_name)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    
                    # Determine if we should retry or fail
                    if len(current_state.errors) >= self.config.max_retries:
                        raise ComponentError(
                            f"Agent {self.name} failed after {len(current_state.errors)} attempts",
                            component_name=self.name,
                            details={"errors": current_state.errors}
                        )
        
        return current_state
    
    @abstractmethod
    def _create_initial_state(self, input_data: TInput) -> TState:
        """Create the initial state from input data."""
        pass
    
    @abstractmethod
    def _extract_output(self, final_state: TState) -> TOutput:
        """Extract the output from the final state."""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Check agent health."""
        health = await super().health_check()
        
        # Add agent-specific metrics
        health.update({
            "invocation_count": self._invocation_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(1, self._invocation_count),
            "avg_duration_ms": self._total_duration_ms / max(1, self._invocation_count)
        })
        
        return health
    
    def get_capabilities(self) -> List[str]:
        """Get list of agent capabilities."""
        return ["base_agent", "observability", "state_management"]