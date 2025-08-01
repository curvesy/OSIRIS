"""
Agent Observability with OpenTelemetry GenAI Semantic Conventions

Implements comprehensive instrumentation for AI agents following
the latest OpenTelemetry GenAI standards for 2025.
"""

from typing import Dict, Any, Optional, Callable, List
from functools import wraps
import time
import asyncio
from contextlib import asynccontextmanager

from opentelemetry import trace, metrics, baggage, context
from opentelemetry.trace import Status, StatusCode, Link
from opentelemetry.metrics import CallbackOptions, Observation
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.semconv.trace import SpanAttributes
import structlog

# GenAI Semantic Conventions (2025 standards)
class GenAIAttributes:
    """OpenTelemetry GenAI semantic conventions for agents."""
    
    # Agent attributes
    AGENT_NAME = "gen_ai.agent.name"
    AGENT_ID = "gen_ai.agent.id"
    AGENT_TYPE = "gen_ai.agent.type"
    AGENT_VERSION = "gen_ai.agent.version"
    AGENT_TASK = "gen_ai.agent.task"
    AGENT_STEP_NUMBER = "gen_ai.agent.step.number"
    AGENT_STEP_NAME = "gen_ai.agent.step.name"
    AGENT_DECISION = "gen_ai.agent.decision"
    AGENT_DECISION_REASON = "gen_ai.agent.decision.reason"
    
    # Tool attributes
    AGENT_TOOL_NAME = "gen_ai.agent.tool.name"
    AGENT_TOOL_INPUT = "gen_ai.agent.tool.input"
    AGENT_TOOL_OUTPUT = "gen_ai.agent.tool.output"
    AGENT_TOOL_DURATION_MS = "gen_ai.agent.tool.duration_ms"
    
    # LLM attributes
    LLM_MODEL_NAME = "gen_ai.llm.model.name"
    LLM_TEMPERATURE = "gen_ai.llm.temperature"
    LLM_MAX_TOKENS = "gen_ai.llm.max_tokens"
    LLM_PROMPT = "gen_ai.llm.prompt"
    LLM_COMPLETION = "gen_ai.llm.completion"
    LLM_PROMPT_TOKENS = "gen_ai.llm.prompt_tokens"
    LLM_COMPLETION_TOKENS = "gen_ai.llm.completion_tokens"
    LLM_TOTAL_TOKENS = "gen_ai.llm.total_tokens"
    LLM_LATENCY_MS = "gen_ai.llm.latency_ms"
    
    # Quality attributes
    RESPONSE_QUALITY_SCORE = "gen_ai.response.quality_score"
    RESPONSE_RELEVANCE_SCORE = "gen_ai.response.relevance_score"
    RESPONSE_SAFETY_SCORE = "gen_ai.response.safety_score"


class AgentMetrics:
    """Metrics collection for agents."""
    
    def __init__(self, meter: metrics.Meter, agent_name: str):
        """Initialize metrics for an agent."""
        self.agent_name = agent_name
        
        # Token metrics
        self.token_counter = meter.create_counter(
            name="gen_ai.agent.tokens",
            description="Total tokens used by agent",
            unit="tokens"
        )
        
        # Decision metrics
        self.decision_counter = meter.create_counter(
            name="gen_ai.agent.decisions",
            description="Number of decisions made by agent",
            unit="1"
        )
        
        # Tool usage metrics
        self.tool_usage_counter = meter.create_counter(
            name="gen_ai.agent.tool_invocations",
            description="Number of tool invocations",
            unit="1"
        )
        
        # Quality metrics
        self.quality_histogram = meter.create_histogram(
            name="gen_ai.agent.response_quality",
            description="Response quality scores",
            unit="score"
        )
        
        # Cost metrics
        self.cost_counter = meter.create_counter(
            name="gen_ai.agent.cost",
            description="Estimated cost of agent operations",
            unit="USD"
        )
    
    def record_tokens(self, prompt_tokens: int, completion_tokens: int, model: str):
        """Record token usage."""
        total = prompt_tokens + completion_tokens
        attributes = {
            "agent.name": self.agent_name,
            "model": model,
            "token.type": "total"
        }
        self.token_counter.add(total, attributes)
        
        # Estimate cost (example rates)
        cost_per_1k = {"gpt-4": 0.03, "gpt-3.5-turbo": 0.002}.get(model, 0.01)
        estimated_cost = (total / 1000) * cost_per_1k
        self.cost_counter.add(estimated_cost, attributes)
    
    def record_decision(self, decision: str, reason: str):
        """Record a decision made by the agent."""
        self.decision_counter.add(1, {
            "agent.name": self.agent_name,
            "decision": decision,
            "has_reason": bool(reason)
        })
    
    def record_tool_usage(self, tool_name: str, duration_ms: float, success: bool):
        """Record tool usage."""
        self.tool_usage_counter.add(1, {
            "agent.name": self.agent_name,
            "tool": tool_name,
            "success": success
        })
    
    def record_quality(self, quality_score: float, relevance_score: float = None):
        """Record response quality metrics."""
        self.quality_histogram.record(quality_score, {
            "agent.name": self.agent_name,
            "metric": "quality"
        })
        if relevance_score is not None:
            self.quality_histogram.record(relevance_score, {
                "agent.name": self.agent_name,
                "metric": "relevance"
            })


class AgentInstrumentor:
    """Instrumentor for adding observability to agents."""
    
    def __init__(self):
        """Initialize the instrumentor."""
        self.tracer = trace.get_tracer(__name__)
        self.meter = metrics.get_meter(__name__)
        self.propagator = TraceContextTextMapPropagator()
        self.logger = structlog.get_logger()
    
    def instrument_agent(self, agent: Any) -> Any:
        """Add instrumentation to an agent."""
        # Add metrics
        agent._metrics = AgentMetrics(self.meter, agent.name)
        
        # Wrap key methods
        agent._original_process = agent._process
        agent._process = self._wrap_process(agent)
        
        if hasattr(agent, '_execute_step'):
            agent._original_execute_step = agent._execute_step
            agent._execute_step = self._wrap_execute_step(agent)
        
        return agent
    
    def _wrap_process(self, agent):
        """Wrap the main process method with instrumentation."""
        @wraps(agent._original_process)
        async def wrapped(input_data):
            with self.tracer.start_as_current_span(
                f"agent.{agent.name}.process",
                kind=trace.SpanKind.SERVER,
                attributes={
                    GenAIAttributes.AGENT_NAME: agent.name,
                    GenAIAttributes.AGENT_TYPE: type(agent).__name__,
                    GenAIAttributes.LLM_MODEL_NAME: agent.config.model,
                    GenAIAttributes.LLM_TEMPERATURE: agent.config.temperature
                }
            ) as span:
                start_time = time.time()
                
                try:
                    # Add input to span
                    span.set_attribute(GenAIAttributes.AGENT_TASK, str(input_data)[:1000])
                    
                    # Execute
                    result = await agent._original_process(input_data)
                    
                    # Record success
                    duration_ms = (time.time() - start_time) * 1000
                    span.set_attribute("duration_ms", duration_ms)
                    span.set_status(Status(StatusCode.OK))
                    
                    return result
                    
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        return wrapped
    
    def _wrap_execute_step(self, agent):
        """Wrap step execution with instrumentation."""
        @wraps(agent._original_execute_step)
        async def wrapped(state, step_name):
            with self.tracer.start_as_current_span(
                f"agent.{agent.name}.step.{step_name}",
                attributes={
                    GenAIAttributes.AGENT_NAME: agent.name,
                    GenAIAttributes.AGENT_STEP_NAME: step_name,
                    GenAIAttributes.AGENT_STEP_NUMBER: len(state.messages)
                }
            ) as span:
                try:
                    # Execute step
                    new_state = await agent._original_execute_step(state, step_name)
                    
                    # Record decision if present
                    if hasattr(new_state, 'last_decision'):
                        span.set_attribute(
                            GenAIAttributes.AGENT_DECISION,
                            new_state.last_decision
                        )
                        agent._metrics.record_decision(
                            new_state.last_decision,
                            getattr(new_state, 'decision_reason', '')
                        )
                    
                    return new_state
                    
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        return wrapped
    
    @asynccontextmanager
    async def trace_llm_call(
        self,
        agent_name: str,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = None
    ):
        """Context manager for tracing LLM calls."""
        with self.tracer.start_as_current_span(
            f"llm.{model}.completion",
            attributes={
                GenAIAttributes.AGENT_NAME: agent_name,
                GenAIAttributes.LLM_MODEL_NAME: model,
                GenAIAttributes.LLM_TEMPERATURE: temperature,
                GenAIAttributes.LLM_MAX_TOKENS: max_tokens or -1,
                GenAIAttributes.LLM_PROMPT: prompt[:500]  # Truncate for safety
            }
        ) as span:
            start_time = time.time()
            
            try:
                yield span
                
                # Success - record latency
                latency_ms = (time.time() - start_time) * 1000
                span.set_attribute(GenAIAttributes.LLM_LATENCY_MS, latency_ms)
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    
    @asynccontextmanager
    async def trace_tool_call(self, agent_name: str, tool_name: str, tool_input: Any):
        """Context manager for tracing tool calls."""
        with self.tracer.start_as_current_span(
            f"tool.{tool_name}",
            attributes={
                GenAIAttributes.AGENT_NAME: agent_name,
                GenAIAttributes.AGENT_TOOL_NAME: tool_name,
                GenAIAttributes.AGENT_TOOL_INPUT: str(tool_input)[:500]
            }
        ) as span:
            start_time = time.time()
            
            try:
                yield span
                
                # Success
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute(GenAIAttributes.AGENT_TOOL_DURATION_MS, duration_ms)
                
                # Record metrics
                if hasattr(self, '_metrics'):
                    self._metrics.record_tool_usage(tool_name, duration_ms, True)
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                
                # Record failure
                if hasattr(self, '_metrics'):
                    duration_ms = (time.time() - start_time) * 1000
                    self._metrics.record_tool_usage(tool_name, duration_ms, False)
                
                raise
    
    def extract_context(self, carrier: Dict[str, str]) -> context.Context:
        """Extract trace context from carrier."""
        return self.propagator.extract(carrier)
    
    def inject_context(self, carrier: Dict[str, str]) -> None:
        """Inject current trace context into carrier."""
        self.propagator.inject(carrier)
    
    def create_span_link(self, trace_id: str, span_id: str) -> Link:
        """Create a link to another span."""
        return Link(
            context={
                "trace_id": trace_id,
                "span_id": span_id
            }
        )