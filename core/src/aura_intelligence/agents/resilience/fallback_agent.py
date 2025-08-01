"""
Fallback Agent Pattern for Graceful Degradation

Provides fallback mechanisms when primary agents fail,
ensuring the system remains operational with reduced functionality.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Callable, TypeVar, Generic
from abc import abstractmethod
import asyncio
from datetime import timedelta

from ..base import AgentBase, AgentConfig, AgentState
from ..observability import AgentInstrumentor, GenAIAttributes
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError

from opentelemetry import trace
import structlog

# Type variables
TInput = TypeVar('TInput')
TOutput = TypeVar('TOutput')
TState = TypeVar('TState', bound=AgentState)

tracer = trace.get_tracer(__name__)


class FallbackStrategy(Enum):
    """Strategies for fallback behavior."""
    
    CACHED_RESPONSE = "cached_response"      # Return cached result
    SIMPLIFIED_MODEL = "simplified_model"     # Use simpler/cheaper model
    DEFAULT_RESPONSE = "default_response"     # Return predefined default
    ALTERNATIVE_AGENT = "alternative_agent"   # Use different agent
    PARTIAL_RESPONSE = "partial_response"     # Return partial results


class FallbackAgent(AgentBase[TInput, TOutput, TState], Generic[TInput, TOutput, TState]):
    """
    Agent wrapper that provides fallback capabilities.
    
    Features:
    - Multiple fallback strategies
    - Circuit breaker integration
    - Response caching
    - Graceful degradation
    """
    
    def __init__(
        self,
        primary_agent: AgentBase[TInput, TOutput, TState],
        fallback_strategy: FallbackStrategy,
        config: Optional[AgentConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    ):
        """
        Initialize fallback agent.
        
        Args:
            primary_agent: The main agent to wrap
            fallback_strategy: Strategy to use on failure
            config: Agent configuration
            circuit_breaker_config: Circuit breaker configuration
        """
        if config is None:
            config = AgentConfig(
                name=f"{primary_agent.name}_fallback",
                model=primary_agent.config.model,
                temperature=primary_agent.config.temperature
            )
        
        super().__init__(config)
        
        self.primary_agent = primary_agent
        self.fallback_strategy = fallback_strategy
        self.logger = structlog.get_logger().bind(
            agent=config.name,
            strategy=fallback_strategy.value
        )
        
        # Set up circuit breaker
        if circuit_breaker_config is None:
            circuit_breaker_config = CircuitBreakerConfig(
                name=f"{primary_agent.name}_circuit",
                failure_threshold=3,
                timeout=timedelta(seconds=30)
            )
        self.circuit_breaker = CircuitBreaker(circuit_breaker_config)
        
        # Cache for fallback responses
        self.response_cache: Dict[str, TOutput] = {}
        self.cache_max_size = 100
        
        # Alternative agents for ALTERNATIVE_AGENT strategy
        self.alternative_agents: List[AgentBase[TInput, TOutput, TState]] = []
        
        # Default responses for DEFAULT_RESPONSE strategy
        self.default_responses: Dict[str, TOutput] = {}
        
        # Instrumentor for observability
        self.instrumentor = AgentInstrumentor()
    
    def add_alternative_agent(self, agent: AgentBase[TInput, TOutput, TState]) -> None:
        """Add an alternative agent for fallback."""
        self.alternative_agents.append(agent)
        self.logger.info(
            "Added alternative agent",
            alternative_agent=agent.name
        )
    
    def set_default_response(self, key: str, response: TOutput) -> None:
        """Set a default response for a given key."""
        self.default_responses[key] = response
    
    def build_graph(self):
        """Use primary agent's graph."""
        return self.primary_agent.build_graph()
    
    async def _execute_step(self, state: TState, step_name: str) -> TState:
        """Execute step with fallback."""
        try:
            # Try primary agent through circuit breaker
            return await self.circuit_breaker.call(
                self.primary_agent._execute_step,
                state,
                step_name
            )
        except (CircuitBreakerError, Exception) as e:
            # Primary failed, use fallback
            return await self._execute_fallback(state, step_name, e)
    
    async def _process(self, input_data: TInput) -> TOutput:
        """Process with fallback handling."""
        with tracer.start_as_current_span(
            f"fallback_agent.{self.name}",
            attributes={
                GenAIAttributes.AGENT_NAME: self.name,
                "fallback.strategy": self.fallback_strategy.value,
                "circuit.state": self.circuit_breaker.get_state().value
            }
        ) as span:
            try:
                # Try primary agent
                result = await self.circuit_breaker.call(
                    self.primary_agent._process,
                    input_data
                )
                
                # Cache successful result
                cache_key = self._get_cache_key(input_data)
                self._update_cache(cache_key, result)
                
                span.set_attribute("fallback.used", False)
                return result
                
            except (CircuitBreakerError, Exception) as e:
                self.logger.warning(
                    "Primary agent failed, using fallback",
                    error=str(e),
                    error_type=type(e).__name__
                )
                
                span.set_attribute("fallback.used", True)
                span.set_attribute("fallback.reason", str(e))
                
                # Execute fallback strategy
                return await self._execute_fallback_strategy(input_data, e)
    
    async def _execute_fallback_strategy(self, input_data: TInput, error: Exception) -> TOutput:
        """Execute the configured fallback strategy."""
        if self.fallback_strategy == FallbackStrategy.CACHED_RESPONSE:
            return await self._fallback_cached_response(input_data)
        
        elif self.fallback_strategy == FallbackStrategy.SIMPLIFIED_MODEL:
            return await self._fallback_simplified_model(input_data)
        
        elif self.fallback_strategy == FallbackStrategy.DEFAULT_RESPONSE:
            return await self._fallback_default_response(input_data)
        
        elif self.fallback_strategy == FallbackStrategy.ALTERNATIVE_AGENT:
            return await self._fallback_alternative_agent(input_data)
        
        elif self.fallback_strategy == FallbackStrategy.PARTIAL_RESPONSE:
            return await self._fallback_partial_response(input_data)
        
        else:
            raise ValueError(f"Unknown fallback strategy: {self.fallback_strategy}")
    
    async def _fallback_cached_response(self, input_data: TInput) -> TOutput:
        """Return cached response if available."""
        cache_key = self._get_cache_key(input_data)
        
        if cache_key in self.response_cache:
            self.logger.info("Returning cached response", cache_key=cache_key)
            return self.response_cache[cache_key]
        
        # No cached response, try partial response
        return await self._fallback_partial_response(input_data)
    
    async def _fallback_simplified_model(self, input_data: TInput) -> TOutput:
        """Use a simplified model configuration."""
        # Create simplified agent with lower cost model
        simplified_config = AgentConfig(
            name=f"{self.name}_simplified",
            model="gpt-3.5-turbo",  # Cheaper model
            temperature=0.3,         # Lower temperature
            max_retries=1           # Fewer retries
        )
        
        # Create temporary simplified agent
        simplified_agent = type(self.primary_agent)(simplified_config)
        
        try:
            return await simplified_agent._process(input_data)
        except Exception as e:
            self.logger.error(
                "Simplified model also failed",
                error=str(e)
            )
            return await self._fallback_partial_response(input_data)
    
    async def _fallback_default_response(self, input_data: TInput) -> TOutput:
        """Return a default response."""
        # Try to find matching default
        for key, response in self.default_responses.items():
            if key in str(input_data):
                return response
        
        # No matching default, return generic partial response
        return await self._fallback_partial_response(input_data)
    
    async def _fallback_alternative_agent(self, input_data: TInput) -> TOutput:
        """Try alternative agents in order."""
        for agent in self.alternative_agents:
            try:
                self.logger.info(
                    "Trying alternative agent",
                    agent=agent.name
                )
                return await agent._process(input_data)
            except Exception as e:
                self.logger.warning(
                    "Alternative agent failed",
                    agent=agent.name,
                    error=str(e)
                )
                continue
        
        # All alternatives failed
        return await self._fallback_partial_response(input_data)
    
    @abstractmethod
    async def _fallback_partial_response(self, input_data: TInput) -> TOutput:
        """
        Generate a partial response as last resort.
        
        This method must be implemented by subclasses to provide
        domain-specific partial responses.
        """
        pass
    
    def _get_cache_key(self, input_data: TInput) -> str:
        """Generate cache key from input."""
        return str(hash(str(input_data)))
    
    def _update_cache(self, key: str, value: TOutput) -> None:
        """Update response cache with LRU eviction."""
        self.response_cache[key] = value
        
        # Simple LRU: remove oldest if over limit
        if len(self.response_cache) > self.cache_max_size:
            oldest_key = next(iter(self.response_cache))
            del self.response_cache[oldest_key]
    
    def _create_initial_state(self, input_data: TInput) -> TState:
        """Create initial state using primary agent."""
        return self.primary_agent._create_initial_state(input_data)
    
    def _extract_output(self, final_state: TState) -> TOutput:
        """Extract output using primary agent."""
        return self.primary_agent._extract_output(final_state)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of fallback system."""
        health = await super().health_check()
        
        # Add circuit breaker status
        cb_stats = self.circuit_breaker.get_stats()
        health["circuit_breaker"] = {
            "state": self.circuit_breaker.get_state().value,
            "failure_rate": cb_stats.failure_rate,
            "total_calls": cb_stats.total_calls
        }
        
        # Add cache status
        health["cache"] = {
            "size": len(self.response_cache),
            "max_size": self.cache_max_size
        }
        
        # Add alternative agents status
        health["alternatives"] = len(self.alternative_agents)
        
        return health