"""
Temporal Activities for AURA Intelligence

Activities are the building blocks of workflows - they perform
the actual work and can be retried independently.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import asyncio
from dataclasses import dataclass

from temporalio import activity
from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode
import structlog
from aiokafka import AIOKafkaProducer
import aioredis

from ...agents.base import AgentBase, AgentState, AgentConfig
from ...agents.observability import AgentInstrumentor, GenAIAttributes
from ...agents.resilience import CircuitBreaker, CircuitBreakerConfig
from ..legacy.core.observer import ObserverAgent as LegacyObserverAgent
from ..v2.observer import ObserverAgentV2
from ..v2.analyst import AnalystAgentV2

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

# Metrics
activity_duration = meter.create_histogram(
    name="temporal.activity.duration",
    description="Duration of activity execution",
    unit="ms"
)

activity_errors = meter.create_counter(
    name="temporal.activity.errors",
    description="Number of activity errors",
    unit="1"
)


class AgentActivity:
    """Activities for agent execution."""
    
    @staticmethod
    @activity.defn
    async def process(
        agent_id: str,
        agent_type: str,
        state: AgentState,
        config: Dict[str, Any],
        span_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute agent processing with full observability."""
        start_time = datetime.utcnow()
        
        # Create span for activity
        with tracer.start_as_current_span(
            "temporal.activity.agent.process",
            attributes={
                GenAIAttributes.AGENT_NAME: agent_id,
                GenAIAttributes.AGENT_TYPE: agent_type,
                "temporal.activity": "agent.process"
            }
        ) as span:
            try:
                # Create agent based on type and config
                agent = await AgentActivity._create_agent(agent_type, config)
                
                # Instrument the agent
                instrumentor = AgentInstrumentor()
                agent = instrumentor.instrument_agent(agent)
                
                # Process with circuit breaker
                circuit_breaker = CircuitBreaker(
                    CircuitBreakerConfig(
                        name=f"agent_{agent_type}",
                        failure_threshold=3,
                        timeout=config.get("circuit_breaker_timeout", 60)
                    )
                )
                
                async def process_with_agent():
                    # Convert state if needed
                    if hasattr(agent, 'process_state'):
                        return await agent.process_state(state)
                    else:
                        # Legacy agent compatibility
                        input_data = state.context.get("input_data", {})
                        result = await agent.process(input_data)
                        return {
                            "output": result,
                            "state": state,
                            "metrics": getattr(agent, '_metrics', {})
                        }
                
                result = await circuit_breaker.call(process_with_agent)
                
                # Record success metrics
                duration = (datetime.utcnow() - start_time).total_seconds() * 1000
                activity_duration.record(
                    duration,
                    {
                        "activity": "agent.process",
                        "agent_type": agent_type,
                        "status": "success"
                    }
                )
                
                span.set_status(Status(StatusCode.OK))
                return result
                
            except Exception as e:
                logger.error(
                    "Agent processing failed",
                    agent_id=agent_id,
                    agent_type=agent_type,
                    error=str(e)
                )
                
                activity_errors.add(
                    1,
                    {
                        "activity": "agent.process",
                        "agent_type": agent_type,
                        "error_type": type(e).__name__
                    }
                )
                
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    
    @staticmethod
    async def _create_agent(agent_type: str, config: Dict[str, Any]) -> AgentBase:
        """Create agent instance based on type."""
        # Check feature flag for v2 agents
        use_v2 = config.get("use_v2_agents", True)
        
        if agent_type == "observer":
            if use_v2:
                return ObserverAgentV2(AgentConfig(
                    name=f"observer_v2_{datetime.utcnow().timestamp()}",
                    **config
                ))
            else:
                return LegacyObserverAgent(config)
                
        elif agent_type == "analyst":
            return AnalystAgentV2(AgentConfig(
                name=f"analyst_v2_{datetime.utcnow().timestamp()}",
                **config
            ))
            
        elif agent_type in ["web_search", "academic_search", "news_search"]:
            # These would be specific search agents
            from ..v2.search import SearchAgentV2
            return SearchAgentV2(AgentConfig(
                name=f"{agent_type}_v2_{datetime.utcnow().timestamp()}",
                search_type=agent_type.replace("_search", ""),
                **config
            ))
            
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    @staticmethod
    @activity.defn
    async def compute_consensus(
        results: Dict[str, Any],
        strategy: str = "majority"
    ) -> Dict[str, Any]:
        """Compute consensus from multiple agent results."""
        with tracer.start_as_current_span("temporal.activity.compute_consensus") as span:
            span.set_attribute("consensus.strategy", strategy)
            span.set_attribute("consensus.participant_count", len(results))
            
            if strategy == "majority":
                # Simple majority voting on key decisions
                votes = {}
                for agent_id, result in results.items():
                    decision = result.get("decision", "unknown")
                    votes[decision] = votes.get(decision, 0) + 1
                
                # Find majority decision
                majority_decision = max(votes, key=votes.get)
                majority_count = votes[majority_decision]
                
                consensus = {
                    "decision": majority_decision,
                    "confidence": majority_count / len(results),
                    "votes": votes,
                    "strategy": strategy
                }
                
            elif strategy == "weighted_average":
                # Weighted average based on agent confidence scores
                total_weight = 0
                weighted_sum = {}
                
                for agent_id, result in results.items():
                    confidence = result.get("confidence", 0.5)
                    total_weight += confidence
                    
                    for key, value in result.items():
                        if isinstance(value, (int, float)):
                            if key not in weighted_sum:
                                weighted_sum[key] = 0
                            weighted_sum[key] += value * confidence
                
                # Calculate weighted averages
                consensus = {
                    key: value / total_weight
                    for key, value in weighted_sum.items()
                }
                consensus["strategy"] = strategy
                consensus["total_weight"] = total_weight
                
            else:
                raise ValueError(f"Unknown consensus strategy: {strategy}")
            
            span.set_attribute("consensus.achieved", True)
            return consensus


class StateManagementActivity:
    """Activities for agent state management."""
    
    @staticmethod
    @activity.defn
    async def create_initial_state(
        agent_id: str,
        input_data: Dict[str, Any]
    ) -> AgentState:
        """Create initial agent state."""
        with tracer.start_as_current_span("temporal.activity.create_state") as span:
            span.set_attribute("agent.id", agent_id)
            
            state = AgentState(
                agent_id=agent_id,
                context={"input_data": input_data},
                messages=[{
                    "role": "system",
                    "content": "Agent workflow started",
                    "timestamp": datetime.utcnow().isoformat()
                }]
            )
            
            # Store in Redis for persistence
            try:
                redis = await aioredis.create_redis_pool('redis://localhost')
                await redis.setex(
                    f"agent:state:{agent_id}",
                    3600,  # 1 hour TTL
                    json.dumps(state.dict())
                )
                redis.close()
                await redis.wait_closed()
            except Exception as e:
                logger.warning(f"Failed to persist initial state to Redis: {e}")
            
            return state
    
    @staticmethod
    @activity.defn
    async def persist_state(
        agent_id: str,
        state: AgentState
    ) -> None:
        """Persist agent state to storage."""
        with tracer.start_as_current_span("temporal.activity.persist_state") as span:
            span.set_attribute("agent.id", agent_id)
            
            try:
                # Persist to Redis
                redis = await aioredis.create_redis_pool('redis://localhost')
                await redis.setex(
                    f"agent:state:{agent_id}",
                    3600,  # 1 hour TTL
                    json.dumps(state.dict())
                )
                
                # Also persist to long-term storage (e.g., S3, database)
                # This would be implemented based on your storage strategy
                
                redis.close()
                await redis.wait_closed()
                
                logger.info(f"Persisted state for agent {agent_id}")
                
            except Exception as e:
                logger.error(f"Failed to persist state: {e}")
                raise
    
    @staticmethod
    @activity.defn
    async def load_state(agent_id: str) -> Optional[AgentState]:
        """Load agent state from storage."""
        with tracer.start_as_current_span("temporal.activity.load_state") as span:
            span.set_attribute("agent.id", agent_id)
            
            try:
                redis = await aioredis.create_redis_pool('redis://localhost')
                state_json = await redis.get(f"agent:state:{agent_id}")
                redis.close()
                await redis.wait_closed()
                
                if state_json:
                    return AgentState(**json.loads(state_json))
                return None
                
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
                return None


class KafkaProducerActivity:
    """Activities for Kafka event publishing."""
    
    _producer: Optional[AIOKafkaProducer] = None
    
    @classmethod
    async def _get_producer(cls) -> AIOKafkaProducer:
        """Get or create Kafka producer."""
        if cls._producer is None:
            cls._producer = AIOKafkaProducer(
                bootstrap_servers='localhost:9092',
                value_serializer=lambda v: json.dumps(v).encode()
            )
            await cls._producer.start()
        return cls._producer
    
    @staticmethod
    @activity.defn
    async def publish_event(
        topic: str,
        event: Dict[str, Any]
    ) -> None:
        """Publish event to Kafka topic."""
        with tracer.start_as_current_span("temporal.activity.publish_event") as span:
            span.set_attribute("kafka.topic", topic)
            span.set_attribute("event.type", event.get("type", "unknown"))
            
            try:
                producer = await KafkaProducerActivity._get_producer()
                
                # Add metadata
                event["timestamp"] = datetime.utcnow().isoformat()
                event["source"] = "temporal.activity"
                
                # Send to Kafka
                await producer.send_and_wait(topic, event)
                
                logger.info(f"Published event to {topic}")
                
            except Exception as e:
                logger.error(f"Failed to publish event: {e}")
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise


class ObservabilityActivity:
    """Activities for observability and monitoring."""
    
    @staticmethod
    @activity.defn
    async def start_workflow_span(
        agent_id: str,
        agent_type: str,
        trace_parent: Optional[str] = None
    ) -> str:
        """Start a workflow span and return context."""
        # In a real implementation, this would create a span
        # and return serialized context for propagation
        span = tracer.start_span(
            f"workflow.{agent_type}",
            attributes={
                GenAIAttributes.AGENT_NAME: agent_id,
                GenAIAttributes.AGENT_TYPE: agent_type
            }
        )
        
        # Return span context as string for propagation
        return f"trace_id={span.get_span_context().trace_id}"
    
    @staticmethod
    @activity.defn
    async def record_workflow_metrics(
        agent_id: str,
        agent_type: str,
        status: str,
        duration_ms: float,
        tokens: Dict[str, int]
    ) -> None:
        """Record workflow execution metrics."""
        with tracer.start_as_current_span("temporal.activity.record_metrics") as span:
            # Record duration
            workflow_duration = meter.create_histogram(
                name="workflow.duration",
                description="Workflow execution duration",
                unit="ms"
            )
            
            workflow_duration.record(
                duration_ms,
                {
                    "agent_type": agent_type,
                    "status": status
                }
            )
            
            # Record token usage if available
            if tokens:
                token_counter = meter.create_counter(
                    name="llm.tokens",
                    description="LLM token usage",
                    unit="1"
                )
                
                for token_type, count in tokens.items():
                    token_counter.add(
                        count,
                        {
                            "agent_type": agent_type,
                            "token_type": token_type
                        }
                    )
            
            # Record workflow completion
            workflow_counter = meter.create_counter(
                name="workflow.completed",
                description="Number of completed workflows",
                unit="1"
            )
            
            workflow_counter.add(
                1,
                {
                    "agent_type": agent_type,
                    "status": status
                }
            )
            
            logger.info(
                "Recorded workflow metrics",
                agent_id=agent_id,
                duration_ms=duration_ms,
                status=status
            )