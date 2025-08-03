# Production LNN Council Agent Implementation Guide

## Overview

This guide provides step-by-step instructions for implementing and deploying the Production LNN Council Agent in the AURA Intelligence system. The Production LNN Council Agent uses real Liquid Neural Networks for adaptive, context-aware decision making.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  Production LNN Council Agent                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │   Config    │  │  LNN Core    │  │  Integration    │  │
│  │  Handler    │  │  Components  │  │   Adapters      │  │
│  └─────────────┘  └──────────────┘  └─────────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              Workflow Pipeline                       │  │
│  │  ┌──────┐ ┌────────┐ ┌─────────┐ ┌──────────┐    │  │
│  │  │Valid.│→│Context │→│Features │→│    LNN    │    │  │
│  │  │ Task │ │Gather  │ │ Prep    │ │ Inference │    │  │
│  │  └──────┘ └────────┘ └─────────┘ └──────────┘    │  │
│  │                                                     │  │
│  │  ┌──────┐ ┌────────┐                              │  │
│  │  │ Vote │←│ Store  │                              │  │
│  │  │ Gen. │ │Decision│                              │  │
│  │  └──────┘ └────────┘                              │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │   Neo4j     │  │    Kafka     │  │      Mem0       │  │
│  │  Adapter    │  │   Events     │  │    Memory       │  │
│  └─────────────┘  └──────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Steps

### Step 1: Environment Setup

```bash
# Install required dependencies
pip install torch>=2.0.0
pip install langgraph>=0.1.0
pip install pydantic>=2.0.0
pip install structlog
pip install opentelemetry-api
pip install opentelemetry-sdk

# For production deployment
pip install aiokafka
pip install neo4j
pip install redis
```

### Step 2: Basic Usage

```python
from aura_intelligence.agents.council import (
    ProductionLNNCouncilAgent,
    CouncilTask,
    AgentConfig
)

# Initialize the agent
config = AgentConfig(
    name="gpu_allocation_council",
    model="lnn-v1",
    temperature=0.7,
    enable_memory=True,
    enable_tools=True
)

agent = ProductionLNNCouncilAgent(config)

# Create a task
task = CouncilTask(
    task_id="req-12345",
    task_type="gpu_allocation",
    payload={
        "gpu_allocation": {
            "gpu_type": "a100",
            "gpu_count": 4,
            "duration_hours": 8,
            "user_id": "user-123",
            "cost_per_hour": 12.8
        }
    },
    context={
        "priority": "high",
        "budget_remaining": 5000.0
    },
    priority=8
)

# Process the task
vote = await agent.process(task)

print(f"Decision: {vote.vote.value}")
print(f"Confidence: {vote.confidence:.2%}")
print(f"Reasoning: {vote.reasoning}")
```

### Step 3: Production Configuration

```python
from aura_intelligence.adapters.neo4j_adapter import Neo4jAdapter, Neo4jConfig
from aura_intelligence.events.producer import EventProducer
from aura_intelligence.memory.mem0_integration import Mem0Manager

# Initialize external adapters
neo4j_adapter = Neo4jAdapter(Neo4jConfig(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="your-password"
))

event_producer = EventProducer(
    bootstrap_servers="localhost:9092",
    topic_prefix="aura.council"
)

memory_manager = Mem0Manager(
    redis_url="redis://localhost:6379",
    namespace="lnn_council"
)

# Set adapters on the agent
agent.set_adapters(
    neo4j=neo4j_adapter,
    events=event_producer,
    memory=memory_manager
)
```

### Step 4: Advanced Configuration

```python
# Custom LNN configuration
custom_config = {
    "name": "advanced_lnn_council",
    "model": "lnn-v2",
    "temperature": 0.5,
    "lnn_config": {
        "input_size": 512,      # Larger input for more context
        "hidden_size": 256,     # Larger hidden state
        "output_size": 4,       # Vote types
        "num_layers": 5,        # Deeper network
        "time_constant": 0.8,   # Faster adaptation
        "ode_solver": "adaptive",
        "solver_steps": 20,     # More precise ODE solving
        "adaptivity_rate": 0.15,
        "sparsity": 0.6,       # Less sparse for more capacity
        "consensus_enabled": True,
        "consensus_threshold": 0.75
    },
    "feature_flags": {
        "use_lnn_inference": True,
        "enable_memory_hooks": True,
        "enable_context_queries": True,
        "enable_event_streaming": True,
        "enable_explanation_generation": True
    },
    "vote_threshold": 0.8,
    "delegation_threshold": 0.4
}

advanced_agent = ProductionLNNCouncilAgent(custom_config)
```

### Step 5: Context Enhancement

```python
# Pre-populate context for better decisions
async def enhance_context(task: CouncilTask, agent: ProductionLNNCouncilAgent):
    """Enhance task context with additional information."""
    
    # Query historical patterns
    if agent.neo4j_adapter:
        user_history = await agent.neo4j_adapter.query("""
            MATCH (u:User {id: $user_id})-[:REQUESTED]->(a:Allocation)
            WHERE a.timestamp > datetime() - duration('P90D')
            RETURN avg(a.actual_usage) as avg_usage,
                   count(a) as total_requests,
                   sum(CASE WHEN a.approved THEN 1 ELSE 0 END) as approved_count
        """, {"user_id": task.payload["gpu_allocation"]["user_id"]})
        
        task.context["user_history"] = user_history[0] if user_history else {}
    
    # Add current resource availability
    task.context["resource_availability"] = {
        "a100_available": 50,
        "v100_available": 100,
        "utilization_rate": 0.75
    }
    
    return task

# Use enhanced context
enhanced_task = await enhance_context(task, agent)
vote = await agent.process(enhanced_task)
```

### Step 6: Monitoring and Observability

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Setup tracing
trace.set_tracer_provider(TracerProvider())
tracer_provider = trace.get_tracer_provider()

# Add OTLP exporter for production
otlp_exporter = OTLPSpanExporter(
    endpoint="http://localhost:4317",
    insecure=True
)
span_processor = BatchSpanProcessor(otlp_exporter)
tracer_provider.add_span_processor(span_processor)

# Monitor decision metrics
async def monitored_decision(agent, task):
    tracer = trace.get_tracer(__name__)
    
    with tracer.start_as_current_span("council_decision") as span:
        span.set_attribute("task.id", task.task_id)
        span.set_attribute("task.type", task.task_type)
        
        vote = await agent.process(task)
        
        span.set_attribute("vote.type", vote.vote.value)
        span.set_attribute("vote.confidence", vote.confidence)
        
        # Log to metrics system
        if vote.confidence < 0.5:
            span.add_event("low_confidence_decision", {
                "confidence": vote.confidence,
                "reasoning": vote.reasoning
            })
        
        return vote
```

### Step 7: Batch Processing

```python
async def process_batch_decisions(agent, tasks):
    """Process multiple tasks efficiently."""
    
    # Process tasks concurrently
    votes = await asyncio.gather(*[
        agent.process(task) for task in tasks
    ], return_exceptions=True)
    
    # Handle results
    results = []
    for i, vote in enumerate(votes):
        if isinstance(vote, Exception):
            logger.error(f"Task {tasks[i].task_id} failed: {vote}")
            results.append(None)
        else:
            results.append(vote)
    
    return results

# Example batch processing
tasks = [
    CouncilTask(...),
    CouncilTask(...),
    CouncilTask(...)
]

votes = await process_batch_decisions(agent, tasks)
```

### Step 8: Custom Decision Logic

```python
class CustomGPUAllocationAgent(ProductionLNNCouncilAgent):
    """Custom agent with domain-specific logic."""
    
    def _generate_reasoning(self, state, vote_type, confidence):
        """Override to add custom reasoning logic."""
        base_reasoning = super()._generate_reasoning(state, vote_type, confidence)
        
        # Add domain-specific insights
        gpu_allocation = state.task.payload.get("gpu_allocation", {})
        if gpu_allocation.get("gpu_type") == "a100":
            base_reasoning += " A100 GPUs are premium resources reserved for ML workloads."
        
        # Add utilization predictions
        if state.context_window and len(state.context_window.historical_patterns) > 10:
            base_reasoning += f" Based on {len(state.context_window.historical_patterns)} historical patterns."
        
        return base_reasoning
    
    def _collect_evidence(self, state, vote_type):
        """Override to add custom evidence collection."""
        evidence = super()._collect_evidence(state, vote_type)
        
        # Add custom evidence
        evidence.append({
            "type": "resource_impact",
            "current_utilization": 0.75,
            "projected_utilization": 0.85,
            "impact_score": 0.1
        })
        
        return evidence
```

## Best Practices

### 1. Resource Management

```python
# Always cleanup resources
try:
    vote = await agent.process(task)
finally:
    await agent.cleanup()

# Use context managers (when available)
async with ProductionLNNCouncilAgent(config) as agent:
    vote = await agent.process(task)
```

### 2. Error Handling

```python
from aura_intelligence.resilience import with_fallback

@with_fallback(fallback_value=CouncilVote(
    agent_id="fallback",
    vote=VoteType.ABSTAIN,
    confidence=0.0,
    reasoning="Failed to process - using fallback",
    supporting_evidence=[],
    timestamp=datetime.now(timezone.utc)
))
async def safe_process(agent, task):
    return await agent.process(task)
```

### 3. Performance Optimization

```python
# Precompile the workflow
agent = ProductionLNNCouncilAgent(config)
_ = agent.build_graph()  # Pre-build the graph

# Batch context queries
tasks_with_context = await asyncio.gather(*[
    enhance_context(task, agent) for task in tasks
])

# Use connection pooling for adapters
neo4j_adapter = Neo4jAdapter(Neo4jConfig(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="your-password",
    max_connection_pool_size=50
))
```

### 4. Testing

```python
import pytest

@pytest.fixture
async def test_agent():
    config = AgentConfig(name="test_agent", model="lnn-v1")
    agent = ProductionLNNCouncilAgent(config)
    yield agent
    await agent.cleanup()

async def test_gpu_allocation_decision(test_agent):
    task = CouncilTask(
        task_id="test-123",
        task_type="gpu_allocation",
        payload={
            "gpu_allocation": {
                "gpu_type": "v100",
                "gpu_count": 2,
                "duration_hours": 4,
                "user_id": "test-user",
                "cost_per_hour": 8.0
            }
        },
        context={},
        priority=5
    )
    
    vote = await test_agent.process(task)
    
    assert vote.vote in [VoteType.APPROVE, VoteType.REJECT]
    assert 0.0 <= vote.confidence <= 1.0
    assert len(vote.reasoning) > 0
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```python
   # If you get import errors, ensure all dependencies are installed
   pip install -r requirements.txt
   ```

2. **Memory Issues**
   ```python
   # Reduce batch sizes
   batch_size = 10  # Instead of 100
   
   # Use smaller LNN configuration
   config["lnn_config"]["hidden_size"] = 64  # Instead of 256
   ```

3. **Slow Inference**
   ```python
   # Use faster ODE solver
   config["lnn_config"]["ode_solver"] = "euler"  # Instead of "rk4"
   config["lnn_config"]["solver_steps"] = 5  # Instead of 20
   ```

4. **Connection Issues**
   ```python
   # Add retry logic
   from tenacity import retry, stop_after_attempt, wait_exponential
   
   @retry(stop=stop_after_attempt(3), wait=wait_exponential())
   async def connect_with_retry():
       return await neo4j_adapter.connect()
   ```

## Performance Metrics

Expected performance characteristics:

- **Inference Latency**: 50-200ms per decision
- **Memory Usage**: 200-500MB per agent instance
- **Throughput**: 50-100 decisions/second (single instance)
- **Context Query Time**: 10-50ms (with proper indexing)
- **Event Publishing**: < 5ms

## Deployment Checklist

- [ ] Install all dependencies
- [ ] Configure external services (Neo4j, Kafka, Redis)
- [ ] Set up monitoring and tracing
- [ ] Configure resource limits
- [ ] Implement error handling
- [ ] Add health checks
- [ ] Set up logging
- [ ] Configure backups for stateful components
- [ ] Test failover scenarios
- [ ] Document custom configurations

## Next Steps

1. Review the [LNN Technical Documentation](./LNN_COUNCIL_INTEGRATION_RESEARCH.md)
2. Set up monitoring dashboards
3. Configure alerting rules
4. Implement custom decision logic for your use case
5. Run load tests to validate performance
6. Set up continuous training pipeline for LNN adaptation

## Support

For issues or questions:
- Check the [troubleshooting guide](#troubleshooting)
- Review logs in OpenTelemetry
- Contact the AURA Intelligence team