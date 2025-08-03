"""
Event Mesh Connectors for AURA Intelligence

Integrates Kafka with:
- Temporal workflows
- State stores (Redis, PostgreSQL)
- Change Data Capture (CDC)
- External systems
"""

from typing import Dict, Any, Optional, List, Callable, AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import json

import structlog
from opentelemetry import trace, metrics
# import aioredis  # Temporarily commented out due to Python 3.13 compatibility issue
import asyncpg
# from debezium import DebeziumClient  # Temporarily commented out - package not available

from .schemas import EventSchema, AgentEvent, WorkflowEvent, SystemEvent, EventType
from .producers import EventProducer, ProducerConfig, TransactionalProducer
from .consumers import EventProcessor, EventConsumer, ConsumerConfig
from ..agents.temporal import TemporalClient, execute_workflow

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

# Metrics
connector_events = meter.create_counter(
    name="connector.events.processed",
    description="Number of events processed by connectors",
    unit="1"
)

connector_errors = meter.create_counter(
    name="connector.errors",
    description="Number of connector errors",
    unit="1"
)


class TemporalKafkaConnector(EventProcessor):
    """
    Bridges Kafka events with Temporal workflows.
    
    Features:
    - Event-triggered workflow execution
    - Workflow result publishing
    - Saga pattern coordination
    - Compensation handling
    """
    
    def __init__(
        self,
        temporal_client: TemporalClient,
        producer: EventProducer,
        config: Optional[Dict[str, Any]] = None
    ):
        self.temporal_client = temporal_client
        self.producer = producer
        self.config = config or {}
        
        # Workflow mappings
        self.event_to_workflow: Dict[str, str] = {
            EventType.AGENT_STARTED: "agent",
            EventType.CONSENSUS_REQUESTED: "consensus",
            "orchestration.requested": "multi_agent"
        }
        
        # Active workflows tracking
        self.active_workflows: Dict[str, Any] = {}
    
    async def process(self, event: EventSchema) -> None:
        """Process event and trigger workflows."""
        with tracer.start_as_current_span(
            "connector.temporal.process",
            attributes={
                "event.type": event.event_type.value,
                "event.id": event.event_id
            }
        ) as span:
            try:
                # Check if event should trigger workflow
                workflow_type = self.event_to_workflow.get(event.event_type.value)
                
                if not workflow_type:
                    return
                
                # Start workflow based on event
                if workflow_type == "agent" and isinstance(event, AgentEvent):
                    await self._start_agent_workflow(event)
                    
                elif workflow_type == "multi_agent":
                    await self._start_multi_agent_workflow(event)
                    
                elif workflow_type == "consensus":
                    await self._start_consensus_workflow(event)
                
                connector_events.add(
                    1,
                    {
                        "connector": "temporal",
                        "event_type": event.event_type.value,
                        "workflow_type": workflow_type
                    }
                )
                
                span.set_status(trace.Status(trace.StatusCode.OK))
                
            except Exception as e:
                connector_errors.add(
                    1,
                    {"connector": "temporal", "error": type(e).__name__}
                )
                
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.record_exception(e)
                
                # Publish error event
                await self._publish_error_event(event, e)
                raise
    
    async def _start_agent_workflow(self, event: AgentEvent) -> None:
        """Start agent workflow from event."""
        # Extract workflow input from event
        workflow_input = {
            "agent_type": event.agent_type,
            "input_data": event.data.get("input", {}),
            "correlation_id": event.correlation_id
        }
        
        # Start workflow
        handle = await self.temporal_client.execute_agent_workflow(
            event.agent_type,
            workflow_input,
            config={
                "workflow_id": f"event-{event.event_id}",
                "search_attributes": {
                    "event_id": event.event_id,
                    "correlation_id": event.correlation_id
                }
            }
        )
        
        # Track active workflow
        self.active_workflows[event.event_id] = {
            "handle": handle,
            "start_time": datetime.utcnow(),
            "event": event
        }
        
        # Start result monitoring
        asyncio.create_task(self._monitor_workflow_result(event.event_id, handle))
    
    async def _start_multi_agent_workflow(self, event: EventSchema) -> None:
        """Start multi-agent orchestration workflow."""
        # Extract agents configuration from event
        agents = event.data.get("agents", [])
        orchestration_type = event.data.get("orchestration_type", "sequential")
        
        handle = await self.temporal_client.execute_multi_agent_workflow(
            agents,
            orchestration_type,
            event.data.get("input", {}),
            config={
                "workflow_id": f"orchestration-{event.event_id}",
                "search_attributes": {
                    "event_id": event.event_id,
                    "orchestration_type": orchestration_type
                }
            }
        )
        
        self.active_workflows[event.event_id] = {
            "handle": handle,
            "start_time": datetime.utcnow(),
            "event": event
        }
        
        asyncio.create_task(self._monitor_workflow_result(event.event_id, handle))
    
    async def _start_consensus_workflow(self, event: EventSchema) -> None:
        """Start consensus workflow."""
        # Would implement consensus workflow triggering
        pass
    
    async def _monitor_workflow_result(self, event_id: str, handle: Any) -> None:
        """Monitor workflow and publish result."""
        try:
            # Wait for workflow result
            result = await handle.result(timeout=timedelta(minutes=30))
            
            # Create completion event
            completion_event = WorkflowEvent.create_completed_event(
                workflow_id=handle.workflow_id,
                workflow_type="agent_workflow",
                workflow_version="1.0.0",
                run_id=handle.run_id,
                output_data=result,
                duration_ms=result.duration_ms,
                causation_id=event_id
            )
            
            # Publish to Kafka
            await self.producer.send_event(
                "workflow.completed",
                completion_event
            )
            
            # Clean up tracking
            self.active_workflows.pop(event_id, None)
            
        except Exception as e:
            logger.error(f"Workflow monitoring failed: {e}", event_id=event_id)
            
            # Publish failure event
            failure_event = WorkflowEvent(
                event_type=EventType.WORKFLOW_FAILED,
                workflow_id=handle.workflow_id,
                workflow_type="agent_workflow",
                workflow_version="1.0.0",
                run_id=handle.run_id,
                data={"error": str(e)},
                causation_id=event_id
            )
            
            await self.producer.send_event(
                "workflow.failed",
                failure_event
            )
    
    async def _publish_error_event(self, original_event: EventSchema, error: Exception) -> None:
        """Publish error event for failed processing."""
        error_event = SystemEvent.create_alert_event(
            component="temporal_connector",
            instance_id="connector-1",
            alert_type="workflow_start_failed",
            message=f"Failed to start workflow for event {original_event.event_id}",
            severity="error",
            details={
                "original_event_id": original_event.event_id,
                "original_event_type": original_event.event_type.value,
                "error": str(error),
                "error_type": type(error).__name__
            }
        )
        
        await self.producer.send_event("system.errors", error_event)


class StateStoreConnector:
    """
    Synchronizes Kafka events with state stores.
    
    Features:
    - Redis state synchronization
    - PostgreSQL event sourcing
    - State snapshots
    - Compaction
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        postgres_url: Optional[str] = None
    ):
        self.redis_url = redis_url
        self.postgres_url = postgres_url
        
        self.redis: Optional[aioredis.Redis] = None
        self.postgres: Optional[asyncpg.Pool] = None
        
        # State compaction settings
        self.compaction_interval = timedelta(hours=1)
        self.retention_period = timedelta(days=7)
        self._compaction_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the state store connector."""
        # Connect to Redis
        self.redis = await aioredis.create_redis_pool(self.redis_url)
        
        # Connect to PostgreSQL if configured
        if self.postgres_url:
            self.postgres = await asyncpg.create_pool(self.postgres_url)
            await self._create_tables()
        
        # Start compaction task
        self._compaction_task = asyncio.create_task(self._run_compaction())
        
        logger.info("State store connector started")
    
    async def stop(self):
        """Stop the state store connector."""
        # Stop compaction
        if self._compaction_task:
            self._compaction_task.cancel()
            try:
                await self._compaction_task
            except asyncio.CancelledError:
                pass
        
        # Close connections
        if self.redis:
            self.redis.close()
            await self.redis.wait_closed()
        
        if self.postgres:
            await self.postgres.close()
        
        logger.info("State store connector stopped")
    
    async def store_agent_state(self, event: AgentEvent) -> None:
        """Store agent state from event."""
        if event.event_type == EventType.AGENT_STATE_CHANGED:
            state_key = f"agent:state:{event.agent_id}"
            
            # Store in Redis
            await self.redis.setex(
                state_key,
                int(self.retention_period.total_seconds()),
                json.dumps(event.state_after or {})
            )
            
            # Store in PostgreSQL for event sourcing
            if self.postgres:
                await self._store_event_postgres(event)
    
    async def store_workflow_state(self, event: WorkflowEvent) -> None:
        """Store workflow state from event."""
        if event.event_type == EventType.WORKFLOW_STATE_CHANGED:
            state_key = f"workflow:state:{event.workflow_id}:{event.run_id}"
            
            # Store current state
            state = {
                "workflow_id": event.workflow_id,
                "run_id": event.run_id,
                "current_step": event.current_step,
                "next_step": event.next_step,
                "steps_completed": event.steps_completed,
                "timestamp": event.timestamp.isoformat()
            }
            
            await self.redis.setex(
                state_key,
                int(self.retention_period.total_seconds()),
                json.dumps(state)
            )
    
    async def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get current agent state."""
        state_json = await self.redis.get(f"agent:state:{agent_id}")
        
        if state_json:
            return json.loads(state_json)
        
        # Try to reconstruct from event store
        if self.postgres:
            return await self._reconstruct_agent_state(agent_id)
        
        return None
    
    async def _create_tables(self) -> None:
        """Create PostgreSQL tables for event sourcing."""
        async with self.postgres.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS event_store (
                    event_id UUID PRIMARY KEY,
                    event_type VARCHAR(100) NOT NULL,
                    source_id VARCHAR(255) NOT NULL,
                    source_type VARCHAR(100) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    data JSONB NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_event_store_source 
                ON event_store(source_id, timestamp DESC);
                
                CREATE INDEX IF NOT EXISTS idx_event_store_type 
                ON event_store(event_type, timestamp DESC);
                
                CREATE TABLE IF NOT EXISTS state_snapshots (
                    id SERIAL PRIMARY KEY,
                    entity_id VARCHAR(255) NOT NULL,
                    entity_type VARCHAR(100) NOT NULL,
                    snapshot_data JSONB NOT NULL,
                    event_id UUID REFERENCES event_store(event_id),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_state_snapshots_entity 
                ON state_snapshots(entity_id, created_at DESC);
            """)
    
    async def _store_event_postgres(self, event: EventSchema) -> None:
        """Store event in PostgreSQL."""
        async with self.postgres.acquire() as conn:
            await conn.execute("""
                INSERT INTO event_store 
                (event_id, event_type, source_id, source_type, timestamp, data, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
                event.event_id,
                event.event_type.value,
                event.source_id,
                event.source_type,
                event.timestamp,
                json.dumps(event.data),
                json.dumps({
                    "correlation_id": event.correlation_id,
                    "causation_id": event.causation_id,
                    "headers": event.headers
                })
            )
    
    async def _reconstruct_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Reconstruct agent state from events."""
        # Check for snapshot first
        async with self.postgres.acquire() as conn:
            snapshot = await conn.fetchrow("""
                SELECT snapshot_data, created_at
                FROM state_snapshots
                WHERE entity_id = $1 AND entity_type = 'agent'
                ORDER BY created_at DESC
                LIMIT 1
            """, agent_id)
            
            if snapshot:
                state = json.loads(snapshot['snapshot_data'])
                
                # Apply events since snapshot
                events = await conn.fetch("""
                    SELECT data
                    FROM event_store
                    WHERE source_id = $1 
                    AND event_type = 'agent.state.changed'
                    AND timestamp > $2
                    ORDER BY timestamp
                """, agent_id, snapshot['created_at'])
                
                for event in events:
                    # Apply event to state
                    event_data = json.loads(event['data'])
                    state.update(event_data.get('state_after', {}))
                
                return state
        
        return None
    
    async def _run_compaction(self) -> None:
        """Periodically compact state and create snapshots."""
        while True:
            try:
                await asyncio.sleep(self.compaction_interval.total_seconds())
                await self._compact_states()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Compaction error: {e}")
    
    async def _compact_states(self) -> None:
        """Create state snapshots and clean old events."""
        if not self.postgres:
            return
        
        async with self.postgres.acquire() as conn:
            # Get entities that need snapshots
            entities = await conn.fetch("""
                SELECT DISTINCT source_id, source_type
                FROM event_store
                WHERE timestamp > NOW() - INTERVAL '1 day'
                AND source_type IN ('agent', 'workflow')
            """)
            
            for entity in entities:
                # Reconstruct current state
                if entity['source_type'] == 'agent':
                    state = await self._reconstruct_agent_state(entity['source_id'])
                    
                    if state:
                        # Create snapshot
                        await conn.execute("""
                            INSERT INTO state_snapshots 
                            (entity_id, entity_type, snapshot_data)
                            VALUES ($1, $2, $3)
                        """, entity['source_id'], 'agent', json.dumps(state))
            
            # Clean old events
            await conn.execute("""
                DELETE FROM event_store
                WHERE timestamp < NOW() - INTERVAL '7 days'
                AND event_id NOT IN (
                    SELECT event_id FROM state_snapshots
                )
            """)


class CDCConnector:
    """
    Change Data Capture connector for database synchronization.
    
    Features:
    - Database change streaming
    - Event generation from changes
    - Schema evolution handling
    - Transactional consistency
    """
    
    def __init__(
        self,
        producer: TransactionalProducer,
        database_config: Dict[str, Any]
    ):
        self.producer = producer
        self.database_config = database_config
        
        # Debezium client for CDC
        self.debezium_client: Optional[DebeziumClient] = None
        
        # Table to topic mappings
        self.table_mappings = {
            "agents": "agent.cdc.events",
            "workflows": "workflow.cdc.events",
            "agent_states": "agent.state.changes"
        }
    
    async def start(self):
        """Start CDC connector."""
        # Initialize Debezium client
        self.debezium_client = DebeziumClient(
            connector_name="aura-cdc",
            database_hostname=self.database_config["host"],
            database_port=self.database_config["port"],
            database_user=self.database_config["user"],
            database_password=self.database_config["password"],
            database_dbname=self.database_config["database"],
            table_whitelist=",".join(self.table_mappings.keys())
        )
        
        # Start consuming changes
        asyncio.create_task(self._consume_changes())
        
        logger.info("CDC connector started")
    
    async def stop(self):
        """Stop CDC connector."""
        if self.debezium_client:
            await self.debezium_client.stop()
        
        logger.info("CDC connector stopped")
    
    async def _consume_changes(self) -> None:
        """Consume database changes and publish events."""
        async for change in self.debezium_client.consume():
            try:
                await self._process_change(change)
            except Exception as e:
                logger.error(f"Error processing CDC change: {e}", change=change)
    
    async def _process_change(self, change: Dict[str, Any]) -> None:
        """Process a database change event."""
        table = change["source"]["table"]
        operation = change["op"]  # c=create, u=update, d=delete
        
        topic = self.table_mappings.get(table)
        if not topic:
            return
        
        # Create event based on change
        if table == "agents" and operation in ["c", "u"]:
            event = self._create_agent_event(change)
        elif table == "workflows":
            event = self._create_workflow_event(change)
        elif table == "agent_states":
            event = self._create_state_change_event(change)
        else:
            return
        
        # Publish event transactionally
        async with self.producer.transaction():
            await self.producer.send_event(topic, event)
            
            # Commit the offset to ensure exactly-once
            # (In real implementation, would commit Debezium offset)
    
    def _create_agent_event(self, change: Dict[str, Any]) -> AgentEvent:
        """Create agent event from database change."""
        after = change.get("after", {})
        before = change.get("before", {})
        
        return AgentEvent(
            event_type=EventType.AGENT_STATE_CHANGED,
            agent_id=after["id"],
            agent_type=after["type"],
            agent_version=after.get("version", "1.0.0"),
            state_before=before,
            state_after=after,
            data={
                "operation": change["op"],
                "source": "cdc",
                "timestamp": change["ts_ms"]
            }
        )
    
    def _create_workflow_event(self, change: Dict[str, Any]) -> WorkflowEvent:
        """Create workflow event from database change."""
        # Similar implementation for workflow changes
        pass
    
    def _create_state_change_event(self, change: Dict[str, Any]) -> EventSchema:
        """Create state change event from database change."""
        # Implementation for state changes
        pass


# Factory functions
def create_temporal_connector(
    temporal_host: str = "localhost:7233",
    kafka_config: Optional[ProducerConfig] = None
) -> TemporalKafkaConnector:
    """Create a Temporal-Kafka connector."""
    temporal_client = TemporalClient(temporal_host)
    producer = EventProducer(kafka_config or ProducerConfig())
    
    return TemporalKafkaConnector(temporal_client, producer)


def create_state_connector(
    redis_url: str = "redis://localhost:6379",
    postgres_url: Optional[str] = None
) -> StateStoreConnector:
    """Create a state store connector."""
    return StateStoreConnector(redis_url, postgres_url)


def create_cdc_connector(
    database_config: Dict[str, Any],
    kafka_config: Optional[ProducerConfig] = None
) -> CDCConnector:
    """Create a CDC connector."""
    producer = TransactionalProducer(kafka_config or ProducerConfig())
    
    return CDCConnector(producer, database_config)