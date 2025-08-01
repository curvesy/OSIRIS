"""
Stream Processing for AURA Intelligence Event Mesh

Implements stream processing patterns:
- Event aggregation and windowing
- State management
- Stream joins and enrichment
- Complex event processing (CEP)
"""

from typing import Dict, Any, Optional, List, Callable, Tuple, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import asyncio
import json

import structlog
from opentelemetry import trace, metrics

from .schemas import EventSchema, AgentEvent, WorkflowEvent, EventType
from .producers import EventProducer, ProducerConfig
from .consumers import EventProcessor, EventConsumer, ConsumerConfig

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

# Metrics
stream_events_processed = meter.create_counter(
    name="stream.events.processed",
    description="Number of events processed by streams",
    unit="1"
)

stream_state_size = meter.create_gauge(
    name="stream.state.size",
    description="Size of stream state"
)

stream_window_latency = meter.create_histogram(
    name="stream.window.latency",
    description="Window processing latency",
    unit="ms"
)


class WindowType(str, Enum):
    """Window types for stream processing."""
    TUMBLING = "tumbling"      # Fixed-size, non-overlapping
    SLIDING = "sliding"        # Fixed-size, overlapping
    SESSION = "session"        # Variable-size, gap-based
    HOPPING = "hopping"        # Fixed-size, fixed overlap


@dataclass
class StreamConfig:
    """Configuration for stream processing."""
    # Window configuration
    window_type: WindowType = WindowType.TUMBLING
    window_size: timedelta = timedelta(minutes=1)
    window_slide: Optional[timedelta] = None  # For sliding windows
    session_gap: Optional[timedelta] = None   # For session windows
    
    # State configuration
    state_backend: str = "memory"  # memory, rocksdb, redis
    state_retention: timedelta = timedelta(hours=24)
    checkpoint_interval: timedelta = timedelta(minutes=5)
    
    # Processing configuration
    parallelism: int = 1
    buffer_size: int = 1000
    late_arrival_grace: timedelta = timedelta(minutes=5)
    
    # Output configuration
    output_topic: Optional[str] = None
    output_batch_size: int = 100
    output_batch_timeout: timedelta = timedelta(seconds=1)


class StateStore:
    """
    In-memory state store for stream processing.
    
    In production, this would be backed by RocksDB or Redis.
    """
    
    def __init__(self, name: str, retention: timedelta):
        self.name = name
        self.retention = retention
        self.store: Dict[str, Any] = {}
        self.timestamps: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from state store."""
        async with self._lock:
            # Check if expired
            if key in self.timestamps:
                if datetime.utcnow() - self.timestamps[key] > self.retention:
                    del self.store[key]
                    del self.timestamps[key]
                    return None
            
            return self.store.get(key)
    
    async def put(self, key: str, value: Any) -> None:
        """Put value into state store."""
        async with self._lock:
            self.store[key] = value
            self.timestamps[key] = datetime.utcnow()
            
            # Update metrics
            stream_state_size.set(len(self.store), {"store": self.name})
    
    async def delete(self, key: str) -> None:
        """Delete value from state store."""
        async with self._lock:
            self.store.pop(key, None)
            self.timestamps.pop(key, None)
    
    async def get_all(self) -> Dict[str, Any]:
        """Get all values from state store."""
        async with self._lock:
            # Clean expired entries
            current_time = datetime.utcnow()
            expired_keys = [
                key for key, timestamp in self.timestamps.items()
                if current_time - timestamp > self.retention
            ]
            
            for key in expired_keys:
                del self.store[key]
                del self.timestamps[key]
            
            return self.store.copy()
    
    async def clear(self) -> None:
        """Clear all values from state store."""
        async with self._lock:
            self.store.clear()
            self.timestamps.clear()


class StreamTopology:
    """
    Defines the stream processing topology.
    
    Similar to Kafka Streams DSL.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.sources: Dict[str, List[str]] = {}  # source_name -> topics
        self.processors: Dict[str, Callable] = {}  # processor_name -> function
        self.sinks: Dict[str, str] = {}  # sink_name -> topic
        self.edges: List[Tuple[str, str]] = []  # (from_node, to_node)
        self.state_stores: Dict[str, StateStore] = {}
    
    def add_source(self, name: str, topics: List[str]) -> "StreamTopology":
        """Add a source node."""
        self.sources[name] = topics
        return self
    
    def add_processor(
        self,
        name: str,
        processor: Callable,
        parents: List[str],
        state_stores: Optional[List[str]] = None
    ) -> "StreamTopology":
        """Add a processor node."""
        self.processors[name] = processor
        
        for parent in parents:
            self.edges.append((parent, name))
        
        if state_stores:
            for store in state_stores:
                if store not in self.state_stores:
                    raise ValueError(f"State store {store} not found")
        
        return self
    
    def add_sink(self, name: str, topic: str, parents: List[str]) -> "StreamTopology":
        """Add a sink node."""
        self.sinks[name] = topic
        
        for parent in parents:
            self.edges.append((parent, name))
        
        return self
    
    def add_state_store(self, name: str, retention: timedelta) -> "StreamTopology":
        """Add a state store."""
        self.state_stores[name] = StateStore(name, retention)
        return self


class AgentEventStream(EventProcessor):
    """
    Stream processor for agent events.
    
    Features:
    - Agent performance tracking
    - Decision pattern analysis
    - Anomaly detection
    - Performance aggregation
    """
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.topology = self._build_topology()
        
        # State stores
        self.agent_metrics = StateStore("agent_metrics", config.state_retention)
        self.decision_patterns = StateStore("decision_patterns", config.state_retention)
        self.anomalies = StateStore("anomalies", config.state_retention)
        
        # Windows
        self.windows: Dict[str, List[AgentEvent]] = defaultdict(list)
        self._window_task: Optional[asyncio.Task] = None
        
        # Output producer
        self.producer: Optional[EventProducer] = None
    
    def _build_topology(self) -> StreamTopology:
        """Build the stream processing topology."""
        topology = StreamTopology("agent_event_stream")
        
        # Add state stores
        topology.add_state_store("agent_metrics", self.config.state_retention)
        topology.add_state_store("decision_patterns", self.config.state_retention)
        topology.add_state_store("anomalies", self.config.state_retention)
        
        # Define topology
        topology.add_source("agent_events", ["agent.events"]) \
            .add_processor(
                "performance_tracker",
                self._track_performance,
                ["agent_events"],
                ["agent_metrics"]
            ) \
            .add_processor(
                "decision_analyzer",
                self._analyze_decisions,
                ["agent_events"],
                ["decision_patterns"]
            ) \
            .add_processor(
                "anomaly_detector",
                self._detect_anomalies,
                ["performance_tracker"],
                ["anomalies"]
            )
        
        if self.config.output_topic:
            topology.add_sink(
                "output",
                self.config.output_topic,
                ["anomaly_detector"]
            )
        
        return topology
    
    async def start(self):
        """Start the stream processor."""
        # Initialize producer if output configured
        if self.config.output_topic:
            producer_config = ProducerConfig()
            self.producer = EventProducer(producer_config)
            await self.producer.start()
        
        # Start window processing
        self._window_task = asyncio.create_task(self._process_windows())
        
        logger.info(f"Agent event stream started")
    
    async def stop(self):
        """Stop the stream processor."""
        # Stop window processing
        if self._window_task:
            self._window_task.cancel()
            try:
                await self._window_task
            except asyncio.CancelledError:
                pass
        
        # Stop producer
        if self.producer:
            await self.producer.stop()
        
        logger.info("Agent event stream stopped")
    
    async def process(self, event: EventSchema) -> None:
        """Process an agent event."""
        if not isinstance(event, AgentEvent):
            return
        
        with tracer.start_as_current_span(
            "stream.agent.process",
            attributes={
                "agent.id": event.agent_id,
                "agent.type": event.agent_type,
                "event.type": event.event_type.value
            }
        ) as span:
            try:
                # Add to window
                window_key = self._get_window_key(event)
                self.windows[window_key].append(event)
                
                # Process through topology
                await self._track_performance(event)
                
                if event.event_type == EventType.AGENT_DECISION_MADE:
                    await self._analyze_decisions(event)
                
                # Check for anomalies
                anomaly = await self._detect_anomalies(event)
                
                if anomaly and self.producer:
                    await self.producer.send_event(
                        self.config.output_topic,
                        anomaly
                    )
                
                stream_events_processed.add(
                    1,
                    {
                        "stream": "agent_events",
                        "event_type": event.event_type.value
                    }
                )
                
                span.set_status(trace.Status(trace.StatusCode.OK))
                
            except Exception as e:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    
    def _get_window_key(self, event: AgentEvent) -> str:
        """Get window key for event."""
        if self.config.window_type == WindowType.TUMBLING:
            window_start = int(
                event.timestamp.timestamp() / self.config.window_size.total_seconds()
            )
            return f"{event.agent_type}:{window_start}"
        else:
            # Simplified - would need more complex logic for other window types
            return f"{event.agent_type}:{event.agent_id}"
    
    async def _track_performance(self, event: AgentEvent) -> None:
        """Track agent performance metrics."""
        if event.event_type != EventType.AGENT_COMPLETED:
            return
        
        # Get current metrics
        metrics_key = f"{event.agent_type}:{event.agent_id}"
        metrics = await self.agent_metrics.get(metrics_key) or {
            "total_executions": 0,
            "total_duration_ms": 0,
            "total_tokens": 0,
            "error_count": 0,
            "last_updated": None
        }
        
        # Update metrics
        metrics["total_executions"] += 1
        
        if event.duration_ms:
            metrics["total_duration_ms"] += event.duration_ms
        
        if event.tokens_used:
            metrics["total_tokens"] += sum(event.tokens_used.values())
        
        metrics["last_updated"] = datetime.utcnow().isoformat()
        
        # Calculate averages
        metrics["avg_duration_ms"] = (
            metrics["total_duration_ms"] / metrics["total_executions"]
        )
        
        # Store updated metrics
        await self.agent_metrics.put(metrics_key, metrics)
    
    async def _analyze_decisions(self, event: AgentEvent) -> None:
        """Analyze agent decision patterns."""
        if event.event_type != EventType.AGENT_DECISION_MADE:
            return
        
        decision_data = event.data
        decision = decision_data.get("decision", "unknown")
        confidence = decision_data.get("confidence", 0)
        
        # Track decision patterns
        pattern_key = f"{event.agent_type}:{decision}"
        pattern = await self.decision_patterns.get(pattern_key) or {
            "count": 0,
            "total_confidence": 0,
            "reasons": defaultdict(int)
        }
        
        pattern["count"] += 1
        pattern["total_confidence"] += confidence
        pattern["avg_confidence"] = pattern["total_confidence"] / pattern["count"]
        
        reason = decision_data.get("reason", "unknown")
        pattern["reasons"][reason] += 1
        
        await self.decision_patterns.put(pattern_key, pattern)
    
    async def _detect_anomalies(self, event: AgentEvent) -> Optional[EventSchema]:
        """Detect anomalies in agent behavior."""
        if event.event_type != EventType.AGENT_COMPLETED:
            return None
        
        # Get agent metrics
        metrics_key = f"{event.agent_type}:{event.agent_id}"
        metrics = await self.agent_metrics.get(metrics_key)
        
        if not metrics or metrics["total_executions"] < 10:
            return None  # Not enough data
        
        # Check for anomalies
        anomalies_detected = []
        
        # Duration anomaly
        if event.duration_ms:
            avg_duration = metrics["avg_duration_ms"]
            if event.duration_ms > avg_duration * 2:
                anomalies_detected.append({
                    "type": "slow_execution",
                    "value": event.duration_ms,
                    "threshold": avg_duration * 2
                })
        
        # Token usage anomaly
        if event.tokens_used:
            total_tokens = sum(event.tokens_used.values())
            avg_tokens = metrics["total_tokens"] / metrics["total_executions"]
            
            if total_tokens > avg_tokens * 3:
                anomalies_detected.append({
                    "type": "high_token_usage",
                    "value": total_tokens,
                    "threshold": avg_tokens * 3
                })
        
        if anomalies_detected:
            # Create anomaly event
            from .schemas import SystemEvent
            
            anomaly_event = SystemEvent.create_alert_event(
                component="agent_stream",
                instance_id=self.topology.name,
                alert_type="agent_anomaly",
                message=f"Anomalies detected in agent {event.agent_id}",
                severity="warning",
                details={
                    "agent_id": event.agent_id,
                    "agent_type": event.agent_type,
                    "anomalies": anomalies_detected
                }
            )
            
            # Store anomaly
            await self.anomalies.put(
                f"{event.agent_id}:{datetime.utcnow().timestamp()}",
                anomaly_event.dict()
            )
            
            return anomaly_event
        
        return None
    
    async def _process_windows(self) -> None:
        """Process completed windows."""
        while True:
            try:
                await asyncio.sleep(self.config.window_size.total_seconds())
                
                # Process completed windows
                current_time = datetime.utcnow()
                completed_windows = []
                
                for window_key, events in self.windows.items():
                    if events:
                        # Check if window is complete
                        window_end = events[0].timestamp + self.config.window_size
                        if window_end <= current_time:
                            completed_windows.append(window_key)
                
                # Process windows
                for window_key in completed_windows:
                    events = self.windows.pop(window_key)
                    await self._process_window(window_key, events)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing windows: {e}")
    
    async def _process_window(
        self,
        window_key: str,
        events: List[AgentEvent]
    ) -> None:
        """Process a completed window of events."""
        start_time = datetime.utcnow()
        
        with tracer.start_as_current_span(
            "stream.window.process",
            attributes={
                "window.key": window_key,
                "window.size": len(events)
            }
        ) as span:
            try:
                # Aggregate window data
                aggregation = {
                    "window_key": window_key,
                    "event_count": len(events),
                    "start_time": min(e.timestamp for e in events),
                    "end_time": max(e.timestamp for e in events),
                    "agent_types": defaultdict(int),
                    "event_types": defaultdict(int),
                    "total_duration_ms": 0,
                    "total_tokens": 0,
                    "error_count": 0
                }
                
                for event in events:
                    aggregation["agent_types"][event.agent_type] += 1
                    aggregation["event_types"][event.event_type.value] += 1
                    
                    if event.duration_ms:
                        aggregation["total_duration_ms"] += event.duration_ms
                    
                    if event.tokens_used:
                        aggregation["total_tokens"] += sum(event.tokens_used.values())
                    
                    if event.event_type == EventType.AGENT_FAILED:
                        aggregation["error_count"] += 1
                
                # Calculate rates
                window_duration = (
                    aggregation["end_time"] - aggregation["start_time"]
                ).total_seconds()
                
                if window_duration > 0:
                    aggregation["events_per_second"] = (
                        aggregation["event_count"] / window_duration
                    )
                    aggregation["error_rate"] = (
                        aggregation["error_count"] / aggregation["event_count"]
                    )
                
                logger.info(
                    "Window processed",
                    window_key=window_key,
                    aggregation=aggregation
                )
                
                # Record metrics
                duration = (datetime.utcnow() - start_time).total_seconds() * 1000
                stream_window_latency.record(
                    duration,
                    {"stream": "agent_events", "window_type": self.config.window_type.value}
                )
                
                span.set_status(trace.Status(trace.StatusCode.OK))
                
            except Exception as e:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise


class WorkflowEventStream(EventProcessor):
    """
    Stream processor for workflow events.
    
    Features:
    - Workflow completion tracking
    - Step duration analysis
    - Bottleneck detection
    - SLA monitoring
    """
    
    def __init__(self, config: StreamConfig):
        self.config = config
        
        # State stores
        self.workflow_state = StateStore("workflow_state", config.state_retention)
        self.step_metrics = StateStore("step_metrics", config.state_retention)
        self.sla_violations = StateStore("sla_violations", config.state_retention)
    
    async def process(self, event: EventSchema) -> None:
        """Process a workflow event."""
        if not isinstance(event, WorkflowEvent):
            return
        
        # Track workflow state
        await self._track_workflow_state(event)
        
        # Analyze step performance
        if event.event_type == EventType.WORKFLOW_STEP_COMPLETED:
            await self._analyze_step_performance(event)
        
        # Check SLA compliance
        await self._check_sla_compliance(event)
    
    async def _track_workflow_state(self, event: WorkflowEvent) -> None:
        """Track workflow execution state."""
        state_key = f"{event.workflow_id}:{event.run_id}"
        
        if event.event_type == EventType.WORKFLOW_STARTED:
            state = {
                "workflow_id": event.workflow_id,
                "workflow_type": event.workflow_type,
                "run_id": event.run_id,
                "start_time": event.timestamp,
                "steps_completed": [],
                "status": "running"
            }
            await self.workflow_state.put(state_key, state)
            
        elif event.event_type == EventType.WORKFLOW_STEP_COMPLETED:
            state = await self.workflow_state.get(state_key)
            if state:
                state["steps_completed"].append({
                    "step": event.current_step,
                    "timestamp": event.timestamp,
                    "duration_ms": event.data.get("duration_ms", 0)
                })
                await self.workflow_state.put(state_key, state)
                
        elif event.event_type in [EventType.WORKFLOW_COMPLETED, EventType.WORKFLOW_FAILED]:
            state = await self.workflow_state.get(state_key)
            if state:
                state["end_time"] = event.timestamp
                state["status"] = "completed" if event.event_type == EventType.WORKFLOW_COMPLETED else "failed"
                state["total_duration_ms"] = (
                    event.timestamp - state["start_time"]
                ).total_seconds() * 1000
                await self.workflow_state.put(state_key, state)
    
    async def _analyze_step_performance(self, event: WorkflowEvent) -> None:
        """Analyze workflow step performance."""
        step_key = f"{event.workflow_type}:{event.current_step}"
        
        metrics = await self.step_metrics.get(step_key) or {
            "count": 0,
            "total_duration_ms": 0,
            "min_duration_ms": float('inf'),
            "max_duration_ms": 0
        }
        
        duration = event.data.get("duration_ms", 0)
        
        metrics["count"] += 1
        metrics["total_duration_ms"] += duration
        metrics["avg_duration_ms"] = metrics["total_duration_ms"] / metrics["count"]
        metrics["min_duration_ms"] = min(metrics["min_duration_ms"], duration)
        metrics["max_duration_ms"] = max(metrics["max_duration_ms"], duration)
        
        await self.step_metrics.put(step_key, metrics)
    
    async def _check_sla_compliance(self, event: WorkflowEvent) -> None:
        """Check if workflow meets SLA requirements."""
        # Example SLA: Workflows should complete within 5 minutes
        if event.event_type == EventType.WORKFLOW_COMPLETED:
            state_key = f"{event.workflow_id}:{event.run_id}"
            state = await self.workflow_state.get(state_key)
            
            if state and state.get("total_duration_ms", 0) > 300000:  # 5 minutes
                violation = {
                    "workflow_id": event.workflow_id,
                    "workflow_type": event.workflow_type,
                    "run_id": event.run_id,
                    "duration_ms": state["total_duration_ms"],
                    "sla_ms": 300000,
                    "timestamp": event.timestamp
                }
                
                await self.sla_violations.put(
                    f"{event.workflow_id}:{event.timestamp.timestamp()}",
                    violation
                )
                
                logger.warning(
                    "SLA violation detected",
                    workflow_id=event.workflow_id,
                    duration_ms=state["total_duration_ms"]
                )


class EventAggregator:
    """
    Aggregates events from multiple streams.
    
    Features:
    - Multi-stream joins
    - Cross-stream correlation
    - Global metrics computation
    """
    
    def __init__(self, streams: List[EventProcessor]):
        self.streams = streams
        self.global_state = StateStore("global_state", timedelta(hours=24))
    
    async def aggregate(self) -> Dict[str, Any]:
        """Aggregate data from all streams."""
        aggregation = {
            "timestamp": datetime.utcnow(),
            "total_events": 0,
            "events_by_type": defaultdict(int),
            "active_agents": set(),
            "active_workflows": set(),
            "error_rate": 0,
            "avg_latency_ms": 0
        }
        
        # Aggregate from each stream's state
        # This is simplified - real implementation would be more sophisticated
        
        return aggregation