"""
🤖 Base Agent Class - Foundation for The Collective

Enterprise-grade base agent implementation with:
- OpenTelemetry instrumentation for observability
- Integration with UnifiedMemory and ACP protocol
- Structured logging and error handling
- Health monitoring and metrics collection
- Lifecycle management and graceful shutdown

Based on the proven patterns from kakakagan.md research.
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, field

try:
    from opentelemetry import trace, metrics
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.metrics import Counter, Histogram, Gauge
except ImportError:
    # OpenTelemetry is optional - provide fallbacks
    trace = None
    metrics = None
    Status = None
    StatusCode = None
    Counter = None
    Histogram = None
    Gauge = None

from ..schemas.acp import ACPEnvelope, ACPEndpoint, ACPResponse, MessageType, Priority
from ..schemas.state import AgentState, DossierEntry, ActionRecord
from ..schemas.log import AgentActionEvent, ActionType, ActionResult, ImpactLevel
from ..memory.unified import UnifiedMemory, MemoryTier, QueryResult
from ..communication.protocol import ACPProtocol, MessageBus

# Initialize OpenTelemetry components with fallbacks
if trace:
    tracer = trace.get_tracer(__name__)
else:
    # Fallback tracer that does nothing
    class NoOpSpan:
        def set_attributes(self, attrs):
            pass
        def set_status(self, status):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    
    class NoOpTracer:
        def start_as_current_span(self, name):
            # Return a decorator function for @tracer.start_as_current_span usage
            def decorator(func):
                return func  # Just return the original function
            return decorator
    tracer = NoOpTracer()

if metrics:
    meter = metrics.get_meter(__name__)
else:
    # Fallback meter that does nothing
    class NoOpMeter:
        def create_counter(self, **kwargs):
            return NoOpCounter()
        def create_histogram(self, **kwargs):
            return NoOpHistogram()
        def create_gauge(self, **kwargs):
            return NoOpGauge()
    meter = NoOpMeter()

# Fallback metric classes
class NoOpCounter:
    def add(self, value, attributes=None):
        pass

class NoOpHistogram:
    def record(self, value, attributes=None):
        pass

class NoOpGauge:
    def set(self, value, attributes=None):
        pass

# Fallback trace functions
if not trace:
    def get_current_span():
        return NoOpSpan()
    trace = type('trace', (), {'get_current_span': get_current_span})()


class AgentRole(str, Enum):
    """Standard agent roles in The Collective."""
    OBSERVER = "observer"
    ANALYST = "analyst"
    EXECUTOR = "executor"
    COORDINATOR = "coordinator"
    ROUTER = "router"
    CONSENSUS = "consensus"
    SUPERVISOR = "supervisor"


class AgentCapability(str, Enum):
    """Agent capabilities for routing and discovery."""
    MONITORING = "monitoring"
    ANALYSIS = "analysis"
    EXECUTION = "execution"
    COORDINATION = "coordination"
    LEARNING = "learning"
    COMMUNICATION = "communication"
    DECISION_MAKING = "decision_making"
    PATTERN_RECOGNITION = "pattern_recognition"


class AgentStatus(str, Enum):
    """Agent operational status."""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


@dataclass
class AgentMetrics:
    """Agent performance metrics."""
    tasks_processed: int = 0
    tasks_successful: int = 0
    tasks_failed: int = 0
    average_processing_time_ms: float = 0.0
    memory_queries: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    uptime_seconds: float = 0.0
    last_activity: Optional[str] = None
    
    def get_success_rate(self) -> float:
        """Calculate task success rate."""
        if self.tasks_processed == 0:
            return 0.0
        return self.tasks_successful / self.tasks_processed
    
    def get_failure_rate(self) -> float:
        """Calculate task failure rate."""
        if self.tasks_processed == 0:
            return 0.0
        return self.tasks_failed / self.tasks_processed


class BaseAgent(ABC):
    """
    Base class for all agents in The Collective.
    
    Provides common functionality for agent lifecycle, communication,
    memory access, and observability.
    """
    
    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        capabilities: List[AgentCapability],
        memory: UnifiedMemory,
        protocol: ACPProtocol,
        message_bus: MessageBus,
        instance_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique agent identifier
            role: Agent role in The Collective
            capabilities: List of agent capabilities
            memory: Unified memory interface
            protocol: ACP protocol for communication
            message_bus: High-level message bus
            instance_id: Instance ID for scaled deployments
            config: Agent configuration
        """
        self.agent_id = agent_id
        self.role = role
        self.capabilities = capabilities
        self.memory = memory
        self.protocol = protocol
        self.message_bus = message_bus
        self.instance_id = instance_id or f"{agent_id}_{uuid.uuid4().hex[:8]}"
        self.config = config or {}
        
        # Agent endpoint
        self.endpoint = ACPEndpoint(
            agent_id=self.agent_id,
            role=self.role.value,
            instance_id=self.instance_id,
            capabilities=[cap.value for cap in self.capabilities]
        )
        
        # State management
        self.status = AgentStatus.INITIALIZING
        self.metrics = AgentMetrics()
        self.start_time = time.time()
        
        # Task management
        self.active_tasks: Dict[str, AgentState] = {}
        self.task_handlers: Dict[str, Callable] = {}
        
        # OpenTelemetry metrics
        self._setup_metrics()
        
        # Background tasks
        self._running = False
        self._background_tasks: Set[asyncio.Task] = set()
    
    def _setup_metrics(self) -> None:
        """Setup OpenTelemetry metrics."""
        self.task_counter = meter.create_counter(
            name=f"agent_tasks_total",
            description="Total number of tasks processed by agent",
            unit="1"
        )
        
        self.task_duration = meter.create_histogram(
            name=f"agent_task_duration_ms",
            description="Task processing duration in milliseconds",
            unit="ms"
        )
        
        self.memory_query_counter = meter.create_counter(
            name=f"agent_memory_queries_total",
            description="Total number of memory queries",
            unit="1"
        )
        
        self.message_counter = meter.create_counter(
            name=f"agent_messages_total",
            description="Total number of messages sent/received",
            unit="1"
        )
        
        self.health_gauge = meter.create_gauge(
            name=f"agent_health_status",
            description="Agent health status (1=healthy, 0=unhealthy)",
            unit="1"
        )
    
    async def start(self) -> None:
        """Start the agent and initialize services."""
        if self._running:
            return
        
        try:
            # Register message handlers
            await self._register_message_handlers()
            
            # Start background tasks
            self._running = True
            
            health_task = asyncio.create_task(self._health_monitor_loop())
            self._background_tasks.add(health_task)
            health_task.add_done_callback(self._background_tasks.discard)
            
            metrics_task = asyncio.create_task(self._metrics_update_loop())
            self._background_tasks.add(metrics_task)
            metrics_task.add_done_callback(self._background_tasks.discard)
            
            # Agent-specific initialization
            await self.initialize()
            
            self.status = AgentStatus.HEALTHY
            self._update_health_metric()
            
        except Exception as e:
            self.status = AgentStatus.UNHEALTHY
            self._update_health_metric()
            raise
    
    async def stop(self) -> None:
        """Stop the agent and cleanup resources."""
        self.status = AgentStatus.SHUTTING_DOWN
        self._update_health_metric()
        
        self._running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Agent-specific cleanup
        await self.cleanup()
        
        self.status = AgentStatus.STOPPED
        self._update_health_metric()
    
    @abstractmethod
    async def initialize(self) -> None:
        """Agent-specific initialization logic."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Agent-specific cleanup logic."""
        pass
    
    @abstractmethod
    async def process_task(self, state: AgentState) -> AgentState:
        """
        Process a task and return updated state.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state
        """
        pass
    
    async def _register_message_handlers(self) -> None:
        """Register ACP message handlers."""
        self.protocol.register_handler(
            MessageType.REQUEST,
            self._handle_request
        )
        
        self.protocol.register_handler(
            MessageType.NOTIFICATION,
            self._handle_notification
        )
        
        self.protocol.register_handler(
            MessageType.BROADCAST,
            self._handle_broadcast
        )
    
    @tracer.start_as_current_span("agent_handle_request")
    async def _handle_request(self, envelope: ACPEnvelope) -> ACPResponse:
        """Handle incoming request messages."""
        span = trace.get_current_span()
        span.set_attributes({
            "agent_id": self.agent_id,
            "agent_role": self.role.value,
            "sender": envelope.sender.agent_id,
            "correlation_id": envelope.correlation_id
        })
        
        start_time = time.time()
        
        try:
            payload = envelope.payload
            method = payload.get('method')
            params = payload.get('params', {})
            
            # Route to appropriate handler
            if method in self.task_handlers:
                result = await self.task_handlers[method](params)
                
                processing_time = (time.time() - start_time) * 1000
                self.task_duration.record(processing_time, {
                    "agent_id": self.agent_id,
                    "method": method,
                    "status": "success"
                })
                
                return ACPResponse(
                    success=True,
                    result=result,
                    processing_time_ms=processing_time
                )
            else:
                return ACPResponse(
                    success=False,
                    error=f"Unknown method: {method}",
                    error_code="METHOD_NOT_FOUND"
                )
                
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.task_duration.record(processing_time, {
                "agent_id": self.agent_id,
                "method": payload.get('method', 'unknown'),
                "status": "error"
            })
            
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            
            return ACPResponse(
                success=False,
                error=str(e),
                error_code="PROCESSING_ERROR",
                processing_time_ms=processing_time
            )
    
    async def _handle_notification(self, envelope: ACPEnvelope) -> None:
        """Handle incoming notification messages."""
        # Default implementation - can be overridden by subclasses
        pass
    
    async def _handle_broadcast(self, envelope: ACPEnvelope) -> None:
        """Handle incoming broadcast messages."""
        # Default implementation - can be overridden by subclasses
        pass
    
    @tracer.start_as_current_span("agent_query_memory")
    async def query_memory(
        self,
        query: str,
        tier: MemoryTier = MemoryTier.AUTO,
        limit: int = 10,
        context: Optional[Dict[str, Any]] = None
    ) -> List[QueryResult]:
        """
        Query the unified memory system.
        
        Args:
            query: Query string
            tier: Memory tier to query
            limit: Maximum results
            context: Additional context
            
        Returns:
            List of query results
        """
        span = trace.get_current_span()
        span.set_attributes({
            "agent_id": self.agent_id,
            "query": query[:100],  # Truncate for logging
            "tier": tier.value,
            "limit": limit
        })
        
        try:
            result = await self.memory.query(
                query=query,
                tier=tier,
                limit=limit,
                context=context
            )
            
            self.memory_query_counter.add(1, {
                "agent_id": self.agent_id,
                "tier": tier.value,
                "status": "success"
            })
            
            self.metrics.memory_queries += 1
            
            span.set_attributes({
                "results_count": len(result.results),
                "fusion_confidence": result.fusion_confidence
            })
            
            return result.results
            
        except Exception as e:
            self.memory_query_counter.add(1, {
                "agent_id": self.agent_id,
                "tier": tier.value,
                "status": "error"
            })
            
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
    
    @tracer.start_as_current_span("agent_log_action")
    async def log_action(
        self,
        action_type: ActionType,
        action_name: str,
        action_taken: Dict[str, Any],
        result: ActionResult,
        task_id: str,
        correlation_id: str,
        confidence: float = 0.5,
        impact_level: ImpactLevel = ImpactLevel.LOW,
        context_used: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Log an action to the memory system.
        
        Args:
            action_type: Type of action
            action_name: Name of the action
            action_taken: Description of action
            result: Action result
            task_id: Task ID
            correlation_id: Correlation ID
            confidence: Confidence in action
            impact_level: Impact level
            context_used: Context used for decision
            
        Returns:
            Action event ID
        """
        event = AgentActionEvent(
            agent_id=self.agent_id,
            agent_role=self.role.value,
            task_id=task_id,
            correlation_id=correlation_id,
            action_type=action_type,
            action_name=action_name,
            action_taken=action_taken,
            action_result=result,
            confidence=confidence,
            impact_level=impact_level,
            context_used=context_used or []
        )
        
        # Store in memory system
        await self.memory.store(
            content=event.to_memory_signature(),
            tier=MemoryTier.HOT,
            metadata={
                'event_type': 'agent_action',
                'agent_id': self.agent_id,
                'action_type': action_type.value
            }
        )
        
        return event.event_id
    
    async def _health_monitor_loop(self) -> None:
        """Background task to monitor agent health."""
        while self._running:
            try:
                # Check agent health
                health_ok = await self._check_health()
                
                if health_ok and self.status == AgentStatus.DEGRADED:
                    self.status = AgentStatus.HEALTHY
                elif not health_ok and self.status == AgentStatus.HEALTHY:
                    self.status = AgentStatus.DEGRADED
                
                self._update_health_metric()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in health monitor: {e}")
                await asyncio.sleep(30)
    
    async def _metrics_update_loop(self) -> None:
        """Background task to update metrics."""
        while self._running:
            try:
                # Update uptime
                self.metrics.uptime_seconds = time.time() - self.start_time
                self.metrics.last_activity = datetime.now(timezone.utc).isoformat()
                
                await asyncio.sleep(60)  # Update every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in metrics update: {e}")
                await asyncio.sleep(60)
    
    async def _check_health(self) -> bool:
        """Check agent health status."""
        try:
            # Basic health checks
            if not self._running:
                return False
            
            # Check memory connectivity
            stats = await self.memory.get_memory_stats()
            if not stats:
                return False
            
            # Check if we have too many failed tasks
            if self.metrics.get_failure_rate() > 0.5:  # More than 50% failure rate
                return False
            
            return True
            
        except Exception:
            return False
    
    def _update_health_metric(self) -> None:
        """Update the health metric."""
        health_value = 1.0 if self.status == AgentStatus.HEALTHY else 0.0
        self.health_gauge.set(health_value, {
            "agent_id": self.agent_id,
            "agent_role": self.role.value,
            "status": self.status.value
        })
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status information."""
        return {
            'agent_id': self.agent_id,
            'role': self.role.value,
            'instance_id': self.instance_id,
            'status': self.status.value,
            'capabilities': [cap.value for cap in self.capabilities],
            'metrics': {
                'tasks_processed': self.metrics.tasks_processed,
                'success_rate': self.metrics.get_success_rate(),
                'uptime_seconds': self.metrics.uptime_seconds,
                'memory_queries': self.metrics.memory_queries,
                'messages_sent': self.metrics.messages_sent,
                'messages_received': self.metrics.messages_received
            },
            'active_tasks': len(self.active_tasks),
            'last_activity': self.metrics.last_activity
        }
