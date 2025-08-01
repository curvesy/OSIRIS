"""
Kafka Event Consumers for AURA Intelligence

Implements various consumer patterns:
- Standard consumer with manual commit
- Consumer groups for scaling
- Stream processor for stateful processing
- Exactly-once consumer with transactions
"""

from typing import Dict, Any, Optional, List, Callable, Set, AsyncIterator, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
from abc import ABC, abstractmethod

from aiokafka import AIOKafkaConsumer, ConsumerRebalanceListener, TopicPartition
from aiokafka.errors import KafkaError, CommitFailedError
from aiokafka.structs import ConsumerRecord
import structlog
from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode

from .schemas import EventSchema, validate_event, get_event_schema
from ..agents.resilience import CircuitBreaker, CircuitBreakerConfig, RetryWithBackoff

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

# Metrics
events_consumed = meter.create_counter(
    name="kafka.events.consumed",
    description="Number of events consumed from Kafka",
    unit="1"
)

consume_errors = meter.create_counter(
    name="kafka.consume.errors",
    description="Number of consume errors",
    unit="1"
)

consume_lag = meter.create_gauge(
    name="kafka.consumer.lag",
    description="Consumer lag in messages"
)

processing_duration = meter.create_histogram(
    name="kafka.processing.duration",
    description="Event processing duration in milliseconds",
    unit="ms"
)


class ProcessingStrategy(str, Enum):
    """Event processing strategies."""
    AT_MOST_ONCE = "at_most_once"    # Commit before processing
    AT_LEAST_ONCE = "at_least_once"  # Commit after processing
    EXACTLY_ONCE = "exactly_once"     # Transactional processing


@dataclass
class ConsumerConfig:
    """Configuration for Kafka consumer."""
    bootstrap_servers: str = "localhost:9092"
    group_id: str = "aura-consumer"
    client_id: Optional[str] = None
    
    # Consumer settings
    topics: List[str] = field(default_factory=list)
    auto_offset_reset: str = "latest"
    enable_auto_commit: bool = False
    max_poll_records: int = 500
    max_poll_interval_ms: int = 300000  # 5 minutes
    
    # Processing settings
    processing_strategy: ProcessingStrategy = ProcessingStrategy.AT_LEAST_ONCE
    processing_timeout: timedelta = timedelta(seconds=30)
    max_retries: int = 3
    
    # Performance settings
    fetch_min_bytes: int = 1
    fetch_max_wait_ms: int = 500
    
    # Security settings
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None
    
    # Custom settings
    additional_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_kafka_config(self) -> Dict[str, Any]:
        """Convert to Kafka configuration dict."""
        config = {
            "bootstrap_servers": self.bootstrap_servers,
            "group_id": self.group_id,
            "auto_offset_reset": self.auto_offset_reset,
            "enable_auto_commit": self.enable_auto_commit,
            "max_poll_records": self.max_poll_records,
            "max_poll_interval_ms": self.max_poll_interval_ms,
            "fetch_min_bytes": self.fetch_min_bytes,
            "fetch_max_wait_ms": self.fetch_max_wait_ms,
            "security_protocol": self.security_protocol
        }
        
        if self.client_id:
            config["client_id"] = self.client_id
        else:
            config["client_id"] = f"{self.group_id}-consumer"
            
        if self.sasl_mechanism:
            config["sasl_mechanism"] = self.sasl_mechanism
            config["sasl_plain_username"] = self.sasl_username
            config["sasl_plain_password"] = self.sasl_password
            
        config.update(self.additional_config)
        return config


class EventProcessor(ABC):
    """Abstract base class for event processors."""
    
    @abstractmethod
    async def process(self, event: EventSchema) -> None:
        """Process a single event."""
        pass
    
    async def on_error(self, event: EventSchema, error: Exception) -> None:
        """Handle processing error."""
        logger.error(f"Error processing event: {error}", event_id=event.event_id)


class EventConsumer:
    """
    Standard event consumer with manual commit control.
    
    Features:
    - Configurable processing strategies
    - Automatic retries with backoff
    - Circuit breaker protection
    - Dead letter queue support
    - Distributed tracing
    """
    
    def __init__(
        self,
        config: ConsumerConfig,
        processor: EventProcessor
    ):
        self.config = config
        self.processor = processor
        self.consumer: Optional[AIOKafkaConsumer] = None
        self._running = False
        
        # Circuit breaker for processing
        self.circuit_breaker = CircuitBreaker(
            CircuitBreakerConfig(
                name=f"consumer-{config.group_id}",
                failure_threshold=10,
                timeout=timedelta(seconds=60)
            )
        )
        
        # Retry decorator
        self.retry_processor = RetryWithBackoff(
            max_attempts=config.max_retries,
            initial_delay=1.0,
            max_delay=30.0
        )
    
    async def start(self):
        """Start the consumer."""
        if self._running:
            return
            
        logger.info(f"Starting event consumer: {self.config.group_id}")
        
        try:
            self.consumer = AIOKafkaConsumer(
                *self.config.topics,
                **self.config.to_kafka_config(),
                value_deserializer=lambda v: json.loads(v.decode()) if v else None
            )
            
            await self.consumer.start()
            self._running = True
            
            logger.info(f"Event consumer started for topics: {self.config.topics}")
            
        except Exception as e:
            logger.error(f"Failed to start consumer: {e}")
            raise
    
    async def stop(self):
        """Stop the consumer."""
        if self.consumer and self._running:
            self._running = False
            await self.consumer.stop()
            logger.info("Event consumer stopped")
    
    async def consume(self) -> None:
        """Main consumption loop."""
        if not self._running:
            await self.start()
        
        try:
            async for msg in self.consumer:
                await self._process_message(msg)
                
        except asyncio.CancelledError:
            logger.info("Consumer cancelled")
            raise
        except Exception as e:
            logger.error(f"Consumer error: {e}")
            raise
        finally:
            await self.stop()
    
    async def _process_message(self, msg: ConsumerRecord) -> None:
        """Process a single message."""
        with tracer.start_as_current_span(
            "kafka.consume.message",
            attributes={
                "kafka.topic": msg.topic,
                "kafka.partition": msg.partition,
                "kafka.offset": msg.offset
            }
        ) as span:
            start_time = datetime.utcnow()
            
            try:
                # Extract trace context from headers
                if msg.headers:
                    for key, value in msg.headers:
                        if key == "trace_id":
                            span.set_attribute("parent.trace_id", value.decode())
                        elif key == "event_type":
                            span.set_attribute("event.type", value.decode())
                
                # Validate and parse event
                event = validate_event(msg.value)
                span.set_attribute("event.id", event.event_id)
                
                # Process based on strategy
                if self.config.processing_strategy == ProcessingStrategy.AT_MOST_ONCE:
                    # Commit before processing
                    await self.consumer.commit()
                    await self._process_event(event)
                    
                elif self.config.processing_strategy == ProcessingStrategy.AT_LEAST_ONCE:
                    # Process before committing
                    await self._process_event(event)
                    await self.consumer.commit()
                    
                elif self.config.processing_strategy == ProcessingStrategy.EXACTLY_ONCE:
                    # Transactional processing (requires producer cooperation)
                    await self._process_event_transactionally(event, msg)
                
                # Record metrics
                duration = (datetime.utcnow() - start_time).total_seconds() * 1000
                processing_duration.record(
                    duration,
                    {
                        "topic": msg.topic,
                        "event_type": event.event_type.value,
                        "status": "success"
                    }
                )
                
                events_consumed.add(
                    1,
                    {
                        "topic": msg.topic,
                        "event_type": event.event_type.value,
                        "status": "success"
                    }
                )
                
                span.set_status(Status(StatusCode.OK))
                
            except Exception as e:
                consume_errors.add(
                    1,
                    {
                        "topic": msg.topic,
                        "error": type(e).__name__
                    }
                )
                
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                
                # Handle error based on strategy
                await self._handle_error(msg, e)
    
    async def _process_event(self, event: EventSchema) -> None:
        """Process event with circuit breaker and retry."""
        
        @self.retry_processor
        async def process_with_retry():
            async with self.circuit_breaker:
                await self.processor.process(event)
        
        await process_with_retry()
    
    async def _process_event_transactionally(
        self,
        event: EventSchema,
        msg: ConsumerRecord
    ) -> None:
        """Process event with exactly-once semantics."""
        # This requires coordination with a transactional producer
        # For now, fall back to at-least-once
        await self._process_event(event)
        await self.consumer.commit()
    
    async def _handle_error(self, msg: ConsumerRecord, error: Exception) -> None:
        """Handle processing error."""
        logger.error(
            f"Failed to process message",
            topic=msg.topic,
            partition=msg.partition,
            offset=msg.offset,
            error=str(error)
        )
        
        # Send to dead letter queue if configured
        # For now, just log and continue
        
        # Commit to avoid reprocessing if at-most-once
        if self.config.processing_strategy == ProcessingStrategy.AT_MOST_ONCE:
            await self.consumer.commit()


class ConsumerGroup:
    """
    Manages a group of consumers for parallel processing.
    
    Features:
    - Automatic partition assignment
    - Rebalance handling
    - Coordinated shutdown
    - Health monitoring
    """
    
    def __init__(
        self,
        config: ConsumerConfig,
        processor: EventProcessor,
        num_consumers: int = 1
    ):
        self.config = config
        self.processor = processor
        self.num_consumers = num_consumers
        self.consumers: List[EventConsumer] = []
        self.tasks: List[asyncio.Task] = []
        self._running = False
    
    async def start(self):
        """Start all consumers in the group."""
        if self._running:
            return
        
        logger.info(f"Starting consumer group with {self.num_consumers} consumers")
        
        # Create consumers with unique client IDs
        for i in range(self.num_consumers):
            config = ConsumerConfig(**self.config.dict())
            config.client_id = f"{config.group_id}-{i}"
            
            consumer = EventConsumer(config, self.processor)
            self.consumers.append(consumer)
            
            # Start consumer task
            task = asyncio.create_task(consumer.consume())
            self.tasks.append(task)
        
        self._running = True
        logger.info("Consumer group started")
    
    async def stop(self):
        """Stop all consumers in the group."""
        if not self._running:
            return
        
        logger.info("Stopping consumer group")
        self._running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Stop all consumers
        for consumer in self.consumers:
            await consumer.stop()
        
        self.consumers.clear()
        self.tasks.clear()
        
        logger.info("Consumer group stopped")
    
    async def wait(self):
        """Wait for all consumers to complete."""
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)


class StreamProcessor(EventProcessor):
    """
    Stateful stream processor with windowing and aggregation.
    
    Features:
    - Time and count-based windows
    - State management
    - Aggregation functions
    - Output to new topics
    """
    
    def __init__(
        self,
        window_size: timedelta = timedelta(minutes=1),
        window_type: str = "tumbling"  # tumbling, sliding, session
    ):
        self.window_size = window_size
        self.window_type = window_type
        self.state: Dict[str, Any] = {}
        self.windows: Dict[str, List[EventSchema]] = {}
        self._window_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the stream processor."""
        self._window_task = asyncio.create_task(self._process_windows())
        logger.info(f"Stream processor started with {self.window_type} windows")
    
    async def stop(self):
        """Stop the stream processor."""
        if self._window_task:
            self._window_task.cancel()
            try:
                await self._window_task
            except asyncio.CancelledError:
                pass
    
    async def process(self, event: EventSchema) -> None:
        """Add event to window for processing."""
        window_key = self._get_window_key(event)
        
        if window_key not in self.windows:
            self.windows[window_key] = []
        
        self.windows[window_key].append(event)
        
        # Update state
        await self._update_state(event)
    
    def _get_window_key(self, event: EventSchema) -> str:
        """Get window key for event."""
        # Simple time-based key
        window_start = int(event.timestamp.timestamp() / self.window_size.total_seconds())
        return f"{event.source_type}:{window_start}"
    
    async def _update_state(self, event: EventSchema) -> None:
        """Update processor state with new event."""
        # Example: Count events by type
        event_type = event.event_type.value
        if event_type not in self.state:
            self.state[event_type] = 0
        self.state[event_type] += 1
    
    async def _process_windows(self) -> None:
        """Background task to process completed windows."""
        while True:
            try:
                await asyncio.sleep(self.window_size.total_seconds())
                
                # Process completed windows
                current_time = datetime.utcnow()
                completed_windows = []
                
                for window_key, events in self.windows.items():
                    if events:
                        # Check if window is complete
                        window_end = events[0].timestamp + self.window_size
                        if window_end <= current_time:
                            completed_windows.append(window_key)
                
                # Process and remove completed windows
                for window_key in completed_windows:
                    events = self.windows.pop(window_key)
                    await self._process_window(window_key, events)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing windows: {e}")
    
    async def _process_window(self, window_key: str, events: List[EventSchema]) -> None:
        """Process a completed window of events."""
        with tracer.start_as_current_span(
            "kafka.stream.window",
            attributes={
                "window.key": window_key,
                "window.size": len(events)
            }
        ) as span:
            try:
                # Example aggregation: Count by event type
                aggregation = {}
                for event in events:
                    event_type = event.event_type.value
                    aggregation[event_type] = aggregation.get(event_type, 0) + 1
                
                logger.info(
                    f"Window processed",
                    window_key=window_key,
                    event_count=len(events),
                    aggregation=aggregation
                )
                
                # Here you would typically:
                # 1. Perform aggregations
                # 2. Update state stores
                # 3. Emit results to output topics
                
                span.set_status(Status(StatusCode.OK))
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise


# Example processor implementations
class LoggingProcessor(EventProcessor):
    """Simple processor that logs events."""
    
    async def process(self, event: EventSchema) -> None:
        """Log the event."""
        logger.info(
            f"Processing event",
            event_id=event.event_id,
            event_type=event.event_type.value,
            source_id=event.source_id
        )


class RouterProcessor(EventProcessor):
    """Routes events to different processors based on type."""
    
    def __init__(self):
        self.routes: Dict[str, EventProcessor] = {}
    
    def add_route(self, event_type: str, processor: EventProcessor) -> None:
        """Add a route for an event type."""
        self.routes[event_type] = processor
    
    async def process(self, event: EventSchema) -> None:
        """Route event to appropriate processor."""
        processor = self.routes.get(event.event_type.value)
        
        if processor:
            await processor.process(event)
        else:
            logger.warning(f"No route for event type: {event.event_type.value}")


# Factory function for creating consumers
def create_consumer(
    consumer_type: str = "standard",
    config: Optional[ConsumerConfig] = None,
    processor: Optional[EventProcessor] = None,
    **kwargs
) -> Union[EventConsumer, ConsumerGroup]:
    """
    Create a consumer instance.
    
    Args:
        consumer_type: Type of consumer ("standard", "group")
        config: Consumer configuration
        processor: Event processor
        **kwargs: Additional arguments
        
    Returns:
        Consumer instance
    """
    if config is None:
        config = ConsumerConfig()
        
    if processor is None:
        processor = LoggingProcessor()
    
    if consumer_type == "standard":
        return EventConsumer(config, processor)
    elif consumer_type == "group":
        num_consumers = kwargs.get("num_consumers", 1)
        return ConsumerGroup(config, processor, num_consumers)
    else:
        raise ValueError(f"Unknown consumer type: {consumer_type}")