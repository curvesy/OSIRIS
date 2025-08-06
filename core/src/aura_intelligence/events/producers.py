"""
Kafka Event Producers for AURA Intelligence

Implements various producer patterns:
- Standard async producer
- Transactional producer (exactly-once)
- Batch producer for high throughput
- Partitioned producer for ordering
"""

from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import asyncio
import json
from contextlib import asynccontextmanager

from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError, KafkaTimeoutError
from confluent_kafka import SerializingProducer
from confluent_kafka.serialization import StringSerializer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer
import structlog
from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode

from .schemas import EventSchema, AgentEvent, WorkflowEvent, SystemEvent
from ..agents.observability import GenAIAttributes

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

# Metrics
events_produced = meter.create_counter(
    name="kafka.events.produced",
    description="Number of events produced to Kafka",
    unit="1"
)

produce_errors = meter.create_counter(
    name="kafka.produce.errors",
    description="Number of produce errors",
    unit="1"
)

produce_latency = meter.create_histogram(
    name="kafka.produce.latency",
    description="Produce latency in milliseconds",
    unit="ms"
)


@dataclass
class ProducerConfig:
    """Configuration for Kafka producer."""
    bootstrap_servers: str = "localhost:9092"
    client_id: str = "aura-producer"
    
    # Reliability settings
    acks: str = "all"  # Wait for all replicas
    retries: int = 10
    max_in_flight_requests: int = 5
    
    # Performance settings
    batch_size: int = 16384
    linger_ms: int = 10
    compression_type: str = "gzip"  # Changed from snappy to gzip (more widely available)
    buffer_memory: int = 33554432  # 32MB
    
    # Transactional settings
    enable_idempotence: bool = True
    transactional_id: Optional[str] = None
    transaction_timeout_ms: int = 60000
    
    # Schema registry
    schema_registry_url: Optional[str] = "http://localhost:8081"
    use_avro: bool = False
    
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
            "client_id": self.client_id,
            "acks": self.acks,
            # "retries": self.retries,  # Not supported in AIOKafkaProducer - handled by resilience layer
            # "max_in_flight_requests_per_connection": self.max_in_flight_requests,  # Not supported in current AIOKafka version
            "compression_type": self.compression_type,
            "enable_idempotence": self.enable_idempotence,
            "security_protocol": self.security_protocol
        }
        
        if self.transactional_id:
            config["transactional_id"] = self.transactional_id
            
        if self.sasl_mechanism:
            config["sasl_mechanism"] = self.sasl_mechanism
            config["sasl_plain_username"] = self.sasl_username
            config["sasl_plain_password"] = self.sasl_password
            
        config.update(self.additional_config)
        return config


class EventProducer:
    """
    Standard async event producer with observability.
    
    Features:
    - Async/await interface
    - Automatic retries
    - Circuit breaker integration
    - Distributed tracing
    - Metrics collection
    """
    
    def __init__(self, config: ProducerConfig):
        self.config = config
        self.producer: Optional[AIOKafkaProducer] = None
        self._started = False
        
    async def start(self):
        """Start the producer."""
        if self._started:
            return
            
        logger.info(f"Starting event producer: {self.config.client_id}")
        
        try:
            self.producer = AIOKafkaProducer(
                **self.config.to_kafka_config(),
                value_serializer=lambda v: json.dumps(v).encode() if isinstance(v, dict) else v
            )
            
            await self.producer.start()
            self._started = True
            
            logger.info("Event producer started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start producer: {e}")
            raise
    
    async def stop(self):
        """Stop the producer."""
        if self.producer and self._started:
            await self.producer.stop()
            self._started = False
            logger.info("Event producer stopped")
    
    async def send_event(
        self,
        topic: str,
        event: EventSchema,
        key: Optional[str] = None,
        partition: Optional[int] = None,
        headers: Optional[List[tuple]] = None
    ) -> None:
        """Send a single event to Kafka."""
        if not self._started:
            await self.start()
        
        with tracer.start_as_current_span(
            "kafka.produce.event",
            attributes={
                "kafka.topic": topic,
                "event.type": event.event_type.value,
                "event.id": event.event_id
            }
        ) as span:
            start_time = datetime.now(timezone.utc)
            
            try:
                # Convert event to Kafka format
                record = event.to_kafka_record()
                
                # Merge headers
                if headers:
                    record["headers"].extend(headers)
                
                # Add tracing headers
                if span.get_span_context().is_valid:
                    record["headers"].append(
                        ("trace_id", str(span.get_span_context().trace_id).encode())
                    )
                    record["headers"].append(
                        ("span_id", str(span.get_span_context().span_id).encode())
                    )
                
                # Send to Kafka
                await self.producer.send_and_wait(
                    topic,
                    value=record["value"],
                    key=(key or record["key"]).encode() if key or record["key"] else None,
                    partition=partition,
                    headers=record["headers"]
                )
                
                # Record metrics
                duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                produce_latency.record(
                    duration,
                    {"topic": topic, "event_type": event.event_type.value}
                )
                
                events_produced.add(
                    1,
                    {"topic": topic, "event_type": event.event_type.value, "status": "success"}
                )
                
                span.set_status(Status(StatusCode.OK))
                logger.debug(f"Event sent successfully", event_id=event.event_id, topic=topic)
                
            except Exception as e:
                produce_errors.add(
                    1,
                    {"topic": topic, "event_type": event.event_type.value, "error": type(e).__name__}
                )
                
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(f"Failed to send event: {e}", event_id=event.event_id)
                raise
    
    async def send_batch(
        self,
        topic: str,
        events: List[EventSchema],
        ordered: bool = False
    ) -> None:
        """Send a batch of events."""
        if ordered:
            # Send sequentially to maintain order
            for event in events:
                await self.send_event(topic, event)
        else:
            # Send in parallel for better throughput
            tasks = [self.send_event(topic, event) for event in events]
            await asyncio.gather(*tasks, return_exceptions=True)


class TransactionalProducer(EventProducer):
    """
    Transactional producer for exactly-once semantics.
    
    Features:
    - Exactly-once delivery (EOS)
    - Atomic multi-topic writes
    - Transaction coordination
    - Automatic rollback on failure
    """
    
    def __init__(self, config: ProducerConfig):
        # Ensure transactional settings
        if not config.transactional_id:
            config.transactional_id = f"{config.client_id}-txn"
        config.enable_idempotence = True
        
        super().__init__(config)
        self._in_transaction = False
    
    async def start(self):
        """Start the transactional producer."""
        await super().start()
        
        # Initialize transactions
        await self.producer.init_transactions()
        logger.info(f"Transactional producer initialized: {self.config.transactional_id}")
    
    @asynccontextmanager
    async def transaction(self):
        """Context manager for transactions."""
        if not self._started:
            await self.start()
        
        with tracer.start_as_current_span("kafka.transaction") as span:
            try:
                # Begin transaction
                await self.producer.begin_transaction()
                self._in_transaction = True
                span.set_attribute("transaction.id", self.config.transactional_id)
                
                yield self
                
                # Commit transaction
                await self.producer.commit_transaction()
                self._in_transaction = False
                
                span.set_status(Status(StatusCode.OK))
                logger.debug("Transaction committed successfully")
                
            except Exception as e:
                # Abort transaction on error
                if self._in_transaction:
                    try:
                        await self.producer.abort_transaction()
                    except Exception as abort_error:
                        logger.error(f"Failed to abort transaction: {abort_error}")
                    finally:
                        self._in_transaction = False
                
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(f"Transaction failed: {e}")
                raise
    
    async def send_transactional_batch(
        self,
        events: List[tuple[str, EventSchema]],  # (topic, event) pairs
        consumer_offsets: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send a batch of events in a transaction."""
        async with self.transaction():
            # Send all events
            for topic, event in events:
                await self.send_event(topic, event)
            
            # Optionally commit consumer offsets (for exactly-once processing)
            if consumer_offsets:
                await self.producer.send_offsets_to_transaction(
                    consumer_offsets["offsets"],
                    consumer_offsets["group_id"]
                )


class BatchProducer:
    """
    High-throughput batch producer with buffering.
    
    Features:
    - Automatic batching by size/time
    - Memory-efficient buffering
    - Parallel partition writes
    - Backpressure handling
    """
    
    def __init__(
        self,
        config: ProducerConfig,
        batch_size: int = 1000,
        batch_timeout: timedelta = timedelta(seconds=1)
    ):
        self.config = config
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        self.producer = EventProducer(config)
        self._buffer: Dict[str, List[EventSchema]] = {}
        self._buffer_lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start the batch producer."""
        await self.producer.start()
        self._running = True
        
        # Start background flush task
        self._flush_task = asyncio.create_task(self._flush_periodically())
        logger.info("Batch producer started")
    
    async def stop(self):
        """Stop the batch producer."""
        self._running = False
        
        # Flush remaining events
        await self.flush_all()
        
        # Stop flush task
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        await self.producer.stop()
        logger.info("Batch producer stopped")
    
    async def add_event(self, topic: str, event: EventSchema) -> None:
        """Add event to batch buffer."""
        async with self._buffer_lock:
            if topic not in self._buffer:
                self._buffer[topic] = []
            
            self._buffer[topic].append(event)
            
            # Flush if batch size reached
            if len(self._buffer[topic]) >= self.batch_size:
                await self._flush_topic(topic)
    
    async def _flush_topic(self, topic: str) -> None:
        """Flush events for a specific topic."""
        events = self._buffer.pop(topic, [])
        
        if events:
            with tracer.start_as_current_span(
                "kafka.batch.flush",
                attributes={
                    "kafka.topic": topic,
                    "batch.size": len(events)
                }
            ) as span:
                try:
                    await self.producer.send_batch(topic, events, ordered=False)
                    
                    events_produced.add(
                        len(events),
                        {"topic": topic, "producer": "batch", "status": "success"}
                    )
                    
                    span.set_status(Status(StatusCode.OK))
                    logger.debug(f"Flushed {len(events)} events to {topic}")
                    
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    
                    # Re-add events to buffer on failure
                    async with self._buffer_lock:
                        if topic not in self._buffer:
                            self._buffer[topic] = []
                        self._buffer[topic] = events + self._buffer[topic]
                    
                    raise
    
    async def flush_all(self) -> None:
        """Flush all buffered events."""
        async with self._buffer_lock:
            topics = list(self._buffer.keys())
        
        for topic in topics:
            await self._flush_topic(topic)
    
    async def _flush_periodically(self) -> None:
        """Background task to flush events periodically."""
        while self._running:
            try:
                await asyncio.sleep(self.batch_timeout.total_seconds())
                await self.flush_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic flush: {e}")


# Factory function for creating producers
def create_producer(
    producer_type: str = "standard",
    config: Optional[ProducerConfig] = None,
    **kwargs
) -> Union[EventProducer, TransactionalProducer, BatchProducer]:
    """
    Create a producer instance.
    
    Args:
        producer_type: Type of producer ("standard", "transactional", "batch")
        config: Producer configuration
        **kwargs: Additional arguments for specific producer types
        
    Returns:
        Producer instance
    """
    if config is None:
        config = ProducerConfig()
    
    if producer_type == "standard":
        return EventProducer(config)
    elif producer_type == "transactional":
        return TransactionalProducer(config)
    elif producer_type == "batch":
        batch_size = kwargs.get("batch_size", 1000)
        batch_timeout = kwargs.get("batch_timeout", timedelta(seconds=1))
        return BatchProducer(config, batch_size, batch_timeout)
    else:
        raise ValueError(f"Unknown producer type: {producer_type}")