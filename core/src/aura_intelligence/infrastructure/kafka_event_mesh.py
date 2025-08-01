"""
🚀 Kafka Event Mesh for High-Throughput Streaming
Atomic module for Kafka-based event streaming with backpressure and fault tolerance
"""

import asyncio
from typing import Protocol, Callable, Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import json

from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.errors import KafkaError
import structlog
from prometheus_client import Counter, Histogram, Gauge

from ..common.circuit_breaker import CircuitBreaker
from ..common.config import get_config
from ..common.errors import AuraError

logger = structlog.get_logger(__name__)

# Metrics
MESSAGES_SENT = Counter('kafka_messages_sent_total', 'Total messages sent to Kafka', ['topic'])
MESSAGES_RECEIVED = Counter('kafka_messages_received_total', 'Total messages received from Kafka', ['topic'])
SEND_LATENCY = Histogram('kafka_send_latency_seconds', 'Kafka send latency', ['topic'])
PROCESSING_ERRORS = Counter('kafka_processing_errors_total', 'Kafka processing errors', ['error_type'])
CONSUMER_LAG = Gauge('kafka_consumer_lag', 'Kafka consumer lag', ['topic', 'partition'])


@dataclass
class KafkaConfig:
    """Kafka configuration"""
    bootstrap_servers: List[str] = field(default_factory=lambda: ['localhost:9092'])
    max_batch_size: int = 16384
    linger_ms: int = 10
    compression_type: str = 'lz4'
    acks: str = 'all'
    enable_idempotence: bool = True
    max_in_flight_requests: int = 5
    request_timeout_ms: int = 30000
    retry_backoff_ms: int = 100
    
    @classmethod
    def from_config(cls) -> 'KafkaConfig':
        """Load from application config"""
        config = get_config()
        return cls(
            bootstrap_servers=config.get('kafka.bootstrap_servers', cls.bootstrap_servers),
            max_batch_size=config.get('kafka.max_batch_size', cls.max_batch_size),
            compression_type=config.get('kafka.compression_type', cls.compression_type)
        )


@dataclass
class Event:
    """Base event structure"""
    id: str
    type: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventMesh(Protocol):
    """Event mesh protocol"""
    async def publish(self, topic: str, event: Event) -> None:
        """Publish event to topic"""
        ...
    
    async def subscribe(self, topic: str, handler: Callable[[Event], None]) -> None:
        """Subscribe to topic with handler"""
        ...
    
    async def create_stream(self, topic: str, partitions: int = 3) -> None:
        """Create topic with partitions"""
        ...


class KafkaEventMesh:
    """High-throughput Kafka event mesh implementation"""
    
    def __init__(self, config: Optional[KafkaConfig] = None):
        self.config = config or KafkaConfig.from_config()
        self.producer: Optional[AIOKafkaProducer] = None
        self.consumers: Dict[str, AIOKafkaConsumer] = {}
        self.handlers: Dict[str, List[Callable]] = {}
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30,
            expected_exception=KafkaError
        )
        self._running = False
        
    async def initialize(self) -> None:
        """Initialize Kafka connections"""
        try:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.config.bootstrap_servers,
                compression_type=self.config.compression_type,
                max_batch_size=self.config.max_batch_size,
                linger_ms=self.config.linger_ms,
                acks=self.config.acks,
                enable_idempotence=self.config.enable_idempotence,
                max_in_flight_requests_per_connection=self.config.max_in_flight_requests,
                request_timeout_ms=self.config.request_timeout_ms,
                retry_backoff_ms=self.config.retry_backoff_ms,
                value_serializer=self._serialize_event
            )
            await self.producer.start()
            self._running = True
            logger.info("Kafka event mesh initialized", servers=self.config.bootstrap_servers)
        except Exception as e:
            logger.error("Failed to initialize Kafka", error=str(e))
            raise AuraError(f"Kafka initialization failed: {e}")
    
    async def publish(self, topic: str, event: Event) -> None:
        """Publish event with circuit breaker protection"""
        if not self._running:
            raise AuraError("Event mesh not initialized")
            
        async def _send():
            with SEND_LATENCY.labels(topic=topic).time():
                await self.producer.send_and_wait(topic, event)
                MESSAGES_SENT.labels(topic=topic).inc()
                
        try:
            await self.circuit_breaker.call(_send)
        except Exception as e:
            PROCESSING_ERRORS.labels(error_type=type(e).__name__).inc()
            logger.error("Failed to publish event", topic=topic, error=str(e))
            raise
    
    async def subscribe(self, topic: str, handler: Callable[[Event], None]) -> None:
        """Subscribe to topic with automatic consumer management"""
        if topic not in self.handlers:
            self.handlers[topic] = []
            await self._create_consumer(topic)
        self.handlers[topic].append(handler)
        
    async def _create_consumer(self, topic: str) -> None:
        """Create and start consumer for topic"""
        consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=self.config.bootstrap_servers,
            group_id=f"aura-streaming-tda-{topic}",
            enable_auto_commit=True,
            auto_commit_interval_ms=1000,
            value_deserializer=self._deserialize_event
        )
        await consumer.start()
        self.consumers[topic] = consumer
        
        # Start consumer loop
        asyncio.create_task(self._consume_loop(topic, consumer))
        
    async def _consume_loop(self, topic: str, consumer: AIOKafkaConsumer) -> None:
        """Consumer loop with error handling"""
        while self._running:
            try:
                async for msg in consumer:
                    MESSAGES_RECEIVED.labels(topic=topic).inc()
                    
                    # Update lag metric
                    lag = consumer.highwater(msg.partition) - msg.offset
                    CONSUMER_LAG.labels(topic=topic, partition=msg.partition).set(lag)
                    
                    # Process with all handlers
                    for handler in self.handlers.get(topic, []):
                        try:
                            await handler(msg.value)
                        except Exception as e:
                            PROCESSING_ERRORS.labels(error_type="handler_error").inc()
                            logger.error("Handler error", topic=topic, error=str(e))
                            
            except Exception as e:
                PROCESSING_ERRORS.labels(error_type="consumer_error").inc()
                logger.error("Consumer error", topic=topic, error=str(e))
                await asyncio.sleep(5)  # Backoff before retry
    
    async def create_stream(self, topic: str, partitions: int = 3) -> None:
        """Create topic (requires admin client in production)"""
        # In production, use KafkaAdminClient
        # This is a placeholder for the interface
        logger.info("Topic creation requested", topic=topic, partitions=partitions)
    
    def _serialize_event(self, event: Event) -> bytes:
        """Serialize event to JSON bytes"""
        data = {
            'id': event.id,
            'type': event.type,
            'timestamp': event.timestamp.isoformat(),
            'data': event.data,
            'metadata': event.metadata
        }
        return json.dumps(data).encode('utf-8')
    
    def _deserialize_event(self, data: bytes) -> Event:
        """Deserialize JSON bytes to event"""
        obj = json.loads(data.decode('utf-8'))
        return Event(
            id=obj['id'],
            type=obj['type'],
            timestamp=datetime.fromisoformat(obj['timestamp']),
            data=obj['data'],
            metadata=obj.get('metadata', {})
        )
    
    async def close(self) -> None:
        """Graceful shutdown"""
        self._running = False
        
        if self.producer:
            await self.producer.stop()
            
        for consumer in self.consumers.values():
            await consumer.stop()
            
        logger.info("Kafka event mesh closed")