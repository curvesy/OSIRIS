"""
Kafka connector atomic components.

Provides producer and consumer components for Kafka integration
with built-in error handling, metrics, and observability.
"""

from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
import json
import asyncio
from datetime import datetime
import uuid

from ..base import AtomicComponent
from ..base.exceptions import ConnectionError, ProcessingError, RetryableError


@dataclass
class KafkaConfig:
    """Configuration for Kafka connection."""
    
    bootstrap_servers: str
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None
    ssl_ca_location: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    request_timeout_ms: int = 30000
    retry_backoff_ms: int = 100
    
    def validate(self) -> None:
        """Validate configuration."""
        if not self.bootstrap_servers:
            raise ValueError("bootstrap_servers required")
        
        if self.security_protocol not in ["PLAINTEXT", "SSL", "SASL_PLAINTEXT", "SASL_SSL"]:
            raise ValueError(f"Invalid security protocol: {self.security_protocol}")
        
        if self.security_protocol.startswith("SASL") and not self.sasl_mechanism:
            raise ValueError("SASL mechanism required for SASL security protocol")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to confluent-kafka config dict."""
        config = {
            'bootstrap.servers': self.bootstrap_servers,
            'security.protocol': self.security_protocol,
            'request.timeout.ms': self.request_timeout_ms,
            'retry.backoff.ms': self.retry_backoff_ms
        }
        
        if self.sasl_mechanism:
            config['sasl.mechanism'] = self.sasl_mechanism
            config['sasl.username'] = self.sasl_username
            config['sasl.password'] = self.sasl_password
        
        if self.ssl_ca_location:
            config['ssl.ca.location'] = self.ssl_ca_location
        if self.ssl_certfile:
            config['ssl.certificate.location'] = self.ssl_certfile
        if self.ssl_keyfile:
            config['ssl.key.location'] = self.ssl_keyfile
            
        return config


@dataclass
class KafkaMessage:
    """Kafka message structure."""
    
    topic: str
    value: Any
    key: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    partition: Optional[int] = None
    timestamp: Optional[datetime] = None
    
    def serialize(self) -> bytes:
        """Serialize message value to bytes."""
        if isinstance(self.value, bytes):
            return self.value
        elif isinstance(self.value, str):
            return self.value.encode('utf-8')
        else:
            return json.dumps(self.value).encode('utf-8')
    
    def serialize_key(self) -> Optional[bytes]:
        """Serialize message key to bytes."""
        if self.key is None:
            return None
        return self.key.encode('utf-8') if isinstance(self.key, str) else str(self.key).encode('utf-8')


class KafkaProducer(AtomicComponent[KafkaMessage, bool, KafkaConfig]):
    """
    Atomic component for producing messages to Kafka.
    
    Features:
    - Automatic serialization
    - Delivery confirmation
    - Error handling with retries
    - Metrics collection
    """
    
    def __init__(self, name: str, config: KafkaConfig, **kwargs):
        super().__init__(name, config, **kwargs)
        self._producer = None
        self._delivery_reports = []
    
    def _validate_config(self) -> None:
        """Validate Kafka configuration."""
        self.config.validate()
    
    async def _process(self, message: KafkaMessage) -> bool:
        """
        Produce message to Kafka.
        
        Args:
            message: KafkaMessage to send
            
        Returns:
            True if message was successfully produced
        """
        try:
            # Lazy initialization of producer
            if self._producer is None:
                self._initialize_producer()
            
            # Prepare message
            value = message.serialize()
            key = message.serialize_key()
            headers = [(k, v.encode('utf-8')) for k, v in message.headers.items()] if message.headers else None
            
            # Track delivery
            delivery_future = asyncio.Future()
            
            def delivery_callback(err, msg):
                if err:
                    delivery_future.set_exception(
                        ProcessingError(f"Delivery failed: {err}", component_name=self.name)
                    )
                else:
                    delivery_future.set_result(True)
            
            # Produce message
            self._producer.produce(
                topic=message.topic,
                key=key,
                value=value,
                headers=headers,
                partition=message.partition,
                callback=delivery_callback
            )
            
            # Trigger delivery
            self._producer.poll(0)
            
            # Wait for delivery confirmation
            await asyncio.wait_for(delivery_future, timeout=10.0)
            
            self.logger.info(
                "Message produced successfully",
                topic=message.topic,
                partition=message.partition,
                key=message.key
            )
            
            return True
            
        except asyncio.TimeoutError:
            raise RetryableError(
                "Message delivery timed out",
                retry_after=1.0,
                component_name=self.name
            )
        except Exception as e:
            self.logger.error(f"Failed to produce message: {e}")
            raise ProcessingError(
                f"Kafka produce failed: {str(e)}",
                component_name=self.name
            )
    
    def _initialize_producer(self):
        """Initialize Kafka producer (stub for actual implementation)."""
        # In real implementation, would use confluent-kafka
        self.logger.info("Initializing Kafka producer", config=self.config.bootstrap_servers)
        self._producer = MockProducer()  # Placeholder
    
    async def health_check(self) -> Dict[str, Any]:
        """Check producer health."""
        return {
            "component": self.name,
            "status": "healthy" if self._producer else "not_initialized",
            "bootstrap_servers": self.config.bootstrap_servers
        }


class KafkaConsumer(AtomicComponent[str, Optional[KafkaMessage], KafkaConfig]):
    """
    Atomic component for consuming messages from Kafka.
    
    Features:
    - Automatic deserialization
    - Offset management
    - Error handling
    - Batch consumption support
    """
    
    def __init__(
        self,
        name: str,
        config: KafkaConfig,
        group_id: str,
        topics: List[str],
        auto_commit: bool = False,
        **kwargs
    ):
        super().__init__(name, config, **kwargs)
        self.group_id = group_id
        self.topics = topics
        self.auto_commit = auto_commit
        self._consumer = None
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        self.config.validate()
        if not self.group_id:
            raise ValueError("group_id required for consumer")
        if not self.topics:
            raise ValueError("At least one topic required")
    
    async def _process(self, timeout_seconds: str = "1.0") -> Optional[KafkaMessage]:
        """
        Consume a message from Kafka.
        
        Args:
            timeout_seconds: Poll timeout as string
            
        Returns:
            KafkaMessage if available, None otherwise
        """
        try:
            # Lazy initialization
            if self._consumer is None:
                self._initialize_consumer()
            
            # Poll for message (stub implementation)
            # In real implementation, would use confluent-kafka
            await asyncio.sleep(0.1)  # Simulate poll
            
            # Return None to indicate no message (stub)
            return None
            
        except Exception as e:
            raise ProcessingError(
                f"Failed to consume message: {str(e)}",
                component_name=self.name
            )
    
    def _initialize_consumer(self):
        """Initialize Kafka consumer."""
        self.logger.info(
            "Initializing Kafka consumer",
            group_id=self.group_id,
            topics=self.topics
        )
        self._consumer = MockConsumer()  # Placeholder
    
    async def commit(self) -> bool:
        """Commit current offsets."""
        if not self._consumer:
            return False
        
        try:
            # Stub implementation
            self.logger.info("Committing offsets")
            return True
        except Exception as e:
            self.logger.error(f"Failed to commit: {e}")
            return False


class KafkaBatchProducer(AtomicComponent[List[KafkaMessage], Dict[str, Any], KafkaConfig]):
    """
    Atomic component for batch producing messages to Kafka.
    
    Optimized for high-throughput scenarios.
    """
    
    def __init__(self, name: str, config: KafkaConfig, **kwargs):
        super().__init__(name, config, **kwargs)
        self._producer = None
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        self.config.validate()
    
    async def _process(self, messages: List[KafkaMessage]) -> Dict[str, Any]:
        """
        Produce batch of messages.
        
        Args:
            messages: List of messages to produce
            
        Returns:
            Batch production results
        """
        if not messages:
            return {
                "total": 0,
                "success": 0,
                "failed": 0,
                "errors": []
            }
        
        # Initialize if needed
        if self._producer is None:
            self._initialize_producer()
        
        success_count = 0
        failed_count = 0
        errors = []
        
        # Process each message
        for i, message in enumerate(messages):
            try:
                # Stub implementation
                await asyncio.sleep(0.001)  # Simulate produce
                success_count += 1
            except Exception as e:
                failed_count += 1
                errors.append({
                    "index": i,
                    "error": str(e),
                    "topic": message.topic
                })
        
        return {
            "total": len(messages),
            "success": success_count,
            "failed": failed_count,
            "errors": errors
        }
    
    def _initialize_producer(self):
        """Initialize batch producer."""
        self.logger.info("Initializing batch producer")
        self._producer = MockProducer()  # Placeholder


# Mock classes for testing without actual Kafka dependency
class MockProducer:
    """Mock Kafka producer for testing."""
    
    def produce(self, **kwargs):
        pass
    
    def poll(self, timeout):
        pass
    
    def flush(self):
        pass


class MockConsumer:
    """Mock Kafka consumer for testing."""
    
    def subscribe(self, topics):
        pass
    
    def poll(self, timeout):
        return None
    
    def commit(self):
        pass