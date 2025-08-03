"""
AURA Intelligence Event Mesh

High-throughput, event-driven architecture with Kafka for:
- Agent communication
- State synchronization
- Event sourcing
- Stream processing
- Distributed coordination

Key Features:
- Exactly-once semantics (EOS v2)
- Schema registry integration
- Stream processing with Kafka Streams
- Dead letter queue handling
- Comprehensive observability
"""

from .schemas import (
    EventSchema,
    AgentEvent,
    WorkflowEvent,
    SystemEvent,
    EventType,
    EventPriority
)

from .producers import (
    EventProducer,
    TransactionalProducer,
    BatchProducer,
    ProducerConfig
)

from .consumers import (
    EventConsumer,
    ConsumerGroup,
    StreamProcessor,
    ConsumerConfig
)

from .streams import (
    AgentEventStream,
    WorkflowEventStream,
    EventAggregator,
    StreamTopology
)

from .connectors import (
    TemporalKafkaConnector,
    StateStoreConnector,
    CDCConnector
)

# from .registry import (
#     SchemaRegistry,
#     EventSerializer,
#     EventDeserializer
# )  # Temporarily commented out - module not available

__version__ = "1.0.0"

__all__ = [
    # Schemas
    "EventSchema",
    "AgentEvent",
    "WorkflowEvent",
    "SystemEvent",
    "EventType",
    "EventPriority",
    
    # Producers
    "EventProducer",
    "TransactionalProducer",
    "BatchProducer",
    "ProducerConfig",
    
    # Consumers
    "EventConsumer",
    "ConsumerGroup",
    "StreamProcessor",
    "ConsumerConfig",
    
    # Streams
    "AgentEventStream",
    "WorkflowEventStream",
    "EventAggregator",
    "StreamTopology",
    
    # Connectors
    "TemporalKafkaConnector",
    "StateStoreConnector",
    "CDCConnector",
    
    # Registry
    "SchemaRegistry",
    "EventSerializer",
    "EventDeserializer"
]