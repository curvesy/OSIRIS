"""
🔌 Event Adapters for Streaming TDA
Integrates with Kafka event mesh for real-time topological analysis
"""

import asyncio
import json
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from abc import ABC, abstractmethod

import structlog
from prometheus_client import Counter, Histogram
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.protobuf import ProtobufSerializer, ProtobufDeserializer
from confluent_kafka.serialization import SerializationContext, MessageField

from ..streaming import StreamingTDAProcessor, TDAStatistics
from .parallel_processor import MultiScaleProcessor, ScaleConfig
from ...infrastructure.kafka_event_mesh import KafkaEventMesh, EventMessage
from ...observability.tracing import get_tracer

logger = structlog.get_logger(__name__)

# Metrics
EVENTS_PROCESSED = Counter('tda_events_processed_total', 'Total events processed', ['event_type'])
EVENT_LATENCY = Histogram('tda_event_latency_seconds', 'Event processing latency', ['event_type'])
SCHEMA_ERRORS = Counter('tda_schema_errors_total', 'Schema validation errors')


@dataclass
class PointCloudEvent:
    """Event containing point cloud data for TDA analysis"""
    timestamp: datetime
    source_id: str
    points: List[List[float]]  # N x D array
    metadata: Dict[str, Any]
    event_id: str
    

@dataclass
class TDAResultEvent:
    """Event containing TDA analysis results"""
    timestamp: datetime
    source_id: str
    event_id: str
    scale: str
    num_features: int
    max_persistence: float
    total_persistence: float
    metadata: Dict[str, Any]
    

class EventAdapter(ABC):
    """Base class for event adapters"""
    
    @abstractmethod
    async def process_event(self, event: EventMessage) -> Optional[EventMessage]:
        """Process an incoming event"""
        pass
        

class PointCloudAdapter(EventAdapter):
    """
    Adapter for converting Kafka events to point cloud data
    Supports multiple serialization formats
    """
    
    def __init__(
        self,
        schema_registry_url: Optional[str] = None,
        schema_registry_config: Optional[Dict[str, Any]] = None
    ):
        self.tracer = get_tracer()
        
        # Schema registry for protobuf support
        if schema_registry_url:
            self.schema_registry = SchemaRegistryClient({
                'url': schema_registry_url,
                **(schema_registry_config or {})
            })
            self._init_serializers()
        else:
            self.schema_registry = None
            
    def _init_serializers(self) -> None:
        """Initialize protobuf serializers"""
        # This would be replaced with actual protobuf schema
        # For now, we'll use JSON
        pass
        
    async def process_event(self, event: EventMessage) -> Optional[PointCloudEvent]:
        """Convert Kafka event to PointCloudEvent"""
        with self.tracer.start_as_current_span("point_cloud_adapter") as span:
            span.set_attribute("event_id", event.key)
            
            try:
                # Deserialize based on content type
                if event.headers.get('content-type') == 'application/protobuf':
                    data = self._deserialize_protobuf(event.value)
                else:
                    data = json.loads(event.value)
                    
                # Validate and convert
                points = np.array(data['points'])
                if len(points.shape) != 2:
                    raise ValueError(f"Invalid points shape: {points.shape}")
                    
                cloud_event = PointCloudEvent(
                    timestamp=datetime.fromisoformat(data['timestamp']),
                    source_id=data['source_id'],
                    points=points.tolist(),
                    metadata=data.get('metadata', {}),
                    event_id=event.key
                )
                
                EVENTS_PROCESSED.labels(event_type='point_cloud').inc()
                return cloud_event
                
            except Exception as e:
                logger.error(
                    "point_cloud_adapter_error",
                    event_id=event.key,
                    error=str(e)
                )
                SCHEMA_ERRORS.inc()
                return None
                
    def _deserialize_protobuf(self, data: bytes) -> Dict[str, Any]:
        """Deserialize protobuf data"""
        # Placeholder - would use actual protobuf deserializer
        return json.loads(data)
        

class TDAEventProcessor:
    """
    Main event processor that integrates Kafka with streaming TDA
    """
    
    def __init__(
        self,
        kafka_mesh: KafkaEventMesh,
        tda_processor: Union[StreamingTDAProcessor, MultiScaleProcessor],
        input_topic: str,
        output_topic: str,
        adapter: Optional[EventAdapter] = None
    ):
        self.kafka_mesh = kafka_mesh
        self.tda_processor = tda_processor
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.adapter = adapter or PointCloudAdapter()
        self.tracer = get_tracer()
        
        # Event handlers
        self.pre_process_hooks: List[Callable] = []
        self.post_process_hooks: List[Callable] = []
        
        logger.info(
            "tda_event_processor_initialized",
            input_topic=input_topic,
            output_topic=output_topic
        )
        
    def add_pre_process_hook(self, hook: Callable) -> None:
        """Add pre-processing hook"""
        self.pre_process_hooks.append(hook)
        
    def add_post_process_hook(self, hook: Callable) -> None:
        """Add post-processing hook"""
        self.post_process_hooks.append(hook)
        
    async def start(self) -> None:
        """Start processing events"""
        logger.info("tda_event_processor_started")
        
        # Subscribe to input topic
        await self.kafka_mesh.subscribe([self.input_topic])
        
        try:
            while True:
                # Consume events
                events = await self.kafka_mesh.consume_batch(
                    max_messages=100,
                    timeout_ms=1000
                )
                
                if events:
                    await self._process_batch(events)
                    
        except asyncio.CancelledError:
            logger.info("tda_event_processor_cancelled")
            raise
        except Exception as e:
            logger.error(
                "tda_event_processor_error",
                error=str(e)
            )
            raise
            
    async def _process_batch(self, events: List[EventMessage]) -> None:
        """Process a batch of events"""
        with self.tracer.start_as_current_span("process_event_batch") as span:
            span.set_attribute("batch_size", len(events))
            
            # Convert events to point clouds
            point_clouds = []
            for event in events:
                # Run pre-process hooks
                for hook in self.pre_process_hooks:
                    event = await hook(event)
                    
                # Convert to point cloud
                cloud = await self.adapter.process_event(event)
                if cloud:
                    point_clouds.append(cloud)
                    
            if not point_clouds:
                return
                
            # Combine points from all clouds
            all_points = []
            for cloud in point_clouds:
                all_points.extend(cloud.points)
                
            points_array = np.array(all_points)
            
            # Process with TDA
            start_time = datetime.now()
            
            if isinstance(self.tda_processor, MultiScaleProcessor):
                # Multi-scale processing
                updates = await self.tda_processor.add_points(points_array)
                
                # Create result events for each scale
                for scale, update in updates.items():
                    result_event = TDAResultEvent(
                        timestamp=datetime.now(),
                        source_id="multi_scale_tda",
                        event_id=f"{scale}_{datetime.now().timestamp()}",
                        scale=scale,
                        num_features=len(update.added_features),
                        max_persistence=max(
                            (f.death - f.birth for f in update.added_features),
                            default=0.0
                        ),
                        total_persistence=sum(
                            f.death - f.birth for f in update.added_features
                        ),
                        metadata={
                            "removed_features": len(update.removed_features),
                            "modified_features": len(update.modified_features)
                        }
                    )
                    
                    # Send result
                    await self._send_result(result_event)
                    
            else:
                # Single scale processing
                diagram = await self.tda_processor.process_batch(points_array)
                
                result_event = TDAResultEvent(
                    timestamp=datetime.now(),
                    source_id="streaming_tda",
                    event_id=f"single_{datetime.now().timestamp()}",
                    scale="default",
                    num_features=len(diagram.features),
                    max_persistence=diagram.max_persistence,
                    total_persistence=diagram.total_persistence,
                    metadata={}
                )
                
                await self._send_result(result_event)
                
            # Record metrics
            latency = (datetime.now() - start_time).total_seconds()
            EVENT_LATENCY.labels(event_type='batch').observe(latency)
            
    async def _send_result(self, result: TDAResultEvent) -> None:
        """Send TDA result to output topic"""
        # Run post-process hooks
        for hook in self.post_process_hooks:
            result = await hook(result)
            
        # Convert to event message
        event = EventMessage(
            key=result.event_id,
            value=json.dumps(asdict(result), default=str).encode(),
            headers={
                'content-type': 'application/json',
                'source': result.source_id,
                'scale': result.scale
            }
        )
        
        # Send to Kafka
        await self.kafka_mesh.send(self.output_topic, event)
        

class SchemaEvolutionHandler:
    """
    Handles schema evolution for streaming TDA events
    """
    
    def __init__(self, schema_registry: SchemaRegistryClient):
        self.schema_registry = schema_registry
        self.schema_cache: Dict[str, Any] = {}
        
    async def handle_evolution(
        self,
        old_schema: str,
        new_schema: str,
        migration_fn: Optional[Callable] = None
    ) -> None:
        """Handle schema evolution"""
        logger.info(
            "handling_schema_evolution",
            old_version=old_schema,
            new_version=new_schema
        )
        
        if migration_fn:
            # Apply custom migration
            await migration_fn(old_schema, new_schema)
        else:
            # Default migration strategy
            logger.warning(
                "no_migration_function_provided",
                old_schema=old_schema,
                new_schema=new_schema
            )
            

# Example usage
async def example_usage():
    """Example of using TDA event processor"""
    
    # Create Kafka mesh
    kafka_mesh = KafkaEventMesh(
        producer_id=1,
        bootstrap_servers="localhost:9092",
        max_concurrent_sends=100
    )
    
    # Create multi-scale TDA processor
    scales = [
        ScaleConfig("1min", window_size=1000, slide_interval=100),
        ScaleConfig("5min", window_size=5000, slide_interval=500),
        ScaleConfig("15min", window_size=15000, slide_interval=1500),
    ]
    tda_processor = MultiScaleProcessor(scales)
    
    # Create event processor
    event_processor = TDAEventProcessor(
        kafka_mesh=kafka_mesh,
        tda_processor=tda_processor,
        input_topic="point_clouds",
        output_topic="tda_results"
    )
    
    # Add hooks for custom processing
    async def log_event(event):
        logger.info("processing_event", event_id=event.key)
        return event
        
    event_processor.add_pre_process_hook(log_event)
    
    # Start processing
    await event_processor.start()
    

if __name__ == "__main__":
    asyncio.run(example_usage())