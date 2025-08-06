"""
AURA Event Bus Protocol
=======================
The nervous system that connects all AURA components.
"""

from typing import Protocol, AsyncIterator, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class EventMetadata:
    """Metadata for every event in the system."""
    id: str
    timestamp: datetime
    stream: str
    type: str
    provenance_id: Optional[str] = None
    retry_count: int = 0


@dataclass
class Event:
    """Standard event structure for AURA."""
    metadata: EventMetadata
    payload: Dict[str, Any]


class EventBus(Protocol):
    """
    The contract for AURA's nervous system.
    
    This protocol ensures we can swap implementations (Redis → NATS)
    without changing any consumer code.
    """
    
    async def publish(self, stream: str, data: Dict[str, Any]) -> str:
        """
        Publish an event to a stream.
        
        Args:
            stream: Target stream name (e.g., "fuzz_failures", "shape_updates")
            data: Event payload
            
        Returns:
            Event ID for tracking
        """
        ...
    
    async def subscribe(
        self, 
        stream: str, 
        group: str, 
        consumer: str
    ) -> AsyncIterator[Event]:
        """
        Subscribe to a stream as part of a consumer group.
        
        Args:
            stream: Stream to subscribe to
            group: Consumer group name (for load balancing)
            consumer: Unique consumer ID within the group
            
        Yields:
            Events from the stream
        """
        ...
    
    async def ack(self, stream: str, group: str, event_id: str) -> bool:
        """
        Acknowledge successful processing of an event.
        
        Args:
            stream: Stream name
            group: Consumer group name
            event_id: Event to acknowledge
            
        Returns:
            True if acknowledged successfully
        """
        ...
    
    async def health_check(self) -> bool:
        """Check if the bus is operational."""
        ...


# LangGraph adapter interface (for future integration)
def adapter_for_langgraph(bus: EventBus):
    """
    Returns a LangGraph-compatible adapter for the event bus.
    This will be implemented when we wire LangGraph orchestration.
    """
    # TODO: Implement in Day 4
    pass