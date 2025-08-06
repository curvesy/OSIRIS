"""
Event Store for AURA Intelligence
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid
import logging

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Event structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


# Alias for compatibility
DomainEvent = Event


class EventStore:
    """
    Simple in-memory event store.
    """
    
    def __init__(self):
        self.events: List[Event] = []
        self.subscribers: Dict[str, List[callable]] = {}
        
    def append(self, event_type: str, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Event:
        """Append an event to the store."""
        event = Event(
            type=event_type,
            data=data,
            metadata=metadata or {}
        )
        self.events.append(event)
        
        # Notify subscribers
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in event subscriber: {e}")
                    
        return event
        
    def get_events(self, event_type: Optional[str] = None, limit: int = 100) -> List[Event]:
        """Get events from the store."""
        if event_type:
            filtered = [e for e in self.events if e.type == event_type]
        else:
            filtered = self.events
            
        return filtered[-limit:]
        
    def subscribe(self, event_type: str, callback: callable):
        """Subscribe to events of a specific type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        
    def clear(self):
        """Clear all events."""
        self.events.clear()
        
    def get_event_count(self) -> int:
        """Get total event count."""
        return len(self.events)