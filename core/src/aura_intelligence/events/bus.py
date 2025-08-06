"""
Event Bus for AURA Intelligence
Simple publish-subscribe implementation for decoupled communication
"""

import asyncio
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Event:
    """Base event class"""
    event_type: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventBus:
    """Simple async event bus implementation"""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the event bus processor"""
        if self._running:
            return
            
        self._running = True
        self._task = asyncio.create_task(self._process_events())
        logger.info("EventBus started")
        
    async def stop(self):
        """Stop the event bus processor"""
        self._running = False
        if self._task:
            await self._task
        logger.info("EventBus stopped")
        
    async def _process_events(self):
        """Process events from the queue"""
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._dispatch_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
                
    async def _dispatch_event(self, event: Event):
        """Dispatch event to subscribers"""
        subscribers = self._subscribers.get(event.event_type, [])
        
        for subscriber in subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(event)
                else:
                    subscriber(event)
            except Exception as e:
                logger.error(f"Error in subscriber {subscriber.__name__}: {e}")
                
    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to an event type"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed {handler.__name__} to {event_type}")
        
    def unsubscribe(self, event_type: str, handler: Callable):
        """Unsubscribe from an event type"""
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(handler)
            
    async def publish(self, event: Event):
        """Publish an event"""
        await self._queue.put(event)
        
    async def publish_event(self, event_type: str, payload: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        """Convenience method to publish an event"""
        event = Event(
            event_type=event_type,
            payload=payload,
            metadata=metadata or {}
        )
        await self.publish(event)


# Global event bus instance
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


__all__ = ["EventBus", "Event", "get_event_bus"]