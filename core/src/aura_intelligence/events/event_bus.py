"""
Simple Event Bus for AURA Intelligence
"""

from typing import Dict, List, Callable, Any
import asyncio
import logging

logger = logging.getLogger(__name__)


class EventBus:
    """Simple event bus for inter-component communication"""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._async_subscribers: Dict[str, List[Callable]] = {}
        
    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to an event type"""
        if asyncio.iscoroutinefunction(handler):
            if event_type not in self._async_subscribers:
                self._async_subscribers[event_type] = []
            self._async_subscribers[event_type].append(handler)
        else:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(handler)
            
    async def publish(self, event_type: str, data: Any):
        """Publish an event"""
        # Handle sync subscribers
        if event_type in self._subscribers:
            for handler in self._subscribers[event_type]:
                try:
                    handler(data)
                except Exception as e:
                    logger.error(f"Error in event handler: {e}")
                    
        # Handle async subscribers
        if event_type in self._async_subscribers:
            tasks = []
            for handler in self._async_subscribers[event_type]:
                tasks.append(handler(data))
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                
    def unsubscribe(self, event_type: str, handler: Callable):
        """Unsubscribe from an event type"""
        if asyncio.iscoroutinefunction(handler):
            if event_type in self._async_subscribers:
                self._async_subscribers[event_type].remove(handler)
        else:
            if event_type in self._subscribers:
                self._subscribers[event_type].remove(handler)